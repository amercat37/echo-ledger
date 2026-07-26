"""web.py — HTTP routes via Flask's test client.

DIRS/CONFIG are monkeypatched to a tmp dir per test, so routes never read or
write the real transcripts, audio, or speakers.json. The background worker is
never started (main() isn't called), so uploads just enqueue.
"""
import io
import json

import pytest

import web
import profiles as prof


@pytest.fixture
def client(dirs, config, profiles, monkeypatch):
    prof.save_profiles(config["speakers_file"], profiles)
    monkeypatch.setattr(web, "DIRS", dirs)
    monkeypatch.setattr(web, "CONFIG", config)
    web._jobs.clear()
    monkeypatch.setattr(web, "_paused", False)
    web.app.config["TESTING"] = True
    return web.app.test_client()


@pytest.fixture
def seed(dirs, data):
    """Put a viewable transcript (m.md + m.json) on disk."""
    (dirs["output"] / "m.json").write_text(json.dumps(data))
    (dirs["output"] / "m.md").write_text("**Speaker 1** · 0:00\nhello\n")
    return "m"


def _upload(client, name, content=b"x"):
    return client.post("/upload", content_type="multipart/form-data",
                       data={"files": (io.BytesIO(content), name)})


def test_upload_rejects_non_audio(client):
    body = _upload(client, "note.txt").get_json()
    assert body["rejected"] == ["note.txt"] and body["accepted"] == []
    assert web._jobs == []


def test_upload_accepts_audio_and_queues(client):
    body = _upload(client, "clip.mp3").get_json()
    assert body["accepted"] == ["clip.mp3"]
    assert len(web._jobs) == 1 and web._jobs[0]["status"] == "queued"


def test_state_shape(client):
    s = client.get("/api/state").get_json()
    assert set(s) == {"paused", "queue", "completed"}


def test_people_lists_enrolled(client):
    names = [p["name"] for p in client.get("/api/people").get_json()["people"]]
    assert names == ["Allen", "Sharon"]


def test_people_rename_and_delete(client):
    assert client.post("/api/people/rename", json={"old": "Allen", "new": "Al"}).get_json()["ok"]
    names = [p["name"] for p in client.get("/api/people").get_json()["people"]]
    assert "Al" in names and "Allen" not in names
    assert client.post("/api/people/delete", json={"name": "Al"}).get_json()["ok"]


def test_people_samples_endpoint(client, seed, dirs):
    (dirs["done"] / "meeting.mp3").write_bytes(b"AUDIO")   # so has_audio is True
    body = client.get("/api/people/samples?name=Allen").get_json()
    assert body["name"] == "Allen" and len(body["samples"]) == 1
    s = body["samples"][0]
    assert s["source_stem"] == "m" and s["has_audio"] is True and s["hash"]


def test_people_samples_unknown_person_404(client):
    assert client.get("/api/people/samples?name=Nobody").status_code == 404


def test_people_sample_delete_removes_person_when_last(client, seed):
    r = client.post("/api/people/sample/delete",
                    json={"name": "Allen", "index": 0}).get_json()
    assert r["ok"] and r["removed_person"] is True
    names = [p["name"] for p in client.get("/api/people").get_json()["people"]]
    assert "Allen" not in names and "Sharon" in names


def test_transcript_speakers_endpoint(client, seed):
    body = client.get(f"/api/transcript/{seed}").get_json()
    assert body["stem"] == "m"
    assert {sp["label"] for sp in body["speakers"]} == {"SPEAKER_00", "SPEAKER_01"}
    for sp in body["speakers"]:                       # snippet == its play segment
        assert sp["snippet"] and sp["segment"]


def test_roster_endpoint_assigns_closed_set(client, seed):
    r = client.post("/api/roster",
                    json={"stem": seed, "people": ["Allen", "Sharon"]}).get_json()
    assert r["ok"]
    labels = {a["label"]: a["name"] for a in r["assignments"]}
    assert labels == {"SPEAKER_00": "Allen", "SPEAKER_01": "Sharon"}


def test_roster_open_mode_leaves_others(client, seed):
    # roster only {Allen}; open mode names Allen and leaves Sharon's voice unassigned
    r = client.post("/api/roster",
                    json={"stem": seed, "people": ["Allen"], "allow_others": True}).get_json()
    assert r["ok"] and r["allow_others"] is True
    labels = {a["label"]: a["name"] for a in r["assignments"]}
    assert labels == {"SPEAKER_00": "Allen"}          # SPEAKER_01 left as an "other"


def test_roster_unknown_stem_404(client):
    assert client.post("/api/roster",
                       json={"stem": "nope", "people": ["Allen"]}).status_code == 404


def test_retag_generic(client, seed):
    r = client.post("/api/retag",
                    json={"stem": seed, "label": "SPEAKER_00", "action": "generic"})
    assert r.get_json()["ok"]


def test_delete_endpoint(client, seed, dirs):
    (dirs["done"] / "meeting.mp3").write_bytes(b"A")
    r = client.post("/api/delete", json={"stem": seed}).get_json()
    assert r["ok"] and ".md" in r["removed"]
    assert not (dirs["output"] / "m.md").exists()


def test_tags_lists_unknown_speakers(client, dirs, data):
    # a transcript whose speakers match nobody -> shows up on the tag list
    prof.save_profiles(web.CONFIG["speakers_file"], {})
    (dirs["output"] / "u.json").write_text(json.dumps(data))
    (dirs["output"] / "u.md").write_text("x")
    body = client.get("/api/tags").get_json()
    stems = [t["stem"] for t in body["transcripts"]]
    assert "u" in stems


def test_reprocess_enqueues_with_exact_count(client, seed, dirs):
    (dirs["done"] / "meeting.mp3").write_bytes(b"A")   # source audio present
    r = client.post("/api/reprocess", json={"stem": seed, "num_speakers": 2}).get_json()
    assert r["ok"] and r["min_speakers"] == 2 and r["max_speakers"] == 2   # exact -> min=max
    job = web._jobs[-1]
    assert job["kind"] == "reprocess" and job["stem"] == "m" and job["status"] == "queued"


def test_reprocess_unknown_stem_404(client):
    assert client.post("/api/reprocess", json={"stem": "nope"}).status_code == 404


def test_reprocess_audio_gone_409(client, seed):
    # transcript exists (seed) but its audio was never put in done/
    assert client.post("/api/reprocess", json={"stem": seed, "num_speakers": 2}).status_code == 409


def test_missing_stem_is_404(client):
    assert client.get("/view/nope").status_code == 404
    assert client.get("/audio/nope").status_code == 404


def test_audio_serves_with_range(client, seed, dirs):
    (dirs["done"] / "meeting.mp3").write_bytes(b"AUDIODATA")
    r = client.get(f"/audio/{seed}", headers={"Range": "bytes=0-3"})
    assert r.status_code == 206
