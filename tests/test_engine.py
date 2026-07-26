"""echo_ledger.py — matching, segment selection, rendering, manage, retention."""
import json
import os
import time

import echo_ledger as engine
import profiles as prof


# ---- matching + overrides ----------------------------------------------------

def test_match_speakers(data, config, profiles):
    name_map, report = engine.match_speakers(data, config, profiles)
    assert name_map == {"SPEAKER_00": "Allen", "SPEAKER_01": "Sharon"}
    assert len(report) == 2


def test_apply_overrides_name_and_generic(data):
    nm = {"SPEAKER_00": "Allen", "SPEAKER_01": "Sharon"}
    data["label_overrides"] = {"SPEAKER_00": "", "SPEAKER_01": "Bob"}
    engine.apply_overrides(nm, data)
    assert "SPEAKER_00" not in nm      # "" forces back to Speaker N
    assert nm["SPEAKER_01"] == "Bob"   # explicit name wins


# ---- the snippet/play alignment fix -----------------------------------------

def test_sample_segment_prefers_first_solid_over_longest(data):
    # SPEAKER_00's first >=3s segment is the clean opening (1-5s), not the 1s "Hi"
    s = engine.sample_segment_for(data["segments"], "SPEAKER_00")
    assert (s["start"], s["end"]) == (1.0, 5.0)
    assert s["text"].startswith("This is a clean opening")


def test_sample_segment_falls_back_to_longest(data):
    # SPEAKER_01 has no <early> solid clip before the long one -> longest is returned
    s = engine.sample_segment_for(data["segments"], "SPEAKER_01")
    assert (s["start"], s["end"]) == (6.0, 20.0)


def test_sample_segment_none_when_silent():
    assert engine.sample_segment_for([], "SPEAKER_00") is None


def test_list_unidentified_snippet_matches_its_segment(data, config):
    # no profiles saved -> everyone is unidentified
    out = {sp["label"]: sp for sp in engine.list_unidentified(data, config, {})}
    s00 = out["SPEAKER_00"]
    # the snippet text is exactly the text of the segment the play button will use
    seg_text = next(seg["text"] for seg in data["segments"]
                    if seg["speaker"] == "SPEAKER_00"
                    and seg["start"] == s00["segment"]["start"])
    assert s00["snippet"] == seg_text


def test_list_unidentified_empty_when_all_matched(data, config, profiles):
    assert engine.list_unidentified(data, config, profiles) == []


def test_transcript_speakers_snippet_matches_segment(data, config, profiles):
    for sp in engine.transcript_speakers(data, config, profiles):
        seg_text = next(seg["text"] for seg in data["segments"]
                        if seg["speaker"] == sp["label"]
                        and seg["start"] == sp["segment"]["start"])
        assert sp["snippet"] == seg_text
        assert sp["matched_name"] in ("Allen", "Sharon")


# ---- rendering ---------------------------------------------------------------

def test_render_markdown_applies_names(data, config, profiles, dirs):
    md = dirs["output"] / "t.md"
    engine.render_markdown(data, md, config, profiles)
    text = md.read_text()
    assert "**Allen**" in text and "**Sharon**" in text


def test_resolve_stem_avoids_collisions(dirs):
    (dirs["output"] / "meeting.md").write_text("x")
    assert engine.resolve_stem(dirs["output"], "meeting") == "meeting-2"
    (dirs["output"] / "meeting-2.json").write_text("x")
    assert engine.resolve_stem(dirs["output"], "meeting") == "meeting-3"


# ---- manage / re-tag ---------------------------------------------------------

def test_all_people_counts_samples(config, profiles):
    prof.save_profiles(config["speakers_file"], profiles)
    people = {p["name"]: p["samples"] for p in engine.all_people(config)}
    assert people == {"Allen": 1, "Sharon": 1}


def test_retag_name_enrolls_and_relabels(data, config, dirs, write_transcript):
    jf = write_transcript("m", data)
    res = engine.set_override(jf, dirs, config, "SPEAKER_00", "name", "Bob")
    assert res["ok"] and "Bob" in prof.load_profiles(config["speakers_file"])


def test_retag_generic_forces_speaker_n(data, config, dirs, write_transcript, profiles):
    prof.save_profiles(config["speakers_file"], profiles)
    jf = write_transcript("m", data)
    engine.set_override(jf, dirs, config, "SPEAKER_00", "generic")
    md = (dirs["output"] / "m.md").read_text()
    assert "Allen" not in md            # forced back to a generic label
    # and the override is persisted in the JSON
    assert json.loads(jf.read_text())["label_overrides"]["SPEAKER_00"] == ""


def test_retag_auto_clears_override(data, config, dirs, write_transcript, profiles):
    prof.save_profiles(config["speakers_file"], profiles)
    data["label_overrides"] = {"SPEAKER_00": ""}
    jf = write_transcript("m", data)
    engine.set_override(jf, dirs, config, "SPEAKER_00", "auto")
    assert "SPEAKER_00" not in json.loads(jf.read_text()).get("label_overrides", {})
    assert "**Allen**" in (dirs["output"] / "m.md").read_text()  # match restored


def test_rename_and_delete_person_all(data, config, dirs, write_transcript, profiles):
    prof.save_profiles(config["speakers_file"], profiles)
    write_transcript("m", data)
    engine.rename_person_all(config, dirs, "Allen", "Al")
    assert "Al" in prof.load_profiles(config["speakers_file"])
    assert "**Al**" in (dirs["output"] / "m.md").read_text()
    engine.delete_person_all(config, dirs, "Al")
    assert "Al" not in prof.load_profiles(config["speakers_file"])
    assert "**Al**" not in (dirs["output"] / "m.md").read_text()


# ---- per-voiceprint: see / hear / delete -------------------------------------

def test_person_samples_reconstructs_origin(config, dirs, write_transcript, data, profiles):
    prof.save_profiles(config["speakers_file"], profiles)
    write_transcript("m", data)                       # SPEAKER_00 == Allen's vector
    (dirs["done"] / "meeting.mp3").write_bytes(b"AUDIO")
    samples = engine.person_samples(config, dirs, "Allen")
    assert len(samples) == 1
    s = samples[0]
    assert s["index"] == 0 and s["hash"]
    assert s["source_stem"] == "m" and s["label"] == "SPEAKER_00"
    assert s["snippet"] and s["segment"] and s["has_audio"] is True


def test_person_samples_origin_lost_when_no_transcript(config, dirs, profiles):
    # Allen is enrolled but no transcript on disk holds his embedding.
    prof.save_profiles(config["speakers_file"], profiles)
    samples = engine.person_samples(config, dirs, "Allen")
    assert len(samples) == 1
    s = samples[0]
    assert s["index"] == 0 and s["hash"]              # still identifiable + deletable
    assert s["source_stem"] is None and s["has_audio"] is False and s["segment"] is None


def test_person_samples_unknown_person_is_none(config, dirs, profiles):
    prof.save_profiles(config["speakers_file"], profiles)
    assert engine.person_samples(config, dirs, "Nobody") is None


def test_delete_sample_removes_one_and_keeps_rest(config, dirs, write_transcript, data):
    prof.save_profiles(config["speakers_file"],
                       {"Allen": [[1.0, 0, 0, 0], [0, 0, 1.0, 0]]})
    write_transcript("m", data)
    res = engine.delete_sample(config, dirs, "Allen", 1)
    assert res["ok"] and res["removed_person"] is False and res["remaining"] == 1
    assert prof.load_profiles(config["speakers_file"])["Allen"] == [[1.0, 0, 0, 0]]


def test_delete_sample_last_removes_person(config, dirs, write_transcript, data, profiles):
    prof.save_profiles(config["speakers_file"], profiles)
    write_transcript("m", data)
    res = engine.delete_sample(config, dirs, "Allen", 0)
    assert res["ok"] and res["removed_person"] is True and res["remaining"] == 0
    assert "Allen" not in prof.load_profiles(config["speakers_file"])
    assert "**Allen**" not in (dirs["output"] / "m.md").read_text()  # relabeled


def test_delete_sample_hash_guard_blocks_wrong_target(config, dirs, profiles):
    prof.save_profiles(config["speakers_file"], profiles)
    res = engine.delete_sample(config, dirs, "Allen", 0, expected_hash="deadbeef")
    assert res["ok"] is False and "changed" in res["error"]
    assert prof.load_profiles(config["speakers_file"])["Allen"] == [[1.0, 0, 0, 0]]  # untouched


def test_delete_sample_hash_guard_allows_matching_hash(config, dirs, write_transcript, data, profiles):
    prof.save_profiles(config["speakers_file"], profiles)
    write_transcript("m", data)
    good = engine.person_samples(config, dirs, "Allen")[0]["hash"]
    res = engine.delete_sample(config, dirs, "Allen", 0, expected_hash=good)
    assert res["ok"] and res["removed_person"] is True


def test_delete_sample_bad_index_and_unknown_person(config, dirs, profiles):
    prof.save_profiles(config["speakers_file"], profiles)
    assert engine.delete_sample(config, dirs, "Allen", 9)["ok"] is False
    assert engine.delete_sample(config, dirs, "Allen", -1)["ok"] is False
    assert engine.delete_sample(config, dirs, "Ghost", 0)["error"] == "no such person"


# ---- delete + retention ------------------------------------------------------

def test_source_audio_path_prefers_recorded_then_falls_back(dirs, write_transcript, data):
    write_transcript("m", data)                      # source_audio == meeting.mp3
    (dirs["done"] / "meeting.mp3").write_bytes(b"AUDIO")
    assert engine.source_audio_path("m", dirs).name == "meeting.mp3"
    # fallback: stem-named audio when no source_audio recorded
    (dirs["output"] / "x.json").write_text("{}")
    (dirs["done"] / "x.wav").write_bytes(b"A")
    assert engine.source_audio_path("x", dirs).name == "x.wav"


def test_delete_transcript_full_and_audio_only(dirs, write_transcript, data):
    write_transcript("m", data)
    (dirs["output"] / "m.md").write_text("x")
    (dirs["done"] / "meeting.mp3").write_bytes(b"A")
    removed = engine.delete_transcript("m", dirs, audio_only=True)
    assert removed == ["audio"]
    assert (dirs["output"] / "m.md").exists()        # transcript kept
    assert not (dirs["done"] / "meeting.mp3").exists()
    # full delete removes the transcript too
    (dirs["done"] / "meeting.mp3").write_bytes(b"A")
    removed = engine.delete_transcript("m", dirs)
    assert set(removed) == {"audio", ".md", ".json"}


def test_sweep_old_audio(dirs, write_transcript, data):
    write_transcript("old", data)
    (dirs["done"] / "meeting.mp3").write_bytes(b"A")
    old = time.time() - 40 * 86400
    os.utime(dirs["output"] / "old.json", (old, old))
    removed = engine.sweep_old_audio(dirs, 30)
    assert removed == ["meeting.mp3"]
    assert (dirs["output"] / "old.json").exists()    # transcript kept
    # fresh audio survives, and days<=0 disables the sweep entirely
    write_transcript("new", {"source_audio": "n.mp3"})
    (dirs["done"] / "n.mp3").write_bytes(b"A")
    assert engine.sweep_old_audio(dirs, 30) == []
    assert engine.sweep_old_audio(dirs, 0) == []


# ---- process_file (WhisperX monkeypatched) -----------------------------------

def test_process_file_end_to_end(dirs, config, monkeypatch):
    canned = {
        "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 4.0,
                      "text": "hello world this is a test",
                      "words": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 4.0,
                                 "word": "hello"}]}],
        "speaker_embeddings": {"SPEAKER_00": [1.0, 0.0, 0.0, 0.0]},
    }

    def fake_whisperx(audio_path, out_dir, cfg, min_speakers=None, max_speakers=None):
        p = out_dir / f"{audio_path.stem}.json"
        p.write_text(json.dumps(canned))
        return p

    monkeypatch.setattr(engine, "run_whisperx", fake_whisperx)
    audio = dirs["input"] / "clip.mp3"
    audio.write_bytes(b"FAKE")
    res = engine.process_file(audio, config, dirs)

    assert res["ok"] and res["stem"] == "clip"
    assert (dirs["output"] / "clip.md").exists()
    saved = json.loads((dirs["output"] / "clip.json").read_text())
    assert saved["source_audio"] == "clip.mp3"       # recorded for ▶ play
    assert (dirs["done"] / "clip.mp3").exists()       # original moved to done/
    assert not audio.exists()


def test_process_file_failure_routes_to_failed(dirs, config, monkeypatch):
    def boom(audio_path, out_dir, cfg, min_speakers=None, max_speakers=None):
        raise RuntimeError("whisperx exploded")

    monkeypatch.setattr(engine, "run_whisperx", boom)
    audio = dirs["input"] / "bad.mp3"
    audio.write_bytes(b"X")
    res = engine.process_file(audio, config, dirs)

    assert res["ok"] is False and "whisperx exploded" in res["error"]
    assert (dirs["failed"] / "bad.mp3").exists()
    assert (dirs["failed"] / "bad.error.txt").exists()


# ---- reprocess (WhisperX monkeypatched) --------------------------------------

def test_reprocess_overwrites_in_place_and_passes_hint(dirs, config, write_transcript,
                                                       data, monkeypatch):
    write_transcript("m", data)                       # source_audio == meeting.mp3
    (dirs["output"] / "m.md").write_text("old markdown")
    (dirs["done"] / "meeting.mp3").write_bytes(b"AUDIO")
    two = {
        "segments": [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.0, "text": "one two three",
             "words": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 4.0, "word": "one"}]},
            {"speaker": "SPEAKER_01", "start": 4.0, "end": 8.0, "text": "four five six",
             "words": [{"speaker": "SPEAKER_01", "start": 4.0, "end": 8.0, "word": "four"}]},
        ],
        "speaker_embeddings": {"SPEAKER_00": [1, 0, 0, 0], "SPEAKER_01": [0, 1, 0, 0]},
    }
    seen = {}

    def fake(audio_path, out_dir, cfg, min_speakers=None, max_speakers=None):
        seen["min"], seen["max"] = min_speakers, max_speakers
        p = out_dir / f"{audio_path.stem}.json"
        p.write_text(json.dumps(two))
        return p

    monkeypatch.setattr(engine, "run_whisperx", fake)
    res = engine.reprocess("m", config, dirs, min_speakers=2, max_speakers=2)

    assert res["ok"] and res["speakers"] == 2
    assert seen == {"min": 2, "max": 2}                       # hint reached whisperx
    saved = json.loads((dirs["output"] / "m.json").read_text())
    assert len(saved["speaker_embeddings"]) == 2              # overwritten in place
    assert (dirs["done"] / "meeting.mp3").exists()             # audio left alone
    assert "four five six" in (dirs["output"] / "m.md").read_text()  # re-rendered


def test_reprocess_missing_audio_is_error(dirs, config, write_transcript, data):
    write_transcript("m", data)                       # no audio in done/
    res = engine.reprocess("m", config, dirs)
    assert res["ok"] is False and "audio" in res["error"].lower()
