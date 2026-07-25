#!/usr/bin/env python3
"""Echo Ledger — web front door (Phase 3).

A small Flask app wrapping the SAME engine the batch CLI uses
(`echo_ledger.process_file`). Drop audio on the page → it is saved to input/
and auto-enqueued → a single background worker transcribes one file at a time
(the M3 is CPU-only, so serial is the honest model) → finished transcripts show
up with View + Download. A second page lets you name unknown speakers, which
enrolls their voiceprint and re-labels every past transcript.

Design invariants (see the project brief):
  * Docker-only run mode; localhost-only is enforced by the compose port publish.
  * One serial worker — a visible queue with per-item cancel + a global pause,
    NOT a per-item transcribe button (a button buys no throughput here).
  * Nothing is written to the Obsidian vault. This app only DISPLAYS and lets
    you DOWNLOAD the markdown; filing it into Obsidian stays a manual step.
"""
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask, jsonify, render_template, request, send_file, abort,
)
from werkzeug.utils import secure_filename

import echo_ledger as engine

app = Flask(__name__)

CONFIG = engine.load_config()
DIRS = engine.load_dirs()

# ---------------------------------------------------------------- job queue
# One process, one worker thread, one lock guarding the job list. Jobs live in
# memory (the queue is transient); finished transcripts live on disk in output/
# and survive a restart, so the "completed" panel is always rebuilt from disk.

_lock = threading.Lock()
_jobs = []              # list of dicts, insertion order == arrival order
_wake = threading.Event()
_paused = False


def _new_job(filename):
    return {
        "id": uuid.uuid4().hex[:12],
        "filename": filename,   # name as staged in input/
        "status": "queued",     # queued | transcribing | done | failed | canceled
        "stem": None,           # output stem once done
        "error": None,
    }


def _free_input_path(filename):
    """A collision-free path in input/ (keeps the original name when possible,
    else appends -2, -3 ...). The output stem follows the staged filename."""
    name = secure_filename(filename) or "audio"
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, n = name, 1
    while (DIRS["input"] / candidate).exists():
        n += 1
        candidate = f"{stem}-{n}{suffix}"
    return DIRS["input"] / candidate


def enqueue(file_storage):
    """Save one uploaded file to input/ and add it to the queue. Returns the job
    (or None if the extension is not an accepted audio type)."""
    ext = Path(file_storage.filename or "").suffix.lower()
    if ext not in engine.AUDIO_EXTS:
        return None
    dest = _free_input_path(file_storage.filename)
    file_storage.save(str(dest))
    job = _new_job(dest.name)
    with _lock:
        _jobs.append(job)
    _wake.set()
    return job


def recover_orphans():
    """Re-enqueue any audio still sitting in input/ at startup. All pending work
    — files queued OR interrupted mid-transcription — lives in input/ until
    process_file finishes and moves it to done/, so anything here after a restart
    is unfinished. This makes a crash/reboot self-healing: the container comes back
    (restart policy) and resumes exactly what it was doing. Called once before the
    worker starts, so no lock is needed."""
    found = [p.name for p in engine.gather_audio(DIRS["input"])]
    for name in found:
        _jobs.append(_new_job(name))
    if found:
        print(f"recovered {len(found)} interrupted file(s) from input/: "
              + ", ".join(found), flush=True)
    return found


def retention_worker():
    """Once a day, delete source audio whose transcript is older than
    RETENTION_DAYS (transcripts are always kept). Disabled when RETENTION_DAYS<=0."""
    days = CONFIG["retention_days"]
    while days > 0:
        removed = engine.sweep_old_audio(DIRS, days)
        if removed:
            print(f"retention: deleted {len(removed)} audio file(s) older than "
                  f"{days}d (transcripts kept): " + ", ".join(removed), flush=True)
        time.sleep(24 * 3600)


def _next_queued():
    for job in _jobs:
        if job["status"] == "queued":
            return job
    return None


def worker():
    """Serial worker: pull one queued job at a time and run it through the engine.
    Respects the global pause flag between jobs (never mid-transcription)."""
    while True:
        with _lock:
            job = None if _paused else _next_queued()
            if job:
                job["status"] = "transcribing"
        if job is None:
            _wake.wait(timeout=2.0)
            _wake.clear()
            continue

        audio_path = DIRS["input"] / job["filename"]
        result = engine.process_file(audio_path, CONFIG, DIRS)
        with _lock:
            if result["ok"]:
                job["status"] = "done"
                job["stem"] = result["stem"]
            else:
                job["status"] = "failed"
                job["error"] = result["error"]


# ---------------------------------------------------------------- disk views

def completed_transcripts():
    """Rebuild the completed list from output/ (survives restarts). Each item:
    stem, has JSON (needed for tagging), and whether unknown speakers remain."""
    profiles = engine.prof.load_profiles(CONFIG["speakers_file"])
    items = []
    for md in sorted(DIRS["output"].glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        stem = md.stem
        jf = DIRS["output"] / f"{stem}.json"
        unnamed = 0
        if jf.is_file():
            try:
                data = json.loads(jf.read_text())
                unnamed = len(engine.list_unidentified(data, CONFIG, profiles))
            except Exception:
                unnamed = 0
        items.append({
            "stem": stem,
            "has_json": jf.is_file(),
            "unnamed": unnamed,
            "mtime": md.stat().st_mtime,
        })
    return items


def _safe_stem(stem):
    """Only allow a stem that maps to a real file in output/ (no path tricks)."""
    stem = Path(stem).name
    if not (DIRS["output"] / f"{stem}.md").is_file():
        abort(404)
    return stem


# ---------------------------------------------------------------- routes

@app.route("/")
def index():
    return render_template("index.html", active="home")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    accepted, rejected = [], []
    for f in files:
        if not f or not f.filename:
            continue
        job = enqueue(f)
        (accepted if job else rejected).append(f.filename)
    return jsonify({"accepted": accepted, "rejected": rejected})


@app.route("/api/state")
def api_state():
    with _lock:
        queue = [
            {"id": j["id"], "filename": j["filename"], "status": j["status"],
             "error": j["error"], "stem": j["stem"]}
            for j in _jobs if j["status"] != "done"
        ]
        paused = _paused
    return jsonify({
        "paused": paused,
        "queue": queue,
        "completed": completed_transcripts(),
    })


@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_cancel(job_id):
    with _lock:
        for job in _jobs:
            if job["id"] == job_id:
                if job["status"] == "queued":
                    job["status"] = "canceled"
                    (DIRS["input"] / job["filename"]).unlink(missing_ok=True)
                    return jsonify({"ok": True})
                # transcribing/done/failed can't be pulled from the queue
                return jsonify({"ok": False, "reason": job["status"]}), 409
    abort(404)


@app.route("/api/dismiss/<job_id>", methods=["POST"])
def api_dismiss(job_id):
    """Drop a finished/failed/canceled job from the activity feed."""
    with _lock:
        before = len(_jobs)
        _jobs[:] = [j for j in _jobs
                    if not (j["id"] == job_id and j["status"] in
                            ("failed", "canceled", "done"))]
    return jsonify({"ok": len(_jobs) != before})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    global _paused
    with _lock:
        _paused = bool(request.json.get("paused")) if request.is_json else not _paused
    _wake.set()
    return jsonify({"paused": _paused})


@app.route("/view/<stem>")
def view(stem):
    stem = _safe_stem(stem)
    md = (DIRS["output"] / f"{stem}.md").read_text()
    return render_template("view.html", stem=stem, turns=_parse_transcript(md))


@app.route("/download/<stem>")
def download(stem):
    stem = _safe_stem(stem)
    return send_file((DIRS["output"] / f"{stem}.md").resolve(),
                     as_attachment=True, download_name=f"{stem}.md")


# ---------------------------------------------------------------- tagging

@app.route("/tag")
def tag():
    return render_template("tag.html", active="tag")


@app.route("/api/tags")
def api_tags():
    """Every transcript that still has unknown speakers with real speech, enriched
    with a ▶ play segment per speaker, plus the shared people dropdown."""
    profiles = engine.prof.load_profiles(CONFIG["speakers_file"])
    out = []
    for jf in sorted(DIRS["output"].glob("*.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        unknown = engine.list_unidentified(data, CONFIG, profiles)
        if not unknown:
            continue
        segments = data.get("segments", [])
        for sp in unknown:
            sp["segment"] = engine.best_segment_for(segments, sp["label"])
        out.append({"stem": jf.stem, "speakers": unknown,
                    "has_audio": _source_audio_path(jf.stem) is not None})
    return jsonify({"transcripts": out, "people": engine.all_people(CONFIG)})


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """Name unknown speakers in one transcript, then re-label every transcript.
    Body: {"stem": "...", "names": {"SPEAKER_01": "Allen", ...}}."""

    payload = request.get_json(force=True)
    stem = Path(payload.get("stem", "")).name
    names = payload.get("names", {}) or {}
    jf = DIRS["output"] / f"{stem}.json"
    if not jf.is_file():
        abort(404)
    data = json.loads(jf.read_text())
    profiles, added = engine.enroll_headless(data, names, CONFIG)
    if not added:
        return jsonify({"ok": False, "added": [], "scanned": 0, "named": 0})
    scanned, named = engine.relabel_all(CONFIG, DIRS, profiles)
    return jsonify({"ok": True, "added": added, "scanned": scanned, "named": named})


# ---------------------------------------------------------------- manage / re-tag

@app.route("/speakers")
def speakers():
    return render_template("speakers.html", active="speakers")


@app.route("/api/people")
def api_people():
    return jsonify({"people": engine.all_people(CONFIG)})


@app.route("/api/people/rename", methods=["POST"])
def api_people_rename():
    p = request.get_json(force=True)
    return jsonify(engine.rename_person_all(CONFIG, DIRS,
                                            p.get("old", ""), p.get("new", "")))


@app.route("/api/people/delete", methods=["POST"])
def api_people_delete():
    p = request.get_json(force=True)
    return jsonify(engine.delete_person_all(CONFIG, DIRS, p.get("name", "")))


@app.route("/api/delete", methods=["POST"])
def api_delete():
    """Delete a transcript and its source audio. Body: {stem, audio_only?}.
    audio_only=true keeps the transcript and just frees the (bulky, biometric) audio."""
    p = request.get_json(force=True)
    stem = Path(p.get("stem", "")).name
    if not (DIRS["output"] / f"{stem}.md").is_file() and not (DIRS["output"] / f"{stem}.json").is_file():
        abort(404)
    removed = engine.delete_transcript(stem, DIRS, audio_only=bool(p.get("audio_only")))
    return jsonify({"ok": True, "removed": removed})


@app.route("/api/transcript/<stem>")
def api_transcript(stem):
    """Everything the per-transcript re-tag panel needs: all speakers + the
    people dropdown."""
    stem = _safe_stem(stem)
    jf = DIRS["output"] / f"{stem}.json"
    if not jf.is_file():
        abort(404)
    data = json.loads(jf.read_text())
    return jsonify({
        "stem": stem,
        "speakers": engine.transcript_speakers(data, CONFIG),
        "people": engine.all_people(CONFIG),
        "has_audio": _source_audio_path(stem) is not None,
    })


@app.route("/api/retag", methods=["POST"])
def api_retag():
    """Re-tag one speaker on one transcript. Body:
    {stem, label, action: 'name'|'generic'|'auto', name?}."""
    p = request.get_json(force=True)
    stem = Path(p.get("stem", "")).name
    jf = DIRS["output"] / f"{stem}.json"
    if not jf.is_file():
        abort(404)
    return jsonify(engine.set_override(
        jf, DIRS, CONFIG, p.get("label", ""), p.get("action", ""), p.get("name")))


def _source_audio_path(stem):
    """Absolute path to a transcript's source audio (for ▶ play), or None if gone."""
    ap = engine.source_audio_path(stem, DIRS)
    return ap.resolve() if ap else None


@app.route("/audio/<stem>")
def audio(stem):
    stem = _safe_stem(stem)
    src = _source_audio_path(stem)
    if src is None:
        abort(404)
    # conditional=True enables HTTP range requests so the browser can seek.
    return send_file(src, conditional=True)


# ---------------------------------------------------------------- md rendering

def _parse_transcript(md):
    """Turn our transcript markdown back into structured turns for the view page.
    Each turn line looks like `**Name** · 0:05` followed by the spoken text."""
    turns = []
    cur = None
    for line in md.splitlines():
        line = line.rstrip()
        if line.startswith("**") and "**" in line[2:]:
            if cur:
                turns.append(cur)
            head = line[2:]
            name, _, rest = head.partition("**")
            ts = rest.replace("·", "").strip()
            cur = {"name": name.strip(), "ts": ts, "text": ""}
        elif line and cur is not None:
            cur["text"] = (cur["text"] + " " + line).strip()
    if cur:
        turns.append(cur)
    return turns


def main():
    recover_orphans()  # resume anything left in input/ from a crash/restart
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=retention_worker, daemon=True).start()
    port = int(os.environ.get("PORT", "5000"))
    # Listen on all interfaces INSIDE the container; the compose port publish
    # (127.0.0.1:5000:5000) is what actually restricts access to localhost.
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    sys.exit(main())
