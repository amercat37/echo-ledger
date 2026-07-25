#!/usr/bin/env python3
"""Echo Ledger — ingest pipeline + speaker identification.

Subcommands:
  (none)            batch: transcribe every audio file in input/, then exit.
  enroll [target]   interactively name the unidentified speakers in a processed
                    transcript, appending their voiceprints to the profile store,
                    then re-render that transcript with the new names.
  relabel [target]  non-interactively re-apply the current profiles to processed
                    transcripts and re-render their markdown (e.g. after enrolling
                    someone, to update past files). No prompts.

`target` is an output stem (e.g. `test2`), a path to an output JSON, or omitted
to act on every JSON in output/.

Transcription runs WhisperX (turbo, diarization + speaker embeddings); matching
is cosine similarity against speakers.json. Config comes from env (see env.sample).
The per-file unit is `process_file()` so a future watcher / web button reuses it.
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import refine_speakers
import profiles as prof

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".mp4", ".aac"}
ECHO_VERSION = "1.0"

log = logging.getLogger("echo.engine")


def setup_logging(level="INFO"):
    """Configure one formatted stdout handler for the whole app (engine + web).

    Every line is `<time> <LEVEL> [<component>] <message>` so a pasted log is
    self-describing and greppable. Shared by the CLI and the web service. The
    chatty werkzeug request log is turned down to WARNING so it doesn't bury the
    meaningful events (the web UI polls /api/state every ~1.5s).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    root.handlers = [handler]  # replace any default/duplicate handlers
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


# ---------------------------------------------------------------- config

def _load_dotenv(path=".env"):
    """Minimal KEY=VALUE loader so local runs work without sourcing activate.sh."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_config():
    """Resolve config from the environment (HF_TOKEN may be None; only the
    transcription path requires it)."""
    if not os.environ.get("HF_TOKEN"):
        _load_dotenv()
    return {
        "hf_token": os.environ.get("HF_TOKEN"),
        "model": os.environ.get("MODEL", "large-v3-turbo"),
        "device": os.environ.get("DEVICE", "cpu"),
        "compute_type": os.environ.get("COMPUTE_TYPE", "int8"),
        "language": os.environ.get("LANGUAGE", "en"),
        "threshold": float(os.environ.get("MATCH_THRESHOLD", "0.5")),
        "speakers_file": os.environ.get("SPEAKERS_FILE", "speakers.json"),
        "retention_days": int(os.environ.get("RETENTION_DAYS", "30")),
        "log_level": os.environ.get("LOG_LEVEL", "INFO"),
    }


def load_dirs():
    dirs = {
        "input": Path(os.environ.get("INPUT_DIR", "input")),
        "output": Path(os.environ.get("OUTPUT_DIR", "output")),
        "done": Path(os.environ.get("DONE_DIR", "done")),
        "failed": Path(os.environ.get("FAILED_DIR", "failed")),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------- matching

def match_speakers(data, config, profiles=None):
    """Return {raw_label: real_name} for speakers whose voiceprint clears the
    threshold, plus a printable report list of (label, name_or_None, score)."""
    if profiles is None:
        profiles = prof.load_profiles(config["speakers_file"])
    emb = data.get("speaker_embeddings", {})
    name_map, report = {}, []
    for label, vec in emb.items():
        name, score = prof.best_match(vec, profiles, config["threshold"])
        if name:
            name_map[label] = name
        report.append((label, name, score))
    return name_map, report


def apply_overrides(name_map, data):
    """Layer per-transcript label overrides on top of the auto-match name_map.
    `data["label_overrides"]` = {raw_label: name}; a name of "" forces that
    speaker back to a generic `Speaker N` on this transcript only."""
    for label, nm in data.get("label_overrides", {}).items():
        if nm:
            name_map[label] = nm
        else:
            name_map.pop(label, None)
    return name_map


def render_markdown(data, md_path, config, profiles=None):
    """Apply current profiles (+ any per-transcript overrides) and write the named
    markdown transcript. Returns the {raw_label: real_name} labels used."""
    segments = data.get("segments", [])
    name_map, _ = match_speakers(data, config, profiles)
    apply_overrides(name_map, data)
    label_map = refine_speakers.display_labels(segments, name_map)
    refine_speakers.write_markdown(segments, md_path, label_map)
    return name_map


# ---------------------------------------------------------------- transcription

def run_whisperx(audio_path, out_dir, config):
    if not config["hf_token"]:
        raise RuntimeError("HF_TOKEN is not set (needed for diarization).")
    cmd = [
        "whisperx", str(audio_path),
        "--model", config["model"],
        "--device", config["device"],
        "--compute_type", config["compute_type"],
        "--language", config["language"],
        "--diarize",
        "--speaker_embeddings",
        "--hf_token", config["hf_token"],
        "-f", "json",
        "--output_dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"whisperx exited {proc.returncode}\n{proc.stderr[-2000:]}")
    produced = out_dir / f"{audio_path.stem}.json"
    if not produced.is_file():
        raise RuntimeError(f"whisperx produced no JSON at {produced}\n{proc.stderr[-2000:]}")
    return produced


def resolve_stem(output_dir, stem):
    """A stem colliding with no existing .md/.json in output_dir (suffix -2, -3...)."""
    candidate, n = stem, 1
    while (output_dir / f"{candidate}.md").exists() or (output_dir / f"{candidate}.json").exists():
        n += 1
        candidate = f"{stem}-{n}"
    return candidate


def process_file(audio_path, config, dirs):
    """Transcribe one audio file end-to-end, matching known speakers by voiceprint.

    Returns a result dict the caller (CLI batch or web worker) can act on:
      success -> {"ok": True,  "stem": <output stem>, "name_map": {...},
                  "report": [(label, name_or_None, score), ...]}
      failure -> {"ok": False, "stem": None, "error": "<Type: message>"}
    """
    audio_path = Path(audio_path)
    stem = audio_path.stem
    log.info("transcribing %s (model=%s device=%s)", audio_path.name,
             config["model"], config["device"])
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            raw_json = run_whisperx(audio_path, Path(tmp), config)
            data = json.loads(raw_json.read_text())

        refine_speakers.refine_data(data)
        data["source_audio"] = audio_path.name  # so the web UI can serve it for ▶ play
        final_stem = resolve_stem(dirs["output"], stem)
        name_map = render_markdown(data, dirs["output"] / f"{final_stem}.md", config)
        refine_speakers.atomic_write_text(
            dirs["output"] / f"{final_stem}.json", json.dumps(data, indent=2))

        shutil.move(str(audio_path), str(dirs["done"] / audio_path.name))
        _, report = match_speakers(data, config)
        dur = time.time() - t0
        log.info("done %s -> output/%s.md (%.1fs, %d speaker(s))",
                 audio_path.name, final_stem, dur, len(report))
        for label, name, score in report:
            # These per-speaker scores are the key diagnostic for "why wasn't X
            # recognized?" — a match needs score >= %.2f (MATCH_THRESHOLD).
            log.info("  %s -> %s (score %.3f, threshold %.2f)",
                     label, name or "unidentified", score, config["threshold"])
        return {"ok": True, "stem": final_stem, "name_map": name_map, "report": report}
    except Exception as exc:
        dest = dirs["failed"] / audio_path.name
        try:
            shutil.move(str(audio_path), str(dest))
        except Exception:
            log.warning("could not move %s to failed/", audio_path.name)
        (dirs["failed"] / f"{stem}.error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        log.error("FAILED %s after %.1fs: %s", audio_path.name, time.time() - t0, exc,
                  exc_info=True)
        return {"ok": False, "stem": None, "error": f"{type(exc).__name__}: {exc}"}


def gather_audio(input_dir):
    return sorted(p for p in input_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


# ---------------------------------------------------------------- enroll / relabel

def resolve_targets(target, dirs):
    """A target stem / json path / None -> list of output JSON paths."""
    if target:
        p = Path(target)
        if p.suffix == ".json" and p.is_file():
            return [p]
        cand = dirs["output"] / f"{Path(target).stem}.json"
        if cand.is_file():
            return [cand]
        print(f"no output JSON found for '{target}'")
        return []
    return sorted(dirs["output"].glob("*.json"))


def sample_segment_for(segments, label, min_dur=3.0):
    """Pick ONE representative segment for a speaker — the clip both SHOWN as the
    snippet and PLAYED by ▶, so what you read is exactly what you hear.

    Prefers the first segment at least `min_dur` seconds long: a clean opening
    ("Hello, thank you for calling...") beats the *longest* segment, which on a
    busy call is often crosstalk. Falls back to the longest segment with speech.
    Returns {"start", "end", "text"}, or None if the speaker has no timed speech.
    """
    with_speech = []
    for seg in segments:
        if seg.get("speaker") != label:
            continue
        txt = seg.get("text", "").strip()
        s, e = seg.get("start"), seg.get("end")
        if txt and s is not None and e is not None:
            with_speech.append({"start": s, "end": e, "text": txt})
    if not with_speech:
        return None
    for seg in with_speech:
        if (seg["end"] - seg["start"]) >= min_dur:
            return seg
    return max(with_speech, key=lambda s: s["end"] - s["start"])


def list_unidentified(data, config, profiles=None):
    """Speakers in one transcript that did NOT match a profile but have real
    speech — the candidates a human can name. Each dict carries the SAME segment
    for its snippet text and its ▶ play range:
      {"label", "snippet", "score", "segment": {"start", "end"}}."""
    if profiles is None:
        profiles = prof.load_profiles(config["speakers_file"])
    segments = data.get("segments", [])
    emb = data.get("speaker_embeddings", {})
    out = []
    for label, vec in emb.items():
        name, score = prof.best_match(vec, profiles, config["threshold"])
        if name:
            continue  # already identified
        seg = sample_segment_for(segments, label)
        if not seg:
            continue  # no speech (automated attendant / empty cluster) — skip
        out.append({"label": label, "snippet": seg["text"], "score": round(score, 3),
                    "segment": {"start": seg["start"], "end": seg["end"]}})
    return out


def enroll_headless(data, names_by_label, config, profiles=None, save=True):
    """Core enrollment shared by the CLI and the web tagging page.

    `names_by_label` = {raw_label: name}; blank/missing names are ignored. Adds
    each named speaker's voiceprint to the profile store and (by default) saves.
    Returns (profiles, added_names)."""
    if profiles is None:
        profiles = prof.load_profiles(config["speakers_file"])
    emb = data.get("speaker_embeddings", {})
    added = []
    for label, name in names_by_label.items():
        name = (name or "").strip()
        vec = emb.get(label)
        if name and vec is not None:
            prof.add_sample(profiles, name, vec)
            added.append(name)
    if added and save:
        prof.save_profiles(config["speakers_file"], profiles)
    return profiles, added


def relabel_all(config, dirs, profiles=None):
    """Re-apply the current profiles to every processed transcript and re-render
    its markdown (cheap — no re-transcription). Returns (scanned, named) where
    `named` counts transcripts that now have at least one real name."""
    if profiles is None:
        profiles = prof.load_profiles(config["speakers_file"])
    scanned = named = 0
    for jf in sorted(dirs["output"].glob("*.json")):
        data = json.loads(jf.read_text())
        md = dirs["output"] / f"{jf.stem}.md"
        name_map = render_markdown(data, md, config, profiles)
        scanned += 1
        if name_map:
            named += 1
    return scanned, named


# ---------------------------------------------------------------- manage / re-tag

def all_people(config, profiles=None):
    """Sorted list of enrolled names, each with how many voiceprints it holds."""
    if profiles is None:
        profiles = prof.load_profiles(config["speakers_file"])
    return [{"name": n, "samples": len(v)} for n, v in sorted(profiles.items())]


def transcript_speakers(data, config, profiles=None):
    """Every speaker with real speech in one transcript, with everything the
    re-tag UI needs: current shown label, profile match, and the SAME segment for
    its snippet + ▶ play range, plus any per-transcript override."""
    if profiles is None:
        profiles = prof.load_profiles(config["speakers_file"])
    segments = data.get("segments", [])
    emb = data.get("speaker_embeddings", {})
    overrides = data.get("label_overrides", {})
    name_map, _ = match_speakers(data, config, profiles)
    display = refine_speakers.display_labels(segments, apply_overrides(dict(name_map), data))
    out = []
    seen = set()
    for seg in segments:
        label = seg.get("speaker")
        if not label or label in seen:
            continue
        sample = sample_segment_for(segments, label)
        if not sample:
            continue  # silent cluster — nothing to identify or play
        seen.add(label)
        matched, score = prof.best_match(emb.get(label, []), profiles, config["threshold"])
        out.append({
            "label": label,
            "display": display.get(label, label),
            "matched_name": matched,
            "score": round(score, 3),
            "snippet": sample["text"],
            "override": overrides.get(label),  # None=auto, ""=forced generic, else name
            "segment": {"start": sample["start"], "end": sample["end"]},
        })
    return out


def set_override(json_path, dirs, config, label, action, name=None, profiles=None):
    """Apply a re-tag action to one speaker on one transcript and re-render.

    action:
      'name'    -> enroll this voiceprint under `name` (existing or new) and
                   relabel EVERY transcript (teaches the system). Also clears any
                   forced-generic override on this label.
      'generic' -> force this speaker to `Speaker N` on THIS transcript only.
      'auto'    -> clear the override; fall back to profile auto-match.
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    overrides = data.setdefault("label_overrides", {})

    if action == "name":
        if not (name or "").strip():
            return {"ok": False, "error": "no name given"}
        overrides.pop(label, None)
        refine_speakers.atomic_write_text(json_path, json.dumps(data, indent=2))
        enroll_headless(data, {label: name}, config, profiles)
        scanned, named = relabel_all(config, dirs)
        return {"ok": True, "action": "name", "name": name.strip(),
                "scanned": scanned, "named": named}

    if action == "generic":
        overrides[label] = ""
    elif action == "auto":
        overrides.pop(label, None)
    else:
        return {"ok": False, "error": f"unknown action '{action}'"}

    refine_speakers.atomic_write_text(json_path, json.dumps(data, indent=2))
    md = dirs["output"] / f"{json_path.stem}.md"
    render_markdown(data, md, config, profiles)
    return {"ok": True, "action": action}


def rename_person_all(config, dirs, old, new):
    """Rename an enrolled person everywhere, then relabel all transcripts."""
    profiles = prof.load_profiles(config["speakers_file"])
    if not prof.rename_person(profiles, old, new):
        return {"ok": False, "error": "nothing to rename"}
    prof.save_profiles(config["speakers_file"], profiles)
    scanned, _ = relabel_all(config, dirs, profiles)
    return {"ok": True, "old": old, "new": new.strip(), "scanned": scanned}


def delete_person_all(config, dirs, name):
    """Delete an enrolled person, then relabel all transcripts (their voice falls
    back to the next match or a generic Speaker N)."""
    profiles = prof.load_profiles(config["speakers_file"])
    if not prof.delete_person(profiles, name):
        return {"ok": False, "error": "no such person"}
    prof.save_profiles(config["speakers_file"], profiles)
    scanned, _ = relabel_all(config, dirs, profiles)
    return {"ok": True, "name": name, "scanned": scanned}


# ---------------------------------------------------------------- delete / retention

def source_audio_path(stem, dirs):
    """Locate the original audio for a transcript in done/. Prefers the recorded
    `source_audio` in the JSON; falls back to matching done/<stem>.*. None if gone."""
    jf = dirs["output"] / f"{stem}.json"
    if jf.is_file():
        try:
            src = json.loads(jf.read_text()).get("source_audio")
        except Exception:
            src = None
        if src:
            cand = dirs["done"] / Path(src).name
            if cand.is_file():
                return cand
    for cand in sorted(dirs["done"].glob(f"{stem}.*")):
        if cand.suffix.lower() in AUDIO_EXTS:
            return cand
    return None


def delete_transcript(stem, dirs, audio_only=False):
    """Delete a transcript's source audio and (unless audio_only) its md + json.
    Returns the list of things removed."""
    removed = []
    ap = source_audio_path(stem, dirs)
    if ap and ap.is_file():
        ap.unlink()
        removed.append("audio")
    if not audio_only:
        for ext in (".md", ".json"):
            f = dirs["output"] / f"{stem}{ext}"
            if f.is_file():
                f.unlink()
                removed.append(ext)
    return removed


def sweep_old_audio(dirs, days, now=None):
    """Delete source audio whose transcript is older than `days` (by the JSON's
    mtime = when it was processed/last re-tagged). Transcripts are kept. days<=0
    disables. Returns the list of deleted audio filenames."""
    if days <= 0:
        return []
    cutoff = (now if now is not None else time.time()) - days * 86400
    removed = []
    for jf in sorted(dirs["output"].glob("*.json")):
        if jf.stat().st_mtime >= cutoff:
            continue
        ap = source_audio_path(jf.stem, dirs)
        if ap and ap.is_file():
            ap.unlink()
            removed.append(ap.name)
    return removed


def enroll_cmd(target, config, dirs):
    profiles = prof.load_profiles(config["speakers_file"])
    targets = resolve_targets(target, dirs)
    if not targets:
        return 1
    for jf in targets:
        data = json.loads(jf.read_text())
        print(f"\n== {jf.name} ==")
        answers = {}
        for cand in list_unidentified(data, config, profiles):
            label, snip, score = cand["label"], cand["snippet"], cand["score"]
            print(f'  {label}: "{snip[:100]}"  (best guess {score:.3f})')
            try:
                ans = input(f"    name for {label} (blank = skip): ").strip()
            except EOFError:
                print()
                ans = ""
            if ans:
                answers[label] = ans
        _, added = enroll_headless(data, answers, config, profiles)
        md = dirs["output"] / f"{jf.stem}.md"
        name_map = render_markdown(data, md, config, profiles)
        print(f"  re-rendered {md.name}: "
              + (", ".join(f"{k}={v}" for k, v in name_map.items()) or "no names matched"))
    return 0


def relabel_cmd(target, config, dirs):
    profiles = prof.load_profiles(config["speakers_file"])
    targets = resolve_targets(target, dirs)
    if not targets:
        return 1
    for jf in targets:
        data = json.loads(jf.read_text())
        md = dirs["output"] / f"{jf.stem}.md"
        name_map = render_markdown(data, md, config, profiles)
        print(f"{md.name}: " + (", ".join(f"{k}={v}" for k, v in name_map.items())
                                 or "no names matched"))
    return 0


# ---------------------------------------------------------------- entrypoint

def run_batch(config, dirs):
    if not config["hf_token"]:
        sys.exit("Error: HF_TOKEN is not set (env or .env). Needed for transcription.")
    files = gather_audio(dirs["input"])
    if not files:
        log.info("nothing to process (input/ is empty)")
        return 0
    log.info("batch: processing %d file(s) from %s/", len(files), dirs["input"])
    ok = failed = 0
    for f in files:
        if process_file(f, config, dirs)["ok"]:
            ok += 1
        else:
            failed += 1
    log.info("batch summary: %d transcribed, %d failed", ok, failed)
    return 1 if failed else 0


def main():
    args = sys.argv[1:]
    config = load_config()
    setup_logging(config["log_level"])
    dirs = load_dirs()
    if args and args[0] == "enroll":
        return enroll_cmd(args[1] if len(args) > 1 else None, config, dirs)
    if args and args[0] == "relabel":
        return relabel_cmd(args[1] if len(args) > 1 else None, config, dirs)
    return run_batch(config, dirs)


if __name__ == "__main__":
    sys.exit(main())
