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
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import refine_speakers
import profiles as prof

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".mp4", ".aac"}


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


def render_markdown(data, md_path, config, profiles=None):
    """Apply current profiles and write the named markdown transcript. Returns
    the {raw_label: real_name} matches used."""
    segments = data.get("segments", [])
    name_map, _ = match_speakers(data, config, profiles)
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
    print(f"  -> {audio_path.name}: transcribing...", flush=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            raw_json = run_whisperx(audio_path, Path(tmp), config)
            data = json.loads(raw_json.read_text())

        refine_speakers.refine_data(data)
        final_stem = resolve_stem(dirs["output"], stem)
        name_map = render_markdown(data, dirs["output"] / f"{final_stem}.md", config)
        (dirs["output"] / f"{final_stem}.json").write_text(json.dumps(data, indent=2))

        shutil.move(str(audio_path), str(dirs["done"] / audio_path.name))
        _, report = match_speakers(data, config)
        print(f"     done: output/{final_stem}.md")
        for label, name, score in report:
            who = name if name else "unidentified"
            print(f"       {label} -> {who} ({score:.3f})")
        return {"ok": True, "stem": final_stem, "name_map": name_map, "report": report}
    except Exception as exc:
        dest = dirs["failed"] / audio_path.name
        try:
            shutil.move(str(audio_path), str(dest))
        except Exception:
            pass
        (dirs["failed"] / f"{stem}.error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        print(f"     FAILED: {exc}", flush=True)
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


def speaker_snippets(segments, limit=160):
    out = {}
    for seg in segments:
        spk, txt = seg.get("speaker"), seg.get("text", "").strip()
        if spk and txt and len(out.get(spk, "")) < limit:
            out[spk] = (out.get(spk, "") + " " + txt).strip()
    return out


def list_unidentified(data, config, profiles=None):
    """Speakers in one transcript that did NOT match a profile but have real
    speech — the candidates a human can name. Returns a list of dicts:
      {"label": raw_label, "snippet": "...", "score": best_score}
    (`score` is the closest existing profile, for reference; usually low.)"""
    if profiles is None:
        profiles = prof.load_profiles(config["speakers_file"])
    segments = data.get("segments", [])
    emb = data.get("speaker_embeddings", {})
    snippets = speaker_snippets(segments)
    out = []
    for label, vec in emb.items():
        name, score = prof.best_match(vec, profiles, config["threshold"])
        if name:
            continue  # already identified
        snip = snippets.get(label, "").strip()
        if not snip:
            continue  # no speech (automated attendant / empty cluster) — skip
        out.append({"label": label, "snippet": snip, "score": round(score, 3)})
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
        print("nothing to process (input/ is empty)")
        return 0
    print(f"processing {len(files)} file(s) from {dirs['input']}/")
    ok = failed = 0
    for f in files:
        if process_file(f, config, dirs)["ok"]:
            ok += 1
        else:
            failed += 1
    print(f"\nsummary: {ok} transcribed, {failed} failed")
    return 1 if failed else 0


def main():
    args = sys.argv[1:]
    config = load_config()
    dirs = load_dirs()
    if args and args[0] == "enroll":
        return enroll_cmd(args[1] if len(args) > 1 else None, config, dirs)
    if args and args[0] == "relabel":
        return relabel_cmd(args[1] if len(args) > 1 else None, config, dirs)
    return run_batch(config, dirs)


if __name__ == "__main__":
    sys.exit(main())
