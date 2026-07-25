#!/usr/bin/env python3
"""Post-process WhisperX diarized output to remove unassigned-speaker gaps.

WhisperX assigns a speaker to each word by max temporal overlap with pyannote
turns. Words that fall in gaps (pauses, crosstalk) get no speaker, and a
segment's label can flicker mid-sentence. The transforms here:

  1. Fill every word missing a speaker from its nearest labeled neighbour.
  2. Re-derive each segment's speaker via duration-weighted majority vote, so
     one segment == one speaker.

Exposed as importable functions (`refine_data`, `write_markdown`, ...) for the
ingest pipeline, plus a standalone CLI that writes `.refined.{txt,srt,json}`
next to a WhisperX JSON.
"""
import json
import os
import sys
import tempfile
from pathlib import Path


def atomic_write_text(path, text):
    """Write a file all-or-nothing (temp in the same dir + rename). Prevents a
    half-written file if two threads render the same transcript concurrently."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def midpoint(item):
    s, e = item.get("start"), item.get("end")
    if s is None or e is None:
        return None
    return (s + e) / 2.0


def fill_word_gaps(segments):
    """Assign a speaker to every word, borrowing from the nearest labeled word."""
    flat = []  # (mid, word_dict) for every word that has a timestamp
    for seg in segments:
        for w in seg.get("words", []):
            m = midpoint(w)
            if m is not None:
                flat.append((m, w))
    labeled = [(m, w["speaker"]) for m, w in flat if w.get("speaker")]
    if not labeled:
        return 0  # nothing to borrow from
    filled = 0
    for m, w in flat:
        if w.get("speaker"):
            continue
        _, spk = min(labeled, key=lambda ls: abs(ls[0] - m))
        w["speaker"] = spk
        filled += 1
    return filled


def revote_segments(segments):
    """Set each segment's speaker to the duration-weighted majority of its words."""
    changed = 0
    for seg in segments:
        tally = {}
        for w in seg.get("words", []):
            spk = w.get("speaker")
            if not spk:
                continue
            dur = (w.get("end", 0) or 0) - (w.get("start", 0) or 0)
            tally[spk] = tally.get(spk, 0.0) + max(dur, 1e-3)
        if not tally:
            continue
        winner = max(tally, key=tally.get)
        if seg.get("speaker") != winner:
            changed += 1
        seg["speaker"] = winner
    # segments still without a speaker (no timestamped words): borrow from neighbour
    last = None
    for seg in segments:
        if seg.get("speaker"):
            last = seg["speaker"]
        elif last is not None:
            seg["speaker"] = last
    nxt = None
    for seg in reversed(segments):
        if seg.get("speaker"):
            nxt = seg["speaker"]
        elif nxt is not None:
            seg["speaker"] = nxt
    return changed


def refine_data(data):
    """Fill speaker gaps and re-vote segments in-place. Returns a stats dict.

    Mutates `data["segments"]` only; any other keys (e.g. `speaker_embeddings`)
    are left untouched so they survive a round-trip.
    """
    segments = data.get("segments", [])
    before_w = sum(1 for s in segments for w in s.get("words", []) if not w.get("speaker"))
    before_s = sum(1 for s in segments if not s.get("speaker"))
    filled = fill_word_gaps(segments)
    changed = revote_segments(segments)
    after_w = sum(1 for s in segments for w in s.get("words", []) if not w.get("speaker"))
    after_s = sum(1 for s in segments if not s.get("speaker"))
    return {
        "words_missing_before": before_w, "words_missing_after": after_w,
        "segments_missing_before": before_s, "segments_missing_after": after_s,
        "words_filled": filled, "segments_revoted": changed,
    }


def friendly_labels(segments):
    """Map raw diarization labels (SPEAKER_00...) to 'Speaker 1', 'Speaker 2'...
    in order of first appearance."""
    mapping = {}
    for seg in segments:
        spk = seg.get("speaker")
        if spk and spk not in mapping:
            mapping[spk] = f"Speaker {len(mapping) + 1}"
    return mapping


def display_labels(segments, name_map):
    """Build the render label map: matched speakers show their real name;
    everyone else is numbered 'Speaker N' among the un-named, by appearance.

    name_map: {raw_label: real_name} for speakers that matched a profile.
    """
    mapping = {}
    counter = 0
    for seg in segments:
        spk = seg.get("speaker")
        if not spk or spk in mapping:
            continue
        if spk in name_map:
            mapping[spk] = name_map[spk]
        else:
            counter += 1
            mapping[spk] = f"Speaker {counter}"
    return mapping


def ts(t):
    """Seconds -> M:SS, or H:MM:SS past an hour."""
    t = int(t or 0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def srt_ts(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_markdown(segments, path, label_map=None):
    """Render a readable transcript: consecutive same-speaker segments merged into
    one turn, a single timestamp per turn, friendly speaker labels."""
    if label_map is None:
        label_map = friendly_labels(segments)
    blocks = []
    cur_spk = None
    cur_start = None
    cur_text = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        spk = seg.get("speaker")
        if spk != cur_spk and cur_text:
            blocks.append((cur_spk, cur_start, " ".join(cur_text)))
            cur_text = []
        if not cur_text:
            cur_spk, cur_start = spk, seg.get("start", 0)
        cur_text.append(text)
    if cur_text:
        blocks.append((cur_spk, cur_start, " ".join(cur_text)))

    lines = []
    for spk, start, text in blocks:
        name = label_map.get(spk, spk or "Unknown")
        lines.append(f"**{name}** · {ts(start)}")
        lines.append(text)
        lines.append("")
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def write_txt(segments, path):
    lines = []
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{spk}]: {text}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_srt(segments, path):
    out = []
    for i, seg in enumerate(segments, 1):
        text = seg.get("text", "").strip()
        if not text:
            continue
        spk = seg.get("speaker", "UNKNOWN")
        out.append(str(i))
        out.append(f"{srt_ts(seg['start'])} --> {srt_ts(seg['end'])}")
        out.append(f"[{spk}]: {text}")
        out.append("")
    Path(path).write_text("\n".join(out) + "\n")


def refine_file(json_path):
    """CLI behaviour: refine a WhisperX JSON and write `.refined.{txt,srt,json}` siblings."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    stats = refine_data(data)
    segments = data.get("segments", [])
    stem = json_path.with_suffix("")
    write_txt(segments, Path(str(stem) + ".refined.txt"))
    write_srt(segments, Path(str(stem) + ".refined.srt"))
    Path(str(stem) + ".refined.json").write_text(json.dumps(data, indent=2))
    print(f"words missing speaker: {stats['words_missing_before']} -> "
          f"{stats['words_missing_after']} (filled {stats['words_filled']})")
    print(f"segments missing speaker: {stats['segments_missing_before']} -> "
          f"{stats['segments_missing_after']}")
    print(f"segment labels changed by majority-vote: {stats['segments_revoted']}")
    print(f"wrote: {stem}.refined.txt / .refined.srt / .refined.json")


def main():
    refine_file(sys.argv[1] if len(sys.argv) > 1 else "output/test.json")


if __name__ == "__main__":
    main()
