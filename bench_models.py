#!/usr/bin/env python3
"""Benchmark Whisper models on test.mp3: speed + divergence-WER.

Runs each model at pure ASR (no alignment, no diarization) so the only thing
varying is the transcription model itself. No ground-truth transcript exists,
so 'accuracy' is measured as word error rate against the strongest model
(large-v3) used as a pseudo-reference. Also dumps the money-figure lines so we
can eyeball what matters for tax content.
"""
import json
import re
import subprocess
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
AUDIO = PROJECT / "test.mp3"
BENCH = PROJECT / "bench"
MODELS = ["medium", "large-v3-turbo", "large-v3"]
REFERENCE = "large-v3"


def run_model(model):
    outdir = BENCH / model.replace("/", "_")
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "whisperx", str(AUDIO),
        "--model", model,
        "--language", "en",
        "--device", "cpu",
        "--compute_type", "int8",
        "--no_align",
        "--output_format", "json",
        "--output_dir", str(outdir),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        print(f"[{model}] FAILED:\n{proc.stderr[-1500:]}")
        return None
    js = outdir / "test.json"
    text = " ".join(s.get("text", "").strip()
                     for s in json.loads(js.read_text()).get("segments", []))
    return {"model": model, "seconds": elapsed, "text": text}


def normalize(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9$ ]", " ", t)   # keep $ and digits — money matters
    t = re.sub(r"\s+", " ", t).strip()
    return t.split()


def wer(ref_words, hyp_words):
    # Levenshtein over word lists
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return float(m > 0)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


def money_lines(text):
    return sorted(set(re.findall(r"\$[\d,]+(?:\.\d+)?(?:\s?(?:thousand|k))?", text.lower())))


def main():
    results = {}
    for model in MODELS:
        print(f"\n=== running {model} ===", flush=True)
        r = run_model(model)
        if r:
            results[model] = r
            print(f"[{model}] {r['seconds']:.1f}s", flush=True)

    ref = results.get(REFERENCE)
    print("\n" + "=" * 60)
    print(f"{'model':18} {'time(s)':>8} {'xRT':>6} {'WER vs '+REFERENCE:>16}")
    print("-" * 60)
    audio_len = 258.0  # test.mp3 duration in seconds
    ref_words = normalize(ref["text"]) if ref else None
    for model in MODELS:
        r = results.get(model)
        if not r:
            print(f"{model:18} {'FAILED':>8}")
            continue
        xrt = audio_len / r["seconds"]
        w = wer(ref_words, normalize(r["text"])) if ref_words else float("nan")
        wtxt = "(reference)" if model == REFERENCE else f"{w*100:5.1f}%"
        print(f"{model:18} {r['seconds']:8.1f} {xrt:5.1f}x {wtxt:>16}")

    print("\n=== money figures detected per model ===")
    for model in MODELS:
        r = results.get(model)
        if r:
            print(f"{model:18} {money_lines(r['text'])}")

    (BENCH / "summary.json").write_text(json.dumps(
        {m: {"seconds": r["seconds"], "text": r["text"]}
         for m, r in results.items()}, indent=2))
    print(f"\nfull transcripts saved under {BENCH}/<model>/test.json")


if __name__ == "__main__":
    main()
