# Echo Ledger — Status Checkpoint

_Last updated: 2026-07-23_

Audio → speaker-tagged transcript tool. Drop audio in, get a clean markdown
transcript out with **known speakers automatically named** (enroll a voice once,
it's recognized in every future recording). Transcripts are meant to land in a
memory vault.

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| **1** | Dockerized batch ingest: drop audio → transcribe → markdown + JSON → done/failed | ✅ **done & verified** |
| **2** | Speaker identification: enroll once, auto-name after | ✅ **done & verified (local + container)** |
| **3** | Web "process" button / folder-watcher (wraps `process_file()`) | ⬜ not started |

Deferred: 30-day retention sweep for aged audio/JSON; auto-push transcripts into
the memory vault; the "ignore/system" bucket for recurring automated attendants.

## What works today

- **Transcription:** WhisperX `large-v3-turbo`, CPU/int8, diarization + speaker
  embeddings. Auto-detects speaker count.
- **Refinement:** fills unassigned-speaker gaps, duration-weighted majority re-vote.
- **Identification:** cosine match of 256-dim voiceprints against `speakers.json`;
  threshold ~0.5 (validated: same person cross-file ≈ 0.84, different people ≤ 0.39).
- **Rendering:** turn-based markdown, one timestamp per turn; matched voices show
  their name, everyone else is a distinguishable `Speaker N`.
- **Folder lifecycle:** `input/ → output/ (.md + .json) + done/ | failed/ (+ .error.txt)`;
  same-stem collisions get a `-2` suffix.

## How to run

```bash
# Local (after: source activate.sh)
python3 src/echo_ledger.py                 # batch: transcribe everything in input/
python3 src/echo_ledger.py enroll <stem>   # name a transcript's unknown speakers
python3 src/echo_ledger.py relabel <stem>  # re-apply profiles to past transcripts

# Docker (run-on-demand)
docker compose build
docker compose run --rm echo-ledger           # batch
docker compose run --rm echo-ledger enroll <stem>   # interactive (has a TTY)
```

## Key files

- `src/echo_ledger.py` — CLI + `process_file()` (the reusable per-file engine).
- `src/profiles.py` — `speakers.json` load/save, cosine match, enrollment.
- `src/refine_speakers.py` — gap-fill/re-vote + `write_markdown` / `display_labels`.
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `env.sample`, `requirements.txt`.
- `run-test.sh`, `bench_models.py` — legacy/dev (single-file test, model benchmark).

## Config (env; see `env.sample`)

`HF_TOKEN` (required), `MODEL`, `DEVICE`, `COMPUTE_TYPE`, `LANGUAGE`,
`MATCH_THRESHOLD` (0.5), `SPEAKERS_FILE`, and `INPUT/OUTPUT/DONE/FAILED_DIR`.

## Operational notes (don't trip on these)

- **Profiles for the container must live at `./state/speakers.json`** (bind-mounted
  to `/data/state`). Local runs default to `./speakers.json` at the repo root.
- **Keep `.env` to just `HF_TOKEN`** — extra vars there can override the container's
  Dockerfile paths.
- **`speakers.json` is biometric** — gitignored, never baked into the image.
- Models (~2 GB) persist in the `hf-cache` named volume across container runs.
- On Mac, a future watcher must **poll** `input/` (Docker bind-mount FS events are
  unreliable across the VM); add a `processing/` atomic-move claim before concurrency.

## Next step

Phase 3: a folder-watcher (poll-based, same image, long-running container) or a
web "process" button (adds a small job/queue layer since transcription is slow).
Both just call the existing `process_file()`.
