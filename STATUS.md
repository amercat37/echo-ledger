# Echo Ledger — Status Checkpoint

_Last updated: 2026-07-25_

Audio → speaker-tagged transcript tool. Drop audio in, get a clean markdown
transcript out with **known speakers automatically named** (enroll a voice once,
it's recognized in every future recording). Transcripts are meant to land in a
memory vault.

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| **1** | Dockerized batch ingest: drop audio → transcribe → markdown + JSON → done/failed | ✅ **done & verified** |
| **2** | Speaker identification: enroll once, auto-name after | ✅ **done & verified (local + container)** |
| **3** | Web UI: upload → serial queue → view/download + speaker tagging | ✅ **done & verified (local + container)** |
| **3.1** | Manage speakers: re-tag/override, rename, delete + ▶ play a voice | ✅ **done & verified (local + container)** |

Deferred: 30-day retention sweep for aged audio/JSON; the "ignore/system" bucket
for recurring automated attendants. **Retired:** auto-push transcripts into the
memory vault (would violate the vault's one-writer rule — the web UI only
displays + downloads; filing into Obsidian stays manual).

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
# Web UI (primary)
docker compose up web            # → http://127.0.0.1:5000  (localhost-only)

# Docker batch / run-on-demand (same image)
docker compose run --rm echo-ledger                 # transcribe input/
docker compose run --rm echo-ledger enroll <stem>   # interactive naming (TTY)
docker compose run --rm echo-ledger relabel         # re-apply profiles to all
```

Web UI (3 pages): **Transcribe** = drag-drop → auto-enqueue → visible queue (per-item
cancel + global pause) → completed list with View + Download. **Tag speakers** = name
unknown voices (dropdown of existing people + Add-new, ▶ play to hear them) → enroll →
auto re-label every transcript. **People** = rename / delete enrolled speakers in-UI.
**View** page also has a per-transcript re-tag panel (override any speaker: name / make
generic / auto) + ▶ play per speaker and per turn. `/audio/<stem>` streams the source
audio from `done/` (persisted `source_audio` in each JSON); ▶ hides if the audio is gone.
Per-transcript overrides live in `data["label_overrides"]` ({label: name}; "" = force
Speaker N). `speakers.json` writes are atomic (temp + rename).

## Key files

- `src/echo_ledger.py` — CLI + `process_file()` (reusable engine; returns a
  result dict) + `list_unidentified` / `enroll_headless` / `relabel_all` (web helpers).
- `src/web.py` — Flask app: upload, serial worker thread + queue, pause/cancel,
  view/download, tagging (`/api/enroll` → `relabel_all`).
- `src/templates/` — `base` / `index` (transcribe) / `view` / `tag` pages.
- `src/profiles.py` — `speakers.json` load/save, cosine match, enrollment.
- `src/refine_speakers.py` — gap-fill/re-vote + `write_markdown` / `display_labels`.
- `Dockerfile`, `docker-compose.yml` (`echo-ledger` batch + `web` service),
  `.dockerignore`, `env.sample`, `requirements.txt` (whisperx + flask).
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
- Models (~2 GB) persist in the `hf-cache` named volume; whisperx's ~360 MB
  English alignment model persists in the `torch-cache` volume (populated on the
  first web run — no re-download after).
- The web worker is a single serial thread (the M3 is CPU-only); pause only takes
  effect between jobs, and cancel only applies to still-queued items.

## Next step

Phase 3 is done. Possible follow-ups (all optional): put Traefik in front (moves
TLS + auth off localhost; must keep the enroll endpoint protected); the 30-day
retention sweep; accumulate more voiceprints per person as they're enrolled from
more files.
