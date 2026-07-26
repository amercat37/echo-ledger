# Echo Ledger — Status Checkpoint

_Last updated: 2026-07-26_

Audio → speaker-tagged transcript tool. Drop audio in, get a clean markdown
transcript out with **known speakers automatically named** (enroll a voice once,
it's recognized in every future recording). Transcripts are meant to land in a
memory vault (filed manually — Livesync stays the vault's only writer).

**Status: complete & production-ready** for its scope (single-user, localhost,
Docker-on-Mac). `main`, 16 commits, **77 passing tests**, verified locally and in
Docker. See the README's "About this project" for the origin story (a test of
whether Claude could build a whole project unaided — it wrote every line).

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| **1** | Dockerized batch ingest: drop audio → transcribe → markdown + JSON → done/failed | ✅ **done & verified** |
| **2** | Speaker identification: enroll once, auto-name after | ✅ **done & verified (local + container)** |
| **3** | Web UI: upload → serial queue → view/download + speaker tagging | ✅ **done & verified (local + container)** |
| **3.1** | Manage speakers: re-tag/override, rename, delete + ▶ play a voice | ✅ **done & verified** |
| **3.2** | Delete + 30-day audio retention + self-healing (restart + orphan recovery) + atomic writes | ✅ **done & verified** |
| **3.3** | Structured logging + troubleshooting docs; pytest suite | ✅ **done & verified** |
| **3.4** | Reprocess (re-run diarization with a speaker-count hint) | ✅ **done & verified** |
| **3.5** | Per-voiceprint see / hear / delete on the People page | ✅ **done & verified** |
| **3.6** | "Who's in this recording?" roster solver (closed + "plus others") | ✅ **done & verified** |
| **3.7** | Tag page: collapse per transcript + reversible "dismiss (not a person)" | ✅ **done & verified** |

**Retired:** auto-push transcripts into the memory vault (would violate the vault's
one-writer rule — the web UI only displays + downloads; filing into Obsidian stays
manual). **Deferred (optional, non-blocking):** fold the roster into the Reprocess
modal so the chosen people also set the speaker count; Traefik front (with a
protected enroll endpoint); a reprocess sensitivity/closeness knob.

## What works today

- **Transcription:** WhisperX `large-v3-turbo`, CPU/int8, diarization + speaker
  embeddings. Auto-detects speaker count; **Reprocess** re-runs it with a count hint.
- **Refinement:** fills unassigned-speaker gaps, duration-weighted majority re-vote.
- **Identification:** cosine match of 256-dim voiceprints against `speakers.json`;
  threshold ~0.5 (validated: same person cross-file ≈ 0.84, different people ≤ 0.39).
- **Rendering:** turn-based markdown, one timestamp per turn; matched voices show
  their name, everyone else is a distinguishable `Speaker N`.
- **Folder lifecycle:** `input/ → output/ (.md + .json) + done/ | failed/ (+ .error.txt)`;
  same-stem collisions get a `-2` suffix.
- **Web UI (4 pages):**
  - **Transcribe** — drag-drop → auto-enqueue → visible serial queue (per-item
    cancel + global pause) → completed list with View / Download / **Delete**.
  - **Tag speakers** — name unknown voices (dropdown of people + Add-new, ▶ play);
    transcripts are **collapsed** to a header (auto-expand when few); a voice that
    isn't a person can be **Dismissed** (reversible flag) with a **Show dismissed**
    restore path. Save → enroll → auto re-label every transcript.
  - **People** — rename / delete a speaker; expand **Voiceprints** to see each voice
    sample's source recording, **▶ hear** it, and **delete** individual samples.
  - **View** — collapsible speaker panel with per-speaker re-tag (name / **Leave
    unnamed** / auto), the **"Who's in this recording?"** roster solver (closed or
    "plus others"), ▶ play per speaker & per turn, **Copy**, Download, **Reprocess**.
- **Reliability:** `restart: unless-stopped` + startup orphan recovery (input/ is a
  durable pending list); daily audio-retention sweep; all profile/markdown/JSON
  writes are atomic (temp + rename).
- **Logging:** `<time> <LEVEL> [component] msg`; startup banner + per-job duration &
  per-speaker match score + every action + FAILED with traceback. Paste
  `docker compose logs web` to self-diagnose.

## How to run

```bash
docker compose up web                               # web UI → http://127.0.0.1:5000
docker compose run --rm echo-ledger                 # batch: transcribe input/
docker compose run --rm echo-ledger enroll <stem>   # interactive naming (TTY)
docker compose run --rm echo-ledger relabel         # re-apply profiles to all
docker compose logs -f web                           # live logs
pytest                                               # 77 tests (dev-only dep)
```

## Key files

- `src/echo_ledger.py` — engine + batch/enroll/relabel CLI. `process_file` /
  `reprocess`; matching, rendering, `list_unidentified` (+ dismiss), `set_ignored`;
  re-tag `set_override`; roster `assign_roster` / `apply_roster`; people
  `person_samples` / `delete_sample`, rename/delete; delete + retention.
- `src/web.py` — Flask app: upload, serial worker (transcribe|reprocess), queue,
  pause/cancel, view/download/delete, tagging (`/api/tags`, `/api/enroll`,
  `/api/ignore`), manage (`/api/people/*`, `/api/roster`), `/audio` (range).
- `src/profiles.py` — `speakers.json` load/save (atomic), cosine match, enrollment.
- `src/refine_speakers.py` — gap-fill/re-vote + `write_markdown` / `atomic_write_text`.
- `src/templates/` — `base` / `index` / `view` / `tag` / `speakers`.
- `tests/` — 77 pytest (no models/audio; monkeypatched, tmp dirs).
- `Dockerfile`, `docker-compose.yml` (`echo-ledger` batch + `web` service),
  `.dockerignore`, `env.sample`, `requirements.txt`, `requirements-dev.txt`.

## Config (env; see `env.sample`)

`HF_TOKEN` (required), `MODEL`, `DEVICE`, `COMPUTE_TYPE`, `LANGUAGE`,
`MATCH_THRESHOLD` (0.5), `SPEAKERS_FILE`, `RETENTION_DAYS` (30), `LOG_LEVEL` (INFO).

## Operational notes (don't trip on these)

- **Profiles for the container must live at `./state/speakers.json`** (bind-mounted
  to `/data/state`). Local runs default to `./speakers.json` at the repo root.
- **Keep `.env` to just `HF_TOKEN`** — extra vars there can override the container's
  Dockerfile paths (both compose services use `env_file: .env`).
- **`speakers.json` is biometric** — gitignored, never baked into the image.
- Models (~2 GB) persist in the `hf-cache` volume; the ~360 MB English alignment
  model persists in `torch-cache` (populated on first web run — no re-download).
- The web worker is a single serial thread (the M3 is CPU-only); pause takes effect
  between jobs, cancel only applies to still-queued items.
