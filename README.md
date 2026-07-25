# Echo Ledger

Drop an audio file in → get a clean, **speaker-tagged** markdown transcript out.
Known voices are named automatically: enroll someone once and Echo Ledger
recognizes them in every future recording.

Built on [WhisperX](https://github.com/m-bain/whisperX) (transcription +
diarization) with cosine-similarity speaker matching. Runs **CPU-only** on Apple
Silicon (CTranslate2 has no Metal backend), so it transcribes one file at a time.

> **Everything runs in Docker.** Nothing is meant to run on the host directly.

---

## Quick start

1. **Create `.env`** from the sample and add your Hugging Face token
   (needed for the gated pyannote diarization model — accept the terms for
   `pyannote/speaker-diarization-community-1` on huggingface.co first):

   ```bash
   cp env.sample .env
   # edit .env → set HF_TOKEN=hf_...
   ```

2. **Start the web UI:**

   ```bash
   docker compose up web
   ```

   Open **http://127.0.0.1:5000**. The port is bound to localhost only.

The first run downloads ~2 GB of models into the `hf-cache` Docker volume; after
that they're reused, so subsequent starts are fast.

---

## Using the web UI

Two pages:

- **Transcribe** — drag audio onto the drop zone (or click to choose). Files
  **auto-start** transcribing; there's no per-file button because the machine
  processes one at a time anyway. Watch the **queue** (cancel anything still
  waiting, or **Pause** the whole worker to drop a batch and step away).
  Finished transcripts appear below with **View** and **Download**.
- **Tag speakers** — any transcript with unknown voices (`Speaker 1`, `Speaker 2`…)
  shows a short speech snippet per unknown speaker with a name box. Name one and
  save: the voiceprint is stored and **every past transcript is re-labeled**
  automatically. No re-transcription — labeling is instant.

Downloaded files keep the original name (`myrecording.md`). Echo Ledger never
writes to your Obsidian vault — filing the `.md` there stays a deliberate manual
step (so Livesync remains the vault's only writer).

---

## Batch / command line (optional)

The same engine runs headless. Drop files in `input/` and:

```bash
docker compose run --rm echo-ledger              # transcribe everything in input/
docker compose run --rm echo-ledger enroll <stem>   # interactively name unknowns (TTY)
docker compose run --rm echo-ledger relabel         # re-apply profiles to all transcripts
```

`<stem>` is an output name, e.g. `meeting` for `output/meeting.json`.

---

## How files flow

```
input/    drop audio here (web UI or CLI)
output/   {stem}.md  (readable transcript)  +  {stem}.json  (full data + voiceprints)
done/     originals, moved here after success
failed/   originals + {stem}.error.txt, if transcription failed
state/    speakers.json — the voiceprint library
```

---

## Logs & troubleshooting

All activity is logged to the container's stdout. To see it:

```bash
docker compose logs web              # everything so far
docker compose logs -f web           # follow live
docker compose logs --tail=100 web   # last 100 lines
```

Each line is `<time> <LEVEL> [<component>] <message>`. A run looks like:

```
2026-07-25 18:42:38 INFO [echo.web] Echo Ledger v1.0 starting on 0.0.0.0:5000 (log level INFO)
2026-07-25 18:42:38 INFO [echo.web] config: model=large-v3-turbo device=cpu ... threshold=0.50 retention_days=30
2026-07-25 18:42:38 INFO [echo.web] HF_TOKEN present: True
2026-07-25 18:42:40 INFO [echo.web] worker: starting job 8a03... (small1.mp3)
2026-07-25 18:42:53 INFO [echo.engine] done small1.mp3 -> output/small1.md (13.8s, 1 speaker(s))
2026-07-25 18:42:53 INFO [echo.engine]   SPEAKER_00 -> Sharon (score 0.572, threshold 0.50)
```

The startup banner records the whole runtime config, and every job logs its
duration and each speaker's **match score vs. the threshold** — so the log is
self-diagnosing. The chatty per-request line is suppressed on purpose (the page
polls every ~1.5s) to keep the log high-signal.

**Getting help:** copy the output of `docker compose logs web` (or the last ~50
lines around the problem) — it almost always contains the answer. Set
`LOG_LEVEL=DEBUG` in `.env` for more detail.

| Symptom | Where to look |
|---------|---------------|
| A file failed to transcribe | Log line `FAILED <file>: ...` (includes the whisperx error) and `failed/<name>.error.txt`. Usually a bad/empty audio file or a diarization model issue. |
| Diarization won't run / auth error | Startup line `HF_TOKEN present: False`, or a token/terms error in the FAILED line. Fix `HF_TOKEN` in `.env` and accept the model terms on huggingface.co. |
| "Why wasn't I recognized?" | The `SPEAKER_xx -> ... (score, threshold)` line. A match needs `score ≥ threshold` (default 0.50). Short/noisy clips score lower — enroll that person from more recordings, or lower `MATCH_THRESHOLD`. |
| ▶ Play does nothing | The audio was deleted (by you or the 30-day retention sweep). The transcript stays, but playback needs the original in `done/`. |
| Web UI won't load | `docker compose ps` (is `web` up?) and `docker compose logs web` for a startup error. |
| Interrupted job after a restart | Look for `recovered N interrupted file(s) from input/` at startup — it resumes automatically. |

## Configuration (`.env`)

| Key | Default | Notes |
|-----|---------|-------|
| `HF_TOKEN` | — | **Required.** Hugging Face token for the diarization model. |
| `MODEL` | `large-v3-turbo` | Benchmarked best speed **and** accuracy for this box. |
| `DEVICE` | `cpu` | Mac has no GPU path for WhisperX. |
| `COMPUTE_TYPE` | `int8` | |
| `LANGUAGE` | `en` | |
| `MATCH_THRESHOLD` | `0.5` | Cosine similarity to auto-name a voice (validated: same person ≈0.84, different people ≤0.39). |
| `SPEAKERS_FILE` | `speakers.json` | Container path is `/data/state/speakers.json`. |
| `RETENTION_DAYS` | `30` | Auto-delete source **audio** older than this (transcripts kept). `0` disables. |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |

Keep `.env` to `HF_TOKEN` only unless you have a reason to override a default —
extra vars can shadow the container paths the image sets.

---

## Privacy note

`speakers.json` holds **biometric voiceprints** and raw audio is personal data.
Both are gitignored and never baked into the Docker image. Audio of any kind is
excluded from version control as a hard rule.

---

## Architecture

One engine, swappable front doors — `src/echo_ledger.py :: process_file()`
transcribes, refines diarization, matches known speakers, renders markdown, and
routes the file. The CLI and the Flask web app (`src/web.py`) both call it.

- `src/echo_ledger.py` — engine + batch/enroll/relabel CLI.
- `src/web.py` — Flask app: upload, serial worker, queue, view/download, tagging.
- `src/profiles.py` — voiceprint store + cosine matching.
- `src/refine_speakers.py` — fill diarization gaps, majority-vote segments, render markdown.
- `src/templates/` — web UI pages.

---

## Tests

Fast unit + route tests (`tests/`), no models or audio needed — WhisperX is
monkeypatched and every test runs against a temp dir, so they never touch your
real `speakers.json` or transcripts.

```bash
uv pip install -r requirements-dev.txt   # or: pip install -r requirements-dev.txt
pytest                                    # ~0.2s, 46 tests
```

They cover matching, the snippet/▶-segment alignment, per-transcript overrides,
enroll/rename/delete, delete + retention, atomic writes, and the HTTP routes.
`pytest` is a dev-only dependency and is not included in the Docker image.
