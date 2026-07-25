FROM python:3.12-slim

# ffmpeg is required by whisperx/pyannote for audio decoding.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, from the PyTorch CPU index — avoids pulling the large
# default CUDA wheel. Pinned to match the tested environment.
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Includes src/templates/ for the web UI.
COPY src/ ./src/

# Container-side folders (bind-mounted from the host via docker-compose).
ENV INPUT_DIR=/data/input \
    OUTPUT_DIR=/data/output \
    DONE_DIR=/data/done \
    FAILED_DIR=/data/failed \
    SPEAKERS_FILE=/data/state/speakers.json

# Web UI port (the `web` compose service overrides ENTRYPOINT to run src/web.py).
EXPOSE 5000

# Default entrypoint = run-on-demand batch: process everything in input/, then exit.
ENTRYPOINT ["python", "src/echo_ledger.py"]
