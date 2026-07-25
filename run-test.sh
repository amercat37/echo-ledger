#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "$PROJECT_DIR/activate.sh"

AUDIO_FILE="$PROJECT_DIR/test.mp3"
OUTPUT_DIR="$PROJECT_DIR/output"

if [[ ! -f "$AUDIO_FILE" ]]; then
    echo "Error: test.mp3 not found in $PROJECT_DIR"
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Error: HF_TOKEN is not loaded."
    exit 1
fi

whisperx "$AUDIO_FILE" \
    --model large-v3-turbo \
    --language en \
    --device cpu \
    --compute_type int8 \
    --diarize \
    --min_speakers 2 \
    --max_speakers 2 \
    --hf_token "$HF_TOKEN" \
    --output_dir "$OUTPUT_DIR"

# Post-process: fill unassigned-speaker gaps and re-vote segment labels.
python3 "$PROJECT_DIR/src/refine_speakers.py" "$OUTPUT_DIR/test.json"
