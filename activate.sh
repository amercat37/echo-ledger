#!/usr/bin/env bash

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    echo "Error: .venv not found."
    return 1 2>/dev/null || exit 1
fi

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    echo "Error: .env not found."
    return 1 2>/dev/null || exit 1
fi

source "$PROJECT_DIR/.venv/bin/activate"

set -a
source "$PROJECT_DIR/.env"
set +a

echo "Echo Ledger environment activated."
