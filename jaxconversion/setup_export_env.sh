#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${1:-$ROOT/.venv-export}"
REQ_FILE="$ROOT/requirements_export.txt"

echo "Creating isolated export env at: $ENV_DIR"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "python3.12 not found. Install python3.12 first."
  exit 1
fi

python3.12 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQ_FILE"

# Install local jax-layers package used by export_gemma_bundle.py
python -m pip install -e "$ROOT/../jax-layers"

echo
echo "Export env ready."
echo "Activate with:"
echo "  source \"$ENV_DIR/bin/activate\""
echo
echo "Then export with:"
echo "  python \"$ROOT/export_gemma_bundle.py\" --tiny-random --out-prefix /private/tmp/gemma2_bundle"
