#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="$ROOT_DIR/wall-x"
MODEL_DIR="${1:-$ROOT_DIR/models/wall-oss-0.5}"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  echo "Missing model weights: $MODEL_DIR/model.safetensors" >&2
  echo "Run: bash '$ROOT_DIR/scripts/download_wall_oss_0_5.sh'" >&2
  exit 1
fi

cd "$REPO_DIR"
python - "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path

script = Path("scripts/fake_inference.py").read_text()
script = script.replace('model_path = "/path/to/model"', f'model_path = "{sys.argv[1]}"')
exec(compile(script, "scripts/fake_inference.py", "exec"))
PY
