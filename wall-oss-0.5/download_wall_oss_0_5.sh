#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT_DIR/models/wall-oss-0.5}"

mkdir -p "$DEST"
python -m pip install -U huggingface_hub

python - "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

dest = sys.argv[1]
snapshot_download(
    repo_id="x-square-robot/wall-oss-0.5",
    local_dir=dest,
    local_dir_use_symlinks=False,
)
print(f"Downloaded x-square-robot/wall-oss-0.5 to {dest}")
PY
