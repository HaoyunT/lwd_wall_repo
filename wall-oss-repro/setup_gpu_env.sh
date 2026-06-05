#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="$ROOT_DIR/wall-x"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or Mambaforge on the GPU server first." >&2
  exit 1
fi

conda create --name wallx python=3.10 -y

echo "Run the following commands in an interactive shell:"
echo
echo "  conda activate wallx"
echo "  cd '$REPO_DIR'"
echo "  pip install -r requirements.txt"
echo "  MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation"
echo "  git submodule update --init --recursive"
echo "  MAX_JOBS=4 pip install --no-build-isolation --verbose -e ."
