#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p dist

echo "[1/2] Building dev-profile wheel (avoids -O3 rustc segfaults seen in some environments)..."
uv run --with maturin maturin build --profile dev -o dist

WHEEL="$(ls -1 dist/iozarrpy-*.whl | tail -n 1)"
echo "Built wheel: $WHEEL"

echo "[2/2] Running Python unittests in a clean uv env with test deps..."
uv run \
  --with "$WHEEL" \
  --with "polars>=0.20.0" \
  --with "numpy>=1.26" \
  --with "xarray>=2024.10.0" \
  --with "zarr>=3.0.0" \
  --with "numcodecs>=0.13.0" \
  python -m unittest discover -s tests -p 'test_*.py' -q

echo "OK"


