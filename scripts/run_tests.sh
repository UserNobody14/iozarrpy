#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p dist

echo "[1/2] Building dev-profile wheel (avoids -O3 rustc segfaults seen in some environments)..."
rm -f dist/iozarrpy-dev-*.whl dist/iozarrpy-0.1.0+dev.*.whl
uv run --with maturin maturin build --profile dev -o dist

WHEEL_BUILT="$(ls -1 dist/iozarrpy-0.1.0-cp*-abi3-*.whl | tail -n 1)"
WHEEL_HASH="$(sha256sum "$WHEEL_BUILT" | awk '{print $1}' | cut -c1-12)"

# Cache-bust while keeping a valid wheel filename (PEP 427):
# {dist}-{version}-{python tag}-{abi tag}-{platform tag}.whl
#
# We insert a local version segment into the version, preserving the tags.
WHEEL_BUILT_BASENAME="$(basename "$WHEEL_BUILT")"
WHEEL_TAGS="${WHEEL_BUILT_BASENAME#iozarrpy-0.1.0-}"
WHEEL="dist/iozarrpy-0.1.0+dev.${WHEEL_HASH}-${WHEEL_TAGS}"

if [[ "$WHEEL_BUILT" != "$WHEEL" ]]; then
  cp -f "$WHEEL_BUILT" "$WHEEL"
fi

echo "Built wheel: $WHEEL_BUILT"
echo "Using wheel: $WHEEL"

echo "[2/2] Running Python unittests in a clean uv env with test deps..."
uv run \
  --with "$WHEEL" \
  --with "polars>=0.20.0" \
  --with "numpy>=1.26" \
  --with "pandas>=2.2.0" \
  --with "xarray>=2024.10.0" \
  --with "zarr>=3.0.0" \
  --with "numcodecs>=0.13.0" \
  python -m unittest discover -s tests -p 'test_*.py' -q

echo "OK"


