#!/usr/bin/env bash
set -euo pipefail

# Run tests from the workspace root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "Running tests from rainbear-tests workspace..."
# Run from workspace root - uv will automatically resolve workspace members
(cd rainbear-tests && uv run python -m unittest discover -s tests -p 'test_*.py' -q)

echo "OK"
