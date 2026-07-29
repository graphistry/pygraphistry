#!/bin/bash
set -ex

# Run from project root
# Non-zero exit code on fail

# Resolve ruff command (prefer uvx, then bare, then python -m)
if command -v uvx >/dev/null 2>&1; then
  RUFF_CMD="uvx ruff"
elif command -v ruff >/dev/null 2>&1; then
  RUFF_CMD="ruff"
elif command -v python >/dev/null 2>&1; then
  RUFF_CMD="python -m ruff"
elif command -v python3 >/dev/null 2>&1; then
  RUFF_CMD="python3 -m ruff"
else
  echo "ruff not found. Install ruff or set it on PATH."
  exit 1
fi

RUFF_CMD="$RUFF_CMD" ./bin/ruff.sh "$@"

# Type-hygiene ratchet (missing annotations, Any/cast, bare generics,
# setattr/attr-writes onto a caller-owned Plottable, closed-vocabulary `str`).
# Per-file counts may shrink but never grow; see DEVELOP.md "Type hygiene guard".
# Stdlib-only, ~1.5s, identical results on py3.8-3.14.
echo "Running type-hygiene guard ..."
if command -v python >/dev/null 2>&1; then
  PY_CMD=python
elif command -v python3 >/dev/null 2>&1; then
  PY_CMD=python3
else
  echo "ERROR: no python interpreter found for the type-hygiene guard."
  exit 1
fi
"$PY_CMD" ./bin/ci_type_hygiene_guard.py

# Check for relative imports with '..' using custom regex
# This will fail if any relative imports with .. are found
echo "Checking for relative imports with '..' ..."
if grep -r "from \.\." graphistry --include="*.py" --exclude-dir="__pycache__"; then
    echo "ERROR: Found relative imports with '..'. Use absolute imports instead."
    exit 1
fi
