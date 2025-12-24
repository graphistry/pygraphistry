#!/bin/bash
set -ex

# Run from project root
# Non-zero exit code on fail

# Resolve flake8 command, then delegate to runner (prefer uvx, then venv)
if command -v uvx >/dev/null 2>&1; then
  FLAKE8_CMD="uvx flake8"
elif command -v python >/dev/null 2>&1; then
  FLAKE8_CMD="python -m flake8"
else
  FLAKE8_CMD="flake8"
fi
FLAKE8_CMD="$FLAKE8_CMD" ./bin/flake8.sh "$@"

# Check for relative imports with '..' using flake8-quotes or custom regex
# This will fail if any relative imports with .. are found
echo "Checking for relative imports with '..' ..."
if grep -r "from \.\." graphistry --include="*.py" --exclude-dir="__pycache__"; then
    echo "ERROR: Found relative imports with '..'. Use absolute imports instead."
    exit 1
fi
