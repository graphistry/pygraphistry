#!/bin/bash
set -ex

# Run from project root
# Non-zero exit code on fail

# Resolve mypy command, then delegate to runner (prefer uvx, then venv)
if command -v uvx >/dev/null 2>&1; then
  MYPY_CMD="uvx mypy"
elif command -v python >/dev/null 2>&1; then
  MYPY_CMD="python -m mypy"
else
  MYPY_CMD="mypy"
fi
MYPY_CMD="$MYPY_CMD" ./bin/mypy.sh "$@"
