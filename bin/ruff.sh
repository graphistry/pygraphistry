#!/bin/bash
set -e

# Minimal resolution: env override or host ruff
RUFF_CMD_ARR=(${RUFF_CMD:-ruff})

if ! "${RUFF_CMD_ARR[@]}" version &> /dev/null; then
    echo "ruff not found. Set RUFF_CMD or install ruff on PATH."
    exit 1
fi

# Get the script directory and repo root
SCRIPT_DIR="$( cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to repo root
cd "$REPO_ROOT"

# Check if specific files were passed as arguments
if [ $# -eq 0 ]; then
    # No arguments, run on entire graphistry directory
    TARGET="graphistry"
else
    # Use provided arguments
    TARGET="$@"
fi

echo "Running ruff on: $TARGET"

# Quick syntax error check (E9/F63/F7/F82)
echo "=== Running quick syntax check ==="
"${RUFF_CMD_ARR[@]}" check \
    $TARGET \
    --select=E9,F63,F7,F82 \
    --output-format=full \
    --no-fix

# Full lint check (uses config from pyproject.toml)
echo "=== Running full lint check ==="
"${RUFF_CMD_ARR[@]}" check \
    $TARGET \
    --output-format=full \
    --statistics \
    --no-fix

echo "Ruff check completed successfully!"
