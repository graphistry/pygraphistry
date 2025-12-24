#!/bin/bash
set -e

# Minimal resolution: env override or host flake8
FLAKE8_CMD_ARR=(${FLAKE8_CMD:-flake8})

if ! "${FLAKE8_CMD_ARR[@]}" --version &> /dev/null; then
    echo "flake8 not found. Set FLAKE8_CMD or install flake8 on PATH."
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

echo "Running flake8 on: $TARGET"

# Quick syntax error check
echo "=== Running quick syntax check ==="
"${FLAKE8_CMD_ARR[@]}" \
    $TARGET \
    --count \
    --select=E9,F63,F7,F82 \
    --show-source \
    --statistics

# Full lint check
echo "=== Running full lint check ==="
"${FLAKE8_CMD_ARR[@]}" \
    $TARGET \
    --exclude=graphistry/graph_vector_pb2.py,graphistry/_version.py \
    --count \
    --ignore=C901,E121,E122,E123,E124,E125,E128,E131,E144,E201,E202,E203,E231,E251,E265,E301,E302,E303,E401,E501,E722,F401,W291,W293,W503 \
    --max-complexity=10 \
    --max-line-length=127 \
    --statistics

echo "Flake8 check completed successfully!"
