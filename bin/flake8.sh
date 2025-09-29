#!/bin/bash
set -e

# Try to find the best Python version available
PYTHON_BIN=""
for ver in 3.14 3.13 3.12 3.11 3.10 3.9 3.8; do
    if command -v python$ver &> /dev/null; then
        PYTHON_BIN=python$ver
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "No suitable Python version found (3.8-3.14)"
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# Install flake8 if not available
if ! $PYTHON_BIN -m flake8 --version &> /dev/null; then
    echo "Installing flake8..."
    $PYTHON_BIN -m pip install flake8 --user
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
$PYTHON_BIN -m flake8 \
    $TARGET \
    --count \
    --select=E9,F63,F7,F82 \
    --show-source \
    --statistics

# Full lint check
echo "=== Running full lint check ==="
$PYTHON_BIN -m flake8 \
    $TARGET \
    --exclude=graphistry/graph_vector_pb2.py,graphistry/_version.py \
    --count \
    --ignore=C901,E121,E122,E123,E124,E125,E128,E131,E144,E201,E202,E203,E231,E251,E265,E301,E302,E303,E401,E501,E722,F401,W291,W293,W503 \
    --max-complexity=10 \
    --max-line-length=127 \
    --statistics

echo "Flake8 check completed successfully!"