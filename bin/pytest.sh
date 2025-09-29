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

# Install pytest if not available
if ! $PYTHON_BIN -m pytest --version &> /dev/null; then
    echo "Installing pytest..."
    $PYTHON_BIN -m pip install pytest pytest-xdist --user
fi

# Get the script directory and repo root
SCRIPT_DIR="$( cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to repo root
cd "$REPO_ROOT"

# Show pytest version
$PYTHON_BIN -m pytest --version

# Default pytest arguments for verbosity and parallel execution
PYTEST_ARGS="-vv"

# Check if any arguments were passed
if [ $# -eq 0 ]; then
    # No arguments, run default test location
    TARGET="graphistry/tests"
else
    # Use provided arguments
    TARGET="$@"
fi

echo "Running pytest on: $TARGET"

# Run pytest with verbose output and any additional arguments
# Using python -B to not write .pyc files
$PYTHON_BIN -B -m pytest $PYTEST_ARGS $TARGET

echo "Pytest completed!"