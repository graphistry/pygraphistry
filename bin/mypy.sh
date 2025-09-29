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

# Install mypy if not available
if ! $PYTHON_BIN -m mypy --version &> /dev/null; then
    echo "Installing mypy..."
    $PYTHON_BIN -m pip install mypy --user
fi

# Get the script directory and repo root
SCRIPT_DIR="$( cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to repo root
cd "$REPO_ROOT"

# Check if specific files were passed as arguments
if [ $# -eq 0 ]; then
    # No arguments, run on entire graphistry directory using config file
    TARGET=""
    CONFIG_ARG="--config-file mypy.ini"
else
    # Use provided arguments
    TARGET="$@"
    # Still use config file for consistency
    CONFIG_ARG="--config-file mypy.ini"
fi

echo "Running mypy..."
if [ -z "$TARGET" ]; then
    echo "Checking entire graphistry directory with mypy.ini config"
else
    echo "Checking: $TARGET"
fi

# Show mypy version
$PYTHON_BIN -m mypy --version

# Run mypy with config file
# If no target specified, mypy will use the paths defined in mypy.ini
if [ -z "$TARGET" ]; then
    $PYTHON_BIN -m mypy $CONFIG_ARG graphistry
else
    $PYTHON_BIN -m mypy $CONFIG_ARG $TARGET
fi

echo "Mypy check completed!"