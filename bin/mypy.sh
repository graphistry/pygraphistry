#!/bin/bash
set -e

# Minimal resolution: env override or host mypy
MYPY_CMD_ARR=(${MYPY_CMD:-mypy})

MYPY_EXTRA_ARGS_ARR=()
if [ -n "${MYPY_EXTRA_ARGS:-}" ]; then
    read -r -a MYPY_EXTRA_ARGS_ARR <<< "$MYPY_EXTRA_ARGS"
fi

# Ensure mypy exists rather than installing (works in CI & avoids PEP 668)
if ! "${MYPY_CMD_ARR[@]}" --version &> /dev/null; then
    echo "mypy not found. Set MYPY_CMD or install mypy on PATH."
    exit 1
fi

# Get the script directory and repo root
SCRIPT_DIR="$( cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to repo root
cd "$REPO_ROOT"

# Check if specific files were passed as arguments
if [ $# -eq 0 ]; then
    TARGET_ARGS=("graphistry")
else
    TARGET_ARGS=("$@")
fi
CONFIG_ARGS=(--config-file mypy.ini)

echo "Running mypy..."
echo "Checking: ${TARGET_ARGS[*]}"

# Show mypy version
"${MYPY_CMD_ARR[@]}" --version

# Run mypy with config file
CMD=( "${MYPY_CMD_ARR[@]}" "${MYPY_EXTRA_ARGS_ARR[@]}" "${CONFIG_ARGS[@]}" )
CMD+=( "${TARGET_ARGS[@]}" )
"${CMD[@]}"

echo "Mypy check completed!"
