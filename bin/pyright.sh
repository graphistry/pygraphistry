#!/bin/bash
set -e

# Run from project root
# Non-zero exit code on fail

# Resolve pyright command, then delegate with repo config.
if pyright --version >/dev/null 2>&1; then
    PYRIGHT_CMD_ARR=(pyright)
elif uvx --from pyright pyright --version >/dev/null 2>&1; then
    PYRIGHT_CMD_ARR=(uvx --from pyright pyright)
elif npx pyright --version >/dev/null 2>&1; then
    PYRIGHT_CMD_ARR=(npx pyright)
else
    echo "pyright could not be executed via pyright, uvx, or npx."
    exit 1
fi

PYRIGHT_EXTRA_ARGS_ARR=()
if [ -n "${PYRIGHT_EXTRA_ARGS:-}" ]; then
    read -r -a PYRIGHT_EXTRA_ARGS_ARR <<< "$PYRIGHT_EXTRA_ARGS"
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
CONFIG_ARGS=(-p pyrightconfig.json)

echo "Running pyright..."
echo "Checking: ${TARGET_ARGS[*]}"

"${PYRIGHT_CMD_ARR[@]}" --version
CMD=( "${PYRIGHT_CMD_ARR[@]}" "${PYRIGHT_EXTRA_ARGS_ARR[@]}" "${CONFIG_ARGS[@]}" )
CMD+=( "${TARGET_ARGS[@]}" )
"${CMD[@]}"

echo "Pyright check completed!"
