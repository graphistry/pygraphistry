#!/bin/bash
# secrets.sh - Run secret detection manually
# Usage: ./scripts/secrets.sh [--update-baseline]

set -euo pipefail

if [ "${1:-}" == "--update-baseline" ]; then
    EXCLUDES_FILE="$(dirname "$0")/ci/secret-excludes.sh"
    if [ -f "$EXCLUDES_FILE" ]; then
        # shellcheck source=scripts/ci/secret-excludes.sh
        source "$EXCLUDES_FILE"
    fi
    SECRET_SCAN_EXCLUDE_REGEX="${SECRET_SCAN_EXCLUDE_REGEX:-^(plans/|tmp/)}"

    PYTHON_CMD=()

    if command -v uv >/dev/null 2>&1; then
        UV_CACHE_DIR="${UV_CACHE_DIR:-$(pwd)/.uv-cache}"
        export UV_CACHE_DIR
        mkdir -p "$UV_CACHE_DIR"
        PYTHON_CMD=(uv run python3)
    fi

    if [ ${#PYTHON_CMD[@]} -eq 0 ] && command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD=(python3)
    fi

    if [ ${#PYTHON_CMD[@]} -eq 0 ] && command -v python >/dev/null 2>&1; then
        PYTHON_CMD=(python)
    fi

    if [ ${#PYTHON_CMD[@]} -eq 0 ]; then
        echo "ERROR: python not found; needed to run detect-secrets" >&2
        exit 1
    fi

    if "${PYTHON_CMD[@]}" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("detect_secrets") else 1)
PY
    then
        :
    else
        echo "ERROR: detect-secrets not found. Install with: pip install detect-secrets" >&2
        exit 1
    fi

    SCRIPT_PATH="$(dirname "$0")/ci/secret-detection.py"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "ERROR: missing $SCRIPT_PATH" >&2
        exit 1
    fi

    echo "Updating secrets baseline..."
    "${PYTHON_CMD[@]}" "$SCRIPT_PATH" create-baseline --baseline .secrets.baseline --exclude-files "$SECRET_SCAN_EXCLUDE_REGEX" .
    echo "Baseline updated. Review changes and commit if appropriate."
    exit 0
fi

exec "$(dirname "$0")/ci/secret-detection.sh"
