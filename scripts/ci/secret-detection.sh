#!/bin/bash
# secret-detection.sh - Centralized secret detection using detect-secrets
# Usage: ./scripts/ci/secret-detection.sh [--check-only]

set -euo pipefail

print_error() {
    echo "ERROR: $1" >&2
    exit 1
}

print_warning() {
    echo "WARN: $1" >&2
}

print_success() {
    echo "OK: $1"
}

CHECK_ONLY=false
if [ "${1:-}" == "--check-only" ]; then
    CHECK_ONLY=true
fi

if [ ! -f "setup.cfg" ] && [ ! -f "pyproject.toml" ] && [ ! -f "setup.py" ]; then
    print_error "Must run from project root (setup.cfg/pyproject.toml/setup.py not found)"
fi

EXCLUDES_FILE="$(dirname "$0")/secret-excludes.sh"
if [ -f "$EXCLUDES_FILE" ]; then
    # shellcheck source=scripts/ci/secret-excludes.sh
    source "$EXCLUDES_FILE"
fi
SECRET_SCAN_EXCLUDE_REGEX="${SECRET_SCAN_EXCLUDE_REGEX:-^(plans/|tmp/)}"

PYTHON_CMD=()
DETECT_SECRETS_AVAILABLE=0

if [ "${PRE_COMMIT:-}" != "1" ] && command -v uv >/dev/null 2>&1; then
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
    print_error "python not found; needed to run detect-secrets"
fi

if "${PYTHON_CMD[@]}" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("detect_secrets") else 1)
PY
then
    DETECT_SECRETS_AVAILABLE=1
fi

BASELINE=".secrets.baseline"
SCRIPT_PATH="scripts/ci/secret-detection.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    print_error "Missing $SCRIPT_PATH"
fi

if [ "$CHECK_ONLY" == true ]; then
    echo "Checking for secrets in staged files..."

    staged_files=()
    while IFS= read -r -d '' file; do
        case "$file" in
            .secrets.baseline)
                continue
                ;;
        esac
        if [[ "$file" =~ $SECRET_SCAN_EXCLUDE_REGEX ]]; then
            continue
        fi
        staged_files+=("$file")
    done < <(git diff --cached --name-only -z --diff-filter=ACM)

    if [ ${#staged_files[@]} -eq 0 ]; then
        print_success "No files to check"
        exit 0
    fi

    for file in "${staged_files[@]}"; do
        if [[ "$file" == "scripts/ci/secret-detection.sh" ]]; then
            continue
        fi
        if grep -qE "(accountaccount|testtest|password123)" "$file" 2>/dev/null; then
            print_error "Found hardcoded test password in $file - use placeholders like '<your-password>'"
        fi
        if grep -qE "graphistry-(dev|test|staging)\.(grph\.xyz|graphistry\.com)" "$file" 2>/dev/null; then
            print_error "Found internal dev server URL in $file - use hub.graphistry.com or localhost"
        fi
        if grep -qE "(GRAPHISTRY_)(USERNAME|PASSWORD|TOKEN|KEY|SECRET)\s*=\s*['\"]?[A-Za-z0-9]+" "$file" 2>/dev/null; then
            print_warning "Possible hardcoded credential in $file - use os.environ.get() with defaults"
        fi
    done

    if [ "$DETECT_SECRETS_AVAILABLE" -ne 1 ]; then
        print_warning "detect-secrets not found; install with: uv pip install detect-secrets (or pip install detect-secrets)"
        exit 0
    fi

    if [ ! -f "$BASELINE" ]; then
        print_warning "No $BASELINE found. Creating initial baseline..."
        "${PYTHON_CMD[@]}" "$SCRIPT_PATH" create-baseline --baseline "$BASELINE" --exclude-files "$SECRET_SCAN_EXCLUDE_REGEX" .
        print_success "Created $BASELINE - please review and commit"
        exit 0
    fi

    if ! "${PYTHON_CMD[@]}" "$SCRIPT_PATH" scan --baseline "$BASELINE" --exclude-files "$SECRET_SCAN_EXCLUDE_REGEX" "${staged_files[@]}"; then
        print_error "New secrets detected. Use clear placeholders like 'sk-XXXX' or '<your-password>'"
    fi

    print_success "No secrets detected"
else
    if [ "$DETECT_SECRETS_AVAILABLE" -ne 1 ]; then
        print_error "detect-secrets not found. Install with: pip install detect-secrets"
    fi

    if [ ! -f "$BASELINE" ]; then
        print_warning "No $BASELINE found. Creating initial baseline..."
        "${PYTHON_CMD[@]}" "$SCRIPT_PATH" create-baseline --baseline "$BASELINE" --exclude-files "$SECRET_SCAN_EXCLUDE_REGEX" .
        print_success "Created $BASELINE - please review and commit"
        exit 0
    fi

    echo "Running full secret detection scan..."

    "${PYTHON_CMD[@]}" "$SCRIPT_PATH" scan --baseline "$BASELINE" --exclude-files "$SECRET_SCAN_EXCLUDE_REGEX" . || {
        print_error "New secrets detected. Remove them or update $BASELINE"
    }

    "${PYTHON_CMD[@]}" "$SCRIPT_PATH" scan --baseline "$BASELINE" --only-verified --exclude-files "$SECRET_SCAN_EXCLUDE_REGEX" . || {
        print_error "High-confidence secrets detected. These must be removed"
    }

    print_success "Secret detection passed - no new secrets found"
fi
