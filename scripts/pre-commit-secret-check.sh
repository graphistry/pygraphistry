#!/bin/bash
# pre-commit-secret-check.sh - Wrapper for pre-commit hook

set -euo pipefail

exec "$(dirname "$0")/ci/secret-detection.sh" --check-only
