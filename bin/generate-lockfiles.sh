#!/bin/bash
set -ex

# Generate hashed, cooldown-enforced lockfiles for all CI profiles.
#
# Usage:
#   ./bin/generate-lockfiles.sh              # 6-day cooldown (default)
#   COOLDOWN_DAYS=0 ./bin/generate-lockfiles.sh  # no cooldown (for urgent patches)
#
# Requires: uv >= 0.11 (for --exclude-newer with date arithmetic)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

COOLDOWN_DAYS="${COOLDOWN_DAYS:-6}"
# Default Python version for lockfiles. Jobs that run on older Pythons
# (3.8-3.11) need lockfiles generated at the lowest version they support.
# AI extras require 3.10+; basic test extras work down to 3.8.
PYTHON_MIN="${PYTHON_MIN:-3.8}"
PYTHON_DOCS="${PYTHON_DOCS:-3.10}"
PYTHON_AI="${PYTHON_AI:-3.10}"

if [ "$COOLDOWN_DAYS" -gt 0 ]; then
    # Portable date arithmetic (works on both GNU and BSD date)
    if date -v -1d >/dev/null 2>&1; then
        # BSD date (macOS)
        EXCLUDE_DATE=$(date -u -v "-${COOLDOWN_DAYS}d" +%Y-%m-%dT%H:%M:%SZ)
    else
        # GNU date (Linux)
        EXCLUDE_DATE=$(date -u -d "${COOLDOWN_DAYS} days ago" +%Y-%m-%dT%H:%M:%SZ)
    fi
    EXCLUDE_ARG="--exclude-newer ${EXCLUDE_DATE}"
    echo "Lockfile cooldown: ${COOLDOWN_DAYS} days (exclude packages uploaded after ${EXCLUDE_DATE})"
else
    EXCLUDE_ARG=""
    echo "Lockfile cooldown: DISABLED"
fi

BASE_ARGS="--generate-hashes ${EXCLUDE_ARG}"
COMMON_ARGS="${BASE_ARGS} --python-version ${PYTHON_MIN}"
DOCS_ARGS="${BASE_ARGS} --python-version ${PYTHON_DOCS}"
AI_ARGS="${BASE_ARGS} --python-version ${PYTHON_AI}"

echo "=== Generating lockfiles ==="

# test: minimal tests, gfql-core, lint-types
uv pip compile setup.py \
    --extra test \
    ${COMMON_ARGS} \
    -o requirements/test.lock

# test-core: core python tests (full extras)
uv pip compile setup.py \
    --extra test --extra build --extra bolt \
    --extra igraph --extra networkx --extra gremlin \
    --extra nodexl --extra jupyter \
    ${COMMON_ARGS} \
    -o requirements/test-core.lock

# test-compat: pandas compatibility tests
uv pip compile setup.py \
    --extra test --extra bolt --extra nodexl \
    ${COMMON_ARGS} \
    -o requirements/test-compat.lock

# test-graphviz: graphviz tests
uv pip compile setup.py \
    --extra test --extra pygraphviz \
    ${COMMON_ARGS} \
    -o requirements/test-graphviz.lock

# test-umap: UMAP/AI core tests (torch installed separately via pip)
# AI deps require Python 3.10+
uv pip compile setup.py \
    --extra test --extra testai --extra umap-learn \
    ${AI_ARGS} \
    -o requirements/test-umap.lock

# test-ai: full AI tests (torch installed separately via pip)
# AI deps require Python 3.10+
uv pip compile setup.py \
    --extra test --extra testai --extra ai \
    ${AI_ARGS} \
    -o requirements/test-ai.lock

# docs: RTD and docs CI (docutils requires 3.9+)
uv pip compile setup.py \
    --extra docs \
    ${DOCS_ARGS} \
    -o requirements/docs.lock

# build: package build tests
uv pip compile setup.py \
    --extra build \
    ${COMMON_ARGS} \
    -o requirements/build.lock

# tck: minimal for TCK runner
uv pip compile setup.py \
    ${COMMON_ARGS} \
    -o requirements/tck.lock

echo "=== Done. Generated lockfiles in requirements/ ==="
ls -la requirements/*.lock
