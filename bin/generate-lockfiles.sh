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
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

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

COMMON_ARGS="--generate-hashes --python-version ${PYTHON_VERSION} ${EXCLUDE_ARG}"

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
uv pip compile setup.py \
    --extra test --extra testai --extra umap-learn \
    ${COMMON_ARGS} \
    -o requirements/test-umap.lock

# test-ai: full AI tests (torch installed separately via pip)
uv pip compile setup.py \
    --extra test --extra testai --extra ai \
    ${COMMON_ARGS} \
    -o requirements/test-ai.lock

# docs: RTD and docs CI
uv pip compile setup.py \
    --extra docs \
    ${COMMON_ARGS} \
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
