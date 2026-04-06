#!/bin/bash
set -ex

# Generate hashed, cooldown-enforced lockfiles for all CI profiles × Python versions.
#
# Usage:
#   ./bin/generate-lockfiles.sh                    # 6-day cooldown (default)
#   COOLDOWN_DAYS=0 ./bin/generate-lockfiles.sh    # no cooldown (urgent patches)
#   PROFILES=test VERSIONS=3.12 ./bin/generate-lockfiles.sh  # single combo
#
# Requires: uv >= 0.11

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

COOLDOWN_DAYS="${COOLDOWN_DAYS:-6}"
OUTPUT_DIR="${OUTPUT_DIR:-requirements}"
mkdir -p "$OUTPUT_DIR"

# Python versions matching the CI matrix
ALL_VERSIONS=(3.8 3.9 3.10 3.11 3.12 3.13 3.14)
VERSIONS=(${VERSIONS:-${ALL_VERSIONS[@]}})

# Profiles and their extras. Format: "name:extras:min_python:extra_flags"
# min_python is the lowest Python version that supports the profile's deps.
# extra_flags are additional uv pip compile flags (e.g., --no-emit-package torch).
PROFILE_DEFS=(
    "test:test:3.8:"
    "test-core:test,build,bolt,igraph,networkx,gremlin,nodexl,jupyter:3.8:"
    "test-compat:test,bolt,nodexl:3.8:"
    "test-graphviz:test,pygraphviz:3.8:"
    "test-umap:test,testai,umap-learn:3.9:--no-emit-package torch"
    "test-ai:test,testai,ai:3.9:--no-emit-package torch"
    "docs:docs:3.10:"
    "build:build:3.8:"
    "tck:test:3.8:"
)
PROFILES=(${PROFILES:-$(printf '%s\n' "${PROFILE_DEFS[@]}" | cut -d: -f1)})

# Cooldown date
if [ "$COOLDOWN_DAYS" -gt 0 ]; then
    if date -v -1d >/dev/null 2>&1; then
        EXCLUDE_DATE=$(date -u -v "-${COOLDOWN_DAYS}d" +%Y-%m-%dT%H:%M:%SZ)
    else
        EXCLUDE_DATE=$(date -u -d "${COOLDOWN_DAYS} days ago" +%Y-%m-%dT%H:%M:%SZ)
    fi
    EXCLUDE_ARG="--exclude-newer ${EXCLUDE_DATE}"
    echo "Cooldown: ${COOLDOWN_DAYS}d (exclude after ${EXCLUDE_DATE})"
else
    EXCLUDE_ARG=""
    echo "Cooldown: DISABLED"
fi

echo "Versions: ${VERSIONS[*]}"
echo "Profiles: ${PROFILES[*]}"
echo "=== Generating lockfiles ==="

for profile_def in "${PROFILE_DEFS[@]}"; do
    IFS=: read -r name extras min_python extra_flags <<< "$profile_def"

    # Skip if not in requested profiles
    if ! printf '%s\n' "${PROFILES[@]}" | grep -qx "$name"; then
        continue
    fi

    # Build extras arg
    EXTRAS_ARG=""
    if [ -n "$extras" ]; then
        IFS=',' read -ra extra_list <<< "$extras"
        for extra in "${extra_list[@]}"; do
            EXTRAS_ARG="$EXTRAS_ARG --extra $extra"
        done
    fi

    for ver in "${VERSIONS[@]}"; do
        # Skip versions below the profile's minimum
        if python3 -c "import sys; sys.exit(0 if tuple(int(x) for x in '$ver'.split('.')) >= tuple(int(x) for x in '$min_python'.split('.')) else 1)" 2>/dev/null; then
            echo "--- ${name}-py${ver}.lock ---"
            uv pip compile setup.py \
                $EXTRAS_ARG \
                --python-version "$ver" \
                --generate-hashes \
                $EXCLUDE_ARG \
                $extra_flags \
                -o "${OUTPUT_DIR}/${name}-py${ver}.lock"
        else
            echo "--- SKIP ${name}-py${ver} (requires >=${min_python}) ---"
        fi
    done
done

echo "=== Done ==="
ls -la "${OUTPUT_DIR}"/*.lock 2>/dev/null | wc -l
echo "lockfiles generated"
