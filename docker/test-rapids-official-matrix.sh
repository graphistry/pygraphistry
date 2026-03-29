#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

VERSIONS=${RAPIDS_VERSIONS:-"25.02 26.02"}
PROFILES=${PROFILES:-"basic gfql ai"}
WITH_GPU=${WITH_GPU:-0}
WITH_IMAGE_BUILD=${WITH_IMAGE_BUILD:-1}
WITH_LINT=${WITH_LINT:-0}
WITH_TYPECHECK=${WITH_TYPECHECK:-0}
WITH_TEST=${WITH_TEST:-1}
WITH_BUILD=${WITH_BUILD:-0}
LOG_LEVEL=${LOG_LEVEL:-DEBUG}
MATRIX_CELLS=${MATRIX_CELLS:-}

cells=()
if [[ -n "${MATRIX_CELLS}" ]]; then
    for cell in ${MATRIX_CELLS}; do
        cells+=("${cell}")
    done
else
    for version in ${VERSIONS}; do
        for profile in ${PROFILES}; do
            cells+=("${version}:${profile}")
        done
    done
fi

if [[ ${#cells[@]} -eq 0 ]]; then
    echo "No matrix cells selected" >&2
    exit 1
fi

results=()
failures=0

echo "CONFIG"
echo "VERSIONS=${VERSIONS}"
echo "PROFILES=${PROFILES}"
echo "WITH_GPU=${WITH_GPU}"
echo "WITH_IMAGE_BUILD=${WITH_IMAGE_BUILD}"
echo "WITH_LINT=${WITH_LINT}"
echo "WITH_TYPECHECK=${WITH_TYPECHECK}"
echo "WITH_TEST=${WITH_TEST}"
echo "WITH_BUILD=${WITH_BUILD}"
echo "LOG_LEVEL=${LOG_LEVEL}"
echo "MATRIX_CELLS=${MATRIX_CELLS:-<default>}"

for cell in "${cells[@]}"; do
    version="${cell%%:*}"
    profile="${cell##*:}"

    echo
    echo "=== CELL ${version}/${profile} ==="

    if RAPIDS_VERSION="${version}" \
        PROFILE="${profile}" \
        WITH_GPU="${WITH_GPU}" \
        WITH_IMAGE_BUILD="${WITH_IMAGE_BUILD}" \
        WITH_LINT="${WITH_LINT}" \
        WITH_TYPECHECK="${WITH_TYPECHECK}" \
        WITH_TEST="${WITH_TEST}" \
        WITH_BUILD="${WITH_BUILD}" \
        LOG_LEVEL="${LOG_LEVEL}" \
        "${SCRIPT_DIR}/test-rapids-official-local.sh"; then
        results+=("${version}/${profile}:PASS")
    else
        results+=("${version}/${profile}:FAIL")
        failures=$((failures + 1))
    fi
done

echo
echo "SUMMARY"
for result in "${results[@]}"; do
    cell="${result%%:*}"
    status="${result##*:}"
    printf '%-18s %s\n' "${cell}" "${status}"
done

if [[ "${failures}" -ne 0 ]]; then
    echo
    echo "Matrix failed: ${failures} cell(s) failed" >&2
    exit 1
fi

echo
echo "Matrix passed"
