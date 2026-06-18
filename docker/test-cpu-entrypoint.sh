#!/bin/bash
set -ex

# Run from project root
# Args get passed to pytest phase

activate_rapids_env() {
    if ! command -v conda >/dev/null 2>&1; then
        return 0
    fi

    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate base || true
    fi
}

if [[ "$RAPIDS" == "1" ]]; then
    activate_rapids_env
else
    source pygraphistry/bin/activate
fi

#echo "=== env ==="
#python --version
#python -m pip --version
#pip show pandas
#pip show numpy
#env

echo "=== Linting ==="
if [[ "$WITH_LINT" != "0" ]]; then
    ./bin/lint.sh
fi

echo "=== Type checking ==="
if [[ "$WITH_TYPECHECK" != "0" ]]; then
    pip show mypy
    ./bin/typecheck.sh
fi

echo "=== Testing ==="
if [[ "${WITH_COVERAGE_AUDIT:-${WITH_GFQL_COVERAGE_AUDIT:-0}}" != "0" ]]; then
    COVERAGE_BASELINE_ARGS=()
    COVERAGE_PROFILE="${COVERAGE_PROFILE:-gfql}"
    COVERAGE_ENGINE_LABEL="${COVERAGE_ENGINE_LABEL:-${GFQL_COVERAGE_ENGINE_LABEL:-rapids-cudf}}"
    COVERAGE_OUTPUT_DIR="${COVERAGE_OUTPUT_DIR:-${GFQL_COVERAGE_OUTPUT_DIR:-/tmp/gfql-coverage-audit}}"
    COVERAGE_BASELINE_FILE="${COVERAGE_BASELINE_FILE:-${GFQL_COVERAGE_BASELINE_FILE:-}}"
    COVERAGE_BASELINE_TOLERANCE="${COVERAGE_BASELINE_TOLERANCE:-${GFQL_COVERAGE_BASELINE_TOLERANCE:-}}"
    if [[ -n "${COVERAGE_BASELINE_FILE}" ]]; then
        COVERAGE_BASELINE_ARGS+=(--baseline-file "${COVERAGE_BASELINE_FILE}")
    fi
    if [[ -n "${COVERAGE_BASELINE_TOLERANCE}" ]]; then
        COVERAGE_BASELINE_ARGS+=(--baseline-tolerance "${COVERAGE_BASELINE_TOLERANCE}")
    fi
    python bin/coverage_audit.py \
        --profile "${COVERAGE_PROFILE}" \
        --engine-label "${COVERAGE_ENGINE_LABEL}" \
        --output-dir "${COVERAGE_OUTPUT_DIR}" \
        "${COVERAGE_BASELINE_ARGS[@]}" \
        -- "$@"
elif [[ "$WITH_TEST" != "0" ]]; then
    ./bin/test.sh $@
fi

echo "=== Building ==="
if [[ "$WITH_BUILD" != "0" ]]; then
    if [[ "$RAPIDS" == "1" ]]; then
        activate_rapids_env
    fi
    OUTPUT_DIR=/tmp ./bin/build.sh
fi
