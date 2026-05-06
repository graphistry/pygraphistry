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
if [[ "$WITH_TEST" != "0" ]]; then
    ./bin/test.sh $@
fi

echo "=== Building ==="
if [[ "$WITH_BUILD" != "0" ]]; then
    if [[ "$RAPIDS" == "1" ]]; then
        activate_rapids_env
    fi
    OUTPUT_DIR=/tmp ./bin/build.sh
fi
