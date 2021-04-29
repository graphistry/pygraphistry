#!/bin/bash
set -ex

# Run from project root
# Args get passed to pytest phase

MAYBE_RAPIDS="echo 'no rapids'"
if [[ "$RAPIDS" == "1" ]]; then
    source activate rapids
    MAYBE_RAPIDS="source activate rapids"
else
    source pygraphistry/bin/activate
fi

echo "=== env ==="
python --version
python -m pip --version
env

echo "=== Linting ==="
if [[ "$WITH_LINT" != "0" ]]; then
    ./bin/lint.sh
fi

echo "=== Type checking ==="
if [[ "$WITH_TYPECHECK" != "0" ]]; then
    ./bin/typecheck.sh
fi

echo "=== Testing ==="
./bin/test.sh $@

echo "=== Building ==="
if [[ "$WITH_BUILD" != "0" ]]; then
    $MAYBE_RAPIDS && OUTPUT_DIR=/tmp ./bin/build.sh
fi