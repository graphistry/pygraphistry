#!/bin/bash
set -ex

# Run from project root
# Args get passed to pytest phase

MAYBE_RAPIDS="echo 'no rapids'"
if [[ "$RAPIDS" == "1" ]]; then
    source activate rapids
    MAYBE_RAPIDS="source activate rapids"
fi

echo "=== env ==="
python --version
python -m pip --version
env

echo "=== Linting ==="
./bin/lint.sh

echo "=== Type checking ==="
./bin/typecheck.sh

echo "=== Testing ==="
./bin/test.sh $@

echo "=== Building ==="
$MAYBE_RAPIDS && OUTPUT_DIR=/tmp ./bin/build.sh