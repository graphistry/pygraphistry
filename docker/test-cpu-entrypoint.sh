#!/bin/bash
set -ex

# Run from project root
# Args get passed to pytest phase

echo "=== env ==="
python --version
python -m pip --version

echo "=== Linting ==="
./bin/lint.sh

echo "=== Type checking ==="
./bin/typecheck.sh

echo "=== Testing ==="
./bin/test.sh $@

echo "=== Building ==="
OUTPUT_DIR=/tmp ./bin/build.sh