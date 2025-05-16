#!/bin/bash
set -ex

# Run from project root
# - Args get passed to pytest phase
# - Enable neo4j tests with WITH_NEO4J=1 (assumes ./test/db/neo4j ./launch.sh)
# Non-zero exit code on fail

python --version
python3 --version

python -m pytest --version

# Set up base pytest arguments
PYTEST_ARGS="-vv"

# Add parallel testing by default when no args are provided
if [ $# -eq 0 ]; then
    PYTEST_ARGS="$PYTEST_ARGS -n auto"
fi

# Run pytest with the computed arguments plus any user-provided args
python -B -m pytest $PYTEST_ARGS $@
