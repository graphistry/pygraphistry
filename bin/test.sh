#!/bin/bash
set -ex

# Run from project root
# - Args get passed to pytest phase
# - Enable neo4j tests with WITH_NEO4J=1 (assumes ./test/db/neo4j ./launch.sh)
# Non-zero exit code on fail

MAYBE_RAPIDS="echo 'no rapids'"
if [[ "$RAPIDS" == "1" ]]; then
    source activate rapids
    MAYBE_RAPIDS="source activate rapids"
else
    source /opt/pygraphistry/pygraphistry/bin/activate
fi

python -m pytest --version

python -B -m pytest -vv $@
