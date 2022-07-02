#!/bin/bash
set -ex

# Run from project root
# - Args get passed to pytest phase
# - Enable neo4j tests with WITH_NEO4J=1 (assumes ./test/db/neo4j ./launch.sh)
# Non-zero exit code on fail

python --version
python3 --version

python -m pytest --version

python -B -m pytest -vv $@
