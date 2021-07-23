#!/bin/bash
set -ex

# Run from project root
# - Args get passed to pytest phase
# Non-zero exit code on fail

# Assume minimal env (pandas); no extras (neo4j, gremlin, ...)

python -m pytest --version

python -B -m pytest -vv \
    --ignore=graphistry/tests/test_bolt_util.py \
    --ignore=graphistry/tests/test_gremlin.py \
    --ignore=graphistry/tests/test_ipython.py \
    --ignore=graphistry/tests/test_nodexl.py \
    --ignore=graphistry/tests/test_tigergraph
