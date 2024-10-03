#!/bin/bash
set -ex

# Run from project root
# - Args get passed to pytest phase
# Non-zero exit code on fail

# Assume [pygraphviz,test], apt-get install graphviz graphviz-dev

python -m pytest --version

python -B -m pytest -vv \
    graphistry/tests/plugins/test_graphviz.py
