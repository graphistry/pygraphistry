#!/bin/bash
set -ex

# Run from project root
# - Args get passed to pytest phase
# Non-zero exit code on fail

# Assume [umap-learn,test]

python -m pytest --version

python -B -m pytest -vv \
    graphistry/tests/test_umap_utils.py