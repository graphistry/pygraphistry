#!/bin/bash
set -ex

# Run from project root
# - Args get passed to pytest phase
# Non-zero exit code on fail

# Assume [polars,test] installed

python -m pytest --version

python -B -m pytest -vv \
    graphistry/tests/compute/test_polars.py \
    graphistry/tests/compute/gfql/test_engine_polars_hop.py \
    graphistry/tests/compute/gfql/test_engine_polars_chain.py \
    graphistry/tests/compute/gfql/test_engine_polars_row_pipeline.py \
    graphistry/tests/compute/gfql/test_engine_polars_cypher_conformance.py
