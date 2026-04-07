#!/bin/bash
set -ex

# Fast sentinel gate: runs the minimal suite minus the heaviest test files.
# Target: <60s. Gates all downstream CI jobs.
#
# Excluded from this script (deferred to test-minimal-python-rest pool):
#   test_hyper_dask.py       — heavy dask.distributed / LocalCUDACluster setup
#   test_compute_chain.py    — large suite (~78 tests)
#   compute/test_chain_let.py — largest suite (~91 tests)
#   compute/test_hop.py       — 8x parametrize combinatorics (~54 tests)
#   test_plotter.py           — large suite with dask refs (~61 tests)
#
# Run from project root
# Args get passed to pytest

python -m pytest --version

python -B -m pytest -vv \
    --ignore=graphistry/tests/test_bolt_util.py \
    --ignore=graphistry/tests/test_gremlin.py \
    --ignore=graphistry/tests/test_ipython.py \
    --ignore=graphistry/tests/test_nodexl.py \
    --ignore=graphistry/tests/test_tigergraph.py \
    --ignore=graphistry/tests/test_feature_utils.py \
    --ignore=graphistry/tests/test_umap_utils.py \
    --ignore=graphistry/tests/test_dgl_utils.py \
    --ignore=graphistry/tests/test_embed_utils.py \
    --ignore=graphistry/tests/test_kusto.py \
    --ignore=graphistry/tests/test_spanner.py \
    --ignore=graphistry/tests/compute/gfql/cypher/test_lowering.py \
    --ignore=graphistry/tests/compute/gfql/test_row_pipeline_ops.py \
    --ignore=graphistry/tests/compute/gfql/cypher/test_parser.py \
    --ignore=graphistry/tests/test_hyper_dask.py \
    --ignore=graphistry/tests/test_compute_chain.py \
    --ignore=graphistry/tests/test_plotter.py \
    --ignore=graphistry/tests/compute/test_chain_let.py \
    --ignore=graphistry/tests/compute/test_hop.py \
    "$@"
