#!/bin/bash
set -ex

# Fast sentinel gate: runs the minimal suite minus the heaviest test files.
# Target: <60s. Gates all downstream CI jobs.
#
# Excluded from this script (deferred to test-minimal-python-rest or test-gfql-core):
#   test_hyper_dask.py       — heavy dask.distributed / LocalCUDACluster setup
#   test_compute_chain.py    — large suite (~78 tests)
#   compute/test_chain_let.py — largest suite (~91 tests)
#   compute/test_hop.py       — 8x parametrize combinatorics (~54 tests)
#   test_plotter.py           — large suite with dask refs (~61 tests)
#   GFQL/core/ref tests       — dedicated test-gfql-core owner
#
# Run from project root
# Args get passed to pytest

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        PYTHON_BIN=python3
    fi
fi

"$PYTHON_BIN" -m pytest --version

"$PYTHON_BIN" -B -m pytest -vv \
    --ignore=plans \
    --ignore=test_env \
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
    --ignore=graphistry/tests/benchmarks/gfql \
    --ignore=graphistry/tests/compute/gfql \
    --ignore=graphistry/tests/compute/test_ast.py \
    --ignore=graphistry/tests/compute/test_chain.py \
    --ignore=graphistry/tests/compute/test_chain_concat.py \
    --ignore=graphistry/tests/compute/test_dataframe_primitives.py \
    --ignore=graphistry/tests/compute/test_gfql.py \
    --ignore=graphistry/tests/compute/test_gfql_call_validation.py \
    --ignore=graphistry/tests/compute/test_gfql_exceptions.py \
    --ignore=graphistry/tests/compute/test_gfql_hypergraph.py \
    --ignore=graphistry/tests/compute/test_gfql_validate_only.py \
    --ignore=graphistry/tests/compute/test_gfql_validation.py \
    --ignore=graphistry/tests/test_gfql_remote_metadata.py \
    --ignore=graphistry/tests/test_gfql_remote_persistence.py \
    --ignore=tests/gfql/ref \
    --ignore=graphistry/tests/test_hyper_dask.py \
    --ignore=graphistry/tests/test_compute_chain.py \
    --ignore=graphistry/tests/test_plotter.py \
    --ignore=graphistry/tests/compute/test_chain_let.py \
    --ignore=graphistry/tests/compute/test_hop.py \
    "$@"
