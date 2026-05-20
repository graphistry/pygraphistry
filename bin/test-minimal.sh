#!/bin/bash
set -ex

# Run from project root
# - Args get passed to pytest phase
# Non-zero exit code on fail

# Assume minimal env (pandas); no extras (neo4j, gremlin, ...)

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
    --ignore=graphistry/tests/compute/test_chain_let.py \
    --ignore=graphistry/tests/compute/test_dataframe_primitives.py \
    --ignore=graphistry/tests/compute/test_gfql.py \
    --ignore=graphistry/tests/compute/test_gfql_call_validation.py \
    --ignore=graphistry/tests/compute/test_gfql_exceptions.py \
    --ignore=graphistry/tests/compute/test_gfql_hypergraph.py \
    --ignore=graphistry/tests/compute/test_gfql_validate_only.py \
    --ignore=graphistry/tests/compute/test_gfql_validation.py \
    --ignore=graphistry/tests/compute/test_hop.py \
    --ignore=graphistry/tests/test_gfql_remote_metadata.py \
    --ignore=graphistry/tests/test_gfql_remote_persistence.py \
    --ignore=tests/gfql/ref \
    "$@"
