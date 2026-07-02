#!/bin/bash
set -ex

# Run from project root
# - Extra args are passed through to the pytest phase
# - Set POLARS_COV=1 to collect coverage over graphistry/compute; the coverage
#   data file location is taken from $COVERAGE_FILE (as the CI py3.12 lane sets it)
# - Non-zero exit code on fail

# Assume [polars,test] installed

python -m pytest --version

# Single source of truth for the polars test file list (CI reuses this script).
POLARS_TEST_FILES=(
    graphistry/tests/compute/test_polars.py
    graphistry/tests/compute/gfql/test_engine_polars_hop.py
    graphistry/tests/compute/gfql/test_engine_polars_chain.py
    graphistry/tests/compute/gfql/test_engine_polars_row_pipeline.py
    graphistry/tests/compute/gfql/test_engine_polars_cypher_conformance.py
    graphistry/tests/compute/gfql/test_engine_polars_conformance_matrix.py
    graphistry/tests/compute/gfql/test_conformance_ledger.py
)

COV_ARGS=()
if [ -n "${POLARS_COV:-}" ]; then
    COV_ARGS=(--cov=graphistry/compute --cov-report=)
fi

python -B -m pytest -vv "${COV_ARGS[@]}" "${POLARS_TEST_FILES[@]}" "$@"

# cypher-lowering polars-parametrized cases (round ties, lower/upper, =~, numeric fns);
# appended into the same coverage data file when POLARS_COV=1 (CI audit reads it)
COV_APPEND_ARGS=()
if [ -n "${POLARS_COV:-}" ]; then
    COV_APPEND_ARGS=(--cov=graphistry/compute --cov-report= --cov-append)
fi
python -B -m pytest -vv "${COV_APPEND_ARGS[@]}" \
    graphistry/tests/compute/gfql/cypher/test_lowering.py -k polars
