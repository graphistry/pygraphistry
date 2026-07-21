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
    graphistry/tests/compute/gfql/test_engine_polars_binding_rows.py
    graphistry/tests/compute/gfql/test_engine_polars_with_match_reentry.py
    graphistry/tests/compute/gfql/test_engine_polars_cypher_conformance.py
    graphistry/tests/compute/gfql/test_engine_polars_conformance_matrix.py
    graphistry/tests/compute/gfql/test_polars_string_predicate_nonstring.py
    graphistry/tests/compute/gfql/cypher/test_order_by_null_placement.py
    graphistry/tests/compute/gfql/test_conformance_ledger.py
    graphistry/tests/compute/gfql/test_polars_nan_clean.py
    graphistry/tests/compute/gfql/test_optional_match_polars_frames.py
    graphistry/tests/compute/gfql/test_polars_rows_entity_groupby.py
    graphistry/tests/compute/gfql/test_seeded_typed_hop_fastpath.py
    # index tests exercise the seeded-index hook in the polars hop entry (hop.py) — without
    # them the hook dominates the now-thin file and trips its per-file coverage floor
    graphistry/tests/compute/gfql/index/test_index.py
    # engine-agnostic frame/series primitives (graphistry/Engine.py) — the polars branches of
    # these dispatch helpers are only measured when this lane covers graphistry (see cov widen below)
    graphistry/tests/test_engine_frame_helpers.py
)

COV_ARGS=()
if [ -n "${POLARS_COV:-}" ]; then
    COV_ARGS=(--cov=graphistry --cov-report=)
fi

python -B -m pytest -vv "${COV_ARGS[@]}" "${POLARS_TEST_FILES[@]}" "$@"

# cypher-lowering polars-parametrized cases (round ties, lower/upper, =~, numeric fns);
# appended into the same coverage data file when POLARS_COV=1 (CI audit reads it)
COV_APPEND_ARGS=()
if [ -n "${POLARS_COV:-}" ]; then
    COV_APPEND_ARGS=(--cov=graphistry --cov-report= --cov-append)
fi
python -B -m pytest -vv "${COV_APPEND_ARGS[@]}" \
    graphistry/tests/compute/gfql/cypher/test_lowering.py -k polars
