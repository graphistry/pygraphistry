#!/bin/bash
set -ex

# Run from project root
# - Extra args are passed through to the pytest phase
# - Set POLARS_COV=1 to collect coverage over graphistry/compute; the coverage
#   data file location is taken from $COVERAGE_FILE (as the CI py3.12 lane sets it)
# - Set POLARS_XDIST to pick the pytest-xdist worker spec (default `auto`); `0` forces
#   the serial path, which is what an A/B of this lane compares against
# - Non-zero exit code on fail

# Assume [polars,test] installed

python -m pytest --version

# Single source of truth for the polars test file list (CI reuses this script).
#
# COMPLETENESS IS ENFORCED, NOT REMEMBERED: polars is installed in NO other CI lane, so a
# module gated by `pytest.importorskip("polars")` that is missing from this array runs
# NOWHERE — it is collection-skipped everywhere else and silently reports green.
# `graphistry/tests/compute/gfql/test_polars_lane_completeness.py` parses this array and
# fails if any module-level polars-gated test file is absent from it (or listed but gone).
POLARS_TEST_FILES=(
    graphistry/tests/compute/test_polars.py
    # cache-coverage lock: its static scans run everywhere, but the functional pin for the
    # polars single-alias lowering memo can only execute where polars is installed
    graphistry/tests/compute/gfql/test_clear_caches_covers_every_cache.py
    graphistry/tests/compute/gfql/test_engine_polars_hop.py
    graphistry/tests/compute/gfql/test_engine_polars_chain.py
    graphistry/tests/compute/gfql/test_engine_polars_row_pipeline.py
    graphistry/tests/compute/gfql/test_engine_polars_binding_rows.py
    # engine-parametrized (pandas/polars/cudf/polars-gpu); the pandas params also run in
    # test-gfql-core, but only this lane has polars installed
    graphistry/tests/compute/gfql/test_varlen_bounded_engine_parity_1787.py
    graphistry/tests/compute/gfql/test_engine_polars_with_match_reentry.py
    # engine-parametrized: its pandas params run in test-gfql-core, but the polars params
    # only ever run here (the file has no module-level importorskip, so nothing else flags it)
    graphistry/tests/compute/gfql/test_exec_context_scoping.py
    graphistry/tests/compute/gfql/test_engine_polars_cypher_conformance.py
    # aggregate x dtype type contract: the polars params of this file are the ONLY lane where the
    # native polars aggregate guard and the raw-polars-exception wrap are exercised
    graphistry/tests/compute/gfql/test_aggregate_type_contract.py
    graphistry/tests/compute/gfql/test_engine_polars_conformance_matrix.py
    graphistry/tests/compute/gfql/test_polars_string_predicate_nonstring.py
    graphistry/tests/compute/gfql/cypher/test_order_by_null_placement.py
    graphistry/tests/compute/gfql/test_conformance_ledger.py
    graphistry/tests/compute/gfql/test_polars_nan_clean.py
    graphistry/tests/compute/gfql/test_optional_match_polars_frames.py
    graphistry/tests/compute/gfql/test_optional_match_semantics.py
    graphistry/tests/compute/gfql/test_hop_boundary_matrix.py
    graphistry/tests/compute/gfql/test_hop_semantics_pins.py
    graphistry/tests/compute/gfql/test_polars_rows_entity_groupby.py
    graphistry/tests/compute/gfql/test_seeded_typed_hop_fastpath.py
    graphistry/tests/compute/gfql/test_residual_polars_native.py
    graphistry/tests/compute/gfql/index/test_auto_engine_agreement.py
    graphistry/tests/compute/gfql/index/test_degree_consult.py
    graphistry/tests/compute/gfql/test_single_alias_cache_key.py
    graphistry/tests/compute/gfql/test_semi_join_key_frame.py
    graphistry/tests/compute/gfql/test_fast_path_engagement.py
    graphistry/tests/compute/gfql/test_known_cross_engine_divergences.py
    graphistry/tests/compute/gfql/test_endpoint_closure_matrix.py
    graphistry/tests/compute/gfql/test_gfql_unified_routing_contracts.py
    graphistry/tests/compute/gfql/test_hop_kernel_contracts.py
    graphistry/tests/compute/gfql/test_polars_dtype_classifier_contracts.py
    graphistry/tests/compute/gfql/cypher/test_grouped_aggregate_fused_polars.py
    graphistry/tests/compute/gfql/cypher/test_grouped_aggregate_lowcard_count.py
    # module-level `importorskip("polars")` files that previously ran in no lane at all
    graphistry/tests/compute/gfql/test_engine_polars_narrow_combine.py
    graphistry/tests/compute/gfql/test_engine_polars_semi_key_dedup.py
    graphistry/tests/compute/gfql/test_engine_polars_call_modality.py
    graphistry/tests/compute/gfql/test_engine_polars_gpu.py
    graphistry/tests/compute/gfql/test_rows_table_named_middle.py
    graphistry/tests/compute/gfql/test_viz_pipeline_conformance.py
    # polars-parametrized cases inside otherwise-pandas modules: these files DO run in the
    # pandas lanes, but their polars/polars-gpu parameters are skipped there for want of the
    # wheel, so the polars lane is the only place those parameters can execute
    graphistry/tests/compute/gfql/test_const_fold_engine_parity.py
    graphistry/tests/compute/gfql/index/test_indexed_bindings.py
    graphistry/tests/compute/gfql/test_reentry_caller_graph_immutability.py
    graphistry/tests/compute/gfql/test_rewrite_param_discard.py
    graphistry/tests/compute/test_engine_coercion.py
    # index tests exercise the seeded-index hook in the polars hop entry (hop.py) — without
    # them the hook dominates the now-thin file and trips its per-file coverage floor
    graphistry/tests/compute/gfql/index/test_index.py
    # engine-agnostic frame/series primitives (graphistry/Engine.py) — the polars branches of
    # these dispatch helpers are only measured when this lane covers graphistry (see cov widen below)
    graphistry/tests/test_engine_frame_helpers.py
    graphistry/tests/test_public_apis_do_not_mutate_inputs.py
)

# PARALLELISM. The py3.12 cell of this lane is the coverage cell and has repeatedly run out
# of its CI budget; xdist is the lever that does not require a workflow edit (pytest-xdist is
# already in the [test] extra, and test-gfql-core already runs `-n auto` under --cov, so
# coverage+xdist is an established combination in this repo).
#   * worker spec `auto` = os.cpu_count(): 4 on a GitHub-hosted ubuntu-latest runner, and it
#     scales DOWN on a 2-vCPU runner where a fixed `-n 4` could be slower than serial.
#   * --maxprocesses caps the count so a 24-core dev box does not fan out 24 polars processes
#     that then oversubscribe polars' own thread pool.
#   * --dist load (xdist's default) balances per test. `loadfile` was measured too: it is
#     bounded by the single largest module and only reaches 1.4x where `load` reaches 3.2x.
#     No test in this lane depends on execution order or on cross-test module state, and the
#     pass/skip node-id sets were compared serial-vs-parallel and are identical; POLARS_XDIST_DIST
#     is the escape hatch if a future order-dependent test needs `loadfile`/`loadscope`.
XDIST_ARGS=()
if [ "${POLARS_XDIST:-auto}" != "0" ]; then
    XDIST_ARGS=(
        -n "${POLARS_XDIST:-auto}"
        --maxprocesses "${POLARS_XDIST_MAX:-4}"
        --dist "${POLARS_XDIST_DIST:-load}"
    )
fi

COV_ARGS=()
if [ -n "${POLARS_COV:-}" ]; then
    COV_ARGS=(--cov=graphistry --cov-report=)
fi

python -B -m pytest -vv "${XDIST_ARGS[@]}" "${COV_ARGS[@]}" "${POLARS_TEST_FILES[@]}" "$@"

# cypher-lowering polars-parametrized cases (round ties, lower/upper, =~, numeric fns);
# appended into the same coverage data file when POLARS_COV=1 (CI audit reads it).
# Left SERIAL on purpose: it is one module and ~8s of the lane, so worker startup would eat
# the gain. Appending into the data file the xdist phase produced is verified — the merged
# result is line-for-line identical to running the whole script in one go.
COV_APPEND_ARGS=()
if [ -n "${POLARS_COV:-}" ]; then
    COV_APPEND_ARGS=(--cov=graphistry --cov-report= --cov-append)
fi
python -B -m pytest -vv "${COV_APPEND_ARGS[@]}" \
    graphistry/tests/compute/gfql/cypher/test_lowering.py -k polars
