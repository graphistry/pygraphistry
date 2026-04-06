# Issue #1047 — Multi-row WITH prefix for scalar reentry
**Branch:** feat/issue-1047-multi-row-scalar-prefix
**PR:** TBD
**Issue:** #1047
**Created:** 2026-04-06

## Objective
Allow `MATCH ... WITH scalar1, scalar2 ... MATCH ...` where the WITH prefix
produces N rows (not just 1). This unblocks IC6 conformance: the UNWIND fanout
before the final tag-cooccurrence MATCH produces 4+ prefix rows.

## Root Cause (confirmed)
`gfql_unified.py` `_compiled_query_scalar_reentry_state()` line 908:
`if prefix_row_count != 1: raise ...`

Current design: broadcasts one row's scalar values as hidden columns across all
base-graph nodes, then runs the suffix MATCH once.

## Fix Strategy
For N-row prefix: iterate over each prefix row, broadcast that row's scalars
as hidden columns, run the suffix MATCH, collect the N result Plottables, then
concat `_nodes` and `_edges` from all runs via `safe_concat`.

This is a clean loop — no change to compilation or lowering, only to
`_compiled_query_scalar_reentry_state()` and the caller
`_execute_compiled_query_with_reentry()`.

### Key detail
The caller `_execute_compiled_query_with_reentry` currently calls
`_compiled_query_scalar_reentry_state` once and gets back `(compiled_base_graph,
start_nodes)`, then calls `_execute_compiled_query` once. For multi-row we need
to move the loop up one level: detect multi-row in the execution path and fan
out. Best approach: extract a helper that runs one suffix iteration, then loop
in the caller.

## Files to change
1. `graphistry/compute/gfql_unified.py`:
   - Remove `!= 1` guard in `_compiled_query_scalar_reentry_state()`
   - Add multi-row fan-out loop in `_execute_compiled_query_with_reentry()`
     for the `scalar_reentry_alias is not None` branch
2. Tests (RED → GREEN):
   - `graphistry/tests/compute/gfql/cypher/test_lowering.py`

## Phases

### Phase A: RED — write failing tests ✅ TODO
### Phase B: GREEN — multi-row fan-out ✅ TODO
### Phase C: dgx-spark GPU validation ✅ TODO
### Phase D: CHANGELOG + PR, CI green, squash-merge ✅ TODO
