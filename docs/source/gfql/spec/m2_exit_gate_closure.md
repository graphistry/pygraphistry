(gfql-spec-m2-exit-gate-closure)=

# M2 Exit-Gate Closure Bundle (PR6 / #1137)

Last updated: 2026-04-19 PT

This document is the in-repo closure artifact for:
- Issue #1137: https://github.com/graphistry/pygraphistry/issues/1137
- Metaissue #1139: https://github.com/graphistry/pygraphistry/issues/1139
- PR6: https://github.com/graphistry/pygraphistry/pull/1151

## Context

Issue text for M2 references `plans/compiler-refactor/milestone-M2.md` and `plans/compiler-refactor/research/rf4-unsupported-scope-table.md`. Those paths are not present in this repository tree. This page provides the equivalent, linkable closure bundle for this repo.

## M2 Exit Checklist

| Exit criterion | Status | Evidence |
|---|---|---|
| All M2 child lanes closed with evidence | Done | #1126, #1135, #1127, #1136, #1128 are closed; lane table below |
| `LogicalPlanner` skeleton + op_id contract | Done | PR #1129: https://github.com/graphistry/pygraphistry/pull/1129 |
| `QueryGraph` extraction + cycle-policy locks | Done | PR #1142: https://github.com/graphistry/pygraphistry/pull/1142 |
| LogicalPlan verifier invariants + cycle-safe traversal | Done | PR #1131: https://github.com/graphistry/pygraphistry/pull/1131 |
| Lowering route through LogicalPlan + escape-hatch reduction | Done | PR #1146: https://github.com/graphistry/pygraphistry/pull/1146 |
| CALL/GRAPH compatibility lane through logical route | Done | PR #1147: https://github.com/graphistry/pygraphistry/pull/1147 |
| Unknown-alias rejection evidence across binder/lowering/planner/verifier flow | Done | Local conformance receipts below (`test_binder.py`, `test_ir_verifier.py`, `test_lowering.py`) |
| No unresolved high-severity planner/verifier contract findings | Done | All targeted conformance tests listed below are green |

## Lane Evidence Matrix (PR1-PR5)

| Lane | Issue | PR | Merged (UTC) | Merge commit |
|---|---|---|---|---|
| PR1 LogicalPlanner skeleton | #1126 | #1129 | 2026-04-18T06:41:47Z | `06a085473b38d0325734c057a4b740e34b32f97a` |
| PR2 QueryGraph extraction | #1135 | #1142 | 2026-04-19T19:27:13Z | `06e485b1f0e208b5b4d1f8bdcbb34bb29dd1416b` |
| PR3 Verifier invariants | #1127 | #1131 | 2026-04-18T05:37:09Z | `0f0428c2026db3f2ff150f51b94d4f6ab71fdc9e` |
| PR4 Lowering route + escape-hatch reduction | #1136 | #1146 | 2026-04-19T04:08:34Z | `f6b26bd2e022ab5842a5f6ecc32605b9e3075bf9` |
| PR5 CALL/GRAPH compatibility | #1128 | #1147 | 2026-04-19T18:26:51Z | `4782b70262ae1e7a4ae16a085c37c7c11831eb4d` |

## RF4 M2-Scope Construct Mapping (Implemented vs Deferred)

| Construct / contract | M2 status | Evidence | Deferred tracking |
|---|---|---|---|
| Logical operator planner spine + deterministic op_id assignment | Implemented | PR #1129 | n/a |
| QueryGraph extraction (`components`, `boundary_aliases`, `optional_arms`) | Implemented | PR #1142 | n/a |
| Cycle-policy locks (same-MATCH alias rewrite; unsupported cross-WITH alias reuse) | Implemented | PR #1142 | n/a |
| Verifier invariants (dangling refs, predicate scope, schema type checks, optional-arm nullability, op_id uniqueness, cycle-safe traversal) | Implemented | PR #1131 | n/a |
| Lowering metadata route (`planned` vs `deferred`) for M2-covered shapes | Implemented | PR #1146 | n/a |
| CALL row-returning and graph-preserving `ProcedureCall` route compatibility | Implemented | PR #1147 | n/a |
| Full logical-route coverage for currently deferred reentry / row-sequence flow classes | Deferred to M3 | Explicit defer reasons in lowering route tests (`test_lowering.py`) | #1154 |
| Recursive deep `ListType.element_type` schema validation in verifier | Deferred follow-up | Current verifier suite contains one documented xfail | #1153 |
| Physical planning / executor redesign | Deferred to M3/M4 | Explicit non-goals in #1139 | #1139 |

## Conformance Receipts (Local)

Run date: 2026-04-19 PT

```bash
PYTHONPATH=. uv run --no-project --with pytest python -m pytest -q graphistry/tests/compute/gfql/test_logical_planner.py
PYTHONPATH=. uv run --no-project --with pytest python -m pytest -q graphistry/tests/compute/gfql/test_ir_query_graph.py
PYTHONPATH=. uv run --no-project --with pytest python -m pytest -q graphistry/tests/compute/gfql/test_ir_verifier.py
PYTHONPATH=. uv run --no-project --with pytest python -m pytest -q graphistry/tests/compute/gfql/cypher/test_cycle_policy.py
PYTHONPATH=. uv run --no-project --with pytest python -m pytest -q graphistry/tests/compute/gfql/cypher/test_binder.py -k "unresolved_name_failure_after_with_scope_reset"
PYTHONPATH=. uv run --no-project --with pytest python -m pytest -q graphistry/tests/compute/gfql/cypher/test_lowering.py -k "logical_plan_route_for_covered_shape or logical_plan_defer_reason_for_optional_shape or logical_plan_route_for_call_shape or rejects_out_of_scope_order_by_after_multiple_with_stages"
```

Observed results:
- `test_logical_planner.py`: `20 passed`
- `test_ir_query_graph.py`: `40 passed`
- `test_ir_verifier.py`: `69 passed, 1 xfailed` (documented deep `ListType` follow-up)
- `test_cycle_policy.py`: `9 passed`
- Binder alias-rejection slice: `1 passed, 33 deselected`
- Lowering route/scope slice: `4 passed, 753 deselected`

## Merged-PR CI Receipts

Merged M2 lane PR check-rollup summaries (all completed, no failures):

| PR | checks_total | success | failure | cancelled | skipped | pending |
|---|---:|---:|---:|---:|---:|---:|
| #1129 | 118 | 113 | 0 | 0 | 5 | 0 |
| #1131 | 118 | 117 | 0 | 0 | 1 | 0 |
| #1142 | 119 | 118 | 0 | 0 | 1 | 0 |
| #1146 | 119 | 118 | 0 | 0 | 1 | 0 |
| #1147 | 120 | 119 | 0 | 0 | 1 | 0 |

## PR6 Completion Checklist

- [x] M2 exit checklist with concrete links
- [x] RF4 implemented-vs-deferred mapping with tracking issues for deferred items
- [x] End-to-end conformance receipts (planner/querygraph/verifier/lowering/scope rejection)
- [ ] Merge PR6 (#1151)
- [ ] Close #1137
- [ ] Close #1139
