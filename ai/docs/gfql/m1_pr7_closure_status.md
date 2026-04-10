# M1 PR-7 Closure Status

This document tracks closure readiness for issue #1115 against the authoritative gate:
- https://github.com/graphistry/pygraphistry/issues/1115#issuecomment-4212386040

Scope is evidence and closure readiness. Binder/lowering implementation work landed via PR-4..PR-6.

## Dependency State (Wave-2 DAG)
- PR-4 issue: #1114 (CLOSED)
- PR-5 issue: #1117 (CLOSED)
- PR-6 issue: #1116 (CLOSED)

All dependency issues are closed; PR-7 is now the closure-evidence bundle for #1115.

## Gate Checklist
- [x] Cypher frontend CI gates green for merged PR-4..PR-6 state
  - strict typing: https://github.com/graphistry/pygraphistry/actions/runs/24253760168/job/70819891400
  - differential parity: https://github.com/graphistry/pygraphistry/actions/runs/24253760168/job/70819891456
  - ci-gates: https://github.com/graphistry/pygraphistry/actions/runs/24253760168/job/70820604298
- [x] TCK conformance run attached and green for merged PR-4..PR-6 state
  - tck-gfql: https://github.com/graphistry/pygraphistry/actions/runs/24253760168/job/70820604313
- [x] Differential suite has no placeholder-only trust checks remaining for M1 scope
  - Trust checks are now concrete assertions in:
    - `graphistry/tests/compute/gfql/cypher/test_m1_differential_scaffold.py`
- [x] shortestPath parity suite unchanged after final rebases/merges
  - Local receipt (2026-04-10):
    - `python -B -m pytest -q graphistry/tests/compute/gfql/cypher/test_shortest_path_parity.py --tb=short` -> `58 passed, 15 xfailed in 7.50s`

## Milestone-M1 Exit Criteria Evidence Map

Authoritative criteria source:
- `/home/lmeyerov/Work/graphc/agents/plans/compiler-refactor/milestone-M1.md`

Evidence mapping:
1. `binder.py` exists and is importable
   - `graphistry/compute/gfql/frontends/cypher/binder.py`
2. `mypy --strict` passes on binder + `gfql/ir`
   - strict typing receipt: https://github.com/graphistry/pygraphistry/actions/runs/24253760168/job/70819891400
3. `Binder.bind(ast, ctx)` returns valid `BoundIR` for corpus queries
   - differential parity receipt: https://github.com/graphistry/pygraphistry/actions/runs/24253760168/job/70819891456
   - scaffold coverage: `graphistry/tests/compute/gfql/cypher/test_m1_differential_scaffold.py`
4. `lowering.py` reads from `BoundIR` for migrated semantics
   - implementation lane merged via PR-6: https://github.com/graphistry/pygraphistry/pull/1121
5. `_StageScope` migration slice applied
   - implementation lane merged via PR-6: https://github.com/graphistry/pygraphistry/pull/1121
6. Differential suite exists and passes
   - differential parity receipt: https://github.com/graphistry/pygraphistry/actions/runs/24253760168/job/70819891456
7. Existing test suite passes for merged state
   - merged-state CI receipts for PR-6 head: https://github.com/graphistry/pygraphistry/pull/1121/checks
8. PR docs include migration accounting
   - PR-6 description + merged artifacts: https://github.com/graphistry/pygraphistry/pull/1121
9. shortestPath parity suite preserved
   - suite: `graphistry/tests/compute/gfql/cypher/test_shortest_path_parity.py`
   - local receipt command above

## Notes
- PR-7 remains docs/evidence-only by design.
- Once this PR is merged, #1115 has complete closure evidence against the authoritative gate.
