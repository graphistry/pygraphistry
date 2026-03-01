# PR 915 Handoff: OTel Context Relative To PR 917

Date: 2026-03-01  
Branch at write time: `restack/where-core`  
Relevant PRs:
- PR 917 (WHERE/validator stack): https://github.com/graphistry/pygraphistry/pull/917
- PR 915 (target home for tracing follow-up): team-owned

## Why this note exists
During PR 917 restacking, OTel-related edits were introduced, then intentionally backed out to keep PR 917 scoped to WHERE validator/executor behavior and test stabilization.  
This note captures the exact sequence, what was removed, and what PR 915 should do next.

## Timeline (with commits)
1. OTel centralization attempt landed on this stack:
- `ef542eee` `refactor(otel): centralize no-op tracing shim in graphistry.otel`

2. That centralization was reverted:
- `24b4c99a` `Revert "refactor(otel): centralize no-op tracing shim in graphistry.otel"`

3. Remaining branch-local OTel instrumentation in WHERE stack files was removed:
- `04b761a6` `chore(gfql): remove lingering otel instrumentation from where stack`

## What was removed from PR 917
OTel instrumentation/backfill was removed from these files:
- `graphistry/compute/chain.py`
- `graphistry/compute/hop.py`
- `graphistry/compute/gfql_unified.py`
- `graphistry/compute/gfql/df_executor.py`

Intent: PR 917 should carry validator/dataflow/test refactors, not tracing behavior.

## Why PR 917 intentionally excludes OTel
- PR 917 already has high churn in WHERE validation timing, schema-flow integration, same-path behavior, and broad test rewrites.
- Mixing tracing into that review makes regression attribution and merge conflict resolution harder.
- Keeping OTel out of PR 917 isolates risk domains:
  - PR 917: correctness + validation/dataflow
  - PR 915: tracing/instrumentation behavior

## Suggested integration strategy for PR 915
1. Rebase PR 915 onto PR 917 after PR 917 merges.
2. Re-introduce OTel/tracing changes in PR 915 only.
3. Keep OTel patch scope explicit:
- instrumentation hooks/spans/attributes
- optional-dependency/no-op behavior
- no functional WHERE/validator semantic changes unless intentionally coupled and tested
4. Add focused validation for PR 915:
- import behavior when OTel deps are unavailable
- span emission shape/keys on representative chain + same-path flows
- no behavior change in WHERE correctness tests

## Historical artifacts (context only)
- `plans/pr-886-where/benchmarks/phase-49-dense-multi-otel.md`
- `plans/pr-886-where/benchmarks/phase-49-dense-multi-otel.log`

These are useful for performance/tracing context, but are not part of the PR 917 deliverable surface.

## Current status snapshot
- PR 917 branch currently reflects OTel backout (`24b4c99a`, `04b761a6`) and does not carry those tracing changes.
- This is deliberate, not accidental drift.
