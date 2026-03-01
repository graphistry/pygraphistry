# PR 915 Handoff Note: OTel + Stack Order

Date: 2026-03-01
Author: restack/where-core workstream

## Purpose
Provide a concise handoff for PR #915 on how to align with PR #917 and where to place OTel-related changes.

## Stack Guidance
- Rebase PR #915 on top of PR #917 once #917 merges.
- Reason: #917 now contains the canonical WHERE validator/dataflow updates and large test refactors. Rebasing reduces conflict churn and avoids duplicate fixes.

## What Was Intentionally Removed From #917
- Branch-local OTel instrumentation added during where-stack development was removed from #917.
- Files where lingering OTel instrumentation was removed:
  - `graphistry/compute/chain.py`
  - `graphistry/compute/hop.py`
  - `graphistry/compute/gfql_unified.py`
  - `graphistry/compute/gfql/df_executor.py`
- Centralization attempt around `graphistry.otel` was also backed out on this stack.

## Recommendation For #915
- If #915 is the intended home for OTel/tracing changes:
  - Re-introduce OTel changes there after rebasing on #917.
  - Keep scope explicit: tracing hooks only, no mixed validator/executor behavior changes.
  - Include focused tests for tracing behavior and import/optional-dependency safety.

## Reference Artifacts
- PR #917 (where-core restack): https://github.com/graphistry/pygraphistry/pull/917
- Prior OTel benchmark/log artifacts (historic context only):
  - `plans/pr-886-where/benchmarks/phase-49-dense-multi-otel.md`
  - `plans/pr-886-where/benchmarks/phase-49-dense-multi-otel.log`

## Notes
- #917 currently keeps runtime behavior and validation concerns separated from OTel concerns.
- This separation is intentional to keep merge/review risk lower across stacked PRs.
