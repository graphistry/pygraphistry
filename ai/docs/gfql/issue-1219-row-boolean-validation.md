# Issue #1219: Row-Boolean WHERE Validation Gap (OR/NOT/XOR)

Issue: <https://github.com/graphistry/pygraphistry/issues/1219>

## Problem Statement

PR #1217 moved Cypher parsing onto Earley, which made row-boolean forms like `a OR b`, `NOT a`, and `a XOR b` parse cleanly in more cases.

Current pipeline behavior is asymmetrical:
- Pattern+row mixed boolean forms are explicitly gated as unsupported (E108) in parser/normalizer flow.
- Row-only boolean forms pass through to `BoundPredicate(expression=...)`, then lower to runtime expression evaluation.

This is acceptable for simple expressions, but it leaves a static-validation gap on edge compositions we have not explicitly validated end-to-end.

## Current Code Anchors

- Parser generic WHERE route: `graphistry/compute/gfql/cypher/parser.py` (`generic_where_clause`, `_mixed_where_pattern_expr_error`)
- Binder split behavior: `graphistry/compute/gfql/frontends/cypher/binder.py` (`_where_predicates`, top-level `AND` split)
- Pushdown null-safety classifier: `graphistry/compute/gfql/ir/pushdown_safety.py` (`is_null_rejecting`)
- Pushdown implementation: `graphistry/compute/gfql/passes/predicate_pushdown.py` (`_split_conjuncts`, `_push_filter_into_pattern`)
- Existing shape tests: `graphistry/tests/compute/gfql/cypher/test_where_bool_conformance.py`

## Risk Surface To Investigate

1. OPTIONAL MATCH null-extension interactions with OR/NOT/XOR predicates.
2. Predicate pushdown behavior when a conjunct contains an OR subgroup.
3. Mixed-type branches (`n.p = 12 OR n.p = 'twelve'`) and coercion semantics.
4. NULL-aware disjunction semantics (`n.p = 12 OR n.p IS NULL`).
5. Nested compositions (`(a OR b) AND (c OR d)`, `NOT (a OR b)`).

## Candidate Solution Directions

### Option A: Broad static gate now

Add a row-boolean validator that rejects unvalidated shapes with E108 before lowering.

- Pros: immediate safety envelope, symmetry with existing pattern-side unsupported gates.
- Cons: may be over-restrictive for already-working cases.

### Option B: Safe-subset allowlist + reject rest

Admit explicitly defined safe forms and block the rest.

- Pros: balanced safety/velocity.
- Cons: higher maintenance burden; subset boundaries can become confusing.

### Option C: Full support across planning/runtime

Audit and harden parser → binder → pushdown → lowering semantics for all row-boolean shapes.

- Pros: strongest long-term result.
- Cons: largest scope and highest schedule risk.

### Option D: Documentation-only defer

Keep behavior, document caveats, and wait for concrete failures.

- Pros: lowest immediate cost.
- Cons: silent correctness risk remains for unvalidated compositions.

## Recommended Path (Staged)

Recommend an A/B hybrid as immediate follow-up, with C as phase-2 hardening:

1. Phase 1: Add a conservative static gate for high-risk row-boolean shapes (especially optional/null-extension sensitive cases).
2. Phase 1: Add tests that encode accepted vs rejected shapes with explicit expected outcomes.
3. Phase 2: Expand support by hardening pushdown/null-safety analysis and relaxing gates only when proven.

This protects correctness while keeping room to broaden support deliberately.

## Draft Implementation Sketch

1. Add a `where` row-boolean validation helper in Cypher frontend path (parser/normalizer boundary).
2. Emit E108 for disallowed shapes with actionable message that references supported subset.
3. Keep existing binder top-level `AND` split behavior unchanged initially.
4. Extend test matrices in:
   - `graphistry/tests/compute/gfql/cypher/test_where_bool_conformance.py`
   - `graphistry/tests/compute/gfql/cypher/test_parser.py`
   - pushdown-focused tests for optional arm + null-rejecting semantics.

## Validation Matrix (Initial)

- Accept (phase 1 target): simple row OR/NOT on non-optional aliases where semantics are already covered by existing evaluator behavior.
- Reject (phase 1 target): compositions that include optional-arm/null-extension ambiguity or unsupported nested shape forms.
- Confirm no regressions on existing pattern-side E108 gates.

## Review Skill Workflow For Implementation/Validation

When implementation starts, run multi-wave review workflow from `review` skill:

1. `mode=findings fixes=deferred` first, with credentials gate in Phase 1e.
2. Add/adjust code and tests.
3. Re-run review waves until convergence criteria are met.
4. Only post PR comments after explicit human confirmation.

## Human Gating Requirements

- No auto-merge for this PR.
- Human approval required before selecting final option scope.
- Human approval required before posting PR comments from review flow.
- Human approval required for merge.
