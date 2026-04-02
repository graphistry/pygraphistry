# GFQL #996 Plan: MATCH + OPTIONAL MATCH Connected Path Support
**THIS PLAN FILE**: `plans/issue-996-optional-match-connected/plan.md`
**Issue**: #996 https://github.com/graphistry/pygraphistry/issues/996
**Priority**: p2
**Branch**: `fix/issue-996-optional-match-connected`
**Base**: `origin/master` at `756cb367c`

## Problem Statement

Direct Cypher execution rejects `MATCH ... OPTIONAL MATCH ... RETURN CASE ...` queries with:

```text
[unsupported-cypher-query] Only node-only pre-binding MATCH clauses are supported
before the final connected MATCH in this phase
```

The IS7 (LDBC SNB Interactive short-7) query shape:

```cypher
MATCH (m:Message {id: $messageId})<-[:REPLY_OF]-(c:Comment)-[:HAS_CREATOR]->(p:Person)
OPTIONAL MATCH (m)-[:HAS_CREATOR]->(a:Person)-[r:KNOWS]-(p)
RETURN c.id AS commentId,
    c.content AS commentContent,
    c.creationDate AS commentCreationDate,
    p.id AS replyAuthorId,
    p.firstName AS replyAuthorFirstName,
    p.lastName AS replyAuthorLastName,
    CASE r
        WHEN null THEN false
        ELSE true
    END AS replyAuthorKnowsOriginalMessageAuthor
ORDER BY commentCreationDate DESC, replyAuthorId
```

## Root Cause Analysis

The error originates in `graphistry/compute/gfql/cypher/lowering.py`:

1. **`_merged_match_clause()` (line 2611)** calls `_seed_node_bindings(query.matches[:-1])` for all-but-last MATCH clauses. It does not distinguish OPTIONAL from non-optional MATCH.

2. **`_seed_node_bindings()` (line 2551)** requires each clause to contain only simple `(node)` patterns. A connected pattern like `(m)<-[:REPLY_OF]-(c)-[:HAS_CREATOR]->(p)` is rejected.

3. The broader `compile_cypher_query()` (line 6424) calls `_merged_match_clause()` unconditionally, so the error fires before any OPTIONAL MATCH handling is reached.

### Currently Supported OPTIONAL MATCH Shapes

The compiler already handles these patterns via `_optional_null_fill_plan()` (line 3221):

- `MATCH (single_node) OPTIONAL MATCH (single_node)-[r]-(m) RETURN <optional_alias_props>`
  - Requires: exactly 2 MATCH clauses, first is non-optional single-node, second is optional
  - Requires: RETURN references only optional-only aliases
  - Uses null-fill alignment to produce left-outer-join semantics
- Top-level `OPTIONAL MATCH ()-[r]->() RETURN r`
  - Uses `empty_result_row` for zero-match cases

### What #996 Requires Beyond Current Support

The IS7 query needs three capabilities the compiler doesn't have today:

1. **Connected-path first MATCH**: The non-optional MATCH is `(m)<-[:REPLY_OF]-(c)-[:HAS_CREATOR]->(p)`, not a single node. `_single_node_seed_alias()` returns None for this shape.

2. **Mixed-alias RETURN projection**: The RETURN references aliases from both the non-optional MATCH (`c`, `p`) and the OPTIONAL MATCH (`r`). The current `_optional_null_fill_plan()` requires `referenced <= optional_aliases` (line 3262), which would fail when RETURN includes `c.id`, `p.id`, etc.

3. **CASE expression over nullable optional binding**: `CASE r WHEN null THEN false ELSE true END` must evaluate correctly when `r` is null (no KNOWS edge found). CASE evaluation itself already works in the row pipeline (line 1041 of `row/pipeline.py`), but the null-fill template needs to produce the right input.

## Architectural Approach

The cleanest path is to extend the existing 2-clause `MATCH + OPTIONAL MATCH` architecture rather than redesigning the multi-MATCH merging system. The key changes:

### A. Separate OPTIONAL from non-optional in `_merged_match_clause()`

`_merged_match_clause()` should only merge non-optional MATCH clauses. OPTIONAL MATCH clauses should be excluded from the seed-binding extraction and left for the downstream optional-handling paths.

### B. Generalize `_optional_null_fill_plan()` for connected-path seeds

The null-fill plan currently requires `_single_node_seed_alias()` to return a single node variable. For connected-path seeds, we need to identify the set of aliases bound by the non-optional MATCH and use them as the seed alignment basis, rather than requiring a single node.

### C. Support mixed-alias RETURN projection with null-extension

When RETURN references aliases from both the seed MATCH and the OPTIONAL MATCH, the null-fill must:
- Preserve seed-alias columns from the first MATCH result
- Fill optional-alias columns with null when the OPTIONAL MATCH doesn't match
- Evaluate CASE expressions over the resulting (possibly null) values

### D. Support ORDER BY after OPTIONAL MATCH with mixed projections

The IS7 query has `ORDER BY commentCreationDate DESC, replyAuthorId`, which references projected column names. This should work naturally if the projection produces the right column names before the ORDER BY step.

## Status Legend

- 📝 TODO
- 🔄 IN_PROGRESS
- ✅ DONE
- ❌ FAILED
- 🚫 BLOCKED

## Phases

### Phase 1: Reproduce and lock the current rejection
**Status:** ✅ DONE
**Description:** Write a minimal test that reproduces the #996 error on current master, confirming the exact error code and message. This becomes the regression gate for the fix.
**Actions:**
```bash
# Add test in graphistry/tests/compute/gfql/cypher/test_lowering.py
# Verify it fails with GFQLValidationError E108
# Also add simplified variants:
#   MATCH (a)-[r1]->(b) OPTIONAL MATCH (a)-[r2:T]->(c) RETURN a.id, r2
#   MATCH (a)-[r1]->(b) OPTIONAL MATCH (a)-[r2:T]->(c) RETURN CASE r2 WHEN null THEN false ELSE true END AS flag
```
**Success Criteria:** Test fails with E108 on current code, passes after the fix.

### Phase 2: Fix `_merged_match_clause()` to skip OPTIONAL MATCH
**Status:** ✅ DONE (approach changed: new `_compile_connected_optional_match` path instead)
**Description:** Modify `_merged_match_clause()` and `_seed_node_bindings()` so they only process non-optional MATCH clauses. OPTIONAL MATCH clauses are preserved separately for downstream handling.
**Actions:**
```bash
# In _merged_match_clause():
#   - Filter query.matches to only non-optional clauses for seed extraction
#   - The "last" clause for merging should be the last non-optional clause
#   - OPTIONAL clauses are not passed to _seed_node_bindings()
#
# This unblocks the query from reaching the optional-handling logic in compile_cypher_query()
```
**Success Criteria:** The query no longer throws "Only node-only pre-binding MATCH clauses" — it either succeeds or reaches a different, more specific error.

### Phase 3: Generalize optional null-fill for connected-path seeds
**Status:** ✅ DONE (implemented via left-outer-join in `_apply_connected_optional_match` instead of generalizing null-fill)
**Description:** Extend `_optional_null_fill_plan()` and the downstream `_apply_optional_null_fill()` to support connected-path seed MATCH patterns, not just single-node seeds.
**Actions:**
```bash
# Key changes:
# 1. Replace _single_node_seed_alias() requirement with a connected-path-aware
#    seed identification that can extract a set of seed aliases from the first MATCH
# 2. Relax the `referenced <= optional_aliases` guard to allow mixed projections
#    where some RETURN items reference seed aliases and others reference optional aliases
# 3. The null-fill row template must include null values for optional aliases
#    while seed-alias values are carried from the seed MATCH result
# 4. The alignment mechanism must use the shared aliases between MATCH and OPTIONAL MATCH
#    (e.g., `m` and `p` appear in both clauses) to join results
```
**Success Criteria:** A simplified `MATCH (a)-[r1]->(b) OPTIONAL MATCH (a)-[r2:T]->(c) RETURN a.id, r2` works correctly.

### Phase 4: Wire up CASE expression evaluation with null optional bindings
**Status:** ✅ DONE
**Description:** Ensure that `CASE r WHEN null THEN false ELSE true END` evaluates correctly in RETURN when `r` comes from an OPTIONAL MATCH that didn't match.
**Actions:**
```bash
# The CASE evaluation in row/pipeline.py already handles null via _gfql_null_mask().
# The main work is ensuring the null-fill template produces the right column
# so the CASE expression can evaluate over it.
# Verify with: MATCH (a)-[]->(b) OPTIONAL MATCH (a)-[r2:T]->(c) RETURN CASE r2 WHEN null THEN false ELSE true END AS flag
```
**Success Criteria:** CASE expressions over optional bindings produce correct boolean flags.

### Phase 5: Support ORDER BY on mixed projections
**Status:** ✅ DONE (implemented in `_apply_connected_optional_match`)
**Description:** Verify or fix ORDER BY support when the query has both seed-alias and optional-alias columns in the result.
**Actions:**
```bash
# The IS7 query uses ORDER BY commentCreationDate DESC, replyAuthorId
# This should work naturally if projection produces named columns before ordering
# Verify with a test that includes ORDER BY on seed-alias columns
```
**Success Criteria:** ORDER BY works on projected columns from mixed MATCH + OPTIONAL MATCH results.

### Phase 6: Full IS7 integration test
**Status:** 📝 TODO (deferred — IS7 uses property predicates and labels which need additional graph setup)
**Description:** Run the full IS7 query shape against a small test graph that exercises all the pieces together: connected-path MATCH, OPTIONAL MATCH with shared aliases, CASE null check, and ORDER BY.
**Actions:**
```bash
# Build a small graph with:
#   - Message, Comment, Person nodes
#   - REPLY_OF, HAS_CREATOR, KNOWS edges
#   - Some comments where the reply author knows the message author (r is non-null)
#   - Some comments where they don't (r is null)
# Run the full IS7 query and verify correct row semantics
```
**Success Criteria:** Full IS7 query returns correct results with proper null-extension and CASE evaluation.

### Phase 7: Regression validation and CI
**Status:** ✅ DONE (local)
**Description:** Run the broader test suites to ensure the changes don't regress existing OPTIONAL MATCH behavior or other Cypher lowering paths.
**Actions:**
```bash
# Run:
#   - graphistry/tests/compute/gfql/cypher/test_lowering.py
#   - graphistry/tests/compute/gfql/cypher/test_parser.py
#   - graphistry/tests/compute/test_gfql.py
#   - graphistry/tests/test_compute_chain.py
#   - graphistry/tests/compute/test_chain_let.py
#   - ./bin/typecheck.sh
# Push and verify CI green
```
**Success Criteria:** All existing tests pass, new tests pass, CI green.

### Phase 8: Update existing rejection test
**Status:** ✅ DONE (no changes needed — existing rejection tests still correctly reject their shapes)
**Description:** The test at `test_lowering.py:2802` (`test_string_cypher_failfast_rejects_optional_match_null_extension_shapes_without_safe_alignment`) currently expects both queries to be rejected. After the fix, some of these may now succeed. Update the test expectations accordingly.
**Actions:**
```bash
# Review which of these queries should now succeed vs. remain rejected:
#   "MATCH (n:Single) OPTIONAL MATCH (n)-[r]-(m) WHERE m:NonExistent RETURN r"
#   "MATCH (a:A), (c:C) OPTIONAL MATCH (a)-->(b)-->(c) RETURN b"
# Update test expectations to match new behavior
```
**Success Criteria:** Tests accurately reflect the new supported vs. unsupported query shapes.

## Key Files

| File | Role |
|------|------|
| `graphistry/compute/gfql/cypher/lowering.py` | Main compilation/lowering — primary edit target |
| `graphistry/compute/gfql_unified.py` | Execution of compiled queries including `_apply_optional_null_fill()` |
| `graphistry/compute/gfql/row/pipeline.py` | Row-level expression evaluation including CASE |
| `graphistry/compute/ast.py` | AST node definitions |
| `graphistry/compute/gfql/cypher/ast.py` | Cypher AST definitions including `MatchClause.optional` |
| `graphistry/tests/compute/gfql/cypher/test_lowering.py` | Primary test file |

## Key Functions

| Function | Location | Change Needed |
|----------|----------|---------------|
| `_merged_match_clause()` | lowering.py:2611 | Filter out OPTIONAL MATCH before seed extraction |
| `_seed_node_bindings()` | lowering.py:2551 | May need to handle case where no pre-binding clauses exist |
| `_optional_null_fill_plan()` | lowering.py:3221 | Generalize beyond single-node seed alias |
| `_apply_optional_null_fill()` | gfql_unified.py:142 | May need changes for connected-path alignment |
| `compile_cypher_query()` | lowering.py:6424 | Orchestration of the new flow |

## Risks and Mitigations

1. **Semantic correctness of left-outer-join**: The null-fill alignment must correctly handle the case where shared aliases between MATCH and OPTIONAL MATCH identify the join key. Mitigation: test with multiple seed rows where some have optional matches and some don't.

2. **Performance**: Connected-path OPTIONAL MATCH may produce large intermediate results if the seed MATCH returns many rows. Mitigation: this is a correctness-first change; optimization follows.

3. **Interaction with existing OPTIONAL MATCH paths**: The change must not break the existing single-node-seed OPTIONAL MATCH support. Mitigation: run the full existing test suite.

4. **CASE expression null semantics**: Cypher `CASE r WHEN null` has specific semantics — `WHEN null` means `r IS NULL`, not `r = null`. Mitigation: verify CASE null evaluation matches Neo4j semantics.

## Scope

### In scope for this PR

All three "beyond current support" capabilities are tightly coupled for IS7 and belong together:

1. **Connected-path first MATCH** — core blocker; `_merged_match_clause()` must stop treating OPTIONAL MATCH clauses as seed-binding sources
2. **Mixed-alias RETURN projection** — IS7 returns `c.id`, `p.firstName`, etc. from the seed MATCH alongside `r`-based CASE from the OPTIONAL MATCH
3. **CASE over nullable optional binding** — CASE evaluation already works in `row/pipeline.py`; this falls out of #1+#2 if the null-fill produces the right columns

### Out of scope — file as follow-up issues

These are natural next steps but not required for the IS7 shape:

- **3+ MATCH clauses with mixed OPTIONAL** — IS7 has exactly 2 (1 non-optional + 1 optional); generalizing to N clauses is separate work
- **WHERE on OPTIONAL MATCH results** — IS7 has no WHERE; the existing `_where_uses_optional_only_label_predicate()` rejection can stay
- **Multiple OPTIONAL MATCH clauses** — IS7 has exactly 1; multi-optional is a superset
- **OPTIONAL MATCH with UNWIND/WITH/SKIP/LIMIT** — `_optional_null_fill_plan()` already rejects these; no change needed for IS7

Follow-up issues to file after this PR lands (Phase 9).

## Related Issues

- `#989`: Row-carrier / seeded-row architecture — broader infrastructure that could simplify this
- `#981`: Multi-binding projection problems — adjacent but distinct
- `#994`: Undirected peer-binding wrong-answer bug — already fixed in master

## Phases (continued)

### Phase 9: File follow-up issues for out-of-scope OPTIONAL MATCH shapes
**Status:** 📝 TODO
**Description:** After the PR lands, file issues for the broader OPTIONAL MATCH shapes that remain unsupported.
**Actions:**
```bash
# File issues for:
#   - 3+ MATCH clauses with mixed OPTIONAL
#   - WHERE on OPTIONAL MATCH results
#   - Multiple OPTIONAL MATCH clauses
#   - OPTIONAL MATCH with UNWIND/WITH/SKIP/LIMIT
# Reference this PR and #996 as prior art
```
**Success Criteria:** Follow-up issues exist and are linked.

## Resume State

**Last updated:** 2026-04-01
**Current phase:** Phase 7 (validation in progress)
**Branch:** `fix/issue-996-optional-match-connected`
**Base:** `origin/master` at `756cb367c`

### Implementation Summary

The fix adds a new compilation path `_compile_connected_optional_match()` in `lowering.py` that:
1. Detects the connected-MATCH + OPTIONAL-MATCH pattern via `_is_connected_optional_match_query()`
2. Lowers each MATCH clause independently using `lower_match_clause()`
3. Emits a `ConnectedOptionalMatchPlan` with base and optional chains
4. At execution time (`_apply_connected_optional_match()` in `gfql_unified.py`):
   - Runs each chain with `rows(binding_ops=...)` to get binding row tables
   - Left-outer-joins on shared node alias columns
   - Evaluates RETURN expressions (including CASE) against the joined rows
   - Applies ORDER BY, DISTINCT, SKIP, LIMIT

Key files changed:
- `graphistry/compute/gfql/cypher/lowering.py`: detection + compilation
- `graphistry/compute/gfql_unified.py`: execution + expression evaluation
- `graphistry/tests/compute/gfql/cypher/test_lowering.py`: tests

### Validation

- `test_lowering.py`: 503 passed, 49 skipped
- `test_gfql.py`, `test_compute_chain.py`, `test_chain_let.py`, `test_parser.py`: 286 passed, 3 skipped
- `./bin/typecheck.sh`: success (204 files)
- `./bin/lint.sh`: all checks passed
