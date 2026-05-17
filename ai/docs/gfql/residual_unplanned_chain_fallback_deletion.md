# Residual unplanned Cypher chain fallback deletion checklist

This checklist tracks the final deletion gate for pygraphistry#1486. It is an
internal coordination artifact for GFQL/Cypher maintainers, not user-facing API
documentation.

Current audit base: `origin/master @ c7d73c762ba9a26c1266ecb93c100e0fa242b785`
on 2026-05-17.

## Current residual inventory

Runtime approval is still controlled by
`graphistry/compute/gfql_unified.py::_APPROVED_UNPLANNED_CHAIN_FALLBACK_CODES`.
On the audit base it contains:

```python
frozenset({
    "multiple_match_stages",
    "optional_match_reentry",
})
```

The literal GFQL/Cypher test-corpus compile scan found:

| metric | count |
|---|---:|
| candidate literals | 1104 |
| compiled literals | 721 |
| compile errors, expected for negative parser/validator tests | 383 |
| `logical_plan is None` without a defer code | 0 |
| `multiple_match_stages` defer-code hits | 2 |
| `optional_match_reentry` defer-code hits | 0 |

The remaining `multiple_match_stages` hits are both guard-test forms of the
same #1495 path-alias carry residual:

```cypher
MATCH path = shortestPath((a)-[*]-(b))
MATCH (b)-->(c)
RETURN c
```

and the aggregate variant:

```cypher
MATCH path = shortestPath((a)-[*]-(b))
MATCH (b)-->(c)
RETURN count(c) AS c
```

The direct optional-reentry smoke query now compiles with a logical plan and no
defer code:

```cypher
MATCH (a:A)
MATCH (a)-[:R*2]->(b)
WITH b
OPTIONAL MATCH (b)-[:R]->(c)
RETURN c.id AS id
```

## Active blockers

- pygraphistry#1495 is open and owns the final path-alias carry decision:
  native-plan the shape or classify it as an explicit fail-fast unsupported
  shape outside the residual chain fallback.
- pygraphistry#1469 is closed and is not a current unplanned-chain deletion
  blocker.
- `optional_match_reentry` appears stale in the runtime allowlist based on this
  corpus scan, but final removal still belongs in the post-#1495 cut so the
  whole allowlist and fallback branch are removed together.

Do not delete the fallback branch until #1495 is resolved and the coordinator
approves the final cut.

## Final deletion checklist

Before deleting the fallback:

- Rebase on latest `origin/master` and re-run this inventory.
- Confirm #1495 is resolved and the path-alias carry query no longer requires
  `_execute_compiled_query_chain_non_union()` through an approved
  `logical_plan is None` route.
- Confirm no corpus query emits `optional_match_reentry`; if any do, either
  native-plan them or classify them as explicit fail-fast unsupported shapes.
- Confirm #1469 remains closed and no row-pipeline map/list AST lane depends on
  this unplanned Cypher chain fallback.

Code removal:

- Delete `_APPROVED_UNPLANNED_CHAIN_FALLBACK_CODES`.
- Delete `_is_approved_unplanned_chain_fallback()`.
- Delete the `logical_plan is None` approved-chain branch in
  `_execute_compiled_query_non_union()`.
- Keep fail-fast behavior for `logical_plan is None` without a covered native
  route, including clear `GFQLValidationError` context.
- Remove fallback-only runtime guard tests and replace them with native-route or
  fail-fast assertions for the formerly residual shapes.
- Remove comments and changelog language that imply approved residual fallback
  buckets remain.

Validation for the final cut:

- Run focused runtime cutover and logical planner tests:
  `graphistry/tests/compute/gfql/test_runtime_physical_cutover.py` and
  `graphistry/tests/compute/gfql/test_logical_planner.py`.
- Run the broad non-cuDF GFQL/Cypher suite touched by the deletion.
- Run `ruff`, `typecheck`, and `git diff --check`.
- Run DGX RAPIDS 25.02 and 26.02 GFQL/cuDF validation for any touched cuDF path.
- Run the review skill until convergence.
- Loop PR CI until green.
- Update #1486, #1468, and historical #1419 with the final deletion receipt.
