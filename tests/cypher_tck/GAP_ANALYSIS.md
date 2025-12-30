# Cypher TCK gap + workaround analysis

This document tracks feature gaps between the openCypher TCK and the current
GFQL-based conformance harness, along with any temporary workarounds and
translation guidelines.

## How to use this document
- When adding or updating a scenario, check if it introduces a new gap.
- If a scenario is marked skip/xfail, add or reference a gap entry below.
- Keep the entry short, with a clear root cause and a proposed path forward.
- Update the entry when the gap is partially or fully addressed.

## Entry format
- **ID**: Short identifier (G1, G2, ...)
- **Status**: Open | Partial | Closed
- **Description**: What is missing or incorrect
- **Affected scenarios**: Scenario keys or TCK paths
- **Workaround**: How we currently cope (xfail/skip/translation hack)
- **Next steps**: What work would close the gap

## Workarounds in use
- **Label matching**: Store labels as lists in a `labels` column and expand to
  boolean columns `label__<name>` in the harness. Translate label predicates as
  `n({"label__A": True, "label__B": True})`.

## Gaps

### G1: Cartesian product + projection results
- **Status**: Open
- **Description**: The harness does not compute cartesian products across
  multiple MATCH clauses or comma-separated patterns, nor validate row-level
  projections in `RETURN`.
- **Affected scenarios**: `match1-5` (Match1 [5]), `match-where3-1`,
  `match-where3-2`, `match-where4-1`, `match-where4-2`
- **Workaround**: Mark as xfail with expected rows captured in the scenario.
- **Next steps**: Add a result comparator for row projections and introduce a
  cartesian product mechanism (likely outside GFQL core) for multiple MATCH
  clauses.

### G2: Scenario-level row expectations
- **Status**: Partial
- **Description**: Scenario `Expected.rows` is recorded but not validated by the
  runner. The harness can now validate alias-tagged node IDs for single-node
  returns, but does not compare row-level combinations or projections.
- **Affected scenarios**: Any scenario that only specifies rows, or uses RETURN
  expressions beyond node/edge identity.
- **Workaround**: Use node_ids/edge_ids or `return_alias` when possible;
  otherwise xfail and note row expectations.
- **Next steps**: Extend the runner with row extraction and normalization
  (alias binding, ordering rules, null handling, projection comparisons).

### G3: CREATE parser coverage
- **Status**: Partial
- **Description**: The minimal CREATE parser handles nodes, labels, simple
  properties, basic relationships, chained relationship patterns, and simple
  relationship properties, but does not parse variable-length relationships or
  advanced Cypher constructs.
- **Affected scenarios**: Any scenario whose setup includes relationship
  properties or complex patterns.
- **Workaround**: Manually craft fixtures or extend parser incrementally.
- **Next steps**: Support variable-length relationships and additional pattern
  forms as needed by the next wave of scenarios.

### G4: Parameter binding
- **Status**: Open
- **Description**: The harness does not support Cypher parameters (e.g. `$param`)
  for query execution or comparison.
- **Affected scenarios**: `match-where1-6`, `match-where1-9`, `match-where2-2`
- **Workaround**: Mark as xfail and capture expected rows in the scenario.
- **Next steps**: Add parameter injection support (scenario metadata -> GFQL
  predicate substitution) and validation for edge-return scenarios.

### G5: Disjunctive WHERE predicates (OR)
- **Status**: Open
- **Description**: The harness does not support OR predicates across node or
  relationship properties/types.
- **Affected scenarios**: `match-where1-10`, `match-where1-11`, `match-where4-2`,
  `match-where5-4`
- **Workaround**: Mark as xfail and capture expected rows in the scenario.
- **Next steps**: Add predicate combinators or explicit OR support in the
  translation layer and runner comparison.

### G6: Path variables + length()
- **Status**: Open
- **Description**: Path variables and path-length predicates (e.g. `length(p)`)
  are not supported in GFQL translations or the harness.
- **Affected scenarios**: `match-where1-12`, `match-where1-13`
- **Workaround**: Mark as xfail and capture expected node sets.
- **Next steps**: Define a path representation in GFQL and implement length
  checks in the reference enumerator + runner.

### G7: Compile-time validation coverage
- **Status**: Open
- **Description**: The harness does not model Cypher compile-time validation
  errors (e.g., invalid path property predicates, aggregation in WHERE).
- **Affected scenarios**: `match-where1-14`, `match-where1-15`
- **Workaround**: Mark as xfail with explicit syntax-error reasons.
- **Next steps**: Map Cypher error classes to GFQL validation and assert
  exception types in the runner.

### G8: Multi-variable WHERE predicates
- **Status**: Open
- **Description**: The harness cannot yet translate conjunctive WHERE predicates
  that bind and filter multiple variables across a complex MATCH pattern (e.g.,
  multiple node aliases, re-used variables, multi-hop joins).
- **Affected scenarios**: `match-where2-1`, `match-where2-2`
- **Workaround**: Mark as xfail and capture expected rows/node sets in the
  scenario metadata.
- **Next steps**: Extend the translation layer to support per-alias predicates
  and multi-pattern binding, then add row-level validation to assert projected
  property outputs.

### G9: Variable comparison joins
- **Status**: Open
- **Description**: The harness cannot express comparisons between variables
  (e.g., `a = b`, `a <> b`, `a.id = b.id`, `n.animal = x.animal`,
  `x.val < y.val`), which requires join semantics across bindings and row-level
  projection validation.
- **Affected scenarios**: `match-where3-1`, `match-where3-2`, `match-where3-3`,
  `match-where4-1`, `match-where6-5`, `match-where6-6`, `match-where6-7`,
  `match-where6-8`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add join-aware translation support (variable comparison
  predicates) and extend the runner to compare projected rows.

### G10: Pattern predicates + variable-length relationships in WHERE
- **Status**: Open
- **Description**: Pattern predicates used as boolean filters in WHERE (e.g.,
  `(a)-[:T]->(b)` or `(a)-[:T*]->(b)`) are not supported, nor are variable-length
  relationship predicates.
- **Affected scenarios**: `match-where4-2`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add translation support for pattern predicates and
  variable-length relationship filters, then validate projected rows.

### G11: Comparison predicates + null semantics
- **Status**: Open
- **Description**: The harness cannot express comparison predicates (e.g.,
  `>`, `<`), label predicates in WHERE (e.g., `i:TextNode`), or null checks
  (`IS NOT NULL`) with Cypher's tri-valued logic.
- **Affected scenarios**: `match-where5-1`, `match-where5-2`, `match-where5-3`,
  `match-where5-4`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add predicate operators and null-handling semantics to the
  translation layer and runner comparisons.

### G12: OPTIONAL MATCH semantics
- **Status**: Open
- **Description**: The harness does not model OPTIONAL MATCH row preservation,
  null propagation, or WHERE filtering behavior for optional patterns.
- **Affected scenarios**: `match-where6-1`, `match-where6-2`, `match-where6-3`,
  `match-where6-4`, `match-where6-5`, `match-where6-6`, `match-where6-7`,
  `match-where6-8`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add OPTIONAL MATCH semantics to the translation layer and
  extend the runner to compare projected rows with null handling.

### G13: WITH/LIMIT pipeline semantics
- **Status**: Open
- **Description**: The harness does not support WITH clauses, LIMIT scoping, or
  mid-query variable pipelines.
- **Affected scenarios**: `match-where6-5`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add pipeline support (WITH, LIMIT, variable scoping) and
  extend row-level validation for projections.

### G14: Multiple relationship types in MATCH
- **Status**: Open
- **Description**: The harness does not support relationship type lists in MATCH
  patterns (e.g., `:KNOWS|HATES`).
- **Affected scenarios**: `match2-6`, `match3-8`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add relationship type set support in the translation layer
  and validate projected relationship outputs.

### G15: Label predicates on relationship endpoints
- **Status**: Open
- **Description**: Label predicates on both sides of relationship patterns are
  not reliably supported; label boolean columns can trigger schema/type
  mismatches during multi-hop GFQL filtering.
- **Affected scenarios**: `match2-2`, `match3-6`, `match3-7`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Ensure label predicate columns remain boolean across GFQL
  chain operations and allow endpoint label filtering on relationship matches.

## Notes
- Keep this doc aligned with `tests/cypher_tck/scenarios.py` and plan updates in
  `plans/cypher-tck-conformance/plan.md`.
