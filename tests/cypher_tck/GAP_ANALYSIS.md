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
  `match-where3-2`, `match-where4-1`, `match-where4-2`, `match8-1`
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
  advanced Cypher constructs (MATCH/DELETE/UNWIND or computed property
  expressions) in setup scripts.
- **Affected scenarios**: `match4-4`, `match5-25`, `match5-26`, `match5-27`,
  `match5-28`, `match5-29`, plus any scenario whose setup includes complex
  patterns.
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
- **Affected scenarios**: `match-where1-14`, `match-where1-15`, `match3-29`,
  `match3-30`, `match4-9`, `match4-10`, `match6-21`, `match6-22`, `return1-2`,
  `return2-18`, `return4-10`
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
  `match-where6-8`, `match7-11`, `match9-8`
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
  `match-where5-4`, `match7-25`, `match9-8`
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
  `match-where6-8`, `match3-27`, `match3-28`, `match7-1..match7-31`,
  `match8-2`, `match9-8`, `match9-9`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add OPTIONAL MATCH semantics to the translation layer and
  extend the runner to compare projected rows with null handling.

### G13: WITH/LIMIT pipeline semantics
- **Status**: Open
- **Description**: The harness does not support WITH clauses, LIMIT scoping, or
  mid-query variable pipelines.
- **Affected scenarios**: `match-where6-5`, `match3-24`, `match3-25`,
  `match3-26`, `match3-27`, `match3-28`, `match4-8`, `match6-18`, `match7-4`,
  `match7-5`, `match7-6`, `match7-10`, `match7-21`, `match7-22`, `match7-27`,
  `match8-1`, `match8-2`, `match8-3`, `match9-6`, `match9-7`, `return4-1`,
  `return4-11`
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
- **Affected scenarios**: `match2-2`, `match3-6`, `match3-7`, `match3-25`,
  `match3-26`, `match7-23`, `match7-28`, `match9-5`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Ensure label predicate columns remain boolean across GFQL
  chain operations and allow endpoint label filtering on relationship matches.

### G16: Pattern-level variable binding
- **Status**: Open
- **Description**: The harness cannot enforce repeated node/relationship
  variables across pattern parts or MATCH clauses (self-references, cyclic
  patterns, and shared-variable joins).
- **Affected scenarios**: `match3-10`, `match3-12`, `match3-14`, `match3-17`,
  `match3-18`, `match3-19`, `match3-20`, `match3-21`, `match3-22`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add variable binding semantics across pattern parts (shared
  variables in a single pattern or across comma-separated patterns/MATCH
  clauses) and extend row-level validation.

### G17: Variable-length patterns in MATCH
- **Status**: Open
- **Description**: The harness does not support variable-length relationship
  patterns (`*`, `*min..max`) in MATCH, including length bounds, zero-length
  paths, and variable-length relationship lists.
- **Affected scenarios**: `match4-1`, `match4-2`, `match4-3`, `match4-4`,
  `match4-5`, `match4-6`, `match4-7`, `match4-8`, `match5-1`, `match5-2`,
  `match5-3`, `match5-4`, `match5-5`, `match5-6`, `match5-7`, `match5-8`,
  `match5-9`, `match5-10`, `match5-11..match5-29`, `match6-14`, `match6-15`,
  `match6-16`, `match6-17`, `match6-19`, `match6-20`, `match7-12`,
  `match7-13`, `match7-14`, `match7-15`, `match7-19`, `match7-20`,
  `match9-1..match9-9`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add variable-length matching support to GFQL translation and
  row-level validation for projected paths/lists.

### G18: Named path returns
- **Status**: Open
- **Description**: The harness cannot return or validate named path values
  (e.g., `p = (...)` with `RETURN p`), including path ordering and formatting.
- **Affected scenarios**: `match6-1`, `match6-2`, `match6-3`, `match6-4`,
  `match6-5`, `match6-6`, `match6-7`, `match6-8`, `match6-9`, `match6-10`,
  `match6-11`, `match6-12`, `match6-13`, `match6-14`, `match6-15`,
  `match6-16`, `match6-17`, `match6-18`, `match6-19`, `match6-20`,
  `match7-16`, `match7-17`, `match7-18`, `match7-19`, `match7-20`, `match9-9`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add path materialization support in GFQL and extend the runner
  to normalize and compare path outputs.

### G19: Aggregations + boolean projections
- **Status**: Open
- **Description**: The harness does not support aggregations (e.g., `count(*)`)
  or boolean projections in `RETURN` (e.g., `s IS NULL`).
- **Affected scenarios**: `match7-29`, `match7-30`, `match7-31`, `match8-2`,
  `match8-3`, `match9-5`, `return2-10`, `return4-4`, `return4-6`, `return4-7`,
  `return4-8`, `return4-9`, `return4-11`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add aggregation support to the translation layer and extend
  the runner to compute and compare aggregated projections.

### G20: IN predicate list membership
- **Status**: Open
- **Description**: The harness does not translate `IN` predicate list
  membership for node/relationship properties.
- **Affected scenarios**: `match7-17`, `match9-9`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add `IN` predicate translation to the GFQL predicate layer
  and update the runner to compare filtered rows.

### G21: MERGE clause semantics
- **Status**: Open
- **Description**: The harness does not support `MERGE` clause write semantics
  or the ability to match-or-create nodes/relationships.
- **Affected scenarios**: `match8-2`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add MERGE support to the translation layer (or precompute
  fixture states) and extend the runner to validate resulting graph state.

### G22: Relationship list returns + list functions
- **Status**: Open
- **Description**: The harness cannot materialize relationship lists for
  variable-length matches, evaluate list functions (e.g., `last(r)`), or return
  list-valued relationship variables in projections.
- **Affected scenarios**: `match9-1`, `match9-2`, `match9-3`, `match9-4`,
  `match9-6`, `match9-7`, `match9-9`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add list materialization support for variable-length
  relationships, define list/relationship projection normalization, and
  implement list function evaluation in the runner.

### G23: RETURN expression projections
- **Status**: Open
- **Description**: The harness does not evaluate RETURN expressions or
  projections, including property access, arithmetic, list/map construction,
  label predicates, and literal expressions.
- **Affected scenarios**: `return2-1..return2-9`, `return2-11..return2-13`,
  `return3-1`, `return3-2`, `return3-3`, `return4-1..return4-9`, `return4-11`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add RETURN expression evaluation and row-level projection
  validation in the runner (property access, arithmetic operators, list/map
  projections, label predicates, and literal expressions).

### G24: DELETE clause semantics + deleted entity access
- **Status**: Open
- **Description**: The harness does not support DELETE clause semantics, side
  effect validation, or runtime errors for accessing deleted nodes and
  relationships.
- **Affected scenarios**: `return2-14`, `return2-15`, `return2-16`,
  `return2-17`
- **Workaround**: Mark as xfail and capture expected rows/side effects in the
  scenario metadata.
- **Next steps**: Add DELETE support to the translation layer, track side
  effects, and validate deleted-entity access errors.

### G25: ORDER BY semantics
- **Status**: Open
- **Description**: The harness does not support ORDER BY clauses or ordering
  guarantees across WITH/RETURN pipelines.
- **Affected scenarios**: `return4-11`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add ORDER BY support to the translation layer and extend the
  runner to validate ordered outputs.

## Notes
- Keep this doc aligned with `tests/cypher_tck/scenarios.py` and plan updates in
  `plans/cypher-tck-conformance/plan.md`.
