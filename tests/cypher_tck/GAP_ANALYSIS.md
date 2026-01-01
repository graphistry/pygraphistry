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
- **Affected scenarios**: `match-where1-6`, `match-where1-9`, `match-where2-2`,
  `return6-17`, `return-orderby6-1`, `return-skip-limit1-2`,
  `return-skip-limit1-6`, `return-skip-limit1-8`, `return-skip-limit2-10`,
  `return-skip-limit2-11`, `return-skip-limit2-14`, `return-skip-limit2-15`,
  `return-skip-limit3-2`, `with6-5`, `with-where2-2`, `with-skip-limit3-2`,
  `unwind1-6`
- **Workaround**: Mark as xfail and capture expected rows in the scenario.
- **Next steps**: Add parameter injection support (scenario metadata -> GFQL
  predicate substitution) and validation for edge-return scenarios.

### G5: Disjunctive WHERE predicates (OR)
- **Status**: Open
- **Description**: The harness does not support OR predicates across node or
  relationship properties/types.
- **Affected scenarios**: `match-where1-10`, `match-where1-11`, `match-where4-2`,
  `match-where5-4`, `with-where4-2`, `with-where7-3`
- **Workaround**: Mark as xfail and capture expected rows in the scenario.
- **Next steps**: Add predicate combinators or explicit OR support in the
  translation layer and runner comparison.

### G6: Path variables + length()
- **Status**: Open
- **Description**: Path variables and path-length predicates (e.g. `length(p)`)
  are not supported in GFQL translations or the harness.
- **Affected scenarios**: `match-where1-12`, `match-where1-13`, `return6-8`,
  `return6-13`, `with1-4`, `with6-4`
- **Workaround**: Mark as xfail and capture expected node sets.
- **Next steps**: Define a path representation in GFQL and implement length
  checks in the reference enumerator + runner.

### G7: Compile-time validation coverage
- **Status**: Open
- **Description**: The harness does not model Cypher compile-time validation
  errors (e.g., invalid path property predicates, aggregation in WHERE).
- **Affected scenarios**: `match-where1-14`, `match-where1-15`, `match3-29`,
  `match3-30`, `match4-9`, `match4-10`, `match6-21`, `match6-22`, `return1-2`,
  `return2-18`, `return4-10`, `return6-14`, `return6-15`, `return6-20`,
  `return6-21`, `return7-2`, `with4-4`, `with4-5`, `with6-8`, `with6-9`,
  `with-orderby1-46-1..with-orderby1-46-10`,
  `with-orderby2-25-1..with-orderby2-25-25`,
  `with-orderby3-8-1..with-orderby3-8-30`
- **Workaround**: Mark as xfail with explicit syntax-error reasons.
- **Next steps**: Map Cypher error classes to GFQL validation and assert
  exception types in the runner.

### G8: Multi-variable WHERE predicates
- **Status**: Open
- **Description**: The harness cannot yet translate conjunctive WHERE predicates
  that bind and filter multiple variables across a complex MATCH pattern (e.g.,
  multiple node aliases, re-used variables, multi-hop joins).
- **Affected scenarios**: `match-where2-1`, `match-where2-2`, `with-where2-1`,
  `with-where2-2`
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
  `match-where6-8`, `match7-11`, `match9-8`, `return6-13`,
  `with-where3-1..with-where3-3`, `with-where4-1`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add join-aware translation support (variable comparison
  predicates) and extend the runner to compare projected rows.

### G10: Pattern predicates + variable-length relationships in WHERE
- **Status**: Open
- **Description**: Pattern predicates used as boolean filters in WHERE (e.g.,
  `(a)-[:T]->(b)` or `(a)-[:T*]->(b)`) are not supported, nor are variable-length
  relationship predicates.
- **Affected scenarios**: `match-where4-2`, `with-where4-2`
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
  `match-where5-4`, `match7-25`, `match9-8`, `with-where5-1..with-where5-4`
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
  `match8-2`, `match9-8`, `match9-9`, `with1-5`, `with1-6`,
  `with-where1-3`, `with-where1-4`
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
  `return4-11`, `return6-3`, `return6-13`, `return6-16`, `return6-18`,
  `return8-1`, `return-orderby4-1`, `return-orderby4-2`, `return-skip-limit1-3`,
  `return-skip-limit2-6`, `with1-1..with1-6`, `with2-1..with2-2`, `with3-1`,
  `with4-1..with4-7`, `with5-1..with5-2`, `with6-1..with6-9`,
  `with7-1..with7-2`, `with-where1-1..with-where1-4`,
  `with-where2-1..with-where2-2`, `with-where3-1..with-where3-3`,
  `with-where4-1..with-where4-2`, `with-where5-1..with-where5-4`,
  `with-where6-1`, `with-where7-1..with-where7-3`,
  `with-skip-limit1-1..with-skip-limit1-2`, `with-skip-limit2-1..with-skip-limit2-4`,
  `with-skip-limit3-1..with-skip-limit3-3`, `with-orderby1-1..with-orderby1-46`,
  `with-orderby2-1..with-orderby2-10`, `with-orderby2-11..with-orderby2-20`,
  `with-orderby2-21..with-orderby2-25`, `with-orderby3-1..with-orderby3-8`,
  `with-orderby4-1..with-orderby4-9`, `unwind1-3..unwind1-5`, `unwind1-7`,
  `unwind1-11..unwind1-13`
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
  `match9-1..match9-9`, `return6-8`, `return6-13`, `with6-4`, `with-where4-2`
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
  `match7-16`, `match7-17`, `match7-18`, `match7-19`, `match7-20`, `match9-9`,
  `return7-1`, `with1-4`, `with6-4`
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
  `return4-8`, `return4-9`, `return4-11`, `return5-1`, `return5-3`,
  `return5-4`, `return5-5`, `return6-1..return6-13`, `return6-16..return6-19`,
  `return8-1`, `with4-6`, `with5-2`, `with6-1..with6-9`, `with7-2`,
  `with-where6-1`, `with-skip-limit1-2`, `with-skip-limit2-4`,
  `with-orderby1-45-1..with-orderby1-45-10`,
  `with-orderby2-22-1..with-orderby2-22-2`,
  `with-orderby2-23-1..with-orderby2-23-2`, `unwind1-4`, `unwind1-5`,
  `unwind1-12`
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
- **Affected scenarios**: `merge1-1..merge1-17`, `merge2-1..merge2-6`,
  `merge3-1..merge3-5`, `merge4-1..merge4-2`, `merge5-1..merge5-25`,
  `merge6-1..merge6-4`, `merge6-6..merge6-7`, `merge7-1..merge7-5`,
  `merge8-1`, `merge9-1..merge9-4`, plus `match8-2`, `unwind1-6`
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
- **Description**: The harness does not evaluate projection expressions
  (RETURN/WITH/ORDER BY), including property access, arithmetic, list/map
  construction, label predicates, and literal expressions.
- **Affected scenarios**: `return2-1..return2-9`, `return2-11..return2-13`,
  `return3-1`, `return3-2`, `return3-3`, `return4-1..return4-9`, `return4-11`,
  `return5-1..return5-5`, `return6-1..return6-13`, `return6-16..return6-19`,
  `return7-1`, `return8-1`, `with-orderby1-23..with-orderby1-45`,
  `with-orderby2-1..with-orderby2-10`, `with-orderby2-11..with-orderby2-20`,
  `with-orderby2-21..with-orderby2-24`, `with-orderby3-1..with-orderby3-7`,
  `with-orderby4-1..with-orderby4-9`, `unwind1-5`, `unwind1-11`,
  `unwind1-13`
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
- **Affected scenarios**: `delete1-1..delete1-8`, `delete2-1..delete2-5`,
  `delete3-1..delete3-2`, `delete4-1..delete4-3`, `delete5-1..delete5-9`,
  `delete6-1..delete6-14`, plus `return2-14`, `return2-15`, `return2-16`,
  `return2-17`
- **Workaround**: Mark as xfail and capture expected rows/side effects in the
  scenario metadata.
- **Next steps**: Add DELETE support to the translation layer, track side
  effects, and validate deleted-entity access errors.

### G25: ORDER BY semantics
- **Status**: Open
- **Description**: The harness does not support ORDER BY clauses, ordering
  guarantees across WITH/RETURN pipelines, or ORDER BY validation (scoping and
  aggregation restrictions).
- **Affected scenarios**: `return4-11`, `return-orderby1-1..return-orderby1-12`,
  `return-orderby2-1..return-orderby2-14`, `return-orderby3-1`,
  `return-orderby4-1..return-orderby4-2`, `return-orderby5-1`,
  `return-orderby6-1..return-orderby6-5`,
  `return-skip-limit1-1..return-skip-limit1-2`, `return-skip-limit2-2`,
  `return-skip-limit2-4`, `return-skip-limit2-5`, `return-skip-limit2-7`,
  `return-skip-limit2-8`, `return-skip-limit2-11`, `return-skip-limit2-15`,
  `return-skip-limit3-1..return-skip-limit3-3`, `with3-1`, `with4-6`,
  `with-skip-limit1-1..with-skip-limit1-2`, `with-skip-limit2-1`,
  `with-skip-limit2-4`, `with-skip-limit3-1..with-skip-limit3-3`,
  `with-orderby1-1..with-orderby1-46`, `with-orderby2-1..with-orderby2-10`,
  `with-orderby2-11..with-orderby2-20`, `with-orderby2-21..with-orderby2-25`,
  `with-orderby3-1..with-orderby3-8`, `with-orderby4-1..with-orderby4-9`,
  `unwind1-6`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add ORDER BY support to the translation layer and extend the
  runner to validate ordered outputs.

### G26: DISTINCT semantics
- **Status**: Open
- **Description**: The harness does not support DISTINCT projections or
  DISTINCT handling inside aggregations.
- **Affected scenarios**: `return4-6`, `return5-1`, `return5-2`, `return5-3`,
  `return5-4`, `return5-5`, `return6-16`, `with5-1`, `with5-2`,
  `with-where1-2`, `with-where4-2`, `with-skip-limit1-1`,
  `with-orderby1-44-1..with-orderby1-44-2`,
  `with-orderby2-24-1..with-orderby2-24-2`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add DISTINCT handling to projection and aggregation paths
  and extend the runner to validate distinct row sets.

### G27: SKIP/LIMIT semantics
- **Status**: Open
- **Description**: The harness does not support SKIP/LIMIT clauses or argument
  validation (including parameterized values).
- **Affected scenarios**: `return-skip-limit1-1..return-skip-limit1-11`,
  `return-skip-limit2-1..return-skip-limit2-17`,
  `return-skip-limit3-1..return-skip-limit3-3`, `with7-1`,
  `with-skip-limit1-1..with-skip-limit1-2`,
  `with-skip-limit2-1..with-skip-limit2-4`,
  `with-skip-limit3-1..with-skip-limit3-3`, `with-orderby1-1..with-orderby1-46`,
  `with-orderby2-1..with-orderby2-10`, `with-orderby2-11..with-orderby2-20`,
  `with-orderby2-21..with-orderby2-24`, `with-orderby3-1..with-orderby3-8`,
  `with-orderby4-1..with-orderby4-9`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add SKIP/LIMIT handling to the translation layer and enforce
  compile/runtime validation of arguments in the runner.

### G28: UNWIND clause support
- **Status**: Open
- **Description**: The harness does not execute UNWIND clauses or list
  expansion semantics.
- **Affected scenarios**: `return-orderby1-1..return-orderby1-12`,
  `return-orderby4-1`, `return-skip-limit1-3`, `return-skip-limit2-1`,
  `return-skip-limit2-6`, `return-skip-limit3-3`, `with-orderby1-1..with-orderby1-22`,
  `with-orderby1-43..with-orderby1-45`, `with-orderby3-7-1..with-orderby3-7-10`,
  `unwind1-1..unwind1-13`, `union1-3`, `union2-3`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add UNWIND translation/execution support and row-level
  comparison for list projections.

### G29: Temporal values and functions
- **Status**: Open
- **Description**: The harness does not parse or evaluate temporal functions
  and values (date, time, localtime, datetime, localdatetime) or their ordering
  semantics.
- **Affected scenarios**: `with-orderby1-11..with-orderby1-20`,
  `with-orderby1-33..with-orderby1-42`, `with-orderby1-45-6..with-orderby1-45-10`,
  `with-orderby2-11..with-orderby2-20`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add temporal literal/function support in the parser and
  expression evaluation/ordering layers.

### G30: CALL clause + procedure support
- **Status**: Open
- **Description**: The harness does not support CALL clause execution,
  procedure registry/lookup, argument validation, or YIELD projections.
- **Affected scenarios**: `call1-1..call1-13`, `call2-1..call2-6`,
  `call3-1..call3-6`, `call4-1..call4-2`, `call5-1..call5-2`,
  `call5-3-1..call5-3-2`, `call5-4-1..call5-4-11`, `call5-5..call5-8`,
  `call6-1..call6-3`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add procedure registry stubs, CALL clause translation, and
  YIELD projection validation, then extend the runner to compare row outputs.

### G31: UNION / UNION ALL semantics
- **Status**: Open
- **Description**: The harness does not support UNION/UNION ALL composition,
  distinct handling, or column alignment validation across union branches.
- **Affected scenarios**: `union1-1..union1-5`, `union2-1..union2-5`,
  `union3-1..union3-2`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add UNION clause translation, row normalization, and
  column alignment validation for UNION / UNION ALL branches.

### G32: CREATE clause semantics
- **Status**: Open
- **Description**: The harness does not support `CREATE` clause write semantics
  (node/relationship creation, property/label side effects).
- **Affected scenarios**: `create1-1..create1-20`, `create2-1..create2-24`,
  `create3-1..create3-13`, `create4-1`, `create5-1..create5-5`,
  `create6-1..create6-10`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add CREATE support to the translation layer and validate
  side effects on nodes/edges/properties.

### G33: SET clause semantics
- **Status**: Open
- **Description**: The harness does not support `SET` clause updates for
  properties or labels, nor side effect validation.
- **Affected scenarios**: `set1-1..set1-11`, `set2-1..set2-3`,
  `set3-1..set3-8`, `set4-1..set4-5`, `set5-1..set5-5`,
  `set6-1..set6-21`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add SET support to the translation layer and validate
  property/label updates with side effects.

### G34: REMOVE clause semantics
- **Status**: Open
- **Description**: The harness does not support `REMOVE` clause updates for
  properties or labels, nor side effect validation.
- **Affected scenarios**: `remove1-1..remove1-7`, `remove2-1..remove2-5`,
  `remove3-1..remove3-21`
- **Workaround**: Mark as xfail and capture expected rows in the scenario
  metadata.
- **Next steps**: Add REMOVE support to the translation layer and validate
  property/label removals with side effects.

### G35: Expression evaluation (expressions suite)
- **Status**: Open
- **Description**: The harness does not evaluate expression-only queries from
  the TCK expressions suite (boolean/list/map/string/null/temporal/etc), so
  standalone RETURN expression semantics are not validated.
- **Affected scenarios**: `tck/features/expressions/**` (keys `expr-*`)
- **Workaround**: Mark as xfail with expected rows captured in scenario
  metadata.
- **Next steps**: Implement expression evaluation + row-level comparison in the
  runner (including list/map/temporal semantics) and hook into GFQL execution.

### G36: UseCases suite coverage
- **Status**: Open
- **Description**: The harness does not execute the TCK use case scenarios
  (counting subgraph matches, triadic selection), which depend on full pattern
  matching, optional matches, aggregation, and row-level projections.
- **Affected scenarios**: `tck/features/useCases/**` (keys `usecase-*`)
- **Workaround**: Mark as xfail with expected rows captured in scenario
  metadata.
- **Next steps**: Implement row-level evaluation + optional match semantics and
  translate use case patterns to GFQL once bindings and projection checks are
  in place.

## Notes
- Keep this doc aligned with `tests/cypher_tck/scenarios.py` and plan updates in
  `plans/cypher-tck-conformance/plan.md`.

## Coverage impact snapshot (tag-based)
Counts below are rough, tag-based xfail totals from `tests/cypher_tck/scenarios.py`.
Tags overlap, so totals are not additive.

- **Expression evaluation (G35)**: 2,599 xfail scenarios tagged `expr` (temporal
  1,069; list 219; string 32; typeConversion 47; boolean 150; comparison 84;
  aggregation 142).
- **Pipeline semantics (G13/G25/G26/G27)**: `with` 372, `orderby` 339, `limit`
  241, `skip` 19, `distinct` 21 (all xfail).
- **Row projection + bindings (G1/G2/G23)**: `match` 138 xfail, `return` 134
  xfail (row-level comparison and join/binding semantics missing).
- **Update clauses (G21/G32/G33/G34/G24)**: `create` 78, `merge` 77, `set` 53,
  `remove` 33, `delete` 45 (all xfail).
- **Procedures / union / unwind (G30/G31/G28)**: `call` 49, `union` 12,
  `unwind` 76 (all xfail).
- **UseCases suite (G36)**: `usecase` 30 (all xfail).

## Implementation lift + risk notes
Heuristic guidance for staging work: prioritize high-lift, low-risk items that
vectorize cleanly in pandas/cuDF. "Lift" reflects rough xfail counts above.

- **Row projection + bindings (G1/G2/G8/G9/G23)**: Lift high (`match` 138 +
  `return` 134 xfail) with spillover into many clause suites. Risk medium-high
  due to join semantics, alias scoping, null handling, and row order rules.
  Vectorization: join/merge + projection, but requires careful row normalization.
- **Pipeline semantics (G13/G25/G26/G27)**: Lift high (`with` 372, `orderby` 339,
  `limit` 241, `skip` 19, `distinct` 21). Risk medium: variable scoping and
  aggregation boundaries are tricky, but core operations map to
  sort/groupby/drop_duplicates/iloc.
- **UNWIND (G28)**: Lift medium (`unwind` 76). Risk medium: explode semantics
  are vector-friendly, but null/empty-list behavior and pipeline placement need
  care.
- **Expression evaluation (G35)**: Lift very high (`expr` 2,599) but risk high.
  Recommend staging: start with pure, deterministic scalar ops (literals,
  boolean/comparison, basic arithmetic) before lists/strings, then temporal and
  error semantics last.
- **OPTIONAL MATCH (G12)**: Lift high (touches many MATCH/WHERE scenarios) with
  high risk: left-join + null propagation across multi-hop bindings; sensitive
  to Cypher's tri-valued logic.
- **UNION (G31)**: Lift low-medium (`union` 12). Risk low-medium: concat +
  distinct with column alignment; vectorization straightforward once row
  projection normalization exists.
- **Update clauses (G21/G32/G33/G34/G24)**: Lift medium (node/edge creation and
  mutation: `create` 78, `merge` 77, `set` 53, `remove` 33, `delete` 45). Risk
  high: side effects, transactional semantics, and post-mutation validation.
- **CALL procedures (G30)**: Lift low-medium (`call` 49). Risk high: external
  procedure registry, signature/side-effect semantics.
- **UseCases suite (G36)**: Lift low-medium (`usecase` 30). Risk high: depends
  on OPTIONAL MATCH, aggregation, and row projections.
