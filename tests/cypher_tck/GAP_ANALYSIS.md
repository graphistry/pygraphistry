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
  multiple MATCH clauses or validate row-level projections in `RETURN`.
- **Affected scenarios**: `match1-5` (Match1 [5])
- **Workaround**: Mark as xfail with expected rows captured in the scenario.
- **Next steps**: Add a result comparator for row projections and introduce a
  cartesian product mechanism (likely outside GFQL core) for multiple MATCH
  clauses.

### G2: Scenario-level row expectations
- **Status**: Open
- **Description**: Scenario `Expected.rows` is recorded but not validated by the
  runner (only node/edge IDs are compared).
- **Affected scenarios**: Any scenario that only specifies rows, or uses RETURN
  expressions beyond node/edge identity.
- **Workaround**: Use node_ids/edge_ids when possible; otherwise xfail and note
  row expectations.
- **Next steps**: Extend the runner with row extraction and normalization (e.g.,
  alias binding, ordering rules, null handling).

### G3: CREATE parser coverage
- **Status**: Partial
- **Description**: The minimal CREATE parser handles nodes, labels, simple
  properties, and basic relationships, but does not parse relationship
  properties, variable-length relationships, or advanced Cypher constructs.
- **Affected scenarios**: Any scenario whose setup includes relationship
  properties or complex patterns.
- **Workaround**: Manually craft fixtures or extend parser incrementally.
- **Next steps**: Support relationship properties and additional pattern forms
  as needed by the next wave of scenarios.

## Notes
- Keep this doc aligned with `tests/cypher_tck/scenarios.py` and plan updates in
  `plans/cypher-tck-conformance/plan.md`.
