# Cypher TCK conformance tests (GFQL)

This suite translates a subset of the openCypher TCK into GFQL AST/wire protocol
queries and validates results against the reference enumerator and pandas, with
optional cuDF runs when enabled.

## Source of truth
- openCypher TCK: https://github.com/opencypher/openCypher/tree/main/tck
- Local clone (gitignored): `plans/cypher-tck-conformance/tck`

## Provenance
- Clone date: 2025-12-29
- Repo commit: `59edf2e1c17b845bf97c334ed06b2eb780950c13`
- License: Apache License 2.0 (`plans/cypher-tck-conformance/tck/LICENSE`)

## Goals
- Translate supported Cypher scenarios into GFQL equivalents.
- Run each translated case on:
  - Reference enumerator (oracle)
  - `engine='pandas'`
  - `engine='cudf'` (only when `TEST_CUDF=1` and cudf is available)
- Record unsupported scenarios with explicit xfail/skip reasons and capability tags.
- Preserve traceability to the original Cypher query and expected results.

## Running
```bash
pytest tests/cypher_tck -xvs
TEST_CUDF=1 pytest tests/cypher_tck -xvs
```

## Notes
- The TCK repo is not vendored; use the local clone under `plans/`.
- Each translated scenario should include a reference back to the TCK path,
  the original Cypher, and the expected rows or aggregates.
- Track feature gaps and workarounds in `tests/cypher_tck/GAP_ANALYSIS.md`.
