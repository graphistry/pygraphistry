# GFQL Cypher TCK Conformance (tck-gfql)

The Cypher TCK conformance harness lives in the standalone repo:
https://github.com/graphistry/tck-gfql

Use it for translating openCypher TCK scenarios into GFQL and validating
results against the oracle, pandas, and optional cuDF runs.

## When to run
- After changes to GFQL core, predicates, models, or validation logic.
- When modifying GFQL execution semantics or return shapes.

## CI behavior
PyGraphistry CI includes a `tck-gfql` job that runs only when GFQL-related
paths change (see `.github/workflows/ci.yml`).

## Local run (from pygraphistry checkout)
```bash
git clone https://github.com/graphistry/tck-gfql.git
cd tck-gfql
PYGRAPHISTRY_PATH=/path/to/pygraphistry PYGRAPHISTRY_INSTALL=1 ./bin/ci.sh
```

## Notes
- The TCK clone is not vendored; see `tck-gfql/tests/cypher_tck/README.md`.
- Keep translations and gap analysis in the tck-gfql repo, not in pygraphistry.
