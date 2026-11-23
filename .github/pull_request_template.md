## Summary

- [ ] Linked issue(s): `Closes #...` (the issue you claimed)
- [ ] Scope is explicit (non-goals included when relevant); one topic, about 400 changed lines or 10 files at most
- [ ] Base branch: `develop` for features and fixes; `master` only for a hotfix (say why); docs-only is fine on `develop`
- [ ] AI assistance: this pull request was written with the help of an AI coding agent (yes / no, and which)

## Validation

Paste the commands you ran and the last lines of their output (not just a tick).

- [ ] `./bin/lint.sh` passes
- [ ] `./bin/typecheck.sh` passes
- [ ] The test that failed before this change and passes after it (name it: `path::test_name`; a fix or a feature needs one), and the tests for the touched area still pass: `./bin/pytest.sh <path>` or `./bin/check-gfql.sh` for GFQL
- [ ] `CHANGELOG.md` has a line for this change

## Engines tested (GFQL changes: `graphistry/compute/`, `graphistry/tests/compute/gfql/`)

- [ ] pandas
- [ ] Polars (CPU)
- [ ] cuDF (`./bin/check-gfql.sh --cudf`, or "no GPU: maintainers please cover")
- [ ] polars-gpu (`./bin/check-gfql.sh --polars-gpu`, or "no GPU: maintainers please cover")
- [ ] New tests run on every engine the change can affect

## Benchmark (required when a GFQL change can affect performance or memory; otherwise write "n/a")

Before and after on the changed case and on two paths you did not change, from the recipe in `benchmarks/gfql/README.md`:

```
(paste the table here)
```

## Cypher Frontend CI Evidence (only when the PR touches `graphistry/compute/gfql/cypher/` or the IR; otherwise leave unchecked)

- [ ] `cypher-frontend-strict-typing (py3.12)` passed (strict typing gate)
- [ ] `cypher-frontend-differential-parity (py3.12)` passed (trust-but-verify gate)
- [ ] `cypher-frontend-ci-gates` passed (includes `test-minimal-python` sentinel)
- [ ] PR body includes links/screenshots/log snippets for any non-obvious gate evidence
