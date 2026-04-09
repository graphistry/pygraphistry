## Summary

- [ ] Linked issue(s): `Closes #...`
- [ ] Scope is explicit (non-goals included when relevant)

## Validation

- [ ] Local lint/type/tests run for touched scope
- [ ] CI is green

## Cypher Frontend CI Evidence (when PR touches cypher frontend / IR scope)

- [ ] `cypher-frontend-strict-typing (py3.12)` passed (strict typing gate)
- [ ] `cypher-frontend-differential-parity (py3.12)` passed (trust-but-verify gate)
- [ ] `cypher-frontend-ci-gates` passed (includes `test-minimal-python` sentinel)
- [ ] PR body includes links/screenshots/log snippets for any non-obvious gate evidence
