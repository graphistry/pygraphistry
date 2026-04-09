# Cypher Frontend CI Gates

This document defines the CI checks introduced in PR-3 (`#1102`, part of `#1101`) for cypher frontend and IR changes.

## Gate names

- `cypher-frontend-strict-typing (py3.12)`
  - Runs `mypy --strict --follow-imports=skip` on:
    - `graphistry/compute/gfql/ir/*.py`
    - `graphistry/compute/gfql/frontends/cypher/binder.py` (when present)

- `cypher-frontend-differential-parity (py3.12)`
  - Runs differential/parity harness checks:
    - Prefer these test patterns when present:
      - `tests/gfql/ref/test_m1_*.py` (legacy filename prefix)
      - `tests/gfql/ref/test_differential*.py`
      - `graphistry/tests/compute/gfql/cypher/test_m1_*.py` (legacy filename prefix)
      - `graphistry/tests/compute/gfql/cypher/test_differential*.py`
    - Falls back to `tests/gfql/ref/test_enumerator_parity.py`.

- `cypher-frontend-ci-gates`
  - Aggregator gate that requires:
    - `cypher-frontend-strict-typing (py3.12)`
    - `cypher-frontend-differential-parity (py3.12)`
    - `test-minimal-python` (full-suite sentinel gate)

## Intended branch-protection required checks

For cypher frontend / IR PRs, require:

1. `cypher-frontend-ci-gates`
2. Existing baseline required checks (repo default set)

This ensures strict typing + differential/parity + full-suite sentinel stay enforced together.
