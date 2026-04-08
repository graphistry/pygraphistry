# M1 CI Gates (Wave 1)

This document defines the CI gate names introduced for M1 wave-1 (`#1101`) and PR-3 (`#1102`).

## Gate names

- `m1-strict-typing (py3.12)`
  - Runs `mypy --strict --follow-imports=skip` on:
    - `graphistry/compute/gfql/ir/*.py`
    - `graphistry/compute/gfql/frontends/cypher/binder.py` (when present)

- `m1-differential-parity (py3.12)`
  - Runs differential/parity harness checks:
    - Prefer `tests/gfql/ref/test_m1_*.py` and `tests/gfql/ref/test_differential*.py` when present.
    - Falls back to `tests/gfql/ref/test_enumerator_parity.py`.

- `m1-ci-gates`
  - Aggregator gate that requires:
    - `m1-strict-typing (py3.12)`
    - `m1-differential-parity (py3.12)`
    - `test-minimal-python` (full-suite sentinel gate)

## Intended branch-protection required checks

For M1 binder/refactor PRs, require:

1. `m1-ci-gates`
2. Existing baseline required checks (repo default set)

This ensures strict typing + differential/parity + full-suite sentinel stay enforced together.
