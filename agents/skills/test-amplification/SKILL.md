---
name: test-amplification
description: Methodical bug-driven test amplification for pygraphistry features. Use when hardening a feature area via user-workflow exploration, 5-Whys retrospectives, bug-taxonomy derivation, concrete test planning, implementation, and safety-gated validation.
---

# Test Amplification Skill

## When to Use

- A pygraphistry feature area is stable enough to use but still yields recurring bugs.
- You need to grow tests from evidence, not random additions.
- You want reusable patterns that apply beyond one subsystem (for example, beyond GFQL).

## Goal

Turn historical and newly found bugs into:
1. A reusable bug taxonomy.
2. Prioritized concrete test tasks.
3. Implemented tests (plus minimal fixes when needed).
4. Clear validation evidence and checkpoint commits.

## Required Inputs

- Current branch context.
- Target feature area(s).
- Existing test locations for those areas.
- Safety constraints (for example: pure vectorized path, pandas+cudf compatibility, remote protocol constraints).

## Multi-Round Artifact Rules (Required)

This workflow is usually iterative. Keep artifacts cleanly separated by round, plus cumulative rollups.

### Round IDs

- Use zero-padded round IDs: `round-001`, `round-002`, ...
- Keep a stable task root:
  - `plans/<task>/`

### Round-Scoped Artifacts

Per round, write under:
- `plans/<task>/rounds/round-00N/`

Recommended files:
- `user_testing_playbook.md`
- `feature-risk-analysis.md`
- `discovery-matrix.md`
- `bug-retro-5whys.md`
- `taxonomy-and-task-plan.md`
- `execution-report.md`
- `metrics.md`

### Cumulative Artifacts

Maintain cumulative rollups under:
- `plans/<task>/cumulative/`

Recommended files:
- `feature-risk-analysis-cumulative.md`
- `bug-retro-5whys-cumulative.md`
- `taxonomy-cumulative.md`
- `round-metrics-ledger.md`
- `continue-or-halt-log.md`

### Naming Conventions

- Prefer deterministic names inside round folders (no random suffixes).
- If date is needed, use ISO format: `YYYY-MM-DD`.
- Every round must update:
  1. one round-scoped `metrics.md`
  2. cumulative `round-metrics-ledger.md`
  3. cumulative `continue-or-halt-log.md`

## Parallel Subagent Protocol (Recommended When Available)

If parallel subagents are available (for example, Claude workers), use them with fixed artifact locations so work is crash-resumable.

### Round folder layout for parallel work

Under each round:
- `plans/<task>/rounds/round-00N/findings/`
- `plans/<task>/rounds/round-00N/findings/agent-01/`
- `plans/<task>/rounds/round-00N/findings/agent-02/`
- `plans/<task>/rounds/round-00N/findings/agent-03/`
- `plans/<task>/rounds/round-00N/merge/`

Required per-agent files:
- `task.md` (assigned scope and constraints)
- `findings.md` (bugs, repros, evidence)
- `proposed-tests.md` (concrete test additions)
- `status.json` (machine-readable state: `in_progress|done|blocked`)

Required merge files:
- `merge/triage.md` (dedupe + severity + priority)
- `merge/final-task-plan.md` (selected tasks for implementation)

### Assignment strategy

- Assign non-overlapping subsystems per agent (for example: remote/auth, connectors, compute, policy).
- Give each agent the same required scorecard format.
- Merge only after all `status.json` files are `done` or explicitly `blocked`.

## Core Workflow

### 0) User-Workflow Exploration (Required)

Use a user-testing exploration pass before adding tests.

Build 3-5 representative user stories from the feature area. For each story capture:
- workflow steps users run,
- expected outcomes,
- likely friction points,
- representative test files,
- likely bug classes.

Artifact:
- `AI_PROGRESS/<task>/user_testing_playbook.md`

### 1) Feature Surface Inventory

Inventory what changed (or is risky) in this area:
- API/AST surface
- runtime/executor/evaluator paths
- validator/schema scanning paths
- transport/session/metadata paths (if remote)
- existing tests and known blind spots

Artifact:
- `plans/<task>/feature-risk-analysis-<date>.md`

### 2) Discovery Matrix

Run a small probe matrix that compares:
- validator accept/reject,
- runtime behavior,
- expected semantics.

Prioritize mismatches:
- validator accepts + runtime fails,
- validator rejects + runtime supports,
- runtime succeeds with wrong semantics.

### 3) 5-Whys Per Bug Class

For each bug class, record:
- Symptom
- How found
- Why #1..#5
- Why prior tests missed it
- Root cause family

Artifact:
- `plans/<task>/bug-retro-5whys-<date>.md`

### 4) Derive Abstract Taxonomy

Convert bug roots into reusable categories. Typical categories:
- precedence/token boundary issues,
- lexical scanning/quote-awareness issues,
- validator/runtime parity drift,
- schema inference lexical issues,
- capability-shape failfast gaps,
- null/empty semantic gaps,
- state/session lifecycle issues,
- remote metadata hydration/transport contract issues.

### 5) Convert to Concrete Tasks

For each category define concrete tests:
- exact file(s),
- test names,
- expected behavior,
- whether minimal fix is expected.

Prefer colocated tests in existing subsystem files.

### 6) Implement Tests + Minimal Fixes

Rules:
- Prefer red->green (add test, then fix).
- Keep fixes narrowly scoped to test-driven findings.
- No broad refactors during amplification phases.

### 7) DRY Pass

Before final validation:
- remove duplication in new tests,
- keep readability high,
- avoid helper sprawl.

### 8) Safety Gates (Required)

Run and record outputs.

Baseline static gates:
```bash
python -m py_compile <touched_files>
ruff check <touched_files>
mypy --ignore-missing-imports --follow-imports=skip --explicit-package-bases <touched_files>
```

Focused tests:
```bash
PYTHONPATH=. pytest -q <focused_test_files>
```

Broad sanity (env-available):
```bash
PYTHONPATH=. pytest -q <broader_test_set>
```

Best-effort backend parity (if applicable):
```bash
PYTHONPATH=. pytest -q <tests> -k cudf
```

### 9) Round Scorecard + Continue/Halt Decision (Required)

Each round must publish a scorecard and an operator-facing decision note.

Required scorecard fields:
- `tests_added` (new test cases)
- `tests_modified` (existing tests changed)
- `new_bug_classes_found`
- `new_bug_instances_found`
- `bug_instances_fixed`
- `bug_instances_open` (found - fixed)
- `bug_fix_rate_pct` = `fixed / found * 100` (if found > 0)
- `new_bug_class_fix_rate_pct` = `classes_fixed / classes_found * 100` (if classes_found > 0)
- `tests_per_fix` = `tests_added / max(fixed, 1)`
- `focus_gate_pass` (yes/no)
- `broad_gate_pass` (yes/no/partial + env gaps)

Decision note must include:
- `continue` or `halt`
- one-paragraph rationale based on marginal ROI trend
- explicit next highest-ROI bug classes if continuing

Important:
- This is a **required decision record**, not a forced halt.
- If the operator/user asked for multiple rounds, choose `continue` unless there is a blocker or clear diminishing-return stop condition.
- `halt` should be used when:
  - explicit user directive to stop,
  - no actionable high-ROI targets remain,
  - hard blocker prevents meaningful progress.

### 10) Checkpointing + Plan Update

After gates pass:
```bash
git add <touched_files>
git commit -m "phase-<id>: <short amplification summary>"
```

Update plan with:
- commands run,
- outputs,
- commit hash,
- env gaps.
- round scorecard values,
- continue/halt decision.

## Domain Packs (PyGraphistry-Specific)

Use the pack(s) matching the work area.

### A) Visualization/Upload/Render
Representative files:
- `graphistry/tests/test_plotter.py`
- `graphistry/tests/test_arrow_uploader.py`
- `graphistry/tests/render/test_resolve_render_mode.py`
- `graphistry/tests/test_dataset_id_invalidation.py`

Bug styles:
- encoding metadata loss,
- dtype coercion regressions,
- dataset lifecycle invalidation bugs.

Useful test styles:
- metadata round-trip,
- dtype matrix tests,
- lifecycle mutation/invalidation tests.

### B) Remote/Auth/Session/Trace
Representative files:
- `graphistry/tests/test_chain_remote_auth.py`
- `graphistry/tests/test_gfql_remote_metadata.py`
- `graphistry/tests/test_gfql_remote_persistence.py`
- `graphistry/tests/test_trace_headers_behavior.py`
- `graphistry/tests/test_certificate_validation_session.py`

Bug styles:
- token/session leakage across clients,
- metadata hydration mismatches,
- missing trace/cert propagation.

Useful test styles:
- request contract assertions,
- multi-client isolation tests,
- hydration parity tests.

### C) Connectors/Ingestion
Representative files:
- `graphistry/tests/test_kusto.py`
- `graphistry/tests/test_spanner.py`
- `graphistry/tests/test_gremlin.py`
- `graphistry/tests/test_tigergraph.py`
- `graphistry/tests/test_nodexl.py`
- `graphistry/tests/test_gexf.py`

Bug styles:
- return-shape ambiguity (single vs multiple tables),
- dynamic column flattening drift,
- credential/no-credential path regressions.

Useful test styles:
- mocked connector contracts,
- optional live credential smoke tests,
- malformed payload tests.

### D) Graph Compute (Hop/Chain/GFQL)
Representative files:
- `graphistry/tests/compute/test_hop.py`
- `graphistry/tests/compute/test_chain.py`
- `graphistry/tests/compute/test_call_operations.py`
- `graphistry/tests/compute/gfql/test_row_pipeline_ops.py`

Bug styles:
- precedence/parser boundary bugs,
- validator/runtime parity drift,
- backend parity divergence.

Useful test styles:
- semantic operator matrix,
- validator/runtime parity table,
- pandas baseline + cudf best-effort checks.

### E) Policy/Governance Hooks
Representative files:
- `graphistry/tests/test_policy_*.py`
- `graphistry/tests/test_policy_integration.py`
- `graphistry/tests/compute/gfql/test_policy_shortcuts.py`

Bug styles:
- phase ordering regressions,
- closure state leakage,
- exception contract drift.

Useful test styles:
- phase-state transition tests,
- negative-path policy exception tests,
- cross-operation coverage matrix.

### F) Layouts/Plugins/Optional Deps
Representative files:
- `graphistry/tests/layout/*`
- `graphistry/tests/plugins/*`
- `graphistry/tests/test_layout.py`

Bug styles:
- optional-dependency behavior drift,
- plugin fallback failures,
- algorithm-specific edge-case regressions.

Useful test styles:
- conditional import/fallback tests,
- deterministic small-graph golden tests,
- edge-case topology fixtures.

## Plan Phase Skeleton

Use this in `plan.md`:

```markdown
### Phase X.Y: Test Amplification Cycle
**Status:** 🔄 IN_PROGRESS

#### X.Y.1 User-workflow exploration
#### X.Y.2 Feature/risk inventory
#### X.Y.3 Discovery matrix
#### X.Y.4 5-Whys capture
#### X.Y.5 Taxonomy derivation
#### X.Y.6 Concrete task plan
#### X.Y.7 Implement tests (+minimal fixes)
#### X.Y.8 DRY pass
#### X.Y.9 Validation
#### X.Y.10 Round scorecard + continue/halt decision
#### X.Y.11 Checkpoint
```

## Round Scorecard Template

Use this in `plans/<task>/rounds/round-00N/metrics.md`:

```markdown
# Round 00N Metrics

- tests_added:
- tests_modified:
- new_bug_classes_found:
- new_bug_instances_found:
- bug_instances_fixed:
- bug_instances_open:
- bug_fix_rate_pct:
- new_bug_class_fix_rate_pct:
- tests_per_fix:
- focus_gate_pass:
- broad_gate_pass:
- env_gaps:

## Continue/Halt
- decision: continue | halt
- rationale:
- next_targets:
```

## Anti-Patterns

- Jumping directly into test writing without bug-class derivation.
- Validator-only checks or runtime-only checks without parity coverage.
- Over-generalized guidance with no concrete files/patterns.
- Large refactors during an amplification pass.

## Done Criteria

- User-workflow exploration artifact exists.
- Feature/risk + 5-Whys artifacts exist (round + cumulative).
- Taxonomy -> concrete tasks mapping documented.
- New high-ROI tests landed (minimal fixes only as needed).
- Focused/broad gates pass or env limitations are clearly documented.
- Round scorecard and continue/halt decision are recorded.
- Plan and commit checkpoints are updated.
