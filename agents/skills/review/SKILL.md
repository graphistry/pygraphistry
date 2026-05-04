---
name: review
description: |
  Structured PR review for pygraphistry. Input: PR number/branch (default current branch PR).
  Output: findings and convergence artifacts under plans/<task>/.
  Method: multi-wave, evidence-first review across spec, correctness, tests, security,
  code quality, DRY, concurrency, performance, architecture, operability, and conventions.
---

# PR Review (pygraphistry)

## Invocation

```text
/review [<PR-number-or-branch>] [mode=findings|pr-comments|both] [fixes=deferred|inline]
```

Defaults:
- Target: current branch PR (`gh pr view --json number,headRefName,baseRefName,url`)
- `mode=findings`
- `fixes=deferred`

`mode`:
- `findings`: local artifacts only under `plans/<task>/`
- `pr-comments`: draft locally, post only after explicit user confirmation in the same session
- `both`: run findings then comment flow

`fixes`:
- `deferred`: read-only review
- `inline`: after each converged wave, apply confirmed `BLOCKER`/`IMPORTANT` fixes in separate commits

## Runtime Assumptions

- Run from repo root.
- `gh` is authenticated (`gh auth status`).
- Local branch reflects PR head (`origin/<base>...HEAD` matches intended review scope).
- `plans/` is local working memory and normally gitignored.

## Plan-First Requirement

Always use the plan skill flow:

1. If `plans/<task>/plan.md` exists, reuse it and append a review section.
2. Else reload `.agents/skills/plan/SKILL.md` and create `plans/<task>/plan.md`.
3. Record PR metadata, `mode`, `fixes`, and timestamp.
4. Reload plan before every step; update plan immediately after every step.

`<task>`: prefer `review-pr-<N>` or `<branch>-review`.

## Phase 0: Resolve Scope + Stack Context

1. Resolve PR context (number/title/url/head/base).
2. Record stack context:

```bash
gh pr view <PR> --json baseRefName,headRefName,title,body
gh pr list --base <headRefName>
```

3. If stacked, explicitly mark out-of-scope upstream/downstream work in `plan.md`.
4. Set diff range reference: `origin/<base>...HEAD`.

## Phase 1: Research Criteria Before Findings

Create `plans/<task>/research/` with:
- `context.md`
- `policies.md`
- `credentials.md`
- `canvas-<dimension>.md` (only for dimensions that apply)

### 1a) Collect context + changed files

```bash
gh pr view <PR> --json number,title,headRefName,baseRefName,url,body
git fetch origin <baseRefName> <headRefName>
git diff --name-only origin/<base>...HEAD
git log --oneline origin/<base>..HEAD
```

Record linked issue/spec refs from PR body and summarize PR intent.

### 1b) Discover applicable repo rules

Always inspect:
- `AGENTS.md`, `DEVELOP.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `README.md`, `CHANGELOG.md`
- `docs/source/**` relevant to changed areas
- CI/workflow context under `.github/workflows/` when checks/publish behavior are touched
- Tooling configs (`pyproject.toml`, `mypy.ini`, `pytest.ini`) and helper scripts in `bin/`

Walk up from each changed file to repo root and include nearby `.md` guidance.

### 1c) Credentials gate (always, early)

Run before wave analysis:

```bash
git diff origin/<base>...HEAD -- '*.env*' 'custom.env*' 'docker-compose*.y*ml' '*.conf' '*.config.*' | \
  grep -iE '(api_key|secret|token|password|bearer|authorization|aws_|azure_|openai_|anthropic_).{0,4}=' || \
  echo "[clean] no obvious credential strings"

git diff origin/<base>...HEAD | \
  grep -oE '(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._-]{20,})' | head
```

If any likely secret is found, raise `BLOCKER` and stop until surfaced.

## Phase 2: Multi-Wave Review Loop

Run waves until 2 consecutive waves show no significant advance.

Severity:
- `BLOCKER`: merge must not proceed
- `IMPORTANT`: should fix before merge
- `SUGGESTION`: non-blocking improvement

Suggested folder layout:

```text
plans/<task>/waves/wave-<N>/
  <dimension>/findings-<file-slug>.md
  <dimension>/report.md
  adversarial/<finding-id>.md
  adversarial/report.md
  wave-report.md
```

### Wave gate (start of each wave)

1. Reload `.agents/skills/plan/SKILL.md`
2. Reload `plans/<task>/plan.md`
3. Confirm current diff SHA and wave number
4. If prior wave had substantive findings, perform both:
   - targeted amplification pass on prior findings/fixes
   - clean-slate full pass on current diff

### Dimensions

Apply only relevant dimensions per PR:
- Spec conformance
- Correctness
- Testing
- Security
- Code quality
- DRY / reuse
- Concurrency / parallelism
- Performance
- Architecture
- Operability
- Repo conventions

Guidance:
- Keep dimensions independent (avoid blended "general review" prompts).
- For file-heavy diffs, parallelize by `(dimension, file)` and aggregate.
- Verify pre-existing patterns are not misreported as regressions:

```bash
git show origin/<base>:<file>
```

### Pygraphistry-specific review checks

- Test coverage should mirror changed areas in `graphistry/tests/**`.
- Prefer behavioral tests over implementation-detail assertions.
- Run focused validation before escalating severity when feasible:

```bash
python -m pytest -q [targeted_test]
./bin/ruff.sh [changed_path_or_pkg]
./bin/typecheck.sh [changed_path_or_pkg]
```

- For GPU-affecting PRs, require GPU-path validation evidence:
  - Local GPU path: `cd docker && ./test-gpu-local.sh [targeted_test_or_path]`
  - No local GPU available: run equivalent GPU validation on `dgx-spark` and record exact command + output artifact path in wave evidence.
- If startup/runtime claims are made, verify entrypoints/scripts in `bin/` and workflow behavior.
- For docs-only PRs, prioritize spec/documentation accuracy and navigability (toctree links, anchors, cross-refs).

### Per-file findings format

```markdown
## <file>
### Findings
- [BLOCKER] <description> — <file>:<line>
- [IMPORTANT] <description> — <file>:<line>
- [SUGGESTION] <description> — <file>:<line>

### Evidence
- <diff/code/test evidence with file:line references>
```

### Adversarial pass (required)

For each finding, attempt disproof and write verdict:
- `CONFIRMED`
- `DOWNGRADED`
- `REJECTED`

Only `CONFIRMED` and `DOWNGRADED` findings carry forward.

Each adversarial file should include:
- original claim
- disprove attempt
- verdict
- concrete proof (`file:line`, diff excerpt, or test output)

### Wave summary

Each wave writes `wave-report.md` with:
- inputs (diff SHA/range, dimensions, files)
- post-adversarial counts by severity
- fixes applied (if `fixes=inline`)
- finding resolution ledger (`FIXED`, `UNFIXED`, `DEFERRED`)
- convergence signal

## Phase 3: Convergence Report

Write `plans/<task>/final-report.md` with:
- summary (waves run, convergence status)
- per-wave table (new findings + advance signal)
- resolution matrix (status of non-rejected findings)
- sections: `Blockers`, `Important`, `Suggestions`, `Rejected/False Positives`, `Methodology`

## Phase 4: Output Behavior

`findings` mode:
- Keep full artifacts in `plans/<task>/`
- Provide concise terminal summary with:
  - target + mode + fixes policy
  - waves + convergence status
  - per-wave counts
  - `FIXED` vs `UNFIXED`/`DEFERRED`
  - path to `final-report.md`

`pr-comments` mode:
1. Draft local comments in `plans/<task>/comment-drafts/`
2. Ask for explicit confirmation
3. Post inline + top-level comments with `gh`

`both` mode:
- Complete findings artifacts first, then comment flow.

## Guardrails

- `fixes=deferred`: read-only; do not edit source files.
- `fixes=inline`: edit only files in PR diff and only for confirmed `BLOCKER`/`IMPORTANT` findings.
- Never skip lint/tests when applying inline fixes.
- Never force-push from review flow.
- Never post PR comments without explicit user confirmation.
- Always run credentials scan in Phase 1 before waves.
- Prefer exact file/line evidence over broad statements.
- If uncertain whether issue is new, compare against `origin/<base>` before filing.
