# AI Contribution Policy

This repository permits both AI-assisted and autonomous AI contributions.

Core rule: if you submit it, you own it. If you merge it, you approve it.

## 1) Contribution modes

### AI-assisted
A human author uses AI to draft/refactor/test/review/explain changes and remains responsible for the final PR.

### Autonomous
An agent/bot opens a PR with little or no human editing before opening the PR.

## 2) Eligibility rules

### AI-assisted eligibility
Allowed by default when the contributor:
- understands all material changes,
- can explain non-trivial changes in their own words,
- provides AI disclosure (including harness and model),
- runs appropriate local validation,
- keeps the PR reviewable.

### Autonomous eligibility
Baseline autonomous eligibility:
- linked issue has **one or more** of: `help wanted`, `good-first-issue`, `ai-friendly`,
- PR includes required autonomous AI disclosure,
- PR follows the required validation loop in this policy.

Standard repository merge gates still apply (required CI and human maintainer approval/merge).

Autonomous PRs may be closed without detailed review if these conditions are not met.

## 3) Labels: current reality and optional controls

### Current labels used in this repo
- `help wanted`: maintainers are open to outside contributions.
- `good-first-issue`: suitable onboarding scope; also allowed for autonomous work.
- `ai-friendly`: narrow, deterministic, low-risk, testable scope for AI-enabled contribution.
- `security`: security-sensitive work; requires extra maintainer caution.

### Optional control labels (if maintainers add them)
If labels like `ai:autonomous-ok`, `ai:autonomous-blocked`, `ai:autonomous` are introduced later, maintainers may use them to tighten or route autonomous flow. Until then, they are not required for eligibility.

## 4) Scope and risk boundaries for autonomous PRs

### Usually allowed
- docs fixes,
- focused test additions/corrections,
- small bug fixes with regression tests,
- narrow refactors without behavior change,
- small DX fixes,
- small measured performance improvements.

### Not autonomous-safe by default (needs explicit maintainer pre-approval)
- authn/authz logic,
- secret handling,
- security-sensitive paths,
- dependency/lockfile churn,
- release/publishing logic,
- CI/workflow logic,
- packaging/bootstrap changes,
- broad API changes,
- multi-area refactors,
- integrations requiring live credentials/paid external services.

## 5) PR size and stacking expectations

Prefer small PRs. For AI-generated change streams, prefer stacked PRs when it improves reviewability.

Reviewability targets are change-type dependent, not a single hard cap.

For docs-only, test-only, and narrow bug-fix PRs:
- prefer very small diffs,
- as a guideline, keep fixes focused (often <100 LOC implementation) with targeted tests/docs (can be larger, for example up to ~500 LOC), and
- keep file count small when possible.

For feature PRs:
- larger diffs may be appropriate,
- split work into coherent, reviewable slices (stacked PRs when helpful),
- keep each slice tied to explicit acceptance criteria and validation.

If a PR becomes too large/noisy to review effectively, maintainers may request it be split before deep review.

## 6) Disclosure requirements

### Material AI use (AI-assisted)
PR must include:
- tools/harness used,
- model(s) used,
- what AI changed,
- what human reviewed/rewrote,
- what was run locally,
- what was deferred to CI.

### Autonomous PRs (mandatory)
PR must include:
- linked issue,
- agent identity,
- tools/harness used,
- model(s) used,
- human merge owner,
- change summary,
- local checks run,
- checks deferred to CI,
- known risks/uncertainty,
- attached execution plan (see section 7).

### Commit attribution
For material AI use, include `Assisted-by:` trailer(s) in relevant commits or equivalent explicit PR disclosure.

## 7) Plan requirement for AI work

When viable, use `ai/prompts/PLAN.md` as the plan template.

Requirements:
- Keep plan file under `plans/<task>/plan.md`.
- Do **not** commit the plan file.
- Attach the plan to the PR (for example, paste plan content in PR comment/body or upload as PR attachment).

Purpose: make it auditable that specification conformance, testing, security review, and DRY/professionalism checks were performed.

## 8) Required validation loop (repeat-until-stable)

Do not run checks once and stop. Iterate until fresh analysis produces no significant new actions.

Required loop:
1. Confirm spec/acceptance criteria.
2. Audit AI suggestions; verify and prioritize before acting.
3. Use red->green flow when applicable:
   - reproduce failure or define intended behavior,
   - add/update failing test first,
   - make smallest fix,
   - add/tighten regression coverage.
4. Run lint/type/tests relevant to scope.
5. Run DRY/professionalism pass:
   - remove duplication,
   - improve naming/clarity,
   - avoid unnecessary branching,
   - split oversized logic when it improves readability.
6. Run security/robustness pass:
   - validate untrusted input,
   - avoid secret leakage,
   - avoid unsafe shell/deserialization,
   - verify filesystem/network/permission boundaries,
   - ensure intentional error handling.
7. Re-run checks after refactors.
8. Repeat steps 2-7 until no significant new findings remain.

## 9) Local vs CI validation

CI is required, but not a substitute for local validation.

PR must explicitly state:
- what ran locally,
- what did not run locally,
- what CI is expected to validate,
- environment limits that blocked broader local validation.

Repository-recommended local commands:

```bash
cd docker

# Small CPU iteration
WITH_BUILD=0 ./test-cpu-local-minimal.sh

# Lint + typecheck
WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh

# Full CPU validation
./test-cpu-local.sh

# GPU-affecting changes
./test-gpu-local.sh
```

## 10) Human review and merge rules

- AI-assisted PRs require a human author of record.
- Autonomous PRs require a named human merge owner.
- No autonomous PR self-approval or self-merge.
- Only human maintainers approve/merge autonomous PRs.
- Maintainers may pause/limit autonomous flow at any time.

## 11) Enforcement

Maintainers may close PRs without detailed review when they are out of scope, undisclosed, low-signal, too large/noisy, or violate this policy/security boundaries.

Repeated abuse may lead to loss of contribution privileges.

## 12) Suggested disclosure block

```md
## AI Disclosure
- Mode: AI-assisted / Autonomous / No material AI use
- Agent identity:
- Tools/harness used:
- Model(s) used:
- Human author of record:
- Human merge owner:
- Linked issue:
- What AI changed:
- What I personally reviewed or rewrote:
- Local checks run:
- Deferred to CI:
- Plan attached (path or PR comment link):
- Known risks / uncertainty:
```

## 13) Policy intent

This policy is pro-quality, pro-accountability, and pro-automation in bounded places.

AI should reduce mechanical work without shifting reviewer burden, increasing security risk, or lowering quality standards.
