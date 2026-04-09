# AI Contribution Policy

This repository allows both AI-assisted contributions and limited autonomous AI contributions.

Core rule: if you submit it, you own it. If you merge it, you approve it.

## 1. Modes of AI contribution

### AI-assisted contributions
AI-assisted pull requests are allowed by default when the contributor:
- understands every material change,
- can explain the change in their own words,
- discloses material AI use,
- runs the required local checks,
- keeps the PR small and reviewable, and
- responds to maintainers personally.

### Autonomous AI contributions
Autonomous pull requests are allowed only for explicitly pre-approved issues and only within the limits of this policy.

An autonomous PR is allowed only when all of the following are true:
- the linked issue is labeled `help wanted`, `ai-friendly`, and `ai:autonomous-ok`,
- the linked issue is not labeled `good first issue`, `security`, or `ai:autonomous-blocked`,
- the PR is labeled `ai:autonomous`,
- the PR includes the required AI disclosure block,
- the change stays within the autonomous scope limits in this policy,
- all required CI checks pass, and
- a human maintainer approves and merges the PR.

Autonomous PRs that do not meet these conditions may be closed without detailed review.

## 2. Definitions

### AI-assisted
A human uses AI to help draft, refactor, test, review, or explain code, tests, docs, or issue text, and the human remains the author of record.

### Autonomous
A bot or agent proposes and opens a PR with little or no human editing before the PR is opened.

### Material AI use
AI generated or substantially rewrote content that remains in the final PR, or materially influenced the implementation.

### Human merge owner
For every autonomous PR, one human must be clearly named in the PR body as the person accountable for review quality and merge decisions.

## 3. Relationship to other repository files

- `AGENTS.md` explains how agents should work in this repository. It is operational guidance for tools and agents.
- `AI_POLICY.md` is the governance policy for whether and how AI contributions are allowed.
- `CONTRIBUTING.md` is the contributor entry point and should link here.
- `SECURITY.md` controls vulnerability reporting and takes precedence for security issues.

## 4. Issue labels and issue eligibility

### Label meanings
- `ai-friendly`: the issue is narrow, deterministic, testable, and low-risk enough for AI help.
- `ai:autonomous-ok`: a maintainer explicitly allows autonomous PRs for this issue.
- `ai:autonomous-blocked`: autonomous PRs are not allowed.
- `help wanted`: maintainers are actively open to outside contributions.
- `good first issue`: reserved as a human learning and onboarding path.

### Important rule for `good first issue`
`good first issue` is allowed for human contributors and AI-assisted work, but not for autonomous agents.

### Maintainer checklist for marking an issue `ai-friendly`
Only mark an issue `ai-friendly` when all of the following are true:
- acceptance criteria are concrete,
- the expected diff is small and bounded,
- the change can be tested without secrets or private infrastructure,
- there is no unresolved architectural or product-design ambiguity,
- success can be validated by unit tests, targeted integration tests, or a narrow benchmark,
- the issue does not require dependency, release, auth, or security-sensitive changes.

## 5. Scope limits for autonomous PRs

### Allowed by default
Autonomous PRs may handle only low-risk work such as:
- documentation fixes,
- test additions or corrections,
- small bug fixes with focused regression tests,
- narrow refactors that do not change behavior,
- small developer-experience fixes,
- small performance improvements with measurable evidence.

### Not allowed without explicit maintainer pre-approval
Autonomous PRs must not modify the following unless the issue explicitly authorizes it and a maintainer has pre-approved the scope:
- authentication or authorization logic,
- secret handling,
- security-sensitive code paths,
- dependency upgrades or lockfile churn,
- release or publishing logic,
- GitHub Actions or other CI workflows,
- packaging, installation, or environment bootstrap logic,
- broad API changes,
- multi-area refactors,
- connectors or integrations that require live credentials or paid external services to validate.

### Reviewability limits
By default, autonomous PRs should stay under roughly:
- 5 changed files, and
- 300 net changed lines,
excluding generated snapshots or mechanical fixture updates.

Anything larger requires explicit maintainer signoff before review.

## 6. Disclosure and authorship

### For AI-assisted PRs
If there was material AI use:
- include an `AI Disclosure` section in the PR body,
- describe what AI was used for,
- state what you personally reviewed or rewrote,
- state what you ran locally.

### For autonomous PRs
The PR body must include:
- issue number,
- agent identity,
- human merge owner,
- tools/models used,
- summary of what the agent changed,
- local checks run,
- checks deferred to CI,
- known risks or uncertainty.

### Commit attribution
For material AI use, add an `Assisted-by:` trailer to the relevant commit(s) or otherwise clearly disclose tool use in the PR body.

Examples:
- `Assisted-by: Claude`
- `Assisted-by: ChatGPT`
- `Assisted-by: GitHub Copilot`

## 7. Required quality workflow

Both AI-assisted and autonomous contributions must follow this quality loop:

1. Reproduce the bug or specify the intended behavior.
2. Add or update a test that fails before the fix, when applicable.
3. Make the smallest change that makes the test pass.
4. Add or tighten regression coverage so the failure does not come back.
5. Run lint, type checks, and relevant tests.
6. Do a DRY and simplification pass:
   - remove duplication,
   - improve naming,
   - reduce unnecessary branching,
   - split oversized functions if it improves clarity,
   - avoid mixing cleanup with unrelated behavior changes.
7. Re-run lint, type checks, and relevant tests after each refactor pass.
8. Perform a security and robustness review:
   - validate untrusted input,
   - avoid logging secrets,
   - avoid unsafe shell execution,
   - avoid unsafe deserialization,
   - check file system, network, and permission boundaries,
   - confirm failures are handled intentionally.
9. If performance is part of the claim, include a benchmark or other measurable before/after evidence.

Do not stop at "tests passed once." The expectation is to iterate until no known bugs remain and the implementation is as simple as it can be without changing behavior.

## 8. Local vs CI testing requirements

CI is a required gate, not a substitute for zero local validation.

### Required local validation before ready-for-review
Run the best applicable local command(s) for the scope of your change.

Recommended local commands for this repository:

```bash
cd docker

# Small CPU-scoped iteration
WITH_BUILD=0 ./test-cpu-local-minimal.sh

# Lint + typecheck only
WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh

# Full CPU validation before commit for non-trivial changes
./test-cpu-local.sh

# GPU validation for GPU-affecting changes before merge
./test-gpu-local.sh
```

### Minimum expectations by change type
- Docs-only changes: explain why code execution was unnecessary.
- Small code changes: run at least the relevant minimal local test path.
- Behavioral changes or bug fixes: add/update tests and run the relevant local suite.
- GPU-affecting changes: run targeted GPU validation before merge.
- Connector or integration changes: run the narrowest relevant integration path available.

### Required PR reporting
Every AI-assisted or autonomous PR must explicitly say:
- what was run locally,
- what was not run locally,
- what is expected to be covered by CI,
- any environment limitations that prevented broader local validation.

## 9. Reviewer interaction rules

- Write the PR description yourself or edit it until it reflects your own understanding.
- Be able to explain every non-trivial change.
- Do not use AI to post review replies on your behalf.
- Do not paste AI answers into review discussions without verifying and owning them.
- If maintainers ask for a narrower PR, split it.

Review is a human conversation.

## 10. Identity, approvals, and merge rules

- AI-assisted PRs must have a human author of record.
- Autonomous PRs may be opened by a bot or agent account, but must name a human merge owner.
- No autonomous PR may self-approve or self-merge.
- Only a human maintainer may approve and merge an autonomous PR.
- Maintainers may require extra review for higher-risk areas.

## 11. Rate limits for autonomous contributions

To protect reviewer bandwidth, the default operating limits are:
- at most 1 open autonomous PR per agent at a time,
- at most 2 open autonomous PRs repo-wide at a time unless maintainers explicitly allow more,
- maintainers may pause or disable autonomous PRs at any time,
- maintainers may temporarily require human-authored follow-up if an agent repeatedly submits low-signal or low-quality work.

## 12. Agent capability and security boundaries

Unless a maintainer explicitly says otherwise in the issue, autonomous agents must not:
- access repository, org, or cloud secrets,
- modify `.github/workflows/`, release automation, publishing config, or security policy files,
- change dependency pins, lockfiles, or package publishing behavior,
- call unapproved remote services,
- exfiltrate code, data, logs, or test output,
- approve, merge, or close their own PRs,
- suppress, edit around, or bypass failing checks.

If a task requires privileged credentials, secret material, or changes to trust boundaries, it is not autonomous-safe by default.

## 13. Enforcement

Maintainers may close a PR without detailed review if it:
- lacks required AI disclosure,
- is obviously not understood by the submitter,
- skips required local validation,
- is too large or too noisy to review,
- contains AI-generated review replies,
- ignores issue scope,
- crosses security or autonomy boundaries,
- repeatedly creates reviewer burden without enough signal.

Repeated abuse may result in loss of contribution privileges.

## 14. Suggested PR disclosure block

```md
## AI Disclosure
- Mode: AI-assisted / Autonomous / No material AI use
- Tools used:
- Human author of record:
- Human merge owner:
- Linked issue:
- What AI changed:
- What I personally reviewed or rewrote:
- Local checks run:
- Deferred to CI:
- Known risks / uncertainty:
```

## 15. Policy intent

This policy is pro-quality, pro-accountability, and pro-automation in bounded places.

We want AI to reduce mechanical work.
We do not want AI to externalize reviewer cost, increase security risk, or lower the quality bar.
