# CI Speedup + Supply Chain Security Plan
🔴 COLD START: reload skill first → `agents/skills/plan/SKILL.md`
File: `plans/ci-speedup-minimal/plan.md` | Date: 2026-04-06 | Branch: `ci/speedup-minimal-tests` | PR#1050 → master | Base: master @ 483f24622

## Context (read-only)
**Prompt**: Speed up CI minimal gate (10-12 min), switch to uv, add lockfiles with 6d cooldown for supply chain security, cover all jobs incl RTD/docs, make it maintainable
**Goal**: Every CI install path uses uv + on-the-fly hashed lockfiles with 6-day dep cooldown. No committed lockfiles (library convention).
**Done when**: All CI jobs use lockfiles from generation job, `UV_EXCLUDE_NEWER` globally, `--only-binary :all:`, DRY install pattern, RTD issue filed, contributor docs, audit, CI green
**Constraints**: No committed lockfiles (library). Python 3.8-3.14 matrix. Torch `+cpu` needs pip. RTD uses pip natively (out of scope for this PR, issue filed).
**Branch**: `ci/speedup-minimal-tests` off `master` @ `483f24622`

## Strategy
**Approach**: uv for speed → GFQL split for gate reduction → on-the-fly per-version lockfiles via CI job → DRY install helper
**Decisions**:
- No committed lockfiles → library convention; lockfiles generated on-the-fly in CI generation job, uploaded as artifacts, downloaded by matrix jobs
- `uv pip compile --python-version X` cross-resolves without needing that Python installed → generation job runs on one Python, produces lockfiles for all 7 versions × 9 profiles
- `--exclude-newer 6d` + `--generate-hashes` at generation time → 6-day cooldown + hashes baked in
- `UV_EXCLUDE_NEWER` env var globally as belt-and-suspenders → cooldown at install time too
- `--require-hashes` at install time for non-torch profiles → tamper-proof
- AI/umap profiles: lockfile WITHOUT `--require-hashes` (torch excluded from lockfile via `--no-emit-package`, but `--require-hashes` still fails because transitive deps reference torch)
- torch CPU stays on pip with `--no-deps` (installed AFTER lockfile so deps are already satisfied)
- `--universal` rejected → doesn't produce version-conditional deps when setup.py lacks markers
- `tck-gfql` checkout must pin the resolved commit SHA, not just the branch name → avoids nondeterministic CI when the external branch moves between reruns
- RTD: accept floating for now, file issue to port to uv later (#TBD)

**CI flow**:
```
generate-lockfiles (~60s, runs once)
  ├─ for profile in test, test-core, test-compat, ...:
  │    for ver in 3.8, 3.9, ..., 3.14:
  │      uv pip compile setup.py --extra <extras> \
  │        --python-version <ver> --exclude-newer "6 days" \
  │        --generate-hashes -o <profile>-py<ver>.lock
  └─ upload-artifact: all lockfiles

test-minimal-python (matrix: 3.8-3.14, needs: generate-lockfiles)
  ├─ download-artifact
  └─ uv pip install --require-hashes -r test-py${{ matrix.python-version }}.lock
     uv pip install -e . --no-deps
```

## Commands
```bash
cat plans/ci-speedup-minimal/plan.md | head -200
./bin/generate-lockfiles.sh
gh run list --repo graphistry/pygraphistry --branch ci/speedup-minimal-tests --limit 3 --json conclusion,name
gh run view <ID> --repo graphistry/pygraphistry --json jobs --jq '.jobs[] | select(.conclusion == "failure") | .name'
```

## Status
⬜ TODO · 🔄 IN_PROGRESS · ✅ DONE · ❌ FAILED · ⏭️ SKIPPED · 🚧 BLOCKED

## Steps

#### Step 1: Baseline measurement
✅ | Result: Gate 10-12 min. test_lowering.py = 698 tests, 15% of time.

#### Step 2: uv install across all CI jobs
✅ | Result: All jobs converted. torch CPU stays on pip. CI green.

#### Step 3: Split GFQL tests out of gate
✅ | Result: Gate 11 min → 6 min (42%). 944 tests split to parallel test-gfql-core.

#### Step 4: On-the-fly lockfile generation + wiring
✅ | Result: `generate-lockfiles` job produces 57 per-version lockfiles, uploads as artifact. All 11 downstream jobs download + install from lockfiles. `UV_EXCLUDE_NEWER=6 days` set globally. AI/umap use lockfile without `--require-hashes` (torch conflict). Torch installed `--no-deps` after lockfile. TCK fixed (tck-gfql#26 match5-8, #27 match5-6). Push CI green `32ff0ae11`.

#### Step 5: `--only-binary :all:`
⏭️ SKIPPED | Impractical — pygraphviz needs source build, some AI deps on older Python need sdist.

#### Step 6: pip cooldown for floating pip installs
✅ | Result: Removed redundant `sentence-transformers --upgrade` (already in lockfile). All remaining pip installs are exact-pinned (torch, DGL). Zero floating pip installs.

#### Step 7: DRY install pattern
⏭️ SKIPPED | Each lockfile line is unique (different profile). Copy-paste of download-artifact block is clear and self-contained. Composite action adds complexity without proportional benefit.

#### Step 8: RTD — file issue, accept floating for now
✅ | Result: Filed #1074. RTD builds docs, not security-critical code. Acceptable as floating for now.

#### Step 9: Docs Docker
✅ | Result: docs/docker/Dockerfile uses floating pip inside an isolated container. Same category as RTD. Covered by #1074.

#### Step 10: Audit
✅ | Result: All CI installs are lockfile-locked or exact-pinned. Exceptions documented: RTD (#1074), docs Docker (#1074), DGL (exact pins). Zero unexpected floating installs.

#### Step 11: Contributor docs
✅ | Result: Added CI lockfile section to DEVELOP.md covering generation, cooldown, hash verification, and emergency override.

#### Step 12: Measured timing comparison
✅ | Result: Gate reduced 34-43% across all Python versions. Lockfile installs are consistently faster (no resolution).

#### Step 13: CI green + merge
🔄 IN_PROGRESS | Result: Investigated red PR status on 2026-04-06. Root cause: `tck-gfql` workflow resolved a branch name, then cloned that mutable branch head later; run `24024821175` failed on `match5-6` while a later rerun `24024822231` on the same PyGraphistry SHA passed after the external branch advanced. Fix staged locally in `.github/workflows/ci.yml`: resolve `tck-gfql` ref to an exact SHA, fetch that SHA, and record both ref + sha in the job summary. Next: push, let CI rerun, confirm the stale red state is cleared.

## Resume State
**Last updated:** 2026-04-06
**Current phase:** VERIFYING CI STABILITY. Workflow fix staged locally to make `tck-gfql` deterministic across reruns; pending push + CI confirmation.
