# PR-F4 Plan (Issue #1205)

## Goal
Complete PR-F4 by documenting release signing and consumer verification for published artifacts, and clarifying where SBOM evidence is stored.

## Branch / PR
- Branch: `chore/1205-pr-f4-signing-verification-docs-v2`
- Base: latest `origin/master` after PR-F3 merge
- Tracking issue: #1205
- Draft PR: #1206

## Plan
- [x] Start from latest remote master on a fresh branch
- [x] Confirm current publish behavior from PR-F3 (attestations + SBOM evidence)
- [x] Add release verification guide for consumers
- [x] Link verification guidance from security-facing docs
- [x] Link guide into docs index for discoverability
- [x] Run review pass and address findings
- [x] Open draft PR linked to #1205

## Notes
- TestPyPI dry run validation already succeeded in run `24877854485`.
- SBOM is emitted as workflow evidence artifact (`release-evidence-<run_id>/sbom-cyclonedx.json`), while PyPI/TestPyPI host release-file attestations/provenance.
- External skill file requested by user (`~/Work/graphistry/.agents/skills/review/SKILL.md`) could not be read in this restricted runtime; manual review pass used as fallback.