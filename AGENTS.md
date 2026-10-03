# AGENTS.md

This file defines AI assistant guidance for this repository.

## Policy

For AI contribution governance (AI-assisted vs autonomous contribution rules), see `AI_POLICY.md`.

## Skill Locations

Canonical skills live under:

- `agents/skills/`

For tools that expect `.agents/skills/`, use a local symlink:

```bash
ln -s agents .agents
```

## Platform Path Conventions

- **OpenAI Codex**: prefers `.agents/skills/`
- **Claude Code**: often uses `.claude/skills/`
- **Other tools (Cursor/Copilot/etc.)**: may use `.agents/skills/` or tool-specific paths

Recommended local setup:

```bash
# Canonical -> tool-expected path
ln -s agents .agents

# Claude path to shared skills
mkdir -p .claude
ln -s ../.agents/skills .claude/skills
```

`/.agents/` is ignored in git to keep local tool wiring out of commits.

## Guidance Priority

1. Follow repository docs and tests as source of truth.
2. Keep edits minimal and verifiable.
3. Run relevant lint/tests for touched areas.

## Release Assistant Reminder

- During publish assistance, after pushing tag `X.Y.Z`, always check `.github/workflows/publish-pypi.yml` run state.
- Treat `Publish distribution to PyPI` in `waiting` as an expected manual gate for environment `pypi-release`; explicitly instruct maintainer to click `Review deployments` before expecting PyPI completion.
- After the workflow completes, verify both TestPyPI and PyPI public metadata report `X.Y.Z`; PyPI may lag briefly after the job succeeds.
- Create or verify the matching GitHub Release for tag `X.Y.Z`, then check Read the Docs. `latest` may update automatically, while tag-specific versions can require maintainer activation in the RTD dashboard.
- Record release follow-ups discovered during readiness audits as issues in the owning repo before final handoff.
