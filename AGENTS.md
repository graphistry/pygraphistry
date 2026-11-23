# AGENTS.md

This file defines AI assistant guidance for this repository.

## Skill Locations

Canonical skills live under:

- `.agents/skills/`

## Platform Path Conventions

- **OpenAI Codex**: uses `.agents/skills/`
- **Claude Code**: often uses `.claude/skills/`
- **Other tools (Cursor/Copilot/etc.)**: may use `.agents/skills/` or tool-specific paths

Recommended local setup:

```bash
# Claude path to shared skills
mkdir -p .claude
ln -s ../.agents/skills .claude/skills
```

## Guidance Priority

1. Follow repository docs and tests as source of truth.
2. Keep edits minimal and verifiable.
3. Run relevant lint/tests for touched areas, and run `./bin/lint.sh` before every push (it is the whole lint gate: ruff plus the type-hygiene and comment-density ratchets; bare `ruff` is not enough).
4. No dynamic typing in code you add under `graphistry/`: no `hasattr`/`getattr`/`setattr`, no `__module__` or `type(x).__name__` sniffing, no `except AttributeError` probes, no bare `# type: ignore`, no `object` annotations. Use the typed helpers in `graphistry/Engine.py` (`is_series_like`, `is_cudf_series`, `is_polars_series`, `is_polars_df`), `isinstance`, or a Protocol. See `DEVELOP.md` "Type Hygiene Guard".

## Release Assistant Reminder

- During publish assistance, after pushing tag `X.Y.Z`, always check `.github/workflows/publish-pypi.yml` run state.
- Treat `Publish distribution to PyPI` in `waiting` as an expected manual gate for environment `pypi-release`; explicitly instruct maintainer to click `Review deployments` before expecting PyPI completion.
- After the workflow completes, verify both TestPyPI and PyPI public metadata report `X.Y.Z`; PyPI may lag briefly after the job succeeds.
- Create or verify the matching GitHub Release for tag `X.Y.Z`, then check Read the Docs. `latest` may update automatically, while tag-specific versions can require maintainer activation in the RTD dashboard.
- Record release follow-ups discovered during readiness audits as issues in the owning repo before final handoff.

## Published Benchmarks

- Never remove or suppress a published benchmark number without the maintainer's explicit approval. Stale numbers remain published with their measurement date, a staleness label, and a linked re-measure issue.
