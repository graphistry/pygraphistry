This file is for Claude Code (a coding agent); people read [CONTRIBUTING.md](CONTRIBUTING.md). See [AGENTS.md](AGENTS.md) for repository AI guidance.

Skill path conventions:

- Codex expects `.agents/skills/`
- Claude commonly uses `.claude/skills/`
- Canonical repo skills are in `.agents/skills/`

Local setup:

```bash
mkdir -p .claude
ln -s ../.agents/skills .claude/skills
```

Additional reference: [ai/README.md](ai/README.md).

## Before every push

Run `./bin/lint.sh` for the full lint gate: ruff, type hygiene, comment density,
relative imports, and the quick CI guards (GPU-gate audit and Cypher-surface
guard). The quick guards run by default (`LINT_QUICK_GUARDS=1`); set
`LINT_QUICK_GUARDS=0` to skip those two checks.

On Windows or without Bash, run from the repository root: `ruff check graphistry`, then
`python bin/ci/ci_type_hygiene_guard.py`, `python bin/ci/ci_comment_density_guard.py`,
`python bin/ci/ci_gpu_gate_audit.py`, and `python bin/ci/ci_cypher_surface_guard.py`.
