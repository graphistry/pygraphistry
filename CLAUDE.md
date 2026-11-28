This file is for Claude Code (a coding agent); people read [CONTRIBUTING.md](CONTRIBUTING.md).

**Mandatory gateway — do not skip:** before editing code, running checks, or opening/reviewing a pull request, read and follow [AGENTS.md](AGENTS.md) and the matching canonical skill under `.agents/skills/`:

- Working on an issue or opening a PR: [.agents/skills/contribute/SKILL.md](.agents/skills/contribute/SKILL.md)
- Reviewing a PR (especially outside/fork PRs): [.agents/skills/screen-untrusted/SKILL.md](.agents/skills/screen-untrusted/SKILL.md) and [.agents/skills/review/SKILL.md](.agents/skills/review/SKILL.md)

Do not improvise local checks from memory — `AGENTS.md` and the skills define the full lint, type, engine, and review gates (including Windows / no-Bash commands).

Skill path conventions:

- Codex expects `.agents/skills/`
- Claude commonly uses `.claude/skills/`
- Canonical repo skills are in `.agents/skills/` (read them directly on Windows or whenever `.claude/skills` is not symlinked)

Optional local symlink (macOS/Linux):

```bash
mkdir -p .claude
ln -s ../.agents/skills .claude/skills
```

Additional reference: [ai/README.md](ai/README.md).
