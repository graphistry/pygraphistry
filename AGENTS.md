# AGENTS.md

This file defines AI assistant guidance for this repository.

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
