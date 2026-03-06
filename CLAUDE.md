See [AGENTS.md](AGENTS.md) for repository AI guidance.

Skill path conventions:

- Codex expects `.agents/skills/`
- Claude commonly uses `.claude/skills/`
- Canonical repo skills are in `agents/skills/`

Local setup:

```bash
ln -s agents .agents
mkdir -p .claude
ln -s ../.agents/skills .claude/skills
```

Additional reference: [ai/README.md](ai/README.md).
