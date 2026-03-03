@AGENTS.md

## Claude Code Setup

If no repo skills appear in Claude Code (check with `/skills`), configure the local symlink:

```bash
ln -s ../.agents/skills .claude/skills
```

If `.claude/skills` already exists and is wrong, replace it with the symlink above.
