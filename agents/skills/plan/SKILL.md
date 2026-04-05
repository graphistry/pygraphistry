---
name: plan
description: File-based planning for multi-session tasks. Only use when user explicitly requests it or work spans multiple sessions.
disable-model-invocation: true
---

# File-Based Task Plan

## When to Use

Only use this when:
- User explicitly requests it
- Work spans multiple sessions and must survive context resets
- Handoff to another AI/human who needs full written context

For single-session work, prefer native planning features.

## Setup

Create `plans/[task_name]/plan.md`. The `plans/` dir is gitignored — it is local only.

## Format

Plans are living narrative snapshots, not checklists. Each plan starts with a status block and grows by appending dated passes as work progresses. See existing plans in `plans/` for examples.

```markdown
Status Snapshot ([issue/task name])

Date
- [YYYY-MM-DD]

Repo state
- Repo: [path]
- Active branch: [branch]
- Base branch: [master or parent]
- Issue: [#N] [title]
- Issue link: [url]
- PR: [#N] [title]  (or "none yet")
- PR link: [url]

What is already landed (master)
- [relevant prior work that informs this task]

What is on this branch
- [what this branch adds — fill in as work progresses]

Explicitly out of scope
- [things we are not doing here]

Key files
- [file] — [one-line purpose]

Next recommended move
- [concrete next action]

Cold-start resume notes
- [anything a fresh context needs to know to pick up without re-reading history]
```

## Updating

Append dated sections as work progresses — don't rewrite history. Each pass should record:
- What was done
- What was found (bugs, decisions, design choices)
- Validation results (test commands + pass counts)
- What comes next

## Critical rules

- **Keep it current**: update after every meaningful action
- **Cold-start safe**: a fresh AI should be able to resume by reading only this file
- **Less is more**: record decisions and findings, not narration of obvious steps
