---
name: plan
description: OPT-IN file-based planning for multi-session tasks. Only use when user explicitly requests it or work spans multiple sessions. For single-session work, prefer your platform's native planning features.
---

# File-Based Task Plan

## Default: Use Platform Planning
Unless explicitly requested ("create a plan.md", "use the plan template"), prefer your AI platform's built-in planning. This file-based approach is only justified for multi-session or handoff work.

## Use When
- User explicitly requests it
- Work spans multiple sessions / context resets
- Handoff to another AI or human
- Complex multi-PR coordination

## Setup
1. Copy template → `plans/[task_name]/plan.md`
2. Replace `[placeholders]`
3. Fill Context sections
4. Mark Step 1 `🔄`

⚠️ **gitignore**: `plans/` is local only — never commit unless user explicitly requests (`git add -f`).

## Writing Style
Terse but recoverable — every token earns its place, but a cold-start agent with zero context must still understand and resume. Abbreviate, omit filler. Test: would a stranger understand this with no other context? If yes, cut further. If no, add the minimum needed.

Emoji: only use where it saves characters or adds clarity not already present in the text (e.g. status icons replace words, ⚠️ flags a genuine warning). Do not add emoji just to have them — if removing one loses nothing, remove it.

## Critical Meta-Goals

This plan MUST be:
1. **Self-describing** — all context to resume is IN THIS FILE and files it points to
2. **Write before run** — record planned action in file *before* executing
3. **Update before continuing** — record results immediately after, *before* next step
4. **Single source of truth** — if not in the plan, it didn't happen
5. **Safe to resume** — any agent can pick up by reading only this file

> ⚠️ External memory is unreliable. This file — and files it points to — are your ONLY memory.

## Anti-Drift Protocol

### Three Commandments
1. **RELOAD** before every action — memory wiped, plan is all you have
2. **UPDATE** after every action — unwritten = didn't happen
3. **TRUST ONLY THE PLAN** — not memory, not assumptions

### Rules
- 🚫 **No assumptions** — plan is truth
- 🚫 **No offroading** — if not in plan, don't do it
- 🔐 **No secrets** — never write passwords, tokens, API keys, or credentials; use `$ENV_VAR`, `<redacted>`, or pointer to env file (e.g. `source .env.local`)
- **Parallel subagents**: before spawning, mark claimed step `🔄(🤖agent_<id>_step_<N>)` and record subagent plan path. Subagents **must NOT edit this file** — main agent updates it from their output.
- **Subagent plans**: each subagent has its own full plan at `plans/[task]/subagents/agent_<id>_step_<N>/plan.md`, same protocol. Subordinate to this file.

### Step Protocol
Per step:
1. `cat plans/[task]/plan.md | head -200` — reload
2. Find current `🔄` step
3. Write planned action in file first
4. Execute only that step
5. Write results before continuing
6. `tail -50 plans/[task]/plan.md` — verify

**Order: write plan → run → write results → continue**

### If Confused
STOP → reload plan → find last ✅ step → continue from there

## Plan Template

```markdown
# [Task] Plan
🔴 COLD START: reload skill first → `agents/skills/plan/SKILL.md`
File: `plans/[task]/plan.md` | Date: [DATE TZ] | Branch: [branch] | PR#[N] → [target] | Base: [branch @ SHA]

## Context (read-only)
**Prompt**: [verbatim user request]
**Goal**: [what we're achieving]
**Done when**: [success criteria]
**Constraints**: [key limits]
**Branch**: `[name]` off `[parent]` @ `[SHA]`

## Strategy
**Approach**: [high-level plan]
**Decisions**: [decision → reason]

## Commands
Record ALL commands verbatim — exact flags, paths, env vars. No paraphrasing. No secrets (use `$VAR`, `<redacted>`, or `source .env.local`).

Preferred:
```bash
cat plans/[task]/plan.md | head -200          # reload plan
PYTHONPATH=. uv run --no-project --with lark --with pytest python -m pytest [test_path] -q
./bin/ruff.sh && ./bin/typecheck.sh           # lint + types
gh pr checks [PR] --watch                     # CI
gh run view [RUN_ID] --json status,conclusion,jobs
git log --oneline -10
```
Add task-specific commands here as discovered.

## Status
⬜ TODO · 🔄 IN_PROGRESS · 🔄(🤖agent_<id>_step_<N>) subagent · ✅ DONE · ❌ FAILED · ⏭️ SKIPPED · 🚧 BLOCKED

## Steps
Hierarchical numbering (1, 1.1, 1.1.a …) — insert substeps freely without renumbering siblings.

#### Step 1: [Description]
🔄 | Do: [action] | OK: [criteria]
```
$ [exact command]
[output]
```
Result: [decisions / errors / next]

##### Step 1.1: [Substep]
⬜ | Do: [action] | OK: [criteria]

#### Step 2: [Description]
⬜ | Do: [action] | OK: [criteria]
```

## Compaction
Trigger when plan file exceeds ~500 lines. Target 200–300 lines after compaction.
1. Archive completed steps → `plans/[task]/history/steps<N>-<M>.md`
2. Replace with tight summary: what was done, key decisions, gotchas + pointer to archive
3. Continue numbering from where you left off

> 🔴 On context compaction/reset: **reload THIS SKILL FILE first** — before the plan, before code. It governs all execution. Skipping it causes silent drift.
