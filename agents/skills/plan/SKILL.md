---
name: plan
description: OPT-IN file-based planning for multi-session tasks. Only use when user explicitly requests it or work spans multiple sessions. For single-session work, prefer your platform's native planning features.
disable-model-invocation: true
---

# File-Based Task Plan Template

## Default: Use Your Platform's Native Planning

**For single-session work, prefer your AI platform's built-in planning features.**
This file-based approach adds overhead that's only justified for specific scenarios.

## When to Use THIS Skill (Opt-In)

Only use this file-based plan template when:
- **User explicitly requests it** ("use the plan template", "create a plan file")
- **Work spans multiple sessions** and must survive context resets
- **Handoff to another AI/human** who needs full written context
- **Complex multi-PR coordination** requiring persistent state

## Setup

1. Copy this template to `plans/[task_name]/plan.md`
2. Replace all `[placeholders]` with actual values
3. Fill out Context sections completely
4. Start with Step 1 marked as IN_PROGRESS

Note: `plans/` is gitignored — plans are local only.

## Critical Meta-Goals

**THIS PLAN MUST BE:**
1. **FULLY SELF-DESCRIBING**: All context needed to resume work is IN THIS FILE
2. **CONSTANTLY UPDATED**: Every action's results recorded IMMEDIATELY in the step
3. **THE SINGLE SOURCE OF TRUTH**: If it's not in the plan, it didn't happen
4. **SAFE TO RESUME**: Any AI can pick up work by reading ONLY this file

**REMEMBER**: External memory is unreliable. This plan is your ONLY memory.

## Anti-Drift Protocol

### The Three Commandments
1. **RELOAD BEFORE EVERY ACTION**: Your memory has been wiped. This plan is all you have.
2. **UPDATE AFTER EVERY ACTION**: If you don't write it down, it never happened.
3. **TRUST ONLY THE PLAN**: Not your memory, not your assumptions, ONLY what's written here.

### Critical Rules
- **ONE TASK AT A TIME** - Never jump ahead
- **NO ASSUMPTIONS** - The plan is the only truth
- **NO OFFROADING** - If it's not in the plan, don't do it

### Step Execution Protocol
**BEFORE EVERY SINGLE ACTION:**
1. **RELOAD PLAN**: `cat plans/[task_name]/plan.md | head -200`
2. **FIND YOUR TASK**: Locate the current IN_PROGRESS step
3. **EXECUTE**: ONLY do what that step says
4. **UPDATE IMMEDIATELY**: Edit this plan with results BEFORE doing anything else
5. **VERIFY**: `tail -50 plans/[task_name]/plan.md`

### If Confused
1. STOP
2. Reload this plan
3. Find the last completed step
4. Continue from there

## Plan Template

```markdown
# [Task Name] Plan
**THIS PLAN FILE**: `plans/[task_name]/plan.md`
**Created**: [DATE TIME TIMEZONE]
**Current Branch**: [from `git branch --show-current`]
**PRs**: [PR number + title + plan role]
**PR Target Branch**: [where this will merge]
**Base branch**: [FILL ME IN, TYPICALLY PR TARGET BRANCH]

## Context (READ-ONLY)

### Plan Overview
**Raw Prompt**: [What the user said, verbatim]
**Goal**: [What we're trying to achieve]
**Description**: [Brief description of the task]
**Success Criteria**: [How we know we're done]
**Key Constraints**: [Important limitations or requirements]

### Technical Context
**Initial State**:
- Working Directory: [pwd output]
- Current Branch: `[branch-name]` (forked from `[parent]` at `[SHA]`)
- Target Branch: `[where this merges to]`

### Strategy
**Approach**: [High-level plan]
**Key Decisions**:
- [Decision 1]: [Reasoning]
- [Decision 2]: [Reasoning]

## Quick Reference
```bash
# Reload plan
cat plans/[task_name]/plan.md | head -200

# Local validation before pushing
PYTHONPATH=. uv run --no-project --with lark --with pytest python -m pytest graphistry/tests/compute/gfql/cypher/test_lowering.py -q

# Lint + types
./bin/ruff.sh && ./bin/typecheck.sh

# CI monitoring
gh pr checks [PR] --watch
```

## Status Legend
- TODO: Not started
- IN_PROGRESS: Currently working on this
- DONE: Completed successfully
- FAILED: Failed, needs retry
- SKIPPED: Not needed
- BLOCKED: Can't proceed

## LIVE PLAN

### Steps

#### Step 1: [Description]
**Status**: IN_PROGRESS
**Action**: [What to do]
**Success Criteria**: [How to verify]
**Result**:
```
[Fill in with commands, output, decisions, errors]
```

#### Step 2: [Description]
**Status**: TODO
**Action**: [What to do]
**Success Criteria**: [How to verify]
```

## Step Compaction

**Every ~30 completed steps, compact the plan:**
1. Create history file: `plans/[task_name]/history/steps<start>-to-<end>.md`
2. Replace archived steps with summary in plan
3. Continue with next step number
