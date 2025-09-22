# Task Plan Template

<!-- ═══════════════════════════════════════════════════════════════════════════
                           TEMPLATE META SECTION
     ═══════════════════════════════════════════════════════════════════════════
     
     DELETE THIS ENTIRE SECTION (everything above the "START OF ACTUAL PLAN")
     WHEN CREATING YOUR PLAN
     
     Instructions for using this template:
     1. Copy this file to AI_PROGRESS/[task_name]/plan.md
     2. Delete this entire meta section
     3. Replace all [placeholders] with actual values
     4. Make sure the context sections are filled out completely
     5. Start with Step 1 marked as 🔄 IN_PROGRESS
     
     Key principle: There is EXACTLY ONE area that gets updated - the Steps section.
     Everything else is static context that never changes.
     
     ═══════════════════════════════════════════════════════════════════════════ -->

<!-- ═══════════════════════════════════════════════════════════════════════════
                            START OF ACTUAL PLAN
     ═══════════════════════════════════════════════════════════════════════════ -->

# [Task Name] Plan
**THIS PLAN FILE**: `AI_PROGRESS/[task_name]/plan.md`
**Created**: [DATE TIME TIMEZONE]
**Current Branch if any**: [from `git branch --show-current`]
**PRs if any**: [PR number + title + plan role]
**PR Target Branch if any**: [where this will merge]
**Base branch if any**: [FILL ME IN, TYPICALLY PR TARGET BRANCH]

See further info in section `## Context`

## CRITICAL META-GOALS OF THIS PLAN
**THIS PLAN MUST BE:**
1. **FULLY SELF-DESCRIBING**: All context needed to resume work is IN THIS FILE
2. **CONSTANTLY UPDATED**: Every action's results recorded IMMEDIATELY in the step
3. **THE SINGLE SOURCE OF TRUTH**: If it's not in the plan, it didn't happen
4. **SAFE TO RESUME**: Any AI can pick up work by reading ONLY this file

**REMEMBER**: External memory is unreliable. This plan is your ONLY memory.

## CRITICAL: NEVER LEAVE THIS PLAN
**YOU WILL FAIL IF YOU DON'T FOLLOW THIS PLAN EXACTLY**
**TO DO DIFFERENT THINGS, YOU MUST FIRST UPDATE THIS PLAN FILE TO ADD STEPS THAT EXPLICITLY DEFINE THOSE CHANGES.**

### Anti-Drift Protocol - READ THIS EVERY TIME
**THIS PLAN IS YOUR ONLY MEMORY. TREAT IT AS SACRED.**

### The Three Commandments:
1. **RELOAD BEFORE EVERY ACTION**: Your memory has been wiped. This plan is all you have.
2. **UPDATE AFTER EVERY ACTION**: If you don't write it down, it never happened.
3. **TRUST ONLY THE PLAN**: Not your memory, not your assumptions, ONLY what's written here.

### Critical Rules:
- **ONE TASK AT A TIME** - Never jump ahead
- **NO ASSUMPTIONS** - The plan is the only truth. If you need new info, update the plan with new steps to investigate, document, replan, act, and validate.
- **NO OFFROADING** - If it's not in the plan, don't do it

### Step Execution Protocol - MANDATORY FOR EVERY ACTION
**BEFORE EVERY SINGLE ACTION, NO EXCEPTIONS:**
1. **RELOAD PLAN**: `cat AI_PROGRESS/[task_name]/plan.md | head -200`
2. **FIND YOUR TASK**: Locate the current 🔄 IN_PROGRESS step
3. **EXECUTE**: ONLY do what that step says
4. **UPDATE IMMEDIATELY**: Edit this plan with results BEFORE doing anything else
5. **VERIFY**: `tail -50 AI_PROGRESS/[task_name]/plan.md`

**THE ONLY SECTION YOU UPDATE IS "Steps" - EVERYTHING ELSE IS READ-ONLY**

**NEVER:**
- Make decisions without reading the plan first
- Create branches without the plan telling you to
- Create PRs without the plan telling you to
- Switch contexts without updating the plan
- Do ANYTHING without the plan

### If Confused:
1. STOP
2. Reload this plan
3. Find the last ✅ completed step
4. Continue from there

## Context (READ-ONLY - Fill in at Plan Creation)

### Plan Overview
**Raw Prompt**: [What the user said, verbatim]
**Goal**: [What we're trying to achieve]
**Description**: [Brief description of the task]
**Context**: [Brief description of useful context, facts, etc]
**Success Criteria**: [How we know we're done]
**Key Constraints**: [Important limitations or requirements]

### Technical Context
**Initial State**:
- Working Directory: [pwd output]
- Current Branch: `[branch-name]` (forked from `[parent]` at `[SHA]`)
- Target Branch: `[where this merges to]`

**Related Work**:
- PR #[NUM]: [description] - [relationship to this work]
- Depends on: [what must happen first]
- Blocks: [what's waiting for this]

### Strategy
**Approach**: [High-level plan]
**Key Decisions**:
- [Decision 1]: [Reasoning]
- [Decision 2]: [Reasoning]

### Git Strategy
**Planned Git Operations**:
1. [Operation 1 - e.g., create branch X from Y]
2. [Operation 2 - e.g., cherry-pick commits A,B,C]
3. [Operation 3 - e.g., rebase onto Z after PR merges]

**Merge Order**: [PR A] → [PR B] → [This PR] → [PR C]

## Code Change Philosophy (READ-ONLY)
**Minimalist approach**: Make surgical changes, not broad refactoring.
- **No syntactic changes** (unless requested) - formatting, renames, imports, etc.
- Research minimal solution first
- Test locally before pushing
- Revert quickly if wrong

## Quick Reference (READ-ONLY)
```bash
# Reload plan
cat AI_PROGRESS/[task_name]/plan.md | head -200

# Local validation before pushing
./bin/ruff check --fix && ./bin/mypy
shellcheck [script.sh]
./bin/pytest [test] -xvs

# CI monitoring (use watch to avoid stopping - NEVER ASK USER)
gh pr checks [PR] --repo [owner/repo] --watch
gh run watch [RUN-ID]
watch -n 30 'gh pr checks [PR] --repo [owner/repo]'
# Detailed monitoring with jq:
gh run view [RUN-ID] --json status,conclusion | jq -r '"\(.status) - \(.conclusion)"'
gh run view [RUN-ID] --json jobs | jq -r '.jobs[0].steps[] | select(.status == "in_progress") | .name'
# With timeout to prevent infinite waiting:
timeout 30m gh run watch [RUN-ID]

# CI debugging with early exit
echo "DEBUG: Early exit" && exit 0  # Add to speed up iteration
git commit -m "DEBUG: Add early exit"
# Remember to remove after fix confirmed
```
## Step protocol

### RULES:
- Only update the current 🔄 IN_PROGRESS step
- Use nested numbering (1, 1.1, 1.1.1) to show hierarchy  
- Each step should be atomic and verifiable
- Include ALL context in the result (commands, output, errors, decisions)
- When adding new steps: Stop, add the step, save, then execute

### NEW STEPS
If you need to do something not in the plan:
1. STOP - Do not execute the action
2. ADD A STEP - Create it with clear description, action, success criteria
3. Mark it as 🔄 IN_PROGRESS
4. SAVE THE PLAN
5. THEN EXECUTE

### STEP COMPACTION

**Every ~30 completed steps, compact the plan:**
1. **CHECK STEP COUNT** - Count completed steps (✅, ❌, ⏭️)
2. **CREATE HISTORY FILE** - Copy oldest 15+ completed steps to:
   - Path: `AI_PROGRESS/[task_name]/history/steps<start>-to-<end>.md`
   - Check existing history files first with `ls AI_PROGRESS/[task_name]/history/`
   - Keep same format as plan.md
3. **REPLACE IN PLAN** - Replace archived steps with:
   ```
   ### Steps 1-15: [Brief Title] ✅ ARCHIVED
   **Archived**: `AI_PROGRESS/[task_name]/history/steps1-to-15.md`
   **Summary**: 
   - Key outcome 1
   - Key outcome 2
   - Important artifacts/PRs created
   ```
4. **ADD COMPACTION TASK** - Before starting compaction, add it as a step
5. **VERIFY** - Ensure plan still makes sense after compaction

Then continue with Step 16...


## Status Legend
- 📝 **TODO**: Not started
- 🔄 **IN_PROGRESS**: Currently working on this
- ✅ **DONE**: Completed successfully  
- ❌ **FAILED**: Failed, needs retry
- ⏭️ **SKIPPED**: Not needed (explain in result)
- 🚫 **BLOCKED**: Can't proceed (explain in result)

## LIVE PLAN (THE ONLY SECTION YOU UPDATE)

Follow `## Step protocol`:

### Context Preservation (Update ONLY if directed by a step)
<!-- Only update these sections if a step specifically says to -->

#### Key Decisions Made
<!-- Document WHY things were done certain ways -->
- [Decision]: [Reasoning]

#### Lessons Learned  
<!-- Document what failed and why to avoid repeating -->
- [What happened]: [Why it failed]: [How to avoid]

#### Important Commands
<!-- Document complex commands that worked -->
```bash
# [Description of what this does]
[command]
```
### Steps

Reminder, follow `## Step protocol`:

#### Step 1.0.0: [Description]
**Status**: 🔄 IN_PROGRESS
**Started**: [timestamp]
**Action**: [What to do]
**Success Criteria**: [How to verify]
**Result**:
```
[Fill this in with commands, output, decisions, errors, etc.]
```

```