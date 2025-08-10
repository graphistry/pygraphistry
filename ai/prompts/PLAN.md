# Task Plan Template

<!-- ═══════════════════════════════════════════════════════════════════════════
                         TEMPLATE META SECTION
     ═══════════════════════════════════════════════════════════════════════════
     
     DELETE THIS ENTIRE SECTION (everything above the "START OF ACTUAL PLAN")
     WHEN CREATING YOUR PLAN
     
     Instructions for using this template:
     1. Copy this file to plans/[task_name]/plan.md
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
**THIS PLAN FILE**: `plans/[task_name]/plan.md`
**Created**: [DATE TIME]
**Current Branch**: [from `git branch --show-current`]
**PR**: [PR number + title if applicable]
**Base Branch**: [what this branches from]

## CRITICAL META-GOALS OF THIS PLAN
**THIS PLAN MUST BE:**
1. **FULLY SELF-DESCRIBING**: All context needed to resume work is IN THIS FILE
2. **CONSTANTLY UPDATED**: Every action's results recorded IMMEDIATELY
3. **THE SINGLE SOURCE OF TRUTH**: If it's not in the plan, it didn't happen
4. **SAFE TO RESUME**: Any AI can pick up work by reading ONLY this file

**REMEMBER**: External memory is unreliable. This plan is your ONLY memory.

## CRITICAL: NEVER LEAVE THIS PLAN
**YOU WILL FAIL IF YOU DON'T FOLLOW THIS PLAN EXACTLY**

### Anti-Drift Protocol - READ THIS EVERY TIME
**THIS PLAN IS YOUR ONLY MEMORY. TREAT IT AS SACRED.**

### The Three Commandments:
1. **RELOAD BEFORE EVERY ACTION**: Your memory has been wiped. This plan is all you have.
2. **UPDATE AFTER EVERY ACTION**: If you don't write it down, it never happened.
3. **TRUST ONLY THE PLAN**: Not your memory, not your assumptions, ONLY what's written here.

### Step Execution Protocol - MANDATORY
**BEFORE EVERY SINGLE ACTION:**
1. **RELOAD PLAN**: Read this file completely
2. **FIND YOUR TASK**: Locate the current 🔄 IN_PROGRESS step
3. **EXECUTE**: ONLY do what that step says
4. **UPDATE IMMEDIATELY**: Record results before anything else
5. **MARK STATUS**: Update step status (✅, ❌, etc.)

## Context (READ-ONLY - Fill at Creation)

### Objective
[Clear, specific description of what needs to be accomplished]

### Current State
[Where we are now - branch, files, dependencies]

### Success Criteria
[How we know when the task is complete]

### Git Strategy (if applicable)
**Branch Strategy**: [How branches will be organized]
**Merge Order**: [If multiple PRs, what order]

## Status Legend
- 📝 **TODO**: Not started
- 🔄 **IN_PROGRESS**: Currently working (max 1 at a time)
- ✅ **DONE**: Completed successfully
- ❌ **FAILED**: Failed, needs retry
- ⏭️ **SKIPPED**: Not needed (explain why)
- 🚫 **BLOCKED**: Can't proceed (explain why)

## Quick Reference

### Key Commands
```bash
# Project validation (PyGraphistry)
cd docker && WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh
WITH_BUILD=0 ./test-cpu-local-minimal.sh

# Type checking and linting
./bin/lint.sh
./bin/typecheck.sh
mypy graphistry/
ruff check graphistry/ --fix

# Testing
WITH_BUILD=0 WITH_LINT=0 WITH_TYPECHECK=0 ./test-cpu-local.sh graphistry/tests/test_file.py
pytest graphistry/tests/ -xvs

# Git operations
git status
git diff --cached
git log --oneline -n 10
```

### Important Paths
- Source: `graphistry/`
- Tests: `graphistry/tests/`
- Docs: `docs/`
- Plans: `plans/`
- AI prompts: `ai/prompts/`
- AI docs: `ai/docs/`

### SECURITY: Credential Storage
**NEVER save secrets to version-controlled directories!**

**What counts as secrets:**
- API keys and tokens
- Passwords
- Server URLs/hostnames
- Organization names
- Any customer-specific identifiers

**Safe locations:**
- ✅ **SAFE**: `plans/` and `tmp/` (gitignored)
- ✅ **SAFE**: `.env` files in project root (gitignored)
- ❌ **UNSAFE**: Any other directory (graphistry/, tests/, docs/, scripts/, etc.)

**Always use .env files for secrets:**
```bash
# Create secrets in tmp/ or plans/
echo "export GRAPHISTRY_API_KEY='secret'" > tmp/.env.local
echo "export GRAPHISTRY_SERVER='https://hub.graphistry.com'" >> tmp/.env.local
echo "export GRAPHISTRY_ORG='org-name'" >> tmp/.env.local
# Source when needed
source tmp/.env.local
```

## Step Protocol

### RULES:
- Only update the current 🔄 IN_PROGRESS step
- Each step should be atomic and verifiable
- Include ALL context in results (commands, output, errors)
- When adding new steps: Stop, add the step, save, then execute

### NEW STEPS
If you need to do something not in the plan:
1. STOP - Do not execute
2. ADD THE STEP - With clear description and success criteria
3. Mark as 🔄 IN_PROGRESS
4. SAVE THE PLAN
5. THEN EXECUTE

### STEP COMPACTION
Every ~20 completed steps:
1. Move old steps to `## Archived Steps` section
2. Keep summary of what was accomplished
3. Continue with fresh step numbers

## Steps

### Step 1: [Title]
**Status:** 📝 TODO
**Description:** [What this step accomplishes]
**Actions:**
```bash
# Specific commands to run
```
**Success Criteria:** [How to verify completion]
**Result:** [To be filled when complete]

### Step 2: [Title]
**Status:** 📝 TODO
**Description:** [What this step accomplishes]
**Actions:**
```bash
# Specific commands to run
```
**Success Criteria:** [How to verify completion]
**Result:** [To be filled when complete]

## Context Preservation
<!-- Update ONLY when directed by a step -->

### Key Decisions Made
- [Decision]: [Reasoning]

### Lessons Learned
- [What happened]: [Why]: [How to avoid]

### Important Commands
```bash
# [Description]
[command that worked]
```

## Archived Steps
<!-- Move completed steps here when plan gets too long -->

---
*Plan created: [date]*
*Last updated: [date]*