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
     5. Start with Phase 1.A marked as 🔄 IN_PROGRESS

     Key principle: There is EXACTLY ONE area that gets updated - the Phases section.
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

## Execution Protocol

**Before each action:**
1. Reload plan → Find 🔄 IN_PROGRESS phase → Execute only that phase → Update result → Mark status
2. If not in plan: STOP → Add phase → Save → Then execute
3. Trust only what's written here

## Bug Fix Protocol (Bug Fixes Only)

**For bug fixes, follow TDD: Investigate → Reproduce → Test → Fix → Validate → Finalize**

1. **Investigate**: Read code, understand patterns, identify root cause
2. **Reproduce**: Create minimal repro script, confirm bug
3. **Test**: Write failing test (GPU/CPU as needed), verify baseline
4. **Fix**: Implement fix, verify test passes
5. **Validate**: Typecheck, lint, full test suite
6. **Finalize**: CHANGELOG, commit, PR

## Context (READ-ONLY - Fill at Creation)

### Objective
[Clear, specific description of what needs to be accomplished]

### Current State
[Where we are now - branch, files, dependencies]

### Success Criteria
[How we know when the task is complete]

### Related Plans (if applicable)
**Previous Plans**: [Link to prior plan files if continuing work or referencing history]
- `plans/[previous_task]/plan.md` - [Brief summary with key entities/terms]

### Git Strategy (if applicable)
**Branch Strategy**: [How branches will be organized]
**PR/Branch Stack**: [If stacked PRs, list order: PR#123 (branch-1) → PR#124 (branch-2)]
**Merge Order**: [Order to merge if multiple PRs]
**Note**: Track branch/PR per phase as they may change (e.g., rebase flows)

## Status Legend
- 📝 **TODO**: Not started
- 🔄 **IN_PROGRESS**: Currently working (max 1 at a time)
- ✅ **DONE**: Completed successfully
- ❌ **FAILED**: Failed, needs retry
- ⏭️ **SKIPPED**: Not needed (explain why)
- 🚫 **BLOCKED**: Can't proceed (explain why)

## Quick Reference

### Key Commands (Waterfall Order)
```bash
# 1. Tests first (catch functional issues early)
WITH_BUILD=0 WITH_LINT=0 WITH_TYPECHECK=0 ./test-cpu-local.sh graphistry/tests/test_file.py
pytest graphistry/tests/ -xvs

# 2. Then types (catch type issues before style)
./bin/mypy.sh
mypy graphistry/

# 3. Finally linting (style last)
./bin/lint.sh
ruff check graphistry/ --fix

# Full validation (PyGraphistry)
cd docker && WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh
WITH_BUILD=0 ./test-cpu-local-minimal.sh

# Docs (typically don't use, run at end if needed)
./docs/html.sh        # Build docs
./docs/ci.sh          # Full validation

# Git operations
git status
git add -p              # build grouped, semantic hunks before committing
git diff --cached
# prefer conventional commit messages and normal pushes;
# avoid force-push unless coordinated (keeps GH CI intact)
git log --oneline -n 10
```

### Important Paths
- Source: `graphistry/`
- Tests: `graphistry/tests/` (mirrors source structure: `graphistry/foo/bar.py` → `graphistry/tests/foo/test_bar.py`)
- Docs: `docs/`
- Plans: `plans/` (gitignored - safe for auxiliary files, temp secrets, working data)
- AI prompts: `ai/prompts/`
- AI docs: `ai/docs/`

### Security & Working Files

**plans/ is gitignored - safe for:**
- Auxiliary working files
- Temporary secrets (NEVER commit secrets anywhere else)
- Scratch data, test outputs, etc.

**Store secrets in plans/ as .env files:**
```bash
echo "export GRAPHISTRY_API_KEY='secret'" > plans/[task]/.env
echo "export GRAPHISTRY_SERVER='https://hub.graphistry.com'" >> plans/[task]/.env
source plans/[task]/.env
```

**For commits:** See [ai/prompts/CONVENTIONAL_COMMITS.md](CONVENTIONAL_COMMITS.md)

## Phase Protocol

### RULES:
- Only update the current 🔄 IN_PROGRESS phase
- Each phase should be atomic and verifiable
- Include ALL context in results (commands, output, errors)
- **RECORD ALL TOOL CALLS**: When documenting results, include the actual tool calls executed (redact secrets with *****)
- When adding new phases: Stop, add the phase, save, then execute

### NEW PHASES
If you need to do something not in the plan:
1. STOP - Do not execute
2. ADD THE PHASE - With clear description and success criteria
3. Mark as 🔄 IN_PROGRESS
4. SAVE THE PLAN
5. THEN EXECUTE

### PHASE COMPACTION
Every ~20 completed phases:
1. Create inline summary at top of `## Phases` section with key accomplishments/decisions
2. Remove completed phase details from `## Phases` section
3. Continue with fresh phase numbers
4. Plan remains linear - no jumping to separate archived section

## Phases

### Completed Phase Summary (if compacted)
**Phases 1.A - 20.Z Summary**: [Brief summary of what was accomplished, key decisions, important results]

### Phase 1.A: [Title]
**Status:** 📝 TODO
**Branch:** [branch name]
**PR:** [#number or N/A]
**Issues:** [#number, #number or N/A]
**Started:** [YYYY-MM-DD HH:MM:SS]
**Completed:** [YYYY-MM-DD HH:MM:SS]
**Description:** [What this phase accomplishes]
**Actions:**
```bash
# Specific commands to run
```
**Success Criteria:** [How to verify completion]
**Result:** [To be filled when complete]

**IMPORTANT**: When filling **Result**, include:
1. What actually happened (success/failure/partial)
2. **Actual tool calls executed** (with secrets redacted: `--api-key=*****`)
3. Key outputs/errors (can be summarized if long)
4. Any deviations from planned actions
5. Links to auxiliary files created in plans/[task]/ if applicable

### Phase 1.B: [Title]
**Status:** 📝 TODO
**Branch:** [branch name]
**PR:** [#number or N/A]
**Issues:** [#number, #number or N/A]
**Started:** [YYYY-MM-DD HH:MM:SS]
**Completed:** [YYYY-MM-DD HH:MM:SS]
**Description:** [What this phase accomplishes]
**Actions:**
```bash
# Specific commands to run
```
**Success Criteria:** [How to verify completion]
**Result:** [To be filled when complete]

**IMPORTANT**: When filling **Result**, include:
1. What actually happened (success/failure/partial)
2. **Actual tool calls executed** (with secrets redacted: `--api-key=*****`)
3. Key outputs/errors (can be summarized if long)
4. Any deviations from planned actions
5. Links to auxiliary files created in plans/[task]/ if applicable

## Context Preservation
<!-- Update ONLY when directed by a phase -->

### Key Decisions Made
- [Decision]: [Reasoning]

### Lessons Learned
- [What happened]: [Why]: [How to avoid]

### Important Commands
```bash
# [Description]
[command that worked]
```


---
*Plan created: [date]*
*Last updated: [date]*