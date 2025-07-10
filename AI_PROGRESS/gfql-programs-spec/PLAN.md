# GFQL Programs Spec Development Plan
**THIS PLAN FILE**: `AI_PROGRESS/gfql-programs-spec/PLAN.md`
**Created**: 2025-07-10 UTC
**Current Branch if any**: dev/gfql-program
**PRs if any**: None yet
**PR Target Branch if any**: master
**Base branch if any**: master

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
1. **RELOAD PLAN**: `cat AI_PROGRESS/gfql-programs-spec/PLAN.md | head -200`
2. **FIND YOUR TASK**: Locate the current 🔄 IN_PROGRESS step
3. **EXECUTE**: ONLY do what that step says
4. **UPDATE IMMEDIATELY**: Edit this plan with results BEFORE doing anything else
5. **VERIFY**: `tail -50 AI_PROGRESS/gfql-programs-spec/PLAN.md`

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
**Raw Prompt**: "in AI_PROGRESS/gfql-programs-spec/ , make a new PLAN.md that goes through some prd work on sketch.md there . First phase (Steps 1.x) should be general analysis of the repo, feature: Do steps around reading our GFQL impl/examples and PyGraphistry regular APIs around GFQL Wire Protocol and Python API, and saving out some relevant knowledge to a lookup file with back references to our repo (file, lineno, snippet). Then enumerate our sketch.md features, and for each, create a sub analysis step (1.X.1, 1.X.2, 1.X.3) about that feature, how it relates, and some critical review of bugs/risks/improvements/etc. After, do a combined critical review, step 1.X+1. Finally, make a new sketch1X.md that supercedes our sketch.md, and is complete unto itself. Do a follow-on step to compare the two and fix sketch1X.md for whatever missed. Once all that is ready, make a new step steries, 2.*, whose first step is to review 1.* and come up with some key different User Personas and key different User Scenarios for each one around these features. Then another step to review those, and if any key gaps, add more Personas/Scenarios. Then, start step seris 3.*. First step is to add a subset for every user persona x user scenario for a role play. Each individual step is to generate a role_play_user_X_scenario_Y.md (catalog these), where you fill out the role play of that scenario getting solved via the wire protocol or python api. End each role play .md with a bit lof localized analysis of what worked, what didn't, and how to improve, with a prioritized breakdown of regular P0 (absolutely & urgenetly required) to P4/P5 (probably won't happen superficial nice-to-haves.) Finally, do a step series 4.X that reviews our 2.* and 3.* to create a sketch3X.md . Make a stp that is a metastep: read all our 2x/3x files & comments, and for each one, create a fresh step to update our 3X.md with appropriate fixes."
**Goal**: Develop comprehensive product specification for GFQL Programs through analysis, user research, and iterative refinement
**Description**: Multi-phase PRD development process starting with technical analysis, moving through user persona development and role-playing, ending with refined specification
**Context**: GFQL is PyGraphistry's declarative query language. The sketch.md proposes extending it from single chains to DAG composition with new features like remote graph loading, graph combinators, and call operations. Binding names must match regex: ^[a-zA-Z_][a-zA-Z0-9_-]*$
**Success Criteria**: 
- Complete technical analysis with code references
- Validated user personas and scenarios
- Role play documents demonstrating API usage
- Final sketch3X.md specification ready for implementation
**Key Constraints**: 
- Steps must be dynamic and self-determining
- Role plays must be separate 100+ LOC files with 3-20 turns
- No timeline estimates
- Must follow functional programming practices per ai_code_notes

### Technical Context
**Initial State**:
- Working Directory: /home/lmeyerov/Work/pygraphistry2
- Current Branch: `dev/gfql-program` (forked from `master` at `[SHA]`)
- Target Branch: `master`

**Related Work**:
- Current GFQL implementation in `/graphistry/compute/`
- sketch.md RFC in `AI_PROGRESS/gfql-programs-spec/`
- Depends on: Understanding current GFQL architecture
- Blocks: Future GFQL DAG implementation

### Strategy
**Approach**: Four-phase iterative development:
1. Technical analysis and initial refinement
2. User persona and scenario development
3. Role-play validation
4. Final synthesis and specification

**Key Decisions**:
- Dynamic step generation: Later steps determined by earlier findings
- Separate files for each role play: Ensures detailed exploration
- Meta-steps for systematic updates: Maintains traceability

### Git Strategy
**Planned Git Operations**:
1. Work on dev/gfql-program branch
2. Commit analysis artifacts and specifications
3. Create PR to master when complete

**Merge Order**: This work → Implementation work

## Quick Reference (READ-ONLY)
```bash
# Reload plan
cat AI_PROGRESS/gfql-programs-spec/PLAN.md | head -200

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
   - Path: `AI_PROGRESS/gfql-programs-spec/history/steps<start>-to-<end>.md`
   - Check existing history files first with `ls AI_PROGRESS/gfql-programs-spec/history/`
   - Keep same format as plan.md
3. **REPLACE IN PLAN** - Replace archived steps with:
   ```
   ### Steps 1-15: [Brief Title] ✅ ARCHIVED
   **Archived**: `AI_PROGRESS/gfql-programs-spec/history/steps1-to-15.md`
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

#### Step 0.1: Create PR for tracking GFQL Programs Spec work
**Status**: 🔄 IN_PROGRESS
**Started**: 2025-07-10 UTC
**Action**: Create PR from dev/gfql-program to master for tracking this PRD work
**Success Criteria**: PR created with description of the 4-phase plan
**Result**:
```
[To be filled]
```

#### Step 1.0: Create GFQL Knowledge Base
**Status**: 📝 TODO
**Started**: 
**Action**: Read core GFQL implementation files and create lookup file with references
**Success Criteria**: gfql_knowledge_base.md created with file:lineno:snippet references
**Result**:
```
[To be filled]
```

#### Step 1.1: Analyze PyGraphistry APIs Around GFQL
**Status**: 📝 TODO
**Started**: 
**Action**: Document Wire Protocol and Python API integration points
**Success Criteria**: Added to knowledge base with clear entry points documented
**Result**:
```
[To be filled]
```

#### Step 1.2: Enumerate sketch.md Features
**Status**: 📝 TODO
**Started**: 
**Action**: Create numbered list of all proposed features from sketch.md
**Success Criteria**: Complete feature inventory for systematic analysis
**Result**:
```
[To be filled]
```

#### Step 1.3: Meta-step - Generate Feature Analysis Steps
**Status**: 📝 TODO
**Started**: 
**Action**: Based on Step 1.2 results, dynamically create Steps 1.3.1 through 1.3.N for each feature
**Success Criteria**: New steps added to plan for each enumerated feature
**Result**:
```
[To be filled]
```

#### Step 1.4: Combined Critical Review
**Status**: 📝 TODO
**Started**: 
**Action**: Synthesize all feature analyses into comprehensive review
**Success Criteria**: Cross-cutting concerns and integration challenges documented
**Result**:
```
[To be filled]
```

#### Step 1.5: Create sketch1X.md
**Status**: 📝 TODO
**Started**: 
**Action**: Write refined specification incorporating all analysis
**Success Criteria**: Complete, self-contained spec created
**Result**:
```
[To be filled]
```

#### Step 1.6: Compare and Refine sketch1X.md
**Status**: 📝 TODO
**Started**: 
**Action**: Diff against original sketch.md and add missing elements
**Success Criteria**: No features lost, all improvements incorporated
**Result**:
```
[To be filled]
```

#### Step 2.0: Meta-step - Generate Phase 2 Steps
**Status**: 📝 TODO
**Started**: 
**Action**: Review Phase 1 results and create user persona/scenario development steps
**Success Criteria**: Dynamic Phase 2 steps added based on Phase 1 findings
**Result**:
```
[To be filled]
```

#### Step 3.0: Meta-step - Generate Phase 3 Role Play Steps
**Status**: 📝 TODO
**Started**: 
**Action**: Based on Phase 2 personas × scenarios, create individual role play steps
**Success Criteria**: Matrix of role play steps created, one per persona/scenario combo
**Result**:
```
[To be filled]
```

#### Step 4.0: Meta-step - Generate Phase 4 Synthesis Steps
**Status**: 📝 TODO
**Started**: 
**Action**: Review Phases 2 & 3, create steps for sketch3X.md development
**Success Criteria**: Systematic update steps for final specification
**Result**:
```
[To be filled]
```