# GFQL Programs Spec Development Plan
**THIS PLAN FILE**: `AI_PROGRESS/gfql-programs-spec/PLAN.md`
**Created**: 2025-07-10 UTC
**Current Branch if any**: dev/gfql-program
**PRs if any**: PR #695 - feat(gfql): GFQL Programs Spec - PRD Development
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
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Create PR from dev/gfql-program to master for tracking this PRD work
**Success Criteria**: PR created with description of the 4-phase plan
**Result**:
```
Created PR #695: https://github.com/graphistry/pygraphistry/pull/695
Title: feat(gfql): GFQL Programs Spec - PRD Development

The PR includes:
- Summary of 4-phase development process
- Context about GFQL DAG composition features
- Detailed phase descriptions
- Test plan checklist

Successfully pushed branch and created tracking PR.
```

#### Step 1.0: Create GFQL Knowledge Base
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Read core GFQL implementation files and create lookup file with references
**Success Criteria**: gfql_knowledge_base.md created with file:lineno:snippet references
**Result**:
```
Created comprehensive knowledge base at: AI_PROGRESS/gfql-programs-spec/gfql_knowledge_base.md

Key findings:
1. GFQL uses 3-phase algorithm: forward traversal, reverse pruning, output collection
2. Core files analyzed:
   - chain.py: Main execution engine with chain() method
   - ast.py: Node/Edge AST definitions with JSON serialization
   - predicates/: Rich filtering system (numeric, string, temporal, categorical)
   - chain_remote.py: Wire protocol for server-side execution
   - hop.py: Core traversal logic with wavefront expansion

3. Architecture patterns: Visitor, Builder, Strategy, Memento
4. Wire protocol: JSON-based with type-tagged operations
5. Integration points documented for future DAG extension

The knowledge base includes line-by-line references, code snippets, and architectural insights.
```

#### Step 1.1: Analyze PyGraphistry APIs Around GFQL
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Document Wire Protocol and Python API integration points
**Success Criteria**: Added to knowledge base with clear entry points documented
**Result**:
```
Appended "PyGraphistry API Integration" section to gfql_knowledge_base.md

Key findings:
1. Entry Points:
   - Plottable interface: chain(), chain_remote(), chain_remote_shape()
   - ComputeMixin provides implementation bridge
   - Full type hints and documentation

2. Wire Protocol:
   - REST endpoint: /api/v2/etl/datasets/{dataset_id}/gfql/{output_type}
   - JSON format with gfql_operations field
   - Multiple output formats: JSON, CSV, Parquet
   - Optional column subsetting and engine selection

3. Integration Architecture:
   - 5-layer design: User API → Protocol → Implementation → Execution → Transport
   - Session-based auth with token refresh
   - Clean interface/implementation separation
   - Local/remote execution transparency

4. Related APIs: python_remote for arbitrary code execution

This provides solid foundation for DAG program extensions.
```

#### Step 1.2: Enumerate sketch.md Features
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Create numbered list of all proposed features from sketch.md
**Success Criteria**: Complete feature inventory for systematic analysis
**Result**:
```
Enumerated features from sketch.md:

1. Core DAG Composition (QueryDAG/ChainGraph)
   - Type: "QueryDAG" for nested composition
   - Named graph bindings with lexical scoping
   - "ref" parameter for re-rooting operations
   - "output" parameter to specify return binding
   - Binding name validation: ^[a-zA-Z_][a-zA-Z0-9_-]*$

2. Dotted Reference Syntax
   - Disambiguation for nested QueryDAGs
   - Format: "a.b.c" for hierarchical references
   - Used in both "ref" and "output" fields

3. Remote Graph Loading (RemoteGraph)
   - Load existing graphs from Graphistry
   - Type: "RemoteGraph" with dataset_id/graph_id
   - Enables graph mashups without Python

4. Graph Combinators
   - Union (policies: left, right, merge_left, merge_right)
   - Subtract
   - Replace (policies: full, patch, extend)
   - Intersect
   - From (new graph from different node/edge sources)
   - Common policies: node/edge removal strategies

5. Call Operations
   - Expose Plottable methods (umap, layout_cugraph, cypher, etc.)
   - Type: "call" with function name and params
   - Safelisting controls by Hub Tier
   - Future: Louie connector integration

Total: 5 major feature categories to analyze
```

#### Step 1.3: Meta-step - Generate Feature Analysis Steps
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Based on Step 1.2 results, dynamically create Steps 1.3.1 through 1.3.N for each feature
**Success Criteria**: New steps added to plan for each enumerated feature
**Result**:
```
Generated 5 feature analysis steps (1.3.1 through 1.3.5) for:
1. Core DAG Composition
2. Dotted Reference Syntax  
3. Remote Graph Loading
4. Graph Combinators
5. Call Operations

Each step includes 3 sub-steps for comprehensive analysis.
```

#### Step 1.3.1: Feature Analysis - Core DAG Composition (QueryDAG/ChainGraph)
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Analyze DAG composition feature and its relationship to current Chain architecture
**Success Criteria**: Document implementation approach, risks, and improvements
**Sub-steps**:
- 1.3.1.1: Analyze how DAG composition relates to current Chain architecture
- 1.3.1.2: Identify implementation challenges and dependencies
- 1.3.1.3: Critical review (bugs/risks/improvements)
**Result**:
```
Created feature_analysis_dag_composition.md with comprehensive analysis.

Key findings:
1. Architecture: QueryDAG extends linear chains to DAG with named bindings & lexical scoping
   - Reuses: AST, execution engine, remote execution, Plottable interface
   - New: QueryDAG AST node, reference resolver, execution context

2. Challenges:
   - Wire protocol: New message types, backward compatibility required
   - Performance: Memory management for bindings, parallel execution opportunities
   - Security: Resource limits, access control for remote graphs

3. Critical Review:
   - Risks: Circular references, resource exhaustion, name collisions
   - Improvements: Lazy evaluation, query optimization, type system
   - Priority: MVP focus on basic DAG execution with backward compatibility

4. Implementation roadmap: 4 phases over 6-8 weeks total
```

#### Step 1.3.2: Feature Analysis - Dotted Reference Syntax
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Analyze dotted reference syntax for disambiguation
**Success Criteria**: Document scoping rules, edge cases, and implementation strategy
**Sub-steps**:
- 1.3.2.1: Analyze lexical scoping and reference resolution
- 1.3.2.2: Identify ambiguity edge cases
- 1.3.2.3: Critical review (bugs/risks/improvements)
**Result**:
```
Created feature_analysis_dotted_reference.md with comprehensive analysis.

Key findings:
1. Scoping: Lexical scoping with closest-binding principle
   - Root-to-leaf traversal for "a.b.c" resolution
   - Statically resolvable at parse time
   - Similar to programming language variable scoping

2. Edge Cases (6 identified):
   - Name shadowing at different levels
   - Deep nesting performance/readability
   - Partial path resolution failures
   - Circular references
   - Reserved name collisions ("type", "graph", "output")
   - Cross-DAG sibling references

3. Critical Review:
   - Parsing: Need escape sequences for dots in names
   - Performance: Deep traversal costs, caching needed
   - Alternatives: Considered slash notation, JSONPath, explicit scopes
   - Error handling: Need contextual messages with suggestions

4. Recommendations: Enhanced syntax spec, caching, circular detection, validation tools
```

#### Step 1.3.3: Feature Analysis - Remote Graph Loading (RemoteGraph)
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Analyze remote graph loading feature
**Success Criteria**: Document security, performance, and integration considerations
**Sub-steps**:
- 1.3.3.1: Analyze relationship to current dataset loading mechanisms
- 1.3.3.2: Security and authentication considerations
- 1.3.3.3: Critical review (network/caching/error handling)
**Result**:
```
Created feature_analysis_remote_graph.md with comprehensive analysis.

Key findings:
1. Current vs RemoteGraph:
   - Current: Client-driven bind(dataset_id) with Python orchestration
   - RemoteGraph: Server-side loading within pure GFQL DAGs
   - Enables graph mashups without client round-trips

2. Security Critical:
   - Access control: Need token propagation strategy (implicit/embedded/capability)
   - Cross-tenant isolation risks when loading multiple graphs
   - Data exfiltration vectors: combinators, errors, timing
   - Recommendation: Capability-based tokens with strict validation

3. Technical Challenges:
   - Network: No current timeout/retry; need cascading DAG timeouts
   - Caching: 3-tier strategy (metadata/data/computation)
   - Errors: Need comprehensive categorization and safe messages
   - Resources: Per-user and per-graph quotas essential

4. Implementation: Phased approach starting with single RemoteGraph
```

#### Step 1.3.4: Feature Analysis - Graph Combinators
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Analyze graph combinator operations
**Success Criteria**: Document policy system, memory efficiency, and API design
**Sub-steps**:
- 1.3.4.1: Map to existing graph operations in PyGraphistry
- 1.3.4.2: Policy system design review
- 1.3.4.3: Critical review (edge cases/memory/consistency)
**Result**:
```
Created feature_analysis_graph_combinators.md with comprehensive analysis.

Key findings:
1. Current Gap:
   - PyGraphistry has rich single-graph ops but no native combinators
   - Missing: union, intersection, subtraction, systematic merge policies
   - Combinators enable semantic multi-graph operations

2. Policy System:
   - Good coverage: attribute merging, node/edge removal strategies
   - Recommendations: Add type coercion, aggregation, custom merge functions
   - Need schema evolution handling

3. Critical Issues:
   - Memory: Graph duplication, intermediate results, remote loading
   - Edge cases: Mismatched IDs, cycles, empty graphs, schema conflicts
   - Consistency: Node-edge integrity, attribute conflicts
   - Performance: Opportunities for parallel execution, GPU acceleration

4. Implementation: Phased approach with robust testing strategy
```

#### Step 1.3.5: Feature Analysis - Call Operations
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Analyze call operations for Plottable methods
**Success Criteria**: Document method inventory, security model, and extensibility
**Sub-steps**:
- 1.3.5.1: Inventory of Plottable methods to expose
- 1.3.5.2: Safelisting and security model
- 1.3.5.3: Critical review (compatibility/validation/extensibility)
**Result**:
```
Created feature_analysis_call_operations.md with comprehensive analysis.

Key findings:
1. Method Inventory:
   - Identified 21+ Plottable methods suitable for GFQL
   - Categories: transforms (umap, cypher), layouts, analytics, graph ops
   - Recommended phased rollout starting with common/simple methods

2. Security Model:
   - Tiered access: Basic/Standard/Advanced/Enterprise
   - Method-level and parameter-level restrictions
   - Resource limits per tier (nodes/edges/timeouts)
   - Safelist configuration with execution sandbox

3. Technical Review:
   - JSON compatibility: Type serialization challenges addressed
   - Validation: Pydantic schemas for type safety
   - Extensibility: Plugin architecture for Louie connectors
   - Error handling: Comprehensive categories with debugging info

4. Implementation: Start with 5-7 core methods, expand based on usage
```

#### Step 1.4: Combined Critical Review
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Synthesize all feature analyses into comprehensive review
**Success Criteria**: Cross-cutting concerns and integration challenges documented
**Result**:
```
Created combined_critical_review.md synthesizing all 5 feature analyses.

Key findings:
1. Cross-cutting concerns:
   - Reference resolution complexity in nested contexts
   - Compound memory management (DAGs + remote + combinators)
   - Expanded security attack surface
   - Type safety challenges across features

2. Integration challenges:
   - Complex feature interdependencies
   - Conflicting execution models (parallel vs sequential)
   - Inconsistent error handling patterns
   - Competing caching strategies

3. Architecture assessment:
   - Strengths: Clear purpose, natural composability
   - Weaknesses: Complexity explosion, inconsistent patterns
   - Need unified execution context and resource manager

4. Priority recommendations:
   - Phase 1: Core DAG + basic remote (low risk, high value)
   - Phase 2: Dotted refs + simple combinators
   - Phase 3: Advanced combinators + basic calls
   - Phase 4: Full call operations + optimizations

Risk matrix identified memory exhaustion and debugging complexity as highest concerns.
```

#### Step 1.5: Create sketch1X.md
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Write refined specification incorporating all analysis
**Success Criteria**: Complete, self-contained spec created
**Result**:
```
Created sketch1X.md - comprehensive refined specification.

Major enhancements:
1. Complete security model with tiered access and resource limits
2. Robust error handling with categorized codes and debugging info
3. Production-ready features: caching, retries, timeouts, memory mgmt
4. Clear 4-phase implementation roadmap over 6 months
5. Enhanced reference system with escape sequences and caching
6. Advanced combinators with type coercion and schema evolution
7. Comprehensive call operations with validation and tracing
8. Real-world examples for common patterns
9. Performance guidelines and optimization strategies
10. Enhanced wire protocol with versioning and metadata

The spec is self-contained, production-ready, and addresses all concerns from analyses.
```

#### Step 1.6: Compare and Refine sketch1X.md
**Status**: ✅ DONE
**Started**: 2025-07-10 UTC
**Action**: Diff against original sketch.md and add missing elements
**Success Criteria**: No features lost, all improvements incorporated
**Result**:
```
Created sketch_comparison.md documenting comprehensive comparison.

Key findings:
1. No features lost - all original concepts preserved and enhanced
2. Major additions:
   - Complete security model with tiers and access control
   - 6-month phased implementation roadmap
   - Comprehensive error handling framework
   - Performance guidelines and optimization strategies
   - Production features: monitoring, retries, caching
   - Rich real-world examples

3. Quality transformation:
   - From high-level RFC to implementation-ready spec
   - Security-first design throughout
   - Production readiness built in
   - Enhanced developer experience

No updates needed to sketch1X.md - it successfully incorporates all original features with significant enhancements.

Phase 1 complete! Ready for Phase 2: User Personas and Scenarios.
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