# Issue 837 – GFQL cuDF Wavefront Executor Plan
**THIS PLAN FILE**: `plans/issue_837_cudf_executor/plan.md`
**Created**: 2025-11-18 02:10 UTC
**Current Branch**: `feat/issue-837-cudf-hop-executor`
**PR**: N/A (new)
**Base Branch**: `master`

## CRITICAL META-GOALS OF THIS PLAN
1. Fully self-describing
2. Constantly updated
3. Single source of truth
4. Safe to resume

## Execution Protocol
Follow template instructions (reload plan, only edit active phase, log tool calls, etc.).

### Divide & Conquer + Checkpoint Policy
- Split every implementation phase into bite-sized sub-steps that can be independently validated (tests + lint) before moving on.
- After each sub-step reaches green, capture the work with a clean semantic conventional commit (e.g., `feat: add cudf forward summaries`) and push to the remote branch (`git status`, `git add`, `git commit -m ...`, `git push origin feat/issue-837-cudf-hop-executor`).
- Never allow unchecked intermediate states to accumulate; rollback locally if tests fail and only checkpoint once stable.
- Document completed sub-steps + commit hashes inside the relevant phase entries here so resuming agents know the precise cut lines.

## Context (READ-ONLY)

### Objective
Implement issue #837 by delivering a cuDF-based forward/backward/forward GFQL executor for linear chains that preserves existing local-only semantics and introduces same-path WHERE predicate enforcement (inequalities via min/max summaries, equality/!= via bitsets or bounded state tables). The executor must remain set-based (returning nodes/edges), enforce null semantics, and include planner hooks to enable predicate-specific structures only when needed.

### Current State
- Branch `feat/issue-837-cudf-hop-executor` freshly created from `master`.
- Reference oracle (`graphistry/gfql/ref/enumerator.py`) + tests in place from issue #836/#835.
- pandas GFQL chain executor exists; cuDF executor currently limited to local predicates without WHERE tracking.

### Success Criteria
1. cuDF F/B/F executor supports local predicates and same-path WHERE comparisons under set semantics.
2. Planner enables min/max, bitsets, or state tables only when referenced (pay-as-you-go switches).
3. Null/NaN comparisons return False consistently.
4. Hybrid strategy planning (sparse gather vs cuDF join) deferred to separate issue unless proven critical.
5. cuDF executor matches oracle outputs on small graphs (unit tests comparing vs enumerator).
6. Documentation/changelog entries describing new executor + planner behavior.

### Related Plans
- Previous: `plans/issue_836_enumerator/plan.md` – delivered reference oracle + tests; this plan will consume that oracle.

### Git Strategy
- Single branch `feat/issue-837-cudf-hop-executor` → one PR targeting `master`.

## Status Legend
📝 TODO · 🔄 IN_PROGRESS · ✅ DONE · ❌ FAILED · ⏭️ SKIPPED · 🚫 BLOCKED

## Phases

### Phase 1.A – Scope & Research Snapshot
**Status:** ✅ DONE  
**Branch:** `feat/issue-837-cudf-hop-executor`  
**PR:** N/A  
**Issues:** #837  
**Started:** 2025-11-18 02:10 UTC  
**Completed:** 2025-11-18 02:35 UTC  
**Description:** Understand existing pandas executor + planner hooks, review cuDF capabilities (joins, groupby apply, bitset ops), and clarify deferred items (hybrid hop selection). Capture open questions about interface, planner toggles, null semantics, and equality strategy (bitset vs state table).  
**Actions:**
```bash
rg -n "class Chain" graphistry/compute
rg -n "def hop" graphistry/compute
rg -n "cuDF" graphistry/compute -g"*.py"
python - <<'PY'
# quick sanity: pandas chain currently supports local predicates only
import pandas as pd
from graphistry.tests.test_compute import CGFull
from graphistry.compute import n, e_forward
nodes_df = pd.DataFrame([
    {"id": "acct1", "type": "account", "owner_id": "user1"},
    {"id": "acct2", "type": "account", "owner_id": "user2"},
    {"id": "user1", "type": "user"},
    {"id": "user2", "type": "user"},
])
edges_df = pd.DataFrame([
    {"edge_id": "e1", "src": "acct1", "dst": "user1", "type": "owns"},
    {"edge_id": "e2", "src": "acct2", "dst": "user2", "type": "owns"},
])
g = CGFull().nodes(nodes_df, "id").edges(edges_df, "src", "dst", edge="edge_id")
ops = [n({"type": "account"}, name="a"), e_forward({"type": "owns"}, name="r"), n({"type": "user"}, name="c")]
res = g.gfql(ops)
print(res._nodes)
print(res._edges)
PY
```
- Recorded planner/engine questions and deferred the hybrid hop-selection bullet per instructions.  
**Success Criteria:** Written summary (in plan) covering current executor behavior, required cuDF features, planner toggles, null semantics, equality strategy, and list of open questions.  
**Result:** pandas executor currently labels nodes via boolean columns but lacks WHERE enforcement; cuDF hop code exists but without same-path summaries. Requirements confirmed from issue #837. cuDF needs joins, groupby aggregations, bitset/state-table structures; null comparisons must stay False. Planner must expose switches to enable min/max or bitsets only when needed; hybrid CSR/CSC gather vs join deferred to separate issue. Open questions captured for Phase 1.B: bitset lane sizing, state-table caps/location, interface for planner hints, GPU-friendly equality fallback.

### Phase 1.B – Design Planner & Executor Interfaces
**Status:** ✅ DONE  
**Description:** Draft architecture: planner API for enabling min/max, bitsets, state tables; executor data structures; hopping flow (forward/backward). Consider deferring hybrid gather vs join to follow-up issue.  
**Actions:**
```bash
# Design-only phase (no code)
```
- Defined planner toggle structs, executor dataflow, state-table/bitset placement, null semantics, and interface updates.  
**Success Criteria:** Documented design in plan (diagrams/notes) + identified modules/files to modify.  
**Result:**
Planner: extend existing GFQL planner (likely `graphistry/compute/chain.py` + `graphistry/compute/plan.py`) with a `SamePathPlan` dataclass capturing `minmax_aliases: Dict[str, Set[str]]`, `bitset_aliases: Dict[str, BitsetPlan]`, `state_tables: Dict[str, StateTablePlan]`, plus `null_policy` (fixed False semantics) and `hybrid_strategy` placeholder (deferred). Planner resolves WHERE clauses → toggle entries; equality domains estimated via stats (sample distinct count or explicit metadata). Interface returns plan to executor along with AST.

Executor architecture (`graphistry/compute/gfql/cudf_executor.py` new module or expansion of existing):
1. **Forward pass**
   - Accept plan toggles.
   - For each step: perform cuDF merge (`frontier` × `edges`) and gather nodes.
   - When alias in `minmax_aliases`, maintain `frontier[['id', col]].groupby('id').agg(['min','max'])` stored in side table keyed by alias.
   - When alias in `bitset_aliases`, maintain `cupy`-backed `uint64[N_lanes]` per node (done via custom kernel or `ufunc`). Keep columns `alias::bitset_lane_k` in `frontier`. Planner gives lane count + value->lane mapping (hash mod lanes for fallback).
   - When alias in `state_tables`, maintain `(alias_id, value)` cuDF DataFrame with cap (drop rows beyond `cap` per alias via `groupby('alias_id').head(cap)`).
   - Early WHERE prune: each iteration, use plan to evaluate clauses whose aliases bound; drop rows.

2. **Backward pass**
   - Start from terminal frontier (after early prune).
   - For inequalities: use stored min/max summaries; when walking backward, drop nodes whose value fails vs partner alias summaries (requires join on alias id).
   - For equality bitsets: propagate bitsets backward (bitwise AND with edge contributions). For state tables: join with `(alias,value)` tables to filter edges/nodes.
   - Continue until reaching first step; final node/edge sets computed via merges.

3. **Planner toggles application**
   - Planner attaches toggles to AST nodes via metadata (e.g., `ast_node.same_path_plan`). Executor reads them to know which summarizers to build.
   - Hybrid gather vs join deferred: include placeholder flag `plan.hybrid_strategy = None` to revisit later (future GH issue).

Modules touched:
   - `graphistry/compute/chain.py` / planner stage for building `SamePathPlan`.
   - `graphistry/compute/gfql/plan.py` (new or existing) for toggle dataclasses.
   - `graphistry/compute/gfql/cudf_executor.py` (new) or similar for F/B/F logic.
   - `graphistry/tests/gfql` (or new `tests/gfql/ref`) for executor tests.

### Phase 1.B.1 – GFQL JSON Syntax Simulations
**Status:** ✅ DONE  (feeds Phase 1.C)  
**Description:** Author user-facing GFQL JSON query scenarios (plans/issue_837_cudf_executor/stories/scenario_*.md) exploring same-path WHERE syntax consistent with existing GFQL JSON and Cypher expectations. Cover diverse user/task goals, write each scenario, document pain points, and iterate until syntax feels natural.  
**Actions:**
```bash
# create scenarios in plans/issue_837_cudf_executor/stories/scenario_XXX.md
```
- Minimum of 3 batches of scenarios (personal/task variations) with conclusions folded back into plan.  
**Success Criteria:** Scenario files documenting JSON snippets + lessons learned; plan updated with chosen syntax conventions / unresolved issues.  
**Result:** Authored scenario batches 01–03 covering fraud investigator, SOC analyst, and compliance auditor use cases (`stories/scenario_batch01.md`–`03.md`). Each proposes GFQL JSON with `chain` entries and `where` array using `alias.column` references and operation objects (`eq`, `gt`, `between`, etc.). Concluded alias.column syntax feels natural; need validation for alias existence/column names; planner must support complex ops (e.g., `between`). Syntax insights recorded for future parser work and will inform Phase 1.C.0.

### Phase 1.C.0 – GFQL WHERE Syntax & Parser Support
**Status:** ✅ DONE  
**Description:** Implement GFQL JSON/GFQL API support for same-path `where` clauses per scenario findings (alias.column references, comparison objects). Update AST (likely `graphistry/compute/ast.py`) and serialization/deserialization so `Chain` captures WHERE metadata.  
**Actions:**
```bash
python3 -m pytest graphistry/tests/compute/test_chain_where.py tests/gfql/ref/test_same_path_plan.py
python3 -m ruff check graphistry/compute/chain.py graphistry/compute/gfql_unified.py graphistry/gfql/same_path_types.py graphistry/tests/compute/test_chain_where.py
```
- Extended `Chain` constructor/to_json/from_json with `where` metadata, added JSON parser (`parse_where_json`) + formatter, and taught `gfql()` to accept dicts of the form `{ "chain": [...], "where": [...] }`.  
**Success Criteria:** GFQL chains accept `where` clauses in JSON and Python APIs, producing `WhereComparison` metadata available to planner/executor.  
**Result:** `graphistry/gfql/same_path_types.py` now exposes `parse_where_json`/`where_to_json`; `Chain` stores `.where`; GFQL dict inputs with `chain`+`where` become `Chain` objects. New tests (`graphistry/tests/compute/test_chain_where.py`, `tests/gfql/ref/test_same_path_plan.py`) cover round-trip parsing. Enumerator/oracle remain unchanged but can consume `WhereComparison` structures later.
**Open Follow-ups:** `call()` mixers or divide-and-conquer shorthand may need WHERE scoping safeguards; capture decisions before planner wiring. Add case-analysis/simulation phase before implementing those patterns.


### Phase 1.B.2 – call()/Divide-and-Conquer Scenarios
**Status:** ✅ DONE  (scenarios in stories/scenario_batch04_call.md)  
**Description:** Simulate GFQL JSON/Python chains involving side-effecting `call()` operations and divide-and-conquer sugar to understand WHERE scoping needs. Document corner cases before planner/executor wiring.  
**Actions:**
```bash
# add scenario files under plans/issue_837_cudf_executor/stories/
```
- Analyze multiple cases (call boundaries, nested sugar) and record conclusions.  
**Success Criteria:** Scenario notes capturing WHERE scoping rules and constraints for `call()`/divide-and-conquer patterns.  
**Result:** Scenario batch 04 highlights that same-path clauses should ignore aliases introduced inside side-effecting `call()` blocks unless explicitly scoped, and each branch of divide-and-conquer sugar must be treated independently. Planner/executor must respect alias locality before enabling summaries.

Open implementation questions recorded for Phase 1.C: where to store alias stats (in executor vs plannner), bitset lane hashing, GPU kernel for OR, memory caps for state tables, fallback for equality domain detection.

### Phase 1.C.1 – Planner Toggle Implementation
**Status:** ✅ DONE  
**Description:** Implement planner data structures + resolution logic that maps WHERE clauses to min/max, bitset, or state-table toggles. Integrate with AST/planner pipeline.  
**Actions:**
```bash
python3 -m pytest tests/gfql/ref/test_same_path_plan.py tests/gfql/ref/test_ref_enumerator.py
```
- Added shared same-path types + planner module; updated enumerator/tests.  \n**Success Criteria:** Planner attaches toggle metadata to chains; unit/integration tests (planner-level) run locally.  \n**Result:** Implemented `SamePathPlan`, `BitsetPlan`, `StateTablePlan`, and `plan_same_path()` heuristics. Enumerator now imports shared types. Planner still awaits upstream WHERE syntax to attach metadata automatically—recorded as follow-up. Tests above pass locally.

### Phase 1.C.2 – cuDF Forward Pass Enhancements
**Status:** ✅ DONE  
**Description:** Implement cuDF forward pass that honors planner toggles (min/max summaries, bitsets/state tables, early WHERE pruning).  
**Actions:**
```bash
# TBD: will add commands once GPU env available
```
- Build cuDF executor scaffolding (`graphistry/compute/gfql/cudf_executor.py`), integrate planner toggles, and implement early WHERE pruning.  
**Sub-Steps (divide & conquer + checkpoint after each):**
1. Scaffold cuDF executor module + stub interfaces (commit `feat: scaffold cudf executor skeleton`).
2. Wire planner toggles + data prep structures, add targeted planner-unit tests (`feat: wire same-path plan into cudf executor`).
3. Implement forward traversal (joins, early WHERE prune) with temporary CPU guard / TODOs for GPU specifics (`feat: implement cudf forward wavefront`).
4. Add minimal parity tests vs oracle for forward-only paths (`test: add cudf forward parity cases`).
5. Each sub-step: run targeted pytest suite + ruff, then clean semantic commit + push noted above.
**Success Criteria:** Forward pass builds required structures and passes targeted unit tests (mock data). Log commit hashes for each sub-step once landed.
- **Progress Log:**
  - ✅ Sub-step 1 scaffolding: Added `graphistry/compute/gfql/cudf_executor.py` with executor skeleton + helper constructors; lint via `python3 -m ruff check graphistry/compute/gfql/cudf_executor.py`; committed as `feat: scaffold cudf executor skeleton` (`84021ad6`) and pushed.
  - ✅ Sub-step 2 planner wiring: collected alias metadata + column requirements in `cudf_executor.py`, added validation helpers + tests (`tests/gfql/ref/test_cudf_executor_inputs.py`), ran `python3 -m pytest tests/gfql/ref/test_cudf_executor_inputs.py` and ruff; committed as `feat: wire same-path plan into cudf executor` (`3aca848c`) and pushed.
  - ✅ Sub-step 3 forward traversal: Implemented `_forward()` with AST execution + alias frame capture + early WHERE pruning (equality + min/max heuristics), added tests ensuring alias frames + pruning, commands: `python3 -m pytest tests/gfql/ref/test_cudf_executor_inputs.py`, `python3 -m ruff check graphistry/compute/gfql/cudf_executor.py tests/gfql/ref/test_cudf_executor_inputs.py`; committed as `feat: implement cudf executor forward pass` (`8131245e`) and pushed.
  - ✅ Sub-step 4 forward parity tests: Added oracle-vs-forward alias comparisons (equality + inequality scenarios) in `tests/gfql/ref/test_cudf_executor_inputs.py`; commands `python3 -m pytest tests/gfql/ref/test_cudf_executor_inputs.py` and `python3 -m ruff check graphistry/compute/gfql/cudf_executor.py tests/gfql/ref/test_cudf_executor_inputs.py`; committed as `test: add cudf forward parity cases` (`4b36220c`) and pushed.

### Phase 1.C.3 – cuDF Backward Pass & Finalization
**Status:** 📝 TODO  
**Description:** Implement backward pass intersection logic using summaries; ensure outputs match expectations; handle null semantics.  
**Sub-Steps (with semantic checkpointing as above):**
1. Backward propagation for inequalities (min/max summaries) – commit `feat: cudf backward inequalities`.
2. Backward propagation for equality/!= via bitsets/state tables – commit `feat: cudf backward equality`.
3. Final F/B/F glue + output materialization – commit `feat: cudf wavefront finalize`.
4. Local parity tests vs oracle – commit `test: add oracle parity for cudf executor`.
5. Document commit IDs + pushes in this phase entry.
**Success Criteria:** Combined F/B/F executor produces expected node/edge sets in unit tests; integration with planner metadata complete and checkpointed commits pushed.

### Phase 1.D – Testing & Oracle Validation
**Status:** 🚫 BLOCKED  
**Description:** Add cuDF-backed tests comparing executor vs oracle on small graphs; property/metamorphic checks.  
**Blocking Reason:** Requires executable cuDF forward/backward path (Phases 1.C.2–1.C.3).  
**Success Criteria:** Tests in `tests/gfql/ref/` or new suite; CI scripts updated if needed.

### Phase 1.E – Docs & Finalization
**Status:** 🚫 BLOCKED  
**Description:** Update docs (GFQL README / AI notes), changelog, PR summary. Final lint/mypy/pytest runs.  
**Blocking Reason:** Depends on completion of execution + test phases.  
**Success Criteria:** Documentation updated; `python -m pytest` (key suites), `ruff`, `mypy` clean; PR ready.

---
*Plan created: 2025-11-18 02:10 UTC*

### Research Notes
- Planner toggle matrix drafted under `plans/issue_837_cudf_executor/stories/planner_toggle_matrix.md`.
- Flow scenario (`hop_where_flow.md`) documents how min/max + equality summaries propagate in cuDF.
- cuDF/CuPy not installed locally: GPU-specific kernels must be validated in CI/docker (noted in story). Pandas prototype confirms min/max aggregation logic.
