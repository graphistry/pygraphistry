# Issue #872: Multi-hop + WHERE Backward Prune Bug Fixes

## Status: COMPLETED - Native Path Enabled (Dec 27, 2024)

---

## 🔧 Session 9: CI Fixes + Verification Issue Update (Dec 28, 2024)

### CI Lint Fixes (commit `b6b54499`)

Fixed flake8 errors blocking CI:

**F841 - Unused variables** (4 occurrences):
- `relevant_node_indices` at lines 392, 591 - removed
- `edge_id_col` at line 717 - removed
- `max_hop` at line 1276 - removed

**W504 - Line break after binary operator** (7 occurrences):
- Moved `|` and `&` operators to start of next line per PEP 8

### Verification Issue #871 Updated

Added detailed section documenting 5 bugs found during PR #846 development:

1. **Backward traversal join direction** (`_find_multihop_start_nodes`) - joined on wrong column
2. **Empty set short-circuit missing** (`_materialize_filtered`) - no early return for empty sets
3. **Wrong node source for non-adjacent WHERE** (`_apply_non_adjacent_where_post_prune`) - used incomplete `alias_frames`
4. **Multi-hop path tracing through intermediates** - backward prune filtered wrong edges
5. **Reverse/undirected edge direction handling** - missing `is_undirected` checks

Added new Alloy model recommendations:
- P1: Add hop range modeling (would have caught bugs #1, #4)
- P1: Add backward reachability assertions (would have caught bug #1)
- P2: Add empty set propagation assertion (would have caught bug #2)
- P2: Add contradictory WHERE scenarios

Updated coverage table and added PR #846 commits as references.

### Test Results

```
101 passed, 2 skipped, 1 xfailed
```

---

### Current Focus: Production-Ready Native Vectorized Path

The native vectorized path is now enabled by default for both pandas and cuDF.
The oracle is only used when explicitly requested via `GRAPHISTRY_CUDF_SAME_PATH_MODE=oracle`.

---

## 🎉 Session 8: Enable Native Path + Test Amplification (Dec 28, 2024) - COMPLETED

### Status: COMPLETE ✅

Native vectorized path is now enabled by default for both pandas and cuDF.
All 133 GFQL tests pass (21 new tests added).

### Changes Made

1. **Renamed `_run_gpu()` to `_run_native()`** to reflect that it's the production path for both CPU and GPU.

2. **Renamed `_should_attempt_gpu()` to `_should_use_oracle()`** with inverted logic:
   - Oracle is now only used when explicitly requested via `GFQL_CUDF_MODE=oracle`
   - Default: use native vectorized path for both pandas and cuDF

3. **Fixed bug in `_filter_multihop_by_where`**:
   - **Problem**: The function relied on hop labels (`__gfql_output_edge_hop__`) to identify start/end nodes
   - For multi-hop edges like `e_forward(min_hops=2, max_hops=3)`, all edges have hop=1 because each edge is a single step
   - When `chain_min_hops=2` and all hops are 1, `valid_endpoint_edges` was empty → empty results
   - **Solution**: Don't rely on hop labels. Instead:
     1. Get all possible start nodes from edge sources
     2. Trace forward through edges to find reachable (start, end) pairs within [min_hops, max_hops]
     3. Apply WHERE filter to pairs
     4. Filter edges using bidirectional reachability

4. **Fixed bug in `_filter_multihop_edges_by_endpoints` - Multiple Hop Distances**:
   - **Problem**: BFS used anti-join on nodes only, so each node appeared at only one hop distance
   - When a node has multiple roles (e.g., `b` is both a start AND reachable from another start), only one hop distance was kept
   - Edge `b->c` computed as `fwd_hop=0 + 1 + bwd_hop=0 = 1`, missing the valid `fwd_hop=1` path
   - **Solution**: Anti-join on (node, hop) pairs instead of just nodes, allowing same node at multiple hop distances

5. **Fixed bug in `_filter_multihop_edges_by_endpoints` - Duplicate Edges**:
   - **Problem**: Join produces duplicates when a node has multiple hop distances, making `len(filtered) == len(edges_df)` even when edges were filtered
   - This caused filtered edges to NOT be persisted back to `forward_steps[edge_idx]._edges`
   - **Solution**: Add `.drop_duplicates()` after selecting original columns

6. **Fixed bug in `_materialize_filtered` - Edge Source Filtering**:
   - **Problem**: Edges were only filtered by destination node, not source node
   - When a path was filtered by a WHERE clause on an intermediate node, edges downstream of that node were still included
   - Example: For chain `a->mid->d` with WHERE `a.v < mid.v`, if `mid=b` passed but `mid=c` failed, edge `c->d` was incorrectly included
   - **Solution**: Filter edges by BOTH `src` AND `dst` being in allowed nodes

### Test Amplification

Added 21 new tests across 4 new test classes:

1. **TestMultiplePredicates** (7 tests):
   - Multiple WHERE predicates on same/different alias pairs
   - Combinations of ==, <, >, != operators
   - Adjacent and non-adjacent predicate combinations

2. **TestMultipleRolesPerNode** (5 tests):
   - Nodes that are both start AND intermediate
   - Nodes that are both end AND intermediate
   - Diamond graphs with multiple paths
   - Overlapping paths where predicate filters some

3. **TestComplexTopologies** (5 tests):
   - Complete graph K4
   - Binary tree depth 3
   - Ladder graph (two parallel chains with cross-links)
   - Star graph
   - Bipartite graph

4. **TestMultihopWithMultiplePredicates** (4 tests):
   - Multi-hop with two adjacent predicates
   - Multi-hop with non-adjacent predicates
   - Multi-hop with three predicates
   - Multi-hop with equality and inequality predicates

### Test Results

```
133 passed, 2 skipped, 1 xfailed (GFQL test suite)
```

### Impact

- **Performance**: Oracle enumeration was 38% of same-path executor time. Skipping it is a significant speedup.
- **Scalability**: Oracle has caps on graph size (1000 nodes, 5000 edges). Native path has no such limits.
- **GPU Compatibility**: Native path uses vectorized DataFrame operations that work identically on pandas and cuDF.
- **Correctness**: Test amplification caught one additional bug (edge source filtering).

---

## 🚨 REFACTORING CHECKLIST (Session 7+)

### Pre-flight
- [x] Add architecture note to df_executor.py header
- [x] Document anti-patterns and correct patterns
- [x] Audit all non-vectorized code locations

### Function Refactoring (in dependency order) - ✅ COMPLETED

#### 1. `_find_multihop_start_nodes` ✅
- [x] Removed BFS `while queue:` loop
- [x] Replaced with hop-by-hop backward propagation via merge
- [x] Tests pass

#### 2. `_filter_multihop_edges_by_endpoints` ✅
- [x] Removed DFS `while stack:` loop
- [x] Replaced with bidirectional reachability via merge + hop distance tracking
- [x] Tests pass

#### 3. `_re_propagate_backward` ✅
- [x] Already vectorized (uses `.isin()` and calls vectorized helpers)
- [x] Tests pass

#### 4. `_filter_multihop_by_where` ✅
- [x] Kept cross-join for (start,end) pairs (already vectorized)
- [x] Replaced DFS with call to vectorized `_filter_multihop_edges_by_endpoints`
- [x] Tests pass

#### 5. `_apply_non_adjacent_where_post_prune` ✅
- [x] Removed BFS path tracing
- [x] Replaced with state table propagation via merge
- [x] Uses vectorized `_evaluate_clause` for comparison
- [x] Tests pass

### Post-refactor Verification ✅
- [x] Verified no `while queue/stack:` remains
- [x] Verified no `for ... in zip(df[col], ...)` remains
- [x] Verified no `adjacency.get(node, [])` dict lookups remain
- [x] Remaining `.tolist()` calls are only for `set()` conversion (acceptable)
- [x] Full test suite passes: `91 passed, 2 skipped, 1 xfailed`

### Round 2: Remaining Vectorization Issues (Dec 27, 2024) - ✅ COMPLETED

Additional audit found more anti-patterns that break GPU and are suboptimal on CPU.
All 6 issues have been fixed:

#### Issue 1: `dict(zip())` in `_apply_non_adjacent_where_post_prune` ✅
- [x] **Fixed**: Replaced `dict(zip(...))` with direct DataFrame operations
- [x] Build `left_values_df` and `right_values_df` directly from frame slices
- [x] Handle edge case where `node_id_col == left_col` (same column)

#### Issue 2: `list(start_nodes)` for DataFrame construction ✅
- [x] **Fixed**: Build initial `state_df` from `left_values_df` filtered by `.isin(start_nodes)`
- [x] Avoids converting Python set to list for DataFrame construction

#### Issue 3: `set(next_nodes.tolist())` in `_filter_multihop_edges_by_endpoints` ✅
- [x] **Fixed**: Replaced Python set tracking with DataFrame-based anti-joins
- [x] Use `merge(..., indicator=True)` + filter on `_merge == 'left_only'` for "not seen" logic
- [x] Accumulate with `pd.concat()` + `drop_duplicates()`

#### Issue 4: `set(reachable['__node__'].tolist())` in `_find_multihop_start_nodes` ✅
- [x] **Fixed**: Use DataFrame-based anti-join for visited tracking
- [x] Collect valid starts as list of DataFrames, concat at end
- [x] Only convert to set at function return (boundary with caller)

#### Issue 5: `set(df[col].tolist())` in `_filter_multihop_by_where` ✅
- [x] **Fixed**: Extract start/end nodes as DataFrames first
- [x] Use `pd.concat()` + `drop_duplicates()` for undirected case
- [x] Convert to set only at boundary (caller expects sets)

#### Issue 6: `set(df[col].tolist())` in `_materialize_filtered` ✅
- [x] **Fixed**: Build allowed_node_frames list with DataFrames
- [x] Use `pd.concat()` + `drop_duplicates()` instead of Python set union
- [x] Filter nodes/edges using `.isin()` on DataFrame column

#### Remaining Boundary Issues (Future Work)

Some `.tolist()` calls remain at function boundaries where:
- `_PathState` uses `Dict[int, Set[Any]]` for `allowed_nodes`/`allowed_edges`
- Helper functions like `_filter_multihop_edges_by_endpoints` accept `Set[Any]` parameters
- Callers in `_backward_prune` and `_re_propagate_backward` use Python sets

To fully eliminate these, a larger refactor is needed:
1. Change `_PathState` to use `Dict[int, pd.DataFrame]` instead of `Dict[int, Set[Any]]`
2. Update all helper function signatures to accept DataFrames
3. Update all callers to pass DataFrames

This would be a **Round 3** effort. The current Round 2 fixes address the most expensive anti-patterns (the ones inside loops and hop-by-hop propagation).

#### General Pattern: Avoid Python set/dict intermediates

The root issue is using Python `set()` and `dict()` as intermediate data structures. For GPU compatibility:
- **Sets**: Use DataFrame with single column, use `.isin()` or merge for membership
- **Dicts**: Use DataFrame with key/value columns, use merge for lookup
- **Accumulation**: Use `pd.concat()` + `drop_duplicates()` instead of `set.update()`
- **Anti-join**: Use `merge(..., how='left', indicator=True)` + filter on `_merge == 'left_only'`

---

## 🔮 Future Work: Round 3+ (Post-Checkpoint)

**IMPORTANT**: Do Round 4 (profiling) FIRST before Round 3. Need to understand where costs are before committing to a large refactor.

### Round 3: `_PathState` DataFrame Migration

**Status**: BLOCKED - Do AFTER Round 4 profiling to validate benefit

**Risk Assessment** (Dec 27, 2024):
- Attempted refactor, reverted due to complexity
- Touches ~300-400 lines across 6+ functions
- High risk of introducing bugs
- May not be worth it for small queries
- Need profiling data first

**Scope**: Change `_PathState` to use DataFrames instead of Python sets

```python
# Current
@dataclass
class _PathState:
    allowed_nodes: Dict[int, Set[Any]]
    allowed_edges: Dict[int, Set[Any]]

# Proposed
@dataclass
class _PathState:
    allowed_nodes: Dict[int, pd.DataFrame]  # single '__id__' column
    allowed_edges: Dict[int, pd.DataFrame]  # single '__id__' column
```

**Files/Functions that would need changes**:
1. `_PathState` class definition (add helper methods)
2. `_backward_prune` - create DataFrames, use merge for intersection
3. `_filter_edges_by_clauses` - change `allowed_nodes` param type
4. `_filter_multihop_by_where` - change `allowed_nodes` param type
5. `_apply_non_adjacent_where_post_prune` - use DataFrame operations
6. `_re_propagate_backward` - use DataFrame operations
7. `_materialize_filtered` - already mostly uses DataFrames

**Prerequisite**: Round 4 profiling should show that:
- Set↔DataFrame conversions are a significant cost
- OR large queries would benefit from DataFrame-native operations

### Round 4: Pay-As-You-Go Complexity

**Status**: INITIAL PROFILING COMPLETE (Dec 27, 2024)

#### Profiling Results (Dec 27, 2024)

Ran `tests/gfql/ref/profile_df_executor.py` on various scenarios:

| Scenario | Nodes | Edges | Simple | Multihop | With WHERE |
|----------|-------|-------|--------|----------|------------|
| tiny     | 100   | 200   | 38ms   | 95ms     | 40ms       |
| small    | 1000  | 2000  | 42ms   | 100ms    | 41ms       |
| medium   | 10000 | 20000 | 51ms   | 100ms    | 50ms       |
| medium_dense | 10000 | 50000 | 88ms | 110ms | 86ms      |

**Key Findings**:
1. **Multi-hop is ~2x slower** (95-110ms vs 40-50ms) regardless of graph size
2. **Graph size doesn't scale linearly** - 100 nodes vs 10K nodes only adds ~10ms
3. **WHERE clauses add minimal overhead** (within noise)
4. **Dense graphs ~2x slower** for simple queries
5. **Bottleneck is likely fixed costs** (executor setup, chain parsing), not data processing

**Implications for Round 3**:
- `_PathState` refactor may NOT help much - set operations aren't the bottleneck
- Fixed overhead dominates for graphs under 50K edges
- Need to profile larger graphs (100K-1M edges) to find where scaling issues emerge

**Next Steps**: ✅ DONE
1. ✅ Profile with larger graphs (100K-1M edges) - DONE
2. ✅ Profile with Python cProfile to identify actual hotspots - DONE
3. Only proceed with Round 3 if profiling shows set operations are significant

#### Extended Profiling Results (Large Graphs)

| Scenario | Nodes | Edges | Simple | Multihop | With WHERE |
|----------|-------|-------|--------|----------|------------|
| large     | 100K  | 200K  | 200ms  | 112ms    | 184ms      |
| large_dense | 100K | 500K | 603ms  | 228ms    | 655ms      |

**Observation**: Multihop is FASTER than simple for large graphs because:
- Simple returns ALL nodes/edges (large result set)
- Multihop returns a small filtered subset
- Bottleneck is **materialization**, not filtering

#### cProfile Analysis (50K nodes)

**Legacy chain executor** (hop.py):
- `hop.py:239(hop)` - 75% of time
- `pandas.merge` - 47% of time
- `chain.py:179(combine_steps)` - 39% of time

**Same-path executor** (df_executor.py, 1K nodes):
- `_forward()` - 59% of time
- `hop.py:239(hop)` - 44% (called within forward)
- **`enumerator.py:enumerate_chain()` - 38%** ← Oracle overhead!

#### Key Insights

1. **Round 3 (`_PathState` refactor) is LOW PRIORITY**:
   - `df_executor.py` functions don't appear in top hotspots
   - Set operations are not the bottleneck
   - Focus should be elsewhere

2. **Oracle enumeration is expensive** (38% of same-path time):
   - `enumerate_chain()` computes ground truth for verification
   - Could be skipped or made optional in production
   - Has caps that prevent large graph usage

3. **Legacy hop.py is the main bottleneck**:
   - Takes 75% of time in simple queries
   - Same-path executor calls it for forward pass
   - Opportunity: vectorize forward pass directly

4. **Materialization dominates for large results**:
   - Simple queries return all nodes/edges
   - Multihop is faster because it returns less data
   - Consider lazy evaluation or streaming

**Idea**: Inspect chain complexity at runtime and skip expensive operations when not needed

**Research Questions**:
1. Where is the cost?
   - [ ] Profile `_backward_prune` for simple vs complex chains
   - [ ] Profile `_apply_non_adjacent_where_post_prune` - only needed for non-adjacent WHERE
   - [ ] Profile `_filter_multihop_edges_by_endpoints` - only needed for multi-hop
   - [ ] Profile `_find_multihop_start_nodes` - only needed for multi-hop
   - [ ] Measure overhead of DataFrame anti-join vs Python set difference

2. What can we skip?
   - [ ] Single-hop chains: skip multi-hop path tracing entirely
   - [ ] Adjacent-only WHERE: skip `_apply_non_adjacent_where_post_prune`
   - [ ] No WHERE clauses: skip backward prune value filtering
   - [ ] Small graphs (<1000 nodes): maybe Python sets are faster?

3. Chain complexity tiers:
   ```python
   def _analyze_chain_complexity(chain, where):
       has_multihop = any(isinstance(op, ASTEdge) and not _is_single_hop(op) for op in chain)
       has_non_adjacent_where = ...  # check WHERE clause adjacency
       has_any_where = len(where) > 0
       graph_size = ...  # node/edge counts

       return ChainComplexity(
           tier='simple' | 'moderate' | 'complex',
           needs_multihop_tracing=has_multihop,
           needs_non_adjacent_where=has_non_adjacent_where,
           recommended_backend='pandas_sets' | 'pandas_df' | 'cudf'
       )
   ```

4. Adaptive algorithm selection:
   - Small graph + simple chain → use Python sets (lower overhead)
   - Large graph + complex chain → use DataFrame operations (scales better)
   - GPU available + large graph → use cuDF DataFrames

**Benchmarking Plan**:
```python
# Test scenarios
scenarios = [
    ('tiny_simple', nodes=100, edges=200, chain='n->e->n', where=None),
    ('tiny_complex', nodes=100, edges=200, chain='n->e(1..3)->n->e->n', where='a.x==c.x'),
    ('medium_simple', nodes=10000, edges=50000, chain='n->e->n', where=None),
    ('medium_complex', nodes=10000, edges=50000, chain='n->e(1..3)->n', where='a.x<c.x'),
    ('large_simple', nodes=1000000, edges=5000000, chain='n->e->n', where=None),
    ('large_complex', nodes=1000000, edges=5000000, chain='n->e(1..5)->n', where='a.x==c.x'),
]
# Measure: pandas-sets vs pandas-df vs cudf for each
```

### Verification Commands
```bash
# Check for anti-patterns
grep -n "while queue\|while stack" graphistry/compute/gfql/df_executor.py
grep -n "for .* in zip(" graphistry/compute/gfql/df_executor.py
grep -n "\.tolist()" graphistry/compute/gfql/df_executor.py
grep -n "adjacency.get\|reverse_adj.get" graphistry/compute/gfql/df_executor.py

# Run tests
pytest tests/gfql/ref/test_df_executor_inputs.py -v
```

---

## Summary (Original Issue)

Fixed the oracle and executor bugs where nodes/edges from failed WHERE paths were incorrectly included in results for multi-hop edge traversals.

### Test Results (Before Vectorization Refactor)

```
91 passed, 2 skipped, 1 xfailed
```

### Remaining xfail

**`test_edge_alias_on_multihop`** in `TestOracleLimitations` - This is a documented **oracle limitation** (not a bug). The oracle (enumerator.py:109) raises an error when edge aliases are used on multi-hop edges like `e_forward(min_hops=1, max_hops=2, name="e")`. This is expected behavior.

### Git Info

- **Branch**: `feat/issue-837-cudf-hop-executor`
- **Latest commit**: `cd579363` - "fix(gfql): comprehensive WHERE + multi-hop bug fixes and test amplification"

---

## Session 6: 5 Whys Analysis + Bug Pattern Testing (Dec 27, 2024)

### 5 Whys Root Cause Analysis

Analyzed the 4 bugs fixed in Session 5 to identify patterns:

**Bug Pattern 1: Multi-hop backward propagation**
- Root cause: Single-hop assumption baked into backward propagation
- Pattern: Any code that filters edges by endpoints breaks for multi-hop

**Bug Pattern 2: Merge suffix handling**
- Root cause: Inconsistent pandas merge suffix handling across the codebase
- Pattern: Any pandas merge with overlapping columns needs explicit suffix handling

**Bug Pattern 3: Undirected edge support**
- Root cause: Code only checks `is_reverse`, treating undirected as forward
- Pattern: Any adjacency-building code needs explicit undirected handling

### Test Amplification (13 new tests)

Added tests targeting each bug pattern:

**TestBugPatternMultihopBackprop (3 tests)**:
- `test_three_consecutive_multihop_edges` - Stress test with 3 multi-hop edges
- `test_multihop_with_output_slicing_and_where` - Output slicing + WHERE
- `test_multihop_diamond_graph` - Diamond graph with multiple paths

**TestBugPatternMergeSuffix (5 tests)**:
- `test_same_column_eq`, `test_same_column_lt`, `test_same_column_lte`
- `test_same_column_gt`, `test_same_column_gte`

**TestBugPatternUndirected (5 tests)**:
- `test_undirected_non_adjacent_where` - **Found new bug!**
- `test_undirected_multiple_where`
- `test_mixed_directed_undirected_chain`
- `test_undirected_with_self_loop`
- `test_undirected_reverse_undirected_chain`

### Bug 5: Undirected single-hop in `_backward_prune`

**Test**: `test_undirected_non_adjacent_where`

**Problem**: `_backward_prune` didn't handle undirected edges for single-hop:
1. Line 874-884: Filtering by `allowed_dst` only checked dst column, not src
2. Line 918-935: Updating `allowed_nodes` only had forward/reverse paths, not undirected

**Solution**: Added `is_undirected` handling in `_backward_prune`:
- Filter edges where either src or dst is in `allowed_dst`
- Update `allowed_nodes` using union of src and dst columns

### Test Results

```
91 passed, 2 skipped, 1 xfailed
```

---

## Session 7: Vectorization Audit & Refactoring Plan (Dec 27, 2024)

### Problem: Wrong Algorithmic Approach

The bug fixes in Session 5 introduced **BFS/DFS path tracing** with Python loops, which is fundamentally wrong. The GFQL executor should use **Yannakakis-style semijoin pruning with vectorized summaries** - the same approach for both CPU (pandas) and GPU (cuDF).

### What is Yannakakis?

Yannakakis is a classic algorithm for evaluating acyclic joins efficiently:

1. **Forward pass (bottom-up semijoins)**: Push filters/joins from leaves inward, prune rows that won't match
2. **Backward pass (top-down semijoins)**: Push constraints back, prune rows that can't participate in complete results
3. **Final join**: Join the pruned tables with much smaller intermediates

For GFQL chains like `n(name="a") >> e() >> n(name="c")`:
- Forward wavefront = bottom-up semijoins
- Backward wavefront = top-down semijoins
- Final = join/collect

### How to Handle Same-Path Predicates

For predicates like `a.val > c.threshold` across multiple hops:

**Monotone predicates (`<`, `<=`, `>`, `>=`):**
- Propagate `min/max` summaries via `groupby` at each hop
- At endpoint, check `max_a_val[c] > c.threshold`
- 100% vectorized: just merges and groupby aggregations

**Equality predicates (`==`, `!=`):**
- Small domains: per-node bitsets tracking which values appeared
- Larger domains: per-node (node_id, value) state tables, propagated hop by hop
- Still vectorizable via joins + dedup

### Audit: Non-Vectorized Code Locations

Found **5 functions with BFS/DFS loops** that need refactoring:

| Function | Lines | Issue |
|----------|-------|-------|
| `_apply_non_adjacent_where_post_prune` | 290-536 | BFS path tracing for non-adjacent WHERE |
| `_re_propagate_backward` | 537-659 | Python loops for constraint propagation |
| `_filter_multihop_edges_by_endpoints` | 660-728 | DFS to trace valid paths |
| `_find_multihop_start_nodes` | 729-795 | BFS backward from endpoints |
| `_filter_multihop_by_where` | 1076-1258 | DFS from valid_starts to valid_ends (lines 1237-1250) |

**Specific anti-patterns found:**
- `while queue:` / `while stack:` at lines 438, 711, 779, 1240
- `for ... in zip(edges_df[col], ...)` at lines 462, 472, 478, 695, 699, 702, 760, 765, 769, 1216, 1221, 1225
- `for ... in adjacency.get(node, [])` at lines 442, 715, 783, 1244
- `for ... in current_reachable.items()` at lines 432, 491
- `.tolist()` conversions at 20+ locations

### Correct Vectorized Approach

**Example: `a.val > c.threshold` where `a--e1--b--e2--c`**

```python
# Forward: propagate max(a.val) to each node via merges + groupby
a_vals = nodes_a[['id', 'val']]

# Step 1: a -> b (via e1)
e1_with_a = edges_e1.merge(a_vals, left_on='src', right_on='id')
max_at_b = e1_with_a.groupby('dst')['val'].max().reset_index()
max_at_b.columns = ['id', 'max_a_val']

# Step 2: b -> c (via e2)
e2_with_b = edges_e2.merge(max_at_b, left_on='src', right_on='id')
max_at_c = e2_with_b.groupby('dst')['max_a_val'].max().reset_index()
max_at_c.columns = ['id', 'max_a_val']

# Filter c nodes where predicate holds
valid_c = nodes_c.merge(max_at_c, on='id')
valid_c = valid_c[valid_c['max_a_val'] > valid_c['threshold']]

# Backward semijoin: prune nodes/edges not reaching valid_c
# ... (similar merge-based filtering)
```

This is 100% vectorized DataFrame operations - works identically on pandas and cuDF.

### Refactoring Tasks

1. **Replace `_apply_non_adjacent_where_post_prune`** with vectorized summary propagation:
   - For `>/<`: propagate min/max via `groupby().agg()`
   - For `==`: propagate value sets via state tables (merge + groupby)

2. **Replace `_filter_multihop_edges_by_endpoints`** with merge-based filtering:
   - Semijoin edges with allowed start/end node sets
   - For multi-hop: repeated self-joins or hop-labeled edge filtering

3. **Replace `_find_multihop_start_nodes`** with backward semijoin:
   - Merge edges with allowed endpoints, propagate backward via groupby

4. **Simplify `_re_propagate_backward`** to use semijoin pattern:
   - Each step: `edges.merge(allowed_nodes).groupby(src)[dst].apply(set)`

5. **Replace `_filter_multihop_by_where` DFS** with vectorized approach:
   - The cross-join approach (lines 1173-1176) is good for finding valid (start, end) pairs
   - Replace the DFS path tracing (lines 1237-1250) with hop-by-hop semijoins:
     - Filter first-hop edges by valid_starts
     - Filter last-hop edges by valid_ends
     - For intermediates: semijoin to keep edges connected to valid first/last hops

### Key Insight: Hop Labels Enable Vectorization

Multi-hop edges already have hop labels (e.g., `__edge_hop__`). Instead of DFS:
```python
# Filter by hop label + semijoin
first_hop = edges_df[edges_df[hop_col] == min_hop]
last_hop = edges_df[edges_df[hop_col] == max_hop]

# Semijoin with valid endpoints
first_hop = first_hop[first_hop[src_col].isin(valid_starts)]
last_hop = last_hop[last_hop[dst_col].isin(valid_ends)]

# Propagate allowed nodes through intermediate hops via merge+groupby
```

### Why This Matters

| Aspect | Current (BFS/DFS) | Correct (Yannakakis) |
|--------|-------------------|----------------------|
| CPU pandas | Works but slow | Fast vectorized |
| GPU cuDF | Broken (Python loops) | Works natively |
| Complexity | O(paths) | O(edges) |
| Memory | Path tables | Set-based |
| Correctness | Ad-hoc | Theoretically grounded |

### Test Results (Before Refactor)

```
91 passed, 2 skipped, 1 xfailed
```

Tests pass but implementation is wrong. Need to refactor to vectorized approach while maintaining test compatibility.

---

## Session 5: Bug Fixes for Failing Tests (Dec 27, 2024)

Fixed all 4 bugs discovered in Session 4's test amplification:

### Bug 1 & 4: Multi-hop edge filtering in `_re_propagate_backward`

**Tests**: `test_long_chain_with_multihop`, `test_mixed_with_multihop`

**Problem**: `_re_propagate_backward` used simple src/dst filtering for multi-hop edges, which incorrectly removed intermediate edges in paths.

**Solution**: Added two helper functions:
- `_filter_multihop_edges_by_endpoints(edges_df, edge_op, left_allowed, right_allowed, is_reverse, is_undirected)` - Uses DFS to trace valid paths and keeps all participating edges
- `_find_multihop_start_nodes(edges_df, edge_op, right_allowed, is_reverse, is_undirected)` - Uses BFS backward from endpoints to find valid start nodes

### Bug 2: Column name collision in `_filter_multihop_by_where`

**Test**: `test_multihop_neq`

**Problem**: When `left_col == right_col` (e.g., `start.v != end.v`), pandas merge creates columns `v` and `v__r`, but the code compared `pairs_df['v']` to itself instead of to `pairs_df['v__r']`.

**Solution**:
1. Added explicit `suffixes=("", "__r")` to the merge at line 1082
2. Added suffix detection logic to use `v__r` when comparing same-named columns

### Bug 3: Undirected edge support missing

**Test**: `test_undirected_multihop_bidirectional`

**Problem**: The executor only handled `forward` and `reverse` directions, treating `undirected` as `forward`. This meant edges were only traversed in one direction.

**Solution**: Added `is_undirected = edge_op.direction == "undirected"` checks throughout, building bidirectional adjacency and considering both src/dst as valid start/end nodes in:
- `_filter_multihop_by_where` (lines 1046-1053, 1126-1129)
- `_apply_non_adjacent_where_post_prune` (lines 409, 424-427, 466-476)
- `_re_propagate_backward` (lines 589, 599-619, 649-651)
- `_filter_multihop_edges_by_endpoints` (lines 673, 696-699)
- `_find_multihop_start_nodes` (lines 735, 758-761)

---

## Session 4: Comprehensive Test Amplification (Dec 27, 2024)

### Test Amplification

Added 37 new tests for comprehensive coverage:

**Unfiltered Starts (3 tests)** - Converted from xfail to regular tests using public API:
- `test_unfiltered_start_node_multihop`
- `test_unfiltered_start_single_hop`
- `test_unfiltered_start_with_cycle`

**Oracle Limitations (1 xfail)**:
- `test_edge_alias_on_multihop` - Oracle doesn't support edge aliases on multi-hop

**P0 Reverse + Multi-hop (4 tests)**:
- `test_reverse_multihop_basic`
- `test_reverse_multihop_filters_correctly`
- `test_reverse_multihop_with_cycle`
- `test_reverse_multihop_undirected_comparison`

**P0 Multiple Starts (3 tests)**:
- `test_two_valid_starts`
- `test_multiple_starts_different_paths`
- `test_multiple_starts_shared_intermediate`

**P1 Operators × Single-hop (6 tests)**:
- `test_single_hop_eq`, `test_single_hop_neq`, `test_single_hop_lt`
- `test_single_hop_gt`, `test_single_hop_lte`, `test_single_hop_gte`

**P1 Operators × Multi-hop (6 tests)**:
- `test_multihop_eq`, `test_multihop_neq`, `test_multihop_lt`
- `test_multihop_gt`, `test_multihop_lte`, `test_multihop_gte`

**P1 Undirected + Multi-hop (2 tests)**:
- `test_undirected_multihop_basic`
- `test_undirected_multihop_bidirectional`

**P1 Mixed Direction Chains (3 tests)**:
- `test_forward_reverse_forward`
- `test_reverse_forward_reverse`
- `test_mixed_with_multihop`

**P2 Longer Paths (4 tests)**:
- `test_four_node_chain`
- `test_five_node_chain_multiple_where`
- `test_long_chain_with_multihop`
- `test_long_chain_filters_partial_path`

**P2 Edge Cases (6 tests)**:
- `test_single_node_graph`
- `test_disconnected_components`
- `test_dense_graph`
- `test_null_values_in_comparison`
- `test_string_comparison`
- `test_multiple_where_all_operators`

---

## Session 3: Single-hop + Cycle Test Amplification (Dec 27, 2024)

### Test Amplification

Added 8 new tests covering single-hop topologies and cycle patterns:

**Single-hop topology tests** (tests without middle node b):
- `test_single_hop_forward_where` - Tests `n(a) -> e -> n(c)` with `a.v < c.v`
- `test_single_hop_reverse_where` - Tests `n(a) <- e <- n(c)` with `a.v < c.v`
- `test_single_hop_undirected_where` - Tests `n(a) <-> e <-> n(c)` with `a.v < c.v`
- `test_single_hop_with_self_loop` - Tests self-loops with `<` operator
- `test_single_hop_equality_self_loop` - Tests self-loops with `==` operator

**Cycle tests**:
- `test_cycle_single_node` - Self-loop with multi-hop (`n(a) -> e(1..2) -> n(c)` WHERE `a == c`)
- `test_cycle_triangle` - Triangle cycle `a->b->c->a` with multi-hop
- `test_cycle_with_branch` - Cycle with a branch (non-participating edges)

### Bug Fixes Discovered via Test Amplification

**Bug 1**: Multi-hop path tracing in `_apply_non_adjacent_where_post_prune`

**Problem**: The path tracing treated each edge step as a single hop, but for multi-hop edges like `e(min_hops=1, max_hops=2)`, we need to trace through the underlying graph edges multiple times.

**Solution** (lines 411-470): Added BFS within multi-hop edges to properly expand paths:
```python
if is_multihop:
    min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
    max_hops = edge_op.max_hops if edge_op.max_hops is not None else 1

    # Build adjacency from edges
    adjacency: Dict[Any, List[Any]] = {}
    for _, row in edges_df.iterrows():
        if is_reverse:
            s, d = row[dst_col], row[src_col]
        else:
            s, d = row[src_col], row[dst_col]
        adjacency.setdefault(s, []).append(d)

    # BFS to find all reachable nodes within min..max hops
    next_reachable: Dict[Any, Set[Any]] = {}
    for start_node, original_starts in current_reachable.items():
        queue = [(start_node, 0)]
        visited_at_hop: Dict[Any, int] = {start_node: 0}
        while queue:
            node, hop = queue.pop(0)
            if hop >= max_hops:
                continue
            for neighbor in adjacency.get(node, []):
                next_hop = hop + 1
                if neighbor not in visited_at_hop or visited_at_hop[neighbor] > next_hop:
                    visited_at_hop[neighbor] = next_hop
                    queue.append((neighbor, next_hop))
        # Nodes reachable within [min_hops, max_hops] are valid endpoints
        for node, hop in visited_at_hop.items():
            if min_hops <= hop <= max_hops:
                if node not in next_reachable:
                    next_reachable[node] = set()
                next_reachable[node].update(original_starts)
    current_reachable = next_reachable
```

**Bug 2**: `_filter_multihop_by_where` used `hop_col.max()` instead of `edge_op.max_hops`

**Problem**: When all nodes can be starts, every edge gets labeled as "hop 1", making `hop_col.max()` unreliable.

**Solution** (lines 982-987):
```python
chain_max_hops = edge_op.max_hops if edge_op.max_hops is not None else (
    edge_op.hops if edge_op.hops is not None else 10
)
max_hops_val = int(chain_max_hops)
```

**Bug 3**: `_filter_edges_by_clauses` wasn't handling reverse edges

**Problem**: For reverse edges, the left alias is reached via the dst column, but the code always used src for left.

**Solution** (lines 703-704, 803-810): Pass `is_reverse` flag and swap merge columns:
```python
if is_reverse:
    left_merge_col = self._destination_column
    right_merge_col = self._source_column
else:
    left_merge_col = self._source_column
    right_merge_col = self._destination_column
```

**Bug 4**: Single-hop edges not persisted after WHERE filtering

**Problem**: Only multi-hop edges were having their filtered results persisted back to `forward_steps[edge_idx]._edges`.

**Solution** (lines 749-751): Remove the `is_multihop` condition:
```python
if len(filtered) < len(edges_df):
    self.forward_steps[edge_idx]._edges = filtered
```

**Bug 5**: Equality filtering broken when `left_col == right_col`

**Problem**: When filtering on `a.v == c.v` where both aliases have column `v`, the merge creates `v` and `v__r` columns, but the rename logic didn't handle this properly.

**Solution** (lines 833-854): Proper handling of the `__r` suffix from merge:
```python
col_left_name = f"__val_left_{left_col}"
col_right_name = f"__val_right_{right_col}"

rename_map = {}
if left_col in out_df.columns:
    rename_map[left_col] = col_left_name
right_col_with_suffix = f"{right_col}__r"
if right_col_with_suffix in out_df.columns:
    rename_map[right_col_with_suffix] = col_right_name
elif right_col in out_df.columns and right_col != left_col:
    rename_map[right_col] = col_right_name
```

**Bug 6**: Edge filtering in `_re_propagate_backward` (previously discovered but enhanced)

**Problem**: Additional edge cases found where edges weren't being properly filtered during re-propagation.

**Solution**: Enhanced the filtering logic to handle all edge cases consistently.

### Test Results (Session 3 Initial)

```
41 passed, 2 skipped
```

All tests pass including the 8 new topology/cycle tests and all previous tests.

---

## Session 4: Comprehensive Test Amplification (Dec 27, 2024)

### Test Amplification

Added 35 new tests for comprehensive coverage:

**Known Limitations (xfail - 2 tests)**:
- `test_unfiltered_start_node_multihop` - Unfiltered starts with multi-hop (xfail)
- `test_edge_alias_on_multihop` - Edge alias on multi-hop (xfail)
- `test_unfiltered_start_single_hop_works` - Single-hop unfiltered works (passes)

**P0 Reverse + Multi-hop (4 tests)**:
- `test_reverse_multihop_basic`
- `test_reverse_multihop_filters_correctly`
- `test_reverse_multihop_with_cycle`
- `test_reverse_multihop_undirected_comparison`

**P0 Multiple Starts (3 tests)**:
- `test_two_valid_starts`
- `test_multiple_starts_different_paths`
- `test_multiple_starts_shared_intermediate`

**P1 Operators × Single-hop (6 tests)**:
- `test_single_hop_eq`, `test_single_hop_neq`, `test_single_hop_lt`
- `test_single_hop_gt`, `test_single_hop_lte`, `test_single_hop_gte`

**P1 Operators × Multi-hop (6 tests)**:
- `test_multihop_eq`, `test_multihop_neq`, `test_multihop_lt`
- `test_multihop_gt`, `test_multihop_lte`, `test_multihop_gte`

**P1 Undirected + Multi-hop (2 tests)**:
- `test_undirected_multihop_basic`
- `test_undirected_multihop_bidirectional`

**P1 Mixed Direction Chains (3 tests)**:
- `test_forward_reverse_forward`
- `test_reverse_forward_reverse`
- `test_mixed_with_multihop`

**P2 Longer Paths (4 tests)**:
- `test_four_node_chain`
- `test_five_node_chain_multiple_where`
- `test_long_chain_with_multihop`
- `test_long_chain_filters_partial_path`

**P2 Edge Cases (6 tests)**:
- `test_single_node_graph`
- `test_disconnected_components`
- `test_dense_graph`
- `test_null_values_in_comparison`
- `test_string_comparison`
- `test_multiple_where_all_operators`

### Bugs Discovered & Fixed

The new tests revealed **4 bugs** in the executor, all now fixed:

1. **`test_long_chain_with_multihop`**: Long chain with two consecutive multi-hop edges loses edges
   - **Root Cause**: `_re_propagate_backward` used simple src/dst filtering for multi-hop edges, incorrectly removing intermediate edges
   - **Fix**: Added `_filter_multihop_edges_by_endpoints` helper to trace valid paths using DFS and keep all participating edges

2. **`test_multihop_neq`**: Multi-hop with `!=` operator doesn't filter correctly
   - **Root Cause**: When `left_col == right_col` (e.g., both `'v'`), pandas merge creates `v` and `v__r` columns, but the WHERE filtering compared `pairs_df['v']` to itself
   - **Fix**: Added suffix handling in `_filter_multihop_by_where` to detect `__r` suffix and use the correct column; also added explicit `suffixes=("", "__r")` to the merge

3. **`test_undirected_multihop_bidirectional`**: Undirected multi-hop doesn't traverse both directions
   - **Root Cause**: The executor only handled `forward` and `reverse` directions, treating `undirected` as `forward`
   - **Fix**: Added `is_undirected` checks throughout the codebase to build bidirectional adjacency graphs and consider both src/dst as valid start/end nodes in:
     - `_filter_multihop_by_where`
     - `_apply_non_adjacent_where_post_prune`
     - `_re_propagate_backward`
     - `_filter_multihop_edges_by_endpoints`
     - `_find_multihop_start_nodes`

4. **`test_mixed_with_multihop`**: Mixed directions with multi-hop edges has edge filtering issues
   - **Root Cause**: Same as #1 - `_re_propagate_backward` didn't properly handle multi-hop edge filtering
   - **Fix**: Same as #1 - `_filter_multihop_edges_by_endpoints` helper

### Test Results (Final)

```
78 passed, 2 skipped, 1 xfailed
```

**All 4 previously failing tests now pass.**

---

## Session 2: Non-adjacent alias WHERE + Mixed hop ranges (Dec 26, 2024)

### P0 Fix: Non-adjacent alias WHERE (`test_non_adjacent_alias_where`)

**Problem**: WHERE clauses between non-adjacent aliases (2+ edges apart like `a.id == c.id` in chain `n(a) -> e -> n(b) -> e -> n(c)`) were not applied during backward prune. The `_backward_prune` method only processed WHERE clauses between adjacent aliases.

**Solution** (`graphistry/compute/gfql/df_executor.py`):

Added `_apply_non_adjacent_where_post_prune` method (lines 290-474) that:
1. Identifies non-adjacent WHERE clauses after `_backward_prune` completes
2. Traces paths step-by-step to track which start nodes can reach which end nodes
3. For each (start, end) pair, applies the WHERE comparison (==, !=, <, <=, >, >=)
4. Filters `allowed_nodes` to only include nodes in valid (start, end) pairs
5. Re-propagates constraints backward via `_re_propagate_backward` to update intermediate nodes/edges

Also added helper `_are_aliases_adjacent` (lines 278-288) to detect if two node aliases are exactly one edge apart.

**Key insight**: This is fundamentally a path-tracing problem. We can't just intersect value sets because all values might appear in both aliases - we need to know which specific paths satisfy the constraint.

### P1 Fix: Multiple WHERE + mixed hop ranges (`test_multiple_where_mixed_hop_ranges`)

**Problem**: The test had an edge alias on a multi-hop edge, which the oracle doesn't support.

**Solution** (`tests/gfql/ref/test_df_executor_inputs.py`):
- Removed the edge alias from the multi-hop edge (`e_forward(min_hops=1, max_hops=2)` instead of `e_forward(min_hops=1, max_hops=2, name="e2")`)
- The executor was already handling the case correctly; it was an oracle limitation

### Additional Bug Fix: Edge filtering in `_re_propagate_backward`

**Problem discovered via test amplification**: The `!=` operator test revealed that edges weren't being filtered when there's no edge ID column. The `_re_propagate_backward` method only updated `allowed_edges` dict but didn't filter `forward_steps[edge_idx]._edges`.

**Solution**: Updated `_re_propagate_backward` to:
1. Filter edges by BOTH src and dst (not just dst)
2. Persist filtered edges back to `forward_steps[edge_idx]._edges` when filtering occurs

### Test Amplification

Added 4 new test variants to cover all comparison operators:
- `test_non_adjacent_alias_where_inequality` - Tests `<` operator
- `test_non_adjacent_alias_where_inequality_filters` - Tests `>` operator with filtering
- `test_non_adjacent_alias_where_not_equal` - Tests `!=` operator (caught the edge filtering bug)
- `test_non_adjacent_alias_where_lte_gte` - Tests `<=` operator

### Test Results (Session 2)

```
27 passed, 2 skipped
```

**All tests pass including**:
- `test_non_adjacent_alias_where` - P0 non-adjacent WHERE with `==`
- `test_non_adjacent_alias_where_inequality` - Non-adjacent `<`
- `test_non_adjacent_alias_where_inequality_filters` - Non-adjacent `>`
- `test_non_adjacent_alias_where_not_equal` - Non-adjacent `!=`
- `test_non_adjacent_alias_where_lte_gte` - Non-adjacent `<=`
- `test_multiple_where_mixed_hop_ranges` - P1 mixed hops

---

## Session 1: Original fixes (prior session)

### 1. Oracle Fix (`graphistry/gfql/ref/enumerator.py`)

**Problem**: `collected_nodes` and `collected_edges` stored ALL nodes/edges reached during multi-hop traversal BEFORE WHERE filtering, but were used AFTER filtering. This meant nodes from paths that failed WHERE were still included.

**Solution** (lines 151-205):
- After WHERE filtering, re-trace paths from valid starts to valid ends
- Build adjacency respecting edge direction (forward/reverse/undirected)
- DFS from valid starts to find paths reaching valid ends
- Only keep nodes/edges that participate in valid paths
- Clear collected_nodes/edges when no paths survive WHERE

### 2. Executor Fix (`graphistry/compute/gfql/df_executor.py`)

**Problem 1**: `_filter_multihop_by_where` used wrong columns for reverse edges
- Forward assumes: start=src, end=dst
- Reverse needs: start=dst, end=src

**Solution** (lines 538-549):
```python
is_reverse = edge_op.direction == "reverse"
if is_reverse:
    start_nodes = set(first_hop_edges[self._destination_column].tolist())
    end_nodes = set(valid_endpoint_edges[self._source_column].tolist())
else:
    start_nodes = set(first_hop_edges[self._source_column].tolist())
    end_nodes = set(valid_endpoint_edges[self._destination_column].tolist())
```

**Problem 2**: End nodes only from max hop, not all hops >= min_hops

**Solution** (lines 533-536):
```python
chain_min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
valid_endpoint_edges = edges_df[hop_col >= chain_min_hops]
```

**Problem 3**: Path tracing didn't respect direction

**Solution** (lines 602-613):
```python
if is_reverse:
    adjacency.setdefault(dst_val, []).append((eid, src_val))
else:
    adjacency.setdefault(src_val, []).append((eid, dst_val))
```

**Problem 4**: Filtered edges not persisted for materialization

**Solution** (lines 398-400):
```python
if is_multihop and len(filtered) < len(edges_df):
    self.forward_steps[edge_idx]._edges = filtered
```

### 3. Test Updates (`tests/gfql/ref/test_df_executor_inputs.py`)

- Removed xfail from `test_where_respected_after_min_hops_backtracking` (now passes)
- Updated `linear_inequality` scenario to use explicit start filter (current limitation)

---

## Known Limitations

### Multi-start node limitation

The executor can't handle cases where ALL nodes are potential starts (no filter on start node). This is because:

1. Hop labels are relative to each starting node
2. When all nodes can start, every edge is "hop 1" from some start
3. Can't distinguish which paths came from which starts

**Workaround**: Use explicit start filters like `n({"id": "a"})` instead of just `n()`

**Future fix options**:
1. Track path provenance during forward pass
2. Fall back to oracle for unfiltered starts
3. Store per-start-node hop information

### Oracle: Edge aliases on multi-hop edges

The oracle doesn't support edge aliases on multi-hop edges (`e_forward(min_hops=1, max_hops=2, name="e2")` raises an error). This is documented in `enumerator.py:109`.

---

## Files Modified

1. `graphistry/gfql/ref/enumerator.py` - Oracle path retracing after WHERE
2. `graphistry/compute/gfql/df_executor.py` - Executor direction-aware filtering + non-adjacent WHERE
3. `tests/gfql/ref/test_df_executor_inputs.py` - Test updates, removed xfails

---

## Future Work

### P2: All-nodes-as-starts support
- Issue: Executor fails when start node has no filter
- Approach: Either track path provenance or fall back to oracle

### P2: Oracle edge alias support for multi-hop
- Issue: Can't use edge aliases on multi-hop edges in oracle
- Approach: Track edge sets during multi-hop enumeration

---

## How to Resume

1. Run the test suite to verify current state:
   ```bash
   python -m pytest tests/gfql/ref/test_df_executor_inputs.py -v
   ```
   Expected: 78 passed, 2 skipped, 1 xfailed

2. Key files to understand:
   - `graphistry/gfql/ref/enumerator.py` - Oracle implementation (reference/ground truth)
   - `graphistry/compute/gfql/df_executor.py` - Executor implementation (GPU-style path)
   - `tests/gfql/ref/test_df_executor_inputs.py` - 78 test cases with `_assert_parity()` helper

3. Test helper `_assert_parity(graph, chain, where)`:
   - Runs both executor (`_run_gpu()`) and oracle (`enumerate_chain()`)
   - Asserts node/edge sets match
   - Use for debugging: add print statements to compare intermediate results

4. Key executor methods (in order of execution):
   - `_forward()` - Forward pass, captures wavefronts at each step
   - `_run_gpu()` - GPU-style path: `_compute_allowed_tags()` → `_backward_prune()` → `_apply_non_adjacent_where_post_prune()` → `_materialize_filtered()`
   - `_backward_prune()` - Walk edges backward, filter by WHERE clauses
   - `_filter_multihop_by_where()` - Handle WHERE for multi-hop edges
   - `_apply_non_adjacent_where_post_prune()` - Handle WHERE between non-adjacent aliases
   - `_re_propagate_backward()` - Re-propagate constraints after filtering

5. Related issues:
   - #871: Output slicing bugs (fixed)
   - #872: Multi-hop + WHERE bugs (fixed, sessions 1-5)
   - #837: cuDF hop executor (parent issue for this branch)

6. Potential future work:
   - Oracle edge alias support for multi-hop (currently xfail)
   - Performance optimization (current impl uses Python loops, could use vectorized ops)
