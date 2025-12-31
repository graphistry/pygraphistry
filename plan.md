# PR #846 Test Amplification Status

## Context
After Alloy verification PR (#852) proved less useful than Python parity tests for catching real bugs, we're amplifying test coverage on PR #846 (executor branch) based on 5-whys analysis of bugs found during development.

## Current Branch
`feat/issue-837-cudf-hop-executor` (PR #846)

## Status: Test Amplification Ongoing

Test amplification found and fixed 4 bugs (Bug 6, Bug 7, Oracle Bug, Bug 8). Added comprehensive test classes for Yannakakis principle, hop labeling patterns, and sensitive phenomena.

## Completed Work

### 1. 5-Whys Analysis & Test Amplification (11 tests)
- Added `TestFiveWhysAmplification` class
- **Commit**: `f7b3faa5`

### 2. Bug 6 Fix: Multi-hop Edge Filtering
- **Commit**: `f7b3faa5`

### 3. Predicate Type Testing (9 tests)
- **Commit**: `a4d39651`

### 4. Min-Hops Edge Filtering Tests (8 tests) + Bug 7 Fix
- Added `TestMinHopsEdgeFiltering` class
- Found and fixed Bug 7
- **Commit**: `48564039`

### 5. Multiple Path Lengths Tests (7 tests) + Oracle Bug Fix
- Added `TestMultiplePathLengths` class (depth-wise 5-whys on Bug 7)
- Found and fixed Oracle Bug: enumerator.py included paths shorter than min_hops
- Tests: diamond with shortcut, triple paths, cycle paths, parallel paths with min_hops, undirected/reverse routes
- **Commit**: `8b1c8539`

### 6. Yannakakis Principle Tests (6 tests)
- Added `TestYannakakisPrinciple` class
- Tests: dead-end branch pruning, all valid paths included, spurious edge exclusion, WHERE prunes intermediate edges, convergent diamond, mixed valid/invalid branches
- **Commit**: `b3d90a28`

### 7. Hop Labeling Pattern Tests (5 tests)
- Added `TestHopLabelingPatterns` class
- Tests: hop labels don't affect validity, multiple seeds, min_hops labeling, edge hop labels consistent, undirected hop labels
- **Commit**: `b3d90a28`

### 8. Dual-Engine Testing (pandas + cudf)
- Modified `_assert_parity` to automatically test with cudf when available
- No code duplication - same tests run on both engines
- Can skip cudf with `GFQL_SKIP_CUDF=1` env var if needed
- **Commit**: `d3e5712f`

### 9. Sensitive Phenomena Tests (14 tests) + Bug 8 Fix
- Added `TestSensitivePhenomena` class based on deep 5-whys analysis
- Found and fixed Bug 8: `_filter_edges_by_clauses` didn't handle undirected edges
- Tests cover: asymmetric reachability, filter cascades, non-adjacent WHERE, path length boundaries, shared edge semantics, self-loops, cycles
- **Commit**: `e8780035`

### 10. Oracle Node/Edge Match Filter Support + Tests (7 tests)
- Fixed oracle enumerator to support `source_node_match`, `destination_node_match`, `edge_match` filters
- Added `TestNodeEdgeMatchFilters` class with 7 tests
- Tests cover: single-hop filters, multi-hop filters, combined filters, edge match, undirected with filters
- **Commit**: (pending)

## Test Results (All Passing)
- `TestFiveWhysAmplification`: 11/11
- `TestPredicateTypes`: 9/9
- `TestMinHopsEdgeFiltering`: 8/8
- `TestMultiplePathLengths`: 7/7
- `TestYannakakisPrinciple`: 6/6
- `TestHopLabelingPatterns`: 5/5
- `TestSensitivePhenomena`: 14/14
- `TestNodeEdgeMatchFilters`: 7/7
- Full `test_df_executor_inputs.py`: 168 passed
- `test_compute_hops.py`: 58 passed

## 5-Whys Summary

| Bug | Root Cause | Status |
|-----|------------|--------|
| 1 | Backward traversal join direction | Fixed |
| 2 | Empty set short-circuit missing | Fixed |
| 3 | Wrong node source for non-adjacent WHERE | Fixed |
| 4 | Diamond/convergent path confusion | Fixed |
| 5 | Undirected treated as forward-only | Fixed |
| 6 | min_hops applied per-edge not per-path | Fixed |
| 7 | min_hops pruning uses lossy min-hop-per-node | Fixed |
| Oracle | enumerator included paths < min_hops | Fixed |
| 8 | `_filter_edges_by_clauses` ignored undirected | Fixed |

## Future Test Amplification Ideas

### Depth-wise 5-Whys on Bug 7

Bug 7's deeper root cause reveals a pattern worth testing:

1. **Why**: goal_nodes missed nodes reachable via longer paths
2. **Why**: Used `node_hop_records` which only tracks min hop
3. **Why**: The anti-join pattern (`_merge == 'left_only'`) discards duplicates
4. **Why**: BFS-style traversal optimizes for "first seen" not "all paths"
5. **Why**: **No test existed for "same node reachable at multiple distances"**

### Implemented: `TestMultiplePathLengths` (7 tests)

Tests for scenarios where same node is reachable at different hop distances:

1. ✓ **Diamond with shortcut** - node reachable at hop 1 AND hop 2
2. ✓ **Triple paths different lengths** - 3 paths of lengths 1, 2, 3
3. ✓ **Triple paths exact min_hops=3** - only include longest path
4. ✓ **Cycle creating multiple path lengths** - a->b->c->a allows reaching 'a' at hop 0 and hop 3
5. ✓ **Parallel paths with min_hops filter** - found Oracle bug!
6. ✓ **Undirected multiple routes** - same node via different paths
7. ✓ **Reverse multiple path lengths** - reverse traversal with shortcuts

### Other Lossy Aggregation Patterns to Audit

The `groupby().min()` and anti-join "first seen" patterns appear in:

| Location | Pattern | Risk |
|----------|---------|------|
| `df_executor.py:791-792` | `groupby().min()` | Safe (documented with Yannakakis comment) |
| `hop.py:766-782` | edge-based goal_nodes | Fixed |
| `hop.py:682` | node anti-join tracking | Display only - safe? |
| `hop.py:661` | edge anti-join tracking | Display only - safe? |

The anti-join patterns in hop.py are used for hop labeling (display), not filtering. But worth a test to confirm they don't affect path validity.

### Implemented: `TestYannakakisPrinciple` (6 tests)

Tests that specifically validate the Yannakakis semijoin property:
- "Edge included iff it participates in at least one valid complete path"
- "No edge excluded that could be part of a valid path"
- "No spurious edges included that aren't on any valid path"

1. ✓ **Dead-end branch pruning** - edges leading to nodes that fail WHERE should be excluded
2. ✓ **All valid paths included** - multiple valid paths, all edges on any valid path included
3. ✓ **Spurious edge exclusion** - edges not on any complete path are excluded
4. ✓ **WHERE prunes intermediate edges** - aggressive WHERE removes edges mid-chain
5. ✓ **Convergent diamond all paths included** - diamond where both paths are valid
6. ✓ **Mixed valid/invalid branches** - only valid branch edges included

### Implemented: `TestHopLabelingPatterns` (5 tests)

Tests for the anti-join patterns used in hop labeling (display only):
1. ✓ **Hop labels don't affect path validity** - nodes with same label from different paths
2. ✓ **Multiple seeds hop labels** - overlapping reachable nodes from multiple seeds
3. ✓ **Hop labels with min_hops** - intermediate nodes still included
4. ✓ **Edge hop labels consistent** - edges labeled with correct hop distance
5. ✓ **Undirected hop labels** - nodes reachable in both directions

### Implemented: `TestSensitivePhenomena` (14 tests) + Bug 8 Fix

Deep 5-whys analysis across all bugs revealed sensitive edge cases. Tests cover:

**Asymmetric Reachability (3 tests):**
1. ✓ **Forward-only node** - node reachable forward but not reverse
2. ✓ **Reverse-only node** - node reachable reverse but not forward
3. ✓ **Undirected finds reverse-only node** - Found Bug 8! Undirected traversal should find "backward" edges

**Filter Cascades (2 tests):**
4. ✓ **Filter eliminates all at step** - node filter returns empty set at intermediate step
5. ✓ **WHERE eliminates all paths** - all paths fail WHERE clause

**Non-Adjacent WHERE (2 tests):**
6. ✓ **Three-step start-to-end comparison** - WHERE on non-adjacent steps
7. ✓ **Multiple non-adjacent constraints** - multiple WHERE clauses on different pairs

**Path Length Boundaries (2 tests):**
8. ✓ **min_hops=0 includes seed** - seed node in output with min_hops=0
9. ✓ **max_hops exceeds graph diameter** - max_hops > actual path length

**Shared Edge Semantics (2 tests):**
10. ✓ **Edge used by multiple destinations** - same edge reaches different valid ends
11. ✓ **Diamond shared edges** - edges shared by multiple valid paths

**Self-Loops and Cycles (3 tests):**
12. ✓ **Self-loop edge** - edge from node to itself
13. ✓ **Small cycle with min_hops** - 3-node cycle with min_hops constraint
14. ✓ **Cycle with branch** - cycle where branch doesn't affect result

**Bug 8 Root Cause:**
`_filter_edges_by_clauses` only tried forward orientation (src=left, dst=right) for undirected edges. For edges where traversal goes "backwards" (e.g., b->a edge traversed from a), the merge failed to find matches. Fixed by trying both orientations and combining results.

## Recent Commits
- `f7b3faa5` - Bug 6 fix + 5-whys tests (11 tests)
- `a4d39651` - Predicate type tests (9 tests)
- `48564039` - Bug 7 fix + min_hops tests (8 tests)
- `8b1c8539` - TestMultiplePathLengths + oracle min_hops fix (7 tests)
- `b3d90a28` - Yannakakis principle + hop labeling tests (11 tests)
- `d3e5712f` - Dual-engine testing (pandas + cudf)

## Related PRs/Issues
- PR #846: cudf same-path executor (this branch)
- PR #852: Alloy verification (stacked, marked experimental)
- Issue #871: Testing & verification roadmap
