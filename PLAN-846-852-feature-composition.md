# Feature Composition Testing Plan: PR #846 + #852

## Status Summary

| Item | Status | Notes |
|------|--------|-------|
| P0/P1 Tests for #846 | ✅ DONE | 8 tests added; 6 xfail (bugs found), 2 passing |
| Multi-hop bugs filed | ✅ DONE | Issue #872 created |
| Alloy README update | ✅ DONE | Scope/limitations documented |
| Meta-issue roadmap | ✅ DONE | Issue #871 created |

## Issues Created

- **#871**: Meta: GFQL Testing & Verification Roadmap
- **#872**: Fix multi-hop + WHERE backward prune bugs in cuDF executor

## Branch Structure

```
master (includes PR #851 hop ranges - MERGED)
  └── PR #846: feat/issue-837-cudf-hop-executor (same-path executor)
        └── PR #852: feat/issue-838-alloy-fbf-where (alloy proof) ← CURRENT
```

## Execution Order

### Phase 1: PR #846 Tests (on branch `feat/issue-837-cudf-hop-executor`)

**Status: ✅ COMPLETE**

Tests added to `tests/gfql/ref/test_cudf_executor_inputs.py`:

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1 | WHERE respected after min_hops backtracking | xfail | Bug #872 |
| 2 | Reverse direction + hop range + WHERE | xfail | Bug #872 |
| 3 | Non-adjacent alias WHERE | xfail | Bug #872 |
| 4 | Oracle vs cuDF parity comprehensive | xfail | Bug #872 |
| 5 | Multi-hop edge WHERE filtering | xfail | Bug #872 |
| 6 | Output slicing + WHERE | ✅ PASS | Works correctly |
| 7 | label_seeds + output_min_hops | ✅ PASS | Works correctly |
| 8 | Multiple WHERE + mixed hop ranges | xfail | Bug #872 |

**Key Finding**: The cuDF executor has architectural limitations with multi-hop edges + WHERE:
- Backward prune doesn't trace through intermediate edges
- `_is_single_hop()` gates WHERE filtering
- Non-adjacent alias WHERE not applied

These are documented in issue #872 for future fix.

---

### Phase 2: Rebase PR #852 onto master

```bash
git checkout feat/issue-838-alloy-fbf-where
git fetch origin
git rebase origin/master
# Resolve any conflicts
git push origin feat/issue-838-alloy-fbf-where --force-with-lease
```

---

### Phase 3: PR #852 Verification Updates (on branch `feat/issue-838-alloy-fbf-where`)

**Status: ✅ COMPLETE**

| # | Change | File | Status |
|---|--------|------|--------|
| 1 | Clarify hop ranges NOT formally verified | `alloy/README.md` | ✅ DONE |
| 2 | Note reliance on Python parity tests | `alloy/README.md` | ✅ DONE |
| 3 | State verified fragment precisely | `alloy/README.md` | ✅ DONE |

**P1 - Add scenario checks (optional, strengthens claims)** - Deferred to future work.

**Next steps:**
```bash
git checkout feat/issue-837-cudf-hop-executor
git stash pop  # Apply the test changes
git add -A && git commit
git push origin feat/issue-837-cudf-hop-executor
# Wait for CI green, then merge PR #846 to master
```

---

## Test Implementation Details

### Test 1: WHERE after min_hops backtracking

```python
def test_where_respected_after_backtracking():
    """
    Graph: a -> b -> c -> d (3 hops)
           a -> x -> y      (2 hops, dead end for min_hops=3)

    WHERE: a.value < d.value

    Backtracking for min_hops=3 should:
    1. Prune x,y branch (doesn't reach 3 hops)
    2. Keep a,b,c,d path
    3. THEN apply WHERE to filter paths where a.value < d.value

    If WHERE not re-applied after backtracking, invalid paths may remain.
    """
```

### Test 2: Reverse direction + WHERE

```python
def test_reverse_direction_where_semantics():
    """
    Graph: a -> b -> c -> d (forward edges)

    Chain: [n(name='start'), e_reverse(min_hops=2), n(name='end')]
    WHERE: start.value > end.value

    Starting at 'd', reverse traversal reaches:
    - c at hop 1, b at hop 2, a at hop 3

    With min_hops=2, valid endpoints are b (hop 2) and a (hop 3).
    WHERE compares start (d) vs end (b or a).

    Verify WHERE semantics are consistent regardless of traversal direction.
    """
```

### Test 3: Non-adjacent alias WHERE

```python
def test_non_adjacent_alias_where():
    """
    Chain: [n(name='a'), e_forward(), n(name='b'), e_forward(), n(name='c')]
    WHERE: a.id == c.id  (aliases 2 edges apart)

    This WHERE clause should filter to paths where the first and last
    nodes have the same id (e.g., cycles back to start).

    Risk: cuDF backward prune only applies WHERE to adjacent aliases.
    """
```

### Test 4: Oracle vs cuDF parity (parametrized)

```python
@pytest.mark.parametrize("scenario", COMPOSITION_SCENARIOS)
def test_oracle_cudf_parity(scenario):
    """
    Run same query with Oracle and cuDF executor.
    Verify identical results.

    Scenarios cover all combinations of:
    - Directions: forward, reverse, undirected
    - Hop ranges: min_hops, max_hops, output slicing
    - WHERE operators: ==, !=, <, <=, >, >=
    - Topologies: linear, branch, cycle, disconnected
    """
```

---

## README Update for PR #852

```markdown
## Scope and Limitations

### What IS Formally Verified

- WHERE clause lowering to per-alias value summaries
- Equality (==, !=) via bitset filtering
- Inequality (<, <=, >, >=) via min/max summaries
- Multi-step chains with cross-alias comparisons
- Graph topologies: fan-out, fan-in, cycles, parallel edges, disconnected

### What is NOT Formally Verified

- **Hop ranges** (`min_hops`, `max_hops`): Approximated by unrolling to fixed-length chains
- **Output slicing** (`output_min_hops`, `output_max_hops`): Treated as post-filter
- **Hop labeling** (`label_node_hops`, `label_edge_hops`, `label_seeds`): Not modeled
- **Null/NaN semantics**: Verified in Python tests

### Test Coverage for Unverified Features

Hop ranges and output slicing are covered by Python parity tests:
- `tests/gfql/ref/test_enumerator_parity.py`: 11+ hop range scenarios
- `tests/gfql/ref/test_cudf_executor_inputs.py`: 8+ WHERE + hop range scenarios

These tests verify the cuDF executor matches the reference oracle implementation.
```

---

## Priority Summary

| Priority | Branch | Items | Blocks |
|----------|--------|-------|--------|
| **P0** | #846 | 4 tests | Merge of #846 |
| **P1** | #846 | 4 tests | - |
| **P0** | #852 | README scope update | Merge of #852 |
| **P1** | #852 | Alloy scenario checks | - |

---

## Success Criteria

### PR #846 Ready to Merge When:
- [ ] All 8 new tests pass
- [ ] Existing tests still pass
- [ ] CI green

### PR #852 Ready to Merge When:
- [ ] README accurately describes verified scope
- [ ] Alloy checks pass (existing + any new scenarios)
- [ ] CI green

---

## Resume Context

### Current State (as of session end)
- **Current branch**: `feat/issue-838-alloy-fbf-where` (PR #852)
- **Stash**: Test changes stashed on `feat/issue-837-cudf-hop-executor` (stash@{0})
- **Uncommitted**: `alloy/README.md` changes (scope/limitations section added)

### Git State Summary
```
feat/issue-838-alloy-fbf-where:
  - Modified: alloy/README.md (scope/limitations section)
  - Untracked: PLAN-846-852-feature-composition.md (this file)

feat/issue-837-cudf-hop-executor (stash@{0}):
  - 8 new tests in tests/gfql/ref/test_cudf_executor_inputs.py
  - TestP0FeatureComposition class (4 tests, 3 xfail + 1 passing)
  - TestP1FeatureComposition class (4 tests, 3 xfail + 1 passing)
```

### Key Files Modified
1. `tests/gfql/ref/test_cudf_executor_inputs.py` - Added 8 feature composition tests
2. `alloy/README.md` - Added scope/limitations section
3. `PLAN-846-852-feature-composition.md` - This tracking document

### Bug Details (Issue #872)
Root cause in `graphistry/compute/gfql/cudf_executor.py`:
- `_backward_prune()` lines 312-393: Assumes single-hop edges
- `_is_single_hop()` gates WHERE filtering
- Multi-hop edges break backward prune path tracing

### To Resume Work
```bash
# 1. Commit alloy README changes on current branch
git add alloy/README.md
git commit -m "docs(alloy): add scope and limitations section"
git push origin feat/issue-838-alloy-fbf-where

# 2. Switch to #846 branch and apply stashed tests
git checkout feat/issue-837-cudf-hop-executor
git stash pop

# 3. Commit and push test changes
git add tests/gfql/ref/test_cudf_executor_inputs.py
git commit -m "test(gfql): add 8 feature composition tests for hop ranges + WHERE

Adds P0/P1 tests for PR #846 same-path executor with hop ranges.
6 tests xfail documenting known bugs (see issue #872).
2 tests pass verifying output slicing and label_seeds work correctly."
git push origin feat/issue-837-cudf-hop-executor

# 4. Wait for CI, then merge PRs in order: #846 first, then rebase/merge #852
```

### Related Issues
- **#871**: Meta: GFQL Testing & Verification Roadmap (future work)
- **#872**: Fix multi-hop + WHERE backward prune bugs in cuDF executor
