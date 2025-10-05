# GFQL Dict-to-AST Conversion Bug Fix Plan

## 🐛 CRITICAL BUG REPORT: GFQL Dict-to-AST Conversion Regression

### Issue Summary
The policy hooks branch breaks existing GFQL functionality by failing to convert dictionary operations to AST objects in list contexts, causing `TypeError: 'dict' object is not callable` errors.

### Root Cause Analysis
- **Working**: Single dict → `ASTLet` conversion ✅
- **Broken**: List of dicts → direct pass to `chain_impl()` without conversion ❌
- **Location**: `graphistry/compute/gfql_unified.py:~247` in `elif isinstance(query, list):` branch

### Bug Details
**What breaks**:
```python
# This works (single dict gets converted to ASTLet)
g.gfql({"type": "Node", "filter_dict": {"name": "A"}})

# This fails (list of dicts not converted to AST objects)
g.gfql([{"type": "Node", "filter_dict": {"name": "A"}}])
```

**Error**: `TypeError: 'dict' object is not callable`

**Fix needed**: Convert dictionaries in lists to AST objects using `from_json()` before passing to `chain_impl()`

## 🔧 IMPLEMENTATION PLAN

### Step 1: Verify Current State
- [ ] Check if fix is already implemented (system reminder suggests it may be)
- [ ] Create reproduction test case
- [ ] Confirm the exact error and location

### Step 2: Implement/Verify Fix
- [ ] Ensure list branch converts dicts to AST objects
- [ ] Use `from_json()` for proper dict-to-AST conversion
- [ ] Handle mixed lists (AST objects + dicts)

### Step 3: Comprehensive Testing
- [ ] Test single dict scenarios (should still work)
- [ ] Test list of dicts scenarios (main fix target)
- [ ] Test mixed list scenarios (AST + dict)
- [ ] Test edge cases (empty lists, nested structures)
- [ ] Run full GFQL test suite

### Step 4: Quality Assurance
- [ ] Run pytest on GFQL modules
- [ ] Run mypy type checking
- [ ] Verify no regressions in existing functionality

### Step 5: Documentation & Release
- [ ] Update any relevant test cases
- [ ] Commit fix with proper conventional commit message
- [ ] Update PR if needed

## 🎯 SUCCESS CRITERIA

1. **Backward Compatibility**: All previously working GFQL patterns continue to work
2. **Dict Conversion**: Lists containing dictionaries are properly converted to AST objects
3. **Test Coverage**: Comprehensive tests prevent future regressions
4. **No Side Effects**: Policy hooks and UMAP functionality remain intact

## 📋 TEST CASES TO VERIFY

```python
import pandas as pd
import graphistry

g = graphistry.nodes(pd.DataFrame({'name': ['A', 'B']}))

# Case 1: Single dict (should work)
result1 = g.gfql({"type": "Node", "filter_dict": {"name": "A"}})

# Case 2: List of dicts (main fix target)
result2 = g.gfql([{"type": "Node", "filter_dict": {"name": "A"}}])

# Case 3: Mixed list (AST + dict)
from graphistry.compute.ast import n
result3 = g.gfql([n({"name": "A"}), {"type": "Node", "filter_dict": {"name": "B"}}])

# Case 4: Multiple dicts in list
result4 = g.gfql([
    {"type": "Node", "filter_dict": {"name": "A"}},
    {"type": "Node", "filter_dict": {"name": "B"}}
])
```

**Priority**: CRITICAL - This is a backward compatibility regression that breaks existing GFQL usage patterns and must be fixed before merge.