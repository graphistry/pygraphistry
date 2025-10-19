# Critical Implementation Requirements - Path Variable Support

**Discovered via adversarial testing**
**Date**: 2025-10-19

---

## Non-Negotiable Requirements

These requirements emerged from tests designed to BREAK naive implementations.
Each requirement addresses a specific failure mode.

### 1. Edge-Based Path IDs (Not Node Tuples)

**Test**: `test_multi_edges()`

**Failure Mode**: Using `(src, dst)` as path identifier

**Problem**: Graphs can have parallel edges between same nodes:
```
A -[e1:type_a]-> B
A -[e2:type_b]-> B
```

**Wrong**: Treating as single path `(A, B)`
**Right**: Two distinct paths `(e1)` and `(e2)`

**Implementation**:
```python
# ✗ WRONG
path_id = (edge['src'], edge['dst'])

# ✓ RIGHT
path_id = edge['edge_id']  # Or tuple of edge IDs for multi-hop
```

**Impact**: Multi-edge graphs (common in real data) would produce INCORRECT results.

---

### 2. Namespaced Path Tracking Columns

**Test**: `test_column_collision()`

**Failure Mode**: User's graph already has column named like tracking columns

**Problem**: User nodes have column 'n1_x' or even '__gfql_path_id__'

**Wrong**: Overwrite or fail
**Right**: Use guaranteed-unique namespace

**Implementation**:
```python
# ✗ WRONG - Collides with user columns
wavefront['n1_x'] = ...
wavefront['__gfql_path_id__'] = ...

# ✓ RIGHT - Reserved namespace
wavefront['__gfql_var_n1_x__'] = ...
wavefront['__gfql_path_id__'] = ...

# Or even safer:
prefix = f'__gfql_{uuid.uuid4().hex[:8]}_'
wavefront[f'{prefix}n1_x'] = ...
```

**Impact**: Production graphs with adversarial column names would CRASH or CORRUPT data.

---

### 3. Path ID Stability Across Passes

**Test**: `test_path_id_collision_after_pruning()`

**Failure Mode**: Reassigning path IDs in final forward pass

**Problem**: GFQL's 3-pass execution:
```
Forward pass 1: Create paths 0, 1, 2, 3, 4
Backward pass:  Prune paths 1, 3 (dead ends)
Forward pass 2: Must use IDs 0, 2, 4 - NOT reassign 0, 1, 2
```

**Wrong**: Reassign sequential IDs in each pass
**Right**: Preserve IDs from first forward pass

**Implementation**:
```python
# ✗ WRONG - Reassigns IDs
final_paths = pruned_paths.copy()
final_paths['__gfql_path_id__'] = range(len(final_paths))

# ✓ RIGHT - Preserves IDs
final_paths = pruned_paths.copy()  # IDs already set from forward pass 1
```

**Impact**: Cross-variable predicates would match WRONG paths after backward pruning.

---

### 4. Variable-Aware Predicate Evaluation

**Test**: `test_predicate_timing()`

**Failure Mode**: Applying predicates before variables are bound

**Problem**: Pattern `(n1)-[]->(n2)-[]->(n3) WHERE n1.x == n3.x`
```
Step 1: n1 matched    [n1 available]
Step 2: hop to n2     [n1, n2 available, n3 NOT YET]
Step 3: hop to n3     [n1, n2, n3 available - NOW can eval]
```

**Wrong**: Try to filter at step 2
**Right**: Only evaluate when ALL referenced variables exist

**Implementation**:
```python
class PathContext:
    def can_evaluate_predicate(self, predicate: str, available_vars: Set[str]) -> bool:
        # Parse predicate to find referenced variables
        referenced = parse_variable_references(predicate)  # {'n1', 'n3'}
        return referenced.issubset(available_vars)

# During execution:
if ctx.can_evaluate_predicate('n1_x == n3_x', {'n1', 'n2'}):  # False!
    # Don't evaluate yet
```

**Impact**: Predicates would FAIL or silently produce wrong results.

---

### 5. Incremental Filtering (Not Deferred)

**Test**: `test_cartesian_explosion()`

**Failure Mode**: Building all paths then filtering at end

**Problem**: Pattern with many branches:
```
HUB -> [A1, A2, A3] -> [B1, B2, B3] each = 9 paths
With filter n1.x == n3.x, maybe only 1 matches
```

**Wrong**: Build all 9, then filter
**Right**: Filter at each step as soon as predicate can be evaluated

**Implementation**:
```python
# After each hop:
if ctx.has_evaluable_predicates(current_vars):
    wavefront = ctx.filter_wavefront(wavefront)  # Prune early!
```

**Impact**: Exponential memory/time blowup on realistic graphs.

---

### 6. Property Column Scoping

**Test**: `test_property_swapping()`

**Failure Mode**: Confusing which variable a property belongs to

**Problem**: Multiple variables, same property name:
```
n1.x, n2.x, n3.x all in same DataFrame
If columns are 'x', 'x', 'x' → collision!
```

**Wrong**: Reuse column name 'x'
**Right**: Prefix with variable name 'n1_x', 'n2_x', 'n3_x'

**Implementation**:
```python
# ✗ WRONG - Which 'x' is this?
wavefront['x'] = node_df['x']

# ✓ RIGHT - Clear ownership
wavefront[f'{var_name}_x'] = node_df['x']
```

**Impact**: Cross-variable predicates would compare WRONG values.

---

### 7. Type Consistency in Comparisons

**Test**: `test_type_confusion()`

**Failure Mode**: Comparing incompatible types

**Problem**:
```
n1.x = 10 (int)
n3.x = "10" (str)
Does n1.x == n3.x?  → No (type mismatch)
```

**Implementation**:
```python
# Option 1: Strict type checking
if type(n1_x) != type(n3_x):
    raise TypeError(f"Cannot compare {type(n1_x)} with {type(n3_x)}")

# Option 2: Coercion with warning
if type(n1_x) != type(n3_x):
    logger.warning("Type coercion in predicate")
    # Try to coerce...

# Option 3: Follow pandas semantics (10 != "10")
# Just use pandas comparison, let it return False
```

**Impact**: Silent type bugs in user queries.

---

## Implementation Checklist

Before claiming path variable support works, verify:

- [ ] Multi-edge graphs produce correct path counts
- [ ] User columns named 'n1_x' don't collide with tracking columns
- [ ] Path IDs don't change across forward→backward→forward passes
- [ ] Predicates only evaluate when all variables are bound
- [ ] Filtering happens incrementally during execution
- [ ] Variable property columns are clearly namespaced
- [ ] Type mismatches in comparisons are handled consistently

---

## Test Suite Usage

```bash
# Basic functionality tests
python docs/research/path_tracking_tests.py

# Adversarial tests (designed to break implementations)
python docs/research/path_adversarial_tests.py

# Both should pass before merging
```

---

## Lessons Learned

1. **Don't trust simple prototypes** - Edge cases matter
2. **Test with adversarial data** - Real graphs are messy
3. **3-pass model is complex** - Path IDs must persist
4. **User data is hostile** - Namespace everything
5. **Performance matters** - Incremental filtering is critical

---

**Status**: Requirements validated via tests ✓
**Next**: Implementation guided by these requirements
