# Mark Mode Design Document

**Issue**: #755
**Branch**: `feat/755-mark-mode`
**Status**: Phase 1 - Design & Validation
**Created**: 2025-10-19

---

## Overview

Mark mode enriches graphs with boolean columns indicating pattern matches **without filtering out non-matches**. This enables:
- Multi-stage pattern detection with accumulating marks
- Visualization of match vs non-match entities
- Conditional logic based on multiple mark columns

---

## API Design

### Instance Method Form

```python
# Mark nodes matching GFQL pattern
g2 = g.mark(gfql=[n({'type': 'person'})], name='is_person')
# Result: g2._nodes has 'is_person' column (True for matches, False for non-matches)

# Mark edges matching GFQL pattern
g2 = g.mark(gfql=[e_forward({'rel': 'friend'})], name='is_friend')
# Result: g2._edges has 'is_friend' column
```

### Call Operation Form

```python
# Use in call() for remote execution
g2 = g.gfql(call('mark', {'gfql': [n({'type': 'person'})], 'name': 'is_person'}))

# Composable in let() DAG
g2 = g.gfql(let({
    'marked_people': call('mark', {
        'gfql': [n({'type': 'person'})],
        'name': 'is_person'
    }),
    'marked_companies': ref('marked_people', [
        call('mark', {
            'gfql': [n({'type': 'company'})],
            'name': 'is_company'
        })
    ]),
    # Result: both 'is_person' and 'is_company' columns present
}))
```

---

## Method Signature

```python
def mark(
    self: Plottable,
    gfql: Union[Chain, List[ASTObject]],
    name: str,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
) -> Plottable:
    """Mark nodes or edges matching GFQL pattern with boolean column.

    Executes GFQL pattern and adds boolean column indicating matches.
    Unlike filtering, ALL entities are preserved.

    Args:
        gfql: GFQL pattern to match (Chain or list of AST objects)
        name: Name for the boolean marker column
        engine: Execution engine (pandas/cudf/dask)

    Returns:
        New Plottable with marker column added to nodes or edges

    Raises:
        GFQLTypeError: If gfql is not Chain or List[ASTObject]
        GFQLSyntaxError: If gfql pattern is invalid
        GFQLSchemaError: If name conflicts with internal columns

    Example:
        # Mark VIP customers
        g2 = g.mark(
            gfql=[n({'customer_type': 'VIP'})],
            name='is_vip'
        )

        # Multiple marks accumulate
        g3 = g2.mark(
            gfql=[n({'region': 'EMEA'})],
            name='is_emea'
        )
        # g3._nodes has both 'is_vip' and 'is_emea' columns
    """
```

---

## Design Decisions

### 1. Nodes vs Edges: Inferred from GFQL

**Decision**: Single `mark()` method that inspects GFQL to determine target

**Rationale**:
- GFQL patterns inherently specify nodes (`n()`) or edges (`e_forward()`, etc.)
- Simpler API: one method instead of `mark_nodes()` + `mark_edges()`
- Consistent with existing `chain()` which handles both

**Implementation**:
```python
# Inspect final operation in GFQL chain
if isinstance(final_op, ASTNode):
    target = 'nodes'
elif isinstance(final_op, ASTEdge):
    target = 'edges'
else:
    raise GFQLSyntaxError("mark() requires node or edge matcher")
```

### 2. Boolean Column Values

**Decision**: `True` for matches, `False` for non-matches

**Rationale**:
- Clear semantics: boolean column with no NaN
- Easier filtering: `g._nodes[g._nodes['is_person']]`
- Better for visualization: can color by boolean directly
- Composable: `is_person & is_vip` works cleanly

**Alternative considered**: `True` for matches, `NaN` for non-matches
- Rejected: Harder to use, requires `.fillna(False)` everywhere

### 3. Column Naming Strategy

**Decision**: User provides exact column name, error on collision

**Rationale**:
- Explicit is better than implicit
- User controls their schema
- Follows pattern from `get_degrees(col='degree')`
- Prevents accidental overwrites

**Name collision behavior**:
```python
# If column already exists, raise error
if name in g._nodes.columns:
    raise GFQLSchemaError(
        f"Column '{name}' already exists",
        suggestion="Choose different name or drop existing column first"
    )
```

**Alternative considered**: Auto-increment (`is_person`, `is_person_1`, ...)
- Rejected: Surprising behavior, user loses control

### 4. Mark Semantics in let()

**Decision**: Marks accumulate across let() bindings

**Rationale**:
- Each binding can add its own mark column
- Referencing a binding inherits all its columns
- Natural composition: `ref('marked_people', [...])` carries forward `is_person` column

**Example**:
```python
g.gfql(let({
    'people': call('mark', {'gfql': [n({'type': 'person'})], 'name': 'is_person'}),
    'vip_people': ref('people', [
        call('mark', {'gfql': [n({'vip': True})], 'name': 'is_vip'})
    ]),
}))
# Result: vip_people has BOTH 'is_person' and 'is_vip' columns
```

### 5. Internal Implementation

**Strategy**: Execute GFQL, identify matches, create boolean column

**Pseudocode**:
```python
def mark(self, gfql, name, engine):
    # 1. Execute GFQL to get matches
    matched_g = self.chain(gfql, engine=engine)

    # 2. Determine target (nodes or edges)
    final_op = gfql[-1] if isinstance(gfql, list) else gfql.chain[-1]
    if isinstance(final_op, ASTNode):
        target_df = self._nodes
        matched_df = matched_g._nodes
        id_col = self._node
    else:
        target_df = self._edges
        matched_df = matched_g._edges
        id_col = [self._source, self._destination]

    # 3. Create boolean column
    matched_ids = set(matched_df[id_col].values) if isinstance(id_col, str) else set(matched_df[id_col].itertuples(index=False, name=None))

    if isinstance(id_col, str):
        mask = target_df[id_col].isin(matched_ids)
    else:
        # For edges, need to match on (source, destination) pairs
        mask = target_df.apply(lambda row: (row[id_col[0]], row[id_col[1]]) in matched_ids, axis=1)

    # 4. Add column to original graph
    enriched_df = target_df.assign(**{name: mask})

    # 5. Return new graph
    if isinstance(final_op, ASTNode):
        return self.nodes(enriched_df)
    else:
        return self.edges(enriched_df)
```

---

## Validation Requirements

### Parameter Validation

1. **gfql parameter**:
   - Must be `Chain` or `List[ASTObject]`
   - Cannot be empty
   - Must end with `ASTNode` or `ASTEdge` (not `ASTCall` or other operations)

2. **name parameter**:
   - Must be string
   - Cannot be empty
   - Cannot start with `__gfql_` (internal column prefix)
   - Cannot already exist in target DataFrame (error, not overwrite)

3. **engine parameter**:
   - Must be valid `EngineAbstract` or string ('auto', 'pandas', 'cudf', etc.)

### Safelist Registration

```python
# In graphistry/compute/gfql/call_safelist.py
SAFELIST_V1 = {
    # ... existing operations ...

    'mark': {
        'allowed_params': {'gfql', 'name', 'engine'},
        'required_params': {'gfql', 'name'},
        'param_validators': {
            'gfql': lambda v: isinstance(v, (list, Chain)),
            'name': lambda v: isinstance(v, str) and len(v) > 0,
            'engine': lambda v: isinstance(v, (str, EngineAbstract))
        },
        'description': 'Mark nodes/edges matching GFQL pattern with boolean column'
    },
}
```

---

## Error Handling

### E301: Column Name Collision
```python
g.mark(gfql=[n()], name='existing_column')
# GFQLSchemaError: Column 'existing_column' already exists
# Suggestion: Choose different name or drop existing column first
```

### E302: Invalid GFQL Type
```python
g.mark(gfql="not a list", name='foo')
# GFQLTypeError: gfql must be Chain or List[ASTObject]
# Suggestion: Use [n()] or chain([n()])
```

### E303: GFQL Ends with Non-Matcher
```python
g.mark(gfql=[n(), call('get_degrees')], name='foo')
# GFQLSyntaxError: mark() requires GFQL ending with node or edge matcher
# Suggestion: Remove call() from end of pattern
```

### E304: Internal Column Name
```python
g.mark(gfql=[n()], name='__gfql_temp__')
# GFQLSchemaError: Column name '__gfql_temp__' conflicts with internal namespace
# Suggestion: Use user-facing column name without '__gfql_' prefix
```

---

## Testing Strategy

### Unit Tests

1. **Basic marking**:
   - `test_mark_nodes_basic()`: Mark nodes with simple pattern
   - `test_mark_edges_basic()`: Mark edges with simple pattern
   - `test_mark_boolean_values()`: Verify True/False values

2. **Error cases**:
   - `test_mark_name_collision()`: Error on existing column
   - `test_mark_invalid_gfql_type()`: Error on non-list/Chain
   - `test_mark_internal_column_name()`: Error on `__gfql_*` prefix

3. **Composition**:
   - `test_mark_accumulation()`: Multiple marks on same graph
   - `test_mark_in_let()`: Marks accumulate across let() bindings
   - `test_mark_then_filter()`: Use mark column for subsequent filtering

### Integration Tests

1. **Remote execution**:
   - `test_mark_call_operation()`: `call('mark', {...})` in chain
   - `test_mark_in_remote_gfql()`: Mark in remote GFQL query

2. **Visualization**:
   - `test_mark_color_encoding()`: Color nodes by mark column
   - `test_multiple_marks_visualization()`: Multiple boolean columns

---

## Documentation Plan

### API Documentation

- Add `mark()` to `ComputeMixin` docstring
- Add `call('mark')` to call operations guide
- Update GFQL reference with mark examples

### User Guide

New section: "Marking Patterns" in GFQL docs
- Introduction to marking vs filtering
- Basic examples
- Multi-mark workflows
- Visualization with marks

### Notebooks

- `demos/gfql/mark_mode_basics.ipynb`: Introduction and basic usage
- `demos/gfql/multi_mark_analysis.ipynb`: Advanced multi-mark patterns

---

## Implementation Phases

### Phase 2: Core Implementation (4-5 days)

1. **Day 1-2**: Implement `mark()` in `ComputeMixin`
   - Add method signature
   - Implement GFQL execution logic
   - Add boolean column creation

2. **Day 3**: Register in call safelist
   - Add to `SAFELIST_V1`
   - Add parameter validators
   - Add call executor logic

3. **Day 4**: Handle edge cases
   - Empty results
   - All matches / no matches
   - Engine-specific DataFrame operations

4. **Day 5**: Code review and refinement

### Phase 3: Testing (3-4 days)

1. **Day 1-2**: Unit tests
   - Basic marking tests
   - Error handling tests
   - Composition tests

2. **Day 3**: Integration tests
   - Remote execution
   - let() composition
   - Visualization integration

3. **Day 4**: Edge case testing and fixes

### Phase 4: Documentation (2 days)

1. **Day 1**: API docs and docstrings
2. **Day 2**: User guide and notebooks

---

## Open Questions

1. **Performance**: Should we optimize for large graphs?
   - Potential: Use index-based matching instead of set membership
   - Defer until profiling shows need

2. **Remote execution**: How to serialize GFQL in call('mark', ...)?
   - Current approach: Pass GFQL as JSON in params
   - May need special handling in call_executor

3. **Multi-hop marks**: What if GFQL has multi-hop pattern?
   - Current design: Mark final wavefront (nodes or edges)
   - Alternative: Allow marking intermediate hops? (Defer to v2)

---

## Related Issues

- #755: Mark mode feature request
- #722: GFQL path support (may want to mark paths in future)
- #791: Interior call mixing (closed - use let() instead)

---

## References

- [ComputeMixin.py](../../graphistry/compute/ComputeMixin.py): Enrichment pattern (`get_degrees()`)
- [call_safelist.py](../../graphistry/compute/gfql/call_safelist.py): Call operation registration
- [ast.py](../../graphistry/compute/ast.py): ASTNode/ASTEdge name parameter behavior

---

**Last Updated**: 2025-10-19
**Next Review**: After Phase 1 completion
