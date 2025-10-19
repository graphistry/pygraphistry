# GFQL JSON Wire Spec Generation - Research Notes

## Discovery 1: Core Serialization Infrastructure

### Source: `graphistry/compute/ASTSerializable.py`

**Key Finding**: All GFQL AST nodes inherit from `ASTSerializable` which provides automatic JSON serialization.

**Method**: `to_json(validate=True) -> Dict[str, JSONVal]`
- Line: 79-90
- Converts any ASTSerializable to JSON format
- Automatically adds `type` field from class name
- Serializes all non-reserved instance fields

**Example Code**:
```python
def to_json(self, validate=True) -> Dict[str, JSONVal]:
    if validate:
        self.validate()
    data: Dict[str, JSONVal] = {'type': self.__class__.__name__}
    for key, value in self.__dict__.items():
        if key not in self.reserved_fields:
            data[key] = serialize_to_json_val(value)
    return data
```

**Key Insight**: The JSON wire protocol uses a discriminated union pattern with `type` field indicating the AST node class.

---

## Discovery 2: Predicate Deserialization Registry

### Source: `graphistry/compute/predicates/from_json.py`

**Key Finding**: There's a registry-based system for deserializing predicates from JSON.

**Components**:
1. **predicates list** (Line 18-26): Registry of all predicate classes
2. **type_to_predicate dict** (Line 28-31): Maps type names to classes
3. **from_json()** function (Line 33-42): Deserializes based on type field

**Predicate Types**:
- Categorical: `Duplicated`
- Membership: `IsIn`
- Numeric: `GT`, `LT`, `GE`, `LE`, `EQ`, `NE`, `Between`, `IsNA`, `NotNA`
- String: `Contains`, `Startswith`, `Endswith`, `Match`, `Fullmatch`, `IsNumeric`, `IsAlpha`, etc.
- Temporal: `IsMonthStart`, `IsMonthEnd`, `IsQuarterStart`, etc.

**Example**:
```python
type_to_predicate: Dict[str, Type[ASTPredicate]] = {
    cls.__name__: cls
    for cls in predicates
}

def from_json(d: Dict[str, JSONVal]) -> ASTPredicate:
    pred = type_to_predicate[d['type']]
    out = pred.from_json(d)
    out.validate()
    return out
```

---

## Discovery 3: Basic AST Node Serialization Examples

### Source: `graphistry/tests/compute/test_ast.py`

**Example 1: Node Matcher (ASTNode)**
```python
# Python form
node = n(query='zzz', name='abc')

# JSON form (from .to_json())
{
    'type': 'Node',
    'query': 'zzz',
    '_name': 'abc'
}

# Round-trip test shows it works bidirectionally
node2 = from_json(o)
```

**Example 2: Edge Matcher (ASTEdge)**
```python
# Python form
edge = e(edge_query='zzz', name='abc')

# JSON form
{
    'type': 'Edge',
    'edge_query': 'zzz',
    '_name': 'abc'
}
```

**Location**: `test_serialization_node()` and `test_serialization_edge()` functions

---

## Discovery 4: Call Operations (ASTCall)

### Source: `graphistry/tests/compute/test_call_operations.py:120-129`

**Example: Call with Parameters**
```python
# Python form
call = ASTCall('get_degrees', {'col': 'degree', 'engine': 'pandas'})

# JSON form
{
    'type': 'Call',
    'function': 'get_degrees',
    'params': {'col': 'degree', 'engine': 'pandas'}
}
```

**Key Insight**: Call operations have:
- `type`: Always "Call"
- `function`: String name of the method
- `params`: Dict of parameters (validated against safelist)

**Validation**: Parameters are validated in `call_executor.py:105` via `validate_call_params()`

---

## Discovery 5: Helper Functions for AST Construction

### Source: `graphistry/compute/ast.py` (from call_executor.py imports)

**Available Helpers**:
- `n()` - Create ASTNode (node matcher)
- `e()` - Create ASTEdge (any direction)
- `e_forward()` - Forward edge
- `e_reverse()` - Reverse edge
- `e_undirected()` - Undirected edge

**Usage Pattern**:
```python
from graphistry.compute.ast import n, e, ASTCall, call

# Method 1: Direct construction
node = n({'type': 'person'})
edge = e({'weight': {'gt': 0.5}})

# Method 2: For calls, use ASTCall or call() helper
call_op = ASTCall('hypergraph', {'entity_types': ['user', 'post']})
# OR
call_op = call('hypergraph', {'entity_types': ['user', 'post']})
```

---

## Next Steps

Need to research:
1. ✅ Basic node/edge serialization
2. ✅ Call operations
3. ✅ Predicates registry
4. ⏳ Chain serialization
5. ⏳ ASTLet (DAG) serialization
6. ⏳ Nested predicates (and_/or_)
7. ⏳ Complete examples for common flows

---

## Questions to Answer
- [ ] How do chains serialize to JSON?
- [ ] How do let bindings (DAGs) serialize?
- [ ] What does a complete hypergraph query look like in JSON?
- [ ] What does a UMAP query look like in JSON?
- [ ] How are predicates nested in node/edge queries?
- [ ] How do references (ASTRef) work in DAGs?
