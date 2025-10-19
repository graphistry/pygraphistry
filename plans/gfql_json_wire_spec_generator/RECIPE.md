# GFQL JSON Wire Spec Generation - Simple Recipe

## Quick Start

To generate JSON wire protocol specifications for GFQL queries, use the `.to_json()` method on any AST object.

```python
from graphistry.compute.ast import n, e_forward, ASTCall, ASTLet
from graphistry.compute.chain import Chain
from graphistry.compute.predicates.is_in import is_in

# Example: Simple node matcher
node = n({'type': 'person'})
json_spec = node.to_json()
# Returns: {"type": "Node", "filter_dict": {"type": "person"}}
```

---

## Core Concepts

### 1. All AST Objects Have `.to_json()`

Every GFQL AST object inherits from `ASTSerializable` (defined in `graphistry/compute/ASTSerializable.py`), which provides automatic JSON serialization.

**Key Method:**
```python
def to_json(self, validate=True) -> Dict[str, JSONVal]:
    """Convert AST object to JSON wire protocol format."""
```

### 2. Discriminated Union Pattern

The JSON format uses a `type` field to indicate the AST node class:

```json
{
  "type": "Node",
  ...other fields...
}
```

**Type Mapping:**
- `ASTNode` → `"type": "Node"`
- `ASTEdge` → `"type": "Edge"`
- `ASTCall` → `"type": "Call"`
- `Chain` → `"type": "Chain"`
- `ASTLet` → `"type": "Let"`
- `ASTRef` → `"type": "Ref"`
- Predicates → `"type": "GT"`, `"IsIn"`, `"Contains"`, etc.

### 3. Automatic Field Serialization

All non-reserved instance fields are automatically serialized:

```python
# Python
e_forward(hops=2, edge_match={'weight': gt(0.5)})

# JSON
{
  "type": "Edge",
  "hops": 2,
  "to_fixed_point": false,
  "direction": "forward",
  "edge_match": {
    "weight": {
      "type": "GT",
      "val": 0.5
    }
  }
}
```

---

## Step-by-Step Recipe

### Step 1: Import Required Components

```python
# Core AST builders
from graphistry.compute.ast import (
    n,              # Node matcher
    e,              # Edge matcher (any direction)
    e_forward,      # Forward edge
    e_reverse,      # Reverse edge
    e_undirected,   # Undirected edge
    ASTCall,        # Call operation
    ASTLet,         # Let binding (DAG)
    ASTRef          # Reference to binding
)

# Chain wrapper
from graphistry.compute.chain import Chain

# Predicates
from graphistry.compute.predicates.is_in import is_in
from graphistry.compute.predicates.numeric import gt, lt, ge, le, eq, ne
from graphistry.compute.predicates.str import contains, startswith, endswith
```

### Step 2: Build Your GFQL Query

Use the Python DSL helpers to construct your query:

```python
# Example: Find high-value users and their purchases
query = Chain([
    n({'type': 'user', 'lifetime_value': gt(1000)}),
    e_forward(edge_match={'action': 'purchased'}),
    n({'type': 'product'})
])
```

### Step 3: Generate JSON Wire Spec

Call `.to_json()` on your query object:

```python
json_spec = query.to_json()
```

### Step 4: Use the JSON

The JSON can be:
- Sent over the wire to a remote GFQL service
- Stored in a database
- Used in API requests
- Validated and executed remotely

```python
import json

# Pretty-print
print(json.dumps(json_spec, indent=2))

# Send to API
# requests.post('/api/gfql', json=json_spec)
```

---

## Common Patterns

### Pattern 1: Simple Chain (Graph Traversal)

```python
from graphistry.compute.ast import n, e_forward
from graphistry.compute.chain import Chain

chain = Chain([
    n({'type': 'person', 'name': 'Alice'}),
    e_forward(),
    n({'type': 'person'})
])

json_spec = chain.to_json()
```

### Pattern 2: Call Operation (Hypergraph)

```python
from graphistry.compute.ast import ASTCall

call = ASTCall('hypergraph', {
    'entity_types': ['user', 'post', 'comment'],
    'direct': True
})

json_spec = call.to_json()
```

### Pattern 3: Call Operation (UMAP)

```python
from graphistry.compute.ast import ASTCall

call = ASTCall('umap', {
    'kind': 'nodes',
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2
})

json_spec = call.to_json()
```

### Pattern 4: Let Binding (Multi-Step Query)

```python
from graphistry.compute.ast import n, ASTCall, ASTLet
from graphistry.compute.chain import Chain
from graphistry.compute.predicates.numeric import gt

dag = ASTLet({
    'users': Chain([n({'type': 'user'})]),
    'with_degrees': ASTCall('get_degrees', {'col': 'degree'}),
    'high_degree': Chain([n({'degree': gt(10)})])
})

json_spec = dag.to_json()
```

### Pattern 5: References in DAG

```python
from graphistry.compute.ast import n, e_forward, ASTLet, ASTRef
from graphistry.compute.chain import Chain

dag = ASTLet({
    'users': Chain([n({'type': 'user'})]),
    'user_friends': ASTRef('users', [
        e_forward(edge_match={'type': 'friend'}),
        n()
    ])
})

json_spec = dag.to_json()
```

### Pattern 6: Predicates

```python
from graphistry.compute.ast import n
from graphistry.compute.predicates.is_in import is_in
from graphistry.compute.predicates.numeric import gt

node = n({
    'age': gt(18),
    'country': is_in(['USA', 'Canada', 'UK'])
})

json_spec = node.to_json()
```

---

## Validation

By default, `.to_json(validate=True)` performs validation:

```python
# With validation (default)
json_spec = query.to_json()

# Skip validation (not recommended)
json_spec = query.to_json(validate=False)
```

**Validation checks:**
- Required fields present
- Correct types
- Call parameters match safelist
- Predicate syntax

---

## Deserialization (Reverse Process)

To convert JSON back to AST objects:

```python
from graphistry.compute.ast import from_json

# For general AST objects
ast_obj = from_json(json_spec)

# For predicates specifically
from graphistry.compute.predicates.from_json import from_json as pred_from_json
predicate = pred_from_json(json_spec)
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `graphistry/compute/ASTSerializable.py` | Base class with `to_json()` / `from_json()` |
| `graphistry/compute/ast.py` | Core AST classes (Node, Edge, Call, Let, Ref) |
| `graphistry/compute/chain.py` | Chain wrapper for sequences |
| `graphistry/compute/predicates/from_json.py` | Predicate registry for deserialization |
| `graphistry/compute/gfql/call_safelist.py` | Call parameter validation |
| `docs/source/gfql/spec/wire_protocol.md` | Wire protocol specification |

---

## Testing Your JSON Generation

Use the provided test suite as examples:

```python
# See these test files for comprehensive examples:
# - graphistry/tests/compute/test_ast.py
# - graphistry/tests/compute/test_chain.py
# - graphistry/tests/compute/test_call_operations.py
# - graphistry/tests/compute/predicates/test_from_json.py
```

Or run the example generator:

```bash
uv run python3.12 plans/gfql_json_wire_spec_generator/generate_examples.py
```

---

## Common Issues

### Issue 1: Unknown Call Parameters

**Error:** `Unknown parameters for 'function_name'`

**Solution:** Check `graphistry/compute/gfql/call_safelist.py` for valid parameters

### Issue 2: Missing Required Fields

**Error:** `Call missing function` or similar

**Solution:** Ensure all required constructor arguments are provided

### Issue 3: Type Validation Errors

**Error:** `Invalid type for parameter`

**Solution:** Check parameter types in the safelist definition

---

## Summary

1. **Import** AST builders and predicates
2. **Build** your query using Python DSL
3. **Call** `.to_json()` to generate wire spec
4. **Use** the JSON for remote execution, storage, or transmission

The entire system is designed to make JSON generation automatic and seamless!
