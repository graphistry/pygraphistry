(gfql-spec-python-embedding)=

# GFQL Python Embedding

This document describes the Python-specific implementation of GFQL using pandas and cuDF dataframes.

## Graph Construction

In Python, graphs are created with user-defined column names:

```python
import graphistry
assert 'src_col' in df.columns and 'dst_col' in df.columns
g = graphistry.edges(df, source='src_col', destination='dst_col')

# Optional; GFQL infers node existence when only edges are provided
assert 'node_col' in df.columns
g2 = graphistry.nodes(df, node='node_col')
```

### Schema Access

The graph schema is accessible via attributes:
- `g._node`: Node ID column name
- `g._source`: Edge source column name  
- `g._destination`: Edge destination column name

Graph nodes can be generically accessed using these attributes:
- `g._nodes`: Node DataFrame
- `g._nodes[g._node]`: Node ID column
- `g._nodes[[attr for attr in g._nodes.columns if attr != g._node]]`: All other node attributes

Graph edges can be accessed similarly:
- `g._edges`: Edge DataFrame
- `g._edges[g._source]`: Edge source column
- `g._edges[g._destination]`: Edge destination column
- `g._edges[[attr for attr in g._edges.columns if attr not in [g._source, g._destination]]]`: All other edge attributes

## Query Execution

```python
from graphistry import n, e_forward

# Execute a chain
result = g.gfql([
    n({"type": "person"}),
    e_forward(),
    n()
])

# Access results
nodes_df = result._nodes  # Filtered nodes DataFrame
edges_df = result._edges  # Filtered edges DataFrame
```

## Engine Selection

GFQL supports multiple execution engines:

- **pandas**: CPU execution (default)
- **cudf**: GPU acceleration
- **auto**: Automatic selection based on data type

```python
# Force specific engine
g.gfql([...], engine='cudf')  # GPU execution
g.gfql([...], engine='pandas')  # CPU execution
g.gfql([...], engine='auto')  # Auto-select
```

## Python-Specific Values

### Temporal Values

```python
import pandas as pd

# Timestamps
pd.Timestamp('2023-01-01')
pd.Timestamp.now()

# Time deltas
pd.Timedelta(days=30)
pd.Timedelta(hours=24)
```

### DataFrame Operations

Results can be further processed using standard pandas operations:

```python
# Using boolean columns from named operations
people_nodes = result._nodes[result._nodes["people"]]

# Using pandas query
active_nodes = result._nodes.query("active == True")

# Standard pandas operations
result._nodes.groupby('type').size()
```

## Validation

GFQL provides comprehensive validation to catch errors early:

### Syntax Validation

Chains validate on construction by default. Nodes, edges, predicates, refs, calls, and remote graphs are validated when a parent `Chain`/`Let` validates them or when you call `.validate()` directly. Schema validation is a separate, data-aware pass.

```python
from graphistry.compute.chain import Chain
from graphistry.compute.ast import n, e_forward

# Automatic validation on construction
chain = Chain([
    n({'type': 'person'}),
    e_forward({'hops': -1})  # Raises GFQLTypeError: hops must be positive
])
```

For advanced flows (large/nested ASTs or staged assembly), you can defer structural validation and run it once after assembly:

```python
# Defer validation while building
chain = Chain([
    n({'type': 'person'}),
    e_forward({'hops': -1})
], validate=False)  # No validation yet

# Later, validate once (or let g.gfql validate it)
chain.validate()  # Raises GFQLTypeError: hops must be positive
```

Use deferred validation to avoid re-validating nested `Chain`/`Let` wrappers during assembly; keep the defaults for typical workflows so mistakes surface immediately.

### Validation Phases

- **Constructor defaults:** `Chain([...])` and `Let(...)` validate immediately; pass `validate=False` to defer.
- **Parent-driven checks:** AST operations (`Node`, `Edge`, predicates, `Ref`, `Call`, `RemoteGraph`) validate when their parent validates, or via explicit `.validate()`.
- **JSON defaults:** `to_json` / `from_json` default to `validate=True`, which runs structural validation during serialization/deserialization.
- **Schema validation:** Use `validate_chain_schema(g, chain)` or `g.gfql(..., validate_schema=True)` to verify column/type compatibility before execution.

### Schema Validation

You have two options for validating queries against your data schema:

1. **Validate-only** (no execution): Use `validate_chain_schema()` to check compatibility without running the query
2. **Validate-and-run**: Use `g.gfql(..., validate_schema=True)` to validate before execution

```python
# Method 1: Validate-only (no execution)
from graphistry.compute.validate_schema import validate_chain_schema

chain = Chain([n({'missing_column': 'value'})])
try:
    validate_chain_schema(g, chain)  # Only validates, doesn't execute
    print("Chain is valid for this graph")
except GFQLSchemaError as e:
    print(f"Schema incompatibility: {e}")
    print("No query was executed")

# Method 2: Runtime validation (automatic)
try:
    result = g.gfql([
        n({'missing_column': 'value'})
    ])  # Validates during execution, raises GFQLSchemaError
except GFQLSchemaError as e:
    print(f"Runtime validation error: {e}")

# Method 3: Validate-and-run (pre-execution validation)
try:
    result = g.gfql([
        n({'missing_column': 'value'})
    ], validate_schema=True)  # Validates first, only executes if valid
except GFQLSchemaError as e:
    print(f"Pre-execution validation failed: {e}")
    print("Query was not executed")
```

### Error Types

GFQL uses structured exceptions with error codes:

- **GFQLSyntaxError** (E1xx): Structural issues
  - E101: Invalid type (e.g., chain not a list)
  - E103: Invalid parameter value (e.g., negative hops)
  - E104: Invalid direction
  - E105: Missing required field

- **GFQLTypeError** (E2xx): Type mismatches
  - E201: Wrong value type (e.g., string instead of dict)
  - E202: Predicate type mismatch
  - E204: Invalid name type

- **GFQLSchemaError** (E3xx): Data-related issues
  - E301: Column not found
  - E302: Incompatible column type (e.g., numeric predicate on string column)

### Validation Modes

```python
# Fail-fast mode (default) - raises on first error
chain.validate()

# Collect-all mode - returns list of all errors
errors = chain.validate(collect_all=True)
for error in errors:
    print(f"[{error.code}] {error.message}")
    if error.suggestion:
        print(f"  Suggestion: {error.suggestion}")

# Pre-validate schema without execution
from graphistry.compute.validate_schema import validate_chain_schema

# Check schema compatibility
errors = validate_chain_schema(g, chain, collect_all=True)
```

### Example: Handling Validation Errors

```python
from graphistry.compute.exceptions import GFQLValidationError, GFQLSchemaError

try:
    result = g.gfql([
        n({'age': 'twenty-five'})  # Type mismatch
    ])
except GFQLSchemaError as e:
    print(f"Schema error [{e.code}]: {e.message}")
    print(f"Field: {e.context.get('field')}")
    print(f"Suggestion: {e.context.get('suggestion')}")
    # Output:
    # Schema error [E302]: Type mismatch: column "age" is numeric but filter value is string
    # Field: age
    # Suggestion: Use a numeric value like age=25
```

## Common Errors and Validation

### Type Mismatches

```python
# Wrong - String predicate on numeric column
n({"age": contains("3")})

# Correct - Use numeric predicate
n({"age": gt(30)})

# Wrong - String comparison on datetime
n({"created": gt("2024-01-01")})

# Correct - Use proper datetime type
n({"created": gt(pd.Timestamp("2024-01-01"))})
```

### Schema Validation

```python
# Check available columns before querying
print(g._nodes.columns)  # ['id', 'type', 'name']

# Wrong - Column doesn't exist
g.gfql([n({"username": "Alice"})])  # KeyError

# Correct - Use existing column
g.gfql([n({"name": "Alice"})])
```

### Unsupported Operations

```python
# Wrong - Can't aggregate in chain
# g.gfql([n(), e(), count()])

# Correct - Aggregate after chain
result = g.gfql([n(), e()])
count = len(result._edges)

# Wrong - OPTIONAL MATCH not supported
# No direct GFQL equivalent

# Correct - Handle optionality in post-processing
result = g.gfql([n(), e_forward()])
# Check for nodes without edges
nodes_with_edges = result._nodes[result._nodes[g._node].isin(result._edges[g._source])]
```

## Best Practices

### Query Construction
```python
# Good: Build queries programmatically
node_filters = {"type": "User"}
if min_age:
    node_filters["age"] = gt(min_age)
g.gfql([n(node_filters)])

# Avoid: Hardcoded query strings
g.gfql([n(query=f"type == 'User' and age > {min_age}")])  # SQL injection risk
```

### Memory Efficiency
```python
# Good: Filter early and use named results
result = g.gfql([
    n({"active": True}, name="active_users"),  # Filter first
    e_forward({"recent": True})
])
# Only access what you need
active_users = result._nodes[result._nodes["active_users"]]

# Avoid: Loading everything then filtering
all_nodes = g._nodes
active = all_nodes[all_nodes["active"] == True]  # Loads entire graph
```

### GPU Best Practices
```python
# Check GPU memory before large operations
if engine == 'cudf':
    import cudf
    print(f"GPU memory: {cudf.cuda.cuda.get_memory_info()}")
    
# Convert results back to pandas if needed for compatibility
result_pandas = result._nodes.to_pandas() if hasattr(result._nodes, 'to_pandas') else result._nodes
```

## DAG Patterns with Let Bindings

GFQL supports directed acyclic graph (DAG) patterns using Let bindings, which allow you to define named graph operations that can reference each other.

### Let Bindings

```python
from graphistry import let, ref, n, e_forward, ge

# Define DAG patterns with named bindings
result = g.gfql(let({
    'persons': n({'type': 'person'}),
    'adults': ref('persons', [n({'age': ge(18)})]),
    'connections': [
        n({'type': 'person', 'age': ge(18)}),
        e_forward({'type': 'knows'}),
        n({'type': 'person', 'age': ge(18)})
    ]
}))

# Access individual binding results
persons_df = result._nodes[result._nodes['persons']]
adults_df = result._nodes[result._nodes['adults']]
connection_edges = result._edges[result._edges['connections']]
```

### Ref (Reference to Named Bindings)

The `ref()` function creates references to named bindings within a Let.
Ref chains run on the referenced graph; bindings created by `n()` contain nodes only,
so edge traversals need a binding that preserves edges (for example, via a list or `Chain([...])`).

```python
# Basic reference - just the binding result
result = g.gfql(let({
    'base': n({'status': 'active'}),
    'extended': ref('base')  # Just references 'base'
}))

# Reference with additional operations (node-only refinements)
result = g.gfql(let({
    'suspects': n({'risk_score': gt(80)}),
    'verified': ref('suspects', [
        n({'verified': True})
    ])
}))

# For traversals, inline the seed filter into a list or Chain binding
result = g.gfql(let({
    'lateral_movement': [
        n({'risk_score': gt(80)}),
        e_forward({'type': 'ssh', 'failed_attempts': gt(5)}),
        n({'type': 'server'})
    ]
}))
```

### Complex DAG Patterns

```python
# Multi-level analysis pattern
result = g.gfql(let({
    # Find high-value accounts
    'high_value': n({'balance': gt(100000)}),

    # Find transactions from high-value accounts
    'large_transfers': [
        n({'balance': gt(100000)}),
        e_forward({'type': 'transfer', 'amount': gt(10000)}),
        n()
    ],

    # Find suspicious patterns
    'suspicious': ref('large_transfers', [
        n({'created_recent': True, 'verified': False})
    ])
}))
```

### Remote Graph References

For distributed computing, `remote()` allows referencing graphs on remote servers:

```python
from graphistry import remote

# Reference a remote dataset
result = g.gfql([
    remote(dataset_id='fraud-network-2024'),
    n({'risk_score': gt(90)}),
    e_forward()
])
```

## Call Operations with Let Bindings

Call operations can be used within Let bindings for complex workflows:

```python
result = g.gfql(let({
    # Initial filtering with edges preserved for graph algorithms
    'suspects': Chain([
        n({'flagged': True}),
        e_undirected(),
        n({'flagged': True})
    ]),

    # Compute PageRank on subgraph
    'ranked': ref('suspects', [
        call('compute_cugraph', {'alg': 'pagerank'})
    ]),

    # Find high PageRank nodes
    'influencers': ref('ranked', [
        n({'pagerank': gt(0.01)})
    ])
}))
```

## See Also

- {ref}`gfql-spec-language` - Core language specification
- [GFQL Quick Reference](../quick.rst) - Python API examples
