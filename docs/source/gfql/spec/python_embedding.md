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

## DAG Patterns with Let Bindings

GFQL supports directed acyclic graph (DAG) patterns using Let bindings, which allow you to define named graph operations that can reference each other.

### Let Bindings

```python
from graphistry import let, ref, n, e_forward

# Define DAG patterns with named bindings
result = g.gfql(let({
    'persons': n({'type': 'person'}),
    'adults': ref('persons', [n({'age': ge(18)})]),
    'connections': ref('adults', [
        e_forward({'type': 'knows'}),
        ref('adults')  # Find connections between adults
    ])
}))

# Access individual binding results
persons_df = result._nodes[result._nodes['persons']]
adults_df = result._nodes[result._nodes['adults']]
connection_edges = result._edges[result._edges['connections']]
```

### Ref (Reference to Named Bindings)

The `ref()` function creates references to named bindings within a Let:

```python
# Basic reference - just the binding result
result = g.gfql(let({
    'base': n({'status': 'active'}),
    'extended': ref('base')  # Just references 'base'
}))

# Reference with additional operations
result = g.gfql(let({
    'suspects': n({'risk_score': gt(80)}),
    'lateral_movement': ref('suspects', [
        e_forward({'type': 'ssh', 'failed_attempts': gt(5)}),
        n({'type': 'server'})
    ])
}))
```

### Complex DAG Patterns

```python
# Multi-level analysis pattern
result = g.gfql(let({
    # Find high-value accounts
    'high_value': n({'balance': gt(100000)}),

    # Find transactions from high-value accounts
    'high_value_txns': ref('high_value', [
        e_forward({'type': 'transaction', 'amount': gt(10000)})
    ])
}))
```

### Remote Dataset Usage

```python
from graphistry import remote_dataset

# Load a public dataset
remote_g = remote_dataset('public-dataset-id')
result = remote_g.gfql([n({'type': 'user'})])

# Load a private dataset with authentication
remote_g = remote_dataset('private-dataset-id', token='auth-token')

# Use remote dataset in Let bindings
result = g.gfql(let({
    'remote_data': remote_dataset('dataset-123'),
    'filtered': ref('remote_data', [n({'active': True})])
}))
```

## Call Operations and Let Bindings

### Call with Let Bindings

```python
from graphistry import let, ref, call

# Combine Let bindings with Call operations
result = g.gfql(let({
    'ranked': call('pagerank'),
    'influencers': ref('ranked', [n(node_query='pagerank > 0.02')])
}))
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

Operations are automatically validated during construction:

```python
from graphistry.compute.chain import Chain
from graphistry.compute.ast import n, e_forward

# Automatic validation on construction
chain = Chain([
    n({'type': 'person'}),
    e_forward({'hops': -1})  # Raises GFQLTypeError: hops must be positive
])
```

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

## See Also

- {ref}`gfql-spec-language` - Core language specification
- [GFQL Quick Reference](../quick.rst) - Python API examples