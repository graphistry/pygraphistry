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
result = g.chain([
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
g.chain([...], engine='cudf')  # GPU execution
g.chain([...], engine='pandas')  # CPU execution
g.chain([...], engine='auto')  # Auto-select
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
g.chain([n({"username": "Alice"})])  # KeyError

# Correct - Use existing column
g.chain([n({"name": "Alice"})])
```

### Unsupported Operations

```python
# Wrong - Can't aggregate in chain
# g.chain([n(), e(), count()])

# Correct - Aggregate after chain
result = g.chain([n(), e()])
count = len(result._edges)

# Wrong - OPTIONAL MATCH not supported
# No direct GFQL equivalent

# Correct - Handle optionality in post-processing
result = g.chain([n(), e_forward()])
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
g.chain([n(node_filters)])

# Avoid: Hardcoded query strings
g.chain([n(query=f"type == 'User' and age > {min_age}")])  # SQL injection risk
```

### Memory Efficiency
```python
# Good: Filter early and use named results
result = g.chain([
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