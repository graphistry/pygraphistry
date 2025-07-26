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

# Execute a GFQL query
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
    ]),
    
    # Find recipients of high-value transactions
    'recipients': ref('high_value_txns', [n()]),
    
    # Find second-hop connections
    'network': ref('recipients', [
        e_forward({'type': 'transaction'}, hops=2)
    ])
}))
```

### RemoteGraph (Load Remote Datasets)

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

## Call Operations

GFQL supports calling Plottable methods through the `call()` function, providing a safe way to invoke graph transformations and analysis operations.

### Basic Call Usage

```python
from graphistry import call

# Calculate node degrees
result = g.gfql([
    n({'type': 'person'}),
    call('get_degrees', {
        'col': 'centrality',
        'col_in': 'in_centrality',
        'col_out': 'out_centrality'
    })
])

# Access degree columns
degree_df = result._nodes[['centrality', 'in_centrality', 'out_centrality']]
```

### Graph Analysis Operations

```python
# PageRank computation
result = g.gfql([
    call('compute_cugraph', {
        'alg': 'pagerank',
        'out_col': 'pagerank_score',
        'params': {'alpha': 0.85}
    })
])

# Community detection
result = g.gfql([
    call('compute_cugraph', {
        'alg': 'louvain',
        'out_col': 'community'
    })
])

# Topological analysis
result = g.gfql([
    call('get_topological_levels', {
        'level_col': 'topo_level',
        'allow_cycles': True
    })
])
```

### Layout Operations

```python
# GPU-accelerated layout
result = g.gfql([
    call('layout_cugraph', {
        'layout': 'force_atlas2',
        'params': {
            'iterations': 500,
            'outbound_attraction_distribution': True
        }
    })
])

# Graphviz layouts
result = g.gfql([
    call('layout_graphviz', {
        'prog': 'dot',
        'directed': True
    })
])
```

### Filtering and Transformation

```python
# Complex filtering
result = g.gfql([
    call('filter_nodes_by_dict', {
        'filter_dict': {'type': 'server', 'critical': True}
    }),
    call('hop', {
        'hops': 2,
        'direction': 'forward',
        'edge_match': {'port': 22}
    })
])

# Node transformations
result = g.gfql([
    call('collapse', {
        'column': 'department',
        'self_edges': False
    })
])
```

### Visual Encoding

```python
# Encode visual properties
result = g.gfql([
    call('encode_point_color', {
        'column': 'risk_score',
        'palette': ['green', 'yellow', 'red'],
        'as_continuous': True
    }),
    call('encode_point_size', {
        'column': 'importance',
        'categorical_mapping': {
            'low': 10,
            'medium': 20,
            'high': 30
        }
    })
])
```

### Call with Let Bindings

```python
from graphistry import let, ref, call

# Combine Let bindings with Call operations
result = g.gfql(let({
    'high_risk': n({'risk_score': gt(80)}),
    'connected': ref('high_risk', [
        e_forward({'type': 'communicates'})
    ]),
    'analyzed': call('compute_cugraph', {
        'alg': 'pagerank',
        'out_col': 'influence'
    })
}))
```

### Available Call Methods

Call operations are restricted to a safelist of Plottable methods:

- **Graph Analysis**: `get_degrees`, `get_indegrees`, `get_outdegrees`, `compute_cugraph`, `compute_igraph`, `get_topological_levels`
- **Filtering**: `filter_nodes_by_dict`, `filter_edges_by_dict`, `hop`, `drop_nodes`, `keep_nodes`
- **Transformation**: `collapse`, `prune_self_edges`, `materialize_nodes`
- **Layout**: `layout_cugraph`, `layout_igraph`, `layout_graphviz`, `fa2_layout`
- **Visual Encoding**: `encode_point_color`, `encode_edge_color`, `encode_point_size`, `encode_point_icon`
- **Metadata**: `name`, `description`

### Call Validation

Call operations are validated at multiple levels:

1. **Function validation**: Only safelist methods allowed
2. **Parameter validation**: Type checking for all parameters
3. **Schema validation**: Ensures required columns exist

```python
try:
    result = g.gfql([
        call('dangerous_method', {})  # Raises E303: not in safelist
    ])
except GFQLTypeError as e:
    print(f"Error: {e.message}")
    
# Parameter type validation
try:
    result = g.gfql([
        call('hop', {'hops': 'two'})  # Raises E201: wrong type
    ])
except GFQLTypeError as e:
    print(f"Error: {e.message}")
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
from graphistry.compute.ast import n, e_forward

# Automatic validation on construction
from graphistry import ASTNode, ASTEdge
operations = [
    n({'type': 'person'}),
    e_forward({'hops': -1})  # Raises GFQLTypeError: hops must be positive
]
# Validation happens when operations are created
```

### Schema Validation

Schema validation happens during execution or can be done pre-emptively:

```python
# Runtime validation (automatic)
result = g.gfql([
    n({'missing_column': 'value'})  # Raises GFQLSchemaError during execution
])

# Pre-execution validation (optional)
result = g.gfql([
    n({'missing_column': 'value'})
], validate_schema=True)  # Raises GFQLSchemaError before execution
```

### Error Types

GFQL uses structured exceptions with error codes:

- **GFQLSyntaxError** (E1xx): Structural issues
  - E101: Invalid type (e.g., operations not a list)
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
for op in operations:
    op.validate()

# Collect-all mode - returns list of all errors
errors = []
for op in operations:
    try:
        op.validate()
    except Exception as e:
        errors.append(e)
for error in errors:
    print(f"[{error.code}] {error.message}")
    if error.suggestion:
        print(f"  Suggestion: {error.suggestion}")

# Pre-validate schema without execution
from graphistry.compute.validate_schema import validate_gfql_schema

# Check schema compatibility
errors = validate_gfql_schema(g, operations, collect_all=True)
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
# Wrong - Can't aggregate in GFQL query
# g.gfql([n(), e(), count()])

# Correct - Aggregate after GFQL query
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