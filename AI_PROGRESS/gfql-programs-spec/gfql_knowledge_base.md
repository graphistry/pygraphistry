# GFQL Knowledge Base

## Overview

GFQL (Graph Query Language) is a declarative graph pattern matching system implemented in PyGraphistry. It provides a composable AST-based approach for expressing complex graph traversal patterns. The core architecture follows a three-phase algorithm for efficient subgraph extraction:

1. **Forward wavefront traversal** - Explores paths from starting nodes
2. **Reverse pruning pass** - Removes dead-end paths to ensure all nodes are on complete paths
3. **Forward output pass** - Collects and labels final results

## Architecture

### Core Components

```
graphistry/compute/
├── chain.py          # Chain execution engine
├── ast.py            # AST node/edge definitions
├── predicates/       # Predicate system for filtering
├── chain_remote.py   # Remote execution via API
├── hop.py           # Single-hop traversal logic
└── ASTSerializable.py # JSON serialization base
```

## File-by-File Analysis

### 1. `/graphistry/compute/chain.py` - Core Chain Execution Engine

**Purpose**: Implements the main chain execution algorithm that processes sequences of AST operations

**Key Classes**:
- `Chain` (file:18-52) - Container for AST operation sequences
  - `__init__`: Stores list of ASTObjects
  - `from_json`: Deserializes from JSON representation 
  - `to_json`: Serializes to JSON wire format
  - `validate`: Ensures all operations are valid ASTObjects

**Key Functions**:
- `chain()` (file:148-360) - Main entry point for chain execution
  - Signature: `chain(self: Plottable, ops: Union[List[ASTObject], Chain], engine: Union[EngineAbstract, str] = EngineAbstract.AUTO) -> Plottable`
  - Implements 3-phase algorithm:
    - Forward pass (file:280-302): Computes path prefixes
    - Backward pass (file:311-349): Prunes incomplete paths
    - Combine phase (file:350-358): Merges and labels results
  - Handles edge cases like chains starting/ending with edges

- `combine_steps()` (file:56-117) - Merges results from multiple operations
  - Deduplicates nodes/edges across steps
  - Tags nodes/edges with operation names when specified
  - Special handling for edge recomputation in reverse pass

**Implementation Details**:
```python
# file:280-302 - Forward wavefront computation
g_stack : List[Plottable] = []
for op in ops:
    prev_step_nodes = (
        None  # first uses full graph
        if len(g_stack) == 0
        else g_stack[-1]._nodes
    )
    g_step = op(
        g=g,
        prev_node_wavefront=prev_step_nodes,
        target_wave_front=None,
        engine=engine_concrete
    )
    g_stack.append(g_step)
```

### 2. `/graphistry/compute/ast.py` - AST Class Definitions

**Purpose**: Defines the abstract syntax tree nodes for graph patterns

**Key Classes**:

- `ASTObject` (file:67-90) - Abstract base for all AST operations
  - Abstract methods: `__call__()`, `reverse()`
  - Stores optional `_name` for result labeling

- `ASTNode` (file:116-193) - Represents node matching operations
  - Parameters:
    - `filter_dict`: Key-value pairs for attribute matching (supports predicates)
    - `query`: Pandas query string for additional filtering
    - `name`: Optional label for matched nodes
  - `__call__()` (file:164-189): Filters nodes based on criteria
  - `reverse()`: Returns self (nodes are direction-agnostic)

- `ASTEdge` (file:206-374) - Represents edge traversal operations
  - Parameters:
    - `direction`: 'forward', 'reverse', or 'undirected'
    - `edge_match`: Filter for edge attributes
    - `hops`: Number of hops (default 1)
    - `to_fixed_point`: Continue until no new nodes
    - `source_node_match/destination_node_match`: Node filters
    - `*_query`: Pandas query strings
  - `__call__()` (file:313-352): Delegates to hop() function
  - `reverse()` (file:354-373): Flips direction and source/dest

**Specialized Edge Classes**:
- `ASTEdgeForward` (file:375-419) - Convenience for forward edges
- `ASTEdgeReverse` (file:421-464) - Convenience for reverse edges  
- `ASTEdgeUndirected` (file:467-511) - Convenience for undirected edges

**Helper Functions**:
- `n()` (file:194) - Shorthand for ASTNode
- `e_forward()`, `e_reverse()`, `e_undirected()`, `e()` - Edge shorthands

**JSON Serialization**:
```python
# file:267-294 - Example of JSON format for ASTEdge
{
    'type': 'Edge',
    'direction': 'forward',
    'hops': 2,
    'edge_match': {'type': 'transaction'},
    'source_node_match': {'risk': {'type': 'GT', 'val': 0.5}}
}
```

### 3. `/graphistry/compute/predicates/` - Predicate System

**Purpose**: Provides columnar predicates for advanced filtering beyond simple equality

**Base Class**:
- `ASTPredicate` (file:predicates/ASTPredicate.py:9-27)
  - Abstract `__call__(s: SeriesT) -> SeriesT` method
  - Inherits from ASTSerializable for JSON support

**Key Predicate Types**:

1. **Categorical** (`predicates/is_in.py`):
   - `IsIn` (file:16-150) - Check if values in list
   - Handles temporal type normalization
   - Example: `{'type': is_in(['person', 'company'])}`

2. **Numeric** (`predicates/numeric.py`):
   - `GT/LT/GE/LE/EQ/NE` - Comparison operators
   - `Between` (file:95-116) - Range checking
   - `IsNA/NotNA` - Null checking
   - Example: `{'score': gt(0.8)}`

3. **String** (`predicates/str.py`):
   - `Contains/Startswith/Endswith` - Pattern matching
   - `Match` - Regex matching
   - Various type checks: `IsNumeric/IsAlpha/IsDigit/etc`

4. **Temporal** (`predicates/temporal.py`):
   - Date/time specific predicates
   - `IsMonthStart/IsYearEnd/IsLeapYear/etc`

**Integration Example**:
```python
# file:ast.py:98-99 - Predicate validation
for k, v in d.items():
    assert isinstance(v, ASTPredicate) or is_json_serializable(v)
```

### 4. `/graphistry/compute/chain_remote.py` - Remote Execution

**Purpose**: Enables server-side GFQL execution for performance

**Key Functions**:

- `chain_remote()` (file:223-320) - Main remote execution entry
  - Auto-uploads graph if needed
  - Sends GFQL operations to server API
  - Returns results as Plottable

- `chain_remote_shape()` (file:168-221) - Fast metadata-only query
  - Returns DataFrame with graph shape info
  - Useful for checking if patterns exist

- `chain_remote_generic()` (file:16-166) - Shared implementation
  - Handles authentication via JWT
  - Supports multiple output formats (parquet, csv, json)
  - Engine selection (pandas vs cudf/GPU)

**Wire Protocol**:
```python
# file:67-80 - Request format
request_body = {
    "gfql_operations": chain_json['chain'],  # List of AST operations
    "format": "parquet",
    "node_col_subset": ["id", "type"],      # Optional column filtering
    "edge_col_subset": ["weight"],
    "engine": "cudf"                        # Force GPU mode
}
```

### 5. `/graphistry/compute/hop.py` - Single Hop Implementation

**Purpose**: Core graph traversal logic used by ASTEdge operations

**Key Function**:
- `hop()` (file:258-624) - Performs k-hop traversal
  - Parameters align with ASTEdge options
  - Handles forward/reverse/undirected traversal
  - Implements wavefront expansion algorithm
  - Column conflict resolution for node/edge ID collisions

**Algorithm Flow**:
1. Initialize wavefront with starting nodes
2. For each hop:
   - Filter source nodes by predicates
   - Follow edges matching criteria
   - Filter destination nodes
   - Update wavefront with newly reached nodes
3. Return subgraph of all traversed nodes/edges

**Helper Functions**:
- `generate_safe_column_name()` (file:15-43) - Avoid column conflicts
- `prepare_merge_dataframe()` (file:46-105) - Setup merge operations
- `process_hop_direction()` (file:113-255) - Direction-specific logic

### 6. `/graphistry/compute/ASTSerializable.py` - Serialization Base

**Purpose**: Base class for JSON serialization of AST components

**Key Methods**:
- `to_json()` (file:19-30) - Generic serialization
  - Adds 'type' field with class name
  - Serializes all non-reserved attributes
  
- `from_json()` (file:33-40) - Generic deserialization
  - Uses 'type' field to determine class
  - Passes remaining fields to constructor

## Design Patterns

### 1. **Visitor Pattern**
AST nodes implement `__call__()` to process graph data, with the chain orchestrating traversal.

### 2. **Builder Pattern**  
Operations compose into chains, building complex queries from simple primitives.

### 3. **Strategy Pattern**
Predicates encapsulate filtering strategies that can be swapped at runtime.

### 4. **Memento Pattern**
JSON serialization enables saving/restoring query state.

## Integration Points

### 1. **Engine Abstraction**
- Supports pandas (CPU) and cudf (GPU) DataFrames
- Engine selection via `resolve_engine()` helper
- Automatic engine detection based on data type

### 2. **Plottable Integration**
- All operations work on Plottable objects
- Maintains node/edge bindings throughout
- Preserves visualization settings

### 3. **Remote Execution**
- Seamless switch between local and remote
- Same API for both modes
- Automatic data upload when needed

## Example Usage Patterns

### Basic Node Filtering
```python
from graphistry import n, e_forward

# Find all person nodes
g.chain([n({"type": "person"})])
```

### Multi-hop Traversal
```python
# Find transactions between risky entities
g.chain([
    n({"risk": gt(0.8)}),
    e_forward({"type": "transfer"}, hops=2),
    n({"risk": gt(0.8)})
])
```

### Named Operations
```python
# Label intermediate results
g.chain([
    n({"type": "account"}, name="source_accounts"),
    e_forward(name="transfers"),
    n({"type": "account"}, name="dest_accounts")
])
# Access via g._nodes.source_accounts, g._edges.transfers
```

### Complex Predicates
```python
# Combine multiple predicate types
g.chain([
    n({
        "created_date": is_month_start(),
        "name": contains("Corp"),
        "value": between(1000, 10000)
    })
])
```

## Wire Protocol Examples

### Local Chain Execution
```python
chain = Chain([
    {"type": "Node", "filter_dict": {"type": "person"}},
    {"type": "Edge", "direction": "forward", "hops": 2},
    {"type": "Node", "filter_dict": {"type": "company"}}
])
g2 = g.chain(chain)
```

### Remote Execution Request
```json
{
    "gfql_operations": [
        {
            "type": "Node",
            "filter_dict": {"risk": {"type": "GT", "val": 0.5}}
        },
        {
            "type": "Edge", 
            "direction": "forward",
            "edge_match": {"amount": {"type": "Between", "lower": 1000, "upper": 5000}}
        }
    ],
    "format": "parquet",
    "engine": "cudf"
}
```

## Performance Considerations

1. **Wavefront Optimization**: The algorithm maintains a wavefront of active nodes rather than revisiting the entire graph

2. **Dead-end Pruning**: The reverse pass ensures only nodes on complete paths are returned

3. **GPU Acceleration**: When using cudf engine, operations leverage GPU parallelism

4. **Remote Execution**: For large graphs, server-side execution avoids data transfer overhead

5. **Column Subsetting**: Remote queries can request only needed columns to reduce bandwidth

## Extension Points for DAG Support

The current architecture could be extended for DAG program support by:

1. **Program AST Node**: New AST type that contains a sequence of operations
2. **Variable Binding**: Allow naming intermediate results for reuse
3. **Conditional Logic**: Add branching based on graph properties
4. **Aggregation Operators**: Support for counting, grouping operations
5. **Subprogram Composition**: Enable nesting of program definitions

The existing serialization framework and remote execution infrastructure provide a solid foundation for these extensions.

## PyGraphistry API Integration

### Entry Points

GFQL functionality is exposed through the PyGraphistry API via several key integration points:

#### 1. **Plottable Interface** (`/graphistry/Plottable.py`)

The `Plottable` protocol (file:48-791) defines the main API contracts:

```python
# Local chain execution (file:419-423)
def chain(self, ops: Union[Any, List[Any]]) -> 'Plottable':
    """ops is Union[List[ASTObject], Chain]"""
    ...

# Remote chain execution (file:425-440)
def chain_remote(
    self: 'Plottable',
    chain: Union[Any, Dict[str, JSONVal]],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    output_type: OutputTypeGraph = "all",
    format: Optional[FormatType] = None,
    df_export_args: Optional[Dict[str, Any]] = None,
    node_col_subset: Optional[List[str]] = None,
    edge_col_subset: Optional[List[str]] = None,
    engine: Optional[Literal["pandas", "cudf"]] = None
) -> 'Plottable':
    ...

# Remote shape query (file:442-456)
def chain_remote_shape(
    self: 'Plottable',
    chain: Union[Any, Dict[str, JSONVal]],
    # ... same parameters as chain_remote
) -> pd.DataFrame:
    ...
```

#### 2. **Plotter Class Hierarchy** (`/graphistry/plotter.py`)

The main `Plotter` class (file:21-93) inherits from multiple mixins:

```python
class Plotter(
    KustoMixin, SpannerMixin,
    CosmosMixin, NeptuneMixin,
    HeterographEmbedModuleMixin,
    SearchToGraphMixin,
    DGLGraphMixin, ClusterMixin,
    UMAPMixin,
    FeatureMixin, ConditionalMixin,
    LayoutsMixin,
    ComputeMixin, PlotterBase  # <-- GFQL exposed via ComputeMixin
):
```

#### 3. **ComputeMixin** (`/graphistry/compute/ComputeMixin.py`)

Provides the actual implementation bridge (file:462-472):

```python
def chain(self, *args, **kwargs):
    return chain_base(self, *args, **kwargs)
chain.__doc__ = chain_base.__doc__

def chain_remote(self, *args, **kwargs) -> Plottable:
    return chain_remote_base(self, *args, **kwargs)
chain_remote.__doc__ = chain_remote_base.__doc__

def chain_remote_shape(self, *args, **kwargs) -> pd.DataFrame:
    return chain_remote_shape_base(self, *args, **kwargs)
chain_remote_shape.__doc__ = chain_remote_shape_base.__doc__
```

### Wire Protocol Format

#### 1. **Request Format** (`/graphistry/compute/chain_remote.py`)

Remote execution sends requests to the server API (file:67-89):

```python
# API endpoint
url = f"{self.base_url_server()}/api/v2/etl/datasets/{dataset_id}/gfql/{output_type}"

# Request body structure
request_body = {
    "gfql_operations": chain_json['chain'],  # List of AST operations
    "format": format,                        # "json", "csv", "parquet"
    "node_col_subset": node_col_subset,      # Optional: limit returned columns
    "edge_col_subset": edge_col_subset,      # Optional: limit returned columns
    "df_export_args": df_export_args,        # Optional: pandas export args
    "engine": engine                         # Optional: "pandas" or "cudf"
}

# Headers
headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json",
}
```

#### 2. **Response Handling** (`/graphistry/compute/chain_remote.py`)

The response format depends on `output_type` and `format` (file:93-166):

- **Shape queries** (`output_type="shape"`): Returns metadata DataFrame
- **Graph queries** (`output_type="all"`): Returns nodes and edges
  - JSON format: `{"nodes": [...], "edges": [...]}`
  - CSV/Parquet: ZIP file containing separate nodes/edges files
- **Partial queries** (`output_type="nodes"` or `"edges"`): Single DataFrame

#### 3. **Output Type Definitions** (`/graphistry/models/compute/chain_remote.py`)

```python
# Graph-oriented outputs (file:5-6)
OutputTypeGraph = Literal["all", "nodes", "edges", "shape"]

# DataFrame outputs (file:11-12)
OutputTypeDf = Literal["table", "shape"]

# JSON outputs (file:14-15)
OutputTypeJson = Literal["json"]

# Format types (file:8-9)
FormatType = Literal["json", "csv", "parquet"]
```

### Authentication and Session Handling

#### 1. **Session Management** (`/graphistry/client_session.py`)

Each Plotter instance maintains session state (file:26-100):

```python
class ClientSession:
    """Holds all configuration and authentication state"""
    
    # Authentication state
    api_key: Optional[str]
    api_token: Optional[str]  # JWT token
    
    # Server configuration
    hostname: str
    protocol: str
    api_version: ApiVersion  # 1 or 3
    
    # Organization settings
    org_name: Optional[str]
    privacy: Optional[Privacy]
```

#### 2. **Token Refresh** (`/graphistry/compute/chain_remote.py`)

Automatic token refresh on remote calls (file:30-33):

```python
if not api_token:
    from graphistry.pygraphistry import PyGraphistry
    PyGraphistry.refresh()  # Refreshes JWT if needed
    api_token = PyGraphistry.api_token()
```

#### 3. **Dataset Upload** (`/graphistry/compute/chain_remote.py`)

Automatic upload if no dataset_id exists (file:35-40):

```python
if not dataset_id:
    dataset_id = self._dataset_id

if not dataset_id:
    self = self.upload(validate=validate)  # Uploads current graph
    dataset_id = self._dataset_id
```

### Error Handling Patterns

#### 1. **Validation** (`/graphistry/compute/chain_remote.py`)

- Input validation (file:42-48): Checks output_type, engine values
- Chain validation (file:64-65): Validates AST structure before sending
- Response validation (file:91): Uses `response.raise_for_status()`

#### 2. **Error Types**

Common error scenarios handled:
- Missing dataset_id (ValueError)
- Invalid output_type or format (ValueError)
- HTTP errors (requests.HTTPError)
- Deserialization errors based on data type detection

### Python Remote Execution

Related API for arbitrary Python execution (`/graphistry/compute/python_remote.py`):

```python
# Execute Python code remotely (file:34-96)
def python_remote_generic(
    self: Plottable,
    code: Union[str, Callable[..., object]],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    format: Optional[FormatType] = 'json',
    output_type: Optional[OutputTypeAll] = 'json',
    engine: Literal["pandas", "cudf"] = "cudf",
    run_label: Optional[str] = None,
    validate: bool = True
) -> Union[Plottable, pd.DataFrame, Any]:
    ...

# Request format (file:133-139)
request_body = {
    "execute": code_indented,  # Python code with task(g) function
    "engine": engine,
    "run_label": run_label,    # Optional job tracking
    "format": format,
    "output_type": output_type
}
```

### Integration Architecture

The API follows a layered architecture:

1. **User API Layer**: `Plotter` class with friendly methods
2. **Protocol Layer**: `Plottable` interface defining contracts
3. **Implementation Layer**: `ComputeMixin` bridging to compute modules
4. **Execution Layer**: `chain.py` (local) and `chain_remote.py` (remote)
5. **Transport Layer**: HTTP/JSON wire protocol to server

This separation enables:
- Clean API surface for users
- Protocol-based testing and mocking
- Engine-agnostic implementation (pandas vs cudf)
- Seamless local/remote execution switching
- Session isolation for multi-tenant scenarios

### Usage Patterns

#### Basic Local Execution
```python
import graphistry
from graphistry import n, e_forward

g = graphistry.edges(df, 'src', 'dst')
g2 = g.chain([n({"type": "person"}), e_forward(), n({"type": "company"})])
```

#### Remote Execution with Auto-upload
```python
# Automatically uploads if needed
g2 = g.chain_remote([n({"type": "person"}), e_forward(), n({"type": "company"})])
```

#### Optimized Remote Query
```python
# Get just shape info without full data transfer
shape_df = g.chain_remote_shape(
    [n({"risk": gt(0.8)}), e_forward(hops=2)],
    engine='cudf',  # Force GPU
    node_col_subset=['id', 'risk']  # Limit columns
)
if len(shape_df) > 0:
    # Fetch full results only if matches exist
    g2 = g.chain_remote([...])
```