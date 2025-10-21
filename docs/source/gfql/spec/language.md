(gfql-spec-language)=

# GFQL Language Specification

## Introduction

GFQL (Graph Frame Query Language) is a DataFrame-native graph query language designed for expressing graph patterns and traversals on tabular data. It operates on node and edge DataFrames, providing a functional, composable approach to graph querying with native GPU acceleration support.

### Design Principles
- **Dataframe-native**: Type-safe functional bulk operations over dataframe libraries like pandas, cuDF
- **Declarative**: Focus on what to retrieve, and give the engine freedom to optimize how
- **Accessible**: Designed for both human readability and machine generation, and building on intuitions from popular tabular and graph systems
- **Performance-oriented**: Vectorized operations by default, including GPU acceleration
- **Embeddable**: Similar to DuckDB, can be embedded in different languages, and initially focused on Python data ecosystem
- **Computer-tier**: Decoupling from storage enables flexible execution - embedded locally or via remote acceleration servers

### Language Forms

GFQL exists in three complementary forms:

1. **Core Language**: Abstract graph pattern matching language defined by this specification
2. **Embedded DSL**: Host language implementations (currently Python with pandas/cuDF)
3. **Wire Protocol**: JSON serialization for client-server communication (see Wire Protocol spec)

This specification focuses on the core language concepts. Examples use Python syntax for concreteness, but the patterns apply to any embedding.

## Language Overview

### Core Concepts

#### Graph Model

Graphs consist of node and edge dataframes:
- Edges: DataFrame with source and destination columns
- Nodes: DataFrame with unique identifier column
- Column names are user-defined globals for the graph:
  - Node ID attribute: `g._node` (e.g., "node_id", "id")
  - Edge source attribute: `g._source` (e.g., "source", "from")
  - Edge destination attribute: `g._destination` (e.g., "destination", "to")
- GFQL infers nodes from edge references when only edges are provided

#### GFQL Programs

GFQL programs are declarative graph-to-graph transformations:
- Enable use cases like search, filter, enrich, and traverse
- Express *what* to find (ex: Cypher), not *how* to find it (ex: Gremlin)

#### Chains

Path pattern expressions for matching graph structures:
- Express graph patterns as sequences of node and edge matching operations
- Similar to Cypher patterns but decomposed into composable steps
- Define paths through the graph: start nodes → edges → end nodes
- Each operation refines the pattern match based on previous results

#### Operations

Act on graph entities (nodes and edges):
- Node matchers: Filter and select nodes
- Edge matchers: Traverse relationships
- Operations work on the graph structure itself

#### Predicates

Act on attributes of nodes and edges:
- Filter based on property values
- Comparison, membership, string matching, temporal checks
- Composable within operations to build complex conditions

#### Values

Type system matching modern data formats:
- Scalars: numbers, strings, booleans, null
- Temporal: ISO datetimes, dates, times with timezone support
- Collections: lists for membership tests
- Compatible with JSON, Arrow, and DataFrame type systems

## Formal Grammar

```{code-block} ebnf
:caption: GFQL Grammar in Extended Backus-Naur Form

(* Entry point *)
query ::= chain

(* Chain - path pattern expression *)
chain ::= "[" operation ("," operation)* "]"

(* Operations *)
operation ::= node_matcher | edge_matcher

(* Node Matcher *)
node_matcher ::= "n(" node_params? ")"
node_params ::= filter_dict ("," name_param)? ("," query_param)?
              | name_param ("," query_param)?
              | query_param

(* Edge Matchers *)
edge_matcher ::= edge_forward | edge_reverse | edge_undirected
edge_forward ::= "e_forward(" edge_params? ")"
edge_reverse ::= "e_reverse(" edge_params? ")"  
edge_undirected ::= ("e" | "e_undirected") "(" edge_params? ")"

(* Parameters *)
edge_params ::= edge_match_params ("," hop_params)? ("," node_filter_params)? ("," name_param)?

filter_dict ::= "{" (property_filter ("," property_filter)*)? "}"
property_filter ::= string ":" (value | predicate)

hop_params ::= "hops=" integer | "to_fixed_point=True"
node_filter_params ::= source_filter ("," dest_filter)?
source_filter ::= "source_node_match=" filter_dict | "source_node_query=" string
dest_filter ::= "destination_node_match=" filter_dict | "destination_node_query=" string

name_param ::= "name=" string
query_param ::= "query=" string
edge_query_param ::= "edge_query=" string
edge_match_params ::= filter_dict | edge_query_param

(* Predicates *)
predicate ::= comparison | membership | range | null_check | string_pred | temporal_pred

comparison ::= ("gt" | "lt" | "ge" | "le" | "eq" | "ne") "(" value ")"
membership ::= "is_in(" "[" value ("," value)* "]" ")"
range ::= "between(" value "," value ("," "inclusive=" boolean)? ")"
null_check ::= "is_null()" | "not_null()" | "is_na()" | "not_na()"
string_pred ::= string_match | string_check
string_match ::= "contains(" string ("," "case=" boolean)? ("," "regex=" boolean)? ")"
              | "match(" string ("," "case=" boolean)? ("," "flags=" integer)? ")"
              | "fullmatch(" string ("," "case=" boolean)? ("," "flags=" integer)? ")"
              | ("startswith" | "endswith") "(" string ("," "case=" boolean)? ")"
string_check ::= ("isalpha" | "isnumeric" | "isdigit" | "isalnum"
               | "isupper" | "islower") "()"
temporal_pred ::= temporal_check "()"
temporal_check ::= "is_month_start" | "is_month_end" | "is_quarter_start" 
                 | "is_quarter_end" | "is_year_start" | "is_year_end" | "is_leap_year"

(* Values *)
value ::= scalar | temporal_value | collection
scalar ::= number | string | boolean | null
temporal_value ::= datetime_value | date_value | time_value
datetime_value ::= "pd.Timestamp(" string ("," "tz=" string)? ")"
                 | "datetime(" datetime_args ")"
date_value ::= "date(" date_args ")"
time_value ::= "time(" time_args ")"
collection ::= "[" (value ("," value)*)? "]"

(* Primitives *)
string ::= '"' [^"]* '"' | "'" [^']* "'"
number ::= integer | float
integer ::= ["-"]? [0-9]+
float ::= ["-"]? [0-9]+ "." [0-9]+
boolean ::= "True" | "False"
null ::= "None"
datetime_args ::= integer ("," integer)*
date_args ::= integer "," integer "," integer
time_args ::= integer "," integer ("," integer)?
```

## Operations

### Node Matcher: `n()`

Filters nodes based on attributes.

**Syntax**: `n(filter_dict?, name?, query?)`

**Parameters**:
- `filter_dict`: Dictionary of attribute filters
- `name`: Optional string label for results
- `query`: Pandas query string expression

**Examples**:
```python
n()                                    # All nodes
n({"type": "person"})                 # Nodes where type='person'
n({"age": gt(30)})                    # Nodes where age > 30
n(name="important")                   # Label matching nodes
n(query="age > 30 and status == 'active'")  # Query string
```

### Edge Matchers

#### Forward Traversal: `e_forward()`

Traverses edges in forward direction (source → destination).

**Syntax**: `e_forward(edge_match?, hops?, to_fixed_point?, source_node_match?, destination_node_match?, name?)`

**Parameters**:
- `edge_match`: Edge attribute filters
- `hops`: Number of hops (default: 1)
- `to_fixed_point`: Continue until no new nodes (default: False)
- `source_node_match`: Filters for source nodes
- `destination_node_match`: Filters for destination nodes
- `name`: Optional label

**Examples**:
```python
e_forward()                           # One hop forward
e_forward(hops=2)                     # Two hops forward
e_forward(to_fixed_point=True)        # All reachable nodes
e_forward({"type": "follows"})        # Only 'follows' edges
e_forward(source_node_match={"active": True})  # From active nodes
```

#### Reverse Traversal: `e_reverse()`

Traverses edges in reverse direction (destination → source).

**Syntax**: Same as `e_forward()`

#### Undirected Traversal: `e()` or `e_undirected()`

Traverses edges in both directions.

**Syntax**: Same as `e_forward()`

## Predicates

### Comparison Predicates

```python
gt(value)    # Greater than
lt(value)    # Less than
ge(value)    # Greater than or equal
le(value)    # Less than or equal
eq(value)    # Equal
ne(value)    # Not equal
```

### Membership Predicate

```python
is_in([value1, value2, ...])  # Value in list
```

### Range Predicate

```python
between(lower, upper, inclusive=True)  # Value in range
```

### String Predicates

Pattern matching predicates:
```python
contains(pat, case=True, regex=True)     # Contains pattern (substring or regex)
startswith(prefix, case=True)            # Starts with prefix
endswith(suffix, case=True)              # Ends with suffix
match(pat, case=True, flags=0)           # Matches regex from start of string
fullmatch(pat, case=True, flags=0)       # Matches regex against entire string
```

String type checking predicates:
```python
isalpha()    # Alphabetic characters only
isnumeric()  # Numeric characters only
isdigit()    # Digits only
isalnum()    # Alphanumeric
isupper()    # All uppercase
islower()    # All lowercase
```

### Null Predicates

```python
is_null()     # Is null/None
not_null()    # Is not null/None
is_na()       # Is NaN (numeric)
not_na()      # Is not NaN
```

### Temporal Predicates

```python
is_month_start()    # First day of month
is_month_end()      # Last day of month
is_quarter_start()  # First day of quarter
is_quarter_end()    # Last day of quarter
is_year_start()     # First day of year
is_year_end()       # Last day of year
is_leap_year()      # Is leap year
```

## Call Operations and Security

### Call Operations

GFQL supports calling Plottable methods through the `call()` operation, providing controlled access to graph transformation and analysis capabilities:

```python
call(function: str, params: dict) -> ASTCall
```

Call operations enable:
- Graph algorithms (PageRank, community detection)
- Layout computations (ForceAtlas2, Graphviz)
- Data transformations (filtering, collapsing)
- Visual encodings (color, size, icons)

### Safelist Architecture

For security and stability, Call operations are restricted to a predefined safelist of methods. This prevents:
- Arbitrary code execution
- Access to filesystem or network operations
- Modification of global state
- Unsafe graph operations

#### Safelist Categories

**Graph Analysis**
- `get_degrees`, `get_indegrees`, `get_outdegrees`: Calculate node degrees
- `compute_cugraph`: Run GPU algorithms (pagerank, louvain, etc.)
- `compute_igraph`: Run CPU algorithms
- `get_topological_levels`: Analyze DAG structure

**Filtering & Transformation**
- `filter_nodes_by_dict`, `filter_edges_by_dict`: Filter by attributes
- `hop`: Traverse graph with conditions
- `drop_nodes`, `keep_nodes`: Node selection
- `collapse`: Merge nodes by attribute
- `prune_self_edges`: Remove self-loops
- `materialize_nodes`: Generate node table

**Layout**
- `layout_cugraph`: GPU-accelerated layouts
- `layout_igraph`: CPU-based layouts
- `layout_graphviz`: Graphviz layouts
- `fa2_layout`: ForceAtlas2 layout
- `ring_continuous_layout`: Radial layout driven by numeric attributes
- `ring_categorical_layout`: Radial layout grouping by categories
- `time_ring_layout`: Time-series radial layout (accepts ISO timestamp bounds)

```{note}
`time_ring_layout` accepts ISO-8601 strings for `time_start` / `time_end` when
sent over the wire. GFQL converts them to `numpy.datetime64` before use so the
behavior matches direct Plotter calls.
```

**Visual Encoding**
- `encode_point_color`: Color nodes/edges
- `encode_point_size`: Size nodes
- `encode_point_icon`: Set icons
- `bind`: Attach visual attributes

**Embeddings & Dimensionality Reduction**
- `umap`: UMAP dimensionality reduction for graph embeddings

### Validation

Call operations undergo multiple validation stages:

1. **Safelist Check**: Function name must be in the safelist
2. **Parameter Validation**: Parameters validated against method signature
3. **Type Checking**: Runtime type validation
4. **Schema Validation**: Compatibility with graph schema

### Error Codes

- **E104**: Function not in safelist
- **E105**: Missing required parameter
- **E201**: Parameter type mismatch
- **E303**: Unknown parameter
- **E301**: Required column not found (runtime)

## Type System

### Value Types

1. **Scalars**
   - `number`: int, float
   - `string`: Text values
   - `boolean`: True/False
   - `null`: None

2. **Temporal Types**
   - `datetime`: Timestamp with optional timezone
   - `date`: Calendar date
   - `time`: Time of day

3. **Collections**
   - `list`: Ordered sequence of values

### Type Coercion

GFQL performs automatic type coercion:
- Python datetime → pandas Timestamp
- Numeric types → appropriate precision
- Collections → lists for `is_in()`

## Execution Model

### Declarative Pattern Matching

GFQL follows a declarative execution model similar to Neo4j's Cypher:

1. **Pattern Declaration**: Chains express path patterns in the graph
   - Users declare graph patterns as sequences of node and edge constraints
   - Patterns specify *what* paths to match, not *how* to find them
   - The engine optimizes pattern matching based on data characteristics

2. **Set-Based Operations**: All operations work on sets of entities
   - No explicit iteration or traversal order
   - Results include all matching patterns in the graph
   - Current GFQL engines use a novel bulk-oriented execution model that is asymptotically faster than traditional iterative approaches used for Cypher, but this is not a requirement of the language itself

3. **Lazy Evaluation**: Chains define pattern transformations without immediate execution
   - Allows engines to optimize path finding and pattern matching strategies
\
### Result Access

Query execution returns filtered node and edge datasets. In the Python embedding:

```python
result = g.gfql([...])
nodes_df = result._nodes  # Filtered nodes
edges_df = result._edges  # Filtered edges
```

### Named Results

Operations with `name` parameter add boolean columns to mark matched entities:

```python
result = g.gfql([
    n({"type": "person"}, name="people"),
    e_forward(name="connections"),
    n({"active": True}, name="active_targets")
])

# Access all matched nodes and edges:
all_nodes = result._nodes
all_edges = result._edges

# Access specific matched nodes/edges using pandas filtering:
people_nodes = result._nodes[result._nodes["people"]]
connection_edges = result._edges[result._edges["connections"]]
active_nodes = result._nodes[result._nodes["active_targets"]]

# Or using standard pandas query syntax:
people_nodes = result._nodes.query("people == True")
```

This pattern is essential for extracting specific subsets from complex graph traversals.

## Best Practices

1. **Use specific filters early**: Filter nodes before traversing edges
2. **Limit hops**: Use reasonable hop limits to avoid explosion
3. **Name important results**: Use `name` parameter for analysis
4. **Prefer filter_dict**: More efficient than query strings
5. **Use appropriate predicates**: Match predicate to column type

## See Also

- {ref}`gfql-spec-python-embedding` - Python implementation details
- {ref}`gfql-spec-wire-protocol` - JSON serialization format
- {ref}`gfql-spec-cypher-mapping` - Cypher to GFQL translation with wire protocol
- [GFQL Quick Reference](../quick.rst) - Comprehensive examples and usage patterns
- [GFQL Validation Guide](../validation/fundamentals.rst) - Learn validation basics
