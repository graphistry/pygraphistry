(gfql-spec-language)=

# GFQL Language Specification

## Introduction

GFQL (Graph Frame Query Language) is a DataFrame-native graph query language designed for expressing graph patterns and traversals on tabular data. It operates on node and edge DataFrames, providing a functional, composable approach to graph querying with native GPU acceleration support.

### Design Principles
- **Dataframe-native**: Works directly with pandas/cuDF dataframes
- **Functional composition**: Queries are composed of chainable operations
- **Type-safe**: Strong typing with clear coercion rules
- **Performance-oriented**: Vectorized operations with GPU support
- **LLM-friendly**: Clear syntax optimized for code generation

## Language Overview

### Core Concepts

1. **Graph Model**: Graphs consist of node and edge dataframes
   - Nodes: DataFrame with unique identifier column
   - Edges: DataFrame with source and destination columns

2. **Operations**: Two types of operations
   - Node matchers: Filter and select nodes
   - Edge matchers: Traverse relationships

3. **Chains**: Sequences of operations that define patterns
   - Execute left-to-right
   - Each operation filters based on previous results

4. **Predicates**: Reusable filtering conditions
   - Comparison, membership, string matching, etc.
   - Composable within operations

## Formal Grammar

```{code-block} ebnf
:caption: GFQL Grammar in Extended Backus-Naur Form

(* Entry point *)
query ::= chain

(* Chain - sequence of operations *)
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
string_pred ::= ("contains" | "startswith" | "endswith" | "match") "(" string ")"
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

**Note**: Use `"type"` for categorical attributes and `"label"` for Cypher-style node labels. The choice depends on your data schema - use what matches your DataFrame column names.

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

```python
contains(pattern)    # Contains substring
startswith(prefix)   # Starts with prefix
endswith(suffix)     # Ends with suffix
match(regex)         # Matches regular expression
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

### Three-Phase Algorithm

1. **Forward Wavefront Pass**
   - Process operations left-to-right
   - Each operation filters based on previous results
   - Build path prefixes (may include dead-ends)

2. **Reverse Pruning Pass**
   - Process operations right-to-left
   - Remove paths that don't reach the end
   - Ensure all results are on complete paths

3. **Forward Output Pass**
   - Collect final results
   - Apply named labels
   - Merge with original dataframes

### Result Access

```python
result = g.chain([...])
nodes_df = result._nodes  # Filtered nodes
edges_df = result._edges  # Filtered edges
```

### Named Results

Operations with `name` parameter add boolean columns:
```python
g.chain([
    n({"type": "person"}, name="people"),
    e_forward(name="connections")
])
# The _nodes DataFrame will have 'people' boolean column
# _edges will have 'connections' boolean column
```

## Examples

### Basic Patterns

```python
# Find all person nodes
g.chain([n({"type": "person"})])

# One-hop neighbors
g.chain([n({"id": "Alice"}), e(), n()])

# Multi-hop paths
g.chain([n({"id": "A"}), e_forward(hops=3), n({"id": "B"})])
```

### User 360 Pattern

```python
# Find customer's recent interactions
g.chain([
    n({"customer_id": "C123"}),
    e_forward({
        "type": "interaction",
        "timestamp": gt(pd.Timestamp.now() - pd.Timedelta(days=30))
    }),
    n(name="touchpoints")
])
```

### Cyber Security Pattern

```python
# Find compromised paths
g.chain([
    n({"status": "compromised"}),
    e_forward(
        edge_match={"protocol": is_in(["HTTP", "SSH"])},
        to_fixed_point=True
    ),
    n({"type": "critical_asset"}, name="at_risk")
])
```

### Complex Filtering

```python
# Combine multiple conditions
g.chain([
    n({
        "account_type": "business",
        "balance": gt(10000),
        "created": between(date(2023, 1, 1), date(2023, 12, 31))
    }),
    e_forward(
        edge_query="amount > 1000 and status == 'completed'",
        source_node_query="region == 'US'",
        destination_node_match={"verified": True}
    )
])
```

### Code Golf Examples

```python
# Friends of friends
g.chain([n({"id": "Bob"}), e({"type": "friend"}, hops=2)])

# All paths between nodes
g.chain([n({"id": "A"}), e(to_fixed_point=True), n({"id": "B"})])

# Recent high-value transactions
g.chain([n(), e({"amount": gt(1000), "date": gt(pd.Timestamp.now() - pd.Timedelta(7))})])
```

## Best Practices

1. **Use specific filters early**: Filter nodes before traversing edges
2. **Limit hops**: Use reasonable hop limits to avoid explosion
3. **Name important results**: Use `name` parameter for analysis
4. **Prefer filter_dict**: More efficient than query strings
5. **Use appropriate predicates**: Match predicate to column type

## Engine Support

GFQL supports multiple execution engines:
- `pandas`: CPU execution (default)
- `cudf`: GPU acceleration
- `auto`: Automatic selection based on data type

```python
g.chain([...], engine='cudf')  # Force GPU execution
```

## See Also

- [GFQL Validation Guide](../validation/fundamentals.rst) - Learn validation basics
- {ref}`gfql-spec-wire-protocol` - JSON serialization format
- {ref}`gfql-spec-cypher-mapping` - Cypher to GFQL translation
- {ref}`gfql-spec-synthesis-examples` - Code generation examples