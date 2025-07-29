(gfql-spec-cypher-mapping)=

# Cypher to GFQL Python & Wire Protocol Mapping

## Introduction

This specification shows how to translate Cypher queries to both GFQL Python code and Wire Protocol JSON, enabling:
- Migration from Cypher-based systems
- Two-stage LLM synthesis: Text → Cypher → GFQL
- Language-agnostic API integration
- Secure query generation without code execution

## Conceptual Framework

### Translation Scenarios

When translating from Cypher, you'll encounter three scenarios:

**1. Direct Translation** - Most pattern matching maps cleanly to pure GFQL  
**2. Hybrid Approach** - Post-processing operations (RETURN clauses) use dataframes  
**3. GFQL Advantages** - Some capabilities go beyond what Cypher offers

### What Translates Directly
- Graph patterns: `(a)-[r]->(b)` → chain operations
- Property filters: WHERE clauses embed into operations
- Path traversals: Variable-length paths use `hops` parameter
- Pattern composition: Multiple patterns become sequential operations

### What Requires DataFrames
- Aggregations: COUNT, SUM, AVG → pandas operations
- Projections: RETURN specific columns → DataFrame selection
- Sorting/limiting: ORDER BY, LIMIT → DataFrame methods
- Joins: Multiple disconnected patterns → pandas merge

### GFQL Advantages Beyond Cypher
- **Rich edge properties**: Query edges as first-class entities
- **Dataframe-native**: Zero-cost transitions between graph and tabular operations
- **GPU acceleration**: Massively parallel execution on NVIDIA hardware
- **Heterogeneous graphs**: No schema constraints on types or properties

## Quick Example

**Cypher:**
```cypher
MATCH (p:Person)-[r:FOLLOWS]->(q:Person) 
WHERE p.age > 30
```

**Python:**
```python
g.chain([
    n({"type": "Person", "age": gt(30)}, name="p"),
    e_forward({"type": "FOLLOWS"}, name="r"),
    n({"type": "Person"}, name="q")
])
```

**Wire Protocol:**
```json
{"type": "Chain", "chain": [
  {"type": "Node", "filter_dict": {"type": "Person", "age": {"type": "GT", "val": 30}}, "name": "p"},
  {"type": "Edge", "direction": "forward", "edge_match": {"type": "FOLLOWS"}, "name": "r"},
  {"type": "Node", "filter_dict": {"type": "Person"}, "name": "q"}
]}
```

## Pattern Translations

### Node Patterns

| Cypher | Python | Wire Protocol |
|--------|--------|---------------|
| `(n)` | `n()` | `{"type": "Node"}` |
| `(n:Label)` | `n({"type": "Label"})` | `{"type": "Node", "filter_dict": {"type": "Label"}}` |
| `(n {prop: val})` | `n({"prop": val})` | `{"type": "Node", "filter_dict": {"prop": val}}` |
| `(n:Person) WHERE n.age > 30` | `n({"type": "Person", "age": gt(30)})` | `{"type": "Node", "filter_dict": {"type": "Person", "age": {"type": "GT", "val": 30}}}` |

### Edge Patterns

| Cypher | Python | Wire Protocol (compact) |
|--------|--------|-------------------------|
| `-[]->` | `e_forward()` | `{"type": "Edge", "direction": "forward"}` |
| `-[r:KNOWS]->` | `e_forward({"type": "KNOWS"}, name="r")` | `{"type": "Edge", "direction": "forward", "edge_match": {"type": "KNOWS"}, "name": "r"}` |
| `<-[r]-` | `e_reverse(name="r")` | `{"type": "Edge", "direction": "reverse", "name": "r"}` |
| `-[r]-` | `e(name="r")` | `{"type": "Edge", "direction": "undirected", "name": "r"}` |
| `-[*2]->` | `e_forward(hops=2)` | `{"type": "Edge", "direction": "forward", "hops": 2}` |
| `-[*1..3]->` | `e_forward(hops=3)` | `{"type": "Edge", "direction": "forward", "hops": 3}` |
| `-[*]->` | `e_forward(to_fixed_point=True)` | `{"type": "Edge", "direction": "forward", "to_fixed_point": true}` |
| `-[r:BOUGHT {amount: gt(100)}]->` | `e_forward({"type": "BOUGHT", "amount": gt(100)}, name="r")` | `{"type": "Edge", "direction": "forward", "edge_match": {"type": "BOUGHT", "amount": {"type": "GT", "val": 100}}, "name": "r"}` |

### Predicates

| Cypher | Python | Wire Protocol |
|--------|--------|---------------|
| `n.age > 30` | `gt(30)` | `{"type": "GT", "val": 30}` |
| `n.age >= 50` | `ge(50)` | `{"type": "GE", "val": 50}` |
| `n.age < 100` | `lt(100)` | `{"type": "LT", "val": 100}` |
| `n.age <= 50` | `le(50)` | `{"type": "LE", "val": 50}` |
| `n.status = 'active'` | `"active"` | `"active"` |
| `n.status <> 'deleted'` | `ne("deleted")` | `{"type": "NE", "val": "deleted"}` |
| `n.id IN [1,2,3]` | `is_in([1,2,3])` | `{"type": "IsIn", "options": [1,2,3]}` |
| `n.score BETWEEN 0 AND 100` | `between(0, 100)` | `{"type": "Between", "lower": 0, "upper": 100}` |
| `n.name =~ '^A.*'` | `match("^A.*")` | `{"type": "Match", "pattern": "^A.*"}` |
| `n.text CONTAINS 'search'` | `contains("search")` | `{"type": "Contains", "pattern": "search"}` |
| `n.name STARTS WITH 'Dr'` | `startswith("Dr")` | `{"type": "Startswith", "pattern": "Dr"}` |
| `n.email ENDS WITH '.com'` | `endswith(".com")` | `{"type": "Endswith", "pattern": ".com"}` |
| `n.val IS NULL` | `is_null()` | `{"type": "IsNull"}` |
| `n.val IS NOT NULL` | `not_null()` | `{"type": "NotNull"}` |

## Complete Examples

### Friend of Friend

**Cypher:**
```cypher
MATCH (u:User {name: 'Alice'})-[:FRIEND*2]->(fof:User)
WHERE fof.active = true
```

**Python:**
```python
g.chain([
    n({"type": "User", "name": "Alice"}),
    e_forward({"type": "FRIEND"}, hops=2),
    n({"type": "User", "active": True}, name="fof")
])
```

**Wire Protocol:**
```json
{"type": "Chain", "chain": [
  {"type": "Node", "filter_dict": {"type": "User", "name": "Alice"}},
  {"type": "Edge", "direction": "forward", "edge_match": {"type": "FRIEND"}, "hops": 2},
  {"type": "Node", "filter_dict": {"type": "User", "active": true}, "name": "fof"}
]}
```

### Fraud Detection

**Cypher:**
```cypher
MATCH (a:Account)-[t:TRANSFER]->(b:Account)
WHERE t.amount > 10000 AND t.date > date('2024-01-01')
```

**Python:**
```python
g.chain([
    n({"type": "Account"}),
    e_forward({
        "type": "TRANSFER", 
        "amount": gt(10000),
        "date": gt(date(2024,1,1))
    }, name="t"),
    n({"type": "Account"})
])
```

**Wire Protocol:**
```json
{"type": "Chain", "chain": [
  {"type": "Node", "filter_dict": {"type": "Account"}},
  {"type": "Edge", "direction": "forward", "edge_match": {
    "type": "TRANSFER",
    "amount": {"type": "GT", "val": 10000},
    "date": {"type": "GT", "val": {"type": "date", "value": "2024-01-01"}}
  }, "name": "t"},
  {"type": "Node", "filter_dict": {"type": "Account"}}
]}
```

### Complex Aggregation Example

**Cypher:**
```cypher
MATCH (u:User)-[t:TRANSACTION]->(m:Merchant)
WHERE t.date > date('2024-01-01')
RETURN m.category, count(*) as cnt, sum(t.amount) as total
ORDER BY total DESC
LIMIT 10
```

**Python:**
```python
# Step 1: Graph pattern
result = g.chain([
    n({"type": "User"}),
    e_forward({"type": "TRANSACTION", "date": gt(date(2024,1,1))}, name="trans"),
    n({"type": "Merchant"})
])

# Step 2: DataFrame operations
trans_df = result._edges[result._edges["trans"]]
merchant_df = result._nodes
analysis = (trans_df
    .merge(merchant_df, left_on=g._destination, right_on=g._node)
    .groupby('category')
    .agg(cnt=('amount', 'count'), total=('amount', 'sum'))
    .nlargest(10, 'total'))
```

**Note:** Wire protocol returns the filtered graph; aggregations require client-side processing.

## WITH Clause Mapping: Let Bindings

Cypher's `WITH` clause for intermediate variables maps to GFQL's Let bindings for reusable patterns.

### Basic WITH Pattern

**Cypher:**
```cypher
MATCH (u:User)-[:FRIEND]->(f)
WITH u, count(f) as friend_count
MATCH (u)-[:TRANSACTION]->(t:Transaction)
WHERE friend_count > 5
```

**Python:**
```python
g.let({
    'social_users': n({'type': 'User'}).chain([e_forward({'type': 'FRIEND'}), n()]),
    'high_social': ref('social_users', [n({'friend_count': gt(5)})]),
    'transactions': ref('high_social').chain([e_forward({'type': 'TRANSACTION'}), n({'type': 'Transaction'})])
})
```

**Wire Protocol:**
```json
{"type": "Let", "bindings": {
  "social_users": {"type": "Chain", "chain": [
    {"type": "Node", "filter_dict": {"type": "User"}},
    {"type": "Edge", "direction": "forward", "edge_match": {"type": "FRIEND"}},
    {"type": "Node"}
  ]},
  "high_social": {"type": "ChainRef", "ref": "social_users", "chain": [
    {"type": "Node", "filter_dict": {"friend_count": {"type": "GT", "val": 5}}}
  ]},
  "transactions": {"type": "ChainRef", "ref": "high_social", "chain": [
    {"type": "Edge", "direction": "forward", "edge_match": {"type": "TRANSACTION"}},
    {"type": "Node", "filter_dict": {"type": "Transaction"}}
  ]}
}}
```

### Pattern Reuse

**Cypher:**
```cypher
MATCH (p:Person {risk_score: > 8})
WITH p as suspects
MATCH (suspects)-[:CONNECTED]-(contacts)
WITH suspects, contacts
MATCH (contacts)-[:TRANSACTION]->(evidence)
```

**Python:**
```python
g.let({
    'suspects': n({'type': 'Person', 'risk_score': gt(8)}),
    'contacts': ref('suspects').chain([e_undirected({'type': 'CONNECTED'}), n()]),
    'evidence': ref('contacts').chain([e_forward({'type': 'TRANSACTION'}), n()])
})
```

**Wire Protocol:**
```json
{"type": "Let", "bindings": {
  "suspects": {"type": "Node", "filter_dict": {"type": "Person", "risk_score": {"type": "GT", "val": 8}}},
  "contacts": {"type": "ChainRef", "ref": "suspects", "chain": [
    {"type": "Edge", "direction": "undirected", "edge_match": {"type": "CONNECTED"}},
    {"type": "Node"}
  ]},
  "evidence": {"type": "ChainRef", "ref": "contacts", "chain": [
    {"type": "Edge", "direction": "forward", "edge_match": {"type": "TRANSACTION"}},
    {"type": "Node"}
  ]}
}}
```

**Note:** GFQL Let bindings provide more flexibility than Cypher WITH - patterns can reference multiple previous bindings and form complex DAG structures.

## DataFrame Operations Mapping

| Cypher Feature | Python DataFrame Operation | Notes |
|----------------|---------------------------|--------|
| `RETURN a, b, c` | `df[['a', 'b', 'c']]` | Column selection |
| `RETURN DISTINCT` | `df.drop_duplicates()` | Remove duplicates |
| `ORDER BY x DESC` | `df.sort_values('x', ascending=False)` | Sort results |
| `LIMIT 10` | `df.head(10)` | Limit rows |
| `count(*)` | `len(df)` or `df.groupby(...).size()` | Count rows |
| `sum(n.val)` | `df['val'].sum()` or `df.groupby(...).agg(sum)` | Aggregation |
| `collect(n.x)` | `df.groupby(...).agg(list)` | Collect to list |
| Named patterns | `df[df['pattern_name']]` | Boolean column filtering |

## Key Differences

| Feature | Python | Wire Protocol |
|---------|--------|---------------|
| **Temporal values** | `pd.Timestamp()`, `date()` | `{"type": "date", "value": "..."}` |
| **Direct equality** | `"active"` | `"active"` (same) |
| **Comparisons** | `gt(30)` | `{"type": "GT", "val": 30}` |
| **Collections** | `is_in([...])` | `{"type": "IsIn", "options": [...]}` |

## Not Supported
- `OPTIONAL MATCH` - No equivalent (would need outer joins)
- `CREATE`, `DELETE`, `SET` - GFQL is read-only
- Multiple disconnected `MATCH` patterns - Use separate chains or joins

## Procedure and Function Mapping

GFQL Call operations provide functionality similar to Neo4j procedures (especially APOC), with additional GPU acceleration and visualization capabilities.

### Basic Procedure Calls

| Cypher | Python | Wire Protocol |
|--------|--------|---------------|
| `CALL algo.pageRank()` | `call('compute_cugraph', {'alg': 'pagerank'})` | `{"type": "ASTCall", "function": "compute_cugraph", "params": {"alg": "pagerank"}}` |
| `CALL apoc.algo.louvain()` | `call('compute_cugraph', {'alg': 'louvain'})` | `{"type": "ASTCall", "function": "compute_cugraph", "params": {"alg": "louvain"}}` |
| `CALL apoc.path.expand(n, '>KNOWS', null, 1, 3)` | `call('hop', {'hops': 3, 'edge_match': {'type': 'KNOWS'}})` | `{"type": "ASTCall", "function": "hop", "params": {"hops": 3, "edge_match": {"type": "KNOWS"}}}` |
| `CALL apoc.degree.in(n)` | `call('get_indegrees')` | `{"type": "ASTCall", "function": "get_indegrees", "params": {}}` |

### GPU vs CPU Decision Guide

Before choosing between `compute_cugraph` (GPU) and `compute_igraph` (CPU), consider:

**When to use GPU (`compute_cugraph`):**
- Large graphs (>100K edges)
- NVIDIA GPU available (CUDA-enabled)
- Batch processing multiple algorithms
- Real-time interactive analytics
- Algorithms: pagerank, louvain, betweenness_centrality, etc.

**When to use CPU (`compute_igraph`):**
- Smaller graphs (<100K edges)
- No GPU available
- Need algorithms not in cuGraph
- Development/testing environments
- Algorithms: all centrality measures, community detection, paths

**Performance Guidelines:**
- GPU can be 10-50x faster on large graphs
- CPU more efficient for graphs <10K edges
- GPU requires data transfer overhead
- CPU has more algorithm variety

### Algorithm Mapping

#### Comprehensive Algorithm Comparison

| APOC/algo.* | GFQL GPU (cuGraph) | GFQL CPU (igraph) | Notes |
|-------------|-------------------|-------------------|--------|
| `apoc.algo.pageRank` | `call('compute_cugraph', {'alg': 'pagerank'})` | `call('compute_igraph', {'alg': 'pagerank'})` | GPU 10-50x faster on large graphs |
| `apoc.algo.betweenness` | `call('compute_cugraph', {'alg': 'betweenness_centrality'})` | `call('compute_igraph', {'alg': 'betweenness'})` | GPU version handles directed graphs better |
| `apoc.algo.closeness` | Not available | `call('compute_igraph', {'alg': 'closeness'})` | CPU-only algorithm |
| `apoc.algo.louvain` | `call('compute_cugraph', {'alg': 'louvain'})` | `call('compute_igraph', {'alg': 'community_multilevel'})` | Different names, same algorithm |
| `algo.shortestPath` | `call('compute_cugraph', {'alg': 'sssp'})` | `call('compute_igraph', {'alg': 'shortest_paths'})` | GPU version is single-source only |
| `algo.unionFind` | `call('compute_cugraph', {'alg': 'connected_components'})` | `call('compute_igraph', {'alg': 'clusters'})` | GPU version faster for large graphs |
| `apoc.algo.eigenvector` | `call('compute_cugraph', {'alg': 'eigenvector_centrality'})` | `call('compute_igraph', {'alg': 'eigenvector_centrality'})` | Similar performance |
| `apoc.algo.katz` | `call('compute_cugraph', {'alg': 'katz_centrality'})` | Not available | GPU-only algorithm |
| `algo.degree` | Not needed - use `call('get_degrees')` | Not needed - use `call('get_degrees')` | Built-in GFQL operation |
| `apoc.algo.hits` | `call('compute_cugraph', {'alg': 'hits'})` | `call('compute_igraph', {'alg': 'hub_score'})` + `authority_score` | GPU computes both, CPU needs two calls |
| `apoc.algo.triangleCount` | `call('compute_cugraph', {'alg': 'triangle_count'})` | `call('compute_igraph', {'alg': 'transitivity_local_undirected'})` | Different output formats |
| `apoc.algo.kcore` | `call('compute_cugraph', {'alg': 'k_core'})` | `call('compute_igraph', {'alg': 'coreness'})` | Similar functionality |

#### Algorithm Availability Matrix

**GPU-Exclusive (cuGraph only):**
- `katz_centrality` - Katz centrality measure
- `bfs` - Breadth-first search from source
- `sssp` - Single-source shortest path
- `strongly_connected_components` - For directed graphs

**CPU-Exclusive (igraph only):**
- `closeness` - Closeness centrality
- `harmonic_centrality` - Harmonic centrality
- `constraint` - Burt's constraint
- `diversity` - Vertex diversity
- `maximal_cliques` - Find all maximal cliques
- `modularity` - Calculate modularity score
- Many statistical and layout algorithms

**Available in Both:**
- PageRank (different parameter names)
- Community detection (louvain/community_multilevel)
- Betweenness centrality
- Eigenvector centrality
- Connected components (connected_components/clusters)
- Degree calculations
- Triangle counting (different output formats)

#### Path Operations

| APOC/apoc.path.* | GFQL Call | Description |
|------------------|-----------|-------------|
| `apoc.path.expand` | `call('hop', {'hops': N})` | N-hop expansion |
| `apoc.path.expandConfig` | `call('hop', {'hops': N, 'edge_match': {...}})` | Filtered expansion |
| `apoc.path.spanningTree` | Use `hop` with `to_fixed_point=True` | Expand to fixed point |
| `apoc.path.subgraphNodes` | `call('hop', {'return_as_wave_front': False})` | All nodes in path |

#### Utility Operations

| APOC/apoc.* | GFQL Call | Description |
|-------------|-----------|-------------|
| `apoc.create.nodes` | `call('materialize_nodes')` | Create nodes from edges |
| `apoc.graph.fromData` | Direct DataFrame construction | Build from data |
| `apoc.degree.*` | `call('get_degrees')`, `call('get_indegrees')`, etc. | Degree calculations |
| `apoc.nodes.collapse` | `call('collapse', {'column': 'attr'})` | Merge nodes by attribute |

### Algorithm Examples: GPU vs CPU Comparison

#### Example 1: PageRank with Filtering

**Cypher with APOC:**
```cypher
MATCH (n:Person) WHERE n.age > 30
WITH collect(n) as nodes
CALL apoc.algo.pageRank(nodes) YIELD node, score
RETURN node.name, score
ORDER BY score DESC LIMIT 10
```

**GFQL GPU Version (for large graphs >100K edges):**
```python
# Use GPU acceleration for large-scale processing
result = g.gfql([
    n({'type': 'Person', 'age': gt(30)}),
    call('compute_cugraph', {
        'alg': 'pagerank',
        'out_col': 'pagerank_score',
        'params': {'alpha': 0.85, 'max_iter': 100}
    })
])
top_10 = result._nodes.nlargest(10, 'pagerank_score')[['name', 'pagerank_score']]
```

**GFQL CPU Version (for smaller graphs or no GPU):**
```python
# Use CPU for smaller graphs or when GPU unavailable
result = g.gfql([
    n({'type': 'Person', 'age': gt(30)}),
    call('compute_igraph', {
        'alg': 'pagerank',
        'out_col': 'pagerank_score',
        'params': {'damping': 0.85}  # Note: igraph uses 'damping' not 'alpha'
    })
])
top_10 = result._nodes.nlargest(10, 'pagerank_score')[['name', 'pagerank_score']]
```

**Wire Protocol (showing both):**
```json
// GPU Version
{
  "type": "Chain",
  "chain": [
    {
      "type": "Node",
      "filter_dict": {"type": "Person", "age": {"type": "GT", "val": 30}}
    },
    {
      "type": "ASTCall",
      "function": "compute_cugraph",
      "params": {
        "alg": "pagerank",
        "out_col": "pagerank_score",
        "params": {"alpha": 0.85}
      }
    }
  ]
}

// CPU Version
{
  "type": "Chain",
  "chain": [
    {
      "type": "Node",
      "filter_dict": {"type": "Person", "age": {"type": "GT", "val": 30}}
    },
    {
      "type": "ASTCall",
      "function": "compute_igraph",
      "params": {
        "alg": "pagerank",
        "out_col": "pagerank_score",
        "params": {"damping": 0.85}
      }
    }
  ]
}
```

#### Example 2: Community Detection

**GFQL GPU Version (Louvain):**
```python
# GPU version for large-scale community detection
g.gfql([
    call('compute_cugraph', {
        'alg': 'louvain',
        'out_col': 'community_id'
    })
])
```

**GFQL CPU Version (Multilevel):**
```python
# CPU version with more configuration options
g.gfql([
    call('compute_igraph', {
        'alg': 'community_multilevel',  # Same as Louvain
        'out_col': 'community_id',
        'params': {'weights': 'weight'}  # Can use edge weights
    })
])
```

**Performance Characteristics:**
- GPU: ~0.2s for 1M edges
- CPU: ~3s for 1M edges
- CPU: More configuration options
- GPU: Better for real-time analysis

### GFQL-Exclusive Features

These Call operations have no direct APOC equivalent:

#### Visual Encoding
```python
# Color nodes by community
call('encode_point_color', {
    'column': 'community',
    'palette': ['blue', 'red', 'green']
})

# Size nodes by importance
call('encode_point_size', {
    'column': 'pagerank_score',
    'as_continuous': True
})

# Set node icons by type
call('encode_point_icon', {
    'column': 'node_type',
    'categorical_mapping': {
        'person': 'user',
        'company': 'building'
    }
})
```

#### GPU-Accelerated Layouts
```python
# Force-directed layout
call('layout_cugraph', {
    'layout': 'force_atlas2',
    'params': {'iterations': 500}
})

# Hierarchical layout
call('layout_graphviz', {
    'prog': 'dot',
    'directed': True
})
```

### Performance Advantages

GFQL Call operations offer significant performance benefits:

1. **GPU Acceleration**: `compute_cugraph` methods run on NVIDIA GPUs
2. **Bulk Operations**: Process entire graphs vs node-by-node iteration
3. **DataFrame Integration**: Zero-copy transitions between graph and tabular operations
4. **Parallel Execution**: All operations vectorized for CPU/GPU parallelism

For complete Call operation reference, see the [Built-in Call Reference](../builtin_calls.rst).

## Best Practices

1. **Direct Translation First**: Try pure GFQL before adding DataFrame operations
2. **Use Named Patterns**: Label important results with `name=` for easy access
3. **Filter Early**: Apply selective node filters before traversing edges
4. **Type Consistency**: Ensure wire protocol types match expected column types
5. **Validate JSON**: Test wire protocol against schema before sending

## LLM Integration Guide

When building translators:

```
Given Cypher: {cypher_query}

Generate both:
1. Python: Human-readable GFQL code
2. Wire Protocol: JSON for API calls

Rules:
- (n:Label) → Python: n({"type": "Label"}) → JSON: {"type": "Node", "filter_dict": {"type": "Label"}}
- WHERE → Embed as predicates in both formats
- Aggregations → Note as requiring DataFrame post-processing
```

## See Also
- {ref}`gfql-spec-wire-protocol` - Full wire protocol specification
- {ref}`gfql-spec-language` - Language specification
- {ref}`gfql-spec-python-embedding` - Python implementation details