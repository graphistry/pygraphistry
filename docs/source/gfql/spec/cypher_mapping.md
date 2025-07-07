(gfql-spec-cypher-mapping)=

# Cypher to GFQL Mapping Specification

## Introduction

This specification defines the mapping between Cypher query language and GFQL for the subset of Cypher patterns that GFQL supports. This enables:
- Leveraging existing Cypher knowledge for GFQL queries
- Two-stage LLM (Large Language Model) synthesis (Text → Cypher → GFQL)
- Migration from Cypher-based systems
- Cross-platform query portability

### Understanding the Relationship

GFQL and Cypher serve different architectural tiers in graph computing:

**Cypher** is a declarative graph query language designed for graph databases. It operates at the storage tier, focusing on:
- Pattern matching across persistent graph stores
- Transactional operations (CREATE, UPDATE, DELETE)
- Complex aggregations and transformations
- Schema constraints and indexes

**GFQL** is a dataframe-native graph query language designed for the compute tier. It operates directly on in-memory dataframes, focusing on:
- High-performance traversals on dataframes (pandas, cuDF, Arrow)
- GPU acceleration for massive parallelism
- Integration with Python data science ecosystem
- Real-time analytics without database overhead

### Translation Targets

When translating Cypher to GFQL, there are two primary targets:

#### 1. Pure GFQL Chains
Standalone GFQL queries that operate entirely within the graph traversal paradigm:
```python
# Pure GFQL - returns filtered subgraph
g.chain([n({"type": "person"}), e_forward(), n()])
```

#### 2. GFQL + PyGraphistry/Pandas Hybrid
GFQL for graph operations combined with dataframe operations for aggregations and transformations:
```python
# Hybrid - GFQL traversal + pandas aggregation
result = g.chain([n({"type": "person"}), e_forward(), n()])
counts = result._nodes.groupby('type').size()
```

This specification focuses on pure GFQL mappings, with notes on when hybrid approaches are needed for full Cypher semantics.

### Design Principles
- **Semantic Preservation**: Maintain query intent where possible
- **Pattern-Based**: Focus on graph patterns, not imperative operations
- **Read-Only**: Support only query operations (no mutations)
- **Explicit Limitations**: Clear documentation of unsupported features
- **Performance First**: Leverage GFQL's compute-tier advantages

## Supported Cypher Subset

### Graph Patterns
- Node patterns: `(n)`, `(n:Label)`, `(n {prop: value})`
- Edge patterns: `-[r]->`, `<-[r]-`, `-[r]-`
- Path patterns: `(a)-[r]->(b)`
- Variable-length paths: `-[*N]-`, `-[*..N]-`, `-[*]-`

### Filtering
- Property filters in patterns
- WHERE clauses with simple predicates
- Comparison operators: `=`, `<>`, `>`, `<`, `>=`, `<=`
- IN operator for membership
- String operations: STARTS WITH, ENDS WITH, CONTAINS

### Limitations
- Read-only (no CREATE, DELETE, SET)
- No aggregations in MATCH
- No WITH clauses
- No OPTIONAL MATCH
- No RETURN transformations

## Mapping Rules

### Core Translation Rules

1. **MATCH Clause → chain()**
   ```
   MATCH pattern → g.chain([...operations...])
   ```

2. **Node Patterns → n()**
   ```
   (n) → n()
   (n:Label) → n({"label": "Label"})
   (n {prop: val}) → n({"prop": val})
   ```

3. **Edge Patterns → Edge Operations**
   ```
   -[r]-> → e_forward()
   <-[r]- → e_reverse()
   -[r]- → e() or e_undirected()
   ```

4. **WHERE Clause → Embedded Filters**
   ```
   WHERE n.prop = val → embed in n({"prop": val})
   WHERE r.prop > val → embed in e_*(**{"prop": gt(val)}**)
   ```

5. **Path Length → hops Parameter**
   ```
   -[*2]-> → e_forward(hops=2)
   -[*]-> → e_forward(to_fixed_point=True)
   ```

## Pattern Translations

### Basic Node Patterns

| Cypher | GFQL |
|--------|------|
| `(n)` | `n()` |
| `(n:Person)` | `n({"type": "Person"})` or `n({"label": "Person"})` |
| `(n {name: 'Alice'})` | `n({"name": "Alice"})` |
| `(n:Person {age: 30})` | `n({"type": "Person", "age": 30})` |

### Edge Patterns

| Cypher | GFQL |
|--------|------|
| `-[r]->` | `e_forward()` |
| `<-[r]-` | `e_reverse()` |
| `-[r]-` | `e()` |
| `-[r:KNOWS]->` | `e_forward({"type": "KNOWS"})` |
| `-[r {since: 2020}]->` | `e_forward({"since": 2020})` |

### Path Patterns

| Cypher | GFQL |
|--------|------|
| `(a)-[]->(b)` | `chain([n(), e_forward(), n()])` |
| `(a)-[*2]->(b)` | `chain([n(), e_forward(hops=2), n()])` |
| `(a)-[*..3]->(b)` | `chain([n(), e_forward(hops=3), n()])` |
| `(a)-[*]->(b)` | `chain([n(), e_forward(to_fixed_point=True), n()])` |

### Complex Patterns

**Cypher**:
```cypher
MATCH (p:Person {name: 'Alice'})-[:KNOWS*2]->(friend:Person)
WHERE friend.age > 25
```

**GFQL**:
```python
g.chain([
    n({"type": "Person", "name": "Alice"}),
    e_forward({"type": "KNOWS"}, hops=2),
    n({"type": "Person", "age": gt(25)})
])
```

## Predicate Mappings

### Comparison Operators

| Cypher | GFQL |
|--------|------|
| `n.prop = value` | `{"prop": value}` |
| `n.prop > value` | `{"prop": gt(value)}` |
| `n.prop < value` | `{"prop": lt(value)}` |
| `n.prop >= value` | `{"prop": ge(value)}` |
| `n.prop <= value` | `{"prop": le(value)}` |
| `n.prop <> value` | `{"prop": ne(value)}` |

### String Operators

| Cypher | GFQL |
|--------|------|
| `n.name STARTS WITH 'A'` | `{"name": startswith("A")}` |
| `n.name ENDS WITH 'z'` | `{"name": endswith("z")}` |
| `n.name CONTAINS 'bob'` | `{"name": contains("bob")}` |

### Collection Operators

| Cypher | GFQL |
|--------|------|
| `n.type IN ['A', 'B']` | `{"type": is_in(["A", "B"])}` |
| `n.val IN range(1, 10)` | `{"val": is_in(list(range(1, 10)))}` |

### Temporal Comparisons

| Cypher | GFQL |
|--------|------|
| `n.date > date('2024-01-01')` | `{"date": gt(date(2024, 1, 1))}` |
| `n.time < time('12:00:00')` | `{"time": lt(time(12, 0, 0))}` |
| `n.timestamp > datetime()` | `{"timestamp": gt(pd.Timestamp.now())}` |

## Unsupported Features

### Cypher Features Without GFQL Equivalent

1. **OPTIONAL MATCH**
   ```cypher
   OPTIONAL MATCH (n)-[r]->(m)  -- No GFQL equivalent
   ```

2. **WITH Clauses**
   ```cypher
   WITH n, count(*) as cnt  -- Use pandas post-processing
   ```

3. **Aggregations**
   ```cypher
   RETURN n, count(r)  -- Use pandas groupby after
   ```

4. **CREATE/DELETE/SET**
   ```cypher
   CREATE (n:Person)  -- GFQL is read-only
   ```

5. **Complex WHERE**
   ```cypher
   WHERE NOT exists(n.prop)  -- Limited support
   WHERE n.prop =~ 'regex'   -- Use match() predicate
   ```

### Workarounds

| Unsupported Feature | GFQL Alternative |
|---------------------|------------------|
| `ORDER BY` | Use pandas: `result._nodes.sort_values()` |
| `LIMIT` | Use pandas: `result._nodes.head(n)` |
| `DISTINCT` | Use pandas: `result._nodes.drop_duplicates()` |
| `count()` | Use pandas: `len(result._nodes)` |
| `collect()` | Use pandas: `result._nodes.groupby()` |

## Translation Examples

### Example 1: Simple Friend Query

**Natural Language**: "Find Alice's friends"

**Cypher**:
```cypher
MATCH (alice:Person {name: 'Alice'})-[:FRIEND]->(friend:Person)
RETURN friend
```

**GFQL**:
```python
g.chain([
    n({"type": "Person", "name": "Alice"}),
    e_forward({"type": "FRIEND"}),
    n({"type": "Person"})
])._nodes
```

### Example 2: Multi-hop with Filtering

**Natural Language**: "Find friends of friends who are developers"

**Cypher**:
```cypher
MATCH (p:Person {id: 123})-[:FRIEND*2]->(fof:Person)
WHERE fof.occupation = 'Developer'
RETURN fof
```

**GFQL**:
```python
g.chain([
    n({"type": "Person", "id": 123}),
    e_forward({"type": "FRIEND"}, hops=2),
    n({"type": "Person", "occupation": "Developer"})
])._nodes
```

### Example 3: Temporal Query

**Natural Language**: "Find recent transactions over $1000"

**Cypher**:
```cypher
MATCH (a:Account)-[t:TRANSACTION]->(b:Account)
WHERE t.amount > 1000 
  AND t.timestamp > datetime() - duration('P7D')
RETURN a, t, b
```

**GFQL**:
```python
week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
g.chain([
    n({"type": "Account"}),
    e_forward({
        "type": "TRANSACTION",
        "amount": gt(1000),
        "timestamp": gt(week_ago)
    }),
    n({"type": "Account"})
])
```

### Example 4: Bidirectional Search

**Natural Language**: "Find all connections between Alice and Bob"

**Cypher**:
```cypher
MATCH (alice:Person {name: 'Alice'})-[*]-(bob:Person {name: 'Bob'})
RETURN alice, bob
```

**GFQL**:
```python
g.chain([
    n({"type": "Person", "name": "Alice"}),
    e(to_fixed_point=True),
    n({"type": "Person", "name": "Bob"})
])
```

### Example 5: Complex Business Query

**Natural Language**: "Find high-value customers connected to fraudulent accounts"

**Cypher**:
```cypher
MATCH (c:Customer)-[:HAS_ACCOUNT]->(a1:Account)-[:TRANSFER*..3]->(a2:Account)<-[:HAS_ACCOUNT]-(f:Customer)
WHERE c.tier = 'Gold' 
  AND f.status = 'Fraudulent'
  AND a1.balance > 10000
RETURN c, a1, a2, f
```

**GFQL**:
```python
g.chain([
    n({"type": "Customer", "tier": "Gold"}),
    e_forward({"type": "HAS_ACCOUNT"}),
    n({"type": "Account", "balance": gt(10000)}),
    e_forward({"type": "TRANSFER"}, hops=3),
    n({"type": "Account"}),
    e_reverse({"type": "HAS_ACCOUNT"}),
    n({"type": "Customer", "status": "Fraudulent"})
])
```

## Best Practices

### For LLM-Based Translation

1. **Start Simple**: Begin with basic patterns before complex queries
2. **Explicit Types**: Always specify node/edge types when known
3. **Embed Filters**: Move WHERE conditions into matchers
4. **Handle Lists**: Convert Cypher lists to Python lists
5. **Post-Process**: Use pandas for sorting, limiting, aggregating

### Common Patterns

1. **Node Type Mapping**:
   - Cypher labels → GFQL type or label property
   - Choose consistent property name across queries

2. **Edge Type Mapping**:
   - Cypher relationship types → GFQL type property
   - Maintain consistent naming

3. **Variable-Length Paths**:
   - Bounded: Use `hops=N`
   - Unbounded: Use `to_fixed_point=True`
   - Upper bound only: Use `hops=N` as maximum

4. **Property Access**:
   - Direct in patterns when possible
   - Query strings for complex expressions

### Error Handling

Common translation errors and fixes:

1. **OPTIONAL MATCH**
   - Error: No direct translation
   - Fix: Split into separate queries or handle nulls in post-processing

2. **Aggregations in MATCH**
   - Error: Not supported in pattern
   - Fix: Move to pandas operations after query

3. **Complex WHERE**
   - Error: Boolean logic not directly supported
   - Fix: Use query strings or multiple queries

4. **Path Variables**
   - Error: Full path not captured
   - Fix: Use named operations to track path components

## Integration Guidelines

### For Tool Builders

1. **Parser Requirements**:
   - Cypher AST parser for pattern extraction
   - Pattern matcher for translation rules
   - Error handler for unsupported features

2. **Translation Pipeline**:
   ```
   Cypher String → Parse AST → Match Patterns → Generate GFQL → Validate
   ```

3. **Validation Steps**:
   - Check for unsupported Cypher features
   - Verify property names against schema
   - Validate predicate types

### For LLM Integration

1. **Prompt Structure**:
   ```
   Given Cypher: {cypher_query}
   
   Translate to GFQL using these rules:
   - (n) → n()
   - -[r]-> → e_forward()
   - WHERE → embedded filters
   
   Handle unsupported features by noting them.
   ```

2. **Few-Shot Examples**:
   - Include 3-5 examples per pattern type
   - Show both successful and error cases
   - Demonstrate workarounds

3. **Validation Prompts**:
   ```
   Validate this GFQL translation:
   Original Cypher: {cypher}
   Generated GFQL: {gfql}
   Schema: {node_columns}, {edge_columns}
   ```

## See Also

- {ref}`gfql-spec-language` - GFQL language specification
- {ref}`gfql-spec-wire-protocol` - Wire protocol format
- {ref}`gfql-spec-synthesis-examples` - More translation examples