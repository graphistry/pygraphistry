(gfql-spec-cypher-mapping)=

# Cypher to GFQL Python & Wire Protocol Mapping

Translate existing Cypher workloads to GPU-accelerated GFQL with minimal code changes.

## Introduction

This specification shows how to translate Cypher queries to both GFQL Python code and :ref:`Wire Protocol <gfql-spec-wire-protocol>` JSON, enabling migration from Cypher-based systems, LLM pipelines (text → Cypher → GFQL), language-agnostic API integration, and secure query generation without code execution.

## What Maps 1-to-1

When translating from Cypher, you'll encounter three scenarios:

**1. Direct Translation** - Most pattern matching maps cleanly to pure GFQL
**2. Hybrid Approach** - Post-processing operations (RETURN clauses with aggregations) use df.groupby/agg
**3. GFQL Advantages** - Some capabilities go beyond what Cypher offers

### Direct Translations
- Graph patterns: `(a)-[r]->(b)` → chain operations
- Property filters: WHERE clauses embed into operations
- Path traversals: Variable-length paths use `hops` parameter
- Pattern composition: Multiple patterns become sequential operations

## When You Need DataFrames
- Aggregations: COUNT, SUM, AVG → pandas operations
- Projections: RETURN specific columns → DataFrame selection
- Sorting/limiting: ORDER BY, LIMIT → DataFrame methods
- Joins: Multiple disconnected patterns → pandas merge

## GFQL-Only Super-Powers
- **Edge properties**: Query edges as first-class entities
- **Dataframe-native**: Zero-cost transitions between graph and tabular operations
- **GPU acceleration**: Parallel execution on NVIDIA hardware
- **Heterogeneous graphs**: No schema constraints on types or properties
- **Integrated visualization**: Layouts like `group_in_a_box_layout` for community visualization
- **Algorithm chaining**: Combine community detection with layout algorithms

## Quick Example

**Cypher:**
```cypher
MATCH (p:Person)-[r:FOLLOWS]->(q:Person) 
WHERE p.age > 30
```

**Python:**
```python
g.gfql([
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

## Translation Tables

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
| `(n1)-[*2]->(n2)` | `e_forward(min_hops=2, max_hops=2)` | `{"type": "Edge", "direction": "forward", "min_hops": 2, "max_hops": 2}` |
| `(n1)-[*1..3]->(n2)` | `e_forward(min_hops=1, max_hops=3)` | `{"type": "Edge", "direction": "forward", "min_hops": 1, "max_hops": 3}` |
| `(n1)-[*3..3]->(n2)` | `e_forward(min_hops=3, max_hops=3)` | `{"type": "Edge", "direction": "forward", "min_hops": 3, "max_hops": 3}` |
| `(n1)-[*2..4]->(n2)` but only show hops 3..4 | `e_forward(min_hops=2, max_hops=4, output_min_hops=3, label_edge_hops="edge_hop")` | `{"type": "Edge", "direction": "forward", "min_hops": 2, "max_hops": 4, "output_min_hops": 3, "label_edge_hops": "edge_hop"}` |
| `(n1)-[*]->(n2)` | `e_forward(to_fixed_point=True)` | `{"type": "Edge", "direction": "forward", "to_fixed_point": true}` |
| `-[r:BOUGHT {amount: gt(100)}]->` | `e_forward({"type": "BOUGHT", "amount": gt(100)}, name="r")` | `{"type": "Edge", "direction": "forward", "edge_match": {"type": "BOUGHT", "amount": {"type": "GT", "val": 100}}, "name": "r"}` |

### Predicates

| Cypher | Python | Wire Protocol |
|--------|--------|---------------|
| `n.status = 'active'` | `"active"` | `"active"` | # literal
| `n.age > 30` | `gt(30)` | `{"type": "GT", "val": 30}` |
| `n.age >= 50` | `ge(50)` | `{"type": "GE", "val": 50}` |
| `n.age < 100` | `lt(100)` | `{"type": "LT", "val": 100}` |
| `n.age <= 50` | `le(50)` | `{"type": "LE", "val": 50}` |
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
g.gfql([
    n({"type": "User", "name": "Alice"}),
    e_forward({"type": "FRIEND"}, min_hops=2, max_hops=2),
    n({"type": "User", "active": True}, name="fof")
])
```

**Wire Protocol:**
```json
{"type": "Chain", "chain": [
  {"type": "Node", "filter_dict": {"type": "User", "name": "Alice"}},
  {"type": "Edge", "direction": "forward", "edge_match": {"type": "FRIEND"}, "min_hops": 2, "max_hops": 2},
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
g.gfql([
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
result = g.gfql([
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
- `WITH` clauses - Requires intermediate variables
- Multiple `MATCH` patterns - Use separate chains or joins

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
