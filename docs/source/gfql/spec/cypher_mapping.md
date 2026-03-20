(gfql-spec-cypher-mapping)=

# Cypher to GFQL Python & Wire Protocol Mapping

GFQL supports Cypher syntax out of the box for a bounded read-only surface on
bound graphs, while executing through GFQL's columnar engine with optional GPU
acceleration. This page explains how to translate familiar Cypher patterns into
native GFQL Python and wire protocol forms when you want more explicit control.

## Introduction

Cypher is a graph query language popularized by Neo4j and related tools. In
PyGraphistry, you can often start with a Cypher string directly through
`g.gfql("MATCH ...")`, then translate that query into native GFQL when you
want direct operator control, {ref}`Wire Protocol <gfql-spec-wire-protocol>`
JSON generation, migration from Cypher-centric systems, language-agnostic API
integration, or secure query generation without code execution.

## Direct ``g.gfql("MATCH ...")`` Note

If you want to **run** a supported Cypher string through ``g.gfql("MATCH ...")``
on a bound graph, use
`g.gfql("MATCH ...")` (or `g.gfql("...", language="cypher")`) and start with
{doc}`/gfql/cypher`. This page stays translation-first: it explains how to
express Cypher semantics in native GFQL operators and wire protocol, not the
primary quickstart for direct Cypher syntax execution.

## What Maps 1-to-1

When translating from Cypher, you'll encounter three scenarios:

1. **Direct Translation**: Most pattern matching maps cleanly to pure GFQL.
2. **Row-Pipeline Translation**: `RETURN/WITH/ORDER BY/SKIP/LIMIT/DISTINCT/GROUP BY` map to GFQL row operators.
3. **GFQL Advantages**: Some capabilities go beyond what Cypher offers.

### Direct Translations
- Graph patterns: `(a)-[r]->(b)` → chain operations
- Property filters: WHERE clauses embed into operations
- Path traversals: direct `g.gfql("MATCH ...")` supports endpoint-only single
  variable-length relationship forms such as `[*2]`, `[*1..3]`, and `[*]`.
  Native GFQL still gives you the full explicit hop surface, including output
  slicing, intermediate-hop aliasing, and rewrites for currently unsupported
  direct-Cypher multihop shapes.
- Pattern composition: Multiple patterns become sequential operations
- Same-path constraints: `WHERE` across steps → `g.gfql([...], where=[...])`

## Row-Pipeline Translation (`MATCH ... RETURN`)
- Row source selection: `rows(table=..., source=...)`
- Row filtering: `where_rows(filter_dict=..., expr=...)`
- Projection: `return_(...)` / `with_(...)` / `select(...)`
- Sorting/paging: `order_by(...)`, `skip(...)`, `limit(...)`
- Deduplication: `distinct()`
- Aggregation: `group_by(keys=[...], aggregations=[...])`

These row-pipeline operators are call steps inside the same chain list passed to
`g.gfql([...])` (or to `Chain([...])`), not top-level `g.gfql()` keyword args:

```python
from graphistry import n, e_forward
from graphistry.compute import rows, where_rows, return_, order_by, limit

g.gfql([
    n({"type": "Person"}, name="p"),
    e_forward({"type": "FOLLOWS"}),
    n({"type": "Person"}, name="q"),
    rows(table="nodes", source="q"),
    where_rows(expr="score >= 50"),
    return_([("id", "id"), ("name", "name"), ("score", "score")]),
    order_by([("score", "desc")]),
    limit(25),
])
```

```python
from graphistry.compute.chain import Chain

query = Chain([
    n({"type": "Person"}, name="p"),
    e_forward({"type": "FOLLOWS"}),
    n({"type": "Person"}, name="q"),
    rows(table="nodes", source="q"),
    where_rows(expr="score >= 50"),
    return_(["id", "name", "score"]),
])
g.gfql(query)
```

Projection sequencing and placement rules:

- Multiple `return_(...)` / `with_(...)` / `select(...)` steps are valid and
  execute in list order; each step projects from the current row table produced
  by previous steps.
- Interior mixing is invalid: do not place call steps between traversal steps
  (`n()/e_*()`), e.g. `[n(...), return_(...), e_forward(...)]`.
  Keep call steps in boundary prefix/suffix segments around traversal blocks.

## When You Still Need DataFrames
- Translation targets outside the current pure GFQL operator surface, such as
  some `OPTIONAL MATCH` null-extension flows
- Arbitrary joins across disconnected intermediate result sets
- Custom functions outside the current row-expression subset

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
| `(n) WHERE n.a > 10` | `n({"a": gt(10)})` | `{"type": "Node", "filter_dict": {"a": {"type": "GT", "val": 10}}}` |
| `(n:Person) WHERE n.age > 30` | `n({"type": "Person", "age": gt(30)})` | `{"type": "Node", "filter_dict": {"type": "Person", "age": {"type": "GT", "val": 30}}}` |

### Row-Pipeline Translation Tables

Use these as chain steps inside `g.gfql([...])` / `Chain([...])`.

| Cypher | Python chain step | Wire Protocol call (compact) |
|--------|-------------------|-------------------------------|
| `RETURN q.id, q.name` | `return_(["id", "name"])` | `{"type":"Call","function":"select","params":{"items":[["id","id"],["name","name"]]}}` |
| `RETURN q.id AS person_id` | `return_([("person_id", "id")])` | `{"type":"Call","function":"select","params":{"items":[["person_id","id"]]}}` |
| `WITH q.id AS id, q.score AS s` | `with_([("id", "id"), ("s", "score")])` | `{"type":"Call","function":"with_","params":{"items":[["id","id"],["s","score"]]}}` |
| `WHERE <row expr>` after `MATCH` | `where_rows(expr="score >= 50")` | `{"type":"Call","function":"where_rows","params":{"expr":"score >= 50"}}` |
| `WHERE` with predicate helpers in row stage | `where_rows(filter_dict={"created_at": gt(ts)})` | `{"type":"Call","function":"where_rows","params":{"filter_dict":{"created_at":{"type":"GT","val":...}}}}` |
| `ORDER BY score DESC, name ASC` | `order_by([("score", "desc"), ("name", "asc")])` | `{"type":"Call","function":"order_by","params":{"keys":[["score","desc"],["name","asc"]]}}` |
| `SKIP 20` | `skip(20)` | `{"type":"Call","function":"skip","params":{"value":20}}` |
| `LIMIT 10` | `limit(10)` | `{"type":"Call","function":"limit","params":{"value":10}}` |
| `RETURN DISTINCT ...` | `distinct()` | `{"type":"Call","function":"distinct","params":{}}` |
| `GROUP BY category` with `count(*)` | `group_by(keys=["category"], aggregations=[("cnt","count")])` | `{"type":"Call","function":"group_by","params":{"keys":["category"],"aggregations":[["cnt","count"]]}}` |
| Scope rows to alias `q` | `rows(table="nodes", source="q")` | `{"type":"Call","function":"rows","params":{"table":"nodes","source":"q"}}` |

### Same-Path WHERE Predicates

Use `g.gfql([...], where=[...])` when the predicate compares multiple steps.

**Cypher:**
```cypher
MATCH (n1)-[e1]->(n2)-[e2]->(n3)
WHERE n1.a > n2.b AND e1.x = e2.y
```

**Python:**
<!-- doc-test: skip -->
```python
from graphistry import n, e_forward, col, compare

g.gfql(
    [n(name="n1"), e_forward(name="e1"), n(name="n2"), e_forward(name="e2"), n(name="n3")],
    where=[
        compare(col("n1", "a"), ">", col("n2", "b")),
        compare(col("e1", "x"), "==", col("e2", "y")),
    ],
)
```

**Wire Protocol:**
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "name": "n1"},
    {"type": "Edge", "direction": "forward", "name": "e1"},
    {"type": "Node", "name": "n2"},
    {"type": "Edge", "direction": "forward", "name": "e2"},
    {"type": "Node", "name": "n3"}
  ],
  "where": [
    {"gt": {"left": "n1.a", "right": "n2.b"}},
    {"eq": {"left": "e1.x", "right": "e2.y"}}
  ]
}
```

### `MATCH ... RETURN` Row Pipeline

Use GFQL call steps after the pattern match to encode Cypher `RETURN` behavior.

**Cypher:**
```cypher
MATCH (p:Person)-[:FOLLOWS]->(q:Person)
WHERE q.score >= 50
RETURN q.id AS id, q.name AS name, q.score AS score
ORDER BY score DESC, name ASC
LIMIT 25
```

**Python:**
```python
from graphistry import n, e_forward, gt
from graphistry.compute import rows, where_rows, return_, order_by, limit

g.gfql([
    n({"type": "Person"}),
    e_forward({"type": "FOLLOWS"}),
    n({"type": "Person", "score": gt(0)}, name="q"),
    rows(table="nodes", source="q"),
    where_rows(expr="score >= 50"),
    return_(["id", "name", "score"]),
    order_by([("score", "desc"), ("name", "asc")]),
    limit(25),
])
```

**Wire Protocol:**
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "filter_dict": {"type": "Person"}},
    {"type": "Edge", "direction": "forward", "edge_match": {"type": "FOLLOWS"}},
    {"type": "Node", "filter_dict": {"type": "Person", "score": {"type": "GT", "val": 0}}, "name": "q"},
    {"type": "Call", "function": "rows", "params": {"table": "nodes", "source": "q"}},
    {"type": "Call", "function": "where_rows", "params": {"expr": "score >= 50"}},
    {"type": "Call", "function": "select", "params": {"items": [["id", "id"], ["name", "name"], ["score", "score"]]}},
    {"type": "Call", "function": "order_by", "params": {"keys": [["score", "desc"], ["name", "asc"]]}},
    {"type": "Call", "function": "limit", "params": {"value": 25}}
  ]
}
```

### Edge Patterns

Rows using `[*...]` below show the native GFQL rewrite for the same traversal
intent. Direct `g.gfql("MATCH ...")` now accepts these endpoint-only
single-variable-length relationship forms, while native GFQL remains the more
explicit option when you need intermediate-hop control or unsupported mixed
pattern shapes.

| Cypher / intent | Python | Wire Protocol (compact) |
|-----------------|--------|-------------------------|
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

When you need constraints on intermediate hops, path/list-carrier semantics, or
mixed connected patterns beyond the current direct-Cypher subset, use repeated
single-hop GFQL steps with aliases instead of collapsing the traversal into one
multihop edge operator.

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
| `n.val IS NULL` | `isnull()` | `{"type": "IsNull"}` |
| `n.val IS NOT NULL` | `notnull()` | `{"type": "NotNull"}` |

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

### Same-Path Constraint

**Cypher:**
```cypher
MATCH (a:Account)-[:TRANSFER]->(c:User)
WHERE a.owner_id = c.owner_id
```

**Python:**
```python
from graphistry import n, e_forward, col, compare

g.gfql(
    [n({"type": "Account"}, name="a"), e_forward(), n({"type": "User"}, name="c")],
    where=[compare(col("a", "owner_id"), "==", col("c", "owner_id"))],
)
```

**Wire Protocol:**
```json
{"type": "Chain", "chain": [
  {"type": "Node", "filter_dict": {"type": "Account"}, "name": "a"},
  {"type": "Edge", "direction": "forward"},
  {"type": "Node", "filter_dict": {"type": "User"}, "name": "c"}
], "where": [{"eq": {"left": "a.owner_id", "right": "c.owner_id"}}]}
```

### Fraud Detection

**Cypher:**
```cypher
MATCH (a:Account)-[t:TRANSFER]->(b:Account)
WHERE t.amount > 10000 AND t.date > date('2024-01-01')
```

**Python:**
<!-- doc-test: skip -->
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
RETURN t.category AS category, count(*) as cnt, sum(t.amount) as total
ORDER BY total DESC
LIMIT 10
```

**Python:**
<!-- doc-test: skip -->
```python
from datetime import date
from graphistry import n, e_forward, gt
from graphistry.compute import rows, where_rows, group_by, return_, order_by, limit

analysis = g.gfql([
    n({"type": "User"}),
    e_forward({"type": "TRANSACTION", "date": gt(date(2024, 1, 1))}, name="t"),
    rows(table="edges", source="t"),
    where_rows(expr="amount IS NOT NULL"),
    group_by(
        keys=["category"],
        aggregations=[
            ("cnt", "count", "amount"),
            ("total", "sum", "amount"),
        ],
    ),
    return_(["category", "cnt", "total"]),
    order_by([("total", "desc")]),
    limit(10),
])
```

**Note:** If the aggregation/function you need is outside the supported
`group_by` subset, fall back to dataframe post-processing.

## Row-Pipeline Operations Mapping

| Cypher Feature | GFQL Python Row Operation | Notes |
|----------------|---------------------------|--------|
| `RETURN a, b, c` | `return_(["a", "b", "c"])` | String item shorthand maps `a -> ("a", "a")` |
| `WITH a, b` | `with_(["a", "b"])` | Same projection semantics as `return_` |
| `RETURN DISTINCT` | `distinct()` | Deduplicate active row table |
| `ORDER BY x DESC` | `order_by([("x", "desc")])` | Multi-key sorting supported |
| `SKIP 20` | `skip(20)` | Row offset |
| `LIMIT 10` | `limit(10)` | Row cap |
| `WHERE <row expr>` | `where_rows(expr="...")` | Scalar expression subset |
| `count(*)` | `group_by(keys=[...], aggregations=[("cnt", "count")])` | Grouped count |
| `sum(n.val)` | `group_by(..., aggregations=[("total", "sum", "val")])` | Grouped sum |
| `collect(n.x)` | `group_by(..., aggregations=[("xs", "collect", "x")])` | Nulls excluded from collection |
| Named patterns | `rows(source="alias")` | Scope row table to a named match alias |

## Key Differences

| Feature | Python | Wire Protocol |
|---------|--------|---------------|
| **Temporal values** | `pd.Timestamp()`, `date()` | `{"type": "date", "value": "..."}` |
| **Direct equality** | `"active"` | `"active"` (same) |
| **Comparisons** | `gt(30)` | `{"type": "GT", "val": 30}` |
| **Collections** | `is_in([...])` | `{"type": "IsIn", "options": [...]}` |

## GFQL Extension: Graph Constructors (`GRAPH { }`)

Standard Cypher and GQL's first edition (ISO/IEC 39075:2024) have no way to
return a graph from a query — every result is a flat table of binding rows.
GFQL extends Cypher with graph constructors that keep results in graph state,
enabling composable graph-in / graph-out pipelines.

### Syntax and Desugaring

Graph constructors compile down to the same Chain/Call wire-protocol primitives
as regular queries — **no new wire types are needed**.

| Cypher (GFQL extension) | Desugars to |
|--------------------------|-------------|
| `GRAPH { MATCH (a)-[r]->(b) WHERE a.x > 10 }` | `Chain([n("a"), e_forward("r"), n("b")], where=[...])` — executed in graph state |
| `GRAPH { CALL graphistry.degree.write() }` | `Call("graphistry.degree.write")` — graph-preserving procedure |
| `GRAPH g = GRAPH { ... }` | Named binding (like `Let`) whose value is the Chain/Call result |
| `USE g MATCH ... RETURN ...` | Execute the following query against binding `g`'s graph |

### Examples

**Standalone graph constructor (graph state out):**

```python
# Cypher extension
g2 = g.gfql("GRAPH { MATCH (a)-[r]->(b) WHERE a.score > 10 }")

# Equivalent native GFQL (inherently graph-returning)
g2 = g.gfql([n({"score": ge(10)}), e_forward(), n()])
```

**Multi-stage pipeline in one expression:**

```python
result = g.gfql(
    "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) WHERE a.score > 10 } "
    "GRAPH g2 = GRAPH { USE g1 CALL graphistry.degree.write() } "
    "USE g2 MATCH (n) RETURN n.id, n.degree ORDER BY n.degree DESC"
)
```

**Wire protocol** — the pipeline above compiles locally to a sequence of
Chain + Call executions against resolved graph bindings. The server sees
ordinary Chain/Call messages, not graph-constructor-specific wire types.

### Design Notes

- `GRAPH { }` is a **closed scope** — pattern variables inside do not leak
- `USE` is **lexically scoped** — bindings must be defined before USE
- Only graph-preserving `CALL ... .write()` procedures are allowed inside
  constructors (row-returning CALL is rejected)
- `cypher_to_gfql()` rejects multi-graph pipelines (they can't be represented
  as a single Chain); use `g.gfql("...")` for direct execution instead

## Not Supported
- `CREATE`, `DELETE`, `SET`: GFQL is read-only.
- `OPTIONAL MATCH`: direct `g.gfql("MATCH ...")` execution supports a bounded subset,
  but pure GFQL translation still has no single general operator for full
  outer-join/null-extension semantics.
- Full Cypher expression/function surface in row expressions: current vectorized subset only.
- Multiple disconnected `MATCH` patterns in one query: use separate GFQL chains and explicit dataframe joins.

Practical fallback: keep pattern traversal and row-pipeline stages in GFQL, then
apply final custom dataframe logic in pandas/cuDF when needed.

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
- Cross-step WHERE → `g.gfql([...], where=[compare(col(...), op, col(...))])`
- RETURN/WITH/ORDER BY/SKIP/LIMIT/DISTINCT/GROUP BY → row-pipeline call steps (`rows`, `where_rows`, `return_`, `order_by`, `skip`, `limit`, `distinct`, `group_by`)
- Unsupported expressions/functions → explicitly mark as unsupported instead of silently rewriting
```

## See Also
- {ref}`gfql-spec-wire-protocol` - Full wire protocol specification
- {ref}`gfql-spec-language` - Language specification
- {ref}`gfql-spec-python-embedding` - Python implementation details
