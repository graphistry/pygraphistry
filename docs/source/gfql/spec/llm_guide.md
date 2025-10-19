(gfql-spec-llm-guide)=

# GFQL JSON Generation Guide for LLMs

## What is GFQL?

**GFQL (GraphFrame Query Language)** is a Cypher-like JSON AST for graph queries with three purposes:

1. **Graph Search**: Pattern matching with node/edge chains (filtering, traversal, etc.)
2. **Graph Algorithms**: Examples: PageRank, Louvain, UMAP, hypergraph, and more
3. **Visualization**: Encodings for color, icon, size, and more

---

## Table of Contents

**Quick Start:**
- [Quick Example](#quick-example-fraud-detection) - Multi-step fraud analysis
- [Core Types](#core-types) - Chain, Node, Edge, Call, Let, ChainRef
- [Predicates](#predicates) - Filters and comparisons

**Common Patterns:**
- [Simple Search](#simple-search) - Filter and traverse
- [Filter After Enrich](#filter-after-enrichment) - Compute then filter
- [Graph Algorithms](#graph-algorithms) - PageRank, Louvain, UMAP, Hypergraph
- [Visualization](#visualization) - Colors, icons, sizes
- [Multi-Step (Let/Ref)](#multi-step-letref) - DAG composition

**Domain Examples:**
- [Cyber](#cyber-security) - Lateral movement
- [Fraud](#fraud-detection) - Transaction chains
- [Supply Chain](#supply-chain) - Tracing
- [Social](#social-media) - Influence

**Reference:**
- [Call Functions](#call-functions) - All available functions
- [Generation Rules](#generation-rules) - Best practices
- [Common Mistakes](#common-mistakes) - Errors to avoid

---

## Quick Example: Fraud Detection

**Dense:** `let({'suspicious': n({'risk_score': gt(80)}), 'flows': ref('suspicious', [e_forward(hops=3), n()]), 'ranked': ref('flows', [call('compute_cugraph', {'alg': 'pagerank'})]), 'viz': ref('ranked', [call('encode_point_color', {...}), call('encode_point_icon', {...})])})`

**JSON:**
```json
{
  "type": "Let",
  "bindings": {
    "suspicious": {
      "type": "Chain",
      "chain": [{"type": "Node", "filter_dict": {"risk_score": {"type": "GT", "val": 80}}}]
    },
    "flows": {
      "type": "ChainRef",
      "ref": "suspicious",
      "chain": [
        {"type": "Edge", "direction": "forward", "hops": 3, "to_fixed_point": false,
         "edge_match": {"amount": {"type": "GT", "val": 10000}}},
        {"type": "Node", "filter_dict": {}}
      ]
    },
    "ranked": {
      "type": "ChainRef",
      "ref": "flows",
      "chain": [{"type": "Call", "function": "compute_cugraph", "params": {"alg": "pagerank"}}]
    },
    "viz": {
      "type": "ChainRef",
      "ref": "ranked",
      "chain": [
        {"type": "Call", "function": "encode_point_color",
         "params": {"column": "risk_score", "palette": ["green", "yellow", "red"], "as_continuous": true}},
        {"type": "Call", "function": "encode_point_icon",
         "params": {"column": "type", "categorical_mapping": {"account": "credit-card"}}}
      ]
    }
  }
}
```

---

## Core Types

### Chain (Container)
```json
{"type": "Chain", "chain": [Node|Edge|Call, ...]}
```

### Node (Filter nodes)
```json
{
  "type": "Node",
  "filter_dict": {"col": value | predicate},  // required
  "name": "label"                              // optional
}
```

### Edge (Traverse)
```json
{
  "type": "Edge",
  "direction": "forward|reverse|undirected",   // required
  "hops": 1,                                   // default: 1
  "to_fixed_point": false,                     // default: false
  "edge_match": {filters},                     // optional
  "source_node_match": {filters},              // optional
  "destination_node_match": {filters},         // optional
  "name": "label"                              // optional
}
```

### Call (Operation)
```json
{"type": "Call", "function": "name", "params": {...}}
```

### Let (Multi-step)
```json
{"type": "Let", "bindings": {"name": Chain | ChainRef}}
```

### ChainRef (Reference)
```json
{"type": "ChainRef", "ref": "name", "chain": [operations]}
```

---

## Predicates

**Comparison:**
```json
{"type": "GT|LT|GE|LE|EQ|NE", "val": value}
```

**Membership:**
```json
{"type": "IsIn", "options": [values]}
{"type": "Between", "lower": 10, "upper": 100}
```

**String:**
```json
{"type": "Contains|Startswith|Endswith", "pat": "text", "case": true, "regex": false}
```

**Null:**
```json
{"type": "IsNull|NotNull"}
```

**Temporal:**
```json
{"type": "datetime", "value": "2024-01-15T10:30:00", "timezone": "UTC"}
{"type": "date", "value": "2024-01-15"}
```

**Note:** Raw values work for equality: `{"age": 30}` equals `{"age": {"type": "EQ", "val": 30}}`

---

## Simple Search

**Basic pattern matching:**

```python
# Dense: [n({'type': 'Person', 'name': 'Alice'}), e_forward(), n({'type': 'Person'})]
```
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "filter_dict": {"type": "Person", "name": "Alice"}},
    {"type": "Edge", "direction": "forward", "hops": 1, "to_fixed_point": false},
    {"type": "Node", "filter_dict": {"type": "Person"}}
  ]
}
```

**Multi-hop (friends of friends):**
```python
# Dense: [n({'name': 'Alice'}), e_forward(hops=2), n()]
```
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "filter_dict": {"name": "Alice"}},
    {"type": "Edge", "direction": "forward", "hops": 2, "to_fixed_point": false},
    {"type": "Node", "filter_dict": {}}
  ]
}
```

**Fixed-point (traverse until no new nodes):**
```python
# Dense: [n({'compromised': True}), e_forward(to_fixed_point=True), n({'critical': True})]
```
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "filter_dict": {"compromised": true}},
    {"type": "Edge", "direction": "forward", "to_fixed_point": true},
    {"type": "Node", "filter_dict": {"critical": true}}
  ]
}
```

**Reverse edges (follow edges backward):**
```python
# Dense: [n({'type': 'product'}), e_reverse(), n({'type': 'supplier'})]
```
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "filter_dict": {"type": "product"}},
    {"type": "Edge", "direction": "reverse", "hops": 1, "to_fixed_point": false},
    {"type": "Node", "filter_dict": {"type": "supplier"}}
  ]
}
```

---

## Filter After Enrichment

**Pattern: Compute metric → Filter by result**

```python
# Dense: let({'enriched': call('get_degrees', {'col': 'degree'}), 'filtered': ref('enriched', [n({'degree': gt(10)})])})
```
```json
{
  "type": "Let",
  "bindings": {
    "enriched": {
      "type": "Chain",
      "chain": [{"type": "Call", "function": "get_degrees", "params": {"col": "degree"}}]
    },
    "filtered": {
      "type": "ChainRef",
      "ref": "enriched",
      "chain": [{"type": "Node", "filter_dict": {"degree": {"type": "GT", "val": 10}}}]
    }
  }
}
```

---

## Graph Algorithms

**PageRank:**
```python
# Dense: call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'score'})
```
```json
{"type": "Call", "function": "compute_cugraph", "params": {"alg": "pagerank", "out_col": "score"}}
```

**Community Detection (Louvain):**
```python
# Dense: call('compute_cugraph', {'alg': 'louvain', 'out_col': 'community'})
```
```json
{"type": "Call", "function": "compute_cugraph", "params": {"alg": "louvain", "out_col": "community"}}
```

**Hypergraph (Events → Entities):**
```python
# Dense: call('hypergraph', {'entity_types': ['user', 'product'], 'direct': True})
```
```json
{"type": "Call", "function": "hypergraph", "params": {"entity_types": ["user", "product"], "direct": true}}
```

**UMAP (2D Layout):**
```python
# Dense: call('umap', {'kind': 'nodes', 'n_neighbors': 15, 'n_components': 2})
```
```json
{"type": "Call", "function": "umap", "params": {"kind": "nodes", "n_neighbors": 15, "n_components": 2}}
```

**Degrees:**
```python
# Dense: call('get_degrees', {'col': 'degree', 'col_in': 'in_deg', 'col_out': 'out_deg'})
```
```json
{"type": "Call", "function": "get_degrees", "params": {"col": "degree", "col_in": "in_deg", "col_out": "out_deg"}}
```

---

## Visualization

**Gradient Color (continuous):**
```python
# Dense: call('encode_point_color', {'column': 'risk', 'palette': ['green','yellow','red'], 'as_continuous': True})
```
```json
{"type": "Call", "function": "encode_point_color",
 "params": {"column": "risk", "palette": ["green", "yellow", "red"], "as_continuous": true}}
```

**Categorical Color:**
```python
# Dense: call('encode_point_color', {'column': 'dept', 'categorical_mapping': {'sales': 'blue', 'eng': 'green'}})
```
```json
{"type": "Call", "function": "encode_point_color",
 "params": {"column": "dept", "categorical_mapping": {"sales": "blue", "eng": "green"}}}
```

**Icons (FontAwesome 4):**
```python
# Dense: call('encode_point_icon', {'column': 'type', 'categorical_mapping': {'server': 'server', 'laptop': 'laptop'}})
```
```json
{"type": "Call", "function": "encode_point_icon",
 "params": {"column": "type", "categorical_mapping": {"server": "server", "laptop": "laptop"}}}
```

**Size:**
```python
# Dense: call('encode_point_size', {'column': 'importance', 'categorical_mapping': {'low': 10, 'high': 40}})
```
```json
{"type": "Call", "function": "encode_point_size",
 "params": {"column": "importance", "categorical_mapping": {"low": 10, "high": 40}}}
```

**Color Palettes:**
- Risk/Heat: `["green", "yellow", "red"]` or `["#00ff00", "#ffff00", "#ff0000"]`
- Cold→Hot: `["blue", "cyan", "yellow", "red"]`
- Grayscale: `["#000000", "#ffffff"]`

---

## Multi-Step (Let/Ref)

**Pattern:** Named DAG stages - `let({'step1': ..., 'step2': ref('step1', [...]), 'step3': ref('step2', [...])})`

See [Quick Example](#quick-example-fraud-detection) and [Social Media](#social-media) for full JSON.

---

## Cyber Security

**Lateral movement detection:**
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "filter_dict": {"compromised": true}},
    {"type": "Edge", "direction": "forward", "to_fixed_point": true,
     "edge_match": {"port": {"type": "IsIn", "options": [22, 23, 3389]}}},
    {"type": "Node", "filter_dict": {"critical": true}},
    {"type": "Call", "function": "encode_point_icon", "params": {
      "column": "type", "categorical_mapping": {"server": "server", "laptop": "laptop", "firewall": "shield"}
    }},
    {"type": "Call", "function": "encode_point_color", "params": {
      "column": "threat", "palette": ["green", "yellow", "red"], "as_continuous": true
    }}
  ]
}
```

**Icons:** `server`, `laptop`, `mobile`, `shield`, `lock`, `bug`, `exclamation-triangle`, `database`

---

## Fraud Detection

**Transaction chains:**
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "filter_dict": {"risk_score": {"type": "GT", "val": 0.8}}},
    {"type": "Edge", "direction": "forward", "hops": 3,
     "edge_match": {"type": "transfer", "amount": {"type": "GT", "val": 10000}}},
    {"type": "Node", "filter_dict": {"country": {"type": "IsIn", "options": ["CN", "RU"]}}},
    {"type": "Call", "function": "encode_point_icon", "params": {
      "column": "type", "categorical_mapping": {"account": "credit-card", "merchant": "shopping-cart"}
    }}
  ]
}
```

**Icons:** `credit-card`, `bank`, `shopping-cart`, `dollar-sign`, `exclamation-triangle`, `flag`

---

## Supply Chain

**Product recall tracing:**
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "filter_dict": {"recall": true}},
    {"type": "Edge", "direction": "reverse", "to_fixed_point": true},
    {"type": "Node", "filter_dict": {"type": "supplier"}},
    {"type": "Call", "function": "encode_point_icon", "params": {
      "column": "type", "categorical_mapping": {"product": "box", "warehouse": "warehouse", "truck": "truck"}
    }}
  ]
}
```

**Icons:** `truck`, `plane`, `ship`, `warehouse`, `box`, `industry`, `shopping-bag`

---

## Social Media

**Influence ranking:**
```json
{
  "type": "Let",
  "bindings": {
    "users": {"type": "Chain", "chain": [{"type": "Node", "filter_dict": {"type": "user"}}]},
    "ranked": {"type": "ChainRef", "ref": "users", "chain": [
      {"type": "Call", "function": "compute_cugraph", "params": {"alg": "pagerank", "out_col": "influence"}}
    ]},
    "viz": {"type": "ChainRef", "ref": "ranked", "chain": [
      {"type": "Call", "function": "encode_point_color", "params": {
        "column": "influence", "palette": ["#ccc", "#08f"], "as_continuous": true
      }},
      {"type": "Call", "function": "encode_point_size", "params": {
        "column": "followers", "categorical_mapping": {"small": 10, "large": 40}
      }}
    ]}
  }
}
```

**Icons:** `user`, `users`, `comment`, `heart`, `share`, `camera`, `envelope`, `bell`

---

## Call Functions

**Algorithms:** `compute_cugraph` (pagerank, louvain, betweenness), `compute_igraph`, `get_degrees`, `get_indegrees`, `get_outdegrees`

**Transforms:** `hypergraph`, `umap`, `collapse`, `materialize_nodes`

**Filters:** `filter_nodes_by_dict`, `filter_edges_by_dict`, `hop`, `drop_nodes`, `keep_nodes`

**Layouts:** `layout_cugraph`, `layout_igraph`, `fa2_layout`, `group_in_a_box_layout`

**Encodings:** `encode_point_color`, `encode_edge_color`, `encode_point_size`, `encode_point_icon`

---

## Generation Rules

1. **Always include `type` field** in every object
2. **Chain wraps operations** - use `{"type": "Chain", "chain": [...]}`
3. **Edge defaults:** `direction: "forward"`, `hops: 1`, `to_fixed_point: false`
4. **Empty filters:** Use `{}` for match-all
5. **Predicates:** Wrap comparisons: `{"type": "GT", "val": 100}`
6. **Temporal:** Tag values: `{"type": "datetime", "value": "...", "timezone": "UTC"}`
7. **ChainRef:** Reference bindings: `{"type": "ChainRef", "ref": "name", "chain": [...]}`

---

## Common Mistakes

❌ Missing type: `{"filter_dict": {...}}`
✅ Correct: `{"type": "Node", "filter_dict": {...}}`

❌ Raw datetime: `{"timestamp": "2024-01-01"}`
✅ Correct: `{"timestamp": {"type": "GT", "val": {"type": "datetime", "value": "2024-01-01T00:00:00"}}}`

❌ Forgot to_fixed_point: `{"hops": 999}` for "traverse all"
✅ Correct: `{"to_fixed_point": true}`

❌ Wrong direction: Using `"backward"` instead of `"reverse"`
✅ Correct: `{"direction": "reverse"}`
