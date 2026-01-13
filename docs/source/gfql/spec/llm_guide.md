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
- [Layouts](#layouts) - FA2 default, ring layouts
- [Multi-Step (Let/Ref)](#let-multi-step) - DAG composition

**Domain Guidance:**
- [Icons & Palettes](#domain-guidance) - By vertical (Cyber, Fraud, Gov, Social, Supply Chain, Events)

**Reference:**
- [Call Functions](#call-functions) - All available functions
- [Generation Rules](#generation-rules) - Best practices
- [Common Mistakes](#common-mistakes) - Errors to avoid

---

## Quick Example: Fraud Detection

**Dense:** `let({'suspicious': n({'risk_score': gt(80)}), 'flows': [n({'risk_score': gt(80)}), e_forward(min_hops=1, max_hops=3), n()], 'ranked': ref('flows', [call('compute_cugraph', {'alg': 'pagerank'})]), 'viz': ref('ranked', [call('encode_point_color', {...}), call('encode_point_icon', {...})])})`

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
      "type": "Chain",
      "chain": [
        {"type": "Node", "filter_dict": {"risk_score": {"type": "GT", "val": 80}}},
        {"type": "Edge", "direction": "forward", "min_hops": 1, "max_hops": 3, "to_fixed_point": false,
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
  "max_hops": 1,                               // default: 1 (hops shorthand)
  "min_hops": 1,                               // optional; default 1 unless max_hops is 0
  "output_min_hops": 1,                        // optional post-filter slice; omit to keep min_hops..max_hops
  "output_max_hops": 1,                        // optional post-filter cap; omit to keep max_hops
  "label_node_hops": "hop",                    // optional; omit/null to skip node hop labels
  "label_edge_hops": "edge_hop",               // optional; omit/null to skip edge hop labels
  "label_seeds": false,                        // optional; when true, label seeds at hop 0
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

**Multi-hop (friends of friends only):**
```python
# Dense: [n({'name': 'Alice'}), e_forward(min_hops=2, max_hops=2), n()]
```
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "filter_dict": {"name": "Alice"}},
    {"type": "Edge", "direction": "forward", "min_hops": 2, "max_hops": 2, "to_fixed_point": false},
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

### Centrality

**PageRank:** `call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'score'})`
```json
{"type": "Call", "function": "compute_cugraph", "params": {"alg": "pagerank", "out_col": "score"}}
```

**Betweenness Centrality:** `call('compute_cugraph', {'alg': 'betweenness_centrality', 'out_col': 'bc'})`
```json
{"type": "Call", "function": "compute_cugraph", "params": {"alg": "betweenness_centrality", "out_col": "bc"}}
```

**Katz Centrality:** `call('compute_cugraph', {'alg': 'katz_centrality', 'out_col': 'katz'})`
```json
{"type": "Call", "function": "compute_cugraph", "params": {"alg": "katz_centrality", "out_col": "katz"}}
```

### Community Detection

**Louvain:** `call('compute_cugraph', {'alg': 'louvain', 'out_col': 'community'})`
```json
{"type": "Call", "function": "compute_cugraph", "params": {"alg": "louvain", "out_col": "community"}}
```

### Similarity

**Jaccard Coefficient:** `call('compute_cugraph', {'alg': 'jaccard', 'out_col': 'similarity'})`
```json
{"type": "Call", "function": "compute_cugraph", "params": {"alg": "jaccard", "out_col": "similarity"}}
```

### Graph Transforms

**Hypergraph (Events → Entities):** `call('hypergraph', {'entity_types': ['user', 'product'], 'direct': True})`
```json
{"type": "Call", "function": "hypergraph", "params": {"entity_types": ["user", "product"], "direct": true}}
```

**UMAP (2D Embedding):** `call('umap', {'kind': 'nodes', 'n_neighbors': 15, 'n_components': 2})`
```json
{"type": "Call", "function": "umap", "params": {"kind": "nodes", "n_neighbors": 15, "n_components": 2}}
```

**Collapse Nodes:** `call('collapse', {'node': 'category', 'attribute': 'type'})`
```json
{"type": "Call", "function": "collapse", "params": {"node": "category", "attribute": "type"}}
```

**Materialize Nodes:** `call('materialize_nodes', {'column': 'relationship'})`
```json
{"type": "Call", "function": "materialize_nodes", "params": {"column": "relationship"}}
```

### Degree Operations

**All Degrees:** `call('get_degrees', {'col': 'degree', 'col_in': 'in_deg', 'col_out': 'out_deg'})`
```json
{"type": "Call", "function": "get_degrees", "params": {"col": "degree", "col_in": "in_deg", "col_out": "out_deg"}}
```

**In-Degrees Only:** `call('get_indegrees', {'col': 'in_degree'})`
```json
{"type": "Call", "function": "get_indegrees", "params": {"col": "in_degree"}}
```

**Out-Degrees Only:** `call('get_outdegrees', {'col': 'out_degree'})`
```json
{"type": "Call", "function": "get_outdegrees", "params": {"col": "out_degree"}}
```

**Topological Levels:** `call('get_topological_levels', {'level_col': 'level'})`
```json
{"type": "Call", "function": "get_topological_levels", "params": {"level_col": "level"}}
```

### Traversals & Filters

**Hop (Multi-step):** `call('hop', {'min_hops': 1, 'max_hops': 3, 'direction': 'forward'})`
```json
{"type": "Call", "function": "hop", "params": {"min_hops": 1, "max_hops": 3, "direction": "forward"}}
```

**Filter Nodes:** `call('filter_nodes_by_dict', {'query': {'type': 'Person', 'age': {'type': 'GT', 'val': 30}}})`
```json
{"type": "Call", "function": "filter_nodes_by_dict", "params": {"query": {"type": "Person", "age": {"type": "GT", "val": 30}}}}
```

**Filter Edges:** `call('filter_edges_by_dict', {'query': {'amount': {'type': 'GT', 'val': 1000}}})`
```json
{"type": "Call", "function": "filter_edges_by_dict", "params": {"query": {"amount": {"type": "GT", "val": 1000}}}}
```

**Drop Nodes:** `call('drop_nodes', {'nodes': ['node_1', 'node_2']})`
```json
{"type": "Call", "function": "drop_nodes", "params": {"nodes": ["node_1", "node_2"]}}
```

**Keep Nodes:** `call('keep_nodes', {'nodes': ['important_1', 'important_2']})`
```json
{"type": "Call", "function": "keep_nodes", "params": {"nodes": ["important_1", "important_2"]}}
```

### igraph Algorithms

**igraph Wrapper:** `call('compute_igraph', {'alg': 'community_leiden', 'out_col': 'community'})`
```json
{"type": "Call", "function": "compute_igraph", "params": {"alg": "community_leiden", "out_col": "community"}}
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

**Edge Color:** `call('encode_edge_color', {'column': 'weight', 'palette': ['#ccc', '#000'], 'as_continuous': True})`
```json
{"type": "Call", "function": "encode_edge_color",
 "params": {"column": "weight", "palette": ["#ccc", "#000"], "as_continuous": true}}
```

**Recommended Palettes:**

*Continuous Gradients:*
- Risk/Heat: `["#00b894", "#fdcb6e", "#d63031"]` (green→yellow→red)
- Cyber Threat: `["#2E86AB", "#A23B72", "#F18F01"]` (cold→warm)
- Influence: `["#95a5a6", "#3498db", "#e74c3c"]` (gray→blue→red)
- Cool→Warm: `["#3498db", "#9b59b6", "#e67e22"]` (blue→purple→orange)

*Categorical (Colorblind-Safe):*
- Okabe-Ito: `["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]`
- Safe 4-color: `["#4477AA", "#EE6677", "#228833", "#CCBB44"]`

*Cool Linking (Same Hue):*
- Blues: `["#EBF5FB", "#85C1E9", "#2E86C1", "#1B4F72"]` (varying saturation)
- Greens: `["#E8F8F5", "#7DCEA0", "#27AE60", "#145A32"]`

See [Domain Guidance](#domain-guidance) for domain-specific palette recommendations.

---

## Layouts

**Default: ForceAtlas2 (FA2)**

Graphistry uses ForceAtlas2 by default, which produces excellent force-directed layouts for most graph structures.

**Recommendation:** Do NOT override layout in most cases. FA2 handles general graphs well.

**When to Override:**
- Time-series data → Use ring layouts
- Hierarchical data → Use tree/graphviz layouts
- Grouped communities → Use group_in_a_box_layout

**Time-Based Ring Layout:** `call('time_ring_layout', {'time_col': 'timestamp', 'num_rings': 10})`
```json
{"type": "Call", "function": "time_ring_layout",
 "params": {"time_col": "timestamp", "num_rings": 10}}
```

> **Tip:** Supply `time_start` / `time_end` as ISO strings (e.g.,
> `"2024-01-01T00:00:00"`) when needed. The executor converts them to
> `numpy.datetime64`, matching the Python Plotter API.

**Categorical Ring Layout:** `call('ring_categorical_layout', {'ring_col': 'category', 'num_rings': 5})`
```json
{"type": "Call", "function": "ring_categorical_layout",
 "params": {"ring_col": "category", "num_rings": 5}}
```

**Continuous Ring Layout:** `call('ring_continuous_layout', {'ring_col': 'score', 'num_rings': 8})`
```json
{"type": "Call", "function": "ring_continuous_layout",
 "params": {"ring_col": "score", "num_rings": 8}}
```

**Other Layouts:**
- `layout_cugraph` - GPU-accelerated cuGraph layouts
- `layout_igraph` - igraph layouts (CPU)
- `layout_graphviz` - Graphviz layouts (hierarchical)
- `group_in_a_box_layout` - Group-in-a-box with community partitioning

---

## Multi-Step (Let/Ref)

**Pattern:** Named DAG stages - `let({'step1': ..., 'step2': ref('step1', [...]), 'step3': ref('step2', [...])})`

See [Quick Example](#quick-example-fraud-detection) for full JSON example.

---

## Domain Guidance

**Cyber/IT Security:**
- Icons: server, laptop, mobile, shield, lock, bug, exclamation-triangle, database, firewall, cloud
- Colors: Blues/grays (trusted), reds/oranges (threats), yellows (warnings)
- Palettes: `["#2E86AB", "#A23B72", "#F18F01"]` (cold→warm threat levels)
- Patterns: Lateral movement (fixed-point), privilege escalation, kill chains

**Fraud/Finance:**
- Icons: credit-card, bank, shopping-cart, dollar-sign, exclamation-triangle, flag, money, exchange
- Colors: Greens (legitimate), yellows (suspicious), reds (fraudulent)
- Palettes: `["#00b894", "#fdcb6e", "#d63031"]` (risk gradients)
- Patterns: Transaction chains, velocity analysis, clustering by behavior

**Government/Enforcement/IC/Military:**
- Icons: shield, flag, star, building, briefcase, user-secret, crosshairs, certificate
- Colors: Blues (official), reds (adversary), greens (allied)
- Palettes: Classification levels, threat actors by affiliation
- Patterns: Attribution chains, actor networks, timeline analysis

**Social Media:**
- Icons: user, users, comment, heart, share, camera, envelope, bell, rss, hashtag
- Colors: Cool blues (low engagement) → warm oranges (high engagement)
- Palettes: `["#95a5a6", "#3498db", "#e74c3c"]` (influence levels)
- Patterns: Influence ranking, community detection, virality tracking

**Supply Chain/Logistics:**
- Icons: truck, plane, ship, warehouse, box, industry, shopping-bag, map-marker, cog
- Colors: Greens (on-time), yellows (delayed), reds (critical)
- Palettes: `["#27ae60", "#f39c12", "#c0392b"]` (status indicators)
- Patterns: Recall tracing (reverse), bottleneck detection, route optimization

**Event/People/Digital:**
- Icons: calendar, clock, user, desktop, mobile, tablet, globe, sitemap
- Colors: Time-based gradients, entity type differentiation
- Palettes: Blues (past) → purples (present) → oranges (future)
- Patterns: Timeline layouts (ring), entity relationships, digital footprints

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
3. **Edge defaults:** `direction: "forward"`, `max_hops: 1` (`hops` shorthand), `min_hops: 1` unless `max_hops` is 0, `to_fixed_point: false`
4. **Output slice defaults:** If `output_min_hops`/`output_max_hops` are omitted, results keep all traversed hops up to `max_hops`; set them to post-filter displayed hops.
5. **Empty filters:** Use `{}` for match-all
5. **Predicates:** Wrap comparisons: `{"type": "GT", "val": 100}`
6. **Temporal:** Tag values: `{"type": "datetime", "value": "...", "timezone": "UTC"}`
7. **ChainRef:** Reference bindings: `{"type": "ChainRef", "ref": "name", "chain": [...]}`

---

## Common Mistakes

**Wrong:** Missing type: `{"filter_dict": {...}}`
**Correct:** `{"type": "Node", "filter_dict": {...}}`

**Wrong:** Raw datetime: `{"timestamp": "2024-01-01"}`
**Correct:** `{"timestamp": {"type": "GT", "val": {"type": "datetime", "value": "2024-01-01T00:00:00"}}}`

**Wrong:** Forgot to_fixed_point: `{"max_hops": 999}` for "traverse all"
**Correct:** `{"to_fixed_point": true}`

**Wrong:** Using `"backward"` instead of `"reverse"`
**Correct:** `{"direction": "reverse"}`
