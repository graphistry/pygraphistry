(gfql-spec-wire-protocol)=

# GFQL Wire Protocol Specification

## Introduction

The GFQL Wire Protocol defines the JSON serialization format for GFQL queries, enabling:
- Client-server communication
- Query persistence and storage
- Cross-language interoperability between Python, JavaScript, and other clients
- Configuration-driven query generation

### Design Principles
- **Type Safety**: Tagged dictionaries preserve type information
- **Self-Describing**: Each object includes type metadata
- **Extensible**: Schema supports future additions
- **Round-Trip Safe**: Lossless serialization/deserialization

## Protocol Overview

### Message Structure

All GFQL wire protocol messages are JSON objects with a `type` field:

```json
{
  "type": "MessageType",
  ...additional fields...
}
```

### Supported Message Types
- `Chain`: Complete query chain
- `Node`: Node matcher operation
- `Edge`: Edge traversal operation
- Predicates: `GT`, `LT`, `EQ`, `IsIn`, `Between`, etc.
- Temporal values: `datetime`, `date`, `time`

## Message Structure

All GFQL wire protocol messages are JSON objects with a `type` field that identifies the message type. The protocol uses discriminated unions for polymorphic types.

### Type Identification

Each object includes a `type` field:
- Operations: `"Node"`, `"Edge"`, `"Chain"`, `"Let"`, `"Ref"`, `"RemoteGraph"`, `"Call"`
- Predicates: `"GT"`, `"LT"`, `"IsIn"`, etc.
- Temporal values: `"datetime"`, `"date"`, `"time"`

This enables unambiguous deserialization and validation.


## Query Structure

GFQL queries can be expressed as either Chains (linear patterns) or Let queries (DAG patterns with named bindings).

### Chain Queries

Chains represent linear graph patterns as sequences of operations:

**Python**:
```python
g.gfql([n({"type": "person"}), e_forward(), n({"type": "company"})])
```

**Wire Format**:
```json
{
  "type": "Chain",
  "ops": [
    {"type": "Node", "filter_dict": {"type": "person"}},
    {"type": "Edge", "direction": "forward"},
    {"type": "Node", "filter_dict": {"type": "company"}}
  ]
}
```

### Let Queries (DAG Patterns)

Let queries enable complex patterns with named bindings and references:

**Python**:
```python
g.gfql({
    "suspects": n({"risk_score": gt(8)}),
    "contacts": ref("suspects").gfql([e(), n()]),
    "transactions": ref("contacts").gfql([e_forward({"type": "transaction"})])
})
```

**Wire Format**:
```json
{
  "type": "Let",
  "bindings": {
    "suspects": {"type": "Node", "filter_dict": {"risk_score": {"type": "GT", "val": 8}}},
    "contacts": {
      "type": "Ref",
      "ref": "suspects",
      "chain": [
        {"type": "Edge", "direction": "undirected"},
        {"type": "Node"}
      ]
    },
    "transactions": {
      "type": "Ref",
      "ref": "contacts",
      "chain": [
        {"type": "Edge", "direction": "forward", "filter_dict": {"type": "transaction"}}
      ]
    }
  }
}
```

## Operation Serialization

### Node Operation

**Python**:
```python
n({"type": "person", "age": gt(30)}, name="adults")
```

**Wire Format**:
```json
{
  "type": "Node",
  "filter_dict": {
    "type": "person",
    "age": {
      "type": "GT",
      "val": 30
    }
  },
  "name": "adults"
}
```

### Edge Operation

**Python**:
```python
e_forward(
    {"type": "transaction"},
    hops=2,
    source_node_match={"active": True},
    name="txns"
)
```

**Wire Format**:
```json
{
  "type": "Edge",
  "direction": "forward",
  "edge_match": {
    "type": "transaction"
  },
  "hops": 2,
  "source_node_match": {
    "active": true
  },
  "name": "txns"
}
```

### Chain

**Python**:
```python
chain([
    n({"id": "Alice"}),
    e_forward({"type": "friend"}),
    n({"status": "active"})
])
```

**Wire Format**:
```json
{
  "type": "Chain",
  "chain": [
    {
      "type": "Node",
      "filter_dict": {"id": "Alice"}
    },
    {
      "type": "Edge",
      "direction": "forward",
      "edge_match": {"type": "friend"}
    },
    {
      "type": "Node",
      "filter_dict": {"status": "active"}
    }
  ]
}
```

### Let Bindings (DAG Patterns)

**Python**:
```python
ASTLet({
    'persons': n({'type': 'Person'}),
    'adults': ASTRef('persons', [n({'age': ge(18)})]),
    'connections': ASTRef('adults', [
        e_forward({'type': 'knows'}),
        ASTRef('adults')
    ])
})
```

**Wire Format**:
```json
{
  "type": "Let",
  "bindings": {
    "persons": {
      "type": "Node",
      "filter_dict": {"type": "Person"}
    },
    "adults": {
      "type": "Ref",
      "ref": "persons",
      "chain": [{
        "type": "Node",
        "filter_dict": {
          "age": {"type": "GE", "val": 18}
        }
      }]
    },
    "connections": {
      "type": "Ref",
      "ref": "adults",
      "chain": [
        {
          "type": "Edge",
          "direction": "forward",
          "edge_match": {"type": "knows"}
        },
        {
          "type": "Ref",
          "ref": "adults",
          "chain": []
        }
      ]
    }
  }
}
```

### Ref (Reference to Named Binding)

**Python**:
```python
ASTRef('base_pattern', [
    e_forward({'status': 'active'}),
    n({'verified': True})
])
```

**Wire Format**:
```json
{
  "type": "Ref",
  "ref": "base_pattern",
  "chain": [
    {
      "type": "Edge",
      "direction": "forward",
      "edge_match": {"status": "active"}
    },
    {
      "type": "Node",
      "filter_dict": {"verified": true}
    }
  ]
}
```

### RemoteGraph (Load Remote Dataset)

**Python**:
```python
ASTRemoteGraph('dataset-123', token='auth-token')
```

**Wire Format**:
```json
{
  "type": "RemoteGraph",
  "dataset_id": "dataset-123",
  "token": "auth-token"
}
```

Without token (public dataset):
```json
{
  "type": "RemoteGraph",
  "dataset_id": "public-dataset-456"
}
```

### Call Operation

**Python**:
```python
ASTCall('get_degrees', {
    'col': 'centrality',
    'col_in': 'in_centrality',
    'col_out': 'out_centrality'
})
```

**Wire Format**:
```json
{
  "type": "Call",
  "function": "get_degrees",
  "params": {
    "col": "centrality",
    "col_in": "in_centrality", 
    "col_out": "out_centrality"
  }
}
```

#### Call Operation Examples

**PageRank computation**:
```json
{
  "type": "Call",
  "function": "compute_cugraph",
  "params": {
    "alg": "pagerank",
    "out_col": "pagerank_score",
    "params": {"alpha": 0.85}
  }
}
```

**Graph layout**:
```json
{
  "type": "Call",
  "function": "layout_cugraph",
  "params": {
    "layout": "force_atlas2",
    "params": {
      "iterations": 500,
      "outbound_attraction_distribution": true,
      "edge_weight_influence": 1.0
    }
  }
}
```

**Node filtering**:
```json
{
  "type": "Call",
  "function": "filter_nodes_by_dict",
  "params": {
    "filter_dict": {
      "type": "person",
      "active": true
    }
  }
}
```

**Complex hop traversal**:
```json
{
  "type": "Call",
  "function": "hop",
  "params": {
    "hops": 3,
    "direction": "forward",
    "edge_match": {"type": "transfer"},
    "destination_node_match": {"account_type": "checking"}
  }
}
```

#### Available Call Methods

The following Plottable methods are available through Call operations:

**Graph Analysis**:
- `get_degrees`: Calculate node degrees
- `get_indegrees`: Calculate in-degrees only
- `get_outdegrees`: Calculate out-degrees only
- `compute_cugraph`: Run GPU algorithms (pagerank, louvain, etc.)
- `compute_igraph`: Run CPU algorithms
- `get_topological_levels`: Analyze DAG structure

**Filtering & Transformation**:
- `filter_nodes_by_dict`: Filter nodes by attributes
- `filter_edges_by_dict`: Filter edges by attributes
- `hop`: Traverse graph with complex conditions
- `drop_nodes`: Remove specified nodes
- `keep_nodes`: Keep only specified nodes
- `collapse`: Merge nodes by attribute
- `prune_self_edges`: Remove self-loops
- `materialize_nodes`: Generate node table from edges

**Layout**:
- `layout_cugraph`: GPU-accelerated layouts
- `layout_igraph`: CPU-based layouts
- `layout_graphviz`: Graphviz layouts (dot, neato, etc.)
- `fa2_layout`: ForceAtlas2 layout

**Visual Encoding**:
- `encode_point_color`: Map node values to colors
- `encode_edge_color`: Map edge values to colors
- `encode_point_size`: Map node values to sizes
- `encode_point_icon`: Map node values to icons

**Metadata**:
- `name`: Set visualization name
- `description`: Set visualization description

## Predicate Serialization

### Comparison Predicates

```json
{"type": "GT", "val": 100}
{"type": "LT", "val": 50.5}
{"type": "GE", "val": "2024-01-01"}
{"type": "LE", "val": true}
{"type": "EQ", "val": "active"}
{"type": "NE", "val": null}
```

### Between Predicate

```json
{
  "type": "Between",
  "lower": 10,
  "upper": 20,
  "inclusive": true
}
```

### IsIn Predicate

```json
{
  "type": "IsIn",
  "options": ["A", "B", "C"]
}
```

### String Predicates

```json
{"type": "Contains", "pattern": "search"}
{"type": "Startswith", "pattern": "prefix"}
{"type": "Endswith", "pattern": "suffix"}
{"type": "Match", "pattern": "^[A-Z]+\\d+$"}
```

### Null Predicates

```json
{"type": "IsNull"}
{"type": "NotNull"}
{"type": "IsNA"}
{"type": "NotNA"}
```

### Temporal Check Predicates

```json
{"type": "IsMonthStart"}
{"type": "IsYearEnd"}
{"type": "IsLeapYear"}
```

## Type Serialization

### Scalar Types

```json
"hello world"        // string
42                   // integer
3.14159             // float
true                // boolean
null                // null
```

### Temporal Types

#### DateTime
```json
{
  "type": "datetime",
  "value": "2024-01-15T10:30:00",
  "timezone": "America/New_York"  // Optional, defaults to "UTC"
}
```

#### Date
```json
{
  "type": "date",
  "value": "2024-01-15"
}
```

#### Time
```json
{
  "type": "time",
  "value": "14:30:00.123456"
}
```

**Note**: The `timezone` field is optional for DateTime values and defaults to "UTC" if omitted. This ensures consistent behavior across systems while allowing explicit timezone specification when needed.

## Examples

### User 360 Query

**Python**:
```python
g.chain([
    n({"customer_id": "C123"}),
    e_forward({
        "type": "purchase",
        "timestamp": gt(pd.Timestamp("2024-01-01"))
    })
])
```

**Wire Format**:
```json
{
  "type": "Chain",
  "chain": [
    {
      "type": "Node",
      "filter_dict": {
        "customer_id": "C123"
      }
    },
    {
      "type": "Edge",
      "direction": "forward",
      "edge_match": {
        "type": "purchase",
        "timestamp": {
          "type": "GT",
          "val": {
            "type": "datetime",
            "value": "2024-01-01T00:00:00",
            "timezone": "UTC"
          }
        }
      }
    }
  ]
}
```

### Cyber Security Pattern

**Python**:
```python
g.chain([
    n({"ip": is_in(["192.168.1.100", "192.168.1.101"])}),
    e_forward(
        edge_query="port IN [22, 23, 3389]",
        to_fixed_point=True
    ),
    n({"type": "server", "critical": True})
])
```

**Wire Format**:
```json
{
  "type": "Chain",
  "chain": [
    {
      "type": "Node",
      "filter_dict": {
        "ip": {
          "type": "IsIn",
          "options": ["192.168.1.100", "192.168.1.101"]
        }
      }
    },
    {
      "type": "Edge",
      "direction": "forward",
      "edge_query": "port IN [22, 23, 3389]",
      "to_fixed_point": true
    },
    {
      "type": "Node",
      "filter_dict": {
        "type": "server",
        "critical": true
      }
    }
  ]
}
```

### Complex DAG Pattern

**Python**:
```python
g.gfql(ASTLet({
    'suspicious_ips': n({'risk_score': gt(80)}),
    'lateral_movement': ASTRef('suspicious_ips', [
        e_forward({'type': 'ssh', 'failed_attempts': gt(5)}),
        n({'type': 'server'})
    ]),
    'escalation': ASTRef('lateral_movement', [
        e_forward({'type': 'privilege_change'}),
        n({'admin': True})
    ])
}))
```

**Wire Format**:
```json
{
  "type": "Let",
  "bindings": {
    "suspicious_ips": {
      "type": "Node",
      "filter_dict": {
        "risk_score": {"type": "GT", "val": 80}
      }
    },
    "lateral_movement": {
      "type": "Ref",
      "ref": "suspicious_ips",
      "chain": [
        {
          "type": "Edge",
          "direction": "forward",
          "edge_match": {
            "type": "ssh",
            "failed_attempts": {"type": "GT", "val": 5}
          }
        },
        {
          "type": "Node",
          "filter_dict": {"type": "server"}
        }
      ]
    },
    "escalation": {
      "type": "Ref",
      "ref": "lateral_movement",
      "chain": [
        {
          "type": "Edge",
          "direction": "forward",
          "edge_match": {"type": "privilege_change"}
        },
        {
          "type": "Node",
          "filter_dict": {"admin": true}
        }
      ]
    }
  }
}
```

### DAG Pattern with Call Operations

**Python**:
```python
g.gfql(ASTLet({
    'high_value': n({'amount': gt(100000)}),
    'connected': ASTRef('high_value', [
        e_forward({'type': 'transfer'}, hops=2)
    ]),
    'analyzed': ASTCall('compute_cugraph', {
        'alg': 'pagerank',
        'out_col': 'influence_score'
    })
}))
```

**Wire Format**:
```json
{
  "type": "Let",
  "bindings": {
    "high_value": {
      "type": "Node",
      "filter_dict": {
        "amount": {"type": "GT", "val": 100000}
      }
    },
    "connected": {
      "type": "Ref",
      "ref": "high_value",
      "chain": [
        {
          "type": "Edge",
          "direction": "forward",
          "edge_match": {"type": "transfer"},
          "hops": 2
        }
      ]
    },
    "analyzed": {
      "type": "Call",
      "function": "compute_cugraph",
      "params": {
        "alg": "pagerank",
        "out_col": "influence_score"
      }
    }
  }
}
```

## Best Practices

1. **Always include type fields**: Every object must have a `type`
2. **Use ISO formats**: Dates and times in ISO 8601
3. **Handle timezones consistently**: Include timezone for datetime values when precision matters (defaults to UTC)
4. **Validate before sending**: Use JSON Schema validation
5. **Handle unknown fields**: Ignore unrecognized fields for compatibility
6. **Let bindings**: Define bindings in dependency order (referenced names must be defined first)
7. **Ref validation**: Ensure referenced names exist in the Let binding scope
8. **RemoteGraph security**: Protect authentication tokens in transit and storage
9. **Call operations**: Only use function names from the safelist
10. **Parameter validation**: Ensure Call parameters match expected types
11. **Error handling**: Call operations may fail if schema requirements aren't met

## See Also

- {ref}`gfql-spec-language` - Language specification
- {ref}`gfql-spec-cypher-mapping` - Cypher to GFQL translation with wire protocol examples