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
- `Let`: DAG pattern with named bindings
- `ChainRef`: Reference to Let binding with optional chain
- `RemoteGraph`: Reference to remote dataset
- `Call`: Algorithm/transformation invocation
- `Node`: Node matcher operation
- `Edge`: Edge traversal operation
- Predicates: `GT`, `LT`, `EQ`, `IsIn`, `Between`, etc.
- Temporal values: `datetime`, `date`, `time`

## Message Structure

All GFQL wire protocol messages are JSON objects with a `type` field that identifies the message type. The protocol uses discriminated unions for polymorphic types.

### Type Identification

Each object includes a `type` field:
- Operations: `"Node"`, `"Edge"`, `"Chain"`, `"Let"`, `"ChainRef"`, `"RemoteGraph"`, `"Call"`
- Predicates: `"GT"`, `"LT"`, `"IsIn"`, etc.
- Temporal values: `"datetime"`, `"date"`, `"time"`

This enables unambiguous deserialization and validation.


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
    min_hops=2,
    max_hops=4,
    output_min_hops=3,
    label_edge_hops="edge_hop",
    source_node_match={"active": True},
    name="txns"
)
```

**Wire Format**:
```json
{
  "type": "Edge",
  "direction": "forward",
  "edge_match": { "type": "transaction" },
  "min_hops": 2,
  "max_hops": 4,
  "output_min_hops": 3,
  "label_edge_hops": "edge_hop",
  "source_node_match": { "active": true },
  "name": "txns"
}
```

Optional fields:
- `hops` (shorthand for `max_hops`)
- `output_min_hops`
- `output_max_hops`
- `label_node_hops`, `label_edge_hops`, `label_seeds`
- `to_fixed_point`

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

### Let Operation

**Python**:
```python
let({
    'persons': n({'type': 'Person'}),
    'adults': ref('persons', [n({'age': ge(18)})])
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
      "type": "ChainRef",
      "ref": "persons",
      "chain": [{
        "type": "Node",
        "filter_dict": {
          "age": {"type": "GE", "val": 18}
        }
      }]
    }
  }
}
```

### ChainRef Operation

ChainRef executes on the referenced graph; bindings used for edge traversal should retain edges
(for example, from an ``Edge`` or ``Chain`` binding).

**Python**:
```python
ref('base_graph', [
    e_forward({'weight': gt(0.5)}),
    n({'status': 'active'})
])
```

**Wire Format**:
```json
{
  "type": "ChainRef",
  "ref": "base_graph",
  "chain": [
    {
      "type": "Edge",
      "direction": "forward",
      "edge_match": {"weight": {"type": "GT", "val": 0.5}}
    },
    {
      "type": "Node",
      "filter_dict": {"status": "active"}
    }
  ]
}
```

### RemoteGraph Operation

**Python**:
```python
remote(dataset_id='fraud-network-2024')
```

**Wire Format**:
```json
{
  "type": "RemoteGraph",
  "dataset_id": "fraud-network-2024"
}
```

### Call Operation

**Python**:
```python
call('compute_cugraph', {'alg': 'pagerank', 'damping': 0.85})
```

**Wire Format**:
```json
{
  "type": "Call",
  "function": "compute_cugraph",
  "params": {
    "alg": "pagerank",
    "damping": 0.85
  }
}
```

```{note}
For the complete list of safelisted layout calls—including the radial
variants—refer to {doc}`/gfql/builtin_calls`.
```

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

**Basic forms** (defaults: `case=true`, `na=null`, `flags=0`):
```json
{"type": "Contains", "pat": "search", "case": true, "flags": 0, "na": null, "regex": true}
{"type": "Startswith", "pat": "prefix", "case": true, "na": null}
{"type": "Endswith", "pat": "suffix", "case": true, "na": null}
{"type": "Match", "pat": "^[A-Z]+\\d+$", "case": true, "flags": 0, "na": null}
{"type": "Fullmatch", "pat": "^[A-Z]+$", "case": true, "flags": 0, "na": null}
```

**Case-insensitive matching** (using `case=false`):
```json
{"type": "Startswith", "pat": "prefix", "case": false, "na": null}
{"type": "Fullmatch", "pat": "^test$", "case": false, "flags": 0, "na": null}
```

**Tuple patterns** (OR logic - match any):
```json
{"type": "Startswith", "pat": ["app", "ban"], "case": true, "na": null}
{"type": "Endswith", "pat": [".jpg", ".png", ".gif"], "case": true, "na": null}
```

**NA handling** (fill value for missing data):
```json
{"type": "Startswith", "pat": "test", "case": true, "na": false}
{"type": "Endswith", "pat": "end", "case": true, "na": true}
```

**Notes**:
- `pat`: Pattern string or array of strings (array uses OR logic)
- `case`: Case-sensitive if `true` (default: `true`)
- `na`: Fill value for null/missing values (default: `null` preserves NA)
- `flags`: Regex flags for `Match`/`Fullmatch` (default: `0`)
- `regex`: Whether pattern is regex for `Contains` (default: `true`)

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

## Collections Payloads

Collections are Graphistry visualization overlays that use GFQL wire protocol operations to define subsets
of nodes, edges, or subgraphs. They are applied in priority order, with earlier collections overriding later
ones for styling.

### Collection Set

Collection sets wrap GFQL operations in a `gfql_chain` object:

```json
{
  "type": "set",
  "id": "purchasers",
  "name": "Purchasers",
  "node_color": "#00BFFF",
  "expr": {
    "type": "gfql_chain",
    "gfql": [
      {"type": "Node", "filter_dict": {"status": "purchased"}}
    ]
  }
}
```

### Collection Intersection

Intersections reference previously defined set IDs:

```json
{
  "type": "intersection",
  "name": "High Value Purchasers",
  "node_color": "#AA00AA",
  "expr": {
    "type": "intersection",
    "sets": ["purchasers", "vip"]
  }
}
```

For Python examples and helper constructors, see the
:doc:`Collections tutorial notebook </demos/more_examples/graphistry_features/collections>`.

## Examples

### User 360 Query

**Python**:
```python
g.gfql([
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
g.gfql([
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


## Best Practices

1. **Always include type fields**: Every object must have a `type`
2. **Use ISO formats**: Dates and times in ISO 8601
3. **Handle timezones consistently**: Include timezone for datetime values when precision matters (defaults to UTC)
4. **Validate before sending**: Use JSON Schema validation
5. **Handle unknown fields**: Ignore unrecognized fields for compatibility

## See Also

- {ref}`gfql-spec-language` - Language specification
- {ref}`gfql-spec-cypher-mapping` - Cypher to GFQL translation with wire protocol examples
