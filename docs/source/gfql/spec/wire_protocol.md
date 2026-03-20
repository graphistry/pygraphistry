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
  "payload": {}
}
```

### Supported Message Types
- `Chain`: Complete query chain
- `Let`: DAG pattern with named bindings
- `Ref`: Reference to Let binding with optional chain
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
- Operations: `"Node"`, `"Edge"`, `"Chain"`, `"Let"`, `"Ref"`, `"RemoteGraph"`, `"Call"`
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
from graphistry import n, e_forward

g.gfql([
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

Optional fields:
- `where`: list of same-path comparisons using `eq`, `neq`, `lt`, `le`, `gt`, `ge`
  with `left`/`right` as `alias.column` strings. Multiple entries are ANDed.
  Operator mapping:
  - `eq` maps to `==`
  - `neq` maps to `!=`
  - `lt` maps to `<`
  - `le` maps to `<=`
  - `gt` maps to `>`
  - `ge` maps to `>=`

**Chain with WHERE (wire format):**
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "filter_dict": {"type": "account"}, "name": "a"},
    {"type": "Edge", "direction": "forward"},
    {"type": "Node", "filter_dict": {"type": "user"}, "name": "c"}
  ],
  "where": [{"eq": {"left": "a.owner_id", "right": "c.owner_id"}}]
}
```

### WHERE Validation Errors

The parser and same-path validator reject malformed or unresolved WHERE clauses
before execution.

Unsupported operator key:
```json
{
  "type": "Chain",
  "chain": [{"type": "Node", "name": "a"}, {"type": "Node", "name": "c"}],
  "where": [{"lte": {"left": "a.owner_id", "right": "c.owner_id"}}]
}
```
Expected error: `Unsupported WHERE operator 'lte'`.

Missing required keys:
```json
{
  "type": "Chain",
  "chain": [{"type": "Node", "name": "a"}, {"type": "Node", "name": "c"}],
  "where": [{"eq": {"left": "a.owner_id"}}]
}
```
Expected error: `WHERE clause must have 'left' and 'right' keys`.

Alias not bound in the chain:
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "name": "a"},
    {"type": "Edge", "direction": "forward", "name": "e"},
    {"type": "Node", "name": "c"}
  ],
  "where": [{"eq": {"left": "missing.owner_id", "right": "c.owner_id"}}]
}
```
Expected error: `WHERE references aliases with no node/edge bindings: missing`.

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
      "type": "Ref",
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

#### Nested Let (Scope Isolation)

A ``Let`` binding value may itself be a ``Let``. The inner ``Let`` executes as an opaque unit: its internal bindings are **not** visible in the outer scope. The outer ``Let`` sees only the binding name and the inner DAG's result.

**Python**:
```python
let({
    'stage1': let({
        'people': n({'type': 'Person'}),
        'friends': ref('people', [e_forward(), n()])
    }),
    'stage2': ref('stage1', [e_forward(), n()])
})
```

**Wire Format**:
```json
{
  "type": "Let",
  "bindings": {
    "stage1": {
      "type": "Let",
      "bindings": {
        "people": {"type": "Node", "filter_dict": {"type": "Person"}},
        "friends": {
          "type": "Ref", "ref": "people",
          "chain": [{"type": "Edge", "direction": "forward"}, {"type": "Node"}]
        }
      }
    },
    "stage2": {
      "type": "Ref", "ref": "stage1",
      "chain": [{"type": "Edge", "direction": "forward"}, {"type": "Node"}]
    }
  }
}
```

**Scope rules** (lexical scoping):
- ``stage2`` can reference ``stage1`` (an outer binding)
- ``stage2`` **cannot** reference ``people`` or ``friends`` (inner bindings — they do not leak upward)
- Inner bindings **can** read outer bindings (e.g., ``people`` could use ``ref('stage2')`` if ``stage2`` had already executed)
- Sibling inner ``Let`` blocks may reuse the same binding names without collision
- If an inner binding has the same name as an outer binding, the inner shadows the outer within its scope without corrupting the outer value
- The inner ``Let`` result is the last executed binding in its own scope

### Ref Operation

Ref executes on the referenced graph; bindings used for edge traversal should retain edges
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
  "type": "Ref",
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

#### Row-Pipeline Call Serialization

Row-pipeline operators use the same existing `Call` envelope. There is no
wire-format envelope change for row pipelines; only `function`/`params` values
vary by operator.

`rows`:
```json
{"type": "Call", "function": "rows", "params": {"table": "nodes", "source": "q"}}
```
`where_rows`:
```json
{"type": "Call", "function": "where_rows", "params": {"expr": "score >= 50"}}
```
`where_rows.expr` supports comparison operators:
`=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`.
`where_rows` can also use predicate dictionaries on the active row table:
```json
{"type": "Call", "function": "where_rows", "params": {"filter_dict": {"score": {"type": "GE", "val": 50}}}}
```

WHERE context summary:
- Chain-level same-path `where` uses lower-case operator keys (`eq`, `neq`,
  `lt`, `le`, `gt`, `ge`) with `left`/`right` alias-column references.
- Row-level `where_rows(filter_dict=...)` uses predicate envelopes like
  `GT`, `GE`, `LT`, `LE`, `EQ`, `NE` on active row-table columns.

`select`:
```json
{"type": "Call", "function": "select", "params": {"items": [["id", "id"], ["score", "score"]]}}
```
`with_`:
```json
{"type": "Call", "function": "with_", "params": {"items": [["id", "id"]]}}
```
`order_by`:
```json
{"type": "Call", "function": "order_by", "params": {"keys": [["score", "desc"], ["name", "asc"]]}}
```
`skip`:
```json
{"type": "Call", "function": "skip", "params": {"value": 20}}
```
`limit`:
```json
{"type": "Call", "function": "limit", "params": {"value": 10}}
```
`distinct`:
```json
{"type": "Call", "function": "distinct", "params": {}}
```
`unwind`:
```json
{"type": "Call", "function": "unwind", "params": {"expr": "tags", "as_": "tag"}}
```
`group_by`:
```json
{"type": "Call", "function": "group_by", "params": {"keys": ["category"], "aggregations": [["cnt", "count"], ["total", "sum", "amount"]]}}
```

`return_(...)` is serialized as `function: "select"` with equivalent `items`.

#### Row-Call Validation Errors

Row-call payloads are validated before execution. Invalid payloads fail fast.

Invalid `rows.table` enum:
```json
{"type": "Call", "function": "rows", "params": {"table": "invalid"}}
```
Expected error: parameter validation failure (`table` must be `"nodes"` or `"edges"`).

Invalid `where_rows.expr` type:
```json
{"type": "Call", "function": "where_rows", "params": {"expr": 123}}
```
Expected error: parameter validation failure (`expr` must be a non-empty string).

Invalid `order_by` direction:
```json
{"type": "Call", "function": "order_by", "params": {"keys": [["score", "up"]]}}
```
Expected error: parameter validation failure (`direction` must be `"asc"` or `"desc"`).

Invalid `group_by` payload shape:
```json
{"type": "Call", "function": "group_by", "params": {"keys": [], "aggregations": []}}
```
Expected error: parameter validation failure (non-empty keys and valid aggregation specs required).

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

Temporal comparisons use standard predicate envelopes over these typed temporal
values:
- `GT`, `GE`, `LT`, `LE`, `EQ`, `NE`

Example:
```json
{
  "type": "GE",
  "val": {
    "type": "date",
    "value": "2024-01-01"
  }
}
```

**Note**: The `timezone` field is optional for DateTime values and defaults to "UTC" if omitted. This ensures consistent behavior across systems while allowing explicit timezone specification when needed.

## Examples

### `MATCH ... RETURN` Row Pipeline

**Python**:
```python
g.gfql([
    n({"type": "Person"}),
    e_forward({"type": "FOLLOWS"}),
    n({"type": "Person"}, name="q"),
    rows(table="nodes", source="q"),
    where_rows(expr="score >= 50"),
    return_(["id", "name", "score"]),
    order_by([("score", "desc"), ("name", "asc")]),
    limit(25),
])
```

**Wire Format**:
```json
{
  "type": "Chain",
  "chain": [
    {"type": "Node", "filter_dict": {"type": "Person"}},
    {"type": "Edge", "direction": "forward", "edge_match": {"type": "FOLLOWS"}},
    {"type": "Node", "filter_dict": {"type": "Person"}, "name": "q"},
    {"type": "Call", "function": "rows", "params": {"table": "nodes", "source": "q"}},
    {"type": "Call", "function": "where_rows", "params": {"expr": "score >= 50"}},
    {"type": "Call", "function": "select", "params": {"items": [["id", "id"], ["name", "name"], ["score", "score"]]}},
    {"type": "Call", "function": "order_by", "params": {"keys": [["score", "desc"], ["name", "asc"]]}},
    {"type": "Call", "function": "limit", "params": {"value": 25}}
  ]
}
```

### User 360 Query

**Python**:
<!-- doc-test: skip -->
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
<!-- doc-test: skip -->
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


## Graph Constructors and the Wire Protocol

GFQL's Cypher extensions (`GRAPH { }` constructors, `GRAPH g = ...` bindings,
`USE g` graph switching) serialize using the existing `Let`, `Chain`, `Call`,
and `Ref` wire-protocol primitives. No new message types are needed.

### Serialization

A multi-stage graph pipeline maps to a `Let` whose bindings are `Chain` or
`Call` values, with `Ref` for `USE` references:

```
GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) WHERE a.score > 10 }
GRAPH g2 = GRAPH { USE g1 CALL graphistry.degree.write() }
USE g2 MATCH (n) RETURN n.id, n.degree ORDER BY n.degree DESC
```

```json
{
  "type": "Let",
  "bindings": {
    "g1": {
      "type": "Chain",
      "chain": [
        {"type": "Node", "filter_dict": {"score": {"type": "GT", "val": 10}}, "name": "a"},
        {"type": "Edge", "direction": "forward", "name": "r"},
        {"type": "Node", "name": "b"}
      ]
    },
    "g2": {
      "type": "Ref",
      "ref": "g1",
      "chain": [
        {"type": "Call", "function": "graphistry.degree.write", "params": {}}
      ]
    },
    "__result__": {
      "type": "Ref",
      "ref": "g2",
      "chain": [
        {"type": "Node", "name": "n"},
        {"type": "Call", "function": "rows", "params": {"table": "nodes", "source": "n"}},
        {"type": "Call", "function": "select", "params": {"items": [["id", "n.id"], ["degree", "n.degree"]]}},
        {"type": "Call", "function": "order_by", "params": {"keys": [["degree", "desc"]]}}
      ]
    }
  }
}
```

The entire pipeline is a single `Let` message — one request, server-side
evaluation.

### Desugaring Reference

| GFQL Extension | Wire Equivalent |
|----------------|-----------------|
| `GRAPH { MATCH ... WHERE ... }` | `{"type": "Chain", "chain": [...], "where": [...]}` |
| `GRAPH { CALL graphistry.*.write() }` | `{"type": "Call", "function": "...", "params": {}}` |
| `GRAPH g = GRAPH { ... }` | Named `Let` binding — body is a `Chain` or `Call` |
| `USE g` | `Ref` with `"ref": "g"` — subsequent operations execute against `g`'s result |
| `USE g MATCH ... RETURN ...` | `Ref` with `"ref": "g"` and the query chain as its body |

## Best Practices

1. **Always include type fields**: Every object must have a `type`
2. **Use ISO formats**: Dates and times in ISO 8601
3. **Handle timezones consistently**: Include timezone for datetime values when precision matters (defaults to UTC)
4. **Validate before sending**: Use JSON Schema validation
5. **Handle unknown fields**: Ignore unrecognized fields for compatibility

## See Also

- {ref}`gfql-spec-language` - Language specification
- {ref}`gfql-spec-cypher-mapping` - Cypher to GFQL translation with wire protocol examples
