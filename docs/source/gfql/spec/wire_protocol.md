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

## JSON Schema

```{code-block} json
:caption: Complete JSON Schema for GFQL Wire Protocol

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://graphistry.com/schemas/gfql/wire-protocol.json",
  
  "definitions": {
    "Chain": {
      "type": "object",
      "properties": {
        "type": {"const": "Chain"},
        "chain": {
          "type": "array",
          "items": {"$ref": "#/definitions/Operation"},
          "minItems": 1
        }
      },
      "required": ["type", "chain"],
      "additionalProperties": false
    },
    
    "Operation": {
      "oneOf": [
        {"$ref": "#/definitions/NodeOperation"},
        {"$ref": "#/definitions/EdgeOperation"}
      ]
    },
    
    "NodeOperation": {
      "type": "object",
      "properties": {
        "type": {"const": "Node"},
        "filter_dict": {"$ref": "#/definitions/FilterDict"},
        "query": {"type": "string"},
        "name": {"type": "string"}
      },
      "required": ["type"],
      "additionalProperties": false
    },
    
    "EdgeOperation": {
      "type": "object",
      "properties": {
        "type": {"const": "Edge"},
        "direction": {
          "enum": ["forward", "reverse", "undirected"]
        },
        "edge_match": {"$ref": "#/definitions/FilterDict"},
        "edge_query": {"type": "string"},
        "hops": {
          "type": "integer",
          "minimum": 1,
          "default": 1
        },
        "to_fixed_point": {
          "type": "boolean",
          "default": false
        },
        "source_node_match": {"$ref": "#/definitions/FilterDict"},
        "source_node_query": {"type": "string"},
        "destination_node_match": {"$ref": "#/definitions/FilterDict"},
        "destination_node_query": {"type": "string"},
        "name": {"type": "string"}
      },
      "required": ["type", "direction"],
      "additionalProperties": false
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

## Error Handling

### Error Response Format

```json
{
  "type": "Error",
  "error_type": "ValidationError",
  "message": "Invalid predicate type 'GT' for string column 'name'",
  "details": {
    "column": "name",
    "predicate": "GT",
    "expected_types": ["string"],
    "actual_type": "comparison"
  }
}
```

### Error Types

1. **ValidationError**: Schema validation failed
2. **ParseError**: JSON parsing failed
3. **TypeError**: Type mismatch
4. **SemanticError**: Logical inconsistency
5. **UnsupportedError**: Feature not supported

## Protocol Extensions

### Future Considerations

The protocol is designed to support future extensions:

1. **Versioning**: Add `"version"` field for protocol versioning
2. **Metadata**: Add `"metadata": {}` for additional context
3. **Streaming**: Support for partial results
4. **Transactions**: Batch multiple operations
5. **Optimization Hints**: Engine-specific parameters

### Custom Predicates

New predicates can be added by extending the schema:

```json
{
  "definitions": {
    "CustomPredicate": {
      "type": "object",
      "properties": {
        "type": {"const": "CustomPredicateName"},
        ...custom fields...
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

## See Also

- {ref}`gfql-spec-language` - Language specification
- {ref}`gfql-spec-cypher-mapping` - Cypher translation guide
- {ref}`gfql-spec-cypher-mapping-wire` - Cypher to GFQL with wire protocol examples