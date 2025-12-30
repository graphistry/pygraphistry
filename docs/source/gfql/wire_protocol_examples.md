# Temporal Predicates Wire Protocol Reference

This document provides a comprehensive reference for how temporal predicates serialize to JSON in the GFQL wire protocol. The wire protocol enables interoperability between Python and other systems.

## Overview

The wire protocol uses tagged dictionaries to preserve type information during JSON serialization. This enables:
- Cross-language compatibility
- Configuration-driven predicate creation
- Network transport of queries
- Storage of predicate definitions

**Key Concept**: Wire protocol dictionaries can be used directly in the Python API:

```python
# These are equivalent:
pred1 = gt(100)
pred2 = gt(pd.Timestamp("2023-01-01"))
pred3 = gt({"type": "datetime", "value": "2023-01-01T00:00:00", "timezone": "UTC"})
```

## 1. DateTime Comparisons

### Python API
```python
import pandas as pd
from datetime import datetime
from graphistry import n
from graphistry.compute import gt, between

# Using pandas Timestamp
filter1 = n(filter_dict={
    "created_at": gt(pd.Timestamp("2023-01-01 12:00:00"))
})

# Using Python datetime
filter2 = n(edge_match={
    "timestamp": between(
        datetime(2023, 1, 1),
        datetime(2023, 12, 31, 23, 59, 59)
    )
})
```

### Wire Protocol (JSON)
```json
// GT with datetime
{
    "type": "ASTNode",
    "filter_dict": {
        "created_at": {
            "type": "GT",
            "val": {
                "type": "datetime",
                "value": "2023-01-01T12:00:00",
                "timezone": "UTC"
            }
        }
    }
}

// Between with datetime range
{
    "type": "ASTNode", 
    "edge_match": {
        "timestamp": {
            "type": "Between",
            "lower": {
                "type": "datetime",
                "value": "2023-01-01T00:00:00",
                "timezone": "UTC"
            },
            "upper": {
                "type": "datetime",
                "value": "2023-12-31T23:59:59",
                "timezone": "UTC"
            },
            "inclusive": true
        }
    }
}
```

### Round-trip Example
```python
# Create predicate
from graphistry.compute import gt
pred = gt(pd.Timestamp("2023-01-01 12:00:00"))

# Serialize to JSON
json_data = pred.to_json()
print(json_data)
# Output: {
#     'type': 'GT',
#     'val': {
#         'type': 'datetime',
#         'value': '2023-01-01T12:00:00',
#         'timezone': 'UTC'
#     }
# }

# Deserialize from JSON
from graphistry.compute.predicates.numeric import GT
pred2 = GT.from_json(json_data)
# pred2 is functionally equivalent to pred
```

## 2. Date-Only Comparisons

### Python API
```python
from datetime import date
from graphistry.compute import eq, ge

# Date equality
filter1 = n(filter_dict={
    "event_date": eq(date(2023, 6, 15))
})

# Date range check
filter2 = n(filter_dict={
    "start_date": ge(date(2023, 1, 1))
})
```

### Wire Protocol (JSON)
```json
// Date equality
{
    "type": "ASTNode",
    "filter_dict": {
        "event_date": {
            "type": "EQ",
            "val": {
                "type": "date",
                "value": "2023-06-15"
            }
        }
    }
}

// Date greater than or equal
{
    "type": "ASTNode",
    "filter_dict": {
        "start_date": {
            "type": "GE", 
            "val": {
                "type": "date",
                "value": "2023-01-01"
            }
        }
    }
}
```

## 3. Time-Only Comparisons

### Python API
```python
from datetime import time
from graphistry.compute import is_in, between

# Specific times
filter1 = n(filter_dict={
    "event_time": is_in([
        time(9, 0, 0),
        time(12, 0, 0),
        time(17, 0, 0)
    ])
})

# Time range
filter2 = n(edge_match={
    "daily_schedule": between(
        time(9, 0, 0),
        time(17, 30, 0)
    )
})
```

### Wire Protocol (JSON)
```json
// IsIn with times
{
    "type": "ASTNode",
    "filter_dict": {
        "event_time": {
            "type": "IsIn",
            "options": [
                {"type": "time", "value": "09:00:00"},
                {"type": "time", "value": "12:00:00"},
                {"type": "time", "value": "17:00:00"}
            ]
        }
    }
}

// Time range
{
    "type": "ASTNode",
    "edge_match": {
        "daily_schedule": {
            "type": "Between",
            "lower": {"type": "time", "value": "09:00:00"},
            "upper": {"type": "time", "value": "17:30:00"},
            "inclusive": true
        }
    }
}
```

## 4. Timezone-Aware DateTime

### Python API
```python
import pytz
from graphistry.compute import DateTimeValue, gt

# Using timezone-aware timestamp
eastern = pytz.timezone('US/Eastern')
filter1 = n(filter_dict={
    "timestamp": gt(
        pd.Timestamp("2023-01-01 09:00:00", tz=eastern)
    )
})

# Using DateTimeValue with explicit timezone
dt_val = DateTimeValue("2023-01-01T09:00:00", "US/Eastern")
filter2 = n(filter_dict={
    "timestamp": gt(dt_val)
})
```

### Wire Protocol (JSON)
```json
// Timezone-aware datetime
{
    "type": "ASTNode",
    "filter_dict": {
        "timestamp": {
            "type": "GT",
            "val": {
                "type": "datetime",
                "value": "2023-01-01T09:00:00",
                "timezone": "US/Eastern"
            }
        }
    }
}
```

## 5. Complex Chain with Temporal Predicates

### Python API
```python
from graphistry import n, e_forward
from graphistry.compute import gt, eq, between
from datetime import datetime, timedelta

# Multi-hop query with temporal filters
chain = g.gfql([
    # Recent transactions
    n(edge_match={
        "timestamp": gt(datetime.now() - timedelta(days=7)),
        "amount": gt(1000)
    }),
    # To active accounts
    n(filter_dict={
        "status": eq("active"),
        "last_login": between(
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
    }),
    # Outgoing transfers
    e_forward(edge_match={
        "type": eq("transfer"),
        "timestamp": gt(datetime.now() - timedelta(days=1))
    })
])
```

### Wire Protocol (JSON)
```json
{
    "type": "Chain",
    "queries": [
        {
            "type": "ASTNode",
            "edge_match": {
                "timestamp": {
                    "type": "GT",
                    "val": {
                        "type": "datetime",
                        "value": "2023-12-18T10:30:00",
                        "timezone": "UTC"
                    }
                },
                "amount": {
                    "type": "GT",
                    "val": 1000
                }
            }
        },
        {
            "type": "ASTNode",
            "filter_dict": {
                "status": {
                    "type": "EQ",
                    "val": "active"
                },
                "last_login": {
                    "type": "Between",
                    "lower": {
                        "type": "datetime",
                        "value": "2023-11-25T10:30:00",
                        "timezone": "UTC"
                    },
                    "upper": {
                        "type": "datetime",
                        "value": "2023-12-25T10:30:00",
                        "timezone": "UTC"
                    },
                    "inclusive": true
                }
            }
        },
        {
            "type": "ASTEdge",
            "direction": "forward",
            "edge_match": {
                "type": {
                    "type": "EQ",
                    "val": "transfer"
                },
                "timestamp": {
                    "type": "GT",
                    "val": {
                        "type": "datetime",
                        "value": "2023-12-24T10:30:00",
                        "timezone": "UTC"
                    }
                }
            }
        }
    ]
}
```

## 6. Temporal Value Classes Direct Usage

### Python API
```python
from graphistry.compute import (
    DateTimeValue, DateValue, TimeValue,
    temporal_value_from_json, gt
)

# Create temporal values
dt_val = DateTimeValue("2023-06-15T14:30:00", "Europe/London")
date_val = DateValue("2023-06-15")
time_val = TimeValue("14:30:00")

# Use in predicates
filter1 = n(filter_dict={"timestamp": gt(dt_val)})
filter2 = n(filter_dict={"event_date": eq(date_val)})
filter3 = n(filter_dict={"daily_time": eq(time_val)})

# Create from JSON
json_dt = {
    "type": "datetime",
    "value": "2023-06-15T14:30:00",
    "timezone": "Europe/London"
}
dt_from_json = temporal_value_from_json(json_dt)
```

### Wire Protocol (JSON)
```json
// DateTimeValue serialization
{
    "type": "datetime",
    "value": "2023-06-15T14:30:00",
    "timezone": "Europe/London"
}

// DateValue serialization
{
    "type": "date",
    "value": "2023-06-15"
}

// TimeValue serialization
{
    "type": "time",
    "value": "14:30:00"
}
```

## 7. Full Round-Trip Example

```python
# 1. Create a complex query with temporal predicates
from graphistry import n, Chain
from graphistry.compute import gt, between, is_in
from datetime import datetime, date, time
import pandas as pd

query = Chain([
    n(filter_dict={
        "created": gt(pd.Timestamp("2023-01-01")),
        "event_date": between(date(2023, 6, 1), date(2023, 6, 30)),
        "event_time": is_in([time(9, 0), time(12, 0), time(17, 0)])
    })
])

# 2. Serialize to JSON
json_query = query.to_json()
print(json_query)

# 3. Send over wire (simulated)
import json
wire_data = json.dumps(json_query)
received_data = json.loads(wire_data)

# 4. Deserialize on receiving end
from graphistry.compute.ast import Chain
reconstructed_query = Chain.from_json(received_data)

# 5. Apply to graph data
result = g.gfql(reconstructed_query.queries)
```

## Wire Protocol Structure

### Temporal Value Types

All temporal values in the wire protocol follow this pattern:

```typescript
// DateTime with timezone
interface DateTimeWire {
    type: "datetime";
    value: string;      // ISO 8601 format
    timezone?: string;  // IANA timezone (default: "UTC")
}

// Date only
interface DateWire {
    type: "date";
    value: string;      // YYYY-MM-DD format
}

// Time only  
interface TimeWire {
    type: "time";
    value: string;      // HH:MM:SS[.ffffff] format
}
```

### Predicate Structure

Predicates containing temporal values serialize as:

```typescript
interface TemporalPredicate {
    type: "GT" | "LT" | "GE" | "LE" | "EQ" | "NE";
    val: DateTimeWire | DateWire | TimeWire;
}

interface BetweenPredicate {
    type: "Between";
    lower: DateTimeWire | DateWire | TimeWire;
    upper: DateTimeWire | DateWire | TimeWire;
    inclusive: boolean;
}

interface IsInPredicate {
    type: "IsIn";
    options: Array<DateTimeWire | DateWire | TimeWire>;
}
```

## Key Points

1. **Type Safety**: Raw strings are rejected in the Python API to avoid ambiguity
2. **Automatic Conversion**: Python datetime objects are automatically converted to appropriate temporal values
3. **Timezone Preservation**: Timezone information is preserved through serialization
4. **Tagged Format**: JSON uses tagged dictionaries to preserve type information
5. **Direct Usage**: Wire protocol dicts can be passed directly to Python predicates

## Error Handling

```python
# Raw strings raise ValueError
try:
    filter_raw = n(filter_dict={"date": gt("2023-01-01")})
except ValueError as e:
    print(e)
    # Output: Raw string '2023-01-01' is ambiguous. Use:
    #   - gt(pd.Timestamp('2023-01-01')) for datetime
    #   - gt({'type': 'datetime', 'value': '2023-01-01T00:00:00'}) for explicit type

# Valid approaches
filter1 = n(filter_dict={"date": gt(pd.Timestamp("2023-01-01"))})
filter2 = n(filter_dict={"date": gt(datetime(2023, 1, 1))})
filter3 = n(filter_dict={"date": gt({"type": "datetime", "value": "2023-01-01T00:00:00", "timezone": "UTC"})})
```

## Performance Considerations

- Temporal predicates leverage pandas' optimized datetime operations
- Timezone conversions are handled efficiently
- For large datasets, ensure datetime columns are properly typed (not object dtype)
- Use `pd.Timestamp` for best performance when creating many predicates programmatically

## Collections and Wire Protocol

Collections accept GFQL wire protocol dicts inside the `expr` field for set definitions.
You can pass the dict directly or through the helper constructors:

```python
import graphistry

collections = [
    graphistry.collection_set(
        expr={
            "type": "gfql_chain",
            "gfql": [
                {"type": "Node", "filter_dict": {"status": {"type": "EQ", "val": "purchased"}}}
            ],
        },
        name="Purchasers",
        node_color="#00BFFF",
    )
]
g.collections(collections=collections)
```

See the [Collections guide](../visualization/layout/settings.html) for full usage details and
intersection examples.
