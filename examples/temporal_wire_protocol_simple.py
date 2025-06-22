#!/usr/bin/env python
"""Simple demonstration of temporal predicate wire protocol"""

import json
from datetime import datetime, date, time
import pandas as pd

# Example 1: DateTime GT predicate
print("=== Example 1: DateTime GT Predicate ===")
print("\nPython API:")
print("pred = gt(pd.Timestamp('2023-06-15 14:30:00'))")

print("\nWire Protocol (JSON):")
datetime_gt_json = {
    "type": "GT",
    "val": {
        "type": "datetime",
        "value": "2023-06-15T14:30:00",
        "timezone": "UTC"
    }
}
print(json.dumps(datetime_gt_json, indent=2))

# Example 2: Date Between predicate
print("\n\n=== Example 2: Date Between Predicate ===")
print("\nPython API:")
print("pred = between(date(2023, 6, 1), date(2023, 6, 30))")

print("\nWire Protocol (JSON):")
date_between_json = {
    "type": "Between",
    "lower": {
        "type": "date",
        "value": "2023-06-01"
    },
    "upper": {
        "type": "date", 
        "value": "2023-06-30"
    },
    "inclusive": True
}
print(json.dumps(date_between_json, indent=2))

# Example 3: Time IsIn predicate
print("\n\n=== Example 3: Time IsIn Predicate ===")
print("\nPython API:")
print("pred = is_in([time(9, 0), time(12, 0), time(17, 0)])")

print("\nWire Protocol (JSON):")
time_isin_json = {
    "type": "IsIn",
    "options": [
        {"type": "time", "value": "09:00:00"},
        {"type": "time", "value": "12:00:00"},
        {"type": "time", "value": "17:00:00"}
    ]
}
print(json.dumps(time_isin_json, indent=2))

# Example 4: AST Node with temporal predicates
print("\n\n=== Example 4: AST Node with Temporal Filters ===")
print("\nPython API:")
print("""node = n(filter_dict={
    "created_at": gt(datetime(2023, 1, 1)),
    "event_date": eq(date(2023, 6, 15)),
    "daily_time": between(time(9, 0), time(17, 0))
})""")

print("\nWire Protocol (JSON):")
ast_node_json = {
    "type": "ASTNode",
    "filter_dict": {
        "created_at": {
            "type": "GT",
            "val": {
                "type": "datetime",
                "value": "2023-01-01T00:00:00",
                "timezone": "UTC"
            }
        },
        "event_date": {
            "type": "EQ",
            "val": {
                "type": "date",
                "value": "2023-06-15"
            }
        },
        "daily_time": {
            "type": "Between",
            "lower": {"type": "time", "value": "09:00:00"},
            "upper": {"type": "time", "value": "17:00:00"},
            "inclusive": True
        }
    }
}
print(json.dumps(ast_node_json, indent=2))

# Example 5: Timezone-aware datetime
print("\n\n=== Example 5: Timezone-Aware DateTime ===")
print("\nPython API:")
print("pred = gt(pd.Timestamp('2023-06-15 09:00:00', tz='US/Eastern'))")

print("\nWire Protocol (JSON):")
tz_aware_json = {
    "type": "GT",
    "val": {
        "type": "datetime",
        "value": "2023-06-15T09:00:00",
        "timezone": "US/Eastern"
    }
}
print(json.dumps(tz_aware_json, indent=2))

# Example 6: Chain with temporal predicates
print("\n\n=== Example 6: Chain with Temporal Predicates ===")
print("\nPython API:")
print("""chain = g.chain([
    n(edge_match={"timestamp": gt(datetime(2023, 6, 1))}),
    n(filter_dict={"last_active": between(date(2023, 1, 1), date(2023, 12, 31))}),
    e_forward(edge_match={"created": ge(datetime.now())})
])""")

print("\nWire Protocol (JSON):")
chain_json = {
    "type": "Chain",
    "queries": [
        {
            "type": "ASTNode",
            "edge_match": {
                "timestamp": {
                    "type": "GT",
                    "val": {
                        "type": "datetime",
                        "value": "2023-06-01T00:00:00",
                        "timezone": "UTC"
                    }
                }
            }
        },
        {
            "type": "ASTNode",
            "filter_dict": {
                "last_active": {
                    "type": "Between",
                    "lower": {"type": "date", "value": "2023-01-01"},
                    "upper": {"type": "date", "value": "2023-12-31"},
                    "inclusive": True
                }
            }
        },
        {
            "type": "ASTEdge",
            "direction": "forward",
            "edge_match": {
                "created": {
                    "type": "GE",
                    "val": {
                        "type": "datetime",
                        "value": "2023-12-22T10:30:00",
                        "timezone": "UTC"
                    }
                }
            }
        }
    ]
}
print(json.dumps(chain_json, indent=2))

print("\n\nKey Points:")
print("1. Temporal values are tagged with type ('datetime', 'date', 'time')")
print("2. Timezone information is preserved for datetime values")
print("3. All temporal predicates serialize to JSON and can be deserialized")
print("4. The wire protocol maintains full fidelity of temporal information")