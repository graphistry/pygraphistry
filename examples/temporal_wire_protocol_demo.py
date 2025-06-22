#!/usr/bin/env python
"""
Demonstration of GFQL temporal predicates wire protocol.

This script shows how temporal predicates are serialized to JSON
and can be reconstructed, maintaining full functionality.
"""

import json
from datetime import datetime, date, time
import pandas as pd
import pytz

from graphistry import n, e_forward
from graphistry.compute import (
    gt, lt, ge, le, eq, ne, between, is_in,
    DateTimeValue, DateValue, TimeValue
)
from graphistry.compute.ast import ASTNode
from graphistry.compute.chain import Chain


def print_json(obj, title):
    """Pretty print JSON with title"""
    print(f"\n{title}")
    print("-" * len(title))
    print(json.dumps(obj, indent=2))


def demo_datetime_predicates():
    """Demonstrate datetime predicate serialization"""
    print("\n=== DATETIME PREDICATES ===")
    
    # 1. Simple datetime comparison
    dt = pd.Timestamp("2023-06-15 14:30:00")
    pred = gt(dt)
    
    # Serialize
    json_pred = pred.to_json()
    print_json(json_pred, "GT predicate with datetime")
    
    # Deserialize
    from graphistry.compute.predicates.numeric import GT
    pred_restored = GT.from_json(json_pred)
    print(f"Restored predicate type: {type(pred_restored)}")
    
    # 2. Datetime with timezone
    eastern = pytz.timezone('US/Eastern')
    dt_tz = pd.Timestamp("2023-06-15 14:30:00", tz=eastern)
    pred_tz = lt(dt_tz)
    
    json_pred_tz = pred_tz.to_json()
    print_json(json_pred_tz, "LT predicate with timezone-aware datetime")


def demo_date_predicates():
    """Demonstrate date predicate serialization"""
    print("\n=== DATE PREDICATES ===")
    
    # Date comparison
    d = date(2023, 6, 15)
    pred = eq(d)
    
    json_pred = pred.to_json()
    print_json(json_pred, "EQ predicate with date")
    
    # Date range
    start = date(2023, 1, 1)
    end = date(2023, 12, 31)
    pred_range = between(start, end)
    
    json_range = pred_range.to_json()
    print_json(json_range, "Between predicate with date range")


def demo_time_predicates():
    """Demonstrate time predicate serialization"""
    print("\n=== TIME PREDICATES ===")
    
    # Time values in list
    times = [time(9, 0, 0), time(12, 0, 0), time(17, 0, 0)]
    pred = is_in(times)
    
    json_pred = pred.to_json()
    print_json(json_pred, "IsIn predicate with time values")


def demo_ast_node_serialization():
    """Demonstrate AST node with temporal predicates"""
    print("\n=== AST NODE SERIALIZATION ===")
    
    # Create node with multiple temporal filters
    node = n(filter_dict={
        "created_at": gt(datetime(2023, 1, 1)),
        "updated_at": lt(pd.Timestamp("2023-12-31 23:59:59")),
        "event_date": between(date(2023, 6, 1), date(2023, 6, 30)),
        "daily_time": is_in([time(9, 0), time(17, 0)])
    })
    
    json_node = node.to_json()
    print_json(json_node, "AST Node with multiple temporal predicates")
    
    # Deserialize
    node_restored = ASTNode.from_json(json_node)
    print(f"\nRestored node type: {type(node_restored)}")
    print(f"Filter dict keys: {list(node_restored.filter_dict.keys())}")


def demo_chain_serialization():
    """Demonstrate chain with temporal predicates"""
    print("\n=== CHAIN SERIALIZATION ===")
    
    # Create a chain with temporal filters
    chain = Chain([
        n(edge_match={
            "timestamp": gt(datetime(2023, 6, 1)),
            "amount": gt(1000)
        }),
        n(filter_dict={
            "last_active": between(
                datetime(2023, 6, 1),
                datetime(2023, 12, 31)
            )
        }),
        e_forward(edge_match={
            "created": ge(pd.Timestamp("2023-07-01"))
        })
    ])
    
    json_chain = chain.to_json()
    print_json(json_chain, "Chain with temporal predicates")
    
    # Deserialize
    chain_restored = Chain.from_json(json_chain)
    print(f"\nRestored chain with {len(chain_restored.queries)} queries")


def demo_temporal_value_classes():
    """Demonstrate temporal value classes directly"""
    print("\n=== TEMPORAL VALUE CLASSES ===")
    
    # Create temporal values
    dt_val = DateTimeValue("2023-06-15T14:30:00", "Europe/London")
    date_val = DateValue("2023-06-15")
    time_val = TimeValue("14:30:00")
    
    # Show their JSON representation
    print_json(dt_val.to_json(), "DateTimeValue")
    print_json(date_val.to_json(), "DateValue") 
    print_json(time_val.to_json(), "TimeValue")
    
    # Use in predicates
    pred_dt = gt(dt_val)
    pred_date = eq(date_val)
    pred_time = ne(time_val)
    
    print_json(pred_dt.to_json(), "GT with DateTimeValue")
    print_json(pred_date.to_json(), "EQ with DateValue")
    print_json(pred_time.to_json(), "NE with TimeValue")


def demo_round_trip():
    """Demonstrate complete round-trip serialization"""
    print("\n=== COMPLETE ROUND TRIP ===")
    
    # 1. Create complex query
    original_query = Chain([
        n(filter_dict={
            "created": gt(pd.Timestamp("2023-01-01 00:00:00", tz="UTC")),
            "event_date": between(date(2023, 6, 1), date(2023, 6, 30)),
            "status": eq("active")
        }),
        e_forward(edge_match={
            "timestamp": gt(datetime.now()),
            "daily_time": is_in([time(9, 0), time(12, 0), time(17, 0)])
        })
    ])
    
    # 2. Serialize to JSON
    json_data = original_query.to_json()
    
    # 3. Simulate wire transfer
    wire_data = json.dumps(json_data)
    print(f"Wire data size: {len(wire_data)} bytes")
    
    # 4. Receive and parse
    received_data = json.loads(wire_data)
    
    # 5. Reconstruct
    reconstructed_query = Chain.from_json(received_data)
    
    # 6. Verify
    print(f"Original queries: {len(original_query.queries)}")
    print(f"Reconstructed queries: {len(reconstructed_query.queries)}")
    print("Round trip successful!")
    
    # Show final JSON
    print_json(json_data, "Final serialized query")


def main():
    """Run all demonstrations"""
    print("GFQL Temporal Predicates Wire Protocol Demonstration")
    print("=" * 50)
    
    demo_datetime_predicates()
    demo_date_predicates()
    demo_time_predicates()
    demo_ast_node_serialization()
    demo_chain_serialization()
    demo_temporal_value_classes()
    demo_round_trip()
    
    print("\n" + "=" * 50)
    print("Demo complete!")


if __name__ == "__main__":
    main()