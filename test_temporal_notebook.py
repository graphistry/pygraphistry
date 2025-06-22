#!/usr/bin/env python
"""Test script to verify temporal predicates notebook code works"""

import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import pytz

# Test 1: Basic imports
print("Testing imports...")
try:
    import graphistry
    from graphistry import n, e_forward
    from graphistry.compute import (
        gt, lt, ge, le, eq, ne, between, is_in,
        DateTimeValue, DateValue, TimeValue
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

# Test 2: Create sample data
print("\nTesting data creation...")
np.random.seed(42)

# Create nodes (accounts)
n_accounts = 100
accounts_df = pd.DataFrame({
    'account_id': [f'ACC_{i:04d}' for i in range(n_accounts)],
    'account_type': np.random.choice(['checking', 'savings', 'business'], n_accounts),
    'created_date': pd.date_range('2020-01-01', periods=n_accounts, freq='W'),
    'last_active': pd.date_range('2023-01-01', periods=n_accounts, freq='D') + 
                   pd.to_timedelta(np.random.randint(0, 365, n_accounts), unit='D')
})

# Create edges (transactions)
n_transactions = 500
transactions_df = pd.DataFrame({
    'transaction_id': [f'TXN_{i:06d}' for i in range(n_transactions)],
    'source': np.random.choice(accounts_df['account_id'], n_transactions),
    'target': np.random.choice(accounts_df['account_id'], n_transactions),
    'amount': np.random.exponential(100, n_transactions).round(2),
    'timestamp': pd.date_range('2023-01-01', periods=n_transactions, freq='H') + 
                 pd.to_timedelta(np.random.randint(0, 8760, n_transactions), unit='H'),
    'transaction_time': [time(np.random.randint(0, 24), np.random.randint(0, 60)) 
                        for _ in range(n_transactions)],
    'transaction_type': np.random.choice(['transfer', 'payment', 'deposit'], n_transactions)
})

print(f"✓ Created {len(accounts_df)} accounts and {len(transactions_df)} transactions")

# Test 3: Create graph
print("\nTesting graph creation...")
g = graphistry.edges(transactions_df, 'source', 'target').nodes(accounts_df, 'account_id')
print(f"✓ Graph: {len(g._nodes)} nodes, {len(g._edges)} edges")

# Test 4: DateTime filtering
print("\nTesting datetime filtering...")
cutoff_date = datetime(2023, 7, 1)
recent_transactions = g.chain([
    e_forward(edge_match={"timestamp": gt(pd.Timestamp(cutoff_date))})
])
print(f"✓ Transactions after {cutoff_date}: {len(recent_transactions._edges)}")

# Test 5: Date range filtering
print("\nTesting date range filtering...")
march_2023 = g.chain([
    e_forward(edge_match={
        "timestamp": between(
            datetime(2023, 3, 1),
            datetime(2023, 3, 31, 23, 59, 59)
        )
    })
])
print(f"✓ Transactions in March 2023: {len(march_2023._edges)}")

# Test 6: Time filtering with IsIn
print("\nTesting time filtering...")
on_the_hour_times = [time(h, 0, 0) for h in range(24)]
on_hour_transactions = g.chain([
    e_forward(edge_match={
        "transaction_time": is_in(on_the_hour_times)
    })
])
print(f"✓ Transactions on the hour: {len(on_hour_transactions._edges)}")

# Test 7: Temporal value classes
print("\nTesting temporal value classes...")
dt_value = DateTimeValue("2023-06-15T14:30:00", "UTC")
date_value = DateValue("2023-06-15")
time_value = TimeValue("14:30:00")

specific_datetime = g.chain([
    e_forward(edge_match={"timestamp": gt(dt_value)})
])
print(f"✓ Transactions after {dt_value.value}: {len(specific_datetime._edges)}")

# Test 8: Complex query
print("\nTesting complex temporal query...")
thirty_days_ago = datetime.now() - timedelta(days=30)
money_flow = g.chain([
    e_forward(edge_match={
        "timestamp": gt(pd.Timestamp(thirty_days_ago))
    }),
    n(filter_dict={"account_type": "business"})
])
print(f"✓ Money flow pattern found: {len(money_flow._nodes)} business accounts")

print("\n✅ All tests passed! The notebook code works correctly.")