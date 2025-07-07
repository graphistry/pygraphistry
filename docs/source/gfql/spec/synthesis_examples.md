(gfql-spec-synthesis-examples)=

# GFQL Synthesis Examples

This document provides comprehensive examples for LLM-based GFQL synthesis, organized by use case and complexity level.

**Note**: The examples assume the following imports:
```python
import pandas as pd
from datetime import date, datetime, timedelta
import graphistry
```

## Basic Patterns

### Single Node Queries

**Natural Language**: "Find all active users"
```python
g.chain([n({"status": "active", "type": "user"})])
```

**Natural Language**: "Show me nodes created today"
```python
# Assumes: from datetime import date
g.chain([n({"created_date": eq(date.today())})])
```

**Natural Language**: "Get all nodes with high priority"
```python
g.chain([n({"priority": gt(8)})])
```

### Simple Traversals

**Natural Language**: "Find direct connections of node A"
```python
g.chain([n({"id": "A"}), e(), n()])
```

**Natural Language**: "Show what Alice purchased"
```python
g.chain([
    n({"name": "Alice"}),
    e_forward({"type": "purchased"}),
    n({"type": "product"})
])
```

**Natural Language**: "Find who reports to Bob"
```python
g.chain([
    n({"name": "Bob"}),
    e_reverse({"type": "reports_to"}),
    n({"type": "employee"})
])
```

## User 360 Patterns

### Customer Journey Analysis

**Natural Language**: "Show me all touchpoints for customer C123 in the last 30 days"
```python
g.chain([
    n({"customer_id": "C123"}),
    e_forward({
        "type": is_in(["purchase", "support", "browse", "email"]),
        "timestamp": gt(pd.Timestamp.now() - pd.Timedelta(days=30))
    }),
    n(name="touchpoints")
])
```

### Customer Segmentation

**Natural Language**: "Find high-value customers who made recent purchases"
```python
g.chain([
    n({"customer_tier": "gold", "type": "customer"}),
    e_forward({
        "type": "purchase",
        "date": gt(pd.Timestamp.now() - pd.Timedelta(days=7)),
        "amount": gt(500)
    }),
    n({"type": "product", "category": is_in(["electronics", "luxury"])})
])
```

### Cross-Sell Opportunities

**Natural Language**: "Find products frequently bought together with product P123"
```python
g.chain([
    n({"product_id": "P123"}),
    e_reverse({"type": "purchased"}),
    n({"type": "customer"}, name="buyers"),
    e_forward({"type": "purchased", "date": gt(pd.Timestamp.now() - pd.Timedelta(days=90))}),
    n({"type": "product", "product_id": ne("P123")}, name="cross_sell")
])
```

### Customer Network Effects

**Natural Language**: "Find customers influenced by top reviewers"
```python
g.chain([
    n({"type": "customer", "reviewer_rank": lt(100)}),
    e_forward({"type": "reviewed", "rating": ge(4)}),
    n({"type": "product"}),
    e_reverse({"type": "viewed", "action": "after_review"}),
    n({"type": "customer"}, name="influenced")
])
```

## Cyber Security Patterns

### Lateral Movement Detection

**Natural Language**: "Track potential lateral movement from compromised accounts"
```python
g.chain([
    n({"type": "account", "status": "compromised"}),
    e_forward({
        "type": "login",
        "timestamp": gt(pd.Timestamp.now() - pd.Timedelta(hours=24)),
        "success": True
    }, hops=3),
    n({"type": "system", "criticality": is_in(["high", "critical"])}, name="at_risk")
])
```

### Anomalous Access Patterns

**Natural Language**: "Find unusual access to sensitive data outside business hours"
```python
g.chain([
    n({"type": "user", "department": ne("IT")}),
    e_forward({
        "type": "access",
        "resource_type": "sensitive",
        "hour": is_in(list(range(0, 6)) + list(range(22, 24)))
    }),
    n({"type": "resource", "classification": is_in(["confidential", "secret"])})
])
```

### Command and Control Detection

**Natural Language**: "Identify potential C2 communication patterns"
```python
g.chain([
    n({"type": "endpoint", "os": is_in(["windows", "linux"])}),
    e_forward({
        "type": "network_connection",
        "port": is_in([443, 8443, 8080]),
        "bytes_sent": gt(1000000),
        "duration": gt(3600)
    }),
    n({"type": "external_ip", "reputation": lt(50)}, name="suspicious_c2")
])
```

### Data Exfiltration Risk

**Natural Language**: "Find paths from compromised users to sensitive data stores"
```python
g.chain([
    n({"type": "user", "risk_score": gt(80)}),
    e_forward(to_fixed_point=True),
    n({"type": "database", "contains_pii": True}, name="exfil_risk")
])
```

## Complex Business Queries

### Supply Chain Analysis

**Natural Language**: "Find all suppliers affected by the NYC warehouse disruption"
```python
g.chain([
    n({"type": "warehouse", "location": "NYC", "status": "disrupted"}),
    e_reverse({"type": "ships_to"}),
    n({"type": "supplier"}),
    e_forward({"type": "supplies"}, hops=2),
    n({"type": "product", "critical": True}, name="affected_products")
])
```

### Fraud Ring Detection

**Natural Language**: "Identify connected accounts with suspicious transaction patterns"
```python
g.chain([
    n({
        "type": "account",
        "fraud_score": gt(0.7),
        "created": gt(pd.Timestamp.now() - pd.Timedelta(days=30))
    }),
    e({
        "type": is_in(["transfer", "shared_device", "same_ip"]),
        "timestamp": gt(pd.Timestamp.now() - pd.Timedelta(days=7))
    }, to_fixed_point=True),
    n({"type": "account"}, name="fraud_ring")
])
```

### Influence Network Analysis

**Natural Language**: "Find key influencers in the communication network"
```python
# First, find highly connected nodes
g.chain([
    n({"type": "person"}),
    e({"type": "communicates"}, hops=2),
    n(name="reachable")
])
# Then post-process to find nodes with high reachability
```

## Code Golf Examples

### Concise Patterns

**Natural Language**: "Friends of friends"
```python
g.chain([n({"id": "Bob"}), e({"type": "friend"}, hops=2)])
```

**Natural Language**: "All paths between A and B"
```python
g.chain([n({"id": "A"}), e(to_fixed_point=True), n({"id": "B"})])
```

**Natural Language**: "Recent high-value transactions"
```python
g.chain([n(), e({"amount": gt(1000), "date": gt(pd.Timestamp.now() - pd.Timedelta(7))})])
```

**Natural Language**: "Compromised device spread"
```python
g.chain([n({"compromised": True}), e(to_fixed_point=True), n(name="infected")])
```

### One-Liners

**Natural Language**: "Active users' recent actions"
```python
g.chain([n({"active": True}), e({"timestamp": gt(pd.Timestamp.now() - pd.Timedelta(1))})])
```

**Natural Language**: "Products in same category as P123"
```python
g.chain([n({"id": "P123"}), e_reverse({"type": "in_category"}), e_forward(), n({"id": ne("P123")})])
```

## Error Cases

### Common Synthesis Errors

**Error**: Using unsupported Cypher features
```python
# Incorrect - OPTIONAL MATCH is not supported
# OPTIONAL MATCH (n)-[r]->(m)

# Correct - Handle optionality in post-processing
g.chain([n(), e_forward()])
# Then check for nulls in results
```

**Error**: Aggregations in chain
```python
# Wrong - Can't aggregate in chain
# g.chain([n(), e(), count()])

# Correct - Aggregate after
result = g.chain([n(), e()])
count = len(result._edges)
```

**Error**: Wrong predicate types
```python
# Wrong - String predicate on number
# n({"age": contains("3")})

# Correct - Use numeric predicate
n({"age": gt(30)})
```

### Validation Examples

**Schema Validation**:
```python
# Given schema: nodes have ['id', 'type', 'name']
# Wrong:
g.chain([n({"username": "Alice"})])  # 'username' doesn't exist

# Correct:
g.chain([n({"name": "Alice"})])
```

**Type Validation**:
```python
# Given: 'created' is a datetime column
# Wrong:
n({"created": gt("2024-01-01")})  # String instead of datetime

# Correct:
n({"created": gt(pd.Timestamp("2024-01-01"))})
```

## Synthesis Tips

### For Natural Language Processing

1. **Identify Entities**: Look for nouns that map to node types
2. **Identify Relationships**: Look for verbs that map to edge types
3. **Extract Filters**: Look for adjectives and conditions
4. **Determine Direction**: "from", "to", "by" indicate direction
5. **Handle Time**: "recent", "last N days", "since" map to temporal predicates

### For Cypher Translation

1. **Pattern Matching**: `(a)-[r]->(b)` → `chain([n(), e_forward(), n()])`
2. **WHERE Embedding**: Move WHERE conditions into matchers
3. **Path Length**: `*N` → `hops=N`, `*` → `to_fixed_point=True`
4. **Labels**: `:Label` → `{"type": "Label"}` or `{"label": "Label"}`

### For Optimization

1. **Filter Early**: Put selective filters in first operations
2. **Limit Hops**: Use specific hop counts when possible
3. **Name Results**: Use `name` parameter for important nodes/edges
4. **Combine Filters**: Put multiple conditions in one operation

## See Also

- [GFQL Validation for LLMs](../validation/llm.rst) - LLM integration patterns
- {ref}`gfql-spec-language` - Language specification
- {ref}`gfql-spec-wire-protocol` - Wire protocol
- {ref}`gfql-spec-cypher-mapping` - Cypher translation guide