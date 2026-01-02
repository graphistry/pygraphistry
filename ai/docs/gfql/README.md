# GFQL AI Assistant Guide

Guide for AI assistants working with GFQL (Graph Frame Query Language) in PyGraphistry.

## 🎯 Quick Reference

### Process & Checklists
- [`calls_checklist.md`](./calls_checklist.md) — Required steps for exposing or updating GFQL `call()` functions.
- [`predicates_checklist.md`](./predicates_checklist.md) — End-to-end checklist for predicate implementations.
- [`conformance.md`](./conformance.md) — Cypher TCK conformance harness and CI wiring.
- [`../prompts/GFQL_LLM_GUIDE_MAINTENANCE.md`](../prompts/GFQL_LLM_GUIDE_MAINTENANCE.md) — Guidance for keeping AI assistants aligned with GFQL changes.

### Essential GFQL Operations
```python
# Node matching
n()                                    # All nodes
n({"type": "person"})                 # Filter by property
n({"age": gt(30)})                    # With predicate
n(name="result")                      # Named results

# Edge traversal
e_forward()                           # Forward direction
e_reverse()                           # Reverse direction
e() or e_undirected()                # Both directions
e_forward(hops=2)                     # Multi-hop
e_forward(to_fixed_point=True)        # All reachable

# Chaining
g.chain([n(), e_forward(), n()])      # Pattern matching
```

### Key Predicates
- Comparison: `gt()`, `lt()`, `ge()`, `le()`, `eq()`, `ne()`
- Membership: `is_in([...])`
- Range: `between(lower, upper)`
- String: `contains()`, `startswith()`, `endswith()`
- Null: `is_null()`, `not_null()`
- Temporal: `is_month_start()`, `is_year_end()`, etc.

### Performance Tips
- Filter early in the chain
- Use specific hop counts vs `to_fixed_point`
- Prefer `filter_dict` over `query` strings
- Use appropriate engine: `pandas` (CPU) or `cudf` (GPU)

### Validation (Built-in)
```python
# Automatic syntax validation
chain = Chain([n(), e_forward(hops=-1)])  # Raises GFQLTypeError

# Schema validation at runtime
g.chain([n({'missing': 'value'})])  # Raises GFQLSchemaError

# Pre-execution validation
g.chain(ops, validate_schema=True)  # Validate before execution

# Collect all errors
errors = chain.validate(collect_all=True)
```

Error types:
- `GFQLSyntaxError` (E1xx): Structural issues
- `GFQLTypeError` (E2xx): Type mismatches  
- `GFQLSchemaError` (E3xx): Missing columns, wrong types

## 📋 When to Use GFQL

### Use GFQL When
- Performing graph traversals or path queries
- Finding patterns in connected data
- Need efficient multi-hop operations
- Working with node/edge dataframes

### Use Pandas/Aggregations When
- Need sorting (`sort_values()`)
- Need limiting (`head()`, `tail()`)
- Aggregating results (`groupby()`, `count()`)
- Complex transformations

## 🚀 Common Patterns

### User 360 Query
```python
# Customer's recent activity
g.chain([
    n({"customer_id": "C123"}),
    e_forward({
        "type": is_in(["purchase", "view", "support"]),
        "timestamp": gt(pd.Timestamp.now() - pd.Timedelta(days=30))
    })
])
```

### Cyber Security Pattern
```python
# Lateral movement detection
g.chain([
    n({"status": "compromised"}),
    e_forward({"type": "login", "success": True}, hops=3),
    n({"criticality": "high"}, name="at_risk")
])
```

### Business Intelligence
```python
# Cross-sell opportunities
g.chain([
    n({"product_id": "P123"}),
    e_reverse({"type": "purchased"}),
    n({"type": "customer"}),
    e_forward({"type": "purchased"}),
    n({"product_id": ne("P123")}, name="also_bought")
])
```

## 🔧 Code Style Guidelines

### Preferred Style
```python
# ✅ Good - Clean, code-golfed chains
g.chain([n({"type": "user"}), e({"active": True}), n()])

# ❌ Avoid - Overly verbose
result = g.chain([
    n(filter_dict={"type": "user"}),
    e_forward(edge_match={"active": True}, hops=1),
    n(filter_dict={})
])
```

### Naming Conventions
- Use descriptive names for `name` parameters
- Keep filter keys consistent with dataframe columns
- Use snake_case for all identifiers

## 🐛 Common Errors and Fixes

### Schema Errors
```python
# ❌ Wrong - Column doesn't exist
n({"username": "Alice"})

# ✅ Fix - Use correct column name
n({"name": "Alice"})
```

### Type Errors
```python
# ❌ Wrong - String predicate on number
n({"age": contains("30")})

# ✅ Fix - Use numeric predicate
n({"age": gt(30)})
```

### Temporal Errors
```python
# ❌ Wrong - Raw string for datetime
n({"created": gt("2024-01-01")})

# ✅ Fix - Use proper datetime
n({"created": gt(pd.Timestamp("2024-01-01"))})
```

## 📝 Natural Language to GFQL

### Translation Patterns
- "recent" → `gt(pd.Timestamp.now() - pd.Timedelta(days=N))`
- "between X and Y" → `between(X, Y)`
- "any of" → `is_in([...])`
- "connected to" → `e()` or `e_undirected()`
- "from X to Y" → X with `e_forward()` to Y
- "within N hops" → `hops=N`

### Example Translations

**NL**: "Find all employees who report to managers in NYC"
```python
g.chain([
    n({"type": "employee"}),
    e_forward({"type": "reports_to"}),
    n({"type": "manager", "office": "NYC"})
])
```

**NL**: "Show me high-value customers from last week"
```python
g.chain([
    n({"customer_tier": "high_value"}),
    e_forward({
        "type": "purchase",
        "date": gt(pd.Timestamp.now() - pd.Timedelta(days=7))
    })
])
```

## 🔄 Cypher to GFQL

### Basic Mappings
| Cypher | GFQL |
|--------|------|
| `(n)` | `n()` |
| `(n:Label)` | `n({"type": "Label"})` |
| `-[r]->` | `e_forward()` |
| `<-[r]-` | `e_reverse()` |
| `-[r*2]-` | `e_forward(hops=2)` |
| `WHERE n.prop = val` | `n({"prop": val})` |

### Unsupported in GFQL
- `OPTIONAL MATCH` - Handle nulls in post-processing
- `WITH` clauses - Use intermediate chains
- `ORDER BY/LIMIT` - Use pandas after
- `CREATE/DELETE` - GFQL is read-only

## 🧪 Validation Checklist

Before generating GFQL:
1. ✓ Check column names exist in schema
2. ✓ Verify predicate types match column types
3. ✓ Ensure temporal values use proper types
4. ✓ Validate operation names (n, e_forward, etc.)
5. ✓ Check chain structure is valid

## 📚 Additional Resources

### For AI Assistants

- **`predicates_checklist.md`** - Complete implementation guide for adding/modifying GFQL predicates
  - Covers all 16 integration points (implementation, JSON, validators, docs)
  - Real-world examples using IsIn predicate
  - Common patterns and mistakes to avoid

### Full Specifications

- Full specifications in: `AI_PROGRESS/gfql_llm_specs/`
  - `gfql_language_spec.md` - Complete language specification
  - `gfql_wire_protocol_spec.md` - JSON wire format
  - `cypher_to_gfql_mapping_spec.md` - Cypher translation

## 🎯 Key Takeaways

1. **GFQL is functional**: Chain operations, don't mutate
2. **Filter early**: Put selective conditions first
3. **Think patterns**: Focus on graph patterns, not procedures
4. **Post-process**: Use pandas for sorting/aggregating
5. **Code golf**: Keep queries concise and elegant
