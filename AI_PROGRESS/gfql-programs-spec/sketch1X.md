# GFQL Programs Specification v1.X

## Executive Summary

GFQL Programs extends PyGraphistry's Graph Query Language from single-chain operations to a full-fledged graph programming environment. This specification enables users to compose multiple graphs, load remote data, apply transformations, and combine results - all within declarative GFQL expressions without requiring Python code.

Key capabilities:
- **DAG Composition**: Express complex multi-graph workflows as directed acyclic graphs
- **Remote Graph Loading**: Access saved graphs directly from GFQL
- **Graph Combinators**: Union, intersect, and subtract graphs with policy controls
- **Call Operations**: Invoke PyGraphistry's rich transformation methods
- **Reference System**: Navigate nested graph structures with dotted paths

This specification incorporates comprehensive security controls, resource management, and error handling based on thorough analysis of implementation requirements.

## Core Concepts

### 1. QueryDAG - The Foundation

QueryDAG (called ChainGraph in Python API) introduces a binding environment where multiple graphs can be named, referenced, and composed:

```json
{
  "type": "QueryDAG",
  "graph": {
    "customers": {"type": "RemoteGraph", "dataset_id": "cust_2024"},
    "transactions": {"type": "RemoteGraph", "dataset_id": "tx_2024"},
    "risky": {
      "type": "Chain",
      "ref": "customers",
      "chain": [
        {"type": "Node", "filter_dict": {"risk_score": {"type": "GT", "val": 0.8}}}
      ]
    },
    "connected": {
      "type": "Chain", 
      "ref": "risky",
      "chain": [
        {"type": "Edge", "direction": "forward", "edge_match": {"type": "transaction"}}
      ]
    }
  },
  "output": "connected"
}
```

### 2. Reference Resolution

References follow lexical scoping rules with dotted path syntax for disambiguation:

- **Simple references**: `"ref": "customers"` - searches from current scope outward
- **Dotted references**: `"ref": "fraud.analysis.results"` - explicit path through nested DAGs
- **Scoping rules**: Closest binding wins, statically resolvable at parse time

### 3. Execution Model

The system maintains these key principles:
- **Lazy evaluation** where possible to optimize resource usage
- **Parallel execution** of independent DAG branches
- **Resource limits** enforced at every level
- **Fail-fast validation** with clear error messages

## Feature Specifications

### 1. DAG Composition

#### Wire Protocol

```json
{
  "type": "QueryDAG",
  "graph": {
    "binding_name": {
      "type": "Chain" | "QueryDAG" | "RemoteGraph" | "GraphCombinator" | "Call",
      ...operation_specific_fields
    }
  },
  "output": "binding_name",
  "resource_limits": {
    "max_memory_gb": 8,
    "max_time_seconds": 300,
    "max_graphs": 10
  }
}
```

#### Python API

```python
from graphistry.gfql import ChainGraph, Chain, RemoteGraph

result = ChainGraph({
    "source": RemoteGraph(dataset_id="abc123"),
    "filtered": Chain(ref="source", chain=[
        n({"type": "person"}),
        e_forward({"amount": gt(1000)})
    ])
}, output="filtered")
```

#### Validation Rules

- Binding names must match: `^[a-zA-Z_][a-zA-Z0-9_-]*$`
- No circular references allowed
- Output must reference existing binding
- Reserved names prohibited: `type`, `graph`, `output`, `ref`

### 2. Remote Graph Loading

#### Wire Protocol

```json
{
  "type": "RemoteGraph",
  "dataset_id": "abc123",
  "cache_policy": {
    "mode": "aggressive",
    "ttl_seconds": 3600,
    "validate": true
  },
  "timeout_ms": 30000,
  "retry_policy": {
    "max_attempts": 3,
    "backoff": "exponential"
  }
}
```

#### Security Model

Authentication follows the execution context:
- Uses current user's permissions implicitly
- No embedded tokens in GFQL programs
- Dataset access validated at load time
- Cross-tenant isolation enforced

#### Error Handling

```json
{
  "error": {
    "type": "RemoteGraphError",
    "code": "GRAPH_NOT_FOUND",
    "graph_ref": "abc123",
    "message": "Dataset not found or access denied",
    "retry_possible": true
  }
}
```

### 3. Graph Combinators

#### Wire Protocol

```json
{
  "type": "GraphCombinator",
  "combinator": "union" | "intersect" | "subtract" | "replace",
  "graphs": ["graph1", "graph2"],
  "policies": {
    "attribute_conflict": "left" | "right" | "merge_left" | "merge_right",
    "node_removal": "drop_edges" | "keep_edges",
    "edge_removal": "drop_all_isolated" | "drop_newly_isolated" | "keep_nodes",
    "type_conflict": "coerce_left" | "coerce_right" | "error",
    "drop_dangling": true
  }
}
```

#### Python API

```python
from graphistry.gfql import GraphUnion, GraphIntersect

# Direct usage
union = GraphUnion(
    Chain([n({"type": "customer"})]),
    RemoteGraph("base_graph"),
    policies={"attribute_conflict": "merge_left"}
)

# In ChainGraph
ChainGraph({
    "a": RemoteGraph("graph_a"),
    "b": RemoteGraph("graph_b"),
    "combined": GraphUnion("a", "b")
})
```

#### Advanced Policies

```python
{
  "aggregation": {
    "duplicate_edges": "first" | "last" | "sum" | "mean",
    "multi_valued": "concat" | "array" | "error"
  },
  "schema": {
    "validation": "strict" | "lenient",
    "evolution": "allow" | "deny"
  }
}
```

### 4. Call Operations

#### Wire Protocol

```json
{
  "type": "Call",
  "function": "umap",
  "ref": "input_graph",
  "params": {
    "X": ["feature1", "feature2"],
    "n_neighbors": 15,
    "min_dist": 0.1,
    "kind": "nodes"
  }
}
```

#### Safelist Configuration

```python
SAFELIST_TIERS = {
    "basic": {
        "methods": ["get_degrees", "filter_nodes_by_dict", "materialize_nodes"],
        "resource_limits": {
            "max_nodes": 10_000,
            "max_edges": 50_000,
            "timeout_seconds": 30
        }
    },
    "standard": {
        "methods": ["basic", "hop", "collapse", "fa2_layout"],
        "parameter_restrictions": {
            "hop": {"hops": {"max": 3}},
            "fa2_layout": {"iterations": {"max": 100}}
        }
    },
    "advanced": {
        "methods": ["standard", "umap", "compute_cugraph", "cypher"],
        "parameter_restrictions": {
            "compute_cugraph": {
                "alg": ["pagerank", "louvain", "betweenness_centrality"]
            }
        }
    },
    "enterprise": {
        "methods": "*",
        "custom_methods": true
    }
}
```

#### Parameter Validation

```python
# Type schemas for each method
METHOD_SCHEMAS = {
    "umap": {
        "X": {"type": "array", "items": "string"},
        "n_neighbors": {"type": "integer", "minimum": 2, "maximum": 200},
        "min_dist": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "kind": {"type": "string", "enum": ["nodes", "edges"]}
    }
}
```

### 5. Dotted Reference System

#### Syntax Rules

```yaml
reference_syntax:
  valid_name: "^[a-zA-Z_][a-zA-Z0-9_-]*$"
  separator: "."
  escape: "\\"  # For dots in names: "my\\.name"
  max_depth: 10
  case_sensitive: true
```

#### Resolution Algorithm

```python
def resolve_reference(ref: str, context: ExecutionContext) -> Plottable:
    """Resolve a reference in the current execution context"""
    
    # Split into components
    components = parse_reference(ref)  # Handles escaping
    
    # Simple reference - lexical search
    if len(components) == 1:
        return context.lexical_lookup(components[0])
    
    # Dotted reference - traverse path
    current = context.lexical_lookup(components[0])
    for component in components[1:]:
        if not hasattr(current, 'graph') or component not in current.graph:
            raise ReferenceError(
                f"Cannot resolve '{component}' in path '{ref}'",
                suggestion=find_similar_names(component, current)
            )
        current = current.graph[component]
    
    return current
```

## Security Model

### 1. Resource Limits

```python
class ResourceLimits:
    # Per-execution limits
    max_memory_gb: int = 8
    max_execution_time: int = 300  # seconds
    max_concurrent_graphs: int = 10
    max_remote_fetches: int = 20
    max_nesting_depth: int = 10
    
    # Per-graph limits
    max_nodes_per_graph: int = 10_000_000
    max_edges_per_graph: int = 100_000_000
    max_graph_size_gb: int = 2
```

### 2. Access Control

```python
class SecurityPolicy:
    # Method access by tier
    tier: Literal["basic", "standard", "advanced", "enterprise"]
    
    # Dataset access
    allowed_datasets: List[str] = None  # None = user's accessible datasets
    denied_datasets: List[str] = []
    
    # Operation limits
    max_call_operations: int = 100
    max_graph_combinations: int = 50
    
    # Audit settings
    log_operations: bool = True
    log_data_access: bool = True
```

### 3. Validation Framework

```python
class DAGValidator:
    """Comprehensive validation before execution"""
    
    def validate(self, dag: QueryDAG, context: SecurityContext) -> ValidationResult:
        checks = [
            self.check_circular_references(dag),
            self.check_reference_resolution(dag),
            self.check_resource_limits(dag, context),
            self.check_method_access(dag, context),
            self.check_dataset_permissions(dag, context),
            self.check_parameter_types(dag),
            self.check_reserved_names(dag)
        ]
        
        errors = [e for check in checks for e in check.errors]
        warnings = [w for check in checks for w in check.warnings]
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
```

## Implementation Roadmap

### Phase 1: Foundation (Months 1-2)

**Goals**: Core DAG execution with basic features

1. **Week 1-2**: AST Extensions
   - QueryDAG/ChainGraph classes
   - Basic reference resolution (no dots)
   - JSON serialization

2. **Week 3-4**: Execution Engine
   - ExecutionContext with binding management
   - Memory management framework
   - Basic validation

3. **Week 5-6**: Remote Graph Loading
   - Integration with existing dataset loading
   - Simple authentication flow
   - Basic error handling

4. **Week 7-8**: Testing & Documentation
   - Unit tests for core functionality
   - Integration tests
   - Basic documentation

**Deliverables**: 
- Working DAG execution locally
- Simple remote graph loading
- Basic Chain operations with refs

### Phase 2: Production Features (Months 2-3)

**Goals**: Security, performance, and core transformations

1. **Week 1-2**: Security Framework
   - Resource limits implementation
   - Safelist enforcement
   - Audit logging

2. **Week 3-4**: Call Operations
   - Core method exposure (10-15 methods)
   - Parameter validation
   - Tier-based access control

3. **Week 5-6**: Performance & Caching
   - Query optimization
   - Caching infrastructure
   - Parallel execution

**Deliverables**:
- Secure execution environment
- Key graph transformations
- Performance optimizations

### Phase 3: Advanced Features (Months 3-4)

**Goals**: Full feature set with graph combinators

1. **Week 1-2**: Graph Combinators
   - Union/Intersect/Subtract
   - Policy system
   - Memory-efficient implementation

2. **Week 3-4**: Dotted References
   - Full path resolution
   - Nested DAG support
   - Enhanced error messages

3. **Week 5-6**: Extended Call Operations
   - Additional methods (20-30 total)
   - Custom method registration
   - Advanced parameter types

**Deliverables**:
- Complete combinator support
- Full reference system
- Extended method library

### Phase 4: Enterprise & Polish (Months 4-6)

**Goals**: Production hardening and advanced capabilities

1. **Week 1-4**: Enterprise Features
   - Custom security policies
   - Advanced caching strategies
   - Distributed execution
   - Monitoring & observability

2. **Week 5-8**: Polish & Migration
   - Performance tuning
   - Migration tools
   - Comprehensive documentation
   - Training materials

**Deliverables**:
- Enterprise-ready system
- Migration guides
- Performance benchmarks
- Full documentation

## Examples

### Example 1: Multi-Source Analysis

```python
# Find connections between high-risk entities across datasets
ChainGraph({
    # Load this year's and last year's data
    "current": RemoteGraph("customers_2024"),
    "previous": RemoteGraph("customers_2023"),
    
    # Find high-risk in each
    "risky_current": Chain(ref="current", chain=[
        n({"risk_score": gt(0.8), "status": "active"})
    ]),
    "risky_previous": Chain(ref="previous", chain=[
        n({"risk_score": gt(0.8)})
    ]),
    
    # Union to get all high-risk entities
    "all_risky": GraphUnion("risky_current", "risky_previous",
                           policies={"attribute_conflict": "merge_left"}),
    
    # Find transaction patterns
    "transactions": Chain(ref="all_risky", chain=[
        e_forward({"type": "transaction", "amount": gt(10000)}, hops=2),
        n({"type": "account"})
    ]),
    
    # Apply UMAP for visualization
    "final": Call("umap", ref="transactions", 
                  X=["risk_score", "transaction_count"],
                  n_neighbors=30)
}, output="final")
```

### Example 2: Graph Enrichment Pipeline

```python
ChainGraph({
    # Start with base graph
    "base": RemoteGraph("network_topology"),
    
    # Load enrichment data
    "metrics": RemoteGraph("performance_metrics"),
    
    # Compute centrality on base
    "analyzed": Call("compute_cugraph", ref="base",
                    alg="betweenness_centrality",
                    out_col="centrality"),
    
    # Combine with metrics
    "enriched": GraphUnion("analyzed", "metrics",
                          policies={
                              "attribute_conflict": "merge_right",
                              "drop_dangling": true
                          }),
    
    # Layout for visualization
    "final": Call("layout_cugraph", ref="enriched",
                 layout="force_atlas2",
                 params={"iterations": 100})
})
```

### Example 3: Nested Analysis Modules

```python
ChainGraph({
    "fraud_analysis": ChainGraph({
        "accounts": RemoteGraph("account_graph"),
        "suspicious": Chain(ref="accounts", chain=[
            n({"account_type": "personal",
               "daily_volume": gt(50000)}),
            e_forward({"type": is_in(["wire", "ach"])}, hops=3)
        ]),
        "clusters": Call("compute_igraph", ref="suspicious",
                        alg="louvain", out_col="community")
    }, output="clusters"),
    
    "aml_analysis": ChainGraph({
        "entities": RemoteGraph("entity_graph"),
        "peps": Chain(ref="entities", chain=[
            n({"is_pep": true}),
            e_forward(to_fixed_point=true)
        ])
    }, output="peps"),
    
    # Combine analyses
    "combined": GraphIntersect("fraud_analysis", "aml_analysis"),
    
    # Final risk scoring
    "scored": Call("dbscan", ref="combined",
                  eps=0.3, min_samples=5)
}, output="scored")
```

## Appendices

### A. Wire Protocol Details

#### Request Structure
```json
{
  "version": "1.0",
  "type": "QueryDAG",
  "metadata": {
    "request_id": "uuid",
    "timestamp": "2024-01-15T10:30:00Z",
    "user_context": {
      "tier": "advanced",
      "org_id": "org123"
    }
  },
  "resource_limits": {...},
  "graph": {...},
  "output": "result"
}
```

#### Response Structure
```json
{
  "version": "1.0",
  "request_id": "uuid",
  "status": "success" | "partial" | "error",
  "result": {
    "nodes": [...],
    "edges": [...],
    "metadata": {
      "execution_time_ms": 1234,
      "memory_used_mb": 567,
      "cache_hits": 3
    }
  },
  "errors": [],
  "warnings": [],
  "debug_info": {...}  // If requested
}
```

### B. Error Codes

```python
ERROR_CODES = {
    # Reference errors (1xxx)
    "1001": "REFERENCE_NOT_FOUND",
    "1002": "CIRCULAR_REFERENCE",
    "1003": "AMBIGUOUS_REFERENCE",
    
    # Security errors (2xxx)
    "2001": "ACCESS_DENIED",
    "2002": "METHOD_NOT_ALLOWED",
    "2003": "PARAMETER_FORBIDDEN",
    
    # Resource errors (3xxx)
    "3001": "MEMORY_LIMIT_EXCEEDED",
    "3002": "TIMEOUT_EXCEEDED",
    "3003": "GRAPH_TOO_LARGE",
    
    # Validation errors (4xxx)
    "4001": "INVALID_PARAMETER_TYPE",
    "4002": "MISSING_REQUIRED_FIELD",
    "4003": "SCHEMA_MISMATCH",
    
    # Execution errors (5xxx)
    "5001": "REMOTE_GRAPH_UNAVAILABLE",
    "5002": "COMPUTATION_FAILED",
    "5003": "COMBINATOR_CONFLICT"
}
```

### C. Performance Guidelines

1. **Memory Usage Estimates**
   - Node: ~100 bytes base + attributes
   - Edge: ~50 bytes base + attributes
   - Overhead: ~20% for indices and metadata

2. **Operation Complexity**
   - Union: O(V₁ + V₂ + E₁ + E₂)
   - Intersection: O(min(V₁, V₂) × log(max(V₁, V₂)))
   - Chain operations: O(V × average_degree × hops)

3. **Optimization Hints**
   - Use column subsetting for remote graphs
   - Prefer intersection before union for filtering
   - Cache frequently used remote graphs
   - Parallelize independent DAG branches

### D. Migration Guide

#### From Current Chain API
```python
# Before
g.chain([n({"type": "customer"}), e_forward()]).chain([n({"risk": gt(0.5)})])

# After - single expression
ChainGraph({
    "result": Chain(chain=[
        n({"type": "customer"}),
        e_forward(),
        n({"risk": gt(0.5)})
    ])
})
```

#### From Multiple Python Steps
```python
# Before
g1 = graphistry.bind(dataset_id="abc").chain([...])
g2 = graphistry.bind(dataset_id="xyz").chain([...])
# Manual combination needed

# After - unified expression
ChainGraph({
    "g1": Chain(ref="RemoteGraph('abc')", chain=[...]),
    "g2": Chain(ref="RemoteGraph('xyz')", chain=[...]),
    "combined": GraphUnion("g1", "g2")
})
```

## Conclusion

GFQL Programs represents a significant evolution in PyGraphistry's capabilities, transforming it from a visualization-focused tool to a comprehensive graph programming platform. This specification provides a solid foundation for implementation while addressing critical concerns around security, performance, and usability.

The phased implementation approach allows for iterative development and validation, ensuring each feature is production-ready before moving to the next. With careful attention to the security model and resource management, GFQL Programs will enable powerful new workflows while maintaining system stability and data isolation.