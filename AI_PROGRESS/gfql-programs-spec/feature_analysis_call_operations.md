# Feature Analysis: Call Operations

## Overview

The Call Operations feature aims to expose PyGraphistry's Plottable interface methods through GFQL, enabling users to invoke graph transformation and analysis methods without requiring Python. This analysis examines the implementation requirements, security considerations, and design decisions for exposing these methods safely and effectively.

## 1.3.5.1: Inventory of Plottable Methods to Expose

### Methods That Return New Plottables

Based on the codebase analysis, the following Plottable methods return new Plottable instances and are candidates for GFQL exposure:

#### Core Graph Transformations
1. **umap()** - UMAP dimensionality reduction for node/edge embeddings
   - Parameters: X, y, kind, scale, n_neighbors, min_dist, spread, etc.
   - Returns: Plottable with UMAP embeddings applied

2. **cypher()** - Execute Cypher queries against Neo4j/Memgraph/Neptune
   - Parameters: query (string), params (dict)
   - Returns: Plottable with query results

3. **hypergraph()** - Convert tabular data to hypergraph representation
   - Parameters: raw_events, entity_types, opts, drop_na, etc.
   - Returns: Plottable with hypergraph structure

#### Layout Methods
4. **layout_cugraph()** - GPU-accelerated graph layouts
   - Parameters: layout, params, kind, directed, G, bind_position, etc.
   - Returns: Plottable with computed positions

5. **layout_igraph()** - IGraph-based layouts
   - Parameters: layout, directed, use_vids, bind_position, params, etc.
   - Returns: Plottable with computed positions

6. **layout_graphviz()** - Graphviz layouts
   - Parameters: prog, args, directed, strict, graph_attr, etc.
   - Returns: Plottable with computed positions

7. **fa2_layout()** - ForceAtlas2 layout
   - Parameters: fa2_params, circle_layout_params, singleton_layout, etc.
   - Returns: Plottable with computed positions

#### Graph Analytics
8. **compute_cugraph()** - GPU-accelerated graph algorithms
   - Parameters: alg, out_col, params, kind, directed, G
   - Returns: Plottable with computed metrics

9. **compute_igraph()** - IGraph algorithms
   - Parameters: alg, out_col, directed, use_vids, params
   - Returns: Plottable with computed metrics

#### Feature Engineering
10. **featurize()** - Extract features from graph structure
    - Parameters: kind, X, y, feature params
    - Returns: Plottable with extracted features

11. **dbscan()** - DBSCAN clustering
    - Parameters: min_dist, min_samples, eps, metric, etc.
    - Returns: Plottable with cluster assignments

12. **embed()** - Graph embeddings
    - Parameters: relation, proto, various embedding params
    - Returns: Plottable with embeddings

#### Data Transformations
13. **transform()** / **transform_umap()** - Transform new data using fitted models
    - Parameters: df, y, kind, min_dist, n_neighbors, etc.
    - Returns: Plottable with transformed data

14. **materialize_nodes()** - Create explicit node table from edges
    - Parameters: reuse, engine
    - Returns: Plottable with materialized nodes

15. **collapse()** - Collapse nodes based on attributes
    - Parameters: node, attribute, column, self_edges, etc.
    - Returns: Plottable with collapsed graph

#### Graph Filtering/Selection
16. **hop()** - Multi-hop traversal
    - Parameters: nodes, hops, direction, edge_match, etc.
    - Returns: Plottable with traversal results

17. **filter_nodes_by_dict()** / **filter_edges_by_dict()** - Dictionary-based filtering
    - Parameters: filter_dict
    - Returns: Plottable with filtered graph

18. **drop_nodes()** / **keep_nodes()** - Node-based filtering
    - Parameters: nodes
    - Returns: Plottable with filtered graph

19. **prune_self_edges()** - Remove self-loops
    - Parameters: none
    - Returns: Plottable without self-edges

#### Graph Metrics
20. **get_degrees()** / **get_indegrees()** / **get_outdegrees()** - Degree calculations
    - Parameters: col names
    - Returns: Plottable with degree metrics

21. **get_topological_levels()** - Topological sorting
    - Parameters: level_col, allow_cycles, warn_cycles, etc.
    - Returns: Plottable with topological levels

### Good Candidates for Initial GFQL Exposure

For the initial implementation, prioritize methods that:
1. Have simple, JSON-serializable parameters
2. Are commonly used in graph analysis workflows
3. Have minimal side effects
4. Don't require complex object initialization

**Recommended Initial Set:**
- `umap()` - Core embedding functionality
- `layout_cugraph()` / `fa2_layout()` - Essential layouts
- `compute_cugraph()` / `compute_igraph()` - Key algorithms
- `hop()` - Graph traversal
- `filter_nodes_by_dict()` / `filter_edges_by_dict()` - Filtering
- `get_degrees()` - Basic metrics
- `materialize_nodes()` - Data preparation

### Parameter Types and Validation Needs

#### Common Parameter Types
1. **Strings**: algorithm names, column names, layout types
2. **Numbers**: int (n_neighbors, hops), float (min_dist, scale)
3. **Booleans**: directed, allow_cycles, drop_na
4. **Dictionaries**: params, filter_dict, opts
5. **Lists**: entity_types, node lists, column lists
6. **Enums**: kind ("nodes"/"edges"), direction ("forward"/"reverse"/"both")

#### Validation Requirements
1. **Type Validation**: Ensure parameters match expected types
2. **Range Validation**: n_neighbors > 0, min_dist >= 0, etc.
3. **Enum Validation**: Check against allowed values
4. **Dictionary Schema**: Validate structure of complex parameters
5. **Column Existence**: Verify referenced columns exist
6. **Compatibility**: Check parameter combinations are valid

## 1.3.5.2: Safelisting and Security Model

### Access Control Levels

#### 1. Method-Level Access Control
```python
SAFELIST_TIERS = {
    "basic": [
        "get_degrees", "get_indegrees", "get_outdegrees",
        "filter_nodes_by_dict", "filter_edges_by_dict",
        "materialize_nodes", "prune_self_edges"
    ],
    "standard": [
        # Includes basic +
        "hop", "collapse", "drop_nodes", "keep_nodes",
        "fa2_layout", "get_topological_levels"
    ],
    "advanced": [
        # Includes standard +
        "umap", "featurize", "dbscan", "embed",
        "layout_cugraph", "compute_cugraph",
        "layout_igraph", "compute_igraph"
    ],
    "enterprise": [
        # Includes advanced +
        "cypher", "hypergraph", "transform", "transform_umap",
        # Custom/experimental methods
    ]
}
```

#### 2. Parameter-Level Restrictions
```python
PARAMETER_RESTRICTIONS = {
    "compute_cugraph": {
        "basic": {"alg": ["pagerank", "degree_centrality"]},
        "standard": {"alg": ["pagerank", "degree_centrality", "betweenness_centrality", "louvain"]},
        "advanced": {"alg": "*"}  # All algorithms
    },
    "umap": {
        "standard": {"n_neighbors": range(5, 50), "min_dist": [0.1, 0.25, 0.5]},
        "advanced": {"n_neighbors": range(2, 200), "min_dist": "*"}
    }
}
```

#### 3. Resource Limits
```python
RESOURCE_LIMITS = {
    "basic": {
        "max_nodes": 10_000,
        "max_edges": 50_000,
        "timeout_seconds": 30
    },
    "standard": {
        "max_nodes": 100_000,
        "max_edges": 1_000_000,
        "timeout_seconds": 300
    },
    "advanced": {
        "max_nodes": 10_000_000,
        "max_edges": 100_000_000,
        "timeout_seconds": 3600
    }
}
```

### Security Implementation

#### 1. Safelist Configuration
```python
class CallSafelist:
    def __init__(self, tier: str, custom_safelist: Optional[Dict] = None):
        self.tier = tier
        self.allowed_methods = set(SAFELIST_TIERS.get(tier, []))
        if custom_safelist:
            self.allowed_methods.update(custom_safelist.get("allow", []))
            self.allowed_methods -= set(custom_safelist.get("deny", []))
    
    def is_allowed(self, method: str, params: Dict) -> Tuple[bool, Optional[str]]:
        if method not in self.allowed_methods:
            return False, f"Method '{method}' not allowed for tier '{self.tier}'"
        
        # Check parameter restrictions
        if method in PARAMETER_RESTRICTIONS:
            restrictions = PARAMETER_RESTRICTIONS[method].get(self.tier, {})
            for param, value in params.items():
                if param in restrictions:
                    allowed = restrictions[param]
                    if allowed != "*" and value not in allowed:
                        return False, f"Parameter '{param}={value}' not allowed"
        
        return True, None
```

#### 2. Execution Sandbox
```python
class CallExecutor:
    def __init__(self, safelist: CallSafelist, resource_limiter: ResourceLimiter):
        self.safelist = safelist
        self.limiter = resource_limiter
    
    def execute(self, plottable: Plottable, method: str, params: Dict) -> Plottable:
        # Security checks
        allowed, error = self.safelist.is_allowed(method, params)
        if not allowed:
            raise SecurityError(error)
        
        # Resource checks
        if not self.limiter.check_limits(plottable):
            raise ResourceError("Graph exceeds resource limits")
        
        # Parameter validation
        validated_params = self.validate_params(method, params)
        
        # Execute with timeout
        with self.limiter.timeout():
            func = getattr(plottable, method)
            return func(**validated_params)
```

### Security Implications

#### 1. Denial of Service (DoS)
- **Risk**: Expensive operations (UMAP on large graphs)
- **Mitigation**: Resource limits, timeouts, rate limiting

#### 2. Data Exfiltration
- **Risk**: Methods revealing sensitive graph structure
- **Mitigation**: Result size limits, output sanitization

#### 3. Code Injection
- **Risk**: String parameters (cypher queries, column names)
- **Mitigation**: Parameter validation, query sanitization

#### 4. Resource Exhaustion
- **Risk**: Memory/CPU intensive operations
- **Mitigation**: Graph size limits, operation timeouts

#### 5. Unauthorized Access
- **Risk**: Accessing methods beyond tier
- **Mitigation**: Strict safelist enforcement, audit logging

## 1.3.5.3: Critical Review

### Compatibility and Validation

#### JSON Serialization Compatibility

**Compatible Parameter Types:**
- Primitives: string, number, boolean, null
- Collections: arrays, objects (dicts)
- Enums: string values from predefined sets

**Incompatible Types Requiring Adaptation:**
```python
# Problem: Functions/Callables
"singleton_layout": Callable  # fa2_layout parameter

# Solution: Predefined function names
"singleton_layout": "circle" | "random" | "grid"

# Problem: Complex objects
"G": cugraph.Graph  # compute_cugraph parameter

# Solution: Reference by ID or auto-detection
"G": "auto" | {"ref": "graph_id"}
```

#### Parameter Validation Framework
```python
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, validator

class CallParameters(BaseModel):
    method: str
    params: Dict[str, Any]
    
    @validator('method')
    def validate_method(cls, v):
        if v not in ALLOWED_METHODS:
            raise ValueError(f"Unknown method: {v}")
        return v
    
    @validator('params')
    def validate_params(cls, v, values):
        method = values.get('method')
        schema = METHOD_SCHEMAS.get(method)
        if schema:
            # Validate against method-specific schema
            schema.validate(v)
        return v

# Method-specific schemas
class UMAPParameters(BaseModel):
    X: Optional[Union[List[str], str]] = None
    y: Optional[Union[List[str], str]] = None
    kind: Literal["nodes", "edges"] = "nodes"
    n_neighbors: int = Field(ge=2, le=200)
    min_dist: float = Field(ge=0.0, le=1.0)
    # ... other parameters
```

### Type Safety

#### Runtime Type Checking
```python
def validate_call_params(method: str, params: Dict) -> Dict:
    """Validate and coerce parameters for a method call"""
    
    # Get method signature
    sig = inspect.signature(getattr(Plottable, method))
    
    # Check required parameters
    for param_name, param in sig.parameters.items():
        if param.default == param.empty and param_name not in params:
            raise ValueError(f"Missing required parameter: {param_name}")
    
    # Validate types
    for param_name, value in params.items():
        if param_name not in sig.parameters:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        # Type coercion/validation
        expected_type = sig.parameters[param_name].annotation
        if expected_type != param.empty:
            params[param_name] = coerce_type(value, expected_type)
    
    return params
```

### Future Extensibility

#### 1. Louie Connector Integration
```python
class LouieConnectorCall:
    """Future extension for Louie connector methods"""
    
    def __init__(self, connector_id: str, method: str, params: Dict):
        self.connector = load_connector(connector_id)
        self.method = method
        self.params = params
    
    def execute(self, plottable: Plottable) -> Plottable:
        # Apply connector-specific transformation
        data = self.connector.transform(plottable, self.method, self.params)
        return plottable.bind(edges=data['edges'], nodes=data['nodes'])
```

#### 2. Custom Method Registration
```python
class MethodRegistry:
    """Extensible method registry for future additions"""
    
    def __init__(self):
        self.methods = {}
        self._register_core_methods()
    
    def register(self, name: str, func: Callable, schema: BaseModel):
        """Register a new method for GFQL exposure"""
        self.methods[name] = {
            'func': func,
            'schema': schema,
            'tier': 'custom'
        }
    
    def execute(self, plottable: Plottable, name: str, params: Dict):
        if name not in self.methods:
            raise ValueError(f"Unknown method: {name}")
        
        method_info = self.methods[name]
        validated = method_info['schema'](**params).dict()
        return method_info['func'](plottable, **validated)
```

#### 3. Plugin Architecture
```python
class CallPlugin:
    """Base class for call operation plugins"""
    
    def get_methods(self) -> Dict[str, Dict]:
        """Return methods exposed by this plugin"""
        raise NotImplementedError
    
    def validate(self, method: str, params: Dict) -> Dict:
        """Validate parameters for a method"""
        raise NotImplementedError
    
    def execute(self, plottable: Plottable, method: str, params: Dict) -> Plottable:
        """Execute the method"""
        raise NotImplementedError
```

### Error Handling and Debugging

#### 1. Error Categories
```python
class CallError(Exception):
    """Base class for call operation errors"""
    pass

class MethodNotFoundError(CallError):
    """Method doesn't exist or isn't exposed"""
    pass

class ParameterValidationError(CallError):
    """Invalid parameters for method"""
    pass

class ExecutionError(CallError):
    """Error during method execution"""
    pass

class SecurityError(CallError):
    """Security policy violation"""
    pass
```

#### 2. Debugging Information
```python
class CallDebugInfo:
    def __init__(self, method: str, params: Dict):
        self.method = method
        self.params = params
        self.start_time = time.time()
        self.graph_shape_before = None
        self.graph_shape_after = None
        self.execution_time = None
        self.error = None
    
    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "params": self.params,
            "execution_time": self.execution_time,
            "graph_shape_change": {
                "before": self.graph_shape_before,
                "after": self.graph_shape_after
            },
            "error": str(self.error) if self.error else None
        }
```

#### 3. Execution Tracing
```python
class TracedCallExecutor:
    def execute(self, plottable: Plottable, call: Dict) -> Tuple[Plottable, Dict]:
        debug_info = CallDebugInfo(call["function"], call["params"])
        
        try:
            # Capture initial state
            debug_info.graph_shape_before = {
                "nodes": len(plottable._nodes) if plottable._nodes is not None else 0,
                "edges": len(plottable._edges) if plottable._edges is not None else 0
            }
            
            # Execute
            result = self._execute_call(plottable, call)
            
            # Capture final state
            debug_info.graph_shape_after = {
                "nodes": len(result._nodes) if result._nodes is not None else 0,
                "edges": len(result._edges) if result._edges is not None else 0
            }
            
            debug_info.execution_time = time.time() - debug_info.start_time
            
            return result, debug_info.to_dict()
            
        except Exception as e:
            debug_info.error = e
            debug_info.execution_time = time.time() - debug_info.start_time
            raise CallExecutionError(
                f"Failed to execute {call['function']}: {str(e)}",
                debug_info=debug_info.to_dict()
            )
```

## Implementation Recommendations

### Phase 1: Core Implementation
1. Implement basic call operation with safelist
2. Add parameter validation for core methods
3. Implement resource limits and timeouts
4. Add comprehensive error handling

### Phase 2: Security Hardening
1. Implement tier-based access control
2. Add parameter-level restrictions
3. Implement audit logging
4. Add rate limiting

### Phase 3: Extended Features
1. Add more methods to safelist
2. Implement custom method registration
3. Add Louie connector support
4. Implement debugging/tracing features

### Best Practices
1. **Default Deny**: Only expose explicitly safelisted methods
2. **Validate Everything**: Never trust client-provided parameters
3. **Resource Limits**: Always enforce limits on expensive operations
4. **Audit Trail**: Log all operations for security monitoring
5. **Graceful Degradation**: Provide clear errors when operations fail
6. **Version Compatibility**: Design for backward compatibility

## Conclusion

The Call Operations feature provides powerful graph transformation capabilities through GFQL while maintaining security and performance. The tiered access model, comprehensive validation, and extensible architecture ensure the feature can grow with user needs while protecting system resources. Careful implementation of the security model and validation framework will be critical for successful deployment.