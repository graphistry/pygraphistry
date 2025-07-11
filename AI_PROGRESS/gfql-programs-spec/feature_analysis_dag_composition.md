# Feature Analysis: Core DAG Composition (QueryDAG/ChainGraph)

## Executive Summary

This analysis examines the proposed QueryDAG/ChainGraph feature that extends GFQL from single-chain operations to directed acyclic graph (DAG) compositions. The feature enables multiple graph bindings, remote graph loading, and complex graph combinators while maintaining backward compatibility with existing Chain operations.

## 1.3.1.1: Relationship to Current Chain Architecture

### How QueryDAG Extends the Single-Chain Model

The current GFQL architecture is built around a linear chain of operations:

```python
# Current: Linear sequence
g.chain([
    n({"type": "person"}),
    e_forward(),
    n({"type": "company"})
])
```

QueryDAG extends this to support:

1. **Multiple Named Graphs**: Instead of operating on a single implicit graph, QueryDAG introduces a binding environment where multiple graphs can be named and referenced:

```python
# Proposed: DAG with multiple graphs
ChainGraph({
    "people": Chain([n({"type": "person"})]),
    "companies": Chain([n({"type": "company"})]),
    "connections": Chain(ref="people", chain=[e_forward()])
})
```

2. **Lexical Scoping**: References follow lexical scoping rules, where the closest binding to a reference is used. This is similar to variable scoping in programming languages.

3. **Explicit Output Selection**: While chains implicitly return the final operation's result, QueryDAG allows explicit output selection via the `output` field.

### Reusable Components from Current Architecture

The following components can be directly reused:

1. **AST Infrastructure**:
   - `ASTSerializable` base class for JSON serialization
   - `ASTObject` interface with `__call__` and `reverse` methods
   - Existing AST nodes (`ASTNode`, `ASTEdge`)
   - Predicate system remains unchanged

2. **Execution Engine**:
   - The 3-phase algorithm (forward/reverse/combine) can be applied to each sub-chain
   - `combine_steps()` function for merging results
   - Engine abstraction (pandas/cudf) works as-is

3. **Remote Execution**:
   - `chain_remote.py` infrastructure can be extended
   - Wire protocol JSON structure naturally extends to nested operations
   - Authentication and session handling remain the same

4. **Plottable Integration**:
   - Results still return Plottable objects
   - Node/edge bindings work the same way
   - Visualization settings preservation

### New Components Required

1. **QueryDAG AST Node**:
```python
class QueryDAG(ASTSerializable):
    def __init__(self, graph: Dict[str, Union[Chain, 'QueryDAG']], output: Optional[str] = None):
        self.graph = graph
        self.output = output or list(graph.keys())[-1]
    
    def validate(self):
        # Validate binding names match pattern ^[a-zA-Z_][a-zA-Z0-9_-]*$
        # Check for circular references
        # Ensure output exists in graph
```

2. **Reference Resolution**:
   - New `ref` parameter on Chain class
   - Dotted reference parser for nested scopes (e.g., "alerts.start")
   - Reference validation during AST construction

3. **Execution Context**:
```python
class ExecutionContext:
    """Manages graph bindings during DAG execution"""
    def __init__(self):
        self.bindings: Dict[str, Plottable] = {}
        self.scopes: List[Dict[str, Plottable]] = []
    
    def bind(self, name: str, value: Plottable):
        self.bindings[name] = value
    
    def resolve(self, ref: str) -> Plottable:
        # Handle dotted references
        # Search scopes in reverse order (lexical scoping)
```

## 1.3.1.2: Implementation Challenges and Dependencies

### Wire Protocol Changes

1. **New Message Types**:
```json
{
  "type": "QueryDAG",
  "graph": {
    "a": {"type": "Chain", "chain": [...]},
    "b": {"type": "Chain", "ref": "a", "chain": [...]}
  },
  "output": "b"
}
```

2. **Extended Chain Format**:
```json
{
  "type": "Chain",
  "ref": "some_graph",  // New optional field
  "chain": [...]
}
```

3. **Remote Graph Loading**:
```json
{
  "type": "RemoteGraph",
  "dataset_id": "abc123"
}
```

### Backward Compatibility Concerns

1. **API Compatibility**:
   - Existing `g.chain([...])` calls must continue working
   - New `ChainGraph({...})` is additive, not breaking
   - Server must handle both old and new wire formats

2. **Wire Protocol Versioning**:
   - Add protocol version negotiation
   - Server should detect message format by presence of "type" field
   - Graceful degradation for older clients

3. **Migration Path**:
```python
# Old code continues to work
g.chain([n({"type": "person"})])

# Can be gradually migrated to
ChainGraph({
    "result": Chain([n({"type": "person"})])
})
```

### Performance Implications

1. **Memory Management**:
   - Each binding holds a full Plottable (nodes + edges DataFrames)
   - Need reference counting or garbage collection for intermediate results
   - Consider lazy evaluation for unused bindings

2. **Execution Optimization**:
   - Parallel execution of independent subgraphs
   - Common subexpression elimination
   - Query planning for optimal execution order

3. **Network Overhead**:
   - Larger wire protocol messages
   - Multiple remote graph fetches
   - Consider batching remote operations

### Memory Management for Multiple Graph Bindings

1. **Binding Lifecycle**:
```python
class GraphBindingManager:
    def __init__(self, max_memory_mb: int = 1000):
        self.bindings: Dict[str, Plottable] = {}
        self.access_count: Dict[str, int] = {}
        self.memory_usage: Dict[str, int] = {}
    
    def add_binding(self, name: str, graph: Plottable):
        # Track memory usage
        # Implement LRU eviction if needed
    
    def get_binding(self, name: str) -> Plottable:
        # Update access count
        # Handle cache misses
```

2. **Resource Limits**:
   - Maximum number of concurrent bindings
   - Total memory usage caps
   - Timeout for long-running DAGs

## 1.3.1.3: Critical Review

### Potential Bugs/Edge Cases

1. **Circular References**:
```python
# This should be detected and rejected
ChainGraph({
    "a": Chain(ref="b", chain=[...]),
    "b": Chain(ref="a", chain=[...])
})
```

2. **Name Collisions**:
```python
# Shadowing in nested scopes
ChainGraph({
    "data": RemoteGraph("abc"),
    "nested": ChainGraph({
        "data": RemoteGraph("xyz"),  # Shadows outer "data"
        "result": Chain(ref="data")  # Which one?
    })
})
```

3. **Missing References**:
```python
# Reference to non-existent binding
Chain(ref="nonexistent", chain=[...])
```

4. **Resource Exhaustion**:
   - Loading too many large remote graphs
   - Deep nesting causing stack overflow
   - Combinatorial explosion in graph combinations

### Security Risks

1. **Resource Exhaustion Attacks**:
   - Malicious DAGs with excessive bindings
   - Recursive structures consuming memory
   - Remote graph fetching as DoS vector

2. **Access Control**:
   - Ensure remote graph access respects permissions
   - Prevent information leakage through error messages
   - Validate dataset_id format to prevent injection

3. **Execution Limits**:
```python
class DAGExecutionLimits:
    max_bindings = 100
    max_nesting_depth = 10
    max_execution_time = 300  # seconds
    max_memory_usage = 1024  # MB
    max_remote_fetches = 20
```

### Suggested Improvements

1. **Lazy Evaluation**:
   - Only compute bindings that are referenced
   - Cache intermediate results
   - Streaming execution for large graphs

2. **Query Optimization**:
```python
class QueryOptimizer:
    def optimize(self, dag: QueryDAG) -> QueryDAG:
        # Common subexpression elimination
        # Dead code elimination
        # Parallel execution planning
        # Push filters down to data sources
```

3. **Better Error Messages**:
```python
class DAGValidationError(ValueError):
    def __init__(self, message: str, location: str, suggestion: str):
        self.location = location  # e.g., "graph.alerts.start"
        self.suggestion = suggestion
        super().__init__(f"{message} at {location}. {suggestion}")
```

4. **Type System**:
   - Add optional type annotations for graph schemas
   - Compile-time validation of operations
   - Better IDE support through type hints

### Priority Assessment

**High Priority** (Required for MVP):
1. Basic QueryDAG execution with single-level bindings
2. Reference resolution (without dotted syntax)
3. RemoteGraph loading
4. Backward compatibility
5. Basic validation (circular refs, missing refs)

**Medium Priority** (Post-MVP):
1. Dotted reference syntax
2. Nested QueryDAG support
3. Performance optimizations
4. Graph combinators (union, intersection)
5. Memory management

**Low Priority** (Future Enhancements):
1. Query optimization
2. Lazy evaluation
3. Type system
4. Advanced debugging tools
5. Visual DAG editor

## Implementation Roadmap

### Phase 1: Core Infrastructure (2-3 weeks)
1. Implement QueryDAG/ChainGraph classes
2. Add ref parameter to Chain
3. Basic execution context
4. Update wire protocol
5. Basic validation

### Phase 2: Remote Graphs (1-2 weeks)
1. Implement RemoteGraph class
2. Integration with existing dataset loading
3. Permission checking
4. Caching layer

### Phase 3: Advanced Features (2-3 weeks)
1. Dotted references
2. Nested QueryDAG
3. Graph combinators
4. Performance optimizations

### Phase 4: Production Hardening (1-2 weeks)
1. Resource limits
2. Security review
3. Performance testing
4. Documentation
5. Migration guide

## Conclusion

The QueryDAG feature is a natural evolution of GFQL that addresses real user needs (JPMC, Louie) while maintaining the elegance of the current design. The implementation can leverage significant portions of the existing codebase, with the main challenges being:

1. Managing multiple graph bindings efficiently
2. Ensuring backward compatibility
3. Preventing resource exhaustion
4. Providing clear error messages

With careful implementation following the suggested phases, this feature can be delivered with minimal risk to existing functionality while opening up powerful new use cases for graph analysis workflows.