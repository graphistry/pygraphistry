# Feature Analysis: Graph Combinators

## Executive Summary

This analysis examines the Graph Combinators feature proposed in the GFQL Programs RFC (sketch.md). Graph combinators would enable users to compose multiple graphs through operations like union, intersection, and subtraction - a significant enhancement to PyGraphistry's current capabilities. While PyGraphistry has rich graph processing methods, it lacks explicit graph combination operations that preserve graph semantics and relationships.

## 1.3.4.1: Mapping to Existing Graph Operations in PyGraphistry

### Current Graph Operations

PyGraphistry currently provides several graph manipulation methods, but they are primarily focused on single-graph transformations:

#### Filtering Operations
- `filter_edges_by_dict()` - Filter edges by attribute values
- `filter_nodes_by_dict()` - Filter nodes by attribute values
- Chain operations with predicates (via GFQL)

#### Graph Transformations
- `materialize_nodes()` - Generate nodes from edges
- `collapse_by()` - Collapse nodes/edges by attributes
- `hop()` - Graph traversal operations
- Layout operations (UMAP, ForceAtlas2, etc.)

#### Data Conversions
- `to_pandas()` / `to_cudf()` - Engine conversions
- Integration with NetworkX, igraph, CuGraph
- Support for various data formats

### What's Missing

Current PyGraphistry lacks explicit graph combination primitives:

1. **No Graph Union** - Cannot merge two graphs while preserving node/edge relationships
2. **No Graph Intersection** - Cannot find common subgraphs
3. **No Graph Subtraction** - Cannot remove one graph from another
4. **No Merge Policies** - No systematic way to handle attribute conflicts
5. **No Graph References** - Cannot compose operations on multiple named graphs

### What Combinators Would Add

Graph combinators would provide:

1. **Semantic Graph Operations** - Operations that understand graph structure, not just dataframes
2. **Multi-Graph Workflows** - Ability to work with multiple graphs in a single expression
3. **Declarative Composition** - Express complex multi-graph operations without imperative code
4. **Remote Graph Integration** - Load and combine graphs from different sources
5. **Policy-Based Conflict Resolution** - Systematic handling of overlapping data

## 1.3.4.2: Policy System Design Review

### Proposed Combination Policies

The RFC proposes several policies for handling data conflicts:

#### Attribute Merge Policies
- **left**: Keep attributes from left graph
- **right**: Keep attributes from right graph  
- **merge_left**: Merge with left graph taking precedence
- **merge_right**: Merge with right graph taking precedence

#### Node Removal Policies
- **drop_edges**: Remove edges connected to removed nodes
- **keep_edges**: Preserve edges (may create dangling references)

#### Edge Removal Policies
- **drop_all_isolated**: Remove all isolated nodes
- **drop_newly_isolated**: Remove only nodes isolated by the operation
- **keep_nodes**: Preserve all nodes regardless of connectivity

#### Additional Policy: drop_dangling
- Remove edges with missing source/destination nodes

### Policy System Analysis

**Strengths:**
- Covers common conflict scenarios
- Provides fine-grained control over graph structure preservation
- Aligns with database join semantics (left/right/merge)

**Gaps:**
- No policy for attribute type conflicts (e.g., string vs. numeric)
- No aggregation policies for duplicate edges
- Missing policies for multi-valued attributes
- No support for custom merge functions

### Recommended Enhancements

1. **Type Coercion Policies**
   - `coerce_left`: Use left type, coerce right
   - `coerce_right`: Use right type, coerce left
   - `coerce_common`: Find common supertype
   - `error`: Fail on type mismatch

2. **Aggregation Policies**
   - `first`: Keep first occurrence
   - `last`: Keep last occurrence
   - `sum/mean/max/min`: Aggregate numeric attributes
   - `concat`: Concatenate string/list attributes

3. **Custom Resolution**
   - Allow user-defined merge functions
   - Support for attribute-specific policies

## 1.3.4.3: Critical Review

### Memory Implications

**Concerns:**
1. **Duplication During Operations** - Combinators may need to copy entire graphs
2. **Intermediate Results** - DAG evaluation creates temporary graphs
3. **Remote Graph Loading** - Loading multiple large graphs simultaneously

**Mitigation Strategies:**
1. Lazy evaluation where possible
2. Streaming operations for large graphs
3. Memory-mapped operations for remote graphs
4. Reference counting for shared data

### Edge Cases

1. **Mismatched Node IDs**
   - Different ID types (string vs. int)
   - Different ID semantics (global vs. local)
   - Solution: ID mapping/normalization phase

2. **Cyclic References in DAG**
   - Detecting cycles in QueryDAG
   - Solution: Static validation at parse time

3. **Empty Graph Handling**
   - Union with empty graph
   - Intersection resulting in empty graph
   - Solution: Well-defined empty graph semantics

4. **Schema Evolution**
   - Graphs with different schemas over time
   - Solution: Schema versioning and migration

### Consistency Issues

1. **Node-Edge Relationship Integrity**
   - Operations may break referential integrity
   - Need validation after each operation
   - Consider transaction-like semantics

2. **Attribute Consistency**
   - Same attribute with different meanings
   - Solution: Namespace attributes by source

3. **Graph Property Preservation**
   - Directed vs. undirected conflicts
   - Multi-graph vs. simple graph
   - Solution: Explicit graph type policies

### Performance Considerations

1. **Index Maintenance**
   - Need efficient lookups for intersection/union
   - Consider maintaining node/edge indices

2. **Parallel Execution**
   - DAG structure enables parallelism
   - Need thread-safe operations

3. **Caching Strategy**
   - Cache intermediate results in DAG
   - Invalidation on source changes

4. **Engine Optimization**
   - Leverage GPU for large-scale operations
   - Push operations to data source when possible

## Implementation Recommendations

### Phase 1: Core Infrastructure
1. Implement QueryDAG/ChainGraph classes
2. Add reference resolution system
3. Create basic combinator operations

### Phase 2: Policy System
1. Implement merge policies
2. Add removal policies
3. Create validation framework

### Phase 3: Optimization
1. Add lazy evaluation
2. Implement caching
3. GPU acceleration

### Phase 4: Advanced Features
1. Custom policies
2. Schema management
3. Distributed execution

## Testing Strategy

### Unit Tests
- Policy behavior validation
- Edge case handling
- Memory leak detection

### Integration Tests
- Multi-graph workflows
- Remote graph loading
- Large graph performance

### Stress Tests
- Memory limits
- Concurrent operations
- Schema conflicts

## Conclusion

Graph combinators represent a significant enhancement to PyGraphistry's capabilities. While the current proposal provides a solid foundation, attention to memory management, edge cases, and consistency will be critical for production use. The phased implementation approach allows for iterative refinement based on user feedback and performance characteristics.