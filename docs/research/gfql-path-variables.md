# GFQL Path Support: Variable Bindings for Cross-Node Predicates

**Issue**: #722
**Created**: 2025-10-19
**Status**: Research Phase

---

## Problem Statement

GFQL currently cannot express multi-hop patterns with **cross-node predicates** like Cypher's:

```cypher
MATCH (n1)-[e1]->(n2)-[e2]->(n3)
WHERE n1.x == n3.x
```

### The Core Issue

**Wavefront-based execution** in GFQL:
- Each hop step operates on a **wavefront** (set of node IDs)
- Only the **current wavefront** is accessible to subsequent operations
- **Previous nodes are lost** - no way to reference them in predicates
- Each operation returns a new graph with only matched entities

**Example**:
```python
# Current GFQL - CAN'T compare n1 and n3
g.gfql([
    n({'type': 'person'}, name='n1'),      # wavefront: person nodes
    e_forward(name='e1'),                   # wavefront: forward neighbors
    n(name='n2'),                           # wavefront: those neighbors
    e_forward(name='e2'),                   # wavefront: 2-hop neighbors
    n(name='n3')                            # wavefront: final nodes
    # ❌ NO WAY to express: WHERE n1.x == n3.x
])
```

### What Users Need

1. **Named variable bindings** for nodes and edges in the pattern
2. **Access to all matched variables** at any point in the pattern
3. **Cross-variable predicates** that can reference multiple nodes
4. **Path-aware execution** that tracks the full match, not just wavefronts

---

## Current GFQL Architecture

### Declarative 3-Pass Execution Model

GFQL is **declarative** and uses a sophisticated 3-pass execution strategy:

**Pass 1: FORWARD** (Optimistic prefix construction)
- Build up growing prefix tries
- Each node visited, expanding possible paths
- Creates wavefront of partial paths

**Pass 2: BACKWARD** (Pruning)
- Identify prefixes that never completed (dead ends)
- Prune unreachable paths from search space
- Uses `target_wave_front` to constrain reachability

**Pass 3: FORWARD** (Final execution)
- Execute with pruned search space
- Much more efficient due to backward pruning
- Returns final matched paths

**From `graphistry/compute/ast.py`**:
```python
class ASTNode:
    def __call__(self, g, prev_node_wavefront, target_wave_front, engine):
        # prev_node_wavefront: Nodes from previous step
        # target_wave_front: Reachable nodes from backward pass (pruning)
        # Returns: graph with matched nodes only

class ASTEdge:
    def __call__(self, g, prev_node_wavefront, target_wave_front, engine):
        # Hops from prev_node_wavefront
        # target_wave_front: Limits reachability (backward pass result)
        # Returns: graph with newly reached nodes as wavefront
```

**Key Limitation**: `prev_node_wavefront` is a DataFrame of node IDs, **not** full path history.

### Architectural Implications for Path Variables

The 3-pass model has critical implications:

1. **Backward Pass Constraint**: Path tracking must work with `target_wave_front` pruning
2. **Path IDs**: Must be stable across forward→backward→forward passes
3. **Partial Paths**: Forward pass creates prefix tries that may be pruned
4. **Predicate Timing**: Cross-variable predicates affect both forward and backward passes

### Current Name Support

GFQL supports `name` parameter:
- `n({'type': 'person'}, name='people')` → adds boolean column `people=True`
- `e_forward(name='e1')` → adds boolean column `e1=True`

**BUT**: These are just markers, not variable bindings you can query against!

---

## How Other Query Languages Handle This

### 1. Cypher (Neo4j)

**Named Variables**:
```cypher
MATCH (n1:Person)-[e1:KNOWS]->(n2:Person)-[e2:KNOWS]->(n3:Person)
WHERE n1.city = n3.city AND n1 <> n3
RETURN n1, e1, n2, e2, n3
```

**Key Features**:
- Each node/edge gets a **named variable** (n1, e1, n2, etc.)
- Variables are **accessible in WHERE clause**
- Can compare properties across variables: `n1.city = n3.city`
- Can return any subset of variables

**Path Variables**:
```cypher
MATCH p = (n1)-[*1..3]->(n2)
WHERE ALL(node IN nodes(p) WHERE node.type = 'person')
RETURN p
```

### 2. GQL (ISO Standard 2024)

**Quantified Path Patterns**:
```gql
MATCH (n1:Person) -[e:KNOWS]->+ (n2:Person)
WHERE n1.city = n2.city
```

**Key Features**:
- Variable-length paths with `+` or `*` quantifiers
- Path-level predicates
- Similar variable semantics to Cypher

### 3. GSQL (TigerGraph)

**Pattern Matching (V3 Syntax)**:
```gsql
SELECT n1, n2, n3
FROM Person:n1 -((KNOWS):e1)-> Person:n2 -((KNOWS):e2)-> Person:n3
WHERE n1.city == n3.city
```

**Key Features**:
- Explicit variable naming in FROM clause
- Cross-variable predicates in WHERE
- Aligned with GQL standard

### 4. Gremlin (Apache TinkerPop)

**Step Labels**:
```gremlin
g.V().hasLabel('person').as('n1')
  .out('knows').as('n2')
  .out('knows').as('n3')
  .where('n1', eq('n3')).by('city')
  .select('n1', 'n2', 'n3')
```

**Key Features**:
- `.as('label')` creates step labels
- `.where('n1', eq('n3'))` references earlier labels
- `.select()` projects named variables

---

## Design Challenges for GFQL

### Challenge 1: DataFrame Model Compatibility

GFQL operates on **full graphs** (node/edge DataFrames), not individual paths.

**Current Return**:
```python
g.gfql([n(), e_forward(), n()])  # Returns: Plottable with filtered nodes/edges
```

**Problem**: How to represent **multiple variable bindings** in a DataFrame?

**Options**:
1. **Path DataFrame**: One row per path, columns for each variable
   ```
   | n1_id | e1_id | n2_id | e2_id | n3_id |
   |-------|-------|-------|-------|-------|
   | 'A'   | 1     | 'B'   | 2     | 'C'   |
   ```
   - ✅ Natural for cross-variable predicates
   - ❌ Breaks existing graph model
   - ❌ Doesn't support mark() and other graph operations

2. **Column Explosion**: Flatten path into node DataFrame
   ```
   _nodes:
   | node | n1_match | n2_match | n3_match | n1_props_x | n3_props_x |
   |------|----------|----------|----------|------------|------------|
   | 'A'  | True     | False    | False    | 10         | None       |
   | 'B'  | False    | True     | False    | None       | None       |
   | 'C'  | False    | False    | True     | None       | 10         |
   ```
   - ❌ Doesn't capture path structure
   - ❌ Can't express cross-node predicates easily

3. **Path Column**: Add path column to edges
   ```
   _edges:
   | src | dst | path_id | step | n1_id | n2_id | n3_id |
   |-----|-----|---------|------|-------|-------|-------|
   | 'A' | 'B' | 0       | 1    | 'A'   | 'B'   | 'C'   |
   | 'B' | 'C' | 0       | 2    | 'A'   | 'B'   | 'C'   |
   ```
   - ✅ Preserves graph model
   - ✅ Allows path-level operations
   - ❌ Denormalized (path data repeated)

### Challenge 2: Execution Strategy

**Wavefront Model**:
- Efficient for reachability queries
- Scales to large graphs
- Enables distributed execution

**Path Tracking**:
- Must maintain full path history
- Potentially expensive for large result sets
- Harder to optimize

**Hybrid Approach**?
- Wavefront by default
- Path tracking only when needed (e.g., cross-variable predicates detected)

### Challenge 3: Syntax Design

How should users express variable bindings?

**Option A: Cypher-like (explicit variables)**
```python
# Hypothetical syntax
g.gfql([
    n({'type': 'person'}, var='n1'),
    e_forward(var='e1'),
    n(var='n2'),
    e_forward(var='e2'),
    n(var='n3'),
    where(eq('n1.x', 'n3.x'))  # Cross-variable predicate
])
```

**Option B: Implicit position-based**
```python
# Variables auto-named: n0, e0, n1, e1, n2
g.gfql([
    n({'type': 'person'}),
    e_forward(),
    n(),
    e_forward(),
    n({'x': ref('n0.x')})  # Reference earlier node
])
```

**Option C: Path object**
```python
# Define pattern with path
p = path([
    n({'type': 'person'}),
    e_forward(),
    n(),
    e_forward(),
    n()
])

# Query with path predicates
g.gfql(p.where(lambda vars: vars['n0']['x'] == vars['n2']['x']))
```

### Challenge 4: Backward Compatibility

- Existing GFQL code must continue to work
- `name` parameter already exists (adds boolean marker)
- Can't break wavefront model for simple queries

---

## Proposed Solutions

### Approach 1: Path Context Object (Recommended)

**Idea**: Introduce optional path tracking that coexists with wavefront model.

**API Design**:
```python
# Opt-in path tracking with vars()
g.gfql(
    vars('n1', 'n2', 'n3'),  # Declare variables to track
    [
        n({'type': 'person'}),      # Binds to n1
        e_forward(),
        n(),                         # Binds to n2
        e_forward(),
        n(),                         # Binds to n3
        # Cross-variable predicate using query expression
        n(query='@n1_x == @n3_x')   # Access via special vars
    ]
)
```

**Execution**:
1. Detect `vars()` → enable path tracking
2. Execute pattern, accumulating path DataFrame
3. Inject variables into query namespace
4. Filter paths based on cross-variable predicates

**Benefits**:
- ✅ Backward compatible (vars() is opt-in)
- ✅ Works with DataFrame query() syntax
- ✅ Clear execution model

**Drawbacks**:
- ❌ Requires path materialization (memory overhead)
- ❌ Query syntax a bit awkward (`@n1_x`)

### Approach 2: Let-based Variable Binding

**Idea**: Use let() DAG with named node sets, then join.

**API Design**:
```python
g.gfql(let({
    'n1': [n({'type': 'person'})],
    'n1_neighbors': [ref('n1'), e_forward(), n()],
    'n1_2hop': [ref('n1_neighbors'), e_forward(), n()],
    # Join n1 and n1_2hop where property matches
    'result': call('join_on_property', {
        'left': 'n1',
        'right': 'n1_2hop',
        'condition': 'x == x'
    })
}))
```

**Benefits**:
- ✅ Uses existing let() infrastructure
- ✅ Explicit variable names
- ✅ Composable

**Drawbacks**:
- ❌ Verbose syntax
- ❌ Requires new join operation
- ❌ Not clear how to handle intermediate nodes

### Approach 3: Path Column Extension

**Idea**: Extend hop() to optionally track path IDs and variable columns.

**Implementation**:
```python
# In hop.py, add path tracking:
def hop(g, nodes, track_path=False, path_vars=None):
    if track_path:
        # Add __gfql_path_id__ column
        # Add var columns for each named variable
        ...
```

**API Design**:
```python
g.gfql([
    n({'type': 'person'}, path_var='n1'),
    e_forward(path_var='e1'),
    n(path_var='n2'),
    e_forward(path_var='e2'),
    n(path_var='n3', query='__gfql_n1_x__ == __gfql_n3_x__')
])
```

**Benefits**:
- ✅ Integrates with existing architecture
- ✅ Uses internal columns for path data
- ✅ Efficient (columns, not separate DataFrame)

**Drawbacks**:
- ❌ Column explosion for complex patterns
- ❌ Internal column management complexity

---

## Next Steps

### Phase 1: Prototype (Current)
- [ ] Create minimal path tracking proof-of-concept
- [ ] Test with simple 2-hop pattern and cross-node predicate
- [ ] Measure memory/performance overhead
- [ ] Choose preferred approach

### Phase 2: Design Refinement
- [ ] Finalize syntax for variable binding
- [ ] Design cross-variable predicate API
- [ ] Specify path DataFrame schema
- [ ] Document execution semantics

### Phase 3: Implementation
- [ ] Extend AST classes with variable binding support
- [ ] Implement path tracking in hop/chain execution
- [ ] Add cross-variable predicate evaluation
- [ ] Update validation and error handling

### Phase 4: Testing & Documentation
- [ ] Comprehensive test suite
- [ ] Performance benchmarks
- [ ] User documentation
- [ ] Migration guide

---

## Key Insight: Path-Aware Wavefront Model

**The Core Technical Challenge** (from user feedback):

Current wavefront tracks **node IDs only**:
```python
wavefront = DataFrame([
    {'node': 'A'},
    {'node': 'B'},
    {'node': 'C'}
])
```

**Needed**: Wavefront must track **growing path expressions**:
```python
# Step 1: After n1 match
wavefront = DataFrame([
    {'path_id': 0, 'n1': 'A', 'n1_x': 3},
    {'path_id': 1, 'n1': 'B', 'n1_x': 4}
])

# Step 2: After e1 hop
wavefront = DataFrame([
    {'path_id': 0, 'n1': 'A', 'n1_x': 3, 'e1': edge_1, 'n2': 'D'},
    {'path_id': 1, 'n1': 'B', 'n1_x': 4, 'e1': edge_2, 'n2': 'E'}
])

# Step 3: After n3 match with WHERE n1.x == n3.x
wavefront = DataFrame([
    {'path_id': 0, 'n1': 'A', 'n1_x': 3, 'e1': edge_1, 'n2': 'D', 'e2': edge_3, 'n3': 'F', 'n3_x': 3},
    # path_id 1 filtered out because n1_x=4 != n3_x=5
])
```

**Execution Model**:
1. Each pattern step **enriches** the wavefront with new variables
2. Cross-variable predicates **filter** wavefront based on full path state
3. Final result: Set of complete paths satisfying all constraints

**Architecture Implications**:
- Wavefront DataFrame grows horizontally (new columns per variable)
- Each row represents a **partial path** being explored
- Path explosion managed through filtering at each step
- Backward compatible: Single-variable queries don't track extra columns

---

## Proposed Approach: Path Context Columns

Based on the wavefront insight, here's the refined approach:

### Data Model

**Extended Wavefront Schema**:
```python
# Example: MATCH (n1)-[e1]->(n2)-[e2]->(n3) WHERE n1.x == n3.x

# Step 1: n1 match → Initialize path context
wavefront: DataFrame
Columns: [__gfql_path_id__, n1, n1_x, ...]
```

**Path Tracking Mechanics**:
1. **Variable Binding**: When `var='n1'` specified, add columns for that variable
   - `n1`: node/edge ID
   - `n1_<prop>`: Denormalized property values needed for cross-variable predicates
2. **Path ID**: Unique identifier per path (`__gfql_path_id__`)
3. **Wavefront Evolution**: Each hop **joins** with previous wavefront to propagate path state

### Implementation Strategy

**Modified hop() signature**:
```python
def hop(
    g,
    nodes,  # Current wavefront (may include path context columns)
    target_wave_front,  # CRITICAL: From backward pass pruning
    ...,
    path_context: Optional[PathContext] = None  # NEW
):
    if path_context is not None:
        # FORWARD PASS: Join results with path context to propagate variable bindings
        # Must respect target_wave_front constraint from backward pass
        hop_result = _hop_impl(g, nodes, target_wave_front, ...)
        new_wavefront = merge_with_path_context(hop_result, nodes, path_context)

        # Apply cross-variable predicates during forward pass
        new_wavefront = path_context.filter_wavefront(new_wavefront)
    else:
        # Existing behavior
        new_wavefront = _hop_impl(g, nodes, target_wave_front, ...)

    return new_wavefront
```

**Modified chain() for 3-pass execution with path context**:
```python
def chain(g, ast_chain, engine):
    if has_path_variables(ast_chain):
        ctx = extract_path_context(ast_chain)

        # PASS 1: FORWARD (optimistic, track path IDs)
        forward_result, path_prefixes = _forward_pass_with_paths(g, ast_chain, ctx, engine)

        # PASS 2: BACKWARD (prune dead-end paths)
        # Must preserve path_id tracking through backward pass
        backward_result, valid_paths = _backward_pass_with_paths(g, ast_chain, forward_result, ctx, engine)

        # PASS 3: FORWARD (with pruned search space and path filtering)
        final_result = _forward_pass_with_paths(g, ast_chain, ctx, engine, valid_paths)

        # Apply final cross-variable predicates
        final_result = ctx.filter_final_paths(final_result)

        return final_result
    else:
        # Existing 3-pass execution without path tracking
        return _chain_impl(g, ast_chain, engine)
```

**PathContext class**:
```python
class PathContext:
    """Tracks variable bindings and cross-variable constraints."""

    def __init__(self):
        self.variables = {}  # {var_name: properties_to_track}
        self.predicates = []  # Cross-variable WHERE clauses
        self.path_id_col = '__gfql_path_id__'

    def add_variable(self, var_name: str, properties: List[str]):
        """Register a variable and which properties to denormalize."""
        ...

    def add_predicate(self, predicate: Callable):
        """Add cross-variable constraint."""
        ...

    def filter_wavefront(self, wavefront: DataFrame) -> DataFrame:
        """Apply cross-variable predicates to filter paths."""
        ...
```

**Example Execution**:
```python
# User query:
g.gfql([
    n({'type': 'person'}, var='n1'),
    e_forward(var='e1'),
    n(var='n2'),
    e_forward(var='e2'),
    n(var='n3', query='@n1_x == @n3_x')  # Cross-variable predicate
])

# Internal execution:
ctx = PathContext()
ctx.add_variable('n1', properties=['x'])
ctx.add_variable('n3', properties=['x'])
ctx.add_predicate(lambda df: df['n1_x'] == df['n3_x'])

# Step 1: Match n1
wavefront = filter_nodes(g._nodes, {'type': 'person'})
wavefront['__gfql_path_id__'] = range(len(wavefront))
wavefront = wavefront.rename(columns={g._node: 'n1', 'x': 'n1_x'})

# Step 2: Hop e1
hop_result = hop(g, wavefront[[g._node]], ...)
wavefront = merge(wavefront, hop_result, on=g._node)  # Propagate path context
wavefront = wavefront.rename(columns={edge_id: 'e1', next_node: 'n2'})

# Step 3: Hop e2
hop_result = hop(g, wavefront[[g._node]], ...)
wavefront = merge(wavefront, hop_result, on=g._node)
wavefront = wavefront.rename(columns={edge_id: 'e2', next_node: 'n3'})

# Step 4: Apply cross-variable predicate
wavefront = ctx.filter_wavefront(wavefront)  # Filter n1_x == n3_x

# Return: Graph constructed from remaining paths
```

### Benefits

1. **Incremental Filtering**: Predicates applied at each step, pruning paths early
2. **DataFrame Native**: Uses merge/join operations familiar to users
3. **Opt-In**: PathContext only created when variables declared
4. **Backward Compatible**: Existing queries work unchanged
5. **Scalable**: Column-based storage more efficient than nested path objects

### Challenges

1. **Column Explosion**: Each variable adds columns (mitigated by only tracking needed properties)
2. **Path Explosion**: Cartesian product in wavefront (managed by incremental filtering)
3. **Memory**: Denormalized path data (trade-off for query expressiveness)
4. **3-Pass Complexity**: Path tracking must survive forward→backward→forward execution
5. **Path ID Stability**: IDs assigned in first forward pass must remain valid after backward pruning
6. **Predicate Placement**: Where to apply cross-variable predicates in 3-pass model?
   - Option A: Only in final forward pass (simpler, less efficient)
   - Option B: In both forward passes (more complex, more efficient)
   - Option C: Analyze dependencies and apply as early as possible

---

## Open Questions

1. **Syntax**: Explicit `var='n1'` or implicit position-based naming?
2. **Scope**: Only for patterns, or also for let() bindings?
3. **Performance**: Path tracking always-on or opt-in?
4. **Return Value**: Return Plottable (current) or Path object (new)?
5. **Predicates**: Query string (`query='n1.x == n3.x'`) or AST-based?
6. **Interop**: How does this relate to future Cypher/GQL support?
7. **Property Selection**: Auto-detect needed properties or explicit declaration?
8. **Path Pruning**: When to apply cross-variable predicates (each step vs end)?

---

## References

- [Issue #722](https://github.com/graphistry/pygraphistry/issues/722) - Original feature request
- [Cypher Path Patterns](https://neo4j.com/docs/cypher-manual/current/patterns/)
- [GQL ISO Standard 2024](https://www.gqlstandards.org/)
- [GSQL Pattern Matching](https://docs.tigergraph.com/gsql-ref/current/tutorials/pattern-matching/)
- [Gremlin Path Steps](https://tinkerpop.apache.org/docs/current/reference/)
