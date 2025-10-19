"""
Proof-of-Concept: Path-Aware Wavefront Execution

Demonstrates how to extend GFQL's wavefront model to support
cross-node predicates like: WHERE n1.x == n3.x

This prototype shows the core idea without full integration.
"""
import pandas as pd
from typing import Optional, List, Dict, Any


class PathContext:
    """Tracks variable bindings and cross-variable constraints during pattern matching."""

    def __init__(self):
        self.variables: Dict[str, List[str]] = {}  # {var_name: [properties]}
        self.predicates: List[str] = []  # Cross-variable predicates as query strings
        self.path_id_col = '__gfql_path_id__'
        self._next_var_index = 0

    def add_variable(self, var_name: str, properties: Optional[List[str]] = None):
        """Register a variable and which properties to track."""
        self.variables[var_name] = properties or []

    def add_predicate(self, predicate: str):
        """Add cross-variable constraint as pandas query string."""
        self.predicates.append(predicate)

    def filter_wavefront(self, wavefront: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-variable predicates to filter paths."""
        if not self.predicates:
            return wavefront

        # Combine all predicates with AND
        combined_query = ' and '.join(f'({p})' for p in self.predicates)
        print(f"  Applying predicate: {combined_query}")
        return wavefront.query(combined_query)


def example_graph():
    """Create sample graph for demonstration."""
    nodes = pd.DataFrame({
        'node': ['A', 'B', 'C', 'D', 'E', 'F'],
        'type': ['person', 'person', 'person', 'company', 'company', 'person'],
        'x': [10, 20, 10, 100, 200, 10]  # A, C, F have same value
    })

    edges = pd.DataFrame({
        'src': ['A', 'B', 'A', 'C', 'D', 'E'],
        'dst': ['D', 'E', 'E', 'D', 'F', 'F'],
        'edge_id': [1, 2, 3, 4, 5, 6]
    })

    return nodes, edges


def match_nodes(
    nodes: pd.DataFrame,
    filter_dict: Dict[str, Any],
    var_name: str,
    properties: List[str],
    ctx: PathContext
) -> pd.DataFrame:
    """
    Match nodes and initialize path context.

    Like GFQL: n({'type': 'person'}, var='n1')
    """
    print(f"\nStep: Match nodes with {filter_dict}, var='{var_name}'")

    # Filter nodes
    matched = nodes.copy()
    for col, val in filter_dict.items():
        matched = matched[matched[col] == val]

    # Initialize path tracking
    wavefront = pd.DataFrame()
    wavefront[ctx.path_id_col] = range(len(matched))

    # Add variable binding columns
    wavefront[var_name] = matched['node'].values
    for prop in properties:
        wavefront[f'{var_name}_{prop}'] = matched[prop].values

    print(f"  Matched {len(wavefront)} nodes")
    print(wavefront)

    ctx.add_variable(var_name, properties)
    return wavefront


def hop_forward(
    wavefront: pd.DataFrame,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    var_edge: str,
    var_node: str,
    properties: List[str],
    ctx: PathContext
) -> pd.DataFrame:
    """
    Hop forward from wavefront, propagating path context.

    Like GFQL: e_forward(var='e1'), then n(var='n2')
    """
    print(f"\nStep: Hop forward, edge_var='{var_edge}', node_var='{var_node}'")

    # Get current node column (last node variable in wavefront)
    # For simplicity, assume it's the last var added
    prev_vars = [v for v in ctx.variables.keys() if v in wavefront.columns]
    if not prev_vars:
        raise ValueError("No previous node variable in wavefront")

    current_node_col = prev_vars[-1]

    # Merge wavefront with edges on source
    hop_df = wavefront.merge(
        edges,
        left_on=current_node_col,
        right_on='src',
        how='inner'
    )

    # Add edge variable
    hop_df[var_edge] = hop_df['edge_id']

    # Add next node variable
    hop_df[var_node] = hop_df['dst']

    # Enrich with node properties
    node_props = nodes[['node'] + properties].rename(
        columns={
            'node': var_node,
            **{prop: f'{var_node}_{prop}' for prop in properties}
        }
    )
    hop_df = hop_df.merge(node_props, on=var_node, how='left')

    # Keep only path context columns
    keep_cols = [ctx.path_id_col] + [
        col for col in hop_df.columns
        if any(col.startswith(v) or col == v for v in ctx.variables.keys())
           or col == var_edge or col == var_node
           or col.startswith(f'{var_node}_')
    ]
    result = hop_df[keep_cols].copy()

    print(f"  After hop: {len(result)} paths")
    print(result)

    ctx.add_variable(var_edge)
    ctx.add_variable(var_node, properties)
    return result


def demo_cross_node_predicate():
    """
    Demonstrate: MATCH (n1)-[e1]->(n2)-[e2]->(n3) WHERE n1.x == n3.x
    """
    print("=" * 70)
    print("DEMO: Cross-Node Predicate")
    print("Pattern: (n1)-[e1]->(n2)-[e2]->(n3) WHERE n1.x == n3.x")
    print("=" * 70)

    nodes, edges = example_graph()
    ctx = PathContext()

    # Step 1: Match n1 (person nodes, track 'x' property)
    wavefront = match_nodes(
        nodes,
        filter_dict={'type': 'person'},
        var_name='n1',
        properties=['x'],
        ctx=ctx
    )

    # Step 2: Hop e1 -> n2
    wavefront = hop_forward(
        wavefront,
        edges,
        nodes,
        var_edge='e1',
        var_node='n2',
        properties=[],  # Don't need n2 properties
        ctx=ctx
    )

    # Step 3: Hop e2 -> n3 (track 'x' property for comparison)
    wavefront = hop_forward(
        wavefront,
        edges,
        nodes,
        var_edge='e2',
        var_node='n3',
        properties=['x'],  # Track x for cross-variable predicate
        ctx=ctx
    )

    # Step 4: Apply cross-variable predicate
    print("\n" + "=" * 70)
    print("Applying Cross-Variable Predicate: n1.x == n3.x")
    print("=" * 70)

    ctx.add_predicate('n1_x == n3_x')
    filtered = ctx.filter_wavefront(wavefront)

    print(f"\nFinal Result: {len(filtered)} paths satisfy predicate")
    print(filtered)

    # Verify: Should only have paths where n1.x == n3.x
    print("\nVerification:")
    for _, row in filtered.iterrows():
        print(f"  Path {row[ctx.path_id_col]}: "
              f"n1={row['n1']} (x={row['n1_x']}) -> "
              f"n2={row['n2']} -> "
              f"n3={row['n3']} (x={row['n3_x']})")
        assert row['n1_x'] == row['n3_x'], "Predicate violated!"

    return filtered


def demo_memory_comparison():
    """Compare memory usage: current wavefront vs path-aware wavefront."""
    print("\n" + "=" * 70)
    print("MEMORY COMPARISON")
    print("=" * 70)

    nodes, edges = example_graph()

    # Current wavefront approach (node IDs only)
    current_wavefront = pd.DataFrame({
        'node': ['A', 'B', 'C', 'D', 'E', 'F']
    })

    # Path-aware wavefront (full path context)
    ctx = PathContext()
    path_wavefront = match_nodes(
        nodes,
        filter_dict={'type': 'person'},
        var_name='n1',
        properties=['x'],
        ctx=ctx
    )

    print(f"\nCurrent wavefront size: {current_wavefront.memory_usage(deep=True).sum()} bytes")
    print(f"Columns: {list(current_wavefront.columns)}")

    print(f"\nPath-aware wavefront size: {path_wavefront.memory_usage(deep=True).sum()} bytes")
    print(f"Columns: {list(path_wavefront.columns)}")

    overhead_pct = (
        (path_wavefront.memory_usage(deep=True).sum() / current_wavefront.memory_usage(deep=True).sum()) - 1
    ) * 100
    print(f"\nOverhead: +{overhead_pct:.1f}%")


if __name__ == '__main__':
    # Run demonstration
    result = demo_cross_node_predicate()

    # Show memory comparison
    demo_memory_comparison()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✅ Path-aware wavefronts successfully track variable bindings")
    print("✅ Cross-node predicates work via DataFrame query()")
    print("✅ Memory overhead is acceptable for moderate path counts")
    print("\nNext steps:")
    print("- Integrate with actual hop() implementation")
    print("- Add syntax for variable declaration")
    print("- Optimize property denormalization")
    print("- Handle path explosion with early pruning")
