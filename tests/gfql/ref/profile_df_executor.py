"""
Profile df_executor to identify optimization opportunities.

Run with:
    python -m tests.gfql.ref.profile_df_executor

Outputs timing data for different chain complexities and graph sizes.
"""
import time
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Import the executor and test utilities
import graphistry
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected
from graphistry.gfql.same_path_types import WhereComparison, StepColumnRef, col, compare, where_to_json


@dataclass
class ProfileResult:
    scenario: str
    nodes: int
    edges: int
    chain_desc: str
    where_desc: str
    time_ms: float
    result_nodes: int
    result_edges: int


def make_linear_graph(n_nodes: int, n_edges: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a linear graph: 0 -> 1 -> 2 -> ... -> n-1"""
    nodes = pd.DataFrame({
        'id': list(range(n_nodes)),
        'v': list(range(n_nodes)),
    })
    # Create edges ensuring we don't exceed available nodes
    edges_list = []
    for i in range(min(n_edges, n_nodes - 1)):
        edges_list.append({'src': i, 'dst': i + 1, 'eid': i})
    edges = pd.DataFrame(edges_list)
    return nodes, edges


def make_dense_graph(n_nodes: int, n_edges: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a denser graph with multiple paths."""
    import random
    random.seed(42)

    nodes = pd.DataFrame({
        'id': list(range(n_nodes)),
        'v': list(range(n_nodes)),
    })

    edges_list = []
    for i in range(n_edges):
        src = random.randint(0, n_nodes - 2)
        dst = random.randint(src + 1, n_nodes - 1)
        edges_list.append({'src': src, 'dst': dst, 'eid': i})
    edges = pd.DataFrame(edges_list).drop_duplicates(subset=['src', 'dst'])

    return nodes, edges


def profile_query(
    g: graphistry.Plottable,
    chain: List[Any],
    where: List[WhereComparison],
    scenario: str,
    n_nodes: int,
    n_edges: int,
    n_runs: int = 3
) -> ProfileResult:
    """Profile a single query, return average time."""

    from graphistry.compute.chain import Chain

    # Convert WHERE to JSON format
    where_json = where_to_json(where) if where else []

    # Warmup
    result = g.gfql({"chain": chain, "where": where_json}, engine="pandas")

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = g.gfql({"chain": chain, "where": where_json}, engine="pandas")
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    avg_time = sum(times) / len(times)

    chain_desc = " -> ".join(str(type(op).__name__) for op in chain)
    where_desc = str(len(where)) + " clauses" if where else "none"

    return ProfileResult(
        scenario=scenario,
        nodes=n_nodes,
        edges=n_edges,
        chain_desc=chain_desc,
        where_desc=where_desc,
        time_ms=avg_time,
        result_nodes=len(result._nodes) if result._nodes is not None else 0,
        result_edges=len(result._edges) if result._edges is not None else 0,
    )


def run_profiles() -> List[ProfileResult]:
    """Run all profiling scenarios."""
    results = []

    # Define scenarios
    scenarios = [
        # (name, n_nodes, n_edges, graph_type)
        ('tiny', 100, 200, 'linear'),
        ('small', 1000, 2000, 'linear'),
        ('medium', 10000, 20000, 'linear'),
        ('medium_dense', 10000, 50000, 'dense'),
        ('large', 100000, 200000, 'linear'),
        ('large_dense', 100000, 500000, 'dense'),
    ]

    for scenario_name, n_nodes, n_edges, graph_type in scenarios:
        print(f"\n=== Scenario: {scenario_name} ({n_nodes} nodes, {n_edges} edges, {graph_type}) ===")

        if graph_type == 'linear':
            nodes_df, edges_df = make_linear_graph(n_nodes, n_edges)
        else:
            nodes_df, edges_df = make_dense_graph(n_nodes, n_edges)

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Chain variants
        chains = [
            ("simple", [n(name="a"), e_forward(name="e"), n(name="c")], []),

            ("with_filter", [
                n({"id": 0}, name="a"),
                e_forward(name="e"),
                n(name="c")
            ], []),

            ("with_where_adjacent", [
                n(name="a"),
                e_forward(name="e"),
                n(name="c")
            ], [compare(col("a", "v"), "<", col("c", "v"))]),

            ("multihop", [
                n({"id": 0}, name="a"),
                e_forward(min_hops=1, max_hops=3, name="e"),
                n(name="c")
            ], []),

            ("multihop_with_where", [
                n({"id": 0}, name="a"),
                e_forward(min_hops=1, max_hops=3, name="e"),
                n(name="c")
            ], [compare(col("a", "v"), "<", col("c", "v"))]),
        ]

        for chain_name, chain, where in chains:
            try:
                result = profile_query(
                    g, chain, where,
                    f"{scenario_name}_{chain_name}",
                    n_nodes, n_edges
                )
                results.append(result)
                print(f"  {chain_name}: {result.time_ms:.2f}ms "
                      f"(nodes={result.result_nodes}, edges={result.result_edges})")
            except Exception as e:
                print(f"  {chain_name}: ERROR - {e}")

    return results


def main():
    print("=" * 60)
    print("GFQL df_executor Profiling")
    print("=" * 60)

    results = run_profiles()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Group by scenario type
    print("\nTiming by scenario:")
    for r in results:
        print(f"  {r.scenario}: {r.time_ms:.2f}ms")

    # Identify hotspots
    print("\nSlowest queries:")
    sorted_results = sorted(results, key=lambda x: x.time_ms, reverse=True)
    for r in sorted_results[:5]:
        print(f"  {r.scenario}: {r.time_ms:.2f}ms")


if __name__ == "__main__":
    main()
