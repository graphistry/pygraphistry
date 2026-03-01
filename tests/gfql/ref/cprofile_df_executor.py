"""
cProfile analysis of df_executor to find hotspots.

Run with:
    python -m tests.gfql.ref.cprofile_df_executor
"""
import cProfile
import pstats
import io

import graphistry
from graphistry.compute.gfql.same_path_types import where_to_json
from tests.gfql.ref.profile_utils import make_dense_graph, multihop_chain, simple_chain, simple_where


def profile_simple_query(g, n_runs=5):
    chain = simple_chain()
    for _ in range(n_runs):
        g.gfql({"chain": chain, "where": []}, engine="pandas")


def profile_multihop_query(g, n_runs=5):
    chain = multihop_chain()
    for _ in range(n_runs):
        g.gfql({"chain": chain, "where": []}, engine="pandas")


def profile_where_query(g, n_runs=5):
    chain = simple_chain()
    where_json = where_to_json(simple_where())
    for _ in range(n_runs):
        g.gfql({"chain": chain, "where": where_json}, engine="pandas")


def profile_samepath_query(g_small, n_runs=5):
    from graphistry.compute.gfql.df_executor import (
        build_same_path_inputs,
        execute_same_path_chain,
    )
    from graphistry.Engine import Engine

    chain = simple_chain()
    where = simple_where()

    for _ in range(n_runs):
        inputs = build_same_path_inputs(
            g_small,
            chain,
            where,
            engine=Engine.PANDAS,
            include_paths=False,
        )
        execute_same_path_chain(
            inputs.graph,
            inputs.chain,
            inputs.where,
            inputs.engine,
            inputs.include_paths,
        )


def run_profile(func, g, name):
    print(f"\n{'='*60}")
    print(f"Profiling: {name}")
    print(f"{'='*60}")

    profiler = cProfile.Profile()
    profiler.enable()
    func(g)
    profiler.disable()

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(30)  # Top 30 functions
    print(s.getvalue())


def main():
    print("Creating large graph: 50K nodes, 200K edges")
    nodes_df, edges_df = make_dense_graph(50000, 200000)
    g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    print(f"Large graph: {len(nodes_df)} nodes, {len(edges_df)} edges")

    print("Creating small graph: 1K nodes, 2K edges")
    nodes_small, edges_small = make_dense_graph(1000, 2000)
    g_small = graphistry.nodes(nodes_small, 'id').edges(edges_small, 'src', 'dst')
    print(f"Small graph: {len(nodes_small)} nodes, {len(edges_small)} edges")

    print("\nWarmup...")
    chain = simple_chain()
    g.gfql({"chain": chain, "where": []}, engine="pandas")

    # Profile legacy chain on large graph
    run_profile(profile_simple_query, g, "Simple query (n->e->n) - legacy chain, 50K nodes")
    run_profile(profile_multihop_query, g, "Multihop query (n->e(1..3)->n) - legacy chain, 50K nodes")
    run_profile(profile_where_query, g, "WHERE query (a.v < c.v) - legacy chain, 50K nodes")

    # Profile same-path executor on small graph (oracle has caps)
    run_profile(lambda g: profile_samepath_query(g_small), g, "Same-path executor (n->e->n, a.v < c.v) - 1K nodes")


if __name__ == "__main__":
    main()
