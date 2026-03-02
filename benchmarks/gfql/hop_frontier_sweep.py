#!/usr/bin/env python3
"""
Frontier-size sweep for hop() on a fixed graph.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd

import graphistry
from graphistry.Engine import Engine


@dataclass
class ResultRow:
    graph: str
    seed_size: int
    ms: Optional[float]


def make_linear_graph(n_nodes: int, n_edges: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame({"id": list(range(n_nodes))})
    edges_list = []
    for i in range(min(n_edges, n_nodes - 1)):
        edges_list.append({"src": i, "dst": i + 1, "eid": i})
    edges = pd.DataFrame(edges_list)
    return nodes, edges


def build_graph(n_nodes: int, n_edges: int, engine: Engine):
    nodes_df, edges_df = make_linear_graph(n_nodes, n_edges)
    if engine == Engine.CUDF:
        import cudf  # type: ignore

        nodes_df = cudf.from_pandas(nodes_df)
        edges_df = cudf.from_pandas(edges_df)
    return graphistry.nodes(nodes_df, "id").edges(edges_df, "src", "dst")


def _time_call(fn, runs: int) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)


def run_sweep(g, seed_sizes: List[int], runs: int) -> Iterable[ResultRow]:
    for seed_size in seed_sizes:
        seed_nodes = g._nodes.head(seed_size)

        def _call() -> None:
            g.hop(
                nodes=seed_nodes,
                hops=2,
                to_fixed_point=False,
                direction="forward",
                return_as_wave_front=True,
            )

        ms = _time_call(_call, runs)
        yield ResultRow(graph="", seed_size=seed_size, ms=ms)


def write_markdown(results: Iterable[ResultRow], output_path: str) -> None:
    header = [
        "# Hop Frontier Sweep",
        "",
        "Notes:",
        "- Fixed linear graph, forward 2-hop, return_as_wave_front=True.",
        "",
        "| Graph | Seed Size | Time |",
        "|-------|-----------|------|",
    ]
    lines = header + [
        f"| {row.graph} | {row.seed_size} | {row.ms:.2f}ms |" for row in results
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hop frontier sweep.")
    parser.add_argument("--engine", default="pandas", choices=["pandas", "cudf"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=100000)
    parser.add_argument("--edges", type=int, default=200000)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--seed-sizes",
        default="1,10,100,1000,10000",
        help="Comma-separated list of seed sizes",
    )
    args = parser.parse_args()

    engine = Engine.CUDF if args.engine == "cudf" else Engine.PANDAS
    seed_sizes = [int(x) for x in args.seed_sizes.split(",") if x.strip()]

    g = build_graph(args.nodes, args.edges, engine)
    results = list(run_sweep(g, seed_sizes, args.runs))
    for row in results:
        row.graph = f"linear_{args.nodes}"

    if args.output:
        write_markdown(results, args.output)

    print("| Graph | Seed Size | Time |")
    print("|-------|-----------|------|")
    for row in results:
        print(f"| {row.graph} | {row.seed_size} | {row.ms:.2f}ms |")


if __name__ == "__main__":
    main()
