#!/usr/bin/env python3
"""
Direct hop() microbenchmarks for common traversal shapes.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd

import graphistry
from graphistry.Engine import Engine


@dataclass(frozen=True)
class Scenario:
    name: str
    hops: int
    direction: str
    seed_mode: str  # "seed0" | "all"
    return_as_wave_front: bool = True


@dataclass(frozen=True)
class GraphSpec:
    name: str
    nodes: int
    edges: int
    kind: str  # "linear" | "dense"


@dataclass
class ResultRow:
    graph: str
    scenario: str
    ms: Optional[float]


def make_linear_graph(n_nodes: int, n_edges: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame({"id": list(range(n_nodes))})
    edges_list = []
    for i in range(min(n_edges, n_nodes - 1)):
        edges_list.append({"src": i, "dst": i + 1, "eid": i})
    edges = pd.DataFrame(edges_list)
    return nodes, edges


def make_dense_graph(n_nodes: int, n_edges: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import random

    random.seed(42)
    nodes = pd.DataFrame({"id": list(range(n_nodes))})
    edges_list = []
    for i in range(n_edges):
        src = random.randint(0, n_nodes - 2)
        dst = random.randint(src + 1, n_nodes - 1)
        edges_list.append({"src": src, "dst": dst, "eid": i})
    edges = pd.DataFrame(edges_list).drop_duplicates(subset=["src", "dst"])
    return nodes, edges


def build_graph(spec: GraphSpec, engine: Engine):
    if spec.kind == "dense":
        nodes_df, edges_df = make_dense_graph(spec.nodes, spec.edges)
    else:
        nodes_df, edges_df = make_linear_graph(spec.nodes, spec.edges)

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


def run_scenarios(g, scenarios: List[Scenario], runs: int) -> Iterable[ResultRow]:
    for scenario in scenarios:
        seed_nodes = None
        if scenario.seed_mode == "seed0":
            seed_nodes = g._nodes[g._nodes["id"] == 0]

        def _call() -> None:
            g.hop(
                nodes=seed_nodes,
                hops=scenario.hops,
                to_fixed_point=False,
                direction=scenario.direction,
                return_as_wave_front=scenario.return_as_wave_front,
            )

        ms = _time_call(_call, runs)
        yield ResultRow(graph="", scenario=scenario.name, ms=ms)


def build_scenarios() -> List[Scenario]:
    return [
        Scenario("2hop_forward_seed0", 2, "forward", "seed0", True),
        Scenario("2hop_forward_all", 2, "forward", "all", True),
        Scenario("2hop_undirected_seed0", 2, "undirected", "seed0", True),
        Scenario("2hop_undirected_all", 2, "undirected", "all", True),
    ]


def build_graph_specs() -> List[GraphSpec]:
    return [
        GraphSpec("small_linear", 1_000, 2_000, "linear"),
        GraphSpec("medium_linear", 10_000, 20_000, "linear"),
        GraphSpec("medium_dense", 10_000, 50_000, "dense"),
    ]


def write_markdown(results: Iterable[ResultRow], output_path: str) -> None:
    header = [
        "# Hop Microbench Results",
        "",
        "Notes:",
        "- Direct hop() calls; no WHERE predicates.",
        "",
        "| Graph | Scenario | Time |",
        "|-------|----------|------|",
    ]
    lines = header + [
        f"| {row.graph} | {row.scenario} | {row.ms:.2f}ms |" for row in results
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hop microbenchmarks.")
    parser.add_argument("--engine", default="pandas", choices=["pandas", "cudf"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    engine = Engine.CUDF if args.engine == "cudf" else Engine.PANDAS
    scenarios = build_scenarios()
    results: List[ResultRow] = []
    for spec in build_graph_specs():
        g = build_graph(spec, engine)
        for row in run_scenarios(g, scenarios, args.runs):
            row.graph = spec.name
            results.append(row)

    if args.output:
        write_markdown(results, args.output)

    print("| Graph | Scenario | Time |")
    print("|-------|----------|------|")
    for row in results:
        print(f"| {row.graph} | {row.scenario} | {row.ms:.2f}ms |")


if __name__ == "__main__":
    main()
