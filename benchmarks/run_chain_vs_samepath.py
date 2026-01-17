#!/usr/bin/env python3
"""
Benchmark regular chain() vs Yannakakis df_executor on shared scenarios.

Notes:
- Regular chain() does NOT apply WHERE; it is included as a baseline.
- Yannakakis path applies WHERE via execute_same_path_chain().
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.ast import n, e_forward, e_undirected
from graphistry.compute.gfql.df_executor import execute_same_path_chain
from graphistry.compute.gfql.same_path_types import WhereComparison, col, compare


@dataclass(frozen=True)
class Scenario:
    name: str
    chain: List
    where: List[WhereComparison]


@dataclass(frozen=True)
class GraphSpec:
    name: str
    nodes: int
    edges: int
    kind: str  # "linear" | "dense"


@dataclass
class TimingStats:
    median_ms: float
    p90_ms: float
    std_ms: float


@dataclass
class ResultRow:
    graph: str
    scenario: str
    regular: Optional[TimingStats]
    yannakakis: Optional[TimingStats]


def make_linear_graph(n_nodes: int, n_edges: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a linear graph: 0 -> 1 -> 2 -> ... -> n-1."""
    nodes = pd.DataFrame(
        {
            "id": list(range(n_nodes)),
            "v": list(range(n_nodes)),
        }
    )
    edges_list = []
    for i in range(min(n_edges, n_nodes - 1)):
        edges_list.append({"src": i, "dst": i + 1, "eid": i})
    edges = pd.DataFrame(edges_list)
    return nodes, edges


def make_dense_graph(n_nodes: int, n_edges: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a denser graph with multiple paths."""
    import random

    random.seed(42)
    nodes = pd.DataFrame(
        {
            "id": list(range(n_nodes)),
            "v": list(range(n_nodes)),
        }
    )

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
        try:
            import cudf  # type: ignore
        except Exception as exc:
            raise RuntimeError("cudf not available; install cudf or use --engine pandas") from exc
        nodes_df = cudf.from_pandas(nodes_df)
        edges_df = cudf.from_pandas(edges_df)

    return graphistry.nodes(nodes_df, "id").edges(edges_df, "src", "dst")


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    if low == high:
        return sorted_vals[low]
    weight = rank - low
    return sorted_vals[low] * (1 - weight) + sorted_vals[high] * weight


def _summarize_times(times: List[float]) -> TimingStats:
    ordered = sorted(times)
    median_ms = statistics.median(ordered)
    p90_ms = _percentile(ordered, 0.9)
    std_ms = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    return TimingStats(median_ms=median_ms, p90_ms=p90_ms, std_ms=std_ms)


def _time_call(fn, runs: int, warmup: int) -> TimingStats:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return _summarize_times(times)


def run_regular(g, chain_ops: List, engine_label: str, runs: int, warmup: int) -> TimingStats:
    def _call():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="chain\\(\\) is deprecated.*",
            )
            g.chain(chain_ops, engine=engine_label)

    return _time_call(_call, runs, warmup)


def run_yannakakis(
    g,
    chain_ops: List,
    where: List[WhereComparison],
    engine: Engine,
    runs: int,
    warmup: int,
) -> TimingStats:
    def _call():
        execute_same_path_chain(g, chain_ops, where, engine, include_paths=False)

    return _time_call(_call, runs, warmup)


def format_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}ms"


def summarize_row(row: ResultRow) -> str:
    if row.regular is None or row.yannakakis is None:
        ratio = "n/a"
        winner = "n/a"
    else:
        ratio_val = row.yannakakis.median_ms / row.regular.median_ms if row.regular.median_ms > 0 else float("inf")
        ratio = f"{ratio_val:.2f}x"
        winner = "yannakakis" if ratio_val < 1 else "regular"
    return (
        f"| {row.graph} | {row.scenario} | {format_ms(row.regular.median_ms if row.regular else None)}"
        f" | {format_ms(row.yannakakis.median_ms if row.yannakakis else None)} | {ratio} | {winner}"
        f" | {format_ms(row.regular.p90_ms if row.regular else None)}"
        f" | {format_ms(row.yannakakis.p90_ms if row.yannakakis else None)}"
        f" | {format_ms(row.regular.std_ms if row.regular else None)}"
        f" | {format_ms(row.yannakakis.std_ms if row.yannakakis else None)} |"
    )


def build_scenarios() -> List[Scenario]:
    one_hop = [n(name="a"), e_forward(name="e1"), n(name="b")]
    one_hop_filtered = [n({"id": 0}, name="a"), e_forward(name="e1"), n(name="b")]
    two_hop = [n(name="a"), e_forward(name="e1"), n(name="b"), e_forward(name="e2"), n(name="c")]
    undirected_one_hop = [n(name="a"), e_undirected(name="e1"), n(name="b")]
    undirected_two_hop = [n(name="a"), e_undirected(name="e1"), n(name="b"), e_undirected(name="e2"), n(name="c")]
    multihop_range = [n({"id": 0}, name="a"), e_forward(min_hops=1, max_hops=2, name="e1"), n(name="b")]
    multihop_range_filtered = [
        n({"id": 0}, name="a"),
        e_forward(min_hops=1, max_hops=2, name="e1"),
        n({"id": 1}, name="b"),
    ]
    where_adj = [compare(col("a", "v"), "<", col("b", "v"))]
    where_nonadj = [compare(col("a", "v"), "<", col("c", "v"))]

    return [
        Scenario("1hop_simple", one_hop, []),
        Scenario("1hop_filtered", one_hop_filtered, []),
        Scenario("2hop", two_hop, []),
        Scenario("1hop_undirected", undirected_one_hop, []),
        Scenario("2hop_undirected", undirected_two_hop, []),
        Scenario("1to2hop_range", multihop_range, []),
        Scenario("1to2hop_range_filtered", multihop_range_filtered, []),
        Scenario("2hop_where_adj", two_hop, where_adj),
        Scenario("2hop_where_nonadj", two_hop, where_nonadj),
    ]


def build_graph_specs() -> List[GraphSpec]:
    return [
        GraphSpec("tiny", 100, 200, "linear"),
        GraphSpec("small", 1000, 2000, "linear"),
        GraphSpec("medium", 10000, 20000, "linear"),
        GraphSpec("medium_dense", 10000, 50000, "dense"),
        GraphSpec("large", 100000, 200000, "linear"),
        GraphSpec("large_dense", 100000, 500000, "dense"),
    ]


def write_markdown(results: Iterable[ResultRow], output_path: str) -> None:
    header = [
        "# Baseline Benchmark Results",
        "",
        "Notes:",
        "- Regular chain() ignores WHERE; Yannakakis path applies WHERE.",
        "- Scenario sizes reuse `baseline-2026-01-12.md` graph specs.",
        "- Values are median over runs; p90 and std columns show variability.",
        "",
        "| Graph | Scenario | Regular | Yannakakis | Ratio | Winner | Reg_p90 | Yann_p90 | Reg_std | Yann_std |",
        "|-------|----------|---------|------------|-------|--------|---------|----------|---------|----------|",
    ]
    lines = header + [summarize_row(row) for row in results]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark chain vs df_executor.")
    parser.add_argument("--engine", default="pandas", choices=["pandas", "cudf"])
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    engine_enum = Engine.CUDF if args.engine == "cudf" else Engine.PANDAS
    scenarios = build_scenarios()
    graph_specs = build_graph_specs()

    results: List[ResultRow] = []
    for spec in graph_specs:
        g = build_graph(spec, engine_enum)
        graph_name = spec.name
        for scenario in scenarios:
            regular_ms = run_regular(g, scenario.chain, args.engine, args.runs, args.warmup)
            yannakakis_ms = run_yannakakis(
                g,
                scenario.chain,
                scenario.where,
                engine_enum,
                args.runs,
                args.warmup,
            )
            results.append(
                ResultRow(
                    graph=f"{graph_name} ({spec.kind})",
                    scenario=scenario.name,
                    regular=regular_ms,
                    yannakakis=yannakakis_ms,
                )
            )

    if args.output:
        write_markdown(results, args.output)

    print("| Graph | Scenario | Regular | Yannakakis | Ratio | Winner | Reg_p90 | Yann_p90 | Reg_std | Yann_std |")
    print("|-------|----------|---------|------------|-------|--------|---------|----------|---------|----------|")
    for row in results:
        print(summarize_row(row))


if __name__ == "__main__":
    main()
