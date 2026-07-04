#!/usr/bin/env python3
"""Benchmark the native polars GFQL engine vs pandas for hop() and chain().

Compares ``engine='pandas'`` against ``engine='polars'`` on synthetic random
graphs across a size sweep, for representative hop and single-hop chain
workloads. Reports a markdown table of median latency and the polars speedup
(pandas_ms / polars_ms; > 1 means polars wins).

Polars wins at scale (joins amortize its fixed per-call overhead); the crossover
is typically ~50-100k rows. On a shared host, interleave is implicit (each
workload times pandas then polars back-to-back per size).

Example::

    python benchmarks/gfql/pandas_vs_polars.py --runs 7 --warmup 2 \
        --sizes 10000,100000,500000 --output /tmp/pandas-vs-polars.md
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

import graphistry
from graphistry.compute.ast import n, e_forward

# (name, builder) — builder takes (graphistry_graph, engine_str) -> Plottable
WORKLOADS: List[Tuple[str, Callable]] = [
    ("hop1", lambda g, eng: g.hop(hops=1, engine=eng)),
    ("hop2", lambda g, eng: g.hop(hops=2, engine=eng)),
    ("chain n-e-n", lambda g, eng: g.chain([n(), e_forward(), n()], engine=eng)),
    ("chain filter", lambda g, eng: g.chain([n({"kind": "x"}), e_forward({"rel": "r1"}), n()], engine=eng)),
    ("chain 2-edge", lambda g, eng: g.chain([n({"kind": "x"}), e_forward(), n(), e_forward(), n()], engine=eng)),
]


@dataclass
class ResultRow:
    workload: str
    n_nodes: int
    n_edges: int
    pandas_ms: Optional[float]
    polars_ms: Optional[float]
    error: Optional[str] = None

    @property
    def speedup(self) -> Optional[float]:
        if self.pandas_ms and self.polars_ms:
            return self.pandas_ms / self.polars_ms
        return None


def make_graph(n_nodes: int, n_edges: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    nodes = pd.DataFrame({
        "id": np.arange(n_nodes),
        "kind": rng.choice(["x", "y", "z"], size=n_nodes),
        "score": rng.integers(0, 100, size=n_nodes),
    })
    edges = pd.DataFrame({
        "s": rng.integers(0, n_nodes, size=n_edges),
        "d": rng.integers(0, n_nodes, size=n_edges),
        "rel": rng.choice(["r1", "r2", "r3"], size=n_edges),
    })
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def timeit(fn: Callable[[], object], runs: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _polars_graph(g):
    """Graph with frames already in polars so polars runs don't pay a per-call
    pandas->polars input coercion the pandas runs avoid (real deployments keep
    the graph in the engine's native frame type)."""
    from graphistry.Engine import Engine, df_to_engine
    return g.nodes(df_to_engine(g._nodes, Engine.POLARS), g._node).edges(
        df_to_engine(g._edges, Engine.POLARS), g._source, g._destination)


def run(sizes: List[Tuple[int, int]], runs: int, warmup: int) -> List[ResultRow]:
    rows: List[ResultRow] = []
    for n_nodes, n_edges in sizes:
        g_pd = make_graph(n_nodes, n_edges)
        g_pl = _polars_graph(g_pd)
        for name, fn in WORKLOADS:
            try:
                pandas_ms = timeit(lambda: fn(g_pd, "pandas"), runs, warmup)
                polars_ms = timeit(lambda: fn(g_pl, "polars"), runs, warmup)
                rows.append(ResultRow(name, n_nodes, n_edges, pandas_ms, polars_ms))
            except Exception as exc:  # noqa: BLE001 - bench harness reports, never crashes the sweep
                rows.append(ResultRow(name, n_nodes, n_edges, None, None, error=f"{type(exc).__name__}: {exc}"))
    return rows


def to_markdown(rows: List[ResultRow]) -> str:
    lines = [
        "| workload | nodes | edges | pandas_ms | polars_ms | speedup |",
        "|----------|-------|-------|-----------|-----------|---------|",
    ]
    for r in rows:
        if r.error:
            lines.append(f"| {r.workload} | {r.n_nodes} | {r.n_edges} | ERROR | ERROR | {r.error} |")
        else:
            lines.append(
                f"| {r.workload} | {r.n_nodes} | {r.n_edges} | "
                f"{r.pandas_ms:.1f} | {r.polars_ms:.1f} | {r.speedup:.2f}x |"
            )
    return "\n".join(lines)


def _parse_sizes(text: str) -> List[Tuple[int, int]]:
    # "nodes:edges,nodes:edges" or "nodes" (edges defaults to 5x nodes)
    out: List[Tuple[int, int]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            nn, ne = chunk.split(":")
            out.append((int(nn), int(ne)))
        else:
            nn = int(chunk)
            out.append((nn, nn * 5))
    return out


def main() -> None:
    try:
        import polars  # noqa: F401
    except ImportError:
        raise SystemExit("polars is not installed; install with `pip install polars`")

    parser = argparse.ArgumentParser(description="Benchmark GFQL pandas vs native polars engine.")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--sizes",
        default="1000,10000,100000",
        help="Comma list of node counts (edges=5x) or nodes:edges pairs.",
    )
    parser.add_argument("--output", default="", help="Optional path to write the markdown table.")
    args = parser.parse_args()

    rows = run(_parse_sizes(args.sizes), args.runs, args.warmup)
    table = to_markdown(rows)
    print(table)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(table + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
