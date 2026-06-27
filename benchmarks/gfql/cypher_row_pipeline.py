#!/usr/bin/env python3
"""Benchmark the native polars GFQL row pipeline vs pandas for cypher queries.

Phase 2 of the polars engine enables cypher RETURN / LIMIT / SKIP / DISTINCT /
single-entity WHERE on ``engine='polars'`` (before this increment these raised
NotImplementedError on polars). The heavy traversal + frame ops (filter, dedup,
slice) run natively in polars; only the final row-wise entity-text projection is
host-bridged to pandas. So polars wins most where a row op reduces the set
before projection (LIMIT, selective WHERE, DISTINCT), and is closest to neutral
on a full-table whole-entity RETURN (projection dominates, bridge roundtrip).

Reports median latency and the polars speedup (pandas_ms / polars_ms; > 1 means
polars wins). On a shared host, interleave is implicit (pandas then polars
back-to-back per query); for regression-grade claims run several times and
compare distributions (see plans/gfql-polars-engine memory).

Example::

    python benchmarks/gfql/cypher_row_pipeline.py --runs 7 --warmup 2 \
        --sizes 10000,100000,1000000 --output /tmp/cypher-row.md
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

# (name, cypher) — exercised on both engines via g.gfql(cypher, engine=...)
# Native = frame ops (rows/limit/skip/distinct) run in polars; Bridged = the
# cypher expression engine (select/order_by/group_by) runs host-bridged to pandas.
WORKLOADS: List[Tuple[str, str]] = [
    # native frame-op path
    ("RETURN n LIMIT 10", "MATCH (n) RETURN n LIMIT 10"),
    ("RETURN n SKIP/LIMIT", "MATCH (n) RETURN n SKIP 5 LIMIT 100"),
    ("WHERE > RETURN LIMIT", "MATCH (n) WHERE n.score > 90 RETURN n LIMIT 50"),
    ("RETURN DISTINCT n", "MATCH (n) RETURN DISTINCT n"),
    ("WHERE > RETURN n", "MATCH (n) WHERE n.score > 50 RETURN n"),
    ("RETURN n (full)", "MATCH (n) RETURN n"),
    ("rel RETURN m LIMIT", "MATCH (n)-[e]->(m) RETURN m LIMIT 100"),
    # host-bridged expression path
    ("select n.score", "MATCH (n) RETURN n.score"),
    ("select 2 cols", "MATCH (n) RETURN n.score, n.kind"),
    ("order_by", "MATCH (n) RETURN n.score ORDER BY n.score DESC"),
    ("where+select+limit", "MATCH (n) WHERE n.score > 50 RETURN n.score ORDER BY n.score LIMIT 100"),
    ("group_by count", "MATCH (n) RETURN n.kind, count(n) AS c"),
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
    """Same graph with node/edge frames already in polars, so the polars runs
    don't pay a per-call pandas->polars input coercion that the pandas runs avoid
    (a real deployment keeps the graph in its engine's native frame type)."""
    from graphistry.Engine import Engine, df_to_engine
    return g.nodes(df_to_engine(g._nodes, Engine.POLARS), g._node).edges(
        df_to_engine(g._edges, Engine.POLARS), g._source, g._destination)


def run(sizes: List[Tuple[int, int]], runs: int, warmup: int) -> List[ResultRow]:
    rows: List[ResultRow] = []
    for n_nodes, n_edges in sizes:
        g_pd = make_graph(n_nodes, n_edges)
        g_pl = _polars_graph(g_pd)
        for name, query in WORKLOADS:
            try:
                pandas_ms = timeit(lambda: g_pd.gfql(query, engine="pandas"), runs, warmup)
                polars_ms = timeit(lambda: g_pl.gfql(query, engine="polars"), runs, warmup)
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

    parser = argparse.ArgumentParser(description="Benchmark GFQL cypher row pipeline pandas vs polars.")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--sizes",
        default="10000,100000,1000000",
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
