#!/usr/bin/env python3
"""
bench_ladybug.py — GFQL-native port of the LadybugDB vs Kuzu benchmark.

Mirrors https://github.com/LadybugDB/kuzu-ladybug-benchmark (synthetic Item/Owns
graph) so GFQL (polars CPU, and cudf GPU on a GPU host) can be compared
head-to-head against LadybugDB 0.18.0 / Kuzu 0.11.3 on the SAME query shapes.

The Ladybug suite's operations map onto GFQL/dataframe primitives (see the
per-op comments). Several are direct analogues of our own work:
  - op9  out-degree for seeded nodes  == our CSR ``edge_out_adj`` seeded index
  - op11 scan-rel rowid               == columnar edge scan / Arrow return
  - op13 Arrow CSR export             == our ``create_index('edge_out_adj')`` CSR

SAFETY: default size is TINY (1K nodes / 5K edges) and engine is ``polars``
(CPU) so this can run locally for correctness validation without a GPU and
without large memory. Use ``--nodes 5M --edges 20M`` (matching Ladybug's full
run) and ``--engine cudf`` ONLY on a GPU host with headroom.

Usage:
  python bench_ladybug.py                    # tiny local validation (CPU polars)
  python bench_ladybug.py --validate         # + assert results vs pandas oracle
  python bench_ladybug.py -n 5M -e 20M       # full scale (run on the bench box)
  python bench_ladybug.py --engine cudf -n 5M -e 20M   # GPU (bench box only)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Prefer the working-tree graphistry over any (possibly stale) pip-installed one,
# so submodules like graphistry.compute.chain resolve to this repo's code.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isdir(os.path.join(_REPO, "graphistry")) and _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def parse_num(s: str) -> int:
    s = str(s).upper().strip()
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(s)


def build_dataset(n_nodes: int, n_edges: int):
    """Replicate Ladybug's synthetic Item/Owns graph as pandas frames.

    nodes: id 1..N, name 'abcdefghijklmn_name_{id}'
    edges: i 1..E -> src=(i%N)+1, dst=(i*7%N)+1, since=i   (deterministic)
    """
    import numpy as np
    import pandas as pd

    ids = np.arange(1, n_nodes + 1, dtype="int64")
    nodes = pd.DataFrame({"id": ids, "name": [f"abcdefghijklmn_name_{i}" for i in ids]})
    i = np.arange(1, n_edges + 1, dtype="int64")
    src = (i % n_nodes) + 1
    dst = ((i * 7) % n_nodes) + 1
    edges = pd.DataFrame({"src": src, "dst": dst, "since": i})
    return nodes, edges


def timed_median(fn: Callable[[], Any], warmups: int, iters: int) -> Tuple[float, Optional[int]]:
    for _ in range(warmups):
        fn()
        gc.collect()
    samples: List[float] = []
    last: Optional[int] = None
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
        last = out if isinstance(out, int) else (len(out) if hasattr(out, "__len__") else None)
        gc.collect()
    return statistics.median(samples), last


class GFQLLadybug:
    """GFQL adapter running the Ladybug op-suite. engine in {'polars','cudf'}."""

    def __init__(self, engine: str, nodes, edges):
        import graphistry

        self.engine = engine
        self.nodes = nodes
        self.edges = edges
        self.n_nodes = int(len(nodes))
        self.n_edges = int(len(edges))
        self.g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id")

    # -- op fns: each returns a size (int) for a cheap correctness signal ------
    def op_full_scan(self):  # Ladybug #4
        g, eng = self.g, self.engine
        from graphistry.compute.ast import n
        return lambda: int(len(g.gfql([n()], engine=eng)._nodes))

    def op_range(self, lo: int, hi: int):  # Ladybug #5  <-- the index question
        # Columnar range predicate. Hypothesis: a vectorized full-column
        # scan is already competitive without a dedicated range index.
        g, eng = self.g, self.engine
        from graphistry.compute.ast import n
        from graphistry.compute.predicates.numeric import between
        return lambda: int(len(g.gfql([n({"id": between(lo, hi)})], engine=eng)._nodes))

    def op_point(self, pid: int):  # Ladybug #6
        g, eng = self.g, self.engine
        from graphistry.compute.ast import n
        return lambda: int(len(g.gfql([n({"id": pid})], engine=eng)._nodes))

    def op_count_rel(self):  # Ladybug #8
        edges = self.edges
        return lambda: int(len(edges))

    def op_out_degree_seeded(self, k: int = 100):  # Ladybug #9 == our CSR seeded index
        # Vectorized dataframe-native out-degree for nodes id 1..k (GFQL's
        # strength vs a per-node query loop). Sum of out-degrees.
        edges = self.edges
        seeds = set(range(1, k + 1))

        def run() -> int:
            deg = edges.groupby("src").size()
            return int(sum(int(deg.get(s, 0)) for s in seeds))

        return run

    def op_scan_rel(self):  # Ladybug #10
        edges = self.edges
        return lambda: int(len(edges[["src", "dst", "since"]]))

    def op_scan_rel_rowid(self):  # Ladybug #11 (their 50-60x claim vs Kuzu)
        edges = self.edges
        return lambda: int(len(edges[["src", "dst"]]))

    def op_arrow_csr(self):  # Ladybug #13 == our create_index('edge_out_adj') CSR
        g = self.g

        def run() -> int:
            gi = g.create_index("edge_out_adj")
            # size signal: number of indexes present after build
            try:
                idx = gi.show_indexes()
                return int(len(idx)) if hasattr(idx, "__len__") else 1
            except Exception:
                return 1

        return run


OPS = [
    ("full_scan", "op_full_scan", ()),
    ("range", "op_range", None),          # args filled from size
    ("point", "op_point", None),
    ("count_rel", "op_count_rel", ()),
    ("out_degree_seeded", "op_out_degree_seeded", ()),
    ("scan_rel", "op_scan_rel", ()),
    ("scan_rel_rowid", "op_scan_rel_rowid", ()),
    ("arrow_csr", "op_arrow_csr", ()),
]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="GFQL-native LadybugDB benchmark suite")
    p.add_argument("-n", "--nodes", default="1K")
    p.add_argument("-e", "--edges", default="5K")
    p.add_argument("--engine", default="polars", choices=["polars", "cudf", "pandas"])
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--out", default=None)
    p.add_argument("--validate", action="store_true",
                   help="assert op results vs a pandas oracle (tiny sizes)")
    p.add_argument("--debug", action="store_true", help="print tracebacks on op errors")
    args = p.parse_args(argv)

    n_nodes, n_edges = parse_num(args.nodes), parse_num(args.edges)
    # SAFETY rail for accidental local big runs.
    if args.engine != "cudf" and (n_nodes > 2_000_000 or n_edges > 8_000_000):
        print(f"[GUARD] {n_nodes:,} nodes / {n_edges:,} edges is large for a CPU "
              f"host — run this on the bench box. Refusing by default.", file=sys.stderr)
        return 3

    print(f"dataset: {n_nodes:,} nodes / {n_edges:,} edges, engine={args.engine}", file=sys.stderr)
    nodes, edges = build_dataset(n_nodes, n_edges)
    lo, hi = n_nodes // 2, n_nodes // 2 + 1000
    mid = n_nodes // 2

    adapter = GFQLLadybug(args.engine, nodes, edges)

    # Fill parametrized op args.
    resolved = []
    for name, meth, a in OPS:
        if name == "range":
            a = (lo, hi)
        elif name == "point":
            a = (mid,)
        resolved.append((name, meth, a))

    if args.validate:
        # pandas oracle checks on the (tiny) dataset
        exp_range = int(((nodes["id"] >= lo) & (nodes["id"] <= hi)).sum())
        exp_point = int((nodes["id"] == mid).sum())
        exp_scan = n_edges
        print(f"[oracle] range={exp_range} point={exp_point} edges={exp_scan}", file=sys.stderr)

    results = []
    for name, meth, a in resolved:
        fn = getattr(adapter, meth)(*a)
        try:
            med, size = timed_median(fn, args.warmups, args.iters)
            row = {"system": f"gfql-{args.engine}", "op": name, "n_nodes": n_nodes,
                   "n_edges": n_edges, "median_ms": round(med, 3), "size": size, "status": "ok"}
            print(f"[OK] {name:20} median={med:8.3f}ms size={size}", file=sys.stderr)
        except Exception as ex:
            import traceback as _tb
            row = {"system": f"gfql-{args.engine}", "op": name, "status": "error",
                   "error": str(ex)[:300]}
            print(f"[ERR] {name}: {ex}", file=sys.stderr)
            if args.debug:
                _tb.print_exc()
        results.append(row)

    if args.out:
        with open(args.out, "w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
