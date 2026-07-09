#!/usr/bin/env python3
"""Small-N pandas-vs-polars CROSSOVER bench (CPU). Answers "where does polars start
beating pandas?" per workload SHAPE, on a real graph subsampled to N edges.

The crossover is shape-dependent: row-pipeline shapes (filter / WHERE+ORDER) cross over
much earlier than traversal (chain orchestration is the residual small-N fixed cost).
CPU only (the crossover question is pandas-CPU vs polars-CPU); no GPU needed.

Env: PARQUET=/data/edges.parquet  EDGES=10000,100000,1000000  REPS=15  WARM=3  OUT=/tmp/x.jsonl
"""
from __future__ import annotations
import json, os, statistics, time
import numpy as np
import pandas as pd
import graphistry
from graphistry.compute.ast import n, e_forward


def med(fn, reps, warm):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return statistics.median(ts)


def main():
    edf_full = pd.read_parquet(os.environ["PARQUET"]).astype({"src": np.int64, "dst": np.int64})
    sizes = [int(x) for x in os.environ.get("EDGES", "10000,100000,1000000").split(",")]
    reps = int(os.environ.get("REPS", "15")); warm = int(os.environ.get("WARM", "3"))
    outf = open(os.environ["OUT"], "a") if os.environ.get("OUT") else None
    print(f"{'shape':10} {'edges':>9} {'pandas_ms':>10} {'polars_ms':>10} {'polars_speedup':>15}")
    for E in sizes:
        edf = edf_full.head(E).reset_index(drop=True)
        nodes = np.unique(np.concatenate([edf["src"].values, edf["dst"].values]))
        ndf = pd.DataFrame({"id": nodes, "val": (nodes % 100).astype(np.int64)})
        g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
        seeds = nodes[: max(1, len(nodes) // 100)].tolist()  # ~1% frontier
        shapes = {
            "filter": lambda eng: g.gfql([n({"val": 50})], engine=eng),
            "hop1": lambda eng: g.gfql([n({"id": seeds}), e_forward()], engine=eng),
            "where_ord": lambda eng: g.gfql(
                "MATCH (a) WHERE a.val > 50 RETURN a.id ORDER BY a.id LIMIT 100", engine=eng),
        }
        for name, fn in shapes.items():
            try:
                rp = fn("pandas"); rl = fn("polars")  # warm + sanity
                pm = med(lambda: fn("pandas"), reps, warm)
                lm = med(lambda: fn("polars"), reps, warm)
                sp = pm / lm if lm else float("nan")
                print(f"{name:10} {E:>9} {pm:>10.3f} {lm:>10.3f} {('polars '+format(sp,'.2f')+'x') if sp>=1 else ('PANDAS '+format(1/sp,'.2f')+'x'):>15}")
                if outf:
                    outf.write(json.dumps(dict(shape=name, edges=E, pandas_ms=pm, polars_ms=lm,
                                               polars_speedup=sp)) + "\n"); outf.flush()
            except Exception as ex:
                print(f"{name:10} {E:>9}  FAILED {type(ex).__name__}: {ex}")
    if outf:
        outf.close()


if __name__ == "__main__":
    main()
