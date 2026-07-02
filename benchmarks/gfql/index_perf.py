#!/usr/bin/env python3
"""GFQL seeded-traversal perf: scan baseline vs warm index, scaling in edges.

The thesis (from HANDOFF-SELECTIVE-TRAVERSAL): GFQL seeded hop is O(E) (scan),
so latency grows linearly with edges; an adjacency index makes it O(degree),
flat in graph size. This bench measures, per engine × size:
  - scan      : seeded hop, no index (the O(E) baseline)
  - build     : one-time index build cost (cold, pay-as-you-go)
  - warm      : seeded hop with resident index (the win)
for SEL1 (1-hop from 1 seed) and SEL2 (2-hop from 1 seed).

Env: ENGINES=pandas,cudf,polars,polars-gpu  NS=100000,1000000  DEG=8
     REPS=10  SEEDS=1  OUT=/path/results.jsonl
Run (dgx-spark container, GPU idle-checked).
"""
from __future__ import annotations

import json
import os
import statistics
import time
import numpy as np
import pandas as pd

import graphistry
from graphistry.Engine import Engine


def make_graph(n_nodes, deg, seed=0):
    rng = np.random.default_rng(seed)
    m = n_nodes * deg
    src = rng.integers(0, n_nodes, size=m, dtype=np.int64)
    dst = rng.integers(0, n_nodes, size=m, dtype=np.int64)
    edf = pd.DataFrame({"src": src, "dst": dst})
    ndf = pd.DataFrame({"id": np.arange(n_nodes, dtype=np.int64)})
    return ndf, edf


def _sync(engine):
    # force device sync so GPU timings are honest
    if engine in ("cudf", "polars-gpu"):
        try:
            import cupy as cp  # type: ignore
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass


def timeit(fn, engine, reps, warmup=1):
    for _ in range(warmup):
        fn(); _sync(engine)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(); _sync(engine)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return statistics.median(ts), ts[0], ts[-1]


def n_rows(g):
    e = g._edges
    return int(e.shape[0]) if e is not None else 0


def main():
    engines = os.environ.get("ENGINES", "pandas,cudf,polars,polars-gpu").split(",")
    NS = [int(x) for x in os.environ.get("NS", "100000,1000000").split(",")]
    DEG = int(os.environ.get("DEG", "8"))
    REPS = int(os.environ.get("REPS", "10"))
    NSEEDS = int(os.environ.get("SEEDS", "1"))
    out = os.environ.get("OUT")
    outf = open(out, "a") if out else None

    seeds = pd.DataFrame({"id": list(range(NSEEDS))})
    print(f"{'engine':11} {'N':>9} {'edges':>10} {'task':5} "
          f"{'scan_ms':>10} {'build_ms':>10} {'warm_ms':>10} {'speedup':>9} {'rows':>7}")
    for N in NS:
        ndf, edf = make_graph(N, DEG)
        g0 = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
        for engine in engines:
            try:
                # coerce once; build index once (cold timing)
                t0 = time.perf_counter()
                gi = g0.gfql_index_all(engine=engine)
                _sync(engine)
                build_ms = (time.perf_counter() - t0) * 1e3
            except Exception as ex:
                print(f"{engine:11} {N:>9} build FAILED {type(ex).__name__}: {ex}")
                continue
            E = n_rows(gi)
            for task, kw in (("SEL1", dict(hops=1)), ("SEL2", dict(hops=2))):
                kw = dict(kw, direction="forward")
                try:
                    scan = timeit(lambda: g0.hop(nodes=seeds, engine=engine, **kw), engine, REPS)
                    warm = timeit(lambda: gi.hop(nodes=seeds, engine=engine, **kw), engine, REPS)
                    rows = n_rows(gi.hop(nodes=seeds, engine=engine, **kw))
                except Exception as ex:
                    print(f"{engine:11} {N:>9} {task} FAILED {type(ex).__name__}: {ex}")
                    continue
                sp = scan[0] / warm[0] if warm[0] > 0 else float("inf")
                print(f"{engine:11} {N:>9} {E:>10} {task:5} "
                      f"{scan[0]:>10.3f} {build_ms:>10.1f} {warm[0]:>10.4f} {sp:>8.1f}x {rows:>7}")
                if outf:
                    outf.write(json.dumps(dict(
                        engine=engine, n=N, edges=E, task=task, deg=DEG, seeds=NSEEDS,
                        scan_ms=scan[0], build_ms=build_ms, warm_ms=warm[0],
                        scan_min=scan[1], warm_min=warm[1], speedup=sp, rows=rows, reps=REPS,
                    )) + "\n")
                    outf.flush()
    if outf:
        outf.close()


if __name__ == "__main__":
    main()
