#!/usr/bin/env python3
"""Adversarial CSR-index bench (takeover). Trust nothing: every timed cell is
GUARDED by (a) the index path was actually taken (index_trace) and (b) the index
result == the scan result. A cell that fails either guard is reported as INVALID,
never as a speedup.

GFQL: index-vs-scan seeded latency, flat-in-N check, 4 engines.
Optional vs-DB: kuzu (embedded, CSR) if installed.

Env: NS=800000,8000000,80000000  DEG=8  REPS=15  ENGINES=pandas,polars,cudf,polars-gpu
     SYSTEMS=gfql,kuzu  OUT=/tmp/idx-bench.jsonl
"""
from __future__ import annotations
import json, os, statistics, time
import numpy as np
import pandas as pd
import graphistry
from graphistry.compute.gfql.index import index_trace


def make_graph(n, deg, seed=0):
    rng = np.random.default_rng(seed)
    m = n * deg
    return (pd.DataFrame({"id": np.arange(n, dtype=np.int64)}),
            pd.DataFrame({"src": rng.integers(0, n, m, dtype=np.int64),
                          "dst": rng.integers(0, n, m, dtype=np.int64)}))


def _sync(engine):
    if engine in ("cudf", "polars-gpu"):
        try:
            import cupy as cp
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass


def timeit(fn, reps, engine, warmup=2):
    for _ in range(warmup):
        fn(); _sync(engine)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); _sync(engine)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return statistics.median(ts)


def _nodeset_edgeset(g):
    n, e = g._nodes, g._edges
    nm, em = type(n).__module__, type(e).__module__
    if "cudf" in nm or "polars" in nm: n = n.to_pandas()
    if "cudf" in em or "polars" in em: e = e.to_pandas()
    return (len(n), len(e),
            int(n["id"].sum()), int(e["src"].sum()) + int(e["dst"].sum()))


def bench_gfql(ndf, edf, N, E, engines, reps, outf):
    g0 = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
    seeds = pd.DataFrame({"id": [0]})  # single seed -> tiny frontier, beats cost gate
    for engine in engines:
        try:
            t0 = time.perf_counter()
            gi = g0.gfql_index_all(engine=engine); _sync(engine)
            build_ms = (time.perf_counter() - t0) * 1e3
        except Exception as ex:
            print(f"  gfql {engine:11} BUILD FAILED: {type(ex).__name__}: {ex}"); continue
        for task, kw in (("SEL1", dict(hops=1, direction="forward")),
                         ("SEL2", dict(hops=2, direction="forward"))):
            # --- correctness + path GUARD (adversarial) ---
            with index_trace() as steps:
                idx_g = gi.hop(nodes=seeds, engine=engine, **kw)
            took_index = any(s.get("path") == "index" for s in steps)
            scan_g = g0.hop(nodes=seeds, engine=engine, **kw)  # no index resident -> scan
            same = _nodeset_edgeset(idx_g) == _nodeset_edgeset(scan_g)
            valid = took_index and same
            if not valid:
                print(f"  gfql {engine:11} {task} INVALID: index_path={took_index} result_match={same} "
                      f"(idx={_nodeset_edgeset(idx_g)} scan={_nodeset_edgeset(scan_g)})")
            warm_idx = timeit(lambda: gi.hop(nodes=seeds, engine=engine, **kw), reps, engine)
            warm_scan = timeit(lambda: g0.hop(nodes=seeds, engine=engine, **kw), reps, engine)
            rows = _nodeset_edgeset(idx_g)[1]
            speedup = warm_scan / warm_idx if warm_idx else float("nan")
            rec = dict(system="gfql", engine=engine, task=task, n=N, edges=E, valid=valid,
                       took_index=took_index, result_match=same, warm_idx_ms=warm_idx,
                       warm_scan_ms=warm_scan, speedup=speedup, build_ms=build_ms, rows=rows)
            tag = "" if valid else "  <<< INVALID"
            print(f"  gfql {engine:11} {task} idx={warm_idx:8.4f}ms scan={warm_scan:10.3f}ms "
                  f"speedup={speedup:6.1f}x rows={rows} build={build_ms:.0f}ms{tag}")
            if outf: outf.write(json.dumps(rec) + "\n"); outf.flush()


def bench_kuzu(ndf, edf, N, E, reps, outf):
    try:
        import kuzu
    except Exception:
        print("  kuzu: NOT INSTALLED (pip install kuzu)"); return
    import shutil, tempfile
    dbp = os.path.join(tempfile.gettempdir(), f"kuzu_bench_{N}")
    shutil.rmtree(dbp, ignore_errors=True)
    db = kuzu.Database(dbp); conn = kuzu.Connection(db)
    conn.execute("CREATE NODE TABLE Node(id INT64, PRIMARY KEY(id))")
    conn.execute("CREATE REL TABLE E(FROM Node TO Node)")
    npath = os.path.join(tempfile.gettempdir(), f"n_{N}.parquet")
    epath = os.path.join(tempfile.gettempdir(), f"e_{N}.parquet")
    ndf.to_parquet(npath); edf.rename(columns={"src": "from", "dst": "to"}).to_parquet(epath)
    t0 = time.perf_counter()
    conn.execute(f'COPY Node FROM "{npath}"')
    conn.execute(f'COPY E FROM "{epath}"')
    build_ms = (time.perf_counter() - t0) * 1e3
    for task, q in (("SEL1", "MATCH (a:Node)-[:E]->(b:Node) WHERE a.id=0 RETURN b.id"),
                    ("SEL2", "MATCH (a:Node)-[:E*1..2]->(b:Node) WHERE a.id=0 RETURN DISTINCT b.id")):
        def run(): return conn.execute(q)
        rows = 0
        res = run()
        while res.has_next(): res.get_next(); rows += 1
        warm = timeit(run, reps, "cpu")
        print(f"  kuzu {'':11} {task} warm={warm:8.4f}ms rows={rows} build={build_ms:.0f}ms")
        if outf: outf.write(json.dumps(dict(system="kuzu", engine="kuzu", task=task, n=N,
                            edges=E, warm_idx_ms=warm, build_ms=build_ms, rows=rows)) + "\n"); outf.flush()


def main():
    NS = [int(x) for x in os.environ.get("NS", "800000,8000000").split(",")]
    DEG = int(os.environ.get("DEG", "8"))
    REPS = int(os.environ.get("REPS", "15"))
    ENGINES = os.environ.get("ENGINES", "pandas,polars").split(",")
    SYSTEMS = os.environ.get("SYSTEMS", "gfql,kuzu").split(",")
    OUT = os.environ.get("OUT", "")
    outf = open(OUT, "w") if OUT else None
    for N in NS:
        E = N * DEG
        print(f"\n===== N={N:,} nodes  E={E:,} edges  deg={DEG} =====")
        ndf, edf = make_graph(N, DEG)
        if "gfql" in SYSTEMS:
            bench_gfql(ndf, edf, N, E, ENGINES, REPS, outf)
        if "kuzu" in SYSTEMS:
            bench_kuzu(ndf, edf, N, E, REPS, outf)
    if outf: outf.close()


if __name__ == "__main__":
    main()
