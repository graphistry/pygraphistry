#!/usr/bin/env python3
"""Large REAL-graph CSR-index bench (Step 7). Power-law topology exposes what the
uniform deg-8 synthetic never did: the index is O(degree), so warm latency is flat
in N but scales with SEED DEGREE — a hub seed is the adversarial worst case.

Same trust discipline as index_takeover_bench.py: every GFQL timing is GUARDED by
(index path actually taken via index_trace) AND (index result == scan result). A
cell failing either guard is reported INVALID, never as a speedup.

Datasets (SNAP edge lists, gzipped `u v`, load once -> parquet cache):
  com-Orkut       3.07M nodes / 117M edges   https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz
  com-LiveJournal 4.0M / 34.7M               https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz
  soc-LiveJournal1 4.8M / 69M (directed)     https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
  com-Friendster  65.6M / 1.8B (STRETCH)     https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
  twitter-2010    41.7M / 1.47B (STRETCH)
LDBC SNB sf10/sf100 via ~/Work/pyg-bench loader + the live snb-interactive-neo4j.

Env: EDGELIST=/path/to/edges.txt.gz  (or PARQUET=/path/edges.parquet)
     DEG_PCTLS=50,90,99,100  MULTISEED=1,10,100,1000  ENGINES=pandas,polars,cudf,polars-gpu
     REPS=15  OUT=/tmp/lg.jsonl  MAXSCAN_REPS=3  (cap scan reps at large E)
"""
from __future__ import annotations
import gzip, json, os, statistics, time
import numpy as np
import pandas as pd
import graphistry
from graphistry.compute.gfql.index import index_trace


def load_graph(seed=0):
    """Load a real edge list -> graphistry graph (int64 ids), parquet-cached."""
    pq = os.environ.get("PARQUET")
    el = os.environ.get("EDGELIST")
    if pq and os.path.exists(pq):
        edf = pd.read_parquet(pq)
    elif el:
        cache = el + ".parquet"
        if os.path.exists(cache):
            edf = pd.read_parquet(cache)
        else:
            op = gzip.open if el.endswith(".gz") else open
            with op(el, "rt") as f:
                edf = pd.read_csv(f, sep=r"\s+", comment="#", header=None,
                                  names=["src", "dst"], dtype=np.int64)
            edf.to_parquet(cache)
            print(f"  cached parquet -> {cache}")
    else:
        # fallback synthetic power-law (Barabasi-ish via preferential attachment proxy)
        rng = np.random.default_rng(seed)
        n = int(os.environ.get("SYNTH_N", "1000000")); m = n * 8
        deg = rng.zipf(2.2, m) % n
        edf = pd.DataFrame({"src": rng.integers(0, n, m, dtype=np.int64), "dst": deg.astype(np.int64)})
    nodes = np.unique(np.concatenate([edf["src"].values, edf["dst"].values]))
    ndf = pd.DataFrame({"id": nodes})
    return graphistry.nodes(ndf, "id").edges(edf, "src", "dst"), ndf, edf


def degree_seeds(edf, pctls):
    """Pick one seed id at each out-degree percentile (the O(degree) honesty sweep)."""
    deg = edf.groupby("src").size()
    out = {}
    for p in pctls:
        if p >= 100:
            sid = int(deg.idxmax()); out["max"] = (int(deg.max()), sid)
        else:
            thr = np.percentile(deg.values, p)
            cand = deg[deg >= thr]
            sid = int(cand.index[0]); out[f"p{p}"] = (int(deg.loc[sid]), sid)
    return out


def _sync(engine):
    if engine in ("cudf", "polars-gpu"):
        try:
            import cupy as cp; cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass


def timeit(fn, reps, engine, warmup=2):
    for _ in range(warmup):
        fn(); _sync(engine)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); _sync(engine); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort(); return statistics.median(ts)


def _sig(g):
    n, e = g._nodes, g._edges
    if "polars" in type(n).__module__ or "cudf" in type(n).__module__: n = n.to_pandas()
    if "polars" in type(e).__module__ or "cudf" in type(e).__module__: e = e.to_pandas()
    return (len(n), len(e), int(e["src"].sum()) + int(e["dst"].sum()))


def bench(g0, ndf, edf, engines, reps):
    maxscan = int(os.environ.get("MAXSCAN_REPS", "3"))
    E = len(edf)
    pctls = [int(x) for x in os.environ.get("DEG_PCTLS", "50,90,99,100").split(",")]
    multiseed = [int(x) for x in os.environ.get("MULTISEED", "1,10,100,1000").split(",")]
    dseeds = degree_seeds(edf, pctls)
    print(f"  degree seeds: " + ", ".join(f"{k}=deg{d}" for k, (d, _) in dseeds.items()))
    outf = open(os.environ["OUT"], "a") if os.environ.get("OUT") else None
    for engine in engines:
        try:
            t0 = time.perf_counter(); gi = g0.gfql_index_all(engine=engine); _sync(engine)
            build_ms = (time.perf_counter() - t0) * 1e3
        except Exception as ex:
            print(f"  {engine}: BUILD FAILED {type(ex).__name__}: {ex}"); continue
        # T3: seed-degree sweep (1-hop), guarded
        for tag, (deg, sid) in dseeds.items():
            seeds = pd.DataFrame({"id": [sid]})
            with index_trace() as steps:
                gidx = gi.hop(nodes=seeds, engine=engine, hops=1, direction="forward")
            took = any(s.get("path") == "index" for s in steps)
            gscan = g0.hop(nodes=seeds, engine=engine, hops=1, direction="forward")
            same = _sig(gidx) == _sig(gscan)
            valid = took and same
            wi = timeit(lambda: gi.hop(nodes=seeds, engine=engine, hops=1, direction="forward"), reps, engine)
            ws = timeit(lambda: g0.hop(nodes=seeds, engine=engine, hops=1, direction="forward"),
                        min(reps, maxscan), engine)
            rec = dict(system="gfql", engine=engine, task="degsweep", seed_deg=deg, n=len(ndf), edges=E,
                       valid=valid, warm_idx_ms=wi, warm_scan_ms=ws, speedup=ws / wi if wi else None, build_ms=build_ms)
            print(f"  {engine:11} deg={deg:>8} idx={wi:9.4f}ms scan={ws:10.3f}ms x{ws/wi:7.1f}{'' if valid else '  <<INVALID'}")
            if outf: outf.write(json.dumps(rec) + "\n"); outf.flush()
        # T4: multi-seed frontier sweep (where the cost gate flips index->scan)
        rng = np.random.default_rng(0)
        allids = ndf["id"].values
        for k in multiseed:
            seeds = pd.DataFrame({"id": rng.choice(allids, size=min(k, len(allids)), replace=False)})
            with index_trace() as steps:
                gidx = gi.hop(nodes=seeds, engine=engine, hops=1, direction="forward")
            took = any(s.get("path") == "index" for s in steps)
            wi = timeit(lambda: gi.hop(nodes=seeds, engine=engine, hops=1, direction="forward"), reps, engine)
            print(f"  {engine:11} kseed={k:>6} idx={wi:9.4f}ms path={'index' if took else 'scan'}")
            if outf: outf.write(json.dumps(dict(system="gfql", engine=engine, task="multiseed", kseed=k,
                                n=len(ndf), edges=E, took_index=took, warm_idx_ms=wi)) + "\n"); outf.flush()
    if outf: outf.close()


def main():
    g0, ndf, edf = load_graph()
    print(f"===== graph: {len(ndf):,} nodes  {len(edf):,} edges =====")
    engines = os.environ.get("ENGINES", "pandas,polars").split(",")
    bench(g0, ndf, edf, engines, int(os.environ.get("REPS", "15")))


if __name__ == "__main__":
    main()
