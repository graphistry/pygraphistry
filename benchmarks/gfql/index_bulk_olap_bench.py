#!/usr/bin/env python3
"""BULK-OLAP head-to-head: GFQL 4 engines vs kuzu on REAL graphs.

Answers "is bulk OLAP better with GFQL (cudf / polars-gpu)?" The seeded CSR index
is O(degree) and wins tiny work; this bench deliberately AVOIDS that path and
measures the BULK regime instead — large-frontier multi-hop + full-graph
aggregation, i.e. the scan/join work where columnar GPU throughput should pay off
and the index does NOT help. We run g0.hop (NO resident index -> engine traversal,
the honest bulk path) so every engine does the same materialized join work.

Tasks (all bulk, all materialized on both sides):
  BULK1   1-hop forward from K seeds         (edge semijoin, frontier=K)
  BULK2   2-hop forward from K seeds         (edge-edge join, frontier blows up)
  DEGALL  full-graph out-degree aggregation  (group_by over ALL edges; pure OLAP)
K frontier sweep: 1k, 10k, 100k seeds. cudf/polars-gpu should overtake pandas as K
(hence work) grows; kuzu is the WCOJ/optimizer peer for the multi-hop join.

Trust: GFQL rows reported per engine (engine parity is separately guaranteed by the
conformance suite); kuzu rows reported alongside with a semantic note. Timing is the
deliverable — rows are the honesty check that each system did real work.

Env: PARQUET=/path/edges.parquet  KS=1000,10000,100000  ENGINES=pandas,polars,cudf,polars-gpu
     SYSTEMS=gfql,kuzu  REPS=10  WARM=2  OUT=/tmp/bulk.jsonl  SEED=0
"""
from __future__ import annotations
import json, os, statistics, time, tempfile, shutil
import numpy as np
import pandas as pd
import graphistry
from graphistry.compute.ast import n, e_forward


def _sync(engine):
    if engine in ("cudf", "polars-gpu"):
        try:
            import cupy as cp  # type: ignore
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass


def timeit(fn, reps, engine="cpu", warmup=2):
    for _ in range(warmup):
        fn(); _sync(engine)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); _sync(engine)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return statistics.median(ts)


def load_graph():
    edf = pd.read_parquet(os.environ["PARQUET"]).astype({"src": np.int64, "dst": np.int64})
    nodes = np.unique(np.concatenate([edf["src"].values, edf["dst"].values]))
    ndf = pd.DataFrame({"id": nodes})
    return ndf, edf, nodes


def gfql_trav(g0, seed_ids, hops, engine):
    """BULK seeded multi-hop via the CHAIN API — the one GFQL surface that supports
    ALL FOUR engines (generic hop() is pandas/cudf only; polars/polars-gpu route
    through engine_polars). n({id:seeds}) = frontier filter, then e_forward()*hops."""
    ops = [n({"id": seed_ids})] + [e_forward() for _ in range(hops)]
    return g0.chain(ops, engine=engine)


def run_gfql(ndf, edf, nodes, ks, engines, reps, warm, outf, seed):
    N, E = len(ndf), len(edf)
    rng = np.random.default_rng(seed)
    seed_sets = {k: rng.choice(nodes, size=min(k, len(nodes)), replace=False).tolist() for k in ks}
    for engine in engines:
        try:
            g0 = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
            # warm/convert frames onto the engine ONCE (exclude H2D/convert from timing)
            _ = gfql_trav(g0, seed_sets[ks[0]], 1, engine)
        except Exception as ex:
            print(f"  gfql {engine}: SETUP FAILED {type(ex).__name__}: {ex}"); continue
        # frontier sweep: BULK1 (1-hop) + BULK2 (2-hop)
        for k in ks:
            sids = seed_sets[k]
            for task, hops in (("BULK1", 1), ("BULK2", 2)):
                try:
                    res = gfql_trav(g0, sids, hops, engine)
                    rows = int(res._edges.shape[0]); nn = int(res._nodes.shape[0])
                    ms = timeit(lambda: gfql_trav(g0, sids, hops, engine), reps, engine, warm)
                except Exception as ex:
                    print(f"  gfql {engine} {task} k={k} FAILED: {type(ex).__name__}: {ex}"); continue
                rec = dict(system="gfql", engine=engine, task=task, k=k, hops=hops,
                           n=N, edges=E, warm_ms=ms, e_rows=rows, n_rows=nn)
                print(f"  gfql {engine:11} {task} k={k:>7} {ms:10.3f}ms  e_rows={rows:>10}  n_rows={nn:>9}")
                if outf: outf.write(json.dumps(rec) + "\n"); outf.flush()
        # DEGALL: full-graph out-degree aggregation (pure columnar OLAP, no traversal)
        try:
            ms, rows = degall(edf, engine, reps, warm)
            rec = dict(system="gfql", engine=engine, task="DEGALL", k=None, hops=0,
                       n=N, edges=E, warm_ms=ms, e_rows=rows, n_rows=rows)
            print(f"  gfql {engine:11} DEGALL{'':>13} {ms:10.3f}ms  groups={rows:>10}")
            if outf: outf.write(json.dumps(rec) + "\n"); outf.flush()
        except Exception as ex:
            print(f"  gfql {engine} DEGALL FAILED: {type(ex).__name__}: {ex}")


def degall(edf, engine, reps, warm):
    """Full-graph out-degree = group_by(src).size() on the chosen engine."""
    if engine == "pandas":
        df = edf
        fn = lambda: df.groupby("src").size()
    elif engine == "cudf":
        import cudf
        df = cudf.from_pandas(edf)
        fn = lambda: df.groupby("src").size()
    elif engine in ("polars", "polars-gpu"):
        import polars as pl
        df = pl.from_pandas(edf)
        if engine == "polars-gpu":
            eng = pl.GPUEngine(executor="in-memory", raise_on_fail=False)
            fn = lambda: df.lazy().group_by("src").len().collect(engine=eng)
        else:
            fn = lambda: df.group_by("src").len()
    else:
        raise ValueError(engine)
    r = fn(); rows = int(r.shape[0])
    ms = timeit(fn, reps, engine, warm)
    return ms, rows


def run_kuzu(ndf, edf, nodes, ks, reps, warm, outf, seed, tmpdir):
    try:
        import kuzu
    except Exception:
        print("  kuzu: NOT AVAILABLE (pip install kuzu)"); return
    rng = np.random.default_rng(seed)
    seed_sets = {k: rng.choice(nodes, size=min(k, len(nodes)), replace=False).tolist() for k in ks}
    dbp = tempfile.mkdtemp(dir=tmpdir)
    db = kuzu.Database(os.path.join(dbp, "kz")); conn = kuzu.Connection(db)
    conn.execute("CREATE NODE TABLE N(id INT64, PRIMARY KEY(id))")
    conn.execute("CREATE REL TABLE E(FROM N TO N)")
    np_path = os.path.join(dbp, "n.parquet"); ep_path = os.path.join(dbp, "e.parquet")
    ndf.to_parquet(np_path)
    edf.rename(columns={"src": "from", "dst": "to"}).to_parquet(ep_path)
    t0 = time.perf_counter()
    conn.execute(f'COPY N FROM "{np_path}"'); conn.execute(f'COPY E FROM "{ep_path}"')
    load_ms = (time.perf_counter() - t0) * 1e3
    print(f"  kuzu load: {load_ms:.0f}ms")
    # BULK1/BULK2: distinct reachable set from K seeds (materialized columnar via get_as_df)
    q1 = conn.prepare("MATCH (a:N)-[:E]->(b:N) WHERE a.id IN $seeds RETURN b.id")
    q2 = conn.prepare("MATCH (a:N)-[:E]->()-[:E]->(b:N) WHERE a.id IN $seeds RETURN b.id")
    for k in ks:
        s = seed_sets[k]
        for task, stmt in (("BULK1", q1), ("BULK2", q2)):
            try:
                rows = len(conn.execute(stmt, {"seeds": s}).get_as_df())
                ms = timeit(lambda: conn.execute(stmt, {"seeds": s}).get_as_df(), reps, "kuzu", warm)
            except Exception as ex:
                print(f"  kuzu {task} k={k} FAILED: {type(ex).__name__}: {ex}"); continue
            rec = dict(system="kuzu", engine="kuzu", task=task, k=k, n=len(ndf), edges=len(edf),
                       warm_ms=ms, e_rows=rows, n_rows=rows, load_ms=load_ms)
            print(f"  kuzu {'':11} {task} k={k:>7} {ms:10.3f}ms  rows={rows:>10}  (b.id, not-distinct)")
            if outf: outf.write(json.dumps(rec) + "\n"); outf.flush()
    # DEGALL: full out-degree aggregation
    try:
        qd = "MATCH (a:N)-[:E]->() RETURN a.id, count(*) AS deg"
        for _ in range(warm): conn.execute(qd).get_as_df()
        rows = len(conn.execute(qd).get_as_df())
        ms = timeit(lambda: conn.execute(qd).get_as_df(), reps, "kuzu", warm)
        rec = dict(system="kuzu", engine="kuzu", task="DEGALL", k=None, n=len(ndf), edges=len(edf),
                   warm_ms=ms, e_rows=rows, n_rows=rows, load_ms=load_ms)
        print(f"  kuzu {'':11} DEGALL{'':>13} {ms:10.3f}ms  groups={rows:>10}")
        if outf: outf.write(json.dumps(rec) + "\n"); outf.flush()
    except Exception as ex:
        print(f"  kuzu DEGALL FAILED: {type(ex).__name__}: {ex}")
    shutil.rmtree(dbp, ignore_errors=True)


def main():
    ndf, edf, nodes = load_graph()
    print(f"===== graph: {len(ndf):,} nodes  {len(edf):,} edges =====")
    ks = [int(x) for x in os.environ.get("KS", "1000,10000,100000").split(",")]
    engines = os.environ.get("ENGINES", "pandas,polars,cudf,polars-gpu").split(",")
    systems = os.environ.get("SYSTEMS", "gfql,kuzu").split(",")
    reps = int(os.environ.get("REPS", "10")); warm = int(os.environ.get("WARM", "2"))
    seed = int(os.environ.get("SEED", "0"))
    tmpdir = os.environ.get("TMPDIR_BENCH", "/tmp/bulkbench"); os.makedirs(tmpdir, exist_ok=True)
    outf = open(os.environ["OUT"], "a") if os.environ.get("OUT") else None
    if "gfql" in systems:
        run_gfql(ndf, edf, nodes, ks, engines, reps, warm, outf, seed)
    if "kuzu" in systems:
        run_kuzu(ndf, edf, nodes, ks, reps, warm, outf, seed, tmpdir)
    if outf: outf.close()


if __name__ == "__main__":
    main()
