#!/usr/bin/env python3
"""Head-to-head: GFQL physical index vs kuzu (CSR) vs neo4j on identical graphs.

Seeded SEL1 (1-hop) / SEL2 (2-hop) from one seed. All engines MATERIALIZE the
result (fair apples-to-apples — the method-guard from the pygraphistry2 bench).
GFQL is measured in two modes:
  - on-the-fly (inclusive): build index + warm query, amortized=1 (cold)  [index_policy=force-ish]
  - preindexed (warm)      : resident index, warm query only

Env: NS=100000,1000000  DEG=8  REPS=20  GFQL_ENGINES=pandas,polars,cudf,polars-gpu
     SYSTEMS=gfql,kuzu,neo4j  NEO4J_URI=bolt://localhost:7688 NEO4J_USER/NEO4J_PASS
     OUT=/results/vs-dbs.jsonl
"""
from __future__ import annotations

import json
import os
import statistics
import time
import numpy as np
import pandas as pd

import graphistry


def make_graph(n, deg, seed=0):
    rng = np.random.default_rng(seed)
    m = n * deg
    src = rng.integers(0, n, size=m, dtype=np.int64)
    dst = rng.integers(0, n, size=m, dtype=np.int64)
    return (pd.DataFrame({"id": np.arange(n, dtype=np.int64)}),
            pd.DataFrame({"src": src, "dst": dst}))


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


def emit(outf, rec):
    print(f"  {rec['system']:14} {rec['engine']:11} {rec['task']:5} "
          f"warm={rec['warm_ms']:>9.4f}ms  cold(build+q)={rec.get('cold_ms', float('nan')):>10.2f}ms  rows={rec['rows']}")
    if outf:
        outf.write(json.dumps(rec) + "\n"); outf.flush()


def run_gfql(ndf, edf, N, E, seeds, reps, engines, outf):
    g0 = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
    for engine in engines:
        try:
            t0 = time.perf_counter()
            gi = g0.gfql_index_all(engine=engine); _sync(engine)
            build_ms = (time.perf_counter() - t0) * 1e3
        except Exception as ex:
            print(f"  gfql {engine} build FAILED: {type(ex).__name__}: {ex}"); continue
        for task, kw in (("SEL1", dict(hops=1)), ("SEL2", dict(hops=2))):
            kw = dict(kw, direction="forward")
            try:
                warm = timeit(lambda: gi.hop(nodes=seeds, engine=engine, **kw), reps, engine)
                rows = int(gi.hop(nodes=seeds, engine=engine, **kw)._edges.shape[0])
            except Exception as ex:
                print(f"  gfql {engine} {task} FAILED: {type(ex).__name__}: {ex}"); continue
            emit(outf, dict(system="gfql", engine=engine, task=task, n=N, edges=E,
                            warm_ms=warm, cold_ms=build_ms + warm, build_ms=build_ms, rows=rows))


def run_kuzu(ndf, edf, N, E, reps, outf, tmpdir):
    try:
        import kuzu
    except Exception:
        print("  kuzu: NOT AVAILABLE (pip install kuzu)"); return
    import shutil
    dbp = os.path.join(tmpdir, f"kuzu_{N}")
    if os.path.exists(dbp):
        shutil.rmtree(dbp)
    ncsv = os.path.join(tmpdir, "kn.csv"); ecsv = os.path.join(tmpdir, "ke.csv")
    ndf.to_csv(ncsv, index=False, header=False)
    edf.to_csv(ecsv, index=False, header=False)
    t0 = time.perf_counter()
    db = kuzu.Database(dbp); conn = kuzu.Connection(db)
    conn.execute("CREATE NODE TABLE N(id INT64, PRIMARY KEY(id))")
    conn.execute("CREATE REL TABLE E(FROM N TO N)")
    conn.execute(f'COPY N FROM "{ncsv}"')
    conn.execute(f'COPY E FROM "{ecsv}"')
    load_ms = (time.perf_counter() - t0) * 1e3
    queries = {
        "SEL1": "MATCH (a:N {id:0})-[:E]->(b:N) RETURN b.id",
        "SEL2": "MATCH (a:N {id:0})-[:E]->()-[:E]->(b:N) RETURN b.id",
    }
    for task, q in queries.items():
        try:
            for _ in range(2):
                conn.execute(q).get_as_df()
            ts = []
            for _ in range(reps):
                t = time.perf_counter(); df = conn.execute(q).get_as_df()
                ts.append((time.perf_counter() - t) * 1e3)
            ts.sort()
            rows = len(df)
            emit(outf, dict(system="kuzu", engine="kuzu", task=task, n=N, edges=E,
                            warm_ms=statistics.median(ts), cold_ms=load_ms, build_ms=load_ms, rows=rows))
        except Exception as ex:
            print(f"  kuzu {task} FAILED: {type(ex).__name__}: {ex}")


def run_neo4j(ndf, edf, N, E, reps, outf):
    uri = os.environ.get("NEO4J_URI")
    if not uri:
        print("  neo4j: skipped (no NEO4J_URI)"); return
    try:
        from neo4j import GraphDatabase
    except Exception:
        print("  neo4j: driver NOT AVAILABLE"); return
    user = os.environ.get("NEO4J_USER", "neo4j"); pw = os.environ.get("NEO4J_PASS", "test")
    drv = GraphDatabase.driver(uri, auth=(user, pw))
    node_ids = ndf["id"].tolist()
    edge_rows = edf.values.tolist()
    BATCH = 20000
    t0 = time.perf_counter()
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        s.run("CREATE INDEX n_id IF NOT EXISTS FOR (n:N) ON (n.id)")
        s.run("CALL db.awaitIndexes(300)")
        for i in range(0, len(node_ids), BATCH):
            s.run("UNWIND $ids AS x CREATE (:N {id:x})", ids=node_ids[i:i + BATCH])
        for i in range(0, len(edge_rows), BATCH):
            s.run("UNWIND $rows AS r MATCH (a:N {id:r[0]}),(b:N {id:r[1]}) CREATE (a)-[:E]->(b)",
                  rows=edge_rows[i:i + BATCH])
        s.run("CALL db.awaitIndexes(300)")
    load_ms = (time.perf_counter() - t0) * 1e3
    qs = {
        "SEL1": "MATCH (a:N {id:0})-[:E]->(b:N) RETURN b.id AS id",
        "SEL2": "MATCH (a:N {id:0})-[:E]->()-[:E]->(b:N) RETURN b.id AS id",
    }
    with drv.session() as s:
        for task, q in qs.items():
            try:
                for _ in range(2):
                    list(s.run(q))
                ts = []
                for _ in range(reps):
                    t = time.perf_counter(); data = list(s.run(q))
                    ts.append((time.perf_counter() - t) * 1e3)
                ts.sort()
                emit(outf, dict(system="neo4j", engine="neo4j", task=task, n=N, edges=E,
                                warm_ms=statistics.median(ts), cold_ms=load_ms, build_ms=load_ms, rows=len(data)))
            except Exception as ex:
                print(f"  neo4j {task} FAILED: {type(ex).__name__}: {ex}")
    drv.close()


def main():
    NS = [int(x) for x in os.environ.get("NS", "100000,1000000").split(",")]
    DEG = int(os.environ.get("DEG", "8"))
    REPS = int(os.environ.get("REPS", "20"))
    engines = os.environ.get("GFQL_ENGINES", "pandas,polars,cudf,polars-gpu").split(",")
    systems = os.environ.get("SYSTEMS", "gfql,kuzu,neo4j").split(",")
    out = os.environ.get("OUT")
    tmpdir = os.environ.get("TMPDIR_BENCH", "/tmp/idxbench")
    os.makedirs(tmpdir, exist_ok=True)
    outf = open(out, "a") if out else None

    seeds = pd.DataFrame({"id": [0]})
    for N in NS:
        ndf, edf = make_graph(N, DEG)
        E = N * DEG
        print(f"\n=== N={N} edges={E} ===")
        if "gfql" in systems:
            run_gfql(ndf, edf, N, E, seeds, REPS, engines, outf)
        if "kuzu" in systems:
            run_kuzu(ndf, edf, N, E, REPS, outf, tmpdir)
        if "neo4j" in systems:
            run_neo4j(ndf, edf, N, E, REPS, outf)
    if outf:
        outf.close()


if __name__ == "__main__":
    main()
