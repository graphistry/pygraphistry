#!/usr/bin/env python3
"""FAIR GFQL-vs-Ladybug: run Ladybug's benchmark ops as GFQL Cypher MATCH...RETURN
(the row pipeline), NOT dataframe shortcuts. Compare to their published 5M/20M numbers.
Ladybug best (5M/20M, ms): full_scan 3789, range 7.5, point 0.3, count 3.3,
  out_degree100 59.8, scan_rel_props 15722, scan_rel_rowid 14562. Kuzu count 46.
"""
import os, time, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, graphistry

N = int(os.environ.get("N", "5000000")); E = int(os.environ.get("E", "20000000"))
REPS = int(os.environ.get("REPS", "5")); WARM = int(os.environ.get("WARM", "2"))
ENGINES = os.environ.get("ENGINES", "pandas,polars").split(",")
mid = N // 2; lo, hi = N // 2, N // 2 + 1000

def build():
    ids = np.arange(1, N + 1, dtype=np.int64)
    nd = pd.DataFrame({"id": ids, "name": ("abcdefghijklmn_name_" + pd.Series(ids).astype(str)).values})
    i = np.arange(1, E + 1, dtype=np.int64)
    ed = pd.DataFrame({"src": (i % N) + 1, "dst": ((i * 7) % N) + 1, "since": i})
    return graphistry.nodes(nd, "id").edges(ed, "src", "dst")

OPS = {
    "full_scan":      "MATCH (i) RETURN i.id, i.name",
    "range":          f"MATCH (i) WHERE i.id >= {lo} AND i.id <= {hi} RETURN i.id, i.name",
    "point":          f"MATCH (i) WHERE i.id = {mid} RETURN i.id, i.name",
    "count":          "MATCH ()-[r]->() RETURN COUNT(*)",
    "scan_rel_props": "MATCH (a)-[o]->(b) RETURN a.id, b.id, o.since",
    "scan_rel_rowid": "MATCH (a)-[r]->(b) RETURN a.id, b.id",
}
LADYBUG = {"full_scan": 3789, "range": 7.5, "point": 0.3, "count": 3.3,
           "scan_rel_props": 15722, "scan_rel_rowid": 14562}

def med(fn):
    for _ in range(WARM): fn()
    t = []
    for _ in range(REPS):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1e3)
    return statistics.median(sorted(t))

def main():
    print(f"N={N:,} E={E:,} REPS={REPS}", flush=True)
    g = build(); print("graph built", flush=True)
    print(f"{'op':16} {'engine':11} {'gfql_ms':>10} {'ladybug_ms':>11} {'vs_ladybug':>11}")
    for name, q in OPS.items():
        for eng in ENGINES:
            try:
                ms = med(lambda: g.gfql(q, engine=eng))
                lb = LADYBUG.get(name)
                vs = f"{lb/ms:.1f}x" if lb else "-"
                print(f"{name:16} {eng:11} {ms:10.3f} {str(lb):>11} {vs:>11}", flush=True)
            except Exception as ex:
                print(f"{name:16} {eng:11} {'NIE/ERR':>10} -> {type(ex).__name__}: {str(ex)[:40]}", flush=True)

if __name__ == "__main__":
    main()
