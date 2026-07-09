#!/usr/bin/env python3
"""Fairest GFQL-vs-kuzu seeded 1-hop: BOTH in-process, warm, kuzu using a PREPARED
statement (its fast path) + result fully materialized on both sides, same seed,
matched answer counts. Removes Cypher-parse-per-call from kuzu so the comparison is
engine-vs-engine, not engine-vs-(parse+engine). kuzu is embedded like GFQL (no bolt
network), so it's the cleanest peer. Env: PARQUET=/data/<edges>.parquet"""
import os, time, statistics, tempfile, shutil
import numpy as np, pandas as pd, graphistry, kuzu


def med(fn, reps=25, warm=4):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    ts.sort(); return statistics.median(ts)


def main():
    edf = pd.read_parquet(os.environ["PARQUET"]).astype({"src": np.int64, "dst": np.int64})
    nodes = np.unique(np.concatenate([edf["src"].values, edf["dst"].values]))
    print(f"graph: {len(nodes):,} nodes / {len(edf):,} edges")
    g = graphistry.nodes(pd.DataFrame({"id": nodes}), "id").edges(edf, "src", "dst")
    gi = g.gfql_index_all(engine="pandas")
    deg = edf.groupby("src").size()
    # typical (median-degree) and hub seeds
    for tag, sid in [("typical", int(deg[deg >= deg.median()].index[0])), ("hub", int(deg.idxmax()))]:
        d = int(deg.loc[sid]); seeds = pd.DataFrame({"id": [sid]})
        gfql_rows = int(gi.hop(nodes=seeds, engine="pandas", hops=1)._edges.shape[0])
        gfql_ms = med(lambda: gi.hop(nodes=seeds, engine="pandas", hops=1))

        dbp = tempfile.mkdtemp()
        db = kuzu.Database(os.path.join(dbp, "kz")); conn = kuzu.Connection(db)
        conn.execute("CREATE NODE TABLE N(id INT64, PRIMARY KEY(id))")
        conn.execute("CREATE REL TABLE E(FROM N TO N)")
        np_path = os.path.join(dbp, "n.parquet"); ep_path = os.path.join(dbp, "e.parquet")
        pd.DataFrame({"id": nodes}).to_parquet(np_path)
        edf.rename(columns={"src": "from", "dst": "to"}).to_parquet(ep_path)
        conn.execute(f'COPY N FROM "{np_path}"'); conn.execute(f'COPY E FROM "{ep_path}"')
        stmt = conn.prepare("MATCH (a:N {id:$sid})-[:E]->(b:N) RETURN b.id")
        # Columnar materialization (kuzu's fast result path) == GFQL's DataFrame output.
        def kq():
            conn.execute(stmt, {"sid": sid}).get_as_df()
        kr = len(conn.execute(stmt, {"sid": sid}).get_as_df())
        kuzu_ms = med(kq)
        ratio = kuzu_ms / gfql_ms if gfql_ms else float("nan")
        print(f"  {tag:8} deg={d:>7}  GFQL-pandas {gfql_ms:8.4f}ms (rows={gfql_rows})  "
              f"kuzu-prepared {kuzu_ms:8.4f}ms (rows={kr})  match={gfql_rows==kr}  "
              f"GFQL {'faster' if ratio>1 else 'SLOWER'} {ratio:.2f}x")
        shutil.rmtree(dbp, ignore_errors=True)


if __name__ == "__main__":
    main()
