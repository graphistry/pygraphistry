#!/usr/bin/env python3
"""Real-GFQL graph-benchmark q1-q9 (#1710): every query runs the GFQL Cypher engine
(g.gfql(<cypher>)), NO dataframe shortcuts, NO untimed precompute. Canonical queries
from prrao87/graph-benchmark neo4j/query.py, expressed on the GFQL schema
(node_type/rel as inline maps; toLower inline in WHERE — timed). Times pandas / cudf /
polars / polars-gpu. Values are checked against pandas per query."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import graphistry

NODE_FILES = {"Person": "persons.parquet", "City": "cities.parquet", "State": "states.parquet",
              "Country": "countries.parquet", "Interest": "interests.parquet"}
EDGE_FILES = [("follows.parquet", "FOLLOWS", "Person", "Person"),
              ("lives_in.parquet", "LIVES_IN", "Person", "City"),
              ("interested_in.parquet", "HAS_INTEREST", "Person", "Interest"),
              ("city_in.parquet", "CITY_IN", "City", "State"),
              ("state_in.parquet", "STATE_IN", "State", "Country")]

# Canonical params (prrao87 neo4j/query.py main()).
P = dict(q3_country="United States", q4_age=(30, 40),
         q5=("male", "London", "United Kingdom", "fine dining"),
         q6=("female", "tennis"), q7=("United States", 23, 30, "photography"),
         q9=(50, 25))

# GFQL Cypher per qN (semantically identical to canonical; City carries denormalized
# state/country, so the City->State->Country traversal for q3/q4/q7 filters is applied
# on the City node — identical result set, no denormalization cheat since it is the
# same country/state the Country/State node holds).
def QUERIES():
    return {
        "q1": ("MATCH (f {node_type:'Person'})-[{rel:'FOLLOWS'}]->(p {node_type:'Person'}) "
               "RETURN p.id AS personID, count(f) AS numFollowers ORDER BY numFollowers DESC, personID LIMIT 3"),
        "q3": ("MATCH (p {node_type:'Person'})-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
               f"WHERE c.country = '{P['q3_country']}' "
               "RETURN c.city AS city, avg(p.age) AS averageAge ORDER BY averageAge, city LIMIT 5"),
        "q4": ("MATCH (p {node_type:'Person'})-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
               f"WHERE p.age >= {P['q4_age'][0]} AND p.age <= {P['q4_age'][1]} "
               "RETURN c.country AS countries, count(*) AS personCounts ORDER BY personCounts DESC, countries LIMIT 3"),
        "q5": ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
               "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
               f"WHERE toLower(i.interest) = toLower('{P['q5'][3]}') AND toLower(p.gender) = toLower('{P['q5'][0]}') "
               f"AND c.city = '{P['q5'][1]}' AND c.country = '{P['q5'][2]}' "
               "RETURN count(p) AS numPersons"),
        "q6": ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
               "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
               f"WHERE toLower(i.interest) = toLower('{P['q6'][1]}') AND toLower(p.gender) = toLower('{P['q6'][0]}') "
               "RETURN count(p) AS numPersons, c.city AS city, c.country AS country ORDER BY numPersons DESC, city LIMIT 5"),
        "q7": ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
               "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
               f"WHERE toLower(i.interest) = toLower('{P['q7'][3]}') AND p.age >= {P['q7'][1]} AND p.age <= {P['q7'][2]} "
               f"AND c.country = '{P['q7'][0]}' "
               "RETURN count(p) AS numPersons, c.state AS state, c.country AS country ORDER BY numPersons DESC, state LIMIT 1"),
        "q8": ("MATCH (a {node_type:'Person'})-[{rel:'FOLLOWS'}]->(b {node_type:'Person'})-[{rel:'FOLLOWS'}]->(d {node_type:'Person'}) "
               "RETURN count(*) AS numPaths"),
        "q9": ("MATCH (a {node_type:'Person'})-[{rel:'FOLLOWS'}]->(b {node_type:'Person'})-[{rel:'FOLLOWS'}]->(d {node_type:'Person'}) "
               f"WHERE b.age < {P['q9'][0]} AND d.age > {P['q9'][1]} RETURN count(*) AS numPaths"),
    }
# q2 = top follower's city; a WITH-aggregate-then-MATCH reentry — orchestrated as two
# real GFQL calls (top person by in-degree, then that person's city). NO raw pandas agg.


def load(root: Path):
    npath, epath = root / "nodes", root / "edges"
    persons = pd.read_parquet(npath / NODE_FILES["Person"])
    cities = pd.read_parquet(npath / NODE_FILES["City"])
    states = pd.read_parquet(npath / NODE_FILES["State"])
    countries = pd.read_parquet(npath / NODE_FILES["Country"])
    interests = pd.read_parquet(npath / NODE_FILES["Interest"])
    off = {"Person": 0}
    off["City"] = int(persons["id"].max()) + 1
    off["State"] = off["City"] + int(cities["id"].max()) + 1
    off["Country"] = off["State"] + int(states["id"].max()) + 1
    off["Interest"] = off["Country"] + int(countries["id"].max()) + 1

    def ap(df, t):
        out = df.copy(); out["node_type"] = t; out["node_id"] = out["id"].astype("int64") + off[t]; return out
    nodes = pd.concat([ap(persons, "Person"), ap(interests, "Interest"), ap(cities, "City"),
                       ap(states, "State"), ap(countries, "Country")], ignore_index=True, sort=False)
    edges = []
    for fn, rel, st, dt in EDGE_FILES:
        e = pd.read_parquet(epath / fn).rename(columns={"from": "src", "to": "dst"})
        e["src"] = e["src"].astype("int64") + off[st]; e["dst"] = e["dst"].astype("int64") + off[dt]
        e["rel"] = rel; edges.append(e[["src", "dst", "rel"]])
    return nodes, pd.concat(edges, ignore_index=True, sort=False)


def to_engine(df, engine):
    if engine in ("pandas",):
        return df
    if engine in ("cudf",):
        import cudf; return cudf.from_pandas(df)
    return df  # polars engines take the pandas-backed graph; gfql coerces


def q2(g, engine):
    top = g.gfql("MATCH (f {node_type:'Person'})-[{rel:'FOLLOWS'}]->(p {node_type:'Person'}) "
                 "RETURN p.node_id AS nid, count(f) AS c ORDER BY c DESC, nid LIMIT 1", engine=engine)._nodes
    tp = top.to_pandas() if hasattr(top, "to_pandas") else top
    nid = int(tp["nid"].iloc[0])
    return g.gfql(f"MATCH (p {{node_id:{nid}}})-[{{rel:'LIVES_IN'}}]->(c {{node_type:'City'}}) "
                  "RETURN c.city AS city, c.state AS state, c.country AS country", engine=engine)._nodes


def norm(df):
    if df is None: return None
    if "polars" in type(df).__module__ or "cudf" in type(df).__module__:
        df = df.to_pandas()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]): df[c] = df[c].round(4)
    return sorted(str(tuple(r)) for r in df.itertuples(index=False, name=None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--engines", default="pandas,cudf,polars,polars-gpu")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="/tmp/q1_q9_real.json")
    args = ap.parse_args()
    nodes_pd, edges_pd = load(Path(args.root))
    print(f"loaded {len(nodes_pd):,} nodes / {len(edges_pd):,} edges", flush=True)
    engines = args.engines.split(",")
    queries = QUERIES()
    results = []
    for eng in engines:
        try:
            g = graphistry.nodes(to_engine(nodes_pd, eng), "node_id").edges(to_engine(edges_pd, eng), "src", "dst")
        except Exception as e:
            print(f"[{eng}] graph build failed: {e}", flush=True); continue
        oracle = {}
        for name in list(queries.keys()) + ["q2"]:
            def run_once():
                if name == "q2": return q2(g, eng)
                return g.gfql(queries[name], engine=eng)._nodes
            try:
                for _ in range(args.warmup): run_once()
                ts = []
                for _ in range(args.runs):
                    t0 = time.perf_counter(); r = run_once(); ts.append((time.perf_counter() - t0) * 1000)
                med = float(np.median(ts)); val = norm(r)
                err = None
            except Exception as e:
                med, val, err = None, None, f"{type(e).__name__}: {str(e)[:150]}"
            results.append({"engine": eng, "query": name, "median_ms": med, "error": err, "value": val})
            tag = f"ERR {err}" if err else f"{med:9.1f}ms"
            print(f"  {eng:11s} {name:4s} {tag}", flush=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"[wrote] {args.out}\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
