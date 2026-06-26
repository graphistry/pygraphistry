"""Differential cypher conformance: engine='polars' == engine='pandas'.

A broad TCK-style conformance lane for the native polars engine: a large curated
corpus plus a seeded query fuzzer, each run on both engines and asserted to
produce identical result tables. Pandas is the oracle. This is the polars
counterpart of the cross-repo Cypher TCK harness (graphistry/tck-gfql) — it
keeps the polars row pipeline honest across the whole cypher surface, native and
host-bridged paths alike. See plans/gfql-polars-engine.
"""
import random

import pandas as pd
import pytest

import graphistry

pl = pytest.importorskip("polars")


def _graph(seed: int = 0, n: int = 12):
    rng = random.Random(seed)
    kinds = ["alpha", "beta", "gamma"]
    nodes = pd.DataFrame({
        "id": list(range(n)),
        "val": [rng.randint(0, 100) for _ in range(n)],
        "score": [round(rng.uniform(0, 10), 2) for _ in range(n)],
        "kind": [rng.choice(kinds) for _ in range(n)],
        "name": [f"node{i}" for i in range(n)],
        "flag": [rng.choice([True, False]) for _ in range(n)],
    })
    src = [rng.randint(0, n - 1) for _ in range(n * 2)]
    dst = [rng.randint(0, n - 1) for _ in range(n * 2)]
    edges = pd.DataFrame({"s": src, "d": dst, "w": [round(rng.uniform(0, 1), 3) for _ in range(n * 2)]})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


BASE = _graph(0)


def _to_pd(df):
    return df.to_pandas() if df is not None and "polars" in type(df).__module__ else df


def _round_floats(df):
    """Dampen last-ULP float differences (e.g. sum/avg summation order) so the
    differential check tests semantics, not IEEE-754 reduction order."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(6)
    return out


def _assert_parity(g, query):
    a = _to_pd(g.gfql(query, engine="pandas")._nodes).reset_index(drop=True)
    b = _to_pd(g.gfql(query, engine="polars")._nodes).reset_index(drop=True)
    assert list(a.columns) == list(b.columns), f"cols differ for {query!r}: {list(a.columns)} vs {list(b.columns)}"
    assert len(a) == len(b), f"row count differs for {query!r}: {len(a)} vs {len(b)}"
    if len(a) == 0:
        return
    # Bare LIMIT without ORDER BY selects an arbitrary k rows (cypher: order
    # undefined) — the engines may legitimately pick different rows, so only the
    # column shape + row count are conformant here.
    if "LIMIT" in query and "ORDER BY" not in query:
        return
    a, b = _round_floats(a), _round_floats(b)
    if "ORDER BY" in query:
        pd.testing.assert_frame_equal(a.astype(str), b.astype(str), check_dtype=False)
    else:
        a_s = a.astype(str).sort_values(list(a.columns)).reset_index(drop=True)
        b_s = b.astype(str).sort_values(list(b.columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(a_s, b_s, check_dtype=False)


CORPUS = [
    # whole-entity
    "MATCH (n) RETURN n",
    "MATCH (n) RETURN n LIMIT 5",
    "MATCH (n) RETURN n SKIP 3",
    "MATCH (n) RETURN n SKIP 2 LIMIT 4",
    "MATCH (n) RETURN DISTINCT n",
    # property projection
    "MATCH (n) RETURN n.val",
    "MATCH (n) RETURN n.val, n.kind, n.score",
    "MATCH (n) RETURN n.val AS v, n.name AS nm",
    "MATCH (n) RETURN n, n.val",
    "MATCH (n) RETURN DISTINCT n.kind",
    # arithmetic / comparison / boolean projection
    "MATCH (n) RETURN n.val + 1 AS p",
    "MATCH (n) RETURN n.val * 2 - 3 AS x",
    "MATCH (n) RETURN n.val % 7 AS r",
    "MATCH (n) RETURN n.score / 2 AS half",
    "MATCH (n) RETURN n.val > 50 AS big, n.kind",
    "MATCH (n) RETURN n.val >= 50 AND n.val <= 80 AS mid",
    # single-entity WHERE (folds into matcher)
    "MATCH (n) WHERE n.val > 40 RETURN n",
    "MATCH (n) WHERE n.kind = 'alpha' RETURN n.val",
    "MATCH (n) WHERE n.val > 20 AND n.val < 90 RETURN n.name",
    "MATCH (n) WHERE n.flag = true RETURN n.val",
    # order_by
    "MATCH (n) RETURN n.val ORDER BY n.val",
    "MATCH (n) RETURN n.val ORDER BY n.val DESC",
    "MATCH (n) RETURN n.kind, n.val ORDER BY n.kind, n.val DESC",
    "MATCH (n) WHERE n.val > 10 RETURN n.val ORDER BY n.val DESC LIMIT 5",
    "MATCH (n) RETURN n.score ORDER BY n.score SKIP 2 LIMIT 4",
    # aggregation
    "MATCH (n) RETURN count(n) AS c",
    "MATCH (n) RETURN n.kind, count(n) AS c",
    "MATCH (n) RETURN n.kind, sum(n.val) AS s",
    "MATCH (n) RETURN n.kind, avg(n.val) AS a, min(n.val) AS mn, max(n.val) AS mx",
    "MATCH (n) RETURN n.kind, count(n) AS c ORDER BY c DESC",
    # unwind
    "MATCH (n) UNWIND [1, 2, 3] AS x RETURN n.val, x",
    "MATCH (n) UNWIND ['a', 'b'] AS t RETURN n.kind, t",
    # relationship patterns
    "MATCH (n)-[e]->(m) RETURN m",
    "MATCH (n)-[e]->(m) RETURN n.val, m.val",
    "MATCH (n)-[e]->(m) WHERE n.val < m.val RETURN n, m",
    "MATCH (a)-[e]->(b) RETURN b LIMIT 5",
]


@pytest.mark.parametrize("query", CORPUS)
def test_cypher_conformance_corpus(query):
    _assert_parity(BASE, query)


@pytest.mark.parametrize("seed", list(range(40)))
def test_cypher_conformance_fuzz(seed):
    """Seeded fuzzer: random RETURN/WHERE/ORDER/LIMIT/agg queries, both engines."""
    rng = random.Random(seed)
    g = _graph(seed % 5, n=rng.choice([6, 12, 20]))
    props = ["n.val", "n.score", "n.kind", "n.name"]
    num_props = ["n.val", "n.score"]

    shape = rng.choice(["project", "where", "order", "agg", "distinct", "limit", "arith"])
    if shape == "project":
        sel = ", ".join(rng.sample(props, rng.randint(1, 3)))
        q = f"MATCH (n) RETURN {sel}"
    elif shape == "where":
        p = rng.choice(num_props)
        op = rng.choice([">", "<", ">=", "<=", "="])
        v = rng.randint(0, 100)
        q = f"MATCH (n) WHERE {p} {op} {v} RETURN n.val, n.kind"
    elif shape == "order":
        p = rng.choice(num_props)
        d = rng.choice(["", " DESC"])
        q = f"MATCH (n) RETURN {p}, n.kind ORDER BY {p}{d}"
    elif shape == "agg":
        fn = rng.choice(["count", "sum", "avg", "min", "max"])
        arg = "n" if fn == "count" else rng.choice(num_props)
        key = rng.choice(["n.kind", None])
        if key:
            q = f"MATCH (n) RETURN {key}, {fn}({arg}) AS r"
        else:
            q = f"MATCH (n) RETURN {fn}({arg}) AS r"
    elif shape == "distinct":
        q = f"MATCH (n) RETURN DISTINCT {rng.choice(props)}"
    elif shape == "limit":
        q = f"MATCH (n) RETURN n.val SKIP {rng.randint(0, 3)} LIMIT {rng.randint(1, 6)}"
    else:  # arith
        p = rng.choice(num_props)
        op = rng.choice(["+", "-", "*"])
        v = rng.randint(1, 9)
        q = f"MATCH (n) RETURN {p} {op} {v} AS x, n.kind"

    _assert_parity(g, q)
