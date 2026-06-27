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


def _normalize_nulls(df):
    """Collapse pandas NaN/None and polars null to a single sentinel so the
    differential check compares null SEMANTICS, not the engines' null repr
    (``nan`` vs ``None``) which astype(str) would otherwise render differently."""
    return df.where(df.notna(), "∅")


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
    a, b = _normalize_nulls(_round_floats(a)), _normalize_nulls(_round_floats(b))
    if "ORDER BY" in query:
        pd.testing.assert_frame_equal(a.astype(str), b.astype(str), check_dtype=False)
    else:
        a_s = a.astype(str).sort_values(list(a.columns)).reset_index(drop=True)
        b_s = b.astype(str).sort_values(list(b.columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(a_s, b_s, check_dtype=False)


# Queries the polars engine runs NATIVELY (property/arith/order/agg/unwind +
# single-entity WHERE returning properties). Run on BASE; parity vs pandas.
CORPUS = [
    # property projection
    "MATCH (n) RETURN n.val",
    "MATCH (n) RETURN n.val, n.kind, n.score",
    "MATCH (n) RETURN n.val AS v, n.name AS nm",
    "MATCH (n) RETURN DISTINCT n.kind",
    # arithmetic / comparison / boolean projection
    "MATCH (n) RETURN n.val + 1 AS p",
    "MATCH (n) RETURN n.val * 2 - 3 AS x",
    "MATCH (n) RETURN n.val % 7 AS r",
    "MATCH (n) RETURN n.score / 2 AS half",
    # whitelisted scalar functions (native lowering)
    "MATCH (n) RETURN coalesce(n.val, 0) AS c",
    "MATCH (n) RETURN abs(n.val - 50) AS d",
    "MATCH (n) RETURN n.val > 50 AS big, n.kind",
    "MATCH (n) RETURN n.val >= 50 AND n.val <= 80 AS mid",
    # 3-valued boolean over bare null literals — must not crash on Null dtype
    # (polars & / | / ~ need Boolean cast). Cypher Kleene logic. Bare RETURN
    # (no MATCH) keeps it a single constant row on both engines.
    "RETURN true AND null AS a, false AND null AS b, null AND null AS c",
    "RETURN true OR null AS a, false OR null AS b, null OR null AS c",
    "RETURN NOT true AS a, NOT false AS b, NOT null AS c",
    "RETURN NOT NOT null AS a",
    # single-entity WHERE (folds into matcher), returning properties
    "MATCH (n) WHERE n.kind = 'alpha' RETURN n.val",
    "MATCH (n) WHERE n.val > 20 AND n.val < 90 RETURN n.name",
    "MATCH (n) WHERE n.flag = true RETURN n.val",
    # single-entity WHERE that does NOT fold (OR / NOT) -> native where_rows filter
    "MATCH (n) WHERE n.val > 80 OR n.kind = 'alpha' RETURN n.val, n.kind",
    "MATCH (n) WHERE n.val < 20 OR n.val > 80 RETURN n.val ORDER BY n.val",
    "MATCH (n) WHERE NOT n.kind = 'beta' RETURN n.kind",
    # native predicate lowering (no pandas bridge): STARTS WITH, range (AllOf)
    "MATCH (n) WHERE n.name STARTS WITH 'node' RETURN n.name",
    "MATCH (n) WHERE n.val > 20 AND n.val < 90 RETURN n.name",
    "MATCH (n) WHERE n.flag = true OR n.val > 50 RETURN n.name ORDER BY n.name",
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
    # whole-entity returns — now FLATTEN to {alias}.{field} columns (#1650
    # structured returns), native for ANY dtype incl BASE.score (float).
    # (Single-MATCH only here: MATCH (n)-[e]->(m) RETURN m is correct on polars
    # but pandas upcasts m.val int->float in the binding merge, so it's not a
    # clean differential case — polars is more correct. See plan.md.)
    "MATCH (n) RETURN n",
    "MATCH (n) RETURN n LIMIT 5",
    "MATCH (n) RETURN DISTINCT n",
    # UNION / UNION ALL — the distinct de-dup must use the polars-native unique()
    # (regression: it called pandas drop_duplicates on a polars frame and crashed).
    "RETURN 1 AS x UNION RETURN 2 AS x",
    "RETURN 1 AS x UNION RETURN 1 AS x",
    "RETURN 1 AS x UNION ALL RETURN 1 AS x",
    "MATCH (n) WHERE n.kind = 'alpha' RETURN n.val UNION MATCH (n) WHERE n.kind = 'beta' RETURN n.val",
]


@pytest.mark.parametrize("query", CORPUS)
def test_cypher_conformance_corpus(query):
    _assert_parity(BASE, query)


# NO-CHEATING (see plan.md): the polars engine has no native implementation for
# these yet, so it must raise NotImplementedError (NOT silently run pandas).
# Whole-entity RETURN over a float column (BASE.score), multi-entity bindings,
# and cross-entity same-path WHERE.
DEFERRED = [
    # Whole-entity RETURN now FLATTENS (#1650 structured returns) instead of
    # rendering text, so float/whole-entity returns are native — moved to CORPUS.
    # These remain deferred (honest NIE, no pandas bridge):
    "MATCH (n) RETURN n, n.val",                            # duplicate output col (polars .select rejects)
    "MATCH (n)-[e]->(m) RETURN n.val, m.val",               # multi-entity bindings
    "MATCH (n)-[e]->(m) WHERE n.val < m.val RETURN n, m",   # cross-entity WHERE
    "MATCH (a)-[e]->(b) WHERE a.val < b.val RETURN a.kind, b.kind",
    "MATCH (a)-[e]->(b) WHERE a.kind = b.kind RETURN a.id, b.id",
    # temporal arithmetic: duration(...) lowers to an ISO string literal, so
    # a.time + duration(...) must NOT silently become string concatenation
    "MATCH (n) RETURN n.val + duration({minutes: 6}) AS t",
    "MATCH (n) WITH n ORDER BY n.val + duration({days: 1}) RETURN n.val",
]


@pytest.mark.parametrize("query", DEFERRED)
def test_cypher_deferred_raises_not_bridges(query):
    with pytest.raises(NotImplementedError):
        BASE.gfql(query, engine="polars")


def test_temporal_constructor_property_declines_honestly():
    """A standalone property projection over a temporal-constructor string column
    (``date({year: 1910, month: 5, day: 6})`` — how Cypher/TCK store temporal
    values) must raise NotImplementedError, not leak the raw constructor text
    (pandas normalizes it to ISO; that normalizer is not yet native)."""
    nodes = pd.DataFrame({
        "id": [0, 1],
        "date": ["date({year: 1910, month: 5, day: 6})", "date({year: 1980, month: 10, day: 24})"],
    })
    edges = pd.DataFrame({"s": [0], "d": [1]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    with pytest.raises(NotImplementedError):
        g.gfql("MATCH (n) RETURN n.date", engine="polars")


def test_optional_match_absent_entity_renders_null():
    """OPTIONAL MATCH miss → the absent whole-entity must render as null, not '()'
    (the alias marker column is null; mirrors pandas _nullify_missing_alias_rows)."""
    empty = pd.DataFrame({"id": pd.Series([], dtype="int64")})
    edges = pd.DataFrame({"s": pd.Series([], dtype="int64"), "d": pd.Series([], dtype="int64")})
    g = graphistry.nodes(empty, "id").edges(edges, "s", "d")
    out = g.gfql("OPTIONAL MATCH (n) RETURN n", engine="polars")._nodes
    out = out.to_pandas() if hasattr(out, "to_pandas") else out
    assert out["n"].tolist() == [None]


def test_mixed_type_column_declines_honestly():
    """A heterogeneous (int+str) object column — legal for dynamically-typed Cypher
    properties in pandas, but unrepresentable in polars/Arrow — must raise a clear
    NotImplementedError (use engine='pandas'), NOT a cryptic pyarrow ArrowInvalid."""
    nodes = pd.DataFrame({"id": [0, 1, 2], "var": [0, "xx", None]})  # int + str + null
    edges = pd.DataFrame({"s": [0], "d": [1]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    with pytest.raises(NotImplementedError):
        g.gfql("MATCH (n) WHERE n.var > 'x' RETURN n.var", engine="polars")


def _nullable_graph():
    """Nulls in numeric/string/bool columns + zero/negative — exercises the
    native lowering's NULL / cypher 3-valued-logic semantics vs pandas."""
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 5, 6],
        "val": [10, None, 30, None, 50, 0, -5],
        "kind": ["a", "b", None, "a", None, "b", "a"],
        "flag": [True, None, False, True, None, False, True],
    })
    edges = pd.DataFrame({"s": [0, 1, 2, 3, 4, 5], "d": [1, 2, 3, 4, 5, 6]})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


NULLABLE = [
    "MATCH (n) WHERE n.val > 25 RETURN n.val",           # null compares -> excluded
    "MATCH (n) WHERE n.val >= 0 RETURN n.id",
    "MATCH (n) RETURN n.val + 1 AS p",                    # null arithmetic -> null
    "MATCH (n) RETURN coalesce(n.val, -1) AS c",          # coalesce fills null
    "MATCH (n) RETURN abs(n.val) AS a",                   # abs over null -> null
    "MATCH (n) RETURN n.val > 25 AS big",                # null comparison projection
    "MATCH (n) WHERE n.val > 5 AND n.kind = 'a' RETURN n.id",   # 3-valued AND (folds)
    "MATCH (n) WHERE n.val > 5 OR n.kind = 'b' RETURN n.id",    # 3-valued OR -> native where_rows
    "MATCH (n) WHERE n.val < 0 OR n.flag = true RETURN n.id",   # null in OR operands
    "MATCH (n) WHERE NOT n.val > 25 RETURN n.id",               # NOT over null -> null dropped
    "MATCH (n) RETURN n.val ORDER BY n.val",             # null sort position
    "MATCH (n) RETURN n.val ORDER BY n.val DESC",
    "MATCH (n) RETURN n.kind, count(n) AS c",            # null group key
    "MATCH (n) RETURN n.kind, sum(n.val) AS s, avg(n.val) AS a",  # null in agg
    "MATCH (n) RETURN DISTINCT n.kind",
    "MATCH (n) WHERE n.flag = true RETURN n.id",         # nullable bool
    "MATCH (n) WHERE n.val IS NULL RETURN n.id",          # IsNA -> is_null (native)
    "MATCH (n) WHERE n.kind IS NOT NULL RETURN n.id",     # NotNA -> is_not_null (native)
    "MATCH (n) WHERE n.val IS NULL OR n.val > 40 RETURN n.id",  # null check in OR
]


@pytest.mark.parametrize("query", NULLABLE)
def test_cypher_conformance_nullable(query):
    _assert_parity(_nullable_graph(), query)


def _scalar_graph():
    """int/string/bool only — eligible for native polars entity-text rendering,
    incl. quote/backslash escaping and null omission."""
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 3],
        "amount": [10, 20, 30, 40],
        "label": ["plain", "has'quote", "back\\slash", None],
        "active": [True, False, True, False],
    })
    edges = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3]})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def test_native_entity_text_parity():
    """Whole-entity RETURN n FLATTENS to a.* columns natively in polars (#1650
    structured returns) and matches pandas. No pandas bridge. The legacy display
    string is presentation-only via render_entity_text()."""
    g = _scalar_graph()
    _assert_parity(g, "MATCH (n) RETURN n")


@pytest.mark.parametrize("seed", list(range(40)))
def test_cypher_conformance_fuzz(seed):
    """Seeded fuzzer: random RETURN/WHERE/ORDER/LIMIT/agg queries, both engines."""
    rng = random.Random(seed)
    g = _graph(seed % 5, n=rng.choice([6, 12, 20]))
    props = ["n.val", "n.score", "n.kind", "n.name"]
    num_props = ["n.val", "n.score"]

    shape = rng.choice(["project", "where", "or_where", "order", "agg", "distinct", "limit", "arith"])
    if shape == "project":
        sel = ", ".join(rng.sample(props, rng.randint(1, 3)))
        q = f"MATCH (n) RETURN {sel}"
    elif shape == "where":
        p = rng.choice(num_props)
        op = rng.choice([">", "<", ">=", "<=", "="])
        v = rng.randint(0, 100)
        q = f"MATCH (n) WHERE {p} {op} {v} RETURN n.val, n.kind"
    elif shape == "or_where":
        # OR doesn't fold into the node matcher -> exercises native where_rows
        p1, p2 = rng.sample(num_props, 2)
        o1, o2 = rng.choice([">", "<", ">=", "<="]), rng.choice([">", "<", ">=", "<="])
        v1, v2 = rng.randint(0, 100), rng.randint(0, 100)
        q = f"MATCH (n) WHERE {p1} {o1} {v1} OR {p2} {o2} {v2} RETURN n.val, n.kind"
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
