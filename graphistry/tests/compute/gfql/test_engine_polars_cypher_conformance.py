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
    # NaN comparison: 0.0/0.0 computes NaN inside polars; polars treats NaN as the
    # LARGEST value (NaN>1 True) but IEEE/pandas/cypher compare any NaN false (!= true)
    "RETURN 0.0 / 0.0 > 1 AS gt, 0.0 / 0.0 >= 1 AS gtE, 0.0 / 0.0 < 1 AS lt, 0.0 / 0.0 <= 1 AS ltE",
    "RETURN 0.0 / 0.0 = 0.0 AS eq, 0.0 / 0.0 <> 0.0 AS ne",
    # NaN from a FUNCTION / division result (AST inference missed these; output-dtype
    # guard catches them — polars NaN-as-largest would otherwise leak)
    "RETURN abs(0.0 / 0.0) > 1 AS a, coalesce(0.0 / 0.0, 0.0) > 1 AS b",
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
    # numeric-vs-string comparison: polars raises ComputeError (pandas/cypher return
    # a value/null), so the lowering must decline rather than crash
    "MATCH (n) RETURN n.val > 'a' AS x",
    "MATCH (n) WHERE n.val < 'z' RETURN n.id",
    # ISO temporal comparison: cypher time()/date()/datetime() lower to ISO strings;
    # polars would compare them lexicographically (wrong across timezones) -> NIE
    "RETURN time({hour: 10, timezone: '+01:00'}) > time({hour: 9, timezone: '+00:00'}) AS x",
    "RETURN date({year: 1984, month: 10, day: 12}) < date({year: 1985, month: 5, day: 6}) AS x",
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


@pytest.mark.parametrize("edges,chain_cypher", [
    # null endpoint promotes the column to float64 vs int64 node ids — the chain's
    # endpoint<->node-id joins used to SchemaError (the hop casts, the chain didn't)
    (pd.DataFrame({"s": [1, 2, None], "d": [2.0, 3, 3]}), "MATCH (a)-[]->(b) RETURN b.id"),
    (pd.DataFrame({"s": [1.0, 2.0], "d": [2.0, 3.0]}), "MATCH (a)-[]->(b)-[]->(c) RETURN c.id"),
])
def test_chain_dtype_mismatched_endpoints_no_crash(edges, chain_cypher):
    """Node-id dtype != edge-endpoint dtype (int vs float, e.g. a null endpoint) must
    not crash the polars chain — align join keys, restore output dtype to match pandas."""
    nodes = pd.DataFrame({"id": [1, 2, 3, 4]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    _assert_parity(g, chain_cypher)


def test_chain_otel_decorator_on_public_chain():
    """The gfql.chain OTel span must wrap the public chain(), not the fast-path probe."""
    from graphistry.compute.chain import chain as _chain, _try_chain_fast_path
    assert hasattr(_chain, "__wrapped__")  # decorated
    assert not hasattr(_try_chain_fast_path, "__wrapped__")  # not decorated


def test_optional_match_absent_entity_renders_null():
    """OPTIONAL MATCH miss → the absent whole-entity must render as null, not '()'
    (the alias marker column is null; mirrors pandas _nullify_missing_alias_rows)."""
    empty = pd.DataFrame({"id": pd.Series([], dtype="int64")})
    edges = pd.DataFrame({"s": pd.Series([], dtype="int64"), "d": pd.Series([], dtype="int64")})
    g = graphistry.nodes(empty, "id").edges(edges, "s", "d")
    out = g.gfql("OPTIONAL MATCH (n) RETURN n", engine="polars")._nodes
    out = out.to_pandas() if hasattr(out, "to_pandas") else out
    # The single absent-entity row must be NULL. polars→pandas renders a null as
    # None or NaN depending on column dtype / polars version (1.40 gives NaN, newer
    # gives None) — both are null, so assert is-null rather than `== [None]`.
    assert len(out) == 1 and pd.isna(out["n"].iloc[0])


@pytest.mark.parametrize("nodes,query", [
    # user List-valued property compared to a scalar — must NOT silently apply
    # list-membership (pandas compares the whole list); decline (not the labels col)
    (pd.DataFrame({"id": [0, 1], "tags": [["a", "b"], ["c"]]}), "MATCH (n) WHERE n.tags = 'a' RETURN n.id"),
    # numeric-vs-string nested in AllOf (x>20 AND x<'z') — would PANIC if not detected
    (pd.DataFrame({"id": [0, 1, 2], "val": [10, 50, 90]}), "MATCH (n) WHERE n.val > 20 AND n.val < 'z' RETURN n.id"),
    # all-null column types as String in from_pandas → numeric arithmetic crashes
    (pd.DataFrame({"id": [0, 1], "val": [None, None]}), "MATCH (n) RETURN n.val + 1 AS x"),
    # categorical column vs numeric — polars ComputeError, must decline
    (pd.DataFrame({"id": [0, 1], "kind": pd.Series(["a", "b"], dtype="category")}), "MATCH (n) WHERE n.kind > 5 RETURN n.id"),
])
def test_polars_engine_declines_cross_type_not_crash(nodes, query):
    """Review-found cases where polars would CRASH/panic or silently misanswer —
    must raise an honest NotImplementedError instead (NO-CHEATING)."""
    g = graphistry.nodes(nodes, "id").edges(pd.DataFrame({"s": [0], "d": [1]}), "s", "d")
    with pytest.raises(NotImplementedError):
        g.gfql(query, engine="polars")


def test_polars_string_column_vs_date_literal_computes():
    """A genuine String property compared to a date-looking literal must COMPUTE
    (lexicographic, like pandas), not be over-declined by the ISO-temporal guard."""
    nodes = pd.DataFrame({"id": [0, 1], "w": ["2020-06-01", "2022-01-01"]})
    g = graphistry.nodes(nodes, "id").edges(pd.DataFrame({"s": [0], "d": [1]}), "s", "d")
    _assert_parity(g, "MATCH (n) RETURN n.w < '2021-01-01' AS x, n.id ORDER BY n.id")


def test_mixed_type_column_declines_honestly():
    """A heterogeneous (int+str) object column — legal for dynamically-typed Cypher
    properties in pandas, but unrepresentable in polars/Arrow — must raise a clear
    NotImplementedError (use engine='pandas'), NOT a cryptic pyarrow ArrowInvalid."""
    nodes = pd.DataFrame({"id": [0, 1, 2], "var": [0, "xx", None]})  # int + str + null
    edges = pd.DataFrame({"s": [0], "d": [1]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    with pytest.raises(NotImplementedError):
        g.gfql("MATCH (n) WHERE n.var > 'x' RETURN n.var", engine="polars")


def test_mixed_type_column_validate_autofix_coerces_to_string():
    """The mixed-type object column honors the repo-wide validate/warn convention.
    Default (strict) raises; validate='autofix' coerces the offending column to string
    and warns (validate=False coerces without warning) — matching the plot()/upload()
    and cuDF-conversion behavior instead of hardcoding one policy."""
    import warnings as _warnings
    from graphistry.Engine import Engine, df_to_engine
    pl = pytest.importorskip("polars")
    df = pd.DataFrame({"id": [0, 1, 2], "var": [0, "xx", None]})  # int + str + null

    # strict (the compute-path default) still declines
    with pytest.raises(NotImplementedError):
        df_to_engine(df, Engine.POLARS)

    # autofix coerces the mixed column to string and warns
    with pytest.warns(RuntimeWarning):
        out = df_to_engine(df, Engine.POLARS, validate="autofix")
    assert isinstance(out, pl.DataFrame)
    assert out.schema["var"] == pl.String

    # validate=False == autofix but suppresses the warning
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning becomes an error
        out2 = df_to_engine(df, Engine.POLARS, validate=False)
    assert out2.schema["var"] == pl.String


def test_polars_duplicate_alias_declines_like_pandas():
    """A chain reusing an alias name (``[n('a'), e(), n('a')]``) must raise the same
    GFQLValidationError E201 as pandas — NOT return a malformed colliding-join schema
    (``a``/``a_right``). NO-CHEATING: decline where the oracle declines."""
    from graphistry.compute.ast import n, e_forward
    from graphistry.compute.exceptions import GFQLValidationError
    g = graphistry.edges(pd.DataFrame({"s": [1, 2, 3], "d": [2, 3, 1]}), "s", "d").materialize_nodes()
    with pytest.raises(GFQLValidationError):
        g.chain([n(name="a"), e_forward(), n(name="a")], engine="pandas")
    with pytest.raises(GFQLValidationError):
        g.chain([n(name="a"), e_forward(), n(name="a")], engine="polars")


def test_polars_integer_literal_division_declines():
    """Cypher folds integer-literal division (``10/4 == 2``, truncating) but polars
    does true division (``2.5``) — a silent wrong answer when embedded in a non-monotonic
    op (``ORDER BY n.val % (10/4)`` sorts differently). Must decline (NIE). Column ``/``
    int stays Float on both engines, so it must NOT be over-declined."""
    g = graphistry.nodes(pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "val": [1, 2, 3, 4, 5, 6]}), "id") \
        .edges(pd.DataFrame({"s": [1], "d": [2]}), "s", "d")
    with pytest.raises(NotImplementedError):
        g.gfql("MATCH (n) RETURN n.val AS v ORDER BY n.val % (10/4)", engine="polars")
    # column / int-literal is true division on BOTH engines — must still compute natively
    _assert_parity(g, "MATCH (n) RETURN n.val / 2 AS h, n.id ORDER BY n.id")


def test_polars_chain_seed_dtype_alignment():
    """An internal ``start_nodes`` seed whose id-column dtype diverges from the node-id
    dtype (e.g. float seed vs int nodes — an empty crossfilter selection defaults to
    float64) must align join keys rather than crash with SchemaError (mirrors hop)."""
    import polars as pl
    from graphistry.compute.gfql.lazy.engine.polars.chain import chain_polars
    from graphistry.compute.ast import n, e_forward
    # polars frames (as the engine boundary hands chain_polars), int node ids
    nodes = pl.DataFrame({"id": [1, 2, 3]})
    edges = pl.DataFrame({"s": [1, 2, 3], "d": [2, 3, 1]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    seed = pl.DataFrame({"id": pl.Series([1.0, 2.0], dtype=pl.Float64)})  # float seed vs int nodes
    out = chain_polars(g, [n(), e_forward(), n()], start_nodes=seed)  # must not raise SchemaError
    out_edges = out._edges.to_pandas() if hasattr(out._edges, "to_pandas") else out._edges
    assert len(out_edges) >= 1


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


def test_native_polars_nan_input_treated_as_missing():
    """Review C1 regression: a NATIVE polars input carrying a real NaN must be treated as
    MISSING (the pandas oracle drops a NaN row under a gt/eq filter), not kept. The
    pandas->polars ingestion nan_to_null's; the native-polars path (skipped by
    _coerce_input_formats as 'already correct') did NOT, keeping the row = silent wrong answer."""
    import pandas as pd
    from graphistry.compute.ast import n
    from graphistry.compute.predicates.numeric import gt
    nodes_data = {"id": [0, 1, 2], "x": [10.0, float("nan"), 30.0]}
    edges_data = {"s": [0], "d": [1]}
    g_pd = graphistry.nodes(pd.DataFrame(nodes_data), "id").edges(pd.DataFrame(edges_data), "s", "d")
    g_pl = graphistry.nodes(pl.DataFrame(nodes_data), "id").edges(pl.DataFrame(edges_data), "s", "d")
    oracle = sorted(g_pd.gfql([n({"x": gt(5)})], engine="pandas")._nodes["id"].tolist())
    got = sorted(g_pl.gfql([n({"x": gt(5)})], engine="polars")._nodes["id"].to_list())
    assert got == oracle == [0, 2]


def test_in_query_nan_aggregation_matches_pandas_skipna():
    """Review I1 regression: an IN-QUERY NaN (0.0/0.0 in a WITH) then aggregated must match the
    pandas oracle's skipna/dropna. polars propagates NaN through sum/mean (and NaN==NaN is True,
    so it can't be detected by self-inequality); the agg lowering now nulls NaN in float columns."""
    import pandas as pd
    for eng_frame in (pd, pl):
        nodes = eng_frame.DataFrame({"id": [0, 1, 2], "a": [0.0, 2.0, 4.0]})
        edges = eng_frame.DataFrame({"s": [0], "d": [1]})
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
        for eng in (["pandas", "polars"] if eng_frame is pd else ["polars"]):
            r = g.gfql("MATCH (n) WITH n.a / n.a AS r RETURN sum(r) AS s", engine=eng)._nodes["s"]
            got = r.to_list()[0] if hasattr(r, "to_list") else r.tolist()[0]
            assert got == 2.0, (eng, got)  # 0/0=NaN dropped; 2/2 + 4/4 = 2.0


def test_bool_modulo_declines_like_pandas():
    """Review S2: pandas declines Boolean modulo (n.flag % 2 -> GFQLTypeError) while polars
    would compute it (bool->int). The polars engine now declines (NIE) to match — bool +,-,*,/
    compute identically on both, so only % diverges."""
    import pandas as pd
    g = graphistry.nodes(pd.DataFrame({"id": [0, 1, 2], "flag": [True, False, True]}), "id").edges(
        pd.DataFrame({"s": [0], "d": [1]}), "s", "d")
    with pytest.raises(NotImplementedError):
        g.gfql("MATCH (n) RETURN n.flag % 2 AS r", engine="polars")
    # bool + int still computes in parity (not over-declined)
    got = g.gfql("MATCH (n) RETURN n.flag + 2 AS r", engine="polars")._nodes["r"].to_list()
    assert got == [3, 2, 3]
