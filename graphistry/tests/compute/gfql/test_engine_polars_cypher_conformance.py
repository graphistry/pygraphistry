"""Differential cypher conformance (TCK-style): engine='polars' == engine='pandas' (the oracle)
on a curated corpus + seeded query fuzzer — identical result tables. Polars counterpart of the
cross-repo Cypher TCK harness (graphistry/tck-gfql); keeps the row pipeline honest across the
whole cypher surface, native and host-bridged alike. See plans/gfql-polars-engine."""
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
    """Dampen non-semantic numeric repr differences so the differential check
    tests semantics: round floats and render non-bool numeric columns as float64.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.astype("float64").round(6)
            continue
        if s.dtype == object:
            # pandas emits object columns of Python ints in some paths (e.g.
            # UNWIND literals); normalize those too — but only when every
            # non-null value coerces cleanly and none are bools.
            non_null = s.dropna()
            if len(non_null) and not non_null.map(lambda v: isinstance(v, bool)).any():
                coerced = pd.to_numeric(s, errors="coerce")
                if coerced.isna().sum() == s.isna().sum():
                    out[col] = coerced.astype("float64").round(6)
    return out


def _normalize_nulls(df):
    """Collapse pandas NaN/None and polars null to one sentinel: compare null SEMANTICS, not the
    engines' null repr (nan vs None), which astype(str) would render differently."""
    return df.where(df.notna(), "∅")


def _assert_parity(g, query):
    a = _to_pd(g.gfql(query, engine="pandas")._nodes).reset_index(drop=True)
    b = _to_pd(g.gfql(query, engine="polars")._nodes).reset_index(drop=True)
    assert list(a.columns) == list(b.columns), f"cols differ for {query!r}: {list(a.columns)} vs {list(b.columns)}"
    assert len(a) == len(b), f"row count differs for {query!r}: {len(a)} vs {len(b)}"
    if len(a) == 0:
        return
    # Bare LIMIT without ORDER BY picks an arbitrary k rows (cypher: order undefined) — engines
    # may legitimately differ, so only column shape + row count are conformant here
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
    # Kleene 3-valued booleans over bare null literals — must not crash on Null dtype (polars
    # &/|/~ need Boolean cast); bare RETURN keeps a single constant row on both engines
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
    # whole-entity returns FLATTEN to {alias}.{field} (#1650 structured returns), native for ANY
    # dtype incl float. Single-MATCH only: on (n)-[e]->(m) RETURN m pandas upcasts m.val
    # int->float in the binding merge (polars is more correct), so not a clean differential case.
    "MATCH (n) RETURN n",
    "MATCH (n) RETURN n LIMIT 5",
    "MATCH (n) RETURN DISTINCT n",
    # UNION / UNION ALL — the distinct de-dup must use the polars-native unique()
    # (regression: it called pandas drop_duplicates on a polars frame and crashed).
    "RETURN 1 AS x UNION RETURN 2 AS x",
    "RETURN 1 AS x UNION RETURN 1 AS x",
    "RETURN 1 AS x UNION ALL RETURN 1 AS x",
    "MATCH (n) WHERE n.kind = 'alpha' RETURN n.val UNION MATCH (n) WHERE n.kind = 'beta' RETURN n.val",
    # multi-entity property projection via native rows(binding_ops) (#1709)
    "MATCH (n)-[e]->(m) RETURN n.val, m.val",
]


@pytest.mark.parametrize("query", CORPUS)
def test_cypher_conformance_corpus(query):
    _assert_parity(BASE, query)


# NO-CHEATING (plan.md): no native polars impl yet -> NotImplementedError, NOT silently pandas.
DEFERRED = [
    # Whole-entity RETURN now FLATTENS (#1650) so float/whole-entity cases moved to CORPUS;
    # these remain deferred (honest NIE, no pandas bridge):
    "MATCH (n) RETURN n, n.val",                            # duplicate output col (polars .select rejects)
    # (multi-entity property bindings now native via rows(binding_ops) — #1709,
    #  moved to CORPUS below; cross-entity same-path WHERE remains deferred:)
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
    """Property projection over a temporal-constructor string column (how Cypher/TCK store temporal
    values) must NIE, not leak raw constructor text (pandas ISO-normalizes; not yet native)."""
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
    # polars->pandas renders the null as None or NaN depending on dtype / polars version
    # (1.40 NaN, newer None) — both are null, so assert is-null rather than `== [None]`
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
    """Review-found polars CRASH/panic/silent-misanswer cases — must raise honest NIE instead (NO-CHEATING)."""
    g = graphistry.nodes(nodes, "id").edges(pd.DataFrame({"s": [0], "d": [1]}), "s", "d")
    with pytest.raises(NotImplementedError):
        g.gfql(query, engine="polars")


def test_polars_string_column_vs_date_literal_computes():
    """A genuine String property vs a date-looking literal must COMPUTE (lexicographic, like
    pandas), not be over-declined by the ISO-temporal guard."""
    nodes = pd.DataFrame({"id": [0, 1], "w": ["2020-06-01", "2022-01-01"]})
    g = graphistry.nodes(nodes, "id").edges(pd.DataFrame({"s": [0], "d": [1]}), "s", "d")
    _assert_parity(g, "MATCH (n) RETURN n.w < '2021-01-01' AS x, n.id ORDER BY n.id")


def test_mixed_type_column_declines_honestly():
    """Heterogeneous (int+str) object column — legal in pandas Cypher, unrepresentable in
    polars/Arrow — must raise a clear NIE (use engine='pandas'), NOT a cryptic ArrowInvalid."""
    nodes = pd.DataFrame({"id": [0, 1, 2], "var": [0, "xx", None]})  # int + str + null
    edges = pd.DataFrame({"s": [0], "d": [1]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    with pytest.raises(NotImplementedError):
        g.gfql("MATCH (n) WHERE n.var > 'x' RETURN n.var", engine="polars")


def test_mixed_type_column_validate_autofix_coerces_to_string():
    """Mixed-type column honors the repo-wide validate/warn convention: strict default raises;
    validate='autofix' coerces to string + warns; validate=False coerces silently — matching
    plot()/upload() and cuDF-conversion behavior instead of hardcoding one policy."""
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
    """A chain reusing an alias must raise the same GFQLValidationError E201 as pandas — NOT a
    malformed colliding-join schema (a/a_right). NO-CHEATING: decline where the oracle does."""
    from graphistry.compute.ast import n, e_forward
    from graphistry.compute.exceptions import GFQLValidationError
    g = graphistry.edges(pd.DataFrame({"s": [1, 2, 3], "d": [2, 3, 1]}), "s", "d").materialize_nodes()
    with pytest.raises(GFQLValidationError):
        g.chain([n(name="a"), e_forward(), n(name="a")], engine="pandas")
    with pytest.raises(GFQLValidationError):
        g.chain([n(name="a"), e_forward(), n(name="a")], engine="polars")


def test_polars_integer_literal_division_declines():
    """Cypher truncates integer-literal division (10/4 == 2) but polars true-divides (2.5) — a
    silent wrong answer inside a non-monotonic op (ORDER BY n.val % (10/4)); must NIE. Column /
    int is Float on both engines, so it must NOT be over-declined."""
    g = graphistry.nodes(pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "val": [1, 2, 3, 4, 5, 6]}), "id") \
        .edges(pd.DataFrame({"s": [1], "d": [2]}), "s", "d")
    with pytest.raises(NotImplementedError):
        g.gfql("MATCH (n) RETURN n.val AS v ORDER BY n.val % (10/4)", engine="polars")
    # column / int-literal is true division on BOTH engines — must still compute natively
    _assert_parity(g, "MATCH (n) RETURN n.val / 2 AS h, n.id ORDER BY n.id")


def test_polars_chain_seed_dtype_alignment():
    """A start_nodes seed whose id dtype diverges from the node-id dtype (float seed vs int
    nodes — empty crossfilter selections default to float64) must align join keys rather than
    crash with SchemaError (mirrors hop)."""
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
    """Nulls in numeric/string/bool + zero/negative — the native lowering's NULL / cypher 3VL semantics vs pandas."""
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
    """int/string/bool only — native entity-text eligible, incl. quote/backslash escaping and null omission."""
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 3],
        "amount": [10, 20, 30, 40],
        "label": ["plain", "has'quote", "back\\slash", None],
        "active": [True, False, True, False],
    })
    edges = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3]})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def test_native_entity_text_parity():
    """RETURN n FLATTENS to a.* columns natively (#1650) == pandas, no bridge; legacy display
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
    """Review C1: a NATIVE polars input carrying a real NaN must be treated as MISSING (pandas drops
    the NaN row under gt/eq), not kept — pandas->polars ingestion nan_to_null's, but the native-polars
    path (skipped by _coerce_input_formats as 'already correct') did NOT: silent wrong answer."""
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
    """Review I1: an IN-QUERY NaN (0.0/0.0 in WITH) then aggregated must match pandas
    skipna/dropna. polars propagates NaN through sum/mean (and NaN==NaN is True, so no
    self-inequality detection); the agg lowering now nulls NaN in float columns."""
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
    """Review S2: pandas declines Boolean modulo (GFQLTypeError) while polars would compute it
    (bool->int) — polars now NIEs to match; bool +,-,*,/ agree, only % diverges."""
    import pandas as pd
    g = graphistry.nodes(pd.DataFrame({"id": [0, 1, 2], "flag": [True, False, True]}), "id").edges(
        pd.DataFrame({"s": [0], "d": [1]}), "s", "d")
    with pytest.raises(NotImplementedError):
        g.gfql("MATCH (n) RETURN n.flag % 2 AS r", engine="polars")
    # bool + int still computes in parity (not over-declined)
    got = g.gfql("MATCH (n) RETURN n.flag + 2 AS r", engine="polars")._nodes["r"].to_list()
    assert got == [3, 2, 3]


class TestAutoEngineRoutesPolarsNative:
    """engine=auto on polars-frame graphs must run the native polars path (frames in = frames
    out), not the silent pandas bridge (~13x on point queries); polars-NIE shapes still answer
    via the legacy AUTO fallback since the user did not pin an engine."""

    def _graph(self):
        nodes = pl.DataFrame({"id": [0, 1, 2, 3], "label__Person": [True] * 4})
        edges = pl.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3], "type": ["KNOWS"] * 3})
        return graphistry.nodes(nodes, "id").edges(edges, "s", "d")

    def test_auto_returns_polars_frames_and_matches_explicit(self):
        g = self._graph()
        q = "MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id AS bid ORDER BY bid"
        r_auto = g.gfql(q)._nodes
        r_expl = g.gfql(q, engine="polars")._nodes
        assert "polars" in type(r_auto).__module__, "auto must not bridge polars graphs to pandas"
        from polars.testing import assert_frame_equal
        assert_frame_equal(r_auto, r_expl)

    def test_auto_falls_back_on_polars_nie(self):
        g = self._graph()
        q = ("MATCH p = shortestPath((a:Person {id: 0})-[:KNOWS*]-(b:Person {id: 3})) "
             "RETURN length(p) AS l")
        with pytest.raises(NotImplementedError):
            g.gfql(q, engine="polars")  # pinned engine: honest NIE
        out = g.gfql(q)._nodes  # auto: answers via fallback
        rows = out.to_dict("records") if hasattr(out, "to_dict") else out.to_pandas().to_dict("records")
        assert rows == [{"l": 3}]

    def test_auto_routes_edges_only_graph(self):
        """``self._nodes is None`` is inside the guard's condition, so it must actually work."""
        edges = pl.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3], "type": ["KNOWS"] * 3})
        g = graphistry.edges(edges, "s", "d")
        out = g.gfql("MATCH (a)-[:KNOWS]->(b) RETURN b.id AS bid ORDER BY bid")._nodes
        assert "polars" in type(out).__module__
        assert out["bid"].to_list() == [1, 2, 3]

    def test_auto_enum_form_routes_too(self):
        """AUTO arrives as either the enum or its string value; both must route."""
        from graphistry.Engine import EngineAbstract
        g = self._graph()
        q = "MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id AS bid"
        for eng in (EngineAbstract.AUTO, EngineAbstract.AUTO.value):
            assert "polars" in type(g.gfql(q, engine=eng)._nodes).__module__


class TestAutoEngineRoutingBoundaries:
    """Where AUTO's polars routing starts and stops: pandas-frame graphs stay
    pandas, explicit engines are honored, mixed frames follow the edges frame,
    and a policy-carrying query keeps the hook-emitting path."""

    NODES = {"id": [0, 1, 2, 3], "label__Person": [True] * 4}
    EDGES = {"s": [0, 1, 2], "d": [1, 2, 3], "type": ["KNOWS"] * 3}
    Q = "MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id AS bid"

    def test_pandas_frames_unaffected(self):
        g = graphistry.nodes(pd.DataFrame(self.NODES), "id").edges(pd.DataFrame(self.EDGES), "s", "d")
        assert "pandas" in type(g.gfql(self.Q)._nodes).__module__

    def test_mixed_frames_follow_the_edges_frame(self):
        """Modern AUTO resolves mixed frames by the edges frame and coerces the
        nodes across. The invariant is the VALUE plus result-engine-follows-
        resolution ('must not try' pinned the absence of a bridge that now
        exists -- accident, not spec)."""
        g = (graphistry
             .nodes(pd.DataFrame(self.NODES), "id")
             .edges(pl.DataFrame(self.EDGES), "s", "d"))
        out = g.gfql(self.Q)
        assert "polars" in type(out._nodes).__module__
        assert out._nodes.to_pandas()["bid"].tolist() == [1]
        g2 = (graphistry
              .nodes(pl.DataFrame(self.NODES), "id")
              .edges(pd.DataFrame(self.EDGES), "s", "d"))
        assert "pandas" in type(g2.gfql(self.Q)._nodes).__module__

    def test_explicit_pandas_engine_still_pandas(self):
        g = graphistry.nodes(pl.DataFrame(self.NODES), "id").edges(pl.DataFrame(self.EDGES), "s", "d")
        assert "pandas" in type(g.gfql(self.Q, engine="pandas")._nodes).__module__

    def test_policy_disables_the_route(self):
        """The native executor does not go through ``chain_impl`` and so never emits
        ``postload``/``postchain``. Diverting a policy-carrying query would silently stop
        enforcing a DENYING postload policy -- measured, not theorised."""
        from graphistry.compute.gfql.policy import PolicyException
        g = graphistry.nodes(pl.DataFrame(self.NODES), "id").edges(pl.DataFrame(self.EDGES), "s", "d")

        def deny(ctx):
            raise PolicyException(phase="postload", reason="denied by test")

        with pytest.raises(PolicyException):
            g.gfql(self.Q, policy={"postload": deny})
        # ...and structurally: a policy-bearing AUTO query stays on the generic path,
        # which is what makes the hook trace identical to the pre-change build.
        assert "pandas" in type(g.gfql(self.Q, policy={"preload": lambda ctx: None})._nodes).__module__
        seen = []
        pol = {h: (lambda ctx, h=h: seen.append(h)) for h in ("preload", "postload", "precompile")}
        g.gfql(self.Q, policy=pol)
        assert "postload" in seen, "postload must still fire under AUTO on a polars graph"

    def test_policy_guard_covers_every_polars_resolution(self):
        """The guard's predicate must be resolve_engine itself: MIXED frames
        (polars edges + pandas nodes) also resolve to polars under AUTO, and a
        frame-shape guard once let them bypass a DENYING postload policy --
        governance silently off is the worst failure mode this surface has."""
        from graphistry.compute.gfql.policy import PolicyException

        def deny(ctx):
            raise PolicyException(phase="postload", reason="denied by test")

        mixed = (graphistry
                 .nodes(pd.DataFrame(self.NODES), "id")
                 .edges(pl.DataFrame(self.EDGES), "s", "d"))
        with pytest.raises(PolicyException):
            mixed.gfql(self.Q, policy={"postload": deny})

        edges_only = graphistry.edges(pl.DataFrame(self.EDGES), "s", "d")
        with pytest.raises(PolicyException):
            edges_only.gfql("MATCH (a)-[:KNOWS]->(b) RETURN b LIMIT 1",
                            policy={"postload": deny})

        nodes_only = graphistry.nodes(pl.DataFrame(self.NODES), "id")
        with pytest.raises(PolicyException):
            nodes_only.gfql("MATCH (a:Person) RETURN a.id LIMIT 1",
                            policy={"postload": deny})

    def test_policy_hooks_fire_once_on_nie_shape(self):
        """A retry-on-NIE route would fire the compile/load hooks twice for one user call."""
        g = graphistry.nodes(pl.DataFrame(self.NODES), "id").edges(pl.DataFrame(self.EDGES), "s", "d")
        q = ("MATCH p = shortestPath((a:Person {id: 0})-[:KNOWS*]-(b:Person {id: 3})) "
             "RETURN length(p) AS l")
        seen = []
        pol = {h: (lambda ctx, h=h: seen.append(h)) for h in ("preload", "precompile", "postcompile")}
        g.gfql(q, policy=pol)
        assert seen.count("precompile") == 1, seen
        assert seen.count("preload") == 1, seen


class TestAutoEngineLazyFrames:
    """LazyFrame is the other member of the PolarsFrame union `is_polars_df` admits, so
    the AUTO route must take it too — and hand back EAGER polars, same as explicit
    engine='polars' does."""

    def test_lazyframe_auto_routes_native_and_matches_explicit(self):
        nodes = pl.DataFrame({"id": [0, 1, 2, 3], "label__Person": [True] * 4}).lazy()
        edges = pl.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3], "type": ["KNOWS"] * 3}).lazy()
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
        q = "MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id AS bid ORDER BY bid"
        r_auto = g.gfql(q)._nodes
        r_expl = g.gfql(q, engine="polars")._nodes
        assert isinstance(r_auto, pl.DataFrame), type(r_auto)  # eager out, not Lazy
        assert isinstance(r_expl, pl.DataFrame), type(r_expl)
        from polars.testing import assert_frame_equal
        assert_frame_equal(r_auto, r_expl)
        assert r_auto["bid"].to_list() == [1]


def _cudf_graph():
    """Small cudf-frame graph; every AUTO-on-cudf test starts here."""
    cudf = pytest.importorskip("cudf")
    nodes = cudf.DataFrame({"id": [0, 1, 2, 3], "label__Person": [True] * 4})
    edges = cudf.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3], "type": ["KNOWS"] * 3})
    return cudf, graphistry.nodes(nodes, "id").edges(edges, "s", "d")


_CUDF_Q = "MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id AS bid"


class TestAutoEngineCudfUntouched:
    """cuDF frames never enter the polars-AUTO guard (`is_polars_df` is a polars module
    check). The cuDF arm of AUTO now PREFERS the polars-gpu route when the cudf-polars
    GPU target is genuinely usable (probe True) — but with the probe False (any box
    without a working cudf-polars stack, including this one) the legacy AUTO->CUDF
    resolution is byte-for-byte what it was, INCLUDING the output frame types:
    cudf frames in must mean cudf frames out. That held before the routing existed
    (the legacy path never converted) and must keep holding after. Skips without
    cudf/GPU; runs on GPU lanes."""

    def test_auto_on_cudf_frames_stays_on_legacy_cudf_path(self, monkeypatch):
        cudf, g = _cudf_graph()
        # pin probe False so this asserts the LEGACY path on GPU lanes too
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: False)
        out = g.gfql(_CUDF_Q)
        # cudf in -> cudf out, on BOTH bound frames — the frame-type contract of the
        # legacy path, pinned as types, not just "guard bypassed"
        assert isinstance(out._nodes, cudf.DataFrame), type(out._nodes)
        assert isinstance(out._edges, cudf.DataFrame), type(out._edges)
        assert out._nodes["bid"].to_pandas().tolist() == [1]

    def test_explicit_engine_cudf_is_cudf_in_cudf_out(self, monkeypatch):
        """Explicit engine= always wins: even with the probe forced True and the route
        armed to explode, engine='cudf' serves on the legacy path, cudf in -> cudf out."""
        cudf, g = _cudf_graph()
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        monkeypatch.setattr(
            "graphistry.compute.gfql_unified._auto_cudf_polars_gpu_route",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("explicit engine must bypass the route")),
        )
        out = g.gfql(_CUDF_Q, engine="cudf")
        assert isinstance(out._nodes, cudf.DataFrame), type(out._nodes)
        assert isinstance(out._edges, cudf.DataFrame), type(out._edges)
        assert out._nodes["bid"].to_pandas().tolist() == [1]


class TestPolarsGpuAvailabilityProbe:
    """lazy.polars_gpu_available: a REAL probe (imports + a genuine GPU collect), cached
    once per process and registered exempt in the GFQL cache registry. These tests are
    CPU-runnable anywhere: they pin the graceful-False paths and the memoization by
    forcing each failure leg, clearing the singleton around every observation."""

    @pytest.fixture(autouse=True)
    def _fresh_probe(self):
        from graphistry.compute.gfql.lazy import polars_gpu_available
        polars_gpu_available.cache_clear()
        yield
        polars_gpu_available.cache_clear()

    def test_probe_never_raises_and_returns_bool(self):
        from graphistry.compute.gfql.lazy import polars_gpu_available
        assert isinstance(polars_gpu_available(), bool)

    def test_probe_false_when_cudf_polars_missing(self, monkeypatch):
        """cudf installed, cudf_polars not: the second spec check must decline. cudf is
        STUBBED present (not looked up for real) so this leg is actually reached on the
        CPU lanes where cudf is absent — a real lookup would short-circuit at the cudf
        check and this test would silently re-pin the cudf-missing leg instead."""
        import importlib.util as ilu
        from graphistry.compute.gfql.lazy import polars_gpu_available
        real_find_spec = ilu.find_spec
        def fake_find_spec(name, *a, **k):
            if name == "cudf":
                return object()
            if name == "cudf_polars":
                return None
            return real_find_spec(name, *a, **k)
        monkeypatch.setattr(ilu, "find_spec", fake_find_spec)
        assert polars_gpu_available() is False

    def test_probe_false_when_spec_lookup_raises(self, monkeypatch):
        """Broken packaging metadata (find_spec itself raising, e.g. a half-uninstalled
        wheel) must probe False through the import-block except, never raise."""
        import importlib.util as ilu
        from graphistry.compute.gfql.lazy import polars_gpu_available
        def _broken(name, *a, **k):
            raise ImportError(f"broken package metadata for {name}")
        monkeypatch.setattr(ilu, "find_spec", _broken)
        assert polars_gpu_available() is False

    def test_probe_false_when_cudf_missing(self, monkeypatch):
        import importlib.util as ilu
        from graphistry.compute.gfql.lazy import polars_gpu_available
        real_find_spec = ilu.find_spec
        monkeypatch.setattr(
            ilu, "find_spec",
            lambda name, *a, **k: None if name == "cudf" else real_find_spec(name, *a, **k),
        )
        assert polars_gpu_available() is False

    def test_probe_false_when_gpu_collect_fails(self, monkeypatch):
        """Packages installed but the GPU genuinely unusable (the broken-libnvrtc class:
        imports succeed, kernels fail) must probe False, not raise."""
        import importlib.util as ilu
        from graphistry.compute.gfql import lazy as lazy_mod
        monkeypatch.setattr(ilu, "find_spec", lambda name, *a, **k: object())  # all "installed"
        def _boom(target):
            raise RuntimeError("libnvrtc.so: cannot open shared object file")
        monkeypatch.setattr(lazy_mod, "_engine_for", _boom)
        assert lazy_mod.polars_gpu_available() is False

    def test_probe_false_when_engine_builder_returns_none(self, monkeypatch):
        """The typing-honesty guard (`_engine_for` -> None) must decline, not collect on
        a None engine. Unreachable through the real `_engine_for` (GPU target always
        builds an engine), so it is pinned through the same seam the failure test uses."""
        import importlib.util as ilu
        from graphistry.compute.gfql import lazy as lazy_mod
        monkeypatch.setattr(ilu, "find_spec", lambda name, *a, **k: object())
        monkeypatch.setattr(lazy_mod, "_engine_for", lambda target: None)
        assert lazy_mod.polars_gpu_available() is False

    def test_probe_true_when_collect_succeeds(self, monkeypatch):
        """The success leg, CPU-runnable through the same `_engine_for` seam the failure
        test uses: hand the probe an engine spec that executes here ('in-memory' — a
        string polars accepts wherever GPUEngine is accepted), and its REAL collect plus
        the value check must run and return True. Only the GPU-ness of the engine object
        is stubbed; the collect itself is genuine."""
        import importlib.util as ilu
        from graphistry.compute.gfql import lazy as lazy_mod
        monkeypatch.setattr(ilu, "find_spec", lambda name, *a, **k: object())
        monkeypatch.setattr(lazy_mod, "_engine_for", lambda target: "in-memory")
        assert lazy_mod.polars_gpu_available() is True

    def test_probe_is_a_process_singleton(self, monkeypatch):
        """Second call must not re-probe: the availability of the GPU stack is a property
        of the process environment, probed once (lru_cache maxsize=1)."""
        import importlib.util as ilu
        from graphistry.compute.gfql.lazy import polars_gpu_available
        calls = []
        real_find_spec = ilu.find_spec
        def counting_find_spec(name, *a, **k):
            if name == "cudf":
                calls.append(name)
            return None if name == "cudf_polars" else real_find_spec(name, *a, **k)
        monkeypatch.setattr(ilu, "find_spec", counting_find_spec)
        assert polars_gpu_available() is False
        assert polars_gpu_available() is False
        assert len(calls) == 1, calls

    def test_probe_registered_exempt_in_cache_registry(self):
        import graphistry.compute.gfql.lazy  # noqa: F401  (registration happens at import)
        from graphistry.compute.gfql.cache_registry import entries
        entry = entries()["polars_gpu_available"]
        assert entry.clear is None, "must be an exempt process singleton, not clearable"
        assert entry.reason and len(entry.reason.split()) >= 6


class TestAutoEngineCudfPolarsGpuGuard:
    """The guard around the cuDF arm: WHO gets routed. All pins go through the two module
    seams (`_polars_gpu_probe`, `_auto_cudf_polars_gpu_route`) so they run on CPU boxes;
    the route's real execution is covered by the CPU seam class below and the DGX class."""

    def test_routed_when_probe_true(self, monkeypatch):
        _, g = _cudf_graph()
        sentinel = object()
        calls = []
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        monkeypatch.setattr(
            "graphistry.compute.gfql_unified._auto_cudf_polars_gpu_route",
            lambda *a, **k: (calls.append(a), sentinel)[1],
        )
        assert g.gfql(_CUDF_Q) is sentinel
        assert len(calls) == 1

    def test_edges_only_cudf_graph_is_routed(self, monkeypatch):
        cudf = pytest.importorskip("cudf")
        edges = cudf.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3], "type": ["KNOWS"] * 3})
        g = graphistry.edges(edges, "s", "d")
        sentinel = object()
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        monkeypatch.setattr(
            "graphistry.compute.gfql_unified._auto_cudf_polars_gpu_route",
            lambda *a, **k: sentinel,
        )
        assert g.gfql("MATCH (a)-[:KNOWS]->(b) RETURN b.id AS bid") is sentinel

    def test_not_routed_when_probe_false(self, monkeypatch):
        cudf, g = _cudf_graph()
        calls = []
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: False)
        monkeypatch.setattr(
            "graphistry.compute.gfql_unified._auto_cudf_polars_gpu_route",
            lambda *a, **k: calls.append(a),
        )
        out = g.gfql(_CUDF_Q)
        assert calls == []
        assert isinstance(out._nodes, cudf.DataFrame)

    def test_policy_bearing_query_not_routed(self, monkeypatch):
        """policy= present bypasses the route (the native executor's postload/postchain
        hook gap must not become the default), same as the polars arm."""
        cudf, g = _cudf_graph()
        calls = []
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        monkeypatch.setattr(
            "graphistry.compute.gfql_unified._auto_cudf_polars_gpu_route",
            lambda *a, **k: calls.append(a),
        )
        seen = []
        out = g.gfql(_CUDF_Q, policy={"preload": lambda ctx: seen.append("preload")})
        assert calls == []
        assert seen == ["preload"]
        assert isinstance(out._nodes, cudf.DataFrame)

    def test_mixed_cudf_pandas_frames_not_routed(self, monkeypatch):
        pytest.importorskip("cudf")
        cudf, g_all = _cudf_graph()
        g = g_all.nodes(pd.DataFrame({"id": [0, 1, 2, 3], "label__Person": [True] * 4}), "id")
        calls = []
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        monkeypatch.setattr(
            "graphistry.compute.gfql_unified._auto_cudf_polars_gpu_route",
            lambda *a, **k: calls.append(a),
        )
        try:
            g.gfql(_CUDF_Q)
        except Exception:
            # mixed-frame legacy execution may fail on limited-GPU boxes; the pin here
            # is only that the polars-gpu route was never consulted for mixed frames
            pass
        assert calls == []

    def test_all_polars_frames_win_over_cudf_arm(self, monkeypatch):
        """Guard ORDER: the all-polars arm fires first; the cudf arm must not even be
        consulted for a polars-frame graph."""
        g = graphistry.nodes(
            pl.DataFrame({"id": [0, 1, 2, 3], "label__Person": [True] * 4}), "id"
        ).edges(pl.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3], "type": ["KNOWS"] * 3}), "s", "d")
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        monkeypatch.setattr(
            "graphistry.compute.gfql_unified._auto_cudf_polars_gpu_route",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("cudf arm must not fire on polars frames")),
        )
        out = g.gfql(_CUDF_Q)
        assert isinstance(out._nodes, pl.DataFrame)

    def test_nie_from_route_falls_back_to_legacy_cudf_values(self, monkeypatch):
        """Decline shape: a NotImplementedError out of the route (engine decline, GPU
        collect failure via _gpu_raise, or the cudf-out boundary) re-serves the query on
        the legacy CUDF path with identical values."""
        cudf, g = _cudf_graph()
        expected = g.gfql(_CUDF_Q)._nodes.to_pandas()  # probe False here: legacy result
        calls = []
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        def declining_route(*a, **k):
            calls.append(a)
            raise NotImplementedError("declined for the test")
        monkeypatch.setattr(
            "graphistry.compute.gfql_unified._auto_cudf_polars_gpu_route", declining_route
        )
        out = g.gfql(_CUDF_Q)
        assert len(calls) == 1
        assert isinstance(out._nodes, cudf.DataFrame)
        pd.testing.assert_frame_equal(
            out._nodes.to_pandas().reset_index(drop=True), expected.reset_index(drop=True)
        )


class TestAutoEngineCudfPolarsGpuRouteCpuSeam:
    """The REAL route body, executable on CPU: `_AUTO_CUDF_ROUTE_ENGINE` swapped to
    'polars' runs guard -> recursion -> cudf->Arrow->polars coercion -> native engine ->
    polars->Arrow->cudf result boundary — everything except the GPU collect itself
    (DGX-deferred). Value tests only where actually executable, per the tiering."""

    def test_cudf_in_cudf_out_values_match_legacy(self, monkeypatch):
        cudf, g = _cudf_graph()
        expected = g.gfql(_CUDF_Q)._nodes.to_pandas()  # legacy path (probe False here)
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        monkeypatch.setattr("graphistry.compute.gfql_unified._AUTO_CUDF_ROUTE_ENGINE", "polars")
        out = g.gfql(_CUDF_Q)
        assert isinstance(out._nodes, cudf.DataFrame), type(out._nodes)
        assert isinstance(out._edges, cudf.DataFrame), type(out._edges)
        pd.testing.assert_frame_equal(
            out._nodes.to_pandas().reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_nullable_values_survive_both_arrow_boundaries(self, monkeypatch):
        """Nulls in a cudf int column must survive cudf->polars->cudf without the pandas
        detour's float64+NaN upcast."""
        cudf = pytest.importorskip("cudf")
        nodes = cudf.DataFrame({"id": [0, 1, 2, 3], "v": [10, None, 30, None]})
        edges = cudf.DataFrame({"s": [0, 1], "d": [1, 2]})
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        monkeypatch.setattr("graphistry.compute.gfql_unified._AUTO_CUDF_ROUTE_ENGINE", "polars")
        out = g.gfql("MATCH (n) RETURN n.id AS nid, n.v AS v ORDER BY nid")
        assert isinstance(out._nodes, cudf.DataFrame)
        got = out._nodes.to_pandas()
        assert got["nid"].tolist() == [0, 1, 2, 3]
        vals = got["v"].tolist()
        assert vals[0] == 10 and vals[2] == 30
        assert pd.isna(vals[1]) and pd.isna(vals[3])
        assert "int" in str(out._nodes["v"].dtype).lower(), out._nodes["v"].dtype

    def test_engine_decline_falls_back_to_legacy_through_real_route(self, monkeypatch):
        """An honest engine NIE (shortestPath has no native polars row op) travels the
        real route and lands on the legacy CUDF path — cudf out, correct value."""
        cudf, g = _cudf_graph()
        q = ("MATCH p = shortestPath((a:Person {id: 0})-[:KNOWS*]-(b:Person {id: 3})) "
             "RETURN length(p) AS l")
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        monkeypatch.setattr("graphistry.compute.gfql_unified._AUTO_CUDF_ROUTE_ENGINE", "polars")
        out = g.gfql(q)
        assert isinstance(out._nodes, cudf.DataFrame), type(out._nodes)
        assert out._nodes["l"].to_pandas().tolist() == [3]

    def test_row_pipeline_predicate_where_declines_to_legacy_cudf(self, monkeypatch):
        """DGX regression (#1743 full-container run): where_rows filter_dict PREDICATES
        (gt/lt/...) are not polars-lowerable — the kernel used to reach ``pl.col == GT(...)``
        and leak a raw polars TypeError, which is NOT NotImplementedError, so the guard's
        decline-and-fall-back contract broke and the query errored instead of re-serving on
        legacy cuDF. Pin the whole failing shape through the REAL route: decline -> legacy
        cudf -> cudf-out with legacy values (mirrors test_row_pipeline_ops.py::
        test_row_pipeline_cudf_where_unwind_group_by_when_available)."""
        cudf = pytest.importorskip("cudf")
        from graphistry.compute.ast import group_by, order_by, rows, unwind, where_rows
        from graphistry.compute.predicates.numeric import gt
        nodes = cudf.DataFrame({
            "id": ["a", "b", "c"],
            "grp": ["x", "x", "y"],
            "vals": [[1, 2], [3], [4, 5]],
            "score": [1, 2, 5],
        })
        edges = cudf.DataFrame({"s": ["a"], "d": ["b"]})
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        monkeypatch.setattr("graphistry.compute.gfql_unified._AUTO_CUDF_ROUTE_ENGINE", "polars")
        out = g.gfql([
            rows(),
            where_rows({"score": gt(1)}),
            unwind("vals", as_="v"),
            group_by(["grp"], [("cnt", "count"), ("sum_v", "sum", "v")]),
            order_by([("grp", "asc")]),
        ])
        assert isinstance(out._nodes, cudf.DataFrame), type(out._nodes)
        assert out._nodes.to_arrow().to_pylist() == [
            {"grp": "x", "cnt": 1, "sum_v": 3},
            {"grp": "y", "cnt": 2, "sum_v": 9},
        ]

    def test_row_pipeline_list_text_order_by_declines_to_legacy_cudf(self, monkeypatch):
        """DGX regression (#1743 full-container run): order_by over stringified-list text —
        the legacy row pipeline PARSES ``"[...]"`` columns and sorts with Cypher
        list-orderability, while a plain polars sort is lexicographic, so the routed result
        silently DIVERGED in values (order_by host-bridge + nested-map parity failures).
        The kernel now sniffs list-like sort keys and declines -> legacy cudf serves with
        legacy values (mirrors test_row_pipeline_ops.py::
        test_row_pipeline_order_by_stringified_list_column_on_cudf_when_available)."""
        cudf = pytest.importorskip("cudf")
        from graphistry.compute.ast import limit, order_by, rows, select
        nodes = cudf.DataFrame({
            "id": ["a", "b", "c", "d", "e"],
            "list": ["[2, -2]", "[1, 2]", "[300, 0]", "[1, -20]", "[2, -2, 100]"],
        })
        edges = cudf.DataFrame({"s": ["a"], "d": ["b"]})
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: True)
        monkeypatch.setattr("graphistry.compute.gfql_unified._AUTO_CUDF_ROUTE_ENGINE", "polars")
        out = g.gfql([rows(), order_by([("list", "asc")]), limit(3), select([("id", "id")])])
        assert isinstance(out._nodes, cudf.DataFrame), type(out._nodes)
        # legacy semantic list order ([1,-20] < [1,2] < [2,-2]); lexicographic would be d,b,e
        assert out._nodes.to_arrow().to_pylist() == [{"id": "d"}, {"id": "b"}, {"id": "a"}]

    def test_route_result_boundary_failure_is_nie(self, monkeypatch):
        """_route_result_frames_to_cudf wraps conversion failures as NotImplementedError
        so the guard's decline path catches them (never a leaked raw error)."""
        pytest.importorskip("cudf")
        from graphistry.compute import gfql_unified as gu
        res = graphistry.nodes(pl.DataFrame({"id": [1]}), "id").edges(
            pl.DataFrame({"s": [1], "d": [1]}), "s", "d"
        )
        def _explode(df, engine):
            raise RuntimeError("no arrow for you")
        monkeypatch.setattr("graphistry.compute.gfql_unified.df_to_engine", _explode)
        with pytest.raises(NotImplementedError, match="could not cross back to cudf"):
            gu._route_result_frames_to_cudf(res)


def _polars_gpu_genuinely_available() -> bool:
    try:
        from graphistry.compute.gfql.lazy import polars_gpu_available
        return polars_gpu_available()
    except Exception:
        return False


class TestAutoEngineCudfPolarsGpuExecutionDGX:
    """DGX-DEFERRED tier: the real cudf-polars GPU collect. Runs only where the probe is
    genuinely True (working RAPIDS cudf_polars stack); everywhere else these skip — same
    convention as test_engine_polars_gpu.py. Owner has explicitly deferred running these
    (GPU box under a locked benchmark)."""

    pytestmark = pytest.mark.skipif(
        not _polars_gpu_genuinely_available(),
        reason="requires a genuinely usable cudf-polars GPU stack (DGX-deferred)",
    )

    def test_auto_on_cudf_routes_polars_gpu_and_returns_cudf(self, monkeypatch):
        cudf, g = _cudf_graph()
        from graphistry.compute import gfql_unified as gu
        calls = []
        real_route = gu._auto_cudf_polars_gpu_route
        def spying_route(*a, **k):
            calls.append(1)
            return real_route(*a, **k)
        monkeypatch.setattr("graphistry.compute.gfql_unified._auto_cudf_polars_gpu_route", spying_route)
        out = g.gfql(_CUDF_Q)
        assert calls, "probe True but the route did not engage"
        assert isinstance(out._nodes, cudf.DataFrame), type(out._nodes)
        assert isinstance(out._edges, cudf.DataFrame), type(out._edges)
        assert out._nodes["bid"].to_pandas().tolist() == [1]

    def test_gpu_route_values_match_legacy_cudf_path(self, monkeypatch):
        cudf, g = _cudf_graph()
        out_routed = g.gfql(_CUDF_Q)._nodes.to_pandas()
        monkeypatch.setattr("graphistry.compute.gfql_unified._polars_gpu_probe", lambda: False)
        out_legacy = g.gfql(_CUDF_Q)._nodes.to_pandas()
        pd.testing.assert_frame_equal(
            out_routed.reset_index(drop=True), out_legacy.reset_index(drop=True),
            check_dtype=False,
        )

    def test_explicit_polars_gpu_still_returns_polars(self):
        """Explicit engine='polars-gpu' is NOT the AUTO route: no cudf-out boundary."""
        _, g = _cudf_graph()
        out = g.gfql(_CUDF_Q, engine="polars-gpu")
        assert isinstance(out._nodes, pl.DataFrame), type(out._nodes)


class TestAutoEnginePandasOracleParity:
    """Routed AUTO answers must equal the pandas oracle in VALUE, not just shape: the
    same query on pandas frames via engine='pandas' is the reference for each routed
    query shape (filter+traverse, aggregate, WHERE-bearing)."""

    ROUTED_SHAPES = [
        # filter + traverse (connected join, multi-alias projection)
        "MATCH (a {kind: 'alpha'})-[]->(b) RETURN a.id AS aid, b.id AS bid ORDER BY aid, bid",
        # aggregate
        "MATCH (n) RETURN n.kind AS kind, count(n) AS c ORDER BY kind",
        # WHERE-bearing (predicate + arithmetic projection)
        "MATCH (a)-[]->(b) WHERE b.val > 50 AND b.flag "
        "RETURN b.id AS bid, b.val + b.score AS v ORDER BY bid, v",
    ]

    @pytest.mark.parametrize("query", ROUTED_SHAPES)
    def test_auto_polars_matches_pandas_oracle(self, query):
        g_pl = graphistry.nodes(pl.from_pandas(BASE._nodes), "id").edges(
            pl.from_pandas(BASE._edges), "s", "d"
        )
        got = g_pl.gfql(query)._nodes  # AUTO: routes native polars
        assert "polars" in type(got).__module__, "shape did not route; oracle test is vacuous"
        oracle = BASE.gfql(query, engine="pandas")._nodes
        a = _normalize_nulls(_round_floats(_to_pd(got).reset_index(drop=True)))
        b = _normalize_nulls(_round_floats(oracle.reset_index(drop=True)))
        assert list(a.columns) == list(b.columns), (list(a.columns), list(b.columns))
        pd.testing.assert_frame_equal(a.astype(str), b.astype(str), check_dtype=False)
