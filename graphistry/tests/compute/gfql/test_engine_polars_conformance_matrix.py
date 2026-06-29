"""Generative differential-conformance matrix for the polars engine (Phase 0 strategy).

THE CORE INVARIANT: for any query, on a non-pandas engine the result is EITHER parity-equal to
the pandas oracle OR an honest NotImplementedError — NEVER silently different, NEVER a silent
bridge, NEVER a non-NIE crash. This is checked across the cross-product of SURFACES
(native chain / Cypher string / let() DAG / call() on both) × OPS/PREDICATES, which is exactly
what the prior chain-only gates missed (the DAG silent-bridge bug et al.).

CPU lane (pandas-vs-polars) runs everywhere; a GPU lane (cudf / polars-gpu) is added on the dgx.
"""
import datetime
import numpy as np
import pandas as pd
import pytest
import graphistry
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected, call, let, rows, with_, return_


# ---- graph with diverse dtypes (int, float w/ null, str, bool) ----
def _graph(seed=0):
    rng = np.random.default_rng(seed)
    nn = 30
    nd = pd.DataFrame({
        "id": np.arange(nn),
        "num": rng.integers(0, 100, nn).astype("int64"),
        "f": rng.normal(0, 1, nn),
        "name": [f"node.{i % 7}" for i in range(nn)],  # has a literal '.' (Contains-regex trap)
        "flag": rng.integers(0, 2, nn).astype(bool),
    })
    ne = 80
    ed = pd.DataFrame({
        "s": rng.integers(0, nn, ne),
        "d": rng.integers(0, nn, ne),
        "eid": np.arange(ne),
        "w": rng.integers(0, 50, ne),
    })
    return graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")


def _to_pd(df):
    """Normalize any engine frame (polars / cudf / pandas) to pandas for comparison."""
    if df is None:
        return None
    if "pandas" in type(df).__module__:
        return df
    if hasattr(df, "to_pandas"):  # polars.DataFrame, cudf.DataFrame
        return df.to_pandas()
    return df


def _frame_repr(df):
    """Canonical VALUE-level repr of a frame for cross-engine comparison: normalize to pandas,
    sort columns, round floats (FP tolerance across engines), sort rows (order-insensitive),
    NaN/NA -> None. Compares actual cell values, not just id-sets/shape."""
    df = _to_pd(df)
    if df is None:
        return None
    import numpy as np
    df = df.reindex(sorted(df.columns), axis=1).copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(6)
    cols = tuple(df.columns)
    # rows as tuples (NaN/NA -> None), then sort the LIST of tuples (order-insensitive) with a
    # None-safe, type-safe key. Avoids per-row agg(join) which is fragile on empty/mixed frames.
    obj = df.where(df.notna(), None)
    rows = [tuple(r) for r in obj.to_numpy().tolist()]
    rows.sort(key=lambda t: tuple((v is None, type(v).__name__, str(v)) for v in t))
    return (cols, tuple(rows))


def _sig(g):
    """Full value-level signature of a graph/row result (both frames, values compared)."""
    return (_frame_repr(g._nodes), _frame_repr(g._edges))


def _run(g, query, engine):
    """('ok', sig) | ('nie',) | ('err', ExcTypeName)."""
    try:
        return ("ok", _sig(g.gfql(query, engine=engine)))
    except NotImplementedError:
        return ("nie",)
    except Exception as ex:  # any non-NIE error is itself a conformance failure to surface
        return ("err", type(ex).__name__)


def _available_nonpandas_engines():
    """polars always; cudf / polars-gpu when the GPU stack is importable (the dgx GPU lane)."""
    engines = ["polars"]
    try:
        import cudf  # noqa: F401
        engines.append("cudf")
    except Exception:
        pass
    import importlib.util
    if importlib.util.find_spec("cudf_polars") is not None:
        engines.append("polars-gpu")
    return engines


_NONPANDAS_ENGINES = _available_nonpandas_engines()


def _assert_invariant(g, query, label):
    """For EVERY available non-pandas engine: result == pandas oracle, OR honest NIE. Never a
    silent divergence / non-NIE crash. Runs polars on every box; cudf + polars-gpu on the dgx."""
    base = _run(g, query, "pandas")
    if base[0] == "err":
        pytest.skip(f"{label}: pandas oracle itself errored ({base[1]})")
    for eng in _NONPANDAS_ENGINES:
        res = _run(g, query, eng)
        if res[0] == "nie":
            continue  # honest decline — allowed
        assert res[0] != "err", f"{label}[{eng}]: non-NIE {res[1]} where pandas={base[0]}"
        assert res == base, f"{label}[{eng}]: SILENT DIVERGENCE {eng}{res} != pandas{base}"


# ---- predicate cross-product (the area that had 2 wrong-answer bugs) ----
def _predicate_queries():
    from graphistry.compute.predicates.numeric import GT, LT, GE, LE, Between
    from graphistry.compute.predicates.is_in import IsIn
    from graphistry.compute.predicates.str import Contains, Startswith, Endswith, Match, Fullmatch
    from graphistry.compute.predicates.numeric import IsNA, NotNA
    out = []
    for P, col, kw in [
        (GT, "num", {"val": 50}), (LT, "num", {"val": 50}), (GE, "num", {"val": 50}),
        (LE, "num", {"val": 50}),
        (Between, "num", {"lower": 20, "upper": 80}),
        (IsIn, "num", {"options": [1, 2, 3, 50, 51, 52]}),
        (Contains, "name", {"pat": "e.1", "regex": False}),   # literal metachar trap
        (Contains, "name", {"pat": "e.1", "regex": True}),
        (Contains, "name", {"pat": "ODE", "regex": False, "case": False}),
        (Startswith, "name", {"pat": "node"}),
        (Endswith, "name", {"pat": ".3"}),
        (Startswith, "name", {"pat": ("node.1", "node.2")}),  # tuple-of-prefixes (OR)
        (Endswith, "name", {"pat": (".1", ".2")}),            # tuple-of-suffixes (OR)
        (Match, "name", {"pat": "node"}),            # anchored start
        (Match, "name", {"pat": "ode"}),             # NOT at start -> no match (parity check)
        (Fullmatch, "name", {"pat": r"node\.\d"}),   # full match 'node.X'
        (IsNA, "name", {}), (NotNA, "name", {}),
    ]:
        out.append((f"pred:{P.__name__}({col},{kw})", [n({col: P(**kw)})]))
    return out


@pytest.mark.parametrize("label,query", _predicate_queries())
def test_conformance_predicates_chain(label, query):
    _assert_invariant(_graph(1), query, f"chain {label}")


@pytest.mark.parametrize("label,query", _predicate_queries())
def test_conformance_predicates_dag(label, query):
    # SAME predicate via a let() DAG binding — must agree with the chain surface (parity or NIE).
    g = _graph(1)
    _assert_invariant(g, let({"a": query}), f"dag {label}")


# ---- traversal cross-product (single-hop parity; multi-hop/undirected-multi-edge NIE) ----
@pytest.mark.parametrize("label,query", [
    ("fwd1", [n({"id": [0]}), e_forward()]),
    ("rev1", [n({"id": [0]}), e_reverse()]),
    ("und1", [n({"id": [0]}), e_undirected()]),
    ("n-e-n", [n(), e_forward(), n({"num": 50})]),
    ("fwd-fwd", [n({"id": [0]}), e_forward(), e_forward()]),
    ("multihop", [n({"id": [0]}), e_forward(hops=2)]),                 # NIE expected
    ("und-multi", [n({"id": [0]}), e_undirected(), e_undirected()]),   # NIE expected
])
def test_conformance_traversals(label, query):
    _assert_invariant(_graph(2), query, f"traversal {label}")


# ---- HOT-PATH CARVE-OUT verification (fast paths bypass the general engine = highest risk) ----
# Adversarial inputs against each special-cased fast path: node-only MATCH, unconstrained 1-hop,
# filtered single-hop, filters on BOTH endpoints, empty results, isolated/self-loop topology.
def _carveout_graph():
    nd = pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 5],
        "k": ["a", "a", "b", "b", "c", "c"],
        "v": [10, 20, 30, 40, 50, 60],
    })
    ed = pd.DataFrame({  # node 5 isolated; 2->2 self-loop; multi-edge 0->1
        "s": [0, 0, 1, 2, 3],
        "d": [1, 1, 2, 2, 4],
        "eid": [0, 1, 2, 3, 4],
    })
    return graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")


@pytest.mark.parametrize("label,query", [
    ("node-only-match", [n({"k": "a"})]),
    ("node-only-empty", [n({"k": "zzz"})]),
    ("unconstrained-1hop", [n(), e_forward(), n()]),
    ("filtered-src", [n({"k": "a"}), e_forward(), n()]),
    ("filtered-dst", [n(), e_forward(), n({"k": "b"})]),
    ("filtered-both-endpoints", [n({"k": "a"}), e_forward(), n({"k": "a"})]),
    ("both-endpoints-empty", [n({"k": "a"}), e_forward(), n({"k": "zzz"})]),
    ("self-loop-reach", [n({"id": [2]}), e_forward(), n()]),
    ("isolated-node-seed", [n({"id": [5]}), e_forward(), n()]),
    ("reverse-filtered", [n({"k": "b"}), e_reverse(), n()]),
    ("undirected-1hop-filtered", [n({"k": "a"}), e_undirected(), n({"k": "b"})]),
])
def test_conformance_hotpath_carveouts(label, query):
    _assert_invariant(_carveout_graph(), query, f"carveout {label}")


# ---- cypher expression / aggregation lowerings (value-level: validates sqrt/sign/count_distinct) ----
@pytest.mark.parametrize("label,query", [
    ("sqrt", "MATCH (n) RETURN n.id AS id, sqrt(n.num) AS sq"),
    ("sign", "MATCH (n) RETURN n.id AS id, sign(n.num - 50) AS sg"),
    ("abs", "MATCH (n) RETURN n.id AS id, abs(n.num - 50) AS ab"),
    ("coalesce", "MATCH (n) RETURN n.id AS id, coalesce(n.num, 0) AS c"),
    ("casewhen", "MATCH (n) RETURN n.id AS id, CASE WHEN n.num > 50 THEN 1 ELSE 0 END AS cw"),
    ("casewhen_str", "MATCH (n) RETURN n.id AS id, CASE WHEN n.flag THEN 'y' ELSE 'n' END AS cw"),
    ("count_distinct_grouped", "MATCH (n) RETURN n.flag AS k, count(DISTINCT n.num) AS cd"),
    ("count_distinct_all", "MATCH (n) RETURN count(DISTINCT n.flag) AS cd"),
    ("count_grouped", "MATCH (n) RETURN n.flag AS k, count(n.num) AS c"),
])
def test_conformance_cypher_expressions(label, query):
    _assert_invariant(_graph(4), query, f"cypher {label}")


# ---- cross-surface call() consistency (the silent-bridge bug class) ----
@pytest.mark.parametrize("fn", ["get_degrees", "hypergraph", "limit"])
def test_conformance_call_chain_vs_dag_consistent(fn):
    """A call must behave the SAME (parity or NIE) on the chain and the DAG surfaces — no surface
    may silently bridge where the other declines."""
    g = _graph(3)
    params = {"value": 2} if fn == "limit" else {}
    chain_q = [call(fn, params)] if params else [call(fn)]
    chain = _run(g, chain_q, "polars")
    dag = _run(g, let({"a": (call(fn, params) if params else call(fn))}), "polars")
    # Both honest-NIE, or both ok-with-equal-sig. Never one ok and the other NIE (the bug).
    if chain[0] == "nie":
        assert dag[0] == "nie", f"call '{fn}': chain NIE but DAG {dag} (silent-bridge regression)"
    elif chain[0] == "ok":
        assert dag[0] == "ok" and dag[1] == chain[1], f"call '{fn}': chain ok but DAG {dag}"


# ---- generative predicate fuzz across surfaces (verify-not-trust: broad, seeded) ----
def test_conformance_predicate_fuzz():
    from graphistry.compute.predicates.numeric import GT, Between
    from graphistry.compute.predicates.is_in import IsIn
    from graphistry.compute.predicates.str import Contains, Startswith, Endswith
    fails = []
    for t in range(60):
        rng = np.random.default_rng(7000 + t)
        g = _graph(7000 + t)
        choice = int(rng.integers(0, 6))
        if choice == 0:
            q = [n({"num": GT(val=int(rng.integers(0, 100)))})]
        elif choice == 1:
            lo = int(rng.integers(0, 50))
            q = [n({"num": Between(lower=lo, upper=lo + 30)})]
        elif choice == 2:
            q = [n({"num": IsIn(options=[int(x) for x in rng.integers(0, 100, 5)])})]
        elif choice == 3:
            q = [n({"name": Contains(pat="e.%d" % int(rng.integers(0, 7)),
                                     regex=bool(rng.integers(0, 2)))})]
        elif choice == 4:
            q = [n({"name": Startswith(pat="node")})]
        else:
            q = [n({"name": Endswith(pat=".%d" % int(rng.integers(0, 7)))})]
        try:
            _assert_invariant(g, q, f"fuzz {t}")
        except AssertionError as e:
            fails.append(str(e)[:120])
    assert not fails, "predicate-conformance fuzz failures:\n" + "\n".join(fails)


# ============================================================================
# Phase 2d native-feature wins (session 2): get_degrees / temporal / with_extend
# Each is native-or-honest; the parity-or-NIE invariant + explicit "must run
# NATIVELY" assertions gate them across all available engines.
# ============================================================================

# ---- NATIVE get_degrees (pure groupby/count — a real native polars win) ----
def _degrees_graph():
    # node 6 isolated; node 5 src-only; node 4 dst-only; 2->2 self-loop (double-counted)
    nd = pd.DataFrame({"id": [0, 1, 2, 3, 4, 5, 6], "k": ["a", "b", "c", "d", "e", "f", "g"]})
    ed = pd.DataFrame({
        "s":   [0, 1, 2, 5, 3],
        "d":   [1, 2, 2, 0, 4],
        "eid": [0, 1, 2, 3, 4],
    })
    return graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")


@pytest.mark.parametrize("label,query", [
    ("default", [call("get_degrees")]),
    ("custom-cols", [call("get_degrees", {"col": "deg", "degree_in": "din", "degree_out": "dout"})]),
])
def test_conformance_get_degrees_chain(label, query):
    _assert_invariant(_degrees_graph(), query, f"get_degrees chain {label}")


@pytest.mark.parametrize("label,binding", [
    ("default", call("get_degrees")),
    ("custom-cols", call("get_degrees", {"col": "deg", "degree_in": "din", "degree_out": "dout"})),
])
def test_conformance_get_degrees_dag(label, binding):
    _assert_invariant(_degrees_graph(), let({"a": binding}), f"get_degrees dag {label}")


def test_get_degrees_runs_natively_on_polars():
    """get_degrees is pure groupby/count -> it MUST run NATIVELY under polars: NOT an
    (honest-but-lazy) NotImplementedError and NOT a silent pandas bridge."""
    if "polars" not in _NONPANDAS_ENGINES:
        pytest.skip("polars not installed")
    g = _degrees_graph()
    base = _run(g, [call("get_degrees")], "pandas")
    res = _run(g, [call("get_degrees")], "polars")
    assert res[0] == "ok", f"get_degrees must be NATIVE on polars, got {res}"
    assert res == base, f"get_degrees polars must match pandas oracle: {res} != {base}"
    out = g.gfql([call("get_degrees")], engine="polars")
    assert "polars" in type(out._nodes).__module__, "get_degrees polars returned non-polars nodes (silent bridge!)"


def test_get_degrees_chain_vs_dag_native_consistent():
    """Cross-surface: chain call() vs let()/ref() DAG must agree — both native-ok with the
    SAME value signature (or both NIE)."""
    g = _degrees_graph()
    chain = _run(g, [call("get_degrees")], "polars")
    dag = _run(g, let({"a": call("get_degrees")}), "polars")
    assert chain[0] == dag[0], f"surface divergence: chain {chain[0]} != dag {dag[0]}"
    if chain[0] == "ok":
        assert chain[1] == dag[1], f"chain/dag sig mismatch: {chain} vs {dag}"


# ---- NATIVE TEMPORAL: SAFE DateValue lowering vs declined tz-aware ----
def _temporal_graph():
    """Graph with a NAIVE datetime64[ns] node column -> pl.Datetime(time_zone=None)."""
    nd = pd.DataFrame({
        "id": np.arange(6),
        "ts": pd.to_datetime([
            "2020-01-01 09:30", "2020-01-15 00:00", "2020-01-15 23:59",
            "2020-02-01 12:00", "2020-03-10 06:00", "2020-12-31 23:59",
        ]),
        "v": [10, 20, 30, 40, 50, 60],
    })
    ed = pd.DataFrame({"s": [0, 1, 2, 3], "d": [1, 2, 3, 4], "eid": [0, 1, 2, 3]})
    return graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")


def _temporal_chain_queries():
    from graphistry.compute.predicates.comparison import GT, GE, EQ, LT
    d = datetime.date(2020, 1, 15)
    return [
        ("gt-date",      [n({"ts": GT(val=d)})]),
        ("ge-date",      [n({"ts": GE(val=d)})]),
        # date truncation: BOTH 2020-01-15 rows (00:00 and 23:59) match the date literal
        ("eq-date",      [n({"ts": EQ(val=d)})]),
        ("lt-date",      [n({"ts": LT(val=d)})]),
        # no-match edge: nothing is after 2099 -> empty result, must still parity/NIE-agree
        ("nomatch-date", [n({"ts": GT(val=datetime.date(2099, 1, 1))})]),
    ]


@pytest.mark.parametrize("label,query", _temporal_chain_queries())
def test_conformance_temporal_datevalue_chain(label, query):
    _assert_invariant(_temporal_graph(), query, f"temporal-chain {label}")


@pytest.mark.parametrize("label,query", [
    ("cy-gt-date", "MATCH (n) WHERE n.ts > date('2020-01-15') RETURN n.id AS id"),
    ("cy-ge-date", "MATCH (n) WHERE n.ts >= date('2020-01-15') RETURN n.id AS id"),
    ("cy-eq-date", "MATCH (n) WHERE n.ts = date('2020-01-15') RETURN n.id AS id"),
])
def test_conformance_temporal_datevalue_cypher(label, query):
    # Cypher date('...') folds to a DateValue in the node filter_dict -> same native lowering.
    _assert_invariant(_temporal_graph(), query, f"temporal-cypher {label}")


def test_temporal_naive_datevalue_runs_natively_polars():
    """The SAFE shape (naive Datetime column + DateValue) MUST lower NATIVELY on polars
    (result 'ok', not an NIE decline) AND match the pandas oracle."""
    from graphistry.compute.predicates.comparison import GT
    g = _temporal_graph()
    q = [n({"ts": GT(val=datetime.date(2020, 1, 15))})]
    res = _run(g, q, "polars")
    assert res[0] == "ok", f"expected native polars lowering (not NIE), got {res}"
    _assert_invariant(g, q, "temporal native gt-date")


def test_temporal_tzaware_datetimevalue_honest_nie_polars():
    """A tz-tagged DateTimeValue (python datetime -> DateTimeValue, default tz=UTC) must stay an
    HONEST NotImplementedError on polars — never a silent tz mishandling, never a non-NIE crash."""
    from graphistry.compute.predicates.comparison import GT
    g = _temporal_graph()
    res = _run(g, [n({"ts": GT(val=datetime.datetime(2020, 1, 15, 12, 0))})], "polars")
    assert res[0] == "nie", f"expected honest NIE for tz-aware DateTimeValue, got {res}"


# ---- WITH ... extend=True (native polars `with_columns` column-extension) ----
def test_conformance_with_extend_arithmetic_native():
    """extend=True adds a computed column while KEEPING existing node columns, and MUST
    run natively on polars (with_columns) — not an honest NIE."""
    g = _graph(5)
    query = [n(), rows(), with_([("p", "num + 1")], extend=True)]
    _assert_invariant(g, query, "with_extend arithmetic")
    res = _run(g, query, "polars")
    assert res[0] == "ok", f"with_(extend=True) arithmetic must run NATIVELY on polars, got {res}"


def test_conformance_with_extend_shadows_existing_column():
    """A projected alias that SHADOWS an existing column overwrites it in place."""
    g = _graph(6)
    query = [n(), rows(), with_([("num", "num * 2"), ("extra", "f + 1.0")], extend=True)]
    _assert_invariant(g, query, "with_extend shadow")
    res = _run(g, query, "polars")
    assert res[0] == "ok", f"with_(extend=True) shadow must run NATIVELY on polars, got {res}"


def test_conformance_with_extend_chain_vs_cypher_consistent():
    """Cross-surface: chain `with_(extend=True)` and Cypher `WITH n, ...` must AGREE —
    both native+equal, or both honest-NIE (never the silent-bridge bug class)."""
    g = _graph(7)
    chain_q = [
        n(), rows(),
        with_([("id", "id"), ("num", "num"), ("p", "num + 1")], extend=True),
        return_([("id", "id"), ("num", "num"), ("p", "p")]),
    ]
    cypher_q = "MATCH (n) WITH n, n.num + 1 AS p RETURN n.id AS id, n.num AS num, p AS p"
    _assert_invariant(g, chain_q, "with_extend chain")
    _assert_invariant(g, cypher_q, "with_extend cypher")
    chain = _run(g, chain_q, "polars")
    cyph = _run(g, cypher_q, "polars")
    if chain[0] == "nie":
        assert cyph[0] == "nie", f"chain NIE but cypher {cyph} (cross-surface divergence)"
    elif chain[0] == "ok":
        assert cyph[0] == "ok" and cyph[1] == chain[1], f"chain ok but cypher {cyph}"


def test_conformance_with_extend_unlowerable_is_honest_nie():
    """An item lower_expr cannot lower (a list literal) must decline as an honest NIE on
    polars (NO silent pandas bridge), while the pandas oracle succeeds. Direct pandas-ok +
    polars-nie check rather than _assert_invariant: the list-literal expr ALSO surfaces an
    orthogonal cudf element-ordering divergence (filed separately, not the polars with_ path
    under test here)."""
    g = _graph(8)
    query = [n(), rows(), with_([("vals", "[num, num + 1, 99]")], extend=True)]
    assert _run(g, query, "pandas")[0] == "ok", "pandas oracle should build the list column"
    assert _run(g, query, "polars")[0] == "nie", "list-literal with_ must be an honest NIE on polars"
