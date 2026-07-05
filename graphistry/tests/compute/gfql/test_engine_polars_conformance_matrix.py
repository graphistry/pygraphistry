"""Generative differential-conformance matrix for the polars engine (Phase 0). CORE INVARIANT:
on any non-pandas engine a query is parity-equal to the pandas oracle OR an honest
NotImplementedError — never silently different, silently bridged, or a non-NIE crash. Checked
across SURFACES (chain / Cypher / let() DAG / call()) × OPS/PREDICATES — the cross-product the
prior chain-only gates missed (DAG silent-bridge bug et al.). CPU lane (pandas-vs-polars) runs
everywhere; the GPU lane (cudf / polars-gpu) joins on the dgx."""
import datetime
import numpy as np
import pandas as pd
import pytest
import graphistry
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected, call, let, rows, with_, return_

# No polars wheel (e.g. cp314) -> skip module cleanly, not per-case non-NIE ImportError "fails".
pytest.importorskip("polars")


# ---- graph with diverse dtypes (int, float, str, bool); NO null cells — null-path
# parity is exercised by test_scalar_fns_null_cells_parity's dedicated fixture ----
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


# Shared suite-wide comparison machinery; loud-failure contracts documented there.
from graphistry.tests.compute.gfql.polars_test_utils import (  # noqa: E402
    to_pandas_any as _to_pd,
    run_status as _run,
    available_nonpandas_engines,
    assert_parity_or_nie,
    assert_surfaces_agree,
)

_NONPANDAS_ENGINES = available_nonpandas_engines()


def _assert_invariant(g, query, label):
    assert_parity_or_nie(g, query, label, _NONPANDAS_ENGINES)


def test_pandas_oracle_sanity():
    """Canary: assert_parity_or_nie SKIPS when the pandas oracle errors, so a GLOBAL oracle
    break would silently skip the whole matrix — this known-good query must stay 'ok'."""
    assert _run(_graph(0), [n()], "pandas")[0] == "ok", "pandas oracle broken — matrix would silently skip"


# ---- predicate cross-product (the area that had 2 wrong-answer bugs) ----
def _predicate_queries():
    from graphistry.compute.predicates.numeric import GT, LT, GE, LE, Between, EQ, NE
    from graphistry.compute.predicates.is_in import IsIn
    from graphistry.compute.predicates.str import (
        Contains, Startswith, Endswith, Match, Fullmatch, IsNull, NotNull,
    )
    from graphistry.compute.predicates.numeric import IsNA, NotNA
    out = []
    for P, col, kw in [
        (GT, "num", {"val": 50}), (LT, "num", {"val": 50}), (GE, "num", {"val": 50}),
        (LE, "num", {"val": 50}),
        (EQ, "num", {"val": 50}), (NE, "num", {"val": 50}),    # scalar equality / inequality
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
        (IsNull, "name", {}), (NotNull, "name", {}),   # str-module null predicates (distinct from numeric IsNA/NotNA)
    ]:
        out.append((f"pred:{P.__name__}({col},{kw})", [n({col: P(**kw)})]))
    return out


@pytest.mark.parametrize("label,query", _predicate_queries())
def test_conformance_predicates_chain(label, query):
    _assert_invariant(_graph(1), query, f"chain {label}")


@pytest.mark.parametrize("label,query", _predicate_queries())
def test_conformance_predicates_dag(label, query):
    # SAME predicate via a let() DAG binding — must agree with the chain surface (parity or NIE)
    g = _graph(1)
    _assert_invariant(g, let({"a": query}), f"dag {label}")


# ---- traversal cross-product (single-hop parity; multi-hop/undirected-multi-edge NIE) ----
_TRAVERSAL_CASES = [
    ("fwd1", [n({"id": [0]}), e_forward()]),
    ("rev1", [n({"id": [0]}), e_reverse()]),
    ("und1", [n({"id": [0]}), e_undirected()]),
    ("n-e-n", [n(), e_forward(), n({"num": 50})]),
    ("fwd-fwd", [n({"id": [0]}), e_forward(), e_forward()]),
    ("multihop", [n({"id": [0]}), e_forward(hops=2)]),                 # NIE expected
    ("und-multi", [n({"id": [0]}), e_undirected(), e_undirected()]),   # NIE expected
    # ---- multi-hop (NIE today @ chain guard; PARITY-NATIVE after Stage 1) ----
    ("fwd-hops2", [n({"id": [0]}), e_forward(hops=2), n()]),
    ("rev-hops2", [n({"id": [0]}), e_reverse(hops=2), n()]),
    ("fwd-maxhops3", [n({"id": [0]}), e_forward(max_hops=3), n()]),     # 1..3 (max_hops wins over default hops=1)
    ("multihop-midfilter", [n({"id": [0]}), e_forward(hops=2), n({"num": 50}), e_forward(hops=2), n()]),
    ("multihop-named", [n({"id": [0]}, name="src"), e_forward(hops=2, name="h"), n(name="dst")]),
    ("sandwiched", [n({"id": [0]}), e_forward(), n(), e_forward(hops=2), n(), e_forward(), n()]),
    # hardening (adversarial-review): stranded-endpoint attr probe + *_match on a multi-hop edge
    ("multihop-deep-midfilter", [n({"id": [0]}), e_forward(hops=3), n({"flag": True}), e_forward(hops=2), n()]),
    ("multihop-srcmatch", [n({"id": [0]}), e_forward(hops=2, source_node_match={"flag": True}), n()]),
    ("multihop-dstmatch", [n({"id": [0]}), e_forward(hops=2, destination_node_match={"flag": True}), n()]),
    ("fwd-tofixed", [n({"id": [0]}), e_forward(to_fixed_point=True), n()]),        # Stage 4: native
    ("rev-tofixed", [n({"id": [0]}), e_reverse(to_fixed_point=True), n()]),        # Stage 4: native
    ("und-hops2", [n({"id": [0]}), e_undirected(hops=2), n()]),                    # undirected multi-hop: native
    ("und-maxhops3", [n({"id": [0]}), e_undirected(max_hops=3), n()]),             # undirected variable-length: native
    # ---- STAYS NIE (separately deferred surfaces) ----
    ("und-tofixed", [n({"id": [0]}), e_undirected(to_fixed_point=True), n()]),  # undirected to_fixed_point
    # NOTE: min_hops>1 excluded — orthogonal cudf bug (seed-node hop label: pandas None vs cudf
    # max_hops; cudf-device-residency-issue.md); polars' NIE-decline: test_engine_polars_chain.py::*deferred_raises
]


@pytest.mark.parametrize("label,query", _TRAVERSAL_CASES)
def test_conformance_traversals(label, query):
    _assert_invariant(_graph(2), query, f"traversal {label}")


@pytest.mark.parametrize("label,query", _TRAVERSAL_CASES)
def test_conformance_traversals_dag(label, query):
    # SAME traversal via let() DAG: guards the silent-bridge class (chain NIEs, DAG bridges to
    # pandas) and proves native multi-hop reaches the DAG/let surface too, not just chain.
    _assert_invariant(_graph(2), let({"a": query}), f"traversal dag {label}")


# ---- HOT-PATH CARVE-OUTS — adversarial inputs per fast path (they bypass the general engine =
# highest risk): node-only MATCH, (un)filtered 1-hop, both-endpoint filters, empties, self-loop/isolated ----
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
def _cypher_expression_queries():
    """(label, cypher) cases; importable so the ledger (test_conformance_ledger.py) DERIVES the
    exercised scalar-function set by parsing the cypher strings — mirrors _predicate_queries()."""
    return [
        ("sqrt", "MATCH (n) RETURN n.id AS id, sqrt(n.num) AS sq"),
        ("sign", "MATCH (n) RETURN n.id AS id, sign(n.num - 50) AS sg"),
        ("abs", "MATCH (n) RETURN n.id AS id, abs(n.num - 50) AS ab"),
        ("coalesce", "MATCH (n) RETURN n.id AS id, coalesce(n.num, 0) AS c"),
        ("casewhen", "MATCH (n) RETURN n.id AS id, CASE WHEN n.num > 50 THEN 1 ELSE 0 END AS cw"),
        ("casewhen_str", "MATCH (n) RETURN n.id AS id, CASE WHEN n.flag THEN 'y' ELSE 'n' END AS cw"),
        ("count_distinct_grouped", "MATCH (n) RETURN n.flag AS k, count(DISTINCT n.num) AS cd"),
        ("count_distinct_all", "MATCH (n) RETURN count(DISTINCT n.flag) AS cd"),
        ("count_grouped", "MATCH (n) RETURN n.flag AS k, count(n.num) AS c"),
        # count(*) short-circuit (count_table fast path): whole-graph node / edge counts.
        ("count_all_nodes", "MATCH (n) RETURN count(*) AS c"),
        ("count_all_edges", "MATCH ()-[r]->() RETURN count(*) AS c"),
        ("size_str", "MATCH (n) RETURN n.id AS id, size(n.name) AS sz"),
        ("substring3", "MATCH (n) RETURN n.id AS id, substring(n.name, 0, 4) AS sub"),
        ("substring2", "MATCH (n) RETURN n.id AS id, substring(n.name, 2) AS sub"),
        ("tointeger_int", "MATCH (n) RETURN n.id AS id, toInteger(n.num) AS i"),
        ("tointeger_float", "MATCH (n) RETURN n.id AS id, toInteger(n.f) AS i"),
        ("tointeger_bool", "MATCH (n) RETURN n.id AS id, toInteger(n.flag) AS i"),
        ("tofloat_int", "MATCH (n) RETURN n.id AS id, toFloat(n.num) AS f"),
        ("tofloat_float", "MATCH (n) RETURN n.id AS id, toFloat(n.f) AS f"),
        ("tofloat_bool", "MATCH (n) RETURN n.id AS id, toFloat(n.flag) AS f"),
        ("toboolean_bool", "MATCH (n) RETURN n.id AS id, toBoolean(n.flag) AS b"),
        ("tostring_bool", "MATCH (n) RETURN n.id AS id, toString(n.flag) AS s"),
        ("tostring_int", "MATCH (n) RETURN n.id AS id, toString(n.num) AS s"),
        ("tostring_str", "MATCH (n) RETURN n.id AS id, toString(n.name) AS s"),
        # #1673 numeric/string scalar fns (native on polars per the #1675 lowering; the
        # coverage ledger requires every GFQL_SCALAR_FUNCTIONS entry exercised-or-waived)
        ("floor", "MATCH (n) RETURN n.id AS id, floor(n.f) AS x"),
        ("ceil", "MATCH (n) RETURN n.id AS id, ceil(n.f) AS x"),
        ("ceiling", "MATCH (n) RETURN n.id AS id, ceiling(n.f) AS x"),
        ("round", "MATCH (n) RETURN n.id AS id, round(n.f) AS x"),
        ("round_p2", "MATCH (n) RETURN n.id AS id, round(n.f, 2) AS x"),
        ("tolower", "MATCH (n) RETURN n.id AS id, toLower(n.name) AS s"),
        ("toupper", "MATCH (n) RETURN n.id AS id, toUpper(n.name) AS s"),
        ("lower_alias", "MATCH (n) RETURN n.id AS id, lower(n.name) AS s"),
        ("upper_alias", "MATCH (n) RETURN n.id AS id, upper(n.name) AS s"),
        # NOTE: toString(float) intentionally absent — polars NIEs (test_tostring_float_honest_nie
        # _polars covers that), and cudf's orthogonal float-repr divergence from pandas would trip
        # _assert_invariant; the dedicated pandas-vs-polars test carries the real intent.
        # #1675 review wave-1: the `=~` family cross-engine. The cudf fullmatch emulation
        # mis-anchored top-level alternation and the (?i) lowercase-fold inverted escape
        # classes (both dgx-repro'd silent wrongs); unsafe folds + lookaround now DECLINE
        # (parity-or-NIE allows the NIE, forbids the wrong answer).
        ("regex_fullmatch_literal", "MATCH (n) WHERE n.name =~ 'node.1' RETURN n.id AS id"),
        ("regex_fullmatch_alternation", "MATCH (n) WHERE n.name =~ 'node.1|node.2' RETURN n.id AS id"),
        ("regex_ci_fold_safe", "MATCH (n) WHERE n.name =~ '(?i)NODE.1' RETURN n.id AS id"),
        ("regex_ci_fold_unsafe_declines", "MATCH (n) WHERE n.name =~ '(?i)\\\\D+' RETURN n.id AS id"),
        ("regex_lookbehind_declines", "MATCH (n) WHERE n.name =~ 'node(?<=e).*' RETURN n.id AS id"),
        ("regex_composed_or", "MATCH (n) WHERE n.name =~ 'node.1' OR n.num > 1000 RETURN n.id AS id"),
        # viz-filter L1: EXISTS { } pattern subqueries — semi_apply_mark/anti_semi_apply
        # run NATIVELY on polars (chain-flag key sets); parity-or-NIE 4-engine.
        ("exists_neighbor", "MATCH (n) WHERE EXISTS { (n)-->() } RETURN n.id AS id"),
        ("exists_neighbor_undirected", "MATCH (n) WHERE EXISTS { (n)--() } RETURN n.id AS id"),
        ("not_exists_neighbor", "MATCH (n) WHERE NOT EXISTS { (n)--() } RETURN n.id AS id"),
        ("exists_far_node_prop", "MATCH (n) WHERE EXISTS { (n)-->({num: 50}) } RETURN n.id AS id"),
        ("exists_composed_and", "MATCH (n) WHERE EXISTS { (n)-->() } AND n.num < 50 RETURN n.id AS id"),
    ]


@pytest.mark.parametrize("label,query", _cypher_expression_queries())
def test_conformance_cypher_expressions(label, query):
    _assert_invariant(_graph(4), query, f"cypher {label}")


# ---- NATIVE-vs-honest-NIE cypher scalar functions: provable sub-cases MUST lower natively (no
# lazy NIE, no silent bridge); unprovable ones MUST decline (never fabricate / replicate a pandas
# quirk); one-line "why" per row, full proofs at lowering sites (lazy/engine/polars/row_pipeline.py) ----
_NATIVE_OK_CYPHER = [
    # (id, cypher over _graph(4), why-native-is-provable)
    ("size_string", "MATCH (n) RETURN n.id AS id, size(n.name) AS sz",
     "str.len_chars == pandas str.len"),
    ("substring_0_4", "MATCH (n) RETURN n.id AS id, substring(n.name, 0, 4) AS sub",
     "non-negative int literals: str.slice(offset,len) == pandas str.slice(start,start+len)"),
    ("tointeger_float", "MATCH (n) RETURN n.id AS id, toInteger(n.f) AS i",
     "truncate toward zero, NaN/null -> null"),
    ("tointeger_int", "MATCH (n) RETURN n.id AS id, toInteger(n.num) AS i",
     "identity cast"),
    ("toboolean_bool", "MATCH (n) RETURN n.id AS id, toBoolean(n.flag) AS b",
     "identity, nulls preserved"),
    ("tostring_bool", "MATCH (n) RETURN n.id AS id, toString(n.flag) AS s",
     "polars Boolean casts to lowercase true/false == pandas astype(str)+rewrite"),
    ("tostring_int", "MATCH (n) RETURN n.id AS id, toString(n.num) AS s",
     "decimal digits identical"),
]

_HONEST_NIE_CYPHER = [
    # (id, cypher over _graph(4), pandas_expect 'ok'|'raises', why-polars-declines)
    ("tofloat_string", "MATCH (n) RETURN n.id AS id, toFloat(n.name) AS f", "raises",
     "pandas astype(float) RAISES on non-numeric strings; strict=False nulls would fabricate data"),
    ("tointeger_string", "MATCH (n) RETURN n.id AS id, toInteger(n.name) AS i", "raises",
     "pandas astype(float) RAISES on non-numeric strings; strict=False nulls would fabricate data"),
    ("size_numeric", "MATCH (n) RETURN n.id AS id, size(n.num) AS sz", "ok",
     "pandas size(non-string/non-list) = ROW-COUNT quirk we refuse to replicate"),
    ("substring_negative_start", "MATCH (n) RETURN n.id AS id, substring(n.name, -2) AS sub", "ok",
     "negative start diverges: pandas Python-slice vs polars offset/length — silent wrong slice"),
    ("tostring_float", "MATCH (n) RETURN n.id AS id, toString(n.f) AS s", "ok",
     "float repr diverges across engines"),
]


def test_round_tie_hazard_values_all_engines():
    """#1677 wave-1: the shared fixture's floats are rng.normal — ties are measure-zero,
    so the GPU engines never saw a tie/hazard value despite ties being the round() fix's
    whole point. Deterministic hazard column, ALL engines via parity-or-NIE: exact ties
    (p=0 ties -> +inf; ±1.25 = EXACT p=1 ties, ×10 = ±12.5 → away-from-zero ±1.3 vs
    half-even ±1.2 — wave-2 caught that no other value ties at p>0; 2^51+0.5 = a
    representable HUGE tie), the 1-ulp-below-tie JDK case, -0.0 (zero-sign normalize),
    1e308 (scaled overflow -> identity), NaN + null, and p=400 (identity on BOTH
    engines — 10.0**p / u32 overflow guards, was uncaught/divergent)."""
    import pandas as pd
    nd = pd.DataFrame({
        "id": list(range(14)),
        "v": [0.5, -0.5, 2.5, -1.5, -2.55, 1.25, -1.25, 0.49999999999999994,
              -0.04, -0.0, 1e308, 2251799813685248.5, float("nan"), None],
    })
    ed = pd.DataFrame({"s": [0], "d": [1], "eid": [0]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")
    for q in [
        "MATCH (n) RETURN n.id AS id, round(n.v) AS x",
        "MATCH (n) RETURN n.id AS id, round(n.v, 1) AS x",
        "MATCH (n) RETURN n.id AS id, round(n.v, 2) AS x",
        "MATCH (n) RETURN n.id AS id, round(n.v, 400) AS x",
    ]:
        _assert_invariant(g, q, f"round-hazard {q}")


def test_round_negative_zero_normalized_all_engines():
    """round(-0.04, 1) must be +0.0 on EVERY engine: the pandas/cuDF kernel's + 0.0
    normalizes, polars' native mode= kept -0.0 (dgx-repro'd) — and value equality
    cannot see it (-0.0 == 0.0), so pin the sign bit explicitly (NIE acceptable)."""
    import math as _math
    import pandas as pd
    nd = pd.DataFrame({"id": [0], "v": [-0.04]})
    ed = pd.DataFrame({"s": [0], "d": [0], "eid": [0]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")
    q = "MATCH (n) RETURN round(n.v, 1) AS x"
    for eng in ["pandas"] + _NONPANDAS_ENGINES:
        if _run(g, q, eng)[0] == "nie":
            continue
        val = float(_to_pd(g.gfql(q, engine=eng)._nodes)["x"].iloc[0])
        assert _math.copysign(1.0, val) == 1.0, f"{eng}: round(-0.04, 1) kept -0.0"


@pytest.mark.parametrize("label,cypher,why", _NATIVE_OK_CYPHER, ids=[c[0] for c in _NATIVE_OK_CYPHER])
def test_scalar_fn_runs_natively_on_polars(label, cypher, why):
    g = _graph(4)
    base = _run(g, cypher, "pandas")
    res = _run(g, cypher, "polars")
    assert res[0] == "ok", f"{label} must run NATIVELY on polars ({why}), got {res}"
    assert res == base, f"{label} polars must match pandas oracle: {res} != {base}"


@pytest.mark.parametrize("label,cypher,pandas_expect,why", _HONEST_NIE_CYPHER,
                         ids=[c[0] for c in _HONEST_NIE_CYPHER])
def test_scalar_fn_honest_nie_on_polars(label, cypher, pandas_expect, why):
    g = _graph(4)
    base = _run(g, cypher, "pandas")[0]
    assert (base == "ok") == (pandas_expect == "ok"), \
        f"{label}: pandas-oracle expectation drifted (expected {pandas_expect}, got {base})"
    assert _run(g, cypher, "polars")[0] == "nie", f"{label} must be an honest NIE on polars: {why}"


def test_regex_cudf_emulation_pinned_repros():
    """Pins the two dgx-repro'd cudf silent-wrongs with DISCRIMINATING fixtures — the
    generic matrix cases pass on pre-fix code (the shared fixture has no
    anchoring-breaking names and no digit-only names; #1675 wave-2). Pre-fix: the
    alternation matched 'abXXX' (bare ^ab|cd$ anchoring); (?i)\\D+ INVERTED the
    result set (pattern .lower() turned it into \\d+). Also pins that lowercase
    escapes ((?i)\\d+) KEEP working on cudf — the guard must not over-decline."""
    import pandas as pd
    ed = pd.DataFrame({"s": [0], "d": [1], "eid": [0]})
    g1 = graphistry.nodes(
        pd.DataFrame({"id": [0, 1, 2, 3], "name": ["ab", "cd", "abXXX", "xcd"]}), "id"
    ).edges(ed, "s", "d").bind(edge="eid")
    _assert_invariant(g1, "MATCH (n) WHERE n.name =~ 'ab|cd' RETURN n.id AS id", "regex-alt-pin")
    g2 = graphistry.nodes(
        pd.DataFrame({"id": [0, 1, 2, 3], "name": ["abc", "123", "XYZ", "7"]}), "id"
    ).edges(ed, "s", "d").bind(edge="eid")
    _assert_invariant(g2, "MATCH (n) WHERE n.name =~ '(?i)\\\\D+' RETURN n.id AS id", "regex-ci-unsafe-pin")
    _assert_invariant(g2, "MATCH (n) WHERE n.name =~ '(?i)\\\\d+' RETURN n.id AS id", "regex-ci-lower-escape-pin")


def test_scalar_fns_null_cells_parity():
    """#1675 wave-1: the shared fixture has NO null cells, so the scalar fns'
    where(~null_mask, pd.NA) paths were never exercised cross-engine — dedicated
    null-bearing columns here (float + string), 4-engine parity-or-NIE."""
    import pandas as pd
    nd = pd.DataFrame({
        "id": [0, 1, 2, 3],
        "fx": [1.5, None, -2.25, None],
        "sx": ["Ab", None, "cD", None],
    })
    ed = pd.DataFrame({"s": [0, 1], "d": [1, 2], "eid": [0, 1]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")
    for q in [
        "MATCH (n) RETURN n.id AS id, floor(n.fx) AS x",
        "MATCH (n) RETURN n.id AS id, ceil(n.fx) AS x",
        "MATCH (n) RETURN n.id AS id, round(n.fx) AS x",
        "MATCH (n) RETURN n.id AS id, toLower(n.sx) AS s",
        "MATCH (n) RETURN n.id AS id, toUpper(n.sx) AS s",
        "MATCH (n) WHERE n.sx =~ '(?i)ab' RETURN n.id AS id",
    ]:
        _assert_invariant(g, q, f"nullcells {q}")


def test_search_any_op_all_engines():
    """viz-filter L2 `search_any` row op (panel-algebra D2 inspector semantics):
    OR-across-columns; case-insensitive substring DEFAULT; regex opt-in; dtype gate =
    string cols always + int cols iff numeric-literal term (floats gated OUT of auto);
    explicit columns= override; nulls never match. Oracle-pinned + 4-engine
    parity-or-NIE via the chain call surface."""
    import pandas as pd
    from graphistry.compute.ast import search_any as search_any_op
    from graphistry.compute.exceptions import GFQLValidationError
    nd = pd.DataFrame({
        "id": [0, 1, 2, 3],
        "name": ["Alpha", "beta", None, "gamma7"],
        "num": pd.Series([7, 77, 3, 4], dtype="int64"),
        "f": [0.5, 1.5, 2.5, 3.5],
    })
    ed = pd.DataFrame({"s": [0, 1], "d": [1, 2], "eid": [0, 1]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")

    def q(**kw):
        return [n(name="a"), search_any_op(alias="a", out_col="__hit__", **kw)]

    oracle = {
        "default_ci": (dict(term="ALPHA"), [True, False, False, False]),
        "case_sensitive": (dict(term="ALPHA", case_sensitive=True), [False, False, False, False]),
        "numeric_int_gate": (dict(term="7"), [True, True, False, True]),  # num=7, num=77, name gamma7; f EXCLUDED
        "explicit_cols": (dict(term="7", columns=["name"]), [False, False, False, True]),
        "regex": (dict(term="a.pha", regex=True), [True, False, False, False]),  # (?i) default
    }
    for label, (kw, expected) in oracle.items():
        out = g.gfql(q(**kw), engine="pandas")
        got = _to_pd(out._nodes).sort_values("id")["__hit__"].tolist()
        assert got == expected, f"search_any oracle {label}: {got} != {expected}"
        _assert_invariant(g, q(**kw), f"search_any {label}")
    with pytest.raises(GFQLValidationError):
        g.gfql(q(term="x", columns=["nope"]), engine="pandas")


def test_search_any_cypher_surface_all_engines():
    """viz-filter L2-b: the cypher WHERE searchAny(...) surface — marker lift +
    composition with other predicates; oracle-pinned + 4-engine parity-or-NIE."""
    import pandas as pd
    nd = pd.DataFrame({
        "id": [0, 1, 2, 3],
        "name": ["Alpha", "beta", None, "gamma7"],
        "num": pd.Series([7, 77, 3, 4], dtype="int64"),
    })
    ed = pd.DataFrame({"s": [0, 1], "d": [1, 2], "eid": [0, 1]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")
    oracle = {
        "MATCH (a) WHERE searchAny(a, 'ALPHA') RETURN a.id AS id": [0],
        "MATCH (a) WHERE searchAny(a, '7') RETURN a.id AS id": [0, 1, 3],
        "MATCH (a) WHERE searchAny(a, '7', {columns: ['name']}) RETURN a.id AS id": [3],
        "MATCH (a) WHERE searchAny(a, 'a.pha', {regex: true}) RETURN a.id AS id": [0],
        "MATCH (a) WHERE searchAny(a, 'a') AND a.num > 10 RETURN a.id AS id": [1],
        "MATCH (a) WHERE NOT searchAny(a, 'a') RETURN a.id AS id": [2],
        # two lifts in one WHERE compose independently (wave-2 S4):
        # 'alpha'~{0}; '7'~{0 (num=7), 1 (77), 3 (gamma7)}; AND => [0]
        "MATCH (a) WHERE searchAny(a, 'alpha') AND searchAny(a, '7') RETURN a.id AS id": [0],
    }
    for q, expected in oracle.items():
        pdf = _to_pd(g.gfql(q, engine="pandas")._nodes)
        assert sorted(pdf["id"].tolist()) == expected, f"oracle drift: {q}"
        _assert_invariant(g, q, f"searchAny-cypher {q}")


def test_search_nodes_edges_twins():
    """g.search_nodes/search_edges (persona-pass python twins): same kernel, own table
    only, Plottable out; polars frames decline honestly."""
    import pandas as pd
    from graphistry.compute.exceptions import GFQLValidationError
    nd = pd.DataFrame({"id": [0, 1, 2], "name": ["Alpha", "beta", None]})
    ed = pd.DataFrame({"s": [0, 1], "d": [1, 2], "etype": ["Knows", "likes"], "eid": [0, 1]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")
    assert sorted(g.search_nodes("ALPHA")._nodes["id"].tolist()) == [0]
    assert sorted(g.search_nodes("alpha", case_sensitive=True)._nodes["id"].tolist()) == []
    assert sorted(g.search_edges("knows")._edges["eid"].tolist()) == [0]
    with pytest.raises(GFQLValidationError):
        g.search_nodes("x", columns=["nope"])
    import polars as pl
    gpl = graphistry.nodes(pl.from_pandas(nd), "id").edges(pl.from_pandas(ed), "s", "d").bind(edge="eid")
    with pytest.raises(NotImplementedError):
        gpl.search_nodes("x")


def test_search_any_literal_and_param_guards():
    """wave-1 B1/I5: `searchAny(...)` spelled INSIDE a string literal is data — no
    lift, no spurious unbound-alias error, literal compares byte-identically; a
    non-literal term ($param) fails loudly at lowering with a usable message
    instead of leaking an unknown function downstream."""
    from graphistry.compute.exceptions import GFQLValidationError
    lit = 'see searchAny(zzz, "x") docs'
    nd = pd.DataFrame({"id": [0, 1], "note": [lit, "plain"]})
    ed = pd.DataFrame({"s": [0], "d": [1], "eid": [0]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")
    # zzz is unbound: if the lift saw inside the literal this would error/rewrite
    q1 = "MATCH (n) WHERE n.note = 'see searchAny(zzz, \"x\") docs' RETURN n.id AS id"
    assert _to_pd(g.gfql(q1, engine="pandas")._nodes)["id"].tolist() == [0]
    _assert_invariant(g, q1, "searchAny-in-literal")
    # real call AND a literal mention in the same WHERE: only the real call lifts
    q2 = ("MATCH (n) WHERE searchAny(n, 'plain') AND n.note <> 'searchAny(zzz, \"y\")' "
          "RETURN n.id AS id")
    assert _to_pd(g.gfql(q2, engine="pandas")._nodes)["id"].tolist() == [1]
    _assert_invariant(g, q2, "searchAny-real-plus-literal")
    # $param term: the lift can't parse it — clear E108, not an obscure failure.
    # Without params the generic missing-param validation fires first (also clear);
    # WITH params supplied, the residual-searchAny guard is the one that speaks.
    qp = "MATCH (n) WHERE searchAny(n, $q) RETURN n.id AS id"
    with pytest.raises(GFQLValidationError, match="Missing Cypher parameter"):
        g.gfql(qp, engine="pandas")
    with pytest.raises(GFQLValidationError, match="string-literal term"):
        g.gfql(qp, engine="pandas", params={"q": "plain"})


def test_search_any_cudf_regex_guards():
    """wave-1 B2: the cuDF search path honors the same decline rules as =~ —
    NIE on lookaround (libcudf kernel-compile crash otherwise) and on unsound
    case-insensitive folds (blind pat.lower() turns \\D into \\d — INVERTS);
    sound patterns still run natively."""
    cudf = pytest.importorskip("cudf")
    from graphistry.compute.gfql.search_any import search_any_mask
    df = cudf.DataFrame({"name": ["Deed", "1234"]})
    with pytest.raises(NotImplementedError):
        search_any_mask(df, r"\D+", regex=True)  # case-insensitive default: unsound fold
    with pytest.raises(NotImplementedError):
        search_any_mask(df, r"x(?=y)", regex=True, case_sensitive=True)  # lookaround
    got = search_any_mask(df, r"de{2}d", regex=True)  # sound fold: native
    assert got.to_pandas().tolist() == [True, False]
    got_cs = search_any_mask(df, r"\d{4}", regex=True, case_sensitive=True)
    assert got_cs.to_pandas().tolist() == [False, True]


def test_search_any_null_and_float_stringify():
    """wave-1 I1/I3: null cells NEVER match — including explicit non-string columns
    where pandas astype(str) would stringify them ('nan'/'<NA>'); and explicit
    float-column stringification is parity-pinned (term '7' discriminates 7.5)."""
    from graphistry.compute.ast import search_any as search_any_op
    from graphistry.compute.exceptions import GFQLValidationError
    nd = pd.DataFrame({
        "id": [0, 1, 2],
        # 7.25 sits on a row with NO int-column '7' so a float wrongly entering the
        # auto gate flips row 2 — the wave-1 fixture (7.5 on the ni=7 row) could not
        # discriminate (wave-2 W2-1)
        "f": [0.5, None, 7.25],
        "ni": pd.array([7, None, 3], dtype="Int64"),
        "flag": pd.array([True, None, False], dtype="boolean"),
    })
    ed = pd.DataFrame({"s": [0], "d": [1], "eid": [0]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")

    def q(**kw):
        return [n(name="a"), search_any_op(alias="a", out_col="__hit__", **kw)]

    for term in ["nan", "NA", "None"]:  # the null-stringification spellings
        kw = dict(term=term, columns=["f", "ni"])
        got = _to_pd(g.gfql(q(**kw), engine="pandas")._nodes).sort_values("id")["__hit__"].tolist()
        assert got == [False, False, False], f"null cells matched {term!r}: {got}"
        _assert_invariant(g, q(**kw), f"search_any nulls {term}")
    # floats stay OUT of the auto gate even when they'd match stringified ('7' in
    # '7.25'); numeric term still probes int cols — ni=7 hits row 0 ONLY
    got = _to_pd(g.gfql(q(term="7"), engine="pandas")._nodes).sort_values("id")["__hit__"].tolist()
    assert got == [True, False, False], f"auto gate drift: {got}"  # NOT [.., .., True]
    _assert_invariant(g, q(term="7"), "search_any float auto-gate")
    # explicit float column: canonical toString on pandas, discriminating pin (I3);
    # polars DECLINES floats — repr diverges in the exponent regime, same decision
    # as the polars toString lowering (wave-3 W3-1); NIE tolerated by the invariant
    kw = dict(term="7", columns=["f"])
    got = _to_pd(g.gfql(q(**kw), engine="pandas")._nodes).sort_values("id")["__hit__"].tolist()
    assert got == [False, False, True], f"float stringify drift: {got}"
    _assert_invariant(g, q(**kw), "search_any float explicit")
    # explicit BOOL column under caseSensitive: pandas renders 'True' — polars must
    # match (canonicalized) or NIE, never silently miss (wave-2 W2-3)
    kw = dict(term="True", columns=["flag"], case_sensitive=True)
    got = _to_pd(g.gfql(q(**kw), engine="pandas")._nodes).sort_values("id")["__hit__"].tolist()
    assert got == [True, False, False], f"bool stringify drift: {got}"
    _assert_invariant(g, q(**kw), "search_any bool explicit cs")
    # missing explicit column raises the SAME E108 natively on polars — a generic
    # NIE here would misreport a user error as an engine gap (wave-2 W2-2)
    import polars as pl
    gpl = graphistry.nodes(pl.from_pandas(nd), "id").edges(
        pl.from_pandas(ed), "s", "d").bind(edge="eid")
    with pytest.raises(GFQLValidationError, match="absent"):
        gpl.gfql(q(term="x", columns=["nope"]), engine="polars")
    # explicit FLOAT column on polars: pinned NIE (exponent-regime repr divergence,
    # wave-3 W3-1 — same decision as the polars toString lowering)
    with pytest.raises(NotImplementedError):
        gpl.gfql(q(term="7", columns=["f"]), engine="polars")
    # explicit TEMPORAL column: stringification is engine-divergent — polars
    # declines honestly (NIE) instead of risking a silent mismatch (wave-2 W2-3)
    ndt = nd.assign(t=pd.to_datetime(["2020-01-02", "2021-03-04", "2022-05-06"]))
    gplt = graphistry.nodes(pl.from_pandas(ndt), "id").edges(
        pl.from_pandas(ed), "s", "d").bind(edge="eid")
    with pytest.raises(NotImplementedError):
        gplt.gfql(q(term="2020", columns=["t"]), engine="polars")
    # precedence: a MISSING column is a user error (E108) even when another listed
    # column would trip the dtype gate — a refactor must not flip it to NIE (wave-3)
    with pytest.raises(GFQLValidationError, match="absent"):
        gplt.gfql(q(term="x", columns=["nope", "t"]), engine="polars")


def test_search_any_edge_alias():
    """wave-1 I4: edge-alias searchAny — correct on pandas; polars declines with an
    honest NIE (documented v1 gap), never a silent wrong answer."""
    nd = pd.DataFrame({"id": [0, 1, 2]})
    ed = pd.DataFrame({
        "s": [0, 1], "d": [1, 2], "eid": [0, 1],
        "etype": ["Knows", "likes"],
    })
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")
    q = "MATCH (a)-[r]->(b) WHERE searchAny(r, 'knows') RETURN r.eid AS eid"
    assert _to_pd(g.gfql(q, engine="pandas")._nodes)["eid"].tolist() == [0]
    _assert_invariant(g, q, "searchAny-edge-alias")


def test_exists_polars_internal_decline_branches():
    """Coverage pins for the honest-decline branches of the polars semi-apply family
    (each returns None -> the boundary lane raises NIE): non-single/unnamed/query=/
    alias==node_id rows shapes; non-endpoint join alias; multi-hop edges; neq aliases
    that aren't the endpoints (wave-2 / changed-line gate)."""
    import polars as pl
    from graphistry.compute.ast import n as _n, e_forward as _ef
    from graphistry.compute.chain import serialize_binding_ops
    from graphistry.compute.gfql.lazy.engine.polars.pattern_apply import (
        rows_binding_ops_polars,
        _pattern_alias_keys_polars,
    )
    import pandas as pd
    nd = pd.DataFrame({"id": [0, 1], "x": [1, 2]})
    ed = pd.DataFrame({"s": [0], "d": [1], "eid": [0]})
    gpl = graphistry.nodes(pl.from_pandas(nd), "id").edges(
        pl.from_pandas(ed), "s", "d").bind(edge="eid")
    assert rows_binding_ops_polars(gpl, []) is None
    assert rows_binding_ops_polars(
        gpl, serialize_binding_ops([_n(name="n"), _n(name="m")])) is None
    assert rows_binding_ops_polars(gpl, serialize_binding_ops([_n()])) is None
    assert rows_binding_ops_polars(
        gpl, serialize_binding_ops([_n(name="n", query="x > 1")])) is None
    assert rows_binding_ops_polars(
        gpl, serialize_binding_ops([_n(name="id")])) is None  # alias == node-id col
    bos = serialize_binding_ops([_n(name="n"), _ef(), _n(name="m")])
    assert _pattern_alias_keys_polars(gpl, bos, "zzz") is None  # not an endpoint
    assert _pattern_alias_keys_polars(
        gpl, serialize_binding_ops([_n(name="n"), _ef(hops=2), _n()]), "n") is None
    assert _pattern_alias_keys_polars(gpl, bos, "n", neq=["n", "zzz"]) is None
    assert _pattern_alias_keys_polars(gpl, bos[:2], "n") is None  # not Node/Edge/Node


def test_prune_isolated_graph_shape_all_engines():
    """LEO steer 2026-07-05: viz needs the FULL GRAPH back — nodes AND edges. THE
    working patterns (dgx-probed): `GRAPH { MATCH (a)-[e]-(b) }` = keep-self
    prune-isolated (self-edged + mutually-connected nodes with ALL their edges;
    true edgeless nodes dropped); `+ WHERE a.id <> b.id` = drop-self variant (note:
    also drops self-EDGES of kept nodes — panel-algebra §3's E₂ keeps them; the
    exact algebra form needs a two-stage pipeline, L3). EXISTS inside GRAPH { } is
    REJECTED fail-fast — it was SILENTLY DROPPED (dgx-repro'd all-nodes wrong
    answer), and node-only GRAPH matches return zero edges."""
    import pandas as pd
    from graphistry.compute.exceptions import GFQLValidationError
    nd = pd.DataFrame({"id": [0, 1, 2, 3, 4]})
    # 0<->1 mutually connected (2 directed edges), 4 self-loop-only, 2+3 edgeless
    ed = pd.DataFrame({"s": [0, 1, 4], "d": [1, 0, 4], "eid": [0, 1, 2]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")
    oracle = {
        "GRAPH { MATCH (a)-[e]-(b) }": ([0, 1, 4], [0, 1, 2]),
        "GRAPH { MATCH (a)-[e]-(b) WHERE a.id <> b.id }": ([0, 1], [0, 1]),
    }
    for q, (exp_nodes, exp_edges) in oracle.items():
        out = g.gfql(q, engine="pandas")
        assert sorted(_to_pd(out._nodes)["id"].tolist()) == exp_nodes, q
        assert sorted(_to_pd(out._edges)["eid"].tolist()) == exp_edges, \
            f"{q}: kept nodes must keep ALL their edges"
    _assert_invariant(g, "GRAPH { MATCH (a)-[e]-(b) }", "prune-graph-shape keep-self")
    # drop-self uses cross-entity same-path WHERE: polars/polars-gpu decline honestly;
    # cuDF has a PRE-EXISTING TypeError in the same-path executor (verified
    # byte-identical on the base tree — repo debt predating this PR, plan-tracked).
    q2 = "GRAPH { MATCH (a)-[e]-(b) WHERE a.id <> b.id }"
    base = _run(g, q2, "pandas")
    for eng in _NONPANDAS_ENGINES:
        res = _run(g, q2, eng)
        if res[0] == "ok":
            assert res == base, f"prune-graph-shape drop-self[{eng}]: silent divergence"
        elif eng == "cudf":
            assert res[0] == "nie" or res == ("err", "TypeError"), \
                f"cudf drop-self: expected NIE or the pre-existing same-path TypeError, got {res}"
        else:
            assert res[0] == "nie", f"{eng} drop-self must decline honestly, got {res}"
    with pytest.raises(GFQLValidationError, match="GRAPH"):
        g.gfql("GRAPH { MATCH (n) WHERE EXISTS { (n)--() } }", engine="pandas")


def test_exists_native_and_boundary_pins_polars():
    """Wave-1 T1/T2/T4 pins: (a) EXISTS stays NATIVE on polars — parity-or-NIE alone
    would let a silent NIE regression pass; (b) multi-alias correlation DECLINES on
    polars (pair bindings inexpressible in the key-set route) while pandas answers;
    (c) self-alias EXISTS { (n)--(n) } never non-NIE crashes (chain's duplicate-alias
    guard is caught into a decline)."""
    g = _graph(4)
    for q in [
        "MATCH (n) WHERE EXISTS { (n)-->() } RETURN n.id AS id",
        "MATCH (n) WHERE NOT EXISTS { (n)--() } RETURN n.id AS id",
    ]:
        assert _run(g, q, "polars")[0] == "ok", f"EXISTS must stay NATIVE on polars: {q}"
    q_multi = "MATCH (a)-->(b) WHERE EXISTS { (a)-->(b) } RETURN a.id AS id"
    assert _run(g, q_multi, "pandas")[0] == "ok"
    assert _run(g, q_multi, "polars")[0] == "nie",         "multi-alias correlation must decline on polars (not crash or diverge)"
    _assert_invariant(g, q_multi, "exists-multi-alias")
    _assert_invariant(g, "MATCH (n) WHERE EXISTS { (n)--(n) } RETURN n.id AS id", "exists-self-alias")


def test_exists_prune_isolated_flavors_all_engines():
    """viz-filter L1 acceptance kernel: BOTH prune-isolated flavors of the panel algebra
    (research/panel-algebra.md §3) on a discriminating graph — node 3 is fully isolated,
    node 4 has ONLY a self-loop. keep-self (EXISTS {(n)--()}) keeps 4; drop-self
    (EXISTS {(n)--(m) WHERE m<>n}) drops 4; 4-engine parity-or-NIE."""
    import pandas as pd
    nd = pd.DataFrame({"id": [0, 1, 2, 3, 4]})
    ed = pd.DataFrame({"s": [0, 1, 4], "d": [1, 2, 4], "eid": [0, 1, 2]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")
    oracle = {
        "MATCH (n) WHERE EXISTS { (n)--() } RETURN n.id AS id": [0, 1, 2, 4],
        "MATCH (n) WHERE NOT EXISTS { (n)--() } RETURN n.id AS id": [3],
        "MATCH (n) WHERE EXISTS { (n)--(m) WHERE m <> n } RETURN n.id AS id": [0, 1, 2],
        "MATCH (n) WHERE NOT EXISTS { (n)--(m) WHERE m <> n } RETURN n.id AS id": [3, 4],
    }
    for q, expected in oracle.items():
        pdf = _to_pd(g.gfql(q, engine="pandas")._nodes)
        assert sorted(pdf["id"].tolist()) == expected, f"oracle drift: {q}"
        _assert_invariant(g, q, f"prune-isolated {q}")


def test_tolower_non_string_declines_never_fabricates():
    """toLower/toUpper on a non-string column must never return values (neo4j: type error).
    Regression (#1675 wave-1, dgx-repro'd): the scalar fallback broadcast the lowercased
    Series REPR to every row on pandas AND cudf; polars-gpu raised a non-NIE SchemaError."""
    g = _graph(4)
    for fn in ["toLower", "toUpper"]:
        q = f"MATCH (n) RETURN n.id AS id, {fn}(n.num) AS s"
        for eng in ["pandas"] + _NONPANDAS_ENGINES:
            res = _run(g, q, eng)
            assert res[0] in ("err", "nie"), \
                f"{fn}(numeric) must decline on {eng}, got {res[0]} (silent fabrication)"


def test_numeric_fn_float_dtype_parity_polars():
    """neo4j floor/ceil/round return Float and the pandas engine astype(float)s; polars must
    too (#1675 wave-1: bare .floor() on Int64 stayed Int64 — the value-level sig is
    dtype-blind (5 == 5.0), so pin dtypes explicitly)."""
    g = _graph(4)
    for fn in ["floor", "ceil", "round"]:
        q = f"MATCH (n) RETURN {fn}(n.num) AS x"
        pdf = _to_pd(g.gfql(q, engine="pandas")._nodes)
        pol = _to_pd(g.gfql(q, engine="polars")._nodes)
        assert pdf["x"].dtype.kind == "f", f"{fn}(int) oracle must be float"
        assert pol["x"].dtype.kind == "f", f"{fn}(int) must be Float64 on polars, got {pol['x'].dtype}"


def test_collect_aggregations_native_parity_polars():
    """collect(x)/collect(DISTINCT x) lower NATIVELY == pandas: nulls dropped, within-group order
    kept (collect keeps dups; distinct dedups keep-first), all-null -> []; list cells normalized."""
    import pandas as pd
    e = pd.DataFrame({"s": [0, 1, 2, 3, 4, 5], "d": [1, 2, 3, 4, 5, 6],
                      "k": ["a", "a", "a", "b", "b", "c"], "v": ["x", "w", None, "y", "y", None]})
    g = graphistry.edges(e, "s", "d")

    def collected(engine, q):
        df = _to_pd(g.gfql(q, engine=engine)._nodes).sort_values("k")
        return {r["k"]: list(r["vs"]) for _, r in df.iterrows()}

    for q, expected in [
        ("MATCH ()-[r]->() RETURN r.k AS k, collect(r.v) AS vs", {"a": ["x", "w"], "b": ["y", "y"], "c": []}),
        ("MATCH ()-[r]->() RETURN r.k AS k, collect(DISTINCT r.v) AS vs", {"a": ["x", "w"], "b": ["y"], "c": []}),
    ]:
        assert collected("pandas", q) == expected
        assert collected("polars", q) == expected


def test_size_list_runs_natively_on_polars():
    """size(<List col>) MUST lower NATIVELY (list.len); operand from the native list-literal with_
    (dtype List); count is order-invariant (no cudf reorder exposure); direct pandas-vs-polars
    because the intermediate List cell makes the generic sig fragile."""
    g = _graph(4)
    query = [
        n(), rows(),
        with_([("lst", "[num, num + 1, 99]")], extend=True),
        with_([("sz", "size(lst)")], extend=True),
    ]
    res = _run(g, query, "polars")
    assert res[0] == "ok", f"size(list) must run NATIVELY on polars, got {res}"
    pol = g.gfql(query, engine="polars")
    assert "polars" in type(pol._nodes).__module__, "size(list) returned non-polars nodes (silent bridge!)"
    pdf = _to_pd(g.gfql(query, engine="pandas")._nodes)
    poldf = _to_pd(pol._nodes)
    assert poldf["sz"].tolist() == pdf["sz"].tolist(), "size(list) diverges polars vs pandas"
    assert set(int(x) for x in poldf["sz"].tolist()) == {3}, "size of a 3-element list literal must be 3"


# ---- cross-surface call() consistency (silent-bridge class); shared constant so the ledger
# derives the exercised call()-safelist from the SAME list the test runs on (no drift) ----
_CALL_CONSISTENCY_FNS = ["get_degrees", "hypergraph", "limit"]


def _call_exercised_functions():
    """call()-safelist names this matrix exercises (importable for the ledger): the consistency
    test drives _CALL_CONSISTENCY_FNS; the degree trio also has dedicated conformance tests."""
    return set(_CALL_CONSISTENCY_FNS) | {"get_degrees", "get_indegrees", "get_outdegrees"}


def _rowop_exercised():
    """ROW_PIPELINE_CALLS ops with a labeled SUBJECT here (importable for the ledger): `with_`
    (with_extend*/in-membership), `unwind` (unwind_* native+NIE), and the _ROW_OP_CASES ops
    (chain+dag). Ops exercised only implicitly via cypher text (RETURN->select etc.) stay ledger
    waivers; semi_apply_mark/anti_semi_apply are now NATIVE on polars and exercised implicitly by
    the EXISTS { } cypher cases (exists_neighbor etc.); join_apply remains the honest-NIE waiver."""
    return {
        "with_", "unwind",
        "rows", "skip", "limit", "distinct", "drop_cols",
        "order_by", "select", "return_", "where_rows", "group_by",
        "count_table", "search_any",
    }


@pytest.mark.parametrize("fn", _CALL_CONSISTENCY_FNS)
def test_conformance_call_chain_vs_dag_consistent(fn):
    """A call must behave the SAME (parity or NIE) on chain and DAG — no silent bridge where the
    other declines; assert_surfaces_agree also fails non-NIE errors (old inline check let 'err' pass)."""
    g = _graph(3)
    params = {"value": 2} if fn == "limit" else {}
    chain_q = [call(fn, params)] if params else [call(fn)]
    chain = _run(g, chain_q, "polars")
    dag = _run(g, let({"a": (call(fn, params) if params else call(fn))}), "polars")
    assert_surfaces_agree(chain, dag, f"call '{fn}' chain-vs-dag")


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


# ==== Phase 2d native wins (session 2): get_degrees / temporal / with_extend — each gated by
# the parity-or-NIE invariant PLUS explicit "must run NATIVELY" assertions ====

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


@pytest.mark.parametrize("fn", ["get_degrees", "get_indegrees", "get_outdegrees"])
def test_degree_fn_runs_natively_on_polars(fn):
    """Degree trio = pure groupby/count -> MUST be NATIVE on polars: no lazy NIE, no silent bridge (frame probe)."""
    g = _degrees_graph()
    base = _run(g, [call(fn)], "pandas")
    res = _run(g, [call(fn)], "polars")
    assert res[0] == "ok", f"{fn} must be NATIVE on polars, got {res}"
    assert res == base, f"{fn} polars must match pandas oracle: {res} != {base}"
    out = g.gfql([call(fn)], engine="polars")
    assert "polars" in type(out._nodes).__module__, f"{fn} polars returned non-polars nodes (silent bridge!)"


@pytest.mark.parametrize("fn", ["get_degrees", "get_indegrees", "get_outdegrees"])
def test_degree_fn_chain_vs_dag_native_consistent(fn):
    """Chain call() vs let()/ref() DAG: both native-ok with the SAME signature (or both NIE; non-NIE errors fail)."""
    g = _degrees_graph()
    assert_surfaces_agree(_run(g, [call(fn)], "polars"),
                          _run(g, let({"a": call(fn)}), "polars"),
                          f"{fn} chain-vs-dag")


# ---- NATIVE get_indegrees/get_outdegrees on _degrees_graph; a self-loop counts ONCE per
# direction (not double, unlike get_degrees) ----
@pytest.mark.parametrize("fn", ["get_indegrees", "get_outdegrees"])
@pytest.mark.parametrize("label,params", [
    ("default", {}),
    ("custom-col", {"col": "mydeg"}),
])
def test_conformance_single_degree_chain(fn, label, params):
    query = [call(fn, params)] if params else [call(fn)]
    _assert_invariant(_degrees_graph(), query, f"{fn} chain {label}")


@pytest.mark.parametrize("fn", ["get_indegrees", "get_outdegrees"])
@pytest.mark.parametrize("label,params", [
    ("default", {}),
    ("custom-col", {"col": "mydeg"}),
])
def test_conformance_single_degree_dag(fn, label, params):
    binding = call(fn, params) if params else call(fn)
    _assert_invariant(_degrees_graph(), let({"a": binding}), f"{fn} dag {label}")


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


def _temporal_between_queries():
    from graphistry.compute.predicates.comparison import Between
    lo, hi = datetime.date(2020, 1, 15), datetime.date(2020, 3, 1)
    return [
        # inclusive lower boundary: BOTH 2020-01-15 rows (00:00 AND 23:59) included (date-truncated)
        ("between-incl",      [n({"ts": Between(lower=lo, upper=hi, inclusive=True)})]),
        # exclusive lower boundary: the 2020-01-15 rows are EXCLUDED (> not >=)
        ("between-excl",      [n({"ts": Between(lower=lo, upper=hi, inclusive=False)})]),
        # single-day window: only the two 2020-01-15 rows
        ("between-singleday",  [n({"ts": Between(lower=lo, upper=lo, inclusive=True)})]),
        # empty / out-of-range window -> no rows, must still parity/NIE-agree
        ("between-empty",     [n({"ts": Between(lower=datetime.date(2099, 1, 1),
                                                upper=datetime.date(2099, 2, 1))})]),
    ]


@pytest.mark.parametrize("label,query", _temporal_between_queries())
def test_conformance_temporal_between_chain(label, query):
    _assert_invariant(_temporal_graph(), query, f"temporal-between {label}")


@pytest.mark.parametrize("label,query", _temporal_between_queries())
def test_conformance_temporal_between_dag(label, query):
    _assert_invariant(_temporal_graph(), let({"a": query}), f"temporal-between dag {label}")


@pytest.mark.parametrize("label,query", [
    ("cy-gt-date", "MATCH (n) WHERE n.ts > date('2020-01-15') RETURN n.id AS id"),
    ("cy-ge-date", "MATCH (n) WHERE n.ts >= date('2020-01-15') RETURN n.id AS id"),
    ("cy-eq-date", "MATCH (n) WHERE n.ts = date('2020-01-15') RETURN n.id AS id"),
])
def test_conformance_temporal_datevalue_cypher(label, query):
    # Cypher date('...') folds to a DateValue in the node filter_dict -> same native lowering
    _assert_invariant(_temporal_graph(), query, f"temporal-cypher {label}")


# ---- WITH ... extend=True (native polars `with_columns` column-extension) ----
def test_conformance_with_extend_arithmetic_native():
    """extend=True adds a computed column KEEPING existing ones; MUST be native (with_columns), not an honest NIE."""
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
    """Chain with_(extend=True) vs Cypher `WITH n, ...`: both native+equal or both NIE (no silent bridge)."""
    g = _graph(7)
    chain_q = [
        n(), rows(),
        with_([("id", "id"), ("num", "num"), ("p", "num + 1")], extend=True),
        return_([("id", "id"), ("num", "num"), ("p", "p")]),
    ]
    cypher_q = "MATCH (n) WITH n, n.num + 1 AS p RETURN n.id AS id, n.num AS num, p AS p"
    _assert_invariant(g, chain_q, "with_extend chain")
    _assert_invariant(g, cypher_q, "with_extend cypher")
    assert_surfaces_agree(_run(g, chain_q, "polars"), _run(g, cypher_q, "polars"),
                          "with_extend chain-vs-cypher")


# ==== Native list-literal CONSTRUCTION + `x IN [literals]` membership. Construction is scoped
# pandas-vs-polars (cudf REORDERS list elements — orthogonal bug; ndarray-vs-list cell repr also
# breaks the generic sig); membership yields a Boolean column (cudf-safe) -> full invariant ====

def _norm_list_col(df, col):
    """Normalize a list-valued column from any engine to plain-python lists of ints,
    id-sorted, so a pandas list cell and a polars (ndarray/list) cell compare cleanly."""
    df = _to_pd(df).sort_values("id").reset_index(drop=True)
    return [None if c is None else [int(x) for x in list(c)] for c in df[col].tolist()]


def test_conformance_list_literal_construction_native_polars():
    """Homogeneous-int list literal builds per-row NATIVELY on polars (pl.concat_list), order
    preserved, element-for-element equal to pandas; scoped pandas-vs-polars (cudf reorder bug)."""
    g = _graph(8)
    query = [n(), rows(), with_([("vals", "[num, num + 1, 99]")], extend=True)]
    res = _run(g, query, "polars")
    assert res[0] == "ok", f"homogeneous-int list literal must run NATIVELY on polars, got {res}"
    poldf = g.gfql(query, engine="polars")._nodes
    assert "polars" in type(poldf).__module__, "list-literal polars returned non-polars nodes (silent bridge!)"
    pdf = g.gfql(query, engine="pandas")._nodes
    assert _norm_list_col(poldf, "vals") == _norm_list_col(pdf, "vals"), "list elements/order diverge polars vs pandas"


@pytest.mark.parametrize("label,query", [
    ("chain-int", [n(), rows(), with_([("hit", "num IN [1, 2, 3, 50, 51, 52]")], extend=True)]),
    ("cypher-int", "MATCH (n) RETURN n.id AS id, n.num IN [1, 2, 3, 50, 51, 52] AS hit"),
    ("cypher-str", "MATCH (n) RETURN n.id AS id, n.name IN ['node.1', 'node.2'] AS hit"),
])
def test_conformance_in_membership(label, query):
    """`x IN [literals]` as a ROW EXPRESSION (vs the WHERE/IsIn predicate path): 3-valued, Boolean
    output (cudf-safe) -> full invariant on every engine."""
    _assert_invariant(_graph(1), query, f"in-membership {label}")


# ---- NATIVE TEMPORAL date-part: IsLeapYear lowered; boundary predicates declined ----
def _leapyear_graph():
    """Naive datetime64 spanning leap/non-leap years incl. century cases (1900 NOT leap; 2000 leap) — real Gregorian calc."""
    nd = pd.DataFrame({
        "id": np.arange(7),
        "ts": pd.to_datetime([
            "1900-06-15", "2000-02-29", "2019-07-01",
            "2020-01-01", "2021-12-31", "2023-03-03", "2024-08-08",
        ]),
        "v": [1, 2, 3, 4, 5, 6, 7],
    })
    ed = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3], "eid": [0, 1, 2]})
    return graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")


def test_conformance_temporal_is_leap_year_parity():
    """IsLeapYear on NAIVE Datetime matches pandas across leap/non-leap (incl. 1900/2000) on every engine, or honest-NIEs."""
    from graphistry.compute.predicates.temporal import is_leap_year
    _assert_invariant(_leapyear_graph(), [n({"ts": is_leap_year()})], "temporal is_leap_year")


@pytest.mark.parametrize("factory", [
    "is_month_start", "is_month_end", "is_quarter_start",
    "is_quarter_end", "is_year_start", "is_year_end",
])
def test_temporal_boundary_predicates_native_parity(factory):
    """Date-part BOUNDARY predicates lower NATIVELY on naive Datetime via a provable calendar-field
    derivation (day==1 / day==days_in_month / month-set, leap-aware, NaT->False): pandas parity on
    every engine AND native on polars (no lazy NIE); _leapyear_graph spans true+false for each."""
    import graphistry.compute.predicates.temporal as T
    g = _leapyear_graph()
    q = [n({"ts": getattr(T, factory)()})]
    assert _run(g, q, "pandas")[0] == "ok", f"{factory} pandas oracle should compute"
    _assert_invariant(g, q, f"temporal boundary {factory}")
    if "polars" in _NONPANDAS_ENGINES:
        assert _run(g, q, "polars")[0] == "ok", f"{factory} must lower NATIVELY on polars, got NIE"


def _tzaware_graph():
    """tz-aware (UTC) Datetime — the shape polars must DECLINE (spans year/quarter/month start, mid-year, year/month end)."""
    nd = pd.DataFrame({
        "id": np.arange(3),
        "ts": pd.to_datetime(["2000-01-01", "2019-06-01", "2020-12-31"]).tz_localize("UTC"),
        "v": [1, 2, 3],
    })
    ed = pd.DataFrame({"s": [0, 1], "d": [1, 2], "eid": [0, 1]})
    return graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")


# ---- NATIVE-vs-honest-NIE over QUERY OBJECTS (predicates / row ops; cypher tables above) ----
def _native_ok_query_cases():
    """(id, graph_fn, query, why-native-is-provable): each MUST lower natively on polars (no lazy
    NIE) AND satisfy the full invariant; proofs at the lowering sites (predicates/row_pipeline)."""
    from graphistry.compute.predicates.comparison import GT, Between
    from graphistry.compute.predicates.temporal import is_leap_year
    return [
        ("temporal-between-naive", _temporal_graph,
         [n({"ts": Between(lower=datetime.date(2020, 1, 15), upper=datetime.date(2020, 3, 1))})],
         "composes two proven date-truncated endpoint compares"),
        ("temporal-gt-datevalue", _temporal_graph,
         [n({"ts": GT(val=datetime.date(2020, 1, 15))})],
         "naive Datetime column + DateValue = the proven SAFE temporal shape"),
        ("leapyear-naive", _leapyear_graph, [n({"ts": is_leap_year()})],
         "provable Gregorian leap-year derivation on a naive Datetime column"),
        ("in-membership-row-expr", lambda: _graph(1),
         [n(), rows(), with_([("hit", "num IN [1, 2, 3, 50, 51, 52]")], extend=True)],
         "homogeneous non-null literal is_in"),
    ]


def _polars_nie_query_cases():
    """(id, graph_fn, query, pandas_expect, why-polars-declines); pandas_expect: 'ok' (oracle
    computes — declining is the only honest polars answer), 'raises', None (not load-bearing).
    Deliberately NOT routed through _assert_invariant: these exprs also surface orthogonal cudf
    divergences (list-literal element order; list-cell repr), not the polars decline under test."""
    from graphistry.compute.predicates.comparison import GT, Between
    from graphistry.compute.predicates.is_in import IsIn
    import graphistry.compute.predicates.temporal as T
    return [
        ("temporal-between-tzaware-bounds", _temporal_graph,
         [n({"ts": Between(lower=datetime.datetime(2020, 1, 15, 0, 0),
                           upper=datetime.datetime(2020, 3, 1, 0, 0))})], None,
         "datetime bounds normalize to tz-tagged DateTimeValue -> an endpoint declines -> the composed Between declines"),
        ("temporal-isin", _temporal_graph,
         [n({"ts": IsIn(options=[datetime.date(2020, 1, 15), datetime.date(2020, 2, 1)])})], None,
         "pandas is EXACT-instant membership (not date-truncated); cross-precision is_in unproven"),
        ("temporal-gt-tzaware-datetimevalue", _temporal_graph,
         [n({"ts": GT(val=datetime.datetime(2020, 1, 15, 12, 0))})], None,
         "python datetime -> tz-tagged DateTimeValue (default UTC): silent tz-mishandling risk"),
        ("leapyear-nontemporal-col", _leapyear_graph, [n({"v": T.is_leap_year()})], None,
         "IsLeapYear on an int column has no proven parity"),
        ("leapyear-tzaware", _tzaware_graph, [n({"ts": T.is_leap_year()})], None,
         "local-time year boundary shifts under tz; parity unproven"),
        ("boundary-tzaware", _tzaware_graph, [n({"ts": T.is_month_start()})], None,
         "wall-clock calendar fields shift under tz; boundary parity unproven"),
        ("with-extend-mixed-list", lambda: _graph(8),
         [n(), rows(), with_([("vals", "[name, 99]")], extend=True)], "ok",
         "mixed-category (str+int) list literal: pandas builds the heterogeneous python list; polars would coerce/raise"),
        ("list-literal-mixed", lambda: _graph(8),
         [n(), rows(), with_([("vals", "[num, name]")], extend=True)], "ok",
         "mixed int+string list literal not provably parity-equal"),
        ("unwind-scalar", lambda: _graph(13),
         [n(), rows(), call("unwind", {"expr": "5", "as_": "x"})], "ok",
         "no ListLiteral lowering for a scalar; pandas broadcasts 1:1"),
        ("unwind-nested-list", lambda: _graph(14),
         [n(), rows(), call("unwind", {"expr": "[[1, 2], [3]]", "as_": "x"})], "ok",
         "list-of-lists is not an all-Literal ListLiteral; pandas one-level explodes to list cells"),
    ]


@pytest.mark.parametrize("label,graph_fn,query,why", _native_ok_query_cases(),
                         ids=[c[0] for c in _native_ok_query_cases()])
def test_query_runs_natively_on_polars(label, graph_fn, query, why):
    g = graph_fn()
    res = _run(g, query, "polars")
    assert res[0] == "ok", f"{label} must lower NATIVELY on polars ({why}), got {res}"
    _assert_invariant(g, query, f"native {label}")


@pytest.mark.parametrize("label,graph_fn,query,pandas_expect,why", _polars_nie_query_cases(),
                         ids=[c[0] for c in _polars_nie_query_cases()])
def test_query_honest_nie_on_polars(label, graph_fn, query, pandas_expect, why):
    g = graph_fn()
    if pandas_expect is not None:
        base = _run(g, query, "pandas")[0]
        assert (base == "ok") == (pandas_expect == "ok"), \
            f"{label}: pandas-oracle expectation drifted (expected {pandas_expect}, got {base})"
    assert _run(g, query, "polars")[0] == "nie", f"{label} must be an honest NIE on polars: {why}"


# ==== NATIVE unwind: scalar-literal list is the ONE native polars path (per-row cross-join via
# unwind_polars); list-column/string-expr/nested/scalar forms honest-NIE. Ledger subject. ====

@pytest.mark.parametrize("label,expr,as_", [
    ("ints", "[1, 2, 3]", "x"),
    ("strings", "['a', 'b']", "t"),
    ("with-null", "[1, NULL, 3]", "x"),
    ("singleton", "[42]", "x"),
])
def test_conformance_unwind_literal_list_native(label, expr, as_):
    """UNWIND of a scalar-literal list expands each active row by the list values (cypher per-row
    cross-join): value-equal to pandas AND native on polars (not an honest NIE)."""
    g = _graph(11)
    query = [n(), rows(), call("unwind", {"expr": expr, "as_": as_})]
    _assert_invariant(g, query, f"unwind literal {label}")
    if "polars" in _NONPANDAS_ENGINES:
        res = _run(g, query, "polars")
        assert res[0] == "ok", f"UNWIND {expr} must run NATIVELY on polars, got {res}"


def test_conformance_unwind_empty_list_drops_all_rows():
    """UNWIND [] AS x -> ZERO rows on every engine (documented pandas semantic); never a silent divergence."""
    g = _graph(12)
    query = [n(), rows(), call("unwind", {"expr": "[]", "as_": "x"})]
    _assert_invariant(g, query, "unwind empty list")


def test_conformance_unwind_chain_vs_cypher_consistent():
    """Chain UNWIND vs Cypher UNWIND: both native+equal or both NIE (no silent bridge)."""
    g = _graph(15)
    chain_q = [n(), rows(), call("unwind", {"expr": "[1, 2, 3]", "as_": "x"})]
    cypher_q = "MATCH (n) UNWIND [1, 2, 3] AS x RETURN n.id AS id, x AS x"
    _assert_invariant(g, chain_q, "unwind chain")
    _assert_invariant(g, cypher_q, "unwind cypher")


# ==== NATIVE row-pipeline OP SUBJECTS (close the ROW_PIPELINE_CALLS ledger gaps). Each op is
# native on the CHAIN surface via _try_native_row_op, and equally native on let() DAG because a
# LIST binding wraps into a Chain (ASTLet.__init__) -> chain_polars — NOT the bare single-call
# execute_row_pipeline_call path (that NIEs for chain-only ops; tracked on the call() axis).
# order_by pairs with limit on the UNIQUE id key so the surviving ROW SET depends on ordering
# (the order-insensitive sig can't see order alone); float `f` dropped to dodge repr noise.
_ROW_OP_CASES = [
    ("rows",       [n(), call("rows", {"table": "nodes"})]),
    ("skip",       [n(), rows(), call("skip", {"value": 5})]),
    ("limit",      [n(), rows(), call("limit", {"value": 3})]),
    ("distinct",   [n(), rows(), call("distinct", {})]),
    ("drop_cols",  [n(), rows(), call("drop_cols", {"cols": ["f", "name"]})]),
    ("order_by",   [n(), rows(), call("order_by", {"keys": [("id", "desc")]}),
                    call("limit", {"value": 5})]),
    ("select",     [n(), rows(), call("select", {"items": [("nid", "id"), ("n2", "num + 1")]})]),
    ("return_",    [n(), rows(), call("return_", {"items": [("nid", "id"), ("k", "flag")]})]),
    ("where_rows", [n(), rows(), call("where_rows", {"expr": "num > 50"})]),
    ("group_by",   [n(), rows(), call("group_by", {"keys": ["flag"],
                    "aggregations": [("c", "count"), ("s", "sum", "num")]})]),
    ("count_table", [n(), rows(), call("count_table", {"table": "nodes", "alias": "cnt"})]),
]


@pytest.mark.parametrize("label,query", _ROW_OP_CASES)
def test_conformance_rowop_chain(label, query):
    """Labeled CHAIN subject per row op: parity-or-NIE on every engine AND native on polars (no lazy NIE / bridge)."""
    g = _graph(20)
    _assert_invariant(g, query, f"rowop chain {label}")
    if "polars" in _NONPANDAS_ENGINES:
        res = _run(g, query, "polars")
        assert res[0] == "ok", f"row op {label} must run NATIVELY on polars (chain), got {res}"
        out = g.gfql(query, engine="polars")
        probe = out._nodes if out._nodes is not None else out._edges
        assert "polars" in type(probe).__module__, f"row op {label} returned non-polars (silent bridge!)"


@pytest.mark.parametrize("label,query", _ROW_OP_CASES)
def test_conformance_rowop_dag(label, query):
    """SAME op via let() DAG (list binding -> Chain -> chain_polars): native + parity too, no silent bridge/divergence."""
    _assert_invariant(_graph(20), let({"a": query}), f"rowop dag {label}")


# group_by is in the 4-engine _ROW_OP_CASES above — cuDF Series-truthiness bug (#1663 finding 4)
# FIXED by projecting to key+value cols before grouping; pandas == cudf == polars == polars-gpu.


def test_cudf_list_literal_order_matches_pandas():
    """#1663 finding 1 FIXED: cuDF list-literal keeps element order (order-deterministic host build, not groupby-collect)."""
    if "cudf" not in _NONPANDAS_ENGINES:
        pytest.skip("cudf not available")
    nd = pd.DataFrame({"id": np.arange(4), "num": [10, 20, 30, 40]})
    ed = pd.DataFrame({"s": [0], "d": [1], "eid": [0]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")
    q = [n(), rows(), call("select", {"items": [("id", "id"), ("lst", "[num, num + 1, 99]")]})]
    assert _run(g, q, "cudf") == _run(g, q, "pandas"), "cudf list-literal order must match pandas (#1663)"


def test_cudf_tostring_float_matches_pandas():
    """#1663 finding 2 FIXED: cuDF toString(float) string-matches pandas (host round-trip via the
    pandas float-repr oracle), incl. the scientific-notation boundary 1e20."""
    if "cudf" not in _NONPANDAS_ENGINES:
        pytest.skip("cudf not available")
    nd = pd.DataFrame({"id": np.arange(5), "f": [0.1, 1.5, -2.0, 1e10, 1e20]})
    ed = pd.DataFrame({"s": [0], "d": [1], "eid": [0]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")
    q = [n(), rows(), call("select", {"items": [("id", "id"), ("sf", "toString(f)")]})]
    assert _run(g, q, "cudf") == _run(g, q, "pandas"), "cudf toString(float) must match pandas (#1663)"
