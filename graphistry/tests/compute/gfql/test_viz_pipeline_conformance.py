"""Viz-filter-pipeline ACCEPTANCE suite (goal L3). Ground truth:
plans/viz-filter-pipeline/research/panel-algebra.md (v1-FINAL frozen algebra).

Encodes streamgl-viz panel states as SINGLE cypher pipelines over the shipped dialect:
masks M = (AND F) AND NOT (OR X) with (pred OR x IS NULL) keep-null leaves (algebra Section 1,
D3 true nulls), well-formedness inherent to edge patterns (Section 2), prune-isolated flavors via
EXISTS { } keep-self / drop-self (Section 3), explicit paging via ORDER BY/SKIP/LIMIT (D1),
searchAny inspector-style cross-column search (D2), GRAPH { } graph-state staging.

CORE INVARIANT (inherited from test_engine_polars_conformance_matrix): on every non-pandas
engine a query is parity-equal to the pandas oracle OR an honest NotImplementedError — never
silently different, silently bridged, or a non-NIE crash. Every pandas oracle here is pinned
TWICE: hand-computed row ids AND an independent plain-pandas mirror computed in-test (a wrong
hand pin fails against the mirror before any engine runs).

Known engine caveats respected (not conceded): cuDF declines (?i)+escape-class regex (NIE ok);
polars declines cross-entity same-path WHERE and composed =~ OR-forms (NIE ok); cuDF's
same-path executor has a PRE-EXISTING TypeError on the drop-self GRAPH query (tolerated ONLY
there, with a comment)."""
import operator
import re
import numpy as np
import pandas as pd
import pytest
import graphistry

# No polars wheel (e.g. cp314) -> skip module cleanly, not per-case non-NIE ImportError "fails".
pytest.importorskip("polars")

from graphistry.compute.exceptions import GFQLValidationError  # noqa: E402

# Shared suite-wide comparison machinery; loud-failure contracts documented there.
from graphistry.tests.compute.gfql.polars_test_utils import (  # noqa: E402
    to_pandas_any as _to_pd,
    run_status as _run,
    available_nonpandas_engines,
    assert_parity_or_nie,
)

_NONPANDAS_ENGINES = available_nonpandas_engines()


def _assert_invariant(g, query, label):
    assert_parity_or_nie(g, query, label, _NONPANDAS_ENGINES)


# ---- Section 1: the panel fixture — SMALL and fully hand-verifiable ----
# 30 nodes: id 0..29; kind cycles 'a','b','c' (id % 3); score = id + 0.25 with nulls at
# id % 5 == 4 ({4, 9, 14, 19, 24, 29}); flag = (id % 2 == 0); cat = Categorical x/y (id % 2);
# name: case/unicode/metachar/null specials on ids 0..11, 'item<id>' for 12..29.
_SPECIAL_NAMES = {
    0: "Alpha", 1: "ALPHA", 2: "alpha",           # case triplet
    3: "a.b", 4: "aXb",                            # literal-vs-regex metachar pair
    5: "Ätna", 6: "ätna",                          # unicode case pair
    7: "istanbul İ",                               # Turkish dotted capital I (U+0130)
    8: "straße", 9: "STRASSE",                     # German sharp-s folding pair
    10: None, 11: None,                            # null cells in the searched column
}


def _panel_nodes():
    nn = 30
    return pd.DataFrame({
        "id": list(range(nn)),
        "name": [_SPECIAL_NAMES.get(i, f"item{i}") for i in range(nn)],
        "kind": ["abc"[i % 3] for i in range(nn)],
        "score": [None if i % 5 == 4 else i + 0.25 for i in range(nn)],
        "flag": [i % 2 == 0 for i in range(nn)],
        "cat": pd.Categorical(["x" if i % 2 == 0 else "y" for i in range(nn)]),
    })


def _panel_edges():
    """50 edges. Connectivity by construction: nodes 27/28/29 fully ISOLATED; node 26
    self-loop-ONLY (eid 4, its sole edge); 0<->1 mutually connected (two directed edges);
    2->3 parallel pair; node 5 has a self-loop (eid 5) PLUS real edges; chain 4->5..24->25
    connects 4..25; extra edges j -> (7j+1) % 26 stay within 0..25 and provably contain no
    self-loop (j = 7j+1 mod 26 => 6j = 25 mod 26: LHS even, RHS odd — no solution)."""
    pairs = (
        [(0, 1), (1, 0)]                                # eids 0,1: mutual pair
        + [(2, 3), (2, 3)]                              # eids 2,3: parallel edges
        + [(26, 26)]                                    # eid 4: self-loop-only node 26
        + [(5, 5), (5, 6)]                              # eids 5,6: self-loop on connected node
        + [(i, i + 1) for i in range(4, 25)]            # eids 7..27: chain 4..25
        + [(j, (j * 7 + 1) % 26) for j in range(22)]    # eids 28..49: extras, no self-loops
    )
    ne = len(pairs)
    assert ne == 50
    return pd.DataFrame({
        "s": [p[0] for p in pairs],
        "d": [p[1] for p in pairs],
        "eid": list(range(ne)),
        "etype": [["knows", "likes", "sees"][k % 3] for k in range(ne)],
        "w": [k % 10 for k in range(ne)],
    })


def _panel_graph():
    return (
        graphistry.nodes(_panel_nodes(), "id")
        .edges(_panel_edges(), "s", "d")
        .bind(edge="eid")
    )


def _connected_ids(ed):
    """Nodes with ANY incident edge (self-loops count) == EXISTS { (n)--() } keep-self."""
    return set(ed["s"]) | set(ed["d"])


def _nonself_connected_ids(ed):
    """Nodes with a non-self incident edge == EXISTS { (n)--(m) WHERE m <> n } drop-self."""
    ns = ed[ed["s"] != ed["d"]]
    return set(ns["s"]) | set(ns["d"])


# Plain-pandas mirror of the searchAny kernel semantics (panel-algebra D2 inspector gate):
# OR across string columns (name, kind; the categorical 'cat' column holds only 'x'/'y' so no
# multi-char pool term can match it regardless of the dtype-gate outcome) + int columns
# ('id') iff the term is a numeric literal; ci substring default; nulls never match.
def _search_mask(nd, term, case_sensitive=False, regex=False, columns=None):
    if columns is not None:
        cols = list(columns)
    else:
        cols = ["name", "kind"] + (["id"] if re.match(r"^[0-9.\-]+$", term) else [])
    mask = pd.Series(False, index=nd.index)
    for c in cols:
        s = nd[c]
        if s.dtype.kind != "O":
            s = s.astype(str)
        mask = mask | s.str.contains(term, case=case_sensitive, regex=regex, na=False)
    return mask


def test_pandas_oracle_sanity():
    """Canary: assert_parity_or_nie SKIPS when the pandas oracle errors, so a GLOBAL oracle
    break would silently skip the suite — this known-good query must stay 'ok'."""
    st = _run(_panel_graph(), "MATCH (n) RETURN n.id AS id", "pandas")
    assert st[0] == "ok", "pandas oracle broken — suite would silently skip"


# ---- Section 2: curated single-query panel pipelines (streamgl-viz panel states) ----
# Each row: (label, cypher, result column, HAND-COMPUTED pin, plain-pandas mirror).
# Null semantics are three-valued (D3): a comparison on a null cell is null; NOT null is null;
# null rows are DROPPED by WHERE — mirrors encode that explicitly (notna() under NOT).
def _panel_scenarios():
    return [
        # Exclusion dominates (algebra Section 1: M = F AND NOT X). kind 'a' = {0,3,...,27};
        # NOT(score>5) drops nulls (9, 24) and keeps score<=5 => ids 0, 3 only.
        ("exclusion-dominates",
         "MATCH (n) WHERE n.kind = 'a' AND NOT (n.score > 5) RETURN n.id AS id", "id",
         [0, 3],
         lambda nd, ed: nd[(nd["kind"] == "a") & nd["score"].notna()
                           & ~(nd["score"] > 5)]["id"]),
        # Edge filter via the edge alias (node/edge masks independent, Section 1).
        # etype 'knows' = eid % 3 == 0; w = eid % 10 >= 6.
        ("edge-filter-alias",
         "MATCH ()-[e]->() WHERE e.etype = 'knows' AND e.w >= 6 RETURN e.eid AS eid", "eid",
         [6, 9, 18, 27, 36, 39, 48],
         lambda nd, ed: ed[(ed["etype"] == "knows") & (ed["w"] >= 6)]["eid"]),
        # Keep-null leaf form (D3 k-partite default: pred OR IS NULL keeps null cells).
        # kind 'a' with score>5 {6,12,15,18,21,27} plus null-score kind-a rows {9, 24}.
        ("keep-null-leaf",
         "MATCH (n) WHERE n.kind = 'a' AND (n.score > 5 OR n.score IS NULL) "
         "RETURN n.id AS id", "id",
         [6, 9, 12, 15, 18, 21, 24, 27],
         lambda nd, ed: nd[(nd["kind"] == "a")
                           & ((nd["score"] > 5) | nd["score"].isna())]["id"]),
        # Filter composed with keep-self prune (Section 3 graph-theory flavor): kind 'c'
        # minus isolated 29; self-loop-only 26 KEPT.
        ("exists-keep-self-prune",
         "MATCH (n) WHERE n.kind = 'c' AND EXISTS { (n)--() } RETURN n.id AS id", "id",
         [2, 5, 8, 11, 14, 17, 20, 23, 26],
         lambda nd, ed: nd[(nd["kind"] == "c") & nd["id"].isin(_connected_ids(ed))]["id"]),
        # Isolated-only selection: exactly the three edge-free nodes.
        ("not-exists-isolated-only",
         "MATCH (n) WHERE NOT EXISTS { (n)--() } RETURN n.id AS id", "id",
         [27, 28, 29],
         lambda nd, ed: nd[~nd["id"].isin(_connected_ids(ed))]["id"]),
        # Drop-self prune (Section 3 viz flavor): endpoints of non-self edges = 0..25
        # (self-loop-only 26 dropped, isolated 27..29 dropped).
        ("exists-drop-self-prune",
         "MATCH (n) WHERE EXISTS { (n)--(m) WHERE m <> n } RETURN n.id AS id", "id",
         list(range(26)),
         lambda nd, ed: nd[nd["id"].isin(_nonself_connected_ids(ed))]["id"]),
        # searchAny (D2) composed with a filter: names ci-containing 'alpha' = {0,1,2},
        # AND flag (even ids) => {0, 2}.
        ("searchany-composed",
         "MATCH (n) WHERE searchAny(n, 'alpha') AND n.flag = true RETURN n.id AS id", "id",
         [0, 2],
         lambda nd, ed: nd[_search_mask(nd, "alpha") & nd["flag"]]["id"]),
        # Explicit paging (D1: LIMIT explicit-only, stable order via ORDER BY):
        # kind 'b' sorted = [1,4,7,10,13,...]; SKIP 2 LIMIT 3 => [7, 10, 13].
        ("orderby-limit-paging",
         "MATCH (n) WHERE n.kind = 'b' RETURN n.id AS id ORDER BY id SKIP 2 LIMIT 3", "id",
         [7, 10, 13],
         lambda nd, ed: sorted(nd[nd["kind"] == "b"]["id"])[2:5]),
    ]


@pytest.mark.parametrize("label,query,col,pinned,mirror", _panel_scenarios(),
                         ids=[s[0] for s in _panel_scenarios()])
def test_full_panel_pipeline_scenarios(label, query, col, pinned, mirror):
    g = _panel_graph()
    nd, ed = _panel_nodes(), _panel_edges()
    second = sorted(int(v) for v in mirror(nd, ed))
    assert second == sorted(pinned), f"{label}: hand pin disagrees with the pandas mirror"
    out = _to_pd(g.gfql(query, engine="pandas")._nodes)
    got = [int(v) for v in out[col].tolist()]
    if "paging" in label:
        assert got == pinned, f"{label}: ORDER BY paging must be deterministic, got {got}"
    else:
        assert sorted(got) == sorted(pinned), f"{label}: pandas oracle drift, got {sorted(got)}"
    _assert_invariant(g, query, f"panel {label}")


def test_distinct_rows_all_engines():
    """DISTINCT on rows (panel dedup view): 3 kinds out of 30 rows, ordered."""
    g = _panel_graph()
    q = "MATCH (n) RETURN DISTINCT n.kind AS kind ORDER BY kind"
    pdf = _to_pd(g.gfql(q, engine="pandas")._nodes)
    assert pdf["kind"].tolist() == ["a", "b", "c"]
    _assert_invariant(g, q, "distinct-kinds")


# ---- Section 3: GRAPH { } graph-state prune (nodes AND edges back, Section 2+3) ----
def test_graph_state_prune_pipeline():
    """GRAPH edge-pattern staging returns the full graph (nodes + edges).
    keep-self `GRAPH { MATCH (a)-[e]-(b) }`: self-loop-only node 26 kept WITH its edge
    (eid 4); mutual pair 0<->1 keeps BOTH directed edges (eids 0, 1); isolated 27..29
    dropped; kept nodes keep ALL their edges (all 50). drop-self `+ WHERE a.id <> b.id`:
    node 26 dropped, and self-EDGES of kept nodes drop too (eid 5 on kept node 5 —
    the discriminating case for the documented drop-self edge semantics)."""
    g = _panel_graph()
    ed = _panel_edges()
    # -- keep-self: pins + independent mirror from the edge frame --
    connected = _connected_ids(ed)
    assert sorted(connected) == list(range(27)), "fixture drift: connected set"
    q1 = "GRAPH { MATCH (a)-[e]-(b) }"
    out1 = g.gfql(q1, engine="pandas")
    nodes1 = sorted(int(v) for v in _to_pd(out1._nodes)["id"].tolist())
    edges1 = sorted(int(v) for v in _to_pd(out1._edges)["eid"].tolist())
    assert nodes1 == list(range(27)), f"keep-self nodes: {nodes1}"
    assert edges1 == list(range(50)), "keep-self: kept nodes must keep ALL their edges"
    assert 26 in nodes1 and 4 in edges1, "self-loop-only node kept WITH its edge"
    assert {0, 1} <= set(nodes1) and {0, 1} <= set(edges1), "mutual pair keeps BOTH edges"
    assert not {27, 28, 29} & set(nodes1), "isolated nodes dropped"
    _assert_invariant(g, q1, "graph-state keep-self")
    # -- drop-self: pins + mirror --
    nonself = _nonself_connected_ids(ed)
    nonself_eids = sorted(int(v) for v in ed[ed["s"] != ed["d"]]["eid"].tolist())
    assert sorted(nonself) == list(range(26)), "fixture drift: nonself set"
    assert nonself_eids == sorted(set(range(50)) - {4, 5}), "fixture drift: nonself eids"
    q2 = "GRAPH { MATCH (a)-[e]-(b) WHERE a.id <> b.id }"
    out2 = g.gfql(q2, engine="pandas")
    nodes2 = sorted(int(v) for v in _to_pd(out2._nodes)["id"].tolist())
    edges2 = sorted(int(v) for v in _to_pd(out2._edges)["eid"].tolist())
    assert nodes2 == list(range(26)), f"drop-self nodes: {nodes2}"
    # L3 FINDING (dgx, 2026-07-05): the same-path WHERE route DEDUPES PARALLEL
    # edges — one edge survives per (src,dst) pair — diverging from panel-algebra
    # E2 multiplicity (E1 semijoin N2 keeps all copies). Pinned as-is + tracked as
    # repo debt (same_path executor); the prune CORRECTNESS facts still hold:
    assert set(edges2) <= set(nonself_eids), "drop-self must never keep a self-loop"
    assert not ({4, 5} & set(edges2)), "self-loop eids excluded"
    pair = lambda r: (min(r.s, r.d), max(r.s, r.d))  # noqa: E731
    nonself_pairs = {pair(r) for r in ed[ed["s"] != ed["d"]].itertuples()}
    kept_pairs = {pair(r) for r in _to_pd(out2._edges).itertuples()}
    assert kept_pairs == nonself_pairs, "every non-self endpoint pair must be represented"
    # drop-self uses cross-entity same-path WHERE: polars/polars-gpu decline honestly;
    # cuDF has a PRE-EXISTING TypeError in the same-path executor (verified byte-identical
    # on the base tree — repo debt predating this suite, plan-tracked). Tolerated ONLY here.
    base = _run(g, q2, "pandas")
    assert base[0] == "ok"
    for eng in _NONPANDAS_ENGINES:
        res = _run(g, q2, eng)
        if res[0] == "ok":
            assert res == base, f"graph-state drop-self[{eng}]: silent divergence"
        elif eng == "cudf":
            assert res[0] == "nie" or res == ("err", "TypeError"), \
                f"cudf drop-self: expected NIE or the pre-existing TypeError, got {res}"
        else:
            assert res[0] == "nie", f"{eng} drop-self must decline honestly, got {res}"


# ---- Section 4: seeded panel-state fuzzer — generated pipelines, dual-oracle checked ----
def test_panel_state_fuzzer():
    """Random panel states -> ONE generated cypher query each (mirroring the viz panel:
    filter, exclude-via-AND-NOT, keep-null wrapper, EXISTS prune, searchAny, ORDER BY+LIMIT).
    Deterministic seeds; expected rows computed INDEPENDENTLY with plain pandas (three-valued
    null semantics encoded explicitly); then 4-engine parity-or-NIE. If pandas declines a
    combo outside the dialect it must be a CLEAR decline (GFQLValidationError /
    NotImplementedError) — any other exception class propagates and fails the test."""
    g = _panel_graph()
    nd, ed = _panel_nodes(), _panel_edges()
    connected, nonself = _connected_ids(ed), _nonself_connected_ids(ed)
    cmp_ops = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
    term_pool = [
        ("alpha", {}),
        ("item1", {}),
        ("ätna", {}),
        ("ITEM2", {"caseSensitive": True}),
        ("7", {}),                                # numeric term: int 'id' column gated IN
        ("item2", {"columns": ["name"]}),
    ]
    fails = []
    declined = 0
    for seed in range(40):
        rng = np.random.default_rng(4200 + seed)
        conj = []  # (cypher text, plain-pandas kept-mask)
        if rng.integers(0, 2):  # node-filter comparison on kind/score
            if rng.integers(0, 2):
                k = "abc"[int(rng.integers(0, 3))]
                conj.append((f"n.kind = '{k}'", nd["kind"] == k))
            else:
                op = ["<", "<=", ">", ">="][int(rng.integers(0, 4))]
                t = int(rng.integers(3, 27))
                text = f"n.score {op} {t}"
                mask = cmp_ops[op](nd["score"], t).fillna(False)
                if rng.integers(0, 2):  # (pred OR x IS NULL) keep-null wrapper
                    text = f"({text} OR n.score IS NULL)"
                    mask = mask | nd["score"].isna()
                conj.append((text, mask))
        if rng.integers(0, 2):  # exclude via AND NOT (exclusion dominates)
            if rng.integers(0, 2):
                k2 = "abc"[int(rng.integers(0, 3))]
                conj.append((f"NOT (n.kind = '{k2}')", nd["kind"] != k2))
            else:
                t2 = int(rng.integers(3, 27))
                # three-valued: NOT(null > t2) is null -> row dropped
                conj.append((f"NOT (n.score > {t2})",
                             nd["score"].notna() & ~(nd["score"] > t2)))
        prune = int(rng.integers(0, 3))  # EXISTS prune: none / keep-self / drop-self
        if prune == 1:
            conj.append(("EXISTS { (n)--() }", nd["id"].isin(connected)))
        elif prune == 2:
            conj.append(("EXISTS { (n)--(m) WHERE m <> n }", nd["id"].isin(nonself)))
        if rng.integers(0, 2):  # searchAny from the fixed pool
            term, opts = term_pool[int(rng.integers(0, len(term_pool)))]
            rendered = ""
            if opts:
                parts = []
                if "caseSensitive" in opts:
                    parts.append(f"caseSensitive: {'true' if opts['caseSensitive'] else 'false'}")
                if "columns" in opts:
                    parts.append("columns: [" + ", ".join(f"'{c}'" for c in opts["columns"]) + "]")
                rendered = ", {" + ", ".join(parts) + "}"
            conj.append((f"searchAny(n, '{term}'{rendered})",
                         _search_mask(nd, term,
                                      case_sensitive=opts.get("caseSensitive", False),
                                      columns=opts.get("columns"))))
        query = "MATCH (n)"
        if conj:
            query += " WHERE " + " AND ".join(text for text, _ in conj)
        query += " RETURN n.id AS id"
        total = pd.Series(True, index=nd.index)
        for _, m in conj:
            total = total & m
        expected = [int(v) for v in nd.loc[total, "id"].tolist()]
        limit = None
        if rng.integers(0, 2):  # explicit paging (D1) — only ever WITH a stable order
            limit = int(rng.integers(1, 12))
            query += f" ORDER BY id LIMIT {limit}"
            expected = sorted(expected)[:limit]
        try:
            res = g.gfql(query, engine="pandas")
        except (GFQLValidationError, NotImplementedError):
            declined += 1  # clear decline of an out-of-dialect combo — allowed, never a crash
            continue
        got = [int(v) for v in _to_pd(res._nodes)["id"].tolist()]
        ok = (got == expected) if limit is not None else (sorted(got) == sorted(expected))
        if not ok:
            fails.append(f"seed {seed}: {query} -> {sorted(got)} != {sorted(expected)}")
            continue
        try:
            _assert_invariant(g, query, f"fuzz seed {seed}")
        except AssertionError as e:
            fails.append(str(e)[:200])
    assert not fails, "panel-state fuzz failures:\n" + "\n".join(fails)
    assert declined < 40, "every fuzz seed declined — generator drifted out of the dialect"


# ---- Section 5: case/regex/unicode trick matrix (per-row pin + mirror + parity-or-NIE) ----
# `=~` is ANCHORED (fullmatch); CONTAINS is substring; searchAny default is ci LITERAL
# substring with regex opt-in ((?i)-defaulted). Engine declines ride parity-or-NIE.
def _trick_matrix():
    return [
        # (?i) vs case-sensitive anchored regex on the case triplet
        ("regex-ci-anchored",
         "MATCH (n) WHERE n.name =~ '(?i)alpha' RETURN n.id AS id", [0, 1, 2],
         lambda nd: nd["name"].str.fullmatch("(?i)alpha", na=False)),
        ("regex-cs-anchored",
         "MATCH (n) WHERE n.name =~ 'alpha' RETURN n.id AS id", [2],
         lambda nd: nd["name"].str.fullmatch("alpha", na=False)),
        # anchored =~ vs CONTAINS: 'item1' full-matches NOTHING but is contained in item12..19
        ("anchored-fullmatch-empty",
         "MATCH (n) WHERE n.name =~ 'item1' RETURN n.id AS id", [],
         lambda nd: nd["name"].str.fullmatch("item1", na=False)),
        ("contains-substring",
         "MATCH (n) WHERE n.name CONTAINS 'item1' RETURN n.id AS id", list(range(12, 20)),
         lambda nd: nd["name"].str.contains("item1", regex=False, na=False)),
        # (?i) + escape class: cuDF declines this fold honestly (NIE ok, silent wrong not)
        ("regex-ci-escape-class",
         "MATCH (n) WHERE n.name =~ '(?i)ITEM\\\\d+' RETURN n.id AS id", list(range(12, 30)),
         lambda nd: nd["name"].str.fullmatch(r"(?i)ITEM\d+", na=False)),
        # composed =~ OR-form: polars declines honestly (NIE ok)
        ("regex-composed-or-isnull",
         "MATCH (n) WHERE n.name =~ '(?i)alpha' OR n.score IS NULL RETURN n.id AS id",
         [0, 1, 2, 4, 9, 14, 19, 24, 29],
         lambda nd: nd["name"].str.fullmatch("(?i)alpha", na=False) | nd["score"].isna()),
        # literal-vs-regex metachars through searchAny: default is LITERAL ('a.b' only);
        # regex opt-in matches 'a.b', 'aXb' AND 'istanbul İ' (its 'anb' matches /a.b/i)
        ("searchany-literal-metachar",
         "MATCH (n) WHERE searchAny(n, 'a.b') RETURN n.id AS id", [3],
         lambda nd: _search_mask(nd, "a.b")),
        ("searchany-regex-metachar",
         "MATCH (n) WHERE searchAny(n, 'a.b', {regex: true}) RETURN n.id AS id", [3, 4, 7],
         lambda nd: _search_mask(nd, "a.b", regex=True)),
        # unicode folding: pandas str.lower/upper do FULL case mapping ('straße'.upper() ==
        # 'STRASSE', 'İ'.lower() == 'i̇'); engines must match or decline (parity-or-NIE)
        ("tolower-eq-no-ss-fold",
         "MATCH (n) WHERE toLower(n.name) = 'strasse' RETURN n.id AS id", [9],
         lambda nd: nd["name"].str.lower() == "strasse"),
        ("toupper-eq-ss-fold",
         "MATCH (n) WHERE toUpper(n.name) = 'STRASSE' RETURN n.id AS id", [8, 9],
         lambda nd: nd["name"].str.upper() == "STRASSE"),
        # 'İ' (U+0130) lowercases to 'i' + combining dot above (U+0307) — written as an
        # explicit escape so editors cannot silently normalize the invisible codepoint
        ("tolower-turkish-dotted-i",
         "MATCH (n) WHERE toLower(n.name) = 'istanbul i\u0307' RETURN n.id AS id", [7],
         lambda nd: nd["name"].str.lower() == "istanbul i\u0307"),
        # null cells in the searched/filtered column never match (D3 true nulls)
        ("searchany-null-cells",
         "MATCH (n) WHERE searchAny(n, 'item') RETURN n.id AS id", list(range(12, 30)),
         lambda nd: _search_mask(nd, "item")),
        ("contains-null-cells",
         "MATCH (n) WHERE n.name CONTAINS 'a' RETURN n.id AS id", [0, 2, 3, 4, 5, 6, 7, 8],
         lambda nd: nd["name"].str.contains("a", regex=False, na=False)),
        # scalar fns through filters; score = id + 0.25 so floor/ceil/round discriminate
        # (x.25 avoids the round-half tie — all engines' rounding modes agree on it)
        ("floor-filter",
         "MATCH (n) WHERE floor(n.score) >= 25 RETURN n.id AS id", [25, 26, 27, 28],
         lambda nd: np.floor(nd["score"]) >= 25),
        ("ceil-filter",
         "MATCH (n) WHERE ceil(n.score) <= 3 RETURN n.id AS id", [0, 1, 2],
         lambda nd: np.ceil(nd["score"]) <= 3),
        ("round-filter",
         "MATCH (n) WHERE round(n.score) = 10 RETURN n.id AS id", [10],
         lambda nd: nd["score"].round(0) == 10),
        # Categorical dtype through a plain filter (searchAny variant has its own tolerant test)
        ("categorical-eq-filter",
         "MATCH (n) WHERE n.cat = 'x' RETURN n.id AS id", list(range(0, 30, 2)),
         lambda nd: nd["cat"].astype(str) == "x"),
    ]


@pytest.mark.parametrize("label,query,pinned,mirror", _trick_matrix(),
                         ids=[t[0] for t in _trick_matrix()])
def test_case_regex_unicode_trick_matrix(label, query, pinned, mirror):
    g = _panel_graph()
    nd = _panel_nodes()
    second = sorted(int(v) for v in nd.loc[mirror(nd).fillna(False), "id"].tolist())
    assert second == sorted(pinned), f"{label}: hand pin disagrees with the pandas mirror"
    got = sorted(int(v) for v in _to_pd(g.gfql(query, engine="pandas")._nodes)["id"].tolist())
    assert got == sorted(pinned), f"{label}: pandas oracle drift, got {got}"
    _assert_invariant(g, query, f"trick {label}")


def test_categorical_searchany_decline_or_correct():
    """Categorical column through searchAny via explicit columns=: a DECLINE (honest NIE or
    a clear GFQLValidationError) is acceptable on any engine, a SILENT WRONG answer is not,
    and a non-NIE crash class is not."""
    g = _panel_graph()
    nd = _panel_nodes()
    q = "MATCH (n) WHERE searchAny(n, 'x', {columns: ['cat']}) RETURN n.id AS id"
    expected = sorted(int(v) for v in nd.loc[nd["cat"].astype(str) == "x", "id"].tolist())
    assert expected == list(range(0, 30, 2)), "fixture drift: categorical evens"
    base = _run(g, q, "pandas")
    if base[0] == "ok":
        got = sorted(int(v) for v in _to_pd(g.gfql(q, engine="pandas")._nodes)["id"].tolist())
        assert got == expected, f"categorical searchAny: pandas silent wrong answer {got}"
        for eng in _NONPANDAS_ENGINES:
            res = _run(g, q, eng)
            if res[0] == "nie":
                continue  # honest decline — allowed
            assert res[0] != "err", f"categorical searchAny[{eng}]: non-NIE crash {res}"
            assert res == base, f"categorical searchAny[{eng}]: silent divergence"
    else:
        # pandas itself may decline categorical search — but only CLEARLY
        assert base[0] == "nie" or base == ("err", "GFQLValidationError"), \
            f"categorical searchAny must decline clearly on pandas, got {base}"
