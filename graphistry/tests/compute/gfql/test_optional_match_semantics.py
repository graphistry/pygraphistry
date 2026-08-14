"""Round-003 OPTIONAL MATCH semantics pins (#1891).

openCypher contract: OPTIONAL MATCH keeps every row produced by the preceding
clauses; rows whose optional pattern finds no match are KEPT with the optional
aliases bound to NULL (left-join semantics, never inner-join).

Every expected value in this file is HAND-COMPUTED against that contract on
the shared fixture below. Cross-engine agreement is NOT evidence of
correctness here: every wrong-value shape originally pinned as a strict xfail
was engine-AGREEING (pandas and polars returned the same wrong answer), which
is exactly why parity suites never saw them (#1891). All 31 strict-xfail pins
flipped green with the #1891 fix (single-node-seed and pure-carry-WITH shapes
routed onto the connected optional-match left-join lowering; typed-null
empty-arm schema fill on polars; aggregate-aware reentry null fill).

Layout:
- section A -- formerly silent-wrong / bare-crash shapes, now pinned green
- green pins -- hand-verified-correct bright spots that must not regress
- gate-or-keep-seeds sweep -- every seed+OPTIONAL projection shape must either
  raise a typed gate / honest NotImplementedError or keep unmatched seeds, so
  FUTURE gate-bypass shapes fail the suite the day they appear
- message audit -- typed gates must not assert falsehoods about the query

Provenance: plans/gfql-release-amplification/rounds/round-003/findings/agent-01/
(probe_matrix.py + cells.jsonl in the plans repo; 88 hand-scored cells).
"""
import json
import math

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]


# ---------------------------------------------------------------- fixture
# persons: 0 alice(score 5), 1 bob(9), 2 carol(7), 3 <null-name>(2)
# things:  10 t1(v=100), 11 t2(v=200), 12 t3(v=300)
# edges: alice-L->t1, alice-L->t2, bob-H->t1, alice-K->bob, carol-K->noname, t1-X->t3
def _nodes_pd() -> pd.DataFrame:
    return pd.DataFrame({
        "id":    [0, 1, 2, 3, 10, 11, 12],
        "kind":  ["person"] * 4 + ["thing"] * 3,
        "name":  ["alice", "bob", "carol", None, "t1", "t2", "t3"],
        "score": [5, 9, 7, 2, None, None, None],
        "v":     [None, None, None, None, 100, 200, 300],
        "label__Person": [True] * 4 + [False] * 3,
        "label__Thing":  [False] * 4 + [True] * 3,
    })


def _edges_pd() -> pd.DataFrame:
    return pd.DataFrame({
        "s": [0, 0, 1, 0, 2, 10],
        "d": [10, 11, 10, 1, 3, 12],
        "t": ["L", "L", "H", "K", "K", "X"],
    })


def _run(query: str, engine: str, edges=None) -> pd.DataFrame:
    nodes = _nodes_pd()
    if edges is None:
        edges = _edges_pd()
    if engine == "polars":
        g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    else:
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    out = g.gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


def _scalar(x):
    """Engine-neutral scalar: None==NaN, ints not floats, lists sorted."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if hasattr(x, "item"):
        try:
            return _scalar(x.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(x, (list, tuple)) or type(x).__name__ == "ndarray":
        return sorted((_scalar(i) for i in x), key=lambda z: (z is None, str(z)))
    return x


def _key(rec):
    return json.dumps(rec, sort_keys=True, default=str)


def _records(df: pd.DataFrame, ordered: bool = False):
    recs = [{k: _scalar(v) for k, v in r.items()} for r in df.to_dict("records")]
    return recs if ordered else sorted(recs, key=_key)


def _assert_rows(df: pd.DataFrame, expected, ordered: bool = False) -> None:
    got = _records(df, ordered=ordered)
    exp = expected if ordered else sorted(expected, key=_key)
    assert got == exp, f"got {got}, expected {exp}"


# ===========================================================================
# A. Formerly silent-wrong / bare-crash shapes (#1891), now pinned green
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_optional_match_seed_ungrouped_count_star_counts_null_rows(engine):
    """B8: the discriminating probe -- grouped shapes can pass by luck, an
    ungrouped count(*) cannot. 5 rows = alice x2 + 3 null-extended seeds."""
    q = ("MATCH (p {kind:'person'}) OPTIONAL MATCH (p)-[{t:'L'}]->(x) "
         "RETURN count(*) AS c")
    _assert_rows(_run(q, engine), [{"c": 5}])


@pytest.mark.parametrize("engine", ENGINES)
def test_optional_match_seed_zero_arm_aggregate_keeps_all_seeds(engine):
    """B3: every seed row must survive a fully-unmatched arm with c=0."""
    q = ("MATCH (p {kind:'person'}) OPTIONAL MATCH (p)-[{t:'NOPE'}]->(x) "
         "RETURN p.name AS n, count(x) AS c")
    _assert_rows(_run(q, engine), [
        {"n": "alice", "c": 0}, {"n": "bob", "c": 0},
        {"n": "carol", "c": 0}, {"n": None, "c": 0},
    ])


@pytest.mark.parametrize("engine", ENGINES)
def test_optional_match_alias_only_projection_null_extends(engine):
    """B10: plain `RETURN x.v` (no aggregate) also used to sail past the seed
    gate (plan.source_alias was x, not the seed) and inner-join to [100, 200]."""
    q = ("MATCH (p {kind:'person'}) OPTIONAL MATCH (p)-[{t:'L'}]->(x) "
         "RETURN x.v AS v")
    _assert_rows(_run(q, engine), [
        {"v": 100}, {"v": 200}, {"v": None}, {"v": None}, {"v": None},
    ])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("proj,expected", [
    ("sum(x.v) AS a",
     [{"n": "alice", "a": 300}, {"n": "bob", "a": 0},
      {"n": "carol", "a": 0}, {"n": None, "a": 0}]),
    ("count(*) AS a",
     [{"n": "alice", "a": 2}, {"n": "bob", "a": 1},
      {"n": "carol", "a": 1}, {"n": None, "a": 1}]),
], ids=["sum_zero_for_unmatched", "count_star_one_for_unmatched"])
def test_optional_match_seed_sum_and_count_star_grouped(proj, expected, engine):
    """B4 + B6: unmatched seeds keep their group row -- sum()=0, count(*)=1."""
    q = ("MATCH (p {kind:'person'}) OPTIONAL MATCH (p)-[{t:'L'}]->(x) "
         "RETURN p.name AS n, " + proj)
    _assert_rows(_run(q, engine), expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_with_prefix_optional_zero_arm_keeps_carried_props(engine):
    """E6: the broken reentry concat had the right row count (4) with every
    value wrong (all-null rows) -- so the pin compares the FULL record set,
    not the count."""
    q = ("MATCH (p {kind:'person'}) WITH p OPTIONAL MATCH (p)-[{t:'NOPE'}]->(x) "
         "RETURN p.name AS n, x.v AS v")
    _assert_rows(_run(q, engine), [
        {"n": "alice", "v": None}, {"n": "bob", "v": None},
        {"n": "carol", "v": None}, {"n": None, "v": None},
    ])


@polars_only
def test_polars_optional_zero_arm_same_answer_as_matched_arm():
    """F-03: identical query text, only the data changes (bob's H edge
    removed). 'Runs natively on polars' must be a property of the QUERY --
    production queries die the day an optional arm comes back empty."""
    q = ("MATCH (m {kind:'person'})-[{t:'K'}]->(p) OPTIONAL MATCH (p)-[{t:'H'}]->(x) "
         "RETURN p.name AS n, x.v AS v")
    # arm-matching data: polars answers correctly (green, CN1)
    _assert_rows(_run(q, "polars"), [{"n": "bob", "v": 100}, {"n": None, "v": None}])
    # arm-empty data: formerly NotImplementedError "cypher row op 'select'"
    edges = _edges_pd()
    edges = edges[~((edges["s"] == 1) & (edges["t"] == "H"))].reset_index(drop=True)
    _assert_rows(_run(q, "polars", edges=edges),
                 [{"n": "bob", "v": None}, {"n": None, "v": None}])


@polars_only
def test_polars_nested_optional_zero_mid_arm_no_bare_crash():
    """H2: pins the CONTRACT (polars matches the pandas oracle). Formerly a
    raw polars.exceptions.SchemaError (Null-typed join key from the empty mid
    arm); fixed by the typed-null empty-arm schema fill. The oracle itself is
    asserted (hand-verified correct)."""
    q = ("MATCH (m {kind:'person'})-[{t:'K'}]->(p) "
         "OPTIONAL MATCH (p)-[{t:'NOPE'}]->(x) OPTIONAL MATCH (x)-[{t:'X'}]->(z) "
         "RETURN p.name AS n, x.v AS xv, z.v AS zv")
    expected = [{"n": "bob", "xv": None, "zv": None},
                {"n": None, "xv": None, "zv": None}]
    oracle = _run(q, "pandas")
    _assert_rows(oracle, expected)  # pandas side is a bright spot; keep it honest
    _assert_rows(_run(q, "polars"), expected)


@pytest.mark.parametrize("seed_filter", ["{kind:'person'}", "{name:'alice'}"],
                         ids=["four_row_prefix", "single_row_prefix"])
def test_with_carried_scalar_aggregate_after_optional_reentry(seed_filter):
    """F-04/E5: the aggregate projection rebuild recomputes group keys from the
    match frame, which no longer carries the WITH scalar. The single-row-prefix
    variant pins that the bug is the aggregate rebuild, NOT the N>1 prefix
    guard (that guard never gets a say for aggregate shapes)."""
    q = ("MATCH (p " + seed_filter + ") WITH p, p.score AS s "
         "OPTIONAL MATCH (p)-[{t:'L'}]->(x) RETURN s, count(x) AS c")
    expected = ([{"s": 5, "c": 2}] if seed_filter == "{name:'alice'}" else
                [{"s": 5, "c": 2}, {"s": 9, "c": 0}, {"s": 7, "c": 0}, {"s": 2, "c": 0}])
    _assert_rows(_run(q, "pandas"), expected)


# ===========================================================================
# B. Green pins: hand-verified-correct bright spots
# ===========================================================================


def _parity_or_nie(q: str, engine: str, expected) -> None:
    """Green-pin helper for shapes polars currently declines honestly: pandas
    must match the hand-computed answer; polars must either match it or raise
    a clean NotImplementedError (parity-or-error contract) -- a future
    silent-wrong polars route cannot land unseen."""
    if engine == "polars":
        try:
            out = _run(q, "polars")
        except NotImplementedError:
            return
        _assert_rows(out, expected)
    else:
        _assert_rows(_run(q, engine), expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_optional_match_first_clause_zero_match_yields_null_row(engine):
    """A1 + A4: OPTIONAL MATCH as the first clause; zero matches yield one
    null row, and count(x) over it is 0 -- both engines."""
    _assert_rows(_run("OPTIONAL MATCH (a {kind:'zzz'}) RETURN a.name AS n", engine),
                 [{"n": None}])
    _assert_rows(_run("OPTIONAL MATCH (a)-[{t:'NOPE'}]->(x) RETURN count(x) AS c", engine),
                 [{"c": 0}])


@pytest.mark.parametrize("engine", ENGINES)
def test_optional_match_where_inside_keeps_null_extended_rows(engine):
    """C1 + C4: WHERE inside the optional clause filters MATCHES but keeps the
    rows null-extended (openCypher), including WHERE on the seed alias only.
    pandas is the oracle; polars is pinned parity-or-NotImplementedError."""
    expected = [{"n": "bob", "v": None}, {"n": None, "v": None}]
    _parity_or_nie(
        "MATCH (m {kind:'person'})-[{t:'K'}]->(p) OPTIONAL MATCH (p)-[{t:'H'}]->(x) "
        "WHERE x.v > 150 RETURN p.name AS n, x.v AS v", engine, expected)
    _parity_or_nie(
        "MATCH (m {kind:'person'})-[{t:'K'}]->(p) OPTIONAL MATCH (p)-[{t:'H'}]->(x) "
        "WHERE p.score > 20 RETURN p.name AS n, x.v AS v", engine, expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_optional_match_two_arms_from_single_node_seed_left_joins(engine):
    """D2, full 5-record compare, both engines. THE D2 PARADOX: this two-arm
    single-node-seed shape is answered correctly, while its one-arm twin (B1)
    is gate-blocked as unsupported -- so the null-extension mechanism the gate
    claims missing demonstrably exists. A gate relaxation must flip the B1
    gate pin (in the sweep below / section-A pins), never this test."""
    q = ("MATCH (p {kind:'person'}) "
         "OPTIONAL MATCH (p)-[{t:'H'}]->(x) OPTIONAL MATCH (p)-[{t:'L'}]->(y) "
         "RETURN p.name AS n, x.v AS xv, y.v AS yv")
    _assert_rows(_run(q, engine), [
        {"n": "alice", "xv": None, "yv": 100}, {"n": "alice", "xv": None, "yv": 200},
        {"n": "bob", "xv": 100, "yv": None}, {"n": "carol", "xv": None, "yv": None},
        {"n": None, "xv": None, "yv": None},
    ])


@pytest.mark.parametrize("engine", ENGINES)
def test_optional_match_order_by_null_placement(engine):
    """F1 + F2: ORDER BY an optional prop -- nulls last ASC, nulls first DESC
    (openCypher null placement), ordered compare, both engines."""
    base = ("MATCH (m {kind:'person'})-[{t:'K'}]->(p) OPTIONAL MATCH (p)-[{t:'H'}]->(x) "
            "RETURN p.id AS pid, x.v AS v ORDER BY v")
    _assert_rows(_run(base, engine),
                 [{"pid": 1, "v": 100}, {"pid": 3, "v": None}], ordered=True)
    _assert_rows(_run(base + " DESC", engine),
                 [{"pid": 3, "v": None}, {"pid": 1, "v": 100}], ordered=True)


@pytest.mark.parametrize("engine", ENGINES)
def test_optional_match_nested_arm_and_label_arm_matched(engine):
    """H1 + L1: nested optionality from an optional alias, and a label on the
    optional alias -- matched-arm data, both engines."""
    _assert_rows(_run(
        "MATCH (m {kind:'person'})-[{t:'K'}]->(p) "
        "OPTIONAL MATCH (p)-[{t:'H'}]->(x) OPTIONAL MATCH (x)-[{t:'X'}]->(z) "
        "RETURN p.name AS n, x.v AS xv, z.v AS zv", engine), [
        {"n": "bob", "xv": 100, "zv": 300}, {"n": None, "xv": None, "zv": None},
    ])
    _assert_rows(_run(
        "MATCH (m:Person)-[{t:'K'}]->(p) OPTIONAL MATCH (p)-[{t:'H'}]->(x:Thing) "
        "RETURN p.name AS n, x.v AS v", engine), [
        {"n": "bob", "v": 100}, {"n": None, "v": None},
    ])


@pytest.mark.parametrize("engine", ENGINES)
def test_with_carried_scalar_props_after_optional_reentry(engine):
    """F-04 control: scalar WITH carry + optional PROPS projection is a correct
    5-row left join with the scalar carried -- protects the working half next
    to the aggregate half (fixed above). polars: parity-or-NIE."""
    q = ("MATCH (p {kind:'person'}) WITH p, p.score AS s "
         "OPTIONAL MATCH (p)-[{t:'L'}]->(x) RETURN s, x.v AS v")
    _parity_or_nie(q, engine, [
        {"s": 5, "v": 100}, {"s": 5, "v": 200},
        {"s": 9, "v": None}, {"s": 7, "v": None}, {"s": 2, "v": None},
    ])


# ===========================================================================
# C. Gate-or-keep-seeds invariant sweep (the anti-blindness test)
# ===========================================================================
# Template: MATCH (p {kind:'person'}) OPTIONAL MATCH (p)-[{t:'L'}]->(x)
# RETURN <proj> (+ WITH-prefix variants). Invariant: the outcome is EITHER a
# typed GFQLValidationError / honest NotImplementedError OR an answer that
# still reflects the unmatched seeds -- never a silent inner-join. Future
# bypass shapes fail this suite the day they appear, without a hand-check.
# The formerly-bypassing cells (silent inner-joins) are green since the
# #1891 fix routed them onto the connected optional-match left-join lowering.


def _has_bob_row(recs):
    return any(r.get("n") == "bob" for r in recs)


_SEED_Q = "MATCH (p {kind:'person'}) OPTIONAL MATCH (p)-[{t:'L'}]->(x) RETURN "
_WITH_Q = "MATCH (p {kind:'person'}) WITH p OPTIONAL MATCH (p)-[{t:'L'}]->(x) RETURN "

SWEEP_CASES = [
    # gate fires today (typed) -- green invariant params
    pytest.param(_SEED_Q + "p.name AS n, x.v AS v", _has_bob_row, id="props"),
    pytest.param(_SEED_Q + "p.name AS n, collect(x.v) AS vs", _has_bob_row, id="collect"),
    pytest.param(_SEED_Q + "DISTINCT p.name AS n", _has_bob_row, id="distinct_seed"),
    # formerly gate-bypassed (silent inner-join), fixed by the connected
    # optional-match left-join route (#1891) -- green invariant params
    pytest.param(_SEED_Q + "p.name AS n, count(x) AS c", _has_bob_row, id="count"),
    pytest.param(_SEED_Q + "p.name AS n, sum(x.v) AS a", _has_bob_row, id="sum"),
    pytest.param(_SEED_Q + "p.name AS n, count(*) AS c", _has_bob_row,
                 id="count_star_grouped"),
    pytest.param(_SEED_Q + "count(*) AS c",
                 lambda recs: recs == [{"c": 5}],
                 id="count_star_ungrouped"),
    pytest.param(_SEED_Q + "x.v AS v",
                 lambda recs: sum(1 for r in recs if r.get("v") is None) == 3,
                 id="alias_only"),
    pytest.param(_WITH_Q + "p.name AS n, x.v AS v", _has_bob_row,
                 id="with_prefix_props"),
    pytest.param(_WITH_Q + "p.name AS n, count(x) AS c",
                 lambda recs: {"n": "bob", "c": 0} in recs,
                 id="with_prefix_count"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,seeds_kept", SWEEP_CASES)
def test_optional_match_seed_shapes_gate_or_keep_seeds(query, seeds_kept, engine):
    try:
        out = _run(query, engine)
    except GFQLValidationError:
        return  # typed gate: acceptable ("not yet supported" beats silent-wrong)
    except NotImplementedError:
        return  # honest engine decline: acceptable (parity-or-error by design)
    recs = _records(out)
    assert seeds_kept(recs), f"unmatched seeds silently dropped (inner-join): {recs}"


# ===========================================================================
# D. Message audit: typed gates must describe the query they reject (F-05)
# ===========================================================================


def test_optional_match_gate_messages_describe_the_query():
    """Typed gates are the good outcome, but their messages must not assert
    falsehoods (F-05). Two former offenders:
    - connected seed + count(x) used to raise 'aggregate ... must be
      top-level' when the aggregate IS a top-level RETURN projection; the
      shape is now served by the connected optional-match lowering, so pin
      the (hand-computed) answer instead.
    - the canonical anti-join used to carry the suggestion 'Use MATCH instead
      of OPTIONAL MATCH', a rewrite that CHANGES semantics (an anti-join can
      never be expressed with non-optional MATCH)."""
    out = _run("MATCH (m {kind:'person'})-[{t:'K'}]->(p) OPTIONAL MATCH (p)-[{t:'H'}]->(x) "
               "RETURN p.name AS n, count(x) AS c", "pandas")
    _assert_rows(out, [{"n": "bob", "c": 1}, {"n": None, "c": 0}])
    # canonical anti-join: still a typed decline, but it must not suggest a
    # semantics-changing rewrite and must not claim the aggregate placement
    # is wrong
    with pytest.raises(GFQLValidationError) as anti_err:
        _run("MATCH (m {kind:'person'})-[{t:'K'}]->(p) OPTIONAL MATCH (p)-[{t:'L'}]->(x) "
             "WITH p, x WHERE x IS NULL RETURN p.name AS n", "pandas")
    assert "Use MATCH instead of OPTIONAL MATCH" not in str(anti_err.value)
    assert "must be top-level" not in str(anti_err.value)


def test_optional_match_varlen_arm_residual_gate_is_honest():
    """Variable-length optional arms are declined by the connected left-join
    lowering, so they exercise the residual general path: each shape must
    either answer with unmatched seeds kept (the null-fill mechanism serves
    the props shape) or fail typed with a message that describes the actual
    limitation -- never the old 'return only the bound seed alias' text
    (false for projections that also referenced the optional alias) and never
    a silent inner-join."""
    for proj in ["p.name AS n, x.v AS v", "x.v AS v", "p.name AS n, count(x) AS c"]:
        try:
            out = _run("MATCH (p {kind:'person'}) OPTIONAL MATCH (p)-[*1..2]->(x) RETURN " + proj,
                       "pandas")
        except GFQLValidationError as err:
            msg = str(err)
            assert "return only the bound seed alias" not in msg, msg
            assert "OPTIONAL MATCH" in msg, msg
            continue
        recs = _records(out)
        # bob's only outgoing path is bob-H->t1(-X->t3); carol reaches only the
        # null-named person -- unmatched/null-extended seeds must be present
        assert any(r.get("n") == "carol" or r.get("v") is None for r in recs), recs


# ===========================================================================
# E. #1891-regression twins: undirected arm + arm WHERE + whole-row RETURN
# ===========================================================================
# The cudf-only test_string_cypher_formats_optional_match_projection_on_cudf
# was the single pin for this shape -- that lane-skew let the #1891 residual
# gate silently decline it on every engine. Pandas/polars twins pin it here.
# Fixture (mirrors the cudf test): s(:Single), a(:A num=42), b(:B num=46),
# c(:C); edges s->a, s->b, a->c, b->b.


def _run_labeled(query: str, engine: str) -> pd.DataFrame:
    nodes = pd.DataFrame({
        "id": ["s", "a", "b", "c"],
        "labels": [["Single"], ["A"], ["B"], ["C"]],
        "label__Single": [True, False, False, False],
        "label__A": [False, True, False, False],
        "label__B": [False, False, True, False],
        "label__C": [False, False, False, True],
        "num": pd.Series([pd.NA, 42, 46, pd.NA], dtype="Int64"),
    })
    edges = pd.DataFrame({
        "s": ["s", "s", "a", "b"],
        "d": ["a", "b", "c", "b"],
        "edge_id": ["rel_1", "rel_2", "rel_3", "rel_4"],
    })
    if engine == "polars":
        g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    else:
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    out = g.gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


@pytest.mark.parametrize("engine", ENGINES)
def test_undirected_arm_where_whole_row_projects_matched_entity(engine):
    """The regressed repro: undirected (s)-[r]-(m) reaches a (s->a) and b
    (s->b); WHERE m.num = 42 keeps only a -> exactly one row, m = a."""
    q = ("MATCH (n:Single) OPTIONAL MATCH (n)-[r]-(m) WHERE m.num = 42 "
         "RETURN m")
    _assert_rows(_run_labeled(q, engine), [{"m": "(:A {num: 42})"}])


@pytest.mark.parametrize("engine", ENGINES)
def test_undirected_arm_where_fails_null_extends_not_drops(engine):
    """WHERE belongs to the optional pattern: no neighbor of s has num=999,
    so the seed row survives with m = null -- never zero rows."""
    q = ("MATCH (n:Single) OPTIONAL MATCH (n)-[r]-(m) WHERE m.num = 999 "
         "RETURN m")
    _assert_rows(_run_labeled(q, engine), [{"m": None}])


@pytest.mark.parametrize("engine", ENGINES)
def test_directed_arm_where_whole_row_matched_and_no_edge_null(engine):
    """Directed sanity: (s)-[r]->(m) still reaches a (matched); (s)<-[r]-(m)
    has no incoming edge at all -> the no-edge arm null-extends too."""
    _assert_rows(
        _run_labeled("MATCH (n:Single) OPTIONAL MATCH (n)-[r]->(m) WHERE m.num = 42 RETURN m", engine),
        [{"m": "(:A {num: 42})"}])
    _assert_rows(
        _run_labeled("MATCH (n:Single) OPTIONAL MATCH (n)<-[r]-(m) WHERE m.num = 42 RETURN m", engine),
        [{"m": None}])
