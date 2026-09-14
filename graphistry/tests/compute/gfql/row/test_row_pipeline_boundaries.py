"""Both-sides pins for every boundary ``compute/gfql/row/pipeline.py`` decides.

Each boundary gets the case just INSIDE it and the case just OUTSIDE it, so a
gate that drifts in either direction goes red:

==============================  =================================  ==========================
boundary                        inside                             outside
==============================  =================================  ==========================
trail tracking engages          plain var-length MATCH             ``shortestPath`` (BFS)
undirected flip-twin dedupe     a self-loop is ONE binding         a non-loop keeps BOTH
same-edge return trip           parallel edge -> legal             the same edge -> illegal
walk scratch columns            live inside the pipeline           never reach a result
arithmetic null mask            computed NaN compares by IEEE      a genuine null stays null
polars var-length gate          served with pandas parity          typed decline, named
==============================  =================================  ==========================

Every expectation is hand-computed from the fixture edge list under openCypher
trail semantics (a relationship binds at most once per path; nodes may repeat).
No expectation was produced by running the code, and no engine is used as
another engine's oracle -- pandas, polars and cuDF are each checked against the
same hand-written literal.

A polars decline is never a silent pass: ``POLARS_DECLINES`` says which shapes
polars is allowed to decline and which words the message must contain, and
:func:`_assert_value_or_named_decline` fails a shape that declines off-table AND
a tabled shape that quietly starts serving.
"""
import os
from typing import Dict, Tuple

import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.identifiers import (
    TRAIL_COLUMN_PREFIX,
    WALK_CURRENT_COL,
    WALK_SCRATCH_COLUMNS,
    is_trail_column,
    is_walk_scratch_column,
)

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
cudf_only = pytest.mark.skipif(
    "TEST_CUDF" not in os.environ, reason="cuDF lane: set TEST_CUDF=1 (e.g. dgx-spark)"
)

ENGINES = [
    "pandas",
    pytest.param("polars", marks=polars_only),
    pytest.param("cudf", marks=cudf_only),
]

# --- fixtures (the edge list IS the oracle's input; ``e<i>`` names the i-th row) ---

#: e1 a->b1, e2 a->b2, e3 b1->c, e4 b2->c, e5 c->d
DIAMOND = (
    pd.DataFrame({"id": ["a", "b1", "b2", "c", "d"]}),
    pd.DataFrame({
        "s": ["a", "a", "b1", "b2", "c"],
        "d": ["b1", "b2", "c", "c", "d"],
        "type": ["KNOWS"] * 5,
    }),
)

#: e1 a->b only, so the sole undirected 2-hop candidate has to reuse e1.
SINGLE = (
    pd.DataFrame({"id": ["a", "b"]}),
    pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["KNOWS"]}),
)

#: e1 a->b, e2 a->b -- a return trip over the OTHER parallel edge is a legal trail.
PARA2 = (
    pd.DataFrame({"id": ["a", "b"]}),
    pd.DataFrame({"s": ["a", "a"], "d": ["b", "b"], "type": ["KNOWS"] * 2}),
)

#: e1 0->0 (self-loop) and e2 0->1, so both sides of the flip-twin rule show at once.
LOOP_AND_EDGE = (
    pd.DataFrame({"id": [0, 1]}),
    pd.DataFrame({"s": [0, 0], "d": [0, 1], "type": ["REL"] * 2}),
)

#: p1->p2->p3->p4 plus a disconnected q1.
LINE = (
    pd.DataFrame({"id": ["p1", "p2", "p3", "p4", "q1"]}),
    pd.DataFrame({"s": ["p1", "p2", "p3"], "d": ["p2", "p3", "p4"], "type": ["KNOWS"] * 3}),
)

#: f = 5.0 / NULL / 0.5 -- the three arithmetic-null-mask cases in one table.
NUMS = (
    pd.DataFrame({"id": ["n1", "n2", "n3"], "f": [5.0, None, 0.5]}),
    pd.DataFrame({"s": ["n1"], "d": ["n3"], "type": ["KNOWS"]}),
)

_VARLEN_GAP = "polars chain engine supports single-hop and multi-hop edges"
_ROWS_GAP = "does not yet natively support cypher row op"

#: query -> the words polars' decline MUST contain. Absent = polars must serve it.
POLARS_DECLINES: Dict[str, Tuple[str, ...]] = {
    "MATCH (x {id:'a'})-[*2]-(y) RETURN y.id AS y": (_VARLEN_GAP, "undirected min_hops>1"),
    "MATCH (x)-[*1..2]->(m)-[]->(y) RETURN x.id AS a, y.id AS b": (_VARLEN_GAP,),
    "MATCH (x {id:'a'})-[*1..2]->(m)-[]->(y) RETURN y.id AS y": (_VARLEN_GAP,),
    "MATCH (a {id:'p1'}), (b {id:'p4'}), p = shortestPath((a)-[*]-(b)) RETURN length(p) AS y":
        (_ROWS_GAP, "engine='pandas'"),
    "MATCH (x)-[*2..3]->(y) RETURN y.id AS y": ("#1748", "hop-gated"),
    "MATCH (x {id:'a'})-[*2..3]->(y) RETURN y.id AS y": ("#1748", "hop-gated"),
}


def _graph(fixture, engine):
    nodes, edges = fixture
    if engine == "polars":
        return graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    if engine == "cudf":
        import cudf
        return graphistry.nodes(cudf.from_pandas(nodes), "id").edges(cudf.from_pandas(edges), "s", "d")
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _rows(fixture, query, engine) -> pd.DataFrame:
    out = _graph(fixture, engine).gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


def _assert_value_or_named_decline(fixture, query, engine, check):
    """Run ``check`` on the result, or -- only for a tabled polars shape -- require
    a decline whose message NAMES the gap. Never a silent skip in either
    direction: an off-table decline fails, and a tabled shape that starts
    serving fails so the table cannot rot."""
    expected_tokens = POLARS_DECLINES.get(query) if engine == "polars" else None
    try:
        got = _rows(fixture, query, engine)
    except NotImplementedError as exc:
        assert expected_tokens is not None, f"unexpected {engine} decline for {query}: {exc}"
        missing = [tok for tok in expected_tokens if tok not in str(exc)]
        assert not missing, f"decline did not name {missing}: {exc}"
        return
    assert expected_tokens is None, f"{engine} now serves a tabled decline; update the table: {query}"
    check(got)


def _bag_check(expected, col="y"):
    def check(got):
        assert sorted(str(v) for v in got[col]) == expected
    return check


# ===========================================================================
# Boundary 1: trail tracking engages on a plain MATCH, NOT in shortestPath mode
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_plain_varlen_binds_each_relationship_at_most_once(engine):
    """INSIDE. PARA2 ``-[*2]-`` from a: hop 1 reaches b over e1 and over e2; hop 2
    may only leave b over the edge that branch has NOT bound, so each returns to
    a. Bag is [a, a] -- walk semantics would also emit b (a->b->a->b) plus the
    two same-edge returns."""
    _assert_value_or_named_decline(
        PARA2, "MATCH (x {id:'a'})-[*2]-(y) RETURN y.id AS y", engine, _bag_check(["a", "a"]))


@pytest.mark.parametrize("engine", ENGINES)
def test_plain_varlen_same_edge_return_trip_is_not_a_trail(engine):
    """OUTSIDE the parallel-edge case, same query shape. SINGLE ``-[*2]-`` from a:
    the only 2-hop candidate a->b->a must reuse e1, so NO row survives."""
    _assert_value_or_named_decline(
        SINGLE, "MATCH (x {id:'a'})-[*2]-(y) RETURN y.id AS y", engine, _bag_check([]))


@pytest.mark.parametrize("engine", ENGINES)
def test_shortest_path_mode_keeps_bfs_length_without_trail_tracking(engine):
    """OUTSIDE. shortestPath does not take the trail lane; LINE p1..p4 is 3 hops."""
    _assert_value_or_named_decline(
        LINE,
        "MATCH (a {id:'p1'}), (b {id:'p4'}), p = shortestPath((a)-[*]-(b)) RETURN length(p) AS y",
        engine,
        lambda got: [int(v) for v in got["y"]] == [3],
    )


# ===========================================================================
# Boundary 2: the undirected flip-twin dedupe is a SELF-LOOP rule
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_undirected_self_loop_is_one_binding_and_non_loop_keeps_both(engine):
    """BOTH SIDES in one fixture. LOOP_AND_EDGE under ``(a)-[r]-(b)``: e1 (0->0)
    yields ONE binding because its two orientations agree on (from, to, edge);
    e2 (0->1) yields TWO, (0,1) and (1,0). Three rows -- not two, not four."""
    def check(got):
        pairs = sorted((str(r["y"]), str(r["z"])) for r in got.to_dict("records"))
        assert pairs == [("0", "0"), ("0", "1"), ("1", "0")]
    _assert_value_or_named_decline(
        LOOP_AND_EDGE, "MATCH (a)-[r]-(b) RETURN a.id AS y, b.id AS z", engine, check)


# ===========================================================================
# Boundary 3: walk scratch columns live INSIDE the pipeline, never in a result
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_multi_element_varlen_result_carries_no_walk_scratch_column(engine):
    """INSIDE the pipeline the pattern binds a ``__gfql_trail_*`` column per hop;
    OUTSIDE (the returned frame) the columns are exactly the projected aliases.

    DIAMOND ``(x)-[*1..2]->(m)-[]->(y)`` enumerated over distinct-edge paths:
    (a,b1|e1)+e3, (a,b2|e2)+e4, (b1,c|e3)+e5, (b2,c|e4)+e5, (a,c|e1,e3)+e5,
    (a,c|e2,e4)+e5. The (c,d|e5) prefix dies because d has no out-edge."""
    def check(got):
        assert sorted((str(r["a"]), str(r["b"])) for r in got.to_dict("records")) == [
            ("a", "c"), ("a", "c"), ("a", "d"), ("a", "d"), ("b1", "d"), ("b2", "d"),
        ]
        assert list(got.columns) == ["a", "b"]
    _assert_value_or_named_decline(
        DIAMOND, "MATCH (x)-[*1..2]->(m)-[]->(y) RETURN x.id AS a, y.id AS b", engine, check)


def test_bindings_state_returns_no_trail_column_but_keeps_the_walk_cursor():
    """The RETURN projection would hide a leak, so pin the boundary where it is
    decided. ``_gfql_connected_bindings_state`` OWNS the drop: the per-hop
    ``__gfql_trail_*`` columns are consumed there and must not escape, while
    ``__current__`` is the cursor its caller still needs -- dropping everything
    would break the join, dropping nothing leaks scratch into every downstream
    op. Same six paths as the query pin above."""
    from graphistry.compute.ast import e_forward, n as node_op
    from graphistry.compute.gfql.row.pipeline import _RowPipelineAdapter

    ops = [node_op(name="x"), e_forward(min_hops=1, max_hops=2),
           node_op(name="m"), e_forward(), node_op(name="y")]
    state, alias_frames = _RowPipelineAdapter(_graph(DIAMOND, "pandas"))._gfql_connected_bindings_state(ops)

    assert len(state) == 6
    assert [c for c in state.columns if is_trail_column(str(c))] == []
    assert WALK_CURRENT_COL in state.columns
    assert sorted(alias_frames) == ["m", "x", "y"]


def test_walk_scratch_vocabulary_is_closed_over_its_predicate():
    """The externed vocabulary and its predicate cannot drift apart."""
    assert all(is_walk_scratch_column(col) for col in WALK_SCRATCH_COLUMNS)
    assert is_walk_scratch_column(f"{TRAIL_COLUMN_PREFIX}0__")
    assert is_walk_scratch_column(f"{TRAIL_COLUMN_PREFIX}17__")
    assert not is_walk_scratch_column(f"{TRAIL_COLUMN_PREFIX}x__")
    assert not is_walk_scratch_column("__gfql_trail__")
    assert not is_walk_scratch_column("from")


# ===========================================================================
# Boundary 4: hop depths mix in one result (the pad-before-concat alignment)
# ===========================================================================


@cudf_only
@pytest.mark.parametrize("query,expected", [
    ("MATCH (x {id:'a'})-[*1..2]->(m)-[]->(y) RETURN y.id AS y", ["c", "c", "d", "d"]),
    ("MATCH (x {id:'a'})-[*2]->(m)-[]->(y) RETURN y.id AS y", ["d", "d"]),
], ids=["mixed_depth", "single_depth"])
def test_cudf_range_varlen_keeps_rows_from_every_hop_depth(query, expected):
    """cuDF-only on purpose: its concat aligns schemas strictly, so hop frames of
    DIFFERENT trail width need an explicit pad or the shallow rows vanish --
    pandas and polars null-fill and cannot see the bug (their values for the
    mixed-depth query are already pinned in test_path_trail_semantics.py).

    INSIDE (mixed depth) DIAMOND from a: hop-1 prefixes (a,b1|e1) and (a,b2|e2)
    reach c,c; hop-2 prefixes (a,c|e1,e3) and (a,c|e2,e4) reach d,d.
    OUTSIDE (single depth) ``*2`` gives one frame width, so d,d with no pad."""
    got = _rows(DIAMOND, query, "cudf")
    assert sorted(str(v) for v in got["y"]) == expected


# ===========================================================================
# Boundary 5: a computed NaN is a VALUE; a genuine null stays null
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_computed_nan_comparison_is_false_not_null(engine):
    """INSIDE the arithmetic-operand branch. ``n.f % 0.0`` is NaN for 5.0 and for
    0.5, and ``NaN > 1`` is FALSE, so ``NOT (...)`` is TRUE and both rows stay.
    n2 (f IS NULL) propagates null through the arithmetic; ``NOT null`` is null
    and WHERE drops it."""
    _assert_value_or_named_decline(
        NUMS, "MATCH (n) WHERE NOT (n.f % 0.0 > 1) RETURN n.id AS y", engine,
        _bag_check(["n1", "n3"]))


@pytest.mark.parametrize("engine", ENGINES)
def test_computed_nan_comparison_matches_nothing_in_positive_form(engine):
    """The same expression un-negated: ``NaN > 1`` is false for n1/n3 and null for
    n2, so NO row matches. Pins that the override makes the comparison FALSE
    rather than merely non-null."""
    _assert_value_or_named_decline(
        NUMS, "MATCH (n) WHERE n.f % 0.0 > 1 RETURN n.id AS y", engine, _bag_check([]))


@pytest.mark.parametrize("engine", ENGINES)
def test_computed_nan_override_applies_to_the_right_operand_too(engine):
    """The override is per-operand, so the RIGHT side needs its own pin: with the
    arithmetic moved across the comparison, ``1 < n.f % 0.0`` is still false for
    n1/n3 (NaN) and null for n2, so ``NOT (...)`` keeps exactly n1 and n3."""
    _assert_value_or_named_decline(
        NUMS, "MATCH (n) WHERE NOT (1 < n.f % 0.0) RETURN n.id AS y", engine,
        _bag_check(["n1", "n3"]))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("MATCH (n) WHERE NOT (n.f > 1) RETURN n.id AS y", ["n3"]),
    ("MATCH (n) WHERE n.f > 1 RETURN n.id AS y", ["n1"]),
], ids=["negated", "plain"])
def test_non_arithmetic_operand_keeps_plain_null_semantics(query, expected, engine):
    """OUTSIDE the arithmetic branch. A bare property reference takes no override,
    so n2's null drops under BOTH polarities. Green before this PR as well --
    kept as the fence that stops the override widening to non-arithmetic
    operands, where it would turn null into false."""
    _assert_value_or_named_decline(NUMS, query, engine, _bag_check(expected))


# ===========================================================================
# Boundary 6: the polars variable-length gate -- served vs TYPED decline
# ===========================================================================


@polars_only
def test_polars_serves_unseeded_min_one_varlen_with_pandas_parity():
    """INSIDE the gate. DIAMOND ``(x)-[*1..2]->(y)`` over all starts: length 1
    gives b1,b2,c,c,d and length 2 gives c,c (via b1/b2) and d,d (b1->c->d and
    b2->c->d) -- nine rows, c four times and d three times."""
    expected = ["b1", "b2", "c", "c", "c", "c", "d", "d", "d"]
    query = "MATCH (x)-[*1..2]->(y) RETURN y.id AS y"
    _assert_value_or_named_decline(DIAMOND, query, "polars", _bag_check(expected))
    _assert_value_or_named_decline(DIAMOND, query, "pandas", _bag_check(expected))


@polars_only
@pytest.mark.parametrize("query", sorted(q for q in POLARS_DECLINES if "#1748" in POLARS_DECLINES[q]))
def test_polars_declines_min_hops_above_one_by_name(query):
    """OUTSIDE the gate. The decline must NAME the missing capability -- a bare
    failure is indistinguishable from a crash, and a silent skip would make
    every polars cell above vacuous."""
    _assert_value_or_named_decline(DIAMOND, query, "polars", _bag_check([]))
