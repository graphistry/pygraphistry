"""The point path narrows its candidate edges on codes before building a frame.

A seeded point query gathers the seed's incident edges and then filters them. When the
edge predicate is one the category index can answer, the candidates are row POSITIONS and
the filter runs on codes, so only the surviving rows are ever materialized. Anything the
index cannot answer falls back to gather-then-filter, which stays the oracle: every case
here is checked against that route and must agree exactly.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n, rows, select
from graphistry.tests.compute.gfql.routes.switch import routes_off

# Module scope, so the lane without polars SKIPS this file instead of failing to import it.
pl = pytest.importorskip("polars")

pytestmark = pytest.mark.route_engaged("polars-point-rows", "point-rows")

NODES = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "label__Person": [True, False, False, False, False],
    "label__City": [False, True, True, False, False],
    "name": ["ann", "paris", "rome", "widget", "gadget"],
})
EDGES = pd.DataFrame({
    "s": [1, 1, 1, 1],
    "d": [2, 3, 4, 5],
    "type": ["IS_LOCATED_IN", "VISITED", "BOUGHT", "BOUGHT"],
    "w": [1, 2, 3, 4],
})


def _graph(nodes=NODES, edges=EDGES):
    g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    return g.gfql_index_all(engine="polars")


def _both(g, ops):
    """(served, canonical) where canonical is gather-then-filter, the unchanged route."""
    import graphistry.compute.gfql.lazy.engine.polars.chain_specializations.point_rows as point_rows

    served = g.gfql(ops, engine="polars")._nodes
    original = point_rows._gathered_edges_matching

    def without_index(g_, adj, seed_ids, xp, engine, edges, edge_match):
        from graphistry.compute.chain_fast_paths import _index_edge_rows
        from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars
        gathered = _index_edge_rows(adj, seed_ids, xp, engine, edges, preserve_input_order=True)
        return None if gathered is None else filter_by_dict_polars(gathered, edge_match)

    point_rows._gathered_edges_matching = without_index
    try:
        canonical = g.gfql(ops, engine="polars")._nodes
    finally:
        point_rows._gathered_edges_matching = original
    return served, canonical


def _ops(edge_match, tail_filter=None):
    return [
        n({"id": 1, "label__Person": True}, name="p"),
        e_forward(edge_match, name="e"),
        n(tail_filter if tail_filter is not None else {}, name="c"),
        rows(),
        select([("cid", "c.id"), ("cname", "c.name")]),
    ]


@pytest.mark.parametrize("edge_match,expect_rows", [
    ({"type": "IS_LOCATED_IN"}, 1),   # the indexed predicate: one surviving edge
    ({"type": "BOUGHT"}, 2),          # indexed, more than one survivor
    ({"type": "NOPE"}, 0),            # indexed value the column never holds
])
def test_an_indexed_edge_predicate_answers_exactly_as_gather_then_filter(edge_match, expect_rows):
    served, canonical = _both(_graph(), _ops(edge_match))
    assert served.columns == canonical.columns
    assert served.schema == canonical.schema
    assert served.rows() == canonical.rows()
    assert canonical.height == expect_rows, "the oracle moved; the case no longer means what it says"


@pytest.mark.parametrize("edge_match", [
    None,                    # no predicate at all
    {},                      # empty predicate
    {"w": 1},                # an integer column with no category index (high cardinality path)
    {"type": 1},             # cross-type scalar against a string column
    {"absent": "x"},         # a column the edge frame does not have
    {"type": "IS_LOCATED_IN", "w": 1},  # one indexed column and one that is not
])
def test_a_predicate_the_index_cannot_answer_falls_back_and_still_agrees(edge_match):
    """Declining must be invisible in the answer, only in how it got there."""
    g = _graph()
    ops = _ops(edge_match)
    try:
        served, canonical = _both(g, ops)
    except Exception as error:  # a malformed predicate must fail identically on both routes
        import graphistry.compute.gfql.lazy.engine.polars.chain_specializations.point_rows as point_rows
        original = point_rows._gathered_edges_matching
        point_rows._gathered_edges_matching = lambda *a, **k: None
        try:
            with pytest.raises(type(error)):
                g.gfql(ops, engine="polars")
        finally:
            point_rows._gathered_edges_matching = original
        return
    assert served.columns == canonical.columns
    assert served.rows() == canonical.rows()


def test_the_narrowing_actually_engages_on_the_indexed_predicate():
    """Without this the suite above could pass with the narrowing never taken."""
    from graphistry.compute.gfql.index import array_bindings

    seen = {"served": 0}
    original = array_bindings._positions_via_category_index

    def counting(*args, **kwargs):
        result = original(*args, **kwargs)
        seen["served"] += result is not None
        return result

    array_bindings._positions_via_category_index = counting
    try:
        out = _graph().gfql(_ops({"type": "IS_LOCATED_IN"}), engine="polars")
    finally:
        array_bindings._positions_via_category_index = original
    assert seen["served"], "the code-compare narrowing never answered; the cases prove nothing"
    assert out._nodes.height == 1


def test_results_are_unchanged_without_any_index():
    """No index at all means no narrowing, and the same answer."""
    g = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    with routes_off([]):
        out = g.gfql(_ops({"type": "IS_LOCATED_IN"}), engine="polars")._nodes
    assert out.rows() == _graph().gfql(_ops({"type": "IS_LOCATED_IN"}), engine="polars")._nodes.rows()
