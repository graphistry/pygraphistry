"""A seeded point query returns the same rows whether or not the index can answer its edge predicate.

Narrowing the candidate edges on codes is an internal choice, so the boundary a CALLER sees is
meant to be invisible: on either side of it the same query returns the same rows, in the same
order, or raises the same error. Every case states the rows it expects outright, and then checks
them against gather-then-filter, the route that answered these queries before the narrowing
existed.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, e_reverse, is_in, n, rows, select
from graphistry.tests.compute.gfql.routes.switch import routes_off

# Module scope, so the lane without polars SKIPS this file instead of failing to import it.
pl = pytest.importorskip("polars")

pytestmark = pytest.mark.route_engaged("polars-point-rows", "point-rows")

# Seed 1's edges are deliberately NON-CONTIGUOUS and not in ascending `w`, so a route that
# gathered or ordered differently shows up as reordered rows rather than as an equal set.
# `amount` is a float and `note` is high-null: neither is a column the category index codes.
NODES = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "label__Person": [True, True, False, False, False, False],
    "name": ["ann", "bob", "paris", "rome", "widget", "gadget"],
})
EDGES = pd.DataFrame({
    "s": [1, 2, 1, 2, 1, 1, 2],
    "d": [5, 3, 5, 5, 4, 6, 6],
    "type": ["BOUGHT", "VISITED", "BOUGHT", "BOUGHT", "VISITED", "BOUGHT", "BOUGHT"],
    "w": [50, 20, 30, 40, 10, 60, 70],
    "amount": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
    "note": ["a", None, "b", None, "c", None, "d"],
})

SEED_1_ALL = [(5, 50), (5, 30), (4, 10), (6, 60)]
SEED_1_BOUGHT = [(5, 50), (5, 30), (6, 60)]


def _indexed(nodes=NODES, edges=EDGES):
    g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    return g.gfql_index_all(engine="polars")


def _ops(seed, edge_match, edge=e_forward):
    return [
        n(seed, name="p"),
        edge(edge_match, name="e"),
        n({}, name="c"),
        rows(),
        select([("cid", "c.id"), ("ew", "e.w")]),
    ]


def _gather_then_filter(g, ops):
    """Run one query on the route that answered before the narrowing existed."""
    import graphistry.compute.gfql.lazy.engine.polars.chain_specializations.point_rows as point_rows

    from graphistry.compute.chain_fast_paths import _index_edge_rows
    from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars

    def unnarrowed(g_, adj, seed_ids, xp, engine, edges, edge_match):
        gathered = _index_edge_rows(adj, seed_ids, xp, engine, edges, preserve_input_order=True)
        return None if gathered is None else filter_by_dict_polars(gathered, edge_match)

    original = point_rows._gathered_edges_matching
    point_rows._gathered_edges_matching = unnarrowed
    try:
        return g.gfql(ops, engine="polars")._nodes
    finally:
        point_rows._gathered_edges_matching = original


def _narrowed_the_edges(ops):
    """Whether the code compare answered for the EDGE frame — it also serves node seeds."""
    from graphistry.compute.gfql.index import array_bindings

    answered = []
    original = array_bindings._positions_via_category_index

    def watching(*args, **kwargs):
        result = original(*args, **kwargs)
        role = args[1] if len(args) > 1 else kwargs.get("role")
        if role == "edges":
            answered.append(result is not None)
        return result

    array_bindings._positions_via_category_index = watching
    try:
        _indexed().gfql(ops, engine="polars")
    finally:
        array_bindings._positions_via_category_index = original
    return any(answered)


# The index CAN answer these, so the narrowing decides the surviving rows.
ANSWERABLE = [
    ("one survivor of several", {"id": 1}, {"type": "VISITED"}, e_forward, [(4, 10)]),
    ("several survivors, source order kept", {"id": 1}, {"type": "BOUGHT"}, e_forward, SEED_1_BOUGHT),
    ("reverse direction", {"id": 6}, {"type": "BOUGHT"}, e_reverse, [(1, 60), (2, 70)]),
    ("two coded columns at once", {"id": 1}, {"type": "BOUGHT", "w": 30}, e_forward, [(5, 30)]),
    ("a high-null column is still coded", {"id": 1}, {"note": "b"}, e_forward, [(5, 30)]),
    ("a value the column never holds", {"id": 1}, {"type": "NOPE"}, e_forward, []),
    ("a seed with no outgoing edges", {"id": 5}, {"type": "BOUGHT"}, e_forward, []),
    ("a seed that matches no node", {"id": 999}, {"type": "BOUGHT"}, e_forward, []),
]

# The index CANNOT answer these, so the canonical filter decides the surviving rows.
NOT_ANSWERABLE = [
    ("no predicate at all", {"id": 1}, None, e_forward, SEED_1_ALL),
    ("an empty predicate", {"id": 1}, {}, e_forward, SEED_1_ALL),
    ("a float column", {"id": 1}, {"amount": 3.5}, e_forward, [(5, 30)]),
    ("one coded column and one not", {"id": 1}, {"type": "BOUGHT", "amount": 3.5}, e_forward, [(5, 30)]),
    ("a non-scalar predicate", {"id": 1}, {"type": is_in(["BOUGHT"])}, e_forward, SEED_1_BOUGHT),
    ("a column the frame does not have", {"id": 1}, {"absent": "x"}, e_forward, []),
    ("several seeds, which this route declines", {"label__Person": True}, {"type": "BOUGHT"},
     e_forward, [(5, 50), (5, 30), (6, 60), (5, 40), (6, 70)]),
]


@pytest.mark.parametrize("case,seed,edge_match,edge,expected", ANSWERABLE + NOT_ANSWERABLE)
def test_the_rows_are_the_same_on_both_sides_of_the_answerable_boundary(
    case, seed, edge_match, edge, expected,
):
    ops = _ops(seed, edge_match, edge)
    served = _indexed().gfql(ops, engine="polars")._nodes
    assert served.rows() == expected, case
    unnarrowed = _gather_then_filter(_indexed(), ops)
    assert served.rows() == unnarrowed.rows(), f"{case}: the two routes disagree"
    assert served.columns == unnarrowed.columns
    assert served.schema == unnarrowed.schema


def test_a_null_predicate_matches_nothing_because_a_null_code_is_never_queried():
    ops = _ops({"id": 1}, {"note": None})
    assert _indexed().gfql(ops, engine="polars")._nodes.rows() == []
    assert _gather_then_filter(_indexed(), ops).rows() == []


def test_a_cross_type_scalar_raises_the_same_error_on_both_routes():
    ops = _ops({"id": 1}, {"type": 1})
    with pytest.raises(Exception) as served_error:
        _indexed().gfql(ops, engine="polars")
    with pytest.raises(type(served_error.value)):
        _gather_then_filter(_indexed(), ops)


def test_parallel_edges_between_one_pair_are_all_kept():
    edges = pd.DataFrame({
        "s": [1, 1, 1], "d": [2, 2, 3],
        "type": ["BOUGHT", "BOUGHT", "BOUGHT"], "w": [1, 2, 3],
    })
    g = _indexed(nodes=NODES.head(3), edges=edges)
    assert g.gfql(_ops({"id": 1}, {"type": "BOUGHT"}), engine="polars")._nodes.rows() == [
        (2, 1), (2, 2), (3, 3),
    ]


def test_a_graph_with_no_index_at_all_answers_the_same():
    plain = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    with routes_off([]):
        assert plain.gfql(_ops({"id": 1}, {"type": "BOUGHT"}), engine="polars")._nodes.rows() == (
            SEED_1_BOUGHT
        )


@pytest.mark.parametrize("case,seed,edge_match,edge,expected", ANSWERABLE)
def test_the_answerable_cases_really_did_narrow(case, seed, edge_match, edge, expected):
    """The one implementation assertion: without it every case above could pass unnarrowed."""
    assert _narrowed_the_edges(_ops(seed, edge_match, edge)), case


@pytest.mark.parametrize("case,seed,edge_match,edge,expected", NOT_ANSWERABLE)
def test_the_unanswerable_cases_really_did_decline(case, seed, edge_match, edge, expected):
    assert not _narrowed_the_edges(_ops(seed, edge_match, edge)), case
