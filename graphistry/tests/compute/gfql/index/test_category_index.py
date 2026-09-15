"""Category indexes answer scalar equality exactly as the canonical filter does.

The index turns a label or edge-type predicate into a code compare. Every case here
checks it against the filter it replaces, including the places where a code compare
could quietly disagree: nulls, a value the column never holds, boolean-versus-integer
conflation, and an index left over from a frame that has since changed.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.ast import e_forward, n, rows, select
from graphistry.compute.gfql.index.api import get_registry
from graphistry.compute.gfql.index.build import build_category_index
from graphistry.tests.compute.gfql.routes.switch import routes_off

pl = pytest.importorskip("polars")

ROUTE = "polars-bindings-select"


def _codes(frame, column):
    return build_category_index(frame, column, "nodes", Engine.POLARS)


@pytest.mark.parametrize("values,dtype", [
    ([True, False, True], pl.Boolean),
    ([True, None, False], pl.Boolean),
    (["a", "b", "a"], pl.String),
    (["a", None, "b"], pl.String),
    ([1, 2, 1], pl.Int64),
    ([1, None, 2], pl.Int64),
])
def test_codes_round_trip_every_indexable_dtype(values, dtype):
    frame = pl.DataFrame({"c": pl.Series(values, dtype=dtype)})
    index = _codes(frame, "c")
    assert index is not None
    assert index.n_rows == len(values)
    for row, value in enumerate(values):
        if value is None:
            # The null code is reserved: it is never handed out for a queried value.
            assert index.codes[row] not in set(index.value_codes.values())
        else:
            assert index.codes[row] == index.value_codes[value]


@pytest.mark.parametrize("column,reason", [
    ("floats", "float equality is not a category"),
    ("wide", "cardinality above the cap"),
    ("absent", "no such column"),
])
def test_declines_what_a_code_compare_should_not_answer(column, reason):
    frame = pl.DataFrame({
        "floats": [1.0, 2.0, 3.0],
        "wide": [str(value) for value in range(3)],
    })
    if column == "wide":
        frame = pl.DataFrame({"wide": [str(value) for value in range(200)]})
    assert _codes(frame, column) is None, reason


def test_declines_on_non_polars_engines():
    frame = pl.DataFrame({"c": [True, False]})
    assert build_category_index(frame, "c", "nodes", Engine.PANDAS) is None
    assert build_category_index(frame, "c", "nodes", Engine.CUDF) is None


def test_index_goes_stale_with_the_frame():
    nodes = pl.DataFrame({"id": [1, 2, 3], "kind": ["a", "b", "a"]})
    edges = pl.DataFrame({"s": [1, 2], "d": [2, 3]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql_index_all(engine="polars")
    registry = get_registry(g)
    assert registry.get_category_valid("nodes", "kind", g._nodes, Engine.POLARS) is not None
    # A different frame object with the same content is still a different binding.
    assert registry.get_category_valid("nodes", "kind", nodes.clone(), Engine.POLARS) is None
    # And a reshaped frame fails the fingerprint.
    assert registry.get_category_valid(
        "nodes", "kind", nodes.head(2), Engine.POLARS,
    ) is None
    assert registry.get_category_valid("nodes", "kind", None, Engine.POLARS) is None


NODES = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "label__Person": [True, None, None, None, True, True],
    "label__Message": [None, True, True, True, None, None],
    "kind": ["person", "msg", "msg", "msg", "person", "person"],
    "flag": [1, 0, 1, 0, 1, 0],
    "name": list("abcdef"),
})
EDGES = pd.DataFrame({
    "s": [2, 3, 4, 4, 2],
    "d": [1, 1, 1, 5, 6],
    "type": ["HAS_CREATOR", "HAS_CREATOR", "HAS_CREATOR", "LIKES", "LIKES"],
    "w": [1, 2, 3, 4, 5],
})


def _graph(nodes=NODES, edges=EDGES):
    g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    return g.gfql_index_all(engine="polars")


def _both(g, ops, **kwargs):
    kwargs.setdefault("index_policy", "force")
    served = g.gfql(ops, engine="polars", **kwargs)
    with routes_off([ROUTE]):
        canonical = g.gfql(ops, engine="polars", **kwargs)
    return served._nodes, canonical._nodes


@pytest.mark.parametrize("seed_filter,edge_match,end_filter", [
    ({"id": 1, "label__Person": True}, {"type": "HAS_CREATOR"}, {"label__Message": True}),
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"kind": "msg"}),
    ({"id": 1}, {}, {}),
    ({"id": 1}, {"type": "NOPE"}, {}),                       # value the column never holds
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"kind": "ghost"}),  # endpoint value never held
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"flag": 1}),        # integer category
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"label__Person": True}),  # all-null-for-these rows
])
def test_indexed_predicates_match_the_canonical_filter(seed_filter, edge_match, end_filter):
    ops = [
        n(seed_filter, name="a"),
        e_forward(edge_match or None, name="e"),
        n(end_filter or None, name="b"),
        rows(),
        select([("aid", "a"), ("bid", "b"), ("bkind", "b.kind")]),
    ]
    served, canonical = _both(_graph(), ops)
    assert served.columns == canonical.columns
    assert served.schema == canonical.schema
    assert served.rows() == canonical.rows()


def test_boolean_predicate_never_matches_an_integer_column_code():
    """``True == 1`` in Python; the canonical filter does not conflate them, nor may we."""
    nodes = NODES.assign(flag=[1, 0, 1, 0, 1, 0])
    ops = [
        n({"id": 1}, name="a"), e_forward({"type": "HAS_CREATOR"}, name="e"),
        n({"flag": True}, name="b"), rows(), select([("bid", "b")]),
    ]
    served, canonical = _both(_graph(nodes=nodes), ops)
    assert served.rows() == canonical.rows()


def test_null_rows_are_never_matched_by_a_scalar_predicate():
    frame = pl.DataFrame({"c": pl.Series([True, None, False], dtype=pl.Boolean)})
    index = _codes(frame, "c")
    assert index is not None
    true_code = index.value_codes[True]
    false_code = index.value_codes[False]
    assert index.codes[1] != true_code and index.codes[1] != false_code
    # The canonical filter agrees: a null row survives neither predicate.
    assert frame.filter(pl.col("c") == True).height == 1  # noqa: E712
    assert frame.filter(pl.col("c") == False).height == 1  # noqa: E712


def test_index_all_builds_categories_and_skips_the_rest():
    g = _graph()
    registry = get_registry(g)
    node_cols = set(registry.category_cols("nodes"))
    edge_cols = set(registry.category_cols("edges"))
    assert {"label__Person", "label__Message", "kind", "flag"} <= node_cols
    assert "type" in edge_cols
    # `name` is all-distinct here but still under the cap, so it is indexed; the cap is
    # what bounds the work, and a wide column is skipped.
    wide = NODES.assign(name=[f"n{value}" for value in range(len(NODES))])
    assert "name" in set(get_registry(_graph(nodes=wide)).category_cols("nodes"))


def test_registry_can_drop_categories():
    g = _graph()
    registry = get_registry(g)
    assert registry.category_cols("nodes")
    assert registry.without_categories().category_cols("nodes") == ()


def _endpoint_fact(g):
    return get_registry(g).get_endpoint_rows_valid(
        str(g._source), str(g._destination), str(g._node), g._edges, g._nodes, Engine.POLARS,
    )


def test_endpoint_rows_resolve_every_edge_endpoint():
    g = _graph()
    fact = _endpoint_fact(g)
    assert fact is not None
    ids = g._nodes.get_column("id").to_numpy()
    assert list(ids[fact.src_rows]) == list(g._edges.get_column("s").to_numpy())
    assert list(ids[fact.dst_rows]) == list(g._edges.get_column("d").to_numpy())


def test_endpoint_rows_decline_when_an_endpoint_has_no_node():
    nodes = pl.DataFrame({"id": [1, 2], "kind": ["a", "b"]})
    edges = pl.DataFrame({"s": [1, 9], "d": [2, 2]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql_index_all(engine="polars")
    assert _endpoint_fact(g) is None


def test_endpoint_rows_go_stale_with_either_frame():
    g = _graph()
    registry = get_registry(g)
    src, dst, node = str(g._source), str(g._destination), str(g._node)
    assert registry.get_endpoint_rows_valid(src, dst, node, g._edges, g._nodes, Engine.POLARS) is not None
    # A clone of either frame is a different binding, and a reshape fails the fingerprint.
    assert registry.get_endpoint_rows_valid(src, dst, node, g._edges.clone(), g._nodes, Engine.POLARS) is None
    assert registry.get_endpoint_rows_valid(src, dst, node, g._edges, g._nodes.clone(), Engine.POLARS) is None
    assert registry.get_endpoint_rows_valid(src, dst, node, g._edges, None, Engine.POLARS) is None
    assert registry.without_endpoint_rows().endpoint_rows == {}


@pytest.mark.parametrize("seed_filter,edge_match,end_filter", [
    ({"id": 1, "label__Person": True}, {"type": "HAS_CREATOR"}, {"label__Message": True}),
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"kind": "msg"}),
    ({"id": 1}, {}, {}),
    ({"id": 1}, {"type": "HAS_CREATOR"}, {"kind": "ghost"}),
])
def test_results_match_with_and_without_the_endpoint_rows_fact(
    seed_filter, edge_match, end_filter, monkeypatch,
):
    """The fact is an accelerator: dropping it must not change a single row."""
    import graphistry.compute.gfql.index.registry as registry_mod
    ops = [
        n(seed_filter, name="a"), e_forward(edge_match or None, name="e"),
        n(end_filter or None, name="b"), rows(),
        select([("aid", "a"), ("bid", "b"), ("bkind", "b.kind")]),
    ]
    g = _graph()
    with_fact = g.gfql(ops, engine="polars", index_policy="force")._nodes
    monkeypatch.setattr(
        registry_mod.GfqlIndexRegistry, "get_endpoint_rows_valid",
        lambda *a, **k: None,
    )
    without_fact = g.gfql(ops, engine="polars", index_policy="force")._nodes
    assert with_fact.columns == without_fact.columns
    assert with_fact.schema == without_fact.schema
    assert with_fact.rows() == without_fact.rows()


def test_temporal_text_verdicts_match_a_direct_scan():
    frame = pl.DataFrame({
        "plain": ["hello", "a (paren) here", "no"],
        "constructor": ["date({year: 1984})", "x", "y"],
        "number": [1, 2, 3],
    })
    g = graphistry.nodes(frame.with_columns(pl.Series("id", [1, 2, 3])), "id").edges(
        pl.DataFrame({"s": [1], "d": [2]}), "s", "d",
    ).gfql_index_all(engine="polars")
    fact = get_registry(g).get_temporal_text_valid("nodes", g._nodes, Engine.POLARS)
    assert fact is not None
    assert fact.verdicts["plain"] is False
    assert fact.verdicts["constructor"] is True
    assert "number" not in fact.verdicts  # not a String column


def test_temporal_text_fact_goes_stale_with_the_frame():
    g = _graph()
    registry = get_registry(g)
    assert registry.get_temporal_text_valid("nodes", g._nodes, Engine.POLARS) is not None
    assert registry.get_temporal_text_valid("nodes", g._nodes.clone(), Engine.POLARS) is None
    assert registry.get_temporal_text_valid("nodes", None, Engine.POLARS) is None
    assert registry.without_temporal_text().temporal_text == {}


def test_projection_of_constructor_text_declines_to_the_canonical_route():
    """A column holding constructor text must not reach the caller raw."""
    nodes = pd.DataFrame({
        "id": [1, 2, 3],
        "label__Person": [True, None, None],
        "label__Message": [None, True, True],
        "kind": ["person", "msg", "msg"],
        "flag": [1, 0, 1],
        "name": ["a", "b", "c"],
        "when": ["date({year: 1984})", "date({year: 1985})", "plain"],
    })
    edges = pd.DataFrame({"s": [2, 3], "d": [1, 1], "type": ["HAS_CREATOR"] * 2, "w": [1, 2]})
    g = _graph(nodes=nodes, edges=edges)
    ops = [
        n({"id": 1}, name="a"), e_forward({"type": "HAS_CREATOR"}, name="e"),
        n({}, name="b"), rows(), select([("w", "b.when")]),
    ]
    import graphistry.compute.gfql.lazy.engine.polars.chain as polars_chain
    assert polars_chain.try_bindings_select_polars(g, ops[:3], ops[3:], None) is None
    # The same query still answers, through the canonical route.
    served = g.gfql(ops, engine="polars", index_policy="force")
    with routes_off([ROUTE]):
        canonical = g.gfql(ops, engine="polars", index_policy="force")
    assert served._nodes.rows() == canonical._nodes.rows()


def test_string_literal_that_is_constructor_text_also_declines():
    g = _graph()
    ops = [
        n({"id": 1}, name="a"), e_forward({"type": "HAS_CREATOR"}, name="e"),
        n({}, name="b"), rows(), select([("lit", "date({year: 1984})"), ("bid", "b")]),
    ]
    import graphistry.compute.gfql.lazy.engine.polars.chain as polars_chain
    # A bare string item is an expression, not a literal, so this declines on the plan;
    # the point is that neither route can emit raw constructor text.
    assert polars_chain.try_bindings_select_polars(g, ops[:3], ops[3:], None) is None
