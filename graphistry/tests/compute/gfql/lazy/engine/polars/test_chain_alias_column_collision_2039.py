"""An alias that shares its name with the column its own step filters on is served on polars
instead of raising, with the user values still readable (#2039): the edge marker of a forward
pass must not be re-filtered as the edge column on the backward pass, and the node marker of
an earlier pass must not be re-filtered as the node column."""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n

pl = pytest.importorskip("polars")


def _pair():
    nodes = pd.DataFrame({"key": [1, 2, 3], "id": [10, 20, 30], "type": ["p", "p", "m"]})
    edges = pd.DataFrame({"s": [3, 3], "d": [1, 2], "type": ["HAS_CREATOR", "OTHER"]})
    g_pd = graphistry.nodes(nodes, "key").edges(edges, "s", "d")
    g_pl = graphistry.nodes(pl.from_pandas(nodes), "key").edges(pl.from_pandas(edges), "s", "d")
    return g_pd, g_pl


SHAPES = {
    "edge alias = filtered edge column": [n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="type"), n(name="p")],
    "node alias = seed property, edge alias = filtered column": [n({"id": 30}, name="id"), e_forward({"type": "HAS_CREATOR"}, name="type"), n(name="p")],
    "node alias = property only": [n({"id": 30}, name="id"), e_forward({"type": "HAS_CREATOR"}, name="e"), n(name="p")],
    "destination alias = its own filtered column": [n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="e"), n({"type": "p"}, name="type")],
    "seed, edge and destination aliases all collide": [n({"id": 30}, name="id"), e_forward({"type": "HAS_CREATOR"}, name="type"), n({"type": "p"}, name="type")],
}


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("index_policy", ["off", "use"])
def test_colliding_aliases_are_served_with_the_same_rows_on_both_engines(shape, index_policy):
    g_pd, g_pl = _pair()
    ops = SHAPES[shape]
    a = g_pd.gfql(ops, engine="pandas", index_policy=index_policy)
    b = g_pl.gfql(ops, engine="polars", index_policy=index_policy)
    assert sorted(a._nodes["key"].tolist()) == sorted(b._nodes["key"].to_list()) == [1, 3]
    assert a._edges[["s", "d"]].values.tolist() == b._edges.select("s", "d").rows() == [[3, 1]] or \
        a._edges[["s", "d"]].values.tolist() == [list(r) for r in b._edges.select("s", "d").rows()]


def test_two_hop_with_a_colliding_edge_alias_is_served_on_polars():
    _, g_pl = _pair()
    ops = [n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="type"), n(name="mid"),
           e_forward(name="e2"), n(name="p")]
    out = g_pl.gfql(ops, engine="polars")
    assert out._edges.select("s", "d").rows() == []  # nodes 1 and 2 have no out-edges


def test_destination_alias_marker_replaces_the_colliding_column_like_pandas():
    g_pd, g_pl = _pair()
    ops = [n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="e"), n({"type": "p"}, name="type")]
    a = g_pd.gfql(ops, engine="pandas")._nodes
    b = g_pl.gfql(ops, engine="polars")._nodes.to_pandas()
    assert "type_right" not in b.columns
    assert a["type"].dtype == bool and b["type"].dtype == bool
    assert sorted(a.loc[a["type"], "key"].tolist()) == sorted(b.loc[b["type"], "key"].tolist()) == [1]
