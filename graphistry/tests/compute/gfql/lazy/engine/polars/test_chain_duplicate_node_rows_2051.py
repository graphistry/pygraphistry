"""The polars chain collapses duplicate node rows on every shape, as pandas' combine does.

Sibling of #1993 (polars ``hop()`` de-dups its node table): the unnamed, untyped single-hop
chain shape still keeps the duplicate (#2051, strict expected failure); the named, typed,
multi-hop and undirected shapes and ``hop()`` already collapse it and are pinned green.
chain shape kept the duplicate (#2051); every shape now collapses it, pinned against pandas.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, e_undirected, n

pl = pytest.importorskip("polars")


def _graph():
    nodes = pd.DataFrame({"key": [1, 2, 3, 4, 5], "id": [10, 20, 30, 40, 50]})
    nodes = pd.concat([nodes, nodes.iloc[[0]]], ignore_index=True)
    edges = pd.DataFrame({"s": [1, 1, 2, 3, 3, 4], "d": [2, 3, 3, 1, 1, 5],
                          "type": ["KNOWS", "KNOWS", "LIKES", "KNOWS", "KNOWS", "LIKES"], "eid": range(6)})
    g_pd = graphistry.nodes(nodes, "key").edges(edges, "s", "d", "eid")
    g_pl = graphistry.nodes(pl.from_pandas(nodes), "key").edges(pl.from_pandas(edges), "s", "d", "eid")
    return g_pd, g_pl


COLLAPSED = {
    "named single hop": [n({"key": 1}, name="a"), e_forward(name="e"), n(name="b")],
    "typed single hop": [n({"key": 1}), e_forward({"type": "KNOWS"}), n()],
    "hops=2": [n({"key": 1}), e_forward(hops=2), n()],
    "undirected": [n({"key": 1}), e_undirected(), n()],
    "seed only": [n({"key": 1})],
}


@pytest.mark.parametrize("shape", list(COLLAPSED))
def test_shapes_that_collapse_the_duplicate_match_pandas(shape):
    g_pd, g_pl = _graph()
    ops = COLLAPSED[shape]
    assert sorted(g_pl.gfql(ops, engine="polars")._nodes["key"].to_list()) == sorted(g_pd.gfql(ops, engine="pandas")._nodes["key"].tolist())


def test_hop_collapses_the_duplicate():
    g_pd, g_pl = _graph()
    seeds_pd = g_pd._nodes[g_pd._nodes["key"] == 1]
    seeds_pl = g_pl._nodes.filter(pl.col("key") == 1)
    assert sorted(g_pl.hop(nodes=seeds_pl, hops=1, engine="polars")._nodes["key"].to_list()) == sorted(g_pd.hop(nodes=seeds_pd, hops=1)._nodes["key"].tolist()) == [1, 2, 3]


@pytest.mark.parametrize("shape", ["unnamed untyped single hop", "unnamed single hop with destination filter"])
@pytest.mark.xfail(strict=True, reason="graphistry/pygraphistry#2051")
def test_unnamed_untyped_single_hop_collapses_the_duplicate(shape):
    g_pd, g_pl = _graph()
    ops = [n({"key": 1}), e_forward(), n()] if shape == "unnamed untyped single hop" else [n({"key": 1}), e_forward(), n({"id": 20})]
    assert sorted(g_pl.gfql(ops, engine="polars")._nodes["key"].to_list()) == sorted(g_pd.gfql(ops, engine="pandas")._nodes["key"].tolist())
