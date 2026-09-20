"""The #2039 column restore is paid only when an alias can actually shadow a column.

`_step_edges_with_source_columns` re-joins the WHOLE edge frame per step to undo an alias
marker stamped over a column its own step filters on. That shadowing needs an edge alias NAMED
like an edge column; paying the restore unconditionally cost LJ 1-hop +27% (34.7M-edge semi-join
twice per hop). These pins hold the guard on both sides of that boundary.
"""

import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import n, e_forward, e_undirected
from graphistry.compute.gfql.lazy.engine.polars.chain import _edge_alias_can_shadow_column


@pytest.fixture()
def g():
    edges = pd.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 0], "rel": ["a", "b", "c"]})
    nodes = pd.DataFrame({"node_id": [0, 1, 2]})
    return graphistry.nodes(nodes, "node_id").edges(edges, "src", "dst")


@pytest.mark.parametrize("ops,expected", [
    ([n(), e_undirected(hops=1), n()], False),
    ([n(), e_forward(), n()], False),
    ([n(), e_forward(name="not_a_column"), n()], False),
    # An alias named like an edge column CAN be stamped over it: the restore is required.
    ([n(), e_forward(name="rel"), n()], True),
    ([n(), e_forward(name="src"), n()], True),
    ([n(), e_forward(name="dst"), n()], True),
    # One colliding alias anywhere in the chain is enough.
    ([n(), e_forward(name="safe"), n(), e_forward(name="rel"), n()], True),
])
def test_restore_is_required_exactly_when_an_alias_can_shadow_a_column(g, ops, expected):
    assert _edge_alias_can_shadow_column(ops, g) is expected


def test_a_graph_without_edges_keeps_the_restore(g):
    """Unprovable input fails toward correctness, never toward speed."""
    assert _edge_alias_can_shadow_column([n(), e_forward(), n()], graphistry.bind()) is True


def test_the_colliding_alias_chain_still_answers(g):
    """The guard must not change what the collision case returns -- only whether the
    restore is skipped for everyone else."""
    out = g.gfql([n(), e_forward(name="rel"), n()], engine="polars")
    assert out._nodes is not None and len(out._nodes) > 0
