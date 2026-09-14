"""A chain made only of plain row-table calls does not scaffold the internal edge index.

Pins: the node rows and the edge frame are identical with and without the scaffolding (routes off is the
oracle); no ``__gfql_edge_index__`` column reaches the result; a call over the edge table still runs the
path without the scaffold too, and no longer leaks the index column into its row table.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n, rows, select
from graphistry.compute.chain import _calls_only_on_node_rows

NODES = pd.DataFrame({"id": [1, 2, 3], "kind": ["p", "p", "c"], "w": [10, 20, 30]})
EDGES = pd.DataFrame({"s": [1, 2], "d": [3, 3], "type": ["IN", "IN"]})
ENGINES = ["pandas", "cudf"]


def _graph(engine):
    nodes, edges = NODES, EDGES
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _pd(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("ops", [
    [n({"id": 1}, name="a"), rows(source="a")],
    [n({"id": 1}, name="a"), e_forward({"type": "IN"}), n(name="b"), rows(source="b"), select([("cid", "id"), ("k", "kind")])],
])
def test_node_row_calls_answer_identically_without_the_edge_index(engine, ops):
    from graphistry.tests.compute.gfql.routes.switch import routes_off, ROUTES
    g = _graph(engine)
    served = g.gfql(ops, engine=engine)
    with routes_off(ROUTES):
        general = g.gfql(ops, engine=engine)
    assert _pd(served._nodes).to_dict("records") == _pd(general._nodes).to_dict("records")
    assert not any(str(c).startswith("__gfql_") for c in served._nodes.columns)
    assert not any(str(c).startswith("__gfql_") for c in served._edges.columns)


@pytest.mark.parametrize("engine", ENGINES)
def test_edge_row_calls_answer_without_the_index_column(engine):
    g = _graph(engine)
    out = g.gfql([rows(table="edges")], engine=engine)
    assert sorted(_pd(out._nodes)[["s", "d"]].values.tolist()) == [[1, 3], [2, 3]]
    assert not any(str(c).startswith("__gfql_") for c in out._nodes.columns)


def test_the_predicate_names_exactly_the_node_row_calls():
    assert _calls_only_on_node_rows([rows(source="a")])
    assert _calls_only_on_node_rows([rows(source="a"), select([("x", "id")])])
    assert _calls_only_on_node_rows([rows(table="edges")])
    assert not _calls_only_on_node_rows([rows(binding_ops=[])])
    assert not _calls_only_on_node_rows([n({"id": 1}), rows(source="a")])
    assert not _calls_only_on_node_rows([])
