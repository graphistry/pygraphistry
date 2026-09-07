"""A colliding alias never leaks an internal column on the op-list surface.

On polars the chain keeps a shadowed user column under the shadow-restore name ONLY while a
compiled Cypher pipeline runs (its row pipeline reads the value back for ``alias.alias``);
on the op-list surface the marker simply shadows the column, as on pandas and cuDF. Nothing
is stripped from results, so user-defined columns (the ``name=`` marker, requested hop
labels, even a caller's own ``__gfql_``-prefixed column) and pipelines composed from
successive ``gfql([...])`` calls are untouched.
"""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n

NODES = pd.DataFrame({"id": [1, 2, 3], "kind": ["P", "P", "C"]})
EDGES = pd.DataFrame({"s": [1, 2], "d": [3, 3], "type": ["K", "L"], "w": [7, 8]})
SHADOW = [n(name="a"), e_forward({"type": "K"}, name="type"), n(name="b")]


def _graph(engine):
    nodes, edges = NODES, EDGES
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("cuDF lane runs with TEST_CUDF=1")
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _pd(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.parametrize("engine", ["pandas", "cudf", "polars"])
def test_op_list_result_has_no_internal_columns_and_keeps_the_user_marker(engine):
    res = _graph(engine).gfql(SHADOW, engine=engine)
    edges = _pd(res._edges)
    assert not [c for c in edges.columns if str(c).startswith("__gfql_")], list(edges.columns)
    assert not [c for c in _pd(res._nodes).columns if str(c).startswith("__gfql_")]
    assert edges["type"].tolist() == [True] and str(edges["type"].dtype) == "bool"
    assert edges["w"].tolist() == [7]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_cypher_still_reads_the_shadowed_user_value(engine):
    rows = _pd(_graph(engine).gfql("MATCH (a)-[type:K]->(b) RETURN type.type AS t, type.w AS w", engine=engine)._nodes)
    assert rows.to_dict("records") == [{"t": "K", "w": 7}]
    assert not [c for c in rows.columns if str(c).startswith("__gfql_")]


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_user_requested_hop_labels_survive(engine):
    res = _graph(engine).gfql([n({"id": 1}), e_forward(hops=2, label_node_hops="hops", label_edge_hops="ehops"), n()], engine=engine)
    assert "hops" in _pd(res._nodes).columns and "ehops" in _pd(res._edges).columns
    assert not [c for c in _pd(res._nodes).columns if str(c).startswith("__gfql_")]


def test_deprecated_chain_surface_is_clean_too():
    pytest.importorskip("polars")
    with pytest.warns(DeprecationWarning):
        res = _graph("polars").chain(SHADOW, engine="polars")
    assert not [c for c in res._edges.columns if str(c).startswith("__gfql_")]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_a_user_column_with_the_internal_prefix_is_kept(engine):
    """Only columns the chain added are hidden; a caller's own `__gfql_`-prefixed column passes through."""
    g = _graph(engine)
    nodes = g._nodes.with_columns(**{"__gfql_mine__": 1}) if engine == "polars" else g._nodes.assign(__gfql_mine__=1)
    res = g.nodes(nodes, "id").gfql([n(), e_forward(), n()], engine=engine)
    assert "__gfql_mine__" in _pd(res._nodes).columns
