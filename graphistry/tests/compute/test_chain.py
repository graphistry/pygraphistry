import os
import pandas as pd
from graphistry.compute.predicates.is_in import is_in
import pytest

from graphistry.compute.ast import ASTNode, ASTEdge, n, e
from graphistry.compute.chain import Chain
from graphistry.tests.test_compute import CGFull


def test_chain_serialization_mt():
    o = Chain([]).to_json()
    d = Chain.from_json(o)
    assert d.chain == []
    assert o['chain'] == []

def test_chain_serialization_node():
    o = Chain([n(query='zzz', name='abc')]).to_json()
    d = Chain.from_json(o)
    assert isinstance(d.chain[0], ASTNode)
    assert d.chain[0].query == 'zzz'
    assert d.chain[0]._name == 'abc'
    o2 = d.to_json()
    assert o == o2

def test_chain_serialization_edge():
    o = Chain([e(edge_query='zzz', name='abc')]).to_json()
    d = Chain.from_json(o)
    assert isinstance(d.chain[0], ASTEdge)
    assert d.chain[0].edge_query == 'zzz'
    assert d.chain[0]._name == 'abc'
    o2 = d.to_json()
    assert o == o2

def test_chain_serialization_multi():
    o = Chain([n(query='zzz', name='abc'), e(edge_query='zzz', name='abc')]).to_json()
    d = Chain.from_json(o)
    assert isinstance(d.chain[0], ASTNode)
    assert d.chain[0].query == 'zzz'
    assert d.chain[0]._name == 'abc'
    assert isinstance(d.chain[1], ASTEdge)
    assert d.chain[1].edge_query == 'zzz'
    assert d.chain[1]._name == 'abc'
    o2 = d.to_json()
    assert o == o2

def test_chain_simple_cudf_pd():
    nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_df = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    #g_nodes = g.chain([n()])
    #assert isinstance(g_nodes._nodes, pd.DataFrame)
    #assert len(g_nodes._nodes) == 3
    g_edges = g.chain([e()])
    assert isinstance(g_edges._edges, pd.DataFrame)
    assert len(g_edges._edges) == 3


@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1",
)
def test_chain_simple_cudf():
    import cudf
    nodes_gdf = cudf.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_gdf = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')
    g_nodes = g.chain([n()])
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 3
    g_edges = g.chain([e()])
    assert isinstance(g_edges._edges, cudf.DataFrame)
    assert len(g_edges._edges) == 3

def test_chain_kv_cudf_pd():
    nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_df = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    g_nodes = g.chain([n({'id': 0})])
    assert isinstance(g_nodes._nodes, pd.DataFrame)
    assert len(g_nodes._nodes) == 1
    g_edges = g.chain([e({'src': 0})])
    assert isinstance(g_edges._edges, pd.DataFrame)
    assert len(g_edges._edges) == 1

@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1",
)
def test_chain_kv_cudf():
    import cudf
    nodes_gdf = cudf.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_gdf = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')
    g_nodes = g.chain([n({'id': 0})])
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 1
    g_edges = g.chain([e({'src': 0})])
    assert isinstance(g_edges._edges, cudf.DataFrame)
    assert len(g_edges._edges) == 1

def test_chain_pred_cudf_pd():
    nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_df = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    g_nodes = g.chain([n({'id': is_in([0])})])
    assert isinstance(g_nodes._nodes, pd.DataFrame)
    assert len(g_nodes._nodes) == 1
    g_edges = g.chain([e({'src': is_in([0])})])
    assert isinstance(g_edges._edges, pd.DataFrame)
    assert len(g_edges._edges) == 1

@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1",
)
def test_chain_pred_cudf():
    import cudf
    nodes_gdf = cudf.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_gdf = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')
    g_nodes = g.chain([n({'id': is_in([0])})])
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 1
    g_edges = g.chain([e({'src': is_in([0])})])
    assert isinstance(g_edges._edges, cudf.DataFrame)
    assert len(g_edges._edges) == 1
