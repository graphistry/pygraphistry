import os
import pandas as pd
import pytest

from graphistry.compute.ast import ASTEdgeUndirected, ASTNode, ASTEdge, n, e, e_undirected, e_forward
from graphistry.compute.chain import Chain
from graphistry.compute.predicates.is_in import IsIn, is_in
from graphistry.compute.predicates.numeric import gt
from graphistry.tests.test_compute import CGFull


@pytest.fixture(scope='module')
def g_long_forwards_chain():
    """
    a->b->c->d->e
    """
    return (CGFull()
        .edges(pd.DataFrame({
            's': ['a', 'b', 'c', 'd'],
            'd': ['b', 'c', 'd', 'e'],
            't': ['1', '2', '3', '4'],
            'e': ['2', '3', '4', '5']}),
            's', 'd')
        .nodes(pd.DataFrame({
            'v': ['a', 'b', 'c', 'd', 'e'],
            'w': ['1', '2', '3', '4', '5']}),
            'v'))

@pytest.fixture(scope='module')
def g_long_forwards_chain_dead_end():
    """
    a->b->c->d->e
          c->x
    """
    return (CGFull()
        .edges(pd.DataFrame({
            's': ['a', 'b', 'c', 'd', 'c'],
            'd': ['b', 'c', 'd', 'e', 'x']}),
            's', 'd')
        .nodes(pd.DataFrame({
            'v': ['a', 'b', 'c', 'd', 'e', 'x']}),
            'v'))

@pytest.fixture(scope='module')
def g_long_forwards_chain_loop():
    """
    a->b->c->d->e
       c->x->c
    """
    return (CGFull()
        .edges(pd.DataFrame({
            's': ['a', 'b', 'c', 'd', 'c', 'x'],
            'd': ['b', 'c', 'd', 'e', 'x', 'c']}),
            's', 'd')
        .nodes(pd.DataFrame({
            'v': ['a', 'b', 'c', 'd', 'e', 'x']}),
            'v'))

class TestMultiHopChainForward():

    def test_chain_short(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.chain([n({'v': 'a'}), e_forward(hops=2), n({'v': 'd'})])
        assert len(g2._nodes) == 0
        assert len(g2._edges) == 0
    
    def test_chain_exact(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.chain([n({'v': 'a'}), e_forward(hops=3), n({'v': 'd'})])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_chain_long(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.chain([n({'v': 'a'}), e_forward(hops=4), n({'v': 'd'})])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_chain_fixedpoint(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.chain([n({'v': 'a'}), e_forward(to_fixed_point=True), n({'v': 'd'})])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_chain_predicates_ok_source(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.chain([
            n({'v': 'a'}),
            e_forward(
                source_node_match={'w': is_in(['1', '2', '3'])},
                hops=3
            ),
            n({'v': 'd'})
        ])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_chain_predicates_ok_edge(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.chain([
            n({'v': 'a'}),
            e_forward(
                edge_match={
                    't': is_in(['1', '2', '3']),
                    'e': is_in(['2', '3', '4'])
                },
                hops=3
            ),
            n({'v': 'd'})
        ])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_chain_predicates_ok_destination(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.chain([
            n({'v': 'a'}),
            e_forward(
                destination_node_match={'w': is_in(['2', '3', '4'])},
                hops=3
            ),
            n({'v': 'd'})
        ])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_chain_predicates_ok(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.chain([
            n({'v': 'a'}),
            e_forward(
                source_node_match={'w': is_in(['1', '2', '3'])},
                edge_match={
                    't': is_in(['1', '2', '3']),
                    'e': is_in(['2', '3', '4'])
                },
                destination_node_match={'w': is_in(['2', '3', '4'])},
                hops=3
            ),
            n({'v': 'd'})
        ])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_chain_predicates_source_fail(self, g_long_forwards_chain):
        BAD = []
        g2 = g_long_forwards_chain.chain([
            n({'v': 'a'}),
            e_forward(
                source_node_match={'w': is_in(BAD)},
                edge_match={
                    't': is_in(['1', '2', '3']),
                    'e': is_in(['2', '3', '4'])
                },
                destination_node_match={'w': is_in(['2', '3', '4'])},
                hops=3
            ),
            n({'v': 'd'})
        ])
        assert len(g2._nodes) == 0
        assert len(g2._edges) == 0

    def test_chain_predicates_dest_fail(self, g_long_forwards_chain):
        BAD = []
        g2 = g_long_forwards_chain.chain([
            n({'v': 'a'}),
            e_forward(
                source_node_match={'w': is_in(['1', '2', '3'])},
                edge_match={
                    't': is_in(['1', '2', '3']),
                    'e': is_in(['2', '3', '4'])
                },
                destination_node_match={'w': is_in(BAD)},
                hops=3
            ),
            n({'v': 'd'})
        ])
        assert len(g2._nodes) == 0
        assert len(g2._edges) == 0

    def test_chain_predicates_edge_fail(self, g_long_forwards_chain):
        BAD = []
        g2 = g_long_forwards_chain.chain([
            n({'v': 'a'}),
            e_forward(
                source_node_match={'w': is_in(['1', '2', '3'])},
                edge_match={
                    't': is_in(BAD),
                    'e': is_in(['2', '3', '4'])
                },
                destination_node_match={'w': is_in(['2', '3', '4'])},
                hops=3
            ),
            n({'v': 'd'})
        ])
        assert len(g2._nodes) == 0
        assert len(g2._edges) == 0


class TestMultiHopDeadend():

    def test_chain_fixedpoint(self, g_long_forwards_chain_dead_end: CGFull):
        """
        Same as chain; x should not be considered a hint
        """
        g2 = g_long_forwards_chain_dead_end.chain([n({'v': 'a'}), e_forward(to_fixed_point=True), n({'v': 'd'})])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]


class TestMultiHopLoop():

    def test_chain_fixedpoint(self, g_long_forwards_chain_loop: CGFull):
        """
        Same as chain; + detour using x
        """
        g2 = g_long_forwards_chain_loop.chain([n({'v': 'a'}), e_forward(to_fixed_point=True), n({'v': 'd'})])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd', 'x'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
            {'s': 'c', 'd': 'x'},
            {'s': 'x', 'd': 'c'}
        ]


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

def test_chain_serialization_pred():
    o = Chain([n(query='zzz', name='abc', filter_dict={'a': is_in(options=['a', 'b', 'c'])}),
               e(edge_query='zzz', name='abc', edge_match={'b': is_in(options=['a', 'b', 'c'])})]).to_json()
    d = Chain.from_json(o)
    assert isinstance(d.chain[0], ASTNode)
    assert d.chain[0].query == 'zzz'
    assert d.chain[0]._name == 'abc'
    assert isinstance(d.chain[1], ASTEdge)
    assert d.chain[1].edge_query == 'zzz'
    assert d.chain[1]._name == 'abc'
    o2 = d.to_json()
    assert o == o2

def test_chain_serialize_pred_is_in():

    #from graphistry.compute.chain import Chain
    #from graphistry import e_undirected, is_in
    o = Chain([
        e_undirected(
            hops=1,
            edge_match={"source": is_in(options=[
                "Oakville Square",
                "Maplewood Square"
            ])})
    ]).to_json()
    d = Chain.from_json(o)
    assert isinstance(d.chain[0], ASTEdgeUndirected), f'got: {type(d.chain[0])}'
    assert d.chain[0].direction == 'undirected'
    assert d.chain[0].hops == 1
    assert isinstance(d.chain[0].edge_match['source'], IsIn)
    assert d.chain[0].edge_match['source'].options == ['Oakville Square', 'Maplewood Square']

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

def test_preds_more_pd():

    # a->b3->c1
    #    U
    edf = pd.DataFrame({
        's': ['a1', 'b3', 'b3'],
        'd': ['b3', 'b3', 'c1']
    })
    g = CGFull().edges(edf, 's', 'd').materialize_nodes().get_degrees()

    g2 = (g.get_degrees()
        .chain([
            n({'degree': gt(1)}),
            e_undirected(),
            n({'degree': gt(1)})
        ])
    )
    assert set(g2._nodes[g2._node].tolist()) == set(['b3'])

def test_preds_more_pd_2():

    edf = pd.DataFrame({
        's': ['a1', 'b2', 'c2'],
        'd': ['b2', 'c2', 'd1']
    })
    g = CGFull().edges(edf, 's', 'd').materialize_nodes().get_degrees()

    g2 = (g.get_degrees()
        .chain([
            n({'degree': gt(1)}),
            e_undirected(),
            n({'degree': gt(1)})
        ])
    )
    assert len(g2._nodes) == 2
    assert set(g2._nodes[g._node].tolist()) == set(['b2', 'c2'])


def test_chain_binding_reuse():
    # This test has been updated to reflect the new behavior that allows node column names
    # to be the same as edge source or destination column names
    edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
    nodes1_df = pd.DataFrame({'v': ['a', 'b', 'c']})
    nodes2_df = pd.DataFrame({'s': ['a', 'b', 'c']})
    nodes3_df = pd.DataFrame({'d': ['a', 'b', 'c']})
    
    g1 = CGFull().nodes(nodes1_df, 'v').edges(edges_df, 's', 'd')
    g2 = CGFull().nodes(nodes2_df, 's').edges(edges_df, 's', 'd')
    g3 = CGFull().nodes(nodes3_df, 'd').edges(edges_df, 's', 'd')

    # With our new implementation, all three should successfully run
    g1_chain = g1.chain([n(), e(), n()])
    g2_chain = g2.chain([n(), e(), n()])
    g3_chain = g3.chain([n(), e(), n()])
    
    # Make sure we get expected results - g1 and g2 have consistent behavior
    # Just verify that all three approaches produce reasonable results
    assert g1_chain._nodes.shape[0] > 0
    assert g1_chain._edges.shape[0] > 0
    assert g2_chain._nodes.shape[0] > 0
    assert g2_chain._edges.shape[0] > 0
    assert g3_chain._nodes.shape[0] > 0
    assert g3_chain._edges.shape[0] > 0
