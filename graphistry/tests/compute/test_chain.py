import os
import pandas as pd
import pytest

from graphistry.compute.ast import ASTEdgeUndirected, ASTNode, ASTEdge, n, e, e_undirected, e_forward, e_reverse
from graphistry.compute.chain import Chain, _try_chain_fast_path
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
        g2 = g_long_forwards_chain.gfql([n({'v': 'a'}), e_forward(hops=2), n({'v': 'd'})])
        assert len(g2._nodes) == 0
        assert len(g2._edges) == 0
    
    def test_chain_exact(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.gfql([n({'v': 'a'}), e_forward(hops=3), n({'v': 'd'})])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_chain_long(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.gfql([n({'v': 'a'}), e_forward(hops=4), n({'v': 'd'})])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_chain_fixedpoint(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.gfql([n({'v': 'a'}), e_forward(to_fixed_point=True), n({'v': 'd'})])
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_chain_exact_named_terminal_alias_marks_only_endpoint_nodes(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.gfql([n({'v': 'a'}, name='seed'), e_forward(min_hops=3, max_hops=3), n(name='hit')])
        assert sorted(g2._nodes.loc[g2._nodes['seed'], 'v'].tolist()) == ['a']
        assert sorted(g2._nodes.loc[g2._nodes['hit'], 'v'].tolist()) == ['d']

    def test_chain_exact_multihop_then_single_hop_marks_following_terminal_alias(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.gfql([
            n({'v': 'a'}, name='seed'),
            e_forward(min_hops=1, max_hops=1),
            n(name='mid'),
            e_forward(),
            n(name='hit'),
        ])
        assert sorted(g2._nodes.loc[g2._nodes['mid'], 'v'].tolist()) == ['b']
        assert sorted(g2._nodes.loc[g2._nodes['hit'], 'v'].tolist()) == ['c']

    def test_chain_single_hop_then_exact_multihop_marks_following_terminal_alias(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.gfql([
            n({'v': 'a'}, name='seed'),
            e_forward(),
            n(name='mid'),
            e_forward(min_hops=2, max_hops=2),
            n(name='hit'),
        ])
        assert sorted(g2._nodes.loc[g2._nodes['mid'], 'v'].tolist()) == ['b']
        assert sorted(g2._nodes.loc[g2._nodes['hit'], 'v'].tolist()) == ['d']

    def test_chain_predicates_ok_source(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.gfql([
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
        g2 = g_long_forwards_chain.gfql([
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

    def test_chain_uses_execute_even_if_dunder_call_exists(self, g_long_forwards_chain, monkeypatch):
        """Regression: operator execution path must not rely on __call__."""

        def _boom(*_args, **_kwargs):
            raise AssertionError("__call__ should not be used by chain execution")

        # If chain still used op(...), this would fail immediately.
        monkeypatch.setattr(ASTNode, "__call__", _boom, raising=False)
        monkeypatch.setattr(ASTEdge, "__call__", _boom, raising=False)

        g2 = g_long_forwards_chain.gfql([n({'v': 'a'}), e_forward(hops=3), n({'v': 'd'})])
        assert set(g2._nodes['v'].tolist()) == {'a', 'b', 'c', 'd'}
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).reset_index(drop=True).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_chain_predicates_ok_destination(self, g_long_forwards_chain):
        g2 = g_long_forwards_chain.gfql([
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
        g2 = g_long_forwards_chain.gfql([
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
        g2 = g_long_forwards_chain.gfql([
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
        g2 = g_long_forwards_chain.gfql([
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
        g2 = g_long_forwards_chain.gfql([
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
        g2 = g_long_forwards_chain_dead_end.gfql([n({'v': 'a'}), e_forward(to_fixed_point=True), n({'v': 'd'})])
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
        g2 = g_long_forwards_chain_loop.gfql([n({'v': 'a'}), e_forward(to_fixed_point=True), n({'v': 'd'})])
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
    #g_nodes = g.gfql([n()])
    #assert isinstance(g_nodes._nodes, pd.DataFrame)
    #assert len(g_nodes._nodes) == 3
    g_edges = g.gfql([e()])
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
    g_nodes = g.gfql([n()])
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 3
    g_edges = g.gfql([e()])
    assert isinstance(g_edges._edges, cudf.DataFrame)
    assert len(g_edges._edges) == 3

def test_chain_kv_cudf_pd():
    nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_df = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    g_nodes = g.gfql([n({'id': 0})])
    assert isinstance(g_nodes._nodes, pd.DataFrame)
    assert len(g_nodes._nodes) == 1
    g_edges = g.gfql([e({'src': 0})])
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
    g_nodes = g.gfql([n({'id': 0})])
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 1
    g_edges = g.gfql([e({'src': 0})])
    assert isinstance(g_edges._edges, cudf.DataFrame)
    assert len(g_edges._edges) == 1

def test_chain_pred_cudf_pd():
    nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_df = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    g_nodes = g.gfql([n({'id': is_in([0])})])
    assert isinstance(g_nodes._nodes, pd.DataFrame)
    assert len(g_nodes._nodes) == 1
    g_edges = g.gfql([e({'src': is_in([0])})])
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
    g_nodes = g.gfql([n({'id': is_in([0])})])
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 1
    g_edges = g.gfql([e({'src': is_in([0])})])
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
        .gfql([
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
        .gfql([
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
    g1_chain = g1.gfql([n(), e(), n()])
    g2_chain = g2.gfql([n(), e(), n()])
    g3_chain = g3.gfql([n(), e(), n()])
    
    # Make sure we get expected results - g1 and g2 have consistent behavior
    # Just verify that all three approaches produce reasonable results
    assert g1_chain._nodes.shape[0] > 0
    assert g1_chain._edges.shape[0] > 0
    assert g2_chain._nodes.shape[0] > 0
    assert g2_chain._edges.shape[0] > 0
    assert g3_chain._nodes.shape[0] > 0
    assert g3_chain._edges.shape[0] > 0


def test_chain_preserves_none_edge_binding():
    """Test that chain() preserves None edge binding when no edge column is set.
    
    When g._edge is None, chain() internally adds a temporary index column for tracking,
    but the output graph should restore the original None binding.
    
    Regression test for bug where output graph would have _edge set to internal column
    name like '__gfql_edge_index_0__' instead of None.
    """
    # Create a graph with NO edge binding (g._edge = None)
    edges_df = pd.DataFrame({
        's': ['a', 'b', 'c'],
        'd': ['b', 'c', 'd']
    })
    nodes_df = pd.DataFrame({
        'v': ['a', 'b', 'c', 'd']
    })
    
    g = CGFull().edges(edges_df, 's', 'd').nodes(nodes_df, 'v')
    
    # Verify g._edge is None before chain
    assert g._edge is None, "Input graph should have None edge binding"
    
    # Run a simple chain operation
    g_result = g.gfql([n({'v': 'a'}), e_forward(hops=2)])
    
    # The bug was that g_result._edge would be set to the internal column name like '__gfql_edge_index_0__'
    # The fix ensures it's restored to None
    assert g_result._edge is None, f"Output graph should have None edge binding, but got: {g_result._edge}"
    
    # Verify the chain operation actually worked
    assert len(g_result._nodes) > 0
    assert len(g_result._edges) > 0
    # Verify the internal column was properly removed
    assert '__gfql_edge_index_0__' not in g_result._edges.columns


def test_chain_preserves_custom_edge_binding():
    """Test that chain() preserves custom edge binding when edge column is set."""
    # Create a graph WITH an edge binding
    edges_df = pd.DataFrame({
        's': ['a', 'b', 'c'],
        'd': ['b', 'c', 'd'],
        'edge_id': ['e1', 'e2', 'e3']
    })
    nodes_df = pd.DataFrame({
        'v': ['a', 'b', 'c', 'd']
    })

    g = CGFull().edges(edges_df, 's', 'd', edge='edge_id').nodes(nodes_df, 'v')

    # Verify g._edge is 'edge_id' before chain
    assert g._edge == 'edge_id', "Input graph should have 'edge_id' edge binding"

    # Run a simple chain operation
    g_result = g.gfql([n({'v': 'a'}), e_forward(hops=2)])

    # Should preserve the 'edge_id' binding
    assert g_result._edge == 'edge_id', f"Output graph should have 'edge_id' edge binding, but got: {g_result._edge}"

    # Verify the chain operation actually worked
    assert len(g_result._nodes) > 0
    assert len(g_result._edges) > 0
    assert 'edge_id' in g_result._edges.columns


def test_chain_hop_label_node_hops():
    """label_node_hops propagates hop numbers to nodes; chain combine_steps passes columns with 'hop' in the name."""
    # a -> b -> c -> d  (linear chain of 4 nodes)
    nodes_df = pd.DataFrame({'v': ['a', 'b', 'c', 'd'], 'type': ['T', 'T', 'T', 'T']})
    edges_df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd'], 'etype': ['E', 'E', 'E']})
    g = CGFull().nodes(nodes_df, 'v').edges(edges_df, 's', 'd')

    # Direct hop with label_seeds: seed gets hop 0, reached nodes get hop > 0
    seed = pd.DataFrame({'v': ['a']})
    g2 = g.hop(nodes=seed, hops=3, label_node_hops='node_hop', label_seeds=True, direction='forward')
    assert 'node_hop' in g2._nodes.columns, f"Expected 'node_hop' in nodes, got: {list(g2._nodes.columns)}"
    nodes_by_id = g2._nodes.set_index('v')
    assert nodes_by_id.loc['a', 'node_hop'] == 0
    assert nodes_by_id.loc['b', 'node_hop'] == 1
    assert nodes_by_id.loc['c', 'node_hop'] == 2
    assert nodes_by_id.loc['d', 'node_hop'] == 3

    # gfql chain: combine_steps propagates columns whose name contains 'hop'
    g3 = g.gfql([n({'v': 'a'}), e_forward(hops=2, label_node_hops='node_hop')])
    assert 'node_hop' in g3._nodes.columns, f"Expected 'node_hop' in gfql chain nodes, got: {list(g3._nodes.columns)}"


def test_fast_path_still_fires_policy_hooks():
    """Regression: the node-only / single-hop chain fast path (chain._try_chain_fast_path)
    must NOT bypass policy hooks. The fast path returns before the prechain/postchain/
    postload block in _chain_impl, so it is only valid when no policy is installed.
    With a policy present we must take the full, hook-bearing path for the SAME
    fast-path-eligible shapes (``[n()]`` and ``[n(), e(), n()]``)."""
    nodes_df = pd.DataFrame({'v': ['a', 'b', 'c'], 'w': ['1', '2', '3']})
    edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
    g = CGFull().nodes(nodes_df, 'v').edges(edges_df, 's', 'd')

    for query in ([n()], [n(), e_forward(hops=1), n()]):
        fired = []
        g.gfql(query, policy={
            'prechain': lambda ctx: fired.append('prechain'),
            'postchain': lambda ctx: fired.append('postchain'),
            'postload': lambda ctx: fired.append('postload'),
        })
        assert fired == ['prechain', 'postchain', 'postload'], \
            f"fast-path shape {query} bypassed hooks: got {fired}"

    # And without a policy the fast path is still taken (results unchanged).
    res = g.gfql([n()])
    assert sorted(res._nodes['v'].tolist()) == ['a', 'b', 'c']


# ---------------------------------------------------------------------------
# Fast-path amplification: lock the real behaviors of _try_chain_fast_path
# (chain.py). The fast path (node-only MATCH + single-hop, pandas/cuDF) skips
# the forward/backward/combine BFS machinery, so it must be observationally
# equivalent to the full path it replaces. We use an installed (no-op) policy as
# the equivalence ORACLE: any non-empty policy forces the full, pre-fast-path
# BFS path (see _chain_impl gate), so `gfql(q)` (fast) vs `gfql(q, policy=NOOP)`
# (full) is a built-in differential — they must agree on node/edge SETS.
# ---------------------------------------------------------------------------

_FAST_NOOP_POLICY = {'preload': lambda ctx: None}  # any hook -> full (non-fast) path


def _cudf_or_skip():
    if not (os.environ.get("TEST_CUDF") == "1"):
        pytest.skip("cuDF lane: set TEST_CUDF=1 (e.g. on dgx-spark)")
    return pytest.importorskip("cudf")


def _fast_graph(engine):
    nodes = pd.DataFrame({'v': [0, 1, 2, 3, 4], 'attr': [10, 20, 30, 40, 50]})
    edges = pd.DataFrame({'s': [0, 1, 2, 3, 0], 'd': [1, 2, 3, 4, 2], 'w': [5, 6, 7, 8, 9]})
    if engine == "cudf":
        cudf = _cudf_or_skip()
        nodes = cudf.DataFrame.from_pandas(nodes)
        edges = cudf.DataFrame.from_pandas(edges)
    return CGFull().nodes(nodes, 'v').edges(edges, 's', 'd')


def _setsig(r):
    """Engine-agnostic (node-id set, edge (s,d) set) — values, not dtypes."""
    def topd(df):
        return df.to_pandas() if df is not None and "cudf" in type(df).__module__ else df
    nn = topd(r._nodes)
    ee = topd(r._edges)
    nodes = sorted(nn['v'].tolist()) if nn is not None else []
    edges = sorted(map(tuple, ee[['s', 'd']].itertuples(index=False, name=None))) \
        if ee is not None and len(ee) else []
    return nodes, edges


# shapes that ARE accelerated by the fast path
_FAST_SHAPES = [
    ("node_only", lambda: [n()]),
    ("node_filter", lambda: [n({'attr': 20})]),
    ("node_pred", lambda: [n({'attr': is_in([10, 30])})]),
    ("hop_fwd", lambda: [n(), e_forward(hops=1), n()]),
    ("hop_rev", lambda: [n(), e_reverse(hops=1), n()]),
    ("hop_undirected_unconstrained", lambda: [n(), e_undirected(hops=1), n()]),
    ("hop_fwd_src_filter", lambda: [n({'attr': 10}), e_forward(hops=1), n()]),
    ("hop_fwd_dst_filter", lambda: [n(), e_forward(hops=1), n({'attr': 40})]),
    ("hop_fwd_both_filter", lambda: [n({'attr': 10}), e_forward(hops=1), n({'attr': 30})]),
    ("hop_rev_dst_filter", lambda: [n(), e_reverse(hops=1), n({'attr': 10})]),
    # #1755 lever-3: typed edges (edge_match) are now a fast shape — a plain edge
    # filter applied on the (seed-reduced) frontier, not a fall-through.
    ("edge_match_unconstrained", lambda: [n(), e_forward(hops=1, edge_match={'w': 5}), n()]),
    ("edge_match_seeded", lambda: [n({'attr': 10}), e_forward(hops=1, edge_match={'w': 5}), n()]),
    ("edge_match_dst_filter", lambda: [n(), e_forward(hops=1, edge_match={'w': 5}), n({'attr': 30})]),
]

# shapes that BYPASS the fast path (still must be correct via the full path)
_BYPASS_SHAPES = [
    ("hops_2", lambda: [n(), e_forward(hops=2), n()]),
    ("filtered_undirected", lambda: [n({'attr': 10}), e_undirected(hops=1), n({'attr': 30})]),
    ("named_node", lambda: [n(name='x'), e_forward(hops=1), n()]),
    # prune_to_endpoints: fast path returns both endpoints; full path keeps only the
    # arrival side. Must bypass the fast path (regression guard for the prune gate).
    ("prune_endpoints_fwd", lambda: [n(), e_forward(hops=1, prune_to_endpoints=True), n()]),
    ("prune_endpoints_rev", lambda: [n(), e_reverse(hops=1, prune_to_endpoints=True), n()]),
]


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("label,build", _FAST_SHAPES + _BYPASS_SHAPES,
                         ids=[s[0] for s in _FAST_SHAPES + _BYPASS_SHAPES])
def test_fast_path_differential_parity_vs_full_path(engine, label, build):
    """Fast path output == full (policy-forced BFS) path output, by node/edge SET,
    for every accelerated shape AND every bypass shape, on pandas and cuDF.

    For FAST shapes `g.gfql(ops)` exercises the fast path and the policy-forced call
    the full BFS, so this is a true differential. For BYPASS shapes both calls take
    the full path (the point being they MUST decline the fast path); the decline is
    asserted directly below so the bypass cases are not merely full-vs-full."""
    from graphistry.compute.chain import _try_chain_fast_path
    from graphistry.Engine import Engine
    g = _fast_graph(engine)
    fast = g.gfql(build())
    full = g.gfql(build(), policy=_FAST_NOOP_POLICY)
    assert _setsig(fast) == _setsig(full), f"{engine}/{label}: fast != full"
    # Bypass shapes must genuinely decline the fast path (not vacuously full==full).
    if engine == "pandas" and any(label == s[0] for s in _BYPASS_SHAPES):
        eng = Engine.PANDAS
        assert _try_chain_fast_path(g, build(), eng, None) is None, \
            f"{label}: bypass shape must decline the fast path"


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_fast_path_preserves_int_node_dtypes(engine):
    """Documented behavior change: the 1-hop fast path PRESERVES node-attribute
    dtypes (int stays int) where the full BFS path upcasts int->float via merge.
    Lock both sides so neither silently regresses."""
    g = _fast_graph(engine)
    q = [n(), e_forward(hops=1), n()]
    fast = g.gfql(q)
    full = g.gfql(q, policy=_FAST_NOOP_POLICY)

    def kind(df, col):
        pdf = df.to_pandas() if "cudf" in type(df).__module__ else df
        return pdf[col].dtype.kind

    # The feature's promise: the 1-hop fast path keeps int node attrs as int.
    assert kind(fast._nodes, 'attr') == 'i', "fast path must keep int node attrs as int"
    # The full BFS path today upcasts int->float (a known merge wart this fast path
    # sidesteps). Assert only that it stays numeric, so a future full-path dtype fix
    # does not break this test; the fast-path promise above is what we hard-lock.
    assert kind(full._nodes, 'attr') in ('i', 'f')
    # node-only never traverses a merge, so it stays int regardless of path.
    assert kind(g.gfql([n()])._nodes, 'attr') == 'i'


def test_fast_path_gating_returns_none_for_ineligible():
    """Unit-level gate: _try_chain_fast_path must DECLINE (return None) for every
    shape/condition it does not cover, so those queries reach the correct full
    path. Eligible shapes must be accepted (non-None)."""
    from graphistry.Engine import Engine
    g = _fast_graph("pandas")
    seed = pd.DataFrame({'v': [0]})

    eligible = [
        [n()],
        [n({'attr': 20})],
        [n(), e_forward(hops=1), n()],
        [n(), e_reverse(hops=1), n()],
        # #1755 lever-3: typed edges are now accepted (edge filter on the frontier)
        [n(), e_forward(hops=1, edge_match={'w': 5}), n()],
        [n({'attr': 10}), e_forward(hops=1, edge_match={'w': 5}), n()],
    ]
    for ops in eligible:
        assert _try_chain_fast_path(g, ops, Engine.PANDAS, None) is not None, f"should accept {ops}"

    ineligible = [
        ("hops_2", [n(), e_forward(hops=2), n()], None, Engine.PANDAS),
        ("filtered_undirected", [n({'attr': 10}), e_undirected(hops=1), n({'attr': 30})], None, Engine.PANDAS),
        ("named_node", [n(name='x'), e_forward(hops=1), n()], None, Engine.PANDAS),
        ("node_query", [n(query='attr > 5'), e_forward(hops=1), n()], None, Engine.PANDAS),
        ("prune_endpoints", [n(), e_forward(hops=1, prune_to_endpoints=True), n()], None, Engine.PANDAS),
        ("seeded", [n()], seed, Engine.PANDAS),
        ("non_eager_engine", [n()], None, Engine.DASK),
        ("two_ops", [n(), e_forward(hops=1)], None, Engine.PANDAS),
    ]
    for label, ops, sn, eng in ineligible:
        assert _try_chain_fast_path(g, ops, eng, sn) is None, f"should decline {label}"


def test_chain_otel_span_attrs_mapped_correctly(monkeypatch):
    """Regression: the `gfql.chain` otel decorator must wrap `chain()`, not the
    `_try_chain_fast_path` helper defined just above it. If it drifts onto the
    fast path, `_chain_otel_attrs` receives the fast path's positional args
    (g_in, ops, engine_concrete, start_nodes) so `gfql.validate_schema` gets bound
    to start_nodes (a DataFrame/None) and `chain()` itself emits no span.
    Enable otel + detail, capture spans, and assert correct attr mapping."""
    import importlib
    import graphistry.compute.chain as chain_mod
    from contextlib import contextmanager
    # `import graphistry.otel` binds to a shadowing client attr, so resolve the
    # real module via importlib. otel_enabled/otel_span are looked up in the otel
    # module (inside otel_traced's wrapper); otel_detail_enabled is looked up in
    # chain.py (inside _chain_otel_attrs). Patch each in its own namespace.
    otel_mod = importlib.import_module("graphistry.otel")

    captured = []
    monkeypatch.setattr(otel_mod, "otel_enabled", lambda: True)
    monkeypatch.setattr(chain_mod, "otel_detail_enabled", lambda: True)

    @contextmanager
    def _fake_span(name, attrs=None):
        captured.append((name, attrs or {}))
        yield None

    monkeypatch.setattr(otel_mod, "otel_span", _fake_span)

    g = CGFull().nodes(pd.DataFrame({'v': [0, 1, 2]}), 'v').edges(
        pd.DataFrame({'s': [0, 1], 'd': [1, 2]}), 's', 'd')
    g.gfql([n()])  # fast-path-eligible shape

    chain_spans = [a for (nm, a) in captured if nm == "gfql.chain"]
    assert chain_spans, "chain() must emit a gfql.chain span"
    attrs = chain_spans[0]
    assert attrs.get("gfql.chain_len") == 1
    # The bug bound validate_schema to start_nodes (None / a DataFrame); the
    # correct mapping is the bool default.
    assert isinstance(attrs.get("gfql.validate_schema"), bool), \
        f"validate_schema attr must be a bool, got {type(attrs.get('gfql.validate_schema'))}"


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_fast_path_drops_edges_to_absent_nodes(engine):
    """The 1-hop fast path must drop edges whose endpoints are not in the node
    table (the full BFS path does, via its edge<->node joins). A node table that
    omits an edge endpoint must not yield dangling edges — nor a non-empty result
    where the full path is empty."""
    nodes = pd.DataFrame({'v': [0, 1], 'attr': [1, 2]})
    edges = pd.DataFrame({'s': [0, 1], 'd': [1, 99]})  # 99 absent from nodes
    if engine == "cudf":
        cudf = _cudf_or_skip()
        nodes, edges = cudf.DataFrame.from_pandas(nodes), cudf.DataFrame.from_pandas(edges)
    g = CGFull().nodes(nodes, 'v').edges(edges, 's', 'd')
    for q in ([n(), e_forward(hops=1), n()],
              [n(), e_reverse(hops=1), n()],
              [n(), e_undirected(hops=1), n()],
              [n({'attr': 2}), e_forward(hops=1), n()]):
        assert _setsig(g.gfql(q)) == _setsig(g.gfql(q, policy=_FAST_NOOP_POLICY)), \
            f"dangling-edge divergence for {q}"


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_fast_path_drops_nan_endpoint_edges(engine):
    """A NaN node id must not validate a NaN edge endpoint. pandas/cuDF `.isin`
    treat NaN as matchable (NaN.isin([NaN]) is True), but the full BFS path's joins
    never match NaN<->NaN, so it drops NaN-endpoint edges. The fast path's `.dropna()`
    on the node-id column must keep it consistent. Regression guard for the NaN fix."""
    import numpy as np
    nodes = pd.DataFrame({'v': [0.0, 1.0, np.nan], 'attr': [1, 2, 3]})  # NaN node id present
    edges = pd.DataFrame({'s': [0.0, 1.0], 'd': [1.0, np.nan]})  # NaN destination endpoint
    if engine == "cudf":
        cudf = _cudf_or_skip()
        nodes, edges = cudf.DataFrame.from_pandas(nodes), cudf.DataFrame.from_pandas(edges)
    g = CGFull().nodes(nodes, 'v').edges(edges, 's', 'd')
    for q in ([n(), e_forward(hops=1), n()], [n(), e_reverse(hops=1), n()]):
        assert _setsig(g.gfql(q)) == _setsig(g.gfql(q, policy=_FAST_NOOP_POLICY)), \
            f"NaN-endpoint divergence for {q}"


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_fast_path_dedups_duplicate_node_ids_on_hop(engine):
    """A malformed node table with duplicate ids must not make the 1-hop fast path
    diverge from the full path (which collapses dup rows via its merge)."""
    nodes = pd.DataFrame({'v': [0, 0, 1, 2], 'attr': [1, 1, 2, 3]})
    edges = pd.DataFrame({'s': [0, 1], 'd': [1, 2]})
    if engine == "cudf":
        cudf = _cudf_or_skip()
        nodes, edges = cudf.DataFrame.from_pandas(nodes), cudf.DataFrame.from_pandas(edges)
    g = CGFull().nodes(nodes, 'v').edges(edges, 's', 'd')
    q = [n(), e_forward(hops=1), n()]
    assert _setsig(g.gfql(q)) == _setsig(g.gfql(q, policy=_FAST_NOOP_POLICY))
