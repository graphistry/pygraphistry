import os
from typing import Callable, List, Sequence, Tuple

import pandas as pd
import pytest

from graphistry.compute.ast import ASTEdgeUndirected, ASTNode, ASTEdge, ASTObject, n, e, e_undirected, e_forward, e_reverse
from graphistry.tests.compute.gfql.routes.registry import Frames, register
from graphistry.compute.chain import Chain, _try_chain_fast_path
from graphistry.compute.typing import DataFrameT
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


def _cudf_at_least_26() -> bool:
    try:
        import cudf
    except ImportError:
        return False
    return int(str(cudf.__version__).split(".")[0]) >= 26


def _cudf_or_skip():
    if not (os.environ.get("TEST_CUDF") == "1"):
        pytest.skip("cuDF lane: set TEST_CUDF=1 (e.g. on dgx-spark)")
    return pytest.importorskip("cudf")


_FAST_FRAMES = Frames(
    pd.DataFrame({'v': [0, 1, 2, 3, 4], 'attr': [10, 20, 30, 40, 50]}),
    pd.DataFrame({'s': [0, 1, 2, 3, 0], 'd': [1, 2, 3, 4, 2], 'w': [5, 6, 7, 8, 9]}),
    'v', 's', 'd')


def _fast_graph(engine):
    nodes, edges = _FAST_FRAMES.nodes, _FAST_FRAMES.edges
    if engine == "cudf":
        cudf = _cudf_or_skip()
        nodes = cudf.from_pandas(nodes)
        edges = cudf.from_pandas(edges)
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
_FAST_SHAPES: List[Tuple[str, Callable[[], List[ASTObject]]]] = register("test_chain.fast", [
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
    # DELIBERATE RULE CHANGE: naming an op is a PROJECTION concern, not a traversal one,
    # so it no longer decides which engine path runs. `_tag_fast_path_aliases` reconstructs
    # the alias flag columns `combine_steps` would have merged in, so these are now FAST
    # shapes. (Was `("named_node", ...)` under _BYPASS_SHAPES: that entry encoded the old
    # "any alias -> decline" gate, not a semantic guarantee.)
    ("named_src", lambda: [n(name='x'), e_forward(hops=1), n()]),
    ("named_dst", lambda: [n(), e_forward(hops=1), n(name='y')]),
    ("named_edge", lambda: [n(), e_forward(hops=1, name='r'), n()]),
    ("named_all_fwd", lambda: [n(name='x'), e_forward(hops=1, name='r'), n(name='y')]),
    ("named_all_rev", lambda: [n(name='x'), e_reverse(hops=1, name='r'), n(name='y')]),
    ("named_filtered", lambda: [n({'attr': 10}, name='x'), e_forward(hops=1), n(name='y')]),
], _FAST_FRAMES, tags=("native-fast",))

# shapes that BYPASS the fast path (still must be correct via the full path)
_BYPASS_SHAPES: List[Tuple[str, Callable[[], List[ASTObject]]]] = register("test_chain.bypass", [
    ("hops_2", lambda: [n(), e_forward(hops=2), n()]),
    ("filtered_undirected", lambda: [n({'attr': 10}), e_undirected(hops=1), n({'attr': 30})]),
    # Named + undirected STAYS a bypass: an undirected edge makes a node reachable as
    # EITHER endpoint, so "which alias does this node carry" is not derivable from the
    # endpoint columns the way it is for a directed hop.
    ("named_undirected", lambda: [n(name='x'), e_undirected(hops=1), n(name='y')]),
    # prune_to_endpoints: fast path returns both endpoints; full path keeps only the
    # arrival side. Must bypass the fast path (regression guard for the prune gate).
    ("prune_endpoints_fwd", lambda: [n(), e_forward(hops=1, prune_to_endpoints=True), n()]),
    ("prune_endpoints_rev", lambda: [n(), e_reverse(hops=1, prune_to_endpoints=True), n()]),
], _FAST_FRAMES, tags=("native-fast-bypass",), row_tags={"prune_endpoints_fwd": ("#2053",), "prune_endpoints_rev": ("#2053",)})


_CUDF_26_DIVERGENT = {"prune_endpoints_fwd", "prune_endpoints_rev"}  # graphistry/pygraphistry#2043


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("label,build", _FAST_SHAPES + _BYPASS_SHAPES,
                         ids=[s[0] for s in _FAST_SHAPES + _BYPASS_SHAPES])
def test_fast_path_differential_parity_vs_full_path(engine, label, build, request):
    if engine == "cudf" and label in _CUDF_26_DIVERGENT and _cudf_at_least_26():
        request.applymarker(pytest.mark.xfail(strict=True, reason="graphistry/pygraphistry#2043"))
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


# Named shapes whose ALIAS FLAG COLUMNS (not merely node/edge sets) must match the full
# path. `_setsig` above compares ids only, so it cannot see a wrong alias tag.
_NAMED_ALIAS_SHAPES: List[Tuple[str, Callable[[], List[ASTObject]]]] = register("test_chain.named_alias", [
    ("src_only", lambda: [n(name='x'), e_forward(hops=1), n()]),
    ("dst_only", lambda: [n(), e_forward(hops=1), n(name='y')]),
    ("edge_only", lambda: [n(), e_forward(hops=1, name='r'), n()]),
    ("all_forward", lambda: [n(name='x'), e_forward(hops=1, name='r'), n(name='y')]),
    ("all_reverse", lambda: [n(name='x'), e_reverse(hops=1, name='r'), n(name='y')]),
    ("src_filtered", lambda: [n({'attr': 10}, name='x'), e_forward(hops=1, name='r'), n(name='y')]),
    ("dst_filtered", lambda: [n(name='x'), e_forward(hops=1, name='r'), n({'attr': 30}, name='y')]),
    ("edge_match", lambda: [n(name='x'), e_forward(hops=1, edge_match={'w': 5}, name='r'), n(name='y')]),
    # DEAD END: attr==50 is node 4, which has no outgoing edge. The tag keys on the
    # SURVIVING EDGES, so the alias must come back False/empty rather than True.
    ("dead_end_seed", lambda: [n({'attr': 50}, name='x'), e_forward(hops=1, name='r'), n(name='y')]),
], _FAST_FRAMES, tags=("alias",))


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("label,build", _NAMED_ALIAS_SHAPES,
                         ids=[s[0] for s in _NAMED_ALIAS_SHAPES])
@pytest.mark.route_engaged("native-fast")
def test_fast_path_named_alias_columns_match_full_path(engine, label, build):
    """The capability this fast-path extension actually adds: when the traversal is
    served without the BFS, the alias flag columns `combine_steps` would have merged in
    are RECONSTRUCTED from the surviving edges, and must be identical to the full path's
    — column set, per-row values and all. Serve-asserted so a gate regression that
    declines everything fails here instead of passing vacuously."""
    from graphistry.compute.chain import _try_chain_fast_path
    from graphistry.Engine import Engine
    g = _fast_graph(engine)
    if engine == "pandas":
        assert _try_chain_fast_path(g, build(), Engine.PANDAS, None) is not None, \
            f"{label}: named shape must be SERVED by the fast path"
    fast = g.gfql(build())
    full = g.gfql(build(), policy=_FAST_NOOP_POLICY)

    def flags(df: DataFrameT, key_cols: Sequence[str]) -> pd.DataFrame:
        # cuDF frames satisfy DataFrameT structurally but only they carry .to_pandas()
        pdf = pd.DataFrame(df.to_pandas() if "cudf" in type(df).__module__ else df)  # type: ignore[attr-defined]
        alias_cols = sorted(set(pdf.columns) & {'x', 'y', 'r'})
        cols = list(key_cols) + alias_cols
        out = pdf[cols].sort_values(cols).reset_index(drop=True)
        # bool vs object is the same merge artifact as int64 vs float64 (see
        # test_fast_path_preserves_int_node_dtypes); this test pins the alias VALUES,
        # and the dtype rule is pinned separately.
        for c in alias_cols:
            out[c] = out[c].astype(bool)
        return out

    pd.testing.assert_frame_equal(flags(fast._nodes, ['v']), flags(full._nodes, ['v']))
    pd.testing.assert_frame_equal(flags(fast._edges, ['s', 'd']), flags(full._edges, ['s', 'd']))


def _norm_all_cols(df: DataFrameT, key_cols: Sequence[str]) -> pd.DataFrame:
    """Full-frame canonicalization: ALL columns, sorted column order, key-sorted rows."""
    # cuDF frames satisfy DataFrameT structurally but only they carry .to_pandas()
    pdf = pd.DataFrame(df.to_pandas() if "cudf" in type(df).__module__ else df)  # type: ignore[attr-defined]
    cols = sorted(pdf.columns)
    return pdf[cols].sort_values(list(key_cols)).reset_index(drop=True)


def _assert_full_frame_value_parity(fast: DataFrameT, full: DataFrameT,
                                    key_cols: Sequence[str]) -> None:
    """Same columns, same per-row VALUES on every column. Dtype is deliberately NOT
    compared: the served lane keeps the Cypher-conformant int64/bool where the full
    path's merges upcast to float64/object (pinned separately)."""
    f, u = _norm_all_cols(fast, key_cols), _norm_all_cols(full, key_cols)
    assert list(f.columns) == list(u.columns), f"column sets differ: {list(f.columns)} vs {list(u.columns)}"
    assert len(f) == len(u), f"row counts differ: {len(f)} vs {len(u)}"
    for c in f.columns:
        pd.testing.assert_series_equal(
            f[c], u[c], check_names=False, check_dtype=False, check_categorical=False)


# Named served shapes for FULL-FRAME parity. `_setsig` compares id sets and the flags
# test compares alias columns, so before this NO test compared the carried DATA columns
# ('attr', 'w') of a named served result against the full path.
_NAMED_VALUE_PARITY_SHAPES: List[Tuple[str, Callable[[], List[ASTObject]]]] = register("test_chain.named_value_parity", [
    ("all_forward", lambda: [n(name='x'), e_forward(hops=1, name='r'), n(name='y')]),
    ("all_reverse", lambda: [n(name='x'), e_reverse(hops=1, name='r'), n(name='y')]),
    ("seed_filtered", lambda: [n({'attr': 10}, name='x'), e_forward(hops=1, name='r'), n(name='y')]),
    ("edge_match", lambda: [n(name='x'), e_forward(hops=1, edge_match={'w': 5}, name='r'), n(name='y')]),
], _FAST_FRAMES, tags=("alias", "values"))


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("label,build", _NAMED_VALUE_PARITY_SHAPES,
                         ids=[s[0] for s in _NAMED_VALUE_PARITY_SHAPES])
@pytest.mark.route_engaged("native-fast")
def test_fast_path_named_full_frame_value_parity(engine, label, build):
    """POSITIVE, whole-frame: a named served result must carry the same VALUES as the
    full path on EVERY column — ids, data columns, and alias flags — not just the id
    sets and flag columns the other tests pin. Also pins the served lane's documented
    dtype promises: data ints stay int64 and alias flags are real bools."""
    from graphistry.compute.chain import _try_chain_fast_path
    from graphistry.Engine import Engine
    g = _fast_graph(engine)
    if engine == "pandas":
        assert _try_chain_fast_path(g, build(), Engine.PANDAS, None) is not None, \
            f"{label}: named shape must be SERVED by the fast path"
    fast = g.gfql(build())
    full = g.gfql(build(), policy=_FAST_NOOP_POLICY)
    _assert_full_frame_value_parity(fast._nodes, full._nodes, ['v'])
    _assert_full_frame_value_parity(fast._edges, full._edges, ['s', 'd'])
    # served-lane dtype promises (the full path upcasts via merge; pinned in
    # test_fast_path_preserves_int_node_dtypes and the CHANGELOG dtype note)
    fn = _norm_all_cols(fast._nodes, ['v'])
    assert fn['attr'].dtype.kind == 'i', "served lane must keep int node attrs int"
    for c in set(fn.columns) & {'x', 'y'}:
        assert fn[c].dtype == bool, f"served alias flag {c} must be bool, got {fn[c].dtype}"
    fe = _norm_all_cols(fast._edges, ['s', 'd'])
    for c in set(fe.columns) & {'r'}:
        assert fe[c].dtype == bool, f"served alias flag {c} must be bool, got {fe[c].dtype}"


# Named shapes whose CORRECT answer is EMPTY. The serve gate cannot see result
# cardinality, so these all engage the fast path — and an empty answer must come back
# as the right empty SHAPE (alias columns present, zero rows), not a throw and not a
# missing-column frame.
_NAMED_EMPTY_SHAPES: List[Tuple[str, Callable[[], List[ASTObject]]]] = register("test_chain.named_empty", [
    # seed filter matches no node at all (distinct from dead_end_seed, which matches
    # a node that has no surviving edge)
    ("zero_seed", lambda: [n({'attr': 999}, name='x'), e_forward(hops=1, name='r'), n(name='y')]),
    ("zero_dst", lambda: [n(name='x'), e_forward(hops=1, name='r'), n({'attr': 999}, name='y')]),
    ("zero_edge_match", lambda: [n(name='x'), e_forward(hops=1, edge_match={'w': 999}, name='r'), n(name='y')]),
], _FAST_FRAMES, tags=("alias", "empty"))


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("label,build", _NAMED_EMPTY_SHAPES,
                         ids=[s[0] for s in _NAMED_EMPTY_SHAPES])
@pytest.mark.route_engaged("native-fast")
def test_fast_path_named_empty_result_matches_full_path(engine, label, build):
    """POSITIVE boundary: named patterns matching ZERO rows are still served, and the
    empty result must be shape-identical to the full path — same columns INCLUDING the
    alias flag columns, zero rows, no exception."""
    from graphistry.compute.chain import _try_chain_fast_path
    from graphistry.Engine import Engine
    g = _fast_graph(engine)
    if engine == "pandas":
        assert _try_chain_fast_path(g, build(), Engine.PANDAS, None) is not None, \
            f"{label}: empty-result named shape must still be SERVED"
    fast = g.gfql(build())
    full = g.gfql(build(), policy=_FAST_NOOP_POLICY)
    fn, un = _norm_all_cols(fast._nodes, ['v']), _norm_all_cols(full._nodes, ['v'])
    fe, ue = _norm_all_cols(fast._edges, ['s', 'd']), _norm_all_cols(full._edges, ['s', 'd'])
    assert len(fn) == 0 and len(fe) == 0, f"{label}: expected empty result"
    assert list(fn.columns) == list(un.columns), "empty nodes must keep alias columns"
    assert list(fe.columns) == list(ue.columns), "empty edges must keep alias columns"
    assert {'x', 'y'} <= set(fn.columns) and 'r' in fe.columns
    assert len(un) == 0 and len(ue) == 0


@pytest.mark.route_engaged("native-fast")
def test_fast_path_named_zero_edge_graph_matches_full_path():
    """POSITIVE boundary: a graph with an EMPTY edge table. The named pattern is served,
    and both lanes must agree on the all-empty answer with alias columns present."""
    from graphistry.compute.chain import _try_chain_fast_path
    from graphistry.Engine import Engine
    nodes = pd.DataFrame({'v': [0, 1], 'attr': [10, 20]})
    edges = pd.DataFrame({'s': pd.Series([], dtype='int64'),
                          'd': pd.Series([], dtype='int64'),
                          'w': pd.Series([], dtype='int64')})
    g = CGFull().nodes(nodes, 'v').edges(edges, 's', 'd')
    ops = [n(name='x'), e_forward(hops=1, name='r'), n(name='y')]
    assert _try_chain_fast_path(g, ops, Engine.PANDAS, None) is not None
    fast = g.gfql(ops)
    full = g.gfql(ops, policy=_FAST_NOOP_POLICY)
    for res in (fast, full):
        assert res._nodes.shape[0] == 0 and res._edges.shape[0] == 0
    assert {'x', 'y'} <= set(fast._nodes.columns) and 'r' in fast._edges.columns
    assert list(sorted(fast._nodes.columns)) == list(sorted(full._nodes.columns))
    assert list(sorted(fast._edges.columns)) == list(sorted(full._edges.columns))


@pytest.mark.parametrize("route", ["fast_default", "full_policy"])
def test_fast_path_duplicate_alias_still_raises_e201(route):
    """NEGATIVE, end-to-end: duplicate NODE alias reuse must still RAISE E201 through
    the public API. The gate-level decline (asserted in
    test_fast_path_gating_returns_none_for_ineligible) exists precisely so
    `combine_steps` stays in charge of this error; if the decline ever regressed, alias
    reuse would silently SUCCEED on the served lane — this pins the user-visible raise
    on both routes so that regression cannot land quietly."""
    from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
    g = _fast_graph("pandas")
    kwargs = {} if route == "fast_default" else {"policy": _FAST_NOOP_POLICY}
    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql([n(name='x'), e_forward(hops=1), n(name='x')], **kwargs)
    assert exc_info.value.code == ErrorCode.E201
    assert "'x'" in str(exc_info.value)


def test_fast_path_cross_type_alias_share_declines_and_matches():
    """NEGATIVE boundary pinning a subtlety found while testing: a NODE alias and an
    EDGE alias sharing one name is NOT an E201 — `combine_steps` checks duplicates
    per frame (node pass vs edge pass), and the two flag columns land on different
    frames. The gate still declines it conservatively (duplicate_alias_edge in the
    ineligible list), which is safe exactly because the full path serves it — so pin
    that the two routes AGREE: no raise, same values, the flag on both frames."""
    g = _fast_graph("pandas")
    ops = lambda: [n(name='r'), e_forward(hops=1, name='r'), n()]  # noqa: E731
    default_route = g.gfql(ops())
    policy_route = g.gfql(ops(), policy=_FAST_NOOP_POLICY)
    for res in (default_route, policy_route):
        assert 'r' in res._nodes.columns and 'r' in res._edges.columns
    _assert_full_frame_value_parity(default_route._nodes, policy_route._nodes, ['v'])
    _assert_full_frame_value_parity(default_route._edges, policy_route._edges, ['s', 'd'])


@pytest.mark.route_engaged("native-fast")
def test_fast_path_named_is_served_with_a_valid_resident_index():
    """A NAMED pattern with BOTH resident indexes validly covering the directed hop is
    served by the chain fast path: by the time it runs, the indexed kernel has already
    declined the middle (``_handle_boundary_calls``), so deferring served nothing. The
    answer must equal the full path's; unnamed and reverse (out-adjacency only) patterns
    keep serving as before."""
    from graphistry.compute.chain import _try_chain_fast_path
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index.api import create_index
    g = _fast_graph("pandas")
    gi = create_index(create_index(g, 'edge_out_adj'), 'node_id')
    named = [n(name='x'), e_forward(hops=1), n(name='y')]
    assert _try_chain_fast_path(gi, named, Engine.PANDAS, None) is not None, \
        "named + valid covering index is served (the kernel already declined)"
    assert _try_chain_fast_path(gi, [n(), e_forward(hops=1), n()], Engine.PANDAS, None) is not None
    assert _try_chain_fast_path(gi, [n(name='x'), e_reverse(hops=1), n(name='y')], Engine.PANDAS, None) is not None
    served = gi.gfql(named)
    full = g.gfql(named, policy=_FAST_NOOP_POLICY)
    assert _setsig(served) == _setsig(full)
    _assert_full_frame_value_parity(served._nodes, full._nodes, ['v'])
    _assert_full_frame_value_parity(served._edges, full._edges, ['s', 'd'])


@pytest.mark.route_engaged("native-fast")
def test_fast_path_named_datetime_categorical_columns_ride_along():
    """POSITIVE dtype edge: datetime64 and categorical NODE columns must ride through
    the served named lane unchanged — same values as the full path, dtypes preserved
    (the served lane never merges, so it must not degrade either dtype) — and a seed
    FILTER over the categorical column must both stay served and agree on values."""
    from graphistry.compute.chain import _try_chain_fast_path
    from graphistry.Engine import Engine
    nodes = pd.DataFrame({
        'v': [0, 1, 2, 3, 4],
        'ts': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
        'cat': pd.Categorical(['a', 'b', 'a', 'c', 'b']),
    })
    edges = pd.DataFrame({'s': [0, 1, 2, 3, 0], 'd': [1, 2, 3, 4, 2], 'w': [5, 6, 7, 8, 9]})
    g = CGFull().nodes(nodes, 'v').edges(edges, 's', 'd')
    ops = [n(name='x'), e_forward(hops=1, name='r'), n(name='y')]
    assert _try_chain_fast_path(g, ops, Engine.PANDAS, None) is not None
    fast = g.gfql(ops)
    full = g.gfql(ops, policy=_FAST_NOOP_POLICY)
    _assert_full_frame_value_parity(fast._nodes, full._nodes, ['v'])
    _assert_full_frame_value_parity(fast._edges, full._edges, ['s', 'd'])
    # dtype preservation is asserted against the INPUT (and the full path), not a
    # hardcoded unit: pandas 2.x infers datetime64[ns] here, pandas 3.x datetime64[us]
    assert fast._nodes['ts'].dtype == nodes['ts'].dtype
    assert fast._nodes['ts'].dtype == full._nodes['ts'].dtype
    assert str(fast._nodes['ts'].dtype).startswith('datetime64[')
    assert str(fast._nodes['cat'].dtype) == 'category'
    assert fast._nodes['cat'].dtype == full._nodes['cat'].dtype
    # categorical seed filter: still served, same answer
    ops2 = [n({'cat': 'a'}, name='x'), e_forward(hops=1), n(name='y')]
    assert _try_chain_fast_path(g, ops2, Engine.PANDAS, None) is not None
    f2 = g.gfql(ops2)
    u2 = g.gfql(ops2, policy=_FAST_NOOP_POLICY)
    _assert_full_frame_value_parity(f2._nodes, u2._nodes, ['v'])
    _assert_full_frame_value_parity(f2._edges, u2._edges, ['s', 'd'])


# Alias names that SHADOW a real column WITHOUT breaking parity. Today BOTH lanes
# overwrite the shadowed column with the flag, identically — that (pre-existing,
# full-path) contract is what these pin, so a lane can't drift to a different
# overwrite/raise behavior alone. The FROM-side binding columns and the node-id
# binding are excluded here: those wrong-served (diverged) before, are now GATED to
# decline, and are pinned by the two regression tests below.
_ALIAS_SHADOW_SHAPES: List[Tuple[str, Callable[[], List[ASTObject]]]] = register("test_chain.alias_shadow", [
    ("node_alias_shadows_node_data_col", lambda: [n(name='attr'), e_forward(hops=1), n()]),
    ("edge_alias_shadows_edge_data_col", lambda: [n(), e_forward(hops=1, name='w'), n()]),
    # edge aliases named like the source/destination/edge-id bindings are rejected before
    # execution on both routes (#2050), pinned below
    # cross-frame names are NOT collisions: nodes have no 'w', edges have no 'v'
    ("node_alias_named_like_edge_col", lambda: [n(name='w'), e_forward(hops=1), n()]),
    ("edge_alias_named_like_node_id", lambda: [n(), e_forward(hops=1, name='v'), n()]),
], _FAST_FRAMES, tags=("alias-collision",))


@pytest.mark.parametrize("label,build", _ALIAS_SHADOW_SHAPES,
                         ids=[s[0] for s in _ALIAS_SHADOW_SHAPES])
def test_fast_path_alias_shadowing_column_matches_full_path(label, build):
    """NEGATIVE-ish boundary: alias names that collide with existing data/binding
    columns must behave IDENTICALLY on both lanes (today: same silent overwrite the
    full path has always done; cross-frame same-name cases are no-ops). Edge key
    columns may themselves be overwritten here, so edges are keyed on 'w' (unique)."""
    g = _fast_graph("pandas")
    fast = g.gfql(build())
    full = g.gfql(build(), policy=_FAST_NOOP_POLICY)
    _assert_full_frame_value_parity(fast._nodes, full._nodes, ['v'])
    _assert_full_frame_value_parity(fast._edges, full._edges, ['w'])


@pytest.mark.parametrize("build", [
    lambda: [n(), e_forward(hops=1, name='d'), n()],
    lambda: [n(), e_reverse(hops=1, name='s'), n()],
], ids=["fwd_alias_is_dst_binding", "rev_alias_is_src_binding"])
def test_edge_alias_named_like_a_to_side_binding_is_rejected_on_both_routes(build):
    """An edge alias equal to the hop's TO-side binding used to be served with the marker
    overwriting the endpoint column on both routes (#2050); it is now the same typed decline
    as the FROM-side and node-id collisions, before either route runs."""
    from graphistry.compute.exceptions import GFQLValidationError
    g = _fast_graph("pandas")
    with pytest.raises(GFQLValidationError):
        g.gfql(build())
    with pytest.raises(GFQLValidationError):
        g.gfql(build(), policy=_FAST_NOOP_POLICY)


@pytest.mark.parametrize("build", [
    lambda: [n(), e_forward(hops=1, name='s'), n()],
    lambda: [n(), e_reverse(hops=1, name='d'), n()],
], ids=["fwd_alias_is_src_binding", "rev_alias_is_dst_binding"])
def test_fast_path_edge_alias_colliding_with_from_binding_matches_full_path(build):
    """REGRESSION (wrong-serve, found by adversarial parity testing, now GATED): an
    edge alias equal to the hop's FROM-side binding column (forward+name=src,
    reverse+name=dst) used to be SERVED with a DIFFERENT node set than the full path
    (fast [0..4] vs full [1..4] on this fixture — the full path's flag overwrite
    corrupts its own node reduction). The gate now DECLINES it; TO-side collisions
    keep parity and stay served (pinned above). Both routes must agree: both raise,
    or both return the same frames."""
    from graphistry.Engine import Engine
    g = _fast_graph("pandas")
    assert _try_chain_fast_path(g, build(), Engine.PANDAS, None) is None, \
        "edge alias == FROM-side binding must DECLINE the fast path"
    try:
        full = g.gfql(build(), policy=_FAST_NOOP_POLICY)
        full_raised = None
    except Exception as ex:  # noqa: BLE001 — parity contract, not error-type contract
        full, full_raised = None, type(ex)
    if full_raised is not None:
        with pytest.raises(full_raised):
            g.gfql(build())
    else:
        fast = g.gfql(build())
        assert full is not None
        _assert_full_frame_value_parity(fast._nodes, full._nodes, ['v'])
        _assert_full_frame_value_parity(fast._edges, full._edges, ['w'])


@pytest.mark.parametrize("build", [
    lambda: [n(name='v'), e_forward(hops=1), n()],
    lambda: [n(), e_forward(hops=1), n(name='v')],
], ids=["n0_alias_is_node_binding", "n2_alias_is_node_binding"])
def test_fast_path_alias_colliding_with_node_id_binding_matches_full_path(build):
    """REGRESSION (wrong-serve, found by adversarial parity testing, now GATED): a
    node alias equal to the NODE ID BINDING column ('v') used to be SERVED, silently
    overwriting the id column with the bool alias flag while the full path raised.
    The gate now DECLINES it, so the two lanes are observationally equivalent: both
    raise, or both return the same frames (today: both raise pandas' ValueError)."""
    from graphistry.Engine import Engine
    g = _fast_graph("pandas")
    assert _try_chain_fast_path(g, build(), Engine.PANDAS, None) is None, \
        "node alias == node-id binding must DECLINE the fast path"
    try:
        full = g.gfql(build(), policy=_FAST_NOOP_POLICY)
        full_raised = None
    except Exception as ex:  # noqa: BLE001 — parity contract, not error-type contract
        full, full_raised = None, type(ex)
    if full_raised is not None:
        with pytest.raises(full_raised):
            g.gfql(build())
    else:
        fast = g.gfql(build())
        assert full is not None
        _assert_full_frame_value_parity(fast._nodes, full._nodes, ['v'])
        _assert_full_frame_value_parity(fast._edges, full._edges, ['s', 'd'])


@pytest.mark.route_engaged("native-fast")
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
        # DELIBERATE RULE CHANGE (was asserted INELIGIBLE as "named_node"): an alias is a
        # PROJECTION concern, so it no longer gates the traversal path. The old assertion
        # encoded the gate itself, not a semantic the fast path could not meet — the alias
        # flag columns are reconstructible from the surviving edges
        # (`_tag_fast_path_aliases`), which is exactly what `combine_steps` computes.
        [n(name='x'), e_forward(hops=1), n()],
        [n(), e_forward(hops=1), n(name='y')],
        [n(), e_forward(hops=1, name='r'), n()],
        [n(name='x'), e_forward(hops=1, name='r'), n(name='y')],
        [n(name='x'), e_reverse(hops=1, name='r'), n(name='y')],
    ]
    for ops in eligible:
        assert _try_chain_fast_path(g, ops, Engine.PANDAS, None) is not None, f"should accept {ops}"

    ineligible = [
        ("hops_2", [n(), e_forward(hops=2), n()], None, Engine.PANDAS),
        ("filtered_undirected", [n({'attr': 10}), e_undirected(hops=1), n({'attr': 30})], None, Engine.PANDAS),
        # NEW declines that come with serving aliases. Undirected: a node is reachable as
        # EITHER endpoint, so alias identity is not derivable from the endpoint columns.
        ("named_undirected", [n(name='x'), e_undirected(hops=1), n(name='y')], None, Engine.PANDAS),
        ("named_undirected_edge", [n(), e_undirected(hops=1, name='r'), n(name='y')], None, Engine.PANDAS),
        # Duplicate alias reuse is E201, and `combine_steps` is what raises it; serving
        # here would BYPASS the check and let alias reuse silently succeed.
        ("duplicate_alias_nodes", [n(name='x'), e_forward(hops=1), n(name='x')], None, Engine.PANDAS),
        ("duplicate_alias_edge", [n(name='r'), e_forward(hops=1, name='r'), n()], None, Engine.PANDAS),
        ("node_query", [n(query='attr > 5'), e_forward(hops=1), n()], None, Engine.PANDAS),
        ("prune_endpoints", [n(), e_forward(hops=1, prune_to_endpoints=True), n()], None, Engine.PANDAS),
        ("seeded", [n()], seed, Engine.PANDAS),
        # MIXED supported + unsupported: an alias is now a served concern, but it must
        # never FLIP an otherwise-ineligible shape to served — the unsupported piece
        # (multi-hop, queries, richer predicates, prune, seeds) still declines the whole op list.
        ("named_hops_2", [n(name='x'), e_forward(hops=2), n(name='y')], None, Engine.PANDAS),
        ("named_node_query", [n(query='attr > 5', name='x'), e_forward(hops=1), n(name='y')], None, Engine.PANDAS),
        ("named_edge_query", [n(name='x'), e_forward(hops=1, edge_query='w > 5'), n(name='y')], None, Engine.PANDAS),
        ("named_source_node_match", [n(name='x'), e_forward(hops=1, source_node_match={'attr': 10}), n(name='y')], None, Engine.PANDAS),
        ("named_prune_endpoints", [n(name='x'), e_forward(hops=1, prune_to_endpoints=True), n(name='y')], None, Engine.PANDAS),
        ("named_seeded", [n(name='x'), e_forward(hops=1), n(name='y')], seed, Engine.PANDAS),
        # BINDING-COLLISION declines (wrong-serves found by adversarial parity testing):
        # a node alias equal to the node-id binding would overwrite the id column where
        # the full path raises; an edge alias equal to the hop's FROM-side binding made
        # the two lanes return different node sets. TO-side collisions keep parity and
        # remain served (see test_fast_path_alias_shadowing_column_matches_full_path).
        ("n0_alias_is_node_binding", [n(name='v'), e_forward(hops=1), n()], None, Engine.PANDAS),
        ("n2_alias_is_node_binding", [n(), e_forward(hops=1), n(name='v')], None, Engine.PANDAS),
        ("edge_alias_is_src_binding_fwd", [n(), e_forward(hops=1, name='s'), n()], None, Engine.PANDAS),
        ("edge_alias_is_dst_binding_rev", [n(), e_reverse(hops=1, name='d'), n()], None, Engine.PANDAS),
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
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
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
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = CGFull().nodes(nodes, 'v').edges(edges, 's', 'd')
    for q in ([n(), e_forward(hops=1), n()], [n(), e_reverse(hops=1), n()]):
        assert _setsig(g.gfql(q)) == _setsig(g.gfql(q, policy=_FAST_NOOP_POLICY)), \
            f"NaN-endpoint divergence for {q}"


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_fast_path_dedups_duplicate_node_ids_on_hop(engine, request):
    if engine == "cudf" and _cudf_at_least_26():
        request.applymarker(pytest.mark.xfail(strict=True, reason="graphistry/pygraphistry#2043"))
    """A malformed node table with duplicate ids must not make the 1-hop fast path
    diverge from the full path (which collapses dup rows via its merge)."""
    nodes = pd.DataFrame({'v': [0, 0, 1, 2], 'attr': [1, 1, 2, 3]})
    edges = pd.DataFrame({'s': [0, 1], 'd': [1, 2]})
    if engine == "cudf":
        cudf = _cudf_or_skip()
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = CGFull().nodes(nodes, 'v').edges(edges, 's', 'd')
    q = [n(), e_forward(hops=1), n()]
    assert _setsig(g.gfql(q)) == _setsig(g.gfql(q, policy=_FAST_NOOP_POLICY))
