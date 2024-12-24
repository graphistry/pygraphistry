import os
import pandas as pd
from graphistry.compute.predicates.is_in import is_in
import pytest

from graphistry.compute.ast import ASTNode, ASTEdge, n, e
from graphistry.tests.test_compute import CGFull


@pytest.fixture(scope='module')
def g_long_forwards_chain() -> CGFull:
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
def n_a(g_long_forwards_chain: CGFull) -> pd.DataFrame:
    return g_long_forwards_chain._nodes.query('v == "a"')


@pytest.fixture(scope='module')
def n_mt(g_long_forwards_chain: CGFull) -> pd.DataFrame:
    return g_long_forwards_chain._nodes[:0]

@pytest.fixture(scope='module')
def n_d(g_long_forwards_chain: CGFull) -> pd.DataFrame:
    return g_long_forwards_chain._nodes.query('v == "d"')


class TestMultiHopForward():
    """
    Test multi-hop as used by chain, corresponding to chain multi-hop tests
    """

    def test_hop_short_forward(self, g_long_forwards_chain: CGFull, n_a):
        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=2,
            to_fixed_point=False,
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'}
        ]

    def test_hop_short_back(self, g_long_forwards_chain: CGFull, n_mt, n_a):
        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b'],
                        'd': ['b', 'c']}),
                    on=['s', 'd'],
                    how='inner'
                ))
        g2 = g_reverse.hop(
            nodes=n_mt,
            hops=2,
            to_fixed_point=False,
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert len(g2._nodes) == 0
        assert len(g2._edges) == 0

    def test_hop_exact_forward(self, g_long_forwards_chain, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=3,
            to_fixed_point=False,
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_hop_exact_back(self, g_long_forwards_chain: CGFull, n_d, n_a):
        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['a', 'b', 'c', 'd'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c'],
                        'd': ['b', 'c', 'd']}),
                    on=['s', 'd'],
                    how='inner'
                ))
        g2 = g_reverse.hop(
            nodes=n_d,
            hops=3,
            to_fixed_point=False,
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]


    def test_hop_long_forward(self, g_long_forwards_chain, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd', 'e'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
            {'s': 'd', 'd': 'e'}
        ]

    def test_hop_long_back(self, g_long_forwards_chain: CGFull, n_d, n_a):
        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c', 'd', 'e'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c', 'd'],
                        'd': ['b', 'c', 'd', 'e']}),
                    on=['s', 'd'],
                    how='inner'
                ))
        g2 = g_reverse.hop(
            nodes=n_d,
            hops=4,
            to_fixed_point=False,
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_hop_predicates_ok_source_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            source_node_match={'w': is_in(['1', '2', '3'])},
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_source_back(self, g_long_forwards_chain: CGFull, n_a, n_d):

        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c', 'd'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c'],
                        'd': ['b', 'c', 'd']}),
                    on=['s', 'd'],
                    how='inner'
                ))

        g2 = g_reverse.hop(
            nodes=n_d,
            hops=4,
            to_fixed_point=False,
            destination_node_match={'w': is_in(['1', '2', '3'])},
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_edge_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            edge_match={
                't': is_in(['1', '2', '3']),
                'e': is_in(['2', '3', '4'])
            },
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_edge_back(self, g_long_forwards_chain: CGFull, n_a, n_d):

        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c', 'd'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c'],
                        'd': ['b', 'c', 'd']}),
                    on=['s', 'd'],
                    how='inner'
                ))

        g2 = g_reverse.hop(
            nodes=n_d,
            hops=4,
            to_fixed_point=False,
            edge_match={
                't': is_in(['1', '2', '3']),
                'e': is_in(['2', '3', '4'])
            },
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_destination_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            destination_node_match={'w': is_in(['2', '3', '4'])},
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_destination_back(self, g_long_forwards_chain: CGFull, n_a, n_d):

        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c', 'd'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c'],
                        'd': ['b', 'c', 'd']}),
                    on=['s', 'd'],
                    how='inner'
                ))

        g2 = g_reverse.hop(
            nodes=n_d,
            hops=4,
            to_fixed_point=False,
            source_node_match={'w': is_in(['2', '3', '4'])},
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            source_node_match={'w': is_in(['1', '2', '3'])},
            edge_match={
                't': is_in(['1', '2', '3']),
                'e': is_in(['2', '3', '4'])
            },
            destination_node_match={'w': is_in(['2', '3', '4'])},
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_back(self, g_long_forwards_chain: CGFull, n_a, n_d):

        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c', 'd'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c'],
                        'd': ['b', 'c', 'd']}),
                    on=['s', 'd'],
                    how='inner'
                ))

        g2 = g_reverse.hop(
            nodes=n_d,
            hops=4,
            to_fixed_point=False,
            destination_node_match={'w': is_in(['1', '2', '3'])},
            edge_match={
                't': is_in(['1', '2', '3']),
                'e': is_in(['2', '3', '4'])
            },
            source_node_match={'w': is_in(['2', '3', '4'])},
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_fail_source_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            source_node_match={'w': is_in([])},
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set([])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == []

    def test_hop_predicates_fail_edge_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            edge_match={
                't': is_in([]),
                'e': is_in([])
            },
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set([])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == []

    def test_hop_predicates_fail_destination_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            destination_node_match={'w': is_in([])},
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set([])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == []


def test_hop_binding_reuse():
    edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
    nodes1_df = pd.DataFrame({'v': ['a', 'b', 'c']})
    nodes2_df = pd.DataFrame({'s': ['a', 'b', 'c']})
    nodes3_df = pd.DataFrame({'d': ['a', 'b', 'c']})
    
    g1 = CGFull().nodes(nodes1_df, 'v').edges(edges_df, 's', 'd')
    g2 = CGFull().nodes(nodes2_df, 's').edges(edges_df, 's', 'd')
    g3 = CGFull().nodes(nodes3_df, 'd').edges(edges_df, 's', 'd')

    try:
        g1_hop = g1.hop()
        g2_hop = g2.hop()
        g3_hop = g3.hop()
    except NotImplementedError:
        return

    assert g1_hop._nodes.shape == g2_hop._nodes.shape
    assert g1_hop._edges.shape == g2_hop._edges.shape    
    assert g1_hop._nodes.shape == g3_hop._nodes.shape
    assert g1_hop._edges.shape == g3_hop._edges.shape    

def test_hop_simple_cudf_pd():
    nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_df = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    g_nodes = g.hop()
    assert isinstance(g_nodes._nodes, pd.DataFrame)
    assert len(g_nodes._nodes) == 3
    g_edges = g.hop()
    assert isinstance(g_edges._edges, pd.DataFrame)
    assert len(g_edges._edges) == 3


@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1",
)
def test_hop_simple_cudf():
    import cudf
    nodes_gdf = cudf.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_gdf = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')
    g_nodes = g.hop()
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 3
    g_edges = g.hop()
    assert isinstance(g_edges._edges, cudf.DataFrame)
    assert len(g_edges._edges) == 3

def test_hop_kv_cudf_pd():
    nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_df = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    g_nodes = g.hop(source_node_match=({'id': 0}))
    assert isinstance(g_nodes._nodes, pd.DataFrame)
    assert len(g_nodes._nodes) == 2
    g_edges = g.hop(edge_match={'src': 0})
    assert isinstance(g_edges._edges, pd.DataFrame)
    assert len(g_edges._edges) == 1

@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1",
)
def test_hop_kv_cudf():
    import cudf
    nodes_gdf = cudf.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_gdf = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')
    g_nodes = g.hop(source_node_match={'id': 0})
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 2
    g_edges = g.hop(edge_match={'src': 0})
    assert isinstance(g_edges._edges, cudf.DataFrame)
    assert len(g_edges._edges) == 1

def test_hop_pred_cudf_pd():
    nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_df = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    g_nodes = g.hop(source_node_match={'id': is_in([0])})
    assert isinstance(g_nodes._nodes, pd.DataFrame)
    assert len(g_nodes._nodes) == 2
    g_edges = g.hop(edge_match={'src': is_in([0])})
    assert isinstance(g_edges._edges, pd.DataFrame)
    assert len(g_edges._edges) == 1

@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1",
)
def test_hop_pred_cudf():
    import cudf
    nodes_gdf = cudf.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_gdf = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')
    g_nodes = g.hop(source_node_match={'id': is_in([0])})
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 2
    g_edges = g.hop(edge_match={'src': is_in([0])})
    assert isinstance(g_edges._edges, cudf.DataFrame)
    assert len(g_edges._edges) == 1
