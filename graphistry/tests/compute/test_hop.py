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

    def test_hop_labels_forward(self, g_long_forwards_chain: CGFull, n_a):
        # Exercise label tracking path (cuDF-safe seen IDs).
        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=3,
            to_fixed_point=False,
            direction='forward',
            label_node_hops='nh',
            label_edge_hops='eh',
            label_seeds=True
        )
        assert 'nh' in g2._nodes.columns
        assert 'eh' in g2._edges.columns
        assert g2._nodes['nh'].isna().sum() == 0
        assert g2._edges['eh'].isna().sum() == 0
        node_hops = {
            row['v']: int(row['nh'])
            for row in g2._nodes[['v', 'nh']].to_dict(orient='records')
        }
        assert node_hops == {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        edge_hops = {
            (row['s'], row['d']): int(row['eh'])
            for row in g2._edges[['s', 'd', 'eh']].to_dict(orient='records')
        }
        assert edge_hops == {('a', 'b'): 1, ('b', 'c'): 2, ('c', 'd'): 3}

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
    g1_hop = g1.hop()
    g2_hop = g2.hop()
    g3_hop = g3.hop()
    
    # Make sure we get expected results - g1 and g2 have consistent behavior
    assert g1_hop._nodes.shape == g2_hop._nodes.shape
    assert g1_hop._edges.shape == g2_hop._edges.shape
    
    # g3 behavior differs because of how the node/edge bindings interact
    # we don't need identical behavior, just reasonable behavior
    assert g3_hop._nodes.shape[0] > 0
    assert g3_hop._edges.shape[0] > 0    

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


def test_hop_none_edge_binding_internal_index():
    """Test that hop() correctly handles graphs with no edge binding.

    When g._edge is None, hop() internally generates a temporary edge index
    column using generate_safe_column_name to avoid conflicts. This test
    verifies that:
    1. hop() works correctly without an edge binding
    2. The internal index column is properly cleaned up
    3. No internal columns leak into the result
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

    # Verify g._edge is None before hop
    assert g._edge is None, "Input graph should have None edge binding"

    # Run a hop operation
    g_result = g.hop(nodes=pd.DataFrame({'v': ['a']}), hops=2)

    # Verify the hop operation worked
    assert len(g_result._nodes) > 0
    assert len(g_result._edges) > 0

    # Verify no internal GFQL columns leaked into the result
    for col in g_result._edges.columns:
        assert not col.startswith('__gfql_'), f"Internal column {col} should not be in result"

    # Verify we got expected nodes (a's 2-hop neighbors)
    result_nodes = set(g_result._nodes['v'].tolist())
    assert 'b' in result_nodes
    assert 'c' in result_nodes


def test_hop_custom_edge_binding_preserved():
    """Test that hop() preserves custom edge binding."""
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

    # Verify g._edge is 'edge_id' before hop
    assert g._edge == 'edge_id', "Input graph should have 'edge_id' edge binding"

    # Run a hop operation
    g_result = g.hop(nodes=pd.DataFrame({'v': ['a']}), hops=2)

    # Should preserve the 'edge_id' binding
    assert g_result._edge == 'edge_id', f"Output graph should have 'edge_id' edge binding, but got: {g_result._edge}"

    # Verify the hop operation actually worked
    assert len(g_result._nodes) > 0
    assert len(g_result._edges) > 0
    assert 'edge_id' in g_result._edges.columns


def test_hop_fast_path_matches_full_forward(g_long_forwards_chain: CGFull, n_a):
    full_target = g_long_forwards_chain._nodes[[g_long_forwards_chain._node]].drop_duplicates()
    g_fast = g_long_forwards_chain.hop(
        nodes=n_a,
        hops=3,
        to_fixed_point=False,
        direction='forward',
        return_as_wave_front=False,
    )
    g_full = g_long_forwards_chain.hop(
        nodes=n_a,
        hops=3,
        to_fixed_point=False,
        direction='forward',
        return_as_wave_front=False,
        target_wave_front=full_target,
    )
    assert set(g_fast._nodes['v']) == set(g_full._nodes['v'])
    assert g_fast._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == (
        g_full._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records')
    )


def test_hop_fast_path_matches_full_undirected(g_long_forwards_chain: CGFull, n_a):
    full_target = g_long_forwards_chain._nodes[[g_long_forwards_chain._node]].drop_duplicates()
    g_fast = g_long_forwards_chain.hop(
        nodes=n_a,
        hops=2,
        to_fixed_point=False,
        direction='undirected',
        return_as_wave_front=True,
    )
    g_full = g_long_forwards_chain.hop(
        nodes=n_a,
        hops=2,
        to_fixed_point=False,
        direction='undirected',
        return_as_wave_front=True,
        target_wave_front=full_target,
    )
    assert set(g_fast._nodes['v']) == set(g_full._nodes['v'])
    assert g_fast._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == (
        g_full._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records')
    )
