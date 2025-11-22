import pandas as pd
from common import NoAuthTestCase
from functools import lru_cache

from graphistry.compute.ast import is_in
from graphistry.tests.test_compute import CGFull

@lru_cache(maxsize=1)
def hops_graph() -> CGFull:
    nodes_df = pd.DataFrame([
        {'node': 'a'},
        {'node': 'b'},
        {'node': 'c'},
        {'node': 'd'},
        {'node': 'e'},
        {'node': 'f'},
        {'node': 'g'},
        {'node': 'h'},
        {'node': 'i'},
        {'node': 'j'},
        {'node': 'k'},
        {'node': 'l'},
        {'node': 'm'},
        {'node': 'n'},
        {'node': 'o'},
        {'node': 'p'}
    ]).assign(type='n')

    edges_df = pd.DataFrame([
        {'s': 'e', 'd': 'l'},
        {'s': 'l', 'd': 'b'},
        {'s': 'k', 'd': 'a'},
        {'s': 'e', 'd': 'g'},
        {'s': 'g', 'd': 'a'},
        {'s': 'd', 'd': 'f'},
        {'s': 'd', 'd': 'c'},
        {'s': 'd', 'd': 'j'},
        {'s': 'd', 'd': 'i'},
        {'s': 'd', 'd': 'h'},
        {'s': 'j', 'd': 'p'},
        {'s': 'i', 'd': 'n'},
        {'s': 'h', 'd': 'm'},
        {'s': 'j', 'd': 'o'},
        {'s': 'o', 'd': 'b'},
        {'s': 'm', 'd': 'a'},
        {'s': 'n', 'd': 'a'},
        {'s': 'p', 'd': 'b'},
    ]).assign(type='e')

    return CGFull().nodes(nodes_df, 'node').edges(edges_df, 's', 'd')


def simple_chain_graph() -> CGFull:
    nodes_df = pd.DataFrame([
        {'node': 'a'},
        {'node': 'b'},
        {'node': 'c'},
        {'node': 'd'},
    ])
    edges_df = pd.DataFrame([
        {'s': 'a', 'd': 'b'},
        {'s': 'b', 'd': 'c'},
        {'s': 'c', 'd': 'd'},
    ])
    return CGFull().nodes(nodes_df, 'node').edges(edges_df, 's', 'd')

class TestComputeHopMixin(NoAuthTestCase):


    def test_hop_0(self):

        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: []}), 0)
        assert g2._nodes.shape == (0, 2)
        assert g2._edges.shape == (0, 3)

    def test_hop_0b(self):

        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['d']}), 0)
        assert g2._nodes.shape == (0, 2)
        assert g2._edges.shape == (0, 3)

    def test_hop_1_1_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['d']}), 1)
        assert g2._nodes.shape == (6, 2)
        assert (g2._nodes[g2._node].sort_values().to_list() ==  # noqa: W504
            sorted(['f', 'j', 'd','i', 'c', 'h']))
        assert g2._edges.shape == (5, 3)

    def test_hop_2_1_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['k', 'd']}), 1)
        assert g2._nodes.shape == (8, 2)
        assert g2._edges.shape == (6, 3)

    def test_hop_2_2_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['k', 'd']}), 2)
        assert g2._nodes.shape == (12, 2)
        assert g2._edges.shape == (10, 3)

    def test_hop_2_all_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['k', 'd']}), to_fixed_point=True)
        assert g2._nodes.shape == (13, 2)
        assert g2._edges.shape == (14, 3)

    def test_hop_1_2_undirected(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['j']}), 2, direction='undirected')
        assert g2._nodes.shape == (9, 2)
        assert g2._edges.shape == (9, 3)

    def test_hop_1_all_reverse(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['b']}), direction='reverse', to_fixed_point=True)
        assert g2._nodes.shape == (7, 2)
        assert g2._edges.shape == (7, 3)

    #edge filter

    def test_hop_1_1_forwards_edge(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['d']}), 1, edge_match={'d': 'f'})
        assert g2._nodes.shape == (2, 2)
        assert (g2._nodes[g2._node].sort_values().to_list() ==  # noqa: W504
            sorted(['f', 'd']))
        assert g2._edges.shape == (1, 3)

    def test_hop_post_match(self):
        g = hops_graph()
        g2 = g.hop(destination_node_match={'node': 'b'})
        assert g2._nodes.shape == (4, 2)
        assert (g2._nodes[g2._node].sort_values().to_list() ==  # noqa: W504
            sorted(['b', 'l', 'o', 'p']))
        assert g2._edges.shape == (3, 3)

    def test_hop_pre_match(self):
        g = hops_graph()
        g2 = g.hop(source_node_match={'node': 'e'})
        assert g2._nodes.shape == (3, 2)
        assert (g2._nodes[g2._node].sort_values().to_list() ==  # noqa: W504
            sorted(['e', 'l', 'g']))
        assert g2._edges.shape == (2, 3)

    def test_hop_pre_post_match_1(self):
        g = hops_graph()
        g2 = g.hop(source_node_match={'node': 'e'}, destination_node_match={'node': 'l'})
        assert g2._nodes.shape == (2, 2)
        assert (g2._nodes[g2._node].sort_values().to_list() ==  # noqa: W504
            sorted(['e', 'l']))
        assert g2._edges.shape == (1, 3)

    def test_hop_filter_types(self):

        e_df = pd.DataFrame({
            's': ['a', 'a', 'd', 'd', 'f', 'f'],
            'd': ['b', 'b', 'e', 'e', 'g', 'g'],
            't': ['x', 'h', 'x', 'h', 'x', 'h']
        })
        n_df = pd.DataFrame({
            'n': ['a', 'b', 'd', 'e', 'f', 'g'],
            't': ['x', 'm', 'x', 'n', 'x', 'o']
        })
        g = CGFull().edges(e_df, 's', 'd').nodes(n_df, 'n')

        g2a = g.hop(source_node_match={'n': 'a'})
        assert g2a._nodes.shape == (2, 2)
        assert g2a._edges.shape == (2, 3)

        g2b = g.hop(source_node_match={'t': 'm'}, direction='forward')
        assert g2b._nodes.shape == (0, 2)
        assert g2b._edges.shape == (0, 3)

        g3a = g.hop(edge_match={'t': 'h', 's': 'a'})
        assert g3a._nodes.shape == (2, 2)
        assert g3a._edges.shape == (1, 3)

        #TODO investigate
        #g4a = g.hop(destination_node_match={'t': 'n'}, direction='reverse')
        #assert g4a._nodes.shape == (2, 2)
        #assert g4a._edges.shape == (2, 3)

        g4a = g.hop(destination_node_match={'t': 'n'})
        assert g4a._nodes.shape == (2, 2)
        assert g4a._edges.shape == (2, 3)

        #TODO investigate setting to reverse
        g5a = g.hop(
            source_node_match={'t': 'x', 'n': 'a'},
            edge_match={'t': 'h'},
            destination_node_match={'t': 'm'})
        assert g5a._nodes.shape == (2, 2)
        assert g5a._edges.shape == (1, 3)

    def test_predicate_is_in(self):
        g = hops_graph()
        assert g.hop(source_node_match={'node': is_in(['e', 'k'])})._edges.shape == (3, 3)

    def test_hop_min_max_range(self):
        g = simple_chain_graph()
        seeds = pd.DataFrame({g._node: ['a']})
        g2 = g.hop(seeds, min_hops=2, max_hops=3)
        assert set(g2._nodes[g2._node].to_list()) == {'c', 'd'}
        assert set(zip(g2._edges['s'], g2._edges['d'])) == {('b', 'c'), ('c', 'd')}

    def test_hop_labels_nodes_edges(self):
        g = simple_chain_graph()
        seeds = pd.DataFrame({g._node: ['a']})
        g2 = g.hop(seeds, min_hops=1, max_hops=3, label_nodes='hop', label_edges='edge_hop', label_seed=True)
        node_hops = dict(zip(g2._nodes[g._node], g2._nodes['hop']))
        assert node_hops == {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        edge_hops = {(row['s'], row['d'], row['edge_hop']) for _, row in g2._edges.iterrows()}
        assert edge_hops == {('a', 'b', 1), ('b', 'c', 2), ('c', 'd', 3)}

    def test_hop_output_slice(self):
        g = simple_chain_graph()
        seeds = pd.DataFrame({g._node: ['a']})
        g2 = g.hop(
            seeds,
            min_hops=1,
            max_hops=3,
            output_min=2,
            output_max=2,
            label_nodes='hop',
            label_edges='edge_hop'
        )
        assert set(g2._nodes[g._node].to_list()) == {'c'}
        assert set(zip(g2._edges['s'], g2._edges['d'])) == {('b', 'c')}
        assert set(g2._edges['edge_hop'].to_list()) == {2}

class TestComputeHopMixinQuery(NoAuthTestCase):

    def test_hop_source_query(self):
        g = hops_graph()
        g2 = g.hop(source_node_query='node == "d"', direction='forward', hops=1)
        assert g2._nodes.shape == (6, 2)
        assert g2._edges.shape == (5, 3)

    def test_hop_destination_query(self):
        g = hops_graph()
        g2 = g.hop(destination_node_query='node == "d"', direction='reverse', hops=1)
        assert g2._nodes.shape == (6, 2)
        assert g2._edges.shape == (5, 3)

    def test_hop_edge_query(self):
        g = hops_graph()
        g2 = g.hop(edge_query='s == "d"', direction='forward', hops=1)
        assert g2._nodes.shape == (6, 2)
        assert g2._edges.shape == (5, 3)
