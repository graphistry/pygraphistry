import pandas as pd
from common import NoAuthTestCase

from graphistry.tests.test_compute import hops_graph

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
