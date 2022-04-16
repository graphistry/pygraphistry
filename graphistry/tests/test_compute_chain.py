import pandas as pd
from common import NoAuthTestCase

from graphistry.tests.test_compute import hops_graph
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected


class TestComputeChainMixin(NoAuthTestCase):

    def test_chain_0(self):

        g = hops_graph()
        g2 = g.chain([])
        assert g2._nodes.shape == g._nodes.shape
        assert g2._edges.shape == g._edges.shape

    def test_chain_node_mt(self):
            
        g = hops_graph()
        g2 = g.chain([n()])
        assert g2._nodes.shape == g._nodes.shape
        assert g2._edges.shape == (0, 3)

    def test_chain_node_filter(self):
            
        g = hops_graph()
        g2 = g.chain([n({"node": "a", "type": "n"})])
        assert g2._nodes.shape == (1, 2)    
        assert g2._edges.shape == (0, 3)

    def test_chain_edge_filter_undirected_all(self):
            
        g = hops_graph()
        g2 = g.chain([e_undirected({})])
        assert g2._nodes.shape == g._nodes.shape
        assert g2._edges.shape == g._edges.shape

    def test_chain_edge_filter_forward_all(self):
            
        g = hops_graph()
        g2 = g.chain([e_forward({})])
        assert g2._nodes.shape == g._nodes.shape
        assert g2._edges.shape == g._edges.shape

    def test_chain_edge_filter_forward_some(self):
            
        g = hops_graph()
        g2 = g.chain([e_forward({g._source: "j"})])
        assert g2._nodes.shape == (3, 2)    
        assert g2._edges.shape == (2, 3)

    def test_chain_edge_filter_reverse_all(self):
            
        g = hops_graph()
        g2 = g.chain([e_reverse({})])
        assert g2._nodes.shape == g._nodes.shape    
        assert g2._edges.shape == g._edges.shape

    def test_chain_edge_filter_reverse_some(self):
            
        g = hops_graph()
        g2 = g.chain([e_reverse({g._destination: "b"})])
        assert g2._nodes.shape == (4, 2)
        assert g2._edges.shape == (3, 3)
    
    def test_chain_multi(self):

        g = hops_graph()

        g2a = g.chain([
            n({g._node: "e"}),
            e_forward({}, hops=2),
        ])
        assert g2a._nodes.shape == (5, 2)  # e, l, g, b, a
        assert g2a._edges.shape == (4, 3)


        g2b = g.chain([
            n({g._node: "e"}),
            e_forward({}, hops=1),
            e_forward({}, hops=1)
        ])

        assert g2b._nodes.equals(g2a._nodes)
        assert g2b._edges.equals(g2b._edges)

    def test_chain_named(self):

        g = hops_graph()

        # e->l->b, e->g->a
        g2 = g.chain([
            n({g._node: "e"}, name="n1"),
            e_forward({}, hops=1),
            e_forward({}, hops=1, name="e2"),
            n(name="n2"),
        ])

        assert g2._nodes[ g2._nodes.n1 ][g2._node].to_list() == ["e"]
        assert sorted(g2._edges[ g2._edges.e2 ][g2._source].to_list()) == ["g", "l"]
        assert sorted(g2._edges[ g2._edges.e2 ][g2._destination].to_list()) == ["a", "b"]
        assert sorted(g2._nodes[ g2._nodes.n2 ][g2._node].to_list()) == ["a", "b"]
