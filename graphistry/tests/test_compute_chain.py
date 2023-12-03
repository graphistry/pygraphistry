from functools import lru_cache
from typing import Dict, List
import pandas as pd
from common import NoAuthTestCase
from graphistry.tests.test_compute import CGFull

from graphistry.tests.test_compute_hops import hops_graph
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected, is_in

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@lru_cache(maxsize=1)
def chain_graph():
    return CGFull().edges(
        pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd']
        }),
        's', 'd'
    ).nodes(
        pd.DataFrame({'n': ['a', 'b', 'c', 'd']}),
        'n'
    )

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

    def test_chain_is_in(self):
        g = hops_graph()
        assert g.chain([n({'node': is_in(['e', 'k'])})])._nodes.shape == (2, 2)

    def test_post_hop_node_match(self):

        ns = pd.DataFrame({
            'n': [1, 5],
            'category': ['Port', 'Other'],
        })

        es = pd.DataFrame({
            's': [1, 1],
            'd': [1, 5]
        })

        g = CGFull().edges(es, 's', 'd').nodes(ns, 'n')

        g2 = g.chain([
            n({'category': 'Port'}),
            e_undirected(),
            n({'category': 'Port'})
        ])
        assert len(g2._nodes) == 1


def compare_graphs(g, nodes: List[Dict[str, str]], edges: List[Dict[str, str]]) -> None:
    assert g._nodes.sort_values(by='n').to_dict(orient='records') == nodes
    assert g._edges.sort_values(by=['s', 'd']).to_dict(orient='records') == edges


class TestComputeChainWavefront1Mixin(NoAuthTestCase):
    """
    Test individual steps for 0-hop and 1-hop
    """

    def test_hop_chain_0(self):

        g = chain_graph()

        g2 = g.chain([
            n({'n': 'a'})
        ])

        assert g2._nodes.to_dict(orient='records') == [{'n': 'a'}]
        assert g2._edges.to_dict(orient='records') == []

        g3 = g.chain([
            n({'n': 'd'})
        ])

        assert g3._nodes.to_dict(orient='records') == [{'n': 'd'}]
        assert g3._edges.to_dict(orient='records') == []

    def test_hop_chain_1_forward(self):

        g = chain_graph()

        g_out_nodes_hop = [{'n': 'b'}]
        g_out_nodes = [{'n': 'a'}, {'n': 'b'}]
        g_out_edges = [{'s': 'a', 'd': 'b'}]

        g2_forward = g.hop(
            nodes = pd.DataFrame({'n': ['a']}),
            hops = 1,
            to_fixed_point = False,
            direction = 'forward',
            source_node_match = None,
            edge_match = None,
            destination_node_match = None,
            return_as_wave_front = True
        )
        compare_graphs(g2_forward, g_out_nodes_hop, g_out_edges)

        g2_forward_triple = g.chain([
            e_forward({}, source_node_match={'n': 'a'}, hops=1)
        ])
        compare_graphs(g2_forward_triple, g_out_nodes, g_out_edges)

        g2_forward_chain = g.chain([
            n({'n': 'a'}),
            e_forward({}, hops=1)
        ])
        compare_graphs(g2_forward_chain, g_out_nodes, g_out_edges)

        g2_forward_chain_closed = g.chain([
            n({'n': 'a'}),
            e_forward({}, hops=1),
            n({})
        ])
        compare_graphs(g2_forward_chain_closed, g_out_nodes, g_out_edges)
    
    def test_hop_chain_1_reverse(self):

        g = chain_graph()

        g_out_nodes_hop = []
        g_out_nodes = []
        g_out_edges = []

        g2_reverse = g.hop(
            nodes = pd.DataFrame({'n': ['a']}),
            hops = 1,
            to_fixed_point = False,
            direction = 'reverse',
            source_node_match = None,
            edge_match = None,
            destination_node_match = None,
            return_as_wave_front = True
        )
        compare_graphs(g2_reverse, g_out_nodes_hop, g_out_edges)

        g2_reverse_triple = g.chain([
            e_reverse({}, source_node_match={'n': 'a'}, hops=1)
        ])
        compare_graphs(g2_reverse_triple, g_out_nodes, g_out_edges)

        g2_reverse_chain = g.chain([
            n({'n': 'a'}),
            e_reverse({}, hops=1)
        ])
        compare_graphs(g2_reverse_chain, g_out_nodes, g_out_edges)

        g2_reverse_chain_closed = g.chain([
            n({'n': 'a'}),
            e_reverse({}, hops=1),
            n({})
        ])
        compare_graphs(g2_reverse_chain_closed, g_out_nodes, g_out_edges)

    def test_hop_chain_1_undirected(self):

        g = chain_graph()

        g_out_nodes_hop = [{'n': 'b'}]
        g_out_nodes = [{'n': 'a'}, {'n': 'b'}]
        g_out_edges = [{'s': 'a', 'd': 'b'}]

        g2_undirected = g.hop(
            nodes = pd.DataFrame({'n': ['a']}),
            hops = 1,
            to_fixed_point = False,
            direction = 'undirected',
            source_node_match = None,
            edge_match = None,
            destination_node_match = None,
            return_as_wave_front = True
        )
        compare_graphs(g2_undirected, g_out_nodes_hop, g_out_edges)

        g2_undirected_triple = g.chain([
            e_undirected({}, source_node_match={'n': 'a'}, hops=1)
        ])
        compare_graphs(g2_undirected_triple, g_out_nodes, g_out_edges)

        g2_undirected_chain = g.chain([
            n({'n': 'a'}),
            e_undirected({}, hops=1)
        ])
        compare_graphs(g2_undirected_chain, g_out_nodes, g_out_edges)

        g2_undirected_chain_closed = g.chain([
            n({'n': 'a'}),
            e_undirected({}, hops=1),
            n({})
        ])
        compare_graphs(g2_undirected_chain_closed, g_out_nodes, g_out_edges)

    def test_hop_chain_1_end_forward(self):

        g = chain_graph()

        g_out_nodes_hop = []
        g_out_nodes = []
        g_out_edges = []

        g3_forward = g.hop(
            nodes = pd.DataFrame({'n': ['d']}),
            hops = 2,
            to_fixed_point = False,
            direction = 'forward',
            source_node_match = None,
            edge_match = None,
            destination_node_match = None,
            return_as_wave_front = True
        )
        compare_graphs(g3_forward, g_out_nodes_hop, g_out_edges)

        g3_forward_triple = g.chain([
            e_forward({}, source_node_match={'n': 'd'}, hops=1)
        ])
        compare_graphs(g3_forward_triple, g_out_nodes, g_out_edges)

        g3_forward_chain = g.chain([
            n({'n': 'd'}),
            e_forward({}, hops=1)
        ])
        compare_graphs(g3_forward_chain, g_out_nodes, g_out_edges)

        g3_forward_chain_closed = g.chain([
            n({'n': 'd'}),
            e_forward({}, hops=1),
            n({})
        ])
        compare_graphs(g3_forward_chain_closed, g_out_nodes, g_out_edges)

    def test_hop_chain_1_end_reverse(self):

        g = chain_graph()

        g_out_nodes_hop = [{'n': 'c'}]
        g_out_nodes = [{'n': 'c'}, {'n': 'd'}]
        g_out_edges = [{'s': 'c', 'd': 'd'}]

        g3_reverse = g.hop(
            nodes = pd.DataFrame({'n': ['d']}),
            hops = 1,
            to_fixed_point = False,
            direction = 'reverse',
            source_node_match = None,
            edge_match = None,
            destination_node_match = None,
            return_as_wave_front = True
        )
        compare_graphs(g3_reverse, g_out_nodes_hop, g_out_edges)

        g3_reverse_triple = g.chain([
            e_reverse({}, source_node_match={'n': 'd'}, hops=1)
        ])
        compare_graphs(g3_reverse_triple, g_out_nodes, g_out_edges)

        g3_reverse_chain = g.chain([
            n({'n': 'd'}),
            e_reverse({}, hops=1)
        ])
        compare_graphs(g3_reverse_chain, g_out_nodes, g_out_edges)

        g3_reverse_chain_closed = g.chain([
            n({'n': 'd'}),
            e_reverse({}, hops=1),
            n({})
        ])
        compare_graphs(g3_reverse_chain_closed, g_out_nodes, g_out_edges)

    def test_hop_chain_1_end_undirected(self):

        g = chain_graph()

        g_out_nodes_hop = [{'n': 'c'}]
        g_out_nodes = [{'n': 'c'}, {'n': 'd'}]
        g_out_edges = [{'s': 'c', 'd': 'd'}]

        g3_undirected = g.hop(
            nodes = pd.DataFrame({'n': ['d']}),
            hops = 1,
            to_fixed_point = False,
            direction = 'undirected',
            source_node_match = None,
            edge_match = None,
            destination_node_match = None,
            return_as_wave_front = True
        )
        compare_graphs(g3_undirected, g_out_nodes_hop, g_out_edges)

        g3_undirected_triple = g.chain([
            e_undirected({}, source_node_match={'n': 'd'}, hops=1)
        ])
        compare_graphs(g3_undirected_triple, g_out_nodes, g_out_edges)

        g3_undirected_chain = g.chain([
            n({'n': 'd'}),
            e_undirected({}, hops=1)
        ])
        compare_graphs(g3_undirected_chain, g_out_nodes, g_out_edges)

        g3_undirected_chain_closed = g.chain([
            n({'n': 'd'}),
            e_undirected({}, hops=1),
            n({})
        ])
        compare_graphs(g3_undirected_chain_closed, g_out_nodes, g_out_edges)


