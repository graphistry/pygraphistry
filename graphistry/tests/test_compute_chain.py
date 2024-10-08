from functools import lru_cache
from typing import Dict, List
import logging
import pandas as pd

from common import NoAuthTestCase
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected, is_in
from graphistry.tests.test_compute import CGFull
from graphistry.tests.test_compute_hops import hops_graph
from graphistry.util import setup_logger

logger = setup_logger(__name__)


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

@lru_cache(maxsize=1)
def chain_graph_rich():
    return CGFull().edges(
        pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd'],
            'u': [0, 1, 2]
        }),
        's', 'd'
    ).nodes(
        pd.DataFrame({
            'n': ['a', 'b', 'c', 'd'],
            'v': [0, 1, 2, 3]
        }),
        'n'
    )

def compare_graphs(g, nodes: List[Dict[str, str]], edges: List[Dict[str, str]]) -> None:
    assert g._nodes.sort_values(by='n').to_dict(orient='records') == nodes
    assert g._edges.sort_values(by=['s', 'd']).to_dict(orient='records') == edges

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

        assert g2b._nodes.sort_values(by=['node']).reset_index(drop=True).equals(g2a._nodes.sort_values(by=['node']).reset_index(drop=True))
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

    def test_chain_predicate_is_in(self):
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

    def test_shortest_path(self):

        g = chain_graph_rich()

        g_out_nodes_1_hop = [{'n': 'a', 'v': 0}, {'n': 'b', 'v': 1}]
        g_out_edges_1_hop = [{'s': 'a', 'd': 'b', 'u': 0}]

        g_out_nodes_2_hops = [{'n': 'a', 'v': 0}, {'n': 'b', 'v': 1}, {'n': 'c', 'v': 2}]
        g_out_edges_2_hops = [{'s': 'a', 'd': 'b', 'u': 0}, {'s': 'b', 'd': 'c', 'u': 1}]

        g2a = g.chain([n({'n': 'a'}), e_forward(hops=1), n()])
        assert g2a._nodes.shape == (2, 2)
        assert g2a._edges.shape == (1, 3)
        compare_graphs(g2a, g_out_nodes_1_hop, g_out_edges_1_hop)

        g2b = g.chain([n({'n': 'a'}), e_forward(hops=2), n()])
        assert g2b._nodes.shape == (3, 2)
        assert g2b._edges.shape == (2, 3)
        compare_graphs(g2b, g_out_nodes_2_hops, g_out_edges_2_hops)

        g3a = g.chain([n({'n': 'a'}), e_forward(hops=1), n({'n': 'b'})])
        assert g3a._nodes.shape == (2, 2)
        assert g3a._edges.shape == (1, 3)
        compare_graphs(g3a, g_out_nodes_1_hop, g_out_edges_1_hop)

        #a->b
        g3ba = g.chain([n({'n': 'a'}), e_forward(hops=1), n({'n': 'b'})])
        assert g3ba._nodes.shape == (2, 2)
        assert g3ba._edges.shape == (1, 3)

        #a->b-c
        g3baa = g.chain([n({'n': 'a'}), e_forward(hops=2)])
        assert g3baa._nodes.shape == (3, 2)
        assert g3baa._edges.shape == (2, 3)

        g3b = g.chain([n({'n': 'a'}), e_forward(hops=2), n({'n': 'c'})])
        assert g3b._nodes.shape == (3, 2), "nodes"
        assert g3b._edges.shape == (2, 3), "edges"
        compare_graphs(g3b, g_out_nodes_2_hops, g_out_edges_2_hops)

        g3c = g.chain([n({'n': 'a'}), e_undirected(hops=2), n({'n': 'c'})])
        assert g3c._nodes.shape == (3, 2)
        assert g3c._edges.shape == (2, 3)
        compare_graphs(g3c, g_out_nodes_2_hops, g_out_edges_2_hops)

        g3d = g.chain([n({'n': 'a'}), e_forward(to_fixed_point=True), n({'n': 'c'})])
        assert g3d._nodes.shape == (3, 2)
        assert g3d._edges.shape == (2, 3)
        compare_graphs(g3d, g_out_nodes_2_hops, g_out_edges_2_hops)

    def test_shortest_path_chained(self):

        g = chain_graph_rich()

        g_out_nodes_2_hops = [{'n': 'a', 'v': 0}, {'n': 'b', 'v': 1}, {'n': 'c', 'v': 2}]
        g_out_edges_2_hops = [{'s': 'a', 'd': 'b', 'u': 0}, {'s': 'b', 'd': 'c', 'u': 1}]

        g_out_nodes_3_hops = [{'n': 'a', 'v': 0}, {'n': 'b', 'v': 1}, {'n': 'c', 'v': 2}, {'n': 'd', 'v': 3}]
        g_out_edges_3_hops = [{'s': 'a', 'd': 'b', 'u': 0}, {'s': 'b', 'd': 'c', 'u': 1}, {'s': 'c', 'd': 'd', 'u': 2}]

        g2a = g.chain([n({'n': 'a'}), e_forward(hops=1), n({'n': 'b'}), e_forward(hops=1), n()])
        assert g2a._nodes.shape == (3, 2)
        assert g2a._edges.shape == (2, 3)
        compare_graphs(g2a, g_out_nodes_2_hops, g_out_edges_2_hops)

        g2b = g.chain([n({'n': 'a'}), e_forward(to_fixed_point=True), n({'n': 'b'}), e_forward(hops=1), n()])
        assert g2b._nodes.shape == (3, 2)
        assert g2b._edges.shape == (2, 3)
        compare_graphs(g2b, g_out_nodes_2_hops, g_out_edges_2_hops)

        g2c = g.chain([n({'n': 'a'}), e_forward(to_fixed_point=True), n({'n': 'b'}), e_forward(hops=1), n({'n': 'c'})])
        assert g2c._nodes.shape == (3, 2)
        assert g2c._edges.shape == (2, 3)
        compare_graphs(g2c, g_out_nodes_2_hops, g_out_edges_2_hops)

        g2d = g.chain([n({'n': 'a'}), e_forward(to_fixed_point=True), n(), e_forward(hops=1), n({'n': 'c'})])
        assert g2d._nodes.shape == (3, 2)
        assert g2d._edges.shape == (2, 3)
        compare_graphs(g2c, g_out_nodes_2_hops, g_out_edges_2_hops)

        g3a = g.chain([n({'n': 'a'}), e_forward(hops=2), n({'n': 'c'}), e_forward(hops=1), n()])
        assert g3a._nodes.shape == (4, 2)
        assert g3a._edges.shape == (3, 3)
        compare_graphs(g3a, g_out_nodes_3_hops, g_out_edges_3_hops)

        g3b = g.chain([n({'n': 'a'}), e_forward(hops=2), n({'n': 'c'}), e_forward(hops=1), n({'n': 'd'})])
        assert g3b._nodes.shape == (4, 2)
        assert g3b._edges.shape == (3, 3)
        compare_graphs(g3b, g_out_nodes_3_hops, g_out_edges_3_hops)

        g3c = g.chain([n({'n': 'a'}), e_forward(to_fixed_point=True), n({'n': 'c'}), e_forward(hops=1), n({'n': 'd'})])
        assert g3c._nodes.shape == (4, 2)
        assert g3c._edges.shape == (3, 3)
        compare_graphs(g3c, g_out_nodes_3_hops, g_out_edges_3_hops)

        g3d = g.chain([n({'n': 'a'}), e_forward(to_fixed_point=True), n({'n': 'c'}), e_forward(to_fixed_point=True), n({'n': 'd'})])
        assert g3d._nodes.shape == (4, 2)
        assert g3d._edges.shape == (3, 3)
        compare_graphs(g3d, g_out_nodes_3_hops, g_out_edges_3_hops)


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

    def test_tricky_topology_1(self):

        nodes = pd.DataFrame({
            'n': ['a1', 'a2', 'b1', 'b2'],
            't': [0, 0, 1, 1]
        })

        edges = pd.DataFrame({
            's': ['a1', 'a1'  ],
            'd': ['a2', 'b1']
        })

        n_out = pd.DataFrame({
            'n': ['a1', 'a2'],
            't': [0, 0]
        })

        e_out = pd.DataFrame({
            's': ['a1'],
            'd': ['a2']
        })

        g = CGFull().edges(edges, 's', 'd').nodes(nodes, 'n')

        g2 = g.chain([
            n({'t': 0}),
            e_undirected(),
            n({'t': 0})
        ])

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('\nNODES\n')
            logger.debug(g2._nodes.to_dict(orient='records'))
            logger.debug('\nEDGES\n')
            logger.debug(g2._edges.to_dict(orient='records'))

        compare_graphs(g2, n_out.to_dict(orient='records'), e_out.to_dict(orient='records'))

class TestComputeChainWavefront2Mixin(NoAuthTestCase):
    """
    Test individual steps for 2-hop
    """

    def test_hop_chain_2(self):

        g = chain_graph()

        g_out_nodes_hop = [{'n': 'b'}, {'n': 'c'}]
        g_out_nodes = [{'n': 'a'}, {'n': 'b'}, {'n': 'c'}]
        g_out_edges = [{'s': 'a', 'd': 'b'}, {'s': 'b', 'd': 'c'}]

        g2_forward = g.hop(
            nodes = pd.DataFrame({'n': ['a']}),
            hops = 2,
            to_fixed_point = False,
            direction = 'forward',
            source_node_match = None,
            edge_match = None,
            destination_node_match = None,
            return_as_wave_front = True
        )
        compare_graphs(g2_forward, g_out_nodes_hop, g_out_edges)

        # source _node_match would require each hop to start with {'n': 'a'}
        #g2_forward_triple = g.chain([
        #    e_forward({}, source_node_match={'n': 'a'}, hops=2)
        #])
        #compare_graphs(g2_forward_triple, g_out_nodes, g_out_edges)

        g2_forward_chain = g.chain([
            n({'n': 'a'}),
            e_forward({}, hops=2)
        ])
        compare_graphs(g2_forward_chain, g_out_nodes, g_out_edges)

        g2_forward_chain_closed = g.chain([
            n({'n': 'a'}),
            e_forward({}, hops=2),
            n({})
        ])
        compare_graphs(g2_forward_chain_closed, g_out_nodes, g_out_edges)


    def test_hop_chain_2_reverse(self):

        g = chain_graph()

        g_out_nodes_hop = []
        g_out_nodes = []
        g_out_edges = []

        g2_reverse = g.hop(
            nodes = pd.DataFrame({'n': ['a']}),
            hops = 2,
            to_fixed_point = False,
            direction = 'reverse',
            source_node_match = None,
            edge_match = None,
            destination_node_match = None,
            return_as_wave_front = True
        )
        compare_graphs(g2_reverse, g_out_nodes_hop, g_out_edges)

        # source _node_match would require each hop to start with {'n': 'a'}
        #g2_reverse_triple = g.chain([
        #    e_reverse({}, source_node_match={'n': 'a'}, hops=2)
        #])
        #compare_graphs(g2_reverse_triple, g_out_nodes, g_out_edges)

        g2_reverse_chain = g.chain([
            n({'n': 'a'}),
            e_reverse({}, hops=2)
        ])
        compare_graphs(g2_reverse_chain, g_out_nodes, g_out_edges)

        g2_reverse_chain_closed = g.chain([
            n({'n': 'a'}),
            e_reverse({}, hops=2),
            n({})
        ])
        compare_graphs(g2_reverse_chain_closed, g_out_nodes, g_out_edges)

    def test_hop_chain_2_undirected(self):

        g = chain_graph()

        g_out_nodes_hop = [{'n': 'a'}, {'n': 'b'}, {'n': 'c'}]
        g_out_nodes = [{'n': 'a'}, {'n': 'b'}, {'n': 'c'}]
        g_out_edges = [{'s': 'a', 'd': 'b'}, {'s': 'b', 'd': 'c'}]

        g2_undirected = g.hop(
            nodes = pd.DataFrame({'n': ['a']}),
            hops = 2,
            to_fixed_point = False,
            direction = 'undirected',
            source_node_match = None,
            edge_match = None,
            destination_node_match = None,
            return_as_wave_front = True
        )
        compare_graphs(g2_undirected, g_out_nodes_hop, g_out_edges)

        # source _node_match would require each hop to start with {'n': 'a'}
        #g2_undirected_triple = g.chain([
        #    e_undirected({}, source_node_match={'n': 'a'}, hops=2)
        #])
        #compare_graphs(g2_undirected_triple, g_out_nodes, g_out_edges)

        g2_undirected_chain = g.chain([
            n({'n': 'a'}),
            e_undirected({}, hops=2)
        ])
        compare_graphs(g2_undirected_chain, g_out_nodes, g_out_edges)

        g2_undirected_chain_closed = g.chain([
            n({'n': 'a'}),
            e_undirected({}, hops=2),
            n({})
        ])
        compare_graphs(g2_undirected_chain_closed, g_out_nodes, g_out_edges)


    def test_hop_chain_2_end(self):

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

        # source _node_match would require each hop to start with {'n': 'd'}
        #g3_forward_triple = g.chain([
        #    e_forward({}, source_node_match={'n': 'd'}, hops=2)
        #])
        #compare_graphs(g3_forward_triple, g_out_nodes, g_out_edges)

        g3_forward_chain = g.chain([
            n({'n': 'd'}),
            e_forward({}, hops=2)
        ])
        compare_graphs(g3_forward_chain, g_out_nodes, g_out_edges)

        g3_forward_chain_closed = g.chain([
            n({'n': 'd'}),
            e_forward({}, hops=2),
            n({})
        ])
        compare_graphs(g3_forward_chain_closed, g_out_nodes, g_out_edges)


    def test_hop_chain_2_end_reverse(self):

        g = chain_graph()

        g_out_nodes_hop = [{'n': 'b'}, {'n': 'c'}]
        g_out_nodes = [{'n': 'b'}, {'n': 'c'}, {'n': 'd'}]
        g_out_edges = [{'s': 'b', 'd': 'c'}, {'s': 'c', 'd': 'd'}]

        g3_reverse = g.hop(
            nodes = pd.DataFrame({'n': ['d']}),
            hops = 2,
            to_fixed_point = False,
            direction = 'reverse',
            source_node_match = None,
            edge_match = None,
            destination_node_match = None,
            return_as_wave_front = True
        )
        compare_graphs(g3_reverse, g_out_nodes_hop, g_out_edges)

        # source _node_match would require each hop to start with {'n': 'd'}
        # g3_reverse_triple = g.chain([
        #    e_reverse({}, source_node_match={'n': 'd'}, hops=2)
        #])
        #compare_graphs(g3_reverse_triple, g_out_nodes, g_out_edges)

        g3_reverse_chain = g.chain([
            n({'n': 'd'}),
            e_reverse({}, hops=2)
        ])
        compare_graphs(g3_reverse_chain, g_out_nodes, g_out_edges)

        g3_reverse_chain_closed = g.chain([
            n({'n': 'd'}),
            e_reverse({}, hops=2),
            n({})
        ])
        compare_graphs(g3_reverse_chain_closed, g_out_nodes, g_out_edges)

    
    def test_hop_chain_2_end_undirected(self):

        g = chain_graph()

        g_out_nodes_hop = [{'n': 'b'}, {'n': 'c'}, {'n': 'd'}]
        g_out_nodes = [{'n': 'b'}, {'n': 'c'}, {'n': 'd'}]
        g_out_edges = [{'s': 'b', 'd': 'c'}, {'s': 'c', 'd': 'd'}]

        g3_undirected = g.hop(
            nodes = pd.DataFrame({'n': ['d']}),
            hops = 2,
            to_fixed_point = False,
            direction = 'undirected',
            source_node_match = None,
            edge_match = None,
            destination_node_match = None,
            return_as_wave_front = True
        )
        compare_graphs(g3_undirected, g_out_nodes_hop, g_out_edges)

        # source _node_match would require each hop to start with {'n': 'd'}
        #g3_undirected_triple = g.chain([
        #    e_undirected({}, source_node_match={'n': 'd'}, hops=2)
        #])
        #compare_graphs(g3_undirected_triple, g_out_nodes, g_out_edges)

        g3_undirected_chain = g.chain([
            n({'n': 'd'}),
            e_undirected({}, hops=2)
        ])
        compare_graphs(g3_undirected_chain, g_out_nodes, g_out_edges)

        g3_undirected_chain_closed = g.chain([
            n({'n': 'd'}),
            e_undirected({}, hops=2),
            n({})
        ])
        compare_graphs(g3_undirected_chain_closed, g_out_nodes, g_out_edges)

class TestComputeChainQuery(NoAuthTestCase):

    def test_node_query(self):

        g = chain_graph()

        g2 = g.chain([
            n(query='n == "a"')
        ])

        assert g2._nodes.to_dict(orient='records') == [{'n': 'a'}]
        assert g2._edges.to_dict(orient='records') == []

    def test_edge_query(self):

        g = chain_graph()

        g2 = g.chain([
            e_forward(edge_query='s == "a"')
        ])

        assert g2._nodes.to_dict(orient='records') == [{'n': 'a'}, {'n': 'b'}]
        assert g2._edges.to_dict(orient='records') == [{'s': 'a', 'd': 'b'}]

    def test_edge_source_query(self):

        g = chain_graph()

        g2 = g.chain([
            e_forward(source_node_query='n == "a"')
        ])
        assert g2._nodes.to_dict(orient='records') == [{'n': 'a'}, {'n': 'b'}]
        assert g2._edges.to_dict(orient='records') == [{'s': 'a', 'd': 'b'}]

    def test_edge_destination_query(self):

        g = chain_graph()

        g2 = g.chain([
            e_forward(destination_node_query='n == "b"')
        ])
        assert g2._nodes.to_dict(orient='records') == [{'n': 'a'}, {'n': 'b'}]
        assert g2._edges.to_dict(orient='records') == [{'s': 'a', 'd': 'b'}]
