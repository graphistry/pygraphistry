from functools import lru_cache
from typing import Dict, List
import logging
import pandas as pd

from common import NoAuthTestCase
import pytest
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected, is_in, rows, select
from graphistry.compute.chain import _inject_binding_ops_if_needed
from graphistry.compute.exceptions import GFQLValidationError
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
        g2 = g.gfql([])
        assert g2._nodes.shape == g._nodes.shape
        assert g2._edges.shape == g._edges.shape

    def test_chain_node_mt(self):
            
        g = hops_graph()
        g2 = g.gfql([n()])
        assert g2._nodes.shape == g._nodes.shape
        assert g2._edges.shape == (0, 3)

    def test_chain_node_filter(self):
            
        g = hops_graph()
        g2 = g.gfql([n({"node": "a", "type": "n"})])
        assert g2._nodes.shape == (1, 2)    
        assert g2._edges.shape == (0, 3)

    def test_chain_edge_filter_undirected_all(self):
            
        g = hops_graph()
        g2 = g.gfql([e_undirected({})])
        assert g2._nodes.shape == g._nodes.shape
        assert g2._edges.shape == g._edges.shape

    def test_chain_edge_filter_forward_all(self):
            
        g = hops_graph()
        g2 = g.gfql([e_forward({})])
        assert g2._nodes.shape == g._nodes.shape
        assert g2._edges.shape == g._edges.shape

    def test_chain_edge_filter_forward_some(self):
            
        g = hops_graph()
        g2 = g.gfql([e_forward({g._source: "j"})])
        assert g2._nodes.shape == (3, 2)    
        assert g2._edges.shape == (2, 3)

    def test_chain_edge_filter_reverse_all(self):
            
        g = hops_graph()
        g2 = g.gfql([e_reverse({})])
        assert g2._nodes.shape == g._nodes.shape    
        assert g2._edges.shape == g._edges.shape

    def test_chain_edge_filter_reverse_some(self):
            
        g = hops_graph()
        g2 = g.gfql([e_reverse({g._destination: "b"})])
        assert g2._nodes.shape == (4, 2)
        assert g2._edges.shape == (3, 3)
    
    def test_chain_multi(self):

        g = hops_graph()

        g2a = g.gfql([
            n({g._node: "e"}),
            e_forward({}, hops=2),
        ])
        assert g2a._nodes.shape == (5, 2)  # e, l, g, b, a
        assert g2a._edges.shape == (4, 3)


        g2b = g.gfql([
            n({g._node: "e"}),
            e_forward({}, hops=1),
            e_forward({}, hops=1)
        ])

        assert g2b._nodes.sort_values(by=['node']).reset_index(drop=True).equals(g2a._nodes.sort_values(by=['node']).reset_index(drop=True))
        assert g2b._edges.equals(g2b._edges)

    def test_chain_named(self):

        g = hops_graph()

        # e->l->b, e->g->a
        g2 = g.gfql([
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
        assert g.gfql([n({'node': is_in(['e', 'k'])})])._nodes.shape == (2, 2)

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

        g2 = g.gfql([
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

        g2a = g.gfql([n({'n': 'a'}), e_forward(hops=1), n()])
        assert g2a._nodes.shape == (2, 2)
        assert g2a._edges.shape == (1, 3)
        compare_graphs(g2a, g_out_nodes_1_hop, g_out_edges_1_hop)

        g2b = g.gfql([n({'n': 'a'}), e_forward(hops=2), n()])
        assert g2b._nodes.shape == (3, 2)
        assert g2b._edges.shape == (2, 3)
        compare_graphs(g2b, g_out_nodes_2_hops, g_out_edges_2_hops)

        g3a = g.gfql([n({'n': 'a'}), e_forward(hops=1), n({'n': 'b'})])
        assert g3a._nodes.shape == (2, 2)
        assert g3a._edges.shape == (1, 3)
        compare_graphs(g3a, g_out_nodes_1_hop, g_out_edges_1_hop)

        #a->b
        g3ba = g.gfql([n({'n': 'a'}), e_forward(hops=1), n({'n': 'b'})])
        assert g3ba._nodes.shape == (2, 2)
        assert g3ba._edges.shape == (1, 3)

        #a->b-c
        g3baa = g.gfql([n({'n': 'a'}), e_forward(hops=2)])
        assert g3baa._nodes.shape == (3, 2)
        assert g3baa._edges.shape == (2, 3)

        g3b = g.gfql([n({'n': 'a'}), e_forward(hops=2), n({'n': 'c'})])
        assert g3b._nodes.shape == (3, 2), "nodes"
        assert g3b._edges.shape == (2, 3), "edges"
        compare_graphs(g3b, g_out_nodes_2_hops, g_out_edges_2_hops)

        g3c = g.gfql([n({'n': 'a'}), e_undirected(hops=2), n({'n': 'c'})])
        assert g3c._nodes.shape == (3, 2)
        assert g3c._edges.shape == (2, 3)
        compare_graphs(g3c, g_out_nodes_2_hops, g_out_edges_2_hops)

        g3d = g.gfql([n({'n': 'a'}), e_forward(to_fixed_point=True), n({'n': 'c'})])
        assert g3d._nodes.shape == (3, 2)
        assert g3d._edges.shape == (2, 3)
        compare_graphs(g3d, g_out_nodes_2_hops, g_out_edges_2_hops)

    def test_shortest_path_chained(self):

        g = chain_graph_rich()

        g_out_nodes_2_hops = [{'n': 'a', 'v': 0}, {'n': 'b', 'v': 1}, {'n': 'c', 'v': 2}]
        g_out_edges_2_hops = [{'s': 'a', 'd': 'b', 'u': 0}, {'s': 'b', 'd': 'c', 'u': 1}]

        g_out_nodes_3_hops = [{'n': 'a', 'v': 0}, {'n': 'b', 'v': 1}, {'n': 'c', 'v': 2}, {'n': 'd', 'v': 3}]
        g_out_edges_3_hops = [{'s': 'a', 'd': 'b', 'u': 0}, {'s': 'b', 'd': 'c', 'u': 1}, {'s': 'c', 'd': 'd', 'u': 2}]

        g2a = g.gfql([n({'n': 'a'}), e_forward(hops=1), n({'n': 'b'}), e_forward(hops=1), n()])
        assert g2a._nodes.shape == (3, 2)
        assert g2a._edges.shape == (2, 3)
        compare_graphs(g2a, g_out_nodes_2_hops, g_out_edges_2_hops)

        g2b = g.gfql([n({'n': 'a'}), e_forward(to_fixed_point=True), n({'n': 'b'}), e_forward(hops=1), n()])
        assert g2b._nodes.shape == (3, 2)
        assert g2b._edges.shape == (2, 3)
        compare_graphs(g2b, g_out_nodes_2_hops, g_out_edges_2_hops)

        g2c = g.gfql([n({'n': 'a'}), e_forward(to_fixed_point=True), n({'n': 'b'}), e_forward(hops=1), n({'n': 'c'})])
        assert g2c._nodes.shape == (3, 2)
        assert g2c._edges.shape == (2, 3)
        compare_graphs(g2c, g_out_nodes_2_hops, g_out_edges_2_hops)

        g2d = g.gfql([n({'n': 'a'}), e_forward(to_fixed_point=True), n(), e_forward(hops=1), n({'n': 'c'})])
        assert g2d._nodes.shape == (3, 2)
        assert g2d._edges.shape == (2, 3)
        compare_graphs(g2c, g_out_nodes_2_hops, g_out_edges_2_hops)

        g3a = g.gfql([n({'n': 'a'}), e_forward(hops=2), n({'n': 'c'}), e_forward(hops=1), n()])
        assert g3a._nodes.shape == (4, 2)
        assert g3a._edges.shape == (3, 3)
        compare_graphs(g3a, g_out_nodes_3_hops, g_out_edges_3_hops)

        g3b = g.gfql([n({'n': 'a'}), e_forward(hops=2), n({'n': 'c'}), e_forward(hops=1), n({'n': 'd'})])
        assert g3b._nodes.shape == (4, 2)
        assert g3b._edges.shape == (3, 3)
        compare_graphs(g3b, g_out_nodes_3_hops, g_out_edges_3_hops)

        g3c = g.gfql([n({'n': 'a'}), e_forward(to_fixed_point=True), n({'n': 'c'}), e_forward(hops=1), n({'n': 'd'})])
        assert g3c._nodes.shape == (4, 2)
        assert g3c._edges.shape == (3, 3)
        compare_graphs(g3c, g_out_nodes_3_hops, g_out_edges_3_hops)

        g3d = g.gfql([n({'n': 'a'}), e_forward(to_fixed_point=True), n({'n': 'c'}), e_forward(to_fixed_point=True), n({'n': 'd'})])
        assert g3d._nodes.shape == (4, 2)
        assert g3d._edges.shape == (3, 3)
        compare_graphs(g3d, g_out_nodes_3_hops, g_out_edges_3_hops)


class TestComputeChainWavefront1Mixin(NoAuthTestCase):
    """
    Test individual steps for 0-hop and 1-hop
    """

    def test_hop_chain_0(self):

        g = chain_graph()

        g2 = g.gfql([
            n({'n': 'a'})
        ])

        assert g2._nodes.to_dict(orient='records') == [{'n': 'a'}]
        assert g2._edges.to_dict(orient='records') == []

        g3 = g.gfql([
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

        g2_forward_triple = g.gfql([
            e_forward({}, source_node_match={'n': 'a'}, hops=1)
        ])
        compare_graphs(g2_forward_triple, g_out_nodes, g_out_edges)

        g2_forward_chain = g.gfql([
            n({'n': 'a'}),
            e_forward({}, hops=1)
        ])
        compare_graphs(g2_forward_chain, g_out_nodes, g_out_edges)

        g2_forward_chain_closed = g.gfql([
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

        g2_reverse_triple = g.gfql([
            e_reverse({}, source_node_match={'n': 'a'}, hops=1)
        ])
        compare_graphs(g2_reverse_triple, g_out_nodes, g_out_edges)

        g2_reverse_chain = g.gfql([
            n({'n': 'a'}),
            e_reverse({}, hops=1)
        ])
        compare_graphs(g2_reverse_chain, g_out_nodes, g_out_edges)

        g2_reverse_chain_closed = g.gfql([
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

        g2_undirected_triple = g.gfql([
            e_undirected({}, source_node_match={'n': 'a'}, hops=1)
        ])
        compare_graphs(g2_undirected_triple, g_out_nodes, g_out_edges)

        g2_undirected_chain = g.gfql([
            n({'n': 'a'}),
            e_undirected({}, hops=1)
        ])
        compare_graphs(g2_undirected_chain, g_out_nodes, g_out_edges)

        g2_undirected_chain_closed = g.gfql([
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

        g3_forward_triple = g.gfql([
            e_forward({}, source_node_match={'n': 'd'}, hops=1)
        ])
        compare_graphs(g3_forward_triple, g_out_nodes, g_out_edges)

        g3_forward_chain = g.gfql([
            n({'n': 'd'}),
            e_forward({}, hops=1)
        ])
        compare_graphs(g3_forward_chain, g_out_nodes, g_out_edges)

        g3_forward_chain_closed = g.gfql([
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

        g3_reverse_triple = g.gfql([
            e_reverse({}, source_node_match={'n': 'd'}, hops=1)
        ])
        compare_graphs(g3_reverse_triple, g_out_nodes, g_out_edges)

        g3_reverse_chain = g.gfql([
            n({'n': 'd'}),
            e_reverse({}, hops=1)
        ])
        compare_graphs(g3_reverse_chain, g_out_nodes, g_out_edges)

        g3_reverse_chain_closed = g.gfql([
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

        g3_undirected_triple = g.gfql([
            e_undirected({}, source_node_match={'n': 'd'}, hops=1)
        ])
        compare_graphs(g3_undirected_triple, g_out_nodes, g_out_edges)

        g3_undirected_chain = g.gfql([
            n({'n': 'd'}),
            e_undirected({}, hops=1)
        ])
        compare_graphs(g3_undirected_chain, g_out_nodes, g_out_edges)

        g3_undirected_chain_closed = g.gfql([
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

        g2 = g.gfql([
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
        #g2_forward_triple = g.gfql([
        #    e_forward({}, source_node_match={'n': 'a'}, hops=2)
        #])
        #compare_graphs(g2_forward_triple, g_out_nodes, g_out_edges)

        g2_forward_chain = g.gfql([
            n({'n': 'a'}),
            e_forward({}, hops=2)
        ])
        compare_graphs(g2_forward_chain, g_out_nodes, g_out_edges)

        g2_forward_chain_closed = g.gfql([
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
        #g2_reverse_triple = g.gfql([
        #    e_reverse({}, source_node_match={'n': 'a'}, hops=2)
        #])
        #compare_graphs(g2_reverse_triple, g_out_nodes, g_out_edges)

        g2_reverse_chain = g.gfql([
            n({'n': 'a'}),
            e_reverse({}, hops=2)
        ])
        compare_graphs(g2_reverse_chain, g_out_nodes, g_out_edges)

        g2_reverse_chain_closed = g.gfql([
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
        #g2_undirected_triple = g.gfql([
        #    e_undirected({}, source_node_match={'n': 'a'}, hops=2)
        #])
        #compare_graphs(g2_undirected_triple, g_out_nodes, g_out_edges)

        g2_undirected_chain = g.gfql([
            n({'n': 'a'}),
            e_undirected({}, hops=2)
        ])
        compare_graphs(g2_undirected_chain, g_out_nodes, g_out_edges)

        g2_undirected_chain_closed = g.gfql([
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
        #g3_forward_triple = g.gfql([
        #    e_forward({}, source_node_match={'n': 'd'}, hops=2)
        #])
        #compare_graphs(g3_forward_triple, g_out_nodes, g_out_edges)

        g3_forward_chain = g.gfql([
            n({'n': 'd'}),
            e_forward({}, hops=2)
        ])
        compare_graphs(g3_forward_chain, g_out_nodes, g_out_edges)

        g3_forward_chain_closed = g.gfql([
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
        # g3_reverse_triple = g.gfql([
        #    e_reverse({}, source_node_match={'n': 'd'}, hops=2)
        #])
        #compare_graphs(g3_reverse_triple, g_out_nodes, g_out_edges)

        g3_reverse_chain = g.gfql([
            n({'n': 'd'}),
            e_reverse({}, hops=2)
        ])
        compare_graphs(g3_reverse_chain, g_out_nodes, g_out_edges)

        g3_reverse_chain_closed = g.gfql([
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
        #g3_undirected_triple = g.gfql([
        #    e_undirected({}, source_node_match={'n': 'd'}, hops=2)
        #])
        #compare_graphs(g3_undirected_triple, g_out_nodes, g_out_edges)

        g3_undirected_chain = g.gfql([
            n({'n': 'd'}),
            e_undirected({}, hops=2)
        ])
        compare_graphs(g3_undirected_chain, g_out_nodes, g_out_edges)

        g3_undirected_chain_closed = g.gfql([
            n({'n': 'd'}),
            e_undirected({}, hops=2),
            n({})
        ])
        compare_graphs(g3_undirected_chain_closed, g_out_nodes, g_out_edges)

class TestComputeChainQuery(NoAuthTestCase):

    def test_node_query(self):

        g = chain_graph()

        g2 = g.gfql([
            n(query='n == "a"')
        ])

        assert g2._nodes.to_dict(orient='records') == [{'n': 'a'}]
        assert g2._edges.to_dict(orient='records') == []

    def test_edge_query(self):

        g = chain_graph()

        g2 = g.gfql([
            e_forward(edge_query='s == "a"')
        ])

        assert g2._nodes.to_dict(orient='records') == [{'n': 'a'}, {'n': 'b'}]
        assert g2._edges.to_dict(orient='records') == [{'s': 'a', 'd': 'b'}]

    def test_edge_source_query(self):

        g = chain_graph()

        g2 = g.gfql([
            e_forward(source_node_query='n == "a"')
        ])
        assert g2._nodes.to_dict(orient='records') == [{'n': 'a'}, {'n': 'b'}]
        assert g2._edges.to_dict(orient='records') == [{'s': 'a', 'd': 'b'}]

    def test_edge_destination_query(self):

        g = chain_graph()

        g2 = g.gfql([
            e_forward(destination_node_query='n == "b"')
        ])
        assert g2._nodes.to_dict(orient='records') == [{'n': 'a'}, {'n': 'b'}]
        assert g2._edges.to_dict(orient='records') == [{'s': 'a', 'd': 'b'}]


class TestChainBindingsTable(NoAuthTestCase):
    """#880: native chain rows() should materialize multi-alias bindings table."""

    def _mk_graph(self, nodes_df, edges_df):
        return CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

    def _mk_cudf_graph(self, nodes_df, edges_df):
        cudf = pytest.importorskip("cudf")
        return CGFull().nodes(cudf.from_pandas(nodes_df), "id").edges(cudf.from_pandas(edges_df), "s", "d")

    def _to_binding_ops(self, match_ops):
        return [op.to_json(validate=False) for op in match_ops]

    def _rows_df(self, g, match_ops, items=None):
        steps = [*match_ops, rows()]
        if items is not None:
            steps.append(select(items))
        return g.gfql(steps)._nodes

    def _rows_records(self, g, match_ops, items=None, sort_by=None):
        df = self._rows_df(g, match_ops, items=items)
        if sort_by is not None:
            df = df.sort_values(sort_by)
        return df.to_dict(orient="records")

    def _binding_rows_records(self, g, binding_ops, items, sort_by=None):
        df = g.gfql([rows(binding_ops=binding_ops), select(items)])._nodes
        if sort_by is not None:
            df = df.sort_values(sort_by)
        return df.to_dict(orient="records")

    def _assert_rows_binding_parity(self, g, match_ops, items, expected, sort_by=None):
        assert self._rows_records(g, match_ops, items=items, sort_by=sort_by) == expected
        assert self._binding_rows_records(g, self._to_binding_ops(match_ops), items, sort_by=sort_by) == expected

    def _mk_forum_moderator_graph(self):
        return self._mk_graph(
            pd.DataFrame(
                [
                    {"id": "c1", "labels": ["Comment"], "label__Comment": True},
                    {"id": "m1", "labels": ["Message"], "label__Message": True},
                    {"id": "p1", "labels": ["Post"], "label__Post": True},
                    {"id": "f1", "labels": ["Forum"], "label__Forum": True, "title": "Forum"},
                    {
                        "id": "u1",
                        "labels": ["Person"],
                        "label__Person": True,
                        "firstName": "Mod",
                        "lastName": "Erator",
                    },
                ]
            ),
            pd.DataFrame(
                [
                    {"s": "c1", "d": "m1", "type": "REPLY_OF"},
                    {"s": "m1", "d": "p1", "type": "REPLY_OF"},
                    {"s": "f1", "d": "p1", "type": "CONTAINER_OF"},
                    {"s": "f1", "d": "u1", "type": "HAS_MODERATOR"},
                ]
            ),
        )

    def _mk_cartesian_node_graph(self):
        return self._mk_graph(
            pd.DataFrame(
                {
                    "id": ["a", "b"],
                    "num": [1, 2],
                    "ts": [10, 20],
                }
            ),
            pd.DataFrame({"s": [], "d": []}),
        )

    def _forum_moderator_match_ops(self, reply_edge):
        return [
            n({"id": "c1", "label__Comment": True}, name="message"),
            reply_edge,
            n({"label__Post": True}, name="post"),
            e_reverse({"type": "CONTAINER_OF"}),
            n({"label__Forum": True}, name="forum"),
            e_forward({"type": "HAS_MODERATOR"}),
            n({"label__Person": True}, name="moderator"),
        ]

    def _mk_reverse_range_continuation_graph(self):
        return self._mk_graph(
            pd.DataFrame(
                [
                    {"id": "x", "label__Extra": True},
                    {"id": "a", "label__Seed": True},
                    {"id": "b", "label__Mid": True},
                    {"id": "c", "label__Mid": True},
                ]
            ),
            pd.DataFrame(
                [
                    {"s": "a", "d": "b", "type": "R"},
                    {"s": "b", "d": "c", "type": "R"},
                    {"s": "x", "d": "a", "type": "S"},
                ]
            ),
        )

    def _reverse_range_continuation_match_ops(self, range_edge):
        return [
            n({"id": "c", "label__Mid": True}, name="tail"),
            range_edge,
            n({"label__Seed": True}, name="seed"),
            e_reverse({"type": "S"}),
            n({"label__Extra": True}, name="extra"),
        ]

    def test_native_chain_rows_bindings_basic(self):
        """Basic: n(a)->e->n(b) with rows() should produce alias-prefixed columns."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "label__X": [True, False], "label__Y": [False, True], "val": [1, 2]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
        )
        result = g.gfql([
            n({"label__X": True}, name="x"),
            e_forward({"type": "R"}),
            n({"label__Y": True}, name="y"),
            rows(),
        ])
        df = result._nodes
        assert len(df) == 1
        assert "x.id" in df.columns
        assert "y.id" in df.columns
        assert df["x.id"].iloc[0] == "a"
        assert df["y.id"].iloc[0] == "b"

    def test_native_chain_rows_bindings_with_select(self):
        """rows() + select() should allow alias-prefixed projection."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "label__X": [True, False], "label__Y": [False, True], "val": [1, 2]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
        )
        records = self._rows_records(
            g,
            [
                n({"label__X": True}, name="x"),
                e_forward({"type": "R"}),
                n({"label__Y": True}, name="y"),
            ],
            items=[("x_val", "x.val"), ("y_val", "y.val")],
        )
        assert len(records) == 1
        assert records[0]["x_val"] == 1
        assert records[0]["y_val"] == 2

    def test_native_chain_rows_bindings_star_graph(self):
        """Star graph: 1 hub -> 3 leaves produces 3 binding rows."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["h", "a", "b", "c"], "label__Hub": [True, False, False, False], "label__Leaf": [False, True, True, True]}),
            pd.DataFrame({"s": ["h", "h", "h"], "d": ["a", "b", "c"], "type": ["R", "R", "R"]}),
        )
        result = g.gfql([
            n({"label__Hub": True}, name="hub"),
            e_forward({"type": "R"}),
            n({"label__Leaf": True}, name="leaf"),
            rows(),
        ])
        df = result._nodes
        assert len(df) == 3
        assert sorted(df["leaf.id"].tolist()) == ["a", "b", "c"]
        assert all(df["hub.id"] == "h")

    def test_native_chain_rows_bindings_undirected(self):
        """#994 shape via native chain: undirected edge with incoming storage."""
        g = self._mk_graph(
            pd.DataFrame({"id": [1, 2], "label__P": [True, True], "name": ["Alice", "Bob"]}),
            pd.DataFrame({"s": [2], "d": [1], "type": ["KNOWS"]}),
        )
        result = g.gfql([
            n({"id": 1, "label__P": True}, name="seed"),
            e_undirected({"type": "KNOWS"}),
            n({"label__P": True}, name="friend"),
            rows(),
        ])
        df = result._nodes
        assert len(df) == 1
        assert df["friend.id"].iloc[0] == 2
        assert df["friend.name"].iloc[0] == "Bob"

    def test_native_chain_rows_select_undirected_edge_alias_projection(self):
        """#982: undirected traversal should project edge alias properties after rows()."""
        g = self._mk_graph(
            pd.DataFrame(
                {
                    "id": ["a", "b"],
                    "label__Person": [True, True],
                    "firstName": ["Alice", "Bob"],
                }
            ),
            pd.DataFrame({"s": ["b"], "d": ["a"], "type": ["KNOWS"], "creationDate": [123]}),
        )
        records = self._rows_records(
            g,
            [
                n({"id": "a", "label__Person": True}, name="n"),
                e_undirected({"type": "KNOWS"}, name="r"),
                n({"label__Person": True}, name="friend"),
            ],
            items=[
                ("personId", "friend.id"),
                ("firstName", "friend.firstName"),
                ("friendshipCreationDate", "r.creationDate"),
            ],
        )
        assert records == [
            {
                "personId": "b",
                "firstName": "Bob",
                "friendshipCreationDate": 123,
            }
        ]

    def test_native_chain_rows_bindings_edge_alias(self):
        """#982: edge alias properties should be accessible."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "label__X": [True, True]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"], "weight": [42]}),
        )
        result = g.gfql([
            n(name="x"),
            e_forward({"type": "R"}, name="r"),
            n(name="y"),
            rows(),
        ])
        df = result._nodes
        assert len(df) >= 1
        assert "r.weight" in df.columns or "r.type" in df.columns

    def test_native_chain_rows_bindings_unnamed_first_node(self):
        """First node unnamed, second named — bindings still produced for named alias."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "val": [1, 2]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
        )
        result = g.gfql([
            n({"id": "a"}),  # unnamed
            e_forward({"type": "R"}, name="r"),
            n(name="y"),
            rows(),
        ])
        df = result._nodes
        assert len(df) == 1
        assert "y.id" in df.columns
        assert df["y.id"].iloc[0] == "b"

    def test_native_chain_rows_without_names_returns_single_table(self):
        """rows() without named ops should return standard single-table view."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "val": [1, 2]}),
            pd.DataFrame({"s": ["a"], "d": ["b"]}),
        )
        result = g.gfql([
            n(),
            e_forward(),
            n(),
            rows(),
        ])
        df = result._nodes
        # Should be single-table (no alias-prefixed columns)
        assert "id" in df.columns
        assert not any("." in str(c) for c in df.columns)

    def test_native_chain_rows_with_source_not_overridden(self):
        """rows(source=...) should NOT be overridden by binding_ops injection."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "label__X": [True, False]}),
            pd.DataFrame({"s": ["a"], "d": ["b"]}),
        )
        result = g.gfql([
            n({"label__X": True}, name="x"),
            e_forward(),
            n(name="y"),
            rows(source="x"),
        ])
        df = result._nodes
        # Should be single-table filtered to source alias, not bindings
        assert "id" in df.columns

    def test_native_chain_rows_bindings_empty_match(self):
        """No matching edges → empty bindings table."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "label__X": [True, False], "label__Y": [False, True]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["NOPE"]}),
        )
        result = g.gfql([
            n({"label__X": True}, name="x"),
            e_forward({"type": "MISSING"}),
            n({"label__Y": True}, name="y"),
            rows(),
        ])
        df = result._nodes
        assert len(df) == 0

    def test_native_chain_rows_bindings_three_hops(self):
        """Three-hop chain: a->b->c->d produces binding rows."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b", "c", "d"], "val": [1, 2, 3, 4]}),
            pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"], "type": ["R", "R", "R"]}),
        )
        result = g.gfql([
            n({"id": "a"}, name="start"),
            e_forward({"type": "R"}),
            n(name="mid"),
            e_forward({"type": "R"}),
            n(name="end"),
            rows(),
        ])
        df = result._nodes
        # a->b->c and a->b->c->d? Only 2-hop paths: a->b->c
        # But this is 2 edges (start->mid->end), so:
        # start=a, mid=b, end=c (one path)
        assert len(df) >= 1
        assert df["start.id"].iloc[0] == "a"

    def test_native_chain_rejects_duplicate_alias_names(self):
        """Duplicate alias names in a chain should raise, not silently overwrite."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "val": [1, 2]}),
            pd.DataFrame({"s": ["a"], "d": ["b"]}),
        )
        with pytest.raises(GFQLValidationError) as exc_info:
            g.gfql([
                n(name="hit"),
                e_forward(),
                n(name="hit"),
            ])
        assert "Duplicate alias" in exc_info.value.message

    def test_native_chain_rejects_duplicate_edge_alias_names(self):
        """Duplicate edge alias names should also be rejected."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b", "c"]}),
            pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]}),
        )
        with pytest.raises(GFQLValidationError) as exc_info:
            g.gfql([
                n(name="x"),
                e_forward(name="r"),
                n(name="y"),
                e_forward(name="r"),
                n(name="z"),
            ])
        assert "Duplicate alias" in exc_info.value.message

    def test_native_chain_rows_bindings_four_hops(self):
        """Four-hop chain: a->b->c->d->e with all aliases."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b", "c", "d", "e"], "val": [1, 2, 3, 4, 5]}),
            pd.DataFrame({
                "s": ["a", "b", "c", "d"],
                "d": ["b", "c", "d", "e"],
                "type": ["R", "R", "R", "R"],
            }),
        )
        result = g.gfql([
            n({"id": "a"}, name="n1"),
            e_forward({"type": "R"}),
            n(name="n2"),
            e_forward({"type": "R"}),
            n(name="n3"),
            e_forward({"type": "R"}),
            n(name="n4"),
            rows(),
        ])
        df = result._nodes
        assert len(df) == 1
        assert df["n1.id"].iloc[0] == "a"
        assert df["n2.id"].iloc[0] == "b"
        assert df["n3.id"].iloc[0] == "c"
        assert df["n4.id"].iloc[0] == "d"

    def test_native_chain_rows_bindings_no_match_first_node(self):
        """First node filter matches nothing → empty bindings."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "label__X": [True, False]}),
            pd.DataFrame({"s": ["a"], "d": ["b"]}),
        )
        result = g.gfql([
            n({"label__X": False, "id": "NOPE"}, name="x"),
            e_forward(),
            n(name="y"),
            rows(),
        ])
        assert len(result._nodes) == 0

    def test_native_chain_rows_bindings_mid_chain_empty(self):
        """Second edge matches nothing → empty bindings."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b", "c"]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
        )
        result = g.gfql([
            n({"id": "a"}, name="x"),
            e_forward({"type": "R"}),
            n(name="y"),
            e_forward({"type": "MISSING"}),
            n(name="z"),
            rows(),
        ])
        assert len(result._nodes) == 0

    def test_native_chain_rows_bindings_multi_hop_edge_alias(self):
        """Multi-hop with edge aliases should produce edge properties in bindings (#880).

        Edge alias names must not collide with the graph's source/destination columns
        ('s'/'d' in this test fixture) — use 'r1'/'r2' instead.
        """
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b", "c"], "label__X": [True, True, True]}),
            pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["R", "S"], "weight": [10, 20]}),
        )
        result = g.gfql([
            n(name="x"),
            e_forward({"type": "R"}, name="r1"),
            n(name="y"),
            e_forward({"type": "S"}, name="r2"),
            n(name="z"),
            rows(),
        ])
        df = result._nodes
        assert len(df) == 1
        assert df["x.id"].iloc[0] == "a"
        assert df["y.id"].iloc[0] == "b"
        assert df["z.id"].iloc[0] == "c"
        assert df["r1.weight"].iloc[0] == 10
        assert df["r2.weight"].iloc[0] == 20

    def test_native_chain_rows_select_edge_alias_projection(self):
        """select() should project edge alias properties from bindings."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "label__X": [True, True]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"], "weight": [42]}),
        )
        records = self._rows_records(
            g,
            [n(name="x"), e_forward({"type": "R"}, name="r"), n(name="y")],
            items=[("w", "r.weight"), ("xid", "x.id"), ("yid", "y.id")],
        )
        assert len(records) == 1
        assert records[0]["w"] == 42
        assert records[0]["xid"] == "a"
        assert records[0]["yid"] == "b"

    def test_native_chain_rows_select_reverse_edge_alias_projection(self):
        """Reverse traversals should project edge alias properties from bindings."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "label__X": [True, True]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"], "creationDate": [77]}),
        )
        records = self._rows_records(
            g,
            [n({"id": "b"}, name="dst"), e_reverse({"type": "R"}, name="r"), n(name="src")],
            items=[("srcId", "src.id"), ("dstId", "dst.id"), ("created", "r.creationDate")],
        )
        assert records == [
            {"srcId": "a", "dstId": "b", "created": 77}
        ]

    def test_native_chain_rows_select_missing_column_returns_null(self):
        """Missing alias-prefixed bindings columns should resolve to null, not error."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"]}),
            pd.DataFrame({"s": ["a"], "d": ["b"]}),
        )
        records = self._rows_records(
            g,
            [n(name="x"), e_forward(), n(name="y")],
            items=[("xid", "x.id"), ("missing", "x.nonexistent")],
        )
        assert len(records) == 1
        assert records[0]["xid"] == "a"
        assert records[0]["missing"] is None or pd.isna(records[0]["missing"])

    def test_native_chain_rows_select_missing_edge_property_returns_null(self):
        """Missing edge alias properties should resolve to null, not error."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
        )
        records = self._rows_records(
            g,
            [n({"id": "a"}, name="x"), e_forward({"type": "R"}, name="r"), n(name="y")],
            items=[("xid", "x.id"), ("missing", "r.nonexistent")],
        )
        assert len(records) == 1
        assert records[0]["xid"] == "a"
        assert records[0]["missing"] is None or pd.isna(records[0]["missing"])

    def test_native_chain_rows_select_parallel_edges_preserve_distinct_rows(self):
        """Duplicate edges between the same nodes should preserve distinct edge-alias rows."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "label__X": [True, True]}),
            pd.DataFrame(
                {"s": ["a", "a"], "d": ["b", "b"], "type": ["R", "R"], "weight": [10, 20]}
            ),
        )
        records = self._rows_records(
            g,
            [n({"id": "a"}, name="x"), e_forward({"type": "R"}, name="r"), n({"id": "b"}, name="y")],
            items=[("w", "r.weight"), ("xid", "x.id"), ("yid", "y.id")],
            sort_by="w",
        )
        assert records == [
            {"w": 10, "xid": "a", "yid": "b"},
            {"w": 20, "xid": "a", "yid": "b"},
        ]

    def test_native_chain_rows_bindings_reverse_edge(self):
        """Reverse edge direction should still produce correct bindings."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "val": [1, 2]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
        )
        result = g.gfql([
            n({"id": "b"}, name="dst"),
            e_reverse({"type": "R"}),
            n(name="src"),
            rows(),
        ])
        df = result._nodes
        assert len(df) == 1
        assert df["dst.id"].iloc[0] == "b"
        assert df["src.id"].iloc[0] == "a"

    def test_native_chain_rows_select_undirected_self_loop_duplicates_both_directions(self):
        """Undirected self-loops should surface both orientations in bindings rows."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a"], "label__Person": [True], "firstName": ["Alice"]}),
            pd.DataFrame({"s": ["a"], "d": ["a"], "type": ["KNOWS"], "weight": [7]}),
        )
        records = self._rows_records(
            g,
            [
                n({"id": "a", "label__Person": True}, name="seed"),
                e_undirected({"type": "KNOWS"}, name="r"),
                n({"label__Person": True}, name="friend"),
            ],
            items=[("seedId", "seed.id"), ("friendId", "friend.id"), ("w", "r.weight")],
        )
        assert records == [
            {"seedId": "a", "friendId": "a", "w": 7},
            {"seedId": "a", "friendId": "a", "w": 7},
        ]

    def test_direct_rows_binding_ops_supports_undirected_edge_alias_projection(self):
        """Direct rows(binding_ops=...) should match chain-injected undirected edge alias behavior."""
        g = self._mk_graph(
            pd.DataFrame(
                {
                    "id": ["a", "b"],
                    "label__Person": [True, True],
                    "firstName": ["Alice", "Bob"],
                }
            ),
            pd.DataFrame({"s": ["b"], "d": ["a"], "type": ["KNOWS"], "creationDate": [123]}),
        )
        match_ops = [
            n({"id": "a", "label__Person": True}, name="n"),
            e_undirected({"type": "KNOWS"}, name="r"),
            n({"label__Person": True}, name="friend"),
        ]
        self._assert_rows_binding_parity(
            g,
            items=[("personId", "friend.id"), ("created", "r.creationDate")],
            match_ops=match_ops,
            expected=[{"personId": "b", "created": 123}],
        )

    def test_native_chain_rows_bindings_open_range_continues_after_multihop(self):
        """IS6-style open-range reply chains should continue into downstream bindings."""
        g = self._mk_forum_moderator_graph()
        records = self._rows_records(
            g,
            self._forum_moderator_match_ops(
                e_forward({"type": "REPLY_OF"}, min_hops=0, to_fixed_point=True)
            ),
            items=[("forumId", "forum.id"), ("moderatorId", "moderator.id")],
        )
        assert records == [{"forumId": "f1", "moderatorId": "u1"}]

    def test_direct_rows_binding_ops_supports_open_range_multihop_continuation(self):
        """Direct rows(binding_ops=...) should preserve open-range multihop semantics."""
        g = self._mk_forum_moderator_graph()
        self._assert_rows_binding_parity(
            g,
            self._forum_moderator_match_ops(
                e_forward({"type": "REPLY_OF"}, min_hops=0, to_fixed_point=True)
            ),
            items=[("forumId", "forum.id"), ("moderatorId", "moderator.id")],
            expected=[{"forumId": "f1", "moderatorId": "u1"}],
        )

    def test_direct_rows_binding_ops_supports_open_range_multihop_continuation_on_cudf(self):
        """Direct rows(binding_ops=...) should stay on cuDF for open-range continuation replay."""
        pandas_graph = self._mk_forum_moderator_graph()
        g = self._mk_cudf_graph(pandas_graph._nodes, pandas_graph._edges)
        binding_ops = self._to_binding_ops(
            self._forum_moderator_match_ops(
                e_forward({"type": "REPLY_OF"}, min_hops=0, to_fixed_point=True)
            )
        )

        result = g.gfql(
            [
                rows(binding_ops=binding_ops),
                select([("forumId", "forum.id"), ("moderatorId", "moderator.id")]),
            ],
            engine="cudf",
        )

        assert type(result._nodes).__module__.startswith("cudf")
        assert result._nodes.to_pandas().to_dict(orient="records") == [
            {"forumId": "f1", "moderatorId": "u1"}
        ]

    def test_direct_rows_binding_ops_supports_bounded_open_range_multihop_continuation(self):
        """Bounded open-range replay should preserve downstream bindings parity."""
        g = self._mk_forum_moderator_graph()
        self._assert_rows_binding_parity(
            g,
            self._forum_moderator_match_ops(
                e_forward({"type": "REPLY_OF"}, min_hops=0, max_hops=2)
            ),
            items=[("forumId", "forum.id"), ("moderatorId", "moderator.id")],
            expected=[{"forumId": "f1", "moderatorId": "u1"}],
        )

    def test_direct_rows_binding_ops_supports_reverse_bounded_range_multihop_continuation(self):
        """Reverse bounded ranges should also replay with downstream bindings intact."""
        g = self._mk_reverse_range_continuation_graph()
        self._assert_rows_binding_parity(
            g,
            self._reverse_range_continuation_match_ops(
                e_reverse({"type": "R"}, min_hops=0, max_hops=2)
            ),
            items=[("seedId", "seed.id"), ("extraId", "extra.id")],
            expected=[{"seedId": "a", "extraId": "x"}],
        )

    def test_direct_rows_binding_ops_supports_undirected_bounded_multihop_without_backtracking(self):
        """Undirected bounded multihop replay should not immediately bounce back to the seed."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b", "c", "d"]}),
            pd.DataFrame(
                {
                    "s": ["a", "b", "c"],
                    "d": ["b", "c", "d"],
                    "type": ["R", "R", "R"],
                }
            ),
        )
        binding_ops = self._to_binding_ops(
            [n({"id": "a"}, name="seed"), e_undirected({"type": "R"}, min_hops=1, max_hops=2), n(name="peer")]
        )
        records = self._binding_rows_records(
            g,
            binding_ops,
            items=[("seedId", "seed.id"), ("peerId", "peer.id")],
            sort_by=["seedId", "peerId"],
        )
        assert records == [
            {"seedId": "a", "peerId": "b"},
            {"seedId": "a", "peerId": "c"},
        ]

    def test_direct_rows_binding_ops_supports_bare_alias_token_expressions(self):
        """Bare alias ids should be available to downstream row expressions."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b", "c", "d"]}),
            pd.DataFrame(
                {
                    "s": ["a", "b", "c"],
                    "d": ["b", "c", "d"],
                    "type": ["R", "R", "R"],
                }
            ),
        )
        binding_ops = self._to_binding_ops(
            [n({"id": "a"}, name="seed"), e_undirected({"type": "R"}, min_hops=1, max_hops=2), n(name="peer")]
        )
        records = self._binding_rows_records(
            g,
            binding_ops,
            items=[("same", "seed = peer")],
            sort_by=["same"],
        )
        assert records == [{"same": False}, {"same": False}]

    def test_direct_rows_binding_ops_rejects_duplicate_alias_names(self):
        """Direct rows(binding_ops=...) should reject duplicate aliases."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b"], "val": [1, 2]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
        )
        binding_ops = [
            n(name="x").to_json(validate=False),
            e_forward().to_json(validate=False),
            n(name="x").to_json(validate=False),
        ]
        with pytest.raises(GFQLValidationError) as exc_info:
            g.gfql([rows(binding_ops=binding_ops)])
        assert "duplicate alias" in exc_info.value.message.lower()

    def test_direct_rows_binding_ops_rejects_named_multihop_edge_alias(self):
        """Direct rows(binding_ops=...) should preserve explicit multihop edge-alias rejection."""
        g = self._mk_graph(
            pd.DataFrame({"id": ["a", "b", "c"], "val": [1, 2, 3]}),
            pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["R", "R"]}),
        )
        binding_ops = [
            n({"id": "a"}, name="x").to_json(validate=False),
            e_forward({"type": "R"}, name="r", min_hops=1, max_hops=2).to_json(validate=False),
            n(name="y").to_json(validate=False),
        ]
        with pytest.raises(Exception, match="variable-length relationship aliases"):
            g.gfql([rows(binding_ops=binding_ops)])

    def test_direct_rows_binding_ops_supports_node_only_cartesian_projection(self):
        """Direct rows(binding_ops=...) should cross-join disconnected node aliases."""
        g = self._mk_cartesian_node_graph()
        binding_ops = self._to_binding_ops([n(name="n"), n(name="m")])
        records = self._binding_rows_records(
            g,
            binding_ops,
            items=[("n_num", "n.num"), ("m_num", "m.num")],
            sort_by=["n_num", "m_num"],
        )
        assert records == [
            {"n_num": 1, "m_num": 1},
            {"n_num": 1, "m_num": 2},
            {"n_num": 2, "m_num": 1},
            {"n_num": 2, "m_num": 2},
        ]

    def test_direct_rows_binding_ops_supports_node_only_cartesian_expression(self):
        """Direct rows(binding_ops=...) should evaluate row expressions across cartesian aliases."""
        g = self._mk_cartesian_node_graph()
        binding_ops = self._to_binding_ops([n(name="n"), n(name="m")])
        records = self._binding_rows_records(
            g,
            binding_ops,
            items=[("lt", "n.ts < m.ts")],
            sort_by=["lt"],
        )
        assert records == [
            {"lt": False},
            {"lt": False},
            {"lt": False},
            {"lt": True},
        ]

    def test_inject_binding_ops_skips_existing_alias_endpoints(self):
        """Injection helper should not override explicit alias_endpoints rows()."""
        middle = [n(name="x"), e_forward(), n(name="y")]
        suffix = [rows(alias_endpoints={"x": "src", "y": "dst"})]
        out = _inject_binding_ops_if_needed(middle, suffix)
        assert out == suffix
        assert out[0].params["alias_endpoints"] == {"x": "src", "y": "dst"}
        assert "binding_ops" not in out[0].params

    def test_inject_binding_ops_skips_existing_binding_ops(self):
        """Injection helper should preserve an explicitly provided binding_ops payload."""
        middle = [n(name="x"), e_forward(), n(name="y")]
        existing = [n(name="seed").to_json(validate=False)]
        suffix = [rows(binding_ops=existing)]
        out = _inject_binding_ops_if_needed(middle, suffix)
        assert out == suffix
        assert out[0].params["binding_ops"] == existing

    def test_inject_binding_ops_skips_non_traversal_middle(self):
        """Injection helper should not serialize non-node/edge middle operations."""
        middle = [n(name="x"), select([("xid", "x.id")]), n(name="y")]
        suffix = [rows()]
        out = _inject_binding_ops_if_needed(middle, suffix)
        assert out == suffix
        assert "binding_ops" not in out[0].params
