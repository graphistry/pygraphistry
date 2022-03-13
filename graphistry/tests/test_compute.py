# -*- coding: utf-8 -*- 
from typing import Iterable
import os, numpy as np, pandas as pd, pyarrow as pa, pytest, queue
from common import NoAuthTestCase
from concurrent.futures import Future
from functools import lru_cache
from mock import patch

from graphistry.plotter import PlotterBase
from graphistry.compute import ComputeMixin

class CG(ComputeMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ComputeMixin.__init__(self, *args, **kwargs)

class CGFull(ComputeMixin, PlotterBase, object):
    def __init__(self, *args, **kwargs):
        print('CGFull init')
        super(CGFull, self).__init__(*args, **kwargs)
        PlotterBase.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)


@lru_cache(maxsize=1)
def hops_graph():
    nodes_df = pd.DataFrame([
        {'node': '431276-1307141'},
        {'node': '431276-1306509'},
        {'node': '431276-1528963'},
        {'node': '431276-1343384'},
        {'node': '431276-1527699'},
        {'node': '431276-1308405'},
        {'node': '431276-1308089'},
        {'node': '431276-1529279'},
        {'node': '431276-1343700'},
        {'node': '431276-1308721'},
        {'node': '431276-1306825'},
        {'node': '431276-1528015'},
        {'node': '1529911-1529595'},
        {'node': '1344016-1342436'},
        {'node': '1421143-1420827'},
        {'node': '1309353-1309037'}
    ]).assign(type='n')

    edges_df = pd.DataFrame([
        {'s': '431276-1527699', 'd': '431276-1528015'},
        {'s': '431276-1528015', 'd': '431276-1306509'},
        {'s': '431276-1306825', 'd': '431276-1307141'},
        {'s': '431276-1527699', 'd': '431276-1308089'},
        {'s': '431276-1308089', 'd': '431276-1307141'},
        {'s': '431276-1343384', 'd': '431276-1308405'},
        {'s': '431276-1343384', 'd': '431276-1528963'},
        {'s': '431276-1343384', 'd': '431276-1308721'},
        {'s': '431276-1343384', 'd': '431276-1343700'},
        {'s': '431276-1343384', 'd': '431276-1529279'},
        {'s': '431276-1308721', 'd': '1309353-1309037'},
        {'s': '431276-1343700', 'd': '1344016-1342436'},
        {'s': '431276-1529279', 'd': '1529911-1529595'},
        {'s': '431276-1308721', 'd': '1421143-1420827'},
        {'s': '1421143-1420827', 'd': '431276-1306509'},
        {'s': '1529911-1529595', 'd': '431276-1307141'},
        {'s': '1344016-1342436', 'd': '431276-1307141'},
        {'s': '1309353-1309037', 'd': '431276-1306509'},
    ]).assign(type='e')

    return CGFull().nodes(nodes_df, 'node').edges(edges_df, 's', 'd')


class TestComputeMixin(NoAuthTestCase):


    def test_materialize(self):
        cg = CGFull()
        g = cg.edges(pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'a', 'd']}), 's', 'd')
        g = g.materialize_nodes()
        assert g._nodes.to_dict(orient='records') == [
            {'id': 'a'},
            {'id': 'b'},
            {'id': 'c'},
            {'id': 'd'}
        ]
        assert g._node == 'id'

    def test_materialize_reuse(self):
        cg = CGFull()
        g = cg.edges(pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'a', 'd']}), 's', 'd')
        g = g.nodes(pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'v': [2, 4, 6, 8]}), 'id')
        g = g.materialize_nodes()
        assert g._nodes.to_dict(orient='records') == [
            {'id': 'a', 'v': 2},
            {'id': 'b', 'v': 4},
            {'id': 'c', 'v': 6},
            {'id': 'd', 'v': 8}
        ]
        assert g._node == 'id'

    def test_degrees_in(self):
        cg = CGFull()
        g = cg.edges(pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'a', 'd']}), 's', 'd')
        g2 = g.get_indegrees()
        assert g2._nodes.to_dict(orient='records') == [
            {'id': 'a', 'degree_in': 1},
            {'id': 'b', 'degree_in': 1},
            {'id': 'c', 'degree_in': 0},
            {'id': 'd', 'degree_in': 1}
        ]
        assert g2._node == 'id'

    def test_degrees_out(self):
        cg = CGFull()
        g = cg.edges(pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'a', 'd']}), 's', 'd')
        g2 = g.get_outdegrees()
        assert g2._nodes.to_dict(orient='records') == [
            {'id': 'b', 'degree_out': 1},
            {'id': 'a', 'degree_out': 1},
            {'id': 'd', 'degree_out': 0},
            {'id': 'c', 'degree_out': 1}
        ]
        assert g2._node == 'id'

    def test_degrees(self):
        cg = CGFull()
        g = cg.edges(pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'a', 'd']}), 's', 'd')
        g2 = g.get_degrees()
        assert g2._nodes.to_dict(orient='records') == [
            {'id': 'a', 'degree_in': 1, 'degree_out': 1, 'degree': 2},
            {'id': 'b', 'degree_in': 1, 'degree_out': 1, 'degree': 2},
            {'id': 'c', 'degree_in': 0, 'degree_out': 1, 'degree': 1},
            {'id': 'd', 'degree_in': 1, 'degree_out': 0, 'degree': 1}
        ]
        assert g2._node == 'id'

    def test_get_topological_levels_mt(self):
        cg = CGFull()
        g = cg.edges(pd.DataFrame({'s': [], 'd': []}), 's', 'd').get_topological_levels()
        assert g._edges is None or len(g._edges) == 0

    def test_get_topological_levels_1(self):
        cg = CGFull()
        g = cg.edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd').get_topological_levels()
        assert g._nodes.to_dict(orient='records') == [
            {'id': 'a', 'level': 0},
            {'id': 'b', 'level': 1}
        ]

    def test_get_topological_levels_1_aliasing(self):
        cg = CGFull()
        g = (cg
            .edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
            .nodes(pd.DataFrame({'n': ['a', 'b'], 'degree': ['x', 'y']}), 'n')
            .get_topological_levels())
        assert g._nodes.to_dict(orient='records') == [
            {'n': 'a', 'level': 0, 'degree': 'x'},
            {'n': 'b', 'level': 1, 'degree': 'y'}
        ]

    def test_get_topological_levels_cycle_exn(self):
        cg = CGFull()
        with pytest.raises(ValueError):
            cg.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'a']}), 's', 'd').get_topological_levels(allow_cycles=False)

    def test_get_topological_levels_cycle_override(self):
        cg = CGFull()
        g = cg.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'a']}), 's', 'd').get_topological_levels(allow_cycles=True)
        assert g._nodes.to_dict(orient='records') == [{'id': 'a', 'level': 0}, {'id': 'b', 'level': 1}]

    def test_drop_nodes(self):
        cg = CGFull()
        g = cg.edges(pd.DataFrame({'x': ['m', 'm', 'n', 'm'], 'y': ['a', 'b', 'c', 'd']}), 'x', 'y')
        g2 = g.drop_nodes(['m'])
        assert g2._edges.to_dict(orient='records') == [{'x': 'n', 'y': 'c'}]


    def test_hop_0(self):

        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: []}), 0)
        assert g2._nodes.shape == (0, 2)
        assert g2._edges.shape == (0, 3)

    def test_hop_0b(self):

        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['431276-1343384']}), 0)
        assert g2._nodes.shape == (1, 2)
        assert g2._edges.shape == (0, 3)

    def test_hop_1_1_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['431276-1343384']}), 1)
        assert g2._nodes.shape == (6, 2)
        assert (g2._nodes[g2._node].sort_values().to_list() ==  # noqa: W504
            sorted(['431276-1308405', '431276-1308721', '431276-1343384','431276-1343700', '431276-1528963', '431276-1529279']))
        assert g2._edges.shape == (5, 3)

    def test_hop_2_1_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['431276-1306825', '431276-1343384']}), 1)
        assert g2._nodes.shape == (8, 2)
        assert g2._edges.shape == (6, 3)

    def test_hop_2_2_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['431276-1306825', '431276-1343384']}), 2)
        assert g2._nodes.shape == (12, 2)
        assert g2._edges.shape == (10, 3)

    def test_hop_2_all_forwards(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['431276-1306825', '431276-1343384']}), to_fixed_point=True)
        assert g2._nodes.shape == (13, 2)
        assert g2._edges.shape == (14, 3)

    def test_hop_1_2_undirected(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['431276-1308721']}), 2, direction='undirected')
        assert g2._nodes.shape == (9, 2)
        assert g2._edges.shape == (9, 3)

    def test_hop_1_all_reverse(self):
        g = hops_graph()
        g2 = g.hop(pd.DataFrame({g._node: ['431276-1306509']}), direction='reverse', to_fixed_point=True)
        assert g2._nodes.shape == (7, 2)
        assert g2._edges.shape == (7, 3)
