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


