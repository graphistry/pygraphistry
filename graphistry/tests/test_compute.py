# -*- coding: utf-8 -*- 
from typing import Iterable
import os, numpy as np, pandas as pd, pyarrow as pa, pytest, queue
from common import NoAuthTestCase
from concurrent.futures import Future
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


def assertFrameEqual(df1, df2, **kwds ):
    """ Assert that two dataframes are equal, ignoring ordering of columns"""

    from pandas.testing import assert_frame_equal
    return assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1), check_names=True, **kwds)

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

    def test_replace_nodes_with_edges_input_list(self):
        cg = CGFull()
        g = cg.edges(pd.DataFrame({'x': ['m', 'm', 'n', 'm'], 'y': ['a', 'b', 'c','d']}), 'x', 'y')
        g2 = g.replace_nodes_with_edges(['m','n'], g._source)
        edges = pd.DataFrame({
            'x': ['a', 'a', 'a', 'b', 'b', 'c'],
            'y': ['b', 'c', 'd', 'c', 'd', 'd'],
            'reduced_from': ['m,n'] * 6
        })
        assertFrameEqual(g2._edges, edges)

    def test_replace_nodes_with_edges_input_series(self):
        cg = CGFull()
        df = pd.DataFrame({'x': ['a', 'b', 'c','d'], 'y': ['m', 'm', 'n', 'm']})
        g = cg.edges(df, 'x', 'y').nodes(df, 'y')
        nodes_series = pd.Series(['m', 'n'])
        g2 = g.replace_nodes_with_edges(nodes_series)
        edges = pd.DataFrame({
            'x': ['a', 'a', 'a', 'b', 'b', 'c'],
            'y': ['b', 'c', 'd', 'c', 'd', 'd'],
            'reduced_from': ['m,n'] * 6
        })
        assertFrameEqual(g2._edges, edges)
