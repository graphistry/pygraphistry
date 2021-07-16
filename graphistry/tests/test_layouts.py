# -*- coding: utf-8 -*- 
from typing import Iterable
import os, numpy as np, pandas as pd, pyarrow as pa, pytest, queue
from common import NoAuthTestCase
from concurrent.futures import Future
from mock import patch

from graphistry.compute import ComputeMixin
from graphistry.layouts import LayoutsMixin
from graphistry.plotter import PlotterBase


class LG(LayoutsMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()
        LayoutsMixin.__init__(self, *args, **kwargs)

class LGFull(LayoutsMixin, ComputeMixin, PlotterBase):
    def __init__(self, *args, **kwargs):
        print('LGFull init')
        super(LGFull, self).__init__(*args, **kwargs)
        PlotterBase.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)
        LayoutsMixin.__init__(self, *args, **kwargs)


class TestComputeMixin(NoAuthTestCase):

    def test_tree_layout_mt(self):
        lg = LGFull()
        g = lg.edges(pd.DataFrame({'s': [], 'd': []}), 's', 'd').tree_layout()
        assert g._edges is None or len(g._edges) == 0

    def test_tree_layout_levels_1(self):
        lg = LGFull()
        g = lg.edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd').tree_layout()
        assert g._nodes.to_dict(orient='records') == [
            {'id': 'a', 'level': 0, 'x': 0, 'y': 0},
            {'id': 'b', 'level': 1, 'x': 0, 'y': -1}]

    def test_tree_layout_levels_1_aliasing(self):
        lg = LGFull()
        g = (lg
            .edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
            .nodes(pd.DataFrame({'n': ['a', 'b'], 'degree': ['x', 'y']}), 'n')
            .tree_layout())
        assert g._nodes.to_dict(orient='records') == [
            {'n': 'a', 'degree': 'x', 'level': 0, 'x': 0, 'y': 0},
            {'n': 'b', 'degree': 'y', 'level': 1, 'x': 0, 'y': -1}]

    def test_tree_layout_cycle_exn(self):
        lg = LGFull()
        with pytest.raises(ValueError):
            lg.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'a']}), 's', 'd').tree_layout(allow_cycles=False)

    def test_tree_layout_cycle_override(self):
        lg = LGFull()
        g = lg.edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd').tree_layout(allow_cycles=True)
        assert g._nodes.to_dict(orient='records') == [
            {'id': 'a', 'level': 0, 'x': 0, 'y': 0},
            {'id': 'b', 'level': 1, 'x': 0, 'y': -1}]

    def test_tree_layout_left_chain(self):
        lg = LGFull()
        g = lg.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd').tree_layout(allow_cycles=True)
        assert g._nodes.to_dict(orient='records') == [
            {'id': 'a', 'level': 0, 'x': 0, 'y': 0},
            {'id': 'b', 'level': 1, 'x': 0, 'y': -1},
            {'id': 'c', 'level': 2, 'x': 0, 'y': -2}
        ]

    def test_tree_layout_center_tree(self):
        lg = LGFull()
        g = (lg
            .edges(pd.DataFrame({'s': ['a', 'a'], 'd': ['b', 'c']}), 's', 'd')
            .tree_layout(allow_cycles=True, level_align='center', width=100))
        #FIXME: x range wrong
        assert g._nodes.to_dict(orient='records') == [
            {'id': 'a', 'level': 0, 'x': 100.0, 'y': 0.0},
            {'id': 'b', 'level': 1, 'x': 50.0, 'y': -100.0},
            {'id': 'c', 'level': 1, 'x': 150.0, 'y': -100.0}
        ]

    def test_tree_layout_sort_ascending(self):
        lg = LGFull()
        g = (lg
            .edges(pd.DataFrame({'s': ['a', 'a'], 'd': ['b', 'c']}), 's', 'd')
            .nodes(pd.DataFrame({'n': ['a', 'b', 'c'], 'v': [0, 1, 2]}), 'n')
            .tree_layout(level_sort_values_by='v'))
        assert g._nodes.to_dict(orient='records') == [
            {'n': 'a', 'v': 0, 'level': 0, 'x': 0, 'y': 0},
            {'n': 'b', 'v': 1, 'level': 1, 'x': 0, 'y': -1},
            {'n': 'c', 'v': 2, 'level': 1, 'x': 1, 'y': -1}
        ]

    def test_tree_layout_sort_descending(self):
        lg = LGFull()
        g = (lg
            .edges(pd.DataFrame({'s': ['a', 'a'], 'd': ['b', 'c']}), 's', 'd')
            .nodes(pd.DataFrame({'n': ['a', 'b', 'c'], 'v': [0, 1, 2]}), 'n')
            .tree_layout(level_sort_values_by='v', level_sort_values_by_ascending=False))
        assert g._nodes.to_dict(orient='records') == [
            {'n': 'c', 'v': 2, 'level': 1, 'x': 0, 'y': -1},
            {'n': 'b', 'v': 1, 'level': 1, 'x': 1, 'y': -1},
            {'n': 'a', 'v': 0, 'level': 0, 'x': 0, 'y': 0}
        ]
