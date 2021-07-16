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

class CGFull(ComputeMixin, PlotterBase):
    def __init__(self, *args, **kwargs):
        print('CGFull init')
        super(CGFull, self).__init__(*args, **kwargs)
        PlotterBase.__init__(self, *args, **kwargs)
        super(ComputeMixin, self).__init__(*args, **kwargs)


class TestComputeMixin(NoAuthTestCase):

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

    def test_get_topological_levels_cycle_exn(self):
        cg = CGFull()
        with pytest.raises(ValueError):
            cg.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'a']}), 's', 'd').get_topological_levels(allow_cycles=False)

    def test_get_topological_levels_cycle_override(self):
        cg = CGFull()
        g = cg.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'a']}), 's', 'd').get_topological_levels(allow_cycles=True)
        assert g._nodes.to_dict(orient='records') == [{'id': 'a', 'level': 0}, {'id': 'b', 'level': 1}]
