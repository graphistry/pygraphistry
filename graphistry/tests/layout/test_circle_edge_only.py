import pandas as pd

from graphistry.compute import ComputeMixin
from graphistry.layouts import LayoutsMixin
from graphistry.plotter import PlotterBase
from graphistry.tests.common import NoAuthTestCase


class LGFull(LayoutsMixin, ComputeMixin, PlotterBase):
    def __init__(self, *args, **kwargs):
        super(LGFull, self).__init__(*args, **kwargs)
        PlotterBase.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)
        LayoutsMixin.__init__(self, *args, **kwargs)


class Test_circle_edge_only(NoAuthTestCase):

    def test_circle_layout_materializes_nodes_from_edges(self):
        # circle_layout used to read len(self._nodes) before materialize_nodes()
        # ran, so an edge-only graph crashed with TypeError: object of type
        # 'NoneType' has no len()
        g = (
            LGFull()
            .edges(pd.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]}), 's', 'd')
            .circle_layout(bounding_box=(0, 0, 10, 10))
        )
        assert isinstance(g._nodes, pd.DataFrame)
        assert len(g._nodes) == 3
        assert 'x' in g._nodes
        assert 'y' in g._nodes
