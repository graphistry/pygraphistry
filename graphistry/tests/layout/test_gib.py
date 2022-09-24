import logging, os, pandas as pd, pytest, warnings
from graphistry.compute import ComputeMixin
from graphistry.layouts import LayoutsMixin
from graphistry.plotter import PlotterBase
from graphistry.tests.common import NoAuthTestCase
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


test_cudf = "TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"

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


class Test_gib(NoAuthTestCase):

    def test_gib_pd(self):
        try:
            import igraph
        except:
            return
        
        lg = LGFull()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            g = (
                lg
                .edges(
                    pd.DataFrame({
                        's': ['a', 'b', 'c', 'd', 'm', 'd1'],
                        'd': ['b', 'c', 'd', 'd', 'm', 'd2']
                    }),
                    's', 'd')
                .group_in_a_box_layout()
            )
        assert isinstance(g._nodes, pd.DataFrame)
        assert isinstance(g._edges, pd.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()

    @pytest.mark.skipif(
        not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
        reason="cudf tests need TEST_CUDF=1")
    def test_gib_cudf(self):
        import cudf
   
        lg = LGFull()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            g = (
                lg
                .edges(
                    cudf.DataFrame({
                        's': ['a', 'b', 'c', 'd', 'm', 'd1'],
                        'd': ['b', 'c', 'd', 'd', 'm', 'd2']
                    }),
                    's', 'd')
                .group_in_a_box_layout()
            )
        print('g ::', type(g))
        print('g._nodes ::', type(g._nodes))
        print('g._edges ::', type(g._edges))
        assert isinstance(g._nodes, cudf.DataFrame)
        assert isinstance(g._edges, cudf.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()
