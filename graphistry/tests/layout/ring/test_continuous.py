from typing import List
import math, os, numpy as np, pandas as pd, pytest, warnings
from graphistry.compute import ComputeMixin
from graphistry.layout.ring.time import MIN_R_DEFAULT, MAX_R_DEFAULT
from graphistry.layouts import LayoutsMixin
from graphistry.plotter import PlotterBase
from graphistry.tests.common import NoAuthTestCase


test_cudf = "TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"

class LG(LayoutsMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()
        LayoutsMixin.__init__(self, *args, **kwargs)


class LGFull(LayoutsMixin, ComputeMixin, PlotterBase):
    def __init__(self, *args, **kwargs):
        super(LGFull, self).__init__(*args, **kwargs)
        PlotterBase.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)
        LayoutsMixin.__init__(self, *args, **kwargs)


class Test_continuous_ring(NoAuthTestCase):

    def test_mt_pd(self):
        
        lg = LGFull()
        #with warnings.catch_warnings():
        #    warnings.filterwarnings("ignore", category=FutureWarning)
        g = (
            lg
            .edges(
                pd.DataFrame({
                    's': ['a', 'b', 'c', 'd', 'm', 'd1'],
                    'd': ['b', 'c', 'd', 'd', 'm', 'd2']
                }),
                's', 'd')
            .nodes(pd.DataFrame({
                'n': ['a', 'b', 'c', 'd', 'm', 'd1', 'd2'],
                't': pd.Series([2, 4, 2, 3, 5, 10, 4])
            }))
            .ring_continuous_layout()
        )
        assert isinstance(g._nodes, pd.DataFrame)
        assert isinstance(g._edges, pd.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()
        rs = (g._nodes['x'] * g._nodes['x'] + g._nodes['y'] * g._nodes['y']).apply(np.sqrt)
        assert rs.min() >= MIN_R_DEFAULT - 1e-10  # Allow for floating point precision
        assert rs.max() <= MAX_R_DEFAULT + 1e-10  # Allow for floating point precision
        assert len(g._complex_encodings and g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) > 0

    def test_configured_pd(self):
        
        lg = LGFull()
        #with warnings.catch_warnings():
        #    warnings.filterwarnings("ignore", category=FutureWarning)
        g = (
            lg
            .edges(
                pd.DataFrame({
                    's': ['a', 'b', 'c', 'd', 'm', 'd1'],
                    'd': ['b', 'c', 'd', 'd', 'm', 'd2']
                }),
                's', 'd')
            .nodes(pd.DataFrame({
                'n': ['a', 'b', 'c', 'd', 'm', 'd1', 'd2'],
                't': pd.Series([2, 4, 2, 3, 5, 10, 4])
            }))
            .ring_continuous_layout(
                ring_col='t',
                min_r=500,
                max_r=900,
                v_start=2,
                v_end=10,
                v_step=2
            )
        )
        assert isinstance(g._nodes, pd.DataFrame)
        assert isinstance(g._edges, pd.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()
        rs = (g._nodes['x'] * g._nodes['x'] + g._nodes['y'] * g._nodes['y']).apply(np.sqrt)
        assert np.isclose(rs.min(), 500, rtol=1e-10)
        assert np.isclose(rs.max(), 900, rtol=1e-10)
        assert len(g._complex_encodings and g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) == 5
        for i, row in enumerate(g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']):
            assert row['r'] == 500 + 100 * i
            assert row['label'] == str(2 + 2 * i)

    @pytest.mark.skipif(
        not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
        reason="cudf tests need TEST_CUDF=1")
    def test_ring_cudf(self):
        import cudf


        lg = LGFull()
        #with warnings.catch_warnings():
        #    warnings.filterwarnings("ignore", category=FutureWarning)
        g = (
            lg
            .edges(
                cudf.DataFrame({
                    's': ['a', 'b', 'c', 'd', 'm', 'd1'],
                    'd': ['b', 'c', 'd', 'd', 'm', 'd2']
                }),
                's', 'd')
            .nodes(cudf.DataFrame({
                'n': ['a', 'b', 'c', 'd', 'm', 'd1', 'd2'],
                't': cudf.Series([2, 4, 2, 3, 5, 10, 4])
            }))
            .ring_continuous_layout(
                ring_col='t',
                min_r=500,
                max_r=900,
                v_start=2,
                v_end=10,
                v_step=2
            )
        )
        assert isinstance(g._nodes, cudf.DataFrame)
        assert isinstance(g._edges, cudf.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()
        g._nodes = g._nodes.to_pandas()
        rs = (g._nodes['x'] * g._nodes['x'] + g._nodes['y'] * g._nodes['y']).apply(np.sqrt)
        assert np.isclose(rs.min(), 500, rtol=1e-10)
        assert np.isclose(rs.max(), 900, rtol=1e-10)
        assert len(g._complex_encodings and g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) == 5
        for i, row in enumerate(g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']):
            assert row['r'] == 500 + 100 * i
            assert row['label'] == str(2 + 2 * i)

    def test_play_ms_preserves_url_param(self):
        lg = LGFull()
        g = (
            lg
            .edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd')
            .nodes(pd.DataFrame({'n': ['a', 'b', 'c'], 't': [1, 2, 3]}))
            .settings(url_params={'play': 6000})
            .ring_continuous_layout('t')
        )
        assert g._url_params.get('play') == 6000

    def test_play_ms_explicit_zero(self):
        lg = LGFull()
        g = (
            lg
            .edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd')
            .nodes(pd.DataFrame({'n': ['a', 'b', 'c'], 't': [1, 2, 3]}))
            .settings(url_params={'play': 6000})
            .ring_continuous_layout('t', play_ms=0)
        )
        assert g._url_params.get('play') == 0

    def test_play_ms_explicit_value(self):
        lg = LGFull()
        g = (
            lg
            .edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd')
            .nodes(pd.DataFrame({'n': ['a', 'b', 'c'], 't': [1, 2, 3]}))
            .settings(url_params={'play': 6000})
            .ring_continuous_layout('t', play_ms=3000)
        )
        assert g._url_params.get('play') == 3000

    def test_play_ms_default_when_no_url_param(self):
        lg = LGFull()
        g = (
            lg
            .edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd')
            .nodes(pd.DataFrame({'n': ['a', 'b', 'c'], 't': [1, 2, 3]}))
            .ring_continuous_layout('t')
        )
        assert g._url_params.get('play') == 0
