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


def _axis_rows(g) -> List[dict]:
    return g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']


def _g_numeric(values, ids=None):
    ids = list(range(len(values))) if ids is None else ids
    return (
        LGFull()
        .edges(pd.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]}), 's', 'd')
        .nodes(pd.DataFrame({'id': ids, 'v': pd.Series(values, dtype='float64')}), 'id')
    )


class Test_continuous_ring_degenerate(NoAuthTestCase):
    """#1964: degenerate ring_col values keep every present value in [min_r, max_r]"""

    def test_constant_col_one_ring(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            g = _g_numeric([2.0, 2.0, 2.0]).ring_continuous_layout('v')
        assert (g._nodes['r'] == MIN_R_DEFAULT).all()
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()
        rs = np.sqrt(g._nodes['x'] ** 2 + g._nodes['y'] ** 2)
        np.testing.assert_allclose(rs, MIN_R_DEFAULT)
        assert any(row['r'] == MIN_R_DEFAULT and row['label'] == '2.0' for row in _axis_rows(g))

    def test_constant_col_reverse(self):
        g = _g_numeric([2.0, 2.0, 2.0]).ring_continuous_layout('v', reverse=True)
        assert (g._nodes['r'] == MAX_R_DEFAULT).all()
        assert any(row['r'] == MAX_R_DEFAULT and row['label'] == '2.0' for row in _axis_rows(g))

    def test_single_node(self):
        g = _g_numeric([5.0], ids=[0]).ring_continuous_layout('v')
        assert g._nodes['r'].tolist() == [MIN_R_DEFAULT]
        assert not g._nodes.x.isna().any()

    def test_int_v_start_equals_v_end(self):
        g = _g_numeric([2.0, 2.0, 2.0]).ring_continuous_layout('v', v_start=2, v_end=2)
        assert (g._nodes['r'] == MIN_R_DEFAULT).all()

    def test_constant_col_v_step(self):
        g = _g_numeric([2.0, 2.0, 2.0]).ring_continuous_layout('v', v_step=1)
        assert (g._nodes['r'] == MIN_R_DEFAULT).all()
        assert [(row['label'], row['r']) for row in _axis_rows(g)] == [
            ('2.0', MIN_R_DEFAULT), ('3.0', MAX_R_DEFAULT)
        ]

    def test_all_nan_col(self):
        g = _g_numeric([np.nan, np.nan, np.nan]).ring_continuous_layout('v')
        assert g._nodes['r'].isna().all()
        assert g._nodes['x'].isna().all()
        assert _axis_rows(g) == []

    def test_empty_nodes(self):
        g = (
            LGFull()
            .edges(pd.DataFrame({'s': [0], 'd': [1]}), 's', 'd')
            .nodes(pd.DataFrame({'id': pd.Series([], dtype='int64'), 'v': pd.Series([], dtype='float64')}), 'id')
            .ring_continuous_layout('v')
        )
        assert len(g._nodes) == 0
        assert _axis_rows(g) == []

    def test_partial_nan_unchanged(self):
        g = _g_numeric([1.0, np.nan, 3.0]).ring_continuous_layout('v')
        assert g._nodes['r'][0] == MIN_R_DEFAULT
        assert np.isnan(g._nodes['r'][1])
        assert g._nodes['r'][2] == MAX_R_DEFAULT
