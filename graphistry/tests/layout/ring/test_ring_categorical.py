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


class Test_categorical_ring(NoAuthTestCase):

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
                't': pd.Series(['a', 'bb', 'a', 'cc', 'bb', 'dd', 'a'])
            }))
            .ring_categorical_layout('t')
        )
        assert isinstance(g._nodes, pd.DataFrame)
        assert isinstance(g._edges, pd.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()
        rs = (g._nodes['x'] * g._nodes['x'] + g._nodes['y'] * g._nodes['y']).apply(np.sqrt)
        assert math.fabs(rs.min() - MIN_R_DEFAULT) < 0.1
        assert math.fabs(rs.max() - MAX_R_DEFAULT) < 0.1
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
                't': pd.Series(['a', 'bb', 'a', 'cc', 'bb', 'dd', 'a'])
            }))
            .ring_categorical_layout(
                ring_col='t',
                order=['a', 'bb', 'cc', 'dd'],
                min_r=500,
                max_r=800
            )
        )
        assert isinstance(g._nodes, pd.DataFrame)
        assert isinstance(g._edges, pd.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()
        assert len(g._complex_encodings and g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) == 4
        for i, row in enumerate(g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']):
            assert row['r'] == 500 + 100 * i
            assert row['label'] == ['a', 'bb', 'cc', 'dd'][i]
        rs = (g._nodes['x'] * g._nodes['x'] + g._nodes['y'] * g._nodes['y']).apply(np.sqrt)
        assert math.fabs(rs.min() - 500) < 0.1
        assert math.fabs(rs.max() - 800) < 0.1

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
                't': pd.Series(['a', 'bb', 'a', 'cc', 'bb', 'dd', 'a'])
            }))
            .ring_categorical_layout(
                ring_col='t',
                min_r=500,
                max_r=800
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
        assert rs.min() == 500
        assert rs.max() == 800
        assert len(g._complex_encodings and g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) == 4
        for i, row in enumerate(g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']):
            assert row['r'] == 500 + 100 * i
            assert row['label'] == ['a', 'bb', 'cc', 'dd'][i]


def _axis_rows(g) -> List[dict]:
    return g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']


def _g_cat(values):
    return (
        LGFull()
        .edges(pd.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]}), 's', 'd')
        .nodes(pd.DataFrame({'id': list(range(len(values))), 'c': pd.Series(values, dtype='object')}), 'id')
    )


class Test_categorical_ring_degenerate(NoAuthTestCase):
    """#1964: one distinct category is a single ring at min_r, not a ZeroDivisionError"""

    def test_single_category(self):
        g = _g_cat(['a', 'a', 'a']).ring_categorical_layout('c')
        assert (g._nodes['r'] == MIN_R_DEFAULT).all()
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()
        assert [(row['label'], row['r']) for row in _axis_rows(g)] == [('a', MIN_R_DEFAULT)]

    def test_single_category_reverse(self):
        g = _g_cat(['a', 'a', 'a']).ring_categorical_layout('c', reverse=True)
        assert (g._nodes['r'] == MIN_R_DEFAULT).all()

    def test_all_nan_category(self):
        g = _g_cat([None, None, None]).ring_categorical_layout('c')
        assert (g._nodes['r'] == MIN_R_DEFAULT).all()
        assert len(_axis_rows(g)) == 1

    def test_single_combined_unhandled_ring(self):
        g = _g_cat(['a', 'b', 'c']).ring_categorical_layout('c', order=['zz'], combine_unhandled=True)
        assert (g._nodes['r'] == MIN_R_DEFAULT).all()

    def test_empty_nodes(self):
        for dtype in ['object', str]:
            g = (
                LGFull()
                .edges(pd.DataFrame({'s': [0], 'd': [1]}), 's', 'd')
                .nodes(pd.DataFrame({'id': pd.Series([], dtype='int64'), 'c': pd.Series([], dtype=dtype)}), 'id')
                .ring_categorical_layout('c')
            )
            assert len(g._nodes) == 0
            assert _axis_rows(g) == []

    def test_partial_nan_unchanged(self):
        g = _g_cat(['a', None, 'b']).ring_categorical_layout('c')
        assert g._nodes['r'].tolist() == [MIN_R_DEFAULT, (MIN_R_DEFAULT + MAX_R_DEFAULT) / 2, MAX_R_DEFAULT]
