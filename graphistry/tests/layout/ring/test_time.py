from typing import List
import math, os, numpy as np, pandas as pd, pytest, warnings
from graphistry.compute import ComputeMixin
from graphistry.layout.ring.time import MIN_R_DEFAULT, MAX_R_DEFAULT, find_round_bin_width
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


class Test_time_ring(NoAuthTestCase):

    def test_ring_mt_pd(self):
        
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
                .nodes(pd.DataFrame({
                    'n': ['a', 'b', 'c', 'd', 'm', 'd1', 'd2'],
                    't': pd.Series([
                        '2015-01-16 18:39:37',
                        '2015-01-29 02:15:35',
                        '2014-12-30 18:59:20',
                        '2015-01-20 08:12:27',
                        '2014-11-22 19:47:15',
                        '2014-11-21 14:38:07',
                        '2014-11-20 15:28:12'
                        ],
                        dtype='datetime64[ns]')
                }))
                .time_ring_layout()
            )
        assert isinstance(g._nodes, pd.DataFrame)
        assert isinstance(g._edges, pd.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()
        #rs = (g._nodes['x'] * g._nodes['x'] + g._nodes['y'] * g._nodes['y']).apply(np.sqrt)
        #assert rs.min() >= MIN_R_DEFAULT
        #assert rs.max() <= MAX_R_DEFAULT
        #print('ce', g._complex_encodings)
        assert len(g._complex_encodings and g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) > 0

    def test_ring_pd(self):
        
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
                .nodes(pd.DataFrame({
                    'n': ['a', 'b', 'c', 'd', 'm', 'd1', 'd2'],
                    't': pd.Series([
                        '2015-01-16 18:39:37',
                        '2015-01-29 02:15:35',
                        '2014-12-30 18:59:20',
                        '2015-01-20 08:12:27',
                        '2014-11-22 19:47:15',
                        '2014-11-21 14:38:07',
                        '2014-11-20 15:28:12'
                        ],
                        dtype='datetime64[ns]')
                }))
                .time_ring_layout('t')
            )
        assert isinstance(g._nodes, pd.DataFrame)
        assert isinstance(g._edges, pd.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()
        #rs = (g._nodes['x'] * g._nodes['x'] + g._nodes['y'] * g._nodes['y']).apply(np.sqrt)
        #assert rs.min() >= MIN_R_DEFAULT
        #assert rs.max() <= MAX_R_DEFAULT
        #print('ce', g._complex_encodings)
        assert len(g._complex_encodings and g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) > 0

    def test_ring_pd_reverse(self):
        
        lg = LGFull()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            g0 = (
                lg
                .edges(
                    pd.DataFrame({
                        's': ['a', 'b', 'c', 'd', 'm', 'd1'],
                        'd': ['b', 'c', 'd', 'd', 'm', 'd2']
                    }),
                    's', 'd')
                .nodes(pd.DataFrame({
                    'n': ['a', 'b', 'c', 'd', 'm', 'd1', 'd2'],
                    't': pd.Series([
                        '2015-01-16 18:39:37',
                        '2015-01-29 02:15:35',
                        '2014-12-30 18:59:20',
                        '2015-01-20 08:12:27',
                        '2014-11-22 19:47:15',
                        '2014-11-21 14:38:07',
                        '2014-11-20 15:28:12'
                        ],
                        dtype='datetime64[ns]')
                }))
            )
            g1 = g0.time_ring_layout('t')
            axis1: List = g1._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
            g2 = g0.time_ring_layout('t', reverse=True)
            axis2: List = g2._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
            assert len(axis1) == len(axis2)


            #rs = (g2._nodes['x'] * g2._nodes['x'] + g2._nodes['y'] * g2._nodes['y']).apply(np.sqrt)
            #assert rs.min() >= MIN_R_DEFAULT
            #assert rs.max() <= MAX_R_DEFAULT

            # same except r are flipped
            #for i, v in enumerate(axis2):
            #    print('rev', i, len(axis1) - i - 1, len(axis1), axis1[len(axis1) - i - 1]['r'], v['r'])
            #    assert v['r'] == axis1[len(axis1) - i - 1]['r']
            for i, v in enumerate(axis2):
                v['r'] = axis1[i]['r']
                assert v == axis1[i]
            g1_r2 = g1._nodes['x'] * g1._nodes['x'] + g1._nodes['y'] * g1._nodes['y']
            g1_r = g1_r2.apply(np.sqrt)
            g2_r2 = g2._nodes['x'] * g2._nodes['x'] + g2._nodes['y'] * g2._nodes['y']
            g2_r = g2_r2.apply(np.sqrt)
            min_g1 = g1_r.min()
            max_g1 = g1_r.max()
            min_g2 = g2_r.min()
            max_g2 = g2_r.max()
            #g2_delta = -100  # extra ring
            assert math.isclose(min_g1 - MIN_R_DEFAULT, MAX_R_DEFAULT - max_g2)
            assert math.isclose(min_g2 - MIN_R_DEFAULT, (MAX_R_DEFAULT - max_g1))

    def test_ring_pd_num_rings(self):

        N_RINGS = 3
        
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
                .nodes(pd.DataFrame({
                    'n': ['a', 'b', 'c', 'd', 'm', 'd1', 'd2'],
                    't': pd.Series([
                        '2015-01-16 18:39:37',
                        '2015-01-29 02:15:35',
                        '2014-12-30 18:59:20',
                        '2015-01-20 08:12:27',
                        '2014-11-22 19:47:15',
                        '2014-11-21 14:38:07',
                        '2014-11-20 15:28:12'
                        ],
                        dtype='datetime64[ns]')
                }))
                .time_ring_layout('t', num_rings=N_RINGS)
            )
        assert len(g._complex_encodings and g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) == N_RINGS + 1

    def test_ring_pd_time_unit(self):

        TIME_UNIT = 'Y'
        
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
                .nodes(pd.DataFrame({
                    'n': ['a', 'b', 'c', 'd', 'm', 'd1', 'd2'],
                    't': pd.Series([
                        '2015-01-16 18:39:37',
                        '2015-01-29 02:15:35',
                        '2014-12-30 18:59:20',
                        '2015-01-20 08:12:27',
                        '2014-11-22 19:47:15',
                        '2014-11-21 14:38:07',
                        '2014-11-20 15:28:12'
                        ],
                        dtype='datetime64[ns]')
                }))
            )
            g0 = g.time_ring_layout('t', time_unit=TIME_UNIT)

        assert len(g0._complex_encodings and g0._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) == 2
        labels = [
            row['label'] for row in g0._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
        ]
        assert labels == ['2014', '2015']

        g1 = g.time_ring_layout('t', time_unit=TIME_UNIT, num_rings=2)
        labels = [
            row['label'] for row in g1._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
        ]
        assert labels == ['2014', '2015', '2016']

        g2 = g.time_ring_layout('t', time_unit='M', num_rings=3)
        labels = [
            row['label'] for row in g2._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
        ]
        assert labels == ['2014-11', '2014-12', '2015-01', '2015-02']

        g2 = g.time_ring_layout(time_unit='M')
        labels = [
            row['label'] for row in g2._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
        ]
        assert labels == ['2014-11', '2014-12', '2015-01', '2015-02']

        g_def2 = g.time_ring_layout('t', num_rings=1, time_unit='Y')
        labels = [
            row['label'] for row in g_def2._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
        ]
        assert labels == ['2014', '2015']

        g_def4 = g.time_ring_layout('t', num_rings=3, time_unit='M')
        labels = [
            row['label'] for row in g_def4._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
        ]
        assert labels == ['2014-11', '2014-12', '2015-01', '2015-02']

        g_def = g.time_ring_layout('t')
        labels = [
            row['label'] for row in g_def._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
        ]
        assert labels == ['2014', '2015'] or labels == ['2014-11', '2014-12', '2015-01', '2015-02', '2015-03', '2015-04']


    def test_ring_pd_us_ns_equivalence(self):

        lg = LGFull()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            edges = pd.DataFrame({
                's': ['a', 'b', 'c', 'd', 'm', 'd1'],
                'd': ['b', 'c', 'd', 'd', 'm', 'd2']
            })
            nodes = pd.DataFrame({
                'n': ['a', 'b', 'c', 'd', 'm', 'd1', 'd2'],
                't': pd.Series([
                    '2015-01-16 18:39:37',
                    '2015-01-29 02:15:35',
                    '2014-12-30 18:59:20',
                    '2015-01-20 08:12:27',
                    '2014-11-22 19:47:15',
                    '2014-11-21 14:38:07',
                    '2014-11-20 15:28:12'
                    ],
                    dtype='datetime64[ns]')
            })
            nodes_us = nodes.copy()
            nodes_us['t'] = nodes_us['t'].astype('datetime64[us]')
            nodes_ms = nodes.copy()
            nodes_ms['t'] = nodes_ms['t'].astype('datetime64[ms]')

            g_ns = lg.edges(edges, 's', 'd').nodes(nodes).time_ring_layout('t')
            g_us = lg.edges(edges, 's', 'd').nodes(nodes_us).time_ring_layout('t')
            g_ms = lg.edges(edges, 's', 'd').nodes(nodes_ms).time_ring_layout('t')

        np.testing.assert_allclose(
            g_ns._nodes['r'].to_numpy(),
            g_us._nodes['r'].to_numpy()
        )
        np.testing.assert_allclose(
            g_ns._nodes['r'].to_numpy(),
            g_ms._nodes['r'].to_numpy()
        )


    def test_ring_pd_axis_positions(self):
        
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
                .nodes(pd.DataFrame({
                    'n': ['a', 'b', 'c', 'd', 'm', 'd1', 'd2'],
                    't': pd.Series([
                        '2015-01-16 18:39:37',
                        '2015-01-29 02:15:35',
                        '2014-12-30 18:59:20',
                        '2015-01-20 08:12:27',
                        '2014-11-22 19:47:15',
                        '2014-11-21 14:38:07',
                        '2014-11-20 15:28:12'
                        ],
                        dtype='datetime64[ns]')
                }))
            )
        
        g0 = g.time_ring_layout('t', time_unit='D', num_rings=60)
        assert len(g0._complex_encodings and g0._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) == 61
        labels = [
            row['r'] for row in g0._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
        ]
        w = (MAX_R_DEFAULT - MIN_R_DEFAULT) / (len(labels) - 1)
        assert labels == [MIN_R_DEFAULT + i * w for i in range(len(labels))]

        g1 = g.time_ring_layout('t', time_unit='M')
        assert len(g1._complex_encodings and g1._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) == 4
        labels = [
            row['r'] for row in g1._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
        ]
        w = (MAX_R_DEFAULT - MIN_R_DEFAULT) / (len(labels) - 1)
        #assert labels == [MIN_R_DEFAULT + i * w for i in range(len(labels))]
        assert labels[0] == 100.0
        expected = [100.0, 390.33, 690.33, 990.33]
        assert all([
            math.fabs(labels[i] - expected[i]) < 0.1
            for i in range(len(labels)) 
        ])

        g2 = g.time_ring_layout('t', time_unit='D', num_rings=60, reverse=True)
        assert len(g2._complex_encodings and g2._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']) == 61
        labels = [
            row['r'] for row in g2._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']
        ]
        w = (MAX_R_DEFAULT - MIN_R_DEFAULT) / (len(labels) - 1)
        assert labels == [MAX_R_DEFAULT - i * w for i in range(len(labels))]

    @pytest.mark.skipif(
        not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
        reason="cudf tests need TEST_CUDF=1")
    def test_ring_cudf(self):
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
                .nodes(cudf.DataFrame({
                    'n': ['a', 'b', 'c', 'd', 'm', 'd1', 'd2'],

                    't': pd.Series([
                        '2015-01-16 18:39:37',
                        '2015-01-29 02:15:35',
                        '2014-12-30 18:59:20',
                        '2015-01-20 08:12:27',
                        '2014-11-22 19:47:15',
                        '2014-11-21 14:38:07',
                        '2014-11-20 15:28:12'
                    ],
                    dtype='datetime64[ns]')
                }))
                .time_ring_layout('t')
            )
        assert isinstance(g._nodes, cudf.DataFrame)
        assert isinstance(g._edges, cudf.DataFrame)
        assert 'x' in g._nodes
        assert 'y' in g._nodes
        assert not g._nodes.x.isna().any()
        assert not g._nodes.y.isna().any()


def _axis_rows(g) -> List[dict]:
    return g._complex_encodings['node_encodings']['default']['pointAxisEncoding']['rows']


def _g_time(values):
    return (
        LGFull()
        .edges(pd.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]}), 's', 'd')
        .nodes(pd.DataFrame({
            'id': list(range(len(values))),
            't': pd.Series(pd.to_datetime(pd.Series(values, dtype='object')), dtype='datetime64[ns]')
        }), 'id')
    )


class Test_time_ring_degenerate(NoAuthTestCase):
    """#1964: NaT gets no position instead of a radius far outside max_r; time_unit is honored"""

    def test_nat_has_no_position(self):
        for reverse in [False, True]:
            g = _g_time(['2020-01-01', None, '2022-12-01']).time_ring_layout('t', reverse=reverse)
            r = g._nodes['r']
            assert np.isnan(r[1]) and np.isnan(g._nodes['x'][1]) and np.isnan(g._nodes['y'][1])
            present = r[[0, 2]]
            assert present.notna().all()
            assert (present >= MIN_R_DEFAULT).all() and (present <= MAX_R_DEFAULT).all()

    def test_nat_does_not_move_present_nodes(self):
        g_nat = _g_time(['2020-01-01', None, '2022-12-01']).time_ring_layout('t')
        g_ok = _g_time(['2020-01-01', '2022-12-01']).time_ring_layout('t')
        assert g_nat._nodes['r'][[0, 2]].tolist() == g_ok._nodes['r'].tolist()
        assert _axis_rows(g_nat) == _axis_rows(g_ok)

    def test_constant_time_unchanged(self):
        g = _g_time(['2020-01-01'] * 3).time_ring_layout('t')
        assert (g._nodes['r'] == MIN_R_DEFAULT).all()
        g_rev = _g_time(['2020-01-01'] * 3).time_ring_layout('t', reverse=True)
        assert (g_rev._nodes['r'] == MAX_R_DEFAULT).all()

    def test_all_nat(self):
        g = _g_time([None, None, None]).time_ring_layout('t')
        assert g._nodes['r'].isna().all()
        assert g._nodes['x'].isna().all()
        assert _axis_rows(g) == []

    def test_empty_nodes(self):
        g = _g_time([]).time_ring_layout('t')
        assert len(g._nodes) == 0
        assert _axis_rows(g) == []

    def test_all_nat_with_explicit_bounds(self):
        g = _g_time([None, None, None]).time_ring_layout('t', time_start='2020-01-01', time_end='2021-01-01')
        assert g._nodes['r'].isna().all()
        assert len(_axis_rows(g)) > 0

    def test_find_round_bin_width_valid_time_unit(self):
        three_days = np.timedelta64(3, 'D')
        for unit in [None, 's', 'm', 'h', 'D', 'W', 'M', 'Y', 'C']:
            assert find_round_bin_width(three_days, time_unit=unit)[0] == 'W'

    def test_unknown_time_unit(self):
        g = _g_time(['2020-01-01', '2020-02-01', '2021-01-01'])
        with pytest.raises(ValueError, match='time_unit'):
            g.time_ring_layout('t', time_unit='d')
        with pytest.raises(ValueError, match='time_unit'):
            g.time_ring_layout('t', time_unit='d', num_rings=3)
        with pytest.raises(ValueError, match='time_unit'):
            find_round_bin_width(np.timedelta64(3, 'D'), time_unit='d')
