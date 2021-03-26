# -*- coding: utf-8 -*-

import datetime as dt, logging, numpy as np, os, pandas as pd, pyarrow as pa, pytest
from common import NoAuthTestCase

from graphistry.pygraphistry import PyGraphistry 
from graphistry.Engine import Engine, DataframeLike
from graphistry.hyper_dask import HyperBindings, hypergraph
from graphistry.tests.test_hypergraph import triangleNodes, assertFrameEqual, hyper_df, squareEvil
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def make_cluster_client():
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster()
    client = Client(cluster)

    return cluster, client


# ###


def assertFrameEqualCudf(df1, df2):
    import cudf
    if isinstance(df1, cudf.DataFrame):
        df1 = df1.to_pandas()
    if isinstance(df2, cudf.DataFrame):
        df2 = df2.to_pandas()
    return assertFrameEqual(df1, df2, check_index_type=False)

def assertFrameEqualDask(df1: DataframeLike, df2: DataframeLike):
    import dask.dataframe as dd
    if isinstance(df1, dd.DataFrame):
        df1 = df1.compute()
    if isinstance(df2, dd.DataFrame):
        df2 = df2.compute()
    return assertFrameEqual(
        df1.reset_index(drop=True).sort_values(by=sorted(df1.columns)),
        df2.reset_index(drop=True).sort_values(by=sorted(df2.columns)),
        check_index_type=False)

def assertFrameEqualDaskCudf(df1: DataframeLike, df2: DataframeLike):
    import cudf, dask_cudf
    if isinstance(df1, dask_cudf.DataFrame):
        df1 = df1.compute()
    if isinstance(df2, dask_cudf.DataFrame):
        df2 = df2.compute()
    return assertFrameEqualCudf(
        df1.reset_index(drop=True).sort_values(by=sorted(df1.columns)),
        df2.reset_index(drop=True).sort_values(by=sorted(df2.columns)))


squareEvil_gdf_friendly = pd.DataFrame({
    'src': [0,1,2,3],
    'dst': [1,2,3,0],
    'colors': [1, 1, 2, 2],
    #'list_int': [ [1], [2, 3], [4], []],
    #'list_str': [ ['x'], ['1', '2'], ['y'], []],
    #'list_bool': [ [True], [True, False], [False], []],
    #'list_date_str': [ ['2018-01-01 00:00:00'], ['2018-01-02 00:00:00', '2018-01-03 00:00:00'], ['2018-01-05 00:00:00'], []],
    #'list_date': [ [pd.Timestamp('2018-01-05')], [pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05')], [], []],
    #'list_mixed': [ [1], ['1', '2'], [False, None], []],
    'bool': [True, False, True, True],
    'char': ['a', 'b', 'c', 'd'],
    'str': ['a', 'b', 'c', 'd'],
    'ustr': [u'a', u'b', u'c', u'd'],
    'emoji': ['😋', '😋😋', '😋', '😋'],
    'int': [0, 1, 2, 3],
    'num': [0.5, 1.5, 2.5, 3.5],
    'date_str': ['2018-01-01 00:00:00', '2018-01-02 00:00:00', '2018-01-03 00:00:00', '2018-01-05 00:00:00'],
    
    'date': [dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1)],
    'time': [pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05')],
    
    'delta': [pd.Timedelta('1 day'), pd.Timedelta('1 day'), pd.Timedelta('1 day'), pd.Timedelta('1 day')]
})

squareEvil_dgdf_friendly = pd.DataFrame({
    'src': [0,1,2,3],
    'dst': [1,2,3,0],
    'colors': [1, 1, 2, 2],
    #'list_int': [ [1], [2, 3], [4], []],
    #'list_str': [ ['x'], ['1', '2'], ['y'], []],
    #'list_bool': [ [True], [True, False], [False], []],
    #'list_date_str': [ ['2018-01-01 00:00:00'], ['2018-01-02 00:00:00', '2018-01-03 00:00:00'], ['2018-01-05 00:00:00'], []],
    #'list_date': [ [pd.Timestamp('2018-01-05')], [pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05')], [], []],
    #'list_mixed': [ [1], ['1', '2'], [False, None], []],
    'bool': [True, False, True, True],
    'char': ['a', 'b', 'c', 'd'],
    'str': ['a', 'b', 'c', 'd'],
    'ustr': [u'a', u'b', u'c', u'd'],
    'emoji': ['😋', '😋😋', '😋', '😋'],
    'int': [0, 1, 2, 3],
    'num': [0.5, 1.5, 2.5, 3.5],
    'date_str': ['2018-01-01 00:00:00', '2018-01-02 00:00:00', '2018-01-03 00:00:00', '2018-01-05 00:00:00'],
    
    'date': [dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1)],
    'time': [pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05')],
    
    #'delta': [pd.Timedelta('1 day'), pd.Timedelta('1 day'), pd.Timedelta('1 day'), pd.Timedelta('1 day')]
})

def hyper_gdf():
    try:
        import cudf
        hyper2_df = hyper_df.assign(cc=hyper_df['cc'].astype(str))
        hyper2_gdf = cudf.DataFrame.from_pandas(hyper2_df)
        logger.debug('hyper2_gdf :: %s', hyper2_gdf.dtypes)
        return hyper2_gdf
    except Exception as e:
        logger.error('Failed to make hyper_gdf fixture..', exc_info=True)
        raise e


# ###


def test_HyperBindings_mt():
    hb = HyperBindings()
    assert hb.title == 'nodeTitle'
    assert hb.skip == []

def test_HyperBindings_override():
    hb = HyperBindings(NODETYPE='abc')
    assert hb.node_type == 'abc'


# ###


@pytest.mark.skipif(
    not ('TEST_PANDAS' in os.environ and os.environ['TEST_PANDAS'] == '1'),
    reason='pandas tests need TEST_PANDAS=1')
class TestHypergraphPandas(NoAuthTestCase):

  
    def test_hyperedges(self):

        h = hypergraph(PyGraphistry.bind(), triangleNodes, verbose=False)
        
        edges = pd.DataFrame({
            'a1': [1, 2, 3] * 4,
            'a2': ['red', 'blue', 'green'] * 4,
            'id': ['a', 'b', 'c'] * 4,
            '🙈': ['æski ēˈmōjē', '😋', 's'] * 4,
            'edgeType': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'id', 'id', 'id', '🙈', '🙈', '🙈'],
            'attribID': [
                'a1::1', 'a1::2', 'a1::3', 
                'a2::red', 'a2::blue', 'a2::green',                 
                'id::a', 'id::b', 'id::c',
                '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s'],
            'EventID': ['EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2']})

        logger.debug('EDGES: %s', h.edges)
        assertFrameEqual(h.edges, edges)
        for (k, v) in [('entities', 12), ('nodes', 15), ('edges', 12), ('events', 3)]:
            self.assertEqual(len(getattr(h, k)), v)

    def test_hyperedges_direct(self):

        h = hypergraph(PyGraphistry.bind(), hyper_df, verbose=False, direct=True)
        
        self.assertEqual(len(h.edges), 9)
        self.assertEqual(len(h.nodes), 9)

    def test_hyperedges_direct_categories(self):

        h = hypergraph(PyGraphistry.bind(), hyper_df, verbose=False, direct=True, opts={'CATEGORIES': {'n': ['aa', 'bb', 'cc']}})
        
        self.assertEqual(len(h.edges), 9)
        self.assertEqual(len(h.nodes), 6)

    def test_hyperedges_direct_manual_shaping(self):

        h1 = hypergraph(PyGraphistry.bind(), hyper_df, verbose=False, direct=True, opts={'EDGES': {'aa': ['cc'], 'cc': ['cc']}})
        self.assertEqual(len(h1.edges), 6)

        h2 = hypergraph(PyGraphistry.bind(), hyper_df, verbose=False, direct=True, opts={'EDGES': {'aa': ['cc', 'bb', 'aa'], 'cc': ['cc']}})
        self.assertEqual(len(h2.edges), 12)


    def test_drop_edge_attrs(self):
    
        h = hypergraph(PyGraphistry.bind(), triangleNodes, ['id', 'a1', '🙈'], verbose=False, drop_edge_attrs=True)

        edges = pd.DataFrame({
            'edgeType': ['a1', 'a1', 'a1', 'id', 'id', 'id', '🙈', '🙈', '🙈'],
            'attribID': [
                'a1::1', 'a1::2', 'a1::3', 
                'id::a', 'id::b', 'id::c',
                '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s'],
            'EventID': ['EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2']})


        assertFrameEqual(h.edges, edges)
        for (k, v) in [('entities', 9), ('nodes', 12), ('edges', 9), ('events', 3)]:
            logger.debug('testing: %s = %s', k, getattr(h, k))
            self.assertEqual(len(getattr(h, k)), v)

    def test_drop_edge_attrs_direct(self):
        
        h = hypergraph(PyGraphistry.bind(), triangleNodes,
            ['id', 'a1', '🙈'],
            verbose=False, direct=True, drop_edge_attrs=True,
            opts = {
                'EDGES': {
                    'id': ['a1'],
                    'a1': ['🙈']
                }
            })

        logger.debug('h.nodes: %s', h.graph._nodes)
        logger.debug('h.edges: %s', h.graph._edges)

        edges = pd.DataFrame({
            'edgeType': ['a1::🙈', 'a1::🙈', 'a1::🙈', 'id::a1', 'id::a1', 'id::a1'],
            'src': [
                'a1::1', 'a1::2', 'a1::3',
                'id::a', 'id::b', 'id::c'],
            'dst': [
                '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s',
                'a1::1', 'a1::2', 'a1::3'],
            'EventID': [
                'EventID::0', 'EventID::1', 'EventID::2',
                'EventID::0', 'EventID::1', 'EventID::2']})

        assertFrameEqual(h.edges, edges)
        for (k, v) in [('entities', 9), ('nodes', 9), ('edges', 6), ('events', 0)]:
            logger.error('testing: %s', k)
            logger.error('actual: %s', getattr(h,k))
            self.assertEqual(len(getattr(h,k)), v)


    def test_drop_na_hyper(self):

        df = pd.DataFrame({
            'a': ['a', None, 'c'],
            'i': [1, 2, None]
        })

        hg = hypergraph(PyGraphistry.bind(), df, drop_na=True)

        assert len(hg.graph._nodes) == 7
        assert len(hg.graph._edges) == 4

    def test_drop_na_direct(self):

        df = pd.DataFrame({
            'a': ['a', None, 'a'],
            'i': [1, 1, None]
        })

        hg = hypergraph(PyGraphistry.bind(), df, drop_na=True, direct=True)

        assert len(hg.graph._nodes) == 2
        assert len(hg.graph._edges) == 1

    def test_skip_na_hyperedge(self):
    
        nans_df = pd.DataFrame({
          'x': ['a', 'b', 'c'],
          'y': ['aa', None, 'cc']
        })
        expected_hits = ['a', 'b', 'c', 'aa', 'cc']

        skip_attr_h_edges = hypergraph(PyGraphistry.bind(), nans_df, drop_edge_attrs=True).edges
        self.assertEqual(len(skip_attr_h_edges), len(expected_hits))

        default_h_edges = hypergraph(PyGraphistry.bind(), nans_df).edges
        self.assertEqual(len(default_h_edges), len(expected_hits))

    def test_hyper_evil(self):
        hypergraph(PyGraphistry.bind(), squareEvil)

    def test_hyper_to_pa_vanilla(self):

        df = pd.DataFrame({
            'x': ['a', 'b', 'c'],
            'y': ['d', 'e', 'f']
        })

        hg = hypergraph(PyGraphistry.bind(), df)
        nodes_arr = pa.Table.from_pandas(hg.graph._nodes)
        assert len(nodes_arr) == 9
        edges_err = pa.Table.from_pandas(hg.graph._edges)
        assert len(edges_err) == 6

    def test_hyper_to_pa_mixed(self):

        df = pd.DataFrame({
            'x': ['a', 'b', 'c'],
            'y': [1, 2, 3]
        })

        hg = hypergraph(PyGraphistry.bind(), df)
        nodes_arr = pa.Table.from_pandas(hg.graph._nodes)
        assert len(nodes_arr) == 9
        edges_err = pa.Table.from_pandas(hg.graph._edges)
        assert len(edges_err) == 6

    def test_hyper_to_pa_na(self):

        df = pd.DataFrame({
            'x': ['a', None, 'c'],
            'y': [1, 2, None]
        })

        hg = hypergraph(PyGraphistry.bind(), df, drop_na=False)
        logger.debug('nodes :: %s => %s', hg.graph._nodes.dtypes, hg.graph._nodes)
        nodes_arr = pa.Table.from_pandas(hg.graph._nodes)
        assert len(hg.graph._nodes) == 9
        assert len(nodes_arr) == 9
        edges_err = pa.Table.from_pandas(hg.graph._edges)
        assert len(hg.graph._edges) == 6
        assert len(edges_err) == 6

    def test_hyper_to_pa_all(self):
        hg = hypergraph(PyGraphistry.bind(), triangleNodes, ['id', 'a1', '🙈'])
        nodes_arr = pa.Table.from_pandas(hg.graph._nodes)
        assert len(hg.graph._nodes) == 12
        assert len(nodes_arr) == 12
        edges_err = pa.Table.from_pandas(hg.graph._edges)
        assert len(hg.graph._edges) == 9
        assert len(edges_err) == 9

    def test_hyper_to_pa_all_direct(self):
        hg = hypergraph(PyGraphistry.bind(), triangleNodes, ['id', 'a1', '🙈'], direct=True)
        nodes_arr = pa.Table.from_pandas(hg.graph._nodes)
        assert len(hg.graph._nodes) == 9
        assert len(nodes_arr) == 9
        edges_err = pa.Table.from_pandas(hg.graph._edges)
        assert len(hg.graph._edges) == 9
        assert len(edges_err) == 9


@pytest.mark.skipif(
    not ('TEST_CUDF' in os.environ and os.environ['TEST_CUDF'] == '1'),
    reason='cudf tests need TEST_CUDF=1')
class TestHypergraphCudf(NoAuthTestCase):

  
    def test_hyperedges(self):
        import cudf

        h = hypergraph(PyGraphistry.bind(), cudf.DataFrame.from_pandas(triangleNodes), verbose=False, engine=Engine.CUDF)
        
        edges = cudf.DataFrame({
            'a1': [1, 2, 3] * 4,
            'a2': ['red', 'blue', 'green'] * 4,
            'id': ['a', 'b', 'c'] * 4,
            '🙈': ['æski ēˈmōjē', '😋', 's'] * 4,
            'edgeType': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'id', 'id', 'id', '🙈', '🙈', '🙈'],
            'attribID': [
                'a1::1', 'a1::2', 'a1::3', 
                'a2::red', 'a2::blue', 'a2::green',                 
                'id::a', 'id::b', 'id::c',
                '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s'],
            'EventID': ['EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2']})

        assertFrameEqualCudf(h.edges, edges)
        for (k, v) in [('entities', 12), ('nodes', 15), ('edges', 12), ('events', 3)]:
            self.assertEqual(len(getattr(h, k)), v)

    def test_hyperedges_direct(self):
        import cudf

        h = hypergraph(PyGraphistry.bind(), hyper_gdf(), verbose=False, direct=True, engine=Engine.CUDF)
        
        self.assertEqual(len(h.edges), 9)
        self.assertEqual(len(h.nodes), 9)

    def test_hyperedges_direct_categories(self):
        import cudf

        h = hypergraph(
            PyGraphistry.bind(), hyper_gdf(),
            verbose=False, direct=True, opts={'CATEGORIES': {'n': ['aa', 'bb', 'cc']}}, engine=Engine.CUDF)
        
        self.assertEqual(len(h.edges), 9)
        self.assertEqual(len(h.nodes), 6)

    def test_hyperedges_direct_manual_shaping(self):
        import cudf

        h1 = hypergraph(
            PyGraphistry.bind(), hyper_gdf(),
            verbose=False, direct=True, opts={'EDGES': {'aa': ['cc'], 'cc': ['cc']}}, engine=Engine.CUDF)
        self.assertEqual(len(h1.edges), 6)

        h2 = hypergraph(
            PyGraphistry.bind(), hyper_gdf(),
            verbose=False, direct=True, opts={'EDGES': {'aa': ['cc', 'bb', 'aa'], 'cc': ['cc']}}, engine=Engine.CUDF)
        self.assertEqual(len(h2.edges), 12)


    def test_drop_edge_attrs(self):
        import cudf
    
        h = hypergraph(
            PyGraphistry.bind(), cudf.DataFrame.from_pandas(triangleNodes), ['id', 'a1', '🙈'],
            verbose=False, drop_edge_attrs=True, engine=Engine.CUDF)

        edges = cudf.DataFrame({
            'edgeType': ['a1', 'a1', 'a1', 'id', 'id', 'id', '🙈', '🙈', '🙈'],
            'attribID': [
                'a1::1', 'a1::2', 'a1::3', 
                'id::a', 'id::b', 'id::c',
                '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s'],
            'EventID': ['EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2']})


        assertFrameEqualCudf(h.edges, edges)
        for (k, v) in [('entities', 9), ('nodes', 12), ('edges', 9), ('events', 3)]:
            logger.debug('testing: %s = %s', k, getattr(h, k))
            self.assertEqual(len(getattr(h, k)), v)

    def test_drop_edge_attrs_direct(self):
        import cudf
        
        h = hypergraph(
            PyGraphistry.bind(), cudf.DataFrame.from_pandas(triangleNodes),
            ['id', 'a1', '🙈'],
            verbose=False, direct=True, drop_edge_attrs=True,
            opts = {
                'EDGES': {
                    'id': ['a1'],
                    'a1': ['🙈']
                }
            },
            engine=Engine.CUDF)

        logger.debug('h.nodes: %s', h.graph._nodes)
        logger.debug('h.edges: %s', h.graph._edges)

        edges = cudf.DataFrame({
            'edgeType': ['a1::🙈', 'a1::🙈', 'a1::🙈', 'id::a1', 'id::a1', 'id::a1'],
            'src': [
                'a1::1', 'a1::2', 'a1::3',
                'id::a', 'id::b', 'id::c'],
            'dst': [
                '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s',
                'a1::1', 'a1::2', 'a1::3'],
            'EventID': [
                'EventID::0', 'EventID::1', 'EventID::2',
                'EventID::0', 'EventID::1', 'EventID::2']})

        assertFrameEqualCudf(h.edges, edges)
        for (k, v) in [('entities', 9), ('nodes', 9), ('edges', 6), ('events', 0)]:
            logger.error('testing: %s', k)
            logger.error('actual: %s', getattr(h,k))
            self.assertEqual(len(getattr(h,k)), v)


    def test_drop_na_hyper(self):
        import cudf

        df = cudf.DataFrame({
            'a': ['a', None, 'c'],
            'i': [1, 2, None]
        })

        hg = hypergraph(PyGraphistry.bind(), df, drop_na=True, engine=Engine.CUDF)

        assert len(hg.graph._nodes) == 7
        assert len(hg.graph._edges) == 4

    @pytest.mark.xfail(reason='https://github.com/rapidsai/cudf/issues/7735')
    def test_drop_na_direct(self):
        import cudf

        df = cudf.DataFrame({
            'a': ['a', None, 'a'],
            'i': [1, 1, None]
        })

        hg = hypergraph(PyGraphistry.bind(), df, drop_na=True, direct=True, engine=Engine.CUDF)

        assert len(hg.graph._nodes) == 2
        assert len(hg.graph._edges) == 1

    def test_drop_nan_direct(self):
        import cudf

        df = cudf.DataFrame({
            'a': ['a', np.nan, 'a'],
            'i': [1, 1, np.nan]
        })

        hg = hypergraph(PyGraphistry.bind(), df, drop_na=True, direct=True, engine=Engine.CUDF)

        assert len(hg.graph._nodes) == 2
        assert len(hg.graph._edges) == 1


    def test_skip_na_hyperedge(self):
        import cudf
    
        nans_df = cudf.DataFrame({
          'x': ['a', 'b', 'c'],
          'y': ['aa', None, 'cc']
        })
        expected_hits = ['a', 'b', 'c', 'aa', 'cc']

        skip_attr_h_edges = hypergraph(
            PyGraphistry.bind(), nans_df, drop_edge_attrs=True,
            engine=Engine.CUDF).edges
        self.assertEqual(len(skip_attr_h_edges), len(expected_hits))

        default_h_edges = hypergraph(
            PyGraphistry.bind(), nans_df,
            engine=Engine.CUDF).edges
        self.assertEqual(len(default_h_edges), len(expected_hits))

    def test_hyper_evil(self):
        import cudf

        hypergraph(
            PyGraphistry.bind(), cudf.DataFrame.from_pandas(squareEvil_gdf_friendly),
            engine=Engine.CUDF)

    def test_hyper_to_pa_vanilla(self):
        import cudf

        df = cudf.DataFrame({
            'x': ['a', 'b', 'c'],
            'y': ['d', 'e', 'f']
        })

        hg = hypergraph(PyGraphistry.bind(), df, engine=Engine.CUDF)
        nodes_arr = hg.graph._nodes.to_arrow()
        assert len(nodes_arr) == 9
        edges_err = hg.graph._edges.to_arrow()
        assert len(edges_err) == 6

    def test_hyper_to_pa_mixed(self):
        import cudf

        df = cudf.DataFrame({
            'x': ['a', 'b', 'c'],
            'y': [1, 2, 3]
        })

        hg = hypergraph(PyGraphistry.bind(), df, engine=Engine.CUDF)
        nodes_arr = hg.graph._nodes.to_arrow()
        assert len(nodes_arr) == 9
        edges_err = hg.graph._edges.to_arrow()
        assert len(edges_err) == 6

    def test_hyper_to_pa_na(self):
        import cudf

        df = cudf.DataFrame({
            'x': ['a', None, 'c'],
            'y': [1, 2, None]
        })

        hg = hypergraph(PyGraphistry.bind(), df, drop_na=False, engine=Engine.CUDF)
        nodes_arr = hg.graph._nodes.to_arrow()
        assert len(hg.graph._nodes) == 9
        assert len(nodes_arr) == 9
        edges_err = hg.graph._edges.to_arrow()
        assert len(hg.graph._edges) == 6
        assert len(edges_err) == 6

    def test_hyper_to_pa_all(self):
        import cudf
        hg = hypergraph(
            PyGraphistry.bind(), cudf.DataFrame.from_pandas(triangleNodes), ['id', 'a1', '🙈'],
            engine=Engine.CUDF)
        nodes_arr = hg.graph._nodes.to_arrow()
        assert len(hg.graph._nodes) == 12
        assert len(nodes_arr) == 12
        edges_err = hg.graph._edges.to_arrow()
        assert len(hg.graph._edges) == 9
        assert len(edges_err) == 9

    def test_hyper_to_pa_all_direct(self):
        import cudf
        hg = hypergraph(
            PyGraphistry.bind(), cudf.DataFrame.from_pandas(triangleNodes), ['id', 'a1', '🙈'],
            direct=True, engine=Engine.CUDF)
        nodes_arr = hg.graph._nodes.to_arrow()
        assert len(hg.graph._nodes) == 9
        assert len(nodes_arr) == 9
        edges_err = hg.graph._edges.to_arrow()
        assert len(hg.graph._edges) == 9
        assert len(edges_err) == 9

    def test_hyperedges_import(self):
        from graphistry.pygraphistry import hypergraph as hypergraph_public

        h = hypergraph_public(hyper_gdf(), verbose=False, direct=True, engine='cudf')

        self.assertEqual(len(h['edges']), 9)
        self.assertEqual(len(h['nodes']), 9)


@pytest.mark.skipif(
    not ('TEST_DASK' in os.environ and os.environ['TEST_DASK'] == '1'),
    reason='dask tests need TEST_DASK=1')
class TestHypergraphDask(NoAuthTestCase):


    def test_hyperedges(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):
            h = hypergraph(PyGraphistry.bind(), triangleNodes, verbose=False, engine=Engine.DASK, npartitions=2, debug=True)
            
            edges = pd.DataFrame({
                'a1': [1, 2, 3] * 4,
                'a2': ['red', 'blue', 'green'] * 4,
                'id': ['a', 'b', 'c'] * 4,
                '🙈': ['æski ēˈmōjē', '😋', 's'] * 4,
                'edgeType': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'id', 'id', 'id', '🙈', '🙈', '🙈'],
                'attribID': [
                    'a1::1', 'a1::2', 'a1::3', 
                    'a2::red', 'a2::blue', 'a2::green',                 
                    'id::a', 'id::b', 'id::c',
                    '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s'],
                'EventID': ['EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2']})

            assertFrameEqualDask(h.edges, edges)
            for (k, v) in [('entities', 12), ('nodes', 15), ('edges', 12), ('events', 3)]:
                self.assertEqual(len(getattr(h, k).compute()), v)

    def test_hyperedges_direct(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):
            h = hypergraph(PyGraphistry.bind(), hyper_df, verbose=False, direct=True, engine=Engine.DASK, npartitions=2, debug=True)
            
            self.assertEqual(len(h.edges.compute()), 9)
            self.assertEqual(len(h.nodes.compute()), 9)

    def test_hyperedges_direct_categories(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):

            h = hypergraph(
                PyGraphistry.bind(), hyper_df,
                verbose=False, direct=True, opts={'CATEGORIES': {'n': ['aa', 'bb', 'cc']}}, engine=Engine.DASK, npartitions=2, debug=True)
            
            self.assertEqual(len(h.edges.compute()), 9)
            self.assertEqual(len(h.nodes.compute()), 6)

    def test_hyperedges_direct_manual_shaping(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):

            h1 = hypergraph(
                PyGraphistry.bind(), hyper_df,
                verbose=False, direct=True, opts={'EDGES': {'aa': ['cc'], 'cc': ['cc']}}, engine=Engine.DASK, npartitions=2, debug=True)
            self.assertEqual(len(h1.edges.compute()), 6)

            h2 = hypergraph(
                PyGraphistry.bind(), hyper_df,
                verbose=False, direct=True, opts={'EDGES': {'aa': ['cc', 'bb', 'aa'], 'cc': ['cc']}}, engine=Engine.DASK, npartitions=2, debug=True)
            self.assertEqual(len(h2.edges.compute()), 12)


    def test_drop_edge_attrs(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):
    
            h = hypergraph(
                PyGraphistry.bind(), triangleNodes, ['id', 'a1', '🙈'],
                verbose=False, drop_edge_attrs=True, engine=Engine.DASK, npartitions=2, debug=True)

            edges = pd.DataFrame({
                'edgeType': ['a1', 'a1', 'a1', 'id', 'id', 'id', '🙈', '🙈', '🙈'],
                'attribID': [
                    'a1::1', 'a1::2', 'a1::3', 
                    'id::a', 'id::b', 'id::c',
                    '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s'],
                'EventID': ['EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2']})

            logger.debug('edges: %s', h.edges.compute())

            assertFrameEqualDask(h.edges, edges)
            for (k, v) in [('entities', 9), ('nodes', 12), ('edges', 9), ('events', 3)]:
                logger.debug('testing raw: %s = %s', k, getattr(h, k))
                logger.debug('actual: %s = %s', k, getattr(h, k).compute())
                self.assertEqual(len(getattr(h, k).compute()), v)

    def test_drop_edge_attrs_direct(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):
        
            h = hypergraph(
                PyGraphistry.bind(), triangleNodes,
                ['id', 'a1', '🙈'],
                verbose=False, direct=True, drop_edge_attrs=True,
                opts = {
                    'EDGES': {
                        'id': ['a1'],
                        'a1': ['🙈']
                    }
                },
                engine=Engine.DASK, npartitions=2, debug=True)

            logger.debug('h.nodes: %s', h.graph._nodes)
            logger.debug('h.edges: %s', h.graph._edges)

            edges = pd.DataFrame({
                'edgeType': ['a1::🙈', 'a1::🙈', 'a1::🙈', 'id::a1', 'id::a1', 'id::a1'],
                'src': [
                    'a1::1', 'a1::2', 'a1::3',
                    'id::a', 'id::b', 'id::c'],
                'dst': [
                    '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s',
                    'a1::1', 'a1::2', 'a1::3'],
                'EventID': [
                    'EventID::0', 'EventID::1', 'EventID::2',
                    'EventID::0', 'EventID::1', 'EventID::2']})

            assertFrameEqualDask(h.edges, edges)
            for (k, v) in [('entities', 9), ('nodes', 9), ('edges', 6), ('events', 0)]:
                logger.error('testing: %s', k)
                logger.error('actual: %s', getattr(h,k).compute())
                self.assertEqual(len(getattr(h,k).compute()), v)


    def test_drop_na_hyper(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):

            df = pd.DataFrame({
                'a': ['a', None, 'c'],
                'i': [1, 2, None]
            })

            hg = hypergraph(PyGraphistry.bind(), df, drop_na=True, engine=Engine.DASK, npartitions=2, debug=True)

            assert len(hg.graph._nodes.compute()) == 7
            assert len(hg.graph._edges.compute()) == 4

    def test_drop_na_direct(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):

            df = pd.DataFrame({
                'a': ['a', None, 'a'],
                'i': [1, 1, None]
            })

            hg = hypergraph(PyGraphistry.bind(), df, drop_na=True, direct=True, engine=Engine.DASK, npartitions=2, debug=True)

            assert len(hg.graph._nodes.compute()) == 2
            assert len(hg.graph._edges.compute()) == 1

    def test_skip_na_hyperedge(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):
    
            nans_df = pd.DataFrame({
                'x': ['a', 'b', 'c'],
                'y': ['aa', None, 'cc']
            })
            expected_hits = ['a', 'b', 'c', 'aa', 'cc']

            skip_attr_h_edges = hypergraph(
                PyGraphistry.bind(), nans_df, drop_edge_attrs=True,
                engine=Engine.DASK, npartitions=2, debug=True).edges
            self.assertEqual(len(skip_attr_h_edges.compute()), len(expected_hits))

            default_h_edges = hypergraph(
                PyGraphistry.bind(), nans_df,
                engine=Engine.DASK, npartitions=2, debug=True).edges
            self.assertEqual(len(default_h_edges.compute()), len(expected_hits))

    def test_hyper_evil(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):

            h = hypergraph(
                PyGraphistry.bind(), squareEvil_gdf_friendly,
                engine=Engine.DASK, npartitions=2, debug=True)
            h.nodes.compute()
            h.edges.compute()
            h.events.compute()
            h.entities.compute()

    def test_hyper_to_pa_vanilla(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):

            df = pd.DataFrame({
                'x': ['a', 'b', 'c'],
                'y': ['d', 'e', 'f']
            })

            hg = hypergraph(PyGraphistry.bind(), df, engine=Engine.DASK, npartitions=2, debug=True)
            nodes_arr = pa.Table.from_pandas(hg.graph._nodes.compute())
            assert len(nodes_arr) == 9
            edges_err = pa.Table.from_pandas(hg.graph._edges.compute())
            assert len(edges_err) == 6

    def test_hyper_to_pa_mixed(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):

            df = pd.DataFrame({
                'x': ['a', 'b', 'c'],
                'y': [1, 2, 3]
            })

            hg = hypergraph(PyGraphistry.bind(), df, engine=Engine.DASK, npartitions=2, debug=True)
            nodes_arr = pa.Table.from_pandas(hg.graph._nodes.compute())
            assert len(nodes_arr) == 9
            edges_err = pa.Table.from_pandas(hg.graph._edges.compute())
            assert len(edges_err) == 6

    def test_hyper_to_pa_na(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):

            df = pd.DataFrame({
                'x': ['a', None, 'c'],
                'y': [1, 2, None]
            })

            hg = hypergraph(PyGraphistry.bind(), df, drop_na=False, npartitions=2, engine=Engine.DASK, debug=True)
            nodes_arr = pa.Table.from_pandas(hg.graph._nodes.compute())
            assert len(hg.graph._nodes) == 9
            assert len(nodes_arr) == 9
            edges_err = pa.Table.from_pandas(hg.graph._edges.compute())
            assert len(hg.graph._edges) == 6
            assert len(edges_err) == 6

    def test_hyper_to_pa_all(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):
            hg = hypergraph(
                PyGraphistry.bind(), triangleNodes, ['id', 'a1', '🙈'],
                engine=Engine.DASK, npartitions=2, debug=True)
            nodes_arr = pa.Table.from_pandas(hg.graph._nodes.compute())
            assert len(hg.graph._nodes) == 12
            assert len(nodes_arr) == 12
            edges_err = pa.Table.from_pandas(hg.graph._edges.compute())
            assert len(hg.graph._edges) == 9
            assert len(edges_err) == 9

    def test_hyper_to_pa_all_direct(self):
        import dask
        from dask.distributed import Client

        with Client(processes=True):
            hg = hypergraph(
                PyGraphistry.bind(), triangleNodes, ['id', 'a1', '🙈'],
                direct=True, engine=Engine.DASK, npartitions=2, debug=True)
            nodes_arr = pa.Table.from_pandas(hg.graph._nodes.compute())
            assert len(hg.graph._nodes) == 9
            assert len(nodes_arr) == 9
            edges_err = pa.Table.from_pandas(hg.graph._edges.compute())
            assert len(hg.graph._edges) == 9
            assert len(edges_err) == 9

    def test_hyperedges_import(self):
        from graphistry.pygraphistry import hypergraph as hypergraph_public

        import dask
        from dask.distributed import Client

        with Client(processes=True):

            h = hypergraph_public(hyper_df, verbose=False, direct=True, engine='dask', npartitions=2)

            self.assertEqual(len(h['edges']), 9)
            self.assertEqual(len(h['nodes']), 9)


@pytest.mark.skipif(
    not ('TEST_DASK_CUDF' in os.environ and os.environ['TEST_DASK_CUDF'] == '1'),
    reason='dask tests need TEST_DASK_CUDF=1')
class TestHypergraphDaskCudf(NoAuthTestCase):

    def test_hyperedges(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                h = hypergraph(PyGraphistry.bind(), triangleNodes, verbose=False, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                edges = pd.DataFrame({
                    'a1': [1, 2, 3] * 4,
                    'a2': ['red', 'blue', 'green'] * 4,
                    'id': ['a', 'b', 'c'] * 4,
                    '🙈': ['æski ēˈmōjē', '😋', 's'] * 4,
                    'edgeType': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'id', 'id', 'id', '🙈', '🙈', '🙈'],
                    'attribID': [
                        'a1::1', 'a1::2', 'a1::3', 
                        'a2::red', 'a2::blue', 'a2::green',                 
                        'id::a', 'id::b', 'id::c',
                        '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s'],
                    'EventID': ['EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2']})
                assertFrameEqualDaskCudf(h.edges, edges)
                for (k, v) in [('entities', 12), ('nodes', 15), ('edges', 12), ('events', 3)]:
                    self.assertEqual(len(getattr(h, k).compute()), v)

    def test_hyperedges_direct(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                h = hypergraph(PyGraphistry.bind(), hyper_df, verbose=False, direct=True, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                
                self.assertEqual(len(h.edges.compute()), 9)
                self.assertEqual(len(h.nodes.compute()), 9)

    def test_hyperedges_direct_categories(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                h = hypergraph(
                    PyGraphistry.bind(), hyper_df,
                    verbose=False, direct=True, opts={'CATEGORIES': {'n': ['aa', 'bb', 'cc']}}, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                self.assertEqual(len(h.edges.compute()), 9)
                self.assertEqual(len(h.nodes.compute()), 6)

    def test_hyperedges_direct_manual_shaping(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                h1 = hypergraph(
                    PyGraphistry.bind(), hyper_df,
                    verbose=False, direct=True, opts={'EDGES': {'aa': ['cc'], 'cc': ['cc']}}, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                self.assertEqual(len(h1.edges.compute()), 6)
                h2 = hypergraph(
                    PyGraphistry.bind(), hyper_df,
                    verbose=False, direct=True, opts={'EDGES': {'aa': ['cc', 'bb', 'aa'], 'cc': ['cc']}}, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                self.assertEqual(len(h2.edges.compute()), 12)

    def test_drop_edge_attrs(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                h = hypergraph(
                    PyGraphistry.bind(), triangleNodes, ['id', 'a1', '🙈'],
                    verbose=False, drop_edge_attrs=True, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                edges = pd.DataFrame({
                    'edgeType': ['a1', 'a1', 'a1', 'id', 'id', 'id', '🙈', '🙈', '🙈'],
                    'attribID': [
                        'a1::1', 'a1::2', 'a1::3', 
                        'id::a', 'id::b', 'id::c',
                        '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s'],
                    'EventID': ['EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2']})
                logger.debug('edges: %s', h.edges.compute())
                assertFrameEqualDaskCudf(h.edges.reset_index(drop=True), edges)
                for (k, v) in [('entities', 9), ('nodes', 12), ('edges', 9), ('events', 3)]:
                    logger.debug('testing: %s = %s', k, getattr(h, k).compute())
                    self.assertEqual(len(getattr(h, k).compute()), v)

    def test_drop_edge_attrs_direct(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                h = hypergraph(
                    PyGraphistry.bind(), triangleNodes,
                    ['id', 'a1', '🙈'],
                    verbose=False, direct=True, drop_edge_attrs=True,
                    opts = {
                        'EDGES': {
                            'id': ['a1'],
                            'a1': ['🙈']
                        }
                    },
                    engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                logger.debug('h.nodes: %s', h.graph._nodes)
                logger.debug('h.edges: %s', h.graph._edges)
                edges = pd.DataFrame({
                    'edgeType': ['a1::🙈', 'a1::🙈', 'a1::🙈', 'id::a1', 'id::a1', 'id::a1'],
                    'src': [
                        'a1::1', 'a1::2', 'a1::3',
                        'id::a', 'id::b', 'id::c'],
                    'dst': [
                        '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s',
                        'a1::1', 'a1::2', 'a1::3'],
                    'EventID': [
                        'EventID::0', 'EventID::1', 'EventID::2',
                        'EventID::0', 'EventID::1', 'EventID::2']})
                assertFrameEqualDaskCudf(h.edges, edges)
                for (k, v) in [('entities', 9), ('nodes', 9), ('edges', 6), ('events', 0)]:
                    logger.error('testing: %s', k)
                    logger.error('actual: %s', getattr(h,k).compute())
                    self.assertEqual(len(getattr(h,k).compute()), v)

    @pytest.mark.xfail(reason='https://github.com/rapidsai/cudf/issues/7735')
    def test_drop_na_hyper(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                df = pd.DataFrame({
                    'a': ['a', None, 'c'],
                    'i': [1, 2, None]
                })
                hg = hypergraph(PyGraphistry.bind(), df, drop_na=True, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                assert len(hg.graph._nodes.compute()) == 7
                assert len(hg.graph._edges.compute()) == 4

    @pytest.mark.xfail(reason='https://github.com/rapidsai/cudf/issues/7735')
    def test_drop_nan_hyper(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                df = pd.DataFrame({
                    'a': ['a', np.nan, 'c'],
                    'i': [1, 2, np.nan]
                })
                hg = hypergraph(PyGraphistry.bind(), df, drop_na=True, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                assert len(hg.graph._nodes.compute()) == 7
                assert len(hg.graph._edges.compute()) == 4

    @pytest.mark.xfail(reason='https://github.com/rapidsai/cudf/issues/7735')
    def test_drop_na_direct(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                df = pd.DataFrame({
                    'a': ['a', None, 'a'],
                    'i': [1, 1, None]
                })
                hg = hypergraph(PyGraphistry.bind(), df, drop_na=True, direct=True, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                logger.debug('nodes: %s', hg.graph._nodes.compute())
                assert len(hg.graph._nodes.compute()) == 2
                assert len(hg.graph._edges.compute()) == 1

    @pytest.mark.xfail(reason='https://github.com/rapidsai/cudf/issues/7735')
    def test_drop_nan_direct(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                df = pd.DataFrame({
                    'a': ['a', np.nan, 'a'],
                    'i': [1, 1, np.nan]
                })
                hg = hypergraph(PyGraphistry.bind(), df, drop_na=True, direct=True, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                logger.debug('nodes: %s', hg.graph._nodes.compute())
                assert len(hg.graph._nodes.compute()) == 2
                assert len(hg.graph._edges.compute()) == 1

    @pytest.mark.xfail(reason='https://github.com/rapidsai/cudf/issues/7735')
    def test_skip_na_hyperedge(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                nans_df = pd.DataFrame({
                    'x': ['a', 'b', 'c'],
                    'y': ['aa', None, 'cc']
                })
                expected_hits = ['a', 'b', 'c', 'aa', 'cc']

                skip_attr_h_edges = hypergraph(
                    PyGraphistry.bind(), nans_df, drop_edge_attrs=True,
                    engine=Engine.DASK_CUDF, npartitions=2, debug=True).edges
                self.assertEqual(len(skip_attr_h_edges.compute()), len(expected_hits))
                default_h_edges = hypergraph(
                    PyGraphistry.bind(), nans_df,
                    engine=Engine.DASK_CUDF, npartitions=2, debug=True).edges
                self.assertEqual(len(default_h_edges.compute()), len(expected_hits))

    @pytest.mark.xfail(reason='https://github.com/rapidsai/cudf/issues/7735')
    def test_skip_nan_hyperedge(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                nans_df = pd.DataFrame({
                    'x': ['a', 'b', 'c'],
                    'y': ['aa', np.nan, 'cc']
                })
                expected_hits = ['a', 'b', 'c', 'aa', 'cc']

                skip_attr_h_edges = hypergraph(
                    PyGraphistry.bind(), nans_df, drop_edge_attrs=True,
                    engine=Engine.DASK_CUDF, npartitions=2, debug=True).edges
                self.assertEqual(len(skip_attr_h_edges.compute()), len(expected_hits))
                default_h_edges = hypergraph(
                    PyGraphistry.bind(), nans_df,
                    engine=Engine.DASK_CUDF, npartitions=2, debug=True).edges
                self.assertEqual(len(default_h_edges.compute()), len(expected_hits))

    def test_hyper_evil(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                h = hypergraph(
                    PyGraphistry.bind(), squareEvil_dgdf_friendly,
                    engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                h.nodes.compute()
                h.edges.compute()
                h.events.compute()
                h.entities.compute()

    def test_hyper_to_pa_vanilla(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                df = pd.DataFrame({
                    'x': ['a', 'b', 'c'],
                    'y': ['d', 'e', 'f']
                })
                hg = hypergraph(PyGraphistry.bind(), df, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                nodes_arr = hg.graph._nodes.compute().to_arrow()
                assert len(nodes_arr) == 9
                edges_err = hg.graph._edges.compute().to_arrow()
                assert len(edges_err) == 6

    def test_hyper_to_pa_mixed(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                df = pd.DataFrame({
                    'x': ['a', 'b', 'c'],
                    'y': [1, 2, 3]
                })

                hg = hypergraph(PyGraphistry.bind(), df, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                nodes_arr = hg.graph._nodes.compute().to_arrow()
                assert len(nodes_arr) == 9
                edges_err = hg.graph._edges.compute().to_arrow()
                assert len(edges_err) == 6


    @pytest.mark.xfail(reason='https://github.com/rapidsai/cudf/issues/7735')
    def test_hyper_to_pa_na(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                df = pd.DataFrame({
                    'x': ['a', None, 'c'],
                    'y': [1, 2, None]
                })

                hg = hypergraph(PyGraphistry.bind(), df, drop_na=False, npartitions=2, engine=Engine.DASK_CUDF, debug=True)
                nodes_arr = hg.graph._nodes.compute().to_arrow()
                assert len(hg.graph._nodes) == 9
                assert len(nodes_arr) == 9
                edges_err = hg.graph._edges.compute().to_arrow()
                assert len(hg.graph._edges) == 6
                assert len(edges_err) == 6
    @pytest.mark.xfail(reason='https://github.com/rapidsai/cudf/issues/7735')
    def test_hyper_to_pa_nan(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                df = pd.DataFrame({
                    'x': ['a', np.nan, 'c'],
                    'y': [1, 2, np.nan]
                })

                hg = hypergraph(PyGraphistry.bind(), df, drop_na=False, npartitions=2, engine=Engine.DASK_CUDF, debug=True)
                nodes_arr = hg.graph._nodes.compute().to_arrow()
                assert len(hg.graph._nodes) == 9
                assert len(nodes_arr) == 9
                edges_err = hg.graph._edges.compute().to_arrow()
                assert len(hg.graph._edges) == 6
                assert len(edges_err) == 6

    def test_hyper_to_pa_all(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                hg = hypergraph(
                    PyGraphistry.bind(), triangleNodes, ['id', 'a1', '🙈'],
                    engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                nodes_arr = hg.graph._nodes.compute().to_arrow()
                assert len(hg.graph._nodes) == 12
                assert len(nodes_arr) == 12
                edges_err = hg.graph._edges.compute().to_arrow()
                assert len(hg.graph._edges) == 9
                assert len(edges_err) == 9

    def test_hyper_to_pa_all_direct(self):
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                hg = hypergraph(
                    PyGraphistry.bind(), triangleNodes, ['id', 'a1', '🙈'],
                    direct=True, engine=Engine.DASK_CUDF, npartitions=2, debug=True)
                nodes_arr = hg.graph._nodes.compute().to_arrow()
                assert len(hg.graph._nodes) == 9
                assert len(nodes_arr) == 9
                edges_err = hg.graph._edges.compute().to_arrow()
                assert len(hg.graph._edges) == 9
                assert len(edges_err) == 9

    def test_hyperedges_import(self):
        from graphistry.pygraphistry import hypergraph as hypergraph_public
        cluster, client = make_cluster_client()
        with cluster:
            with client:
                h = hypergraph_public(hyper_df, verbose=False, direct=True, engine='dask_cudf', npartitions=2)
                self.assertEqual(len(h['edges']), 9)
                self.assertEqual(len(h['nodes']), 9)
