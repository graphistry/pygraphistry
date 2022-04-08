# -*- coding: utf-8 -*- 

import copy, datetime as dt, graphistry, os, pandas as pd, pyarrow as pa, pytest

from common import NoAuthTestCase
from mock import patch
from graphistry.tests.test_hyper_dask import assertFrameEqualDask

maybe_cudf = None
if 'TEST_CUDF' in os.environ and os.environ['TEST_CUDF'] == '1':
    import cudf
    maybe_cudf = cudf


triangleEdges = pd.DataFrame({'src': ['a', 'b', 'c'], 'dst': ['b', 'c', 'a']})
triangleNodes = pd.DataFrame({'id': ['a', 'b', 'c'], 'a1': [1, 2, 3], 'a2': ['red', 'blue', 'green']})
triangleNodesRich = pd.DataFrame({
    'id': ['a', 'b', 'c'], 
    'a1': [1, 2, 3], 
    'a2': ['red', 'blue', 'green'],
    'a3': [True, False, False],
    'a4': [0.5, 1.5, 1000.3],
    'a5': [dt.datetime.fromtimestamp(x) for x in [1440643875, 1440644191, 1440645638]],    
    'a6': [u'æski ēˈmōjē', u'😋', 's']
})

squareEvil = pd.DataFrame({
    'src': [0,1,2,3],
    'dst': [1,2,3,0],
    'colors': [1, 1, 2, 2],
    'list_int': [ [1], [2, 3], [4], []],
    'list_str': [ ['x'], ['1', '2'], ['y'], []],
    'list_bool': [ [True], [True, False], [False], []],
    'list_date_str': [ ['2018-01-01 00:00:00'], ['2018-01-02 00:00:00', '2018-01-03 00:00:00'], ['2018-01-05 00:00:00'], []],
    'list_date': [ [pd.Timestamp('2018-01-05')], [pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05')], [], []],
    'list_mixed': [ [1], ['1', '2'], [False, None], []],
    'bool': [True, False, True, True],
    'char': ['a', 'b', 'c', 'd'],
    'str': ['a', 'b', 'c', 'd'],
    'ustr': [u'a', u'b', u'c', u'd'],
    'emoji': ['😋', '😋😋', '😋', '😋'],
    'int': [0, 1, 2, 3],
    'num': [0.5, 1.5, 2.5, 3.5],
    'date_str': ['2018-01-01 00:00:00', '2018-01-02 00:00:00', '2018-01-03 00:00:00', '2018-01-05 00:00:00'],
    
    # API 1 BUG: Try with https://github.com/graphistry/pygraphistry/pull/126
    'date': [dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1)],
    'time': [pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05')],
    
    # API 2 BUG: Need timedelta in https://github.com/graphistry/pygraphistry/blob/master/graphistry/vgraph.py#L108
    'delta': [pd.Timedelta('1 day'), pd.Timedelta('1 day'), pd.Timedelta('1 day'), pd.Timedelta('1 day')]
})
for c in squareEvil.columns:
    try:
        squareEvil[c + '_cat'] = squareEvil[c].astype('category')
    except:
        # lists aren't categorical
        #print('could not make categorical', c)
        1



class Fake_Response(object):
    def raise_for_status(self):
        pass
    def json(self):
        return {'success': True, 'dataset': 'fakedatasetname', 'viztoken': 'faketoken'}


def assertFrameEqual(df1, df2, **kwds ):
    """ Assert that two dataframes are equal, ignoring ordering of columns"""

    from pandas.testing import assert_frame_equal
    return assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1), check_names=True, **kwds)


@patch('webbrowser.open')
@patch.object(graphistry.pygraphistry.PyGraphistry, '_etl1')
class TestPlotterBindings_API_1(NoAuthTestCase):

    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.register(api=1)


    def test_no_src_dst(self, mock_etl, mock_open):
        with self.assertRaises(ValueError):
            graphistry.bind().plot(triangleEdges)
        with self.assertRaises(ValueError):
            graphistry.bind(source='src').plot(triangleEdges)
        with self.assertRaises(ValueError):
            graphistry.bind(destination='dst').plot(triangleEdges)
        with self.assertRaises(ValueError):
            graphistry.bind(source='doesnotexist', destination='dst').plot(triangleEdges)


    def test_no_nodeid(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst')
        with self.assertRaises(ValueError):
            plotter.plot(triangleEdges, triangleNodes)


    def test_triangle_edges(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst')
        plotter.plot(triangleEdges)
        self.assertTrue(mock_etl.called)


    def test_bind_edges(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', edge_title='src')
        plotter.plot(triangleEdges)
        self.assertTrue(mock_etl.called)

    def test_bind_graph_short(self, mock_etl, mock_open):
        g = graphistry\
            .nodes(pd.DataFrame({'n': []}), 'n')\
            .edges(pd.DataFrame({'a': [], 'b': []}), 'a', 'b')
        assert g._node == 'n'
        assert g._source == 'a'
        assert g._destination == 'b'
        g2 = \
            graphistry \
            .bind(source='a', destination='b', node='n') \
            .nodes(pd.DataFrame({'n': []})) \
            .edges(pd.DataFrame({'a': [], 'b': []}))
        assert g2._node == 'n'
        assert g2._source == 'a'
        assert g2._destination == 'b'

    def test_bind_nodes(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', node='id', point_title='a2')
        plotter.plot(triangleEdges, triangleNodes)
        self.assertTrue(mock_etl.called)


    def test_bind_nodes_rich(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', node='id', point_title='a2')
        plotter.plot(triangleEdges, triangleNodesRich)
        self.assertTrue(mock_etl.called)

    def test_bind_edges_rich_2(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst')
        plotter.plot(squareEvil)
        self.assertTrue(mock_etl.called)

    def test_unknown_col_edges(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', edge_title='doesnotexist')
        with pytest.warns(RuntimeWarning):
            plotter.plot(triangleEdges)
        self.assertTrue(mock_etl.called)


    def test_unknown_col_nodes(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', node='id', point_title='doesnotexist')
        with pytest.warns(RuntimeWarning):
            plotter.plot(triangleEdges, triangleNodes)
        self.assertTrue(mock_etl.called)


    @patch('requests.post', return_value=Fake_Response())
    def test_empty_graph(self, mock_post, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst')
        with pytest.warns(RuntimeWarning):
            plotter.plot(pd.DataFrame({'src': [], 'dst': []}))
        self.assertTrue(mock_etl.called)

    def test_edges(self, mock_etl, mock_open):
        df = pd.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]})
        g = graphistry.edges(df)
        assert g._edges is df
        g = graphistry.edges(df, 's')
        assert g._source == 's'
        g = graphistry.edges(df, 's', 'd')
        assert g._destination == 'd'
        df2 = pd.DataFrame({'s': [2, 4, 6], 'd': [1, 2, 0]})
        g2 = graphistry.edges(lambda g: g.edges(df2)._edges)
        assert g2._edges is df2
        g3 = graphistry.edges(lambda g: g.edges(df2)._edges, source='s')
        assert g3._source == 's'
        g4 = graphistry.edges(
            (lambda g, s: g.edges(df2)._edges.assign(**{s: 1})),
            None, None, None,
            's2')
        assert (g4._edges.columns == ['s', 'd', 's2']).all()

    def test_nodes(self, mock_etl, mock_open):
        df = pd.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]})
        g = graphistry.nodes(df)
        assert g._nodes is df
        g = graphistry.nodes(df, 's')
        assert g._node == 's'
        df2 = pd.DataFrame({'s': [2, 4, 6], 'd': [1, 2, 0]})
        g2 = g.nodes(lambda g: g.nodes(df2)._nodes)
        assert g2._nodes is df2
        assert g2._node == 's'
        g3 = graphistry.nodes((lambda g, n: g.nodes(df2, n)._nodes.assign(**{n: 1})), None, 'n2')
        assert (g3._nodes.columns == ['s', 'd', 'n2']).all()
    
    def test_pipe(self, mock_etl, mock_open):
        df = pd.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]})
        g = graphistry.nodes(df, 's')
        df2 = pd.DataFrame({'s': [2, 4, 6], 'd': [1, 2, 0]})
        g2 = g.pipe((lambda g, s, d: g.edges(df2, s, d)), 's', 'd')
        assert g2._nodes is df
        assert g2._edges is df2
        assert g2._source == 's'
        assert g2._destination == 'd'
        df3 = pd.DataFrame({'s': [3, 6, 9], 'd': [1, 2, 0]})
        g3 = g2.pipe((lambda g: g.edges(df3)))
        assert g3._edges is df3


@patch('webbrowser.open')
@patch('graphistry.pygraphistry.PyGraphistry._etl2')
class TestPlotterBindings_API_2(NoAuthTestCase):

    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.register(api=2)


    def test_no_src_dst(self, mock_etl, mock_open):
        with self.assertRaises(ValueError):
            graphistry.bind().plot(triangleEdges)
        with self.assertRaises(ValueError):
            graphistry.bind(source='src').plot(triangleEdges)
        with self.assertRaises(ValueError):
            graphistry.bind(destination='dst').plot(triangleEdges)
        with self.assertRaises(ValueError):
            graphistry.bind(source='doesnotexist', destination='dst').plot(triangleEdges)


    def test_no_nodeid(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst')
        with self.assertRaises(ValueError):
            plotter.plot(triangleEdges, triangleNodes)


    def test_triangle_edges(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst')
        plotter.plot(triangleEdges)
        self.assertTrue(mock_etl.called)


    def test_bind_edges(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', edge_title='src')
        plotter.plot(triangleEdges)
        self.assertTrue(mock_etl.called)


    def test_bind_nodes(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', node='id', point_title='a2')
        plotter.plot(triangleEdges, triangleNodes)
        self.assertTrue(mock_etl.called)

    def test_bind_nodes_rich(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', node='id', point_title='a2')
        plotter.plot(triangleEdges, triangleNodesRich)
        self.assertTrue(mock_etl.called)

    def test_bind_edges_rich_2(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst')
        plotter.plot(squareEvil)
        self.assertTrue(mock_etl.called)

    def test_unknown_col_edges(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', edge_title='doesnotexist')
        with pytest.warns(RuntimeWarning):
            plotter.plot(triangleEdges)
        self.assertTrue(mock_etl.called)


    def test_unknown_col_nodes(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', node='id', point_title='doesnotexist')
        with pytest.warns(RuntimeWarning):
            plotter.plot(triangleEdges, triangleNodes)            
        self.assertTrue(mock_etl.called)


    def test_empty_graph(self, mock_etl, mock_open):
        plotter = graphistry.bind(source='src', destination='dst')
        with pytest.warns(RuntimeWarning):
            with self.assertRaises(ValueError):
                plotter.plot(pd.DataFrame([]))


@patch('webbrowser.open')
@patch.object(graphistry.pygraphistry.PyGraphistry, '_etl2')
class TestPlotterCallChaining(NoAuthTestCase):

    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.register(api=2)

    def test_bind_chain(self, mock_etl2, mock_open):
        plotter0 = graphistry.bind(source='caca').bind(destination='dst', source='src')
        plotter0.plot(triangleEdges)
        self.assertTrue(mock_etl2.called)


    def test_bind_edges_nodes(self, mock_etl2, mock_open):
        plotter0 = graphistry.bind(source='src').bind(destination='dst')
        plotter1 = plotter0.bind(node='id').bind(point_title='a2')
        plotter1.edges(triangleEdges).nodes(triangleNodes).plot()
        self.assertTrue(mock_etl2.called)


class TestPlotterConversions(NoAuthTestCase):

    @pytest.mark.xfail(raises=ModuleNotFoundError)
    def test_igraph2pandas(self):
        import igraph
        ig = igraph.Graph.Tree(4, 2)
        ig.vs['vattrib'] = 0
        ig.es['eattrib'] = 1
        (e, n) = graphistry.bind(source='src', destination='dst').igraph2pandas(ig)

        edges = pd.DataFrame({
            'dst': {0: 1, 1: 2, 2: 3},
            'src': {0: 0, 1: 0, 2: 1},
            'eattrib': {0: 1, 1: 1, 2: 1}
        })
        nodes = pd.DataFrame({
            '__nodeid__': {0: 0, 1: 1, 2: 2, 3: 3},
            'vattrib': {0: 0, 1: 0, 2: 0, 3: 0}
        })

        assertFrameEqual(e, edges)
        assertFrameEqual(n, nodes)

    @pytest.mark.xfail(raises=ModuleNotFoundError)
    def test_pandas2igraph(self):
        plotter = graphistry.bind(source='src', destination='dst', node='id')
        ig = plotter.pandas2igraph(triangleEdges)
        (e, n) = plotter.igraph2pandas(ig)
        assertFrameEqual(e, triangleEdges[['src', 'dst']])
        assertFrameEqual(n, triangleNodes[['id']])

    @pytest.mark.xfail(raises=ModuleNotFoundError)
    @pytest.mark.xfail(raises=ImportError)
    def test_networkx2igraph(self):
        import networkx as nx
        ng = nx.complete_graph(3)
        vs = [int(x) for x in nx.__version__.split('.')]
        x = vs[0]
        if x == 1:
            nx.set_node_attributes(ng, 'vattrib', 0)
            nx.set_edge_attributes(ng, 'eattrib', 1)
        else:
            nx.set_node_attributes(ng, 0, 'vattrib')
            nx.set_edge_attributes(ng, 1, 'eattrib')
        (e, n) = graphistry.bind(source='src', destination='dst').networkx2pandas(ng)

        edges = pd.DataFrame({
            'dst': {0: 1, 1: 2, 2: 2},
            'src': {0: 0, 1: 0, 2: 1},
            'eattrib': {0: 1, 1: 1, 2: 1}
        })
        nodes = pd.DataFrame({
            '__nodeid__': {0: 0, 1: 1, 2: 2},
            'vattrib': {0: 0, 1: 0, 2: 0}
        })

        assertFrameEqual(e, edges)
        assertFrameEqual(n, nodes)

    def test_cypher_unconfigured(self):
        with pytest.raises(ValueError):
            graphistry.bind().cypher("MATCH (a)-[b]-(c) RETURN a,b,c")


class TestPlotterNameBindings(NoAuthTestCase):

    def test_bind_name(self):
        plotter = graphistry.bind().name('n')
        assert plotter._name == 'n'

    def test_bind_description(self):
        plotter = graphistry.bind().description('d')
        assert plotter._description == 'd'


class TestPlotterPandasConversions(NoAuthTestCase):

    def test_table_to_pandas_from_none(self):
        plotter = graphistry.bind()
        assert plotter._table_to_pandas(None) is None

    def test_table_to_pandas_from_pandas(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({'x': []})
        assert isinstance(plotter._table_to_pandas(df), pd.DataFrame)

    def test_table_to_pandas_from_arrow(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({'x': []})
        arr = pa.Table.from_pandas(df)
        assert isinstance(plotter._table_to_pandas(arr), pd.DataFrame)

    @pytest.mark.skipif(
        not ('TEST_CUDF' in os.environ and os.environ['TEST_CUDF'] == '1'),
        reason='cudf tests need TEST_CUDF=1')
    def test_table_to_pandas_from_cudf(self):
        import cudf
        plotter = graphistry.bind()
        df = pd.DataFrame({'x': [1, 2, 3]})
        gdf = cudf.from_pandas(df)
        out = plotter._table_to_pandas(gdf)
        assert isinstance(out, pd.DataFrame)
        assertFrameEqual(out, df)

    @pytest.mark.skipif(
        not ('TEST_DASK' in os.environ and os.environ['TEST_DASK'] == '1'),
        reason='dask tests need TEST_DASK=1')
    def test_table_to_pandas_from_dask(self):
        import dask, dask.dataframe
        from dask.distributed import Client
        with Client(processes=True):
            plotter = graphistry.bind()
            df = pd.DataFrame({'x': [1, 2, 3]})
            ddf = dask.dataframe.from_pandas(df, npartitions=2)
            out = plotter._table_to_pandas(ddf)
            assert isinstance(out, pd.DataFrame)
            assertFrameEqual(out, df)

    @pytest.mark.skipif(
        not ('TEST_DASK_CUDF' in os.environ and os.environ['TEST_DASK_CUDF'] == '1'),
        reason='dask_cudf tests need TEST_DASK_CUDF=1')
    def test_table_to_pandas_from_dask_cudf(self):
        import cudf, dask_cudf
        plotter = graphistry.bind()
        df = pd.DataFrame({'x': [1, 2, 3]})
        gdf = cudf.from_pandas(df)
        dgdf = dask_cudf.from_pandas(gdf, npartitions=2)
        out = plotter._table_to_pandas(dgdf)
        assert isinstance(out, pd.DataFrame)
        assertFrameEqual(out, df)


class TestPlotterLabelInference(NoAuthTestCase):

    def test_infer_labels_predefined_title(self):
        g = (graphistry
            .nodes(pd.DataFrame({'x': [1,2], 'y': [1, 2]}))
            .bind(point_title='y'))
        g2 = g.infer_labels()
        assert g2._point_title == 'y'

    def test_infer_labels_predefined_label(self):
        g = (graphistry
            .nodes(pd.DataFrame({'x': [1,2], 'y': [1, 2]}))
            .bind(point_label='y'))
        g2 = g.infer_labels()
        assert g2._point_label == 'y'
    
    def test_infer_labels_infer_name_exact(self):
        g = graphistry.nodes(pd.DataFrame({'x': [1,2], 'name': [1, 2]}))
        g2 = g.infer_labels()
        assert g2._point_title == 'name'

    def test_infer_labels_infer_name_substr(self):
        g = graphistry.nodes(pd.DataFrame({'x': [1,2], 'thename': [1, 2]}))
        g2 = g.infer_labels()
        assert g2._point_title == 'thename'

    def test_infer_labels_infer_name_id_fallback(self):
        g = graphistry.nodes(pd.DataFrame({'x': [1,2], 'y': [1, 2]}), 'y')
        g2 = g.infer_labels()
        assert g2._point_title == 'y'

    def test_infer_labels_exn_unknown(self):
        g = graphistry.nodes(pd.DataFrame({'x': [1,2], 'y': [1, 2]}))
        with pytest.raises(ValueError):
            g.infer_labels()


class TestPlotterArrowConversions(NoAuthTestCase):

    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.pygraphistry.PyGraphistry.store_token_creds_in_memory(True)
        graphistry.pygraphistry.PyGraphistry.relogin = lambda: True
        graphistry.register(api=3)

    def test_table_to_arrow_from_none(self):
        plotter = graphistry.bind()
        assert plotter._table_to_arrow(None) is None

    def test_table_to_arrow_from_pandas(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({'x': []})
        assert isinstance(plotter._table_to_arrow(df), pa.Table)

    def test_table_to_arrow_from_arrow(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({'x': []})
        arr = pa.Table.from_pandas(df)
        assert isinstance(plotter._table_to_arrow(arr), pa.Table)

    def test_api3_plot_from_pandas(self):
        g = graphistry.edges(pd.DataFrame({'s': [0], 'd': [0]})).bind(source='s', destination='d')
        ds = g.plot(skip_upload=True)
        assert isinstance(ds.edges, pa.Table)

    @pytest.mark.xfail(raises=ModuleNotFoundError)
    def test_api3_plot_from_igraph(self):
        g = graphistry.bind(source='src', destination='dst', node='id')
        ig = g.pandas2igraph(triangleEdges)
        (e, n) = g.igraph2pandas(ig)
        g = g.edges(e).nodes(n)
        ds = g.plot(skip_upload=True)
        assert isinstance(ds.edges, pa.Table)
        assert isinstance(ds.nodes, pa.Table)

    def test_api3_plot_from_arrow(self):
        g = graphistry.edges(pa.Table.from_pandas(pd.DataFrame({'s': [0], 'd': [0]}))).bind(source='s', destination='d')
        ds = g.plot(skip_upload=True)
        assert isinstance(ds.edges, pa.Table)

    def test_api3_pdf_to_arrow_memoization(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({'x': [1]})
        arr1 = plotter._table_to_arrow(df)
        arr2 = plotter._table_to_arrow(df)
        assert isinstance(arr1, pa.Table)
        assert arr1 is arr2

        arr3 = plotter._table_to_arrow(pd.DataFrame({'x': [1]}))
        assert arr1 is arr3

    def test_api3_pdf_to_renamed_arrow_memoization(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({'x': [1]})
        arr1 = plotter._table_to_arrow(df)
        arr2 = plotter._table_to_arrow(df)
        assert isinstance(arr1, pa.Table)
        assert arr1 is arr2

        df2 = df.rename(columns={'x': 'y'})
        arr3 = plotter._table_to_arrow(df2)
        assert not (arr1 is arr3)

    def test_api3_pdf_to_arrow_memoization_forgets(self):
        plotter = graphistry.bind()
        df = pd.DataFrame({'x': [0]})
        arr1 = plotter._table_to_arrow(df)
        for i in range(1, 110):
            plotter._table_to_arrow(pd.DataFrame({'x': [i]}))

        assert not (arr1 is plotter._table_to_arrow(df))

    @pytest.mark.skipif(
        not ('TEST_CUDF' in os.environ and os.environ['TEST_CUDF'] == '1'),
        reason='cudf tests need TEST_CUDF=1')
    def test_api3_cudf_to_arrow_memoization(self):
        maybe_cudf = None
        try:
            import cudf
            maybe_cudf = cudf
        except ImportError:
            1
        if maybe_cudf is None:
            return

        plotter = graphistry.bind()
        df = maybe_cudf.DataFrame({'x': [1]})
        arr1 = plotter._table_to_arrow(df)
        arr2 = plotter._table_to_arrow(df)
        assert isinstance(arr1, pa.Table)
        assert arr1 is arr2

        arr3 = plotter._table_to_arrow(maybe_cudf.DataFrame({'x': [1]}))
        assert arr1 is arr3

    @pytest.mark.skipif(
        not ('TEST_CUDF' in os.environ and os.environ['TEST_CUDF'] == '1'),
        reason='cudf tests need TEST_CUDF=1')
    def test_api3_cudf_to_arrow_memoization_forgets(self):
        maybe_cudf = None
        try:
            import cudf
            maybe_cudf = cudf
        except ImportError:
            1
        if maybe_cudf is None:
            return

        plotter = graphistry.bind()
        df = maybe_cudf.DataFrame({'x': [0]})
        arr1 = plotter._table_to_arrow(df)
        for i in range(1, 110):
            plotter._table_to_arrow(maybe_cudf.DataFrame({'x': [i]}))

        assert not (arr1 is plotter._table_to_arrow(df))

    @pytest.mark.skipif(
        not ('TEST_DASK' in os.environ and os.environ['TEST_DASK'] == '1'),
        reason='dask tests need TEST_DASK=1')
    def test_api3_dask_to_arrow_memoization(self):
        import dask, dask.dataframe
        from dask.distributed import Client
        with Client(processes=True):
            plotter = graphistry.bind()
            df = pd.DataFrame({'x': [1, 2]})
            ddf = dask.dataframe.from_pandas(df, npartitions=2)
            arr1 = plotter._table_to_arrow(ddf)
            arr2 = plotter._table_to_arrow(ddf)
            assert isinstance(arr1, pa.Table)
            assert arr1 is arr2

            arr3 = plotter._table_to_arrow(dask.dataframe.from_pandas(pd.DataFrame({'x': [1, 2]}), npartitions=2))
            assert arr1 is arr3

    @pytest.mark.skipif(
        not ('TEST_DASK' in os.environ and os.environ['TEST_DASK'] == '1'),
        reason='dask tests need TEST_DASK=1')
    def test_api3_dask_to_arrow_memoization_forgets(self):
        import dask, dask.dataframe
        from dask.distributed import Client
        with Client(processes=True):
            plotter = graphistry.bind()
            df = pd.DataFrame({'x': [0]})
            ddf = dask.dataframe.from_pandas(df, npartitions=2)
            arr1 = plotter._table_to_arrow(ddf)
            for i in range(1, 110):
                ddf_i = dask.dataframe.from_pandas(pd.DataFrame({'x': [0, i]}), npartitions=2)
                plotter._table_to_arrow(ddf_i)
            assert not (arr1 is plotter._table_to_arrow(ddf))

    @pytest.mark.skipif(
        not ('TEST_DASK_CUDF' in os.environ and os.environ['TEST_DASK_CUDF'] == '1'),
        reason='dask_cudf tests need TEST_DASK_CUDF=1')
    def test_api3_dask_cudf_to_arrow_memoization(self):
        import cudf, dask_cudf
        plotter = graphistry.bind()
        gdf = cudf.DataFrame({'x': [1, 2]})
        dgdf = dask_cudf.from_cudf(gdf, npartitions=2)
        arr1 = plotter._table_to_arrow(dgdf)
        arr2 = plotter._table_to_arrow(dgdf)
        assert isinstance(arr1, pa.Table)
        assert arr1 is arr2

        arr3 = plotter._table_to_arrow(dask_cudf.from_cudf(cudf.DataFrame({'x': [1, 2]}), npartitions=2))
        assert arr1 is arr3

    @pytest.mark.skipif(
        not ('TEST_DASK_CUDF' in os.environ and os.environ['TEST_DASK_CUDF'] == '1'),
        reason='dask_cudf tests need TEST_DASK_CUDF=1')
    def test_api3_dask_cudf_to_arrow_memoization_forgets(self):
        import cudf, dask_cudf
        plotter = graphistry.bind()
        gdf = cudf.DataFrame({'x': [1, 0]})
        dgdf = dask_cudf.from_cudf(gdf, npartitions=2)
        arr1 = plotter._table_to_arrow(dgdf)
        for i in range(1, 110):
            dgdf_i = dask_cudf.from_cudf(cudf.DataFrame({'x': [0, i]}), npartitions=2)
            plotter._table_to_arrow(dgdf_i)
        assert not (arr1 is plotter._table_to_arrow(dgdf))


class TestPlotterStylesArrow(NoAuthTestCase):

    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.pygraphistry.PyGraphistry.store_token_creds_in_memory(True)
        graphistry.pygraphistry.PyGraphistry.relogin = lambda: True
        graphistry.register(api=3)

    def test_init(self):
        g = graphistry.bind()
        assert g._style is None

    def test_style_good(self):
        g = graphistry.bind()

        bg = {'color': 'red'}
        fg = {'blendMode': 1}
        logo = {'url': 'zzz'}
        page = {'title': 'zzz'}

        assert g.style()._style == {}

        g.style(fg={'blendMode': 'screen'})
        assert g.style()._style == {}

        assert g.style(bg=copy.deepcopy(bg))._style == {'bg': bg}
        assert g.style(bg={'color': 'blue'}).style(bg=copy.deepcopy(bg))._style == {'bg': bg}
        assert g.style(bg={'image': {'url': 'http://asdf.com/b.png'}}).style(bg=copy.deepcopy(bg))._style == {'bg': bg}
        assert g.style(bg=copy.deepcopy(bg), fg=copy.deepcopy(fg), logo=copy.deepcopy(logo), page=copy.deepcopy(page))._style == {
            'bg': bg, 'fg': fg, 'logo': logo, 'page': page
        }
        assert g.style(bg=copy.deepcopy(bg), fg=copy.deepcopy(fg), logo=copy.deepcopy(logo), page=copy.deepcopy(page))\
                .style(bg={'color': 'green'})._style == {'bg': {'color': 'green'}, 'fg': fg, 'logo': logo, 'page': page}

        g2 = graphistry.edges(pd.DataFrame({'s': [0], 'd': [0]})).bind(source='s', destination='d')
        ds = g2.style(bg=copy.deepcopy(bg), fg=copy.deepcopy(fg), page=copy.deepcopy(page), logo=copy.deepcopy(logo)).plot(skip_upload=True)
        assert ds.metadata['bg'] == bg
        assert ds.metadata['fg'] == fg
        assert ds.metadata['logo'] == logo
        assert ds.metadata['page'] == page

    def test_addStyle_good(self):
        g = graphistry.bind()

        bg = {'color': 'red'}
        fg = {'blendMode': 1}
        logo = {'url': 'zzz'}
        page = {'title': 'zzz'}

        assert g.addStyle()._style == {}

        g.addStyle(fg={'blendMode': 'screen'})
        assert g.addStyle()._style == {}

        assert g.addStyle(bg=copy.deepcopy(bg))._style == {'bg': bg}
        assert g.addStyle(bg={'color': 'blue'}).addStyle(bg=copy.deepcopy(bg))._style == {'bg': bg}
        assert g.addStyle(bg={'image': {'url': 'http://asdf.com/b.png'}}).addStyle(bg=copy.deepcopy(bg))._style == {'bg': {**bg, 'image': {'url': 'http://asdf.com/b.png'}}}
        assert g.addStyle(bg=copy.deepcopy(bg), fg=copy.deepcopy(fg), logo=copy.deepcopy(logo), page=copy.deepcopy(page))._style == {
            'bg': bg, 'fg': fg, 'logo': logo, 'page': page
        }
        assert g.addStyle(bg=copy.deepcopy(bg), fg=copy.deepcopy(fg), logo=copy.deepcopy(logo), page=copy.deepcopy(page))\
                .addStyle(bg={'color': 'green'})._style == {'bg': {'color': 'green'}, 'fg': fg, 'logo': logo, 'page': page}

        g2 = graphistry.edges(pd.DataFrame({'s': [0], 'd': [0]})).bind(source='s', destination='d')
        ds = g2.addStyle(bg=copy.deepcopy(bg), fg=copy.deepcopy(fg), page=copy.deepcopy(page), logo=copy.deepcopy(logo)).plot(skip_upload=True)
        assert ds.metadata['bg'] == bg
        assert ds.metadata['fg'] == fg
        assert ds.metadata['logo'] == logo
        assert ds.metadata['page'] == page

class TestPlotterStylesJSON(NoAuthTestCase):

    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.pygraphistry.PyGraphistry.store_token_creds_in_memory(True)
        graphistry.pygraphistry.PyGraphistry.relogin = lambda: True
        graphistry.register(api=1)

    def test_styleApi_reject(self):
        bg = {'color': 'red'}
        fg = {'blendMode': 1}
        logo = {'url': 'zzz'}
        page = {'title': 'zzz'}
        g2 = graphistry.edges(pd.DataFrame({'s': [0], 'd': [0]})).bind(source='s', destination='d')
        g3 = g2.addStyle(bg=copy.deepcopy(bg), fg=copy.deepcopy(fg), page=copy.deepcopy(page), logo=copy.deepcopy(logo))
        
        with pytest.raises(ValueError):
            g3.plot(skip_upload=True)


class TestPlotterStylesVgraph(NoAuthTestCase):

    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.pygraphistry.PyGraphistry.store_token_creds_in_memory(True)
        graphistry.pygraphistry.PyGraphistry.relogin = lambda: True
        graphistry.register(api=2)

    def test_styleApi_reject(self):
        bg = {'color': 'red'}
        fg = {'blendMode': 1}
        logo = {'url': 'zzz'}
        page = {'title': 'zzz'}
        g2 = graphistry.edges(pd.DataFrame({'s': [0], 'd': [0]})).bind(source='s', destination='d')
        g3 = g2.addStyle(bg=copy.deepcopy(bg), fg=copy.deepcopy(fg), page=copy.deepcopy(page), logo=copy.deepcopy(logo))
        
        with pytest.raises(ValueError):
            g3.plot(skip_upload=True)




class TestPlotterEncodings(NoAuthTestCase):

    COMPLEX_EMPTY = {
        'node_encodings': {'current': {}, 'default': {} },
        'edge_encodings': {'current': {}, 'default': {} }
    }

    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.pygraphistry.PyGraphistry.store_token_creds_in_memory(True)
        graphistry.pygraphistry.PyGraphistry.relogin = lambda: True
        graphistry.register(api=3)

    def test_init_mt(self):
        assert graphistry.bind()._complex_encodings == TestPlotterEncodings.COMPLEX_EMPTY

    def test_point_color(self):
        assert graphistry.bind().encode_point_color('z')._point_color == 'z'
        assert graphistry.bind().encode_point_color('z', ["red", "blue"], as_continuous=True)._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'node_encodings': {
                    'default': {
                        'pointColorEncoding': {
                            'graphType': 'point',
                            'encodingType': 'color',
                            'attribute': 'z',
                            'variation': 'continuous',
                            'colors': ['red', 'blue']
                        }
                    },
                    'current': {}
                }
            }
        assert graphistry.bind().encode_point_color('z', ["red", "blue"], as_categorical=True)._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'node_encodings': {
                    'default': {
                        'pointColorEncoding': {
                            'graphType': 'point',
                            'encodingType': 'color',
                            'attribute': 'z',
                            'variation': 'categorical',
                            'colors': ['red', 'blue']
                        }
                    },
                    'current': {}
                }
            }
        assert graphistry.bind().encode_point_color('z', categorical_mapping={'truck': 'red'})._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'node_encodings': {
                    'default': {
                        'pointColorEncoding': {
                            'graphType': 'point',
                            'encodingType': 'color',
                            'attribute': 'z',
                            'variation': 'categorical',
                            'mapping': { 'categorical': { 'fixed': { 'truck': 'red' } } }
                        }
                    },
                    'current': {}
                }
            }
        assert graphistry.bind().encode_point_color('z', categorical_mapping={'truck': 'red'}, default_mapping='blue')._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'node_encodings': {
                    'default': {
                        'pointColorEncoding': {
                            'graphType': 'point',
                            'encodingType': 'color',
                            'attribute': 'z',
                            'variation': 'categorical',
                            'mapping': {
                                'categorical': {
                                    'fixed': { 'truck': 'red' },
                                    'other': 'blue'
                                }
                            }
                        }
                    },
                    'current': {}
                }
            }


    def test_point_size(self):
        assert graphistry.bind().encode_point_size('z')._point_size == 'z'

    def test_point_icon(self):
        assert graphistry.bind().encode_point_icon('z')._point_icon == 'z'

        assert graphistry.bind().encode_point_icon('z', categorical_mapping={}, as_text=True, blend_mode='color-dodge')\
            ._complex_encodings['node_encodings'] == {
                'current': {},
                'default': {
                    'pointIconEncoding': {
                        'graphType': 'point',
                        'encodingType': 'icon',
                        'attribute': 'z',
                        'variation': 'categorical',
                        'mapping': { 'categorical': {'fixed': {} }},
                        'asText': True,
                        'blendMode': 'color-dodge'
                    }
                }
            }

        assert graphistry.bind().encode_point_icon('z', continuous_binning=[], comparator='<=', as_text=True, blend_mode='color-dodge')\
            ._complex_encodings['node_encodings'] == {
                'current': {},
                'default': {
                    'pointIconEncoding': {
                        'graphType': 'point',
                        'encodingType': 'icon',
                        'attribute': 'z',
                        'variation': 'continuous',
                        'mapping': { 'continuous': {'bins': [], 'comparator': '<=' }},
                        'asText': True,
                        'blendMode': 'color-dodge'
                    }
                }
            }

    def test_edge_icon(self):
        assert graphistry.bind().encode_edge_icon('z')._edge_icon == 'z'

    def test_edge_color(self):
        assert graphistry.bind().encode_edge_color('z')._edge_color == 'z'


    def test_badge(self):

        assert graphistry.bind().encode_point_badge('z', position='Top')._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'node_encodings': {
                    'default': {
                        'pointBadgeTopEncoding': {
                            'graphType': 'point',
                            'encodingType': 'badgeTop',
                            'attribute': 'z',
                            'variation': 'categorical'
                        }
                    },
                    'current': {}
                }
            }

        assert graphistry.bind().encode_edge_badge('z', position='Top')._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'edge_encodings': {
                    'default': {
                        'edgeBadgeTopEncoding': {
                            'graphType': 'edge',
                            'encodingType': 'badgeTop',
                            'attribute': 'z',
                            'variation': 'categorical'
                        }
                    },
                    'current': {}
                }
            }

        assert graphistry.bind().encode_point_badge('z', position='Top',
            continuous_binning=[[None, 'a']], default_mapping='zz', comparator='<=',
            color='red', bg={'color': 'green'}, fg={'style': {'opacity': 0.5}},
            as_text=True, blend_mode='color-dodge', style={'opacity': 0.5},
            border={'width': 10, 'color': 'green', 'stroke': 'dotted'})._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'node_encodings': {
                    'default': {
                        'pointBadgeTopEncoding': {
                            'graphType': 'point',
                            'encodingType': 'badgeTop',
                            'attribute': 'z',
                            'variation': 'continuous',
                            'mapping': { 
                                'continuous': {
                                    'bins': [[None, 'a']],
                                    'comparator': '<=',
                                    'other': 'zz'
                                }
                            },
                            'color': 'red',
                            'bg': {'color': 'green'},
                            'fg': {'style': {'opacity': 0.5}},
                            'asText': True, 'blendMode': 'color-dodge',
                            'style': {'opacity': 0.5},
                            'border': {'width': 10, 'color': 'green', 'stroke': 'dotted'}
                        }
                    },
                    'current': {}
                }
            }

        assert graphistry.bind().encode_edge_badge('z', position='Right', 
            categorical_mapping={'a': 'b'}, for_default=False, for_current=True)._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'edge_encodings': {
                    'default': {},
                    'current': {
                        'edgeBadgeRightEncoding': {
                            'graphType': 'edge',
                            'encodingType': 'badgeRight',
                            'attribute': 'z',
                            'variation': 'categorical',
                            'mapping': { 'categorical': {'fixed': {'a': 'b'}}}
                        }
                    }
                }
            }

    def test_set_mode(self):
        assert graphistry.bind().encode_point_color('z', categorical_mapping={'a': 'b'})._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'node_encodings': {
                    'default': {
                        'pointColorEncoding': {
                            'graphType': 'point',
                            'encodingType': 'color',
                            'attribute': 'z',
                            'variation': 'categorical',
                            'mapping': { 'categorical': {'fixed': { 'a': 'b' } } }
                        }
                    },
                    'current': {}
                }
            }

        assert graphistry.bind().encode_point_color('z', categorical_mapping={'a': 'b'}, for_default=False, for_current=False)._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'node_encodings': {
                    'default': {},
                    'current': {}
                }
            }

        assert graphistry.bind().encode_point_color('z', categorical_mapping={'a': 'b'}, for_default=True, for_current=False)._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'node_encodings': {
                    'default': {
                        'pointColorEncoding': {
                            'graphType': 'point',
                            'encodingType': 'color',
                            'attribute': 'z',
                            'variation': 'categorical',
                            'mapping': { 'categorical': {'fixed': { 'a': 'b' } } }
                        }
                    },
                    'current': {}
                }
            }

        assert graphistry.bind().encode_point_color('z', categorical_mapping={'a': 'b'}, for_default=False, for_current=True)._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'node_encodings': {
                    'default': { },
                    'current': {
                        'pointColorEncoding': {
                            'graphType': 'point',
                            'encodingType': 'color',
                            'attribute': 'z',
                            'variation': 'categorical',
                            'mapping': { 'categorical': { 'fixed': { 'a': 'b' } } }
                        }
                    }
                }
            }

        assert graphistry.bind().encode_point_color('z', categorical_mapping={'a': 'b'}, for_default=True, for_current=True)._complex_encodings \
            == {
                **TestPlotterEncodings.COMPLEX_EMPTY,
                'node_encodings': {
                    'default': {
                        'pointColorEncoding': {
                            'graphType': 'point',
                            'encodingType': 'color',
                            'attribute': 'z',
                            'variation': 'categorical',
                            'mapping': { 'categorical': { 'fixed': { 'a': 'b' } } }
                        }
                    },
                    'current': {
                        'pointColorEncoding': {
                            'graphType': 'point',
                            'encodingType': 'color',
                            'attribute': 'z',
                            'variation': 'categorical',
                            'mapping': { 'categorical': { 'fixed': { 'a': 'b' } } }
                        }
                    }
                }
            }


    def test_composition(self):
        # chaining + overriding
        out = graphistry.bind()\
            .encode_point_size('z', categorical_mapping={'m': 2})\
            .encode_point_color('z', categorical_mapping={'a': 'b'}, for_current=True)\
            .encode_point_color('z', categorical_mapping={'a': 'b2'})\
            .encode_edge_color( 'z', categorical_mapping={'x': 'y'}, for_current=True)\
            ._complex_encodings
        assert out['edge_encodings']['default'] == {
            'edgeColorEncoding': {
                'graphType': 'edge',
                'encodingType': 'color',
                'attribute': 'z',
                'variation': 'categorical',
                'mapping': { 'categorical': { 'fixed': { 'x': 'y' } } }
            }
        }
        assert out['edge_encodings']['current'] == {
            'edgeColorEncoding': {
                'graphType': 'edge',
                'encodingType': 'color',
                'attribute': 'z',
                'variation': 'categorical',
                'mapping': { 'categorical': { 'fixed': { 'x': 'y' } } }
            }
        }
        assert out['node_encodings']['default'] == {
            'pointSizeEncoding': {
                'graphType': 'point',
                'encodingType': 'size',
                'attribute': 'z',
                'variation': 'categorical',
                'mapping': { 'categorical': { 'fixed': { 'm': 2 } } }
            },
            'pointColorEncoding': {
                'graphType': 'point',
                'encodingType': 'color',
                'attribute': 'z',
                'variation': 'categorical',
                'mapping': { 'categorical': { 'fixed': { 'a': 'b2' } } }
            }
        }
        assert out['node_encodings']['current'] == {
            'pointColorEncoding': {
                'graphType': 'point',
                'encodingType': 'color',
                'attribute': 'z',
                'variation': 'categorical',
                'mapping': { 'categorical': { 'fixed': { 'a': 'b' } } }
            }
        }


class TestPlotterMixins(NoAuthTestCase):

    @classmethod
    def setUpClass(cls):
        1

    def test_has_degree(self):
        g = graphistry.bind().edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'a']}), 's', 'd')
        g2 = g.get_degrees()
        assert g2._nodes.to_dict(orient='records') == [
            {'id': 'a', 'degree_in': 1, 'degree_out': 1, 'degree': 2},
            {'id': 'b', 'degree_in': 1, 'degree_out': 1, 'degree': 2},
        ]
