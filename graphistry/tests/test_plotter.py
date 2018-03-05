from builtins import object

import unittest
import pandas
import requests
import IPython
import igraph
import networkx as nx
import graphistry
import datetime as dt
from mock import patch
from common import NoAuthTestCase


triangleEdges = pandas.DataFrame({'src': ['a', 'b', 'c'], 'dst': ['b', 'c', 'a']})
triangleNodes = pandas.DataFrame({'id': ['a', 'b', 'c'], 'a1': [1, 2, 3], 'a2': ['red', 'blue', 'green']})
triangleNodesRich = pandas.DataFrame({
    'id': ['a', 'b', 'c'], 
    'a1': [1, 2, 3], 
    'a2': ['red', 'blue', 'green'],
    'a3': [True, False, False],
    'a4': [0.5, 1.5, 1000.3],
    'a5': [dt.datetime.fromtimestamp(x) for x in [1440643875, 1440644191, 1440645638]]    
})

class Fake_Response(object):
    def raise_for_status(self):
        pass
    def json(self):
        return {'success': True, 'dataset': 'fakedatasetname', 'viztoken': 'faketoken'}


def assertFrameEqual(df1, df2, **kwds ):
    """ Assert that two dataframes are equal, ignoring ordering of columns"""

    from pandas.util.testing import assert_frame_equal
    return assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1), check_names=True, **kwds)



@patch('webbrowser.open')
@patch.object(graphistry.util, 'warn')
@patch.object(graphistry.pygraphistry.PyGraphistry, '_etl2')
class TestPlotterBindings(NoAuthTestCase):

    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.register(api=2)


    def test_no_src_dst(self, mock_etl2, mock_warn, mock_open):
        with self.assertRaises(ValueError):
            graphistry.bind().plot(triangleEdges)
        with self.assertRaises(ValueError):
            graphistry.bind(source='src').plot(triangleEdges)
        with self.assertRaises(ValueError):
            graphistry.bind(destination='dst').plot(triangleEdges)
        with self.assertRaises(ValueError):
            graphistry.bind(source='doesnotexist', destination='dst').plot(triangleEdges)


    def test_no_nodeid(self, mock_etl2, mock_warn, mock_open):
        plotter = graphistry.bind(source='src', destination='dst')
        with self.assertRaises(ValueError):
            plotter.plot(triangleEdges, triangleNodes)


    def test_triangle_edges(self, mock_etl2, mock_warn, mock_open):
        plotter = graphistry.bind(source='src', destination='dst')
        plotter.plot(triangleEdges)
        self.assertTrue(mock_etl2.called)
        self.assertFalse(mock_warn.called)


    def test_bind_edges(self, mock_etl2, mock_warn, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', edge_title='src')
        plotter.plot(triangleEdges)
        self.assertTrue(mock_etl2.called)
        self.assertFalse(mock_warn.called)


    def test_bind_nodes(self, mock_etl2, mock_warn, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', node='id', point_title='a2')
        plotter.plot(triangleEdges, triangleNodes)
        self.assertTrue(mock_etl2.called)
        self.assertFalse(mock_warn.called)


    def test_bind_nodes_rich(self, mock_etl2, mock_warn, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', node='id', point_title='a2')
        plotter.plot(triangleEdges, triangleNodesRich)
        self.assertTrue(mock_etl2.called)
        self.assertFalse(mock_warn.called)


    def test_unknown_col_edges(self, mock_etl2, mock_warn, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', edge_title='doesnotexist')
        plotter.plot(triangleEdges)
        self.assertTrue(mock_etl2.called)
        self.assertTrue(mock_warn.called)


    def test_unknown_col_nodes(self, mock_etl2, mock_warn, mock_open):
        plotter = graphistry.bind(source='src', destination='dst', node='id', point_title='doesnotexist')
        plotter.plot(triangleEdges, triangleNodes)
        self.assertTrue(mock_etl2.called)
        self.assertTrue(mock_warn.called)


    @patch.object(graphistry.util, 'error')
    def test_empty_graph(self, mock_error, mock_etl2, mock_warn, mock_open):
        mock_error.side_effect = ValueError('error')
        plotter = graphistry.bind(source='src', destination='dst')
        with self.assertRaises(ValueError):
            plotter.plot(pandas.DataFrame([]))
        self.assertFalse(mock_etl2.called)
        self.assertTrue(mock_error.called)



@patch('webbrowser.open')
@patch('requests.post', return_value=Fake_Response())
class TestPlotterReturnValue(NoAuthTestCase):

    def test_no_ipython(self, mock_post, mock_open):
        url = graphistry.bind(source='src', destination='dst').plot(triangleEdges)
        self.assertIn('fakedatasetname', url)
        self.assertIn('faketoken', url)
        self.assertTrue(mock_open.called)
        self.assertTrue(mock_post.called)


    @patch.object(graphistry.util, 'in_ipython', return_value=True)
    def test_ipython(self, mock_util, mock_post, mock_open):
        widget = graphistry.bind(source='src', destination='dst').plot(triangleEdges)
        self.assertIsInstance(widget, IPython.core.display.HTML)



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

    def test_igraph2pandas(self):
        ig = igraph.Graph.Tree(4, 2)
        ig.vs['vattrib'] = 0
        ig.es['eattrib'] = 1
        (e, n) = graphistry.bind(source='src', destination='dst').igraph2pandas(ig)

        edges = pandas.DataFrame({
            'dst': {0: 1, 1: 2, 2: 3},
            'src': {0: 0, 1: 0, 2: 1},
            'eattrib': {0: 1, 1: 1, 2: 1}
        })
        nodes = pandas.DataFrame({
            '__nodeid__': {0: 0, 1: 1, 2: 2, 3: 3},
            'vattrib': {0: 0, 1: 0, 2: 0, 3: 0}
        })

        assertFrameEqual(e, edges)
        assertFrameEqual(n, nodes)


    def test_pandas2igraph(self):
        plotter = graphistry.bind(source='src', destination='dst', node='id')
        ig = plotter.pandas2igraph(triangleEdges)
        (e, n) = plotter.igraph2pandas(ig)
        assertFrameEqual(e, triangleEdges[['src', 'dst']])
        assertFrameEqual(n, triangleNodes[['id']])


    def test_networkx2igraph(self):
        ng = nx.complete_graph(3)
        [x, y] = [int(x) for x in nx.__version__.split('.')]
        if x == 1:
            nx.set_node_attributes(ng, 'vattrib', 0)
            nx.set_edge_attributes(ng, 'eattrib', 1)
        else:
            nx.set_node_attributes(ng, 0, 'vattrib')
            nx.set_edge_attributes(ng, 1, 'eattrib')
        (e, n) = graphistry.bind(source='src', destination='dst').networkx2pandas(ng)

        edges = pandas.DataFrame({
            'dst': {0: 1, 1: 2, 2: 2},
            'src': {0: 0, 1: 0, 2: 1},
            'eattrib': {0: 1, 1: 1, 2: 1}
        })
        nodes = pandas.DataFrame({
            '__nodeid__': {0: 0, 1: 1, 2: 2},
            'vattrib': {0: 0, 1: 0, 2: 0}
        })

        assertFrameEqual(e, edges)
        assertFrameEqual(n, nodes)
