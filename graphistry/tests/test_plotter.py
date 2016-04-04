import unittest
import pandas
import graphistry
from mock import patch


triangleEdges = pandas.DataFrame({'src': ['a', 'b', 'c'], 'dst': ['b', 'c', 'a']})
triangleNodes = pandas.DataFrame({'id': ['a', 'b', 'c'], 'a1': [1, 2, 3], 'a2': ['red', 'blue', 'green']})


@patch.object(graphistry.util, 'warn')
@patch.object(graphistry.pygraphistry.PyGraphistry, '_etl2')
class TestPlotterBindings(unittest.TestCase):

    def test_no_src_dst(self, mock_etl2, mock_warn):
        with self.assertRaises(ValueError):
            graphistry.bind().plot(triangleEdges)
        with self.assertRaises(ValueError):
            graphistry.bind(source='src').plot(triangleEdges)
        with self.assertRaises(ValueError):
            graphistry.bind(destination='dst').plot(triangleEdges)


    def test_no_nodeid(self, mock_etl2, mock_warn):
        plotter = graphistry.bind(source='src', destination='dst')
        with self.assertRaises(ValueError):
            plotter.plot(triangleEdges, triangleNodes)


    def test_triangle_edges(self, mock_etl2, mock_warn):
        plotter = graphistry.bind(source='src', destination='dst')
        plotter.plot(triangleEdges)
        self.assertTrue(mock_etl2.called)
        self.assertFalse(mock_warn.called)


    def test_bind_edges(self, mock_etl2, mock_warn):
        plotter = graphistry.bind(source='src', destination='dst', edge_title='src')
        plotter.plot(triangleEdges)
        self.assertTrue(mock_etl2.called)
        self.assertFalse(mock_warn.called)


    def test_bind_nodes(self, mock_etl2, mock_warn):
        plotter = graphistry.bind(source='src', destination='dst', node='id', point_title='a2')
        plotter.plot(triangleEdges, triangleNodes)
        self.assertTrue(mock_etl2.called)
        self.assertFalse(mock_warn.called)


    def test_unknown_col_edges(self, mock_etl2, mock_warn):
        plotter = graphistry.bind(source='src', destination='dst', edge_title='doesnotexist')
        plotter.plot(triangleEdges)
        self.assertTrue(mock_etl2.called)
        self.assertTrue(mock_warn.called)


    def test_unknown_col_nodes(self, mock_etl2, mock_warn):
        plotter = graphistry.bind(source='src', destination='dst', node='id', point_title='doesnotexist')
        plotter.plot(triangleEdges, triangleNodes)
        self.assertTrue(mock_etl2.called)
        self.assertTrue(mock_warn.called)


@patch.object(graphistry.pygraphistry.PyGraphistry, '_etl2')
class TestPlotterCallChaining(unittest.TestCase):

    def test_bind_chain(self, mock_etl2):
        plotter0 = graphistry.bind(source='caca').bind(destination='dst', source='src')
        plotter0.plot(triangleEdges)
        self.assertTrue(mock_etl2.called)


    def test_bind_edges_nodes(self, mock_etl2):
        plotter0 = graphistry.bind(source='src').bind(destination='dst')
        plotter1 = plotter0.bind(node='id').bind(point_title='a2')
        plotter1.edges(triangleEdges).nodes(triangleNodes).plot()
        self.assertTrue(mock_etl2.called)
