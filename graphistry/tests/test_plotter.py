import unittest
import pandas
import igraph
import networkx
import graphistry
from mock import patch


triangleEdges = pandas.DataFrame({'src': ['a', 'b', 'c'], 'dst': ['b', 'c', 'a']})
triangleNodes = pandas.DataFrame({'id': ['a', 'b', 'c'], 'a1': [1, 2, 3], 'a2': ['red', 'blue', 'green']})


def assertFrameEqual(df1, df2, **kwds ):
    """ Assert that two dataframes are equal, ignoring ordering of columns"""

    from pandas.util.testing import assert_frame_equal
    return assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1), check_names=True, **kwds)


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


class TestPlotterConversions(unittest.TestCase):

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
        ng = networkx.complete_graph(3)
        networkx.set_node_attributes(ng, 'vattrib', 0)
        networkx.set_edge_attributes(ng, 'eattrib', 1)
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
