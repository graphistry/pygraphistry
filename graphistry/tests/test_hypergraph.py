# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy
import datetime
import graphistry
import graphistry.plotter
from mock import patch
from common import NoAuthTestCase

nid = graphistry.plotter.Plotter._defaultNodeId

triangleNodesDict = {
    'id': ['a', 'b', 'c'], 
    'a1': [1, 2, 3], 
    'a2': ['red', 'blue', 'green'],
    '🙈': ['æski ēˈmōjē', '😋', 's']
}
triangleNodes = pd.DataFrame(triangleNodesDict)


def assertFrameEqual(df1, df2, **kwds ):
    """ Assert that two dataframes are equal, ignoring ordering of columns"""

    from pandas.util.testing import assert_frame_equal
    return assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1), check_names=True, **kwds)


@patch('webbrowser.open')
class TestHypergraphPlain(NoAuthTestCase):

  
    def test_hyperedges(self, mock_open):

        h = graphistry.hypergraph(triangleNodes, verbose=False)
        
        self.assertEqual(len(h.keys()), len(['entities', 'nodes', 'edges', 'events', 'graph']))

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

        assertFrameEqual(h['edges'], edges)
        for (k, v) in [('entities', 12), ('nodes', 15), ('edges', 12), ('events', 3)]:
            self.assertEqual(len(h[k]), v)

    def test_drop_edge_attrs(self, mock_open):
    
        h = graphistry.hypergraph(triangleNodes, verbose=False, drop_edge_attrs=True)

        self.assertEqual(len(h.keys()), len(['entities', 'nodes', 'edges', 'events', 'graph']))

        edges = pd.DataFrame({
            'edgeType': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'id', 'id', 'id', '🙈', '🙈', '🙈'],
            'attribID': [
                'a1::1', 'a1::2', 'a1::3', 
                'a2::red', 'a2::blue', 'a2::green',                 
                'id::a', 'id::b', 'id::c',
                '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s'],
            'EventID': ['EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2']})


        assertFrameEqual(h['edges'], edges)
        for (k, v) in [('entities', 12), ('nodes', 15), ('edges', 12), ('events', 3)]:
            self.assertEqual(len(h[k]), v)
