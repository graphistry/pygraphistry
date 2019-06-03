# -*- coding: utf-8 -*-

import unittest
import pandas
import numpy
import datetime
import graphistry
import graphistry.plotter
from mock import patch
from common import NoAuthTestCase

nid = graphistry.plotter.Plotter._defaultNodeId

triangleEdges = pandas.DataFrame({'src': ['a', 'b', 'c'], 'dst': ['b', 'c', 'a']})
triangleNodes = pandas.DataFrame({'id': ['a', 'b', 'c'], 'a1': [1, 2, 3], 'a2': ['red', 'blue', 'green']})



def assertFrameEqual(df1, df2, **kwds ):
    """ Assert that two dataframes are equal, ignoring ordering of columns"""

    from pandas.util.testing import assert_frame_equal
    return assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1), check_names=True, **kwds)



@patch('webbrowser.open')
@patch.object(graphistry.pygraphistry.PyGraphistry, '_etl2')
class TestEtl2Metadata(NoAuthTestCase):

    def test_metadata_mandatory_fields(self, mock_etl2, mock_open):
        graphistry.bind(source='src', destination='dst').plot(triangleEdges)
        dataset = mock_etl2.call_args[0][0]
        self.assertListEqual(list(dataset['attributes']['nodes']), [nid])
        self.assertListEqual(list(dataset['attributes']['edges']), [])

        self.assertEqual(dataset['encodings']['nodes'], {
            'pointTitle': {'attributes': [nid]},
            'nodeId': {'attributes': [nid]}
        })
        self.assertEqual(dataset['encodings']['edges'], {
            'source': {'attributes': ['src']},
            'destination': {'attributes': ['dst']}
        })


    def test_metadata_double_name(self, mock_etl2, mock_open):
        edges = triangleEdges.copy()
        edges['a1'] = triangleNodes.a1.map(lambda x: x+10)
        graphistry.bind(source='src', destination='dst', node='id').plot(edges, triangleNodes)
        dataset = mock_etl2.call_args[0][0]

        self.assertIn('a1', dataset['attributes']['nodes'])
        self.assertIn('a1', dataset['attributes']['edges'])


    def test_metadata_no_nan(self, mock_etl2, mock_open):
        edges = triangleEdges.copy()
        edges['testNone'] = triangleNodes.a1.map(lambda x: numpy.nan)
        edges['testNone'] = pandas.to_numeric(edges.testNone, errors='ignore')
        edges['testInt'] = triangleNodes.a1.map(lambda x: numpy.nan if x%2 == 1 else 0)
        edges['testFloat'] = triangleNodes.a1.map(lambda x: numpy.nan if x%2 == 1 else 0.5)
        edges['testString'] = triangleNodes.a1.map(lambda x: numpy.nan if x%2 == 1 else 'foo')
        edges['testBool'] = triangleNodes.a1.map(lambda x: numpy.nan if x%2 == 1 else True)
        graphistry.bind(source='src', destination='dst', node='id').plot(edges)
        dataset = mock_etl2.call_args[0][0]

        #for attrib in ['testInt', 'testFloat', 'testString', 'testBool', 'testNone']:
        #    for entry in list(dataset['attributes']['edges'][attrib]['aggregations'].values()):
        #        if entry is None or isinstance(entry, str):
        #            pass
        #        else:
        #            self.assertFalse(numpy.isnan(entry))

    def test_metadata_no_nat(self, mock_etl2, mock_open):
        edges = triangleEdges.copy()
        edges['testDate'] = triangleNodes.a1.map(lambda x: None if x == 1 else datetime.datetime.now())
        edges['testDate2'] = triangleNodes.a1.map(lambda x: None)
        edges['testDate'] = pandas.to_datetime(edges.testDate, errors='ignore')
        edges['testDate2'] = pandas.to_datetime(edges.testDate2, errors='ignore')
        graphistry.bind(source='src', destination='dst', node='id').plot(edges)
        dataset = mock_etl2.call_args[0][0]

        #for attrib in ['testDate', 'testDate2']:
        #    for entry in list(dataset['attributes']['edges'][attrib]['aggregations'].values()):
        #        self.assertFalse(isinstance(entry, type(pandas.NaT)))


@patch('webbrowser.open')
@patch.object(graphistry.pygraphistry.PyGraphistry, '_etl2')
class TestEtl2Unicode(NoAuthTestCase):

    def test_metadata_mandatory_fields(self, mock_etl2, mock_open):
        edges = triangleEdges.copy()
        edges['ustring'] = [u'abcdef', u'тйîbàüd', u'♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜']

        graphistry.bind(source='src', destination='dst').plot(edges)
        self.assertTrue(mock_etl2.called)
