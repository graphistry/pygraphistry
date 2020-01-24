# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import graphistry
from mock import patch
from common import NoAuthTestCase

class TestNodexlBindings(NoAuthTestCase):

    def test_from_xls_default(self):
        xls = pd.ExcelFile('graphistry/tests/data/NodeXLWorkbook-220237-twitter.xlsx', engine='openpyxl')
        nodes_df = pd.read_excel(xls, 'Vertices')
        edges_df = pd.read_excel(xls, 'Edges')
        g = graphistry.nodexl(xls)
        self.assertEqual(len(g._nodes), len(nodes_df) - 1)
        self.assertEqual(len(g._edges), len(edges_df) - 1)
        self.assertEqual(g._node, 'Vertex')
        self.assertEqual(g._source, 'Vertex 1')
        self.assertEqual(g._destination, 'Vertex 2')

    def test_from_xls_twitter(self):
        xls = pd.ExcelFile('graphistry/tests/data/NodeXLWorkbook-220237-twitter.xlsx', engine='openpyxl')
        nodes_df = pd.read_excel(xls, 'Vertices')
        edges_df = pd.read_excel(xls, 'Edges')
        g = graphistry.nodexl(xls, 'twitter')
        self.assertEqual(len(g._nodes), len(nodes_df) - 1)
        self.assertEqual(len(g._edges), len(edges_df) - 1)
        self.assertEqual(g._node, 'Vertex')
        self.assertEqual(g._source, 'Vertex 1')
        self.assertEqual(g._destination, 'Vertex 2')
