# -*- coding: utf-8 -*-

import pandas as pd
import pytest
import graphistry
from common import NoAuthTestCase


try:
    import openpyxl
    has_openpyxl = True
except (ImportError, ModuleNotFoundError):
    has_openpyxl = False


@pytest.mark.skipif(not has_openpyxl, reason="openpyxl not installed")
class TestNodexlBindings(NoAuthTestCase):
    def test_from_xls_default(self):
        xls = pd.ExcelFile(
            "graphistry/tests/data/NodeXLWorkbook-220237-twitter.xlsx",
            engine="openpyxl",
        )
        nodes_df = pd.read_excel(xls, "Vertices")
        edges_df = pd.read_excel(xls, "Edges")
        g = graphistry.nodexl(xls)
        self.assertEqual(len(g._nodes), len(nodes_df) - 1)
        self.assertEqual(len(g._edges), len(edges_df) - 1)
        self.assertEqual(g._node, "Vertex")
        self.assertEqual(g._source, "Vertex 1")
        self.assertEqual(g._destination, "Vertex 2")
        assert g._nodes["Color2"].dtype.name == "int32"
        assert g._edges["ColorInt"].dtype.name == "int32"
        assert g._edge_title is None

    def test_from_xls_twitter(self):
        xls = pd.ExcelFile(
            "graphistry/tests/data/NodeXLWorkbook-220237-twitter.xlsx",
            engine="openpyxl",
        )
        nodes_df = pd.read_excel(xls, "Vertices")
        edges_df = pd.read_excel(xls, "Edges")
        g = graphistry.nodexl(xls, "twitter")
        self.assertEqual(len(g._nodes), len(nodes_df) - 1)
        self.assertEqual(len(g._edges), len(edges_df) - 1)
        self.assertEqual(g._node, "Vertex")
        self.assertEqual(g._source, "Vertex 1")
        self.assertEqual(g._destination, "Vertex 2")
        assert g._nodes["Color2"].dtype.name == "int32"
        assert g._edges["ColorInt"].dtype.name == "int32"
        assert g._edge_title == "Relationship"
