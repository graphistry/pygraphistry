# -*- coding: utf-8 -*-
"""
Tests for uniform DataFrame type handling across compute and hypergraph paths.
Covers graphistry/pygraphistry#1132.

pa.Table should work everywhere pd.DataFrame works for:
  - materialize_nodes()
  - get_degrees() / get_indegrees() / get_outdegrees()
  - hypergraph()
"""
import importlib
import pandas as pd
import pyarrow as pa
import pytest
import unittest

import graphistry
from graphistry.tests.common import NoAuthTestCase
from graphistry.tests.test_compute import CGFull


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EDGES_PD = pd.DataFrame({"src": ["a", "b", "c"], "dst": ["b", "c", "a"]})
EDGES_ARROW = pa.table({"src": ["a", "b", "c"], "dst": ["b", "c", "a"]})

NODES_PD = pd.DataFrame({"id": ["a", "b", "c"], "v": [1, 2, 3]})
NODES_ARROW = pa.table({"id": ["a", "b", "c"], "v": [1, 2, 3]})

EVENTS_PD = pd.DataFrame({"user": ["alice", "bob", "alice"], "action": ["click", "view", "buy"]})
EVENTS_ARROW = pa.table({"user": ["alice", "bob", "alice"], "action": ["click", "view", "buy"]})


# ---------------------------------------------------------------------------
# Regression: existing pandas behavior is unchanged
# ---------------------------------------------------------------------------

class TestRegressionPandas(NoAuthTestCase):

    def test_materialize_nodes_pandas(self):
        g = CGFull().edges(EDGES_PD, "src", "dst").materialize_nodes()
        self.assertIsInstance(g._nodes, pd.DataFrame)
        self.assertIn("id", g._nodes.columns)
        self.assertEqual(sorted(g._nodes["id"].tolist()), ["a", "b", "c"])

    def test_get_degrees_pandas(self):
        g = CGFull().edges(EDGES_PD, "src", "dst").get_degrees()
        self.assertIsInstance(g._nodes, pd.DataFrame)
        self.assertIn("degree_in", g._nodes.columns)
        self.assertIn("degree_out", g._nodes.columns)

    def test_hypergraph_pandas(self):
        h = graphistry.hypergraph(EVENTS_PD, verbose=False)
        self.assertIn("graph", h)
        g = h["graph"]
        self.assertIsInstance(g._nodes, pd.DataFrame)
        self.assertIsInstance(g._edges, pd.DataFrame)
        self.assertGreater(len(g._nodes), 0)
        self.assertGreater(len(g._edges), 0)


# ---------------------------------------------------------------------------
# Arrow: compute path
# ---------------------------------------------------------------------------

class TestArrowCompute(NoAuthTestCase):

    def test_materialize_nodes_arrow_edges_only(self):
        """pa.Table edges → materialize_nodes produces pandas _nodes."""
        g = CGFull().edges(EDGES_ARROW, "src", "dst").materialize_nodes()
        self.assertIsInstance(g._nodes, pd.DataFrame)
        self.assertIn("id", g._nodes.columns)
        self.assertEqual(sorted(g._nodes["id"].tolist()), ["a", "b", "c"])

    def test_materialize_nodes_arrow_edges_and_nodes(self):
        """pa.Table edges + nodes → materialize_nodes reuses nodes, returns pandas."""
        g = (
            CGFull()
            .edges(EDGES_ARROW, "src", "dst")
            .nodes(NODES_ARROW, "id")
            .materialize_nodes()
        )
        self.assertIsInstance(g._nodes, pd.DataFrame)
        self.assertEqual(sorted(g._nodes["id"].tolist()), ["a", "b", "c"])

    def test_get_degrees_arrow(self):
        """pa.Table edges → get_degrees produces degree columns."""
        g = CGFull().edges(EDGES_ARROW, "src", "dst").get_degrees()
        self.assertIsInstance(g._nodes, pd.DataFrame)
        self.assertIn("degree_in", g._nodes.columns)
        self.assertIn("degree_out", g._nodes.columns)

    def test_get_indegrees_arrow(self):
        g = CGFull().edges(EDGES_ARROW, "src", "dst").get_indegrees()
        self.assertIsInstance(g._nodes, pd.DataFrame)
        self.assertIn("degree_in", g._nodes.columns)

    def test_get_outdegrees_arrow(self):
        g = CGFull().edges(EDGES_ARROW, "src", "dst").get_outdegrees()
        self.assertIsInstance(g._nodes, pd.DataFrame)
        self.assertIn("degree_out", g._nodes.columns)


# ---------------------------------------------------------------------------
# Arrow: hypergraph path
# ---------------------------------------------------------------------------

class TestArrowHypergraph(NoAuthTestCase):

    def test_hypergraph_arrow_returns_graph(self):
        """pa.Table events → hypergraph returns a valid graph."""
        h = graphistry.hypergraph(EVENTS_ARROW, verbose=False)
        self.assertIn("graph", h)
        g = h["graph"]
        self.assertIsInstance(g._nodes, pd.DataFrame)
        self.assertIsInstance(g._edges, pd.DataFrame)
        self.assertGreater(len(g._nodes), 0)
        self.assertGreater(len(g._edges), 0)

    def test_hypergraph_arrow_columns_preserved(self):
        """Column names and unique values survive conversion."""
        h = graphistry.hypergraph(EVENTS_ARROW, verbose=False)
        nodes = h["graph"]._nodes
        # Both 'user' and 'action' entity types should appear as node types
        node_types = nodes["type"].unique().tolist() if "type" in nodes.columns else []
        self.assertTrue(
            any("user" in str(t) or "action" in str(t) for t in node_types),
            f"Expected user/action entity types in nodes, got: {node_types}"
        )

    def test_hypergraph_arrow_matches_pandas(self):
        """Arrow and pandas inputs to hypergraph produce same node/edge counts."""
        h_pd = graphistry.hypergraph(EVENTS_PD, verbose=False)
        h_arrow = graphistry.hypergraph(EVENTS_ARROW, verbose=False)
        self.assertEqual(
            len(h_pd["graph"]._nodes),
            len(h_arrow["graph"]._nodes),
            "Node count should match between pandas and Arrow inputs"
        )
        self.assertEqual(
            len(h_pd["graph"]._edges),
            len(h_arrow["graph"]._edges),
            "Edge count should match between pandas and Arrow inputs"
        )


# ---------------------------------------------------------------------------
# Spark: stubs (skipped without pyspark)
# ---------------------------------------------------------------------------

_pyspark_available = importlib.util.find_spec("pyspark") is not None


@pytest.mark.skipif(not _pyspark_available, reason="pyspark not installed")
class TestSparkCompute(NoAuthTestCase):

    def _make_spark_edges(self):
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local").appName("test").getOrCreate()
        return spark.createDataFrame(
            [("a", "b"), ("b", "c"), ("c", "a")], ["src", "dst"]
        )

    def test_materialize_nodes_spark(self):
        sdf = self._make_spark_edges()
        g = CGFull().edges(sdf, "src", "dst").materialize_nodes()
        self.assertIsInstance(g._nodes, pd.DataFrame)
        self.assertIn("id", g._nodes.columns)

    def test_get_degrees_spark(self):
        sdf = self._make_spark_edges()
        g = CGFull().edges(sdf, "src", "dst").get_degrees()
        self.assertIsInstance(g._nodes, pd.DataFrame)
        self.assertIn("degree_in", g._nodes.columns)


@pytest.mark.skipif(not _pyspark_available, reason="pyspark not installed")
class TestSparkHypergraph(NoAuthTestCase):

    def test_hypergraph_spark(self):
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local").appName("test").getOrCreate()
        sdf = spark.createDataFrame(
            [("alice", "click"), ("bob", "view")], ["user", "action"]
        )
        h = graphistry.hypergraph(sdf, verbose=False)
        self.assertIn("graph", h)
        self.assertGreater(len(h["graph"]._nodes), 0)


if __name__ == "__main__":
    unittest.main()
