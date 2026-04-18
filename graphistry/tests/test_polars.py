# -*- coding: utf-8 -*-
"""
Tests for Polars DataFrame support across plot, compute, and hypergraph paths.
Covers graphistry/pygraphistry#1133.

polars.DataFrame and polars.LazyFrame should work everywhere pd.DataFrame works for:
  - plot() upload path (via _table_to_arrow)
  - _table_to_pandas()
  - materialize_nodes() / get_degrees() / get_indegrees() / get_outdegrees()
  - hypergraph()
"""
import pandas as pd
import pyarrow as pa
import unittest

polars = pytest = None
try:
    import polars as pl
    import pytest
except ImportError:
    pass

if pl is None or pytest is None:
    import sys
    print("polars not installed — skipping test_polars.py", file=sys.stderr)
    # Make the module importable but empty when polars is absent
    class _Skip(unittest.TestCase):
        @unittest.skip("polars not installed")
        def test_placeholder(self):
            pass
else:
    import graphistry
    from graphistry.PlotterBase import PlotterBase
    from graphistry.tests.common import NoAuthTestCase
    from graphistry.tests.test_compute import CGFull

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    EDGES_PL = pl.DataFrame({"src": ["a", "b", "c"], "dst": ["b", "c", "a"]})
    EDGES_LAZY = EDGES_PL.lazy()

    NODES_PL = pl.DataFrame({"id": ["a", "b", "c"], "v": [1, 2, 3]})
    NODES_LAZY = NODES_PL.lazy()

    EVENTS_PL = pl.DataFrame({"user": ["alice", "bob", "alice"], "action": ["click", "view", "buy"]})
    EVENTS_LAZY = EVENTS_PL.lazy()

    EDGES_PD = pd.DataFrame({"src": ["a", "b", "c"], "dst": ["b", "c", "a"]})
    NODES_PD = pd.DataFrame({"id": ["a", "b", "c"], "v": [1, 2, 3]})

    # ------------------------------------------------------------------
    # PlotterBase internals — _table_to_arrow and _table_to_pandas
    # ------------------------------------------------------------------

    class TestPolarsInternals(NoAuthTestCase):

        def _plotter(self):
            return CGFull()

        def test_table_to_pandas_dataframe(self):
            g = self._plotter()
            result = g._table_to_pandas(EDGES_PL)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(list(result.columns), ["src", "dst"])

        def test_table_to_pandas_lazyframe(self):
            g = self._plotter()
            result = g._table_to_pandas(EDGES_LAZY)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(list(result.columns), ["src", "dst"])

        def test_table_to_arrow_dataframe(self):
            g = self._plotter()
            result = g._table_to_arrow(EDGES_PL, memoize=False)
            self.assertIsInstance(result, pa.Table)
            self.assertFalse(result.schema.metadata)  # empty dict or None — no polars metadata

        def test_table_to_arrow_lazyframe(self):
            g = self._plotter()
            result = g._table_to_arrow(EDGES_LAZY, memoize=False)
            self.assertIsInstance(result, pa.Table)

        def test_table_to_arrow_memoization(self):
            """Same polars frame → same Arrow object returned (memoized)."""
            g = self._plotter()
            r1 = g._table_to_arrow(EDGES_PL, memoize=True)
            r2 = g._table_to_arrow(EDGES_PL, memoize=True)
            self.assertIs(r1, r2)

        def test_table_to_arrow_schema_metadata_stripped(self):
            """Polars attaches polars-specific schema metadata; it must be stripped."""
            g = self._plotter()
            result = g._table_to_arrow(NODES_PL, memoize=False)
            self.assertIsInstance(result, pa.Table)
            self.assertFalse(result.schema.metadata)  # empty dict or None — no polars metadata

    # ------------------------------------------------------------------
    # Compute path
    # ------------------------------------------------------------------

    class TestPolarsCompute(NoAuthTestCase):

        def test_materialize_nodes_polars_edges(self):
            g = CGFull().edges(EDGES_PL, "src", "dst").materialize_nodes()
            self.assertIsInstance(g._nodes, pd.DataFrame)
            self.assertIn("id", g._nodes.columns)
            self.assertEqual(sorted(g._nodes["id"].tolist()), ["a", "b", "c"])

        def test_materialize_nodes_polars_lazy_edges(self):
            g = CGFull().edges(EDGES_LAZY, "src", "dst").materialize_nodes()
            self.assertIsInstance(g._nodes, pd.DataFrame)
            self.assertEqual(sorted(g._nodes["id"].tolist()), ["a", "b", "c"])

        def test_materialize_nodes_polars_edges_and_nodes(self):
            g = (CGFull()
                 .edges(EDGES_PL, "src", "dst")
                 .nodes(NODES_PL, "id")
                 .materialize_nodes())
            self.assertIsInstance(g._nodes, pd.DataFrame)
            self.assertIn("v", g._nodes.columns)

        def test_materialize_nodes_polars_edges_pandas_nodes(self):
            """Mixed: Polars edges + pandas nodes."""
            g = (CGFull()
                 .edges(EDGES_PL, "src", "dst")
                 .nodes(NODES_PD, "id")
                 .materialize_nodes())
            self.assertIsInstance(g._nodes, pd.DataFrame)
            self.assertIn("v", g._nodes.columns)

        def test_get_degrees_polars(self):
            g = CGFull().edges(EDGES_PL, "src", "dst").get_degrees()
            self.assertIsInstance(g._nodes, pd.DataFrame)
            self.assertIn("degree", g._nodes.columns)
            self.assertIn("degree_in", g._nodes.columns)
            self.assertIn("degree_out", g._nodes.columns)
            self.assertTrue(
                (g._nodes["degree"] == g._nodes["degree_in"] + g._nodes["degree_out"]).all()
            )

        def test_get_indegrees_polars(self):
            g = CGFull().edges(EDGES_PL, "src", "dst").get_indegrees()
            self.assertIsInstance(g._nodes, pd.DataFrame)
            self.assertIn("degree_in", g._nodes.columns)

        def test_get_outdegrees_polars(self):
            g = CGFull().edges(EDGES_PL, "src", "dst").get_outdegrees()
            self.assertIsInstance(g._nodes, pd.DataFrame)
            self.assertIn("degree_out", g._nodes.columns)

    # ------------------------------------------------------------------
    # Hypergraph path
    # ------------------------------------------------------------------

    class TestPolarsHypergraph(NoAuthTestCase):

        def test_hypergraph_polars_dataframe(self):
            h = graphistry.hypergraph(EVENTS_PL, verbose=False)
            self.assertIn("graph", h)
            g = h["graph"]
            self.assertIsInstance(g._nodes, pd.DataFrame)
            self.assertIsInstance(g._edges, pd.DataFrame)
            self.assertGreater(len(g._nodes), 0)
            self.assertGreater(len(g._edges), 0)

        def test_hypergraph_polars_lazy(self):
            h = graphistry.hypergraph(EVENTS_LAZY, verbose=False)
            g = h["graph"]
            self.assertIsInstance(g._nodes, pd.DataFrame)
            self.assertGreater(len(g._nodes), 0)

        def test_hypergraph_polars_matches_pandas(self):
            import pandas as pd_mod
            events_pd = EVENTS_PL.to_pandas()
            h_pd = graphistry.hypergraph(events_pd, verbose=False)
            h_pl = graphistry.hypergraph(EVENTS_PL, verbose=False)
            self.assertEqual(len(h_pd["graph"]._nodes), len(h_pl["graph"]._nodes))
            self.assertEqual(len(h_pd["graph"]._edges), len(h_pl["graph"]._edges))

        def test_hypergraph_polars_with_entity_types(self):
            h = graphistry.hypergraph(EVENTS_PL, entity_types=["user"], verbose=False)
            g = h["graph"]
            self.assertIsInstance(g._nodes, pd.DataFrame)
            self.assertGreater(len(g._nodes), 0)


if __name__ == "__main__":
    unittest.main()
