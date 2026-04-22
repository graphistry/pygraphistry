# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import pyarrow as pa

from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.ComputeMixin import _coerce_to_pandas, _coerce_input_formats
from graphistry.tests.common import NoAuthTestCase
from graphistry.tests.test_compute import CGFull


try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

try:
    import dask.dataframe as dd
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

try:
    import dask_cudf
    HAS_DASK_CUDF = True
except ImportError:
    HAS_DASK_CUDF = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    from pyspark.sql import SparkSession
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False


EDGES_PD = pd.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]})
EDGES_PA = pa.table({"src": ["a", "b"], "dst": ["b", "c"]})


class TestDfToEnginePandas(NoAuthTestCase):

    def test_pandas_identity(self):
        result = df_to_engine(EDGES_PD, Engine.PANDAS)
        self.assertIs(result, EDGES_PD)

    def test_arrow(self):
        result = df_to_engine(EDGES_PA, Engine.PANDAS)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["src", "dst"])
        self.assertEqual(result["src"].tolist(), ["a", "b"])

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            df_to_engine("not a dataframe", Engine.PANDAS)
        self.assertIn("str", str(ctx.exception))

    def test_unknown_type_raises_list(self):
        with self.assertRaises(ValueError):
            df_to_engine([1, 2, 3], Engine.PANDAS)

    @unittest.skipUnless(HAS_SPARK, "pyspark not installed")
    def test_spark(self):
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(EDGES_PD)
        result = df_to_engine(sdf, Engine.PANDAS)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(sorted(result.columns.tolist()), ["dst", "src"])

    @unittest.skipUnless(HAS_DASK, "dask not installed")
    def test_dask(self):
        ddf = dd.from_pandas(EDGES_PD, npartitions=1)
        result = df_to_engine(ddf, Engine.PANDAS)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result["src"].tolist(), ["a", "b"])

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_cudf(self):
        cdf = cudf.from_pandas(EDGES_PD)
        result = df_to_engine(cdf, Engine.PANDAS)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result["src"].tolist(), ["a", "b"])


class TestDfToEngineCudf(NoAuthTestCase):

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_cudf_identity(self):
        cdf = cudf.from_pandas(EDGES_PD)
        result = df_to_engine(cdf, Engine.CUDF)
        self.assertIsInstance(result, cudf.DataFrame)
        self.assertIs(result, cdf)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_arrow_to_cudf(self):
        result = df_to_engine(EDGES_PA, Engine.CUDF)
        self.assertIsInstance(result, cudf.DataFrame)
        self.assertEqual(sorted(result.columns.tolist()), ["dst", "src"])

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_pandas_to_cudf(self):
        result = df_to_engine(EDGES_PD, Engine.CUDF)
        self.assertIsInstance(result, cudf.DataFrame)
        self.assertEqual(result["src"].to_pandas().tolist(), ["a", "b"])


class TestCoerceToPandas(NoAuthTestCase):

    def _g(self, edges=None, nodes=None):
        g = CGFull().edges(edges if edges is not None else EDGES_PD, "src", "dst")
        if nodes is not None:
            g = g.nodes(nodes, "id")
        return g

    def test_pandas_edges_unchanged(self):
        g = self._g(EDGES_PD)
        result = _coerce_to_pandas(g)
        self.assertIs(result._edges, EDGES_PD)

    def test_arrow_edges_coerced(self):
        g = self._g(EDGES_PA)
        result = _coerce_to_pandas(g)
        self.assertIsInstance(result._edges, pd.DataFrame)
        self.assertEqual(list(result._edges.columns), ["src", "dst"])

    def test_arrow_nodes_coerced(self):
        nodes_pa = pa.table({"id": ["a", "b", "c"]})
        g = self._g(EDGES_PD, nodes=nodes_pa)
        result = _coerce_to_pandas(g)
        self.assertIsInstance(result._nodes, pd.DataFrame)

    def test_none_edges_untouched(self):
        g = CGFull()  # no edges set
        result = _coerce_to_pandas(g)
        self.assertIsNone(result._edges)

    def test_none_nodes_untouched(self):
        g = self._g(EDGES_PD)
        self.assertIsNone(g._nodes)
        result = _coerce_to_pandas(g)
        self.assertIsNone(result._nodes)

    def test_idempotent(self):
        """Calling twice is a no-op."""
        g = self._g(EDGES_PA)
        once = _coerce_to_pandas(g)
        twice = _coerce_to_pandas(once)
        self.assertIs(twice._edges, once._edges)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_cudf_edges_not_coerced(self):
        """cuDF is a compute engine — _coerce_to_pandas must leave it alone."""
        cdf = cudf.from_pandas(EDGES_PD)
        g = self._g(cdf)
        result = _coerce_to_pandas(g)
        self.assertIsInstance(result._edges, cudf.DataFrame)

    @unittest.skipUnless(HAS_DASK, "dask not installed")
    def test_dask_edges_coerced(self):
        """dask is an input format — must be coerced to pandas."""
        ddf = dd.from_pandas(EDGES_PD, npartitions=1)
        g = self._g(ddf)
        result = _coerce_to_pandas(g)
        self.assertIsInstance(result._edges, pd.DataFrame)
        self.assertEqual(result._edges["src"].tolist(), ["a", "b"])


class TestCoerceInputFormats(NoAuthTestCase):
    """Unit tests for _coerce_input_formats — the engine-aware replacement for _coerce_to_pandas."""

    def _g(self, edges=None, nodes=None):
        g = CGFull().edges(edges if edges is not None else EDGES_PD, "src", "dst")
        if nodes is not None:
            g = g.nodes(nodes, "id")
        return g

    # --- Engine.PANDAS ---

    def test_pandas_engine_pandas_noop(self):
        g = self._g(EDGES_PD)
        result = _coerce_input_formats(g, Engine.PANDAS)
        self.assertIs(result._edges, EDGES_PD)

    def test_pandas_engine_arrow_coerced(self):
        g = self._g(EDGES_PA)
        result = _coerce_input_formats(g, Engine.PANDAS)
        self.assertIsInstance(result._edges, pd.DataFrame)
        self.assertEqual(result._edges["src"].tolist(), ["a", "b"])

    def test_pandas_engine_arrow_nodes_coerced(self):
        nodes_pa = pa.table({"id": ["a", "b", "c"]})
        g = self._g(EDGES_PD, nodes=nodes_pa)
        result = _coerce_input_formats(g, Engine.PANDAS)
        self.assertIsInstance(result._nodes, pd.DataFrame)

    def test_pandas_engine_none_edges_untouched(self):
        g = CGFull()
        result = _coerce_input_formats(g, Engine.PANDAS)
        self.assertIsNone(result._edges)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_pandas_engine_cudf_preserved(self):
        """cuDF is a GPU compute engine — must not be coerced to pandas."""
        cdf = cudf.from_pandas(EDGES_PD)
        g = self._g(cdf)
        result = _coerce_input_formats(g, Engine.PANDAS)
        self.assertIsInstance(result._edges, cudf.DataFrame)

    @unittest.skipUnless(HAS_DASK_CUDF, "dask_cudf not installed")
    def test_pandas_engine_dask_cudf_preserved(self):
        """dask_cudf is a GPU compute engine — must not be coerced to pandas."""
        if cudf.__version__.startswith("25."):
            self.skipTest(f"dask_cudf.from_cudf segfaults in cuDF {cudf.__version__} (RAPIDS 25.x numba tokenization bug)")
        cdf = cudf.from_pandas(EDGES_PD)
        dcdf = dask_cudf.from_cudf(cdf, npartitions=1)
        g = self._g(dcdf)
        result = _coerce_input_formats(g, Engine.PANDAS)
        self.assertIn("cudf", str(type(result._edges).__module__))

    @unittest.skipUnless(HAS_POLARS, "polars not installed")
    def test_pandas_engine_polars_coerced(self):
        pldf = pl.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]})
        g = self._g(pldf)
        result = _coerce_input_formats(g, Engine.PANDAS)
        self.assertIsInstance(result._edges, pd.DataFrame)
        self.assertEqual(result._edges["src"].tolist(), ["a", "b"])

    # --- Engine.CUDF ---

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_cudf_engine_cudf_noop(self):
        cdf = cudf.from_pandas(EDGES_PD)
        g = self._g(cdf)
        result = _coerce_input_formats(g, Engine.CUDF)
        self.assertIs(result._edges, cdf)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_cudf_engine_pandas_coerced(self):
        g = self._g(EDGES_PD)
        result = _coerce_input_formats(g, Engine.CUDF)
        self.assertIsInstance(result._edges, cudf.DataFrame)
        self.assertEqual(result._edges["src"].to_pandas().tolist(), ["a", "b"])

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_cudf_engine_arrow_coerced(self):
        g = self._g(EDGES_PA)
        result = _coerce_input_formats(g, Engine.CUDF)
        self.assertIsInstance(result._edges, cudf.DataFrame)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    @unittest.skipUnless(HAS_POLARS, "polars not installed")
    def test_cudf_engine_polars_coerced(self):
        pldf = pl.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]})
        g = self._g(pldf)
        result = _coerce_input_formats(g, Engine.CUDF)
        self.assertIsInstance(result._edges, cudf.DataFrame)
        self.assertEqual(result._edges["src"].to_pandas().tolist(), ["a", "b"])

    @unittest.skipUnless(HAS_DASK_CUDF, "dask_cudf not installed")
    def test_cudf_engine_dask_cudf_preserved(self):
        """dask_cudf is already a GPU compute engine — must not be re-coerced."""
        # dask_cudf.from_cudf tokenizes cuDF frames via numba, which segfaults
        # in RAPIDS 25.x due to a CUDA-context bug in that release line.
        if cudf.__version__.startswith("25."):
            self.skipTest(f"dask_cudf.from_cudf segfaults in cuDF {cudf.__version__} (RAPIDS 25.x numba tokenization bug)")
        cdf = cudf.from_pandas(EDGES_PD)
        dcdf = dask_cudf.from_cudf(cdf, npartitions=1)
        g = self._g(dcdf)
        result = _coerce_input_formats(g, Engine.CUDF)
        self.assertIn("cudf", str(type(result._edges).__module__))


class TestCoerceInputFormatsPolars(NoAuthTestCase):
    """Polars-specific _coerce_input_formats tests."""

    @unittest.skipUnless(HAS_POLARS, "polars not installed")
    def test_polars_edges_coerced_to_pandas(self):
        pldf = pl.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]})
        g = CGFull().edges(pldf, "src", "dst")
        result = _coerce_input_formats(g, Engine.PANDAS)
        self.assertIsInstance(result._edges, pd.DataFrame)
        self.assertEqual(sorted(result._edges.columns.tolist()), ["dst", "src"])

    @unittest.skipUnless(HAS_POLARS, "polars not installed")
    def test_polars_nodes_coerced_to_pandas(self):
        pldf_nodes = pl.DataFrame({"id": ["a", "b", "c"]})
        g = CGFull().edges(EDGES_PD, "src", "dst").nodes(pldf_nodes, "id")
        result = _coerce_input_formats(g, Engine.PANDAS)
        self.assertIsInstance(result._nodes, pd.DataFrame)

    @unittest.skipUnless(HAS_POLARS, "polars not installed")
    def test_polars_idempotent(self):
        pldf = pl.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]})
        g = CGFull().edges(pldf, "src", "dst")
        once = _coerce_input_formats(g, Engine.PANDAS)
        twice = _coerce_input_formats(once, Engine.PANDAS)
        self.assertIs(twice._edges, once._edges)


class TestToPandas(NoAuthTestCase):
    """to_pandas() method — converts all supported input types to pandas."""

    def _g(self, edges=None, nodes=None):
        g = CGFull().edges(edges if edges is not None else EDGES_PD, "src", "dst")
        if nodes is not None:
            g = g.nodes(nodes, "id")
        return g

    def test_pandas_identity(self):
        g = self._g(EDGES_PD)
        result = g.to_pandas()
        self.assertIsInstance(result._edges, pd.DataFrame)
        self.assertEqual(result._edges["src"].tolist(), ["a", "b"])

    def test_arrow_coerced(self):
        g = self._g(EDGES_PA)
        result = g.to_pandas()
        self.assertIsInstance(result._edges, pd.DataFrame)
        self.assertEqual(result._edges["src"].tolist(), ["a", "b"])

    def test_none_edges_untouched(self):
        g = CGFull()
        result = g.to_pandas()
        self.assertIsNone(result._edges)

    def test_none_nodes_untouched(self):
        g = self._g(EDGES_PD)
        result = g.to_pandas()
        self.assertIsNone(result._nodes)

    @unittest.skipUnless(HAS_POLARS, "polars not installed")
    def test_polars_coerced(self):
        pldf = pl.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]})
        g = self._g(pldf)
        result = g.to_pandas()
        self.assertIsInstance(result._edges, pd.DataFrame)
        self.assertEqual(result._edges["src"].tolist(), ["a", "b"])

    @unittest.skipUnless(HAS_POLARS, "polars not installed")
    def test_polars_nodes_coerced(self):
        pldf_nodes = pl.DataFrame({"id": ["a", "b", "c"]})
        g = self._g(EDGES_PD, nodes=pldf_nodes)
        result = g.to_pandas()
        self.assertIsInstance(result._nodes, pd.DataFrame)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_cudf_coerced(self):
        cdf = cudf.from_pandas(EDGES_PD)
        g = self._g(cdf)
        result = g.to_pandas()
        self.assertIsInstance(result._edges, pd.DataFrame)
        self.assertEqual(result._edges["src"].tolist(), ["a", "b"])

    @unittest.skipUnless(HAS_DASK, "dask not installed")
    def test_dask_coerced(self):
        ddf = dd.from_pandas(EDGES_PD, npartitions=1)
        g = self._g(ddf)
        result = g.to_pandas()
        self.assertIsInstance(result._edges, pd.DataFrame)
        self.assertEqual(result._edges["src"].tolist(), ["a", "b"])


class TestGPUOutputPreservation(NoAuthTestCase):
    """End-to-end tests verifying GPU (cuDF) input produces GPU output through chain/hop/gfql."""

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_chain_node_filter_cudf_output(self):
        """chain([n()]) with cuDF edges must return cuDF output."""
        from graphistry.compute.ast import n
        cdf_e = cudf.from_pandas(EDGES_PD)
        cdf_n = cudf.DataFrame({"id": ["a", "b", "c"]})
        g = CGFull().edges(cdf_e, "src", "dst").nodes(cdf_n, "id")
        result = g.chain([n()])
        self.assertIsInstance(result._edges, cudf.DataFrame)
        self.assertIsInstance(result._nodes, cudf.DataFrame)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_chain_edge_traversal_cudf_output(self):
        """chain([n, e_forward, n]) exercises hop.py — output must stay cuDF."""
        from graphistry.compute.ast import n, e_forward
        cdf_e = cudf.from_pandas(EDGES_PD)
        cdf_n = cudf.DataFrame({"id": ["a", "b", "c"]})
        g = CGFull().edges(cdf_e, "src", "dst").nodes(cdf_n, "id")
        result = g.chain([n(), e_forward(), n()])
        self.assertIsInstance(result._edges, cudf.DataFrame)
        self.assertIsInstance(result._nodes, cudf.DataFrame)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_gfql_edge_traversal_cudf_output(self):
        """gfql([n, e_forward, n]) with cuDF edges must return cuDF output."""
        from graphistry.compute.ast import n, e_forward
        cdf_e = cudf.from_pandas(EDGES_PD)
        cdf_n = cudf.DataFrame({"id": ["a", "b", "c"]})
        g = CGFull().edges(cdf_e, "src", "dst").nodes(cdf_n, "id")
        result = g.gfql([n(), e_forward(), n()])
        self.assertIsInstance(result._edges, cudf.DataFrame)
        self.assertIsInstance(result._nodes, cudf.DataFrame)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_hop_cudf_input_cudf_output(self):
        """hop() with cuDF edges must return cuDF output, not pandas."""
        cdf_e = cudf.from_pandas(EDGES_PD)
        cdf_n = cudf.DataFrame({"id": ["a", "b", "c"]})
        g = CGFull().edges(cdf_e, "src", "dst").nodes(cdf_n, "id")
        result = g.hop(cdf_n)
        self.assertIsInstance(result._edges, cudf.DataFrame)
        self.assertIsInstance(result._nodes, cudf.DataFrame)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_materialize_nodes_cudf_engine_pandas_input_becomes_cudf(self):
        """materialize_nodes(engine='cudf') with pandas input must produce cuDF output."""
        g = CGFull().edges(EDGES_PD, "src", "dst")
        result = g.materialize_nodes(engine="cudf")
        self.assertIsInstance(result._nodes, cudf.DataFrame)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_materialize_nodes_cudf_input_cudf_output(self):
        """materialize_nodes() with cuDF input must preserve cuDF output."""
        cdf_e = cudf.from_pandas(EDGES_PD)
        g = CGFull().edges(cdf_e, "src", "dst")
        result = g.materialize_nodes()
        self.assertIsInstance(result._nodes, cudf.DataFrame)

    def test_get_indegrees_arrow_input(self):
        """get_indegrees() with Arrow edges must coerce to pandas and return pandas nodes."""
        g = CGFull().edges(EDGES_PA, "src", "dst")
        result = g.get_indegrees()
        self.assertIsInstance(result._nodes, pd.DataFrame)
        self.assertIn("degree_in", result._nodes.columns)

    def test_get_outdegrees_arrow_input(self):
        """get_outdegrees() with Arrow edges must coerce to pandas and return pandas nodes."""
        g = CGFull().edges(EDGES_PA, "src", "dst")
        result = g.get_outdegrees()
        self.assertIsInstance(result._nodes, pd.DataFrame)
        self.assertIn("degree_out", result._nodes.columns)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_get_indegrees_cudf_input_cudf_output(self):
        """get_indegrees() with cuDF edges must preserve cuDF output."""
        cdf_e = cudf.from_pandas(EDGES_PD)
        g = CGFull().edges(cdf_e, "src", "dst")
        result = g.get_indegrees()
        self.assertIsInstance(result._nodes, cudf.DataFrame)
        self.assertIn("degree_in", result._nodes.columns)


class TestCombineStepsEdgeCases(NoAuthTestCase):
    """Tests for specific code paths in combine_steps / apply_output_slice."""

    NODES = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    EDGES = pd.DataFrame({"src": ["a", "b", "c"], "dst": ["b", "c", "d"]})

    def _g(self):
        return CGFull().nodes(self.NODES, "id").edges(self.EDGES, "src", "dst")

    def test_output_max_hops_isin_path(self):
        """combine_steps isin([]) accumulation path: output_max_hops set + has_na in hop column."""
        from graphistry.compute.ast import n, e_forward
        result = self._g().gfql([
            n({"id": "a"}),
            e_forward(hops=2, label_node_hops="node_hop", output_max_hops=2),
            n(),
        ])
        self.assertIsInstance(result._nodes, pd.DataFrame)
        self.assertIn("node_hop", result._nodes.columns)

    def test_undirected_df_concat_path(self):
        """combine_steps df_concat(engine) undirected path: named node followed by e_undirected."""
        from graphistry.compute.ast import n, e_undirected
        result = self._g().gfql([n(name="start"), e_undirected(), n()])
        self.assertIsInstance(result._edges, pd.DataFrame)
        self.assertIn("start", result._nodes.columns)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_output_max_hops_isin_path_cudf(self):
        """isin([]) path must work with cuDF — result stays cuDF."""
        from graphistry.compute.ast import n, e_forward
        cdf_n = cudf.from_pandas(self.NODES)
        cdf_e = cudf.from_pandas(self.EDGES)
        g = CGFull().nodes(cdf_n, "id").edges(cdf_e, "src", "dst")
        result = g.gfql([
            n({"id": "a"}),
            e_forward(hops=2, label_node_hops="node_hop", output_max_hops=2),
            n(),
        ])
        self.assertIsInstance(result._nodes, cudf.DataFrame)
        self.assertIn("node_hop", result._nodes.columns)

    @unittest.skipUnless(HAS_CUDF, "cuDF not installed")
    def test_undirected_df_concat_path_cudf(self):
        """df_concat undirected path must work with cuDF — result stays cuDF."""
        from graphistry.compute.ast import n, e_undirected
        cdf_n = cudf.from_pandas(self.NODES)
        cdf_e = cudf.from_pandas(self.EDGES)
        g = CGFull().nodes(cdf_n, "id").edges(cdf_e, "src", "dst")
        result = g.gfql([n(name="start"), e_undirected(), n()])
        self.assertIsInstance(result._edges, cudf.DataFrame)
        self.assertIn("start", result._nodes.columns)


if __name__ == "__main__":
    unittest.main()
