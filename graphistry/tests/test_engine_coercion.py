# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import pyarrow as pa

from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.ComputeMixin import _coerce_to_pandas
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


if __name__ == "__main__":
    unittest.main()
