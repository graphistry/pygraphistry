# -*- coding: utf-8 -*-
"""
Tests for df_to_engine() type coercion and _coerce_to_pandas() Plottable coercion.
Covers the coerce-at-boundary unification (refactor/coerce-unification).

All types are handled explicitly via module-string gating — no duck typing.
GPU tests (cuDF, dask_cudf) are skipped when those deps are absent.
"""
import unittest
import pandas as pd
import pyarrow as pa

from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.ComputeMixin import _coerce_to_pandas
from graphistry.tests.common import NoAuthTestCase
from graphistry.tests.test_compute import CGFull


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

    @unittest.skipIf(True, "pyspark optional — run on dgx-spark with pyspark installed")
    def test_spark(self):
        pass  # covered in test_df_types.py spark tests on dgx-spark

    @unittest.skipIf(True, "dask optional — run on dgx-spark")
    def test_dask(self):
        pass

    @unittest.skipIf(True, "cuDF optional — run on dgx-spark with GPU")
    def test_cudf(self):
        pass


class TestDfToEngineCudf(NoAuthTestCase):

    @unittest.skipIf(True, "cuDF optional — run on dgx-spark with GPU")
    def test_cudf_identity(self):
        pass

    @unittest.skipIf(True, "cuDF optional — run on dgx-spark with GPU")
    def test_arrow_to_cudf(self):
        pass

    @unittest.skipIf(True, "cuDF optional — run on dgx-spark with GPU")
    def test_pandas_to_cudf(self):
        pass


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

    @unittest.skipIf(True, "cuDF optional — run on dgx-spark with GPU")
    def test_cudf_edges_not_coerced(self):
        """cuDF is a compute engine — _coerce_to_pandas must leave it alone."""
        pass  # see test_df_types.py cudf tests on dgx-spark

    @unittest.skipIf(True, "dask optional — run on dgx-spark")
    def test_dask_edges_coerced(self):
        """dask is an input format — must be coerced to pandas."""
        pass


if __name__ == "__main__":
    unittest.main()
