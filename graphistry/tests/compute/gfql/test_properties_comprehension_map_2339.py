import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLUnsupportedError


@pytest.fixture(params=["pandas", "cudf", "polars", "polars-gpu"])
def engine(request):
    if request.param == "cudf":
        pytest.importorskip("cudf")
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("cudf tests need TEST_CUDF=1")
    elif request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
    return request.param


def _graph(engine):
    nodes = pd.DataFrame({"id": [0]})
    edges = pd.DataFrame({"s": [0], "d": [1], "eid": [0]})
    if engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d").bind(edge="eid")


def test_properties_of_comprehension_local_map_is_identity(engine):
    query = "MATCH (n) RETURN [m IN [{a: 1}] | properties(m)][0] AS v"
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises((NotImplementedError, GFQLUnsupportedError)):
            _graph(engine).gfql(query, engine=engine)
        return

    result = _graph(engine).gfql(query, engine=engine)._nodes
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["v"].tolist() == [{"a": 1}]
