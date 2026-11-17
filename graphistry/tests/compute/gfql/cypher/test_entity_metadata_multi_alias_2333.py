import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLUnsupportedError


def _engine_frame(engine):
    nodes = pd.DataFrame({
        "id": ["a", "b"],
        "tags": ["left", "right"],
        "shared": ["n-left", "n-right"],
    })
    edges = pd.DataFrame({
        "s": ["a", "b"],
        "d": ["b", "a"],
        "type": ["KNOWS", "KNOWS"],
        "weight": [5, 7],
        "shared": ["e-left", "e-right"],
    })
    if engine == "cudf":
        import cudf

        return cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return nodes, edges


def _records(frame):
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    return frame.to_dict("records")


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
    elif request.param == "polars":
        pytest.importorskip("polars")
    return request.param


def test_keys_and_properties_of_node_alias_ignore_other_bindings(engine):
    nodes, edges = _engine_frame(engine)
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

    query = "MATCH (n),(m) RETURN DISTINCT keys(n) AS k, properties(n) AS p"
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises((NotImplementedError, GFQLUnsupportedError)):
            graph.gfql(query, engine=engine)
        return

    rows = _records(graph.gfql(query, engine=engine)._nodes)
    assert rows == [
        {"k": ["tags", "shared"], "p": "{tags: 'left', shared: 'n-left'}"},
        {"k": ["tags", "shared"], "p": "{tags: 'right', shared: 'n-right'}"},
    ]
