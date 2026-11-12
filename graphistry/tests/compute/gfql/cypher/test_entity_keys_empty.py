"""Ordinary entity keys preserve empty lists across native nullable rows."""
import os

import pandas as pd
import pytest

import graphistry


@pytest.mark.parametrize("engine", [
    "pandas",
    pytest.param("cudf", marks=pytest.mark.skipif(os.environ.get("TEST_CUDF") != "1", reason="needs TEST_CUDF=1")),
])
@pytest.mark.parametrize("values,expected", [
    ([None, "x"], [[], ["p"]]),
    ([None, None], [[], []]),
    (["x", "y"], [["p"], ["p"]]),
])
def test_entity_keys_with_empty_property_rows(engine, values, expected):
    nodes = pd.DataFrame({"id": ["a", "b"], "p": values})
    if engine == "cudf":
        import cudf
        nodes = cudf.from_pandas(nodes)
    before = nodes.to_arrow().to_pylist() if engine == "cudf" else nodes.to_dict("records")
    source = graphistry.nodes(nodes, "id")
    result = source.gfql("MATCH (n) RETURN n.id AS identity, keys(n) AS k", engine=engine)
    rows = result._nodes.to_arrow().to_pylist() if engine == "cudf" else result._nodes.to_dict("records")
    assert rows == [{"identity": identity, "k": keys} for identity, keys in zip(["a", "b"], expected)]
    assert type(result._nodes).__module__.split(".")[0] == engine
    after = nodes.to_arrow().to_pylist() if engine == "cudf" else nodes.to_dict("records")
    assert after == before
    assert source._nodes is nodes and source._node == "id"
