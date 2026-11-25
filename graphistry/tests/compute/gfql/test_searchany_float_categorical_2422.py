"""searchAny over an explicit categorical-of-floats column, through the query engines: pandas, Polars and polars-gpu answer it, cuDF declines by name."""
import os

import pandas as pd
import pytest

import graphistry


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


_QUERY = "MATCH (n) WHERE searchAny(n, '1', {columns: ['catf']}) RETURN n.id AS id ORDER BY id"


def _nodes(engine):
    pdf = pd.DataFrame({"id": [0, 1, 2], "catf": pd.Categorical([1.5, 2.5, 1.5])})
    if engine == "cudf":
        import cudf

        return cudf.from_pandas(pdf)
    return pdf


def test_an_explicit_float_categorical_is_answered_or_declined_by_name_on_every_engine(engine):
    graph = graphistry.nodes(_nodes(engine), "id")
    if engine == "cudf":
        with pytest.raises(Exception) as caught:
            graph.gfql(_QUERY, engine=engine)
        assert "Value inf not found" not in str(caught.value) and "searchAny explicit columns" in str(caught.value)
    else:
        out = graph.gfql(_QUERY, engine=engine)._nodes
        out = out.to_pandas() if hasattr(out, "to_pandas") else out
        assert list(out["id"]) == [0, 2]
