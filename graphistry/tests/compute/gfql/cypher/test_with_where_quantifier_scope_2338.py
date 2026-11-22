import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLUnsupportedError


def _nodes(engine):
    nodes = pd.DataFrame({
        "id": ["a", "b"],
        "tags": ["left", "right"],
    })
    if engine == "cudf":
        import cudf

        return cudf.from_pandas(nodes)
    if engine == "polars":
        import polars as pl

        return pl.from_pandas(nodes)
    return nodes


def _rows(frame):
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


def test_with_where_quantifier_variable_shadows_match_alias(engine):
    graph = graphistry.nodes(_nodes(engine), "id")
    query = (
        "MATCH (n) WITH n.tags AS t "
        "WHERE any(n IN [{tags: 'z'}] WHERE n.tags = 'z') "
        "RETURN t"
    )
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises((NotImplementedError, GFQLUnsupportedError)):
            graph.gfql(query, engine=engine)
        return

    rows = _rows(graph.gfql(query, engine=engine)._nodes)
    assert [(row["t"]) for row in rows] == ["left", "right"]


def test_with_where_list_comprehension_variable_shadows_match_alias(engine):
    graph = graphistry.nodes(_nodes(engine), "id")
    query = (
        "MATCH (n) WITH n.tags AS t "
        "WHERE size([n IN [{tags: 'z'}] WHERE n.tags = 'z']) = 1 "
        "RETURN t"
    )
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises((NotImplementedError, GFQLUnsupportedError)):
            graph.gfql(query, engine=engine)
        return

    rows = _rows(graph.gfql(query, engine=engine)._nodes)
    assert [(row["t"]) for row in rows] == ["left", "right"]


def test_with_where_unshadowed_outer_output_stays_visible(engine):
    graph = graphistry.nodes(_nodes(engine), "id")
    query = (
        "MATCH (n) WITH n.tags AS t "
        "WHERE any(x IN [{tags: 'left'}] WHERE t = x.tags) "
        "RETURN DISTINCT t"
    )
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises((NotImplementedError, GFQLUnsupportedError)):
            graph.gfql(query, engine=engine)
        return

    rows = _rows(graph.gfql(query, engine=engine)._nodes)
    assert [(row["t"]) for row in rows] == ["left"]
