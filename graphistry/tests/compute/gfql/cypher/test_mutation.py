from __future__ import annotations

import os
from typing import Any

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher.api import cypher_to_gfql


ENGINES = [
    pytest.param("pandas"),
    pytest.param("cudf", marks=pytest.mark.skipif(
        os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"
    )),
    pytest.param("polars-gpu", marks=pytest.mark.skipif(
        os.environ.get("TEST_POLARS_GPU") != "1", reason="requires actual polars-gpu"
    )),
]


def _records(frame: Any) -> list[dict[str, Any]]:
    if hasattr(frame, "to_pandas"):
        rows = frame.to_pandas().to_dict(orient="records")
    else:
        rows = frame.to_dict(orient="records")
    return [
        {key: None if pd.isna(value) else value for key, value in row.items()}
        for row in rows
    ]


def _assert_served(engine: str, frame: Any) -> None:
    expected = "pandas" if engine == "pandas" else engine
    assert type(frame).__module__.split(".")[0] == expected.split("-")[0]


def _graph(engine: str) -> Any:
    nodes = pd.DataFrame({"id": [1, 2], "label__Person": [True, False]})
    edges = pd.DataFrame({"src": [1], "dst": [2], "type": ["KNOWS"]})
    if engine == "cudf":
        import cudf
        nodes = cudf.from_pandas(nodes)
        edges = cudf.from_pandas(edges)
    elif engine == "polars-gpu":
        import polars
        nodes = polars.from_pandas(nodes)
        edges = polars.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


@pytest.mark.parametrize("engine", ENGINES)
def test_create_nodes_adds_identity_labels_and_properties(engine: str) -> None:
    source = _graph(engine)

    result = source.gfql(
        "CREATE (a:Admin {name: 'Ada', age: 36}), (b:Admin {name: $name})",
        params={"name": "Grace"},
        engine=engine,
        language="cypher",
    )

    assert isinstance(result, type(source))
    assert result._edges is source._edges
    assert _records(result._nodes) == [
        {"id": 1, "label__Person": True, "label__Admin": False, "name": None, "age": None},
        {"id": 2, "label__Person": False, "label__Admin": False, "name": None, "age": None},
        {"id": 3, "label__Person": False, "label__Admin": True, "name": "Ada", "age": 36},
        {"id": 4, "label__Person": False, "label__Admin": True, "name": "Grace", "age": None},
    ]
    assert _records(source._nodes) == [
        {"id": 1, "label__Person": True},
        {"id": 2, "label__Person": False},
    ]
    assert result.gfql_graph is result
    assert source.gfql_graph is None
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_create_nodes_without_existing_nodes_uses_identity_zero(engine: str) -> None:
    edges = pd.DataFrame({"src": [7], "dst": [8], "type": ["R"]})
    source = graphistry.edges(edges, "src", "dst")

    result = source.gfql("CREATE (a:L {name: 'x'})", engine=engine, language="cypher")

    assert _records(result._nodes) == [{"_node": 0, "label__L": True, "name": "x"}]
    assert result._edges is source._edges
    assert result.gfql_graph is result
    _assert_served(engine, result._nodes)


def test_create_rejects_non_numeric_existing_identity() -> None:
    source = graphistry.nodes(pd.DataFrame({"id": ["a", "b"]}), "id")

    with pytest.raises(GFQLValidationError, match="numeric node identities"):
        source.gfql("CREATE (a:L)", engine="pandas", language="cypher")


def test_create_rejects_missing_existing_identity_column() -> None:
    source = graphistry.bind(node="id").nodes(pd.DataFrame({"name": ["Ada"]}))

    with pytest.raises(GFQLValidationError, match="missing identity column 'id'"):
        source.gfql("CREATE (a:L)", engine="pandas", language="cypher")


def test_create_return_is_typed_supported_gap() -> None:
    source = _graph("pandas")

    with pytest.raises(GFQLValidationError, match="CREATE with RETURN"):
        source.gfql("CREATE (a:L) RETURN a", engine="pandas", language="cypher")


def test_cypher_to_gfql_rejects_standalone_create() -> None:
    with pytest.raises(GFQLValidationError, match="Standalone Cypher CREATE"):
        cypher_to_gfql("CREATE (a:L)")


def test_gfql_graph_is_read_only() -> None:
    result = _graph("pandas").gfql(
        "CREATE (a:L)", engine="pandas", language="cypher"
    )

    with pytest.raises(AttributeError):
        result.gfql_graph = result
