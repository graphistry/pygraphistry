from __future__ import annotations

import os
from typing import Any, cast

import pandas as pd
import pytest

import graphistry
import graphistry.compute.gfql.cypher as cypher_module
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher.api import cypher_to_gfql
from graphistry.compute.gfql.cypher.lowering import CompiledCypherMutationQuery
from graphistry.compute.gfql.mutation import _numeric_id_seed, _promote_shared_column_dtype


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
def test_create_nodes_without_existing_nodes_preserves_inferred_nodes(engine: str) -> None:
    edges = pd.DataFrame({"src": [7], "dst": [8], "type": ["R"]})
    source = graphistry.edges(edges, "src", "dst")

    result = source.gfql("CREATE (a:L {name: 'x'})", engine=engine, language="cypher")

    assert _records(result._nodes) == [
        {"_node": 7, "label__L": False, "name": None},
        {"_node": 8, "label__L": False, "name": None},
        {"_node": 9, "label__L": True, "name": "x"},
    ]
    assert result._edges is source._edges
    assert result.gfql_graph is result
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_create_preserves_edges_only_inferred_nodes(engine: str) -> None:
    edges = pd.DataFrame({"src": [7], "dst": [8], "type": ["R"]})
    source = graphistry.edges(edges, "src", "dst")

    result = source.gfql("CREATE (a:L)", engine=engine, language="cypher")

    assert result._edges is source._edges
    assert _records(result._nodes) == [
        {"_node": 7, "label__L": False},
        {"_node": 8, "label__L": False},
        {"_node": 9, "label__L": True},
    ]


def test_create_promotes_existing_numeric_and_bool_columns_losslessly() -> None:
    nodes = pd.DataFrame({"id": [1], "score": [1], "flag": [True]})
    source = graphistry.nodes(nodes, "id")

    result = source.gfql(
        "CREATE (a:L {score: 1.5, flag: 2})",
        engine="pandas",
        language="cypher",
    )

    rows = _records(result._nodes)
    assert rows[0]["score"] == 1
    assert rows[1]["score"] == 1.5
    assert rows[0]["flag"] == 1
    assert rows[1]["flag"] == 2
    assert str(result._nodes["score"].dtype) == "Float64"
    assert str(result._nodes["flag"].dtype) == "Int64"


def test_create_rejects_integer_identity_overflow() -> None:
    nodes = pd.DataFrame({"id": pd.array([-128, 126], dtype="int8")})
    source = graphistry.nodes(nodes, "id")

    with pytest.raises(GFQLValidationError, match="cannot hold 2 new IDs"):
        source.gfql("CREATE (a:L), (b:L)", engine="pandas", language="cypher")


def test_create_rejects_unsupported_mixed_program() -> None:
    source = _graph("pandas")

    with pytest.raises(GFQLValidationError, match="CREATE with CALL"):
        source.gfql(
            "CREATE (a:L) CALL degree.write()",
            engine="pandas",
            language="cypher",
        )


def test_create_declines_unavailable_polars_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("polars")
    from graphistry.compute.gfql.lazy import polars_gpu_available

    monkeypatch.setattr(
        "graphistry.compute.gfql.lazy.polars_gpu_available",
        lambda: False,
    )
    source = _graph("polars-gpu")

    with pytest.raises(GFQLValidationError, match="requires a usable GPU"):
        source.gfql("CREATE (a:L)", engine="polars-gpu", language="cypher")


def test_validate_mutation_does_not_touch_chain() -> None:
    source = _graph("pandas")

    report = source.gfql_validate("CREATE (a:L)", engine="pandas", language="cypher")

    assert report["ok"] is True
    assert report["compiled_kind"] == "mutation"


def test_deprecated_compiled_mutation_lazy_export_matches_lowering() -> None:
    with pytest.warns(DeprecationWarning, match="CompiledCypherMutationQuery"):
        lazy_class = cypher_module.CompiledCypherMutationQuery

    assert lazy_class is CompiledCypherMutationQuery


def test_numeric_id_seed_with_missing_identity_starts_at_zero() -> None:
    assert _numeric_id_seed(pd.DataFrame({"other": []}), "id", line=1, column=1) == 0


def test_shared_property_dtype_promotes_bool_and_int_losslessly() -> None:
    promoted = _promote_shared_column_dtype(
        bool, int, column="id", line=1, column_position=1
    )

    assert promoted == "Int64"
    assert _promote_shared_column_dtype(
        bool, bool, column="id", line=1, column_position=1
    ) is bool


def test_create_rejects_non_numeric_existing_identity() -> None:
    source = graphistry.nodes(pd.DataFrame({"id": ["a", "b"]}), "id")

    with pytest.raises(GFQLValidationError, match="numeric node identities"):
        cast(Any, source).gfql("CREATE (a:L)", engine="pandas", language="cypher")


def test_create_rejects_missing_existing_identity_column() -> None:
    source = graphistry.bind(node="id").nodes(pd.DataFrame({"name": ["Ada"]}))

    with pytest.raises(GFQLValidationError, match="missing identity column 'id'"):
        cast(Any, source).gfql("CREATE (a:L)", engine="pandas", language="cypher")


def test_create_return_is_typed_supported_gap() -> None:
    source = _graph("pandas")

    with pytest.raises(GFQLValidationError, match="CREATE with RETURN"):
        cast(Any, source).gfql("CREATE (a:L) RETURN a", engine="pandas", language="cypher")


def test_cypher_to_gfql_rejects_standalone_create() -> None:
    with pytest.raises(GFQLValidationError, match="Standalone Cypher CREATE"):
        cypher_to_gfql("CREATE (a:L)")


def test_gfql_graph_is_read_only() -> None:
    result = _graph("pandas").gfql(
        "CREATE (a:L)", engine="pandas", language="cypher"
    )

    with pytest.raises(AttributeError):
        result.gfql_graph = result
