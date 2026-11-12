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
    pytest.param("polars"),
]


def _records(frame: Any) -> list[dict[str, Any]]:
    if isinstance(frame, pd.DataFrame):
        rows = frame.to_dict(orient="records")
    else:
        rows = frame.to_arrow().to_pylist()
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
    elif engine == "polars":
        polars = pytest.importorskip("polars")
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
    assert source.gfql_graph is source
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_create_nodes_without_existing_nodes_preserves_inferred_graph(engine: str) -> None:
    edges = pd.DataFrame({"src": [7], "dst": [8], "type": ["R"]})
    source = graphistry.edges(edges, "src", "dst")

    result = source.gfql("CREATE (a:L {name: 'x'})", engine=engine, language="cypher")

    assert _records(result._nodes) == [
        {"id": 7, "label__L": False, "name": None},
        {"id": 8, "label__L": False, "name": None},
        {"id": 9, "label__L": True, "name": "x"},
    ]
    assert _records(result._edges) == [{"src": 7, "dst": 8, "type": "R"}]
    assert source._nodes is None
    count = result.gfql("MATCH (a)-[r]->(b) RETURN count(*) AS count", engine=engine)
    assert _records(count._nodes) == [{"count": 1}]
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


@pytest.mark.parametrize("engine", ENGINES)
def test_create_preserves_fractional_new_values_and_original_frame(engine: str) -> None:
    source = _graph(engine)
    nodes = source._nodes
    if engine == "polars":
        import polars as pl
        nodes = nodes.with_columns(pl.lit(1).alias("score"))
    else:
        nodes = nodes.assign(score=1)
    source = source.nodes(nodes)
    before = _records(source._nodes)
    result = source.gfql("CREATE (a {score: 1.5})", engine=engine)
    assert [row["score"] for row in _records(result._nodes)] == [1, 1, 1.5]
    assert _records(source._nodes) == before
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_sparse_created_integer_is_not_rounded_before_frame_construction(engine: str) -> None:
    wide = (1 << 60) + 1
    result = _graph(engine).gfql(
        "CREATE (a:Wide {serial: $wide}), (b:Other {name: 'other'})",
        params={"wide": wide}, engine=engine,
    )
    serials = [row["serial"] for row in _records(result._nodes) if row["serial"] is not None]
    assert serials == [wide]
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_create_declines_boolean_numeric_column_union(engine: str) -> None:
    source = _graph(engine)
    nodes = source._nodes
    if engine == "polars":
        import polars as pl
        nodes = nodes.with_columns(pl.lit(True).alias("flag"))
    else:
        nodes = nodes.assign(flag=True)
    source = source.nodes(nodes)
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (a {flag: 2})", engine=engine)
    assert exc.value.context["field"] == "flag"
    assert _records(source._nodes) == before


@pytest.mark.parametrize("dtype,maximum", [("int8", 127), ("int16", 32767), ("int64", (1 << 63) - 1)])
def test_create_checks_the_entire_allocated_id_range(dtype: str, maximum: int) -> None:
    source = graphistry.nodes(pd.DataFrame({"id": pd.Series([maximum - 1], dtype=dtype)}), "id")
    before = source._nodes.copy(deep=True)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (a), (b)", engine="pandas")
    assert exc.value.context["field"] == "id"
    pd.testing.assert_frame_equal(source._nodes, before)


@pytest.mark.parametrize("suffix", [
    "CALL graphistry.degree.write()",
    "UNWIND [1] AS x",
    "MATCH (n) RETURN n",
    "WITH 1 AS x RETURN x",
])
def test_create_rejects_the_entire_unsupported_mixed_program(suffix: str) -> None:
    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    with pytest.raises(GFQLValidationError):
        source.gfql(f"CREATE (a:L) {suffix}")
    pd.testing.assert_frame_equal(source._nodes, before)


def test_strict_polars_gpu_creation_has_explicit_capability_decline() -> None:
    from graphistry.compute.exceptions import ErrorCode, GFQLUnsupportedError
    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    with pytest.raises(GFQLUnsupportedError) as exc:
        source.gfql("CREATE (a:L)", engine="polars-gpu")
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "engine"
    pd.testing.assert_frame_equal(source._nodes, before)


@pytest.mark.parametrize("transform", ["nodes", "edges", "bind"])
def test_ordinary_and_transformed_graph_accessors_return_their_own_graph(transform: str) -> None:
    source = _graph("pandas")
    assert source.gfql_graph is source
    created = source.gfql("CREATE (a:L)")
    if transform == "nodes":
        result = created.nodes(created._nodes.iloc[:1].copy())
    elif transform == "edges":
        result = created.edges(created._edges.iloc[:0].copy())
    else:
        result = created.bind()
    assert result is not created and result.gfql_graph is result
    assert created.gfql_graph is created and len(created._nodes) == 3


def test_create_preflight_is_validate_only_and_matches_execution(monkeypatch) -> None:
    import graphistry.compute.gfql_unified as unified
    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    with monkeypatch.context() as context:
        def reject_execution(*args, **kwargs):
            pytest.fail("preflight dispatched a mutation executor")
        context.setattr(unified, "execute_create_nodes", reject_execution)
        assert source.gfql_validate("CREATE (a:L)")["compiled_kind"] == "mutation"
    pd.testing.assert_frame_equal(source._nodes, before)
    ordinary = source.gfql("CREATE (a:L)")
    checked = source.gfql("CREATE (a:L)", validate=True)
    pd.testing.assert_frame_equal(ordinary._nodes, checked._nodes)


@pytest.mark.parametrize("validate", [False, True])
def test_remote_mutation_declines_before_any_request(monkeypatch, validate: bool) -> None:
    from importlib import import_module
    remote_module = import_module("graphistry.compute.chain_remote")
    def reject_io(*args, **kwargs):
        pytest.fail("unsupported remote mutation reached HTTP")
    monkeypatch.setattr(remote_module.requests, "post", reject_io)
    with pytest.raises(GFQLValidationError) as exc:
        _graph("pandas").gfql_remote(
            "CREATE (a:L)", dataset_id="fixture", api_token="fixture-token",
            format="json", validate=validate,
        )
    assert exc.value.context["value"] == "mutation"


@pytest.mark.parametrize("engine", ENGINES)
def test_create_converts_pandas_input_to_the_explicit_native_engine(engine: str) -> None:
    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    result = source.gfql("CREATE (a:L)", engine=engine)
    assert len(result._nodes) == 3
    _assert_served(engine, result._nodes)
    _assert_served(engine, result._edges)
    pd.testing.assert_frame_equal(source._nodes, before)


@pytest.mark.parametrize("engine", ENGINES)
def test_wide_existing_integer_with_missing_new_property_remains_exact(engine: str) -> None:
    source = _graph(engine)
    wide = (1 << 60) + 1
    if engine == "polars":
        import polars as pl
        nodes = source._nodes.with_columns(pl.lit(wide).alias("serial"))
    else:
        nodes = source._nodes.assign(serial=wide)
    source = source.nodes(nodes)
    result = source.gfql("CREATE (a:L)", engine=engine)
    assert [row["serial"] for row in _records(result._nodes)] == [wide, wide, None]


@pytest.mark.parametrize("engine", ENGINES)
def test_new_float_does_not_round_existing_wide_integer(engine: str) -> None:
    source = _graph(engine)
    wide = (1 << 60) + 1
    if engine == "polars":
        import polars as pl
        nodes = source._nodes.with_columns(pl.lit(wide).alias("serial"))
    else:
        nodes = source._nodes.assign(serial=wide)
    source = source.nodes(nodes)
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (a {serial: 1.5})", engine=engine)
    assert exc.value.context["field"] == "serial"
    assert _records(source._nodes) == before
