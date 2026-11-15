"""Read-side regressions: comprehension locals shadow matched aliases (#2325); string params are literals (#2324)."""
from __future__ import annotations

import importlib.util
import os
from typing import Any

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError


ENGINES = [
    pytest.param("pandas"),
    pytest.param("cudf", marks=pytest.mark.skipif(
        os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"
    )),
    pytest.param("polars", marks=pytest.mark.skipif(
        importlib.util.find_spec("polars") is None, reason="polars is not installed",
    )),
]


def _graph(engine: str) -> Any:
    frame = pd.DataFrame({"id": [1, 2], "tags": ["a", "b"]})
    if engine == "cudf":
        import cudf
        frame = cudf.from_pandas(frame)
    elif engine == "polars":
        frame = pytest.importorskip("polars").from_pandas(frame)
    return graphistry.nodes(frame, "id")


def _rows(frame: Any) -> list[dict[str, Any]]:
    if isinstance(frame, pd.DataFrame):
        return frame.to_dict(orient="records")
    return frame.to_arrow().to_pylist()


def _run(engine: str, query: str, params: Any = None) -> list[dict[str, Any]]:
    """Rows of a read query; native Polars raises its parity-or-error NotImplementedError for forms it lacks."""
    try:
        return _rows(_graph(engine).gfql(query, engine=engine, params=params)._nodes)
    except NotImplementedError:
        if engine != "polars":
            raise
        pytest.skip("native Polars row pipeline does not support this row expression (parity-or-error)")
        raise  # unreachable


# #2325: a comprehension or quantifier variable shadows a matched alias inside its scope.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression,expected", [
    ("[n IN ['x'] | [n]][0][0]", "x"),
    ("[n IN ['x'] | {k: n}][0].k", "x"),
    ("[n IN [1, 2] | [n IN ['a'] | n][0]][1]", "a"),
    ("head([n IN ['x'] | [n]])[0]", "x"),
])
def test_comprehension_local_shadows_the_matched_alias_in_list_and_map_positions(engine, expression, expected) -> None:
    rows = _run(engine, f"MATCH (n) RETURN {expression} AS v")
    assert [row["v"] for row in rows] == [expected, expected]


@pytest.mark.parametrize("engine", ENGINES)
def test_quantifier_local_shadows_the_matched_alias(engine) -> None:
    rows = _run(engine, "MATCH (n) RETURN any(n IN ['x'] WHERE [n][0] = 'x') AS v")
    assert [row["v"] for row in rows] == [True, True]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression", [
    "size([n IN [1] | keys(n)][0])",
    "[n IN ['x'] | labels(n)][0]",
    "any(n IN [1, 2] WHERE keys(n) = [])",
])
def test_keys_and_labels_of_a_shadowing_local_are_declined_not_answered_from_the_outer_node(engine, expression) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        _graph(engine).gfql(f"MATCH (n) RETURN {expression} AS v", engine=engine)
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["value"] == "shadowed_alias"


@pytest.mark.parametrize("engine", ENGINES)
def test_unshadowed_outer_alias_still_resolves_next_to_a_local(engine) -> None:
    rows = _run(engine, "MATCH (n) RETURN [m IN ['x'] | m][0] AS v, n.tags AS t")
    assert [(row["v"], row["t"]) for row in rows] == [("x", "a"), ("x", "b")]


# #2324: a string parameter is a string literal in every row-expression position.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value", ["x", "tags", "id", "n.tags"])
def test_string_parameter_is_a_literal_in_return(engine, value) -> None:
    rows = _run(engine, "MATCH (n) RETURN $v AS v", {"v": value})
    assert [row["v"] for row in rows] == [value, value]


@pytest.mark.parametrize("engine", ENGINES)
def test_string_parameter_equal_to_a_column_name_is_a_literal_in_with(engine) -> None:
    rows = _run(engine, "MATCH (n) WITH $v AS w RETURN w", {"v": "tags"})
    assert [row["w"] for row in rows] == ["tags", "tags"]


@pytest.mark.parametrize("engine", ENGINES)
def test_string_parameter_in_a_predicate_and_in_an_expression(engine) -> None:
    rows = _run(engine, "MATCH (n) WHERE n.tags = $v RETURN n.id AS id, $v + 'y' AS s", {"v": "a"})
    assert rows == [{"id": 1, "s": "ay"}]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value", [5, 2.5, True, None])
def test_non_string_parameters_in_return_are_unchanged(engine, value) -> None:
    rows = _run(engine, "MATCH (n) RETURN $v AS v", {"v": value})
    assert [row["v"] for row in rows] == [value, value]
