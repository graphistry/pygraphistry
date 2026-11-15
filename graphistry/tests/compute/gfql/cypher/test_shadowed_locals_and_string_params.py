"""Read-side regressions: comprehension locals shadow matched aliases (#2325); scalar params are literals (#2324)."""
from __future__ import annotations

import importlib.util
import os
from typing import Any, List, Optional

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

# native Polars' parity-or-error decline for row operations it lacks: asserted, never skipped
POLARS_ROW_OP = r"does not yet natively support cypher row op"


def _graph(engine: str) -> Any:
    frame = pd.DataFrame({"id": [1, 2], "tags": ["a", "b"]})
    if engine == "cudf":
        import cudf
        frame = cudf.from_pandas(frame)
    elif engine == "polars":
        frame = pytest.importorskip("polars").from_pandas(frame)
    return graphistry.nodes(frame, "id")


def _rows(frame: Any) -> List[dict[str, Any]]:
    if isinstance(frame, pd.DataFrame):
        return frame.to_dict(orient="records")
    return frame.to_arrow().to_pylist()


def _check(
    engine: str,
    query: str,
    expected: List[dict[str, Any]],
    params: Any = None,
    polars_declines: Optional[str] = None,
    ordered: bool = True,
) -> None:
    """Rows of a read query equal ``expected``; with ``polars_declines`` native Polars must raise exactly that decline."""
    graph = _graph(engine)
    if engine == "polars" and polars_declines is not None:
        with pytest.raises(NotImplementedError, match=polars_declines):
            graph.gfql(query, engine=engine, params=params)
        return
    rows = _rows(graph.gfql(query, engine=engine, params=params)._nodes)
    if not ordered:
        rows, expected = sorted(rows, key=repr), sorted(expected, key=repr)
    assert rows == expected


def _declined(engine: str, query: str, value: str) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        _graph(engine).gfql(query, engine=engine)
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["value"] == value


# #2325: a comprehension or quantifier variable shadows a matched alias inside its scope.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression,expected", [
    ("[n IN ['x'] | [n]][0][0]", "x"),
    ("[n IN ['x'] | {k: n}][0].k", "x"),
    ("[n IN [1, 2] | [n IN ['a'] | n][0]][1]", "a"),
    ("head([n IN ['x'] | [n]])[0]", "x"),
])
def test_comprehension_local_shadows_the_matched_alias_in_list_and_map_positions(engine, expression, expected) -> None:
    _check(engine, f"MATCH (n) RETURN {expression} AS v", [{"v": expected}] * 2, polars_declines=POLARS_ROW_OP)


@pytest.mark.parametrize("engine", ENGINES)
def test_quantifier_local_shadows_the_matched_alias(engine) -> None:
    _check(
        engine, "MATCH (n) RETURN any(n IN ['x'] WHERE [n][0] = 'x') AS v", [{"v": True}] * 2,
        polars_declines=POLARS_ROW_OP,
    )


# a property of a shadowing local must not read the outer node's column, with one or several matched aliases
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("match", ["(n)", "(n),(m)"])
def test_property_of_a_comprehension_local_does_not_read_the_outer_alias(engine, match) -> None:
    rows = 2 if match == "(n)" else 4
    _check(
        engine, f"MATCH {match} RETURN [n IN [{{tags: 'z'}}] | n.tags][0] AS v", [{"v": "z"}] * rows,
        polars_declines=POLARS_ROW_OP,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_property_of_a_quantifier_local_does_not_read_the_outer_alias(engine) -> None:
    _check(
        engine, "MATCH (n),(m) RETURN any(n IN [{tags: 'z'}] WHERE n.tags = 'z') AS v", [{"v": True}] * 4,
        polars_declines=POLARS_ROW_OP,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_shadowing_is_scoped_the_outer_alias_still_resolves_next_to_the_local(engine) -> None:
    _check(
        engine, "MATCH (n),(m) RETURN [m IN [{tags: 'z'}] | m.tags][0] AS v, n.tags AS t, m.tags AS u",
        [{"v": "z", "t": t, "u": u} for t in "ab" for u in "ab"], polars_declines=POLARS_ROW_OP, ordered=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_unshadowed_outer_alias_still_resolves_next_to_a_local(engine) -> None:
    _check(
        engine, "MATCH (n) RETURN [m IN ['x'] | m][0] AS v, n.tags AS t",
        [{"v": "x", "t": "a"}, {"v": "x", "t": "b"}], polars_declines=POLARS_ROW_OP,
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("match", ["(n)", "(n),(m)"])
def test_outer_alias_property_inside_a_comprehension_over_another_variable(engine, match) -> None:
    rows = [{"v": [1, 1]}, {"v": [2, 2]}] if match == "(n)" else [{"v": [i, i]} for i in (1, 1, 2, 2)]
    _check(engine, f"MATCH {match} RETURN [x IN [1, 2] | n.id] AS v", rows, polars_declines=POLARS_ROW_OP, ordered=False)


@pytest.mark.parametrize("engine", ENGINES)
def test_comprehension_variable_does_not_leak_into_outer_alias_keys(engine) -> None:
    _check(
        engine, "MATCH (n) RETURN [x IN [1, 2] | size(keys(n))][0] AS v",
        [{"v": 2}] * 2, polars_declines=POLARS_ROW_OP,
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression", [
    "size([n IN [1] | keys(n)][0])",
    "[n IN ['x'] | labels(n)][0]",
    "any(n IN [1, 2] WHERE keys(n) = [])",
    "0 < size([n IN [1] | keys(n)][0]) < 5",
    "n:Foo AND size([n IN [1] | keys(n)][0]) = 0",
    "size([n IN [1] | labels(n)][0]) >= 0 AND n:Foo",
])
def test_keys_and_labels_of_a_shadowing_local_are_declined_not_answered_from_the_outer_node(engine, expression) -> None:
    _declined(engine, f"MATCH (n) RETURN {expression} AS v", "shadowed_alias")


# #2324: a scalar parameter is a literal in every row-expression position.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value", ["x", "tags", "id", "n.tags"])
def test_string_parameter_is_a_literal_in_return(engine, value) -> None:
    _check(engine, "MATCH (n) RETURN $v AS v", [{"v": value}] * 2, {"v": value})


@pytest.mark.parametrize("engine", ENGINES)
def test_string_parameter_equal_to_a_column_name_is_a_literal_in_with(engine) -> None:
    _check(engine, "MATCH (n) WITH $v AS w RETURN w", [{"w": "tags"}] * 2, {"v": "tags"})


@pytest.mark.parametrize("engine", ENGINES)
def test_string_parameter_in_a_predicate_and_in_an_expression(engine) -> None:
    _check(
        engine, "MATCH (n) WHERE n.tags = $v RETURN n.id AS id, $v + 'y' AS s", [{"id": 1, "s": "ay"}], {"v": "a"},
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value", ["x", "tags", "it's \"q\" \\ \n x"])
def test_string_parameter_in_collect_and_as_a_grouping_key_and_in_unwind(engine, value) -> None:
    params = {"v": value}
    _check(engine, "MATCH (n) RETURN collect($v) AS v", [{"v": [value, value]}], params)
    _check(engine, "MATCH (n) RETURN $v AS k, count(*) AS c", [{"k": value, "c": 2}], params)
    _check(engine, "UNWIND [$v] AS u RETURN u", [{"u": value}], params)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value", [5, 2.5, True, None])
def test_non_string_parameters_in_return_are_unchanged(engine, value) -> None:
    _check(engine, "MATCH (n) RETURN $v AS v", [{"v": value}] * 2, {"v": value})


# a scalar parameter projected next to a whole entity keeps its value through the next stage
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value", [5, 2.5, True, None, "x"])
def test_scalar_parameter_projected_next_to_a_whole_entity(engine, value) -> None:
    _check(engine, "MATCH (n) WITH n, $v AS w RETURN w", [{"w": value}] * 2, {"v": value})


@pytest.mark.parametrize("engine", ENGINES)
def test_integer_parameter_projected_next_to_a_whole_entity_is_usable_downstream(engine) -> None:
    _check(engine, "MATCH (n) WITH n, $v AS w RETURN w + 1 AS r", [{"r": 6}] * 2, {"v": 5})
