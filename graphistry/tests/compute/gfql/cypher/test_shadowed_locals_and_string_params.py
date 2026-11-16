"""Read-side regressions: comprehension locals shadow matched aliases (#2325); scalar params are literals (#2324)."""
from __future__ import annotations

import importlib.util
import math
import os
from typing import Any, List, Optional

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import (
    ErrorCode, GFQLCallExecutionError, GFQLTypeError, GFQLUnsupportedError, GFQLValidationError,
)


ENGINES = [
    pytest.param("pandas"),
    pytest.param("cudf", marks=pytest.mark.skipif(
        os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"
    )),
    pytest.param("polars", marks=pytest.mark.skipif(
        importlib.util.find_spec("polars") is None, reason="polars is not installed",
    )),
    pytest.param("polars-gpu", marks=pytest.mark.skipif(
        os.environ.get("TEST_POLARS_GPU") != "1", reason="requires actual polars-gpu (TEST_POLARS_GPU=1)",
    )),
]
POLARS_ENGINES = ("polars", "polars-gpu")

# native Polars' parity-or-error declines (a row operation it lacks, or a plan polars-gpu cannot run on the GPU): asserted, never skipped
POLARS_ROW_OP = r"does not yet natively support cypher row op"
GPU_DECLINE = "not fully GPU-executable"


def _graph(engine: str) -> Any:
    frame = pd.DataFrame({"id": [1, 2], "tags": ["a", "b"]})
    if engine == "cudf":
        import cudf
        frame = cudf.from_pandas(frame)
    elif engine in POLARS_ENGINES:
        frame = pytest.importorskip("polars").from_pandas(frame)
    return graphistry.nodes(frame, "id")


def _graph3(engine: str) -> Any:
    """Three nodes joined by two ``T`` edges, for edge-alias and multi-alias shapes."""
    nodes = pd.DataFrame({"id": [1, 2, 3], "tags": ["a", "b", "c"]})
    edges = pd.DataFrame({"s": [1, 2], "d": [2, 3], "type": ["T", "T"]})
    if engine == "cudf":
        import cudf
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in POLARS_ENGINES:
        polars = pytest.importorskip("polars")
        nodes, edges = polars.from_pandas(nodes), polars.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _rows(frame: Any) -> List[dict[str, Any]]:
    if isinstance(frame, pd.DataFrame):
        return frame.to_dict(orient="records")
    return frame.to_arrow().to_pylist()


def _gfql_rows(engine: str, graph: Any, query: str, params: Any = None) -> Optional[List[dict[str, Any]]]:
    """Rows, or None when polars-gpu declines with its named GPU decline (older cudf-polars cannot run some plans on the GPU)."""
    try:
        return _rows(graph.gfql(query, engine=engine, params=params)._nodes)
    except NotImplementedError as exc:  # GFQLUnsupportedError is a NotImplementedError
        if engine == "polars-gpu" and GPU_DECLINE in str(exc):
            return None
        raise


def _run(engine: str, query: str, params: Any = None, polars_declines: Optional[str] = None) -> Optional[List[dict[str, Any]]]:
    """Rows of a read query; with ``polars_declines`` native Polars must raise exactly that decline (returns None)."""
    graph = _graph(engine)
    if engine in POLARS_ENGINES and polars_declines is not None:
        with pytest.raises(NotImplementedError, match=polars_declines):
            graph.gfql(query, engine=engine, params=params)
        return None
    return _gfql_rows(engine, graph, query, params)


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
    if engine in POLARS_ENGINES and polars_declines is not None:
        decline = f"{GPU_DECLINE}|{polars_declines}" if engine == "polars-gpu" else polars_declines
        with pytest.raises(NotImplementedError, match=decline):
            graph.gfql(query, engine=engine, params=params)
        return
    rows = _gfql_rows(engine, graph, query, params)
    if rows is None:
        return
    if not ordered:
        rows, expected = sorted(rows, key=repr), sorted(expected, key=repr)
    assert rows == expected


def _declined(engine: str, query: str, value: str, graph: Any | None = None) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        (graph or _graph(engine)).gfql(query, engine=engine)
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"].startswith("return")
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
    if engine == "cudf" and expression.startswith("head("):
        import cudf
        if int(cudf.__version__.split(".")[0]) < 26:
            pytest.skip("cuDF < 26: head() over a list comprehension fails in the read pipeline at master too (#2337)")
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
    if engine == "polars-gpu":
        import cudf
        if int(cudf.__version__.split(".")[0]) < 26:
            for query in ("MATCH (n) RETURN collect($v) AS v", "MATCH (n) RETURN $v AS k, count(*) AS c"):
                with pytest.raises(GFQLUnsupportedError) as exc:
                    _graph(engine).gfql(query, engine=engine, params=params)
                assert exc.value.code == ErrorCode.E110
                assert exc.value.context["field"] == "function"
                assert exc.value.context["value"] == "group_by"
            _check(engine, "UNWIND [$v] AS u RETURN u", [{"u": value}], params)
            return
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
@pytest.mark.parametrize("value,expected_dtype", [
    (5, {"pandas": "int64", "cudf": "int64", "polars": "int32", "polars-gpu": "int32"}),
    (2.5, {"pandas": "float64", "cudf": "float64", "polars": "float64", "polars-gpu": "float64"}),
    (True, {"pandas": "bool", "cudf": "bool", "polars": "boolean", "polars-gpu": "boolean"}),
    ("x", {"pandas": "object", "cudf": "object", "polars": "string", "polars-gpu": "string"}),
])
def test_scalar_parameter_projection_keeps_its_dtype(engine, value, expected_dtype) -> None:
    out = _graph(engine).gfql("MATCH (n) RETURN $v AS v", engine=engine, params={"v": value})
    frame = out._nodes
    dtype = str(frame.schema["v"]).lower() if hasattr(frame, "schema") else str(frame.dtypes["v"]).lower()
    assert dtype == expected_dtype[engine]


@pytest.mark.parametrize("engine", ENGINES)
def test_integer_parameter_projected_next_to_a_whole_entity_is_usable_downstream(engine) -> None:
    _check(engine, "MATCH (n) WITH n, $v AS w RETURN w + 1 AS r", [{"r": 6}] * 2, {"v": 5})


# A local lives only in an internal column, so it can never collide with a property column or an alias.
@pytest.mark.parametrize("engine", ENGINES)
def test_local_named_like_an_outer_property_does_not_change_keys_of_the_entity(engine) -> None:
    _check(
        engine, "MATCH (n) RETURN [tags IN [1] | size(keys(n))][0] AS v", [{"v": 2}] * 2, polars_declines=POLARS_ROW_OP,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_local_named_like_a_property_does_not_clobber_the_property_column(engine) -> None:
    _check(
        engine, "MATCH (n) RETURN [tags IN ['q'] | n.tags][0] AS v", [{"v": "a"}, {"v": "b"}], polars_declines=POLARS_ROW_OP,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_quantifier_local_named_like_a_property_filters_by_the_outer_property(engine) -> None:
    _check(
        engine, "MATCH (n) WHERE any(id IN $ids WHERE id = n.id) RETURN n.id AS id", [{"id": 1}], {"ids": [1]},
        polars_declines=POLARS_ROW_OP,
    )
    _check(
        engine, "MATCH (n) WHERE any(tags IN $names WHERE n.tags = tags) RETURN n.id AS id", [{"id": 1}],
        {"names": ["a"]}, polars_declines=POLARS_ROW_OP,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_comprehension_local_does_not_leak_into_properties_of_the_outer_alias(engine) -> None:
    rows = _run(
        engine, "MATCH (n) RETURN [x IN ['q'] | properties(n)][0] AS p, properties(n) AS q", polars_declines=POLARS_ROW_OP,
    )
    if rows is not None:
        assert rows and all(row["p"] == row["q"] for row in rows)


@pytest.mark.parametrize("engine", ENGINES)
def test_nested_scopes_bind_the_innermost_local(engine) -> None:
    _check(
        engine, "MATCH (n) WHERE n.id = 1 RETURN [x IN [1, 2] | [x IN [10, 20] | x]][0] AS v", [{"v": [10, 20]}],
        polars_declines=POLARS_ROW_OP,
    )
    _check(
        engine, "MATCH (n) WHERE n.id = 1 RETURN [x IN [1, 2] | [y IN [10, 20] | x + y]] AS v", [{"v": [[11, 21], [12, 22]]}],
        polars_declines=POLARS_ROW_OP,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_entity_local_reads_its_bound_value(engine) -> None:
    _check(
        engine, "MATCH (n) RETURN [x IN [n] | x.tags][0] AS v", [{"v": "a"}, {"v": "b"}],
        polars_declines=POLARS_ROW_OP,
    )


# Non-finite and edge floats and integers survive the whole-parameter projection next to an entity.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), -0.0, 2 ** 53 + 1])
def test_whole_parameter_projected_next_to_an_entity_keeps_edge_numbers(engine, value) -> None:
    rows = _run(engine, "MATCH (n) WITH n, $v AS w RETURN w", {"v": value})
    assert rows is not None and [row["w"] for row in rows] == [value, value]
    if value == 0:
        assert all(math.copysign(1.0, row["w"]) == -1.0 for row in rows)


def normalize(cell: Any) -> Any:
    return cell.tolist() if hasattr(cell, "tolist") else cell


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value", [[1, "tags"], {"k": "tags"}])
def test_list_and_map_parameters_projected_next_to_an_entity(engine, value) -> None:
    rows = _run(engine, "MATCH (n) WITH n, $v AS w RETURN w", {"v": value}, polars_declines=POLARS_ROW_OP)
    assert rows is None or [normalize(row["w"]) for row in rows] == [value] * 2


@pytest.mark.parametrize("engine", ENGINES)
def test_unwind_of_a_scalar_string_parameter_is_a_literal(engine) -> None:
    _check(engine, "UNWIND $v AS u RETURN u", [{"u": "tags"}], {"v": "tags"}, polars_declines=r"does not yet natively support cypher row op 'unwind'")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value", ["", "é"])
def test_empty_and_non_ascii_string_parameters_are_literals(engine, value) -> None:
    _check(engine, "MATCH (n) RETURN $v AS v", [{"v": value}] * 2, {"v": value})


# Every entity-recognizing branch defers to the scope-local column, so a shadowing local is never the outer node.
@pytest.mark.parametrize("engine", ENGINES)
def test_subscript_of_a_shadowing_local_reads_the_local_not_the_outer_node(engine) -> None:
    _check(
        engine, "MATCH (n) RETURN [n IN [{tags: 'z'}] | n['tags']][0] AS v", [{"v": "z"}] * 2, polars_declines=POLARS_ROW_OP,
    )


def _expect(engine, graph, query, params=None, *, rows=None, error=None, polars_native=False):
    """One outcome per engine: rows, or ``error=(exception type, ErrorCode or None)``; Polars declines unless ``polars_native``."""
    if engine in POLARS_ENGINES and not polars_native:
        decline = f"{GPU_DECLINE}|{POLARS_ROW_OP}" if engine == "polars-gpu" else POLARS_ROW_OP
        with pytest.raises(NotImplementedError, match=decline):
            graph.gfql(query, engine=engine, params=params)
        return
    if error is not None:
        exc_type, code = error
        with pytest.raises(exc_type) as exc:
            graph.gfql(query, engine=engine, params=params)
        if code is not None:
            assert exc.value.code == code
        return
    got = _gfql_rows(engine, graph, query, params)
    if got is not None:
        assert sorted(got, key=repr) == sorted(rows, key=repr)


# generalization: locals in every clause position, with empty/null sources, on a 3-node graph with edges
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,rows", [
    ("MATCH ()-[r]->() RETURN [r IN [1] | r] AS v", [{"v": [1]}] * 2),
    ("MATCH (n) WHERE any(n IN [1] WHERE n = 1) RETURN n.id AS id", [{"id": 1}, {"id": 2}, {"id": 3}]),
    ("MATCH (n) WITH n.id AS x RETURN [x IN [9] | x] AS v", [{"v": [9]}] * 3),
    (
        "MATCH (n) WHERE n.id = 1 RETURN all(n IN [1, 2] WHERE n > 0) AS a, none(n IN [1, 2] WHERE n > 5) AS b,"
        " single(n IN [1, 2] WHERE n = 1) AS c",
        [{"a": True, "b": True, "c": True}],
    ),
    ("MATCH (n) WHERE n.id = 1 RETURN [n IN [] | n] AS e, [n IN null | n] AS z", [{"e": [], "z": None}]),
])
def test_shadowing_locals_in_every_clause_position(engine, query, rows) -> None:
    _expect(engine, _graph3(engine), query, rows=rows)


@pytest.mark.parametrize("engine", ENGINES)
def test_label_predicate_on_a_shadowing_local_is_a_structured_error(engine) -> None:
    """master returned null (it read the outer node's label flags); a local integer has no labels."""
    _expect(
        engine, _graph3(engine), "MATCH (n) WHERE n.id = 1 RETURN [n IN [1] | n:Foo] AS v",
        error=(GFQLCallExecutionError, ErrorCode.E306),
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_numeric_local_named_like_an_alias_is_not_taken_for_the_node_marker(engine) -> None:
    """master returned the outer tags (the local 1 read as the node marker); now a structured error."""
    _expect(
        engine, _graph(engine), "MATCH (n) RETURN [n IN [1] | n.tags][0] AS v", error=(GFQLCallExecutionError, ErrorCode.E306),
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_properties_of_a_shadowing_local_is_never_the_outer_nodes_properties(engine) -> None:
    """The evaluator has no properties() of a local map. pandas: structured E306. cuDF: a bare NotImplementedError from
    cudf Column.as_string_column (cuDF cannot cast a struct column to string); tracked in #2339."""
    _expect(
        engine, _graph(engine), "MATCH (n) RETURN [n IN [{a: 1}] | properties(n)][0] AS p",
        error=(NotImplementedError, None) if engine == "cudf" else (GFQLCallExecutionError, ErrorCode.E306),
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_type_of_a_shadowing_local_is_not_the_outer_relationship(engine) -> None:
    """master returned the outer relationship's type ('T'); now a structured error."""
    _expect(
        engine, _graph3(engine), "MATCH ()-[r]->() RETURN [r IN [1] | type(r)][0] AS t",
        error=(GFQLCallExecutionError, ErrorCode.E306),
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_shadowing_local_in_a_multi_alias_path_is_a_structured_decline(engine) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        _graph3(engine).gfql(
            "MATCH (a)-[r]->(b) RETURN [a IN [{tags: 'z'}] | a.tags][0] AS v, b.tags AS bt", engine=engine,
        )
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "return"


# string and non-finite parameters in WHERE / ORDER BY
@pytest.mark.parametrize("engine", ENGINES)
def test_string_parameter_in_where_and_order_by(engine) -> None:
    graph = _graph3(engine)
    _expect(engine, graph, "MATCH (n) WHERE n.tags = $v RETURN n.id AS id", {"v": "b"}, rows=[{"id": 2}], polars_native=True)
    if engine in POLARS_ENGINES:
        # native Polars cannot order by a constant parameter (a structured decline at master too; the exact error varies by version)
        with pytest.raises((NotImplementedError, GFQLValidationError, GFQLTypeError)):
            graph.gfql("MATCH (n) RETURN n.id AS id ORDER BY $v DESC, n.id", engine=engine, params={"v": "x"})
    else:
        got = _rows(graph.gfql("MATCH (n) RETURN n.id AS id ORDER BY $v DESC, n.id", engine=engine, params={"v": "x"})._nodes)
        assert got == [{"id": 1}, {"id": 2}, {"id": 3}]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value,ids", [(float("inf"), [1, 2, 3]), (float("-inf"), [])])
def test_infinite_parameter_in_where_compares_as_infinity(engine, value, ids) -> None:
    _expect(
        engine, _graph3(engine), "MATCH (n) WHERE n.id < $v RETURN n.id AS id", {"v": value},
        rows=[{"id": i} for i in ids], polars_native=True,
    )


# NaN parameters: raw in a plain RETURN (as master), null inside an expression (as master), a structured E108 through WITH
# or next to a whole entity (where master failed or, with the new rendering, would have silently returned null)
@pytest.mark.parametrize("engine", ENGINES)
def test_nan_parameter_in_a_plain_return_stays_nan(engine) -> None:
    rows = _run(engine, "MATCH (n) WHERE n.id = 1 RETURN $v AS v", {"v": float("nan")})
    assert rows is not None and len(rows) == 1 and math.isnan(rows[0]["v"])


@pytest.mark.parametrize("engine", ENGINES)
def test_nan_parameter_inside_an_expression_is_null(engine) -> None:
    _check(engine, "MATCH (n) WHERE n.id = 1 RETURN $v + 1 AS v", [{"v": None}], {"v": float("nan")})


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", [
    "MATCH (n) WITH n, $v AS w RETURN w",
    "MATCH (n) WITH $v AS w RETURN w",
])
def test_nan_parameter_through_with_is_a_structured_decline(engine, query) -> None:
    with pytest.raises(GFQLValidationError) as exc:
        _graph(engine).gfql(query, engine=engine, params={"v": float("nan")})
    assert exc.value.code == ErrorCode.E108 and exc.value.context["value"] == "nan_parameter"
