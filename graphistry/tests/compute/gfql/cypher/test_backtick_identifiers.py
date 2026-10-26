from __future__ import annotations

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.identifiers import CYPHER_RESERVED_WORDS
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.expr_parser import Identifier, parse_expr
from graphistry.compute.gfql.cypher.expression_text import render_expr_node


@pytest.fixture
def odd_columns_graph():
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "analysis.group.Score": [0.2, 0.9],
            "a-b": [10, 20],
            "a`b": [1, 2],
            "a b": [30, 40],
        }
    )
    edges = pd.DataFrame({"src": ["a", "b"], "dst": ["b", "a"]})
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def test_backtick_property_parser_preserves_dotted_name() -> None:
    parsed = parse_cypher("MATCH (n) RETURN n.`analysis.group.Score` AS x")

    assert parsed.return_.items[0].expression.text == "n.`analysis.group.Score`"
    assert parsed.return_.items[0].alias == "x"


def test_backtick_expression_ast_preserves_dotted_and_escaped_names() -> None:
    dotted = parse_expr("n.`analysis.group.Score`")
    escaped = parse_expr("n.`a``b`")

    assert dotted.property == "analysis.group.Score"
    assert escaped.property == "a`b"


def test_backtick_identifiers_execute_on_pandas(odd_columns_graph) -> None:
    out = odd_columns_graph.gfql(
        "MATCH (n) WHERE n.`analysis.group.Score` > 0.5 RETURN n.`a-b` AS y",
        engine="pandas",
    )

    assert out._nodes["y"].tolist() == [20]


def test_backtick_escape_executes_on_pandas(odd_columns_graph) -> None:
    out = odd_columns_graph.gfql(
        "MATCH (n) RETURN n.`a``b` AS escaped",
        engine="pandas",
    )

    assert out._nodes["escaped"].tolist() == [1, 2]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("MATCH (n:`Person X`) RETURN n AS x", ("Person X",)),
        ("MATCH (a)-[r:`REL X`]->(b) RETURN r", ("REL X",)),
        ("UNWIND [1, 2] AS `y.z` RETURN `y.z`", "y.z"),
    ],
)
def test_backtick_labels_types_and_aliases_parse(query, expected):
    parsed = parse_cypher(query)

    if query.startswith("MATCH (a)"):
        assert parsed.matches[0].patterns[0][1].types == expected
    elif isinstance(expected, tuple):
        assert parsed.matches[0].patterns[0][0].labels == expected
    else:
        assert parsed.unwinds[0].alias == expected


def test_backtick_alias_survives_where_order_and_projection(odd_columns_graph):
    out = odd_columns_graph.gfql(
        "MATCH (n) "
        "WITH n.`a-b` AS `y.z` "
        "WHERE `y.z` > 10 "
        "ORDER BY `y.z` DESC "
        "RETURN `y.z` AS out",
        engine="pandas",
    )

    assert out._nodes["out"].tolist() == [20]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_quoted_return_alias_survives_order(odd_columns_graph, engine):
    if engine == "polars":
        pytest.importorskip("polars")

    out = odd_columns_graph.gfql(
        "MATCH (n) RETURN n.`a-b` AS `sort id` ORDER BY `sort id` DESC",
        engine=engine,
    )

    assert out._nodes["sort id"].to_list() == [20, 10]


def test_quoted_return_alias_survives_order_on_cudf():
    cudf = pytest.importorskip("cudf")
    nodes = cudf.DataFrame({"id": ["a", "b"], "a-b": [10, 20]})
    edges = cudf.DataFrame({"src": ["a", "b"], "dst": ["b", "a"]})
    graph = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")

    out = graph.gfql(
        "MATCH (n) RETURN n.`a-b` AS `sort id` ORDER BY `sort id` DESC",
        engine="cudf",
    )

    assert type(out._nodes).__module__.startswith("cudf")
    assert out._nodes["sort id"].to_pandas().tolist() == [20, 10]


def test_quoted_output_alias_does_not_collide_with_expression_text():
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "num": [3, 1, 2]})
    edges = pd.DataFrame({"src": ["a", "b", "c"], "dst": ["b", "c", "a"]})
    graph = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")

    out = graph.gfql(
        "MATCH(a) RETURN a, a.num + 1 AS `(a.num + 1)` "
        "ORDER BY `(a.num + 1)` DESC",
        engine="pandas",
    )

    assert out._nodes["(a.num + 1)"].tolist() == [4, 3, 2]


def test_quoted_output_alias_survives_explicit_with_realias():
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "num": [3, 1, 2]})
    edges = pd.DataFrame({"src": ["a", "b", "c"], "dst": ["b", "c", "a"]})
    graph = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")

    out = graph.gfql(
        "MATCH(a) WITH a, a.num + 1 AS `(a.num + 1)` "
        "WITH a, `(a.num + 1)` AS `(a.num + 1)` "
        "ORDER BY `(a.num + 1)` DESC "
        "RETURN a, `(a.num + 1)` AS `(a.num + 1)`",
        engine="pandas",
    )

    assert out._nodes["(a.num + 1)"].tolist() == [4, 3, 2]


def test_quoted_hidden_order_key_survives_order_on_cudf():
    cudf = pytest.importorskip("cudf")
    nodes = cudf.DataFrame({"id": ["a", "b", "c"], "score": [3, 1, 2]})
    edges = cudf.DataFrame({"src": ["a"], "dst": ["b"]})
    graph = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")

    out = graph.gfql(
        "MATCH(n) "
        "WITH n.id AS id, n.score AS `sort id` "
        "RETURN id ORDER BY `sort id` DESC",
        engine="cudf",
    )

    assert type(out._nodes).__module__.startswith("cudf")
    assert out._nodes["id"].to_pandas().tolist() == ["a", "c", "b"]


def test_quoted_aggregate_alias_survives_order(odd_columns_graph):
    out = odd_columns_graph.gfql(
        "MATCH (n) RETURN count(*) AS `count value` ORDER BY `count value` DESC",
        engine="pandas",
    )

    assert out._nodes["count value"].tolist() == [2]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("query", "expected_category", "expected_aggregate"),
    [
        (
            "MATCH(n) RETURN n.category AS category, count(*) AS `count(*)` ORDER BY `count(*)` ASC, category ASC",
            ["y", "x"],
            [1, 3],
        ),
        (
            "MATCH(n) RETURN n.category AS category, sum(n.num) AS `sum(n.num)` ORDER BY `sum(n.num)` ASC, category ASC",
            ["y", "x"],
            [5, 6],
        ),
    ],
)
def test_quoted_aggregate_output_aliases_survive_order(
    aggregate_graph,
    engine,
    query,
    expected_category,
    expected_aggregate,
):
    if engine == "polars":
        pytest.importorskip("polars")

    out = aggregate_graph.gfql(query, engine=engine)

    assert out._nodes["category"].to_list() == expected_category
    assert out._nodes["count(*)" if "count(*)" in query else "sum(n.num)"].to_list() == expected_aggregate


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "query",
    [
        "MATCH(n) RETURN n.category AS category ORDER BY count(*) ASC",
        "MATCH(n) RETURN n.category AS category ORDER BY sum(n.num) ASC",
    ],
)
def test_unprojected_aggregate_order_is_rejected(boundary_graph, engine, query):
    if engine == "polars":
        pytest.importorskip("polars")

    with pytest.raises(GFQLValidationError) as raised:
        boundary_graph.gfql(query, engine=engine)

    error = raised.value
    assert error.code == "unsupported-cypher-query"
    assert error.context["field"] == "order_by"


@pytest.fixture
def unwind_aggregate_graph():
    nodes = pd.DataFrame({"id": [0]})
    return graphistry.nodes(nodes, "id")


@pytest.fixture
def unwind_match_graph():
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame({"src": ["a"], "dst": ["b"]})
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("query_prefix", "order_key", "expected_g", "expected_count", "expected_sum"),
    [
        ("UNWIND [1, 1, 10] AS x", "`sum(x)`", [10, 1], [1, 2], [10, 2]),
        ("UNWIND [1, 1, 10] AS x", "sum(x)", [1, 10], [2, 1], [2, 10]),
        ("UNWIND [1, 1, 10] AS x", "`count(*)`", [1, 10], [2, 1], [2, 10]),
        ("UNWIND [1, 1, 10] AS x", "count(*)", [10, 1], [1, 2], [10, 2]),
        ("MATCH(n) UNWIND [1, 1, 10] AS x", "`sum(x)`", [10, 1], [2, 4], [20, 4]),
        ("MATCH(n) UNWIND [1, 1, 10] AS x", "sum(x)", [1, 10], [4, 2], [4, 20]),
        ("MATCH(n) UNWIND [1, 1, 10] AS x", "`count(*)`", [1, 10], [4, 2], [4, 20]),
        ("MATCH(n) UNWIND [1, 1, 10] AS x", "count(*)", [10, 1], [2, 4], [20, 4]),
    ],
)
def test_swapped_aggregate_aliases_across_callers(
    unwind_aggregate_graph,
    unwind_match_graph,
    engine,
    query_prefix,
    order_key,
    expected_g,
    expected_count,
    expected_sum,
):
    if engine == "polars":
        pytest.importorskip("polars")

    graph = unwind_match_graph if query_prefix.startswith("MATCH") else unwind_aggregate_graph
    query = (
        f"{query_prefix} "
        "RETURN x AS g, count(*) AS `sum(x)`, sum(x) AS `count(*)` "
        f"ORDER BY {order_key} ASC"
    )

    out = graph.gfql(query, engine=engine)

    assert out._nodes["g"].to_list() == expected_g
    assert out._nodes["sum(x)"].to_list() == expected_count
    assert out._nodes["count(*)"].to_list() == expected_sum


def test_swapped_unwind_aggregate_aliases_on_cudf():
    cudf = pytest.importorskip("cudf")
    graph = graphistry.nodes(cudf.DataFrame({"id": [0]}), "id")

    out = graph.gfql(
        "UNWIND [1, 1, 10] AS x "
        "RETURN x AS g, count(*) AS `sum(x)`, sum(x) AS `count(*)` "
        "ORDER BY sum(x) ASC",
        engine="cudf",
    )

    assert type(out._nodes).__module__.startswith("cudf")
    assert out._nodes["g"].to_pandas().tolist() == [1, 10]
    assert out._nodes["sum(x)"].to_pandas().tolist() == [2, 1]
    assert out._nodes["count(*)"].to_pandas().tolist() == [2, 10]


def test_swapped_match_unwind_aggregate_aliases_on_cudf():
    cudf = pytest.importorskip("cudf")
    graph = graphistry.nodes(cudf.DataFrame({"id": ["a", "b"]}), "id").edges(
        cudf.DataFrame({"src": ["a"], "dst": ["b"]}), "src", "dst"
    )

    out = graph.gfql(
        "MATCH(n) UNWIND [1, 1, 10] AS x "
        "RETURN x AS g, count(*) AS `sum(x)`, sum(x) AS `count(*)` "
        "ORDER BY `sum(x)` ASC",
        engine="cudf",
    )

    assert type(out._nodes).__module__.startswith("cudf")
    assert out._nodes["g"].to_pandas().tolist() == [10, 1]
    assert out._nodes["sum(x)"].to_pandas().tolist() == [2, 4]
    assert out._nodes["count(*)"].to_pandas().tolist() == [20, 4]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "query",
    [
        "UNWIND [1, 1, 10] AS x RETURN x AS g ORDER BY count(*) ASC",
        "MATCH(n) UNWIND [1, 1, 10] AS x RETURN n.id AS g ORDER BY count(*) ASC",
    ],
)
def test_unprojected_aggregate_order_is_rejected_across_callers(
    unwind_aggregate_graph,
    unwind_match_graph,
    engine,
    query,
):
    if engine == "polars":
        pytest.importorskip("polars")

    graph = unwind_match_graph if query.startswith("MATCH") else unwind_aggregate_graph

    with pytest.raises(GFQLValidationError) as raised:
        graph.gfql(query, engine=engine)

    error = raised.value
    assert error.code == "unsupported-cypher-query"
    assert error.context["field"] == "order_by"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("order_key", "expected_category", "expected_count", "expected_sum"),
    [
        ("`sum(n.num)`", ["y", "x"], [1, 3], [10, 3]),
        ("`count(*)`", ["x", "y"], [3, 1], [3, 10]),
        ("count(*)", ["y", "x"], [1, 3], [10, 3]),
        ("sum(n.num)", ["x", "y"], [3, 1], [3, 10]),
    ],
)
def test_aggregate_output_alias_kinds_keep_independent_sort_order(
    aggregate_alias_graph,
    engine,
    order_key,
    expected_category,
    expected_count,
    expected_sum,
):
    if engine == "polars":
        pytest.importorskip("polars")

    out = aggregate_alias_graph.gfql(
        "MATCH(n) "
        "RETURN n.category AS category, count(*) AS `sum(n.num)`, sum(n.num) AS `count(*)` "
        f"ORDER BY {order_key} ASC",
        engine=engine,
    )

    assert out._nodes["category"].to_list() == expected_category
    assert out._nodes["sum(n.num)"].to_list() == expected_count
    assert out._nodes["count(*)"].to_list() == expected_sum


@pytest.mark.parametrize(
    ("order_key", "expected_category", "expected_count", "expected_sum"),
    [
        ("`sum(n.num)`", ["y", "x"], [1, 3], [10, 3]),
        ("`count(*)`", ["x", "y"], [3, 1], [3, 10]),
        ("count(*)", ["y", "x"], [1, 3], [10, 3]),
        ("sum(n.num)", ["x", "y"], [3, 1], [3, 10]),
    ],
)
def test_aggregate_output_alias_kinds_on_cudf(
    order_key,
    expected_category,
    expected_count,
    expected_sum,
):
    cudf = pytest.importorskip("cudf")
    nodes = cudf.DataFrame(
        {"id": ["a", "b", "c", "d"], "num": [1, 1, 1, 10], "category": ["x", "x", "x", "y"]}
    )
    edges = cudf.DataFrame({"src": ["a"], "dst": ["b"]})
    graph = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")

    out = graph.gfql(
        "MATCH(n) "
        "RETURN n.category AS category, count(*) AS `sum(n.num)`, sum(n.num) AS `count(*)` "
        f"ORDER BY {order_key} ASC",
        engine="cudf",
    )

    assert type(out._nodes).__module__.startswith("cudf")
    assert out._nodes["category"].to_pandas().tolist() == expected_category
    assert out._nodes["sum(n.num)"].to_pandas().tolist() == expected_count
    assert out._nodes["count(*)"].to_pandas().tolist() == expected_sum


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("alias", "order_key", "expected_category", "expected_aggregate"),
    [
        ("c", "`c` + 1", ["y", "x"], [1, 3]),
        ("the count", "`the count` + 1", ["y", "x"], [1, 3]),
        ("sum(n.num)", "`sum(n.num)` + 1", ["y", "x"], [1, 3]),
        ("c", "-`c`", ["x", "y"], [3, 1]),
        ("the count", "-`the count`", ["x", "y"], [3, 1]),
        ("sum(n.num)", "-`sum(n.num)`", ["x", "y"], [3, 1]),
        ("c", "abs(`c`)", ["y", "x"], [1, 3]),
        ("the count", "abs(`the count`)", ["y", "x"], [1, 3]),
        ("sum(n.num)", "abs(`sum(n.num)`)", ["y", "x"], [1, 3]),
        ("c", "count(*) + 1", ["y", "x"], [1, 3]),
        ("the count", "count(*) + 1", ["y", "x"], [1, 3]),
        ("sum(n.num)", "count(*) + 1", ["y", "x"], [1, 3]),
    ],
)
def test_compound_aggregate_order_keys_preserve_output_bindings(
    aggregate_alias_graph,
    engine,
    alias,
    order_key,
    expected_category,
    expected_aggregate,
):
    if engine == "polars":
        pytest.importorskip("polars")

    out = aggregate_alias_graph.gfql(
        "MATCH(n) "
        f"RETURN n.category AS category, count(*) AS `{alias}` "
        f"ORDER BY {order_key} ASC",
        engine=engine,
    )

    assert out._nodes["category"].to_list() == expected_category
    assert out._nodes[alias].to_list() == expected_aggregate


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("order_key", "expected_category", "expected_count", "expected_sum"),
    [
        ("`sum(n.num)` + 1", ["y", "x"], [1, 3], [10, 3]),
        ("`count(*)` + 1", ["x", "y"], [3, 1], [3, 10]),
        ("count(*) + 1", ["y", "x"], [1, 3], [10, 3]),
        ("sum(n.num) + 1", ["x", "y"], [3, 1], [3, 10]),
        ("-`sum(n.num)`", ["x", "y"], [3, 1], [3, 10]),
        ("-`count(*)`", ["y", "x"], [1, 3], [10, 3]),
        ("-count(*)", ["x", "y"], [3, 1], [3, 10]),
        ("-sum(n.num)", ["y", "x"], [1, 3], [10, 3]),
        ("abs(`sum(n.num)`)", ["y", "x"], [1, 3], [10, 3]),
        ("abs(`count(*)`)", ["x", "y"], [3, 1], [3, 10]),
        ("abs(count(*))", ["y", "x"], [1, 3], [10, 3]),
        ("abs(sum(n.num))", ["x", "y"], [3, 1], [3, 10]),
    ],
)
def test_compound_swapped_aggregate_order_keys_preserve_bindings(
    aggregate_alias_graph,
    engine,
    order_key,
    expected_category,
    expected_count,
    expected_sum,
):
    if engine == "polars":
        pytest.importorskip("polars")

    out = aggregate_alias_graph.gfql(
        "MATCH(n) "
        "RETURN n.category AS category, "
        "count(*) AS `sum(n.num)`, sum(n.num) AS `count(*)` "
        f"ORDER BY {order_key} ASC",
        engine=engine,
    )

    assert out._nodes["category"].to_list() == expected_category
    assert out._nodes["sum(n.num)"].to_list() == expected_count
    assert out._nodes["count(*)"].to_list() == expected_sum


@pytest.fixture
def shadowed_aggregate_alias_graph():
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "num": [1, 1, 1, 10],
            "score": [1, 1, 1, 1],
        }
    )
    edges = pd.DataFrame({"src": ["a"], "dst": ["b"]})
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("order_key", "expected_group", "expected_aggregate"),
    [
        ("v", [10, 1], [1, 3]),
        ("`v`", [10, 1], [1, 3]),
        ("v + 1", [10, 1], [1, 3]),
        ("`v` + 1", [10, 1], [1, 3]),
        ("sum(w) + 1", [10, 1], [1, 3]),
        ("group_value + 1", [1, 10], [3, 1]),
    ],
)
def test_projected_identifier_precedes_aggregate_alias_rewrite(
    shadowed_aggregate_alias_graph,
    engine,
    order_key,
    expected_group,
    expected_aggregate,
):
    if engine == "polars":
        pytest.importorskip("polars")

    out = shadowed_aggregate_alias_graph.gfql(
        "MATCH(n) "
        "WITH n.num AS v, n.score AS w "
        "RETURN v AS group_value, sum(w) AS v "
        f"ORDER BY {order_key} ASC",
        engine=engine,
    )

    assert out._nodes["group_value"].to_list() == expected_group
    assert out._nodes["v"].to_list() == expected_aggregate


@pytest.mark.parametrize(
    "order_key",
    ["v", "`v`", "v + 1", "`v` + 1"],
)
def test_projected_identifier_precedes_aggregate_alias_rewrite_on_cudf(
    order_key,
):
    cudf = pytest.importorskip("cudf")
    nodes = cudf.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "num": [1, 1, 1, 10],
            "score": [1, 1, 1, 1],
        }
    )
    edges = cudf.DataFrame({"src": ["a"], "dst": ["b"]})
    graph = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")

    out = graph.gfql(
        "MATCH(n) "
        "WITH n.num AS v, n.score AS w "
        "RETURN v AS group_value, sum(w) AS v "
        f"ORDER BY {order_key} ASC",
        engine="cudf",
    )

    assert type(out._nodes).__module__.startswith("cudf")
    assert out._nodes["group_value"].to_pandas().tolist() == [10, 1]
    assert out._nodes["v"].to_pandas().tolist() == [1, 3]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("alias", ["c", "the count"])
@pytest.mark.parametrize(("factor", "expected"), [(1, ["y", "x"]), (-1, ["x", "y"])])
def test_quoted_aggregate_order_accepts_parameters(
    aggregate_graph,
    engine,
    alias,
    factor,
    expected,
):
    if engine == "polars":
        pytest.importorskip("polars")

    out = aggregate_graph.gfql(
        f"MATCH(n) RETURN n.category AS category, count(*) AS `{alias}` "
        f"ORDER BY `{alias}` * $factor ASC",
        params={"factor": factor},
        engine=engine,
    )

    assert out._nodes["category"].to_list() == expected


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("alias", ["c", "the count"])
def test_quoted_aggregate_order_reports_missing_parameter(
    aggregate_graph,
    engine,
    alias,
):
    if engine == "polars":
        pytest.importorskip("polars")

    with pytest.raises(GFQLValidationError) as raised:
        aggregate_graph.gfql(
            f"MATCH(n) RETURN n.category AS category, count(*) AS `{alias}` "
            f"ORDER BY `{alias}` * $factor ASC",
            engine=engine,
        )

    error = raised.value
    assert error.code == "missing-required-field"
    assert error.context["field"] == "order_by"
    assert error.context["value"] == "factor"


def test_compound_aggregate_order_key_survives_order_on_cudf():
    cudf = pytest.importorskip("cudf")
    nodes = cudf.DataFrame(
        {"id": ["a", "b", "c", "d"], "num": [1, 1, 1, 10], "category": ["x", "x", "x", "y"]}
    )
    edges = cudf.DataFrame({"src": ["a"], "dst": ["b"]})
    graph = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")

    out = graph.gfql(
        "MATCH(n) "
        "RETURN n.category AS category, count(*) AS `the count` "
        "ORDER BY `the count` + 1 ASC",
        engine="cudf",
    )

    assert type(out._nodes).__module__.startswith("cudf")
    assert out._nodes["category"].to_pandas().tolist() == ["y", "x"]
    assert out._nodes["the count"].to_pandas().tolist() == [1, 3]


def test_escaped_quoted_alias_survives_order(odd_columns_graph):
    out = odd_columns_graph.gfql(
        "MATCH (n) RETURN n.`a``b` AS `sort``id` ORDER BY `sort``id` DESC",
        engine="pandas",
    )

    assert out._nodes["sort`id"].to_list() == [2, 1]


def test_unknown_quoted_order_alias_is_rejected(odd_columns_graph):
    with pytest.raises(GFQLValidationError) as raised:
        odd_columns_graph.gfql(
            "MATCH (n) RETURN n.`a-b` AS x ORDER BY `missing id` DESC",
            engine="pandas",
        )

    error = raised.value
    assert error.code == "invalid-name-type"
    assert error.context["field"] == "identifier"
    assert error.context["value"] == "missing id"


def test_unquoted_identifiers_are_not_quoted():
    parsed = parse_expr("n.analysis")
    assert parsed.value.name == "n"
    assert parsed.value.quoted is False
    assert parsed.property == "analysis"


@pytest.fixture
def boundary_graph():
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"], "score": [2, None, 2, 1], "a": [1, 5, 2, 4], "b": [0, 0, 0, 0], "category": ["x", "x", "x", "y"]})
    edges = pd.DataFrame({"src": ["a"], "dst": ["b"]})
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


@pytest.fixture
def aggregate_graph():
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"], "num": [1, 3, 2, 5], "category": ["x", "x", "x", "y"]})
    edges = pd.DataFrame({"src": ["a"], "dst": ["b"]})
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


@pytest.fixture
def aggregate_alias_graph():
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "num": [1, 1, 1, 10],
            "category": ["x", "x", "x", "y"],
        }
    )
    edges = pd.DataFrame({"src": ["a"], "dst": ["b"]})
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("case", "query", "expected"),
    [
        ("null_desc", "MATCH(n) RETURN n.id AS id, n.score AS `the id` ORDER BY `the id` DESC, id ASC", ["b", "a", "c", "d"]),
        ("null_asc", "MATCH(n) RETURN n.id AS id, n.score AS `the id` ORDER BY `the id` ASC, id ASC", ["d", "a", "c", "b"]),
        ("aggregate_asc", "MATCH(n) RETURN n.category AS id, count(*) AS `count value` ORDER BY `count value` ASC", ["y", "x"]),
        ("aggregate_desc", "MATCH(n) RETURN n.category AS id, count(*) AS `count value` ORDER BY `count value` DESC", ["x", "y"]),
        ("empty", "MATCH(n) WHERE n.score > 100 RETURN n.id AS id, n.score AS `the id` ORDER BY `the id` DESC", []),
        ("limit", "MATCH(n) RETURN n.id AS id, n.score AS `the id` ORDER BY `the id` DESC, id ASC LIMIT 2", ["b", "a"]),
        ("expression_alias", "MATCH(n) RETURN n.id AS id, n.a AS a, n.b AS b, n.score AS `a-b` ORDER BY `a-b` DESC, id ASC", ["b", "a", "c", "d"]),
        ("expression_key", "MATCH(n) RETURN n.id AS id, n.a AS a, n.b AS b, n.score AS `a-b` ORDER BY a-b DESC", ["b", "d", "c", "a"]),
    ],
)
def test_quoted_order_boundaries(boundary_graph, engine, case, query, expected):
    if engine == "polars":
        pytest.importorskip("polars")

    out = boundary_graph.gfql(query, engine=engine)

    assert out._nodes["id"].to_list() == expected


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "query",
    [
        "MATCH(n) WITH n.id AS id, n.score AS `hidden value` RETURN id ORDER BY `hidden value` DESC",
        "MATCH(n) WITH n, n.score + 1 AS `hidden value` RETURN n.id AS id ORDER BY `hidden value` DESC",
    ],
)
def test_quoted_hidden_order_keys_survive_final_projection(boundary_graph, engine, query):
    if engine == "polars":
        pytest.importorskip("polars")

    out = boundary_graph.gfql(query, engine=engine)

    assert out._nodes["id"].to_list() == ["b", "a", "c", "d"]


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) RETURN n.`unterminated",
    ],
)
def test_malformed_backtick_identifiers_rejected(query):
    with pytest.raises(Exception):
        parse_cypher(query)

@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("RETURN {`abc`: 1} AS x", {"abc": 1}),
        ("RETURN {`a-b`: 1} AS x", {"a-b": 1}),
        ("RETURN {`a``b`: 1} AS x", {"a`b": 1}),
    ],
)
def test_quoted_map_keys_construct_and_access(odd_columns_graph, query, expected):
    constructed = odd_columns_graph.gfql(query, engine="pandas")
    assert constructed._nodes["x"].tolist() == [expected]

    key = next(iter(expected))
    rendered_key = key.replace("`", "``")
    accessed = odd_columns_graph.gfql(
        f"RETURN {{`{rendered_key}`: 1}}.`{rendered_key}` AS x",
        engine="pandas",
    )
    assert accessed._nodes["x"].tolist() == [next(iter(expected.values()))]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("a-b", 20),
        ("a b", 40),
        ("a``b", 2),
    ],
)
def test_quoted_match_aliases_survive_where_order_projection(odd_columns_graph, engine, alias, expected):
    if engine == "polars":
        pytest.importorskip("polars")
    out = odd_columns_graph.gfql(
        f"MATCH (`{alias}`) "
        f"WHERE `{alias}`.`analysis.group.Score` > 0.5 "
        f"RETURN `{alias}`.`{alias}` AS out "
        f"ORDER BY out DESC",
        engine=engine,
    )

    assert out._nodes["out"].to_list() == [expected]


@pytest.mark.parametrize(
    "word",
    ["true", "null", "case", "WITH"],
)
def test_reserved_word_backtick_identifiers_round_trip(word):
    ast = parse_expr(f"`{word}`")

    assert ast == Identifier(name=word, quoted=True)
    assert render_expr_node(ast) == f"`{word}`"
    assert parse_expr(render_expr_node(ast)) == ast


@pytest.mark.parametrize(
    "word",
    sorted(CYPHER_RESERVED_WORDS | {"ANY", "ALL", "NONE", "SINGLE"}),
)
def test_expression_reserved_word_identifiers_round_trip(word):
    for spelling in {word, word.lower(), word.title()}:
        ast = parse_expr(f"`{spelling}`")

        assert ast == Identifier(name=spelling, quoted=True)
        assert render_expr_node(ast) == f"`{spelling}`"
        assert parse_expr(render_expr_node(ast)) == ast


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("alias", ["any", "ALL"])
def test_expression_reserved_word_aliases_project(odd_columns_graph, engine, alias):
    if engine == "polars":
        pytest.importorskip("polars")
    out = odd_columns_graph.gfql(
        f"MATCH (`{alias}`) RETURN `{alias}`.`a-b` AS out",
        engine=engine,
    )

    assert out._nodes["out"].to_list() == [10, 20]


@pytest.mark.parametrize(
    ("local", "expected"),
    [
        ("a-b", [[2, 3]]),
        ("a b", [[2, 3]]),
        ("a``b", [[2, 3]]),
        ("any", [[2, 3]]),
    ],
)
def test_quoted_comprehension_locals_survive_rewrite(odd_columns_graph, local, expected):
    escaped = local.replace("`", "``")
    out = odd_columns_graph.gfql(
        f"WITH [1, 2] AS xs RETURN [`{escaped}` IN xs | `{escaped}` + 1] AS out",
        engine="pandas",
    )

    assert out._nodes["out"].to_list() == expected


def test_quoted_quantifier_local_round_trips_through_rewrite():
    from graphistry.compute.gfql.cypher.lowering import _rewrite_expr_identifiers

    ast = parse_expr("any(`a-b` IN xs WHERE `a-b` > 1)")
    rewritten = _rewrite_expr_identifiers(ast, {})
    rendered = render_expr_node(rewritten)

    assert rendered == "ANY(`a-b` IN xs WHERE (`a-b` > 1))"
    assert parse_expr(rendered) == rewritten
