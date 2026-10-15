from __future__ import annotations

import pandas as pd
import pytest

import graphistry
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


def test_unquoted_identifiers_are_not_quoted():
    parsed = parse_expr("n.analysis")
    assert parsed.value.name == "n"
    assert parsed.value.quoted is False
    assert parsed.property == "analysis"


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
