from typing import cast

import pytest

from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError
from graphistry.compute.gfql.cypher import (
    ExpressionText,
    LabelRef,
    NodePattern,
    ParameterRef,
    PropertyRef,
    RelationshipPattern,
    WherePatternPredicate,
    parse_cypher,
)


def test_parse_minimal_match_return() -> None:
    parsed = parse_cypher("MATCH (n) RETURN n")

    assert parsed.match is not None
    assert parsed.match.pattern[0].variable == "n"
    assert parsed.unwinds == ()
    assert parsed.return_.distinct is False
    assert parsed.return_.kind == "return"
    assert parsed.return_.items[0].expression.text == "n"
    assert parsed.return_.items[0].alias is None
    assert parsed.order_by is None
    assert parsed.skip is None
    assert parsed.limit is None
    assert parsed.trailing_semicolon is False


def test_parse_linear_pattern_with_labels_properties_and_aliases() -> None:
    parsed = parse_cypher(
        'MATCH (p:Person {id: $person_id})-[r:FOLLOWS]->(q:Person {active: true}) RETURN p.name AS person_name, q'
    )

    assert parsed.match is not None
    assert len(parsed.match.pattern) == 3

    left = parsed.match.pattern[0]
    rel = parsed.match.pattern[1]
    right = parsed.match.pattern[2]

    assert isinstance(left, NodePattern)
    assert isinstance(rel, RelationshipPattern)
    assert isinstance(right, NodePattern)

    assert left.variable == "p"
    assert left.labels == ("Person",)
    assert left.properties[0].key == "id"
    assert isinstance(left.properties[0].value, ParameterRef)
    assert left.properties[0].value.name == "person_id"

    assert rel.variable == "r"
    assert rel.direction == "forward"
    assert rel.types == ("FOLLOWS",)

    assert right.variable == "q"
    assert right.labels == ("Person",)
    assert right.properties[0].key == "active"
    assert right.properties[0].value is True

    assert parsed.return_.items[0].expression.text == "p.name"
    assert parsed.return_.items[0].alias == "person_name"
    assert parsed.return_.items[1].expression.text == "q"


@pytest.mark.parametrize(
    "query,direction",
    [
        ("MATCH (a)<-[r:KNOWS]-(b) RETURN r", "reverse"),
        ("MATCH (a)-[r:KNOWS]-(b) RETURN r;", "undirected"),
        ("MATCH (a)-->(b) RETURN b", "forward"),
        ("MATCH (a)<--(b) RETURN b", "reverse"),
        ("MATCH (a)--(b) RETURN b", "undirected"),
        ("MATCH (a)<-->(b) RETURN b", "undirected"),
    ],
)
def test_parse_relationship_directions(query: str, direction: str) -> None:
    parsed = parse_cypher(query)

    assert parsed.match is not None
    rel = parsed.match.pattern[1]
    assert isinstance(rel, RelationshipPattern)
    assert rel.direction == direction


def test_parse_reports_trailing_semicolon() -> None:
    parsed = parse_cypher("MATCH (n) RETURN n;")
    assert parsed.trailing_semicolon is True


def test_parse_where_clause() -> None:
    parsed = parse_cypher("MATCH (a)-[r]->(b) WHERE a.team = b.team AND b.score >= 10 RETURN a")

    assert parsed.where is not None
    assert len(parsed.where.predicates) == 2
    pred0 = parsed.where.predicates[0]
    pred1 = parsed.where.predicates[1]
    assert not isinstance(pred0, WherePatternPredicate)
    assert not isinstance(pred1, WherePatternPredicate)
    left0 = cast(PropertyRef, pred0.left)
    left1 = cast(PropertyRef, pred1.left)
    assert left0.alias == "a"
    assert left0.property == "team"
    assert pred0.op == "=="
    assert left1.alias == "b"
    assert left1.property == "score"
    assert pred1.op == ">="


def test_parse_where_null_predicates() -> None:
    parsed = parse_cypher("MATCH (a)-[r]->(b) WHERE a.deleted IS NULL AND b.name IS NOT NULL RETURN a")

    assert parsed.where is not None
    pred0 = parsed.where.predicates[0]
    pred1 = parsed.where.predicates[1]
    assert not isinstance(pred0, WherePatternPredicate)
    assert not isinstance(pred1, WherePatternPredicate)
    assert pred0.op == "is_null"
    assert pred0.right is None
    assert pred1.op == "is_not_null"
    assert pred1.right is None


def test_parse_return_xor_precedence_expression() -> None:
    parsed = parse_cypher("RETURN true OR true XOR false AND false AS result")

    assert parsed.return_.items[0].expression.text == "true OR true XOR false AND false"


def test_parse_return_searched_case_expression() -> None:
    parsed = parse_cypher("RETURN CASE WHEN score > 1 THEN true ELSE false END AS result")

    assert parsed.return_.items[0].expression.text == "CASE WHEN score > 1 THEN true ELSE false END"
    assert parsed.return_.items[0].alias == "result"


def test_parse_return_simple_case_expression() -> None:
    parsed = parse_cypher("RETURN CASE score WHEN 1 THEN 'one' ELSE 'other' END AS result")

    assert parsed.return_.items[0].expression.text == "CASE score WHEN 1 THEN 'one' ELSE 'other' END"
    assert parsed.return_.items[0].alias == "result"


@pytest.mark.parametrize(
    "query,expr_text",
    [
        ("RETURN 0x1 AS literal", "0x1"),
        ("RETURN -0x1 AS literal", "-0x1"),
        ("RETURN 0o7 AS literal", "0o7"),
        ("RETURN -0o7 AS literal", "-0o7"),
        ("RETURN .1e9 AS literal", ".1e9"),
        ("RETURN -.1e-5 AS literal", "-.1e-5"),
    ],
)
def test_parse_numeric_literal_forms(query: str, expr_text: str) -> None:
    parsed = parse_cypher(query)

    assert parsed.return_.items[0].expression.text == expr_text


def test_parse_with_where_pipeline() -> None:
    parsed = parse_cypher("UNWIND [true, false, null] AS a WITH a WHERE a IS NULL RETURN a")

    assert len(parsed.with_stages) == 1
    assert parsed.with_stages[0].where is not None
    assert parsed.with_stages[0].where.text == "a IS NULL"


def test_parse_match_after_with_reentry_shape() -> None:
    parsed = parse_cypher("MATCH (a:A) WITH a ORDER BY a.name LIMIT 1 MATCH (a)-->(b) RETURN a")

    assert len(parsed.matches) == 1
    assert len(parsed.with_stages) == 1
    assert len(parsed.reentry_matches) == 1
    assert parsed.reentry_where is None


def test_parse_where_label_predicate() -> None:
    parsed = parse_cypher("MATCH (a)-->(b) WHERE b:Foo:Bar RETURN b")

    assert parsed.where is not None
    assert len(parsed.where.predicates) == 1
    predicate = parsed.where.predicates[0]
    assert not isinstance(predicate, WherePatternPredicate)
    assert predicate.op == "has_labels"
    left = cast(LabelRef, predicate.left)
    assert left.alias == "b"
    assert left.labels == ("Foo", "Bar")


def test_parse_reserved_keyword_labels_and_relationship_types() -> None:
    parsed = parse_cypher("MATCH (n:Single)-[r:SINGLE]->(m:End) RETURN m:TYPE, n:Single")

    assert parsed.match is not None
    left = parsed.match.pattern[0]
    rel = parsed.match.pattern[1]
    right = parsed.match.pattern[2]
    assert isinstance(left, NodePattern)
    assert isinstance(rel, RelationshipPattern)
    assert isinstance(right, NodePattern)
    assert left.labels == ("Single",)
    assert rel.types == ("SINGLE",)
    assert right.labels == ("End",)
    assert parsed.return_.items[0].expression.text == "m:TYPE"
    assert parsed.return_.items[1].expression.text == "n:Single"


def test_parse_relationship_type_alternation() -> None:
    parsed = parse_cypher("MATCH (a)-[r:KNOWS|HATES]->(b) RETURN r")

    assert parsed.match is not None
    rel = parsed.match.pattern[1]
    assert isinstance(rel, RelationshipPattern)
    assert rel.types == ("KNOWS", "HATES")


def test_parse_relationship_type_alternation_with_repeated_colon() -> None:
    parsed = parse_cypher("MATCH (a)-[:T|:OTHER]->(b) RETURN b")

    assert parsed.match is not None
    rel = parsed.match.pattern[1]
    assert isinstance(rel, RelationshipPattern)
    assert rel.types == ("T", "OTHER")


def test_parse_return_pipeline_clauses() -> None:
    parsed = parse_cypher(
        "MATCH (p:Person) RETURN DISTINCT p.name AS person_name ORDER BY person_name DESC, p.id ASC SKIP 1 LIMIT 2;"
    )

    assert parsed.return_.kind == "return"
    assert parsed.return_.distinct is True
    assert parsed.return_.items[0].expression.text == "p.name"
    assert parsed.return_.items[0].alias == "person_name"
    assert parsed.order_by is not None
    assert [item.expression.text for item in parsed.order_by.items] == ["person_name", "p.id"]
    assert [item.direction for item in parsed.order_by.items] == ["desc", "asc"]
    assert parsed.skip is not None and isinstance(parsed.skip.value, ExpressionText)
    assert parsed.skip.value.text == "1"
    assert parsed.limit is not None and isinstance(parsed.limit.value, ExpressionText)
    assert parsed.limit.value.text == "2"
    assert parsed.trailing_semicolon is True


def test_parse_terminal_with_clause() -> None:
    parsed = parse_cypher("MATCH (p) WITH p.name AS person_name ORDER BY person_name ASC LIMIT 5")

    assert parsed.with_stages == ()
    assert parsed.return_.kind == "with"
    assert parsed.order_by is not None
    assert parsed.order_by.items[0].expression.text == "person_name"
    assert parsed.limit is not None and isinstance(parsed.limit.value, ExpressionText)
    assert parsed.limit.value.text == "5"


def test_parse_with_then_return_pipeline() -> None:
    parsed = parse_cypher("UNWIND [1, 3, 2] AS ints WITH ints ORDER BY ints DESC LIMIT 2 RETURN ints")

    assert len(parsed.with_stages) == 1
    with_stage = parsed.with_stages[0]
    assert with_stage.clause.kind == "with"
    assert with_stage.clause.items[0].expression.text == "ints"
    assert with_stage.order_by is not None
    assert with_stage.order_by.items[0].expression.text == "ints"
    assert with_stage.order_by.items[0].direction == "desc"
    assert with_stage.limit is not None and isinstance(with_stage.limit.value, ExpressionText)
    assert with_stage.limit.value.text == "2"
    assert parsed.return_.kind == "return"
    assert parsed.return_.items[0].expression.text == "ints"


def test_parse_multiple_with_stages_and_long_order_directions() -> None:
    parsed = parse_cypher("WITH 1 AS a, 2 AS b WITH a ORDER BY a ASCENDING WITH a ORDER BY a DESCENDING RETURN a")

    assert len(parsed.with_stages) == 3
    assert len(parsed.row_sequence) == 4
    assert parsed.with_stages[0].clause.items[0].expression.text == "1"
    assert parsed.with_stages[1].order_by is not None
    assert parsed.with_stages[1].order_by.items[0].direction == "asc"
    assert parsed.with_stages[2].order_by is not None
    assert parsed.with_stages[2].order_by.items[0].direction == "desc"
    assert parsed.return_.items[0].expression.text == "a"


def test_parse_interleaved_row_only_with_unwind_sequence() -> None:
    parsed = parse_cypher(
        "WITH [0, 1] AS prows, [[2], [3, 4]] AS qrows "
        "UNWIND prows AS p "
        "UNWIND qrows[p] AS q "
        "WITH p, count(q) AS rng "
        "RETURN p "
        "ORDER BY rng"
    )

    assert parsed.match is None
    assert len(parsed.row_sequence) == 5
    assert parsed.return_.items[0].expression.text == "p"
    assert parsed.order_by is not None
    assert parsed.order_by.items[0].expression.text == "rng"


def test_parse_unwind_without_match() -> None:
    parsed = parse_cypher("UNWIND [1, 2, 3] AS x RETURN x ORDER BY x")

    assert parsed.match is None
    assert len(parsed.unwinds) == 1
    assert parsed.unwinds[0].expression.text == "[1, 2, 3]"
    assert parsed.unwinds[0].alias == "x"
    assert parsed.return_.items[0].expression.text == "x"
    assert parsed.order_by is not None
    assert parsed.order_by.items[0].expression.text == "x"


def test_parse_match_then_unwind() -> None:
    parsed = parse_cypher("MATCH (p) UNWIND p.vals AS v RETURN v")

    assert parsed.match is not None
    assert len(parsed.unwinds) == 1
    assert parsed.unwinds[0].expression.text == "p.vals"
    assert parsed.unwinds[0].alias == "v"


def test_parse_match_with_comma_connected_patterns() -> None:
    parsed = parse_cypher("MATCH (a)-[:A]->(b), (b)-[:B]->(c) RETURN c")

    assert parsed.match is not None
    assert len(parsed.match.patterns) == 2
    assert cast(NodePattern, parsed.match.patterns[0][0]).variable == "a"
    assert cast(NodePattern, parsed.match.patterns[0][2]).variable == "b"
    assert cast(NodePattern, parsed.match.patterns[1][0]).variable == "b"
    assert cast(NodePattern, parsed.match.patterns[1][2]).variable == "c"


def test_parse_multiple_match_clauses() -> None:
    parsed = parse_cypher("MATCH (a {name: 'A'}), (b {name: 'B'}) MATCH (a)-->(x)<--(b) RETURN x")

    assert len(parsed.matches) == 2
    assert len(parsed.matches[0].patterns) == 2
    assert len(parsed.matches[1].patterns) == 1
    assert cast(NodePattern, parsed.matches[0].patterns[0][0]).variable == "a"
    assert cast(NodePattern, parsed.matches[0].patterns[1][0]).variable == "b"
    assert cast(NodePattern, parsed.matches[1].patterns[0][2]).variable == "x"


def test_parse_optional_match_clause() -> None:
    parsed = parse_cypher("OPTIONAL MATCH (n) RETURN n.exists IS NULL AS missing")

    assert len(parsed.matches) == 1
    assert parsed.matches[0].optional is True
    assert cast(NodePattern, parsed.matches[0].patterns[0][0]).variable == "n"


def test_parse_aggregate_projection_items() -> None:
    parsed = parse_cypher(
        "MATCH (n) RETURN n.division AS division, count(*) AS cnt, max(n.age) AS max_age ORDER BY division, cnt DESC"
    )

    assert [item.expression.text for item in parsed.return_.items] == [
        "n.division",
        "count(*)",
        "max(n.age)",
    ]
    assert [item.alias for item in parsed.return_.items] == ["division", "cnt", "max_age"]
    assert parsed.order_by is not None
    assert [item.expression.text for item in parsed.order_by.items] == ["division", "cnt"]


def test_parse_top_level_projection_only() -> None:
    parsed = parse_cypher("RETURN [1, 2, 3] AS xs LIMIT 1")

    assert parsed.match is None
    assert parsed.unwinds == ()
    assert parsed.return_.items[0].expression.text == "[1, 2, 3]"
    assert parsed.return_.items[0].alias == "xs"
    assert parsed.limit is not None and isinstance(parsed.limit.value, ExpressionText)
    assert parsed.limit.value.text == "1"


def test_parse_top_level_quantifier_expression() -> None:
    parsed = parse_cypher("RETURN none(x IN [true, false] WHERE x) AS result")

    assert parsed.match is None
    assert parsed.return_.items[0].expression.text == "none(x IN [true, false] WHERE x)"
    assert parsed.return_.items[0].alias == "result"


def test_parse_top_level_membership_and_null_expression() -> None:
    parsed = parse_cypher("RETURN 3 IN [1, 2, 3] AS hit, null IS NULL AS empty")

    assert [item.expression.text for item in parsed.return_.items] == [
        "3 IN [1, 2, 3]",
        "null IS NULL",
    ]


def test_parse_top_level_list_comprehension_expression() -> None:
    parsed = parse_cypher("RETURN [x IN [1, 2, 3] WHERE x > 1 | x + 10] AS vals")

    assert parsed.return_.items[0].expression.text == "[x IN [1, 2, 3] WHERE x > 1 | x + 10]"


def test_invalid_syntax_reports_line_and_column() -> None:
    with pytest.raises(GFQLSyntaxError) as exc_info:
        parse_cypher("MATCH (n RETURN n")

    exc = exc_info.value
    assert exc.code == ErrorCode.E107
    assert exc.context["line"] == 1
    assert exc.context["column"] >= 1
    assert "Invalid Cypher query syntax" in str(exc)


def test_multi_statement_rejected() -> None:
    with pytest.raises(GFQLSyntaxError) as exc_info:
        parse_cypher("MATCH (n) RETURN n; MATCH (m) RETURN m")

    assert exc_info.value.code == ErrorCode.E107


def test_or_where_not_yet_supported() -> None:
    with pytest.raises(GFQLSyntaxError) as exc_info:
        parse_cypher("MATCH (a)-[r]->(b) WHERE a.team = b.team OR a.name = 'Alice' RETURN a")

    assert exc_info.value.code == ErrorCode.E107
