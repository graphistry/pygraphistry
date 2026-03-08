from typing import cast

import pytest

from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError
from graphistry.compute.gfql.cypher import (
    LabelRef,
    NodePattern,
    ParameterRef,
    PropertyRef,
    RelationshipPattern,
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
    left0 = cast(PropertyRef, parsed.where.predicates[0].left)
    left1 = cast(PropertyRef, parsed.where.predicates[1].left)
    assert left0.alias == "a"
    assert left0.property == "team"
    assert parsed.where.predicates[0].op == "=="
    assert left1.alias == "b"
    assert left1.property == "score"
    assert parsed.where.predicates[1].op == ">="


def test_parse_where_null_predicates() -> None:
    parsed = parse_cypher("MATCH (a)-[r]->(b) WHERE a.deleted IS NULL AND b.name IS NOT NULL RETURN a")

    assert parsed.where is not None
    assert parsed.where.predicates[0].op == "is_null"
    assert parsed.where.predicates[0].right is None
    assert parsed.where.predicates[1].op == "is_not_null"
    assert parsed.where.predicates[1].right is None


def test_parse_where_label_predicate() -> None:
    parsed = parse_cypher("MATCH (a)-->(b) WHERE b:Foo:Bar RETURN b")

    assert parsed.where is not None
    assert len(parsed.where.predicates) == 1
    assert parsed.where.predicates[0].op == "has_labels"
    left = cast(LabelRef, parsed.where.predicates[0].left)
    assert left.alias == "b"
    assert left.labels == ("Foo", "Bar")


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
    assert parsed.skip is not None and parsed.skip.value == 1
    assert parsed.limit is not None and parsed.limit.value == 2
    assert parsed.trailing_semicolon is True


def test_parse_terminal_with_clause() -> None:
    parsed = parse_cypher("MATCH (p) WITH p.name AS person_name ORDER BY person_name ASC LIMIT 5")

    assert parsed.return_.kind == "with"
    assert parsed.order_by is not None
    assert parsed.order_by.items[0].expression.text == "person_name"
    assert parsed.limit is not None and parsed.limit.value == 5


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
    assert parsed.limit is not None and parsed.limit.value == 1


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
