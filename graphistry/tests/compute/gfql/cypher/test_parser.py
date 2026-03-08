import pytest

from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError
from graphistry.compute.gfql.cypher import (
    NodePattern,
    ParameterRef,
    RelationshipPattern,
    parse_cypher,
)


def test_parse_minimal_match_return() -> None:
    parsed = parse_cypher("MATCH (n) RETURN n")

    assert parsed.match.pattern[0].variable == "n"
    assert parsed.return_.distinct is False
    assert parsed.return_.items[0].expression.text == "n"
    assert parsed.return_.items[0].alias is None
    assert parsed.trailing_semicolon is False


def test_parse_linear_pattern_with_labels_properties_and_aliases() -> None:
    parsed = parse_cypher(
        'MATCH (p:Person {id: $person_id})-[r:FOLLOWS]->(q:Person {active: true}) RETURN p.name AS person_name, q'
    )

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
    ],
)
def test_parse_relationship_directions(query: str, direction: str) -> None:
    parsed = parse_cypher(query)

    rel = parsed.match.pattern[1]
    assert isinstance(rel, RelationshipPattern)
    assert rel.direction == direction


def test_parse_reports_trailing_semicolon() -> None:
    parsed = parse_cypher("MATCH (n) RETURN n;")
    assert parsed.trailing_semicolon is True


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
