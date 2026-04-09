import pytest

from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher.ast import CypherQuery
from graphistry.compute.gfql.cypher.ast_normalizer import ASTNormalizer
from graphistry.compute.gfql.cypher.parser import parse_cypher


def _normalize(query: str) -> CypherQuery:
    parsed = parse_cypher(query)
    assert isinstance(parsed, CypherQuery)
    return ASTNormalizer().normalize(parsed)


def test_normalizer_rewrites_shortest_path_seed_and_projection() -> None:
    normalized = _normalize(
        "MATCH (a:Person), path = shortestPath((a)-[:KNOWS*]-(b)) "
        "RETURN length(path) AS hops"
    )

    assert len(normalized.matches) == 2
    assert normalized.matches[0].pattern_alias_kinds == ("pattern",)
    assert normalized.matches[1].pattern_alias_kinds == ("shortestPath",)
    assert "__cypher_shortest_path_hops__path" in normalized.return_.items[0].expression.text


def test_normalizer_rewrites_where_pattern_predicate_into_extra_match() -> None:
    normalized = _normalize(
        "MATCH (a:Person), (b:Person) "
        "WHERE (a)-[:KNOWS]->(b) AND a.id = 'a' "
        "RETURN a"
    )

    assert len(normalized.matches) == 2
    assert len(normalized.matches[-1].patterns) == 1
    assert normalized.where is not None
    assert normalized.where.expr is not None
    assert "a.id" in normalized.where.expr.text


def test_normalizer_rejects_shortest_path_inside_optional_match() -> None:
    with pytest.raises(GFQLValidationError, match="OPTIONAL MATCH"):
        _normalize(
            "MATCH (person1:Person {id: 'p1'}) "
            "OPTIONAL MATCH (person2:Person {id: 'p2'}), "
            "path = shortestPath((person1)-[:KNOWS*]-(person2)) "
            "RETURN CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength"
        )


def test_normalizer_rejects_where_pattern_predicate_new_alias_introduction() -> None:
    parsed = parse_cypher("MATCH (a:Person) WHERE (a)-[:KNOWS]->(b) RETURN a")
    assert isinstance(parsed, CypherQuery)

    with pytest.raises(GFQLValidationError, match="cannot introduce new aliases"):
        ASTNormalizer().rewrite_where_pattern_predicates(parsed)
