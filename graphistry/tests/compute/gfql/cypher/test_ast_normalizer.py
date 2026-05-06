from __future__ import annotations

import pytest

from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher._boolean_expr_text import boolean_expr_to_text
from graphistry.compute.gfql.cypher.ast import CypherQuery
from graphistry.compute.gfql.cypher.api import compile_cypher
from graphistry.compute.gfql.cypher.ast_normalizer import ASTNormalizer
from graphistry.compute.gfql.cypher.lowering import lower_match_query
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
    assert normalized.where.expr_tree is not None
    assert "a.id" in boolean_expr_to_text(normalized.where.expr_tree)


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


def _capture_normalizer_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def _record_shortest(self: ASTNormalizer, query: CypherQuery) -> CypherQuery:
        calls.append("shortest")
        return query

    def _record_where(self: ASTNormalizer, query: CypherQuery) -> CypherQuery:
        calls.append("where")
        return query

    monkeypatch.setattr(ASTNormalizer, "rewrite_shortest_path", _record_shortest)
    monkeypatch.setattr(ASTNormalizer, "rewrite_where_pattern_predicates", _record_where)
    return calls


def test_compile_cypher_invokes_ast_normalizer_rewrites(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_normalizer_calls(monkeypatch)
    compiled = compile_cypher("RETURN 1 AS x")
    assert compiled is not None
    assert "shortest" in calls
    assert "where" in calls


def test_compile_cypher_graph_constructor_invokes_ast_normalizer_rewrites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_normalizer_calls(monkeypatch)
    compiled = compile_cypher("GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'a' }")
    assert compiled is not None
    assert "shortest" in calls
    assert "where" in calls


def test_lower_match_query_invokes_ast_normalizer_rewrites(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = parse_cypher("MATCH (n) RETURN n")
    assert isinstance(parsed, CypherQuery)
    calls = _capture_normalizer_calls(monkeypatch)

    lowered = lower_match_query(parsed)
    assert lowered is not None
    assert calls[:2] == ["shortest", "where"]
