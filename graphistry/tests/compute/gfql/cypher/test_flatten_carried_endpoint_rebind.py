"""Direct unit tests for the carried-endpoint rebind flattener (#1341).

Anchors each disqualification branch in ``flatten_carried_endpoint_rebind``
so future edits to the narrow shape boundary are caught by tests rather
than indirectly via reentry-path regressions.
"""
from __future__ import annotations

from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.cypher.reentry.flatten import (
    flatten_carried_endpoint_rebind,
)


def _parse(query: str):
    return parse_cypher(query)


def test_flatten_admits_ic1_shortest_path_shape() -> None:
    q = _parse(
        "MATCH (p:Person {id: 'p1'}), (friend:Person) "
        "WHERE NOT p = friend "
        "WITH p, friend "
        "MATCH path = shortestPath((p)-[:KNOWS*1..3]-(friend)) "
        "RETURN friend.id AS friendId"
    )
    flattened = flatten_carried_endpoint_rebind(q)
    assert flattened is not None
    assert flattened.with_stages == ()
    assert flattened.reentry_matches == ()
    # 3 patterns: (p), (friend), shortestPath
    assert len(flattened.matches) == 1
    assert len(flattened.matches[0].patterns) == 3


def test_flatten_admits_simple_rebind_with_edge() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id AS bid"
    )
    flattened = flatten_carried_endpoint_rebind(q)
    assert flattened is not None
    assert flattened.with_stages == ()


def test_flatten_disqualifies_no_relationship_in_trailing_pattern() -> None:
    """``WITH a MATCH (a) RETURN a`` — no edge, would create redundant alias binding."""
    q = _parse("MATCH (a:A) WITH a MATCH (a) RETURN a")
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_distinct_with() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH DISTINCT a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_alias_rename_in_with() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b AS bb "
        "MATCH (bb)-[:S]->(a) "
        "RETURN bb.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_with_order_by() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b ORDER BY b.id "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_with_limit() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b LIMIT 5 "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_fresh_trailing_alias() -> None:
    """Trailing pattern introduces a fresh node alias (``c``); not a rebind."""
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN c.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_optional_trailing_match() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "OPTIONAL MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_optional_prefix_match() -> None:
    q = _parse(
        "OPTIONAL MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_when_query_has_no_reentry() -> None:
    q = _parse("MATCH (a:A {id: 'a1'})-[:R]->(b:B) RETURN b.id")
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_multiple_with_stages() -> None:
    """Two-stage WITH chain → flatten only handles single-WITH narrow shape."""
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH a, b, c "
        "MATCH (c)-[:T]->(a) "
        "RETURN a.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_reentry_where_present() -> None:
    """A WHERE between trailing MATCH and RETURN is a reentry_where → disqualify."""
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "WHERE b.score > 5 "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_partial_carry_drops_prefix_alias() -> None:
    """WITH drops a prefix-bound alias; the existing reentry path emits a clean
    scope error if RETURN references the dropped alias. Flatten must not
    silently admit the merged form, since merging would re-introduce the
    dropped alias into RETURN scope (#1341 wave-2 review)."""
    q = _parse(
        "MATCH (a:A)-[:R]->(c:C) "
        "WITH a "
        "MATCH (a)-[:S]->(a) "
        "RETURN c.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None
