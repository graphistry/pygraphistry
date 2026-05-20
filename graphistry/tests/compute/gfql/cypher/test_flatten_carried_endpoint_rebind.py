"""Direct tests for carried-endpoint reentry flattening."""
from __future__ import annotations

import dataclasses
import typing
from typing import Any, Dict, Literal

import pytest

from graphistry.compute.gfql.cypher.ast import (
    CypherQuery,
    MatchClause,
    PatternElement,
    SourceSpan,
    WhereClause,
)
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.cypher.reentry.flatten import (
    flatten_carried_endpoint_rebind,
    _all_pattern_aliases,
)


def _parse(query: str) -> CypherQuery:
    parsed = parse_cypher(query)
    assert isinstance(parsed, CypherQuery)
    return parsed


@pytest.mark.parametrize(
    "query,pattern_count",
    [
        (
            "MATCH (p:Person {id: 'p1'}), (friend:Person) "
            "WHERE NOT p = friend "
            "WITH p, friend "
            "MATCH path = shortestPath((p)-[:KNOWS*1..3]-(friend)) "
            "RETURN friend.id AS friendId",
            3,
        ),
        (
            "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
            "WITH a, b "
            "MATCH (b)-[:S]->(a) "
            "RETURN b.id AS bid",
            2,
        ),
        (
            "MATCH (a:A)-[r:R]->(b:B) "
            "WITH a, b, r "
            "MATCH (b)-[:S]->(a) "
            "RETURN r.weight",
            2,
        ),
    ],
)
def test_flatten_admits_rebind_shapes(query: str, pattern_count: int) -> None:
    flattened = flatten_carried_endpoint_rebind(_parse(query))
    assert flattened is not None
    assert flattened.with_stages == ()
    assert flattened.reentry_matches == ()
    assert len(flattened.matches) == 1
    assert len(flattened.matches[0].patterns) == pattern_count


def test_flatten_preserves_prefix_match_where_on_merged_match() -> None:
    q = _parse(
        "MATCH (p:Person {id: 'p1'}), (friend:Person) "
        "WHERE NOT p = friend "
        "WITH p, friend "
        "MATCH path = shortestPath((p)-[:KNOWS*1..3]-(friend)) "
        "RETURN friend.id AS friendId"
    )
    flattened = flatten_carried_endpoint_rebind(q)
    assert flattened is not None
    assert flattened.matches[0].where is q.matches[0].where


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a:A) WITH a MATCH (a) RETURN a",
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) WITH DISTINCT a, b MATCH (b)-[:S]->(a) RETURN b.id",
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) WITH a, b AS bb MATCH (bb)-[:S]->(a) RETURN bb.id",
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) WITH a, b ORDER BY b.id MATCH (b)-[:S]->(a) RETURN b.id",
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) WITH a, b LIMIT 5 MATCH (b)-[:S]->(a) RETURN b.id",
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) WITH a, b SKIP 5 MATCH (b)-[:S]->(a) RETURN b.id",
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) WITH a, b.id MATCH (a)-[:S]->(b) RETURN a.id",
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) WITH a, b MATCH (b)-[:S]->(c:C) RETURN c.id",
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) RETURN b.id",
        (
            "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
            "WITH a, b MATCH (b)-[:S]->(c:C) "
            "WITH a, b, c MATCH (c)-[:T]->(a) RETURN a.id"
        ),
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) WITH a, b MATCH (b)-[:S]->(a) WHERE b.score > 5 RETURN b.id",
    ],
)
def test_flatten_disqualifies_non_rebind_shapes(query: str) -> None:
    assert flatten_carried_endpoint_rebind(_parse(query)) is None


def test_flatten_alias_kind_enumeration_locks_in_with_ast() -> None:
    """Lock prefix-alias collection to current variable-bearing AST surfaces."""
    pattern_subtypes = typing.get_args(PatternElement)
    assert pattern_subtypes, "PatternElement Union must have at least one member"
    span = SourceSpan(line=1, column=1, end_line=1, end_column=1, start_pos=0, end_pos=0)
    asserted_subtype = False
    asserted_subtype_names: set = set()
    for subtype in pattern_subtypes:
        type_hints = typing.get_type_hints(subtype)
        if "variable" not in type_hints:
            continue
        kwargs: Dict[str, Any] = {"variable": "probe", "span": span}
        for fld in dataclasses.fields(subtype):
            if fld.name in kwargs:
                continue
            if fld.default is not dataclasses.MISSING:
                continue
            if fld.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                continue
            ftype = type_hints.get(fld.name)
            origin = typing.get_origin(ftype)
            if origin is tuple:
                kwargs[fld.name] = ()
                continue
            if origin is Literal:
                literal_args = typing.get_args(ftype)
                if literal_args:
                    kwargs[fld.name] = literal_args[0]
                    continue
            raise AssertionError(
                f"Lock-in test cannot synthesize required field {fld.name!r} of "
                f"type {ftype!r} on {subtype.__name__}; extend the filler logic "
                f"and verify ``_all_pattern_aliases`` handles the new shape."
            )
        instance = subtype(**kwargs)
        observed = _all_pattern_aliases((instance,))
        assert "probe" in observed, (
            f"_all_pattern_aliases must collect variable from {subtype.__name__} "
            f"(got {observed!r})"
        )
        asserted_subtype = True
        asserted_subtype_names.add(subtype.__name__)
    assert asserted_subtype, (
        "PatternElement enumeration produced no variable-bearing subtype probe; "
        "the lock-in test must assert against at least one concrete subtype."
    )
    assert {"NodePattern", "RelationshipPattern"}.issubset(asserted_subtype_names), (
        f"Lock-in must probe both NodePattern and RelationshipPattern; "
        f"got {sorted(asserted_subtype_names)}"
    )

    matchclause_field_names = {f.name for f in dataclasses.fields(MatchClause)}
    assert "pattern_aliases" in matchclause_field_names, (
        "MatchClause.pattern_aliases removed/renamed — flatten prefix-alias "
        "union step must be updated to track the new field."
    )
    q = _parse(
        "MATCH path = (a:A)-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN length(path) AS n"
    )
    assert flatten_carried_endpoint_rebind(q) is None, (
        "Path alias dropped at WITH must disqualify flatten — prefix-alias "
        "union step must observe MatchClause.pattern_aliases."
    )


@pytest.mark.parametrize(
    "kind,prefix,carry,trailing,return_clause",
    [
        (
            "node",
            "MATCH (a:A)-[:R]->(b:B), (c:C)",
            "WITH a, b",
            "MATCH (b)-[:S]->(a)",
            "RETURN c.id",
        ),
        (
            "rel",
            "MATCH (a:A)-[r:R]->(b:B)",
            "WITH a, b",
            "MATCH (b)-[:S]->(a)",
            "RETURN r.weight",
        ),
        (
            "path",
            "MATCH path = (a:A)-[:R]->(b:B)",
            "WITH a, b",
            "MATCH (b)-[:S]->(a)",
            "RETURN length(path) AS n",
        ),
    ],
)
def test_flatten_disqualifies_partial_carry_for_each_alias_kind(
    kind: str, prefix: str, carry: str, trailing: str, return_clause: str
) -> None:
    q = _parse(f"{prefix} {carry} {trailing} {return_clause}")
    assert flatten_carried_endpoint_rebind(q) is None, (
        f"Dropping a prefix-bound {kind} alias must disqualify flatten"
    )


@pytest.mark.parametrize(
    "prefix_optional,trailing_optional",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_flatten_disqualifies_optional_match_combinations(
    prefix_optional: bool, trailing_optional: bool
) -> None:
    p = "OPTIONAL " if prefix_optional else ""
    t = "OPTIONAL " if trailing_optional else ""
    q = _parse(
        f"{p}MATCH (a:A {{id: 'a1'}})-[:R]->(b:B) "
        f"WITH a, b "
        f"{t}MATCH (b)-[:S]->(a) "
        f"RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_both_inline_wheres_via_built_ast() -> None:
    base = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    parsed_with_where = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WHERE a.score > 5 "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    prefix_where: WhereClause = parsed_with_where.matches[0].where  # type: ignore[assignment]
    assert prefix_where is not None
    trailing = base.reentry_matches[0]
    forged_trailing = dataclasses.replace(trailing, where=prefix_where)
    forged_query = dataclasses.replace(
        base,
        matches=(dataclasses.replace(base.matches[0], where=prefix_where),),
        reentry_matches=(forged_trailing,),
    )
    assert flatten_carried_endpoint_rebind(forged_query) is None
