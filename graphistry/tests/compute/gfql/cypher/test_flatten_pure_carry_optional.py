"""Direct tests for pure-carry OPTIONAL MATCH reentry flattening (#1891).

Every admit/decline branch of ``flatten_pure_carry_optional_reentry`` gets a
direct pin, mirroring ``test_flatten_carried_endpoint_rebind.py``: the
transform is pure AST -> AST, so shapes parse (or are surgically built with
``dataclasses.replace``) and the result is asserted structurally, no engine
execution involved.
"""
from __future__ import annotations

import dataclasses

import pytest

from graphistry.compute.gfql.cypher.ast import CypherQuery
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.cypher.reentry.flatten import (
    _normalized_aliases,
    _normalized_kinds,
    flatten_carried_endpoint_rebind,
    flatten_pure_carry_optional_reentry,
)


def _parse(query: str) -> CypherQuery:
    parsed = parse_cypher(query)
    assert isinstance(parsed, CypherQuery)
    return parsed


def test_flatten_admits_pure_carry_optional() -> None:
    q = _parse(
        "MATCH (a:A) WITH a OPTIONAL MATCH (a)-[r:R]->(b) RETURN b"
    )
    flattened = flatten_pure_carry_optional_reentry(q)
    assert flattened is not None
    assert flattened.with_stages == ()
    assert flattened.reentry_matches == ()
    assert flattened.reentry_wheres == ()
    assert len(flattened.matches) == 2
    assert not flattened.matches[0].optional
    assert flattened.matches[1].optional


def test_flatten_attaches_reentry_where_to_optional_clause() -> None:
    q = _parse(
        "MATCH (a:A) WITH a OPTIONAL MATCH (a)-[r:R]->(b) WHERE b.v = 1 RETURN b"
    )
    assert any(w is not None for w in q.reentry_wheres)
    flattened = flatten_pure_carry_optional_reentry(q)
    assert flattened is not None
    assert flattened.matches[1].where is not None


def test_flatten_declines_no_reentry_matches() -> None:
    q = _parse("MATCH (a:A) RETURN a")
    assert flatten_pure_carry_optional_reentry(q) is None


def test_flatten_declines_multiple_with_stages() -> None:
    q = _parse(
        "MATCH (a:A) WITH a WITH a OPTIONAL MATCH (a)-[r:R]->(b) RETURN b"
    )
    assert flatten_pure_carry_optional_reentry(q) is None


def test_flatten_declines_multiple_prefix_matches() -> None:
    q = _parse(
        "MATCH (a:A) MATCH (c:C) WITH a, c "
        "OPTIONAL MATCH (a)-[r:R]->(b) RETURN b"
    )
    assert flatten_pure_carry_optional_reentry(q) is None


def test_flatten_declines_unwind() -> None:
    q = _parse(
        "MATCH (a:A) UNWIND [1, 2] AS x WITH a, x "
        "OPTIONAL MATCH (a)-[r:R]->(b) RETURN b"
    )
    assert flatten_pure_carry_optional_reentry(q) is None


def test_flatten_declines_row_sequence() -> None:
    q = _parse("MATCH (a:A) WITH a OPTIONAL MATCH (a)-[r:R]->(b) RETURN b")
    hacked = dataclasses.replace(q, row_sequence=q.row_sequence or ("row",))
    assert flatten_pure_carry_optional_reentry(hacked) is None


def test_flatten_declines_optional_prefix_match() -> None:
    q = _parse(
        "OPTIONAL MATCH (a:A) WITH a OPTIONAL MATCH (a)-[r:R]->(b) RETURN b"
    )
    assert flatten_pure_carry_optional_reentry(q) is None


def test_flatten_declines_non_optional_trailing_match() -> None:
    q = _parse("MATCH (a:A) WITH a MATCH (a)-[r:R]->(b) RETURN b")
    assert flatten_pure_carry_optional_reentry(q) is None


@pytest.mark.parametrize(
    "query",
    [
        # WITH ... WHERE is not a pure carry
        "MATCH (a:A) WITH a WHERE a.v > 1 OPTIONAL MATCH (a)-[r:R]->(b) RETURN b",
        # ORDER BY / SKIP / LIMIT on the WITH stage
        "MATCH (a:A) WITH a ORDER BY a.v OPTIONAL MATCH (a)-[r:R]->(b) RETURN b",
        "MATCH (a:A) WITH a LIMIT 1 OPTIONAL MATCH (a)-[r:R]->(b) RETURN b",
        # DISTINCT disqualifies
        "MATCH (a:A) WITH DISTINCT a OPTIONAL MATCH (a)-[r:R]->(b) RETURN b",
        # rename is not a bare carry
        "MATCH (a:A) WITH a AS x OPTIONAL MATCH (x)-[r:R]->(b) RETURN b",
        # non-bare projection item
        "MATCH (a:A) WITH a.v AS v OPTIONAL MATCH (a)-[r:R]->(b) RETURN b",
    ],
)
def test_flatten_declines_impure_with_stage(query: str) -> None:
    assert flatten_pure_carry_optional_reentry(_parse(query)) is None


def test_flatten_declines_with_dropping_prefix_alias() -> None:
    # WITH drops r and c: not scope-preserving.
    q = _parse(
        "MATCH (a:A)-[r:R]->(c:C) WITH a "
        "OPTIONAL MATCH (a)-[r2:S]->(b) RETURN b"
    )
    assert flatten_pure_carry_optional_reentry(q) is None


def test_flatten_declines_misaligned_reentry_wheres() -> None:
    q = _parse("MATCH (a:A) WITH a OPTIONAL MATCH (a)-[r:R]->(b) RETURN b")
    hacked = dataclasses.replace(q, reentry_wheres=(None, None))
    assert flatten_pure_carry_optional_reentry(hacked) is None


def test_flatten_declines_disconnected_trailing_pattern() -> None:
    q = _parse(
        "MATCH (a:A) WITH a OPTIONAL MATCH (x:X)-[r:R]->(y) RETURN y"
    )
    assert flatten_pure_carry_optional_reentry(q) is None


def test_flatten_declines_trailing_where_conflict() -> None:
    q = _parse(
        "MATCH (a:A) WITH a OPTIONAL MATCH (a)-[r:R]->(b) WHERE b.v = 1 RETURN b"
    )
    assert any(w is not None for w in q.reentry_wheres)
    conflicted = dataclasses.replace(
        q,
        reentry_matches=(
            dataclasses.replace(q.reentry_matches[0], where=q.reentry_wheres[0]),
        ),
    )
    assert flatten_pure_carry_optional_reentry(conflicted) is None


# --- sibling #1341 rebind guards sharing this module -------------------------


def test_rebind_declines_multiple_with_stages() -> None:
    q = _parse("MATCH (a:A) WITH a WITH a MATCH (a)-[r:R]->(b) RETURN b")
    assert flatten_carried_endpoint_rebind(q) is None


def test_rebind_declines_multiple_prefix_matches() -> None:
    q = _parse(
        "MATCH (a:A) MATCH (b:B) WITH a, b MATCH (a)-[r:R]->(b) RETURN b"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_rebind_declines_unwind() -> None:
    q = _parse(
        "MATCH (a:A) UNWIND [1, 2] AS x WITH a, x MATCH (a)-[r:R]->(a) RETURN a"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_rebind_declines_row_sequence() -> None:
    q = _parse("MATCH (a:A) WITH a MATCH (a)-[r:R]->(a) RETURN a")
    hacked = dataclasses.replace(q, row_sequence=q.row_sequence or ("row",))
    assert flatten_carried_endpoint_rebind(hacked) is None


# --- helper backfills --------------------------------------------------------


def test_normalized_aliases_backfills_none_per_pattern() -> None:
    assert _normalized_aliases((), ((), ())) == (None, None)
    assert _normalized_aliases(("p",), ((),)) == ("p",)


def test_normalized_kinds_backfills_pattern_kind() -> None:
    assert _normalized_kinds((), ((), ())) == ("pattern", "pattern")
    assert _normalized_kinds(("shortestPath",), ((),)) == ("shortestPath",)
