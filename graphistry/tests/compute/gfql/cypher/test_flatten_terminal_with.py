"""Direct tests for terminal-WITH-over-OPTIONAL flattening (#1896).

Every admit/decline branch of ``flatten_terminal_with_over_optional`` gets a
direct AST-level pin, mirroring ``test_flatten_pure_carry_optional.py``.
"""
from __future__ import annotations

import dataclasses

import pytest

from graphistry.compute.gfql.cypher.ast import CypherQuery
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.cypher.reentry.flatten import (
    flatten_terminal_with_over_optional,
)


def _parse(query: str) -> CypherQuery:
    parsed = parse_cypher(query)
    assert isinstance(parsed, CypherQuery)
    return parsed


BASE = "MATCH (a:P) OPTIONAL MATCH (a)-[r:KNOWS]->(b) "


def test_admits_pure_carry_with_where() -> None:
    q = _parse(BASE + "WITH a, b WHERE a.v <= 2 RETURN a.id AS aid, b.id AS bid")
    out = flatten_terminal_with_over_optional(q)
    assert out is not None
    flattened, row_filter = out
    assert flattened.with_stages == ()
    assert row_filter is not None


def test_admits_pure_carry_without_where() -> None:
    q = _parse(BASE + "WITH a, b RETURN a.id AS aid, b.id AS bid")
    out = flatten_terminal_with_over_optional(q)
    assert out is not None
    _, row_filter = out
    assert row_filter is None


def test_admits_terminal_aggregate_substitution() -> None:
    q = _parse(BASE + "WITH a.id AS aid, count(b) AS cnt RETURN aid, cnt")
    out = flatten_terminal_with_over_optional(q)
    assert out is not None
    flattened, row_filter = out
    assert flattened.with_stages == ()
    assert row_filter is None
    texts = [i.expression.text for i in flattened.return_.items]
    assert any("count" in t for t in texts)


@pytest.mark.parametrize(
    "query",
    [
        # no OPTIONAL MATCH anywhere
        "MATCH (a:P) WITH a RETURN a.id",
        # first MATCH optional
        "OPTIONAL MATCH (a:P) WITH a RETURN a.id",
        # two WITH stages
        BASE + "WITH a, b WITH a, b RETURN a.id",
        # UNWIND present
        "MATCH (a:P) OPTIONAL MATCH (a)-->(b) UNWIND [1] AS x WITH a, b, x RETURN a.id",
        # ORDER BY / SKIP / LIMIT on the stage
        BASE + "WITH a, b ORDER BY a.v RETURN a.id",
        BASE + "WITH a, b LIMIT 2 RETURN a.id",
        # DISTINCT on stage and on RETURN
        BASE + "WITH DISTINCT a, b RETURN a.id",
        BASE + "WITH a, b RETURN DISTINCT a.id",
        # subset carry + downstream reference to dropped alias
        BASE + "WITH a WHERE a.v <= 2 RETURN a.id, b.id",
        # non-pure stage (rename) with a stage WHERE
        BASE + "WITH a.id AS aid WHERE a.v <= 2 RETURN aid",
        # substitution: RETURN references something not in the stage
        BASE + "WITH a.id AS aid RETURN aid, b.id",
        # whole-row carried alias next to an aggregate stays typed-declined
        BASE + "WITH b, count(a) AS cnt RETURN b, cnt",
    ],
)
def test_declines(query: str) -> None:
    assert flatten_terminal_with_over_optional(_parse(query)) is None


def test_declines_reentry_shape() -> None:
    # WITH before the OPTIONAL MATCH makes it a reentry query, not terminal-WITH.
    q = _parse("MATCH (a:P) WITH a OPTIONAL MATCH (a)-->(b) RETURN a.id, b.id")
    assert q.reentry_matches
    assert flatten_terminal_with_over_optional(q) is None


def test_declines_row_sequence() -> None:
    q = _parse(BASE + "WITH a, b RETURN a.id")
    hacked = dataclasses.replace(q, row_sequence=q.row_sequence or ("row",))
    assert flatten_terminal_with_over_optional(hacked) is None


def test_declines_duplicate_stage_output_names() -> None:
    q = _parse(BASE + "WITH a.v AS x, b.v AS y RETURN x, y")
    dup = dataclasses.replace(
        q.with_stages[0].clause,
        items=(
            q.with_stages[0].clause.items[0],
            dataclasses.replace(
                q.with_stages[0].clause.items[1], alias="x"
            ),
        ),
    )
    hacked = dataclasses.replace(
        q, with_stages=(dataclasses.replace(q.with_stages[0], clause=dup),)
    )
    assert flatten_terminal_with_over_optional(hacked) is None


def test_admits_carried_property_projection_through_substitution() -> None:
    # `o.prop` of a bare-carried alias output passes through.
    q = _parse(BASE + "WITH b, a.id AS aid RETURN aid, b.id")
    out = flatten_terminal_with_over_optional(q)
    assert out is not None


def test_order_by_downstream_scope_check() -> None:
    # ORDER BY referencing a non-carried match alias declines the pure carry.
    q = _parse(BASE + "WITH a WHERE a.v <= 2 RETURN a.id AS aid ORDER BY b.v")
    assert flatten_terminal_with_over_optional(q) is None
