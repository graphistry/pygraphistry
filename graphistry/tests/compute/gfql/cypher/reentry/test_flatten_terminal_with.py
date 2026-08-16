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
        pytest.param("MATCH (a:P) WITH a RETURN a.id", id="no_optional_match_anywhere"),
        pytest.param("OPTIONAL MATCH (a:P) WITH a RETURN a.id", id="lead_match_is_optional"),
        pytest.param(BASE + "WITH a, b WITH a, b RETURN a.id", id="two_with_stages"),
        pytest.param(
            "MATCH (a:P) OPTIONAL MATCH (a)-->(b) UNWIND [1] AS x WITH a, b, x RETURN a.id",
            id="unwind_present",
        ),
        pytest.param(BASE + "WITH a, b ORDER BY a.v RETURN a.id", id="order_by_on_stage"),
        pytest.param(BASE + "WITH a, b LIMIT 2 RETURN a.id", id="limit_on_stage"),
        pytest.param(BASE + "WITH DISTINCT a, b RETURN a.id", id="distinct_on_stage"),
        pytest.param(BASE + "WITH a, b RETURN DISTINCT a.id", id="distinct_on_return"),
        pytest.param(
            BASE + "WITH a WHERE a.v <= 2 RETURN a.id, b.id",
            id="subset_carry_with_downstream_reference_to_dropped_alias",
        ),
        pytest.param(
            BASE + "WITH a.id AS aid WHERE a.v <= 2 RETURN aid",
            id="renaming_stage_with_a_stage_where",
        ),
        pytest.param(
            BASE + "WITH a.id AS aid RETURN aid, b.id",
            id="return_references_something_not_in_the_stage",
        ),
        pytest.param(
            BASE + "WITH b, count(a) AS cnt RETURN b, cnt",
            id="whole_row_carried_alias_next_to_an_aggregate",
        ),
        # every aggregate name the stage-aggregate detector knows: dropping any
        # one of them from its pattern admits a whole-row carry beside it
        pytest.param(
            BASE + "WITH b, collect(a.id) AS ids RETURN b, ids",
            id="whole_row_carried_alias_next_to_collect",
        ),
        pytest.param(
            BASE + "WITH b, sum(a.v) AS total RETURN b, total",
            id="whole_row_carried_alias_next_to_sum",
        ),
        pytest.param(
            BASE + "WITH b, max(a.v) AS hi RETURN b, hi",
            id="whole_row_carried_alias_next_to_max",
        ),
    ],
)
def test_declines(query: str) -> None:
    assert flatten_terminal_with_over_optional(_parse(query)) is None


def test_declines_when_with_precedes_the_optional_match_reentry_shape() -> None:
    q = _parse("MATCH (a:P) WITH a OPTIONAL MATCH (a)-->(b) RETURN a.id, b.id")
    assert q.reentry_matches
    assert flatten_terminal_with_over_optional(q) is None


def test_declines_a_trailing_match_reentry_even_with_an_optional_arm_in_front() -> None:
    """The reentry-matches guard, isolated. The shape above declines on the
    ``any(m.optional)`` test instead -- its OPTIONAL MATCH moved into
    ``reentry_matches``, so ``query.matches`` holds no optional at all and the
    reentry guard is never the deciding one. Here the OPTIONAL arm stays in
    ``query.matches`` and only the trailing MATCH is a reentry, so every other
    precondition passes and ``reentry_matches`` alone must stop the flatten --
    dropping the stage would silently discard the trailing MATCH's re-entry."""
    q = _parse("MATCH (a:P) OPTIONAL MATCH (a)-[:KNOWS]->(b) WITH a, b "
               "MATCH (b)-[:KNOWS]->(c) RETURN a.id AS aid, c.id AS cid")
    assert q.reentry_matches
    assert len(q.with_stages) == 1
    assert any(m.optional for m in q.matches) and not q.matches[0].optional
    assert flatten_terminal_with_over_optional(q) is None


def test_declines_row_sequence() -> None:
    q = _parse(BASE + "WITH a, b RETURN a.id")
    hacked = dataclasses.replace(q, row_sequence=q.row_sequence or ("row",))
    assert flatten_terminal_with_over_optional(hacked) is None


def test_declines_a_call_clause() -> None:
    """A CALL feeds rows the flatten's pattern-only alias analysis cannot see,
    so its presence must veto the rewrite -- the same reason row_sequence does."""
    call = _parse("CALL db.labels() YIELD label RETURN label").call
    assert call is not None
    q = _parse(BASE + "WITH a, b RETURN a.id")
    assert flatten_terminal_with_over_optional(dataclasses.replace(q, call=call)) is None


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


def test_admits_property_of_a_bare_carried_alias_through_substitution() -> None:
    q = _parse(BASE + "WITH b, a.id AS aid RETURN aid, b.id")
    out = flatten_terminal_with_over_optional(q)
    assert out is not None


def test_declines_pure_carry_when_order_by_references_an_uncarried_alias() -> None:
    q = _parse(BASE + "WITH a WHERE a.v <= 2 RETURN a.id AS aid ORDER BY b.v")
    assert flatten_terminal_with_over_optional(q) is None


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(BASE + "WITH a, b WHERE a.v <= 2 RETURN a.id AS aid, b.id AS bid",
                     id="pure_carry_with_where"),
        pytest.param(BASE + "WITH a, b RETURN a.id AS aid, b.id AS bid",
                     id="pure_carry_without_where"),
        pytest.param(BASE + "WITH a.id AS aid, count(b) AS cnt RETURN aid, cnt",
                     id="terminal_aggregate_substitution"),
        pytest.param(BASE + "WITH b, a.id AS aid RETURN aid, b.id",
                     id="property_of_a_bare_carried_alias"),
    ],
)
def test_admitted_query_never_retains_a_with_stage_so_recompiling_it_terminates(query: str) -> None:
    out = flatten_terminal_with_over_optional(_parse(query))
    assert out is not None
    flattened, _ = out
    assert flattened.with_stages == ()
    assert flatten_terminal_with_over_optional(flattened) is None
