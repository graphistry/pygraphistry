from __future__ import annotations

import pytest

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.api import cypher_to_gfql
from graphistry.compute.gfql.cypher.lowering import _MAX_MATCH_CLAUSES


def _many_matches_query(clause_count: int) -> str:
    return " ".join(f"MATCH (n{index})" for index in range(clause_count)) + " RETURN n0"


def test_match_clause_budget_admits_supported_limit_before_binder() -> None:
    query = _many_matches_query(_MAX_MATCH_CLAUSES)

    with pytest.raises(GFQLValidationError) as raised:
        cypher_to_gfql(query)

    assert raised.value.code == ErrorCode.E108
    assert raised.value.context["field"] == "match"


def test_match_clause_budget_fails_fast_beyond_limit() -> None:
    clause_count = _MAX_MATCH_CLAUSES + 1
    query = _many_matches_query(clause_count)

    with pytest.raises(GFQLValidationError) as raised:
        cypher_to_gfql(query)

    assert raised.value.code == ErrorCode.E108
    assert raised.value.context["field"] == "matches"
    assert raised.value.context["value"] == clause_count
