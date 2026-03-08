from __future__ import annotations

from typing import Any, Mapping, Optional

from graphistry.compute.chain import Chain

from .lowering import lower_cypher_query
from .parser import parse_cypher


def cypher_to_gfql(
    query: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Chain:
    parsed = parse_cypher(query)
    return lower_cypher_query(parsed, params=params)


def gfql_from_cypher(
    query: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Chain:
    return cypher_to_gfql(query, params=params)
