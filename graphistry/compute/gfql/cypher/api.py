from __future__ import annotations

from typing import Any, Mapping, Optional

from graphistry.compute.chain import Chain

from .lowering import CompiledCypherQuery, compile_cypher_query
from .parser import parse_cypher


def cypher_to_gfql(
    query: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Chain:
    parsed = parse_cypher(query)
    return compile_cypher_query(parsed, params=params).chain


def gfql_from_cypher(
    query: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Chain:
    return cypher_to_gfql(query, params=params)


def compile_cypher(
    query: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> CompiledCypherQuery:
    parsed = parse_cypher(query)
    return compile_cypher_query(parsed, params=params)
