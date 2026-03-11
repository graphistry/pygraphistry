from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError

from .lowering import CompiledCypherQuery, CompiledCypherUnionQuery, compile_cypher_query
from .parser import parse_cypher


def cypher_to_gfql(
    query: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Chain:
    parsed = parse_cypher(query)
    compiled = compile_cypher_query(parsed, params=params)
    if isinstance(compiled, CompiledCypherUnionQuery):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher UNION cannot be represented as a single GFQL Chain",
            field="union",
            value=compiled.union_kind,
            suggestion="Execute the query through g.gfql(\"...\", language=\"cypher\") instead of cypher_to_gfql().",
            language="cypher",
        )
    if compiled.procedure_call is not None:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher CALL cannot be represented as a single GFQL Chain",
            field="call",
            value=compiled.procedure_call.procedure,
            suggestion="Execute the query through g.gfql(\"...\", language=\"cypher\") instead of cypher_to_gfql().",
            language="cypher",
        )
    return compiled.chain


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
) -> Union[CompiledCypherQuery, CompiledCypherUnionQuery]:
    parsed = parse_cypher(query)
    return compile_cypher_query(parsed, params=params)
