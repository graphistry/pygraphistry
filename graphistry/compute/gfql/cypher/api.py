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
    """Compile a supported Cypher query on GFQL's current Cypher surface into a single GFQL Chain.

    Use this helper when you want the translated GFQL chain object rather than
    running the query immediately. Queries that require a union program or a
    row-returning procedure flow cannot be represented as a single ``Chain``;
    use :func:`compile_cypher` when you want to inspect those compiled program
    shapes, or execute them directly through ``g.gfql("...", language="cypher")``.

    :param query: Cypher text to parse and lower.
    :param params: Optional parameter dictionary used during lowering.
    :returns: A GFQL ``Chain`` equivalent to the supported query.
    :raises GFQLValidationError: If the query cannot be represented as a single
        ``Chain``.
    """
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
    """Alias for :func:`cypher_to_gfql` for callers that prefer GFQL-first naming."""
    return cypher_to_gfql(query, params=params)


def compile_cypher(
    query: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Union[CompiledCypherQuery, CompiledCypherUnionQuery]:
    """Parse and lower a supported Cypher query into a compiled program.

    This is the lowest-level public helper for inspecting GFQL's Cypher
    compiler output before execution.

    :param query: Cypher text to parse and lower.
    :param params: Optional parameter dictionary used during lowering.
    :returns: A compiled single-query or union-query program.
    """
    parsed = parse_cypher(query)
    return compile_cypher_query(parsed, params=params)
