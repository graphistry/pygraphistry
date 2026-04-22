from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional, Union

from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError

from .lowering import CompiledCypherGraphQuery, CompiledCypherQuery, CompiledCypherUnionQuery, compile_cypher_query
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
    if isinstance(compiled, CompiledCypherGraphQuery):
        if compiled.graph_bindings:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Multi-graph GRAPH binding pipelines cannot be represented as a single GFQL Chain",
                suggestion="Execute the query through g.gfql(\"...\", language=\"cypher\") instead of cypher_to_gfql().",
                language="cypher",
            )
        return compiled.chain
    if compiled.procedure_call is not None:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher CALL cannot be represented as a single GFQL Chain",
            field="call",
            value=compiled.procedure_call.procedure,
            suggestion="Execute the query through g.gfql(\"...\", language=\"cypher\") instead of cypher_to_gfql().",
            language="cypher",
        )
    if compiled.graph_bindings:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Multi-graph GRAPH binding pipelines cannot be represented as a single GFQL Chain",
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
    _warn_deprecated: bool = True,
) -> Union[CompiledCypherQuery, CompiledCypherUnionQuery, CompiledCypherGraphQuery]:
    """Deprecated compatibility helper for inspecting compiled Cypher internals.

    .. deprecated:: 2.8.0
       ``compile_cypher()`` and ``CompiledCypher*`` return-shape internals are
       deprecated compatibility surfaces scheduled for removal in v2.8.0. Prefer
       ``g.gfql(..., language="cypher")`` for execution and
       :func:`cypher_to_gfql` / :func:`gfql_from_cypher` for single-chain
       translation. Tracked: https://github.com/graphistry/pygraphistry/issues/1169

    Parse and lower a supported Cypher query into a compiled program.

    This is the lowest-level public helper for inspecting GFQL's Cypher
    compiler output before execution.

    :param query: Cypher text to parse and lower.
    :param params: Optional parameter dictionary used during lowering.
    :returns: A compiled single-query or union-query program.
    """
    if _warn_deprecated:
        warnings.warn(
            "compile_cypher() is deprecated and will be removed in v2.8.0; "
            "prefer g.gfql(..., language='cypher') for execution or cypher_to_gfql() for translation. "
            "See https://github.com/graphistry/pygraphistry/issues/1169",
            DeprecationWarning,
            stacklevel=2,
        )
    parsed = parse_cypher(query)
    return compile_cypher_query(parsed, params=params)
