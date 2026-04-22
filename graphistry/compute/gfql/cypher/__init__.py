"""Cypher-string parsing and lowering for local GFQL execution."""

from typing import Any
import warnings

from .ast import (
    CallClause,
    CypherGraphQuery,
    CypherQuery,
    CypherUnionQuery,
    CypherYieldItem,
    ExpressionText,
    GraphBinding,
    GraphConstructor,
    LabelRef,
    LimitClause,
    MatchClause,
    NodePattern,
    OrderByClause,
    OrderItem,
    ParameterRef,
    PropertyRef,
    PropertyEntry,
    RelationshipPattern,
    ReturnClause,
    ReturnItem,
    SkipClause,
    SourceSpan,
    UnwindClause,
    UseClause,
    WhereClause,
    WherePatternPredicate,
    WherePredicate,
)
from .ast_normalizer import ASTNormalizer
from .api import compile_cypher, cypher_to_gfql, gfql_from_cypher
from .lowering import (
    LoweredCypherMatch,
    lower_cypher_query,
    lower_match_clause,
    lower_match_query,
)
from .parser import parse_cypher

__all__ = [
    "CypherQuery",
    "CypherUnionQuery",
    "CallClause",
    "compile_cypher",
    "cypher_to_gfql",
    "ASTNormalizer",
    "ExpressionText",
    "gfql_from_cypher",
    "CypherYieldItem",
    "LabelRef",
    "LimitClause",
    "LoweredCypherMatch",
    "lower_cypher_query",
    "lower_match_clause",
    "lower_match_query",
    "MatchClause",
    "NodePattern",
    "OrderByClause",
    "OrderItem",
    "ParameterRef",
    "PropertyRef",
    "PropertyEntry",
    "RelationshipPattern",
    "ReturnClause",
    "ReturnItem",
    "SkipClause",
    "SourceSpan",
    "UnwindClause",
    "WhereClause",
    "WherePatternPredicate",
    "WherePredicate",
    "parse_cypher",
]


def __getattr__(name: str) -> Any:
    if name in {"CompiledCypherProcedureCall", "CompiledCypherQuery", "CompiledCypherUnionQuery", "compile_cypher_query"}:
        warnings.warn(
            f"{name} is deprecated as a public cypher export and will be removed in 0.55.0; "
            "prefer g.gfql(..., language='cypher') or cypher_to_gfql(). "
            "See https://github.com/graphistry/pygraphistry/issues/1169",
            DeprecationWarning,
            stacklevel=2,
        )
        if name == "CompiledCypherProcedureCall":
            from .call_procedures import CompiledCypherProcedureCall
            return CompiledCypherProcedureCall
        from .lowering import CompiledCypherQuery, CompiledCypherUnionQuery, compile_cypher_query
        if name == "CompiledCypherQuery":
            return CompiledCypherQuery
        if name == "CompiledCypherUnionQuery":
            return CompiledCypherUnionQuery
        return compile_cypher_query
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
