"""Validate-only GFQL/Cypher preflight helpers (no query execution)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union, cast

from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTLet, ASTObject, ASTNode, ASTEdge, from_json
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.lowering import (
    CompiledCypherGraphQuery,
    CompiledCypherQuery,
    CompiledCypherUnionQuery,
    compile_cypher_query,
)
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.compilation import GraphSchemaCatalog, PlanContext
from graphistry.compute.gfql.same_path_types import (
    WhereComparison,
    normalize_where_entries,
    parse_where_json,
)
from graphistry.compute.validate.validate_schema import validate_chain_schema


GFQLValidationQuery = Union[ASTObject, List[ASTObject], ASTLet, Chain, dict, str]

_CYPHER_LEAD_RE = re.compile(
    r"^\s*(?:MATCH|OPTIONAL\s+MATCH|WITH|RETURN|UNWIND|CALL|CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|FOREACH|GRAPH|USE)\b",
    re.IGNORECASE,
)


def _looks_like_cypher_query(query: str) -> bool:
    return _CYPHER_LEAD_RE.match(query) is not None


def _serialize_error(exc: Exception, *, stage: str) -> Dict[str, Any]:
    if hasattr(exc, "to_dict") and callable(getattr(exc, "to_dict")):
        out = cast(Dict[str, Any], exc.to_dict())  # GFQLValidationError surface
    elif hasattr(exc, "code") and hasattr(exc, "message"):
        out = {
            "code": cast(Any, getattr(exc, "code")),
            "message": cast(Any, getattr(exc, "message")),
        }
        context = cast(Any, getattr(exc, "context", None))
        if isinstance(context, dict):
            out.update(context)
    else:
        out = {
            "code": ErrorCode.E108,
            "message": str(exc),
        }
    out["stage"] = stage
    return out


def _build_schema_catalog(g: Plottable, *, strict: bool) -> GraphSchemaCatalog:
    node_columns = ()
    edge_columns = ()
    if getattr(g, "_nodes", None) is not None:
        node_columns = tuple(str(c) for c in cast(Any, g)._nodes.columns)
    if getattr(g, "_edges", None) is not None:
        edge_columns = tuple(str(c) for c in cast(Any, g)._edges.columns)
    return GraphSchemaCatalog.from_schema_parts(
        node_columns=node_columns,
        edge_columns=edge_columns,
        node_id_column=getattr(g, "_node", None),
        edge_source_column=getattr(g, "_source", None),
        edge_destination_column=getattr(g, "_destination", None),
        metadata={"strict": strict},
    )


def _validate_cypher(
    g: Plottable,
    query: str,
    *,
    params: Optional[Mapping[str, Any]],
    strict: bool,
) -> Dict[str, Any]:
    parsed = parse_cypher(query)
    if strict:
        strict_ctx = PlanContext(catalog=_build_schema_catalog(g, strict=True))
        FrontendBinder().bind(parsed, strict_ctx, strict_name_resolution=True)
    compiled = compile_cypher_query(parsed, params=params)
    compiled_kind: Literal["query", "union", "graph"] = "query"
    if isinstance(compiled, CompiledCypherUnionQuery):
        compiled_kind = "union"
    elif isinstance(compiled, CompiledCypherGraphQuery):
        compiled_kind = "graph"
    else:
        compiled = cast(CompiledCypherQuery, compiled)
    return {
        "ok": True,
        "query_type": "chain",
        "language": "cypher",
        "diagnostics": [],
        "compiled_kind": compiled_kind,
    }


def _coerce_non_string_query(
    query: GFQLValidationQuery,
    *,
    where: Optional[Sequence[WhereComparison]],
) -> Union[ASTObject, ASTLet, Chain]:
    where_param: Optional[List[WhereComparison]] = None
    if where is not None:
        if isinstance(where, (list, tuple)):
            where_param = normalize_where_entries(where)
        else:
            raise ValueError(f"where must be a list of comparisons, got {type(where).__name__}")

    out: Union[ASTObject, ASTLet, Chain, dict, List[ASTObject], str] = query
    if isinstance(out, dict) and out.get("type") == "Let":
        out = ASTLet.from_json(out)
    elif isinstance(out, dict) and "chain" in out:
        chain_items: List[ASTObject] = []
        for item in cast(List[Any], out["chain"]):
            if isinstance(item, dict):
                chain_items.append(from_json(item))
            elif isinstance(item, ASTObject):
                chain_items.append(item)
            else:
                raise TypeError(f"Unsupported chain entry type: {type(item)}")
        dict_where = parse_where_json(cast(Any, out).get("where"))
        if where_param is not None and dict_where:
            raise ValueError("where cannot be combined with dict chain that already includes where")
        effective_where = where_param if where_param is not None else dict_where
        if not chain_items and effective_where:
            raise ValueError("where requires at least one named node/edge step; empty chains have no aliases")
        out = Chain(chain_items, where=effective_where)
    elif isinstance(out, dict):
        wrapped_dict: Dict[str, Any] = {}
        for key, value in out.items():
            if isinstance(value, (ASTNode, ASTEdge)):
                wrapped_dict[key] = Chain([value])
            else:
                wrapped_dict[key] = value
        out = ASTLet(wrapped_dict)  # type: ignore[arg-type]
    elif isinstance(out, Chain):
        if where_param:
            if out.where:
                raise ValueError("where provided for Chain that already includes where")
            out = Chain(out.chain, where=where_param)
    elif isinstance(out, ASTObject):
        out = Chain([out], where=where_param)
    elif isinstance(out, list):
        converted_query: List[ASTObject] = []
        for item in out:
            if isinstance(item, dict):
                converted_query.append(from_json(item))
            else:
                converted_query.append(item)
        if not converted_query and where_param:
            raise ValueError("where requires at least one named node/edge step; empty chains have no aliases")
        out = Chain(converted_query, where=where_param)
    else:
        raise TypeError(
            f"Query must be ASTObject, List[ASTObject], Chain, ASTLet, dict, or string. "
            f"Got {type(out).__name__}"
        )

    if isinstance(out, (Chain, ASTLet, ASTObject)):
        return out
    raise TypeError(
        f"Query must be ASTObject, List[ASTObject], Chain, ASTLet, dict, or string. Got {type(out).__name__}"
    )


def _validate_non_string_query(
    g: Plottable,
    query: GFQLValidationQuery,
    *,
    where: Optional[Sequence[WhereComparison]],
    collect_all: bool,
) -> Dict[str, Any]:
    coerced = _coerce_non_string_query(query, where=where)
    if isinstance(coerced, Chain):
        if collect_all:
            errors = validate_chain_schema(g, coerced.chain, collect_all=True) or []
            return {
                "ok": len(errors) == 0,
                "query_type": "chain",
                "language": "gfql",
                "diagnostics": [cast(Any, e).to_dict() for e in errors],
            }
        validate_chain_schema(g, coerced.chain, collect_all=False)
        return {
            "ok": True,
            "query_type": "chain",
            "language": "gfql",
            "diagnostics": [],
        }

    # For DAG/non-chain AST forms, preserve existing AST structural validation
    # surface without introducing a new schema simulator for chain-let graphs.
    if collect_all:
        errors = cast(Any, coerced).validate(collect_all=True) or []
        return {
            "ok": len(errors) == 0,
            "query_type": "dag" if isinstance(coerced, ASTLet) else "single",
            "language": "gfql",
            "diagnostics": [cast(Any, e).to_dict() for e in errors],
        }
    cast(Any, coerced).validate(collect_all=False)
    return {
        "ok": True,
        "query_type": "dag" if isinstance(coerced, ASTLet) else "single",
        "language": "gfql",
        "diagnostics": [],
    }


def gfql_validate(
    g: Plottable,
    query: GFQLValidationQuery,
    *,
    where: Optional[Sequence[WhereComparison]] = None,
    language: Optional[Literal["cypher", "gremlin"]] = None,
    params: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
    collect_all: bool = False,
) -> Dict[str, Any]:
    """Validate a GFQL/Cypher query without executing it.

    Returns structured diagnostics and never dispatches query execution operators.
    """
    try:
        if isinstance(query, str):
            if where is not None:
                raise ValueError("where cannot be combined with string queries; embed Cypher predicates in the query itself")
            query_language = language or "cypher"
            if query_language != "cypher":
                raise GFQLValidationError(
                    ErrorCode.E108,
                    f"Unsupported GFQL string language '{query_language}'",
                    field="language",
                    value=query_language,
                    suggestion="Use language='cypher' for now; Gremlin string compilation is not implemented yet.",
                    language="gfql",
                )
            if language is None and not _looks_like_cypher_query(query):
                raise TypeError("Query must be ASTObject, List[ASTObject], Chain, ASTLet, or dict. Got str")
            return _validate_cypher(g, query, params=params, strict=strict)

        if language is not None:
            raise ValueError("language is only supported when query is a string")
        if params is not None:
            raise ValueError("params is only supported when query is a string")
        return _validate_non_string_query(g, query, where=where, collect_all=collect_all)
    except Exception as exc:
        return {
            "ok": False,
            "query_type": "chain" if isinstance(query, str) else "single",
            "language": "cypher" if isinstance(query, str) else "gfql",
            "diagnostics": [_serialize_error(exc, stage="validate")],
        }

