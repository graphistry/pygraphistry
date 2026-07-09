"""Validate-only GFQL/Cypher preflight helpers (no query execution)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, NoReturn, Optional, Sequence, Tuple, Union, cast
from dataclasses import replace

from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTLet, ASTObject, ASTNode, ASTEdge, ASTCall, ASTRef, from_json
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError, GFQLValidationError
from graphistry.compute.gfql.cypher.lowering import (
    CompiledCypherGraphQuery,
    CompiledCypherQuery,
    CompiledCypherUnionQuery,
    compile_cypher_query,
)
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.compilation import GraphSchemaCatalog, PlanContext
from graphistry.compute.gfql.query_types import GFQLQuery
from graphistry.compute.gfql.same_path_types import (
    WhereComparison,
    normalize_where_entries,
    parse_where_json,
)
from graphistry.compute.validate.validate_schema import validate_chain_schema


GFQLValidationQuery = GFQLQuery

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


def _raise_diagnostics(
    diagnostics: List[Dict[str, Any]],
    *,
    query_type: str,
    language: str,
) -> NoReturn:
    first = diagnostics[0]
    code = cast(Any, first.get("code")) or ErrorCode.E108
    message = cast(Any, first.get("message")) or "GFQL validation failed"
    if len(diagnostics) > 1:
        message = f"GFQL validation failed with {len(diagnostics)} errors; first: {message}"
    extra = {
        key: value
        for key, value in first.items()
        if key not in {"code", "message", "field", "value", "suggestion", "operation_index"}
    }
    exc_cls = GFQLSyntaxError if code == ErrorCode.E107 else GFQLValidationError
    raise exc_cls(
        code,
        message,
        field=cast(Optional[str], first.get("field")),
        value=first.get("value"),
        suggestion=cast(Optional[str], first.get("suggestion")),
        operation_index=cast(Optional[int], first.get("operation_index")),
        diagnostics=diagnostics,
        query_type=query_type,
        language=language,
        **extra,
    )


def _build_schema_catalog(g: Plottable, *, strict: Optional[bool]) -> GraphSchemaCatalog:
    bound_schema = getattr(g, "_gfql_schema", None)
    if bound_schema is not None:
        node_id_column = getattr(g, "_node", None)
        edge_source_column = getattr(g, "_source", None)
        edge_destination_column = getattr(g, "_destination", None)
        if hasattr(bound_schema, "to_catalog") and callable(getattr(bound_schema, "to_catalog")):
            return cast(Any, bound_schema).to_catalog(
                node_id_column=node_id_column,
                edge_source_column=edge_source_column,
                edge_destination_column=edge_destination_column,
                strict=strict,
            )
        if isinstance(bound_schema, GraphSchemaCatalog):
            metadata = dict(bound_schema.metadata)
            if strict is not None:
                metadata["strict"] = bool(strict)
            return replace(
                bound_schema,
                node_id_column=node_id_column or bound_schema.node_id_column,
                edge_source_column=edge_source_column or bound_schema.edge_source_column,
                edge_destination_column=edge_destination_column or bound_schema.edge_destination_column,
                metadata=metadata,
            )

    strict_value = True if strict is None else bool(strict)
    node_columns: Tuple[str, ...] = tuple()
    edge_columns: Tuple[str, ...] = tuple()
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
        metadata={"strict": strict_value},
    )


def _resolve_strict_mode(g: Plottable, *, strict: Optional[bool]) -> bool:
    if strict is not None:
        return bool(strict)
    bound_schema = getattr(g, "_gfql_schema", None)
    if bound_schema is not None:
        schema_strict = getattr(bound_schema, "strict", None)
        if schema_strict is not None:
            return bool(schema_strict)
        metadata = getattr(bound_schema, "metadata", None)
        if isinstance(metadata, Mapping) and "strict" in metadata:
            return bool(metadata["strict"])
    return True


def _validate_cypher(
    g: Plottable,
    query: str,
    *,
    params: Optional[Mapping[str, Any]],
    strict: Optional[bool],
) -> Dict[str, Any]:
    parsed = parse_cypher(query)
    strict_mode = _resolve_strict_mode(g, strict=strict)
    if strict_mode:
        strict_ctx = PlanContext(catalog=_build_schema_catalog(g, strict=strict))
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
    elif isinstance(out, ASTLet):
        pass
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
    schema: bool,
) -> Dict[str, Any]:
    coerced = _coerce_non_string_query(query, where=where)
    if isinstance(coerced, Chain):
        if not schema:
            if collect_all:
                errors = cast(Any, coerced).validate(collect_all=True) or []
                diagnostics = [cast(Any, e).to_dict() for e in errors]
                if diagnostics:
                    _raise_diagnostics(diagnostics, query_type="chain", language="gfql")
                return {
                    "ok": True,
                    "query_type": "chain",
                    "language": "gfql",
                    "diagnostics": [],
                }
            cast(Any, coerced).validate(collect_all=False)
            return {
                "ok": True,
                "query_type": "chain",
                "language": "gfql",
                "diagnostics": [],
            }
        if collect_all:
            errors = validate_chain_schema(g, coerced.chain, collect_all=True) or []
            diagnostics = [cast(Any, e).to_dict() for e in errors]
            if diagnostics:
                _raise_diagnostics(diagnostics, query_type="chain", language="gfql")
            return {
                "ok": True,
                "query_type": "chain",
                "language": "gfql",
                "diagnostics": [],
            }
        validate_chain_schema(g, coerced.chain, collect_all=False)
        return {
            "ok": True,
            "query_type": "chain",
            "language": "gfql",
            "diagnostics": [],
        }

    if isinstance(coerced, ASTLet):
        return _validate_let_query(g, coerced, collect_all=collect_all, schema=schema)

    # For non-chain/non-let AST forms, preserve existing AST structural validation
    # surface without introducing a new schema simulator.
    if collect_all:
        errors = cast(Any, coerced).validate(collect_all=True) or []
        diagnostics = [cast(Any, e).to_dict() for e in errors]
        if diagnostics:
            _raise_diagnostics(diagnostics, query_type="single", language="gfql")
        return {
            "ok": True,
            "query_type": "single",
            "language": "gfql",
            "diagnostics": [],
        }
    cast(Any, coerced).validate(collect_all=False)
    return {
        "ok": True,
        "query_type": "single",
        "language": "gfql",
        "diagnostics": [],
    }


def _validate_let_binding_schema_errors(g: Plottable, value: Any) -> List[Any]:
    # Structural validation for AST forms is handled by ASTSerializable.validate();
    # this helper adds best-effort schema validation for bindings that execute
    # directly against dataframe-like tables.
    errors: List[Any] = []

    if isinstance(value, ASTLet):
        for nested in value.bindings.values():
            errors.extend(_validate_let_binding_schema_errors(g, nested))
        return errors

    if isinstance(value, Chain):
        return validate_chain_schema(g, value.chain, collect_all=True) or []

    if isinstance(value, (ASTNode, ASTEdge, ASTCall)):
        return validate_chain_schema(g, [value], collect_all=True) or []

    # ASTRef bindings execute against prior DAG bindings and may have schema
    # transformations not visible from root graph statically; keep structural
    # checks only to avoid false positives.
    if isinstance(value, ASTRef):
        return []

    return []


def _validate_let_query(
    g: Plottable,
    let_query: ASTLet,
    *,
    collect_all: bool,
    schema: bool,
) -> Dict[str, Any]:
    if collect_all:
        errors = cast(Any, let_query).validate(collect_all=True) or []
        if schema:
            for value in let_query.bindings.values():
                errors.extend(_validate_let_binding_schema_errors(g, value))
        diagnostics = [cast(Any, e).to_dict() for e in errors]
        if diagnostics:
            _raise_diagnostics(diagnostics, query_type="dag", language="gfql")
        return {
            "ok": True,
            "query_type": "dag",
            "language": "gfql",
            "diagnostics": [],
        }

    cast(Any, let_query).validate(collect_all=False)
    if schema:
        for value in let_query.bindings.values():
            binding_errors = _validate_let_binding_schema_errors(g, value)
            if binding_errors:
                raise cast(Any, binding_errors[0])
    return {
        "ok": True,
        "query_type": "dag",
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
    strict: Optional[bool] = None,
    collect_all: bool = False,
    schema: bool = True,
) -> Dict[str, Any]:
    """Validate a GFQL/Cypher query without executing it.

    Raises structured GFQL exceptions on validation failures and never dispatches
    query execution operators.
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
            return _validate_cypher(g, query, params=params, strict=strict)

        if language is not None:
            raise ValueError("language is only supported when query is a string")
        if params is not None:
            raise ValueError("params is only supported when query is a string")
        return _validate_non_string_query(g, query, where=where, collect_all=collect_all, schema=schema)
    except GFQLValidationError:
        raise
    except Exception as exc:
        diagnostic = _serialize_error(exc, stage="validate")
        _raise_diagnostics(
            [diagnostic],
            query_type="chain" if isinstance(query, str) else "single",
            language="cypher" if isinstance(query, str) else "gfql",
        )
