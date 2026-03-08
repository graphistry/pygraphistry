from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, cast

from graphistry.compute import ge, gt, isna, le, lt, ne, notna
from graphistry.compute.ast import ASTEdge, ASTObject, ASTNode, e_forward, e_reverse, e_undirected
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.ast import (
    CypherLiteral,
    CypherQuery,
    MatchClause,
    NodePattern,
    ParameterRef,
    PropertyRef,
    PropertyEntry,
    RelationshipPattern,
)
from graphistry.compute.gfql.same_path_types import WhereComparison, col, compare


@dataclass(frozen=True)
class LoweredCypherMatch:
    query: List[ASTObject]
    where: List[WhereComparison]


def _unsupported(message: str, *, field: str, value: Any, line: int, column: int) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        message,
        field=field,
        value=value,
        suggestion="Use a subset currently supported by the local Cypher compiler.",
        line=line,
        column=column,
        language="cypher",
    )


def _resolve_literal(
    value: CypherLiteral,
    *,
    params: Optional[Mapping[str, Any]],
    field: str,
) -> Any:
    if isinstance(value, ParameterRef):
        if params is None or value.name not in params:
            raise GFQLValidationError(
                ErrorCode.E105,
                f"Missing Cypher parameter '${value.name}'",
                field=field,
                value=value.name,
                suggestion=f"Pass params={{'{value.name}': ...}} when compiling or executing the query.",
                line=value.span.line,
                column=value.span.column,
                language="cypher",
            )
        return params[value.name]
    return value


def _filter_dict_from_entries(
    *,
    discriminator_key: Optional[str],
    discriminator_values: Sequence[str],
    properties: Sequence[PropertyEntry],
    params: Optional[Mapping[str, Any]],
    field_prefix: str,
    line: int,
    column: int,
) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    if len(discriminator_values) > 1:
        raise _unsupported(
            "Multiple labels/types are not yet supported by local Cypher lowering",
            field=field_prefix,
            value=list(discriminator_values),
            line=line,
            column=column,
        )
    if discriminator_key is not None and len(discriminator_values) == 1:
        out[discriminator_key] = discriminator_values[0]
    for entry in properties:
        out[entry.key] = _resolve_literal(
            entry.value,
            params=params,
            field=f"{field_prefix}.{entry.key}",
        )
    return out or None


def _lower_node(node: NodePattern, *, params: Optional[Mapping[str, Any]]) -> ASTNode:
    filter_dict = _filter_dict_from_entries(
        discriminator_key="type",
        discriminator_values=node.labels,
        properties=node.properties,
        params=params,
        field_prefix=f"node.{node.variable or '_'}",
        line=node.span.line,
        column=node.span.column,
    )
    return ASTNode(filter_dict=filter_dict, name=node.variable)


def _lower_relationship(
    relationship: RelationshipPattern,
    *,
    params: Optional[Mapping[str, Any]],
) -> ASTObject:
    edge_match = _filter_dict_from_entries(
        discriminator_key="type",
        discriminator_values=relationship.types,
        properties=relationship.properties,
        params=params,
        field_prefix=f"edge.{relationship.variable or '_'}",
        line=relationship.span.line,
        column=relationship.span.column,
    )
    if relationship.direction == "forward":
        return cast(ASTObject, e_forward(edge_match=edge_match, name=relationship.variable))
    if relationship.direction == "reverse":
        return cast(ASTObject, e_reverse(edge_match=edge_match, name=relationship.variable))
    return cast(ASTObject, e_undirected(edge_match=edge_match, name=relationship.variable))


def _alias_target(ops: Sequence[ASTObject]) -> Dict[str, ASTObject]:
    targets: Dict[str, ASTObject] = {}
    for op in ops:
        alias = getattr(op, "_name", None)
        if alias is None:
            continue
        if alias in targets:
            raise GFQLValidationError(
                ErrorCode.E108,
                f"Duplicate Cypher alias '{alias}' is not yet supported",
                field="alias",
                value=alias,
                suggestion="Use unique aliases per node/relationship pattern.",
                language="cypher",
            )
        targets[alias] = op
    return targets


def _target_filter_dict(target: ASTObject) -> Optional[Dict[str, Any]]:
    if isinstance(target, ASTNode):
        return cast(Optional[Dict[str, Any]], target.filter_dict)
    if isinstance(target, ASTEdge):
        return cast(Optional[Dict[str, Any]], target.edge_match)
    raise _unsupported(
        "Only node and edge aliases are supported in Cypher MATCH lowering",
        field="alias",
        value=type(target).__name__,
        line=1,
        column=1,
    )


def _set_target_filter_dict(target: ASTObject, filter_dict: Dict[str, Any]) -> None:
    if isinstance(target, ASTNode):
        target.filter_dict = filter_dict
        return
    if isinstance(target, ASTEdge):
        target.edge_match = filter_dict
        return
    raise _unsupported(
        "Only node and edge aliases are supported in Cypher MATCH lowering",
        field="alias",
        value=type(target).__name__,
        line=1,
        column=1,
    )


def _predicate_value(op: str, value: Any) -> Any:
    if op == "==":
        return value
    if op == "!=":
        return ne(value)
    if op == "<":
        return lt(value)
    if op == "<=":
        return le(value)
    if op == ">":
        return gt(value)
    if op == ">=":
        return ge(value)
    if op == "is_null":
        return isna()
    if op == "is_not_null":
        return notna()
    raise ValueError(f"Unsupported predicate op: {op}")


def _apply_literal_where(
    targets: Dict[str, ASTObject],
    *,
    left: PropertyRef,
    op: str,
    right: Optional[CypherLiteral],
    params: Optional[Mapping[str, Any]],
) -> None:
    if left.alias not in targets:
        raise GFQLValidationError(
            ErrorCode.E108,
            f"Unknown Cypher alias '{left.alias}' in WHERE clause",
            field="where.left.alias",
            value=left.alias,
            suggestion="Reference an alias declared in the MATCH pattern.",
            line=left.span.line,
            column=left.span.column,
            language="cypher",
        )
    target = targets[left.alias]
    filter_dict = dict(_target_filter_dict(target) or {})
    if left.property in filter_dict:
        raise GFQLValidationError(
            ErrorCode.E108,
            f"Duplicate filtering on '{left.alias}.{left.property}' is not yet supported",
            field=f"where.{left.alias}.{left.property}",
            value=left.property,
            suggestion="Move the filter to one location or wait for predicate merging support.",
            line=left.span.line,
            column=left.span.column,
            language="cypher",
        )
    resolved = None if right is None else _resolve_literal(
        right,
        params=params,
        field=f"where.{left.alias}.{left.property}",
    )
    filter_dict[left.property] = _predicate_value(op, resolved)
    _set_target_filter_dict(target, filter_dict)


def lower_match_clause(
    clause: MatchClause,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> List[ASTObject]:
    out: List[ASTObject] = []
    for element in clause.pattern:
        if isinstance(element, NodePattern):
            out.append(_lower_node(element, params=params))
        else:
            out.append(_lower_relationship(element, params=params))
    return out


def lower_match_query(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> LoweredCypherMatch:
    ops = lower_match_clause(query.match, params=params)
    alias_targets = _alias_target(ops)
    where_out: List[WhereComparison] = []

    if query.where is not None:
        for predicate in query.where.predicates:
            if isinstance(predicate.right, PropertyRef):
                where_out.append(
                    compare(
                        col(predicate.left.alias, predicate.left.property),
                        cast(Any, predicate.op),
                        col(predicate.right.alias, predicate.right.property),
                    )
                )
                continue
            _apply_literal_where(
                alias_targets,
                left=predicate.left,
                op=predicate.op,
                right=cast(Optional[CypherLiteral], predicate.right),
                params=params,
            )

    return LoweredCypherMatch(query=ops, where=where_out)
