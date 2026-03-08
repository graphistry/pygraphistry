from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, cast

from graphistry.compute.ast import ASTObject, ASTNode, e_forward, e_reverse, e_undirected
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.ast import (
    CypherLiteral,
    MatchClause,
    NodePattern,
    ParameterRef,
    PropertyEntry,
    RelationshipPattern,
)


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
