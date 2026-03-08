from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, cast

from graphistry.compute.ast import (
    ASTEdge,
    ASTObject,
    ASTNode,
    distinct,
    e_forward,
    e_reverse,
    e_undirected,
    limit,
    order_by,
    return_,
    rows,
    skip,
    with_,
)
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.predicates.comparison import ge, gt, isna, le, lt, ne, notna
from graphistry.compute.gfql.cypher.ast import (
    CypherLiteral,
    CypherQuery,
    MatchClause,
    NodePattern,
    ParameterRef,
    OrderByClause,
    PropertyRef,
    PropertyEntry,
    RelationshipPattern,
    ReturnClause,
)
from graphistry.compute.gfql.same_path_types import WhereComparison, col, compare


@dataclass(frozen=True)
class LoweredCypherMatch:
    query: List[ASTObject]
    where: List[WhereComparison]


@dataclass(frozen=True)
class _ProjectionPlan:
    source_alias: str
    table: str
    whole_row: bool
    clause_kind: str
    projection_items: List[Tuple[str, str]]
    available_columns: Set[str]


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


def _resolve_page_value(
    value: Any,
    *,
    params: Optional[Mapping[str, Any]],
    field: str,
    line: int,
    column: int,
) -> int:
    if isinstance(value, ParameterRef):
        resolved = _resolve_literal(value, params=params, field=field)
    else:
        resolved = value
    if isinstance(resolved, bool) or not isinstance(resolved, int):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher SKIP/LIMIT values must be integers",
            field=field,
            value=resolved,
            suggestion="Use a non-negative integer literal or parameter.",
            line=line,
            column=column,
            language="cypher",
        )
    if resolved < 0:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher SKIP/LIMIT values must be non-negative",
            field=field,
            value=resolved,
            suggestion="Use 0 or a positive integer.",
            line=line,
            column=column,
            language="cypher",
        )
    return resolved


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


def _alias_table(target: ASTObject, *, alias: str, line: int, column: int) -> str:
    if isinstance(target, ASTNode):
        return "nodes"
    if isinstance(target, ASTEdge):
        return "edges"
    raise _unsupported(
        "Only node and edge aliases are supported in Cypher row projections",
        field="return.alias",
        value=alias,
        line=line,
        column=column,
    )


def _split_qualified_name(expr: str, *, line: int, column: int) -> Tuple[str, Optional[str]]:
    parts = expr.split(".")
    if len(parts) == 1:
        return parts[0], None
    if len(parts) == 2:
        return parts[0], parts[1]
    raise _unsupported(
        "Only simple aliases and alias.property expressions are supported in Cypher RETURN/ORDER BY",
        field="expression",
        value=expr,
        line=line,
        column=column,
    )


def _build_projection_plan(
    clause: ReturnClause,
    *,
    alias_targets: Dict[str, ASTObject],
) -> _ProjectionPlan:
    source_alias: Optional[str] = None
    whole_row = False
    projection_items: List[Tuple[str, str]] = []
    available_columns: Set[str] = set()

    for item in clause.items:
        alias_name, prop = _split_qualified_name(
            item.expression.text,
            line=item.span.line,
            column=item.span.column,
        )
        if alias_name not in alias_targets:
            raise GFQLValidationError(
                ErrorCode.E108,
                f"Unknown Cypher alias '{alias_name}' in {clause.kind.upper()} clause",
                field=f"{clause.kind}.alias",
                value=alias_name,
                suggestion="Reference an alias declared in the MATCH pattern.",
                line=item.span.line,
                column=item.span.column,
                language="cypher",
            )
        if source_alias is None:
            source_alias = alias_name
        elif source_alias != alias_name:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher row projection currently supports one source alias at a time",
                field=f"{clause.kind}.items",
                value=[entry.expression.text for entry in clause.items],
                suggestion="Project from one alias only, or wait for multi-alias row projection support.",
                line=item.span.line,
                column=item.span.column,
                language="cypher",
            )
        if prop is None:
            if len(clause.items) != 1 or item.alias is not None:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Whole-row alias projection is only supported as a single bare alias",
                    field=f"{clause.kind}.items",
                    value=item.expression.text,
                    suggestion="Use a single alias like RETURN p, or project individual properties.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
            whole_row = True
            continue
        output_name = item.alias or prop
        projection_items.append((output_name, prop))
        available_columns.add(output_name)
        available_columns.add(prop)

    if source_alias is None:
        raise GFQLValidationError(
            ErrorCode.E108,
            f"Cypher {clause.kind.upper()} clause must project at least one supported expression",
            field=clause.kind,
            value=None,
            suggestion="Project a match alias or alias.property expression.",
            line=clause.span.line,
            column=clause.span.column,
            language="cypher",
        )
    if whole_row and len(projection_items) > 0:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cannot mix whole-row alias projection with property projections",
            field=f"{clause.kind}.items",
            value=[entry.expression.text for entry in clause.items],
            suggestion="Use either RETURN p or RETURN p.id, p.name, ...",
            line=clause.span.line,
            column=clause.span.column,
            language="cypher",
        )
    if whole_row:
        available_columns = set()
    table = _alias_table(
        alias_targets[source_alias],
        alias=source_alias,
        line=clause.span.line,
        column=clause.span.column,
    )
    return _ProjectionPlan(
        source_alias=source_alias,
        table=table,
        whole_row=whole_row,
        clause_kind=clause.kind,
        projection_items=projection_items,
        available_columns=available_columns,
    )


def _lower_order_by_clause(
    clause: OrderByClause,
    *,
    plan: _ProjectionPlan,
) -> ASTObject:
    keys: List[Tuple[str, str]] = []
    for item in clause.items:
        alias_name, prop = _split_qualified_name(
            item.expression.text,
            line=item.span.line,
            column=item.span.column,
        )
        if prop is None:
            order_key = alias_name
        else:
            if alias_name != plan.source_alias:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "ORDER BY expressions must reference the active RETURN/WITH source alias",
                    field="order_by",
                    value=item.expression.text,
                    suggestion=f"Use columns from alias '{plan.source_alias}' only in this phase.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
            order_key = prop
        if not plan.whole_row and order_key not in plan.available_columns:
            raise GFQLValidationError(
                ErrorCode.E108,
                "ORDER BY column must exist after RETURN/WITH projection in this phase",
                field="order_by",
                value=item.expression.text,
                suggestion="Order by a projected output column or alias.",
                line=item.span.line,
                column=item.span.column,
                language="cypher",
            )
        keys.append((order_key, item.direction))
    return order_by(keys)


def _lower_projection_chain(
    query: CypherQuery,
    lowered: LoweredCypherMatch,
    *,
    params: Optional[Mapping[str, Any]],
) -> List[ASTObject]:
    alias_targets = _alias_target(lowered.query)
    plan = _build_projection_plan(query.return_, alias_targets=alias_targets)

    row_steps: List[ASTObject] = [rows(table=plan.table, source=plan.source_alias)]

    if not plan.whole_row:
        projection_fn = with_ if plan.clause_kind == "with" else return_
        row_steps.append(projection_fn(plan.projection_items))

    if query.return_.distinct:
        row_steps.append(distinct())
    if query.order_by is not None:
        row_steps.append(_lower_order_by_clause(query.order_by, plan=plan))
    if query.skip is not None:
        row_steps.append(
            skip(
                _resolve_page_value(
                    query.skip.value,
                    params=params,
                    field="skip",
                    line=query.skip.span.line,
                    column=query.skip.span.column,
                )
            )
        )
    if query.limit is not None:
        row_steps.append(
            limit(
                _resolve_page_value(
                    query.limit.value,
                    params=params,
                    field="limit",
                    line=query.limit.span.line,
                    column=query.limit.span.column,
                )
            )
        )
    return lowered.query + row_steps


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


def lower_cypher_query(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Chain:
    lowered = lower_match_query(query, params=params)
    row_query = _lower_projection_chain(query, lowered, params=params)
    return Chain(row_query, where=lowered.where)
