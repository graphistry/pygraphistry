from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, cast

from graphistry.compute.ast import (
    ASTEdge,
    ASTObject,
    ASTNode,
    distinct,
    e_forward,
    e_reverse,
    e_undirected,
    group_by,
    limit,
    order_by,
    return_,
    rows,
    skip,
    unwind,
    with_,
)
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.predicates.comparison import ge, gt, isna, le, lt, ne, notna
from graphistry.compute.predicates.is_in import is_in
from graphistry.compute.gfql.expr_parser import (
    FunctionCall,
    GFQLExprParseError,
    Identifier,
    Wildcard,
    collect_identifiers,
    parse_expr,
)
from graphistry.compute.gfql.cypher.ast import (
    CypherLiteral,
    CypherQuery,
    ExpressionText,
    LabelRef,
    MatchClause,
    NodePattern,
    ParameterRef,
    OrderByClause,
    PropertyRef,
    PropertyEntry,
    RelationshipPattern,
    ReturnClause,
    ReturnItem,
    SourceSpan,
    UnwindClause,
)
from graphistry.compute.gfql.same_path_types import WhereComparison, col, compare


@dataclass(frozen=True)
class LoweredCypherMatch:
    query: List[ASTObject]
    where: List[WhereComparison]


@dataclass(frozen=True)
class CompiledCypherQuery:
    chain: Chain
    seed_rows: bool = False


@dataclass(frozen=True)
class _ProjectionPlan:
    source_alias: str
    table: str
    whole_row: bool
    clause_kind: str
    projection_items: List[Tuple[str, str]]
    available_columns: Set[str]
    projected_property_outputs: Dict[str, str]


@dataclass(frozen=True)
class _AggregateSpec:
    source_text: str
    output_name: str
    func: str
    expr_text: Optional[str]
    span_line: int
    span_column: int


_CYPHER_PARAM_RE = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")
_CYPHER_AGGREGATES = frozenset({"count", "sum", "min", "max", "avg", "collect"})


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


def _parse_row_expr(
    expr_text: str,
    *,
    field: str,
    line: int,
    column: int,
) -> Any:
    try:
        return parse_expr(expr_text)
    except (GFQLExprParseError, ImportError) as exc:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher expression is outside the currently supported local GFQL subset",
            field=field,
            value=expr_text,
            suggestion="Use column references, supported GFQL scalar expressions, or supported aggregate functions.",
            line=line,
            column=column,
            language="cypher",
        ) from exc


def _expr_match_aliases(
    expr_text: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    field: str,
    line: int,
    column: int,
) -> Set[str]:
    node = _parse_row_expr(expr_text, field=field, line=line, column=column)
    return {
        ident.split(".", 1)[0]
        for ident in collect_identifiers(node)
        if ident.split(".", 1)[0] in alias_targets
    }


def _validate_row_expr_scope(
    expr_text: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    active_match_alias: Optional[str],
    unwind_aliases: Iterable[str],
    field: str,
    line: int,
    column: int,
) -> None:
    allowed_unwind_roots = set(unwind_aliases)
    for root in _expr_match_aliases(
        expr_text,
        alias_targets=alias_targets,
        field=field,
        line=line,
        column=column,
    ):
        if active_match_alias is None:
            raise _unsupported(
                "Cypher row expressions cannot reference MATCH aliases when no row source is active",
                field=field,
                value=expr_text,
                line=line,
                column=column,
            )
        if root != active_match_alias and root not in allowed_unwind_roots:
            raise _unsupported(
                "Cypher row lowering currently supports one MATCH source alias at a time",
                field=field,
                value=expr_text,
                line=line,
                column=column,
            )


def _whole_param_name(expr_text: str) -> Optional[str]:
    match = _CYPHER_PARAM_RE.match(expr_text.strip())
    if match is None:
        return None
    return match.group(1)


def _row_expr_arg(
    expr: ExpressionText,
    *,
    params: Optional[Mapping[str, Any]],
    field: str,
) -> Any:
    param_name = _whole_param_name(expr.text)
    if param_name is not None:
        return _resolve_literal(
            ParameterRef(name=param_name, span=expr.span),
            params=params,
            field=field,
        )
    if "$" in expr.text:
        raise _unsupported(
            "Cypher parameters are currently only supported as whole row expressions in local lowering",
            field=field,
            value=expr.text,
            line=expr.span.line,
            column=expr.span.column,
        )
    _parse_row_expr(expr.text, field=field, line=expr.span.line, column=expr.span.column)
    return expr.text


def _aggregate_spec(item: ReturnItem) -> Optional[_AggregateSpec]:
    node = _parse_row_expr(
        item.expression.text,
        field="return.item",
        line=item.span.line,
        column=item.span.column,
    )
    if not isinstance(node, FunctionCall) or node.name not in _CYPHER_AGGREGATES:
        return None

    if len(node.args) == 0:
        raise _unsupported(
            "Cypher aggregate functions require an argument or '*'",
            field="return.item",
            value=item.expression.text,
            line=item.span.line,
            column=item.span.column,
        )
    if len(node.args) != 1:
        raise _unsupported(
            "Cypher local aggregate lowering currently supports single-argument aggregate functions only",
            field="return.item",
            value=item.expression.text,
            line=item.span.line,
            column=item.span.column,
        )

    arg = node.args[0]
    if isinstance(arg, Wildcard):
        if node.name != "count":
            raise _unsupported(
                "Only count(*) supports '*' in the local Cypher aggregate subset",
                field="return.item",
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )
        expr_text: Optional[str] = None
    else:
        open_paren = item.expression.text.find("(")
        close_paren = item.expression.text.rfind(")")
        if open_paren < 0 or close_paren <= open_paren:
            raise _unsupported(
                "Invalid Cypher aggregate function syntax",
                field="return.item",
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )
        expr_text = item.expression.text[open_paren + 1:close_paren].strip()
        if expr_text == "":
            raise _unsupported(
                "Cypher aggregate functions require a non-empty argument",
                field="return.item",
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )

    return _AggregateSpec(
        source_text=item.expression.text,
        output_name=item.alias or item.expression.text,
        func=node.name,
        expr_text=expr_text,
        span_line=item.span.line,
        span_column=item.span.column,
    )


def _active_match_alias(
    query: CypherQuery,
    *,
    alias_targets: Mapping[str, ASTObject],
) -> Optional[str]:
    if not alias_targets:
        return None

    expr_texts: List[Tuple[str, int, int, str]] = []
    for unwind_clause in query.unwinds:
        expr_texts.append(
            (
                unwind_clause.expression.text,
                unwind_clause.span.line,
                unwind_clause.span.column,
                "unwind",
            )
        )
    for return_item in query.return_.items:
        expr_texts.append(
            (
                return_item.expression.text,
                return_item.span.line,
                return_item.span.column,
                query.return_.kind,
            )
        )
    if query.order_by is not None:
        for order_item in query.order_by.items:
            expr_texts.append(
                (
                    order_item.expression.text,
                    order_item.span.line,
                    order_item.span.column,
                    "order_by",
                )
            )

    referenced: Set[str] = set()
    for expr_text, line, column, field in expr_texts:
        referenced.update(
            _expr_match_aliases(
                expr_text,
                alias_targets=alias_targets,
                field=field,
                line=line,
                column=column,
            )
        )

    if len(referenced) > 1:
        raise _unsupported(
            "Cypher row lowering currently supports one MATCH source alias at a time",
            field="return",
            value=sorted(referenced),
            line=query.return_.span.line,
            column=query.return_.span.column,
        )
    if len(referenced) == 1:
        return next(iter(referenced))
    return next(iter(alias_targets))

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
    if discriminator_key is not None and len(discriminator_values) == 1:
        out[discriminator_key] = discriminator_values[0]
    elif discriminator_key is not None and len(discriminator_values) > 1:
        out[discriminator_key] = is_in(list(discriminator_values))
    for entry in properties:
        out[entry.key] = _resolve_literal(
            entry.value,
            params=params,
            field=f"{field_prefix}.{entry.key}",
        )
    return out or None


def _lower_node(node: NodePattern, *, params: Optional[Mapping[str, Any]]) -> ASTNode:
    filter_dict = _filter_dict_from_entries(
        discriminator_key=None,
        discriminator_values=(),
        properties=node.properties,
        params=params,
        field_prefix=f"node.{node.variable or '_'}",
        line=node.span.line,
        column=node.span.column,
    )
    if node.labels:
        if len(node.labels) > 1:
            raise _unsupported(
                "Multiple node labels are not yet supported by local Cypher lowering",
                field=f"node.{node.variable or '_'}",
                value=list(node.labels),
                line=node.span.line,
                column=node.span.column,
            )
        filter_dict = dict(filter_dict or {})
        filter_dict[f"label__{node.labels[0]}"] = True
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


def _projection_ref_from_expr(
    expr: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    field: str,
    line: int,
    column: int,
) -> Tuple[str, Optional[str]]:
    node = _parse_row_expr(expr, field=field, line=line, column=column)
    if isinstance(node, Identifier):
        return _split_qualified_name(node.name, line=line, column=column)
    if isinstance(node, FunctionCall) and node.name == "type":
        if len(node.args) != 1 or not isinstance(node.args[0], Identifier):
            raise _unsupported(
                "type(...) is only supported with a single relationship alias argument in this phase",
                field=field,
                value=expr,
                line=line,
                column=column,
            )
        alias_name, prop = _split_qualified_name(node.args[0].name, line=line, column=column)
        if prop is not None:
            raise _unsupported(
                "type(...) only supports bare relationship aliases in this phase",
                field=field,
                value=expr,
                line=line,
                column=column,
            )
        target = alias_targets.get(alias_name)
        if not isinstance(target, ASTEdge):
            raise _unsupported(
                "type(...) is only supported for relationship aliases in this phase",
                field=field,
                value=expr,
                line=line,
                column=column,
            )
        return alias_name, "type"
    raise _unsupported(
        "Only simple aliases, alias.property, and type(rel_alias) expressions are supported in Cypher RETURN/ORDER BY",
        field=field,
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
    projected_property_outputs: Dict[str, str] = {}

    for item in clause.items:
        alias_name, prop = _projection_ref_from_expr(
            item.expression.text,
            alias_targets=alias_targets,
            field=f"{clause.kind}.items",
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
        output_name = item.alias or item.expression.text
        projection_items.append((output_name, prop))
        available_columns.add(output_name)
        projected_property_outputs.setdefault(prop, output_name)

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
        projected_property_outputs=projected_property_outputs,
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
            order_key = prop if plan.whole_row else plan.projected_property_outputs.get(prop, prop)
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


def _lower_order_by_outputs(
    clause: OrderByClause,
    *,
    available_columns: Set[str],
    expr_to_output: Mapping[str, str],
) -> ASTObject:
    keys: List[Tuple[str, str]] = []
    for item in clause.items:
        order_key = expr_to_output.get(item.expression.text, item.expression.text)
        if order_key not in available_columns:
            raise _unsupported(
                "ORDER BY column must exist after RETURN/WITH projection in this phase",
                field="order_by",
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )
        keys.append((order_key, item.direction))
    return order_by(keys)


def _append_page_ops(
    row_steps: List[ASTObject],
    *,
    query: CypherQuery,
    params: Optional[Mapping[str, Any]],
) -> None:
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
    _append_page_ops(row_steps, query=query, params=params)
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


def _apply_label_where(
    targets: Dict[str, ASTObject],
    *,
    left: LabelRef,
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
    if not isinstance(target, ASTNode):
        raise _unsupported(
            "Cypher label predicates in WHERE currently support node aliases only",
            field="where",
            value=left.alias,
            line=left.span.line,
            column=left.span.column,
        )

    filter_dict = dict(target.filter_dict or {})
    for label in left.labels:
        filter_dict[f"label__{label}"] = True
    target.filter_dict = filter_dict


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
    if query.match is None:
        raise _unsupported(
            "Cypher MATCH lowering requires a MATCH clause",
            field="match",
            value=None,
            line=query.return_.span.line,
            column=query.return_.span.column,
        )
    ops = lower_match_clause(query.match, params=params)
    alias_targets = _alias_target(ops)
    where_out: List[WhereComparison] = []

    if query.where is not None:
        for predicate in query.where.predicates:
            if isinstance(predicate.left, LabelRef):
                _apply_label_where(alias_targets, left=predicate.left)
                continue
            if isinstance(predicate.right, PropertyRef):
                assert isinstance(predicate.left, PropertyRef)
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
                left=cast(PropertyRef, predicate.left),
                op=predicate.op,
                right=cast(Optional[CypherLiteral], predicate.right),
                params=params,
            )

    return LoweredCypherMatch(query=ops, where=where_out)


def _fresh_temp_name(existing: Set[str], prefix: str) -> str:
    candidate = prefix
    counter = 0
    while candidate in existing:
        counter += 1
        candidate = f"{prefix}{counter}"
    existing.add(candidate)
    return candidate


def _add_output_mapping(
    expr_to_output: Dict[str, str],
    *,
    source_expr: str,
    output_name: str,
    alias_name: Optional[str],
) -> None:
    expr_to_output[source_expr] = output_name
    if alias_name is not None:
        expr_to_output[alias_name] = output_name


def _lower_general_row_projection(
    query: CypherQuery,
    lowered: LoweredCypherMatch,
    *,
    params: Optional[Mapping[str, Any]],
) -> CompiledCypherQuery:
    alias_targets = _alias_target(lowered.query) if query.match is not None else {}
    active_match_alias = _active_match_alias(query, alias_targets=alias_targets)
    seed_rows = query.match is None

    if active_match_alias is None:
        row_steps: List[ASTObject] = [rows(table="nodes")]
    else:
        row_steps = [
            rows(
                table=_alias_table(
                    alias_targets[active_match_alias],
                    alias=active_match_alias,
                    line=query.return_.span.line,
                    column=query.return_.span.column,
                ),
                source=active_match_alias,
            )
        ]

    unwind_aliases: Set[str] = set()
    for unwind_clause in query.unwinds:
        if unwind_clause.alias in alias_targets or unwind_clause.alias in unwind_aliases:
            raise _unsupported(
                "Cypher UNWIND alias collides with an existing alias in this local subset",
                field="unwind.alias",
                value=unwind_clause.alias,
                line=unwind_clause.span.line,
                column=unwind_clause.span.column,
            )
        _validate_row_expr_scope(
            unwind_clause.expression.text,
            alias_targets=alias_targets,
            active_match_alias=active_match_alias,
            unwind_aliases=unwind_aliases,
            field="unwind",
            line=unwind_clause.span.line,
            column=unwind_clause.span.column,
        )
        row_steps.append(
            unwind(
                _row_expr_arg(
                    unwind_clause.expression,
                    params=params,
                    field="unwind",
                ),
                as_=unwind_clause.alias,
            )
        )
        unwind_aliases.add(unwind_clause.alias)

    aggregate_specs: List[_AggregateSpec] = []
    non_aggregate_items: List[ReturnItem] = []
    for item in query.return_.items:
        agg_spec = _aggregate_spec(item)
        if agg_spec is None:
            non_aggregate_items.append(item)
        else:
            aggregate_specs.append(agg_spec)

    if aggregate_specs and query.return_.distinct:
        raise _unsupported(
            "Cypher DISTINCT with aggregate RETURN/WITH is not yet supported in local lowering",
            field=query.return_.kind,
            value=[item.expression.text for item in query.return_.items],
            line=query.return_.span.line,
            column=query.return_.span.column,
        )

    expr_to_output: Dict[str, str] = {}
    available_columns: Set[str] = set()

    if aggregate_specs:
        pre_items: List[Tuple[str, Any]] = []
        key_names: List[str] = []
        temp_names: Set[str] = set()

        for item in non_aggregate_items:
            if item.expression.text in alias_targets:
                raise _unsupported(
                    "Cypher aggregate RETURN/WITH does not yet support whole-row alias grouping",
                    field=query.return_.kind,
                    value=item.expression.text,
                    line=item.span.line,
                    column=item.span.column,
                )
            _validate_row_expr_scope(
                item.expression.text,
                alias_targets=alias_targets,
                active_match_alias=active_match_alias,
                unwind_aliases=unwind_aliases,
                field=query.return_.kind,
                line=item.span.line,
                column=item.span.column,
            )
            output_name = item.alias or item.expression.text
            if output_name in available_columns:
                raise _unsupported(
                    "Duplicate Cypher projection names are not yet supported in local lowering",
                    field=query.return_.kind,
                    value=output_name,
                    line=item.span.line,
                    column=item.span.column,
                )
            pre_items.append(
                (
                    output_name,
                    _row_expr_arg(
                        item.expression,
                        params=params,
                        field=query.return_.kind,
                    ),
                )
            )
            key_names.append(output_name)
            available_columns.add(output_name)
            _add_output_mapping(
                expr_to_output,
                source_expr=item.expression.text,
                output_name=output_name,
                alias_name=item.alias,
            )

        aggregations: List[Sequence[Any]] = []
        for agg_spec in aggregate_specs:
            if agg_spec.output_name in available_columns:
                raise _unsupported(
                    "Duplicate Cypher projection names are not yet supported in local lowering",
                    field=query.return_.kind,
                    value=agg_spec.output_name,
                    line=agg_spec.span_line,
                    column=agg_spec.span_column,
                )
            if agg_spec.expr_text is None:
                aggregations.append((agg_spec.output_name, agg_spec.func))
            else:
                expr_text_obj = ExpressionText(text=agg_spec.expr_text, span=SourceSpan(
                    line=agg_spec.span_line,
                    column=agg_spec.span_column,
                    end_line=agg_spec.span_line,
                    end_column=agg_spec.span_column,
                    start_pos=0,
                    end_pos=0,
                ))
                _validate_row_expr_scope(
                    agg_spec.expr_text,
                    alias_targets=alias_targets,
                    active_match_alias=active_match_alias,
                    unwind_aliases=unwind_aliases,
                    field=query.return_.kind,
                    line=agg_spec.span_line,
                    column=agg_spec.span_column,
                )
                temp_name = _fresh_temp_name(temp_names, "__cypher_agg__")
                pre_items.append(
                    (
                        temp_name,
                        _row_expr_arg(
                            expr_text_obj,
                            params=params,
                            field=query.return_.kind,
                        ),
                    )
                )
                aggregations.append((agg_spec.output_name, agg_spec.func, temp_name))
            available_columns.add(agg_spec.output_name)
            _add_output_mapping(
                expr_to_output,
                source_expr=agg_spec.source_text,
                output_name=agg_spec.output_name,
                alias_name=agg_spec.output_name,
            )

        if key_names:
            if len(pre_items) > 0:
                row_steps.append(with_(pre_items))
            row_steps.append(group_by(key_names, aggregations))
        else:
            global_key = _fresh_temp_name(temp_names, "__cypher_group__")
            row_steps.append(with_([(global_key, 1)] + pre_items))
            row_steps.append(group_by([global_key], aggregations))
            row_steps.append(return_([(agg.output_name, agg.output_name) for agg in aggregate_specs]))
            available_columns = {agg.output_name for agg in aggregate_specs}
    else:
        if query.match is not None and not query.unwinds:
            return CompiledCypherQuery(
                Chain(_lower_projection_chain(query, lowered, params=params), where=lowered.where),
                seed_rows=False,
            )
        projection_fn = with_ if query.return_.kind == "with" else return_
        projection_items: List[Tuple[str, Any]] = []
        for item in query.return_.items:
            if item.expression.text in alias_targets:
                raise _unsupported(
                    "Whole-row alias RETURN/WITH is not supported once UNWIND/top-level row execution is active",
                    field=query.return_.kind,
                    value=item.expression.text,
                    line=item.span.line,
                    column=item.span.column,
                )
            _validate_row_expr_scope(
                item.expression.text,
                alias_targets=alias_targets,
                active_match_alias=active_match_alias,
                unwind_aliases=unwind_aliases,
                field=query.return_.kind,
                line=item.span.line,
                column=item.span.column,
            )
            output_name = item.alias or item.expression.text
            if output_name in available_columns:
                raise _unsupported(
                    "Duplicate Cypher projection names are not yet supported in local lowering",
                    field=query.return_.kind,
                    value=output_name,
                    line=item.span.line,
                    column=item.span.column,
                )
            projection_items.append(
                (
                    output_name,
                    _row_expr_arg(
                        item.expression,
                        params=params,
                        field=query.return_.kind,
                    ),
                )
            )
            available_columns.add(output_name)
            _add_output_mapping(
                expr_to_output,
                source_expr=item.expression.text,
                output_name=output_name,
                alias_name=item.alias,
            )
        row_steps.append(projection_fn(projection_items))
        if query.return_.distinct:
            row_steps.append(distinct())

    if query.order_by is not None:
        row_steps.append(
            _lower_order_by_outputs(
                query.order_by,
                available_columns=available_columns,
                expr_to_output=expr_to_output,
            )
        )
    _append_page_ops(row_steps, query=query, params=params)
    return CompiledCypherQuery(
        Chain(lowered.query + row_steps, where=lowered.where),
        seed_rows=seed_rows,
    )


def lower_cypher_query(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Chain:
    return compile_cypher_query(query, params=params).chain


def compile_cypher_query(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> CompiledCypherQuery:
    lowered = (
        lower_match_query(query, params=params)
        if query.match is not None
        else LoweredCypherMatch(query=[], where=[])
    )

    if query.match is not None and not query.unwinds:
        has_aggregates = any(_aggregate_spec(item) is not None for item in query.return_.items)
        if not has_aggregates:
            return CompiledCypherQuery(
                Chain(_lower_projection_chain(query, lowered, params=params), where=lowered.where),
                seed_rows=False,
            )

    return _lower_general_row_projection(query, lowered, params=params)
