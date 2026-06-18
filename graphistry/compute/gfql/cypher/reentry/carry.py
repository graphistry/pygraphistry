"""Carry-column and prefix-order helpers for bounded reentry compilation."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple

from graphistry.compute.gfql.cypher.ast import (
    CypherQuery,
    ExpressionText,
    LimitClause,
    ParameterRef,
    ProjectionStage,
)

if TYPE_CHECKING:
    from graphistry.compute.gfql.cypher.lowering import ResultProjectionPlan

__all__ = [
    "_bounded_reentry_carry_columns",
    "_bounded_reentry_scalar_prefix_columns",
    "_bounded_reentry_prefix_order_is_safe",
    "_literal_limit_value",
    "_resolved_limit_value",
]


def _bounded_reentry_carry_columns(
    prefix_projection: "ResultProjectionPlan",
    *,
    projection_items: Sequence[str],
    query: CypherQuery,
    prefix_stage: ProjectionStage,
    reentry_alias_hint: Optional[str] = None,
) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """Return (reentry_alias, carried_scalar_columns, non_source_alias_names)."""
    from graphistry.compute.gfql.cypher import lowering as _lowering

    whole_row_columns = tuple(
        column.output_name for column in prefix_projection.columns if column.kind == "whole_row"
    )
    if not whole_row_columns:
        raise _lowering._unsupported_at_span(
            "Cypher MATCH after WITH currently requires the prefix WITH stage to project at least one whole-row alias",
            field="with",
            value=projection_items,
            span=prefix_stage.span,
        )
    if reentry_alias_hint is not None and reentry_alias_hint in whole_row_columns:
        reentry_alias = reentry_alias_hint
    else:
        reentry_alias = whole_row_columns[0]
    non_source_aliases = tuple(name for name in whole_row_columns if name != reentry_alias)
    carried_columns = tuple(
        column.output_name for column in prefix_projection.columns if column.kind != "whole_row"
    )
    if not carried_columns:
        return reentry_alias, (), non_source_aliases
    invalid_output = next((name for name in carried_columns if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)), None)
    if invalid_output is not None:
        raise _lowering._unsupported_at_span(
            "Cypher MATCH after WITH carried scalar columns currently require identifier-style WITH aliases",
            field="with",
            value=invalid_output,
            span=prefix_stage.span,
        )
    if len(set(carried_columns)) != len(carried_columns):
        raise _lowering._unsupported_at_span(
            "Cypher MATCH after WITH carried scalar columns currently require distinct WITH aliases",
            field="with",
            value=carried_columns,
            span=prefix_stage.span,
        )
    return reentry_alias, carried_columns, non_source_aliases


def _bounded_reentry_scalar_prefix_columns(
    prefix_stage: ProjectionStage,
    *,
    projection_items: Sequence[str],
) -> Tuple[str, ...]:
    from graphistry.compute.gfql.cypher import lowering as _lowering

    if prefix_stage.order_by is not None or prefix_stage.skip is not None or prefix_stage.limit is not None:
        raise _lowering._unsupported_at_span(
            "Cypher MATCH after WITH scalar-only prefix stages do not yet support ORDER BY, SKIP, or LIMIT",
            field="with",
            value=projection_items,
            span=prefix_stage.span,
        )
    carried_columns = tuple(
        item.alias or item.expression.text
        for item in prefix_stage.clause.items
    )
    if not carried_columns:
        raise _lowering._unsupported_at_span(
            "Cypher MATCH after WITH scalar-only prefix stages require at least one scalar output",
            field="with",
            value=projection_items,
            span=prefix_stage.span,
        )
    invalid_output = next((name for name in carried_columns if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)), None)
    if invalid_output is not None:
        raise _lowering._unsupported_at_span(
            "Cypher MATCH after WITH scalar-only prefix stages currently require identifier-style WITH aliases",
            field="with",
            value=invalid_output,
            span=prefix_stage.span,
        )
    if len(set(carried_columns)) != len(carried_columns):
        raise _lowering._unsupported_at_span(
            "Cypher MATCH after WITH scalar-only prefix stages currently require distinct WITH aliases",
            field="with",
            value=carried_columns,
            span=prefix_stage.span,
        )
    return carried_columns


def _literal_limit_value(limit_clause: Optional[LimitClause]) -> Optional[int]:
    if limit_clause is None:
        return None
    value = limit_clause.value
    if isinstance(value, int):
        return value
    if isinstance(value, ParameterRef):
        return None
    text = value.text.strip()
    if not re.fullmatch(r"\d+", text):
        return None
    return int(text)


def _resolved_limit_value(
    limit_clause: Optional[LimitClause],
    *,
    params: Optional[Mapping[str, object]] = None,
) -> Optional[int]:
    value = _literal_limit_value(limit_clause)
    if value is not None:
        return value
    if limit_clause is None:
        return None
    raw = limit_clause.value
    param_name: Optional[str]
    if isinstance(raw, ParameterRef):
        param_name = raw.name
    elif isinstance(raw, ExpressionText):
        match = re.fullmatch(r"\$([A-Za-z_][A-Za-z0-9_]*)", raw.text.strip())
        if match is None:
            return None
        param_name = match.group(1)
    else:
        return None
    if params is None or param_name not in params:
        return None
    param_value = params[param_name]
    if isinstance(param_value, bool):
        return None
    if not isinstance(param_value, int):
        return None
    return param_value


def _bounded_reentry_prefix_order_is_safe(
    *,
    prefix_stage: ProjectionStage,
    query: CypherQuery,
    params: Optional[Mapping[str, object]] = None,
) -> bool:
    if prefix_stage.order_by is None:
        return True
    if query.order_by is not None:
        return True
    resolved_limit = _resolved_limit_value(prefix_stage.limit, params=params)
    return prefix_stage.skip is None and resolved_limit is not None and resolved_limit >= 0
