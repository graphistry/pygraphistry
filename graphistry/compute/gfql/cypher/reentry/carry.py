"""Carry-column / prefix-order helpers for bounded reentry compilation.

Decides which prefix WITH outputs survive into the trailing MATCH as carried
scalar columns and validates that the prefix's ordering is reentry-safe.

Extracted from ``cypher.lowering`` (#1295, #1260 S2). The heavy
``_compile_bounded_reentry_query`` orchestrator that consumes these helpers
intentionally stays in ``lowering.py`` for this slice — it shares heavy edit
surface with the sibling #1294 cleanup (PR #1297). A follow-on slice can move
the orchestrator once #1297 has landed.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

from graphistry.compute.gfql.cypher.ast import (
    CypherQuery,
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
]


def _bounded_reentry_carry_columns(
    prefix_projection: "ResultProjectionPlan",
    *,
    projection_items: Sequence[str],
    query: CypherQuery,
    prefix_stage: ProjectionStage,
    reentry_alias_hint: Optional[str] = None,
) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """Return (reentry_alias, carried_scalar_columns, non_source_alias_names).

    Today's caller continues to consume the first two fields. The third lists
    other whole-row aliases the prefix carries that aren't the trailing-MATCH
    source — those are recorded on the ``ReentryPlan`` for future use and
    drive a compile-time failfast if any of them are referenced downstream
    (carrying non-source whole-row aliases through reentry is the slice
    handled by `#989` follow-up work).
    """
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


def _bounded_reentry_prefix_order_is_safe(
    *,
    prefix_stage: ProjectionStage,
    query: CypherQuery,
) -> bool:
    if prefix_stage.order_by is None:
        return True
    if query.order_by is not None:
        return True
    return prefix_stage.skip is None and _literal_limit_value(prefix_stage.limit) == 1


