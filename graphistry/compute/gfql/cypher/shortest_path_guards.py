from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Set

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher._boolean_expr_text import boolean_expr_to_text
from graphistry.compute.gfql.cypher.ast import CypherQuery
from graphistry.compute.gfql.expr_parser import GFQLExprParseError, collect_identifiers, parse_expr


def _expr_mentions_alias(expr_text: str, aliases: Set[str]) -> bool:
    try:
        identifiers = collect_identifiers(parse_expr(expr_text))
    except GFQLExprParseError:
        return False
    return any(
        identifier == alias or identifier.startswith(f"{alias}.")
        for identifier in identifiers
        for alias in aliases
    )


def reject_shortest_path_alias_references_after_follow_on_match(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]],
) -> None:
    _ = params
    if len(query.matches) < 2:
        return

    alias_match_index: Dict[str, int] = {}
    for match_index, clause in enumerate(query.matches):
        pattern_aliases = clause.pattern_aliases or tuple(None for _ in clause.patterns)
        pattern_kinds = clause.pattern_alias_kinds or tuple("pattern" for _ in clause.patterns)
        for alias, kind in zip(pattern_aliases, pattern_kinds):
            if alias is not None and kind == "shortestPath":
                alias_match_index[alias] = match_index

    aliases = {
        alias for alias, match_index in alias_match_index.items()
        if match_index < len(query.matches) - 1
    }
    if not aliases:
        return

    def _check_expr(expr_text: str, *, field: str, line: int, column: int) -> None:
        if not _expr_mentions_alias(expr_text, aliases):
            return
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher shortestPath() named path aliases cannot yet be projected or used after a follow-on MATCH",
            field=field,
            value=expr_text,
            suggestion="Use a subset currently supported by the local Cypher compiler.",
            line=line,
            column=column,
            language="cypher",
        )

    for clause in query.matches:
        if clause.where is not None and clause.where.expr_tree is not None:
            _check_expr(
                boolean_expr_to_text(clause.where.expr_tree),
                field="where",
                line=clause.where.span.line,
                column=clause.where.span.column,
            )
    if query.where is not None and query.where.expr_tree is not None:
        _check_expr(
            boolean_expr_to_text(query.where.expr_tree),
            field="where",
            line=query.where.span.line,
            column=query.where.span.column,
        )
    for stage in query.with_stages:
        for projection_item in stage.clause.items:
            _check_expr(
                projection_item.expression.text,
                field=stage.clause.kind,
                line=projection_item.span.line,
                column=projection_item.span.column,
            )
        if stage.where is not None:
            _check_expr(
                stage.where.text,
                field="with.where",
                line=stage.where.span.line,
                column=stage.where.span.column,
            )
        if stage.order_by is not None:
            for order_item in stage.order_by.items:
                _check_expr(
                    order_item.expression.text,
                    field="order_by",
                    line=order_item.span.line,
                    column=order_item.span.column,
                )
    for return_item in query.return_.items:
        _check_expr(
            return_item.expression.text,
            field=query.return_.kind,
            line=return_item.span.line,
            column=return_item.span.column,
        )
    if query.order_by is not None:
        for order_item in query.order_by.items:
            _check_expr(
                order_item.expression.text,
                field="order_by",
                line=order_item.span.line,
                column=order_item.span.column,
            )
