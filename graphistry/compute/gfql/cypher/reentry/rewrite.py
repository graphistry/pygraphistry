"""AST/query rewriters that retarget reentry expressions onto carried columns."""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Callable, Mapping, Optional, Sequence, Tuple

from graphistry.compute.gfql.expr_parser import (
    ExprNode,
    GFQLExprParseError,
    Identifier,
    PropertyAccessExpr,
    collect_identifiers,
    parse_expr,
)
from graphistry.compute.gfql.cypher.ast import (
    CypherQuery,
    ExpressionText,
    MatchClause,
    PatternElement,
    ProjectionStage,
    PropertyEntry,
    ReturnClause,
)
from graphistry.compute.gfql.cypher.reentry.naming import (
    _reentry_hidden_column_name,
    _reentry_property_carry_name,
)
from graphistry.compute.gfql.cypher.reentry.lowering_support import _first_pattern_node_alias

__all__ = [
    "_rewrite_reentry_expr_to_hidden_properties",
    "_rewrite_reentry_projection_clause",
    "_rewrite_reentry_property_entry",
    "_rewrite_reentry_pattern_element",
    "_rewrite_reentry_match_clause",
    "_rewrite_reentry_projection_stage",
    "_rewrite_collect_unwind_reentry_query",
]

def _rewrite_reentry_expr_to_hidden_properties(
    expr: ExpressionText,
    *,
    carried_alias: str,
    carried_columns: Sequence[str],
    field: str,
    non_source_carried_props: Optional[Mapping[str, Tuple[str, ...]]] = None,
) -> ExpressionText:
    from graphistry.compute.gfql.cypher import lowering as _lowering

    has_non_source = bool(non_source_carried_props)
    if not carried_columns and not has_non_source:
        return expr
    needs_scalar_rewrite = any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(output_name)}(?![A-Za-z0-9_])", expr.text)
        or _reentry_hidden_column_name(output_name) in expr.text
        for output_name in carried_columns
    )
    needs_non_source_rewrite = False
    if non_source_carried_props:
        needs_non_source_rewrite = any(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(alias_name)}\.{re.escape(prop)}(?![A-Za-z0-9_])",
                expr.text,
            )
            for alias_name, props in non_source_carried_props.items()
            for prop in props
        )
    if not needs_scalar_rewrite and not needs_non_source_rewrite:
        return expr
    normalized_text = expr.text
    for output_name in carried_columns:
        hidden_name = _reentry_hidden_column_name(output_name)
        normalized_text = re.sub(
            rf"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*\.{re.escape(hidden_name)}(?![A-Za-z0-9_])",
            f"{carried_alias}.{hidden_name}",
            normalized_text,
        )
    try:
        node = parse_expr(normalized_text)
    except (GFQLExprParseError, ImportError) as exc:
        raise _lowering._unsupported(
            "Cypher MATCH after WITH carried-column rewrite requires a locally supported scalar expression",
            field=field,
            value=normalized_text,
            line=expr.span.line,
            column=expr.span.column,
        ) from exc

    # Slice 4.3b: rewrite `<non_source>.<prop>` PropertyAccessExpr nodes to
    # `<carried_alias>.<__cypher_reentry_<non_source>__<prop>__>` BEFORE the
    # bare-identifier substitution runs. This way `x.id` lowers to a property
    # access on the reentry-alias's row table where the prefix rewrite has
    # planted the hidden column.
    if has_non_source:
        assert non_source_carried_props is not None

        def _rewrite_property_access(child: ExprNode) -> ExprNode:
            if (
                isinstance(child, PropertyAccessExpr)
                and isinstance(child.value, Identifier)
                and child.value.name in non_source_carried_props
                and child.property in non_source_carried_props[child.value.name]
            ):
                return PropertyAccessExpr(
                    Identifier(carried_alias),
                    _reentry_hidden_column_name(
                        _reentry_property_carry_name(child.value.name, child.property)
                    ),
                )
            return _lowering._rebuild_expr_node(
                child,
                rewrite=_rewrite_property_access,
                error_context="reentry property carry rewrite",
            )

        node = _rewrite_property_access(node)

    replacements = {
        output_name: f"{carried_alias}.{_reentry_hidden_column_name(output_name)}"
        for output_name in carried_columns
    }
    identifiers = collect_identifiers(node)
    if not any(identifier in replacements for identifier in identifiers):
        rendered = _lowering._render_expr_node(node)
        if rendered == expr.text:
            return expr
        return ExpressionText(text=rendered, span=expr.span)
    return ExpressionText(
        text=_lowering._render_expr_node(_lowering._rewrite_expr_identifiers(node, replacements)),
        span=expr.span,
    )


def _rewrite_reentry_projection_clause(
    clause: ReturnClause,
    *,
    rewrite_expr: Callable[[ExpressionText, str], ExpressionText],
) -> ReturnClause:
    return replace(
        clause,
        items=tuple(
            replace(
                item,
                expression=rewritten_expr,
                alias=item.alias or (item.expression.text if rewritten_expr.text != item.expression.text else None),
            )
            for item in clause.items
            for rewritten_expr in (rewrite_expr(item.expression, clause.kind),)
        ),
    )


def _rewrite_reentry_property_entry(
    entry: PropertyEntry,
    *,
    rewrite_expr: Callable[[ExpressionText, str], ExpressionText],
) -> PropertyEntry:
    if not isinstance(entry.value, ExpressionText):
        return entry
    return replace(
        entry,
        value=rewrite_expr(entry.value, "match.property"),
    )


def _rewrite_reentry_pattern_element(
    element: PatternElement,
    *,
    rewrite_expr: Callable[[ExpressionText, str], ExpressionText],
) -> PatternElement:
    rewritten_properties = tuple(
        _rewrite_reentry_property_entry(entry, rewrite_expr=rewrite_expr)
        for entry in element.properties
    )
    return replace(element, properties=rewritten_properties)


def _rewrite_reentry_match_clause(
    clause: MatchClause,
    *,
    rewrite_expr: Callable[[ExpressionText, str], ExpressionText],
) -> MatchClause:
    from graphistry.compute.gfql.cypher import lowering as _lowering

    rewritten_where = None
    if clause.where is not None:
        rewritten_where = _lowering._rewrite_where_clause_and_resync(clause.where, rewrite_expr, "where")
    return replace(
        clause,
        patterns=tuple(
            tuple(
                _rewrite_reentry_pattern_element(element, rewrite_expr=rewrite_expr)
                for element in pattern
            )
            for pattern in clause.patterns
        ),
        where=rewritten_where if rewritten_where is not None else clause.where,
    )


def _rewrite_reentry_projection_stage(
    stage: ProjectionStage,
    *,
    rewrite_expr: Callable[[ExpressionText, str], ExpressionText],
) -> ProjectionStage:
    rewritten_order_by = None
    if stage.order_by is not None:
        rewritten_order_by = replace(
            stage.order_by,
            items=tuple(
                replace(
                    item,
                    expression=rewrite_expr(item.expression, "order_by"),
                )
                for item in stage.order_by.items
            ),
        )
    return replace(
        stage,
        clause=_rewrite_reentry_projection_clause(stage.clause, rewrite_expr=rewrite_expr),
        where=None if stage.where is None else rewrite_expr(stage.where, "where"),
        order_by=rewritten_order_by,
    )


def _rewrite_collect_unwind_reentry_query(query: CypherQuery) -> Optional[CypherQuery]:
    from graphistry.compute.gfql.cypher import lowering as _lowering

    if not query.with_stages or len(query.unwinds) != 1 or len(query.reentry_matches) != 1:
        return None
    prefix_stage = query.with_stages[0]
    remaining_with_stages = query.with_stages[1:]
    if (
        prefix_stage.where is not None
        or prefix_stage.order_by is not None
        or prefix_stage.skip is not None
        or prefix_stage.limit is not None
        or len(prefix_stage.clause.items) < 1
    ):
        return None
    unwind_clause = query.unwinds[0]
    # Find the collect(...) item that feeds the UNWIND
    collected_idx: Optional[int] = None
    collected_match_result: Optional[re.Match[str]] = None
    for idx, item in enumerate(prefix_stage.clause.items):
        output_name = item.alias or item.expression.text
        if output_name != unwind_clause.expression.text:
            continue
        m = re.fullmatch(
            r"collect\(\s*(distinct\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\)",
            item.expression.text,
            flags=re.IGNORECASE,
        )
        if m is not None:
            collected_idx = idx
            collected_match_result = m
            break
    if collected_idx is None or collected_match_result is None:
        return None
    reentry_alias = _first_pattern_node_alias(query.reentry_matches[0])
    if reentry_alias is None or reentry_alias != unwind_clause.alias:
        return None
    collected_item = prefix_stage.clause.items[collected_idx]
    source_alias = collected_match_result.group(2)
    rewritten_item = replace(
        collected_item,
        expression=ExpressionText(text=source_alias, span=collected_item.expression.span),
        alias=unwind_clause.alias,
    )
    # Rebuild items: put the whole-row alias first, then carried scalars.
    # The reentry machinery expects the whole-row alias to be the primary
    # projection source, so it must come first.
    other_items = tuple(
        item for i, item in enumerate(prefix_stage.clause.items) if i != collected_idx
    )
    rewritten_items = (rewritten_item,) + other_items
    rewritten_prefix_stage = replace(
        prefix_stage,
        clause=replace(
            prefix_stage.clause,
            items=rewritten_items,
            distinct=bool(collected_match_result.group(1)),
        ),
    )
    return replace(
        query,
        with_stages=(rewritten_prefix_stage,) + remaining_with_stages,
        unwinds=(),
    )
