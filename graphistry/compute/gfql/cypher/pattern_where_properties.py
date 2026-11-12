from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Set, Tuple

from graphistry.compute.ast import ASTObject
from graphistry.compute.gfql.cypher.ast import (
    PropertyRef,
    SourceSpan,
    WhereClause,
    WherePredicate,
)
from graphistry.compute.gfql.expr_parser import (
    ExprNode,
    GFQLExprParseError,
    Identifier,
    PropertyAccessExpr,
    walk_expr_nodes,
)
from graphistry.compute.exceptions import GFQLValidationError

RowExprParser = Callable[..., ExprNode]


def expr_property_access_node_aliases(
    expr_text: str,
    *,
    parse_row_expr: RowExprParser,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],  # hygiene-ok: explicit-any -- heterogeneous Cypher bound parameters
    field: str,
    line: int,
    column: int,
) -> Set[str]:
    """Node aliases that appear in ``alias.prop`` access, including inside aggregates."""
    node = parse_row_expr(
        expr_text,
        params=params,
        alias_targets=alias_targets,
        allow_missing_params=True,
        field=field,
        line=line,
        column=column,
    )
    aliases: Set[str] = set()

    def _visit(node: ExprNode) -> None:
        if isinstance(node, PropertyAccessExpr) and isinstance(node.value, Identifier):
            root = node.value.name.split(".", 1)[0]
            if root in alias_targets:
                aliases.add(root)

    walk_expr_nodes(node, enter=_visit)
    return aliases


def expr_property_access_columns(
    expr_text: str,
    *,
    parse_row_expr: RowExprParser,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],  # hygiene-ok: explicit-any -- heterogeneous Cypher bound parameters
    field: str,
    line: int,
    column: int,
) -> Set[Tuple[str, str]]:
    """Typed ``(alias, property)`` pairs accessed anywhere in an expression."""
    node = parse_row_expr(
        expr_text,
        params=params,
        alias_targets=alias_targets,
        allow_missing_params=True,
        field=field,
        line=line,
        column=column,
    )
    accessed: Set[Tuple[str, str]] = set()

    def _visit(node: ExprNode) -> None:
        if isinstance(node, PropertyAccessExpr) and isinstance(node.value, Identifier):
            root = node.value.name.split(".", 1)[0]
            if root in alias_targets:
                accessed.add((root, node.property))

    walk_expr_nodes(node, enter=_visit)
    return accessed


def where_pattern_property_columns(
    where: WhereClause,
    *,
    parse_row_expr: RowExprParser,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],  # hygiene-ok: explicit-any -- heterogeneous Cypher bound parameters
    extra_expressions: Sequence[Tuple[str, SourceSpan]] = (),
) -> Dict[str, list[str]]:
    """Bare property columns needed by a pattern-WHERE and aggregate projections."""
    accessed: Set[Tuple[str, str]] = set()
    for predicate in where.predicates:
        if isinstance(predicate, WherePredicate):
            if isinstance(predicate.left, PropertyRef):
                accessed.add((predicate.left.alias, predicate.left.property))
            if isinstance(predicate.right, PropertyRef):
                accessed.add((predicate.right.alias, predicate.right.property))
    if where.expr_tree is not None:
        stack = [where.expr_tree]
        while stack:
            node = stack.pop()
            if node.op == "atom" and node.atom_text:
                try:
                    accessed.update(
                        expr_property_access_columns(
                            node.atom_text,
                            parse_row_expr=parse_row_expr,
                            alias_targets=alias_targets,
                            params=params,
                            field="where",
                            line=(node.atom_span or node.span).line,
                            column=(node.atom_span or node.span).column,
                        )
                    )
                except (GFQLValidationError, RuntimeError):
                    continue
            for child in (node.left, node.right):
                if child is not None:
                    stack.append(child)
    for text, span in extra_expressions:
        try:
            accessed.update(
                expr_property_access_columns(
                    text,
                    parse_row_expr=parse_row_expr,
                    alias_targets=alias_targets,
                    params=params,
                    field="return",
                    line=span.line,
                    column=span.column,
                )
            )
        except (GFQLValidationError, RuntimeError):
            continue
    by_alias: Dict[str, Set[str]] = {}
    for alias, column in accessed:
        by_alias.setdefault(alias, set()).add(column)
    return {alias: sorted(columns) for alias, columns in by_alias.items()}
