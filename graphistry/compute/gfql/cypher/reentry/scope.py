"""Alias-scope traversal helpers for hidden reentry property references."""
from __future__ import annotations

__all__ = [
    "_expr_hidden_reentry_aliases",
    "_binding_row_aliases_for_hidden_reentry_refs",
]

from typing import Any, List, Mapping, Optional, Sequence, Set, Tuple

from graphistry.compute.ast import ASTObject
from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    CaseWhen,
    ExprNode,
    FunctionCall,
    Identifier,
    IsNullOp,
    ListComprehension,
    ListLiteral,
    Literal as ExprLiteral,
    MapLiteral,
    PropertyAccessExpr,
    QuantifierExpr,
    SliceExpr,
    SubscriptExpr,
    UnaryOp,
    Wildcard,
)
from graphistry.compute.gfql.cypher.ast import (
    OrderByClause,
    ReturnClause,
    UnwindClause,
)
from graphistry.compute.gfql.cypher.reentry.naming import _is_hidden_reentry_property


def _expr_hidden_reentry_aliases(
    expr_text: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]] = None,
    field: str,
    line: int,
    column: int,
) -> Set[str]:
    from graphistry.compute.gfql.cypher import lowering as _lowering

    node = _lowering._parse_row_expr(
        expr_text,
        params=params,
        alias_targets=alias_targets,
        allow_missing_params=True,
        field=field,
        line=line,
        column=column,
    )
    aliases: Set[str] = set()

    def _visit(node_in: ExprNode) -> None:
        if isinstance(node_in, Identifier):
            return
        if isinstance(node_in, ExprLiteral):
            return
        if isinstance(node_in, UnaryOp):
            _visit(node_in.operand)
            return
        if isinstance(node_in, BinaryOp):
            _visit(node_in.left)
            _visit(node_in.right)
            return
        if isinstance(node_in, IsNullOp):
            _visit(node_in.value)
            return
        if isinstance(node_in, FunctionCall):
            for arg in node_in.args:
                _visit(arg)
            return
        if isinstance(node_in, Wildcard):
            return
        if isinstance(node_in, CaseWhen):
            _visit(node_in.condition)
            _visit(node_in.when_true)
            _visit(node_in.when_false)
            return
        if isinstance(node_in, QuantifierExpr):
            _visit(node_in.source)
            _visit(node_in.predicate)
            return
        if isinstance(node_in, ListComprehension):
            _visit(node_in.source)
            if node_in.predicate is not None:
                _visit(node_in.predicate)
            if node_in.projection is not None:
                _visit(node_in.projection)
            return
        if isinstance(node_in, ListLiteral):
            for item in node_in.items:
                _visit(item)
            return
        if isinstance(node_in, MapLiteral):
            for _key, value in node_in.items:
                _visit(value)
            return
        if isinstance(node_in, SubscriptExpr):
            _visit(node_in.value)
            _visit(node_in.key)
            return
        if isinstance(node_in, SliceExpr):
            _visit(node_in.value)
            if node_in.start is not None:
                _visit(node_in.start)
            if node_in.stop is not None:
                _visit(node_in.stop)
            return
        if isinstance(node_in, PropertyAccessExpr):
            if _is_hidden_reentry_property(node_in.property) and isinstance(node_in.value, Identifier):
                root = node_in.value.name.split(".", 1)[0]
                if root in alias_targets:
                    aliases.add(root)
                return
            _visit(node_in.value)
            return

    _visit(node)
    return aliases


def _binding_row_aliases_for_hidden_reentry_refs(
    *,
    unwinds: Sequence[UnwindClause],
    clause: ReturnClause,
    order_by_clause: Optional[OrderByClause],
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> Set[str]:
    if not unwinds and clause.kind != "with":
        # Hidden reentry refs in plain MATCH→WITH/RETURN reentry flows do not
        # require the bindings-row path. The forced promotion is only needed
        # for collect/UNWIND carry forwarding corridors and trailing WITH
        # narrowing stages that still reference hidden carry columns.
        return set()
    aliases: Set[str] = set()
    expr_texts: List[Tuple[str, int, int, str]] = []

    for unwind_clause in unwinds:
        expr_texts.append(
            (
                unwind_clause.expression.text,
                unwind_clause.span.line,
                unwind_clause.span.column,
                "unwind",
            )
        )
    for item in clause.items:
        expr_texts.append(
            (
                item.expression.text,
                item.span.line,
                item.span.column,
                clause.kind,
            )
        )
    if order_by_clause is not None:
        for order_item in order_by_clause.items:
            expr_texts.append(
                (
                    order_item.expression.text,
                    order_item.span.line,
                    order_item.span.column,
                    "order_by",
                )
            )

    for expr_text, line, column, field_name in expr_texts:
        if expr_text == "*":
            continue
        aliases.update(
            _expr_hidden_reentry_aliases(
                expr_text,
                alias_targets=alias_targets,
                params=params,
                field=field_name,
                line=line,
                column=column,
            )
        )
    return aliases
