from __future__ import annotations

from typing import Final, cast

from graphistry.compute.gfql.expr_parser import (
    ExprNode,
    FunctionCall,
    Identifier,
    ListComprehension,
    ListLiteral,
    MapLiteral,
    QuantifierExpr,
    SliceExpr,
    SubscriptExpr,
    Wildcard,
    iter_expr_children,
    is_expr_node,
)


_ORDER_AGG_ALIAS_FUNCS: Final[frozenset[str]] = frozenset(
    {"count", "sum", "min", "max", "avg", "mean", "collect"}
)
_ORDER_UNSUPPORTED_NODE_TYPES: Final[tuple[type, ...]] = (
    QuantifierExpr,
    ListComprehension,
    ListLiteral,
    MapLiteral,
    SubscriptExpr,
    SliceExpr,
)


def is_order_aggregate_alias_ast(node: object) -> bool:
    if not isinstance(node, FunctionCall):
        return False
    if node.name.lower() not in _ORDER_AGG_ALIAS_FUNCS or len(node.args) != 1:
        return False

    arg = node.args[0]
    return isinstance(arg, (Wildcard, Identifier)) and (
        not isinstance(arg, Identifier) or arg.name != ""
    )


def order_expr_ast_static_supported(node: object) -> bool:
    if not is_expr_node(node):
        return True
    expr_node = cast(ExprNode, node)
    if isinstance(expr_node, _ORDER_UNSUPPORTED_NODE_TYPES):
        return False
    return all(order_expr_ast_static_supported(child) for child in iter_expr_children(expr_node))
