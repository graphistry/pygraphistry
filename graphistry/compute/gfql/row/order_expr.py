from __future__ import annotations

from typing import Final, cast

from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    ExprNode,
    FunctionCall,
    Identifier,
    ListComprehension,
    Literal,
    MapLiteral,
    QuantifierExpr,
    Wildcard,
    iter_expr_children,
    is_expr_node,
)
from graphistry.compute.gfql.language_defs import GFQL_ORDER_AGG_ALIAS_FUNCTIONS

_ORDER_UNSUPPORTED_NODE_TYPES: Final[tuple[type, ...]] = (
    QuantifierExpr,
    ListComprehension,
    MapLiteral,
)


def is_order_aggregate_alias_ast(node: object) -> bool:
    if not isinstance(node, FunctionCall):
        return False
    if node.name.lower() not in GFQL_ORDER_AGG_ALIAS_FUNCTIONS or len(node.args) != 1:
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


def extract_temporal_duration_sort_ast(node: object) -> tuple[ExprNode, str, int] | None:
    if not isinstance(node, BinaryOp) or node.op not in {"+", "-"}:
        return None
    if isinstance(node.right, Literal) and isinstance(node.right.value, str):
        duration_text = node.right.value.strip()
        if duration_text.startswith(("P", "-P")):
            return node.left, duration_text, (1 if node.op == "+" else -1)
    if node.op == "+" and isinstance(node.left, Literal) and isinstance(node.left.value, str):
        duration_text = node.left.value.strip()
        if duration_text.startswith(("P", "-P")):
            return node.right, duration_text, 1
    return None
