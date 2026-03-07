"""Compatibility shim for row ORDER BY helpers."""

from graphistry.compute.gfql.row.order_expr import (
    is_order_aggregate_alias_ast,
    order_expr_ast_static_supported,
)

__all__ = [
    "is_order_aggregate_alias_ast",
    "order_expr_ast_static_supported",
]
