"""Native polars lowering for the cypher row pipeline (Phase 2, vectorized).

The host-bridge in ``chain._run_calls_polars`` runs not-yet-native row ops via
the pandas expression engine. This module lowers the *common* cypher
expressions to native polars expressions so those ops stay vectorized on polars
(no pandas round-trip). It is deliberately CONSERVATIVE: ``lower_expr`` returns
``None`` for anything it can't prove equivalent to pandas, and the caller falls
back to the bridge. Differential parity vs pandas is the correctness gate.

Currently lowered: property access (``alias.prop`` → column), bare columns,
literals, arithmetic/comparison/boolean ``BinaryOp``, ``UnaryOp``, ``IsNullOp``.
Ops wired to native: ``select``/``with_``/``return_`` projection, ``order_by``.
Everything else (CASE, list/map, subscript, functions, temporal) → bridge.
"""
from typing import Any, List, Optional, Sequence, Tuple

from graphistry.Plottable import Plottable


def _parser():
    from graphistry.compute.gfql.row.pipeline import _gfql_expr_runtime_parser_bundle
    bundle = _gfql_expr_runtime_parser_bundle()
    if bundle is None:
        return None
    parse_expr, _validate, _mod = bundle
    return parse_expr


# Cypher binary operators → polars expression methods. Comparison/boolean use
# polars' null-propagating semantics, which match pandas for these scalar cases
# (verified by differential parity); anything subtler returns None upstream.
def _apply_binop(op: str, left: Any, right: Any) -> Optional[Any]:
    o = op.upper()
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return left / right
    if op == "%":
        return left % right
    if op in ("=", "=="):
        return left == right
    if op in ("<>", "!="):
        return left != right
    if op == "<":
        return left < right
    if op == ">":
        return left > right
    if op == "<=":
        return left <= right
    if op == ">=":
        return left >= right
    if o == "AND":
        return left & right
    if o == "OR":
        return left | right
    return None


def _resolve_property(alias: str, prop: str, columns: Sequence[str]) -> Optional[str]:
    """Resolve ``alias.prop`` to a row-table column (None if ambiguous/absent).

    Multi-entity bindings tables prefix columns (``n.val``); single-entity row
    tables expose the bare property column (``val``) plus an ``alias`` marker
    column. Prefer the prefixed form to avoid cross-entity collisions.
    """
    prefixed = f"{alias}.{prop}"
    if prefixed in columns:
        return prefixed
    if prop in columns and alias in columns:
        return prop
    return None


def lower_expr(node: Any, columns: Sequence[str]) -> Optional[Any]:
    """Lower a parsed cypher ExprNode to a polars expression, or None to bridge."""
    import polars as pl
    from graphistry.compute.gfql.expr_parser import (
        Identifier, Literal, BinaryOp, UnaryOp, IsNullOp, PropertyAccessExpr,
    )

    if isinstance(node, Literal):
        return pl.lit(node.value)
    if isinstance(node, Identifier):
        return pl.col(node.name) if node.name in columns else None
    if isinstance(node, PropertyAccessExpr):
        if isinstance(node.value, Identifier):
            src = _resolve_property(node.value.name, node.property, columns)
            if src is not None:
                return pl.col(src)
        return None
    if isinstance(node, BinaryOp):
        left = lower_expr(node.left, columns)
        right = lower_expr(node.right, columns)
        if left is None or right is None:
            return None
        return _apply_binop(node.op, left, right)
    if isinstance(node, UnaryOp):
        operand = lower_expr(node.operand, columns)
        if operand is None:
            return None
        if node.op == "-":
            return -operand
        if node.op.upper() == "NOT":
            return ~operand
        return None
    if isinstance(node, IsNullOp):
        value = lower_expr(node.value, columns)
        if value is None:
            return None
        return value.is_not_null() if node.negated else value.is_null()
    return None


def lower_expr_str(expr: str, columns: Sequence[str]) -> Optional[Any]:
    """Parse + lower an expression string; None if unparseable or not lowerable."""
    import polars as pl
    if expr in columns:
        return pl.col(expr)
    parse = _parser()
    if parse is None:
        return None
    try:
        node = parse(expr)
    except Exception:
        return None
    return lower_expr(node, columns)


def lower_select_items(items: Sequence[Any], columns: Sequence[str]) -> Optional[List[Any]]:
    """Lower projection items [(alias, expr) | 'col'] to polars exprs, or None."""
    out: List[Any] = []
    for item in items:
        if isinstance(item, str):
            alias, expr = item, item
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            alias, expr = str(item[0]), item[1]
        else:
            return None
        if not isinstance(expr, str):
            return None
        lowered = lower_expr_str(expr, columns)
        if lowered is None:
            return None
        out.append(lowered.alias(alias))
    return out


def lower_order_by_keys(keys: Sequence[Any], columns: Sequence[str]) -> Optional[Tuple[List[Any], List[bool]]]:
    """Lower order_by [(expr, direction)] to (polars exprs, descending flags)."""
    exprs: List[Any] = []
    descending: List[bool] = []
    for key in keys:
        if not isinstance(key, (list, tuple)) or len(key) != 2:
            return None
        expr, direction = key
        if not isinstance(expr, str) or not isinstance(direction, str):
            return None
        lowered = lower_expr_str(expr, columns)
        if lowered is None:
            return None
        exprs.append(lowered)
        descending.append(direction.lower() == "desc")
    return exprs, descending


def _active_table(g: Plottable) -> Any:
    if g._nodes is not None:
        return g._nodes
    return g._edges


def _rewrap(g: Plottable, table_df: Any) -> Plottable:
    """Set the new active row table (mirrors frame_ops.row_table for polars)."""
    from graphistry.compute.gfql.row import frame_ops
    from graphistry.compute.gfql.row.pipeline import _RowPipelineAdapter
    return frame_ops.row_table(_RowPipelineAdapter(g), table_df)


def select_polars(g: Plottable, items: Sequence[Any]) -> Optional[Plottable]:
    """Native polars projection; None if any item isn't lowerable."""
    table = _active_table(g)
    exprs = lower_select_items(items, list(table.columns))
    if exprs is None:
        return None
    return _rewrap(g, table.select(exprs))


def order_by_polars(g: Plottable, keys: Sequence[Any]) -> Optional[Plottable]:
    """Native polars sort; None if any key isn't lowerable."""
    table = _active_table(g)
    lowered = lower_order_by_keys(keys, list(table.columns))
    if lowered is None:
        return None
    exprs, descending = lowered
    # nulls_last=False matches pandas sort_values default (NaN last only for asc);
    # cypher ORDER BY puts NULLs last — polars default is nulls_last=False, so set
    # it explicitly to match the pandas engine's na_position='last'.
    return _rewrap(g, table.sort(exprs, descending=descending, nulls_last=True))


def can_select_native(items: Sequence[Any], columns: Sequence[str]) -> bool:
    return lower_select_items(items, columns) is not None


def can_order_by_native(keys: Sequence[Any], columns: Sequence[str]) -> bool:
    return lower_order_by_keys(keys, columns) is not None
