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
from . import gpu  # GPU-collects these ops when POLARS_GPU active, else eager (CPU unchanged)


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


def _lower_function(node: Any, columns: Sequence[str]) -> Optional[Any]:
    """Lower a whitelisted scalar cypher function to polars, or None to defer.

    Only functions whose polars mapping matches the pandas engine's semantics
    (verified by differential parity) are admitted; everything else returns None
    so the caller raises NotImplementedError rather than guessing.
    """
    name = node.name.lower()
    args: List[Any] = []
    for arg in node.args:
        lowered = lower_expr(arg, columns)
        if lowered is None:
            return None
        args.append(lowered)
    if name == "coalesce" and args:
        import polars as pl
        # cypher coalesce = first non-null; pl.coalesce has identical semantics.
        return pl.coalesce(args)
    if name == "abs" and len(args) == 1:
        return args[0].abs()
    return None


def lower_expr(node: Any, columns: Sequence[str]) -> Optional[Any]:
    """Lower a parsed cypher ExprNode to a polars expression, or None to defer."""
    import polars as pl
    from graphistry.compute.gfql.expr_parser import (
        Identifier, Literal, BinaryOp, UnaryOp, IsNullOp, PropertyAccessExpr, FunctionCall,
    )

    if isinstance(node, Literal):
        return pl.lit(node.value)
    if isinstance(node, FunctionCall):
        return _lower_function(node, columns)
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
            # Non-string projection value = constant literal (e.g. the synthetic
            # ``__cypher_group__`` = 1 for keyless aggregation).
            import polars as pl
            out.append(pl.lit(expr).alias(alias))
            continue
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
    return _rewrap(g, gpu.select(table, exprs))


def where_rows_polars(
    g: Plottable,
    filter_dict: Optional[dict] = None,
    expr: Optional[str] = None,
) -> Optional[Plottable]:
    """Native polars row-table WHERE; None if the predicate isn't lowerable.

    Cypher's 3-valued WHERE keeps only rows whose predicate is TRUE (NULL and
    FALSE are both dropped) — polars ``DataFrame.filter`` has exactly this
    semantics, and polars boolean ``|``/``&`` use Kleene logic, so a lowered
    ``pl.Expr`` predicate matches the pandas engine / cypher NULL handling
    without special-casing. filter_dict entries are scalar-equality conjuncts.
    """
    import polars as pl
    table = _active_table(g)
    columns = list(table.columns)
    preds: List[Any] = []
    if filter_dict:
        for col, val in filter_dict.items():
            if col not in columns or isinstance(val, (list, tuple, set, dict)):
                return None  # missing column / IN-list etc. -> defer (NIE)
            preds.append(pl.col(col) == val)
    if expr is not None:
        if not isinstance(expr, str):
            return None
        lowered = lower_expr_str(expr, columns)
        if lowered is None:
            return None
        preds.append(lowered)
    if not preds:
        return g  # empty WHERE -> identity
    combined = preds[0]
    for pred in preds[1:]:
        combined = combined & pred
    return _rewrap(g, gpu.where(table, combined))


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
    return _rewrap(g, gpu.sort(table, exprs, descending=descending, nulls_last=True))


# Aggregation funcs lowered to native polars; collect/collect_distinct/stdev/
# percentile etc. return None → bridge.
def _agg_expr(func: str, expr: Optional[str], columns: Sequence[str], alias: str) -> Optional[Any]:
    import polars as pl
    func = func.lower()
    if func == "count" and (expr is None or expr == "*"):
        return pl.len().alias(alias)
    if not isinstance(expr, str) or expr not in columns:
        return None
    col = pl.col(expr)
    if func == "count":
        return col.count().alias(alias)
    if func == "sum":
        return col.sum().alias(alias)
    if func in ("avg", "mean"):
        return col.mean().alias(alias)
    if func == "min":
        return col.min().alias(alias)
    if func == "max":
        return col.max().alias(alias)
    return None


def group_by_polars(g: Plottable, keys: Sequence[Any], aggregations: Sequence[Any]) -> Optional[Plottable]:
    """Native polars group-by; None if a key/agg isn't lowerable.

    Matches the pandas engine's ``dropna=False`` (null keys kept) and non-null
    aggregation semantics. Output order is first-occurrence (maintain_order),
    though the differential parity gate compares order-insensitively.
    """
    table = _active_table(g)
    cols = list(table.columns)
    if not keys or not all(isinstance(k, str) and k in cols for k in keys):
        return None
    aggs: List[Any] = []
    for agg in aggregations:
        if not isinstance(agg, (list, tuple)) or len(agg) not in (2, 3):
            return None
        alias = str(agg[0])
        func = str(agg[1])
        expr = agg[2] if len(agg) == 3 else None
        lowered = _agg_expr(func, expr, cols, alias)
        if lowered is None:
            return None
        aggs.append(lowered)
    out = gpu.group_agg(table, list(keys), aggs, maintain_order=True)
    return _rewrap(g, out)


def unwind_polars(g: Plottable, expr: str, as_: str = "value") -> Optional[Plottable]:
    """Native polars UNWIND for a literal list (cross-join); None to bridge.

    ``UNWIND [a, b, ...] AS x`` cross-joins each active row with the list values
    (matching cypher's per-row expansion and empty-list → 0 rows). List-column /
    expression unwinds (null/empty-element semantics) bridge for now.
    """
    import polars as pl
    from graphistry.compute.gfql.expr_parser import ListLiteral, Literal

    if not isinstance(expr, str):
        return None
    parse = _parser()
    if parse is None:
        return None
    try:
        node = parse(expr)
    except Exception:
        return None
    if not isinstance(node, ListLiteral) or not all(isinstance(it, Literal) for it in node.items):
        return None
    table = _active_table(g)
    if as_ in table.columns:
        return None
    values = [it.value for it in node.items if isinstance(it, Literal)]
    rhs = pl.DataFrame({as_: values})
    return _rewrap(g, gpu.join(table, rhs, how="cross"))


def can_select_native(items: Sequence[Any], columns: Sequence[str]) -> bool:
    return lower_select_items(items, columns) is not None


def can_order_by_native(keys: Sequence[Any], columns: Sequence[str]) -> bool:
    return lower_order_by_keys(keys, columns) is not None
