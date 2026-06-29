"""Native polars lowering for the cypher row pipeline (Phase 2, vectorized).

This module lowers the *common* cypher expressions to native polars expressions
so the row ops stay vectorized on polars (no pandas round-trip). It is
deliberately CONSERVATIVE: ``lower_expr`` returns ``None`` for anything it can't
prove equivalent to pandas, and ``chain._run_calls_polars`` then raises an honest
``NotImplementedError`` (NO-CHEATING — there is NO pandas bridge; a row op the
polars engine can't lower natively declines, pointing at ``engine='pandas'``).
Differential parity vs pandas is the correctness gate.

Currently lowered: property access (``alias.prop`` → column), bare columns,
literals, arithmetic/comparison/boolean ``BinaryOp``, ``UnaryOp``, ``IsNullOp``,
``coalesce``/``abs``. Ops wired to native: ``select``/``with_``/``return_``
projection, ``order_by``, ``where_rows``, ``group_by``, ``unwind``. Everything
else (CASE, list/map, subscript, other functions, temporal arithmetic) → NIE.
"""
import contextvars
import re
from typing import Any, List, Optional, Sequence, Tuple

from graphistry.Plottable import Plottable
from .dtypes import is_float as _dtype_is_float, is_numeric as _dtype_is_numeric, is_stringlike as _dtype_is_stringlike


# Active row-table schema (col -> polars dtype), set by select/where/order_by around
# lowering so lower_expr can infer FLOAT operands and apply the NaN-comparison guard
# (below). Free to populate — the schema is already on the table, no scan.
_SCHEMA: "contextvars.ContextVar[dict]" = contextvars.ContextVar("gfql_polars_schema", default={})

# Comparison ops needing the NaN guard. Polars defines NaN as the LARGEST value, so
# NaN compares >/>=/== as TRUE — but IEEE/Python/pandas/Cypher compare any NaN as
# FALSE (and != as TRUE; the Neo4j TCK agrees). For float operands we mask the
# polars result to the IEEE answer. ``is_nan()`` is float-only, hence the inference.
_NAN_GUARD_OPS = frozenset({"<", ">", "<=", ">=", "=", "==", "<>", "!="})
_NAN_NE_OPS = frozenset({"<>", "!="})
_ORDER_OPS = frozenset({"<", ">", "<=", ">="})
# ops whose numeric-vs-string operands make polars raise (compare AND arithmetic)
_NUMSTR_OPS = _NAN_GUARD_OPS | frozenset({"+", "-", "*", "/", "%"})


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
    if o in ("AND", "OR"):
        # Cypher AND/OR are boolean operators with Kleene 3-valued logic (polars
        # Boolean & / | already match: true|null=true, false&null=false,
        # null&null=null). Cast operands to Boolean so a bare ``null`` literal
        # (lowered to a Null-dtype lit) doesn't raise `bitand not supported for
        # dtype null`; casting a real Boolean column is a no-op.
        import polars as pl
        lb, rb = left.cast(pl.Boolean), right.cast(pl.Boolean)
        return lb & rb if o == "AND" else lb | rb
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


_ISO_DURATION_RE = re.compile(r"^-?P(?=[0-9T])")

# ISO-8601 date / datetime / time-with-seconds-or-timezone. Cypher ``date({...})`` /
# ``time({...})`` / ``datetime({...})`` are lowered to these ISO strings; comparing
# them with polars string ``</>`` is LEXICOGRAPHIC (wrong across timezones/precision).
# Requires seconds or a timezone on bare times so ordinary ``'10:00'`` strings don't match.
_ISO_TEMPORAL_RE = re.compile(
    r"""^(
        \d{4}-\d{2}-\d{2}([T\ ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?
      | \d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})
      | \d{2}:\d{2}:\d{2}(\.\d+)?
    )$""",
    re.VERBOSE,
)


def _is_int_literal(node: Any) -> bool:
    """True if ``node`` is an integer Literal (not bool). Cypher integer division
    (``5/2 == 2``, truncating) diverges from polars true division (``2.5``) ONLY for
    constant integer operands — a column ``/`` int already returns Float on both
    engines (so it matches). Used to decline (NIE) literal/literal int division."""
    from graphistry.compute.gfql.expr_parser import Literal
    return isinstance(node, Literal) and isinstance(node.value, int) and not isinstance(node.value, bool)


def _is_iso_duration_literal(node: Any) -> bool:
    """True if ``node`` is a string Literal holding an ISO-8601 duration (``PT6M``,
    ``P1Y``, …) — what cypher ``duration({...})`` translates to. ``^-?P(?=[0-9T])``
    matches a duration without misfiring on ordinary strings like ``'Prefix'``."""
    from graphistry.compute.gfql.expr_parser import Literal
    return (
        isinstance(node, Literal)
        and isinstance(node.value, str)
        and _ISO_DURATION_RE.match(node.value) is not None
    )


def _is_iso_temporal_literal(node: Any) -> bool:
    """True if ``node`` is a string Literal holding an ISO date/datetime/time — what
    cypher ``date()``/``time()``/``datetime()`` constructors lower to. Used to decline
    (NIE) temporal comparison, which polars would do lexicographically (wrong)."""
    from graphistry.compute.gfql.expr_parser import Literal
    return (
        isinstance(node, Literal)
        and isinstance(node.value, str)
        and _ISO_TEMPORAL_RE.match(node.value) is not None
    )


def _expr_output_dtype(expr: Any) -> Any:
    """Output dtype of a lowered ``pl.Expr`` under the active table schema, or None if
    unresolvable. Schema-only (no data) on an empty LazyFrame — robust where AST-level
    type inference misses cases: ``int/int`` → Float (NaN-capable), function results
    (``abs``/``coalesce``), and Categorical/Enum columns. Drives the NaN + cross-type
    guards from real dtypes instead of re-deriving types from the parse tree."""
    import polars as pl
    try:
        return pl.LazyFrame(schema=_SCHEMA.get()).select(expr.alias("__gfql_dt__")).collect_schema()["__gfql_dt__"]
    except Exception:
        return None


def _is_cross_type(ldt: Any, rdt: Any) -> bool:
    """Numeric operand vs string-like operand — polars raises (compare:
    ``cannot compare string with numeric``; arithmetic: ``InvalidOperationError``;
    incl. all-null columns, which ``from_pandas`` types as String). pandas/cypher
    return a value/null, so decline natively. None dtype = unknown → not flagged."""
    if ldt is None or rdt is None:
        return False
    return (_dtype_is_numeric(ldt) and _dtype_is_stringlike(rdt)) or (_dtype_is_stringlike(ldt) and _dtype_is_numeric(rdt))


def _nan_guard(result: Any, op: str, left: Any, right: Any, ldt: Any, rdt: Any) -> Any:
    """Mask a comparison so NaN operands compare like IEEE/pandas/Cypher (always false;
    ``!=`` true) instead of polars (NaN = largest value). ``is_nan()`` is applied only
    to operands whose OUTPUT dtype is float — safe on the lowered float expr, never on
    a non-float one; int/string/bool comparisons are a no-op."""
    nan_terms = []
    if _dtype_is_float(ldt):
        nan_terms.append(left.is_nan())
    if _dtype_is_float(rdt):
        nan_terms.append(right.is_nan())
    if not nan_terms:
        return result
    any_nan = nan_terms[0]
    for term in nan_terms[1:]:
        any_nan = any_nan | term
    return (result | any_nan) if op in _NAN_NE_OPS else (result & ~any_nan)


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
        # Temporal arithmetic: cypher ``duration({...})`` is translated to an ISO
        # duration string literal (e.g. ``'PT6M'``), so ``a.time + duration(...)``
        # would lower to STRING CONCATENATION and sort/compare lexicographically —
        # a silent wrong answer. Decline natively (NIE) when ``+``/``-`` has an ISO
        # duration literal operand; the pandas engine handles temporal arithmetic.
        if node.op in ("+", "-") and (_is_iso_duration_literal(node.left) or _is_iso_duration_literal(node.right)):
            return None
        # ISO temporal ORDERING of two constructor-string literals lowers to
        # LEXICOGRAPHIC string ordering (wrong across timezones). Only ordering of
        # two temporal literals is declined: ``=``/``<>`` are lexicographically
        # correct, and a literal-vs-real-string-column compare must NOT be declined.
        if node.op in _ORDER_OPS and _is_iso_temporal_literal(node.left) and _is_iso_temporal_literal(node.right):
            return None
        # Integer-literal division: Cypher folds ``5/2`` to integer division (``2``,
        # truncating toward zero; ``x/0`` errors) but polars does true division
        # (``2.5``) — a silent wrong answer when embedded in a non-monotonic op (e.g.
        # ``ORDER BY n.val % (10/4)`` sorts differently). Decline natively (NIE); the
        # pandas engine folds it. (Column ``/`` int is Float on both, so not declined.)
        if node.op == "/" and _is_int_literal(node.left) and _is_int_literal(node.right):
            return None
        left = lower_expr(node.left, columns)
        right = lower_expr(node.right, columns)
        if left is None or right is None:
            return None
        ldt = rdt = None
        if node.op in _NUMSTR_OPS:
            # Numeric-vs-string-like operands make polars raise (compare AND
            # arithmetic, incl. AllOf-nested, all-null→String, Categorical). Decline
            # natively. Output dtypes catch int/int→Float division + function results
            # that AST inference missed.
            ldt, rdt = _expr_output_dtype(left), _expr_output_dtype(right)
            if _is_cross_type(ldt, rdt):
                return None
        result = _apply_binop(node.op, left, right)
        if result is not None and node.op in _NAN_GUARD_OPS:
            result = _nan_guard(result, node.op, left, right, ldt, rdt)
        return result
    if isinstance(node, UnaryOp):
        operand = lower_expr(node.operand, columns)
        if operand is None:
            return None
        if node.op == "-":
            return -operand
        if node.op.upper() == "NOT":
            # Cast to Boolean so ``NOT null`` (Null-dtype lit) yields null instead
            # of raising `dtype Null not supported in 'not' operation`; Cypher NOT
            # is 3-valued (NOT null = null). No-op on a real Boolean column.
            return ~operand.cast(pl.Boolean)
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


def _lower_with_schema(table: Any, fn):
    """Run a lowering callable with the active table schema published to ``_SCHEMA``
    (for the float-operand inference behind the NaN-comparison guard)."""
    token = _SCHEMA.set(dict(table.schema))
    try:
        return fn()
    finally:
        _SCHEMA.reset(token)


def select_polars(g: Plottable, items: Sequence[Any]) -> Optional[Plottable]:
    """Native polars projection; None if any item isn't lowerable."""
    table = _active_table(g)
    exprs = _lower_with_schema(table, lambda: lower_select_items(items, list(table.columns)))
    if exprs is None:
        return None
    out = table.select(exprs)
    if _select_emits_temporal_constructor_text(out):
        # A projected String column holds Cypher temporal-constructor text
        # (date({...}) etc.); the pandas projection normalizes it to ISO, not yet
        # native — decline honestly rather than leak the raw text (NO-CHEATING).
        # Only scans String columns, so numeric/bool projections pay nothing.
        return None
    return _rewrap(g, out)


def _select_emits_temporal_constructor_text(out: Any) -> bool:
    import polars as pl
    from graphistry.compute.gfql.lazy.engine.polars.projection import _has_temporal_constructor_text
    for name, dtype in out.schema.items():
        if dtype == pl.String and _has_temporal_constructor_text(out, name):
            return True
    return False


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
        lowered = _lower_with_schema(table, lambda: lower_expr_str(expr, columns))
        if lowered is None:
            return None
        preds.append(lowered)
    if not preds:
        return g  # empty WHERE -> identity
    combined = preds[0]
    for pred in preds[1:]:
        combined = combined & pred
    return _rewrap(g, table.filter(combined))


def order_by_polars(g: Plottable, keys: Sequence[Any]) -> Optional[Plottable]:
    """Native polars sort; None if any key isn't lowerable."""
    table = _active_table(g)
    lowered = _lower_with_schema(table, lambda: lower_order_by_keys(keys, list(table.columns)))
    if lowered is None:
        return None
    exprs, descending = lowered
    # nulls_last=False matches pandas sort_values default (NaN last only for asc);
    # cypher ORDER BY puts NULLs last — polars default is nulls_last=False, so set
    # it explicitly to match the pandas engine's na_position='last'.
    return _rewrap(g, table.sort(exprs, descending=descending, nulls_last=True))


# Aggregation funcs lowered to native polars; collect/collect_distinct/stdev/
# percentile etc. return None → caller declines (NIE, no pandas bridge).
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
    out = table.group_by(list(keys), maintain_order=True).agg(aggs)
    return _rewrap(g, out)


def unwind_polars(g: Plottable, expr: str, as_: str = "value") -> Optional[Plottable]:
    """Native polars UNWIND for a literal list (cross-join); None → caller NIEs.

    ``UNWIND [a, b, ...] AS x`` cross-joins each active row with the list values
    (matching cypher's per-row expansion and empty-list → 0 rows). List-column /
    expression unwinds (null/empty-element semantics) decline (NIE) for now.
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
    return _rewrap(g, table.join(rhs, how="cross"))


def can_select_native(items: Sequence[Any], columns: Sequence[str]) -> bool:
    return lower_select_items(items, columns) is not None


def can_order_by_native(keys: Sequence[Any], columns: Sequence[str]) -> bool:
    return lower_order_by_keys(keys, columns) is not None
