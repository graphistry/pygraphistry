"""Vectorized polars filter_by_dict for the native polars GFQL engine.

Predicates are lowered to native polars expressions (no pandas round-trip) for
the common comparison / membership / string cases; exotic predicates fall back
to a single-column pandas evaluation. All filtering is a single vectorized
``df.filter(expr)`` — no per-row work, no Python materialization.
"""
import operator
from typing import Any, Dict, List, Optional

from graphistry.compute.predicates.ASTPredicate import ASTPredicate
from graphistry.compute.filter_by_dict import resolve_filter_column


def _cmp_expr(col_expr, op, val):
    if op is operator.gt:
        return col_expr > val
    if op is operator.lt:
        return col_expr < val
    if op is operator.ge:
        return col_expr >= val
    if op is operator.le:
        return col_expr <= val
    if op is operator.eq:
        return col_expr == val
    if op is operator.ne:
        return col_expr != val
    return None


def predicate_to_expr(col: str, pred: ASTPredicate):
    """Lower an ASTPredicate to a polars boolean expression, or None if unsupported."""
    import polars as pl

    c = pl.col(col)
    name = type(pred).__name__

    op = getattr(pred, "op", None)
    if op is not None and hasattr(pred, "val"):
        expr = _cmp_expr(c, op, pred.val)
        if expr is not None:
            return expr

    if name == "Between" and hasattr(pred, "lower") and hasattr(pred, "upper"):
        lo, hi = pred.lower, pred.upper
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if getattr(pred, "inclusive", True):
                return (c >= lo) & (c <= hi)
            return (c > lo) & (c < hi)

    if name == "IsIn" and hasattr(pred, "options"):
        opts = list(pred.options)
        if all(isinstance(o, (int, float, str, bool)) for o in opts):
            return c.is_in(opts)

    if name == "Contains" and hasattr(pred, "pat") and isinstance(pred.pat, str):
        case = getattr(pred, "case", True)
        pat = pred.pat if case else f"(?i){pred.pat}"
        return c.str.contains(pat, literal=False)

    if name in ("Startswith", "Endswith") and hasattr(pred, "pat") and isinstance(pred.pat, str):
        if getattr(pred, "case", True):
            return c.str.starts_with(pred.pat) if name == "Startswith" else c.str.ends_with(pred.pat)

    return None


def _is_membership(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, frozenset))


def filter_by_dict_polars(df, filter_dict: Optional[dict]):
    """Return rows of polars ``df`` matching all entries in ``filter_dict`` via one filter."""
    import polars as pl

    if not filter_dict:
        return df

    exprs: List[Any] = []
    temp_cols: Dict[str, Any] = {}
    for col, val in filter_dict.items():
        resolved_col, resolved_val = resolve_filter_column(df, col, val)
        if isinstance(resolved_val, ASTPredicate):
            expr = predicate_to_expr(resolved_col, resolved_val)
            if expr is not None:
                exprs.append(expr)
            else:
                # Rare/exotic predicate: evaluate the single column via pandas,
                # carry the mask as a temp column so it joins the single filter.
                col_pd = df.select(pl.col(resolved_col)).to_pandas()[resolved_col]
                tname = f"__gfql_mask_{len(temp_cols)}__"
                temp_cols[tname] = pl.Series(tname, resolved_val(col_pd).to_numpy())
                exprs.append(pl.col(tname))
        elif _is_membership(resolved_val):
            exprs.append(pl.col(resolved_col).is_in(list(resolved_val)))
        else:
            exprs.append(pl.col(resolved_col) == resolved_val)

    if not exprs:
        return df
    combined = exprs[0]
    for e in exprs[1:]:
        combined = combined & e

    if temp_cols:
        return df.with_columns(list(temp_cols.values())).filter(combined).drop(list(temp_cols))
    return df.filter(combined)
