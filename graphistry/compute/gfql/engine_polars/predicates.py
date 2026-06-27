"""Vectorized polars filter_by_dict for the native polars GFQL engine.

Predicates are lowered to native polars expressions (no pandas round-trip) for
the common comparison / membership / string / null cases. A predicate with no
native lowering raises ``NotImplementedError`` (NO-CHEATING: no pandas bridge —
silently evaluating one column via pandas would misrepresent pandas behavior as
polars and break the columnar/GPU assumptions; use ``engine='pandas'``). All
filtering is a single vectorized ``df.filter(expr)`` — no per-row work, no Python
materialization.
"""
import operator
import re
from typing import Any, List, Optional

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

    if name == "AllOf" and hasattr(pred, "predicates"):
        # Conjunction (e.g. ``n.val > 20 AND n.val < 90`` folds to AllOf[GT, LT]).
        # Lower each child natively and AND them; if ANY child can't lower, the
        # whole predicate can't (caller raises NIE — no pandas bridge).
        child_exprs = [predicate_to_expr(col, p) for p in pred.predicates]
        if child_exprs and all(e is not None for e in child_exprs):
            combined = child_exprs[0]
            for e in child_exprs[1:]:
                combined = combined & e
            return combined
        return None

    if name in ("IsNull", "IsNA"):
        return c.is_null()
    if name in ("NotNull", "NotNA"):
        return c.is_not_null()

    if name == "Contains" and hasattr(pred, "pat") and isinstance(pred.pat, str):
        case = getattr(pred, "case", True)
        pat = pred.pat if case else f"(?i){pred.pat}"
        return c.str.contains(pat, literal=False)

    if name in ("Startswith", "Endswith") and hasattr(pred, "pat") and isinstance(pred.pat, str):
        if getattr(pred, "case", True):
            return c.str.starts_with(pred.pat) if name == "Startswith" else c.str.ends_with(pred.pat)
        # Case-insensitive: anchored regex on the escaped literal (the literal pat
        # is treated literally; (?i) makes it case-insensitive). Matches the pandas
        # boundary predicate's lowercase-both-sides semantics for a single str pat.
        anchored = f"(?i)^{re.escape(pred.pat)}" if name == "Startswith" else f"(?i){re.escape(pred.pat)}$"
        return c.str.contains(anchored, literal=False)

    return None


def _is_membership(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, frozenset))


def filter_by_dict_polars(df, filter_dict: Optional[dict]):
    """Return rows of polars ``df`` matching all entries in ``filter_dict`` via one filter."""
    import polars as pl

    if not filter_dict:
        return df

    exprs: List[Any] = []
    for col, val in filter_dict.items():
        resolved_col, resolved_val = resolve_filter_column(df, col, val)
        if isinstance(resolved_val, ASTPredicate):
            expr = predicate_to_expr(resolved_col, resolved_val)
            if expr is None:
                # NO-CHEATING: no native lowering for this predicate, and we will
                # NOT bridge through pandas (evaluating one column via pandas would
                # present pandas semantics as polars). Decline honestly.
                raise NotImplementedError(
                    f"polars engine does not yet natively support the "
                    f"{type(resolved_val).__name__} predicate on column "
                    f"{resolved_col!r}; use engine='pandas' for this query "
                    f"(no pandas fallback — see plans/gfql-polars-engine NO-CHEATING)"
                )
            exprs.append(expr)
        elif _is_membership(resolved_val):
            exprs.append(pl.col(resolved_col).is_in(list(resolved_val)))
        elif isinstance(df.schema.get(resolved_col), pl.List):
            # Cypher label membership: ``MATCH (n:Label)`` lowers to a scalar match
            # on the reserved ``labels`` List column. A plain ``==`` would try to
            # cast the List to String and crash; ``list.contains`` is the correct
            # semantics (Label ∈ node's labels) and gives empty for a non-existent
            # label, matching pandas.
            exprs.append(pl.col(resolved_col).list.contains(resolved_val))
        else:
            exprs.append(pl.col(resolved_col) == resolved_val)

    if not exprs:
        return df
    combined = exprs[0]
    for e in exprs[1:]:
        combined = combined & e
    return df.filter(combined)
