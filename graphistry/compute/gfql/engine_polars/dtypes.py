"""Shared polars dtype classifiers for the native polars GFQL engine.

These encode the engine's cross-type / NaN-guard correctness CONTRACT (which dtypes
are numeric, float, or string-like), used by the predicate lowering, the expression
lowering, and the result projection. Keeping ONE definition avoids the guards
silently diverging when a polars dtype is added or a classification is fixed at one
site only. Polars is imported lazily (optional dependency), matching the engine's
convention.
"""
from typing import Any


def is_int(dt: Any) -> bool:
    """Signed/unsigned integer dtype (not bool, not float)."""
    import polars as pl
    return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def is_float(dt: Any) -> bool:
    import polars as pl
    return dt in (pl.Float32, pl.Float64)


def is_numeric(dt: Any) -> bool:
    """Integer or float — the operand types polars arithmetic/comparison accepts."""
    return is_int(dt) or is_float(dt)


def is_stringlike(dt: Any) -> bool:
    """String / Categorical / Enum — all compare/order like strings and all raise vs a
    numeric operand in polars (so all must trip the cross-type guard)."""
    import polars as pl
    if dt == pl.String:
        return True
    for name in ("Categorical", "Enum"):
        t = getattr(pl, name, None)
        if t is not None and (dt == t or isinstance(dt, t)):
            return True
    return False
