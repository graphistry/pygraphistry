"""Shared polars dtype classifiers for the native polars GFQL engine.

These encode the engine's cross-type / NaN-guard correctness CONTRACT (which dtypes
are numeric, float, or string-like), used by the predicate lowering, the expression
lowering, and the result projection. Keeping ONE definition avoids the guards
silently diverging when a polars dtype is added or a classification is fixed at one
site only. Polars is imported lazily (optional dependency), matching the engine's
convention.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, TypeVar, Union

if TYPE_CHECKING:
    import polars as pl
    PolarsFrame = Union["pl.DataFrame", "pl.LazyFrame"]
    # eager-in→eager-out / lazy-in→lazy-out (a Union return would type-error at call sites)
    PolarsT = TypeVar("PolarsT", "pl.DataFrame", "pl.LazyFrame")


def is_int(dt: "Optional[pl.DataType]") -> bool:
    """Signed/unsigned integer dtype (not bool, not float)."""
    import polars as pl
    return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def is_float(dt: "Optional[pl.DataType]") -> bool:
    import polars as pl
    return dt in (pl.Float32, pl.Float64)


def is_numeric(dt: "Optional[pl.DataType]") -> bool:
    """Integer or float — the operand types polars arithmetic/comparison accepts."""
    return is_int(dt) or is_float(dt)


def is_stringlike(dt: "Optional[pl.DataType]") -> bool:
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


# --- frame-shape helpers (lazy/eager agnostic) -----------------------------------
# Shared so both the chain orchestration and the degree helpers introspect frames the
# same way regardless of DataFrame-vs-LazyFrame.

def is_lazy(df: "PolarsFrame") -> bool:
    """True for a ``pl.LazyFrame`` (vs an eager ``pl.DataFrame``)."""
    import polars as pl
    return isinstance(df, pl.LazyFrame)


def colnames(df: "PolarsFrame") -> List[str]:
    """Column names for an eager or lazy polars frame (no collect for lazy)."""
    return df.collect_schema().names() if is_lazy(df) else df.columns


def col_dtype(df: "PolarsFrame", col: str) -> "pl.DataType":
    """One column's dtype for an eager or lazy polars frame (no collect for lazy)."""
    return (df.collect_schema() if is_lazy(df) else df.schema)[col]


def endpoint_ids(frame: "PolarsT", src: str, dst: str, out_col: str,
                 dtype: "Optional[pl.DataType]" = None) -> "PolarsT":
    """One-column frame of edge endpoints (src stacked on dst) as ``out_col`` — the
    engine's node-id-universe builder, shared by hop/hop_eager/chain. ``dtype``
    casts both sides to the node-id join dtype (polars won't coerce int/float join
    keys like pandas does). NOT deduplicated: each caller applies its own
    ``.unique(...)`` variant (plain vs ``subset=`` differs and is load-bearing for
    lazy ``maintain_order`` behavior). Eager/lazy agnostic."""
    import polars as pl

    def _side(c: str) -> "pl.Expr":
        e = pl.col(c)
        return (e.cast(dtype) if dtype is not None else e).alias(out_col)
    return pl.concat([frame.select(_side(src)), frame.select(_side(dst))],
                     how="vertical_relaxed")
