"""Shared polars dtype classifiers for the native polars GFQL engine.

Encodes the cross-type / NaN-guard correctness CONTRACT (which dtypes are numeric, float,
string-like) used by predicate lowering, expression lowering, and result projection — ONE
definition so the guards can't silently diverge when a dtype is added or a classification is
fixed at one site. Polars imported lazily (optional dependency), per engine convention.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import polars as pl
    # TypeIs (PEP 742), not TypeGuard: it narrows the NEGATIVE branch too, which is the whole
    # point for `is_lazy` — every eager-side caller sits in the `else`. TYPE_CHECKING-only so
    # no runtime typing_extensions>=4.10 floor is introduced (this module is `from __future__
    # import annotations`, so the annotation is never evaluated).
    from typing_extensions import TypeIs
    # `PolarsFrame` / `PolarsT` were defined here first; they now live in the canonical
    # engine-typing module (graphistry.compute.typing) and are re-exported so this module's
    # existing importers keep working off ONE definition.
    from graphistry.compute.typing import PolarsFrame, PolarsT


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


# --- frame-shape helpers (lazy/eager agnostic), shared by chain orchestration + degree
# helpers so frame introspection is uniform across DataFrame-vs-LazyFrame ------------

def is_lazy(df: "PolarsFrame") -> "TypeIs[pl.LazyFrame]":
    """True for a ``pl.LazyFrame`` (vs an eager ``pl.DataFrame``).

    Declared ``TypeIs``, not ``bool``: ``PolarsFrame`` is a two-member union and this predicate
    decides WHICH member, so the type checker can carry that decision into BOTH branches — lazy
    in the ``if``, eager in the ``else``. As a plain ``bool`` the else-branch fact was invisible
    and every eager-only attribute access after a lazy guard (``.height``, ``.columns``,
    ``.schema``) had to be re-asserted with a ``cast`` at each call site. ``TypeGuard`` would not
    do: it narrows only the positive branch, and the eager side is the one that needs it."""
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
    """One-column frame of edge endpoints (src stacked on dst) as ``out_col`` — the engine's
    node-id-universe builder, shared by hop/hop_eager/chain; eager/lazy agnostic. ``dtype`` casts
    both sides to the node-id join dtype (polars won't coerce int/float join keys like pandas).
    NOT deduplicated: each caller applies its own ``.unique(...)`` variant, preserved verbatim
    from pre-refactor sites (plain vs ``subset=`` are equivalent on this one-column output —
    kept per-site for a byte-identical diff, not semantics)."""
    import polars as pl

    def _side(c: str) -> "pl.Expr":
        e = pl.col(c)
        return (e.cast(dtype) if dtype is not None else e).alias(out_col)
    return pl.concat([frame.select(_side(src)), frame.select(_side(dst))],
                     how="vertical_relaxed")
