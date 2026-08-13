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
    from typing_extensions import TypeGuard, TypeIs
    # `PolarsFrame` / `PolarsT` were defined here first; they now live in the canonical
    # engine-typing module (graphistry.compute.typing) and are re-exported so this module's
    # existing importers keep working off ONE definition.
    from graphistry.compute.typing import PolarsDType, PolarsFrame, PolarsT


def is_polars_dtype(dt: object) -> "TypeGuard[PolarsDType]":
    """True if ``dt`` is a polars dtype (class or instance -- the metaclass puts
    both under the polars module). Import-light module sniff, mirroring
    ``Engine.is_polars_df`` for frames, and declared ``TypeGuard`` for the same
    reason: callers hold a dtype typed ``Any``/``object`` (pandas' ``DType``
    alias is ``Any``), and the guard is what proves the polars dtype API is
    available in the branch it opens."""
    return "polars" in str(type(dt).__module__)


def is_int(dt: "Optional[PolarsDType]") -> bool:
    """Signed/unsigned integer dtype (not bool, not float)."""
    import polars as pl
    return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def is_float(dt: "Optional[PolarsDType]") -> bool:
    import polars as pl
    return dt in (pl.Float32, pl.Float64)


def is_numeric(dt: "Optional[PolarsDType]") -> bool:
    """Integer or float — the operand types polars arithmetic/comparison accepts."""
    return is_int(dt) or is_float(dt)


def is_stringlike(dt: "Optional[PolarsDType]") -> bool:
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


# --- cross-ENGINE dtype classification (pandas/numpy/arrow/polars), the pushdown
# planner's contract. Lives with the polars vocabulary because the polars arm is
# the one that made single-sourcing load-bearing: pandas' classifiers return a
# confident False for polars dtypes, so per-site fallbacks silently diverged. ---

def dtype_text(dtype: object) -> str:
    try:
        return str(dtype).lower()
    except Exception:
        return ""


def is_numeric_dtype_safe(dtype: object) -> bool:
    # polars first: pandas' is_numeric_dtype returns a confident False for polars
    # dtypes (no exception, so a fallback never runs). Polars' own is_numeric(),
    # not is_numeric above: Decimal is in scope for this planner.
    import pandas as pd
    if is_polars_dtype(dtype):
        try:
            return bool(dtype.is_numeric())  # type: ignore[union-attr]  # class-form calls raise -> except arm
        except Exception:
            return any(t in str(dtype).lower() for t in ("int", "float", "decimal"))
    try:
        return bool(pd.api.types.is_numeric_dtype(dtype))
    except Exception:
        kind = getattr(dtype, "kind", None)
        if isinstance(kind, str) and kind in {"b", "i", "u", "f", "c"}:
            return True
        dtype_txt = dtype_text(dtype)
        return any(token in dtype_txt for token in ("bool", "int", "float", "double", "decimal"))


def is_string_dtype_safe(dtype: object) -> bool:
    import pandas as pd
    if is_polars_dtype(dtype):
        return is_stringlike(dtype)
    try:
        return bool(pd.api.types.is_string_dtype(dtype))
    except Exception:
        dtype_txt = dtype_text(dtype)
        # "str" exact: pandas 3-era default string dtype reprs as "str" (not
        # "object"/"string[...]"); exact match so "struct" stays non-string.
        return (dtype_txt in ("object", "str") or "string" in dtype_txt
                or dtype_txt.endswith("[python]"))


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
