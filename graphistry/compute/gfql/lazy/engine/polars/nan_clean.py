"""NaN -> null coercion for polars frames entering GFQL (pandas missing-semantics parity).

pandas treats float NaN as MISSING (skipna/dropna drop it); polars distinguishes NaN from
null. Frames entering the GFQL surface are coerced here so polars/Arrow/cuDF input carrying
genuine NaN is treated as MISSING like the pandas oracle. Without this, ``engine='polars'``
on a frame with a real NaN keeps rows a filter/aggregation should drop (silent divergence
from pandas). Polars imported lazily (optional dependency), per engine convention.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    from .dtypes import PolarsFrame


# Ids of polars frames already verified NaN-free (or produced NaN-free by cleaning).
# Recycle-safe: a weakref.finalize on each cached frame evicts its id on GC, so a reused
# id can never be a stale hit while the original frame is alive. This turns the repeated
# per-hop NaN probe on a RESIDENT graph (seeded Search / native-hop hammers the same edge
# frame every call) from O(E)-per-call into O(1) after the first check — the dominant
# per-call cost for polars/polars-gpu seeded traversal on float-column (i.e. real) graphs.
#
# Freshness contract (release decision, 2026-08-13): the UNDECLARED ingest path
# sees the frame's CURRENT content on every call -- no cross-call verdicts. A
# clean-verdict memo keyed by id() returned stale "clean" after polars' in-place
# mutation APIs (extend/replace_column/insert_column/hstack(in_place=True)) and
# served WRONG ROWS through the public API. The probe below IS the cheapest
# freshness check, so the memo could not be validated, only removed. Cross-call
# reuse belongs to the DECLARED index layer (explicit gfql_index_all), which
# carries a documented frame-immutability assumption.


def _pl_nan_to_null(df: "PolarsFrame") -> "PolarsFrame":
    """Convert NaN -> null in float columns of a polars frame.

    Takes/returns the ``PolarsFrame`` UNION rather than the constrained ``PolarsT`` TypeVar
    (which would preserve eager-in -> eager-out): callers behind ``is_polars_df`` hold a frame
    they only know to be *polars*, and a constrained TypeVar rejects a union argument outright.
    An ``@overload`` set would recover the per-flavour precision, but it cannot be used here --
    polars is TYPE_CHECKING-only, so on the polars-less type-lint lane every signature collapses
    to ``Any -> Any`` and mypy rejects the set as unmatchable.

    Matches ``pl.from_pandas(nan_to_null=True)`` (the pandas-input path) so a *native*
    polars / Arrow / cuDF input carrying genuine NaN is treated as MISSING like the pandas
    oracle (which skipna/dropna's NaN). No-op when there are no float columns.

    Identity-stable: an eager DataFrame is probed for real NaN per call
    (``is_nan().any()`` per float column -- the freshness contract above), a clean
    frame is returned UNCHANGED (same object), and only columns that genuinely
    carry NaN are rewritten (values identical to the old unconditional
    ``fill_nan`` -- it never touches non-NaN cells). Frames without float columns
    short-circuit before any probe."""
    import polars as pl
    # collect_schema(): resolves the LazyFrame schema without a PerformanceWarning
    # (LazyFrame.schema is deprecated for that); on eager DataFrames .schema is free.
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    float_cols = [c for c, dt in schema.items() if dt in (pl.Float32, pl.Float64)]
    if not float_cols:
        return df
    if isinstance(df, pl.DataFrame):
        nan_cols = [c for c in float_cols if df.get_column(c).is_nan().any()]
        if not nan_cols:
            return df
        return df.with_columns([pl.col(c).fill_nan(None) for c in nan_cols])
    # LazyFrame (rare): no cheap eager NaN probe -> keep the unconditional rewrite.
    return df.with_columns([pl.col(c).fill_nan(None) for c in float_cols])
