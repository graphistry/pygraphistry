"""NaN -> null coercion for polars frames entering GFQL (pandas missing-semantics parity).

pandas treats float NaN as MISSING (skipna/dropna drop it); polars distinguishes NaN from
null. Frames entering the GFQL surface are coerced here so polars/Arrow/cuDF input carrying
genuine NaN is treated as MISSING like the pandas oracle. Without this, ``engine='polars'``
on a frame with a real NaN keeps rows a filter/aggregation should drop (silent divergence
from pandas). Polars imported lazily (optional dependency), per engine convention.
"""
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    import polars as pl
    from .dtypes import PolarsFrame


# Ids of polars frames already verified NaN-free (or produced NaN-free by cleaning).
# Bound frames are treated as IMMUTABLE: after mutating one in place, rebind
# (g.nodes(df)/g.edges(df)) or call gfql_clear_caches().
_nan_free_frame_id_cache: Set[int] = set()


def _mark_pl_nan_clean(df: "pl.DataFrame") -> None:
    frame_id = id(df)
    _nan_free_frame_id_cache.add(frame_id)
    try:
        weakref.finalize(df, _nan_free_frame_id_cache.discard, frame_id)
    except TypeError:  # pragma: no cover - pl.DataFrame is weakref-able; guard anyway
        _nan_free_frame_id_cache.discard(frame_id)


from graphistry.compute.gfql.cache_registry import register_clearable_dict as _register_clearable
_register_clearable("_nan_free_frame_id_cache", _nan_free_frame_id_cache)


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

    Identity-stable + O(1)-repeat: an eager DataFrame is probed once for real NaN
    (``is_nan().any()`` per float column); a clean frame is returned UNCHANGED
    (same object) and its verdict cached (sound under the immutability contract
    above; ``gfql_clear_caches()`` flushes it). Only columns that genuinely carry
    NaN are rewritten -- values identical to the old unconditional ``fill_nan``.
    Frames without float columns short-circuit before any probe."""
    import polars as pl
    # collect_schema(): resolves the LazyFrame schema without a PerformanceWarning
    # (LazyFrame.schema is deprecated for that); on eager DataFrames .schema is free.
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    float_cols = [c for c, dt in schema.items() if dt in (pl.Float32, pl.Float64)]
    if not float_cols:
        return df
    if isinstance(df, pl.DataFrame):
        if id(df) in _nan_free_frame_id_cache:
            return df
        nan_cols = [c for c in float_cols if df.get_column(c).is_nan().any()]
        if not nan_cols:
            _mark_pl_nan_clean(df)
            return df
        cleaned = df.with_columns([pl.col(c).fill_nan(None) for c in nan_cols])
        _mark_pl_nan_clean(cleaned)
        return cleaned
    # LazyFrame (rare): no cheap eager NaN probe -> keep the unconditional rewrite.
    return df.with_columns([pl.col(c).fill_nan(None) for c in float_cols])
