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
    from .dtypes import PolarsT


# Ids of polars frames already verified NaN-free (or produced NaN-free by cleaning).
# Recycle-safe: a weakref.finalize on each cached frame evicts its id on GC, so a reused
# id can never be a stale hit while the original frame is alive. This turns the repeated
# per-hop NaN probe on a RESIDENT graph (seeded Search / native-hop hammers the same edge
# frame every call) from O(E)-per-call into O(1) after the first check — the dominant
# per-call cost for polars/polars-gpu seeded traversal on float-column (i.e. real) graphs.
_PL_NAN_CLEAN_IDS: Set[int] = set()


def _mark_pl_nan_clean(df: "pl.DataFrame") -> None:
    key = id(df)
    _PL_NAN_CLEAN_IDS.add(key)
    try:
        weakref.finalize(df, _PL_NAN_CLEAN_IDS.discard, key)
    except TypeError:  # pragma: no cover - pl.DataFrame is weakref-able; guard anyway
        _PL_NAN_CLEAN_IDS.discard(key)  # can't track lifetime -> don't cache (stay correct)


def _pl_nan_to_null(df: "PolarsT") -> "PolarsT":
    """Convert NaN -> null in float columns of a polars frame.

    Matches ``pl.from_pandas(nan_to_null=True)`` (the pandas-input path) so a *native*
    polars / Arrow / cuDF input carrying genuine NaN is treated as MISSING like the pandas
    oracle (which skipna/dropna's NaN). No-op when there are no float columns.

    Identity-stable + O(1)-repeat: an eager DataFrame is probed once for real NaN
    (``is_nan().any()`` per float column). A frame verified clean is returned UNCHANGED
    (same object) and its id is cached so subsequent calls skip the O(E) probe entirely;
    only columns that genuinely carry NaN are rewritten (values identical to the old
    unconditional ``fill_nan`` — it never touches non-NaN cells). This restores the #1726
    identity guard (reverted by #1731) AND removes the per-call O(E) re-scan that made
    polars/polars-gpu seeded Search grow with edge count (see plans/gfql-benchmark-numbers)."""
    import polars as pl
    float_cols = [c for c, dt in df.schema.items() if dt in (pl.Float32, pl.Float64)]
    if not float_cols:
        return df
    if isinstance(df, pl.DataFrame):
        if id(df) in _PL_NAN_CLEAN_IDS:
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
