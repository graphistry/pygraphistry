"""Cardinality-aware lean-combine DataFrameT helpers for the seeded chain executor (#1755).

Extracted from chain.py to keep the executor readable (mirrors the gfql_fast_paths.py
#1731 convention: one-directional import, no back-edge). These are byte-identical
replacements for the two-pass chain executor's full-frame `safe_merge`/`pandas.merge`
reconciliations when the small side is much smaller + unique on the id — an `isin`
membership filter instead of the join machinery. pandas-only (see `_lean_engine_ok`);
`GFQL_LEAN_COMBINE=0` restores the legacy merges. Imports only leaf modules.
"""
import os as _os
from typing import Optional
from graphistry.Engine import Engine
from graphistry.compute.typing import DataFrameT, SeriesT


def _lean_combine_enabled() -> bool:
    """Fast path is on by default; set GFQL_LEAN_COMBINE=0 to force the legacy
    full-frame merges (used by the differential parity harness)."""
    return _os.environ.get('GFQL_LEAN_COMBINE', '1') != '0'


# Only engage when the small side is at least this many times smaller than the
# full frame — below the gate the isin pre-pass is pure overhead vs the merge.
_LEAN_SHRINK_RATIO = 4


def _is_unique_ids(ids: SeriesT) -> bool:
    """Cheap uniqueness probe on a (small) id Series, engine-agnostic."""
    try:
        n = len(ids)
        if n <= 1:
            return True
        # pandas/cuDF both expose is_unique; fall back to nunique.
        iu = getattr(ids, 'is_unique', None)
        if iu is not None:
            return bool(iu)
        return int(ids.nunique()) == n
    except Exception:
        return False


def _lean_engine_ok(engine: Engine) -> bool:
    """Only pandas benefits: pandas' merge machinery (join-indexer build + block
    consolidation) dominates the seeded chain, so an isin pre-filter is a large
    win. On cuDF the GPU hash-join is already sub-ms and the extra isin pass +
    boolean-mask gather is net-negative (measured dgx-spark 26.02, ~0.95-0.99x),
    so the fast path stays off there. polars/dask/spark take their own engines."""
    return engine == Engine.PANDAS


def _lean_intersect_full(full: DataFrameT, key_frame: DataFrameT, key: str, engine: Engine) -> Optional[DataFrameT]:
    """Byte-identical replacement for ``safe_merge(full, key_frame[[key]], on=key,
    how='inner')`` when ``key_frame`` carries only ``key`` and is small + unique.

    ``full`` is the whole node frame (unique on ``key``); the inner merge keeps
    exactly the ``full`` rows whose id is in ``key_frame`` (1:1, ``full`` order,
    ``full`` columns), which is precisely an isin filter. Returns ``None`` to
    signal "not applicable — use the merge" (guards fan-out / column carry).
    """
    if not _lean_combine_enabled() or not _lean_engine_ok(engine):
        return None
    if key not in full.columns or key not in key_frame.columns:
        return None
    try:
        full_len = len(full)
        small_len = len(key_frame)
    except Exception:
        return None
    if small_len == 0:
        return None
    if small_len * _LEAN_SHRINK_RATIO > full_len:
        return None  # not enough size gap to be worth the isin pass
    # key_frame must contribute no columns beyond key, and be unique on key, else
    # the inner merge would fan out / add columns and diverge from an isin filter.
    if list(key_frame.columns) != [key]:
        return None
    if not _is_unique_ids(key_frame[key]):
        return None
    # isin() matches nulls, but merge's null-key semantics are version-dependent;
    # only the (small) key_frame need be null-free for equivalence: if it carries
    # no null, both isin and inner-merge drop full's null-id rows identically,
    # regardless of nulls in full. Checking the small side keeps this O(small).
    if bool(key_frame[key].isna().any()):
        return None
    out = full[full[key].isin(key_frame[key])]
    # match the merge's RangeIndex so any positional downstream use is identical.
    return out.reset_index(drop=True)


def _lean_prefilter_right(left: DataFrameT, right: DataFrameT, key: str, engine: Engine) -> DataFrameT:
    """Shrink ``right`` to the keys present in ``left`` before a ``how='left'``
    merge. A left merge discards unmatched ``right`` rows anyway, so this is
    byte-identical (row order = ``left`` order; matched right rows preserved,
    including any fan-out). Only shrinks when ``left`` is materially smaller.
    """
    if not _lean_combine_enabled() or not _lean_engine_ok(engine):
        return right
    if key not in left.columns or key not in right.columns:
        return right
    try:
        left_len = len(left)
        right_len = len(right)
    except Exception:
        return right
    if left_len == 0 or left_len * _LEAN_SHRINK_RATIO > right_len:
        return right
    return right[right[key].isin(left[key])]
