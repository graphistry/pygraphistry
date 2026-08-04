"""_pl_nan_to_null: NaN->null correctness + identity-stable + O(1)-repeat clean cache."""
import pytest

pl = pytest.importorskip("polars")

from graphistry.compute.gfql.lazy.engine.polars.nan_clean import (  # noqa: E402
    _PL_NAN_CLEAN_IDS,
    _pl_nan_to_null,
)


def test_no_float_cols_is_noop_same_object():
    df = pl.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]})
    out = _pl_nan_to_null(df)
    assert out is df


def test_nan_present_is_cleaned_to_null():
    df = pl.DataFrame({"w": [1.0, float("nan"), 3.0]})
    out = _pl_nan_to_null(df)
    # NaN -> null: null_count reflects the converted cell; no NaN remains.
    assert out.get_column("w").null_count() == 1
    assert not bool(out.get_column("w").is_nan().any())


def test_clean_float_frame_identity_stable_and_cached():
    df = pl.DataFrame({"w": [1.0, 2.0, 3.0]})  # float col, no NaN
    out1 = _pl_nan_to_null(df)
    assert out1 is df                        # clean -> unchanged (identity-stable)
    assert id(df) in _PL_NAN_CLEAN_IDS       # cached
    out2 = _pl_nan_to_null(df)
    assert out2 is df                        # O(1) cache hit -> same object


def test_distinct_frames_do_not_cross_contaminate():
    clean = pl.DataFrame({"w": [1.0, 2.0]})
    _pl_nan_to_null(clean)                          # caches `clean`
    dirty = pl.DataFrame({"w": [float("nan"), 2.0]})  # different object, has NaN
    out = _pl_nan_to_null(dirty)                    # must NOT be skipped by the cache
    assert out.get_column("w").null_count() == 1


def test_cleaned_output_is_cached_and_gc_evicts():
    import gc

    dirty = pl.DataFrame({"w": [float("nan"), 2.0]})
    cleaned = _pl_nan_to_null(dirty)
    assert id(cleaned) in _PL_NAN_CLEAN_IDS  # output cached too (resident-graph reuse)
    assert _pl_nan_to_null(cleaned) is cleaned
    key = id(cleaned)
    del cleaned
    gc.collect()
    assert key not in _PL_NAN_CLEAN_IDS      # weakref.finalize evicted -> no stale id reuse


def test_lazyframe_keeps_unconditional_rewrite():
    lf = pl.DataFrame({"w": [1.0, float("nan")]}).lazy()
    out = _pl_nan_to_null(lf)
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().get_column("w").null_count() == 1
