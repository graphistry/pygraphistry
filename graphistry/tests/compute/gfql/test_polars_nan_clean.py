"""_pl_nan_to_null: NaN->null correctness + the bound-frame immutability contract.

Frames handed to GFQL are immutable by contract; the clean-verdict cache is
sound under it AND is registered, so the supported after-mutation recipes
(gfql_clear_caches(), or hand a new frame) actually restore freshness -- the
pre-#1883 cache was unregistered, which turned a contract violation into an
unflushable wrong answer.
"""
import pytest

pl = pytest.importorskip("polars")

from graphistry.compute.gfql.lazy.engine.polars.nan_clean import (  # noqa: E402
    _PL_NAN_CLEAN_CACHE_IDS,
    _pl_nan_to_null,
)


def test_no_float_cols_is_noop_same_object():
    df = pl.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]})
    assert _pl_nan_to_null(df) is df


def test_nan_present_is_cleaned_to_null():
    df = pl.DataFrame({"w": [1.0, float("nan"), 3.0]})
    out = _pl_nan_to_null(df)
    assert out.get_column("w").null_count() == 1
    assert not bool(out.get_column("w").is_nan().any())


def test_clean_float_frame_identity_stable_and_cached():
    df = pl.DataFrame({"w": [1.0, 2.0, 3.0]})
    assert _pl_nan_to_null(df) is df
    assert id(df) in _PL_NAN_CLEAN_CACHE_IDS
    assert _pl_nan_to_null(df) is df  # O(1) repeat


def test_verdict_cache_is_registered_and_flushable():
    """THE F-01 FIX under the immutability contract: the cache may exist, but it
    must be visible to gfql_clear_caches so the after-mutation recipe works."""
    from graphistry.compute.gfql_unified import gfql_clear_caches

    df = pl.DataFrame({"w": [1.0, 2.0, 3.0]})
    _pl_nan_to_null(df)
    assert id(df) in _PL_NAN_CLEAN_CACHE_IDS
    df.replace_column(0, pl.Series("w", [1.0, float("nan"), 3.0]))  # contract violation
    gfql_clear_caches()  # the supported recovery recipe
    out = _pl_nan_to_null(df)
    assert out.get_column("w").null_count() == 1, "clear_caches did not restore freshness"


def test_new_frame_is_never_served_by_a_dead_frames_verdict():
    import gc

    dirty = pl.DataFrame({"w": [float("nan"), 2.0]})
    cleaned = _pl_nan_to_null(dirty)
    assert id(cleaned) in _PL_NAN_CLEAN_CACHE_IDS
    key = id(cleaned)
    del cleaned
    gc.collect()
    assert key not in _PL_NAN_CLEAN_CACHE_IDS  # weakref.finalize evicted


def test_distinct_frames_do_not_cross_contaminate():
    clean = pl.DataFrame({"w": [1.0, 2.0]})
    _pl_nan_to_null(clean)
    dirty = pl.DataFrame({"w": [float("nan"), 2.0]})
    assert _pl_nan_to_null(dirty).get_column("w").null_count() == 1


def test_lazyframe_keeps_unconditional_rewrite():
    lf = pl.DataFrame({"w": [1.0, float("nan")]}).lazy()
    out = _pl_nan_to_null(lf)
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().get_column("w").null_count() == 1
