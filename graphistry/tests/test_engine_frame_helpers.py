"""Unit tests for the engine-agnostic frame/series primitives in ``graphistry.Engine``.

These helpers (moved out of the cypher reentry executor) dispatch across
pandas/cuDF/polars, so both branches of each are exercised here. Polars cases are
skipped where polars is not installed (the pandas-only CI lanes); the polars lane
(``bin/test-polars.sh``) runs this file with polars present.
"""
import pandas as pd
import pytest

from graphistry.Engine import (
    assign_constant_columns,
    drop_columns,
    frame_filter,
    is_series_like,
    ordered_left_join,
    row_as_mapping,
    series_filter,
    series_not_null_mask,
    series_to_pylist,
)

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

polars_only = pytest.mark.skipif(not _HAS_POLARS, reason="polars not installed")


# --- pandas branches (run on every lane) ---------------------------------

def test_is_series_like_pandas() -> None:
    assert is_series_like(pd.Series([1, 2])) is True
    assert is_series_like(object()) is False


def test_series_not_null_mask_pandas() -> None:
    assert list(series_not_null_mask(pd.Series([1, None, 3]))) == [True, False, True]


def test_series_filter_pandas_drops_index() -> None:
    s = pd.Series([10, 20, 30])
    out = series_filter(s, pd.Series([True, False, True]))
    assert list(out) == [10, 30]
    assert list(out.index) == [0, 1]  # index reset


def test_frame_filter_pandas_drops_index() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = frame_filter(df, pd.Series([False, True, True]))
    assert out["a"].tolist() == [2, 3]
    assert list(out.index) == [0, 1]


def test_ordered_left_join_pandas_preserves_left_order() -> None:
    left = pd.DataFrame({"k": [3, 1, 2]})
    right = pd.DataFrame({"k": [1, 2, 3], "v": [10, 20, 30]})
    out = ordered_left_join(left, right, on="k")
    assert out["v"].tolist() == [30, 10, 20]


def test_row_as_mapping_pandas() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert dict(row_as_mapping(df, 1)) == {"a": 2, "b": 4}


def test_assign_constant_columns_pandas_and_empty() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    assert assign_constant_columns(df, {"c": 9})["c"].tolist() == [9, 9]
    # empty values short-circuits and returns the frame unchanged
    same = assign_constant_columns(df, {})
    assert same is df


def test_drop_columns_pandas() -> None:
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert list(drop_columns(df, ["b", "c"]).columns) == ["a"]


def test_series_to_pylist_pandas_and_fallbacks() -> None:
    # pandas Series has no to_arrow -> tolist branch
    assert series_to_pylist(pd.Series([1, 2, 3])) == [1, 2, 3]

    # object exposing to_pandas but not to_arrow (cuDF-shaped) -> to_pandas branch
    class _ToPandasOnly:
        def to_pandas(self) -> pd.Series:
            return pd.Series([7, 8])

    assert series_to_pylist(_ToPandasOnly()) == [7, 8]

    # to_arrow that raises -> falls through the except to the next branch
    class _ArrowRaises:
        def to_arrow(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        def tolist(self):  # type: ignore[no-untyped-def]
            return [5, 6]

    assert series_to_pylist(_ArrowRaises()) == [5, 6]

    # to_pandas that raises -> falls through its except to the tolist branch
    class _PandasRaises:
        def to_pandas(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        def tolist(self):  # type: ignore[no-untyped-def]
            return [3, 4]

    assert series_to_pylist(_PandasRaises()) == [3, 4]

    # tolist that raises -> falls through its except to the final list() fallback
    class _TolistRaisesIterable:
        def tolist(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([9, 10])

    assert series_to_pylist(_TolistRaisesIterable()) == [9, 10]

    # no arrow/pandas/tolist -> final list() fallback over an iterable
    assert series_to_pylist([1, 2]) == [1, 2]


# --- polars branches (polars lane only) ----------------------------------

@polars_only
def test_is_series_like_polars() -> None:
    assert is_series_like(pl.Series([1, 2])) is True


@polars_only
def test_series_not_null_mask_polars() -> None:
    assert series_not_null_mask(pl.Series([1, None, 3])).to_list() == [True, False, True]


@polars_only
def test_series_filter_polars() -> None:
    out = series_filter(pl.Series([10, 20, 30]), pl.Series([True, False, True]))
    assert out.to_list() == [10, 30]


@polars_only
def test_frame_filter_polars() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    out = frame_filter(df, pl.Series([False, True, True]))
    assert out["a"].to_list() == [2, 3]


@polars_only
def test_ordered_left_join_polars_preserves_order_and_coerces_right() -> None:
    left = pl.DataFrame({"k": [3, 1, 2]})
    # right is pandas -> must be coerced to polars before the join
    right = pd.DataFrame({"k": [1, 2, 3], "v": [10, 20, 30]})
    out = ordered_left_join(left, right, on="k")
    assert out["v"].to_list() == [30, 10, 20]


@polars_only
def test_row_as_mapping_polars() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert dict(row_as_mapping(df, 1)) == {"a": 2, "b": 4}


@polars_only
def test_assign_constant_columns_polars() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    assert assign_constant_columns(df, {"c": 9})["c"].to_list() == [9, 9]


@polars_only
def test_drop_columns_polars() -> None:
    df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert list(drop_columns(df, ["b", "c"]).columns) == ["a"]


@polars_only
def test_series_to_pylist_polars_to_arrow() -> None:
    assert series_to_pylist(pl.Series([1, 2, 3])) == [1, 2, 3]
