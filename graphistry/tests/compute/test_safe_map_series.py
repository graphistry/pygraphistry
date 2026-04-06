"""
Tests for Engine.safe_map_series — cudf-safe Series.map(dict) bridge (#977).
"""
import pytest
import pandas as pd
from graphistry.Engine import safe_map_series


def test_safe_map_series_pandas_dict() -> None:
    """pandas Series + dict mapping — native path."""
    s = pd.Series(["a", "b", "a", "c"])
    result = safe_map_series(s, {"a": 1, "b": 2, "c": 3})
    assert result.tolist() == [1, 2, 1, 3]


def test_safe_map_series_pandas_series_index() -> None:
    """pandas Series + Series-as-mapping (set_index result)."""
    s = pd.Series(["a", "b", "c"])
    mapping = pd.Series([10, 20, 30], index=["a", "b", "c"])
    result = safe_map_series(s, mapping)
    assert result.tolist() == [10, 20, 30]


def test_safe_map_series_pandas_missing_keys() -> None:
    """Missing keys produce NaN (same as native pandas .map)."""
    s = pd.Series(["a", "x"])
    result = safe_map_series(s, {"a": 1})
    assert result.iloc[0] == 1
    assert pd.isna(result.iloc[1])


def test_safe_map_series_pandas_empty() -> None:
    """Empty series maps to empty series."""
    s = pd.Series([], dtype=object)
    result = safe_map_series(s, {"a": 1})
    assert len(result) == 0


def test_safe_map_series_pandas_regression_no_sigsegv() -> None:
    """Regression: pandas path must not SIGSEGV (trivially true but documents intent)."""
    s = pd.Series(["a", "b", "a"])
    result = safe_map_series(s, {"a": True, "b": False})
    assert result.tolist() == [True, False, True]


def test_safe_map_series_cudf_dict() -> None:
    """cudf Series + dict mapping — bridges through pandas, no SIGSEGV."""
    cudf = pytest.importorskip("cudf")
    s = cudf.Series(["a", "b", "a", "c"])
    result = safe_map_series(s, {"a": 1, "b": 2, "c": 3})
    assert result.to_pandas().tolist() == [1, 2, 1, 3]


def test_safe_map_series_cudf_series_mapping() -> None:
    """cudf Series + cudf Series-as-mapping."""
    cudf = pytest.importorskip("cudf")
    s = cudf.Series(["a", "b", "c"])
    mapping = cudf.Series([10, 20, 30], index=["a", "b", "c"])
    result = safe_map_series(s, mapping)
    assert result.to_pandas().tolist() == [10, 20, 30]


def test_safe_map_series_cudf_missing_keys() -> None:
    """cudf: missing keys produce null."""
    cudf = pytest.importorskip("cudf")
    s = cudf.Series(["a", "x"])
    result = safe_map_series(s, {"a": 1})
    vals = result.to_pandas()
    assert vals.iloc[0] == 1
    assert pd.isna(vals.iloc[1])
