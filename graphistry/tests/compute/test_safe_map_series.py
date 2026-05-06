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


# ---------------------------------------------------------------------------
# New edge-case tests (pandas path)
# ---------------------------------------------------------------------------


def test_safe_map_series_pandas_non_default_index() -> None:
    """Non-default index on source series is preserved in the result."""
    s = pd.Series(["a", "b"], index=[10, 20])
    result = safe_map_series(s, {"a": 1, "b": 2})
    assert result.index.tolist() == [10, 20]
    assert result.tolist() == [1, 2]


def test_safe_map_series_pandas_empty_mapping() -> None:
    """Empty mapping dict — all keys miss, result is all NaN."""
    s = pd.Series(["a", "b", "c"])
    result = safe_map_series(s, {})
    assert len(result) == 3
    assert result.isna().all()


def test_safe_map_series_pandas_nan_value_in_series() -> None:
    """Series containing None/NaN as a value (not a key) maps to NaN for that position."""
    s = pd.Series(["a", None, "b"])
    result = safe_map_series(s, {"a": 1, "b": 2})
    assert result.iloc[0] == 1
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 2


def test_safe_map_series_pandas_mixed_value_types() -> None:
    """Mapping values of different types produces an object-dtype result with correct values."""
    s = pd.Series(["a", "b", "c", "d"])
    mapping = {"a": 1, "b": "two", "c": None, "d": 3.14}
    result = safe_map_series(s, mapping)
    assert result.iloc[0] == 1
    assert result.iloc[1] == "two"
    assert pd.isna(result.iloc[2])
    assert result.iloc[3] == 3.14
    assert result.dtype == object


def test_safe_map_series_pandas_series_mapping_non_default_index() -> None:
    """Series-as-mapping with non-default index (set_index pattern from hop.py)."""
    hop_records = pd.DataFrame({"id": ["x", "y", "z"], "hop_num": [0, 1, 2]})
    mapping = hop_records.set_index("id")["hop_num"]
    s = pd.Series(["x", "z", "y"])
    result = safe_map_series(s, mapping)
    assert result.tolist() == [0, 2, 1]


# ---------------------------------------------------------------------------
# New edge-case tests (cudf path — skipped when cudf is not available)
# ---------------------------------------------------------------------------


def test_safe_map_series_cudf_non_default_index() -> None:
    """cudf: non-default index on source series is preserved in the result."""
    cudf = pytest.importorskip("cudf")
    s = cudf.Series(["a", "b"], index=[10, 20])
    result = safe_map_series(s, {"a": 1, "b": 2})
    assert result.index.to_pandas().tolist() == [10, 20]
    assert result.to_pandas().tolist() == [1, 2]


def test_safe_map_series_cudf_empty_mapping() -> None:
    """cudf: empty mapping dict — all keys miss, result is all null."""
    cudf = pytest.importorskip("cudf")
    s = cudf.Series(["a", "b", "c"])
    result = safe_map_series(s, {})
    vals = result.to_pandas()
    assert len(vals) == 3
    assert vals.isna().all()


def test_safe_map_series_cudf_pd_series_mapping() -> None:
    """cudf Series mapped through a pandas Series (set_index pattern from hop.py).

    Exercises Engine.py:399-400 — the branch triggered when a cudf source series
    is mapped through a pandas Series (e.g. hop_map = df.set_index(node_col)[hop_col]).
    """
    cudf = pytest.importorskip("cudf")
    s = cudf.Series(["x", "z", "y", "x"])
    mapping = pd.Series([0, 1, 2], index=["x", "y", "z"])  # pandas, non-default index
    result = safe_map_series(s, mapping)
    vals = result.to_pandas()
    assert vals.tolist() == [0, 2, 1, 0]
