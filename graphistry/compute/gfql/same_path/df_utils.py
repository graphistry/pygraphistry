"""DataFrame utility functions for same-path execution.

Contains pure functions for series/dataframe operations used across the executor.
"""

from typing import Any, Optional, Sequence, Set

import pandas as pd

from graphistry.compute.typing import DataFrameT


def to_pandas_series(series: Any) -> pd.Series:
    """Convert any series-like object to pandas Series."""
    if hasattr(series, "to_pandas"):
        return series.to_pandas()
    if isinstance(series, pd.Series):
        return series
    return pd.Series(series)


def series_values(series: Any) -> Set[Any]:
    """Extract unique non-null values from a series as a set."""
    pandas_series = to_pandas_series(series)
    return set(pandas_series.dropna().unique().tolist())


def common_values(series_a: Any, series_b: Any) -> Set[Any]:
    """Return intersection of unique values from two series."""
    vals_a = series_values(series_a)
    vals_b = series_values(series_b)
    return vals_a & vals_b


def safe_min(series: Any) -> Optional[Any]:
    """Return minimum value of series, or None if empty/all-null."""
    pandas_series = to_pandas_series(series).dropna()
    if pandas_series.empty:
        return None
    value = pandas_series.min()
    if pd.isna(value):
        return None
    return value


def safe_max(series: Any) -> Optional[Any]:
    """Return maximum value of series, or None if empty/all-null."""
    pandas_series = to_pandas_series(series).dropna()
    if pandas_series.empty:
        return None
    value = pandas_series.max()
    if pd.isna(value):
        return None
    return value


def filter_by_values(
    frame: DataFrameT, column: str, values: Set[Any]
) -> DataFrameT:
    """Filter dataframe to rows where column value is in the given set."""
    if not values:
        return frame.iloc[0:0]
    allowed = list(values)
    mask = frame[column].isin(allowed)
    return frame[mask]


def evaluate_clause(series_left: Any, op: str, series_right: Any) -> Any:
    """Evaluate comparison clause between two series.

    Args:
        series_left: Left operand series
        op: Comparison operator ('==', '!=', '>', '>=', '<', '<=')
        series_right: Right operand series

    Returns:
        Boolean series with comparison result
    """
    if op == "==":
        return series_left == series_right
    if op == "!=":
        return series_left != series_right
    if op == ">":
        return series_left > series_right
    if op == ">=":
        return series_left >= series_right
    if op == "<":
        return series_left < series_right
    if op == "<=":
        return series_left <= series_right
    return False


def concat_frames(frames: Sequence[DataFrameT]) -> Optional[DataFrameT]:
    """Concatenate frames, returning None if empty.

    Handles both pandas and cudf DataFrames automatically.
    """
    non_empty = [f for f in frames if f is not None and len(f) > 0]
    if not non_empty:
        return None
    if len(non_empty) == 1:
        return non_empty[0]
    # Check if cudf
    first = non_empty[0]
    if first.__class__.__module__.startswith("cudf"):
        import cudf  # type: ignore
        return cudf.concat(non_empty, ignore_index=True)
    return pd.concat(non_empty, ignore_index=True)
