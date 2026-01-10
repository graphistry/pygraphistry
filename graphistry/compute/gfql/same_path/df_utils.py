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


def evaluate_clause(
    series_left: Any, op: str, series_right: Any, *, null_safe: bool = False
) -> Any:
    """Evaluate comparison clause between two series.

    Args:
        series_left: Left operand series
        op: Comparison operator ('==', '!=', '>', '>=', '<', '<=')
        series_right: Right operand series
        null_safe: If True, use SQL NULL semantics where NULL comparisons return False

    Returns:
        Boolean series with comparison result
    """
    if null_safe:
        # SQL NULL semantics: any comparison with NULL is NULL (treated as False)
        # pandas != returns True for X != NaN, so we need to check for NULL first
        valid = series_left.notna() & series_right.notna()
        if op == "==":
            return valid & (series_left == series_right)
        if op == "!=":
            return valid & (series_left != series_right)
        if op == ">":
            return valid & (series_left > series_right)
        if op == ">=":
            return valid & (series_left >= series_right)
        if op == "<":
            return valid & (series_left < series_right)
        if op == "<=":
            return valid & (series_left <= series_right)
        return valid & False
    else:
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
