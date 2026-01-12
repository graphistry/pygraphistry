"""DataFrame utility functions for same-path execution.

Contains pure functions for series/dataframe operations used across the executor.
"""

from typing import Any, Optional, Sequence, Set

import pandas as pd

from graphistry.compute.typing import DataFrameT


def df_cons(template_df: DataFrameT, data: dict) -> DataFrameT:
    """Construct a DataFrame of the same type as template_df.

    Args:
        template_df: DataFrame to use as type template (pandas or cudf)
        data: Dictionary of column data for new DataFrame

    Returns:
        New DataFrame of same type as template_df
    """
    if template_df.__class__.__module__.startswith("cudf"):
        import cudf  # type: ignore
        return cudf.DataFrame(data)
    return pd.DataFrame(data)


def make_bool_series(template_df: DataFrameT, value: bool) -> Any:
    """Create a boolean Series matching template_df's type and length.

    Args:
        template_df: DataFrame to use as type template
        value: Boolean value to fill series with

    Returns:
        Boolean series of same type and length as template_df
    """
    if template_df.__class__.__module__.startswith("cudf"):
        import cudf  # type: ignore
        return cudf.Series([value] * len(template_df))
    return pd.Series(value, index=template_df.index)


def to_pandas_series(series: Any) -> pd.Series:
    """Convert any series-like object to pandas Series."""
    if hasattr(series, "to_pandas"):
        return series.to_pandas()
    if isinstance(series, pd.Series):
        return series
    return pd.Series(series)


def series_unique(series: Any) -> Any:
    """Extract unique non-null values from a series as an array.

    Returns a numpy array (or cudf array) that can be passed directly to .isin().
    This is ~2x faster than series_values() because it avoids Python set construction.

    For set operations (intersection, union), use series_values() instead.
    """
    if hasattr(series, 'dropna'):
        return series.dropna().unique()
    pandas_series = to_pandas_series(series)
    return pandas_series.dropna().unique()


def series_values(series: Any) -> pd.Index:
    """Extract unique non-null values from a series as a pd.Index.

    Returns pd.Index which supports:
    - .intersection() for & operations
    - .union() for | operations
    - Direct use in .isin() (no conversion needed)

    This is ~9x faster than the previous set-based approach.
    """
    pandas_series = to_pandas_series(series)
    return pd.Index(pandas_series.dropna().unique())


# Standard column name for ID DataFrames used in semi-joins
_ID_COL = "__id__"


def series_to_id_df(series: Any, id_col: str = _ID_COL) -> DataFrameT:
    """Extract unique non-null values from a series as a single-column DataFrame.

    This is the DF-based alternative to series_values() for use with merge-based
    semi-joins instead of .isin() filtering.

    Args:
        series: Series to extract unique values from
        id_col: Column name for the output DataFrame

    Returns:
        Single-column DataFrame with unique values (same type as input series)
    """
    # Handle cuDF
    if hasattr(series, '__class__') and series.__class__.__module__.startswith("cudf"):
        return series.dropna().drop_duplicates().to_frame(name=id_col)

    # Handle pandas
    pandas_series = to_pandas_series(series)
    return pd.DataFrame({id_col: pandas_series.dropna().unique()})


def semi_join_filter(
    df: DataFrameT,
    allowed_df: DataFrameT,
    df_col: str,
    allowed_col: str = _ID_COL,
) -> DataFrameT:
    """Filter df to rows where df[df_col] is in allowed_df[allowed_col].

    This is the DF-based alternative to df[df[col].isin(set)] for vectorized
    semi-join filtering.

    Args:
        df: DataFrame to filter
        allowed_df: DataFrame containing allowed values
        df_col: Column in df to filter on
        allowed_col: Column in allowed_df containing allowed values

    Returns:
        Filtered DataFrame (same type as input)
    """
    if allowed_df is None or len(allowed_df) == 0:
        return df

    # Rename allowed column to match df column for merge
    if allowed_col != df_col:
        allowed_df = allowed_df.rename(columns={allowed_col: df_col})

    # Semi-join: inner merge keeps only matching rows
    return df.merge(allowed_df[[df_col]], on=df_col, how="inner")


def union_id_dfs(df1: Optional[DataFrameT], df2: DataFrameT, id_col: str = _ID_COL) -> DataFrameT:
    """Union two ID DataFrames, returning unique values.

    Args:
        df1: First DataFrame (can be None)
        df2: Second DataFrame
        id_col: Column name containing IDs

    Returns:
        DataFrame with union of unique IDs
    """
    if df1 is None or len(df1) == 0:
        return df2[[id_col]].drop_duplicates() if id_col in df2.columns else df2.drop_duplicates()

    # Handle cuDF
    if hasattr(df1, '__class__') and df1.__class__.__module__.startswith("cudf"):
        import cudf  # type: ignore
        return cudf.concat([df1, df2]).drop_duplicates(subset=[id_col])

    return pd.concat([df1, df2]).drop_duplicates(subset=[id_col])


def intersect_id_dfs(
    df1: Optional[DataFrameT],
    df2: DataFrameT,
    id_col: str = _ID_COL,
) -> DataFrameT:
    """Intersect two ID DataFrames.

    Args:
        df1: First DataFrame (if None, returns df2)
        df2: Second DataFrame
        id_col: Column name containing IDs

    Returns:
        DataFrame with intersection of IDs
    """
    if df1 is None or len(df1) == 0:
        return df2[[id_col]].drop_duplicates() if id_col in df2.columns else df2.drop_duplicates()

    return df1.merge(df2[[id_col]], on=id_col, how="inner")


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
