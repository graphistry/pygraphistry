"""DataFrame utility functions for same-path execution.

Contains pure functions for series/dataframe operations used across the executor.
"""

from typing import Any, Optional, Sequence, Union

import pandas as pd

from graphistry.compute.typing import DataFrameT, SeriesT, DomainT

SeriesLike = Union[SeriesT, DomainT]


def _is_cudf_obj(obj: object) -> bool:
    return hasattr(obj, "__class__") and obj.__class__.__module__.startswith("cudf")


def _cudf_index_op(left: DomainT, right: DomainT, op: str) -> DomainT:
    method = getattr(left, op)
    try:
        return method(right, sort=False)
    except TypeError:
        return method(right)


def df_cons(template_df: DataFrameT, data: dict) -> DataFrameT:
    if _is_cudf_obj(template_df):
        import cudf  # type: ignore
        return cudf.DataFrame(data)
    return pd.DataFrame(data)


def make_bool_series(template_df: DataFrameT, value: bool) -> SeriesT:
    if _is_cudf_obj(template_df):
        import cudf  # type: ignore
        return cudf.Series([value] * len(template_df))
    return pd.Series(value, index=template_df.index)


def to_pandas_series(series: SeriesLike) -> pd.Series:
    if hasattr(series, "to_pandas"):
        return series.to_pandas()
    if isinstance(series, pd.Series):
        return series
    return pd.Series(series)


def series_values(series: SeriesLike) -> DomainT:
    if _is_cudf_obj(series):
        import cudf  # type: ignore
        if isinstance(series, cudf.Index):
            return series.dropna().unique()
        return cudf.Index(series.dropna().unique())
    if isinstance(series, pd.Index):
        return series.dropna().unique()
    pandas_series = to_pandas_series(series)
    return pd.Index(pandas_series.dropna().unique())


def domain_empty(template: Optional[Any] = None) -> DomainT:
    if _is_cudf_obj(template):
        import cudf  # type: ignore
        return cudf.Index([])
    return pd.Index([])


def domain_is_empty(domain: Optional[DomainT]) -> bool:
    return domain is None or len(domain) == 0


def domain_from_values(values: Any, template: Optional[Any] = None) -> DomainT:
    if domain_is_empty(values):
        return domain_empty(template)
    if _is_cudf_obj(values):
        import cudf  # type: ignore
        if isinstance(values, cudf.Index):
            return values
        return cudf.Index(values)
    if isinstance(values, pd.Index):
        return values
    if _is_cudf_obj(template):
        import cudf  # type: ignore
        return cudf.Index(values)
    return pd.Index(values)


def domain_intersect(left: Optional[DomainT], right: Optional[DomainT]) -> DomainT:
    if left is None or right is None:
        return domain_empty(left if left is not None else right)
    if len(left) == 0 or len(right) == 0:
        return domain_empty(left)
    if isinstance(left, pd.Index):
        return left.intersection(right)
    if _is_cudf_obj(left):
        return _cudf_index_op(left, right, "intersection")
    return left.intersection(right)


def domain_union(left: Optional[DomainT], right: Optional[DomainT]) -> DomainT:
    if left is None or len(left) == 0:
        return right if right is not None else domain_empty(left)
    if right is None or len(right) == 0:
        return left
    if isinstance(left, pd.Index):
        return left.union(right)
    if _is_cudf_obj(left):
        return _cudf_index_op(left, right, "union")
    return left.union(right)


def domain_diff(left: Optional[DomainT], right: Optional[DomainT]) -> DomainT:
    if left is None or len(left) == 0:
        return domain_empty(left)
    if right is None or len(right) == 0:
        return left
    if isinstance(left, pd.Index):
        return left.difference(right)
    if _is_cudf_obj(left):
        return _cudf_index_op(left, right, "difference")
    return left.difference(right)


def domain_to_frame(template_df: DataFrameT, domain: Optional[DomainT], col: str) -> DataFrameT:
    if domain is None:
        return df_cons(template_df, {col: []})
    return df_cons(template_df, {col: domain})


_ID_COL = "__id__"


def series_to_id_df(series: SeriesLike, id_col: str = _ID_COL) -> DataFrameT:
    if hasattr(series, '__class__') and series.__class__.__module__.startswith("cudf"):
        return series.dropna().drop_duplicates().to_frame(name=id_col)

    pandas_series = to_pandas_series(series)
    return pd.DataFrame({id_col: pandas_series.dropna().unique()})


def evaluate_clause(
    series_left: Any, op: str, series_right: Any, *, null_safe: bool = False
) -> Any:
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
    non_empty = [f for f in frames if f is not None and len(f) > 0]
    if not non_empty:
        return None
    if len(non_empty) == 1:
        return non_empty[0]
    first = non_empty[0]
    if first.__class__.__module__.startswith("cudf"):
        import cudf  # type: ignore
        return cudf.concat(non_empty, ignore_index=True)
    return pd.concat(non_empty, ignore_index=True)
