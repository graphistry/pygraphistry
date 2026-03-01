from typing import Any, Optional, Sequence

import pandas as pd

try:  # Optional GPU dependency
    import cudf  # type: ignore
except Exception:  # pragma: no cover
    cudf = None  # type: ignore

from graphistry.compute.typing import DataFrameT, DomainT, SeriesT


def is_cudf_obj(obj: object) -> bool:
    return cudf is not None and obj.__class__.__module__.startswith("cudf")


def df_cons(template_df: DataFrameT, data: dict) -> DataFrameT:
    if is_cudf_obj(template_df):
        return cudf.DataFrame(data)  # type: ignore[call-arg]
    return pd.DataFrame(data)


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


def series_values(series: SeriesT) -> DomainT:
    if is_cudf_obj(series):
        if isinstance(series, cudf.Index):
            return series.dropna().unique()
        return cudf.Index(series.dropna().unique())  # type: ignore[call-arg]
    if isinstance(series, pd.Index):
        return series.dropna().unique()
    if hasattr(series, "to_pandas"):
        pandas_series = series.to_pandas()
    elif isinstance(series, pd.Series):
        pandas_series = series
    else:
        pandas_series = pd.Series(series)
    return pd.Index(pandas_series.dropna().unique())


def domain_empty(template: Optional[Any] = None) -> DomainT:
    if is_cudf_obj(template):
        return cudf.Index([])  # type: ignore[call-arg]
    return pd.Index([])


def domain_is_empty(domain: Optional[DomainT]) -> bool:
    return domain is None or len(domain) == 0


def domain_from_values(values: Any, template: Optional[Any] = None) -> DomainT:
    if domain_is_empty(values):
        return domain_empty(template)
    if is_cudf_obj(values):
        if isinstance(values, cudf.Index):
            return values
        return cudf.Index(values)  # type: ignore[call-arg]
    if isinstance(values, pd.Index):
        return values
    if is_cudf_obj(template):
        return cudf.Index(values)  # type: ignore[call-arg]
    return pd.Index(values)


def _domain_op(left: Optional[DomainT], right: Optional[DomainT], op: str) -> DomainT:
    if left is None or len(left) == 0:
        if op == "union":
            return right if right is not None else domain_empty(left)
        if op == "intersection":
            return domain_empty(left if left is not None else right)
        return domain_empty(left)
    if right is None or len(right) == 0:
        return domain_empty(left) if op == "intersection" else left
    if isinstance(left, pd.Index) or not is_cudf_obj(left):
        return getattr(left, op)(right)
    method = getattr(left, op)
    try:
        return method(right, sort=False)
    except TypeError:
        return method(right)


def domain_intersect(left: Optional[DomainT], right: Optional[DomainT]) -> DomainT:
    return _domain_op(left, right, "intersection")


def domain_union(left: Optional[DomainT], right: Optional[DomainT]) -> DomainT:
    return _domain_op(left, right, "union")


def domain_diff(left: Optional[DomainT], right: Optional[DomainT]) -> DomainT:
    return _domain_op(left, right, "difference")


def domain_union_all(domains: Sequence[Optional[DomainT]]) -> Optional[DomainT]:
    out = None
    for domain in domains:
        if not domain_is_empty(domain):
            out = domain if out is None else domain_union(out, domain)
    return out


def domain_to_frame(template_df: DataFrameT, domain: Optional[DomainT], col: str) -> DataFrameT:
    if domain is None:
        return df_cons(template_df, {col: []})
    return df_cons(template_df, {col: domain})
