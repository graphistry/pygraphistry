import operator
import os
from typing import Any, Optional, Sequence, Tuple, Union

import pandas as pd

try:  # Optional GPU dependency
    import cudf  # type: ignore
except Exception:  # pragma: no cover
    cudf = None  # type: ignore

from graphistry.compute.typing import DataFrameT, DomainT, SeriesT

_BOOL_TRUE = {"1", "true", "yes", "on"}
_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}
OP_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}


def env_lower(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip().lower()


def env_flag(name: str, default: bool = False) -> bool:
    raw = env_lower(name)
    return default if not raw else raw in _BOOL_TRUE


def env_optional_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def normalize_limit(value: Optional[float], default: Optional[float]) -> Optional[float]:
    value = default if value is None else value
    return None if value is not None and value <= 0 else value


def _is_cudf_obj(obj: object) -> bool:
    return cudf is not None and obj.__class__.__module__.startswith("cudf")


def df_cons(template_df: DataFrameT, data: dict) -> DataFrameT:
    if _is_cudf_obj(template_df):
        return cudf.DataFrame(data)  # type: ignore[call-arg]
    return pd.DataFrame(data)


def series_values(series: Union[SeriesT, DomainT]) -> DomainT:
    if _is_cudf_obj(series):
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
    if _is_cudf_obj(template):
        return cudf.Index([])  # type: ignore[call-arg]
    return pd.Index([])


def domain_is_empty(domain: Optional[DomainT]) -> bool:
    return domain is None or len(domain) == 0


def domain_from_values(values: Any, template: Optional[Any] = None) -> DomainT:
    if domain_is_empty(values):
        return domain_empty(template)
    if _is_cudf_obj(values):
        if isinstance(values, cudf.Index):
            return values
        return cudf.Index(values)  # type: ignore[call-arg]
    if isinstance(values, pd.Index):
        return values
    if _is_cudf_obj(template):
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
    if isinstance(left, pd.Index) or not _is_cudf_obj(left):
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


def project_node_attrs(frame: DataFrameT, node_col: str, cols: Sequence[str], *, id_label: str, prefix: str = "", labels: Optional[Sequence[str]] = None, node_domain: Optional[DomainT] = None, dedupe: bool = False, drop_nulls: bool = False) -> DataFrameT:
    df = frame[frame[node_col].isin(node_domain)] if node_domain is not None else frame
    data_cols = [node_col] + [col for col in cols if col != node_col]
    rename_map = {node_col: id_label}
    if labels is not None:
        rename_map.update({col: label for col, label in zip(cols, labels) if col != node_col})
    else:
        rename_map.update({col: f"{prefix}{col}" for col in cols if col != node_col})
    df = df[data_cols].rename(columns=rename_map)
    if labels is not None and node_col in cols:
        df[labels[cols.index(node_col)]] = df[id_label]
    if drop_nulls and labels is not None:
        df = df[df[list(labels)].notna().all(axis=1)]
    return df.drop_duplicates() if dedupe else df


def evaluate_clause(series_left: Any, op: str, series_right: Any, *, null_safe: bool = False) -> Any:
    fn = _OPS.get(op)
    if fn is None:
        if null_safe:
            return (series_left.notna() & series_right.notna()) & False
        return False
    if not null_safe:
        return fn(series_left, series_right)
    valid = series_left.notna() & series_right.notna()
    return valid & fn(series_left, series_right)


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


def ineq_eval_pairs(left_pairs: DataFrameT, right_pairs: DataFrameT, op: str, *, group_cols: Optional[Sequence[str]] = None, left_value: str = "__value__", right_value: str = "__value__") -> Tuple[DataFrameT, DataFrameT]:
    group_cols = list(group_cols) if group_cols is not None else ["__mid__"]
    if op in {"<", "<="}:
        right_bound = right_pairs.groupby(group_cols)[right_value].max().reset_index().rename(columns={right_value: "__right_bound__"})
        left_bound = left_pairs.groupby(group_cols)[left_value].min().reset_index().rename(columns={left_value: "__left_bound__"})
        left_eval = left_pairs.merge(right_bound, on=group_cols, how="inner")
        right_eval = right_pairs.merge(left_bound, on=group_cols, how="inner")
        left_cmp = operator.lt if op == "<" else operator.le
        right_cmp = operator.gt if op == "<" else operator.ge
    else:
        right_bound = right_pairs.groupby(group_cols)[right_value].min().reset_index().rename(columns={right_value: "__right_bound__"})
        left_bound = left_pairs.groupby(group_cols)[left_value].max().reset_index().rename(columns={left_value: "__left_bound__"})
        left_eval = left_pairs.merge(right_bound, on=group_cols, how="inner")
        right_eval = right_pairs.merge(left_bound, on=group_cols, how="inner")
        left_cmp = operator.gt if op == ">" else operator.ge
        right_cmp = operator.lt if op == ">" else operator.le
    left_eval = left_eval[left_cmp(left_eval[left_value], left_eval["__right_bound__"])]
    right_eval = right_eval[right_cmp(right_eval[right_value], right_eval["__left_bound__"])]
    return left_eval, right_eval


def semijoin_eval_pairs(left_pairs: DataFrameT, right_pairs: DataFrameT, op: str, *, left_value: str = "__value__", right_value: str = "__value__", left_unique_col: str = "__left_unique__", right_unique_col: str = "__right_unique__", left_only_col: str = "__left_only__", right_only_col: str = "__right_only__", left_keep: Optional[Sequence[str]] = None, right_keep: Optional[Sequence[str]] = None) -> Tuple[Optional[DataFrameT], Optional[DataFrameT], Optional[DataFrameT]]:
    mid_values = None
    if op == "==":
        left_mid = left_pairs[["__mid__", left_value]].drop_duplicates().rename(columns={left_value: "__value__"})
        right_mid = right_pairs[["__mid__", right_value]].drop_duplicates().rename(columns={right_value: "__value__"})
        mid_values = left_mid.merge(right_mid, on=["__mid__", "__value__"], how="inner")
        if len(mid_values) == 0:
            return None, None, mid_values
        left_eval = left_pairs.merge(mid_values.rename(columns={"__value__": left_value}), on=["__mid__", left_value], how="inner")
        right_eval = right_pairs.merge(mid_values.rename(columns={"__value__": right_value}), on=["__mid__", right_value], how="inner")
    elif op == "!=":
        def _uniq(df: DataFrameT, col: str, name: str) -> DataFrameT: return df[["__mid__", col]].drop_duplicates().groupby("__mid__").size().reset_index(name=name)

        left_unique = _uniq(left_pairs, left_value, left_unique_col)
        right_unique = _uniq(right_pairs, right_value, right_unique_col)
        right_only = right_pairs[["__mid__", right_value]].drop_duplicates().merge(right_unique[right_unique[right_unique_col] == 1], on="__mid__", how="inner")[["__mid__", right_value]].rename(columns={right_value: right_only_col})
        left_only = left_pairs[["__mid__", left_value]].drop_duplicates().merge(left_unique[left_unique[left_unique_col] == 1], on="__mid__", how="inner")[["__mid__", left_value]].rename(columns={left_value: left_only_col})
        left_eval = left_pairs.merge(right_unique, on="__mid__", how="inner").merge(right_only, on="__mid__", how="left")
        left_eval = left_eval[(left_eval[right_unique_col] > 1) | left_eval[right_only_col].isna() | (left_eval[right_only_col] != left_eval[left_value])]
        right_eval = right_pairs.merge(left_unique, on="__mid__", how="inner").merge(left_only, on="__mid__", how="left")
        right_eval = right_eval[(right_eval[left_unique_col] > 1) | right_eval[left_only_col].isna() | (right_eval[left_only_col] != right_eval[right_value])]
    else:
        left_eval, right_eval = ineq_eval_pairs(left_pairs, right_pairs, op, left_value=left_value, right_value=right_value)
    if left_eval is None or right_eval is None:
        return None, None, mid_values
    if left_keep:
        left_eval = left_eval[list(left_keep)]
    if right_keep:
        right_eval = right_eval[list(right_keep)]
    return left_eval, right_eval, mid_values
