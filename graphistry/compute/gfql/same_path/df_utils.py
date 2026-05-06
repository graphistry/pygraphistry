import operator
from typing import Any, Optional, Sequence, Tuple

from graphistry.compute.typing import DataFrameT, DomainT
from graphistry.compute.gfql.same_path_types import OP_FLIP, SUPPORTED_WHERE_OPS
from graphistry.compute.dataframe_utils import (
    concat_frames,
    df_cons,
    domain_diff,
    domain_empty,
    domain_from_values,
    domain_intersect,
    domain_is_empty,
    domain_to_frame,
    domain_union,
    domain_union_all,
    series_values,
)

_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


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
        def _uniq(df: DataFrameT, col: str, name: str) -> DataFrameT:
            counts = (
                df[["__mid__", col]]
                .drop_duplicates()
                .groupby("__mid__")
                .size()
                .reset_index()
            )
            return counts.rename(columns={0: name, "size": name})

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
