import operator
from typing import Any

from graphistry.compute.typing import DataFrameT
from graphistry.compute.dataframe import (
    ineq_eval_pairs,
    project_node_attrs,
    semijoin_eval_pairs,
)
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
