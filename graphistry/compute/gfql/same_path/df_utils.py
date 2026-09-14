import operator
from typing import Any, Tuple

from graphistry.compute.typing import DataFrameT, SeriesT
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


def _align_mixed_tz_datetimes(series_left: SeriesT, series_right: SeriesT) -> Tuple[SeriesT, SeriesT]:
    """Normalize a tz-aware/tz-naive datetime pair onto UTC-naive: GFQL reads naive
    datetimes as UTC (as the row pipeline's ``_native_epoch_ticks`` does), and the
    raw pandas compare of the mixed pair raises."""
    left_dtype = getattr(series_left, "dtype", None)
    right_dtype = getattr(series_right, "dtype", None)
    if getattr(left_dtype, "kind", None) != "M" or getattr(right_dtype, "kind", None) != "M":
        return series_left, series_right
    left_tz = getattr(left_dtype, "tz", None)
    right_tz = getattr(right_dtype, "tz", None)
    if (left_tz is None) == (right_tz is None):
        return series_left, series_right
    try:
        if left_tz is not None:
            return series_left.dt.tz_convert("UTC").dt.tz_localize(None), series_right
        return series_left, series_right.dt.tz_convert("UTC").dt.tz_localize(None)
    except (AttributeError, TypeError):  # pragma: no cover - engine without tz_convert
        return series_left, series_right


def evaluate_clause(series_left: Any, op: str, series_right: Any, *, null_safe: bool = False) -> Any:
    fn = _OPS.get(op)
    if fn is None:
        if null_safe:
            return (series_left.notna() & series_right.notna()) & False
        return False
    series_left, series_right = _align_mixed_tz_datetimes(series_left, series_right)
    if not null_safe:
        return fn(series_left, series_right)
    valid = series_left.notna() & series_right.notna()
    return valid & fn(series_left, series_right)
