"""Vectorized row ORDER BY helpers for GFQL row-pipeline execution."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from graphistry.compute.gfql.temporal_text import (
    DATETIME_CALL_TEXT_RE,
    DATE_CALL_TEXT_RE,
    LOCALDATETIME_CALL_TEXT_RE,
    LOCALTIME_CALL_TEXT_RE,
    TIME_CALL_TEXT_RE,
)


_GFQL_LIST_NUMERIC_TEXT_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_GFQL_TIME_TEXT_RE = re.compile(
    r"^(?P<h>\d{2}):(?P<m>\d{2})"
    r"(?::(?P<s>\d{2})(?:\.(?P<f>\d{1,9}))?)?"
    r"(?:(?P<off_sign>[+-])(?P<off_h>\d{2}):(?P<off_m>\d{2}))?$"
)
_GFQL_DATE_TEXT_RE = re.compile(r"^(?P<y>-?\d{4,9})-(?P<mo>\d{2})-(?P<d>\d{2})$")
_GFQL_DATETIME_TEXT_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})T"
    r"(?P<h>\d{2}):(?P<m>\d{2})"
    r"(?::(?P<s>\d{2})(?:\.(?P<f>\d{1,9}))?)?"
    r"(?:(?P<off_sign>[+-])(?P<off_h>\d{2}):(?P<off_m>\d{2}))?$"
)

NullMaskFn = Callable[[Any, Any], Any]
BroadcastScalarFn = Callable[[Any, Any], Any]
FreshColNameFn = Callable[[Any, str], str]


def is_null_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        marker = pd.isna(value)
    except Exception:
        return False
    return bool(marker) if isinstance(marker, bool) else False


def is_nan_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isnan(value)
    except Exception:
        return False


def order_value_family(value: Any) -> Optional[str]:
    if is_null_scalar(value) or is_nan_scalar(value):
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time, pd.Timestamp)):
        return "datetime"
    type_name = type(value).__name__.lower()
    if "datetime64" in type_name or "timedelta64" in type_name:
        return "datetime"
    if isinstance(value, (list, tuple, dict, set)):
        return "unsupported"
    return "unsupported"


def validate_order_series_vector_safe(series: Any, expr: str) -> None:
    dtype_txt = str(getattr(series, "dtype", "")).lower()
    if dtype_txt != "object":
        return
    non_null = series.dropna()
    sample = non_null.head(128)
    if hasattr(sample, "to_pandas"):
        sample = sample.to_pandas()
    values = sample.tolist() if hasattr(sample, "tolist") else list(sample)
    families = {fam for fam in (order_value_family(v) for v in values) if fam is not None}
    if len(families) == 0:
        return
    if "unsupported" in families or len(families) > 1:
        fams = ", ".join(sorted(families))
        raise ValueError(
            "unsupported order_by expression for vectorized execution; "
            f"mixed/dynamic value families ({fams}) in {expr!r}"
        )


def order_sample_values(series: Any) -> List[Any]:
    sample = series.dropna().head(128)
    if hasattr(sample, "to_pandas"):
        sample = sample.to_pandas()
    if hasattr(sample, "tolist"):
        return list(sample.tolist())
    return list(sample)


def order_detect_list_series(series: Any) -> bool:
    sample_values = order_sample_values(series)
    return len(sample_values) > 0 and all(isinstance(v, (list, tuple)) for v in sample_values)


def order_detect_temporal_mode(series: Any) -> Optional[str]:
    if not hasattr(series, "dropna"):
        return None
    non_null = series.dropna()
    if len(non_null) == 0 or not hasattr(non_null, "astype"):
        return None
    text = non_null.astype(str)
    if not hasattr(text, "str"):
        return None
    if bool(text.str.fullmatch(_GFQL_DATE_TEXT_RE.pattern, na=False).all()):
        return "date"
    if bool(text.str.fullmatch(_GFQL_DATETIME_TEXT_RE.pattern, na=False).all()):
        return "datetime"
    if bool(text.str.fullmatch(_GFQL_TIME_TEXT_RE.pattern, na=False).all()):
        return "time"
    if bool(text.str.fullmatch(DATE_CALL_TEXT_RE.pattern, na=False).all()):
        return "date_constructor"
    datetime_local = text.str.fullmatch(LOCALDATETIME_CALL_TEXT_RE.pattern, na=False)
    datetime_tz = text.str.fullmatch(DATETIME_CALL_TEXT_RE.pattern, na=False)
    if bool((datetime_local | datetime_tz).all()):
        return "datetime_constructor"
    time_local = text.str.fullmatch(LOCALTIME_CALL_TEXT_RE.pattern, na=False)
    time_tz = text.str.fullmatch(TIME_CALL_TEXT_RE.pattern, na=False)
    if bool((time_local | time_tz).all()):
        return "time_constructor"
    return None


def build_list_sort_columns(
    work_df: Any,
    sort_col: str,
    key_prefix: str,
    *,
    null_mask_fn: NullMaskFn,
    broadcast_scalar_fn: BroadcastScalarFn,
    fresh_col_name_fn: FreshColNameFn,
) -> Tuple[Any, List[str]]:
    row_col = fresh_col_name_fn(work_df.columns, f"{key_prefix}__row")
    list_col = fresh_col_name_fn(work_df.columns, f"{key_prefix}__list")
    len_col = fresh_col_name_fn(work_df.columns, f"{key_prefix}__len")
    pos_col = fresh_col_name_fn(work_df.columns, f"{key_prefix}__pos")
    tok_col = fresh_col_name_fn(work_df.columns, f"{key_prefix}__tok")

    base = work_df.assign(**{row_col: range(len(work_df)), list_col: work_df[sort_col]})[[row_col, list_col]]
    if not hasattr(base[list_col], "str") or not hasattr(base[list_col].str, "len"):
        raise ValueError("order_by list sorting requires string/list accessor support")
    lengths = base[list_col].str.len()
    base = base.assign(**{len_col: lengths})
    expanded = base[[row_col, list_col, len_col]].explode(list_col)
    if len(expanded) > 0:
        expanded = expanded.assign(**{pos_col: expanded.groupby(row_col, sort=False).cumcount()})
        keep = null_mask_fn(expanded, expanded[len_col]) | (expanded[pos_col] < expanded[len_col])
        expanded = expanded.loc[keep]

    if len(expanded) == 0:
        key_cols = [f"{key_prefix}_0"]
        key_frame = base[[row_col]].assign(**{key_cols[0]: ""})
    else:
        value = expanded[list_col]
        value_str = value.astype(str)
        null_mask = null_mask_fn(expanded, value)
        lower_str = value_str.str.lower() if hasattr(value_str, "str") else value_str
        bool_mask = (~null_mask) & lower_str.isin(["true", "false"])
        num_mask = (~null_mask) & (~bool_mask) & value_str.str.fullmatch(_GFQL_LIST_NUMERIC_TEXT_RE.pattern)
        str_mask = (~null_mask) & (~bool_mask) & (~num_mask)

        num_values = value_str.where(num_mask, None).astype("float64")
        num_rank = num_values.rank(method="dense")
        str_values = value_str.where(str_mask, None)
        str_rank = str_values.rank(method="dense")

        token = broadcast_scalar_fn(expanded, "9:000000000000")
        if hasattr(str_mask, "any") and bool(str_mask.any()):
            str_token = "5:" + str_rank.fillna(0).astype("int64").astype(str).str.zfill(12)
            token = token.where(~str_mask, str_token)
        if hasattr(num_mask, "any") and bool(num_mask.any()):
            num_token = "7:" + num_rank.fillna(0).astype("int64").astype(str).str.zfill(12)
            token = token.where(~num_mask, num_token)
        if hasattr(bool_mask, "any") and bool(bool_mask.any()):
            bool_token = "6:" + lower_str.where(bool_mask, "false").replace({"false": "0", "true": "1"})
            token = token.where(~bool_mask, bool_token)

        expanded = expanded.assign(**{tok_col: token})
        key_wide = expanded.pivot(index=row_col, columns=pos_col, values=tok_col).sort_index(axis=1)
        key_wide = key_wide.reset_index()
        rename_map: Dict[Any, str] = {}
        for col in key_wide.columns:
            if col == row_col:
                continue
            rename_map[col] = f"{key_prefix}_{int(col)}"
        key_wide = key_wide.rename(columns=rename_map)
        key_cols = [col for col in key_wide.columns if col != row_col]
        key_frame = base[[row_col]].merge(key_wide, on=row_col, how="left", sort=False)
        for col in key_cols:
            key_frame[col] = key_frame[col].fillna("")

    merged = work_df.assign(**{row_col: range(len(work_df))}).merge(
        key_frame[[row_col] + key_cols],
        on=row_col,
        how="left",
        sort=False,
    )
    merged = merged.drop(columns=[row_col])
    return merged, key_cols


def build_temporal_sort_columns(
    work_df: Any,
    sort_col: str,
    key_prefix: str,
    mode: str,
    *,
    month_shift: int = 0,
    nanosecond_shift: int = 0,
    null_mask_fn: NullMaskFn,
    fresh_col_name_fn: FreshColNameFn,
) -> Tuple[Any, List[str]]:
    value = work_df[sort_col]
    text = value.astype(str)
    null_mask = null_mask_fn(work_df, value)
    if mode == "date":
        parts = text.str.extract(_GFQL_DATE_TEXT_RE)
        year = parts["y"].fillna("0").astype("int64")
        month = parts["mo"].fillna("1").astype("int64")
        day = parts["d"].fillna("1").astype("int64")
        hour = minute = second = nanos = off_hours = off_minutes = None
        off_sign = None
    elif mode == "datetime":
        parts = text.str.extract(_GFQL_DATETIME_TEXT_RE)
        year = parts["y"].fillna("0").astype("int64")
        month = parts["mo"].fillna("1").astype("int64")
        day = parts["d"].fillna("1").astype("int64")
        hour = parts["h"].fillna("0").astype("int64")
        minute = parts["m"].fillna("0").astype("int64")
        second = parts["s"].fillna("0").astype("int64")
        frac = parts["f"].fillna("").str.pad(9, side="right", fillchar="0").replace("", "0")
        nanos = frac.astype("int64")
        off_sign = parts["off_sign"].fillna("+")
        off_hours = parts["off_h"].fillna("0").astype("int64")
        off_minutes = parts["off_m"].fillna("0").astype("int64")
    elif mode == "time":
        parts = text.str.extract(_GFQL_TIME_TEXT_RE)
        year = month = day = None
        hour = parts["h"].fillna("0").astype("int64")
        minute = parts["m"].fillna("0").astype("int64")
        second = parts["s"].fillna("0").astype("int64")
        frac = parts["f"].fillna("").str.pad(9, side="right", fillchar="0").replace("", "0")
        nanos = frac.astype("int64")
        off_sign = parts["off_sign"].fillna("+")
        off_hours = parts["off_h"].fillna("0").astype("int64")
        off_minutes = parts["off_m"].fillna("0").astype("int64")
    elif mode == "date_constructor":
        parts = text.str.extract(DATE_CALL_TEXT_RE.pattern)
        year = parts["year"].fillna("0").astype("int64")
        month = parts["month"].fillna("1").astype("int64")
        day = parts["day"].fillna("1").astype("int64")
        hour = minute = second = nanos = off_hours = off_minutes = None
        off_sign = None
    elif mode == "datetime_constructor":
        dt_parts = text.str.extract(DATETIME_CALL_TEXT_RE.pattern)
        local_parts = text.str.extract(LOCALDATETIME_CALL_TEXT_RE.pattern)
        use_dt = text.str.match(DATETIME_CALL_TEXT_RE.pattern, na=False)
        year = dt_parts["year"].where(use_dt, local_parts["year"]).fillna("0").astype("int64")
        month = dt_parts["month"].where(use_dt, local_parts["month"]).fillna("1").astype("int64")
        day = dt_parts["day"].where(use_dt, local_parts["day"]).fillna("1").astype("int64")
        hour = dt_parts["hour"].where(use_dt, local_parts["hour"]).fillna("0").astype("int64")
        minute = dt_parts["minute"].where(use_dt, local_parts["minute"]).fillna("0").astype("int64")
        second = dt_parts["second"].where(use_dt, local_parts["second"]).fillna("0").astype("int64")
        frac = (
            dt_parts["nano"].where(use_dt, local_parts["nano"]).fillna("").str.pad(9, side="right", fillchar="0").replace("", "0")
        )
        nanos = frac.astype("int64")
        timezone = dt_parts["tz"].fillna("")
        off_sign = timezone.str[:1].where(use_dt, "+").replace("", "+")
        off_hours = timezone.str[1:3].where(use_dt, "0").replace("", "0").astype("int64")
        off_minutes = timezone.str[4:6].where(use_dt, "0").replace("", "0").astype("int64")
    else:
        time_parts = text.str.extract(TIME_CALL_TEXT_RE.pattern)
        local_parts = text.str.extract(LOCALTIME_CALL_TEXT_RE.pattern)
        use_tz = text.str.match(TIME_CALL_TEXT_RE.pattern, na=False)
        year = month = day = None
        hour = time_parts["hour"].where(use_tz, local_parts["hour"]).fillna("0").astype("int64")
        minute = time_parts["minute"].where(use_tz, local_parts["minute"]).fillna("0").astype("int64")
        second = time_parts["second"].where(use_tz, local_parts["second"]).fillna("0").astype("int64")
        frac = (
            time_parts["nano"].where(use_tz, local_parts["nano"]).fillna("").str.pad(9, side="right", fillchar="0").replace("", "0")
        )
        nanos = frac.astype("int64")
        timezone = time_parts["tz"].fillna("")
        off_sign = timezone.str[:1].where(use_tz, "+").replace("", "+")
        off_hours = timezone.str[1:3].where(use_tz, "0").replace("", "0").astype("int64")
        off_minutes = timezone.str[4:6].where(use_tz, "0").replace("", "0").astype("int64")

    day_nanos = 86_400 * 1_000_000_000
    if mode in {"date", "date_constructor"}:
        time_nanos = month.astype("int64") * 0
        if nanosecond_shift % day_nanos != 0:
            raise ValueError("date order_by duration support currently requires whole-day offsets")
        day_time_shift = nanosecond_shift // day_nanos
    else:
        assert (
            hour is not None
            and minute is not None
            and second is not None
            and nanos is not None
            and off_sign is not None
            and off_hours is not None
            and off_minutes is not None
        )
        sign_mult = off_sign.eq("-").astype("int64")
        sign_mult = sign_mult.where(sign_mult == 0, -1)
        sign_mult = sign_mult.where(sign_mult != 0, 1)
        offset_total_minutes = sign_mult * (off_hours * 60 + off_minutes)
        time_nanos = (
            (hour * 3600 + minute * 60 + second) * 1_000_000_000
            + nanos
            - offset_total_minutes * 60 * 1_000_000_000
        )
        if nanosecond_shift != 0:
            time_nanos = time_nanos + nanosecond_shift
        day_time_shift = 0

    if mode in {"time", "time_constructor"}:
        if month_shift != 0:
            raise ValueError("time order_by duration support currently rejects year/month offsets")
        key_col = fresh_col_name_fn(work_df.columns, f"{key_prefix}_time_ns")
        normalized_time_nanos = ((time_nanos % day_nanos) + day_nanos) % day_nanos
        out = work_df.assign(**{key_col: normalized_time_nanos.where(~null_mask, 9_223_372_036_854_775_000)})
        return out, [key_col]

    assert year is not None and month is not None and day is not None
    if month_shift != 0:
        total_months = year * 12 + (month - 1) + month_shift
        year = total_months // 12
        month = (total_months % 12) + 1
        leap = ((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))).astype("int64")
        max_day = month * 0 + 31
        max_day = max_day.where(~month.isin([4, 6, 9, 11]), 30)
        max_day = max_day.where(month != 2, 28 + leap)
        day = day.where(day <= max_day, max_day)
    a = (14 - month) // 12
    y2 = year + 4800 - a
    m2 = month + 12 * a - 3
    julian_day = day + ((153 * m2 + 2) // 5) + (365 * y2) + (y2 // 4) - (y2 // 100) + (y2 // 400) - 32045
    if mode in {"date", "date_constructor"}:
        day_adjust = day_time_shift
        nanos_of_day = month * 0
    else:
        day_adjust = (time_nanos // day_nanos) + day_time_shift
        nanos_of_day = time_nanos - ((time_nanos // day_nanos) * day_nanos)
    day_key = julian_day + day_adjust
    day_col = fresh_col_name_fn(work_df.columns, f"{key_prefix}_day")
    nanos_col = fresh_col_name_fn(work_df.columns, f"{key_prefix}_ns")
    out = work_df.assign(
        **{
            day_col: day_key.where(~null_mask, 9_223_372_036_854_775_000),
            nanos_col: nanos_of_day.where(~null_mask, day_nanos + 1),
        }
    )
    return out, [day_col, nanos_col]
