from __future__ import annotations

from typing import Any, Literal, Sequence, cast

import pandas as pd

from graphistry.compute.dataframe_utils import df_cons as template_df_cons
from graphistry.compute.gfql.series_str_compat import series_str_contains, series_str_extract, series_str_match
from graphistry.compute.gfql.temporal_text import (
    DATETIME_CALL_TEXT_RE,
    DATE_CALL_TEXT_RE,
    LOCALDATETIME_CALL_TEXT_RE,
    LOCALTIME_CALL_TEXT_RE,
    TIME_CALL_TEXT_RE,
)
from graphistry.compute.typing import DataFrameT, IndexT, SeriesT


_NODE_INTERNAL_COLS = frozenset({"id", "labels", "type"})
_EDGE_INTERNAL_COLS = frozenset({"s", "d", "src", "dst", "edge_id", "type", "__gfql_edge_index_0__", "undirected"})


def _fresh_col_name(columns: Sequence[object] | IndexT, prefix: str) -> str:
    existing = {str(col) for col in columns}
    candidate = prefix
    counter = 0
    while candidate in existing:
        counter += 1
        candidate = f"{prefix}{counter}"
    return candidate


def _include_numeric_id_as_property(df: DataFrameT) -> bool:
    if "id" not in df.columns:
        return False
    try:
        return bool(pd.api.types.is_numeric_dtype(df["id"]))
    except Exception:
        return False


def _list_series_from_python_rows(df: DataFrameT, col_name: str, row_count: int) -> SeriesT:
    return cast(SeriesT, template_df_cons(df, {col_name: [[] for _ in range(row_count)]})[col_name])


def node_property_columns(df: DataFrameT, alias_col: str, excluded: Sequence[str]) -> list[str]:
    include_id = _include_numeric_id_as_property(df)
    return [
        str(col)
        for col in df.columns
        if str(col) != alias_col
        and str(col) not in excluded
        and not str(col).startswith("__")
        and not str(col).startswith("label__")
        and (str(col) not in _NODE_INTERNAL_COLS or (include_id and str(col) == "id"))
    ]


def edge_property_columns(df: DataFrameT, alias_col: str, excluded: Sequence[str]) -> list[str]:
    return [
        str(col)
        for col in df.columns
        if str(col) != alias_col
        and str(col) not in excluded
        and not str(col).startswith("__")
        and str(col) not in _EDGE_INTERNAL_COLS
    ]


def _object_text(series: SeriesT) -> SeriesT:
    out = cast(SeriesT, series)
    if hasattr(out, "astype"):
        out = cast(SeriesT, out.astype("object"))
    return out


def _empty_text(df: DataFrameT, alias_col: str) -> SeriesT:
    base = cast(SeriesT, df[alias_col])
    text = cast(SeriesT, base.astype(str))
    null_preserving = cast(SeriesT, cast(Any, text).where(~base.isna(), None))
    return _object_text(cast(SeriesT, null_preserving.where(base.isna(), "")))


def _const_text(df: DataFrameT, alias_col: str, value: str) -> SeriesT:
    return cast(SeriesT, _empty_text(df, alias_col) + value)


def _false_mask(df: DataFrameT, alias_col: str) -> SeriesT:
    base = cast(SeriesT, df[alias_col])
    return cast(SeriesT, (base == True) & False)  # noqa: E712


def _is_null_mask(series: SeriesT) -> SeriesT:
    return cast(SeriesT, series.isna())


def _nullify_missing_alias_rows(df: DataFrameT, alias_col: str, rendered: SeriesT) -> SeriesT:
    mask = _is_null_mask(cast(SeriesT, df[alias_col]))
    if hasattr(rendered, "where"):
        return cast(SeriesT, cast(Any, rendered).where(~mask, None))
    out = cast(SeriesT, rendered.copy())
    out.loc[mask] = None
    return out


def _all_non_null_match(mask: SeriesT, non_null: SeriesT) -> bool:
    if not hasattr(mask, "where"):
        return False
    return bool(mask.where(non_null, True).all())


def _normalize_zero_offset_suffix(timezone: SeriesT) -> SeriesT:
    if not hasattr(timezone, "where") or not hasattr(timezone, "isin"):
        return timezone
    zero_offset = timezone.isin(["+00:00", "-00:00"])
    return cast(SeriesT, timezone.where(~zero_offset, "Z"))


def _quote_text_series(df: DataFrameT, alias_col: str, text: SeriesT) -> SeriesT:
    escaped = cast(SeriesT, text.str.replace("\\", "\\\\", regex=False))
    escaped = cast(SeriesT, escaped.str.replace("'", "\\'", regex=False))
    escaped = _object_text(escaped)
    return cast(SeriesT, _const_text(df, alias_col, "'") + escaped + "'")


def _const_text_from_template(text: SeriesT, value: str) -> SeriesT:
    return cast(SeriesT, text.where(text.isna(), "") + value)


def _zero_text_from_counts(text: SeriesT, counts: SeriesT) -> SeriesT:
    out = _const_text_from_template(text, "")
    max_count = int(counts.max()) if hasattr(counts, "max") else 0
    for count in range(max_count + 1):
        out = cast(SeriesT, out.where(counts != count, _const_text_from_template(text, "0" * count)))
    return out


def _slice_text_by_position(text: SeriesT, digits: SeriesT, positions: SeriesT, *, left: bool) -> SeriesT:
    out = _const_text_from_template(text, "")
    max_pos = int(positions.max()) if hasattr(positions, "max") else 0
    for pos in range(max_pos + 1):
        piece = cast(SeriesT, digits.str.slice(0, pos) if left else digits.str.slice(pos, None))
        out = cast(SeriesT, out.where(positions != pos, piece))
    return out


def _trim_decimal_trailing_zeroes(text: SeriesT) -> SeriesT:
    decimal_mask = cast(SeriesT, series_str_contains(text, r"\.", na=False))
    if not hasattr(decimal_mask, "any") or not bool(decimal_mask.any()):
        return text

    trimmed = cast(SeriesT, text.str.rstrip("0"))
    trailing_dot = cast(SeriesT, series_str_match(trimmed, r"^[+-]?\d+\.$", na=False))
    if hasattr(trailing_dot, "any") and bool(trailing_dot.any()):
        trimmed = cast(SeriesT, trimmed.where(~trailing_dot, trimmed.str.slice(0, -1)))
    return cast(SeriesT, text.where(~decimal_mask, trimmed))


def _normalize_scientific_numeric_text(text: SeriesT) -> SeriesT:
    sci_mask = cast(SeriesT, series_str_contains(text, r"[eE]", na=False))
    out = cast(SeriesT, text.str.replace(r"\.0+$", "", regex=True))
    if not hasattr(sci_mask, "any") or not bool(sci_mask.any()):
        return out

    parts = series_str_extract(text, r"^(?P<sign>[+-]?)(?P<int>\d+)(?:\.(?P<frac>\d+))?[eE](?P<exp>[+-]?\d+)$")
    valid = cast(SeriesT, parts["int"].notna())
    if not hasattr(valid, "any") or not bool(valid.any()):
        return out

    sign = cast(SeriesT, parts["sign"].fillna(""))
    int_part = cast(SeriesT, parts["int"].fillna(""))
    frac_part = cast(SeriesT, parts["frac"].fillna(""))
    digits = cast(SeriesT, int_part + frac_part)
    int_len = cast(SeriesT, int_part.str.len().fillna(0).astype("int64"))
    digit_len = cast(SeriesT, digits.str.len().fillna(0).astype("int64"))
    exponent = cast(SeriesT, parts["exp"].fillna("0").astype("int64"))
    decimal_pos = cast(SeriesT, int_len + exponent)

    neg_mask = cast(SeriesT, valid & (decimal_pos <= 0))
    if hasattr(neg_mask, "any") and bool(neg_mask.any()):
        zero_prefix = _zero_text_from_counts(
            text,
            cast(SeriesT, (-decimal_pos).where(neg_mask, 0).astype("int64")),
        )
        neg_out = cast(SeriesT, sign + _const_text_from_template(text, "0.") + zero_prefix + digits)
        out = cast(SeriesT, out.where(~neg_mask, neg_out))

    tail_mask = cast(SeriesT, valid & (decimal_pos >= digit_len))
    if hasattr(tail_mask, "any") and bool(tail_mask.any()):
        zero_suffix = _zero_text_from_counts(
            text,
            cast(SeriesT, (decimal_pos - digit_len).where(tail_mask, 0).astype("int64")),
        )
        tail_out = cast(SeriesT, sign + digits + zero_suffix)
        out = cast(SeriesT, out.where(~tail_mask, tail_out))

    mid_mask = cast(SeriesT, valid & ~(neg_mask | tail_mask))
    if hasattr(mid_mask, "any") and bool(mid_mask.any()):
        positions = cast(SeriesT, decimal_pos.where(mid_mask, 0).astype("int64"))
        left = _slice_text_by_position(text, digits, positions, left=True)
        right = _slice_text_by_position(text, digits, positions, left=False)
        mid_out = cast(SeriesT, sign + left + _const_text_from_template(text, ".") + right)
        out = cast(SeriesT, out.where(~mid_mask, mid_out))

    out = _trim_decimal_trailing_zeroes(out)
    neg_zero = cast(SeriesT, out.isin(["-0", "-0.0", "-0."]))
    return cast(SeriesT, out.where(~neg_zero, "0"))


def _normalize_temporal_constructor_series(
    df: DataFrameT,
    alias_col: str,
    source_series: SeriesT,
    text: SeriesT,
    *,
    quoted: bool,
) -> SeriesT | None:
    non_null = cast(SeriesT, ~_is_null_mask(source_series))
    stripped = cast(SeriesT, text.str.strip())

    def _format(series: SeriesT) -> SeriesT:
        series = _object_text(series)
        if not quoted:
            return series
        return cast(SeriesT, _const_text(df, alias_col, "'") + series + "'")

    date_mask = cast(SeriesT, series_str_match(stripped, DATE_CALL_TEXT_RE.pattern, na=False))
    if _all_non_null_match(date_mask, non_null):
        parts = series_str_extract(stripped, DATE_CALL_TEXT_RE.pattern)
        year = parts["year"].fillna("0").astype("int64").astype(str).str.zfill(4)
        month = parts["month"].fillna("0").astype("int64").astype(str).str.zfill(2)
        day = parts["day"].fillna("0").astype("int64").astype(str).str.zfill(2)
        return _format(cast(SeriesT, year + "-" + month + "-" + day))

    localtime_mask = cast(SeriesT, series_str_match(stripped, LOCALTIME_CALL_TEXT_RE.pattern, na=False))
    if _all_non_null_match(localtime_mask, non_null):
        parts = series_str_extract(stripped, LOCALTIME_CALL_TEXT_RE.pattern)
        hour = parts["hour"].fillna("0").astype("int64").astype(str).str.zfill(2)
        minute = parts["minute"].fillna("0").astype("int64").astype(str).str.zfill(2)
        second = parts["second"].fillna("")
        nanos = parts["nano"].fillna("").str.zfill(9).str.rstrip("0")
        base = cast(SeriesT, hour + ":" + minute)
        has_seconds = (second != "") | (nanos != "")
        second_text = cast(SeriesT, ":" + second.where(second != "", "00").astype(str).str.zfill(2))
        frac = cast(SeriesT, "." + nanos)
        base = cast(SeriesT, base + second_text.where(has_seconds, ""))
        base = cast(SeriesT, base + frac.where(nanos != "", ""))
        return _format(base)

    time_mask = cast(SeriesT, series_str_match(stripped, TIME_CALL_TEXT_RE.pattern, na=False))
    if _all_non_null_match(time_mask, non_null):
        parts = series_str_extract(stripped, TIME_CALL_TEXT_RE.pattern)
        hour = parts["hour"].fillna("0").astype("int64").astype(str).str.zfill(2)
        minute = parts["minute"].fillna("0").astype("int64").astype(str).str.zfill(2)
        second = parts["second"].fillna("")
        nanos = parts["nano"].fillna("").str.zfill(9).str.rstrip("0")
        timezone = _normalize_zero_offset_suffix(cast(SeriesT, parts["tz"].fillna("")))
        base = cast(SeriesT, hour + ":" + minute)
        has_seconds = (second != "") | (nanos != "")
        second_text = cast(SeriesT, ":" + second.where(second != "", "00").astype(str).str.zfill(2))
        frac = cast(SeriesT, "." + nanos)
        base = cast(SeriesT, base + second_text.where(has_seconds, ""))
        base = cast(SeriesT, base + frac.where(nanos != "", ""))
        base = cast(SeriesT, base + timezone)
        return _format(base)

    localdatetime_mask = cast(SeriesT, series_str_match(stripped, LOCALDATETIME_CALL_TEXT_RE.pattern, na=False))
    if _all_non_null_match(localdatetime_mask, non_null):
        parts = series_str_extract(stripped, LOCALDATETIME_CALL_TEXT_RE.pattern)
        year = parts["year"].fillna("0").astype("int64").astype(str).str.zfill(4)
        month = parts["month"].fillna("0").astype("int64").astype(str).str.zfill(2)
        day = parts["day"].fillna("0").astype("int64").astype(str).str.zfill(2)
        hour = parts["hour"].fillna("0").astype("int64").astype(str).str.zfill(2)
        minute = parts["minute"].fillna("0").astype("int64").astype(str).str.zfill(2)
        second = parts["second"].fillna("")
        nanos = parts["nano"].fillna("").str.zfill(9).str.rstrip("0")
        base = cast(SeriesT, year + "-" + month + "-" + day + "T" + hour + ":" + minute)
        has_seconds = (second != "") | (nanos != "")
        second_text = cast(SeriesT, ":" + second.where(second != "", "00").astype(str).str.zfill(2))
        frac = cast(SeriesT, "." + nanos)
        base = cast(SeriesT, base + second_text.where(has_seconds, ""))
        base = cast(SeriesT, base + frac.where(nanos != "", ""))
        return _format(base)

    datetime_mask = cast(SeriesT, series_str_match(stripped, DATETIME_CALL_TEXT_RE.pattern, na=False))
    if _all_non_null_match(datetime_mask, non_null):
        parts = series_str_extract(stripped, DATETIME_CALL_TEXT_RE.pattern)
        year = parts["year"].fillna("0").astype("int64").astype(str).str.zfill(4)
        month = parts["month"].fillna("0").astype("int64").astype(str).str.zfill(2)
        day = parts["day"].fillna("0").astype("int64").astype(str).str.zfill(2)
        hour = parts["hour"].fillna("0").astype("int64").astype(str).str.zfill(2)
        minute = parts["minute"].fillna("0").astype("int64").astype(str).str.zfill(2)
        second = parts["second"].fillna("")
        nanos = parts["nano"].fillna("").str.zfill(9).str.rstrip("0")
        timezone = _normalize_zero_offset_suffix(cast(SeriesT, parts["tz"].fillna("")))
        base = cast(SeriesT, year + "-" + month + "-" + day + "T" + hour + ":" + minute)
        has_seconds = (second != "") | (nanos != "")
        second_text = cast(SeriesT, ":" + second.where(second != "", "00").astype(str).str.zfill(2))
        frac = cast(SeriesT, "." + nanos)
        base = cast(SeriesT, base + second_text.where(has_seconds, ""))
        base = cast(SeriesT, base + frac.where(nanos != "", ""))
        base = cast(SeriesT, base + timezone)
        return _format(base)

    return None


def render_scalar_value_text(df: DataFrameT, alias_col: str, series: SeriesT) -> SeriesT:
    text = _object_text(cast(SeriesT, series.astype(str)))
    dtype_txt = str(getattr(series, "dtype", "")).lower()
    if "bool" in dtype_txt and hasattr(text, "str"):
        return cast(SeriesT, text.str.lower())
    if "float" in dtype_txt and hasattr(text, "str"):
        return _normalize_scientific_numeric_text(text)
    if any(token in dtype_txt for token in ("int", "double", "decimal")):
        return text
    if hasattr(text, "str"):
        stripped = cast(SeriesT, text.str.strip())
        non_null = cast(SeriesT, ~_is_null_mask(series))
        list_like = cast(SeriesT, series_str_match(stripped, r"^\[.*\]$", na=False))
        if _all_non_null_match(list_like, non_null):
            return stripped
        map_like = cast(SeriesT, series_str_match(stripped, r"^\{.*\}$", na=False))
        if _all_non_null_match(map_like, non_null):
            return stripped
        temporal = _normalize_temporal_constructor_series(df, alias_col, series, text, quoted=True)
        if temporal is not None:
            return temporal
        non_null = cast(SeriesT, ~_is_null_mask(series))
        bool_like = cast(SeriesT, series_str_match(text, r"^(True|False)$", na=False))
        num_like = cast(SeriesT, series_str_match(text, r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$", na=False))
        if hasattr(bool_like, "where") and bool(bool_like.where(non_null, True).all()):
            return cast(SeriesT, text.str.lower())
        if hasattr(num_like, "where") and bool(num_like.where(non_null, True).all()):
            return _normalize_scientific_numeric_text(text)
        out = _quote_text_series(df, alias_col, text)
        out = cast(SeriesT, out.where(~bool_like, text.str.lower()))
        out = cast(SeriesT, out.where(~num_like, text.str.replace(r"\.0+$", "", regex=True)))
        out = cast(SeriesT, out.where(~list_like, stripped))
        out = cast(SeriesT, out.where(~map_like, stripped))
        return out
    if hasattr(text, "str"):
        return _quote_text_series(df, alias_col, text)
    return cast(SeriesT, _const_text(df, alias_col, "'") + text + "'")


def append_property_segments(
    df: DataFrameT,
    alias_col: str,
    property_columns: Sequence[str],
) -> tuple[SeriesT, SeriesT]:
    text = _empty_text(df, alias_col)
    has_props = _false_mask(df, alias_col)
    for col in property_columns:
        series = cast(SeriesT, df[col])
        include = cast(SeriesT, ~_is_null_mask(series))
        value_text = render_scalar_value_text(df, alias_col, series)
        segment = _const_text(df, alias_col, f"{col}: ") + value_text
        prefix = cast(SeriesT, _const_text(df, alias_col, ", ").where(has_props & include, ""))
        append = cast(SeriesT, (prefix + segment).where(include, ""))
        text = cast(SeriesT, text + append)
        has_props = cast(SeriesT, has_props | include)
    return text, has_props


def format_edge_entity_text(
    df: DataFrameT,
    *,
    alias_col: str,
    property_columns: Sequence[str],
    type_col: str = "type",
    nullify_missing_alias_rows: bool = True,
) -> SeriesT:
    if type_col in df.columns:
        type_series = cast(SeriesT, df[type_col])
        type_text = _object_text(cast(SeriesT, type_series.astype(str)))
        include_type = cast(SeriesT, ~_is_null_mask(type_series))
        if hasattr(type_text, "str"):
            non_blank = cast(SeriesT, type_text.str.strip() != "")
            include_type = cast(SeriesT, include_type & non_blank)
        type_part = cast(SeriesT, (_const_text(df, alias_col, ":") + type_text).where(include_type, ""))
    else:
        type_part = _empty_text(df, alias_col)
    prop_text, has_props = append_property_segments(df, alias_col, property_columns)
    type_present = cast(SeriesT, type_part != "")
    prop_block = cast(SeriesT, _const_text(df, alias_col, "{") + prop_text + "}")
    prop_suffix = cast(
        SeriesT,
        (_const_text(df, alias_col, " ").where(has_props & type_present, "") + prop_block).where(has_props, ""),
    )
    rendered = cast(SeriesT, _const_text(df, alias_col, "[") + type_part + prop_suffix + "]")
    if nullify_missing_alias_rows:
        return _nullify_missing_alias_rows(df, alias_col, rendered)
    return rendered


def entity_property_columns(
    df: DataFrameT,
    *,
    alias_col: str,
    table: Literal["nodes", "edges"],
    excluded: Sequence[str],
) -> list[str]:
    if table == "nodes":
        return node_property_columns(df, alias_col, excluded)
    return edge_property_columns(df, alias_col, excluded)


def entity_keys_series(
    df: DataFrameT,
    *,
    alias_col: str,
    table: Literal["nodes", "edges"],
    excluded: Sequence[str],
) -> SeriesT:
    property_cols = entity_property_columns(
        df,
        alias_col=alias_col,
        table=table,
        excluded=excluded,
    )
    if len(property_cols) == 0:
        empty_col = _fresh_col_name(df.columns, "__gfql_entity_key_empty__")
        return _list_series_from_python_rows(df, empty_col, len(df))

    row_col = _fresh_col_name(df.columns, "__gfql_entity_key_row__")
    key_col = _fresh_col_name(df.columns, "__gfql_entity_key_name__")
    present_col = _fresh_col_name(df.columns, "__gfql_entity_key_present__")

    presence = cast(DataFrameT, df[property_cols].notna().copy())
    presence[row_col] = range(len(df))

    melted = cast(
        DataFrameT,
        presence.melt(
            id_vars=[row_col],
            value_vars=property_cols,
            var_name=key_col,
            value_name=present_col,
        ),
    )
    non_null = cast(DataFrameT, melted.loc[melted[present_col]])
    grouped = cast(
        DataFrameT,
        non_null.groupby(row_col, sort=False)[key_col].agg(list).reset_index(),
    )

    base_rows = cast(DataFrameT, template_df_cons(df, {row_col: range(len(df))}))
    merged = cast(DataFrameT, base_rows.merge(grouped, on=row_col, how="left", sort=False))
    out = cast(SeriesT, merged[key_col])
    missing_mask = cast(SeriesT, out.isna())
    if bool(missing_mask.any()):
        fill_col = _fresh_col_name(df.columns, "__gfql_entity_key_fill__")
        filled = _list_series_from_python_rows(df, fill_col, len(merged))
        out = cast(SeriesT, out.where(~missing_mask, filled))
    return out.reset_index(drop=True)
