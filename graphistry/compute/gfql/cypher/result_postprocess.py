from __future__ import annotations

from typing import Any, Literal, Sequence, TypedDict, cast

import pandas as pd

from graphistry.Plottable import Plottable
from graphistry.compute.gfql.series_str_compat import series_str_contains, series_str_extract, series_str_match
from graphistry.compute.typing import DataFrameT, SeriesT

from .lowering import ResultProjectionColumn, ResultProjectionPlan
from graphistry.compute.gfql.temporal_text import (
    DATETIME_CALL_TEXT_RE,
    DATE_CALL_TEXT_RE,
    LOCALDATETIME_CALL_TEXT_RE,
    LOCALTIME_CALL_TEXT_RE,
    TIME_CALL_TEXT_RE,
)
from graphistry.compute.gfql.row.entity_props import edge_property_columns, node_property_columns
from graphistry.compute.gfql.row.pipeline import _RowPipelineAdapter


class WholeRowProjectionMeta(TypedDict):
    table: Literal["nodes", "edges"]
    alias: str
    id_column: str
    ids: SeriesT


def _object_text(series: SeriesT) -> SeriesT:
    out = cast(SeriesT, series)
    if hasattr(out, "astype"):
        out = cast(SeriesT, out.astype("object"))
    return out


def _empty_text(df: DataFrameT, alias_col: str) -> SeriesT:
    base = cast(SeriesT, df[alias_col])
    text = cast(SeriesT, base.astype(str))
    null_preserving = cast(SeriesT, text.where(~base.isna(), None))
    return _object_text(cast(SeriesT, null_preserving.where(base.isna(), "")))


def _const_text(df: DataFrameT, alias_col: str, value: str) -> SeriesT:
    return cast(SeriesT, _empty_text(df, alias_col) + value)


def _false_mask(df: DataFrameT, alias_col: str) -> SeriesT:
    base = cast(SeriesT, df[alias_col])
    return cast(SeriesT, (base == True) & False)  # noqa: E712


def _is_null_mask(series: SeriesT) -> SeriesT:
    return cast(SeriesT, series.isna())


def _bool_mask(series: SeriesT) -> SeriesT:
    return cast(SeriesT, series == True)  # noqa: E712


def _nullify_missing_alias_rows(df: DataFrameT, alias_col: str, rendered: SeriesT) -> SeriesT:
    mask = _is_null_mask(cast(SeriesT, df[alias_col]))
    if hasattr(rendered, "where"):
        return cast(SeriesT, rendered.where(~mask, None))
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


def _render_scalar_value_text(df: DataFrameT, alias_col: str, series: SeriesT) -> SeriesT:
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


def _append_property_segments(
    df: DataFrameT,
    alias_col: str,
    property_columns: Sequence[str],
) -> tuple[SeriesT, SeriesT]:
    text = _empty_text(df, alias_col)
    has_props = _false_mask(df, alias_col)
    for col in property_columns:
        series = cast(SeriesT, df[col])
        include = cast(SeriesT, ~_is_null_mask(series))
        value_text = _render_scalar_value_text(df, alias_col, series)
        segment = _const_text(df, alias_col, f"{col}: ") + value_text
        prefix = cast(SeriesT, _const_text(df, alias_col, ", ").where(has_props & include, ""))
        append = cast(SeriesT, (prefix + segment).where(include, ""))
        text = cast(SeriesT, text + append)
        has_props = cast(SeriesT, has_props | include)
    return text, has_props


def _node_label_text(df: DataFrameT, alias_col: str) -> SeriesT:
    label_cols = [
        col
        for col in df.columns
        if str(col).startswith("label__")
        and str(col).split("label__", 1)[1] not in {"<NA>", "None", "nan"}
    ]
    if label_cols:
        labels = _empty_text(df, alias_col)
        for col in label_cols:
            mask = _bool_mask(cast(SeriesT, df[col]))
            label = ":" + str(col).split("label__", 1)[1]
            labels = cast(SeriesT, labels + _const_text(df, alias_col, label).where(mask, ""))
        return labels
    if "type" in df.columns:
        type_series = cast(SeriesT, df["type"])
        include = cast(SeriesT, ~_is_null_mask(type_series))
        rendered = _object_text(cast(SeriesT, type_series.where(include, "").astype(str)))
        return cast(SeriesT, (_const_text(df, alias_col, ":") + rendered).where(include, ""))
    return _empty_text(df, alias_col)


def _node_property_columns(df: DataFrameT, alias_col: str, excluded: Sequence[str]) -> list[str]:
    return node_property_columns(df, alias_col, excluded)


def _edge_property_columns(df: DataFrameT, alias_col: str, excluded: Sequence[str]) -> list[str]:
    return edge_property_columns(df, alias_col, excluded)


def _format_node_entities(df: DataFrameT, projection: ResultProjectionPlan) -> SeriesT:
    alias_col = projection.alias
    labels = _node_label_text(df, alias_col)
    prop_text, has_props = _append_property_segments(
        df,
        alias_col,
        _node_property_columns(df, alias_col, projection.exclude_columns),
    )
    label_present = cast(SeriesT, labels != "")
    prop_block = cast(SeriesT, _const_text(df, alias_col, "{") + prop_text + "}")
    prop_suffix = cast(
        SeriesT,
        (_const_text(df, alias_col, " ").where(has_props & label_present, "") + prop_block).where(has_props, ""),
    )
    rendered = cast(SeriesT, _const_text(df, alias_col, "(") + labels + prop_suffix + ")")
    return _nullify_missing_alias_rows(df, alias_col, rendered)


def _format_edge_entities(df: DataFrameT, projection: ResultProjectionPlan) -> SeriesT:
    alias_col = projection.alias
    if "type" in df.columns:
        type_series = cast(SeriesT, df["type"])
        type_text = _object_text(cast(SeriesT, type_series.astype(str)))
        type_part = cast(SeriesT, (_const_text(df, alias_col, ":") + type_text).where(~_is_null_mask(type_series), ""))
    else:
        type_part = _empty_text(df, alias_col)
    prop_text, has_props = _append_property_segments(
        df,
        alias_col,
        _edge_property_columns(df, alias_col, projection.exclude_columns),
    )
    type_present = cast(SeriesT, type_part != "")
    prop_block = cast(SeriesT, _const_text(df, alias_col, "{") + prop_text + "}")
    prop_suffix = cast(
        SeriesT,
        (_const_text(df, alias_col, " ").where(has_props & type_present, "") + prop_block).where(has_props, ""),
    )
    rendered = cast(SeriesT, _const_text(df, alias_col, "[") + type_part + prop_suffix + "]")
    return _nullify_missing_alias_rows(df, alias_col, rendered)


def _project_property_column(
    rows_df: DataFrameT,
    *,
    column: ResultProjectionColumn,
) -> SeriesT:
    if column.source_name is None or column.source_name not in rows_df.columns:
        raise ValueError(f"projection source column not found: {column.source_name!r}")
    series = cast(SeriesT, rows_df[column.source_name])
    if hasattr(series, "astype") and hasattr(cast(SeriesT, series.astype(str)), "str"):
        normalized = _normalize_temporal_constructor_series(
            rows_df,
            column.output_name,
            series,
            cast(SeriesT, series.astype(str)),
            quoted=False,
        )
        if normalized is not None:
            return normalized
    return series


def _broadcast_projected_scalar(adapter: _RowPipelineAdapter, rows_df: DataFrameT, value: Any) -> SeriesT:
    if rows_df.__class__.__module__.startswith("cudf") and isinstance(value, (list, tuple, dict)):
        return cast(SeriesT, pd.Series([value for _ in range(len(rows_df))], dtype="object"))
    return cast(SeriesT, adapter._gfql_broadcast_scalar(rows_df, value))


def _project_expr_column(
    result: Plottable,
    rows_df: DataFrameT,
    *,
    column: ResultProjectionColumn,
) -> SeriesT:
    if column.source_name is None:
        raise ValueError(f"projection expression not found: {column.output_name!r}")
    adapter = _RowPipelineAdapter(result)
    value = adapter._gfql_eval_string_expr(rows_df, column.source_name)
    return cast(SeriesT, value if hasattr(value, "astype") else _broadcast_projected_scalar(adapter, rows_df, value))


def _whole_row_projection_meta(
    result: Plottable,
    rows_df: DataFrameT,
    projection: ResultProjectionPlan,
) -> WholeRowProjectionMeta | None:
    id_column = getattr(result, "_node" if projection.table == "nodes" else "_edge", None)
    if id_column is None or id_column not in rows_df.columns:
        return None
    return {
        "table": projection.table,
        "alias": projection.alias,
        "id_column": id_column,
        "ids": cast(SeriesT, rows_df[id_column]).copy(),
    }


def apply_result_projection(result: Plottable, projection: ResultProjectionPlan) -> Plottable:
    rows_df = cast(DataFrameT, getattr(result, "_nodes", None))
    if rows_df is None or projection.alias not in rows_df.columns:
        return result

    entity_series = (
        _format_node_entities(rows_df, projection)
        if projection.table == "nodes"
        else _format_edge_entities(rows_df, projection)
    )
    projected_data: dict[str, SeriesT] = {}
    whole_row_meta = _whole_row_projection_meta(result, rows_df, projection)
    projected_entity_meta: dict[str, WholeRowProjectionMeta] = {}
    for column in projection.columns:
        if column.kind == "whole_row":
            projected_data[column.output_name] = entity_series
            if whole_row_meta is not None:
                projected_entity_meta[column.output_name] = {
                    "table": whole_row_meta["table"],
                    "alias": whole_row_meta["alias"],
                    "id_column": whole_row_meta["id_column"],
                    "ids": whole_row_meta["ids"],
                }
        else:
            projected_data[column.output_name] = (
                _project_property_column(rows_df, column=column)
                if column.kind == "property"
                else _project_expr_column(result, rows_df, column=column)
            )
    projected_rows = rows_df
    if rows_df.__class__.__module__.startswith("cudf") and any(isinstance(value, pd.Series) for value in projected_data.values()):
        projected_rows = cast(DataFrameT, rows_df.to_pandas())
        projected_data = {
            key: cast(SeriesT, value.to_pandas() if hasattr(value, "to_pandas") else value)
            for key, value in projected_data.items()
        }
    projected_nodes = cast(DataFrameT, projected_rows.assign(**projected_data)[[column.output_name for column in projection.columns]])

    out = result.bind()
    out._nodes = projected_nodes
    if projected_entity_meta:
        setattr(out, "_cypher_entity_projection_meta", projected_entity_meta)
    edges_df = getattr(result, "_edges", None)
    if edges_df is not None:
        out._edges = edges_df[:0]
    return out
