from __future__ import annotations

from typing import Sequence, cast

from graphistry.Plottable import Plottable
from graphistry.compute.typing import DataFrameT, SeriesT

from .lowering import WholeRowProjection


_NODE_INTERNAL_COLS = frozenset({"id", "labels", "type"})
_EDGE_INTERNAL_COLS = frozenset({"s", "d", "src", "dst", "edge_id", "type", "__gfql_edge_index_0__", "undirected"})


def _empty_text(df: DataFrameT, alias_col: str) -> SeriesT:
    base = cast(SeriesT, df[alias_col])
    return cast(SeriesT, base.astype(str).where(base.isna(), ""))


def _const_text(df: DataFrameT, alias_col: str, value: str) -> SeriesT:
    return cast(SeriesT, _empty_text(df, alias_col) + value)


def _false_mask(df: DataFrameT, alias_col: str) -> SeriesT:
    base = cast(SeriesT, df[alias_col])
    return cast(SeriesT, (base == True) & False)  # noqa: E712


def _is_null_mask(series: SeriesT) -> SeriesT:
    return cast(SeriesT, series.isna())


def _bool_mask(series: SeriesT) -> SeriesT:
    return cast(SeriesT, series == True)  # noqa: E712


def _render_scalar_value_text(df: DataFrameT, alias_col: str, series: SeriesT) -> SeriesT:
    text = cast(SeriesT, series.astype(str))
    dtype_txt = str(getattr(series, "dtype", "")).lower()
    if "bool" in dtype_txt and hasattr(text, "str"):
        return cast(SeriesT, text.str.lower())
    if "float" in dtype_txt and hasattr(text, "str"):
        return cast(SeriesT, text.str.replace(r"\.0+$", "", regex=True))
    if any(token in dtype_txt for token in ("int", "double", "decimal")):
        return text
    if hasattr(text, "str"):
        escaped = cast(SeriesT, text.str.replace("'", "\\'", regex=False))
        return cast(SeriesT, _const_text(df, alias_col, "'") + escaped + "'")
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
    label_cols = [col for col in df.columns if str(col).startswith("label__")]
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
        return cast(SeriesT, (_const_text(df, alias_col, ":") + type_series.astype(str)).where(include, ""))
    return _empty_text(df, alias_col)


def _node_property_columns(df: DataFrameT, alias_col: str, excluded: Sequence[str]) -> list[str]:
    return [
        str(col)
        for col in df.columns
        if str(col) != alias_col
        and str(col) not in excluded
        and not str(col).startswith("__")
        and not str(col).startswith("label__")
        and str(col) not in _NODE_INTERNAL_COLS
    ]


def _edge_property_columns(df: DataFrameT, alias_col: str, excluded: Sequence[str]) -> list[str]:
    return [
        str(col)
        for col in df.columns
        if str(col) != alias_col
        and str(col) not in excluded
        and not str(col).startswith("__")
        and str(col) not in _EDGE_INTERNAL_COLS
    ]


def _format_node_entities(df: DataFrameT, projection: WholeRowProjection) -> SeriesT:
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
    return cast(SeriesT, _const_text(df, alias_col, "(") + labels + prop_suffix + ")")


def _format_edge_entities(df: DataFrameT, projection: WholeRowProjection) -> SeriesT:
    alias_col = projection.alias
    if "type" in df.columns:
        type_series = cast(SeriesT, df["type"])
        type_part = cast(SeriesT, (_const_text(df, alias_col, ":") + type_series.astype(str)).where(~_is_null_mask(type_series), ""))
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
    return cast(SeriesT, _const_text(df, alias_col, "[") + type_part + prop_suffix + "]")


def apply_whole_row_projection(result: Plottable, projection: WholeRowProjection) -> Plottable:
    rows_df = cast(DataFrameT, getattr(result, "_nodes", None))
    if rows_df is None or projection.alias not in rows_df.columns:
        return result

    entity_series = (
        _format_node_entities(rows_df, projection)
        if projection.table == "nodes"
        else _format_edge_entities(rows_df, projection)
    )
    projected_nodes = cast(DataFrameT, rows_df.assign(**{projection.output_name: entity_series})[[projection.output_name]])

    out = result.bind()
    out._nodes = projected_nodes
    edges_df = getattr(result, "_edges", None)
    if edges_df is not None:
        out._edges = edges_df[:0]
    return out
