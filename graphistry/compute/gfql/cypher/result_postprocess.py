from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, Optional, TypedDict, cast

import pandas as pd

from graphistry.Plottable import Plottable
from graphistry.compute.typing import DataFrameT, SeriesT

from .lowering import ResultProjectionColumn, ResultProjectionPlan
from graphistry.compute.gfql.row.entity_props import (
    _const_text,
    _empty_text,
    _is_null_mask,
    _normalize_temporal_constructor_series,
    _nullify_missing_alias_rows,
    _object_text,
    append_property_segments,
    edge_property_columns,
    format_edge_entity_text,
    node_property_columns,
)
from graphistry.compute.gfql.row.pipeline import _RowPipelineAdapter


class WholeRowProjectionMeta(TypedDict):
    table: Literal["nodes", "edges"]
    alias: str
    id_column: str
    ids: SeriesT


def entity_projection_meta_entry(
    result: Plottable,
    *,
    output_name: str,
    field: str,
    message: str,
    suggestion: str,
) -> WholeRowProjectionMeta:
    """Look up a whole-row projection-meta entry on a Plottable result.

    Raises a Cypher-language ``GFQLValidationError`` (E108) when the
    side-channel metadata is missing or does not contain ``output_name``.
    Shared by the connected-OPTIONAL-MATCH and bounded-reentry execution
    paths in ``gfql_unified``.
    """
    from graphistry.compute.exceptions import ErrorCode, GFQLValidationError

    entity_meta = cast(
        Optional[dict[str, WholeRowProjectionMeta]],
        getattr(result, "_cypher_entity_projection_meta", None),
    )
    if not isinstance(entity_meta, dict) or output_name not in entity_meta:
        raise GFQLValidationError(
            ErrorCode.E108,
            message,
            field=field,
            value=output_name,
            suggestion=suggestion,
            language="cypher",
        )
    return entity_meta[output_name]


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
            mask = cast(SeriesT, cast(SeriesT, df[col]) == True)  # noqa: E712
            label = ":" + str(col).split("label__", 1)[1]
            labels = cast(SeriesT, labels + _const_text(df, alias_col, label).where(mask, ""))
        return labels
    if "type" in df.columns:
        type_series = cast(SeriesT, df["type"])
        include = cast(SeriesT, ~_is_null_mask(type_series))
        rendered = _object_text(cast(SeriesT, type_series.where(include, "").astype(str)))
        return cast(SeriesT, (_const_text(df, alias_col, ":") + rendered).where(include, ""))
    return _empty_text(df, alias_col)


def _format_node_entities(df: DataFrameT, projection: ResultProjectionPlan) -> SeriesT:
    alias_col = projection.alias
    labels = _node_label_text(df, alias_col)
    prop_text, has_props = append_property_segments(
        df,
        alias_col,
        node_property_columns(df, alias_col, projection.exclude_columns),
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
    return format_edge_entity_text(
        df,
        alias_col=alias_col,
        property_columns=edge_property_columns(df, alias_col, projection.exclude_columns),
        type_col="type",
        nullify_missing_alias_rows=True,
    )


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
    if column.source_name in rows_df.columns:
        return cast(SeriesT, rows_df[column.source_name])
    adapter = _RowPipelineAdapter(result)
    value = adapter._gfql_eval_string_expr(rows_df, column.source_name)
    return cast(SeriesT, value if hasattr(value, "astype") else _broadcast_projected_scalar(adapter, rows_df, value))


def _projection_alias_rows(
    rows_df: DataFrameT,
    *,
    alias: str,
) -> Optional[DataFrameT]:
    prefix = f"{alias}."
    alias_columns = [column for column in rows_df.columns if str(column).startswith(prefix)]
    if alias_columns:
        alias_rows = cast(
            DataFrameT,
            rows_df[alias_columns].rename(columns={column: str(column)[len(prefix):] for column in alias_columns}),
        )
        if alias in rows_df.columns and alias not in alias_rows.columns:
            alias_rows = cast(DataFrameT, alias_rows.assign(**{alias: rows_df[alias]}))
        if alias in alias_rows.columns:
            return alias_rows
    if alias in rows_df.columns:
        return rows_df
    return None


def apply_result_projection(result: Plottable, projection: ResultProjectionPlan) -> Plottable:
    rows_df = cast(DataFrameT, getattr(result, "_nodes", None))
    if rows_df is None:
        return result
    alias_rows_df = _projection_alias_rows(rows_df, alias=projection.alias)
    if alias_rows_df is None or projection.alias not in alias_rows_df.columns:
        return result
    projected_data: dict[str, SeriesT] = {}
    projected_entity_meta: dict[str, WholeRowProjectionMeta] = {}
    for column in projection.columns:
        if column.kind == "whole_row":
            source_alias = column.source_name or projection.alias
            source_rows_df = _projection_alias_rows(rows_df, alias=source_alias)
            if source_rows_df is None or source_alias not in source_rows_df.columns:
                raise ValueError(f"whole-row projection source alias not found: {source_alias!r}")
            source_projection = projection if source_alias == projection.alias else replace(projection, alias=source_alias)
            projected_data[column.output_name] = (
                _format_node_entities(source_rows_df, source_projection)
                if projection.table == "nodes"
                else _format_edge_entities(source_rows_df, source_projection)
            )
            id_column = getattr(result, "_node" if source_projection.table == "nodes" else "_edge", None)
            if id_column is not None and id_column in source_rows_df.columns:
                projected_entity_meta[column.output_name] = {
                    "table": source_projection.table,
                    "alias": source_projection.alias,
                    "id_column": id_column,
                    "ids": cast(SeriesT, source_rows_df[id_column]).copy(),
                }
        else:
            if column.kind == "property":
                property_rows_df = alias_rows_df
                if (
                    column.source_name is not None
                    and column.source_name not in alias_rows_df.columns
                    and column.source_name in rows_df.columns
                ):
                    property_rows_df = rows_df
                projected_data[column.output_name] = _project_property_column(property_rows_df, column=column)
            else:
                projected_data[column.output_name] = _project_expr_column(result, rows_df, column=column)
    projected_rows = alias_rows_df
    if rows_df.__class__.__module__.startswith("cudf") and any(isinstance(value, pd.Series) for value in projected_data.values()):
        projected_rows = cast(DataFrameT, cast(Any, alias_rows_df).to_pandas())
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
