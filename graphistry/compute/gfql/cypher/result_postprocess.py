from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional, Set, TypedDict, cast

import pandas as pd

from graphistry.Plottable import Plottable
from graphistry.compute.typing import DataFrameT, SeriesT
from graphistry.Engine import is_polars_df
from graphistry.compute.gfql.series_str_compat import is_non_textual_scalar_dtype

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
        Optional[Dict[str, WholeRowProjectionMeta]],
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


def _label_flag_columns(df: DataFrameT) -> list[str]:
    return [
        str(col)
        for col in df.columns
        if str(col).startswith("label__")
        and str(col).split("label__", 1)[1] not in {"<NA>", "None", "nan"}
    ]


def _flat_entity_field_names(
    source_rows_df: DataFrameT, projection: ResultProjectionPlan, id_column: Optional[str]
) -> list[str]:
    """Ordered field names for a flattened whole-entity projection (#1650).

    Mirrors the renderer's column selection (``node_property_columns`` /
    ``edge_property_columns`` honor ``exclude_columns`` so sibling aliases and
    engine-internal columns are not pulled in), then prepends the entity id and
    (nodes) appends ``label__*`` flags / (edges) the ``type`` column so
    :func:`render_entity_text` can losslessly reconstruct the Cypher form.
    """
    alias_col = projection.alias
    if projection.table == "nodes":
        prop_cols = node_property_columns(source_rows_df, alias_col, projection.exclude_columns)
        # label sources for faithful reconstruction: label__* flags and/or the
        # node ``type`` column (both consumed by _node_label_text).
        extra = _label_flag_columns(source_rows_df)
        if "type" in source_rows_df.columns:
            extra = [*extra, "type"]
    else:
        prop_cols = edge_property_columns(source_rows_df, alias_col, projection.exclude_columns)
        extra = ["type"] if "type" in source_rows_df.columns else []

    fields: list[str] = []
    for col in [id_column, *prop_cols, *extra]:
        if col is not None and col in source_rows_df.columns and col not in fields:
            fields.append(str(col))
    return fields


def _flat_entity_columns(
    source_rows_df: DataFrameT,
    projection: ResultProjectionPlan,
    output_name: str,
    id_column: Optional[str],
) -> Dict[str, SeriesT]:
    """Structured (flattened) whole-entity projection (issue #1650).

    Emit one ``{output_name}.{field}`` column per aliased field instead of
    collapsing the entity into a single Cypher display string. The per-field
    columns already exist on ``source_rows_df`` (gathered by
    ``_projection_alias_rows``), so this is "stop collapsing", not "rebuild":
    near-free, lossless, and directly usable without re-parsing a string.
    """
    return {
        f"{output_name}.{field}": cast(SeriesT, source_rows_df[field])
        for field in _flat_entity_field_names(source_rows_df, projection, id_column)
    }


def render_entity_text(
    result: Plottable, alias: str, *, table: Literal["nodes", "edges"] = "nodes"
) -> SeriesT:
    """Render a structured whole-entity projection back to Cypher display text.

    Presentation helper: given a result whose ``RETURN <alias>`` was emitted as
    flattened ``{alias}.{field}`` columns (the default since #1650), reconstruct
    the Cypher display string (``(:Label {..})`` / ``[:TYPE {..}]``). Used by the
    conformance/TCK driver and by callers who want the human-readable form. The
    structured data path itself never pays this cost.
    """
    rows_df = cast(DataFrameT, result._nodes)
    if rows_df is None:
        raise ValueError("result has no _nodes frame to render")
    prefix = f"{alias}."
    field_cols = [col for col in rows_df.columns if str(col).startswith(prefix)]
    if not field_cols:
        raise ValueError(f"no flattened columns found for alias {alias!r}")
    frame = cast(
        DataFrameT,
        rows_df[field_cols].rename(columns={col: str(col)[len(prefix):] for col in field_cols}),
    )
    # An OPTIONAL-MATCH miss flattens to a row whose fields are all null; such
    # rows must render as null, not "()". Track presence (any field non-null).
    present: Optional[SeriesT] = None
    for field in frame.columns:
        not_na = cast(SeriesT, frame[field].notna())
        present = not_na if present is None else cast(SeriesT, present | not_na)
    # _format_*_entities anchors length/null on a bare alias column; render every
    # row, then null absent rows below.
    frame = cast(DataFrameT, frame.assign(**{alias: True}))
    projection = ResultProjectionPlan(alias=alias, table=table, columns=(), exclude_columns=())
    rendered = _format_node_entities(frame, projection) if table == "nodes" else _format_edge_entities(frame, projection)
    if present is not None and hasattr(rendered, "where"):
        # Null absent rows. ``other=None`` fills NaN/None (valid pandas/cuDF);
        # the pandas-stubs ``where`` overload is stricter than runtime here.
        rendered = cast(SeriesT, rendered.where(present, None))  # type: ignore[call-overload]
    return rendered


def _project_property_column(
    rows_df: DataFrameT,
    *,
    column: ResultProjectionColumn,
) -> SeriesT:
    if column.source_name is None or column.source_name not in rows_df.columns:
        raise ValueError(f"projection source column not found: {column.source_name!r}")
    series = cast(SeriesT, rows_df[column.source_name])
    # Temporal-constructor normalization only applies to strings; numeric/bool/complex
    # can't hold temporal text, so skip the astype(str)+scan (byte-identical). #1650 gate.
    if is_non_textual_scalar_dtype(getattr(series, "dtype", None)):
        return series
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


def apply_result_projection(
    result: Plottable, projection: ResultProjectionPlan, *, structured: bool = True
) -> Plottable:
    """Project Cypher RETURN columns onto ``result._nodes``.

    ``structured=True`` (#1650 default) emits whole-entity returns as flattened
    ``{alias}.{field}`` columns. ``structured=False`` keeps the legacy single
    Cypher-display-string column; the reentry / OPTIONAL-MATCH null-fill machinery
    (which still assumes a single-column entity value) opts out via this flag until
    it is unified onto the structured path.

    For ``engine='polars'`` the native projection lives in gfql.lazy.engine.polars (not this
    pandas-audited module); it renders natively or raises NotImplementedError — NO
    pandas bridge (no-silent-fallback policy).
    """
    rows_df = result._nodes
    if is_polars_df(rows_df):
        from graphistry.compute.gfql.lazy.engine.polars.projection import apply_result_projection_polars
        return apply_result_projection_polars(result, projection, structured=structured)
    return _apply_result_projection_pandas(result, projection, structured=structured)


def _apply_result_projection_pandas(
    result: Plottable, projection: ResultProjectionPlan, *, structured: bool = True
) -> Plottable:
    rows_df = cast(Optional[DataFrameT], result._nodes)
    if rows_df is None:
        return result
    alias_rows_df = _projection_alias_rows(rows_df, alias=projection.alias)
    if alias_rows_df is None or projection.alias not in alias_rows_df.columns:
        return result
    projected_data: Dict[str, SeriesT] = {}
    projected_entity_meta: Dict[str, WholeRowProjectionMeta] = {}
    output_columns: list[str] = []
    for column in projection.columns:
        if column.kind == "whole_row":
            source_alias = column.source_name or projection.alias
            source_rows_df = _projection_alias_rows(rows_df, alias=source_alias)
            if source_rows_df is None or source_alias not in source_rows_df.columns:
                raise ValueError(f"whole-row projection source alias not found: {source_alias!r}")
            source_projection = projection if source_alias == projection.alias else replace(projection, alias=source_alias)
            id_column = result._node if source_projection.table == "nodes" else result._edge
            flat_columns = (
                _flat_entity_columns(source_rows_df, source_projection, column.output_name, id_column)
                if structured
                else {}
            )
            if structured and flat_columns:
                # Structured (flattened) emission (#1650): one column per field; text
                # stays available via render_entity_text().
                projected_data.update(flat_columns)
                output_columns.extend(flat_columns.keys())
            elif structured:
                # ⚠️ REGRESSION GUARD — DO NOT REMOVE (#1650). This fallback fixes
                # two regressions: top-level OPTIONAL-MATCH miss, and
                # OPTIONAL-WITH-reentry no-match. It looks redundant but is not.
                # No flattenable fields: the entity has no materialized property/id
                # columns on this frame. This is the synthesized null/absent-entity
                # row (top-level OPTIONAL-MATCH miss / reentry no-match, built by
                # _apply_empty_result_row as a single ``{alias: None}`` column). There
                # is nothing to flatten, so emit the single-column text form (which
                # renders to ``None`` here) — preserving the legacy shape the
                # OPTIONAL/reentry machinery consumes. Real rows always carry flat
                # field columns and take the branch above.
                projected_data[column.output_name] = (
                    _format_node_entities(source_rows_df, source_projection)
                    if source_projection.table == "nodes"
                    else _format_edge_entities(source_rows_df, source_projection)
                )
                output_columns.append(column.output_name)
            else:
                projected_data[column.output_name] = (
                    _format_node_entities(source_rows_df, source_projection)
                    if source_projection.table == "nodes"
                    else _format_edge_entities(source_rows_df, source_projection)
                )
                output_columns.append(column.output_name)
            if id_column is not None and id_column in source_rows_df.columns:
                projected_entity_meta[column.output_name] = {
                    "table": source_projection.table,
                    "alias": source_projection.alias,
                    "id_column": id_column,
                    # Snapshot the id Series: the bounded-reentry path recovers
                    # carried node identities from this meta and must not alias the
                    # live working frame (see #1356).
                    "ids": cast(SeriesT, source_rows_df[id_column]).copy(),
                }
        else:
            output_columns.append(column.output_name)
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
    # De-dup output columns (#1650): a flattened whole entity `a` (-> a.id, a.val, ...)
    # collides by name with an explicit property projection (`RETURN a, a.val`). Both
    # read the same source field (dotted aliases are rejected), so values are identical
    # — keep first occurrence; a duplicate name would drop data on to_dict/serialization.
    if len(set(output_columns)) != len(output_columns):
        seen: Set[str] = set()
        deduped: List[str] = []
        for c in output_columns:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        output_columns = deduped
    projected_rows = alias_rows_df
    if rows_df.__class__.__module__.startswith("cudf") and any(isinstance(value, pd.Series) for value in projected_data.values()):
        projected_rows = cast(DataFrameT, cast(Any, alias_rows_df).to_pandas())
        projected_data = {
            key: cast(SeriesT, value.to_pandas() if hasattr(value, "to_pandas") else value)
            for key, value in projected_data.items()
        }
    projected_nodes = cast(DataFrameT, projected_rows.assign(**projected_data)[output_columns])

    out = result.bind()
    out._nodes = projected_nodes
    if projected_entity_meta:
        setattr(out, "_cypher_entity_projection_meta", projected_entity_meta)
    edges_df = result._edges
    if edges_df is not None:
        out._edges = edges_df[:0]
    return out
