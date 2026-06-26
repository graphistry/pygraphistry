"""Native polars cypher result projection (Phase 2).

Lives in ``engine_polars`` (not the pandas-audited ``cypher`` package) so the
polars-only rendering doesn't depress the pandas gfql coverage audit. Handles
the result projection for ``engine='polars'``: native ``rows_df.select`` for
property/expr columns and native ``({prop: val, ...})`` entity text for
single-entity int/string/bool nodes; bridges (polars→pandas→polars) for the
formatting the pandas renderer must do (whole-row floats/temporal/nested,
labels, multi-entity, edges, exotic expressions). Differential-conformance
gated. See plans/gfql-polars-engine.
"""
from typing import Any, Callable, Optional

import pandas as pd

from graphistry.Plottable import Plottable


def _is_polars_frame(df: Any) -> bool:
    return df is not None and "polars" in type(df).__module__


def _bridge_result_frames(result: Plottable, to: str) -> Plottable:
    """Convert a result's node/edge frames between polars and pandas."""
    out = result.bind()
    for attr in ("_nodes", "_edges"):
        df = getattr(result, attr, None)
        if df is None:
            continue
        if to == "pandas" and _is_polars_frame(df):
            setattr(out, attr, df.to_pandas())
        elif to == "polars" and isinstance(df, pd.DataFrame):
            import polars as pl
            setattr(out, attr, pl.from_pandas(df))
    return out


def _native_scalar_text_expr(col: str, dtype: Any) -> Optional[Any]:
    """Per-dtype cypher value rendering as a polars expression, or None to bail.

    Matches the pandas entity renderer for the safe scalar dtypes: ints raw,
    bools lowercased, strings single-quoted with ``\\``→``\\\\`` then ``'``→``\\'``.
    Floats (scientific/NaN repr diverges from pandas), temporal and nested types
    return None so the caller host-bridges those entities.
    """
    import polars as pl
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return pl.col(col).cast(pl.String)
    if dtype == pl.Boolean:
        return pl.when(pl.col(col)).then(pl.lit("true")).otherwise(pl.lit("false"))
    if dtype == pl.String:
        escaped = pl.col(col).str.replace_all("\\", "\\\\", literal=True).str.replace_all("'", "\\'", literal=True)
        return pl.lit("'") + escaped + pl.lit("'")
    return None


def _native_node_entity_text_expr(rows_df: Any, alias: str, exclude: Any) -> Optional[Any]:
    """Native polars ``({prop: val, ...})`` node entity text for the single-entity
    case with int/string/bool properties and no labels; None → bridge.

    ``pl.concat_str(..., ignore_nulls=True)`` joins only the non-null property
    segments with ``", "``, exactly matching the pandas renderer's null-omission.
    """
    import polars as pl

    cols = list(rows_df.columns)
    if alias not in cols:
        return None
    # single-entity only (no prefixed alias columns), no label rendering
    if any(str(c).startswith(f"{alias}.") for c in cols):
        return None
    if "type" in cols or any(str(c).startswith("label__") for c in cols):
        return None
    schema = rows_df.schema
    _int_dtypes = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    # Mirror entity_props.node_property_columns but with a polars-aware "numeric
    # id is a property" check (the pandas helper's pd.api.types check drops id).
    internal = {"id", "labels", "type"}
    excluded = set(str(c) for c in (exclude or ()))
    include_id = "id" in cols and schema["id"] in _int_dtypes
    prop_cols = [
        str(c) for c in cols
        if str(c) != alias and str(c) not in excluded
        and not str(c).startswith("__") and not str(c).startswith("label__")
        and (str(c) not in internal or (include_id and str(c) == "id"))
    ]
    segments = []
    for col in prop_cols:
        val = _native_scalar_text_expr(col, schema[col])
        if val is None:
            return None
        segments.append(pl.when(pl.col(col).is_null()).then(None).otherwise(pl.lit(f"{col}: ") + val))
    if not segments:
        return pl.lit("()")
    props = pl.concat_str(segments, separator=", ", ignore_nulls=True)
    has_props = props.str.len_chars() > 0
    return pl.lit("(") + pl.when(has_props).then(pl.lit("{") + props + pl.lit("}")).otherwise(pl.lit("")) + pl.lit(")")


def _try_native_projection(result: Plottable, rows_df: Any, projection: Any) -> Optional[Plottable]:
    """Native polars projection for property/expr columns already present in the
    (polars) row table + entity text for int/string/bool nodes. None → bridge."""
    import polars as pl

    exprs = []
    for column in projection.columns:
        if column.kind == "whole_row":
            if projection.table != "nodes":
                return None  # edge entity rendering -> bridge
            source_alias = column.source_name or projection.alias
            ent = _native_node_entity_text_expr(rows_df, source_alias, projection.exclude_columns)
            if ent is None:
                return None
            exprs.append(ent.alias(column.output_name))
            continue
        src = column.source_name
        if src is None or src not in rows_df.columns:
            return None  # expression needing evaluation / missing -> bridge
        dtype = rows_df.schema[src]
        if dtype in (pl.Date, pl.Datetime, pl.Duration, pl.Time) or isinstance(dtype, (pl.List, pl.Struct, pl.Object)):
            return None  # temporal/nested rendering -> bridge
        exprs.append(pl.col(src).alias(column.output_name))
    out = result.bind()
    out._nodes = rows_df.select(exprs)
    edges_df = getattr(result, "_edges", None)
    if edges_df is not None:
        out._edges = edges_df.clear() if _is_polars_frame(edges_df) else edges_df[:0]
    return out


def apply_result_projection_polars(
    result: Plottable,
    projection: Any,
    pandas_fallback: Callable[[Plottable, Any], Plottable],
) -> Plottable:
    """Entry point: native projection where possible, else host-bridge the pandas
    renderer and convert back to polars."""
    rows_df = getattr(result, "_nodes", None)
    native = _try_native_projection(result, rows_df, projection)
    if native is not None:
        return native
    bridged = _bridge_result_frames(result, to="pandas")
    out = pandas_fallback(bridged, projection)
    return _bridge_result_frames(out, to="polars")
