"""Native polars cypher result projection (Phase 2).

Lives in ``gfql.lazy.engine.polars`` (not the pandas-audited ``cypher`` package) so polars-only
rendering doesn't depress the pandas gfql coverage audit. Parity-or-NIE: no pandas bridge;
differential parity vs pandas is the release gate. The #1650 default (``structured=True``)
FLATTENS whole-entity ``RETURN n`` to ``{output}.{field}`` columns natively for ANY dtype
(float/temporal/nested just become columns, no rendering). Legacy display-string rendering
(``structured=False``) is native only for single-entity int/string/bool nodes; float/temporal/
nested entity text, labels, multi-entity, edges, and exotic expressions raise NotImplementedError.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graphistry.Plottable import Plottable

if TYPE_CHECKING:
    import polars as pl
    from graphistry.compute.gfql.cypher.lowering import ResultProjectionPlan


from graphistry.Engine import is_polars_df as _is_polars_frame


def _has_temporal_constructor_text(rows_df: pl.DataFrame, col: str) -> bool:
    """True if a String property column holds Cypher temporal-constructor text (``date({...})``,
    ``datetime({...})``, …). The TCK graph builder stores temporal properties as these strings;
    the pandas projection normalizes them to ISO ('1910-05-06') via
    _normalize_temporal_constructor_series, not yet ported — so standalone temporal-property
    projection declines (NIE) rather than leak raw constructor text. Cheap native scan. Only
    standalone property projection needs this guard: whole-entity returns flatten the same raw
    column but are re-rendered downstream via render_entity_text."""
    import polars as pl
    from graphistry.compute.gfql.temporal.constructors import TEMPORAL_CALL_EXPR_RE
    # ^-anchored so values merely CONTAINING "date" (update({...}), candidate(...),
    # my date({x})) don't false-positive — these columns hold a WHOLE constructor string.
    pattern = r"^\s*" + TEMPORAL_CALL_EXPR_RE.pattern
    try:
        return bool(
            rows_df.select(pl.col(col).str.contains(pattern).any()).item()
        )
    except Exception:
        return False


def _native_scalar_text_expr(col: str, dtype: Any) -> Optional[Any]:
    """Per-dtype cypher value rendering as a polars expression, or None to bail. Matches the
    pandas entity renderer for safe scalars: ints raw, bools lowercased, strings single-quoted
    with ``\\``→``\\\\`` then ``'``→``\\'``. Floats (scientific/NaN repr diverges from pandas),
    temporal, and nested types return None → caller NIEs for those entities."""
    import polars as pl
    from .dtypes import is_int
    if is_int(dtype):
        return pl.col(col).cast(pl.String)
    if dtype == pl.Boolean:
        return pl.when(pl.col(col)).then(pl.lit("true")).otherwise(pl.lit("false"))
    if dtype == pl.String:
        escaped = pl.col(col).str.replace_all("\\", "\\\\", literal=True).str.replace_all("'", "\\'", literal=True)
        return pl.lit("'") + escaped + pl.lit("'")
    return None


def _native_node_entity_text_expr(rows_df: Any, alias: str, exclude: Any) -> Optional[Any]:
    """Native ``({prop: val, ...})`` node entity text for the single-entity int/string/bool
    no-labels case; None → caller raises. ``pl.concat_str(..., ignore_nulls=True)`` joins only
    non-null property segments with ", ", exactly matching the pandas renderer's null-omission."""
    import polars as pl

    cols = list(rows_df.columns)
    if alias not in cols:
        return None
    # single-entity only (no prefixed alias columns), no label rendering
    if any(str(c).startswith(f"{alias}.") for c in cols):
        return None
    if "type" in cols or any(str(c).startswith("label__") for c in cols):
        return None
    from .dtypes import is_int
    schema = rows_df.schema
    # Mirror entity_props.node_property_columns with a polars-aware "numeric id is a property"
    # check (the pandas helper's pd.api.types check drops id).
    internal = {"id", "labels", "type"}
    excluded = set(str(c) for c in (exclude or ()))
    include_id = "id" in cols and is_int(schema["id"])
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
        rendered = pl.lit("()")
    else:
        props = pl.concat_str(segments, separator=", ", ignore_nulls=True)
        has_props = props.str.len_chars() > 0
        rendered = pl.lit("(") + pl.when(has_props).then(pl.lit("{") + props + pl.lit("}")).otherwise(pl.lit("")) + pl.lit(")")
    # Nullify absent (OPTIONAL-MATCH miss) rows — alias marker is null there and an absent
    # entity must render null, not "()" (mirrors pandas _nullify_missing_alias_rows); a real
    # property-less node keeps "()".
    return pl.when(pl.col(alias).is_null()).then(None).otherwise(rendered)


def _flat_entity_exprs_polars(rows_df: pl.DataFrame, projection: ResultProjectionPlan, source_alias: str, output_name: str, id_column: Optional[str]) -> Optional[List[pl.Expr]]:
    """Structured (flattened) whole-entity projection (#1650), polars edition. Mirrors pandas
    ``_flat_entity_columns`` exactly (same field selection + ordering via the shared
    ``_flat_entity_field_names``): one ``pl.col(field).alias("{output}.{field}")`` per field.
    Single-entity only (None on multi-entity prefixed columns or absent fields). Works for ANY
    dtype (float/temporal/nested just become columns), covering cases entity-text defers."""
    import polars as pl
    from dataclasses import replace
    from graphistry.compute.gfql.cypher.result_postprocess import _flat_entity_field_names

    cols = list(rows_df.columns)
    if source_alias not in cols:
        return None
    if any(str(c).startswith(f"{source_alias}.") for c in cols):
        return None  # multi-entity binding -> defer (NIE), matches the text path
    source_projection = projection if source_alias == projection.alias else replace(projection, alias=source_alias)
    fields = _flat_entity_field_names(rows_df, source_projection, id_column)
    if not fields:
        return None  # synthesized absent entity -> caller falls back to text
    out = []
    for field in fields:
        if field not in cols:
            return None
        out.append(pl.col(field).alias(f"{output_name}.{field}"))
    return out


def _record_entity_meta(
    entity_meta: Dict[str, Dict[str, Any]],
    rows_df: pl.DataFrame,
    projection: ResultProjectionPlan,
    source_alias: str,
    output_name: str,
    id_column: Optional[str],
) -> None:
    """Record whole-entity projection metadata for one column, mirroring the pandas projector.

    ``_try_native_projection`` reaches this only in the single-entity branch (flat exprs and the
    entity-text path both decline multi-entity prefixed columns), so ``rows_df`` is the aligned
    source frame and ``rows_df[id_column]`` is the carried alias's id column, row-aligned with the
    projected output. Snapshot (``.clone()``) the id column so downstream reentry recovery never
    aliases a later-mutated working frame (see #1356)."""
    if id_column is None or id_column not in rows_df.columns:  # pragma: no cover - defensive: node re-entry always carries the id column
        return
    entity_meta[output_name] = {
        "table": projection.table,
        "alias": source_alias,
        "id_column": id_column,
        "ids": rows_df.get_column(id_column).clone(),
    }


def _try_native_projection(result: Plottable, rows_df: pl.DataFrame, projection: ResultProjectionPlan, structured: bool) -> Optional[Plottable]:
    """Native projection for property/expr columns already in the polars row table + structured-
    flat or entity-text whole-entity returns; None → caller raises NIE."""
    import polars as pl

    exprs = []
    # Whole-entity projection metadata side-channel (#1273 WITH->MATCH re-entry): mirror the
    # pandas projector (result_postprocess._apply_result_projection_pandas), which records the
    # carried alias's id column so the bounded-reentry executor can recover carried node
    # identities. Without it a WITH-projected node alias feeding a trailing MATCH declines.
    entity_meta: Dict[str, Dict[str, Any]] = {}
    id_column = result._node
    for column in projection.columns:
        if column.kind == "whole_row":
            if projection.table != "nodes":
                return None  # edge entity rendering -> defer (NIE)
            source_alias = column.source_name or projection.alias
            if structured:
                # #1650 default: flatten to {output}.{field} (near-free, any dtype);
                # text fallback only for synthesized-absent rows.
                flat = _flat_entity_exprs_polars(rows_df, projection, source_alias, column.output_name, id_column)
                if flat is not None:
                    exprs.extend(flat)
                    _record_entity_meta(entity_meta, rows_df, projection, source_alias, column.output_name, id_column)
                    continue
            ent = _native_node_entity_text_expr(rows_df, source_alias, projection.exclude_columns)
            if ent is None:
                return None
            exprs.append(ent.alias(column.output_name))
            _record_entity_meta(entity_meta, rows_df, projection, source_alias, column.output_name, id_column)
            continue
        src = column.source_name
        if src is None or src not in rows_df.columns:
            return None  # expression needing evaluation / missing -> defer (NIE)
        dtype = rows_df.schema[src]
        if dtype in (pl.Date, pl.Datetime, pl.Duration, pl.Time) or isinstance(dtype, (pl.List, pl.Struct, pl.Object)):
            return None  # temporal/nested rendering -> defer (NIE)
        if dtype == pl.String and _has_temporal_constructor_text(rows_df, src):
            return None  # temporal-constructor-string property -> defer (NIE)
        exprs.append(pl.col(src).alias(column.output_name))
    # decline (NIE): duplicate output names — pandas tolerates them (RETURN n, n.val emits n.val
    # twice: flattened entity + explicit) but polars .select rejects them; don't diverge or crash.
    out_names = [e.meta.output_name() for e in exprs]
    if len(out_names) != len(set(out_names)):
        return None
    out = result.bind()
    out._nodes = rows_df.select(exprs)
    if entity_meta:
        setattr(out, "_cypher_entity_projection_meta", entity_meta)
    edges_df = result._edges
    if edges_df is not None:
        out._edges = edges_df.clear() if _is_polars_frame(edges_df) else edges_df[:0]
    return out


def apply_result_projection_polars(
    result: Plottable,
    projection: ResultProjectionPlan,
    *,
    structured: bool = True,
) -> Plottable:
    """Native polars result projection, or honest NotImplementedError (no pandas fallback).

    ``structured=True`` (#1650 default): flatten whole-entity returns to ``{output}.{field}``
    columns (any dtype, near-free). ``structured=False``: legacy Cypher display string, native
    for int/string/bool single-entity nodes. Multi-entity bindings, edge entity-text, and (text
    mode) float/temporal/nested/label columns are not yet native → raise rather than secretly
    run the pandas renderer.
    """
    rows_df = result._nodes
    native = _try_native_projection(result, rows_df, projection, structured)
    if native is not None:
        return native
    raise NotImplementedError(
        "polars engine does not yet natively render this cypher result projection "
        "(whole-entity RETURN over float/temporal/nested/label/multi-entity columns); "
        "use engine='pandas' for this query "
        "(no pandas fallback; parity-or-error by design)"
    )
