"""Native polars cypher result projection (Phase 2).

Lives in ``gfql.lazy.engine.polars`` (not the pandas-audited ``cypher`` package) so polars-only
rendering doesn't depress the pandas gfql coverage audit. Parity-or-NIE: no pandas bridge;
differential parity vs pandas is the release gate. The #1650 default (``structured=True``)
FLATTENS whole-entity ``RETURN n`` to ``{output}.{field}`` columns natively for ANY dtype
(float/temporal/nested just become columns, no rendering). Legacy display-string rendering
(``structured=False``) is native for int/string/bool node entities, including multi-node
binding rows (boolean ``label__*`` flags included); float/temporal/nested entity text, edge
entities, and exotic expressions raise NotImplementedError.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graphistry.Plottable import Plottable

if TYPE_CHECKING:
    import polars as pl
    from graphistry.compute.gfql.cypher.lowering import ResultProjectionPlan


from graphistry.Engine import is_polars_df as _is_polars_frame
from graphistry.compute.gfql.row.entity_props import (
    LABEL_FLAG_PREFIX,
    NODE_INTERNAL_COLS,
    label_flag_columns,
)


@dataclass(frozen=True)
class _AliasView:
    """One alias's rows under bare field names, plus ``field -> source column`` in the original."""
    frame: pl.DataFrame
    columns: Dict[str, str]


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


def _alias_view_polars(rows_df: pl.DataFrame, alias: str) -> Optional[_AliasView]:
    """Per-alias un-prefixed view of a row frame, plus field -> source-column map.

    polars edition of the pandas ``_projection_alias_rows``. A binding-row frame binds several
    aliases at once and spells each entity's fields ``{alias}.{field}`` alongside a bare
    ``{alias}`` id marker; every projection helper below wants the single-entity shape (bare
    field names). The view is a plain ``select`` of existing columns (Arrow buffers are shared,
    so no data is copied) and stays row-aligned with ``rows_df``; ``columns`` maps each view
    field back to its column in ``rows_df`` so the emitted expressions can be selected from the
    original frame.
    """
    import polars as pl

    prefix = f"{alias}."
    prefixed = [str(c) for c in rows_df.columns if str(c).startswith(prefix)]
    if not prefixed:
        if alias not in rows_df.columns:
            return None
        return _AliasView(frame=rows_df, columns={str(c): str(c) for c in rows_df.columns})
    columns = {c[len(prefix):]: c for c in prefixed}
    if alias in rows_df.columns:
        columns.setdefault(alias, alias)
    if alias not in columns:
        return None
    frame = rows_df.select([pl.col(src).alias(field) for field, src in columns.items()])
    return _AliasView(frame=frame, columns=columns)


def _native_node_entity_text_expr(view: _AliasView, alias: str, exclude: Any) -> Optional[Any]:
    """Native ``(:Label {prop: val, ...})`` node entity text; ``None`` → caller raises.

    Reads field names/dtypes off the un-prefixed ``view`` but emits ``pl.col`` against the
    original frame's column names, so multi-entity binding rows render as well as single-entity
    ones."""
    import polars as pl

    rows_df = view.frame
    cols = list(rows_df.columns)
    if alias not in cols or "type" in cols:
        return None  # typed (edge-ish) rows -> defer (NIE)

    def _c(field: str) -> pl.Expr:
        return pl.col(view.columns.get(field, field))

    from .dtypes import is_int
    schema = rows_df.schema
    excluded = set(str(c) for c in (exclude or ()))
    include_id = "id" in cols and is_int(schema["id"])
    prop_cols = [
        str(c) for c in cols
        if str(c) != alias and str(c) not in excluded
        and not str(c).startswith("__") and not str(c).startswith(LABEL_FLAG_PREFIX)
        and (str(c) not in NODE_INTERNAL_COLS or (include_id and str(c) == "id"))
    ]
    label_cols = label_flag_columns(cols)
    if any(schema[c] != pl.Boolean for c, _ in label_cols):
        return None  # non-boolean label flags -> defer (NIE)
    labels = (
        pl.concat_str([
            pl.when(_c(c).fill_null(False)).then(pl.lit(":" + label_name)).otherwise(pl.lit(""))
            for c, label_name in label_cols
        ], separator="")
        if label_cols else pl.lit("")
    )
    segments = []
    for col in prop_cols:
        val = _native_scalar_text_expr(view.columns.get(col, col), schema[col])
        if val is None:
            return None
        segments.append(pl.when(_c(col).is_null()).then(None).otherwise(pl.lit(f"{col}: ") + val))
    if not segments:
        rendered = pl.lit("(") + labels + pl.lit(")")
    else:
        props = pl.concat_str(segments, separator=", ", ignore_nulls=True)
        has_props = props.str.len_chars() > 0
        label_sep = pl.when(has_props & (labels.str.len_chars() > 0)).then(pl.lit(" ")).otherwise(pl.lit(""))
        prop_suffix = pl.when(has_props).then(label_sep + pl.lit("{") + props + pl.lit("}")).otherwise(pl.lit(""))
        rendered = pl.lit("(") + labels + prop_suffix + pl.lit(")")
    # Nullify absent (OPTIONAL-MATCH miss) rows — alias marker is null there and an absent
    # entity must render null, not "()" (mirrors pandas _nullify_missing_alias_rows); a real
    # property-less node keeps "()".
    return pl.when(_c(alias).is_null()).then(None).otherwise(rendered)


def _flat_entity_exprs_polars(view: _AliasView, projection: ResultProjectionPlan, source_alias: str, output_name: str, id_column: Optional[str]) -> Optional[List[pl.Expr]]:
    """Structured (flattened) whole-entity projection (#1650), polars edition. Mirrors pandas
    ``_flat_entity_columns`` exactly (same field selection + ordering via the shared
    ``_flat_entity_field_names``): one ``{output}.{field}`` column per field, read off the
    alias view's source column so single-entity and multi-entity binding rows both render.
    Works for ANY dtype (float/temporal/nested just become columns), covering cases entity-text
    defers."""
    import polars as pl
    from dataclasses import replace
    from graphistry.compute.gfql.cypher.result_postprocess import _flat_entity_field_names

    source_projection = projection if source_alias == projection.alias else replace(projection, alias=source_alias)
    fields = _flat_entity_field_names(view.frame, source_projection, id_column)
    if not fields:
        return None  # synthesized absent entity -> caller falls back to text
    out = []
    for field in fields:
        src = view.columns.get(field)
        if src is None:
            return None
        out.append(pl.col(src).alias(f"{output_name}.{field}"))
    return out


def _record_entity_meta(
    entity_meta: Dict[str, Dict[str, Any]],
    view: _AliasView,
    projection: ResultProjectionPlan,
    source_alias: str,
    output_name: str,
    id_column: Optional[str],
) -> None:
    """Record whole-entity projection metadata for one column, mirroring the pandas projector.

    ``view.frame`` is this alias's rows under bare field names and stays row-aligned with the
    projected output, so ``view.frame[id_column]`` is the carried alias's id column on both the
    single-entity and the multi-entity binding shape. Snapshot (``.clone()``) the id column so
    downstream reentry recovery never aliases a later-mutated working frame (see #1356)."""
    rows_df = view.frame
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
    # Property/expr source names are alias-relative ("id" is "{alias}.id" on binding rows).
    primary = _alias_view_polars(rows_df, projection.alias)
    primary_columns = primary.columns if primary is not None else {}
    for column in projection.columns:
        if column.kind == "whole_row":
            if projection.table != "nodes":
                return None  # edge entity rendering -> defer (NIE)
            source_alias = column.source_name or projection.alias
            view = _alias_view_polars(rows_df, source_alias)
            if view is None:
                return None
            if structured:
                # #1650 default: flatten to {output}.{field} (near-free, any dtype);
                # text fallback only for synthesized-absent rows.
                flat = _flat_entity_exprs_polars(view, projection, source_alias, column.output_name, id_column)
                if flat is not None:
                    exprs.extend(flat)
                    _record_entity_meta(entity_meta, view, projection, source_alias, column.output_name, id_column)
                    continue
            ent = _native_node_entity_text_expr(view, source_alias, projection.exclude_columns)
            if ent is None:
                return None
            exprs.append(ent.alias(column.output_name))
            _record_entity_meta(entity_meta, view, projection, source_alias, column.output_name, id_column)
            continue
        src = column.source_name
        if src is not None:
            src = primary_columns.get(src, src)
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
    for int/string/bool node entities, including multi-node binding rows, with boolean
    ``label__*`` flags. Edge entity-text and (text mode) float/temporal/nested columns are not yet
    native → raise rather than secretly run the pandas renderer.
    """
    rows_df = result._nodes
    native = _try_native_projection(result, rows_df, projection, structured)
    if native is not None:
        return native
    raise NotImplementedError(
        "polars engine does not yet natively render this cypher result projection "
        "(unsupported node entity text, edge entities, or exotic expressions); "
        "use engine='pandas' or engine='cudf' for this query "
        "(no silent fallback; parity-or-error by design)"
    )
