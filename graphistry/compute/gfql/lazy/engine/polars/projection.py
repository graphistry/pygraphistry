"""Native polars cypher result projection (Phase 2).

Lives in ``gfql.lazy.engine.polars`` (not the pandas-audited ``cypher`` package) so the
polars-only rendering doesn't depress the pandas gfql coverage audit. Handles
the result projection for ``engine='polars'``. The #1650 default (``structured=True``)
FLATTENS a whole-entity ``RETURN n`` to ``{output}.{field}`` columns natively for ANY
dtype (float/temporal/nested included — they just become columns, no rendering), so the
common whole-entity case is native regardless of dtype. The legacy Cypher display-string
rendering (``structured=False``) stays native only for single-entity int/string/bool
nodes; it raises NotImplementedError (NO pandas bridge — no-silent-fallback policy) for
float/temporal/nested entity text, labels, multi-entity, edges, and exotic expressions.
Differential-conformance gated. Differential parity vs pandas is the release gate.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from graphistry.Plottable import Plottable

if TYPE_CHECKING:
    import polars as pl
    from graphistry.compute.gfql.cypher.lowering import ResultProjectionPlan


from graphistry.Engine import is_polars_df as _is_polars_frame


def _has_temporal_constructor_text(rows_df: pl.DataFrame, col: str) -> bool:
    """True if a String property column holds Cypher temporal-constructor text
    (``date({...})``, ``datetime({...})``, …).

    The TCK graph builder stores temporal property values as these constructor
    strings; the pandas projection normalizes them to ISO (``'1910-05-06'``) via
    ``_normalize_temporal_constructor_series``. That normalizer is not yet ported
    natively, so a standalone temporal-property projection must decline honestly
    (NIE) rather than leak the raw constructor text. Cheap native scan — no pandas
    bridge. (Whole-entity returns flatten the same raw column but are re-rendered
    correctly downstream via ``render_entity_text``, so only standalone property
    projection needs this guard.)"""
    import polars as pl
    from graphistry.compute.gfql.temporal.constructors import TEMPORAL_CALL_EXPR_RE
    # Anchor with ^ so ordinary string values whose text merely CONTAINS a substring
    # like "date" (``update({...})``, ``candidate(...)``, ``my date({x})``) don't
    # false-positive — these columns hold a WHOLE constructor string, not embedded text.
    pattern = r"^\s*" + TEMPORAL_CALL_EXPR_RE.pattern
    try:
        return bool(
            rows_df.select(pl.col(col).str.contains(pattern).any()).item()
        )
    except Exception:
        return False


def _native_scalar_text_expr(col: str, dtype: Any) -> Optional[Any]:
    """Per-dtype cypher value rendering as a polars expression, or None to bail.

    Matches the pandas entity renderer for the safe scalar dtypes: ints raw,
    bools lowercased, strings single-quoted with ``\\``→``\\\\`` then ``'``→``\\'``.
    Floats (scientific/NaN repr diverges from pandas), temporal and nested types
    return None so the caller raises NotImplementedError for those entities.
    """
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
    """Native polars ``({prop: val, ...})`` node entity text for the single-entity
    case with int/string/bool properties and no labels; None → caller raises.

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
    from .dtypes import is_int
    schema = rows_df.schema
    # Mirror entity_props.node_property_columns but with a polars-aware "numeric
    # id is a property" check (the pandas helper's pd.api.types check drops id).
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
    # Nullify absent (OPTIONAL-MATCH miss) rows: the alias marker column is null
    # there, and an absent entity must render as null, not "()" (mirrors the pandas
    # renderer's _nullify_missing_alias_rows). A real property-less node keeps "()".
    return pl.when(pl.col(alias).is_null()).then(None).otherwise(rendered)


def _flat_entity_exprs_polars(rows_df: pl.DataFrame, projection: ResultProjectionPlan, source_alias: str, output_name: str, id_column: Optional[str]) -> Optional[List[pl.Expr]]:
    """Structured (flattened) whole-entity projection (#1650), polars edition.

    Mirrors the pandas ``_flat_entity_columns`` exactly (same field selection +
    ordering via the shared ``_flat_entity_field_names``) so polars == pandas:
    one ``pl.col(field).alias("{output}.{field}")`` per entity field. Single-entity
    only (bail to None on multi-entity prefixed columns or absent fields). Works
    for ANY dtype (float/temporal/nested just become columns — no rendering), so
    structured returns cover cases the entity-text path had to defer."""
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


def _try_native_projection(result: Plottable, rows_df: pl.DataFrame, projection: ResultProjectionPlan, structured: bool) -> Optional[Plottable]:
    """Native polars projection for property/expr columns already present in the
    (polars) row table + structured-flat or entity-text whole-entity returns.
    None → caller raises NIE."""
    import polars as pl

    exprs = []
    for column in projection.columns:
        if column.kind == "whole_row":
            if projection.table != "nodes":
                return None  # edge entity rendering -> defer (NIE)
            source_alias = column.source_name or projection.alias
            if structured:
                # #1650 default: flatten to {output}.{field} columns (near-free,
                # any dtype). Falls back to text only for synthesized-absent rows.
                id_column = result._node
                flat = _flat_entity_exprs_polars(rows_df, projection, source_alias, column.output_name, id_column)
                if flat is not None:
                    exprs.extend(flat)
                    continue
            ent = _native_node_entity_text_expr(rows_df, source_alias, projection.exclude_columns)
            if ent is None:
                return None
            exprs.append(ent.alias(column.output_name))
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
    # pandas tolerates duplicate output column names (e.g. RETURN n, n.val emits
    # n.val twice — once from the flattened entity, once explicit); polars .select
    # rejects duplicate names. Defer (NIE) rather than diverge or crash.
    out_names = [e.meta.output_name() for e in exprs]
    if len(out_names) != len(set(out_names)):
        return None
    out = result.bind()
    out._nodes = rows_df.select(exprs)
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
    """Native polars result projection, or honest NotImplementedError.

    NO pandas fallback (no-silent-fallback policy). ``structured=True`` (#1650
    default) flattens whole-entity returns to ``{output}.{field}`` columns (any
    dtype, near-free); ``structured=False`` renders the legacy Cypher display
    string natively for int/string/bool single-entity nodes. Multi-entity
    bindings, edge entity-text, and (text mode) float/temporal/nested/label
    columns are not yet native → raise rather than secretly run the pandas renderer.
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
