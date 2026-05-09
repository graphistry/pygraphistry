"""Bounded-reentry data-frame execution helpers.

Extracted from ``graphistry/compute/gfql_unified.py`` under #987 Step 3
(`move runtime stitching into a dedicated reentry module`). Pure-move
refactor — no semantic changes vs the prior in-line definitions.

Together with ``cypher/reentry_plan.py`` (compile-time contract),
``cypher/reentry/compiletime.py`` (compile-time query rewrites), and the
``CompiledCypherQuery.start_nodes_query`` chain, this module owns the
*data-frame side* of bounded ``MATCH ... WITH ... MATCH ...`` re-entry:
preparing the dispatch base graph, building the seeded ``start_nodes``
table, fanning out multi-row scalar / free-form prefixes, and null-filling
unmatched rows for the OPTIONAL re-entry case.

Callers (currently only ``gfql_unified.py``):

- ``_compiled_query_reentry_state``           — whole-row carry
- ``_compiled_query_scalar_reentry_state``    — scalar-only prefix carry
- ``_compiled_query_freeform_reentry_state``  — free-form intermediate MATCH (single-row)
- ``_freeform_broadcast_row_to_nodes``        — single-row broadcast (also used per-row in #1285 multi-row)
- ``_union_scalar_reentry_results``           — concat per-row dispatches
- ``_apply_optional_reentry_null_fill``       — OPTIONAL re-entry null fill
"""
# ruff: noqa: E501
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

from graphistry.Engine import EngineAbstract, df_concat, df_cons, resolve_engine, safe_merge
from graphistry.Plottable import Plottable
from graphistry.compute.exceptions import GFQLValidationError, ErrorCode
from graphistry.compute.gfql.cypher.lowering import (
    CompiledCypherQuery,
    _reentry_hidden_column_name,
)
from graphistry.compute.gfql.cypher.reentry_plan import ReentryPlan
from graphistry.compute.gfql.cypher.result_postprocess import (
    entity_projection_meta_entry,
)
from graphistry.compute.typing import DataFrameT, SeriesT


REENTRY_WHOLE_ROW_SUGGESTION = "Carry a whole-row node alias through WITH before MATCH re-entry."
REENTRY_SCALAR_SUGGESTION = "Carry scalar columns through WITH before MATCH re-entry."


def reentry_validation_error(
    message: str,
    *,
    value: Any,
    suggestion: str,
    field: str = "with",
) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        message,
        field=field,
        value=value,
        suggestion=suggestion,
        language="cypher",
    )


def apply_optional_reentry_null_fill(
    result: Plottable,
    *,
    prefix_result: Plottable,
    engine: Union[EngineAbstract, str],
    empty_result_row: Optional[Dict[str, Any]] = None,
) -> Plottable:
    """Null-fill result rows for prefix rows that the optional reentry didn't match."""
    prefix_df = getattr(prefix_result, "_nodes", None)
    result_df = getattr(result, "_nodes", None)

    if prefix_df is None or len(prefix_df) == 0:
        return result

    prefix_rows = len(prefix_df)
    result_rows = 0 if result_df is None else len(result_df)

    if result_rows >= prefix_rows:
        return result

    concrete_engine = resolve_engine(cast(Any, engine), result)
    df_ctor = df_cons(concrete_engine)
    concat = df_concat(concrete_engine)

    # Use the compiled empty_result_row template (correct projected column names)
    # or fall back to the result's own columns.
    if empty_result_row is not None:
        null_row = dict(empty_result_row)
    elif result_df is not None and len(result_df.columns) > 0:
        null_row = {col: None for col in result_df.columns}
    else:
        null_row = {}

    if result_df is None or len(result_df) == 0:
        if null_row:
            out = result.bind()
            out._nodes = df_ctor([dict(null_row) for _ in range(prefix_rows)])
            return out
        return result

    missing_count = prefix_rows - result_rows
    fill_rows = [dict(null_row) for _ in range(missing_count)]

    fill_df = df_ctor(fill_rows)
    out = result.bind()
    out._nodes = concat([result_df, fill_df], ignore_index=True, sort=False)
    edges_df = getattr(result, "_edges", None)
    if edges_df is not None:
        out._edges = edges_df[:0]
    return out


def compiled_query_reentry_state(
    base_graph: Plottable,
    plan: ReentryPlan,
    prefix_result: Plottable,
    *,
    engine: Union[EngineAbstract, str],
) -> Tuple[Plottable, DataFrameT]:
    if plan.scalar_only or plan.free_form:
        raise reentry_validation_error(
            "Cypher whole-row reentry dispatcher received a non-whole-row ReentryPlan",
            value=plan.reentry_alias_name,
            suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
        )
    output_name = plan.reentry_alias_name
    carried_columns = tuple(plan.scalar_columns)
    prefix_rows = cast(Optional[DataFrameT], getattr(prefix_result, "_nodes", None))
    prefix_alias_values: Optional[SeriesT] = None
    if prefix_rows is not None and output_name in prefix_rows.columns:
        prefix_alias_values = cast(SeriesT, prefix_rows[output_name])
    entity_meta = cast(Optional[Dict[str, Any]], getattr(prefix_result, "_cypher_entity_projection_meta", None))
    has_projection_meta = isinstance(entity_meta, dict) and output_name in entity_meta
    if not has_projection_meta and prefix_alias_values is not None and hasattr(prefix_alias_values, "notna"):
        # #1356: OPTIONAL prefix + no matches can produce a single null carry row
        # without whole-row projection metadata. Treat it as an empty seed set
        # for re-entry instead of raising "could not recover carried identities".
        non_null_mask = cast(SeriesT, prefix_alias_values.notna())
        has_non_null_ids = bool(non_null_mask.any()) if hasattr(non_null_mask, "any") else True
        if not has_non_null_ids:
            base_nodes = cast(Optional[DataFrameT], getattr(base_graph, "_nodes", None))
            id_column = cast(Optional[str], getattr(base_graph, "_node", None))
            if base_nodes is None or id_column is None or id_column not in base_nodes.columns:
                raise reentry_validation_error(
                    "Cypher MATCH after WITH could not recover the base node table for re-entry",
                    value=id_column,
                    suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
                )
            concrete_engine = resolve_engine(cast(Any, engine), base_graph)
            carried_node_ids = cast(DataFrameT, df_cons(concrete_engine)({id_column: []}))
            return base_graph, ordered_reentry_start_nodes(
                node_rows=base_nodes,
                carried_node_ids=carried_node_ids,
                id_column=id_column,
            )
    meta = entity_projection_meta_entry(
        prefix_result,
        output_name=output_name,
        field="with",
        message="Cypher MATCH after WITH could not recover carried node identities from the prefix stage",
        suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
    )
    if meta["table"] != "nodes":
        raise reentry_validation_error(
            "Cypher MATCH after WITH currently supports node re-entry only",
            value=output_name,
            suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
        )
    ids = meta["ids"]
    id_column = meta["id_column"]
    if not hasattr(ids, "dropna"):
        raise reentry_validation_error(
            "Cypher MATCH after WITH could not recover carried node identities from the prefix stage",
            value=output_name,
            suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
        )
    base_nodes = getattr(base_graph, "_nodes", None)
    if base_nodes is None or id_column not in base_nodes.columns:
        raise reentry_validation_error(
            "Cypher MATCH after WITH could not recover the base node table for re-entry",
            value=id_column,
            suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
        )
    concrete_engine = resolve_engine(cast(Any, engine), base_graph)
    carried_ids, aligned_prefix_rows = aligned_reentry_rows(
        ids=cast(SeriesT, ids),
        prefix_rows=getattr(prefix_result, "_nodes", None),
        output_name=output_name,
    )
    carried_node_ids = cast(DataFrameT, df_cons(concrete_engine)({id_column: carried_ids}))
    if not carried_columns:
        return base_graph, ordered_reentry_start_nodes(
            node_rows=base_nodes,
            carried_node_ids=carried_node_ids,
            id_column=id_column,
        )
    if aligned_prefix_rows is None:
        raise reentry_validation_error(
            "Cypher MATCH after WITH could not recover carried row columns from the prefix stage",
            value=output_name,
            suggestion=REENTRY_SCALAR_SUGGESTION,
        )
    duplicate_mask = carried_ids.duplicated()
    if bool(duplicate_mask.any()) if hasattr(duplicate_mask, "any") else False:
        raise reentry_validation_error(
            "Cypher MATCH after WITH carried scalar columns currently require unique carried node rows",
            value=output_name,
            suggestion="Use a single-node seed WITH shape, or avoid carrying scalar columns into MATCH re-entry.",
        )

    carry_payload = reentry_carry_payload(
        carried_node_ids=carried_node_ids,
        prefix_rows=aligned_prefix_rows,
        carried_columns=carried_columns,
    )
    hidden_columns = [name for name in map(_reentry_hidden_column_name, carried_columns) if name in base_nodes.columns]
    merge_base = cast(DataFrameT, base_nodes.drop(columns=hidden_columns)) if hidden_columns else base_nodes
    node_rows = cast(DataFrameT, safe_merge(merge_base, carry_payload, on=id_column, how="left"))

    dispatch_graph = base_graph.bind()
    dispatch_graph._nodes = node_rows
    edges_df = getattr(base_graph, "_edges", None)
    if edges_df is not None:
        dispatch_graph._edges = edges_df
    return dispatch_graph, ordered_reentry_start_nodes(
        node_rows=node_rows,
        carried_node_ids=carried_node_ids,
        id_column=id_column,
    )


def union_scalar_reentry_results(
    row_results: List[Plottable],
    *,
    base_graph: Plottable,
    engine: Union[EngineAbstract, str],
) -> Plottable:
    """Union per-row suffix results from a multi-row scalar prefix (#1047)."""
    node_frames = []
    for r in row_results:
        nodes = getattr(r, "_nodes", None)
        if nodes is not None and len(cast(Any, nodes)) > 0:
            node_frames.append(nodes)
    result = base_graph.bind()
    if node_frames:
        concrete_engine = resolve_engine(cast(Any, engine), node_frames[0])
        concat = df_concat(concrete_engine)
        result._nodes = cast(DataFrameT, concat(node_frames, ignore_index=True))
    else:
        base_nodes = getattr(base_graph, "_nodes", None)
        result._nodes = cast(DataFrameT, base_nodes.iloc[0:0]) if base_nodes is not None else None
    result._edges = getattr(base_graph, "_edges", None)
    return result


def compiled_query_scalar_reentry_state(
    base_graph: Plottable,
    prefix_result: Plottable,
    *,
    carried_columns: Sequence[str],
    engine: Union[EngineAbstract, str],
    row_index: int = 0,
) -> Tuple[Plottable, Optional[DataFrameT]]:
    prefix_rows = getattr(prefix_result, "_nodes", None)
    if prefix_rows is None:
        raise reentry_validation_error(
            "Cypher MATCH after WITH scalar-only prefix stages could not recover prefix rows",
            value=None,
            suggestion="Project scalar columns directly before MATCH re-entry.",
        )
    prefix_row_count = len(prefix_rows)
    base_nodes = getattr(base_graph, "_nodes", None)
    if prefix_row_count == 0:
        if base_nodes is None:
            return base_graph, None
        dispatch_graph = base_graph.bind()
        dispatch_graph._nodes = cast(DataFrameT, base_nodes.iloc[0:0])
        edges_df = getattr(base_graph, "_edges", None)
        if edges_df is not None:
            dispatch_graph._edges = cast(DataFrameT, edges_df.iloc[0:0])
        return dispatch_graph, None
    if base_nodes is None:
        raise reentry_validation_error(
            "Cypher MATCH after WITH scalar-only prefix stages could not recover the base node table for re-entry",
            value=None,
            suggestion="Retry with a node-backed graph before MATCH re-entry.",
        )
    if not carried_columns:
        # Scalar-only prefix with zero carried scalars: keep the full node table.
        # Row fan-out/union for multi-row prefixes happens in the caller.
        dispatch_graph = base_graph.bind()
        dispatch_graph._nodes = base_nodes
        edges_df = getattr(base_graph, "_edges", None)
        if edges_df is not None:
            dispatch_graph._edges = edges_df
        return dispatch_graph, None
    missing_column = next((name for name in carried_columns if name not in prefix_rows.columns), None)
    if missing_column is not None:
        raise reentry_validation_error(
            "Cypher MATCH after WITH scalar-only prefix stages could not recover a carried scalar column from the prefix stage",
            value=missing_column,
            suggestion="Project the scalar column explicitly before MATCH re-entry.",
        )
    row = prefix_rows.iloc[row_index]
    node_rows = cast(
        DataFrameT,
        base_nodes.assign(
            **{
                _reentry_hidden_column_name(output_name): row[output_name]
                for output_name in carried_columns
            }
        ),
    )
    dispatch_graph = base_graph.bind()
    dispatch_graph._nodes = node_rows
    edges_df = getattr(base_graph, "_edges", None)
    if edges_df is not None:
        dispatch_graph._edges = edges_df
    return dispatch_graph, None


def freeform_broadcast_row_to_nodes(
    base_graph: Plottable,
    base_nodes: DataFrameT,
    prefix_rows: DataFrameT,
    plan: ReentryPlan,
    *,
    row_index: int,
) -> Plottable:
    """Build a dispatch graph for a single free-form prefix row.

    Broadcasts that row's carried hidden columns onto every base node so the
    trailing MATCH (running global, with `start_nodes=None`) inherits the
    carried values via the row pipeline. Used for both single-prefix-row and
    multi-prefix-row (#1285) free-form lanes.
    """
    row = prefix_rows.iloc[row_index]
    broadcast_values: Dict[str, Any] = {}
    # Top-level scalar carries (e.g. ``WITH a, b.id AS bid``): the prefix row
    # exposes them under their output names; the runtime hidden column on the
    # base node table is keyed by ``_reentry_hidden_column_name``.
    for col in plan.scalar_columns:
        if col in prefix_rows.columns:
            broadcast_values[_reentry_hidden_column_name(col)] = row[col]
    # Non-source whole-row property carries (slice 4.3b from #1248): the prefix
    # row already exposes these under their `__cypher_reentry_*` names; copy
    # them across as-is.
    for col in prefix_rows.columns:
        if isinstance(col, str) and col.startswith("__cypher_reentry_"):
            broadcast_values[col] = row[col]

    if broadcast_values:
        existing_hidden = [
            c for c in base_nodes.columns
            if isinstance(c, str) and c.startswith("__cypher_reentry_")
        ]
        node_rows = (
            cast(DataFrameT, base_nodes.drop(columns=existing_hidden))
            if existing_hidden
            else base_nodes
        )
        node_rows = cast(DataFrameT, node_rows.assign(**broadcast_values))
    else:
        node_rows = cast(DataFrameT, base_nodes)

    dispatch_graph = base_graph.bind()
    dispatch_graph._nodes = node_rows
    edges_df = getattr(base_graph, "_edges", None)
    if edges_df is not None:
        dispatch_graph._edges = edges_df
    return dispatch_graph


def compiled_query_freeform_reentry_state(
    base_graph: Plottable,
    compiled_query: CompiledCypherQuery,
    prefix_result: Plottable,
    *,
    engine: Union[EngineAbstract, str],
) -> Tuple[Plottable, Optional[DataFrameT]]:
    """#1263 free-form intermediate MATCH (LDBC SNB IC3 endpoint), single-row.

    The trailing MATCH binds aliases that are NOT in the prefix WITH's carried
    whole-row set, so it must run against the full base graph (no carried-id
    seed filter). Carried hidden columns from the prefix row are broadcast
    onto every base node so the row pipeline carries them through whichever
    alias the trailing MATCH binds; downstream WHERE/RETURN expressions
    referencing carried-alias properties resolve through those broadcast
    columns.

    Single-prefix-row dispatch only. Multi-prefix-row free-form (#1285) is
    handled at the caller via a per-row union loop (mirror of the scalar-only
    multi-row pattern at ``_execute_compiled_query_with_reentry``).
    """
    prefix_rows = getattr(prefix_result, "_nodes", None)
    base_nodes = getattr(base_graph, "_nodes", None)
    if base_nodes is None:
        raise reentry_validation_error(
            "Cypher MATCH after WITH (free-form intermediate MATCH; #1263) "
            "could not recover the base node table for re-entry",
            value=None,
            suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
        )
    if prefix_rows is None or len(prefix_rows) == 0:
        # Empty prefix → empty result. Return a graph with empty nodes/edges
        # so the suffix produces no rows.
        dispatch_graph = base_graph.bind()
        dispatch_graph._nodes = cast(DataFrameT, base_nodes.iloc[0:0])
        edges_df = getattr(base_graph, "_edges", None)
        if edges_df is not None:
            dispatch_graph._edges = cast(DataFrameT, edges_df.iloc[0:0])
        return dispatch_graph, None
    # Single-row dispatch only; the caller routes multi-row through the
    # per-row union loop in ``_execute_compiled_query_with_reentry``.
    if len(prefix_rows) > 1:
        raise reentry_validation_error(
            "Cypher MATCH after WITH (free-form intermediate MATCH) single-row "
            "dispatcher invoked with a multi-row prefix; the caller should "
            "route multi-row free-form through the per-row union loop.",
            value=len(prefix_rows),
            suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
        )

    plan = compiled_query.reentry_plan
    if plan is None:
        # Defensive: caller already gated on plan.free_form, so reaching here
        # without a plan is a programmer error.
        raise reentry_validation_error(
            "Cypher free-form intermediate MATCH dispatched without a ReentryPlan",
            value=None,
            suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
        )

    dispatch_graph = freeform_broadcast_row_to_nodes(
        base_graph, cast(DataFrameT, base_nodes), cast(DataFrameT, prefix_rows), plan, row_index=0,
    )
    return dispatch_graph, None


def aligned_reentry_rows(
    *,
    ids: SeriesT,
    prefix_rows: Optional[DataFrameT],
    output_name: Optional[str],
) -> Tuple[SeriesT, Optional[DataFrameT]]:
    if prefix_rows is not None and len(prefix_rows) != len(ids):
        raise reentry_validation_error(
            "Cypher MATCH after WITH metadata row counts disagreed with prefix rows during re-entry",
            value=output_name,
            suggestion="Retry with a direct whole-row carry through WITH or inspect intermediate row-shaping before MATCH re-entry.",
        )
    if not hasattr(ids, "notna"):
        raise reentry_validation_error(
            "Cypher MATCH after WITH could not align carried node identities from the prefix stage",
            value=output_name,
            suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
        )

    non_null_mask = cast(SeriesT, ids.notna())
    carried_ids = cast(SeriesT, ids[non_null_mask].reset_index(drop=True))
    if prefix_rows is None:
        return carried_ids, None
    return carried_ids, cast(DataFrameT, prefix_rows.loc[non_null_mask].reset_index(drop=True))


def reentry_carry_payload(
    *,
    carried_node_ids: DataFrameT,
    prefix_rows: DataFrameT,
    carried_columns: Sequence[str],
) -> DataFrameT:
    missing_column = next((name for name in carried_columns if name not in prefix_rows.columns), None)
    if missing_column is not None:
        raise reentry_validation_error(
            "Cypher MATCH after WITH could not recover a carried scalar column from the prefix stage",
            value=missing_column,
            suggestion="Project the scalar column explicitly before MATCH re-entry.",
        )
    return cast(
        DataFrameT,
        carried_node_ids.assign(
            **{
                _reentry_hidden_column_name(output_name): cast(SeriesT, prefix_rows[output_name]).reset_index(drop=True)
                for output_name in carried_columns
            }
        ),
    )


def ordered_reentry_start_nodes(
    *,
    node_rows: DataFrameT,
    carried_node_ids: DataFrameT,
    id_column: str,
) -> DataFrameT:
    # MATCH re-entry must preserve the WITH row order, not the base node-table order.
    return cast(DataFrameT, safe_merge(carried_node_ids, node_rows, on=id_column, how="left"))
