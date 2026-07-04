"""DataFrame execution helpers for bounded ``MATCH ... WITH ... MATCH`` reentry."""
# ruff: noqa: E501
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

from graphistry.Engine import EngineAbstract, df_concat, df_cons, resolve_engine, safe_merge
from graphistry.Plottable import Plottable
from graphistry.compute.exceptions import GFQLValidationError, ErrorCode
from graphistry.compute.gfql.cypher.reentry.naming import _reentry_hidden_column_name
from graphistry.compute.gfql.cypher.reentry_plan import ReentryPlan
from graphistry.compute.gfql.cypher.result_postprocess import (
    entity_projection_meta_entry,
)
from graphistry.compute.typing import DataFrameT, SeriesT


REENTRY_WHOLE_ROW_SUGGESTION = "Carry a whole-row node alias through WITH before MATCH re-entry."
REENTRY_SCALAR_SUGGESTION = "Carry scalar columns through WITH before MATCH re-entry."
REENTRY_DUPLICATE_CARRIED_ROWS_REASON = "duplicate_carried_node_rows"


def _is_polars_df(df: Any) -> bool:
    return df is not None and "polars" in type(df).__module__


def _reentry_row(prefix_rows: Any, row_index: int) -> Any:
    """One prefix row as a col->scalar mapping, engine-aware (``row[col]`` works for
    both the pandas Series and the polars named-row dict)."""
    if _is_polars_df(prefix_rows):
        return prefix_rows.row(row_index, named=True)
    return prefix_rows.iloc[row_index]


def _assign_constant_columns(df: Any, values: Dict[str, Any]) -> Any:
    """Broadcast scalar ``values`` as constant columns, engine-aware."""
    if not values:
        return df
    if _is_polars_df(df):
        import polars as pl
        return df.with_columns([pl.lit(v).alias(k) for k, v in values.items()])
    return df.assign(**values)


def _drop_columns(df: Any, cols: Sequence[str]) -> Any:
    if _is_polars_df(df):
        return df.drop(list(cols))
    return df.drop(columns=list(cols))


def _bind_reentry_graph(graph: Plottable, node_rows: Optional[DataFrameT], *, empty_edges: bool = False) -> Plottable:
    out = graph.bind()
    out._nodes = node_rows
    if empty_edges:
        edges_df = graph._edges
        if edges_df is not None:
            out._edges = cast(DataFrameT, edges_df.head(0) if _is_polars_df(edges_df) else edges_df.iloc[0:0])
    return out


def reentry_validation_error(
    message: str,
    *,
    value: Any,
    suggestion: str,
    field: str = "with",
    **extra_context: Any,
) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        message,
        field=field,
        value=value,
        suggestion=suggestion,
        language="cypher",
        **extra_context,
    )


def apply_optional_reentry_null_fill(
    result: Plottable,
    *,
    prefix_result: Plottable,
    engine: Union[EngineAbstract, str],
    empty_result_row: Optional[Dict[str, Any]] = None,
    reentry_plan: Optional[ReentryPlan] = None,
) -> Plottable:
    """Null-fill result rows for prefix rows that the optional reentry didn't match."""
    prefix_df = prefix_result._nodes
    result_df = result._nodes

    if prefix_df is None or len(prefix_df) == 0:
        return result

    prefix_rows = len(prefix_df)
    result_rows = 0 if result_df is None else len(result_df)

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

    fill_rows = _optional_reentry_carried_null_rows(
        prefix_df=prefix_df,
        result_df=result_df,
        null_row=null_row,
        reentry_plan=reentry_plan,
    )
    if fill_rows is None:
        if result_rows >= prefix_rows:
            return result
        missing_count = prefix_rows - result_rows
        fill_rows = [dict(null_row) for _ in range(missing_count)]
    elif not fill_rows:
        return result

    if result_df is None or len(result_df) == 0:
        return _bind_reentry_graph(result, df_ctor(fill_rows))

    fill_df = df_ctor(fill_rows)
    return _bind_reentry_graph(
        result,
        concat([result_df, fill_df], ignore_index=True, sort=False),
        empty_edges=True,
    )


def _optional_reentry_carried_null_rows(
    *,
    prefix_df: DataFrameT,
    result_df: Optional[DataFrameT],
    null_row: Dict[str, Any],
    reentry_plan: Optional[ReentryPlan],
) -> Optional[List[Dict[str, Any]]]:
    if reentry_plan is None or not reentry_plan.scalar_columns:
        return None

    prefix_columns = set(prefix_df.columns)
    result_columns = set(result_df.columns) if result_df is not None else set(null_row)
    carried_columns = tuple(
        col
        for col in reentry_plan.scalar_columns
        if col in prefix_columns and col in result_columns
    )
    if not carried_columns:
        return None

    prefix_records = _records_for_columns(prefix_df, carried_columns)
    prefix_keys = [_optional_reentry_key(record, carried_columns) for record in prefix_records]
    if len(set(prefix_keys)) != len(prefix_keys):
        return None
    if result_df is None or len(result_df) == 0:
        missing_records = prefix_records
    else:
        matched_keys = {
            _optional_reentry_key(record, carried_columns)
            for record in _records_for_columns(result_df, carried_columns)
        }
        missing_records = [
            record
            for record, key in zip(prefix_records, prefix_keys)
            if key not in matched_keys
        ]

    fill_rows: List[Dict[str, Any]] = []
    for record in missing_records:
        row = dict(null_row)
        for col in carried_columns:
            row[col] = record[col]
        fill_rows.append(row)
    return fill_rows


def _optional_reentry_key(record: Dict[str, Any], columns: Tuple[str, ...]) -> Tuple[Any, ...]:
    return tuple(_optional_reentry_key_value(record[col]) for col in columns)


def _records_for_columns(df: DataFrameT, columns: Tuple[str, ...]) -> List[Dict[str, Any]]:
    values_by_column = {column: _series_to_pylist(cast(SeriesT, df[column])) for column in columns}
    row_count = len(df)
    return [
        {column: values_by_column[column][row_index] for column in columns}
        for row_index in range(row_count)
    ]


def _series_to_pylist(values: SeriesT) -> List[Any]:
    if hasattr(values, "to_arrow"):
        return cast(List[Any], values.to_arrow().to_pylist())
    if hasattr(values, "tolist"):
        return cast(List[Any], values.tolist())
    return list(values)


def _optional_reentry_key_value(value: Any) -> Any:
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        return value
    return value


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
    prefix_rows = cast(Optional[DataFrameT], prefix_result._nodes)
    prefix_alias_values: Optional[SeriesT] = None
    if prefix_rows is not None and output_name in prefix_rows.columns:
        prefix_alias_values = cast(SeriesT, prefix_rows[output_name])
    entity_meta = cast(Optional[Dict[str, Any]], getattr(prefix_result, "_cypher_entity_projection_meta", None))
    has_projection_meta = isinstance(entity_meta, dict) and output_name in entity_meta
    has_secondary_carried_alias = any(not alias.is_reentry_alias for alias in plan.aliases)
    if (
        not has_projection_meta
        and has_secondary_carried_alias
        and prefix_alias_values is not None
        and hasattr(prefix_alias_values, "notna")
    ):
        # #1356: OPTIONAL prefix + no matches can produce a single null carry row
        # without whole-row projection metadata. Treat it as an empty seed set
        # for re-entry instead of raising "could not recover carried identities".
        non_null_mask = cast(SeriesT, prefix_alias_values.notna())
        has_non_null_ids = bool(non_null_mask.any()) if hasattr(non_null_mask, "any") else True
        if not has_non_null_ids:
            base_nodes = cast(Optional[DataFrameT], base_graph._nodes)
            id_column = cast(Optional[str], base_graph._node)
            if base_nodes is None or id_column is None or id_column not in base_nodes.columns:
                raise reentry_validation_error(
                    "Cypher MATCH after WITH could not recover the base node table for re-entry",
                    value=id_column,
                    suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
                )
            carried_node_ids = cast(
                DataFrameT,
                cast(DataFrameT, base_nodes[[id_column]]).iloc[0:0].reset_index(drop=True),
            )
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
    base_nodes = base_graph._nodes
    if base_nodes is None or id_column not in base_nodes.columns:
        raise reentry_validation_error(
            "Cypher MATCH after WITH could not recover the base node table for re-entry",
            value=id_column,
            suggestion=REENTRY_WHOLE_ROW_SUGGESTION,
        )
    concrete_engine = resolve_engine(cast(Any, engine), base_graph)
    carried_ids, aligned_prefix_rows = aligned_reentry_rows(
        ids=cast(SeriesT, ids),
        prefix_rows=prefix_result._nodes,
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
            reason=REENTRY_DUPLICATE_CARRIED_ROWS_REASON,
            carried_row_count=len(carried_ids),
            carried_scalar_columns=tuple(carried_columns),
        )

    carry_payload = reentry_carry_payload(
        carried_node_ids=carried_node_ids,
        prefix_rows=aligned_prefix_rows,
        carried_columns=carried_columns,
    )
    hidden_columns = [name for name in map(_reentry_hidden_column_name, carried_columns) if name in base_nodes.columns]
    merge_base = cast(DataFrameT, base_nodes.drop(columns=hidden_columns)) if hidden_columns else base_nodes
    node_rows = cast(DataFrameT, safe_merge(merge_base, carry_payload, on=id_column, how="left"))

    dispatch_graph = _bind_reentry_graph(base_graph, node_rows)
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
    """Union per-row suffix results from a multi-row scalar prefix."""
    node_frames = []
    for r in row_results:
        nodes = r._nodes
        if nodes is not None and len(cast(Any, nodes)) > 0:
            node_frames.append(nodes)
    if node_frames:
        concrete_engine = resolve_engine(cast(Any, engine), node_frames[0])
        concat = df_concat(concrete_engine)
        node_rows = cast(DataFrameT, concat(node_frames, ignore_index=True))
    else:
        base_nodes = base_graph._nodes
        node_rows = cast(DataFrameT, base_nodes.iloc[0:0]) if base_nodes is not None else None
    return _bind_reentry_graph(base_graph, node_rows)


def compiled_query_scalar_reentry_state(
    base_graph: Plottable,
    prefix_result: Plottable,
    *,
    carried_columns: Sequence[str],
    row_index: int = 0,
) -> Tuple[Plottable, Optional[DataFrameT]]:
    prefix_rows = prefix_result._nodes
    if prefix_rows is None:
        raise reentry_validation_error(
            "Cypher MATCH after WITH scalar-only prefix stages could not recover prefix rows",
            value=None,
            suggestion="Project scalar columns directly before MATCH re-entry.",
        )
    prefix_row_count = len(prefix_rows)
    base_nodes = base_graph._nodes
    if prefix_row_count == 0:
        if base_nodes is None:
            return base_graph, None
        return _bind_reentry_graph(
            base_graph,
            cast(DataFrameT, base_nodes.iloc[0:0]),
            empty_edges=True,
        ), None
    if base_nodes is None:
        raise reentry_validation_error(
            "Cypher MATCH after WITH scalar-only prefix stages could not recover the base node table for re-entry",
            value=None,
            suggestion="Retry with a node-backed graph before MATCH re-entry.",
        )
    if not carried_columns:
        # Scalar-only prefix with zero carried scalars: keep the full node table.
        # Row fan-out/union for multi-row prefixes happens in the caller.
        return _bind_reentry_graph(base_graph, base_nodes), None
    missing_column = next((name for name in carried_columns if name not in prefix_rows.columns), None)
    if missing_column is not None:
        raise reentry_validation_error(
            "Cypher MATCH after WITH scalar-only prefix stages could not recover a carried scalar column from the prefix stage",
            value=missing_column,
            suggestion="Project the scalar column explicitly before MATCH re-entry.",
        )
    row = _reentry_row(prefix_rows, row_index)
    node_rows = cast(
        DataFrameT,
        _assign_constant_columns(
            base_nodes,
            {
                _reentry_hidden_column_name(output_name): row[output_name]
                for output_name in carried_columns
            },
        ),
    )
    return _bind_reentry_graph(base_graph, node_rows), None


def freeform_broadcast_row_to_nodes(
    base_graph: Plottable,
    base_nodes: DataFrameT,
    prefix_rows: DataFrameT,
    plan: ReentryPlan,
    *,
    row_index: int,
) -> Plottable:
    """Broadcast one free-form prefix row's hidden carries onto the base nodes."""
    row = _reentry_row(prefix_rows, row_index)
    broadcast_values: Dict[str, Any] = {
        _reentry_hidden_column_name(col): row[col]
        for col in plan.scalar_columns
        if col in prefix_rows.columns
    }
    broadcast_values.update({
        col: row[col]
        for col in prefix_rows.columns
        if isinstance(col, str) and col.startswith("__cypher_reentry_")
    })

    if broadcast_values:
        existing_hidden = [c for c in base_nodes.columns if isinstance(c, str) and c.startswith("__cypher_reentry_")]
        node_rows = (
            cast(DataFrameT, _drop_columns(base_nodes, existing_hidden))
            if existing_hidden
            else base_nodes
        )
        node_rows = cast(DataFrameT, _assign_constant_columns(node_rows, broadcast_values))
    else:
        node_rows = cast(DataFrameT, base_nodes)

    return _bind_reentry_graph(base_graph, node_rows)


def compiled_query_freeform_reentry_state(
    base_graph: Plottable,
    prefix_result: Plottable,
    *,
    plan: ReentryPlan,
) -> Tuple[Plottable, Optional[DataFrameT]]:
    """Build the single-row dispatch state for free-form intermediate MATCH."""
    prefix_rows = prefix_result._nodes
    base_nodes = base_graph._nodes
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
        return _bind_reentry_graph(
            base_graph,
            cast(DataFrameT, base_nodes.iloc[0:0]),
            empty_edges=True,
        ), None
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
