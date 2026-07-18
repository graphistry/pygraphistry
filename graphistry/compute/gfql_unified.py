"""GFQL unified entrypoint for chains, DAGs, and local string-compiled queries."""
# ruff: noqa: E501

from dataclasses import replace
import pandas as pd
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union, cast
from graphistry.Plottable import Plottable
from graphistry.Engine import Engine, EngineAbstract, POLARS_ENGINES, df_concat, df_cons, df_to_engine, df_unique, resolve_engine
from graphistry.util import setup_logger
from .ast import ASTObject, ASTLet, ASTNode, ASTEdge, ASTCall
from .chain import Chain, chain as chain_impl
from .gfql.query_types import GFQLQuery
from .chain_let import chain_let as chain_let_impl
from .execution_context import ExecutionContext
from .gfql.policy import (
    CompileSummary,
    PolicyContext,
    PolicyException,
    PolicyFunction,
    PolicyDict,
    QueryType,
    expand_policy
)
from graphistry.compute.gfql.same_path_types import (
    NODE_IDENTITY_COLUMN,
    WhereComparison,
    normalize_where_entries,
    parse_where_json,
)
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.cypher.lowering import (
    ConnectedMatchJoinPlan,
    CompiledCypherGraphQuery,
    CompiledCypherQuery,
    CompiledCypherUnionQuery,
    CompiledGraphResidualFilter,
    ConnectedOptionalMatchPlan,
    compile_cypher_query,
)
from graphistry.compute.filter_by_dict import _node_dtypes_for_pushdown
from graphistry.compute.gfql.cypher.reentry.execution import (
    REENTRY_DUPLICATE_CARRIED_ROWS_REASON as _REENTRY_DUPLICATE_CARRIED_ROWS_REASON,
    REENTRY_WHOLE_ROW_SUGGESTION as _REENTRY_WHOLE_ROW_SUGGESTION,
    apply_optional_reentry_null_fill as _apply_optional_reentry_null_fill,
    compiled_query_freeform_reentry_state as _compiled_query_freeform_reentry_state,
    compiled_query_reentry_state as _compiled_query_reentry_state,
    compiled_query_scalar_reentry_state as _compiled_query_scalar_reentry_state,
    freeform_broadcast_row_to_nodes as _freeform_broadcast_row_to_nodes,
    reentry_validation_error as _reentry_validation_error,
    union_scalar_reentry_results as _union_scalar_reentry_results,
)
from graphistry.compute.gfql.cypher.call_procedures import execute_cypher_call
from graphistry.compute.gfql.cypher.result_postprocess import (
    apply_result_projection,
    entity_projection_meta_entry as _entity_projection_meta_entry,
)
from graphistry.compute.gfql.df_executor import (
    DFSamePathExecutor,
    build_same_path_inputs,
    execute_same_path_chain,
)
from graphistry.compute.dataframe import (
    binding_join_columns as _binding_join_columns,
    connected_inner_join_rows as _connected_inner_join_rows,
    joined_alias_columns as _joined_alias_columns,
    joined_hidden_scalar_columns as _joined_hidden_scalar_columns,
)
from graphistry.compute.filter_by_dict import filter_by_dict
from graphistry.compute.gfql.ir.compilation import PhysicalPlan, PlanContext
from graphistry.compute.gfql.ir.logical_plan import LogicalPlan
from graphistry.compute.gfql.physical_planner import PhysicalPlanner
from graphistry.compute.gfql.passes import DEFAULT_LOGICAL_PASSES, DEFAULT_TIER2_PASSES, PassManager
from graphistry.compute.gfql.row.pipeline import _RowPipelineAdapter, is_row_pipeline_call
from graphistry.compute.gfql.search_any import search_any_mask
from graphistry.compute.typing import DataFrameT, SeriesT, NodeDtypes
from graphistry.compute.util.generate_safe_column_name import generate_safe_column_name
from graphistry.compute.validate.validate_schema import validate_chain_schema
from graphistry.compute.gfql_validate import gfql_validate as gfql_preflight_validate
from graphistry.otel import otel_traced, otel_detail_enabled

logger = setup_logger(__name__)


def _series_to_pylist(values: Any) -> List[Any]:
    if hasattr(values, "to_arrow"):
        try:
            return list(values.to_arrow().to_pylist())
        except Exception:
            pass
    if hasattr(values, "to_pandas"):
        try:
            return list(values.to_pandas().tolist())
        except Exception:
            pass
    if hasattr(values, "tolist"):
        try:
            return list(values.tolist())
        except Exception:
            pass
    return list(values)


def _is_duplicate_carried_rows_reentry_error(exc: GFQLValidationError) -> bool:
    context = getattr(exc, "context", None)
    if exc.code != ErrorCode.E108 or not isinstance(context, dict):
        return False
    return context.get("reason") == _REENTRY_DUPLICATE_CARRIED_ROWS_REASON


def _slice_reentry_prefix_result_row(
    prefix_result: Plottable,
    *,
    output_name: str,
    row_index: int,
) -> Plottable:
    rows_df = cast(Optional[DataFrameT], prefix_result._nodes)
    if rows_df is None:
        return prefix_result
    out = prefix_result.bind()
    out._nodes = cast(DataFrameT, rows_df.iloc[row_index:row_index + 1].reset_index(drop=True))
    entity_meta = getattr(prefix_result, "_cypher_entity_projection_meta", None)
    if isinstance(entity_meta, dict):
        entry = entity_meta.get(output_name)
        if isinstance(entry, dict):
            sliced_entry = dict(entry)
            ids = sliced_entry.get("ids")
            if ids is not None and hasattr(ids, "iloc"):
                ids_obj = cast(Any, ids)
                sliced_entry["ids"] = cast(Any, ids_obj.iloc[row_index:row_index + 1]).reset_index(drop=True)
            setattr(out, "_cypher_entity_projection_meta", {output_name: sliced_entry})
    return out


def _apply_empty_result_row(
    result: Plottable,
    *,
    engine: Union[EngineAbstract, str],
    empty_result_row: Mapping[str, Any],
) -> Plottable:
    rows_df = result._nodes
    if rows_df is not None and len(rows_df) > 0:
        return result
    concrete_engine = resolve_engine(cast(Any, engine), result)
    df_ctor = df_cons(concrete_engine)
    out = result.bind()
    out._nodes = df_ctor({key: [value] for key, value in empty_result_row.items()})
    edges_df = result._edges
    if edges_df is not None:
        out._edges = edges_df[:0]
    return out


def _apply_optional_null_fill(
    result: Plottable,
    *,
    base_result: Plottable,
    alignment_result: Plottable,
    alignment_output_name: str,
    engine: Union[EngineAbstract, str],
    null_row: Mapping[str, Any],
) -> Plottable:
    base_rows_df = base_result._nodes
    expected_rows = 0 if base_rows_df is None else len(base_rows_df)
    if expected_rows == 0:
        return result

    rows_df = result._nodes
    actual_rows = 0 if rows_df is None else len(rows_df)
    # The null-fill alignment machinery below (matched-id meta, .iloc row slicing,
    # per-segment concat) is not yet native on polars: the polars OPTIONAL MATCH
    # does not populate the matched-seed `_cypher_entity_projection_meta["ids"]`
    # this path needs. Decline honestly (NO-CHEATING) rather than raising a
    # misleading "unsupported-cypher-query" validation error — pandas handles it.
    if resolve_engine(cast(Any, engine), result) in POLARS_ENGINES:
        meta = getattr(alignment_result, "_cypher_entity_projection_meta", None)
        if not isinstance(meta, dict) or alignment_output_name not in meta or "ids" not in meta[alignment_output_name]:
            raise NotImplementedError(
                "polars engine does not yet natively support this OPTIONAL MATCH "
                "null-row fill alignment shape; use engine='pandas' for this query "
                "(no pandas fallback; parity-or-error by design)"
            )
    matched_ids = _entity_projection_meta_entry(
        alignment_result,
        output_name=alignment_output_name,
        field="match",
        message="Cypher OPTIONAL MATCH null-row alignment could not recover matched seed identities",
        suggestion="Use a simpler OPTIONAL MATCH projection shape in the local compiler.",
    )["ids"]
    if not hasattr(matched_ids, "tolist"):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher OPTIONAL MATCH null-row alignment could not recover matched seed identities",
            field="match",
            value=alignment_output_name,
            suggestion="Use a simpler OPTIONAL MATCH projection shape in the local compiler.",
            language="cypher",
        )
    if actual_rows != len(matched_ids):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher OPTIONAL MATCH null-row alignment produced inconsistent row counts",
            field="match",
            value={"matched_rows": actual_rows, "aligned_ids": len(matched_ids)},
            suggestion="Retry with a simpler OPTIONAL MATCH projection shape in the local compiler.",
            language="cypher",
        )
    node_col = base_result._node
    if node_col is None or base_rows_df is None or node_col not in base_rows_df.columns:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher OPTIONAL MATCH null-row alignment could not recover base seed identities",
            field="match",
            value=node_col,
            suggestion="Use a simpler OPTIONAL MATCH projection shape in the local compiler.",
            language="cypher",
        )

    base_ids = _series_to_pylist(base_rows_df[node_col])
    matched_id_list = _series_to_pylist(matched_ids)
    if len(base_ids) == actual_rows and base_ids == matched_id_list:
        return result

    concrete_engine = resolve_engine(cast(Any, engine), result)
    df_ctor = df_cons(concrete_engine)
    concat = df_concat(concrete_engine)
    fill_df = df_ctor({key: [value] for key, value in null_row.items()})
    segments = []
    matched_idx = 0
    for base_id in base_ids:
        group_start = matched_idx
        while matched_idx < len(matched_id_list) and matched_id_list[matched_idx] == base_id:
            matched_idx += 1
        if matched_idx > group_start:
            if rows_df is None:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Cypher OPTIONAL MATCH null-row alignment lost the projected result rows",
                    field="match",
                    value=None,
                    suggestion="Retry with a simpler OPTIONAL MATCH projection shape in the local compiler.",
                    language="cypher",
                )
            segments.append(rows_df.iloc[group_start:matched_idx])
        else:
            segments.append(fill_df)
    if matched_idx != len(matched_id_list):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher OPTIONAL MATCH null-row alignment could not map matched rows back to the seed MATCH order",
            field="match",
            value={"mapped_rows": matched_idx, "matched_rows": len(matched_id_list)},
            suggestion="Use a simpler OPTIONAL MATCH projection shape in the local compiler.",
            language="cypher",
        )

    out = result.bind()
    out._nodes = concat(segments, ignore_index=True, sort=False) if segments else df_ctor()
    edges_df = result._edges
    if edges_df is not None:
        out._edges = edges_df[:0]
    return out


def _apply_optional_projection_row_guard(
    result: Plottable,
    *,
    expected_rows: int,
) -> Plottable:
    if expected_rows == 0:
        return result

    rows_df = result._nodes
    actual_rows = 0 if rows_df is None else len(rows_df)
    if actual_rows >= expected_rows:
        return result

    raise GFQLValidationError(
        ErrorCode.E108,
        "Cypher MATCH ... OPTIONAL MATCH projections over optional aliases would need null-extension rows that the local compiler cannot synthesize for this query shape",
        field="match",
        value={"expected_rows": expected_rows, "actual_rows": actual_rows},
        suggestion="Use a simpler OPTIONAL MATCH projection shape in the local compiler.",
        language="cypher",
    )


def _apply_connected_optional_match(
    base_graph: Plottable,
    plan: ConnectedOptionalMatchPlan,
    *,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
) -> Plottable:
    """Execute 1 non-optional MATCH + N OPTIONAL MATCH as chained left-outer-joins.

    1. Run the base chain with rows(binding_ops) to produce base binding rows.
    2. For each OPTIONAL MATCH arm, run its chain, left-outer-join onto the
       accumulated result on shared node aliases.
    3. Delegate RETURN / ORDER BY / SKIP / LIMIT to the standard row pipeline.
    """
    from graphistry.compute.ast import ASTCall, serialize_binding_ops

    def _split_binding_and_post_ops(ops: Sequence[ASTObject]) -> Tuple[List[ASTObject], List[ASTObject]]:
        """Split ops into contiguous binding path ops and post-row ops."""
        binding_ops: List[ASTObject] = []
        post_ops: List[ASTObject] = []
        saw_post = False

        for op in ops:
            is_binding = isinstance(op, (ASTNode, ASTEdge))
            if is_binding and not saw_post:
                binding_ops.append(op)
                continue
            saw_post = True
            post_ops.append(op)

        if not binding_ops:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Connected OPTIONAL MATCH lowering requires at least one ASTNode/ASTEdge binding op",
                field="match",
                value=[type(op).__name__ for op in ops],
                suggestion="Ensure MATCH/OPTIONAL MATCH clauses lower to path bindings before row-only operations.",
                language="cypher",
            )

        if any(isinstance(op, (ASTNode, ASTEdge)) for op in post_ops):
            raise GFQLValidationError(
                ErrorCode.E108,
                "Connected OPTIONAL MATCH lowering requires binding ops to be contiguous",
                field="match",
                value=[type(op).__name__ for op in ops],
                suggestion="Keep node/edge bindings contiguous; apply row-only operations after rows(binding_ops).",
                language="cypher",
            )

        return binding_ops, post_ops

    concrete_engine = resolve_engine(cast(Any, engine), base_graph)
    df_ctor = df_cons(concrete_engine)
    node_col = str(getattr(base_graph, "_node", "id"))

    def _optional_arm_start_nodes(
        binding_ops: Sequence[ASTObject],
        shared_node_aliases: Sequence[str],
        joined_rows: DataFrameT,
    ) -> Optional[DataFrameT]:
        """Seed optional-arm materialization when the first node is already bound."""
        if not binding_ops:
            return None
        first_op = binding_ops[0]
        if not isinstance(first_op, ASTNode):
            return None
        first_alias = getattr(first_op, "_name", None)
        if not isinstance(first_alias, str) or first_alias not in shared_node_aliases:
            return None

        base_nodes_raw = cast(Optional[DataFrameT], base_graph._nodes)
        base_nodes = None if base_nodes_raw is None else cast(DataFrameT, df_to_engine(base_nodes_raw, concrete_engine))
        if base_nodes is None or node_col not in base_nodes.columns:
            return None

        joined_col = next(
            (
                col
                for col in (f"{first_alias}.{node_col}", first_alias)
                if col in joined_rows.columns
            ),
            None,
        )
        if joined_col is None:
            return None

        seed_frame = cast(
            DataFrameT,
            df_to_engine(
                joined_rows[[joined_col]].dropna().drop_duplicates().rename(columns={joined_col: node_col}),
                concrete_engine,
            ),
        )
        seed_ids = cast(SeriesT, seed_frame[node_col])
        node_ids = cast(SeriesT, base_nodes[node_col])
        return cast(DataFrameT, base_nodes[node_ids.isin(seed_ids)].copy())

    # Run base chain to get binding rows.
    base_binding_chain, base_post_ops = _split_binding_and_post_ops(plan.base_chain.chain)
    base_binding_ops = serialize_binding_ops(base_binding_chain)
    base_with_rows = Chain(
        list(base_binding_chain) + [ASTCall("rows", {"binding_ops": base_binding_ops})] + base_post_ops,
        where=plan.base_chain.where,
    )
    base_rows_result = _chain_dispatch(base_graph, base_with_rows, engine, policy, context)
    joined = base_rows_result._nodes

    if joined is None or len(joined) == 0:
        out = base_graph.bind()
        out._nodes = df_ctor()
        out._edges = df_ctor()
        return out

    # Chained left-outer-join: one pass per OPTIONAL MATCH arm.
    for arm in plan.arms:
        opt_binding_chain, opt_post_ops = _split_binding_and_post_ops(arm.chain.chain)
        opt_binding_ops = serialize_binding_ops(opt_binding_chain)
        opt_with_rows = Chain(
            list(opt_binding_chain) + [ASTCall("rows", {"binding_ops": opt_binding_ops})] + opt_post_ops,
            where=arm.chain.where,
        )
        opt_start_nodes = None if arm.chain.where else _optional_arm_start_nodes(
            opt_binding_chain,
            arm.shared_node_aliases,
            joined,
        )
        opt_rows_result = _chain_dispatch(
            base_graph,
            opt_with_rows,
            engine,
            policy,
            context,
            start_nodes=opt_start_nodes,
        )
        opt_rows_df = opt_rows_result._nodes

        # Determine join columns from shared node aliases.
        join_cols = [
            f"{alias}.{node_col}"
            for alias in arm.shared_node_aliases
            if f"{alias}.{node_col}" in joined.columns
            and opt_rows_df is not None
            and f"{alias}.{node_col}" in opt_rows_df.columns
        ]
        if not join_cols:
            join_cols = [
                alias for alias in arm.shared_node_aliases
                if alias in joined.columns
                and opt_rows_df is not None
                and alias in opt_rows_df.columns
            ]

        if opt_rows_df is not None and len(opt_rows_df) > 0 and join_cols:
            opt_only_cols = [c for c in opt_rows_df.columns if c not in joined.columns or c in join_cols]
            # Semi-join filter: restrict opt rows to join-key values present in base
            # result before materialization. Prevents cross-product blowup when the
            # OPTIONAL MATCH arm produces far more rows than the base MATCH. (#1052)
            if len(join_cols) == 1:
                jc = join_cols[0]
                opt_rows_df = opt_rows_df[opt_rows_df[jc].isin(joined[jc])]
            else:
                join_keys = joined[join_cols].drop_duplicates()
                opt_rows_df = opt_rows_df.merge(join_keys, on=join_cols, how="inner")
            joined = joined.merge(opt_rows_df[opt_only_cols], on=join_cols, how="left")
        else:
            for alias in arm.opt_only_aliases:
                if alias not in joined.columns:
                    joined[alias] = None

        # Synthesize bare alias columns for edge aliases in this arm.
        for alias in arm.opt_only_aliases:
            if alias in joined.columns:
                continue
            prefix = f"{alias}."
            marker_col = next((c for c in joined.columns if c.startswith(prefix)), None)
            if marker_col is not None:
                marker = joined[marker_col]
                joined[alias] = marker.where(marker.notna(), other=None)

    # Delegate RETURN / ORDER BY / SKIP / LIMIT to the standard row pipeline.
    joined_plottable = base_graph.bind()
    joined_plottable._nodes = joined
    joined_plottable._edges = df_ctor()

    return _chain_dispatch(joined_plottable, plan.post_join_chain, engine, policy, context)



def _is_connected_fast_single_hop(edge_op: ASTEdge) -> bool:
    return (
        getattr(edge_op, "direction", None) == "forward"
        and getattr(edge_op, "hops", None) in (None, 1)
        and getattr(edge_op, "min_hops", None) is None
        and getattr(edge_op, "max_hops", None) is None
        and getattr(edge_op, "output_min_hops", None) is None
        and getattr(edge_op, "output_max_hops", None) is None
        and getattr(edge_op, "label_node_hops", None) is None
        and getattr(edge_op, "label_edge_hops", None) is None
        and not bool(getattr(edge_op, "label_seeds", False))
        and not bool(getattr(edge_op, "to_fixed_point", False))
        and getattr(edge_op, "source_node_match", None) is None
        and getattr(edge_op, "destination_node_match", None) is None
        and getattr(edge_op, "source_node_query", None) is None
        and getattr(edge_op, "destination_node_query", None) is None
        and getattr(edge_op, "edge_query", None) is None
        and getattr(edge_op, "_name", None) is None
        and not bool(getattr(edge_op, "include_zero_hop_seed", False))
    )




_CACHE_MISSING = object()


def _connected_join_filter_value_cache_key(value: Any) -> Optional[Tuple[str, str]]:
    if isinstance(value, bool):
        return "bool", "true" if value else "false"
    if isinstance(value, str):
        return "str", value
    if isinstance(value, int) and not isinstance(value, bool):
        return "int", str(value)
    if isinstance(value, float):
        return "float", repr(value)
    if value is None:
        return "none", ""

    predicate_value = getattr(value, "val", _CACHE_MISSING)
    if predicate_value is not _CACHE_MISSING:
        scalar_key = _connected_join_filter_value_cache_key(predicate_value)
        if scalar_key is None:
            return None
        return f"{type(value).__module__}.{type(value).__qualname__}", f"{scalar_key[0]}:{scalar_key[1]}"

    regex_pattern = getattr(value, "pat", _CACHE_MISSING)
    regex_case = getattr(value, "case", _CACHE_MISSING)
    regex_flags = getattr(value, "flags", _CACHE_MISSING)
    regex_na = getattr(value, "na", _CACHE_MISSING)
    if (
        regex_pattern is not _CACHE_MISSING
        and regex_case is not _CACHE_MISSING
        and regex_flags is not _CACHE_MISSING
        and regex_na is not _CACHE_MISSING
    ):
        pattern_key = _connected_join_filter_value_cache_key(regex_pattern)
        case_key = _connected_join_filter_value_cache_key(regex_case)
        flags_key = _connected_join_filter_value_cache_key(regex_flags)
        na_key = _connected_join_filter_value_cache_key(regex_na)
        if pattern_key is None or case_key is None or flags_key is None or na_key is None:
            return None
        return (
            f"{type(value).__module__}.{type(value).__qualname__}",
            f"pat={pattern_key[0]}:{pattern_key[1]}|case={case_key[0]}:{case_key[1]}|flags={flags_key[0]}:{flags_key[1]}|na={na_key[0]}:{na_key[1]}",
        )

    predicates = getattr(value, "predicates", _CACHE_MISSING)
    if predicates is not _CACHE_MISSING and isinstance(predicates, (list, tuple)):
        child_keys = []
        for predicate in predicates:
            child_key = _connected_join_filter_value_cache_key(predicate)
            if child_key is None:
                return None
            child_keys.append(f"{child_key[0]}:{child_key[1]}")
        return f"{type(value).__module__}.{type(value).__qualname__}", "|".join(child_keys)

    return None


def _connected_join_simple_filter_cache_key(filter_dict: Optional[dict]) -> Optional[Tuple[Tuple[str, str, str], ...]]:
    if not filter_dict:
        return ()
    items: List[Tuple[str, str, str]] = []
    for key, value in filter_dict.items():
        if not isinstance(key, str):
            return None
        value_key = _connected_join_filter_value_cache_key(value)
        if value_key is None:
            return None
        items.append((key, value_key[0], value_key[1]))
    return tuple(sorted(items))


def _connected_join_cached_node_filter(
    base_graph: Plottable,
    nodes_obj: DataFrameT,
    node_match: Optional[dict],
    *,
    engine: Engine,
) -> DataFrameT:
    cache_key = _connected_join_simple_filter_cache_key(node_match)
    if cache_key is None:
        nodes = df_to_engine(nodes_obj, engine)
        if engine in POLARS_ENGINES:
            from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars
            return cast(DataFrameT, filter_by_dict_polars(nodes, node_match))
        return filter_by_dict(nodes, node_match, engine=EngineAbstract(engine.value))

    cache_attr = "_gfql_connected_join_node_filter_cache"
    cache = getattr(base_graph, cache_attr, None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(base_graph, cache_attr, cache)
        except Exception:
            cache = None
    full_key = (id(nodes_obj), engine.value, cache_key)
    if cache is not None and full_key in cache:
        return cast(DataFrameT, cache[full_key])

    nodes = df_to_engine(nodes_obj, engine)
    if engine in POLARS_ENGINES:
        from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars
        filtered = cast(DataFrameT, filter_by_dict_polars(nodes, node_match))
    else:
        filtered = filter_by_dict(nodes, node_match, engine=EngineAbstract(engine.value))
    if cache is not None:
        cache[full_key] = filtered
    return cast(DataFrameT, filtered)


def _connected_join_cached_node_ids(
    base_graph: Plottable,
    nodes_obj: DataFrameT,
    node_matches: Sequence[Optional[dict]],
    *,
    node_col: str,
    engine: Engine,
) -> Optional[DataFrameT]:
    if engine not in POLARS_ENGINES:
        return None

    match_keys: List[Tuple[Tuple[str, str, str], ...]] = []
    for node_match in node_matches:
        match_key = _connected_join_simple_filter_cache_key(node_match)
        if match_key is None:
            return None
        match_keys.append(match_key)

    cache_attr = "_gfql_connected_join_node_ids_cache"
    cache = getattr(base_graph, cache_attr, None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(base_graph, cache_attr, cache)
        except Exception:
            cache = None
    full_key = (id(nodes_obj), engine.value, node_col, tuple(match_keys))
    if cache is not None and full_key in cache:
        return cast(DataFrameT, cache[full_key])

    from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars

    nodes = df_to_engine(nodes_obj, engine)
    filtered = nodes
    for node_match in node_matches:
        filtered = cast(DataFrameT, filter_by_dict_polars(filtered, node_match))
    ids = cast(DataFrameT, filtered.select(node_col).unique())
    if cache is not None:
        cache[full_key] = ids
    return ids


def _connected_join_cached_edge_filter(
    base_graph: Plottable,
    edges_obj: DataFrameT,
    edge_match: Optional[dict],
    *,
    engine: Engine,
) -> DataFrameT:
    cache_key = _connected_join_simple_filter_cache_key(edge_match)
    if cache_key is None:
        edges = df_to_engine(edges_obj, engine)
        if engine in POLARS_ENGINES:
            from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars
            return cast(DataFrameT, filter_by_dict_polars(edges, edge_match))
        return filter_by_dict(edges, edge_match, engine=EngineAbstract(engine.value))

    cache_attr = "_gfql_connected_join_edge_filter_cache"
    cache = getattr(base_graph, cache_attr, None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(base_graph, cache_attr, cache)
        except Exception:
            cache = None
    full_key = (id(edges_obj), engine.value, cache_key)
    if cache is not None and full_key in cache:
        return cast(DataFrameT, cache[full_key])

    edges = df_to_engine(edges_obj, engine)
    if engine in POLARS_ENGINES:
        from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars
        filtered = cast(DataFrameT, filter_by_dict_polars(edges, edge_match))
    else:
        filtered = filter_by_dict(edges, edge_match, engine=EngineAbstract(engine.value))
    if cache is not None:
        cache[full_key] = filtered
    return cast(DataFrameT, filtered)


def _connected_join_cached_singleton_dst_source_counts(
    base_graph: Plottable,
    edges_obj: DataFrameT,
    edge_domain: DataFrameT,
    edge_match: Optional[dict],
    singleton_dst: Any,
    *,
    src_col: str,
    dst_col: str,
    engine: Engine,
) -> Optional[DataFrameT]:
    edge_key = _connected_join_simple_filter_cache_key(edge_match)
    dst_key = _connected_join_filter_value_cache_key(singleton_dst)
    if edge_key is None or dst_key is None:
        return None

    cache_attr = "_gfql_connected_join_singleton_dst_source_counts_cache"
    cache = getattr(base_graph, cache_attr, None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(base_graph, cache_attr, cache)
        except Exception:
            cache = None
    full_key = (id(edges_obj), engine.value, src_col, dst_col, edge_key, dst_key)
    if cache is not None and full_key in cache:
        return cast(DataFrameT, cache[full_key])

    if engine in POLARS_ENGINES:
        import polars as pl
        counts = cast(
            DataFrameT,
            edge_domain
            .filter(pl.col(dst_col) == singleton_dst)
            .group_by(src_col)
            .len("__left_count__"),
        )
    else:
        filtered = edge_domain[edge_domain[dst_col] == singleton_dst]
        counts = cast(DataFrameT, filtered.groupby(src_col, sort=False).size().reset_index(name="__left_count__"))

    if cache is not None:
        cache[full_key] = counts
    return counts


def _connected_join_cached_first_arm_shared_counts(
    base_graph: Plottable,
    nodes_obj: DataFrameT,
    edges_obj: DataFrameT,
    first_edges: DataFrameT,
    shared_ids: DataFrameT,
    first_edge_match: Optional[dict],
    first_start_match: Optional[dict],
    second_start_match: Optional[dict],
    singleton_dst: Any,
    *,
    shared_alias: str,
    src_col: str,
    dst_col: str,
    engine: Engine,
) -> Optional[DataFrameT]:
    if engine not in POLARS_ENGINES:
        return None

    edge_key = _connected_join_simple_filter_cache_key(first_edge_match)
    first_start_key = _connected_join_simple_filter_cache_key(first_start_match)
    second_start_key = _connected_join_simple_filter_cache_key(second_start_match)
    dst_key = _connected_join_filter_value_cache_key(singleton_dst)
    if edge_key is None or first_start_key is None or second_start_key is None or dst_key is None:
        return None

    cache_attr = "_gfql_connected_join_first_arm_shared_counts_cache"
    cache = getattr(base_graph, cache_attr, None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(base_graph, cache_attr, cache)
        except Exception:
            cache = None
    full_key = (
        id(nodes_obj),
        id(edges_obj),
        engine.value,
        shared_alias,
        src_col,
        dst_col,
        edge_key,
        first_start_key,
        second_start_key,
        dst_key,
    )
    if cache is not None and full_key in cache:
        return cast(DataFrameT, cache[full_key])

    import polars as pl

    counts = (
        first_edges
        .filter(pl.col(dst_col) == singleton_dst)
        .join(shared_ids, left_on=src_col, right_on=shared_alias, how="semi")
        .group_by(src_col)
        .len("__left_count__")
        .rename({src_col: shared_alias})
    )
    if cache is not None:
        cache[full_key] = counts
    return cast(DataFrameT, counts)


def _connected_join_cached_second_arm_group_rows(
    base_graph: Plottable,
    nodes_obj: DataFrameT,
    edges_obj: DataFrameT,
    second_edges: DataFrameT,
    shared_ids: DataFrameT,
    second_leaf_ids: DataFrameT,
    second_leaf_nodes: DataFrameT,
    first_start_match: Optional[dict],
    second_start_match: Optional[dict],
    second_leaf_match: Optional[dict],
    second_edge_match: Optional[dict],
    second_leaf_singleton: Any,
    group_prop_refs: List[Tuple[str, str]],
    output_group_keys: List[str],
    *,
    node_col: str,
    src_col: str,
    dst_col: str,
    engine: Engine,
) -> Optional[DataFrameT]:
    if engine not in POLARS_ENGINES:
        return None

    first_start_key = _connected_join_simple_filter_cache_key(first_start_match)
    second_start_key = _connected_join_simple_filter_cache_key(second_start_match)
    second_leaf_key = _connected_join_simple_filter_cache_key(second_leaf_match)
    second_edge_key = _connected_join_simple_filter_cache_key(second_edge_match)
    if first_start_key is None or second_start_key is None or second_leaf_key is None or second_edge_key is None:
        return None

    prop_key = tuple((str(out_col), str(prop)) for out_col, prop in group_prop_refs)
    output_key = tuple(str(key) for key in output_group_keys)
    cache_attr = "_gfql_connected_join_second_arm_group_rows_cache"
    cache = getattr(base_graph, cache_attr, None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(base_graph, cache_attr, cache)
        except Exception:
            cache = None
    full_key = (
        id(nodes_obj),
        id(edges_obj),
        engine.value,
        node_col,
        src_col,
        dst_col,
        first_start_key,
        second_start_key,
        second_leaf_key,
        second_edge_key,
        prop_key,
        output_key,
    )
    if cache is not None and full_key in cache:
        return cast(DataFrameT, cache[full_key])

    import polars as pl

    if second_leaf_singleton is _CACHE_MISSING:
        right_base = (
            second_edges
            .join(shared_ids, left_on=src_col, right_on=node_col, how="semi")
            .join(second_leaf_ids, left_on=dst_col, right_on=node_col, how="semi")
        )
    else:
        right_base = (
            second_edges
            .filter(pl.col(dst_col) == second_leaf_singleton)
            .join(shared_ids, left_on=src_col, right_on=node_col, how="semi")
        )
    if group_prop_refs:
        lookup_key = "__gfql_fast_second_leaf_id__"
        lookup_exprs = [pl.col(node_col).alias(lookup_key)] + [pl.col(prop).alias(out_col) for out_col, prop in group_prop_refs]
        second_lookup = second_leaf_nodes.select(lookup_exprs)
        right_base = right_base.join(second_lookup, left_on=dst_col, right_on=lookup_key, how="inner")
    right_rows = right_base.select([pl.col(src_col)] + [pl.col(key) for key in output_group_keys])
    if cache is not None:
        cache[full_key] = right_rows
    return cast(DataFrameT, right_rows)


def _property_ref(expr: Any, valid_aliases: Sequence[str]) -> Optional[Tuple[str, str]]:
    if not isinstance(expr, str) or "." not in expr:
        return None
    alias, prop = expr.split(".", 1)
    if alias not in valid_aliases or not prop:
        return None
    return alias, prop


def _connected_join_post_property_columns(plan: ConnectedMatchJoinPlan, alias: str) -> List[str]:
    prefix = f"{alias}."
    out: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if value.startswith(prefix):
                prop = value[len(prefix):]
                if prop and prop not in out:
                    out.append(prop)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                walk(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for op in plan.post_join_chain.chain:
        if isinstance(op, ASTCall):
            walk(op.params)
    return out


def _connected_join_two_star_split_residuals(
    plan: ConnectedMatchJoinPlan,
    alias_targets: Mapping[str, ASTObject],
    materialized_aliases: Set[str],
) -> Optional[Tuple[Dict[str, List[str]], "Chain"]]:
    """Split leading post-join ``where_rows`` residuals by single node alias.

    #1729's connected-join lowering emits row predicates it cannot push into ``filter_dict``
    (e.g. ``toLower(i.interest) = toLower('fine dining')``) as leading ``where_rows`` ops in
    ``post_join_chain``. The structural fast paths cannot apply a residual to the aggregated
    frame, but a residual that references exactly ONE node alias the fast path materializes can
    be applied to that alias's node set before counting. Returns ``(alias -> [expr, ...],
    remaining_chain)`` when every leading residual is so attributable, or ``None`` when any
    leading residual spans >1 alias, references a non-materialized alias, or is not a bare
    ``expr`` residual -- the caller then declines to the slow path.
    """
    from graphistry.compute.gfql.cypher.lowering import _expr_match_aliases

    ops = list(plan.post_join_chain.chain)
    residuals: Dict[str, List[str]] = {}
    consumed = 0
    for op in ops:
        if not (isinstance(op, ASTCall) and op.function == "where_rows"):
            break
        params = op.params or {}
        expr = params.get("expr")
        if params.get("filter_dict") or not isinstance(expr, str):
            return None
        try:
            aliases = _expr_match_aliases(
                expr, alias_targets=alias_targets, params=None, field="where", line=0, column=0
            )
        except Exception:
            return None
        attributable = {alias for alias in aliases if alias in materialized_aliases}
        if len(aliases) != 1 or len(attributable) != 1:
            return None
        residuals.setdefault(next(iter(attributable)), []).append(expr)
        consumed += 1
    rest = Chain(ops[consumed:], where=plan.post_join_chain.where)
    return residuals, rest


def _connected_join_apply_node_residuals(
    base_graph: Plottable,
    node_frame: DataFrameT,
    alias: str,
    exprs: Sequence[str],
    node_col: str,
    *,
    engine: Engine,
) -> DataFrameT:
    """Filter a fast-path node frame by single-alias post-join residual expressions.

    Reuses the row pipeline's ``where_rows`` evaluator (identical semantics to the slow path,
    so toLower/etc. behave exactly as they would post-join) by aliasing the node columns to
    ``alias.col`` and dispatching a where_rows chain, then renaming back. ``validate_schema`` is
    disabled because the residual references flat ``alias.col`` columns rather than a bound
    graph element.
    """
    is_polars = "polars" in type(node_frame).__module__
    if is_polars:
        aliased = node_frame.rename({col: f"{alias}.{col}" for col in node_frame.columns})
    else:
        aliased = node_frame.rename(columns={col: f"{alias}.{col}" for col in node_frame.columns})
    from graphistry.compute.chain import chain as _chain_fn

    aliased_graph = base_graph.nodes(aliased, f"{alias}.{node_col}")
    filtered_graph = _chain_fn(
        aliased_graph,
        [ASTCall("where_rows", {"expr": expr}) for expr in exprs],
        engine=EngineAbstract(engine.value),
        validate_schema=False,
    )
    filtered = cast(DataFrameT, filtered_graph._nodes)
    if is_polars:
        return cast(DataFrameT, filtered.rename({f"{alias}.{col}": col for col in node_frame.columns}))
    return cast(DataFrameT, filtered.rename(columns={f"{alias}.{col}": col for col in node_frame.columns}))


def _connected_join_filter_node_frames_by_residuals(
    base_graph: Plottable,
    residual_map: Mapping[str, Sequence[str]],
    frames: Mapping[str, DataFrameT],
    node_col: str,
    *,
    engine: Engine,
) -> Dict[str, DataFrameT]:
    """Apply each alias's post-join residuals to its materialized node frame."""
    out: Dict[str, DataFrameT] = dict(frames)
    for alias, exprs in residual_map.items():
        if alias in out and exprs:
            out[alias] = _connected_join_apply_node_residuals(
                base_graph, out[alias], alias, exprs, node_col, engine=engine
            )
    return out


def _connected_join_two_star_fast_grouped_count(
    base_graph: Plottable,
    plan: ConnectedMatchJoinPlan,
    *,
    engine: Engine,
) -> Optional[DataFrameT]:
    if len(plan.pattern_chains) != 2 or len(plan.pattern_shared_node_aliases) != 1:
        return None
    if len(plan.pattern_shared_node_aliases[0]) != 1:
        return None
    shared_alias = plan.pattern_shared_node_aliases[0][0]

    parsed = []
    for pattern_chain in plan.pattern_chains:
        if pattern_chain.where:
            return None
        ops = list(pattern_chain.chain)
        if len(ops) != 3:
            return None
        start_op, edge_op, end_op = ops
        if not isinstance(start_op, ASTNode) or not isinstance(edge_op, ASTEdge) or not isinstance(end_op, ASTNode):
            return None
        if getattr(start_op, "query", None) is not None or getattr(end_op, "query", None) is not None:
            return None
        if not _is_connected_fast_single_hop(edge_op):
            return None
        start_alias = getattr(start_op, "_name", None)
        end_alias = getattr(end_op, "_name", None)
        if start_alias != shared_alias or not isinstance(end_alias, str):
            return None
        parsed.append((start_op, edge_op, end_op, end_alias))

    first_start, first_edge, first_end, first_end_alias = parsed[0]
    second_start, second_edge, second_end, second_end_alias = parsed[1]
    if first_end_alias == second_end_alias:
        return None

    alias_targets: Dict[str, ASTObject] = {
        shared_alias: first_start,
        first_end_alias: first_end,
        second_end_alias: second_end,
    }
    materialized_aliases = {shared_alias, first_end_alias, second_end_alias}
    split = _connected_join_two_star_split_residuals(plan, alias_targets, materialized_aliases)
    if split is None:
        return None
    residual_map, rest_chain = split

    post_ops = [op for op in rest_chain.chain if isinstance(op, ASTCall)]
    if len(post_ops) not in (2, 3, 4):
        return None
    if post_ops[0].function != "with_" or post_ops[1].function != "group_by":
        return None
    suffix = post_ops[2:]
    order_call: Optional[ASTCall] = None
    limit_call: Optional[ASTCall] = None
    select_call: Optional[ASTCall] = None
    for op in suffix:
        if op.function == "order_by" and order_call is None and limit_call is None and select_call is None:
            order_call = op
        elif op.function == "limit" and limit_call is None and select_call is None:
            limit_call = op
        elif op.function == "select" and select_call is None:
            select_call = op
        else:
            return None

    with_items_raw = post_ops[0].params.get("items")
    group_keys_raw = post_ops[1].params.get("keys")
    aggs_raw = post_ops[1].params.get("aggregations")
    if not isinstance(with_items_raw, list) or not isinstance(group_keys_raw, list) or not isinstance(aggs_raw, list):
        return None
    if len(aggs_raw) != 1:
        return None
    agg = aggs_raw[0]
    if not isinstance(agg, (tuple, list)) or len(agg) != 3 or str(agg[1]).lower() != "count":
        return None
    agg_alias = str(agg[0])
    agg_input = agg[2]

    with_items: Dict[str, Any] = {}
    for item in with_items_raw:
        if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[0], str):
            return None
        with_items[item[0]] = item[1]
    # count(p) over the shared alias now lowers `p` to its identity column
    # `p.__gfql_node_id__` (see _connected_join_alias_identity_expr); accept either the
    # bare shared alias or that identity form -- the fast count is shared-node multiplicity
    # regardless, so the identity rewrite does not change the computation.
    shared_identity = f"{shared_alias}.{NODE_IDENTITY_COLUMN}"
    if not isinstance(agg_input, str) or with_items.get(agg_input) not in (shared_alias, shared_identity):
        return None

    group_keys = [str(key) for key in group_keys_raw]
    group_prop_refs: List[Tuple[str, str]] = []
    output_group_keys: List[str] = []
    for key in group_keys:
        expr = with_items.get(key)
        if key == "__cypher_group__" and expr == 1:
            continue
        prop_ref = _property_ref(expr, (second_end_alias,))
        if prop_ref is None:
            return None
        output_group_keys.append(key)
        group_prop_refs.append((key, prop_ref[1]))

    order_keys: List[Tuple[str, bool]] = []
    if order_call is not None:
        raw_order = order_call.params.get("keys")
        if not isinstance(raw_order, list):
            return None
        available = set(output_group_keys) | {agg_alias}
        for item in raw_order:
            if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[0], str):
                return None
            if item[0] not in available:
                return None
            direction = str(item[1]).lower()
            if direction not in {"asc", "ascending", "desc", "descending"}:
                return None
            order_keys.append((item[0], direction in {"desc", "descending"}))

    limit_value: Optional[int] = None
    if limit_call is not None:
        raw_limit = limit_call.params.get("value")
        if not isinstance(raw_limit, int) or raw_limit < 0:
            return None
        limit_value = raw_limit

    select_items: Optional[List[Tuple[str, str]]] = None
    if select_call is not None:
        raw_select = select_call.params.get("items")
        if not isinstance(raw_select, list):
            return None
        select_items = []
        available = set(output_group_keys) | {agg_alias}
        for item in raw_select:
            if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[0], str) or not isinstance(item[1], str):
                return None
            if item[0] not in available:
                return None
            select_items.append((item[0], item[1]))

    nodes_obj = getattr(base_graph, "_nodes", None)
    edges_obj = getattr(base_graph, "_edges", None)
    node_col = getattr(base_graph, "_node", None)
    src_col = getattr(base_graph, "_source", None)
    dst_col = getattr(base_graph, "_destination", None)
    if nodes_obj is None or edges_obj is None or node_col is None or src_col is None or dst_col is None:
        return None
    node_col = str(node_col)
    src_col = str(src_col)
    dst_col = str(dst_col)
    if node_col not in nodes_obj.columns or src_col not in edges_obj.columns or dst_col not in edges_obj.columns:
        return None

    nodes_source = cast(DataFrameT, nodes_obj)
    nodes = df_to_engine(nodes_source, engine)
    edges = cast(DataFrameT, edges_obj)

    if engine in POLARS_ENGINES:
        import polars as pl
        from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars

        # Residual node filters (e.g. toLower(...)) must be applied to the materialized node
        # frames, so residual queries use the direct (non-cached) path -- the id/count caches
        # re-derive node sets from filter_dict alone and would drop the residual.
        if isinstance(nodes_obj, (pl.DataFrame, pl.LazyFrame)) and not residual_map:
            shared_ids = _connected_join_cached_node_ids(
                base_graph,
                nodes_source,
                [cast(Optional[dict], first_start.filter_dict), cast(Optional[dict], second_start.filter_dict)],
                node_col=node_col,
                engine=engine,
            )
            first_leaf_ids = _connected_join_cached_node_ids(
                base_graph,
                nodes_source,
                [cast(Optional[dict], first_end.filter_dict)],
                node_col=node_col,
                engine=engine,
            )
            second_leaf_ids = _connected_join_cached_node_ids(
                base_graph,
                nodes_source,
                [cast(Optional[dict], second_end.filter_dict)],
                node_col=node_col,
                engine=engine,
            )
            first_leaf_nodes = _connected_join_cached_node_filter(base_graph, nodes_source, cast(Optional[dict], first_end.filter_dict), engine=engine)
            second_leaf_nodes = _connected_join_cached_node_filter(base_graph, nodes_source, cast(Optional[dict], second_end.filter_dict), engine=engine)
            if shared_ids is None or first_leaf_ids is None or second_leaf_ids is None:
                return None
        else:
            shared_nodes = filter_by_dict_polars(nodes, cast(Optional[dict], first_start.filter_dict))
            shared_nodes = filter_by_dict_polars(shared_nodes, cast(Optional[dict], second_start.filter_dict))
            first_leaf_nodes = filter_by_dict_polars(nodes, cast(Optional[dict], first_end.filter_dict))
            second_leaf_nodes = filter_by_dict_polars(nodes, cast(Optional[dict], second_end.filter_dict))
            if residual_map:
                _frames = _connected_join_filter_node_frames_by_residuals(
                    base_graph,
                    residual_map,
                    {shared_alias: shared_nodes, first_end_alias: first_leaf_nodes, second_end_alias: second_leaf_nodes},
                    node_col,
                    engine=engine,
                )
                shared_nodes, first_leaf_nodes, second_leaf_nodes = _frames[shared_alias], _frames[first_end_alias], _frames[second_end_alias]
            shared_ids = shared_nodes.select(node_col).unique()
            first_leaf_ids = first_leaf_nodes.select(node_col).unique()
            second_leaf_ids = second_leaf_nodes.select(node_col).unique()
        first_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], first_edge.edge_match), engine=engine)
        second_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], second_edge.edge_match), engine=engine)

        for _, prop in group_prop_refs:
            if prop not in second_leaf_nodes.columns:
                return None

        def singleton_id(ids: Any) -> Any:
            if not isinstance(ids, pl.DataFrame) or len(ids) != 1:
                return _CACHE_MISSING
            return ids.get_column(node_col)[0]

        first_leaf_singleton = singleton_id(first_leaf_ids)
        second_leaf_singleton = singleton_id(second_leaf_ids)

        cached_left_counts: Optional[DataFrameT] = None
        if first_leaf_singleton is not _CACHE_MISSING and not residual_map:
            shared_ids_for_left = shared_ids.rename({node_col: shared_alias}) if node_col != shared_alias else shared_ids
            cached_left_counts = _connected_join_cached_first_arm_shared_counts(
                base_graph,
                nodes_source,
                edges,
                first_edges,
                shared_ids_for_left,
                cast(Optional[dict], first_edge.edge_match),
                cast(Optional[dict], first_start.filter_dict),
                cast(Optional[dict], second_start.filter_dict),
                first_leaf_singleton,
                shared_alias=shared_alias,
                src_col=src_col,
                dst_col=dst_col,
                engine=engine,
            )
            if cached_left_counts is None:
                cached_left_counts = _connected_join_cached_singleton_dst_source_counts(
                    base_graph,
                    edges,
                    first_edges,
                    cast(Optional[dict], first_edge.edge_match),
                    first_leaf_singleton,
                    src_col=src_col,
                    dst_col=dst_col,
                    engine=engine,
                )
        if cached_left_counts is not None:
            if shared_alias in cached_left_counts.columns:
                left_counts = cached_left_counts
            else:
                left_counts = (
                    cached_left_counts
                    .join(shared_ids, left_on=src_col, right_on=node_col, how="semi")
                    .rename({src_col: shared_alias})
                )
        elif first_leaf_singleton is _CACHE_MISSING:
            left_edges = (
                first_edges
                .join(shared_ids, left_on=src_col, right_on=node_col, how="semi")
                .join(first_leaf_ids, left_on=dst_col, right_on=node_col, how="semi")
            )
            left_counts = (
                left_edges
                .group_by(src_col)
                .len("__left_count__")
                .rename({src_col: shared_alias})
            )
        else:
            left_edges = (
                first_edges
                .filter(pl.col(dst_col) == first_leaf_singleton)
                .join(shared_ids, left_on=src_col, right_on=node_col, how="semi")
            )
            left_counts = (
                left_edges
                .group_by(src_col)
                .len("__left_count__")
                .rename({src_col: shared_alias})
            )
        cached_right_rows = None if residual_map else _connected_join_cached_second_arm_group_rows(
            base_graph,
            nodes_source,
            edges,
            second_edges,
            shared_ids,
            second_leaf_ids,
            second_leaf_nodes,
            cast(Optional[dict], first_start.filter_dict),
            cast(Optional[dict], second_start.filter_dict),
            cast(Optional[dict], second_end.filter_dict),
            cast(Optional[dict], second_edge.edge_match),
            second_leaf_singleton,
            group_prop_refs,
            output_group_keys,
            node_col=node_col,
            src_col=src_col,
            dst_col=dst_col,
            engine=engine,
        )
        if cached_right_rows is not None:
            right_rows = cached_right_rows if src_col == shared_alias else cached_right_rows.rename({src_col: shared_alias})
        else:
            if second_leaf_singleton is _CACHE_MISSING:
                right_base = (
                    second_edges
                    .join(shared_ids, left_on=src_col, right_on=node_col, how="semi")
                    .join(second_leaf_ids, left_on=dst_col, right_on=node_col, how="semi")
                )
            else:
                right_base = (
                    second_edges
                    .filter(pl.col(dst_col) == second_leaf_singleton)
                    .join(shared_ids, left_on=src_col, right_on=node_col, how="semi")
                )
            if group_prop_refs:
                lookup_key = "__gfql_fast_second_leaf_id__"
                lookup_exprs = [pl.col(node_col).alias(lookup_key)] + [pl.col(prop).alias(out_col) for out_col, prop in group_prop_refs]
                second_lookup = second_leaf_nodes.select(lookup_exprs)
                right_base = right_base.join(second_lookup, left_on=dst_col, right_on=lookup_key, how="inner")
            right_rows = right_base.select([pl.col(src_col).alias(shared_alias)] + [pl.col(key) for key in output_group_keys])
        if not output_group_keys and len(left_counts) > 0 and bool(left_counts.select((pl.col("__left_count__") == 1).all()).item()):
            matched_rows = right_rows.join(left_counts.select(shared_alias), on=shared_alias, how="semi")
            out_df = matched_rows.select(pl.len().cast(pl.Int64).alias(agg_alias))
        else:
            joined = right_rows.join(left_counts, on=shared_alias, how="inner")
            if len(joined) == 0:
                return cast(DataFrameT, joined.select([]))
            if output_group_keys:
                out_df = joined.group_by(output_group_keys, maintain_order=True).agg(pl.col("__left_count__").sum().cast(pl.Int64).alias(agg_alias))
            else:
                out_df = joined.select(pl.col("__left_count__").sum().cast(pl.Int64).alias(agg_alias))
        if order_keys:
            out_df = out_df.sort([key for key, _ in order_keys], descending=[desc for _, desc in order_keys])
        if limit_value is not None:
            out_df = out_df.head(limit_value)
        if select_items is not None:
            out_df = out_df.select([pl.col(src).alias(dst) for src, dst in select_items])
        return cast(DataFrameT, out_df)

    filter_engine = EngineAbstract(engine.value)
    shared_nodes = filter_by_dict(nodes, cast(Optional[dict], first_start.filter_dict), engine=filter_engine)
    shared_nodes = filter_by_dict(shared_nodes, cast(Optional[dict], second_start.filter_dict), engine=filter_engine)
    first_leaf_nodes = filter_by_dict(nodes, cast(Optional[dict], first_end.filter_dict), engine=filter_engine)
    second_leaf_nodes = filter_by_dict(nodes, cast(Optional[dict], second_end.filter_dict), engine=filter_engine)
    if residual_map:
        _frames = _connected_join_filter_node_frames_by_residuals(
            base_graph,
            residual_map,
            {shared_alias: shared_nodes, first_end_alias: first_leaf_nodes, second_end_alias: second_leaf_nodes},
            node_col,
            engine=engine,
        )
        shared_nodes, first_leaf_nodes, second_leaf_nodes = _frames[shared_alias], _frames[first_end_alias], _frames[second_end_alias]
    first_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], first_edge.edge_match), engine=engine)
    second_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], second_edge.edge_match), engine=engine)

    shared_ids = shared_nodes[node_col].drop_duplicates()
    first_leaf_ids = first_leaf_nodes[node_col].drop_duplicates()
    second_leaf_ids = second_leaf_nodes[node_col].drop_duplicates()
    for _, prop in group_prop_refs:
        if prop not in second_leaf_nodes.columns:
            return None

    left_edges = first_edges[first_edges[src_col].isin(shared_ids) & first_edges[dst_col].isin(first_leaf_ids)]
    left_counts = left_edges.groupby(src_col, sort=False).size().reset_index(name="__left_count__").rename(columns={src_col: shared_alias})
    right_base = second_edges[second_edges[src_col].isin(shared_ids) & second_edges[dst_col].isin(second_leaf_ids)]
    if group_prop_refs:
        lookup_key = "__gfql_fast_second_leaf_id__"
        prop_cols = []
        for _, prop in group_prop_refs:
            if prop not in prop_cols:
                prop_cols.append(prop)
        second_lookup = second_leaf_nodes[[node_col] + prop_cols].drop_duplicates(subset=[node_col]).rename(columns={node_col: lookup_key})
        for out_col, prop in group_prop_refs:
            second_lookup[out_col] = second_lookup[prop]
        right_base = right_base.merge(second_lookup[[lookup_key] + output_group_keys], left_on=dst_col, right_on=lookup_key, how="inner")
    right_rows = right_base[[src_col] + output_group_keys].rename(columns={src_col: shared_alias})
    joined = right_rows.merge(left_counts, on=shared_alias, how="inner")
    if len(joined) == 0:
        return df_cons(engine)()
    if output_group_keys:
        out_df = joined.groupby(output_group_keys, sort=False, dropna=False)["__left_count__"].sum().reset_index(name=agg_alias)
    else:
        out_df = pd.DataFrame({agg_alias: [int(joined["__left_count__"].sum())]})
    if order_keys:
        out_df = cast(DataFrameT, out_df.sort_values(by=[key for key, _ in order_keys], ascending=[not desc for _, desc in order_keys]))
    if limit_value is not None:
        out_df = cast(DataFrameT, out_df.head(limit_value))
    if select_items is not None:
        out_df = out_df[[src for src, _ in select_items]].rename(columns={src: dst for src, dst in select_items})
    return df_to_engine(out_df.reset_index(drop=True), engine)


def _connected_join_two_star_fast_rows(
    base_graph: Plottable,
    plan: ConnectedMatchJoinPlan,
    *,
    engine: Engine,
) -> Optional[DataFrameT]:
    """Fast rows for T1 two-star connected joins.

    Supported shape: ``(p)-[r1]->(i), (p)-[r2]->(c)`` where both arms are fixed
    forward one-hop patterns, share the same start node alias, have no residual
    per-arm row filters, and downstream projections need at most properties from
    the second leaf alias. The returned frame preserves row multiplicity, then
    the normal row pipeline handles RETURN/GROUP/ORDER/LIMIT.
    """
    if len(plan.pattern_chains) != 2 or len(plan.pattern_shared_node_aliases) != 1:
        return None
    if len(plan.pattern_shared_node_aliases[0]) != 1:
        return None
    shared_alias = plan.pattern_shared_node_aliases[0][0]

    parsed = []
    for pattern_chain in plan.pattern_chains:
        if pattern_chain.where:
            return None
        ops = list(pattern_chain.chain)
        if len(ops) != 3:
            return None
        start_op, edge_op, end_op = ops
        if not isinstance(start_op, ASTNode) or not isinstance(edge_op, ASTEdge) or not isinstance(end_op, ASTNode):
            return None
        if getattr(start_op, "query", None) is not None or getattr(end_op, "query", None) is not None:
            return None
        if not _is_connected_fast_single_hop(edge_op):
            return None
        start_alias = getattr(start_op, "_name", None)
        end_alias = getattr(end_op, "_name", None)
        if start_alias != shared_alias or not isinstance(end_alias, str):
            return None
        parsed.append((start_op, edge_op, end_op, end_alias))

    first_start, first_edge, first_end, first_end_alias = parsed[0]
    second_start, second_edge, second_end, second_end_alias = parsed[1]
    if first_end_alias == second_end_alias:
        return None

    attach_aliases = plan.pattern_attach_prop_aliases
    if not attach_aliases or len(attach_aliases) != 2:
        return None
    normalized_attach_aliases: List[Tuple[str, ...]] = []
    for aliases in attach_aliases:
        if aliases is None:
            return None
        normalized_attach_aliases.append(aliases)
    needed_attach_aliases = {alias for aliases in normalized_attach_aliases for alias in aliases}
    if any(alias != second_end_alias for alias in needed_attach_aliases):
        return None
    second_props_needed = _connected_join_post_property_columns(plan, second_end_alias)
    if second_end_alias in needed_attach_aliases and not second_props_needed:
        return None

    nodes_obj = getattr(base_graph, "_nodes", None)
    edges_obj = getattr(base_graph, "_edges", None)
    node_col = getattr(base_graph, "_node", None)
    src_col = getattr(base_graph, "_source", None)
    dst_col = getattr(base_graph, "_destination", None)
    if nodes_obj is None or edges_obj is None or node_col is None or src_col is None or dst_col is None:
        return None
    node_col = str(node_col)
    src_col = str(src_col)
    dst_col = str(dst_col)
    if node_col not in nodes_obj.columns or src_col not in edges_obj.columns or dst_col not in edges_obj.columns:
        return None

    nodes = df_to_engine(cast(DataFrameT, nodes_obj), engine)
    edges = cast(DataFrameT, edges_obj)

    if engine in POLARS_ENGINES:
        import polars as pl
        from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars

        shared_nodes = filter_by_dict_polars(nodes, cast(Optional[dict], first_start.filter_dict))
        shared_nodes = filter_by_dict_polars(shared_nodes, cast(Optional[dict], second_start.filter_dict))
        first_leaf_nodes = filter_by_dict_polars(nodes, cast(Optional[dict], first_end.filter_dict))
        second_leaf_nodes = filter_by_dict_polars(nodes, cast(Optional[dict], second_end.filter_dict))
        first_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], first_edge.edge_match), engine=engine)
        second_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], second_edge.edge_match), engine=engine)

        shared_ids = shared_nodes.select(node_col).unique()
        first_leaf_ids = first_leaf_nodes.select(node_col).unique()
        second_leaf_ids = second_leaf_nodes.select(node_col).unique()
        second_props = [col for col in second_props_needed if col in second_leaf_nodes.columns]

        left_rows = (
            first_edges
            .join(shared_ids, left_on=src_col, right_on=node_col, how="semi")
            .join(first_leaf_ids, left_on=dst_col, right_on=node_col, how="semi")
            .select(pl.col(src_col).alias(shared_alias))
        )
        right_base = (
            second_edges
            .join(shared_ids, left_on=src_col, right_on=node_col, how="semi")
            .join(second_leaf_ids, left_on=dst_col, right_on=node_col, how="semi")
        )
        if second_props:
            second_lookup_key = "__gfql_fast_second_leaf_id__"
            second_lookup = second_leaf_nodes.select([node_col] + second_props).rename({node_col: second_lookup_key})
            second_lookup = second_lookup.rename({col: f"{second_end_alias}.{col}" for col in second_props})
            right_base = right_base.join(second_lookup, left_on=dst_col, right_on=second_lookup_key, how="inner")
        right_select = [pl.col(src_col).alias(shared_alias)] + [pl.col(f"{second_end_alias}.{col}") for col in second_props]
        right_rows = right_base.select(right_select)
        return cast(DataFrameT, left_rows.join(right_rows, on=shared_alias, how="inner"))

    filter_engine = EngineAbstract(engine.value)
    shared_nodes = filter_by_dict(nodes, cast(Optional[dict], first_start.filter_dict), engine=filter_engine)
    shared_nodes = filter_by_dict(shared_nodes, cast(Optional[dict], second_start.filter_dict), engine=filter_engine)
    first_leaf_nodes = filter_by_dict(nodes, cast(Optional[dict], first_end.filter_dict), engine=filter_engine)
    second_leaf_nodes = filter_by_dict(nodes, cast(Optional[dict], second_end.filter_dict), engine=filter_engine)
    first_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], first_edge.edge_match), engine=engine)
    second_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], second_edge.edge_match), engine=engine)

    shared_ids = shared_nodes[node_col].drop_duplicates()
    first_leaf_ids = first_leaf_nodes[node_col].drop_duplicates()
    second_leaf_ids = second_leaf_nodes[node_col].drop_duplicates()
    second_props = [col for col in second_props_needed if col in second_leaf_nodes.columns]

    left_edges = first_edges[first_edges[src_col].isin(shared_ids) & first_edges[dst_col].isin(first_leaf_ids)]
    left_rows = left_edges[[src_col]].rename(columns={src_col: shared_alias})
    right_base = second_edges[second_edges[src_col].isin(shared_ids) & second_edges[dst_col].isin(second_leaf_ids)]
    if second_props:
        second_lookup_key = "__gfql_fast_second_leaf_id__"
        second_lookup_cols = [node_col] + second_props
        second_lookup = second_leaf_nodes[second_lookup_cols].drop_duplicates(subset=[node_col]).rename(
            columns={
                node_col: second_lookup_key,
                **{col: f"{second_end_alias}.{col}" for col in second_props},
            }
        )
        right_base = right_base.merge(second_lookup, left_on=dst_col, right_on=second_lookup_key, how="inner")
    right_cols = [src_col] + [f"{second_end_alias}.{col}" for col in second_props]
    right_rows = right_base[right_cols].rename(columns={src_col: shared_alias})
    return cast(DataFrameT, left_rows.merge(right_rows, on=shared_alias, how="inner"))

def _apply_connected_match_join(
    base_graph: Plottable,
    plan: ConnectedMatchJoinPlan,
    *,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
) -> Plottable:
    from graphistry.compute.ast import ASTCall, ASTNode as _ASTNode, serialize_binding_ops

    requested_engine = resolve_engine(cast(Any, engine), base_graph)
    dispatch_engine: Union[EngineAbstract, str] = engine
    df_ctor = df_cons(requested_engine)
    node_col = getattr(base_graph, "_node", "id")

    fast_grouped_count = _connected_join_two_star_fast_grouped_count(base_graph, plan, engine=requested_engine)
    if fast_grouped_count is not None:
        out = base_graph.bind()
        out._nodes = fast_grouped_count
        out._edges = df_ctor()
        return out

    fast_rows = _connected_join_two_star_fast_rows(base_graph, plan, engine=requested_engine)
    if fast_rows is not None:
        if len(fast_rows) == 0:
            out = base_graph.bind()
            out._nodes = df_ctor()
            out._edges = df_ctor()
            return out
        fast_rows = _joined_hidden_scalar_columns(fast_rows)
        fast_rows = _joined_alias_columns(fast_rows)
        joined_plottable = base_graph.bind()
        joined_plottable._nodes = fast_rows
        joined_plottable._edges = df_ctor()
        return _chain_dispatch(joined_plottable, plan.post_join_chain, dispatch_engine, policy, context)

    joined_rows: Optional[DataFrameT] = None
    pattern_attach_prop_aliases = plan.pattern_attach_prop_aliases or tuple(None for _ in plan.pattern_chains)
    for idx, pattern_chain in enumerate(plan.pattern_chains):
        rows_params: Dict[str, Any] = {"binding_ops": serialize_binding_ops(pattern_chain.chain)}
        if idx < len(pattern_attach_prop_aliases) and pattern_attach_prop_aliases[idx] is not None:
            rows_params["attach_prop_aliases"] = list(cast(Tuple[str, ...], pattern_attach_prop_aliases[idx]))
        with_rows = Chain(
            list(pattern_chain.chain) + [ASTCall("rows", rows_params)],
            where=pattern_chain.where,
        )
        pattern_result = _chain_dispatch(base_graph, with_rows, dispatch_engine, policy, context)
        pattern_rows = cast(Optional[DataFrameT], pattern_result._nodes)
        if pattern_rows is None:
            out = base_graph.bind()
            out._nodes = df_ctor()
            out._edges = df_ctor()
            return out
        # The rows op now emits the full binding schema even at 0 rows (#25), so an emptied
        # pattern carries its columns and flows through post_join_chain -- which is where the
        # aggregate RETURN lives; short-circuiting here dropped that column. Beyond the join
        # columns, rung-3 execution also keeps the bare node-alias columns (e.g. `count(i)`
        # needs the `i` binding column downstream in post_join_chain).
        node_aliases = [
            cast(str, getattr(op, "_name"))
            for op in pattern_chain.chain
            if isinstance(op, _ASTNode) and isinstance(getattr(op, "_name", None), str)
        ]
        keep_binding_columns = _binding_join_columns(pattern_rows) + [
            alias for alias in node_aliases if alias in pattern_rows.columns
        ]
        pattern_rows = cast(DataFrameT, pattern_rows[keep_binding_columns])
        if joined_rows is None:
            joined_rows = pattern_rows
            continue
        shared_aliases = plan.pattern_shared_node_aliases[idx - 1]
        join_cols = [
            alias
            for alias in shared_aliases
            if alias in joined_rows.columns and alias in pattern_rows.columns
        ]
        if not join_cols:
            join_cols = [
                f"{alias}.{node_col}"
                for alias in shared_aliases
                if f"{alias}.{node_col}" in joined_rows.columns and f"{alias}.{node_col}" in pattern_rows.columns
            ]
        if not join_cols:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher connected comma-pattern join lowering could not recover shared node identity columns for the runtime join",
                field="match",
                value=list(shared_aliases),
                suggestion="Use a simpler connected MATCH shape in the local compiler.",
                language="cypher",
            )
        keep_cols = [column for column in pattern_rows.columns if column in join_cols or column not in joined_rows.columns]
        joined_rows = _connected_inner_join_rows(
            cast(DataFrameT, joined_rows),
            cast(DataFrameT, pattern_rows),
            join_cols=join_cols,
            keep_cols=keep_cols,
            engine=requested_engine,
        )

    if joined_rows is None:
        out = base_graph.bind()
        out._nodes = df_ctor()
        out._edges = df_ctor()
        return out

    joined_rows = _joined_hidden_scalar_columns(joined_rows)
    joined_rows = _joined_alias_columns(joined_rows)
    joined_plottable = base_graph.bind()
    joined_plottable._nodes = joined_rows
    joined_plottable._edges = df_ctor()
    return _chain_dispatch(joined_plottable, plan.post_join_chain, dispatch_engine, policy, context)


def _graph_residual_eval_frame(df: DataFrameT, alias: str) -> DataFrameT:
    return cast(DataFrameT, df.assign(**{alias: True}))


def _evaluate_graph_residual_mask(
    graph: Plottable,
    df: DataFrameT,
    residual: CompiledGraphResidualFilter,
) -> Any:
    eval_df = _graph_residual_eval_frame(df, residual.alias)
    for pre_filter in residual.pre_filters:
        if pre_filter.function != "search_any":
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher GRAPH residual pre-filter is not supported as a graph mask",
                field="graph_constructor",
                value=pre_filter.function,
                language="cypher",
            )
        params = pre_filter.params
        marker_col = cast(str, params.get("out_col"))
        search_df = eval_df[[
            col for col in eval_df.columns
            if not str(col).startswith("__gfql_") and col != residual.alias
        ]]
        marker_mask = search_any_mask(
            cast(DataFrameT, search_df),
            cast(str, params.get("term")),
            case_sensitive=bool(params.get("case_sensitive", False)),
            regex=bool(params.get("regex", False)),
            columns=cast(Optional[List[str]], params.get("columns")),
        )
        if marker_mask is None:
            raise GFQLValidationError(
                ErrorCode.E108,
                "searchAny columns= includes a column absent from the searched table",
                field="columns",
                value=params.get("columns"),
                suggestion="List only columns present on the searched entity.",
                language="cypher",
            )
        eval_df = cast(DataFrameT, eval_df.assign(**{marker_col: marker_mask}))

    adapter = _RowPipelineAdapter(graph)
    value = adapter._gfql_eval_string_expr(eval_df, residual.expr)
    return adapter._gfql_bool_mask(eval_df, value)


def _apply_graph_residual_filters(
    base_graph: Plottable,
    residual_filters: Tuple[CompiledGraphResidualFilter, ...],
    *,
    engine: Union[EngineAbstract, str],
) -> Plottable:
    if not residual_filters:
        return base_graph
    concrete_engine = resolve_engine(cast(Any, engine), base_graph)
    if concrete_engine in POLARS_ENGINES:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher GRAPH residual predicates are not yet supported on polars graph execution",
            field="graph_constructor",
            value=[residual.expr for residual in residual_filters],
            suggestion="Use engine='pandas' or engine='cudf' for GRAPH residual predicates.",
            language="cypher",
        )

    graph = base_graph
    for residual in residual_filters:
        if residual.kind == "node":
            graph_with_nodes = graph if graph._nodes is not None else graph.materialize_nodes(engine=EngineAbstract(concrete_engine.value))
            nodes_df = cast(DataFrameT, graph_with_nodes._nodes)
            node_mask = _evaluate_graph_residual_mask(graph_with_nodes, nodes_df, residual)
            filtered_nodes = cast(DataFrameT, nodes_df.loc[node_mask])
            graph = graph_with_nodes.nodes(filtered_nodes)
            if graph._edges is not None and graph._node is not None and graph._source is not None and graph._destination is not None:
                node_ids = filtered_nodes[graph._node]
                edges_df = cast(DataFrameT, graph._edges)
                edge_mask = edges_df[graph._source].isin(node_ids) & edges_df[graph._destination].isin(node_ids)
                graph = graph.edges(cast(DataFrameT, edges_df.loc[edge_mask]))
        else:
            if graph._edges is None:
                continue
            edges_df = cast(DataFrameT, graph._edges)
            edge_mask = _evaluate_graph_residual_mask(graph, edges_df, residual)
            graph = graph.edges(cast(DataFrameT, edges_df.loc[edge_mask]))
    return graph


def _execute_graph_constructor_compiled(
    base_graph: Plottable,
    chain: Chain,
    *,
    procedure_call: Any = None,
    graph_residual_filters: Tuple[CompiledGraphResidualFilter, ...] = (),
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
) -> Plottable:
    """Execute a compiled graph constructor (MATCH-based or CALL-based)."""
    if procedure_call is not None:
        return execute_cypher_call(base_graph, procedure_call)
    filtered_graph = _apply_graph_residual_filters(
        base_graph, graph_residual_filters, engine=engine
    )
    return _chain_dispatch(filtered_graph, chain, engine, policy, context)


def _resolve_graph_bindings(
    base_graph: Plottable,
    bindings: tuple,
    scope: Optional[Dict[str, Plottable]] = None,
    *,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
) -> Dict[str, Plottable]:
    """Execute graph bindings in order, building a scope of named graphs.

    Each binding's USE clause (if present) is resolved against previously
    bound graphs in the scope. The resolved graph becomes the base for
    that binding's execution.
    """
    if scope is None:
        scope = {}
    for binding in bindings:
        target_graph = base_graph
        # USE ref inside the binding's constructor was already validated at
        # parse time. At runtime, resolve it against the scope.
        if binding.use_ref is not None:
            target_graph = scope.get(binding.use_ref.lower(), base_graph)
        result = _execute_graph_constructor_compiled(
            target_graph, binding.chain,
            procedure_call=binding.procedure_call,
            graph_residual_filters=binding.graph_residual_filters,
            engine=engine, policy=policy, context=context,
        )
        scope[binding.name.lower()] = result
    return scope


def _execute_graph_query(
    base_graph: Plottable,
    compiled: CompiledCypherGraphQuery,
    *,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
) -> Plottable:
    """Execute a standalone GRAPH { ... } query (returns graph state)."""
    scope = _resolve_graph_bindings(
        base_graph, compiled.graph_bindings,
        engine=engine, policy=policy, context=context,
    )
    # Resolve USE for the final constructor
    target_graph = base_graph
    if compiled.use_ref is not None:
        target_graph = scope.get(compiled.use_ref.lower(), base_graph)
    return _execute_graph_constructor_compiled(
        target_graph, compiled.chain,
        procedure_call=compiled.procedure_call,
        graph_residual_filters=compiled.graph_residual_filters,
        engine=engine, policy=policy, context=context,
    )


def _execute_query_with_graph_context(
    base_graph: Plottable,
    compiled: CompiledCypherQuery,
    *,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
) -> Plottable:
    """Execute a query that has GRAPH bindings and/or USE."""
    scope = _resolve_graph_bindings(
        base_graph, compiled.graph_bindings,
        engine=engine, policy=policy, context=context,
    )
    # If USE is specified, execute the main query against the USE'd graph
    if compiled.use_ref is not None:
        target_graph = scope.get(compiled.use_ref.lower(), base_graph)
    else:
        target_graph = base_graph
    # Strip graph context from the compiled query and execute normally
    plain_query = replace(compiled, graph_bindings=(), use_ref=None)
    return _execute_compiled_query(
        target_graph,
        compiled_query=plain_query,
        engine=engine,
        policy=policy,
        context=context,
    )


def _execute_compiled_query(
    base_graph: Plottable,
    *,
    compiled_query: Union[CompiledCypherQuery, CompiledCypherUnionQuery],
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
    start_nodes: Optional[DataFrameT] = None,
) -> Plottable:
    if isinstance(compiled_query, CompiledCypherUnionQuery):
        concrete_engine = resolve_engine(cast(Any, engine), base_graph)
        df_ctor = df_cons(concrete_engine)
        concat = df_concat(concrete_engine)
        branch_results = [
            _execute_compiled_query(
                base_graph,
                compiled_query=branch,
                engine=engine,
                policy=policy,
                context=context,
                start_nodes=start_nodes,
            )
            for branch in compiled_query.branches
        ]
        row_frames = [cast(DataFrameT, result._nodes) for result in branch_results if result._nodes is not None]
        union_rows = df_ctor() if not row_frames else concat(row_frames, ignore_index=True, sort=False)
        if compiled_query.union_kind == "distinct" and len(union_rows) > 0:
            union_rows = cast(DataFrameT, df_unique(union_rows, concrete_engine))
        out = base_graph.bind()
        out._nodes = union_rows
        out._edges = df_ctor()
        return out

    return _execute_compiled_query_non_union(
        base_graph,
        compiled_query=compiled_query,
        engine=engine,
        policy=policy,
        context=context,
        start_nodes=start_nodes,
    )


def _execute_compiled_query_non_union(
    base_graph: Plottable,
    *,
    compiled_query: CompiledCypherQuery,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
    start_nodes: Optional[DataFrameT] = None,
) -> Plottable:
    compiled_extras = compiled_query.execution_extras
    logical_plan = compiled_query.logical_plan
    if logical_plan is None:
        defer_reason = compiled_query.logical_plan_defer_reason
        defer_code = compiled_query.logical_plan_defer_code
        if compiled_query.procedure_call is not None:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher CALL queries must use the procedure physical route",
                field="procedure_call",
                value=compiled_query.procedure_call.procedure,
                suggestion="Compile CALL queries with a LogicalPlan before runtime dispatch.",
                language="cypher",
            )
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher query reached runtime without a logical plan",
            field="logical_plan",
            logical_plan_defer_code=defer_code,
            logical_plan_defer_reason=defer_reason,
            suggestion="Compile this Cypher shape through a LogicalPlan route before chain execution.",
            language="cypher",
        )

    ctx = PlanContext(scope_stack=() if compiled_extras is None else compiled_extras.scope_stack)
    logical_plan = _run_logical_pass_pipeline(logical_plan, ctx)

    try:
        physical_plan = PhysicalPlanner().plan(logical_plan, ctx)
    except GFQLValidationError as exc:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher planned route could not be lowered to a supported physical execution path",
            field="logical_plan",
            value=exc.message,
            suggestion="Use a covered M3 query shape (same-path / wavefront / row-pipeline) or retain compatibility shims for this lane.",
            language="cypher",
        ) from exc

    return _execute_compiled_query_via_physical_plan(
        base_graph,
        compiled_query=compiled_query,
        physical_plan=physical_plan,
        engine=engine,
        policy=policy,
        context=context,
        start_nodes=start_nodes,
    )


def _run_logical_pass_pipeline(logical_plan: LogicalPlan, ctx: PlanContext) -> LogicalPlan:
    """Run logical pass pipeline: Tier 1 structural passes then Tier 2 fixed-point rewrite loop."""
    return PassManager(DEFAULT_LOGICAL_PASSES, DEFAULT_TIER2_PASSES).run(logical_plan, ctx).plan


def _execute_compiled_query_via_physical_plan(
    base_graph: Plottable,
    *,
    compiled_query: CompiledCypherQuery,
    physical_plan: PhysicalPlan,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
    start_nodes: Optional[DataFrameT] = None,
) -> Plottable:
    compiled_extras = compiled_query.execution_extras
    connected_match_join = None if compiled_extras is None else compiled_extras.connected_match_join
    connected_optional_match = None if compiled_extras is None else compiled_extras.connected_optional_match

    if connected_match_join is not None:
        return _apply_connected_match_join(
            base_graph,
            connected_match_join,
            engine=engine,
            policy=policy,
            context=context,
        )

    if connected_optional_match is not None:
        return _apply_connected_optional_match(
            base_graph,
            connected_optional_match,
            engine=engine,
            policy=policy,
            context=context,
        )

    if physical_plan.route in ("same_path", "row_pipeline"):
        return _execute_compiled_query_chain_non_union(
            base_graph,
            compiled_query=compiled_query,
            engine=engine,
            policy=policy,
            context=context,
            start_nodes=start_nodes,
        )

    if physical_plan.route == "procedure_call":
        if compiled_query.procedure_call is None:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher procedure physical route selected without a compiled procedure call",
                field="procedure_call",
                value=None,
                suggestion="Compile CALL queries with procedure metadata before physical dispatch.",
                language="cypher",
            )
        dispatch_graph = execute_cypher_call(base_graph, compiled_query.procedure_call)
        return _execute_compiled_query_chain_non_union(
            base_graph,
            compiled_query=compiled_query,
            dispatch_graph=dispatch_graph,
            engine=engine,
            policy=policy,
            context=context,
            start_nodes=start_nodes,
        )

    if physical_plan.route == "wavefront":
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher wavefront physical route selected but compiled query has no connected join payload to execute",
            field="physical_plan.route",
            value=physical_plan.route,
            suggestion="Use a supported connected MATCH/OPTIONAL MATCH lowering shape for wavefront execution.",
            language="cypher",
        )

    raise GFQLValidationError(
        ErrorCode.E108,
        "Cypher physical plan produced an unknown route",
        field="physical_plan.route",
        value=physical_plan.route,
        suggestion="Use a covered M3 route or extend the runtime dispatcher.",
        language="cypher",
    )


def _seeded_dispatch_graph(
    base_graph: Plottable,
    *,
    compiled_query: CompiledCypherQuery,
    engine: Union[EngineAbstract, str],
) -> Plottable:
    if not compiled_query.seed_rows:
        return base_graph

    concrete_engine = resolve_engine(cast(Any, engine), base_graph)
    df_ctor = df_cons(concrete_engine)
    dispatch_graph = base_graph.bind()
    dispatch_graph._nodes = df_ctor({"__cypher_seed_row__": [True]})
    dispatch_graph._edges = df_ctor()
    return dispatch_graph


def _execute_compiled_query_chain_non_union(
    base_graph: Plottable,
    *,
    compiled_query: CompiledCypherQuery,
    dispatch_graph: Optional[Plottable] = None,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
    start_nodes: Optional[DataFrameT] = None,
) -> Plottable:
    if dispatch_graph is None:
        dispatch_graph = _seeded_dispatch_graph(
            base_graph,
            compiled_query=compiled_query,
            engine=engine,
        )

    # #1712: a bounded-reentry main chain that is a binding-ops row pipeline
    # (rows(binding_ops) -> group_by -> ...) must seed its first alias from the
    # prefix WITH result. The chain ``start_nodes`` carry that seed, but the binding
    # builder reads it from ``_gfql_start_nodes`` (propagated through row ops by
    # row_table). The boundary-call handler only set that for the traversal->suffix-
    # call shape; an all-call reentry chain otherwise re-matched the carried alias
    # from the WHOLE graph, dropping the prefix filter (silent wrong count).
    if start_nodes is not None:
        _chain_ops = getattr(compiled_query.chain, "chain", None) or []
        _first_op = _chain_ops[0] if _chain_ops else None
        if (
            _first_op is not None
            and getattr(_first_op, "function", None) == "rows"
            and getattr(_first_op, "params", {}).get("binding_ops") is not None
        ):
            setattr(dispatch_graph, "_gfql_start_nodes", start_nodes)

    result = _chain_dispatch(dispatch_graph, compiled_query.chain, engine, policy, context, start_nodes=start_nodes)
    if compiled_query.empty_result_row is not None:
        result = _apply_empty_result_row(
            result,
            engine=engine,
            empty_result_row=compiled_query.empty_result_row,
        )
    if compiled_query.result_projection is not None:
        # OPTIONAL null-fill / row-guard still consumes a single-column entity value,
        # so those keep the legacy text form; plain terminal RETURN flattens (#1650).
        structured_projection = (
            compiled_query.optional_projection_row_guard is None
            and compiled_query.optional_null_fill is None
        )
        result = apply_result_projection(
            result, compiled_query.result_projection, structured=structured_projection
        )
    if compiled_query.optional_projection_row_guard is not None:
        expected_rows = 1
        for base_chain in compiled_query.optional_projection_row_guard.base_chains:
            base_result = _chain_dispatch(
                base_graph,
                base_chain,
                engine,
                policy,
                context,
            )
            base_rows_df = base_result._nodes
            expected_rows *= 0 if base_rows_df is None else len(base_rows_df)
            if expected_rows == 0:
                break
        result = _apply_optional_projection_row_guard(
            result,
            expected_rows=expected_rows,
        )
    if compiled_query.optional_null_fill is not None:
        base_result = _chain_dispatch(
            base_graph,
            compiled_query.optional_null_fill.base_chain,
            engine,
            policy,
            context,
        )
        alignment_result = apply_result_projection(
            _chain_dispatch(
                base_graph,
                compiled_query.optional_null_fill.alignment_chain,
                engine,
                policy,
                context,
            ),
            compiled_query.optional_null_fill.alignment_projection,
            structured=False,
        )
        result = _apply_optional_null_fill(
            result,
            base_result=base_result,
            alignment_result=alignment_result,
            alignment_output_name=compiled_query.optional_null_fill.alignment_output_name,
            engine=engine,
            null_row=compiled_query.optional_null_fill.null_row,
        )
    return result


def _execute_compiled_query_with_reentry(
    base_graph: Plottable,
    *,
    compiled_query: Union[CompiledCypherQuery, CompiledCypherUnionQuery],
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
) -> Plottable:
    if isinstance(compiled_query, CompiledCypherUnionQuery):
        return _execute_compiled_query(
            base_graph,
            compiled_query=compiled_query,
            engine=engine,
            policy=policy,
            context=context,
        )

    compiled_base_graph = base_graph
    start_nodes = None
    if compiled_query.start_nodes_query is not None:
        prefix_result = _execute_compiled_query_with_reentry(
            base_graph,
            compiled_query=compiled_query.start_nodes_query,
            engine=engine,
            policy=policy,
            context=context,
        )
        plan = compiled_query.reentry_plan
        if plan is None:
            raise _reentry_validation_error(
                "Cypher MATCH after WITH reentry dispatched without a ReentryPlan",
                value=None,
                suggestion=_REENTRY_WHOLE_ROW_SUGGESTION,
            )
        if plan.scalar_only:
            prefix_rows = prefix_result._nodes
            prefix_row_count = len(prefix_rows) if prefix_rows is not None else 0
            if prefix_row_count > 1:
                # Multi-row scalar prefix (#1047): run suffix once per prefix row, union results.
                if compiled_query.optional_reentry:
                    raise _reentry_validation_error(
                        "Cypher OPTIONAL MATCH after a multi-row scalar WITH prefix is not yet supported"
                        " — null-fill for unmatched prefix rows is not implemented for N>1 prefix rows",
                        value=prefix_row_count,
                        suggestion="Use MATCH instead of OPTIONAL MATCH, or reduce the WITH prefix to a single row",
                        field="optional_reentry",
                    )
                row_results = []
                for i in range(prefix_row_count):
                    row_graph, row_start = _compiled_query_scalar_reentry_state(
                        base_graph,
                        prefix_result,
                        carried_columns=plan.scalar_columns,
                        row_index=i,
                    )
                    row_result = _execute_compiled_query(
                        row_graph,
                        compiled_query=compiled_query,
                        engine=engine,
                        policy=policy,
                        context=context,
                        start_nodes=row_start,
                    )
                    row_results.append(row_result)
                result = _union_scalar_reentry_results(row_results, base_graph=base_graph, engine=engine)
                return result
            else:
                compiled_base_graph, start_nodes = _compiled_query_scalar_reentry_state(
                    base_graph,
                    prefix_result,
                    carried_columns=plan.scalar_columns,
                )
        elif plan.free_form:
            # #1263 (LDBC SNB IC3 endpoint): trailing MATCH binds aliases
            # none of which is in the prefix's carried set. Broadcast the
            # carried hidden columns onto every base node so the row
            # pipeline carries them through whichever alias the trailing
            # MATCH binds; the suffix runs as a global MATCH (no seed).
            prefix_rows_for_freeform = prefix_result._nodes
            prefix_row_count_freeform = (
                len(prefix_rows_for_freeform) if prefix_rows_for_freeform is not None else 0
            )
            if prefix_row_count_freeform > 1:
                # #1285: multi-prefix-row free-form intermediate MATCH —
                # run suffix once per prefix row with that row's hidden
                # carry values broadcast, then union per-row results.
                # Mirrors the scalar-only multi-row pattern at lines 916-945
                # above; reuses ``_union_scalar_reentry_results`` (engine-
                # polymorphic concat).
                if compiled_query.optional_reentry:
                    raise _reentry_validation_error(
                        "Cypher OPTIONAL MATCH after a multi-row free-form WITH prefix is not yet supported"
                        " — null-fill for unmatched prefix rows is not implemented for N>1 prefix rows",
                        value=prefix_row_count_freeform,
                        suggestion="Use MATCH instead of OPTIONAL MATCH, or reduce the WITH prefix to a single row",
                        field="optional_reentry",
                    )
                base_nodes_for_freeform = base_graph._nodes
                if base_nodes_for_freeform is None:
                    raise _reentry_validation_error(
                        "Cypher MATCH after WITH (free-form intermediate MATCH; #1285) "
                        "could not recover the base node table for re-entry",
                        value=None,
                        suggestion=_REENTRY_WHOLE_ROW_SUGGESTION,
                    )
                row_results = []
                for i in range(prefix_row_count_freeform):
                    row_graph = _freeform_broadcast_row_to_nodes(
                        base_graph,
                        cast(DataFrameT, base_nodes_for_freeform),
                        cast(DataFrameT, prefix_rows_for_freeform),
                        plan,
                        row_index=i,
                    )
                    row_result = _execute_compiled_query(
                        row_graph,
                        compiled_query=compiled_query,
                        engine=engine,
                        policy=policy,
                        context=context,
                        start_nodes=None,
                    )
                    row_results.append(row_result)
                return _union_scalar_reentry_results(
                    row_results, base_graph=base_graph, engine=engine
                )
            compiled_base_graph, start_nodes = _compiled_query_freeform_reentry_state(
                base_graph,
                prefix_result,
                plan=plan,
            )
        else:
            prefix_rows_for_whole_row = cast(Optional[DataFrameT], prefix_result._nodes)
            prefix_row_count_for_whole_row = (
                len(prefix_rows_for_whole_row) if prefix_rows_for_whole_row is not None else 0
            )
            try:
                compiled_base_graph, start_nodes = _compiled_query_reentry_state(
                    base_graph,
                    plan,
                    prefix_result,
                    engine=engine,
                )
            except GFQLValidationError as exc:
                if not (
                    plan.scalar_columns
                    and prefix_row_count_for_whole_row > 1
                    and _is_duplicate_carried_rows_reentry_error(exc)
                ):
                    raise
                if compiled_query.optional_reentry:
                    raise _reentry_validation_error(
                        "Cypher OPTIONAL MATCH after a multi-row whole-row WITH prefix is not yet supported"
                        " — null-fill for unmatched prefix rows is not implemented for N>1 prefix rows",
                        value=prefix_row_count_for_whole_row,
                        suggestion="Use MATCH instead of OPTIONAL MATCH, or reduce the WITH prefix to a single row",
                        field="optional_reentry",
                    ) from exc
                row_results = []
                for i in range(prefix_row_count_for_whole_row):
                    row_prefix_result = _slice_reentry_prefix_result_row(
                        prefix_result,
                        output_name=plan.reentry_alias_name,
                        row_index=i,
                    )
                    row_graph, row_start = _compiled_query_reentry_state(
                        base_graph,
                        plan,
                        row_prefix_result,
                        engine=engine,
                    )
                    row_result = _execute_compiled_query(
                        row_graph,
                        compiled_query=compiled_query,
                        engine=engine,
                        policy=policy,
                        context=context,
                        start_nodes=row_start,
                    )
                    row_results.append(row_result)
                return _union_scalar_reentry_results(
                    row_results, base_graph=base_graph, engine=engine
                )
    result = _execute_compiled_query(
        compiled_base_graph,
        compiled_query=compiled_query,
        engine=engine,
        policy=policy,
        context=context,
        start_nodes=start_nodes,
    )

    # Optional reentry null-fill: if the reentry MATCH is OPTIONAL, prefix
    # rows that didn't match need null-filled entries in the result.
    if compiled_query.optional_reentry and compiled_query.start_nodes_query is not None:
        result = _apply_optional_reentry_null_fill(
            result,
            prefix_result=prefix_result,  # type: ignore[possibly-undefined]
            engine=engine,
            empty_result_row=compiled_query.empty_result_row,
            reentry_plan=compiled_query.reentry_plan,
        )

    return result


def _materialize_split_alias_columns(
    result: Plottable,
    executor: DFSamePathExecutor,
) -> Plottable:
    if result._edges is not None and result._edge is None:
        edge_id_col = generate_safe_column_name("edge_index", result._edges, prefix="__gfql_", suffix="__")
        result._edges = result._edges.assign(**{edge_id_col: range(len(result._edges))})
        result._edge = edge_id_col

    node_updates: Dict[str, Any] = {}
    edge_updates: Dict[str, Any] = {}

    for alias, binding in executor.inputs.alias_bindings.items():
        frame = executor.alias_frames.get(alias)
        if frame is None:
            continue
        if binding.kind == "node":
            df = result._nodes
            result_id_col = result._node
            frame_id_col = executor._node_column
        else:
            df = result._edges
            result_id_col = result._edge
            frame_id_col = executor._edge_column
        if (
            df is None
            or result_id_col is None
            or frame_id_col is None
            or result_id_col not in df.columns
            or frame_id_col not in frame.columns
        ):
            continue
        mask = df[result_id_col].isin(frame[frame_id_col])
        if binding.kind == "node":
            node_updates[alias] = mask
        else:
            edge_updates[alias] = mask

    if node_updates and result._nodes is not None:
        result._nodes = result._nodes.assign(**node_updates)
    if edge_updates and result._edges is not None:
        result._edges = result._edges.assign(**edge_updates)
    return result


def _gfql_otel_attrs(
    self: Plottable,
    query: GFQLQuery,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
    output: Optional[str] = None,
    policy: Optional[Dict[str, PolicyFunction]] = None,
    where: Optional[Sequence[WhereComparison]] = None,
    language: Optional[Literal["cypher", "gremlin"]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(query, dict):
        query_type = "chain" if "chain" in query else "dag"
    else:
        query_type = detect_query_type(query)
    attrs: Dict[str, Any] = {"gfql.query_type": query_type}
    if isinstance(query, Chain):
        attrs["gfql.chain_len"] = len(query.chain)
        attrs["gfql.has_where"] = bool(query.where)
    elif isinstance(query, list):
        attrs["gfql.chain_len"] = len(query)
        if where:
            attrs["gfql.has_where"] = True
    elif isinstance(query, ASTLet):
        attrs["gfql.binding_count"] = len(query.bindings)
    elif isinstance(query, dict):
        attrs["gfql.binding_count"] = len(query)
        if "chain" in query and isinstance(query["chain"], list):
            attrs["gfql.chain_len"] = len(query["chain"])
    if otel_detail_enabled():
        attrs["gfql.output"] = output is not None
        attrs["gfql.policy"] = policy is not None
        attrs["gfql.engine"] = str(engine)
        if isinstance(query, str):
            attrs["gfql.language"] = language or "cypher"
            attrs["gfql.has_params"] = params is not None
    return attrs


def detect_query_type(query: Any) -> QueryType:
    if isinstance(query, ASTLet):
        return "dag"
    elif isinstance(query, str):
        return "chain"
    elif isinstance(query, (list, Chain)):
        return "chain"
    else:
        return "single"


def _compile_string_query(
    query: str,
    *,
    language: Optional[Literal["cypher", "gremlin"]],
    params: Optional[Mapping[str, Any]],
    node_dtypes: Optional[NodeDtypes] = None,
) -> Any:
    query_language = language or "cypher"
    if query_language != "cypher":
        raise GFQLValidationError(
            ErrorCode.E108,
            f"Unsupported GFQL string language '{query_language}'",
            field="language",
            value=query_language,
            suggestion="Use language='cypher' for now; Gremlin string compilation is not implemented yet.",
            language="gfql",
        )
    return compile_cypher_query(parse_cypher(query), params=params, node_dtypes=node_dtypes)


def _compile_value_repr(value: Any) -> str:
    try:
        rendered = repr(value)
    except Exception:
        rendered = f"<unrepresentable {type(value).__name__}>"
    if len(rendered) > 200:
        return f"{rendered[:197]}..."
    return rendered


def _compile_context_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _compile_context_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_compile_context_value(v) for v in value)
    return _compile_value_repr(value)


def _compiler_phase_for_error(exc: GFQLValidationError) -> str:
    if exc.code == ErrorCode.E107:
        return "parse"
    context = getattr(exc, "context", {})
    if isinstance(context, dict) and (
        "visible_scope" in context
        or "existing_kind" in context
        or "new_kind" in context
    ):
        return "bind"
    if exc.code == ErrorCode.E108:
        return "lower"
    return "compile"


def _compile_summary(
    *,
    query_language: str,
    params: Optional[Mapping[str, Any]],
    exc: Optional[GFQLValidationError] = None,
) -> CompileSummary:
    if exc is None:
        return CompileSummary(
            language=query_language,
            success=True,
            param_keys=tuple(sorted(str(key) for key in params.keys())) if params else (),
        )

    context = getattr(exc, "context", {})
    error_context = context if isinstance(context, dict) else {}
    public_context = MappingProxyType({str(k): _compile_context_value(v) for k, v in error_context.items()})
    return CompileSummary(
        language=query_language,
        success=False,
        error_type=type(exc).__name__,
        message=exc.message,
        compiler_phase=_compiler_phase_for_error(exc),
        code=exc.code,
        context=public_context,
        field=error_context.get("field"),
        suggestion=error_context.get("suggestion"),
        line=error_context.get("line"),
        column=error_context.get("column"),
        value_repr=(
            _compile_value_repr(error_context["value"])
            if "value" in error_context
            else None
        ),
        param_keys=tuple(sorted(str(key) for key in params.keys())) if params else (),
    )


def _base_compile_policy_context(
    *,
    hook: Literal["precompile", "postcompile"],
    query: str,
    query_language: str,
    policy_depth: int,
    execution_depth: int,
    operation_path: str,
) -> PolicyContext:
    return {
        "phase": hook,
        "hook": hook,
        "query": query,
        "current_ast": None,
        "query_type": "chain",
        "compile_language": query_language,
        "execution_depth": execution_depth,
        "operation_path": operation_path,
        "parent_operation": "query" if execution_depth == 0 else operation_path.rsplit(".", 1)[0],
        "_policy_depth": policy_depth,
    }


def _fire_precompile_policy(
    policy: Optional[PolicyDict],
    *,
    query: str,
    query_language: str,
    policy_depth: int,
    execution_depth: int,
    operation_path: str,
) -> None:
    if not policy or "precompile" not in policy:
        return
    policy_context = _base_compile_policy_context(
        hook="precompile",
        query=query,
        query_language=query_language,
        policy_depth=policy_depth,
        execution_depth=execution_depth,
        operation_path=operation_path,
    )
    try:
        policy["precompile"](policy_context)
    except PolicyException as policy_exc:
        if policy_exc.query_type is None:
            policy_exc.query_type = policy_context.get("query_type")
        raise


def _fire_postcompile_policy(
    policy: Optional[PolicyDict],
    *,
    query: str,
    query_language: str,
    exc: Optional[GFQLValidationError],
    policy_depth: int,
    execution_depth: int,
    operation_path: str,
    params: Optional[Mapping[str, Any]],
) -> None:
    if not policy or "postcompile" not in policy:
        return
    summary = _compile_summary(
        query_language=query_language,
        params=params,
        exc=exc,
    )
    policy_context = _base_compile_policy_context(
        hook="postcompile",
        query=query,
        query_language=query_language,
        policy_depth=policy_depth,
        execution_depth=execution_depth,
        operation_path=operation_path,
    )
    policy_context["compile"] = summary
    policy_context["success"] = exc is None
    if exc is not None:
        policy_context["error"] = str(exc)
        policy_context["error_type"] = type(exc).__name__
    try:
        policy["postcompile"](policy_context)
    except PolicyException as policy_exc:
        if policy_exc.query_type is None:
            policy_exc.query_type = policy_context.get("query_type")
        if exc is not None:
            raise policy_exc from exc
        raise


@otel_traced("gfql.run", attrs_fn=_gfql_otel_attrs)
def gfql(self: Plottable,
         query: GFQLQuery,
         engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
         output: Optional[str] = None,
         policy: Optional[Dict[str, PolicyFunction]] = None,
         where: Optional[Sequence[WhereComparison]] = None,
         language: Optional[Literal["cypher", "gremlin"]] = None,
         params: Optional[Mapping[str, Any]] = None,
         validate: bool = False,
         shortest_path_backend: str = "auto") -> Plottable:
    """
    Execute a GFQL query - either a chain or a DAG

    Unified entrypoint that automatically detects query type and
    dispatches to the appropriate execution engine.

    :param query: GFQL query - ASTObject, List[ASTObject], Chain, ASTLet, dict, or supported query string
    :param engine: Execution engine (auto, pandas, cudf)
    :param output: For DAGs, name of binding to return (default: last executed)
    :param policy: Optional policy hooks for external control (preload, postload, precall, postcall phases)
    :param where: Optional same-path constraints for list/Chain queries
    :param language: Optional string-query language selector. Defaults to ``"cypher"`` when ``query`` is a string.
    :param params: Optional parameter dictionary for string-query compilation
    :param validate: When ``True``, run local preflight validation before execution via ``g.gfql_validate(...)``.
    :param shortest_path_backend: Backend for shortestPath execution: ``"auto"`` (default),
        ``"igraph"`` (require igraph, raise if missing), ``"cugraph"`` (require cugraph,
        raise if missing), or ``"bfs"`` (always use DataFrame BFS). ``"auto"`` tries
        cugraph on CUDF engine, igraph on pandas, falls back to BFS silently.
    :returns: Resulting Plottable
    :rtype: Plottable
    """
    context = ExecutionContext()

    if policy and context.policy_depth >= 1:
        logger.debug('Policy disabled due to recursion depth limit (depth=%d)', context.policy_depth)
        policy = None

    policy_depth = context.policy_depth
    if policy:
        context.policy_depth = policy_depth + 1

    expanded_policy: Optional[PolicyDict] = None
    if policy:
        expanded_policy = expand_policy(policy)

    try:
        where_param: Optional[List[WhereComparison]] = None
        if where is not None:
            if isinstance(where, (list, tuple)):
                where_param = normalize_where_entries(where)
            else:
                raise ValueError(f"where must be a list of comparisons, got {type(where).__name__}")

        current_depth = context.execution_depth
        current_path = context.operation_path

        if expanded_policy and 'preload' in expanded_policy:
            policy_context: PolicyContext = {
                'phase': 'preload',
                'hook': 'preload',
                'query': query,
                'current_ast': query,  # For top-level, current == query
                'query_type': detect_query_type(query),
                'execution_depth': current_depth,  # Add execution depth
                'operation_path': current_path,  # Add operation path
                'parent_operation': 'query' if current_depth == 0 else current_path.rsplit('.', 1)[0],
                '_policy_depth': policy_depth
            }

            try:
                expanded_policy['preload'](policy_context)
            except PolicyException as e:
                if e.query_type is None:
                    e.query_type = policy_context.get('query_type')
                raise

        dispatch_self = self
        setattr(dispatch_self, "_gfql_shortest_path_backend", shortest_path_backend)
        compiled_query = None

        if where_param and isinstance(query, (dict, ASTLet)):
            raise ValueError("where must be provided inside dict chain under the 'where' key")
        if not isinstance(query, str):
            if language is not None:
                raise ValueError("language is only supported when query is a string")
            if params is not None:
                raise ValueError("params is only supported when query is a string")
        if isinstance(query, str):
            if where_param:
                raise ValueError("where cannot be combined with string queries; embed Cypher predicates in the query itself")
            query_language = language or "cypher"
            _fire_precompile_policy(
                expanded_policy,
                query=query,
                query_language=query_language,
                policy_depth=policy_depth,
                execution_depth=current_depth,
                operation_path=current_path,
            )

        if validate:
            try:
                gfql_preflight_validate(
                    dispatch_self,
                    query,
                    where=where_param,
                    language=language,
                    params=params,
                    strict=True,
                    schema=True,
                    collect_all=False,
                )
            except GFQLValidationError as exc:
                if isinstance(query, str):
                    _fire_postcompile_policy(
                        expanded_policy,
                        query=query,
                        query_language=language or "cypher",
                        exc=exc,
                        policy_depth=policy_depth,
                        execution_depth=current_depth,
                        operation_path=current_path,
                        params=params,
                    )
                raise

        if isinstance(query, str):
            query_language = language or "cypher"
            try:
                compiled_query = _compile_string_query(
                    query,
                    language=language,
                    params=params,
                    node_dtypes=_node_dtypes_for_pushdown(self, engine),
                )
            except GFQLValidationError as exc:
                _fire_postcompile_policy(
                    expanded_policy,
                    query=query,
                    query_language=query_language,
                    exc=exc,
                    policy_depth=policy_depth,
                    execution_depth=current_depth,
                    operation_path=current_path,
                    params=params,
                )
                raise
            _fire_postcompile_policy(
                expanded_policy,
                query=query,
                query_language=query_language,
                exc=None,
                policy_depth=policy_depth,
                execution_depth=current_depth,
                operation_path=current_path,
                params=params,
            )
            if isinstance(compiled_query, CompiledCypherGraphQuery):
                return _execute_graph_query(self, compiled_query, engine=engine, policy=expanded_policy, context=context)
            if isinstance(compiled_query, CompiledCypherQuery):
                if compiled_query.graph_bindings or compiled_query.use_ref:
                    return _execute_query_with_graph_context(self, compiled_query, engine=engine, policy=expanded_policy, context=context)
                query = compiled_query.chain

        if isinstance(query, dict) and query.get("type") == "Let":
            from .ast import ASTLet as _ASTLet
            query = _ASTLet.from_json(query)
        elif isinstance(query, dict) and "chain" in query:
            chain_items: List[ASTObject] = []
            for item in query["chain"]:
                if isinstance(item, dict):
                    from .ast import from_json
                    chain_items.append(from_json(item))
                elif isinstance(item, ASTObject):
                    chain_items.append(item)
                else:
                    raise TypeError(f"Unsupported chain entry type: {type(item)}")
            dict_where = parse_where_json(query.get("where"))
            if not chain_items and dict_where:
                raise ValueError("where requires at least one named node/edge step; empty chains have no aliases")
            query = Chain(chain_items, where=dict_where)
        elif isinstance(query, dict):
            wrapped_dict = {}
            for key, value in query.items():
                if isinstance(value, (ASTNode, ASTEdge)):
                    logger.debug(f'Auto-wrapping {type(value).__name__} in Chain for dict key "{key}"')
                    wrapped_dict[key] = Chain([value])
                else:
                    wrapped_dict[key] = value
            query = ASTLet(wrapped_dict)  # type: ignore

        context.push_depth()

        query_segment = 'dag' if isinstance(query, ASTLet) else 'chain'
        context.push_path(query_segment)

        try:
            if compiled_query is not None and not isinstance(query, Chain):
                logger.debug('GFQL executing compiled string program')
                return _execute_compiled_query(
                    self,
                    compiled_query=compiled_query,
                    engine=engine,
                    policy=expanded_policy,
                    context=context,
                )
            if isinstance(query, ASTLet):
                logger.debug('GFQL executing as DAG')
                return chain_let_impl(dispatch_self, query, engine, output, policy=expanded_policy, context=context)
            elif isinstance(query, Chain):
                logger.debug('GFQL executing as Chain')
                if output is not None:
                    logger.warning('output parameter ignored for chain queries')
                if where_param:
                    if query.where:
                        raise ValueError("where provided for Chain that already includes where")
                    query = Chain(query.chain, where=where_param)
                if compiled_query is not None:
                    return _execute_compiled_query_with_reentry(
                        self,
                        compiled_query=compiled_query,
                        engine=engine,
                        policy=expanded_policy,
                        context=context,
                    )
                return _chain_dispatch(dispatch_self, query, engine, expanded_policy, context)
            elif isinstance(query, ASTObject):
                logger.debug('GFQL executing single ASTObject as chain')
                if output is not None:
                    logger.warning('output parameter ignored for chain queries')
                return _chain_dispatch(dispatch_self, Chain([query], where=where_param), engine, expanded_policy, context)
            elif isinstance(query, list):
                logger.debug('GFQL executing list as chain')
                if output is not None:
                    logger.warning('output parameter ignored for chain queries')

                if not query and where_param:
                    raise ValueError("where requires at least one named node/edge step; empty chains have no aliases")

                converted_query: List[ASTObject] = []
                for item in query:
                    if isinstance(item, dict):
                        from .ast import from_json
                        converted_query.append(from_json(item))
                    else:
                        converted_query.append(item)

                return _chain_dispatch(
                    dispatch_self,
                    Chain(converted_query, where=where_param),
                    engine,
                    expanded_policy,
                    context,
                )
            else:
                raise TypeError(
                    f"Query must be ASTObject, List[ASTObject], Chain, ASTLet, dict, or string. "
                    f"Got {type(query).__name__}"
                )
        finally:
            context.pop_depth()
            context.pop_path()
    finally:
        if policy:
            context.policy_depth = policy_depth


def _chain_dispatch(
    g: Plottable,
    chain_obj: Chain,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
    start_nodes: Optional[DataFrameT] = None,
) -> Plottable:
    engine_name = engine.value if hasattr(engine, "value") else str(engine)
    if chain_obj.where and engine_name in (Engine.POLARS.value, Engine.POLARS_GPU.value):
        # Cross-entity / same-path WHERE routes through DFSamePathExecutor
        # (df_executor.py), which has no native polars implementation. NO pandas
        # fallback (no-silent-fallback policy) — raise honestly.
        raise NotImplementedError(
            "polars engine does not yet natively support cross-entity (same-path) "
            "WHERE; use engine='pandas' for this query "
            "(no pandas fallback; parity-or-error by design)"
        )
    if chain_obj.where:
        if start_nodes is not None:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher MATCH after WITH does not yet support re-entry into MATCH chains with same-path WHERE constraints",
                field="match",
                value="where",
                suggestion="Use a simpler trailing MATCH without additional same-path WHERE constraints.",
                language="cypher",
            )
        first_row_pipeline_idx = next(
            (
                idx
                for idx, op in enumerate(chain_obj.chain)
                if isinstance(op, ASTCall) and is_row_pipeline_call(op.function)
            ),
            None,
        )
        if first_row_pipeline_idx is not None:
            same_path_prefix = chain_obj.chain[:first_row_pipeline_idx]
            row_suffix = chain_obj.chain[first_row_pipeline_idx:]
            validate_chain_schema(g, same_path_prefix, collect_all=False)
            is_cudf = engine == EngineAbstract.CUDF or engine == "cudf"
            engine_enum = Engine.CUDF if is_cudf else Engine.PANDAS
            inputs = build_same_path_inputs(
                g,
                same_path_prefix,
                chain_obj.where,
                engine=engine_enum,
                include_paths=False,
            )
            executor = DFSamePathExecutor(inputs)
            matched = _materialize_split_alias_columns(executor.run(), executor)
            return chain_impl(matched, row_suffix, engine, policy=policy, context=context)
        validate_chain_schema(g, chain_obj.chain, collect_all=False)
        is_cudf = engine == EngineAbstract.CUDF or engine == "cudf"
        engine_enum = Engine.CUDF if is_cudf else Engine.PANDAS
        inputs = build_same_path_inputs(
            g,
            chain_obj.chain,
            chain_obj.where,
            engine=engine_enum,
            include_paths=False,
        )
        return execute_same_path_chain(
            inputs.graph,
            inputs.chain,
            inputs.where,
            inputs.engine,
            inputs.include_paths,
        )
    return chain_impl(g, chain_obj.chain, engine, policy=policy, context=context, start_nodes=start_nodes)
