"""GFQL unified entrypoint for chains, DAGs, and local string-compiled queries."""
# ruff: noqa: E501

from collections import OrderedDict
from dataclasses import replace
import re
import threading
import pandas as pd
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
from graphistry.Plottable import Plottable
from graphistry.Engine import Engine, EngineAbstract, POLARS_ENGINES, df_concat, df_cons, df_to_engine, df_unique, is_polars_df, is_series_like, resolve_engine, series_to_pylist
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
from graphistry.compute.gfql.identifiers import TRAIL_ARM_EDGE_ALIAS_PREFIX
from graphistry.compute.gfql.same_path_types import (
    EDGE_IDENTITY_COLUMN,
    NODE_IDENTITY_COLUMN,
    WhereComparison,
    normalize_where_entries,
    parse_where_json,
)
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.ast import CypherParams
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.exec_context import attach_row_exec_context, clear_row_exec_context
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
from graphistry.compute.gfql.cypher.reentry.carried_outputs import (
    carried_output_sources as _carried_output_sources,
    optional_reentry_aggregate_fill_values as _optional_reentry_aggregate_fill_values,
)
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
from graphistry.compute.typing import DataFrameT, FilterDict, SeriesT, NodeDtypes
from graphistry.compute.util.generate_safe_column_name import (
    generate_safe_column_name,
    generate_safe_column_name_from,
)
from graphistry.compute.gfql.identifiers import EDGE_INDEX_BASE
from graphistry.compute.validate.validate_schema import validate_chain_schema
from graphistry.compute.gfql_validate import gfql_validate as gfql_preflight_validate
from graphistry.otel import otel_traced, otel_detail_enabled

logger = setup_logger(__name__)


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


def _slice_rows(rows_df: DataFrameT, start: int, stop: int) -> DataFrameT:
    """Positional half-open row slice ``[start, stop)``, engine-dispatched."""
    if is_polars_df(rows_df):
        return rows_df.slice(start, stop - start)
    return rows_df.iloc[start:stop]


def _projector_recorded_matched_seed_ids(
    alignment_result: Plottable,
    alignment_output_name: str,
) -> bool:
    meta = getattr(alignment_result, "_cypher_entity_projection_meta", None)
    return (
        isinstance(meta, dict)
        and alignment_output_name in meta
        and "ids" in meta[alignment_output_name]
    )


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
    if (
        resolve_engine(cast(Any, engine), result) in POLARS_ENGINES
        and not _projector_recorded_matched_seed_ids(alignment_result, alignment_output_name)
    ):
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
    if not is_series_like(matched_ids):
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

    base_ids = series_to_pylist(base_rows_df[node_col])
    matched_id_list = series_to_pylist(matched_ids)
    if len(base_ids) == actual_rows and base_ids == matched_id_list:
        return result

    concrete_engine = resolve_engine(cast(Any, engine), result)
    df_ctor = df_cons(concrete_engine)
    concat = df_concat(concrete_engine)
    fill_columns_spanning_projected_frame = (
        list(rows_df.columns) if rows_df is not None else list(null_row.keys())
    )
    fill_df = df_ctor({col: [null_row.get(col)] for col in fill_columns_spanning_projected_frame})
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
            segments.append(_slice_rows(rows_df, group_start, matched_idx))
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


def _semi_join_prune_arm_rows_to_base_keys(
    opt_rows_df: DataFrameT,
    joined: DataFrameT,
    join_cols: List[str],
) -> DataFrameT:
    """Arm rows restricted to join-key values already present in the accumulated result."""
    if is_polars_df(joined):
        import polars as pl
        if len(join_cols) == 1:
            # polars-stub gap: ``is_polars_df`` cannot narrow the eager-or-lazy union.
            return opt_rows_df.filter(pl.col(join_cols[0]).is_in(joined[join_cols[0]]))  # type: ignore[index,arg-type]
        return opt_rows_df.join(joined.select(join_cols).unique(), on=join_cols, how="inner")
    if len(join_cols) == 1:
        return opt_rows_df[opt_rows_df[join_cols[0]].isin(joined[join_cols[0]])]
    return opt_rows_df.merge(joined[join_cols].drop_duplicates(), on=join_cols, how="inner")


def _null_extend_full_arm_binding_schema(
    joined: DataFrameT,
    opt_rows_df: Optional[DataFrameT],
    opt_only_aliases: Sequence[str],
) -> DataFrameT:
    """Every column the arm would have bound, as a typed null — not just its bare aliases."""
    if is_polars_df(joined):
        import polars as pl
        if opt_rows_df is not None:
            arm_schema = opt_rows_df.schema
            joined = joined.with_columns([
                pl.lit(None, dtype=arm_schema[col]).alias(col)
                for col in opt_rows_df.columns
                if col not in joined.columns
            ])
        for alias in opt_only_aliases:
            if alias not in joined.columns:
                joined = joined.with_columns(pl.lit(None).alias(alias))
        return joined
    if opt_rows_df is not None:
        for col in opt_rows_df.columns:
            if col not in joined.columns:
                joined[col] = None
    for alias in opt_only_aliases:
        if alias not in joined.columns:
            joined[alias] = None
    return joined


def _synthesize_bare_alias_from_prefixed_column(
    joined: DataFrameT,
    opt_only_aliases: Sequence[str],
) -> DataFrameT:
    """Bare ``alias`` column mirrored from that alias's first ``alias.`` prefixed column."""
    polars = is_polars_df(joined)
    if polars:
        import polars as pl
    for alias in opt_only_aliases:
        if alias in joined.columns:
            continue
        prefix = f"{alias}."
        marker_col = next((c for c in joined.columns if c.startswith(prefix)), None)
        if marker_col is None:
            continue
        if polars:
            joined = joined.with_columns(pl.col(marker_col).alias(alias))
        else:
            marker = joined[marker_col]
            joined[alias] = marker.where(marker.notna(), other=None)
    return joined


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

    def _optional_arm_membership_chain(
        binding_ops: Sequence[ASTObject],
        shared_node_aliases: Sequence[str],
        joined_rows: DataFrameT,
    ) -> Optional[List[ASTObject]]:
        """Polars twin of the start_nodes pruning: rewrite the arm's first node op with an
        id-membership filter (the shared alias's bound ids) instead of seeding via
        start_nodes, which the polars bindings-row path declines by contract. Returns the
        pruned binding chain, or None when the shape doesn't qualify (run unseeded)."""
        if not binding_ops:
            return None
        first_op = binding_ops[0]
        if not isinstance(first_op, ASTNode):
            return None
        first_alias = getattr(first_op, "_name", None)
        if not isinstance(first_alias, str) or first_alias not in shared_node_aliases:
            return None
        if first_op.query is not None:
            return None
        filter_dict = dict(first_op.filter_dict or {})
        if node_col in filter_dict:
            return None  # id already constrained; don't intersect two conditions on one key
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
        seed_col = joined_rows[joined_col]
        if is_polars_df(joined_rows):
            seed_ids = seed_col.drop_nulls().unique().to_list()
        else:
            seed_ids = seed_col.dropna().drop_duplicates().tolist()
        if not seed_ids:
            return None
        pruned_first = ASTNode(
            filter_dict={**filter_dict, node_col: seed_ids},
            name=first_alias,
        )
        return [pruned_first, *binding_ops[1:]]

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

        seed_src = joined_rows[[joined_col]]
        # each branch builds ``seed_frame`` directly rather than rebinding ``seed_src``: the
        # polars result would otherwise widen the variable and break the pandas/cuDF branch below
        # (``is_polars_df`` is a TypeGuard -- it does not narrow the negative branch back).
        if is_polars_df(seed_src):
            seed_frame = cast(DataFrameT, df_to_engine(
                seed_src.drop_nulls().unique().rename({joined_col: node_col}), concrete_engine))
        else:
            seed_frame = cast(DataFrameT, df_to_engine(
                seed_src.dropna().drop_duplicates().rename(columns={joined_col: node_col}), concrete_engine))
        # Declared, not cast: selecting one column off a frame is a Series on every engine, so
        # the annotation states that directly instead of re-asserting it at the call site.
        seed_ids: SeriesT = seed_frame[node_col]
        node_ids: SeriesT = base_nodes[node_col]
        if is_polars_df(base_nodes):
            return cast(DataFrameT, base_nodes.filter(node_ids.is_in(seed_ids)))
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
        opt_start_nodes = None
        if not arm.chain.where:
            if concrete_engine in POLARS_ENGINES:
                pruned_chain = _optional_arm_membership_chain(
                    opt_binding_chain,
                    arm.shared_node_aliases,
                    joined,
                )
                if pruned_chain is not None:
                    opt_binding_chain = pruned_chain
            else:
                opt_start_nodes = _optional_arm_start_nodes(
                    opt_binding_chain,
                    arm.shared_node_aliases,
                    joined,
                )
        opt_binding_ops = serialize_binding_ops(opt_binding_chain)
        opt_with_rows = Chain(
            list(opt_binding_chain) + [ASTCall("rows", {"binding_ops": opt_binding_ops})] + opt_post_ops,
            where=arm.chain.where,
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
            opt_rows_df = _semi_join_prune_arm_rows_to_base_keys(opt_rows_df, joined, join_cols)
            if is_polars_df(joined):
                joined = joined.join(opt_rows_df.select(opt_only_cols), on=join_cols, how="left")
            else:
                joined = joined.merge(opt_rows_df[opt_only_cols], on=join_cols, how="left")
        else:
            joined = _null_extend_full_arm_binding_schema(joined, opt_rows_df, arm.opt_only_aliases)

        joined = _synthesize_bare_alias_from_prefixed_column(joined, arm.opt_only_aliases)

    # Delegate RETURN / ORDER BY / SKIP / LIMIT to the standard row pipeline.
    joined_plottable = base_graph.bind()
    joined_plottable._nodes = joined
    joined_plottable._edges = df_ctor()

    return _chain_dispatch(joined_plottable, plan.post_join_chain, engine, policy, context)




# Fast-path specialization helpers extracted to gfql_fast_paths.py (pure move; #1731).
from .gfql_fast_paths import (
    _connected_join_two_star_fast_grouped_count,
    _connected_join_two_star_fast_rows,
    _execute_seeded_typed_hop_fast_path,
    _execute_single_hop_grouped_aggregate_fast_path,
    _execute_two_hop_count_fast_path,
)

def _filter_dicts_provably_disjoint(first: Optional[FilterDict], second: Optional[FilterDict]) -> bool:
    if not first or not second:
        return False
    return any(
        isinstance(first[key], (str, int, float, bool)) and isinstance(second[key], (str, int, float, bool))
        and first[key] != second[key]
        for key in set(first) & set(second)
    )


def _connected_join_trail_arms(
    plan: ConnectedMatchJoinPlan,
    *,
    identity_col: str,
) -> Optional[Tuple[Tuple[Chain, ...], Tuple[Tuple[str, ...], ...]]]:
    """Rewritten arm chains + per-arm relationship identity columns, or None.

    openCypher relationship uniqueness spans the WHOLE match clause, but the arm join
    is a cartesian product that drops edge identity, so an edge fitting two arms binds
    twice. Naming every anonymous arm edge surfaces its ``<alias>.<identity_col>`` identity
    in the arm rows so the join can drop those bindings. Returns None when no arm pair can
    share an edge (nothing to enforce) or an arm is variable-length (no per-hop identity).
    """
    chains = plan.pattern_chains
    if len(chains) < 2:
        return None
    arm_edges: List[List[ASTEdge]] = []
    for pattern_chain in chains:
        edges = [op for op in pattern_chain.chain if isinstance(op, ASTEdge)]
        if not edges:
            return None
        for edge_op in edges:
            if edge_op.to_fixed_point or edge_op.hops != 1 or edge_op.min_hops is not None or edge_op.max_hops is not None:
                return None
        arm_edges.append(edges)
    can_share = any(
        not _filter_dicts_provably_disjoint(first.edge_match, second.edge_match)
        for index, edges in enumerate(arm_edges)
        for other in arm_edges[index + 1:]
        for first in edges
        for second in other
    )
    if not can_share:
        return None

    from graphistry.compute.ast import from_json as ast_from_json

    rewritten: List[Chain] = []
    identity_columns: List[Tuple[str, ...]] = []
    for index, pattern_chain in enumerate(chains):
        ops: List[ASTObject] = []
        columns: List[str] = []
        for position, op in enumerate(pattern_chain.chain):
            if isinstance(op, ASTEdge):
                alias = getattr(op, "_name", None) or f"{TRAIL_ARM_EDGE_ALIAS_PREFIX}{index}_{position}__"
                if getattr(op, "_name", None) is None:
                    cloned = ast_from_json({**op.to_json(), "name": alias}, validate=False)
                    assert isinstance(cloned, ASTEdge)
                    op = cloned
                columns.append(f"{alias}.{identity_col}")
            ops.append(op)
        rewritten.append(Chain(ops, where=pattern_chain.where))
        identity_columns.append(tuple(columns))
    return tuple(rewritten), tuple(identity_columns)


def _is_polars_frame(frame: object) -> bool:
    return "polars" in type(frame).__module__


def _trail_edge_identity_col(base_graph: Plottable) -> str:
    """Name for the per-arm relationship identity column, safe against user edge columns."""
    edges = base_graph._edges
    if edges is None:
        return generate_safe_column_name_from(EDGE_INDEX_BASE, ())
    # The bound frame may still be arrow here (df_to_engine runs later in _with_edge_identity).
    return generate_safe_column_name(EDGE_INDEX_BASE, edges)


def _with_edge_identity(base_graph: Plottable, *, engine: Engine, identity_col: str) -> Plottable:
    edges_obj = base_graph._edges
    if edges_obj is None:
        return base_graph
    edges = df_to_engine(edges_obj, engine, warn=False)
    if identity_col in edges.columns:
        return base_graph
    if _is_polars_frame(edges):
        import polars as pl
        return base_graph.edges(edges.with_columns(pl.int_range(pl.len()).alias(identity_col)))
    return base_graph.edges(edges.assign(**{identity_col: range(len(edges))}))


def _drop_shared_relationship_bindings(
    joined_rows: DataFrameT,
    left_columns: Sequence[str],
    right_columns: Sequence[str],
) -> DataFrameT:
    pairs = [
        (left, right)
        for left in left_columns
        for right in right_columns
        if left in joined_rows.columns and right in joined_rows.columns
    ]
    if not pairs:
        return joined_rows
    if _is_polars_frame(joined_rows):
        import polars as pl
        pl_rows: "pl.DataFrame" = joined_rows  # engine seam: polars frame rides engine-agnostic DataFrameT
        expr = pl.lit(True)
        for left, right in pairs:
            expr = expr & (pl.col(left) != pl.col(right))
        return pl_rows.filter(expr)
    mask = None
    for left, right in pairs:
        keep = joined_rows[left] != joined_rows[right]
        mask = keep if mask is None else (mask & keep)
    return joined_rows[mask]


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

    # One cache scope per connected-join execution: shared across the fast paths but never
    # persisted on the caller's Plottable, so a second gfql() after an in-place mutation
    # recomputes instead of returning a stale cached answer (BLOCKER 1).
    cache_store: Dict[str, Any] = {}

    trail_identity_col = _trail_edge_identity_col(base_graph)
    trail_arms = _connected_join_trail_arms(plan, identity_col=trail_identity_col)
    arms_may_share_an_edge = trail_arms is not None
    arm_chains = plan.pattern_chains if trail_arms is None else trail_arms[0]
    arm_identity_columns: Tuple[Tuple[str, ...], ...] = (
        tuple(() for _ in plan.pattern_chains) if trail_arms is None else trail_arms[1]
    )
    if trail_arms is not None:
        base_graph = _with_edge_identity(
            base_graph, engine=requested_engine, identity_col=trail_identity_col
        )

    # Both two-star fast paths emit the raw arm product, so they serve disjoint arms only.
    fast_grouped_count = (
        None if arms_may_share_an_edge
        else _connected_join_two_star_fast_grouped_count(base_graph, plan, engine=requested_engine, cache_store=cache_store)
    )
    if fast_grouped_count is not None:
        out = base_graph.bind()
        out._nodes = fast_grouped_count
        out._edges = df_ctor()
        return out

    fast_rows = (
        None if arms_may_share_an_edge
        else _connected_join_two_star_fast_rows(base_graph, plan, engine=requested_engine, cache_store=cache_store)
    )
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
    joined_identity_columns: List[str] = []
    pattern_attach_prop_aliases = plan.pattern_attach_prop_aliases or tuple(None for _ in plan.pattern_chains)
    for idx, pattern_chain in enumerate(arm_chains):
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
            op._name
            for op in pattern_chain.chain
            if isinstance(op, _ASTNode) and isinstance(op._name, str)
        ]
        identity_columns = [
            column for column in arm_identity_columns[idx] if column in pattern_rows.columns
        ]
        keep_binding_columns = [
            column for column in _binding_join_columns(pattern_rows)
            if column in identity_columns or not str(column).startswith(TRAIL_ARM_EDGE_ALIAS_PREFIX)
        ] + [alias for alias in node_aliases if alias in pattern_rows.columns]
        pattern_rows = cast(DataFrameT, pattern_rows[keep_binding_columns])
        if joined_rows is None:
            joined_rows = pattern_rows
            joined_identity_columns = list(identity_columns)
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
        if identity_columns and joined_identity_columns:
            joined_rows = _drop_shared_relationship_bindings(
                joined_rows, joined_identity_columns, identity_columns
            )
        joined_identity_columns.extend(identity_columns)

    if joined_rows is None:
        out = base_graph.bind()
        out._nodes = df_ctor()
        out._edges = df_ctor()
        return out

    if trail_arms is not None:
        drop_columns = [
            column for column in joined_rows.columns
            if str(column).startswith(TRAIL_ARM_EDGE_ALIAS_PREFIX)
        ]
        if drop_columns:
            joined_rows = (joined_rows.drop(drop_columns) if _is_polars_frame(joined_rows)
                           else joined_rows.drop(columns=drop_columns))
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


def _polars_union_dtype_is_numeric(dtype: Any) -> bool:  # hygiene-ok: explicit-any -- polars DataType, imported lazily
    import polars as pl
    if dtype == pl.Boolean:
        return False
    checker = getattr(dtype, "is_numeric", None)
    if callable(checker):
        return bool(checker())
    return False


def _reject_unrepresentable_polars_union(frames: List[DataFrameT]) -> None:
    """Decline a UNION whose branches disagree on a column's VALUE TYPE.

    polars' ``vertical_relaxed`` concat coerces to a common supertype, which
    stringifies an Int64 branch next to a String branch (and turns ``true`` into
    ``1`` next to an Int64 branch) — silently changing values and then deleting
    rows via the DISTINCT that follows. openCypher keeps the branch
    values distinct; a polars column cannot, so decline typed rather than answer.
    Numeric-vs-numeric widening stays served (openCypher ``1 = 1.0`` is true).
    """
    import polars as pl
    for name in frames[0].columns:
        dtypes = []
        for frame in frames:
            if name not in frame.columns:
                continue
            dtype = frame.schema[name]
            if dtype == pl.Null:
                continue  # all-null column adopts any branch's type
            if dtype not in dtypes:
                dtypes.append(dtype)
        if len(dtypes) <= 1:
            continue
        if all(_polars_union_dtype_is_numeric(dtype) for dtype in dtypes):
            continue
        raise NotImplementedError(
            "polars engine does not yet natively support UNION over branches with "
            f"different value types for column {name!r} ({', '.join(str(d) for d in dtypes)}); "
            "use engine='pandas' for this query (no pandas fallback; parity-or-error by design)"
        )


def _pandas_union_widen_boolean_columns(frames: List[DataFrameT]) -> List[DataFrameT]:
    """Keep BOOLEAN branch values distinguishable from numeric ones.

    pandas concat of a bool column next to an int column upcasts ``True`` to ``1``,
    after which DISTINCT collapses two openCypher-distinct values (``true = 1`` is
    false) into one row. Widening to object keeps ``True`` a bool.
    """
    widen: Set[Any] = set()
    for name in frames[0].columns:
        kinds = {
            getattr(frame[name].dtype, "kind", None)
            for frame in frames
            if name in frame.columns
        }
        if "b" in kinds and len(kinds) > 1:
            widen.add(name)
    if not widen:
        return frames
    return [
        frame.assign(**{name: frame[name].astype(object) for name in widen if name in frame.columns})
        for frame in frames
    ]


def _concat_union_branch_rows(
    row_frames: List[DataFrameT],
    concrete_engine: Engine,
    concat: Callable[..., DataFrameT],
) -> DataFrameT:
    """Row-concat UNION branch frames without letting a branch's dtype rewrite another's values."""
    # UNION aligns columns by NAME (Neo4j); the output keeps the first branch's order.
    first_columns = list(row_frames[0].columns)
    frames = [
        frame if list(frame.columns) == first_columns else frame[first_columns]
        for frame in row_frames
    ]
    non_empty = [frame for frame in frames if len(frame) > 0]
    if non_empty and len(non_empty) != len(frames):
        # A 0-row branch contributes no rows, and its dtype must not drag the union supertype.
        frames = non_empty
    if len(frames) == 1:
        return frames[0]
    if concrete_engine in POLARS_ENGINES:
        _reject_unrepresentable_polars_union(frames)
    elif concrete_engine == Engine.PANDAS:
        frames = _pandas_union_widen_boolean_columns(frames)
    return concat(frames, ignore_index=True, sort=False)


def _union_dedup_key(value: Any) -> Any:  # hygiene-ok: explicit-any -- arbitrary cypher cell value
    """Hashable openCypher-identity key for a UNION DISTINCT cell.

    Two rules the raw pandas value does not carry: a float NaN in an object column is
    the pandas spelling of a missing value and must dedup against ``None``,
    and a BOOLEAN is never equal to a NUMBER even though ``True == 1`` in Python
   .
    """
    if value is None:
        return ("null",)
    if isinstance(value, float) and value != value:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (list, tuple)):
        return ("list", tuple(_union_dedup_key(item) for item in value))
    if isinstance(value, dict):
        return ("map", tuple(sorted((str(k), _union_dedup_key(v)) for k, v in value.items())))
    return ("value", value)


def _union_distinct_rows(union_rows: DataFrameT, concrete_engine: Engine) -> DataFrameT:
    """UNION DISTINCT dedup under openCypher value identity (pandas object columns)."""
    if concrete_engine != Engine.PANDAS:
        return df_unique(union_rows, concrete_engine)
    object_cols = [
        name for name in union_rows.columns
        if getattr(union_rows[name].dtype, "kind", None) == "O"
    ]
    if not object_cols:
        return df_unique(union_rows, concrete_engine)
    try:
        key_frame = pd.DataFrame(
            {
                name: (
                    union_rows[name].map(_union_dedup_key)
                    if name in object_cols
                    else union_rows[name]
                )
                for name in union_rows.columns
            },
            index=union_rows.index,
        )
        keep = ~key_frame.duplicated()
    except (TypeError, ValueError):
        return df_unique(union_rows, concrete_engine)
    return union_rows.loc[keep].reset_index(drop=True)


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
        union_rows = (
            df_ctor() if not row_frames
            else _concat_union_branch_rows(row_frames, concrete_engine, concat)
        )
        if compiled_query.union_kind == "distinct" and len(union_rows) > 0:
            union_rows = cast(DataFrameT, _union_distinct_rows(union_rows, concrete_engine))
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


if TYPE_CHECKING:
    from graphistry.compute.gfql.lazy import ExecutionTarget


def _policied_auto_serves_via_pandas_until_the_polars_route_emits_hooks(
    engine: Union[EngineAbstract, str],
    policy: Optional[Dict[str, PolicyFunction]],
    g: Plottable,
) -> bool:
    """Transitional: the polars route does not emit the postload/postchain policy hooks yet."""
    return (
        (engine == EngineAbstract.AUTO or engine == EngineAbstract.AUTO.value)
        and policy is not None
        and resolve_engine(EngineAbstract.AUTO, g) == Engine.POLARS
    )


def _fast_path_execution_target_ignoring_requested_engine(
    engine: Union[EngineAbstract, Engine, str],
) -> "ExecutionTarget":
    """Not GPU until every fast-path arm is GPU-or-decline (#1824)."""
    from graphistry.compute.gfql.lazy import ExecutionTarget
    return ExecutionTarget.CPU


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
        # Record served/declined at the CALL SITE rather than inside each fast path:
        # this is where the decision is consumed, it is one place instead of N return
        # paths, and it cannot be bypassed the way patching a directly-imported name is.
        from graphistry.compute.gfql.index.api import record_fast_path_decision
        from graphistry.compute.gfql.lazy import ExecutionTarget, target_mode
        _fp_target = _fast_path_execution_target_ignoring_requested_engine(engine)

        _FastPathName = Literal["single_hop_grouped_aggregate", "two_hop_count", "seeded_typed_hop"]

        def _try_fast(path_name: _FastPathName, run: Callable[[], Optional[Plottable]]) -> Optional[Plottable]:
            try:
                with target_mode(_fp_target):
                    out = run()
                reason = "served" if out is not None else "declined; caller falls back"
            except NotImplementedError:
                if _fp_target != ExecutionTarget.GPU:
                    raise
                out = None
                reason = "declined; plan not GPU-executable, chain route answers"
            record_fast_path_decision(
                path=path_name, engine=engine, served=out is not None, reason=reason)
            return out

        fast_grouped = _try_fast(
            "single_hop_grouped_aggregate",
            lambda: _execute_single_hop_grouped_aggregate_fast_path(base_graph, compiled_query.chain, engine=engine))
        if fast_grouped is not None:
            return fast_grouped
        fast_count = _try_fast(
            "two_hop_count",
            lambda: _execute_two_hop_count_fast_path(base_graph, compiled_query.chain, engine=engine))
        if fast_count is not None:
            return fast_count
        fast_hop = _try_fast(
            "seeded_typed_hop",
            lambda: _execute_seeded_typed_hop_fast_path(
                base_graph, compiled_query, physical_plan,
                engine=engine, policy=policy, context=context, start_nodes=start_nodes))
        if fast_hop is not None:
            return fast_hop
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
        _chain_ops = list(compiled_query.chain.chain) if compiled_query.chain is not None else []
        _first_op = _chain_ops[0] if _chain_ops else None
        if (
            isinstance(_first_op, ASTCall)
            and _first_op.function == "rows"
            and _first_op.params.get("binding_ops") is not None
        ):
            # #1786: on the no-seed-rows path `_seeded_dispatch_graph` hands back
            # `base_graph` ITSELF (the caller's object), so this must land on a copy.
            dispatch_graph = attach_row_exec_context(dispatch_graph, start_nodes=start_nodes)

    result = _chain_dispatch(dispatch_graph, compiled_query.chain, engine, policy, context, start_nodes=start_nodes)
    # Attach/detach pair (#1786): the chain has run, so the seed is spent and must not
    # ride out on the result -- a follow-up query on it is about a DIFFERENT graph.
    result = clear_row_exec_context(result)
    if compiled_query.empty_result_row is not None:
        result = _apply_empty_result_row(
            result,
            engine=engine,
            empty_result_row=compiled_query.empty_result_row,
        )
    if compiled_query.result_projection is not None:
        row_guard_needs_single_column_entity_text = (
            compiled_query.optional_projection_row_guard is not None
        )
        result = apply_result_projection(
            result,
            compiled_query.result_projection,
            structured=not row_guard_needs_single_column_entity_text,
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
            aggregate_fill_values=_optional_reentry_aggregate_fill_values(compiled_query),
            carried_outputs=_carried_output_sources(compiled_query),
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
    params: Optional[CypherParams] = None,
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


_COMPILED_STRING_QUERY_CACHE_MAX = 128

#: Compiled plans, keyed by everything compilation actually depends on.
#:
#: ``compile_cypher_query(parse_cypher(query), params=..., node_dtypes=...)`` takes exactly
#: those inputs and never sees the graph, so the plan for a given key is the same plan
#: no matter which ``Plottable`` asked for it. This cache used to hang off the caller's
#: Plottable by ``setattr``, which partitioned it by something that cannot change the
#: answer -- so a ONE-SHOT query on a fresh graph recompiled a plan the process was already
#: holding, so the FIRST query on a Plottable paid a recompile the second did not --
#: measurably, across every benchmarked query shape.
#:
#: Bounded LRU rather than clear-on-full so a hot query cannot be evicted by a burst of
#: cold ones. Values are ``@dataclass(frozen=True)`` chains and plans -- no DataFrame is
#: reachable from a compiled query, so a process-lifetime cache cannot pin user data.
_COMPILED_STRING_QUERY_CACHE: "OrderedDict[Tuple[str, str, Tuple[Tuple[str, Any], ...], Optional[Tuple[Tuple[str, str], ...]], str], Any]" = OrderedDict()
#: Guards the LRU. Plain dict ops are individually atomic under the GIL, but
#: get-then-move_to_end and insert-then-evict are not, and this cache is now shared across
#: threads rather than isolated per graph.
_COMPILED_STRING_QUERY_CACHE_LOCK = threading.Lock()


def _clear_compiled_string_query_cache() -> None:
    with _COMPILED_STRING_QUERY_CACHE_LOCK:
        _COMPILED_STRING_QUERY_CACHE.clear()


from graphistry.compute.gfql.cache_registry import register_clearable_callable  # noqa: E402
register_clearable_callable("_COMPILED_STRING_QUERY_CACHE", _clear_compiled_string_query_cache)


def _compile_cache_value_key(value: Any) -> Optional[Any]:
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            item_key = _compile_cache_value_key(item)
            if item_key is None:
                return None
            items.append(item_key)
        return ("list", tuple(items))
    if isinstance(value, Mapping):
        items = []
        seen_keys = set()
        for key, item in value.items():
            key_str = str(key)
            if key_str in seen_keys:
                return None
            seen_keys.add(key_str)
            item_key = _compile_cache_value_key(item)
            if item_key is None:
                return None
            items.append((key_str, item_key))
        return ("mapping", tuple(sorted(items)))
    return None


def _compile_cache_params_key(params: Optional[CypherParams]) -> Optional[Tuple[Tuple[str, Any], ...]]:
    if not params:
        return ()
    items = []
    seen_keys = set()
    for key, value in params.items():
        key_str = str(key)
        if key_str in seen_keys:
            return None
        seen_keys.add(key_str)
        value_key = _compile_cache_value_key(value)
        if value_key is None:
            return None
        items.append((key_str, value_key))
    return tuple(sorted(items))


def _node_dtypes_cache_key(
    node_dtypes: Optional[NodeDtypes],
) -> Optional[Tuple[Tuple[str, str], ...]]:
    """Hashable key for node_dtypes so the compile cache never reuses an engine-specific
    pushdown plan across differing dtype views (e.g. pandas numpy dtypes vs polars dtypes)."""
    if node_dtypes is None:
        return None
    return tuple(sorted((str(col), str(dtype)) for col, dtype in node_dtypes.items()))


def gfql_clear_caches() -> None:
    """Empty every PROCESS-LIFETIME GFQL cache.

    GFQL memoizes work that is a pure function of its inputs: compiled plans here, the
    parse caches behind ``parse_cypher`` and the row-expression parser, and the polars
    single-alias predicate-lowering memo. All are bounded, so
    this is not a leak valve -- it exists because a process-lifetime cache that cannot be
    emptied is untestable (results become order-dependent) and unbudgetable (a long-lived
    server cannot reclaim the memory on demand).

    Caches keyed to a specific graph are NOT touched: they live on their ``Plottable`` and
    die with it. This function is therefore NOT the recovery from an in-place mutation of a
    bound frame -- the resident adjacency-index registry survives it. Rebind a fresh frame object
    or ``drop_index`` instead (see :meth:`ComputeMixin.gfql`, which names the same two).
    Neither are the process-lifetime *singletons* -- Lark parser objects,
    compiled regexes, dependency probes -- which are a function of the code, not of any
    input; see ``clear_cypher_parser_caches`` and
    ``graphistry/tests/compute/gfql/test_clear_caches_covers_every_cache.py``, which fails
    if a new memo appears and is neither cleared here nor exempted with a written reason.

    Every clear is UNCONDITIONAL and runs through the cache registry
    (``graphistry/compute/gfql/cache_registry.py``): each cache registers its own bound
    clear handle at its definition site, so there is no later name lookup to get wrong.
    An earlier version looked targets up with ``getattr(obj, "cache_clear", None)`` and
    skipped whatever came back ``None``; naming the wrong object -- ``parse_cypher``,
    whose memo actually lives on the ``_parse_cypher_cached`` body -- turned the whole
    call into a silent no-op for days. ``clear_all`` raises when the registry is empty.
    """
    # Importing a cache-hosting module is what registers its caches; import the
    # clearable hosts here so a caller who never touched them still empties them.
    # All three imports are safe on minimal installs (polars/lark guarded inside).
    import graphistry.compute.gfql.cypher.parser  # noqa: F401
    import graphistry.compute.gfql.expr_parser  # noqa: F401
    import graphistry.compute.gfql.lazy.engine.polars.row_pipeline  # noqa: F401
    from graphistry.compute.gfql.cache_registry import clear_all
    clear_all()


def _compile_string_query(
    query: str,
    *,
    language: Optional[Literal["cypher", "gremlin"]],
    params: Optional[CypherParams],
    engine_key: str,
    node_dtypes: Optional[NodeDtypes] = None,
) -> Any:
    query_language = language or "cypher"
    if query_language != "cypher":
        raise GFQLValidationError(
            ErrorCode.E108,
            f"Unsupported GFQL string language {query_language!r}",
            field="language",
            value=query_language,
            suggestion="Use language=\"cypher\" for now; Gremlin string compilation is not implemented yet.",
            language="gfql",
        )
    params_key = _compile_cache_params_key(params)
    # node_dtypes (from #1730/#1729 pushdown) makes compilation engine-dependent: the same
    # (query, params) yields a different pushdown plan under pandas/cuDF vs polars dtypes. The
    # compile cache (#1731) must therefore key on node_dtypes too, or a plan compiled for one
    # engine would be wrongly reused for another on the same graph.
    node_dtypes_key = _node_dtypes_cache_key(node_dtypes)
    # params that cannot be keyed (unhashable / opaque values) are simply not cached.
    cache_key = (
        (query_language, query, params_key, node_dtypes_key, engine_key)
        if params_key is not None else None)
    if cache_key is not None:
        with _COMPILED_STRING_QUERY_CACHE_LOCK:
            hit = _COMPILED_STRING_QUERY_CACHE.get(cache_key)
            if hit is not None:
                _COMPILED_STRING_QUERY_CACHE.move_to_end(cache_key)
                return hit

    compiled = compile_cypher_query(parse_cypher(query), params=params, node_dtypes=node_dtypes)
    if cache_key is not None:
        with _COMPILED_STRING_QUERY_CACHE_LOCK:
            _COMPILED_STRING_QUERY_CACHE[cache_key] = compiled
            _COMPILED_STRING_QUERY_CACHE.move_to_end(cache_key)
            while len(_COMPILED_STRING_QUERY_CACHE) > _COMPILED_STRING_QUERY_CACHE_MAX:
                _COMPILED_STRING_QUERY_CACHE.popitem(last=False)
    return compiled


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
    params: Optional[CypherParams],
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
    params: Optional[CypherParams],
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


def _is_cudf_frame(df: Any) -> bool:
    """Import-light cudf.DataFrame check, matching the legacy AUTO->CUDF resolution's own
    module test (``resolve_engine``): ``cudf.core.dataframe`` in the type's module. This
    deliberately excludes dask_cudf (``dask_cudf.core``) and cudf.pandas proxies — only
    frames the legacy arm would have resolved to Engine.CUDF are candidates for the
    polars-gpu preference, so the two arms agree on what "a cudf-frame graph" is."""
    return df is not None and 'cudf.core.dataframe' in type(df).__module__


def _polars_gpu_probe() -> bool:
    """Seam for the AUTO cuDF guard: is the cudf-polars GPU target genuinely usable?

    Thin indirection over the process-singleton probe (``lazy.polars_gpu_available``,
    registered exempt in the GFQL cache registry) so tests can pin guard behavior by
    monkeypatching THIS name without touching the cached probe itself."""
    from graphistry.compute.gfql.lazy import polars_gpu_available
    return polars_gpu_available()


# Engine the AUTO cuDF route pins on its recursion. A module-level seam (not an inline
# literal) so CPU-only tests can swap in 'polars' and exercise the ENTIRE route —
# guard -> recursion -> coercion -> result frames back to cudf — everything but the GPU
# collect itself, which only a GPU lane can genuinely test.
_AUTO_CUDF_ROUTE_ENGINE: str = Engine.POLARS_GPU.value


def _route_result_frames_to_cudf(res: Plottable) -> Plottable:
    """cudf-frames-in must mean cudf-frames-out: cross the routed result's polars frames
    back to cuDF via Arrow (``df_to_engine(..., Engine.CUDF)`` — lossless nulls/dtypes,
    no pandas detour). A conversion failure raises ``NotImplementedError`` so the guard's
    decline path re-serves the query on the legacy CUDF route instead of leaking polars
    frames or a raw conversion error."""
    try:
        out = res
        if out._edges is not None and is_polars_df(out._edges):
            out = out.edges(df_to_engine(out._edges, Engine.CUDF), out._source, out._destination)
        if out._nodes is not None and is_polars_df(out._nodes):
            out = out.nodes(df_to_engine(out._nodes, Engine.CUDF), out._node)
        return out
    except NotImplementedError:
        raise
    except Exception as ex:
        raise NotImplementedError(
            "AUTO cudf->polars-gpu route: result frames could not cross back to cudf "
            f"via Arrow: {type(ex).__name__}: {ex}"
        ) from ex


def _auto_cudf_polars_gpu_route(
    self: Plottable,
    query: GFQLQuery,
    *,
    output: Optional[str],
    where: Optional[Sequence[WhereComparison]],
    language: Optional[Literal["cypher", "gremlin"]],
    params: Optional[CypherParams],
    validate: bool,
    shortest_path_backend: str,
) -> Plottable:
    """The routed attempt for AUTO on a cudf-frame graph, as ONE seam: recurse with the
    engine pinned to the lazy polars engine's GPU target (cudf frames cross to polars via
    Arrow inside chain dispatch's ``_coerce_input_formats``), then cross the result frames
    back to cuDF. Raises ``NotImplementedError`` — from the engine's honest declines, from
    GPU-collect failures (``lazy._gpu_raise`` translates those to NIE), or from the
    cudf-out boundary — and the caller's guard falls back to the legacy CUDF path."""
    routed = gfql(
        self, query, engine=_AUTO_CUDF_ROUTE_ENGINE, output=output, policy=None,
        where=where, language=language, params=params, validate=validate,
        shortest_path_backend=shortest_path_backend,
    )
    return _route_result_frames_to_cudf(routed)


@otel_traced("gfql.run", attrs_fn=_gfql_otel_attrs)
def gfql(self: Plottable,
         query: GFQLQuery,
         engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
         output: Optional[str] = None,
         policy: Optional[Dict[str, PolicyFunction]] = None,
         where: Optional[Sequence[WhereComparison]] = None,
         language: Optional[Literal["cypher", "gremlin"]] = None,
         params: Optional[CypherParams] = None,
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
    if _policied_auto_serves_via_pandas_until_the_polars_route_emits_hooks(engine, policy, self):
        engine = Engine.PANDAS.value

    if (
        (engine == EngineAbstract.AUTO or engine == EngineAbstract.AUTO.value)
        and policy is None
        and is_polars_df(self._edges) and (self._nodes is None or is_polars_df(self._nodes))
    ):
        try:
            return gfql(
                self, query, engine=Engine.POLARS.value, output=output, policy=policy,
                where=where, language=language, params=params, validate=validate,
                shortest_path_backend=shortest_path_backend,
            )
        except NotImplementedError:
            logger.debug('AUTO polars-native attempt declined; serving via pandas')
            from graphistry.compute.ComputeMixin import _coerce_input_formats
            return gfql(
                _coerce_input_formats(self, Engine.PANDAS), query,
                engine=Engine.PANDAS.value, output=output, policy=policy,
                where=where, language=language, params=params, validate=validate,
                shortest_path_backend=shortest_path_backend,
            )

    # engine inference, cuDF arm (owner-directed policy addition, 2026-08-02; supersedes the
    # earlier "AUTO never selects polars-gpu" doctrine for THIS arm only): when every bound
    # frame is cuDF AND the cudf-polars GPU target is GENUINELY usable (probed once per
    # process — polars imports, cudf + cudf_polars installed, and a real GPU collect
    # succeeds; see lazy.polars_gpu_available), prefer the native lazy polars engine on its
    # GPU execution target over the legacy CUDF path. Both serve cudf->cudf: inputs cross
    # cudf -> Arrow -> polars in chain dispatch's _coerce_input_formats, results cross back
    # polars -> Arrow -> cudf in _route_result_frames_to_cudf — lossless nulls/dtypes both
    # ways, no pandas detour. Decline shape mirrors the polars arm above: any
    # NotImplementedError (which is also how GPU-collect failures surface, via
    # lazy._gpu_raise) falls back to the legacy CUDF path with identical values. Explicit
    # engine= always wins (guard requires AUTO), and ``policy is None`` is REQUIRED for the
    # same postload/postchain hook-gap reason documented on the polars arm. Guard ORDER:
    # the all-polars arm above ran first; a graph reaches here only if its frames are not
    # all-polars, and routes only if they are all-cudf.
    if (
        (engine == EngineAbstract.AUTO or engine == EngineAbstract.AUTO.value)
        and policy is None
        and _is_cudf_frame(self._edges) and (self._nodes is None or _is_cudf_frame(self._nodes))
        and _polars_gpu_probe()
    ):
        try:
            return _auto_cudf_polars_gpu_route(
                self, query, output=output, where=where, language=language,
                params=params, validate=validate,
                shortest_path_backend=shortest_path_backend,
            )
        except NotImplementedError:
            logger.debug('AUTO cudf polars-gpu attempt declined; falling back to legacy CUDF path')

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

        # #1786: `shortest_path_backend` is an argument to THIS call, not a property of
        # the caller's graph, so it may not be written onto `self`. Copy only when the
        # value actually differs: the default call then keeps `self`'s identity, and the
        # compiled-query memo cache below is owned by `self` either way.
        dispatch_self = self
        if self._gfql_shortest_path_backend != shortest_path_backend:
            dispatch_self = self.bind()
            dispatch_self._gfql_shortest_path_backend = shortest_path_backend
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
                    # The RESOLVED engine, not the requested one: `auto` resolves per graph,
                    # so keying on the request would let two graphs that resolve differently
                    # share a plan. node_dtypes does NOT stand in for this -- polars and
                    # polars-gpu report identical dtypes and compile differently, which the
                    # old per-Plottable cache hid rather than avoided.
                    # `engine` is EngineAbstract | str here; resolve_engine's parameter is
                    # EngineAbstract | Literal[...], so normalize the str arm first. This
                    # mirrors resolve_engine's own first two lines, so an unknown engine
                    # name still raises ValueError from the same place it always did.
                    engine_key=resolve_engine(
                        EngineAbstract(engine) if isinstance(engine, str) else engine,
                        self).value,
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


def _reject_node_alias_shadowing_id_binding(g: Plottable, chain_obj: Chain) -> None:
    """Typed decline for a node alias named after the node-ID binding column.

    The alias marker is stamped as ``<alias> = True``, so an alias equal to the node-id
    column overwrites the ids themselves: pandas then died with a raw
    ``ValueError: The column label 'id' is not unique`` from the chain's own merge while
    polars answered ``True``. Neither is a usable result; decline the same way on both.
    """
    node_id = getattr(g, "_node", None)
    if not isinstance(node_id, str):
        return
    for op in chain_obj.chain:
        if isinstance(op, ASTNode) and getattr(op, "_name", None) == node_id:
            raise GFQLValidationError(
                ErrorCode.E108,
                "A node alias cannot be named after the node-ID binding column",
                field="chain.name",
                value=node_id,
                suggestion=(
                    f"The alias flag is materialized as a column named '{node_id}', which would "
                    f"overwrite the node-ID binding. Rename the alias."
                ),
            )


def _chain_dispatch(
    g: Plottable,
    chain_obj: Chain,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
    start_nodes: Optional[DataFrameT] = None,
) -> Plottable:
    _reject_node_alias_shadowing_id_binding(g, chain_obj)
    engine_name = engine.value if hasattr(engine, "value") else str(engine)
    if chain_obj.where and engine_name in (Engine.POLARS.value, Engine.POLARS_GPU.value):
        # Cross-entity / same-path WHERE routes through DFSamePathExecutor
        # (df_executor.py, serving pandas AND cuDF), which has no native polars
        # implementation. No silent fallback — raise honestly.
        raise NotImplementedError(
            "polars engine does not yet natively support cross-entity (same-path) "
            "WHERE; use engine='pandas' or engine='cudf' for this query "
            "(no silent fallback; parity-or-error by design)"
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
