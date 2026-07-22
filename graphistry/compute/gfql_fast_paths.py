"""Fast-path specialization helpers for the unified GFQL executor.

Extracted verbatim from gfql_unified.py (#1731) to keep that orchestrator readable: the
connected-join single/two-star fast paths and the grouped-aggregate / two-hop-count fast
paths, plus the seeded typed-hop cypher dispatcher (#1755, _execute_seeded_typed_hop_fast_path).
Pure code moves, no behavior change. gfql_unified imports from here (one direction);
no back-edge into gfql_unified (the .chain import is a leaf-ward edge, cycle-free).
"""
# ruff: noqa: E501

from dataclasses import replace
import pandas as pd
import re
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
from graphistry.Plottable import Plottable

if TYPE_CHECKING:
    import polars as pl
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
    cache_store: Optional[Dict[str, Any]] = None,
) -> DataFrameT:
    cache_key = _connected_join_simple_filter_cache_key(node_match)
    if cache_key is None:
        nodes = df_to_engine(nodes_obj, engine)
        if engine in POLARS_ENGINES:
            from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars
            return cast(DataFrameT, filter_by_dict_polars(nodes, node_match))
        return filter_by_dict(nodes, node_match, engine=EngineAbstract(engine.value))

    cache_attr = "_gfql_connected_join_node_filter_cache"
    # Per-execution cache only (threaded via cache_store); NEVER setattr onto the caller's
    # Plottable -- that leaked results across gfql() calls keyed by id(), returning stale
    # answers after an in-place edge/node mutation (BLOCKER 1). None => no caching.
    cache = cache_store.setdefault(cache_attr, {}) if cache_store is not None else None
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


def _connected_join_cached_node_ids(  # pragma: no cover - polars-only, covered by polars lane
    base_graph: Plottable,
    nodes_obj: DataFrameT,
    node_matches: Sequence[Optional[dict]],
    *,
    node_col: str,
    engine: Engine,
    cache_store: Optional[Dict[str, Any]] = None,
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
    # Per-execution cache only (threaded via cache_store); NEVER setattr onto the caller's
    # Plottable -- that leaked results across gfql() calls keyed by id(), returning stale
    # answers after an in-place edge/node mutation (BLOCKER 1). None => no caching.
    cache = cache_store.setdefault(cache_attr, {}) if cache_store is not None else None
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
    cache_store: Optional[Dict[str, Any]] = None,
) -> DataFrameT:
    cache_key = _connected_join_simple_filter_cache_key(edge_match)
    if cache_key is None:
        edges = df_to_engine(edges_obj, engine)
        if engine in POLARS_ENGINES:
            from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars
            return cast(DataFrameT, filter_by_dict_polars(edges, edge_match))
        return filter_by_dict(edges, edge_match, engine=EngineAbstract(engine.value))

    cache_attr = "_gfql_connected_join_edge_filter_cache"
    # Per-execution cache only (threaded via cache_store); NEVER setattr onto the caller's
    # Plottable -- that leaked results across gfql() calls keyed by id(), returning stale
    # answers after an in-place edge/node mutation (BLOCKER 1). None => no caching.
    cache = cache_store.setdefault(cache_attr, {}) if cache_store is not None else None
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


def _connected_join_cached_singleton_dst_source_counts(  # pragma: no cover
    base_graph: Plottable,
    edges_obj: DataFrameT,
    edge_domain: DataFrameT,
    edge_match: Optional[dict],
    singleton_dst: Any,
    *,
    src_col: str,
    dst_col: str,
    engine: Engine,
    cache_store: Optional[Dict[str, Any]] = None,
) -> Optional[DataFrameT]:
    edge_key = _connected_join_simple_filter_cache_key(edge_match)
    dst_key = _connected_join_filter_value_cache_key(singleton_dst)
    if edge_key is None or dst_key is None:
        return None

    cache_attr = "_gfql_connected_join_singleton_dst_source_counts_cache"
    # Per-execution cache only (threaded via cache_store); NEVER setattr onto the caller's
    # Plottable -- that leaked results across gfql() calls keyed by id(), returning stale
    # answers after an in-place edge/node mutation (BLOCKER 1). None => no caching.
    cache = cache_store.setdefault(cache_attr, {}) if cache_store is not None else None
    full_key = (id(edges_obj), engine.value, src_col, dst_col, edge_key, dst_key)
    if cache is not None and full_key in cache:
        return cast(DataFrameT, cache[full_key])

    if engine in POLARS_ENGINES:  # pragma: no cover - polars-only, covered by polars lane
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


def _connected_join_cached_first_arm_shared_counts(  # pragma: no cover - polars-only, covered by polars lane
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
    cache_store: Optional[Dict[str, Any]] = None,
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
    # Per-execution cache only (threaded via cache_store); NEVER setattr onto the caller's
    # Plottable -- that leaked results across gfql() calls keyed by id(), returning stale
    # answers after an in-place edge/node mutation (BLOCKER 1). None => no caching.
    cache = cache_store.setdefault(cache_attr, {}) if cache_store is not None else None
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


def _connected_join_cached_second_arm_group_rows(  # pragma: no cover - polars-only, covered by polars lane
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
    cache_store: Optional[Dict[str, Any]] = None,
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
    # Per-execution cache only (threaded via cache_store); NEVER setattr onto the caller's
    # Plottable -- that leaked results across gfql() calls keyed by id(), returning stale
    # answers after an in-place edge/node mutation (BLOCKER 1). None => no caching.
    cache = cache_store.setdefault(cache_attr, {}) if cache_store is not None else None
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
        # Dedup by node identity (matches pandas drop_duplicates(subset=[node_col])); a
        # duplicate node row would otherwise multiply the join and over-count.
        second_lookup = second_leaf_nodes.select(lookup_exprs).unique(subset=[lookup_key])
        right_base = right_base.join(second_lookup, left_on=dst_col, right_on=lookup_key, how="inner")
    right_rows = right_base.select([pl.col(src_col)] + [pl.col(key) for key in output_group_keys])
    if cache is not None:
        cache[full_key] = right_rows
    return cast(DataFrameT, right_rows)


def _two_hop_cached_equal_domain_degree_counts(
    base_graph: Plottable,
    nodes_obj: DataFrameT,
    edges_obj: DataFrameT,
    domain_nodes: DataFrameT,
    edge_domain: DataFrameT,
    *,
    node_match: Optional[dict],
    edge_match: Optional[dict],
    node_col: str,
    src_col: str,
    dst_col: str,
    engine: Engine,
) -> Optional[Tuple[DataFrameT, DataFrameT]]:
    node_key = _connected_join_simple_filter_cache_key(node_match)
    edge_key = _connected_join_simple_filter_cache_key(edge_match)
    if node_key is None or edge_key is None:
        return None

    cache_attr = "_gfql_two_hop_equal_domain_degree_counts_cache"
    cache = getattr(base_graph, cache_attr, None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(base_graph, cache_attr, cache)
        except Exception:
            cache = None
    full_key = (id(nodes_obj), id(edges_obj), engine.value, node_col, src_col, dst_col, node_key, edge_key)
    if cache is not None and full_key in cache:
        return cast(Tuple[DataFrameT, DataFrameT], cache[full_key])

    if engine in POLARS_ENGINES:
        domain_ids = domain_nodes.select(node_col).unique()
        filtered_edges = (
            edge_domain
            .join(domain_ids, left_on=src_col, right_on=node_col, how="semi")
            .join(domain_ids, left_on=dst_col, right_on=node_col, how="semi")
        )
        counts = (
            cast(DataFrameT, filtered_edges.group_by(dst_col).len("__in_count__")),
            cast(DataFrameT, filtered_edges.group_by(src_col).len("__out_count__")),
        )
    else:
        domain_ids = domain_nodes[node_col].drop_duplicates()
        filtered_edges = edge_domain[edge_domain[src_col].isin(domain_ids) & edge_domain[dst_col].isin(domain_ids)]
        counts = (
            cast(DataFrameT, filtered_edges.groupby(dst_col, sort=False).size().reset_index(name="__in_count__")),
            cast(DataFrameT, filtered_edges.groupby(src_col, sort=False).size().reset_index(name="__out_count__")),
        )

    if cache is not None:
        cache[full_key] = counts
    return counts


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


# The simple residual shapes the connected-join lowering emits for scalar predicates it
# cannot push into filter_dict (see #1729): case-insensitive equality and scalar
# equality/range on a single aliased column. Anything else falls back to the where_rows
# chain evaluator. Literals: single-quoted strings (no embedded quotes) or numbers.
_RESIDUAL_TOLOWER_EQ = re.compile(
    r"^\(tolower\((?P<alias>\w+)\.(?P<col>\w+)\) = tolower\('(?P<lit>[^']*)'\)\)$"
)
_RESIDUAL_SCALAR_CMP = re.compile(
    r"^\((?P<alias>\w+)\.(?P<col>\w+) (?P<op>=|>=|<=|>|<) "
    r"(?:'(?P<slit>[^']*)'|(?P<nlit>-?\d+(?:\.\d+)?))\)$"
)


def _residual_polars_expr(
    expr: str, alias: str, schema: Mapping[str, Any]
) -> Optional['pl.Expr']:
    """Translate a simple residual to a native polars expression, or None to fall back.

    ``expr`` is a *string* by contract: the #1729 connected-join lowering serializes
    residual predicates into ASTCall params as canonical predicate strings (e.g.
    ``(tolower(a.col) = tolower('lit'))``), not typed AST terms — so string parsing here
    is the honest interface; a typed term would require a lowering-level refactor.

    Covered (exactly the #1729 scalar-residual shapes): ``(tolower(a.col) = tolower('lit'))``
    and ``(a.col <op> literal)`` for ``= >= <= > <``. Semantics match the where_rows
    evaluator on these shapes: string compares are null-safe (null -> filtered out, since
    polars comparisons on null yield null which ``filter`` drops, same as the evaluator's
    null-propagating comparisons); toLower equality lowercases the column via polars
    ``str.to_lowercase()`` and the literal via Python ``str.lower()`` (empirically equal
    on the ASCII/latin shapes the lowering emits; a divergence would need a Rust-vs-Python
    Unicode table drift). Float NaN ranking differs between polars and the evaluator, but
    gfql ingest normalizes NaN->null (``_pl_nan_to_null``) so NaN never reaches this
    filter through ``gfql()``. Declines (returns None, caller uses the chain fallback) on:
    any other shape, non-matching alias, a column absent from the schema, an ESCAPED
    string literal (``\\`` — the renderer escapes ``' \\ \\n`` etc. to ``\\uXXXX`` which the
    evaluator unescapes; raw comparison would silently mismatch), and dtype-incompatible
    column/literal pairs (string predicate on non-string column and vice versa — the
    lowering deliberately keeps those residual so the evaluator can raise its designed
    parity-or-error NotImplementedError rather than a raw polars ComputeError).
    """
    import polars as pl

    def _is_string_dtype(dtype: Any) -> bool:
        return dtype == pl.Utf8 or dtype == pl.String

    def _is_numeric_dtype(dtype: Any) -> bool:
        return dtype.is_numeric() if hasattr(dtype, "is_numeric") else False

    m = _RESIDUAL_TOLOWER_EQ.match(expr)
    if m is not None:
        col_name = m.group("col")
        tolower_lit = m.group("lit")
        if m.group("alias") != alias or col_name not in schema:
            return None
        if "\\" in tolower_lit:
            return None  # escaped literal: let the evaluator unescape it
        if not _is_string_dtype(schema[col_name]):
            return None  # tolower on non-string column: evaluator raises designed NIE
        return pl.col(col_name).str.to_lowercase() == tolower_lit.lower()
    m = _RESIDUAL_SCALAR_CMP.match(expr)
    if m is not None:
        col_name = m.group("col")
        if m.group("alias") != alias or col_name not in schema:
            return None
        lit: Any
        if m.group("slit") is not None:
            lit = m.group("slit")
            if "\\" in lit:
                return None  # escaped literal: let the evaluator unescape it
            if not _is_string_dtype(schema[col_name]):
                return None  # string literal vs non-string column: designed NIE path
        else:
            raw = m.group("nlit")
            lit = float(raw) if "." in raw else int(raw)
            if not _is_numeric_dtype(schema[col_name]):
                return None  # numeric literal vs non-numeric column: designed NIE path
        col = pl.col(col_name)
        op = m.group("op")
        if op == "=":
            return col == lit
        if op == ">=":
            return col >= lit
        if op == "<=":
            return col <= lit
        if op == ">":
            return col > lit
        return col < lit
    return None


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

    Fast lane (polars): the simple scalar shapes the #1729 lowering emits
    (``tolower(a.col) = tolower('lit')``, ``a.col <op> literal``) translate directly to
    native polars filters — no chain dispatch (the where_rows chain costs ~1.7ms/alias,
    the dominant cost of the residual OLAP fast path). Any expression outside those
    shapes falls back to the chain evaluator below, so semantics never diverge.

    Fallback: reuses the row pipeline's ``where_rows`` evaluator (identical semantics to
    the slow path, so toLower/etc. behave exactly as they would post-join) by aliasing
    the node columns to ``alias.col`` and dispatching a where_rows chain, then renaming
    back. ``validate_schema`` is disabled because the residual references flat
    ``alias.col`` columns rather than a bound graph element.
    """
    is_polars = "polars" in type(node_frame).__module__
    if is_polars:
        translated = [_residual_polars_expr(e, alias, dict(node_frame.schema)) for e in exprs]
        if all(t is not None for t in translated):
            out = node_frame
            for t in translated:
                out = out.filter(t)
            return cast(DataFrameT, out)
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



def _connected_join_two_star_fused_polars(
    nodes: DataFrameT,
    edges: DataFrameT,
    *,
    node_col: str,
    src_col: str,
    dst_col: str,
    residual_map: Dict[str, List[Any]],
    shared_alias: str,
    first_end_alias: str,
    second_end_alias: str,
    first_start_fd: Optional[dict],
    second_start_fd: Optional[dict],
    first_end_fd: Optional[dict],
    second_end_fd: Optional[dict],
    first_edge_match: Optional[dict],
    second_edge_match: Optional[dict],
    group_prop_refs: List[Tuple[str, str]],
    output_group_keys: List[str],
    agg_alias: str,
    order_keys: List[Tuple[str, bool]],
    limit_value: Optional[int],
    select_items: Optional[List[Tuple[str, str]]],
) -> Optional[DataFrameT]:
    """FUSED lazy lane (#1755 lane-1): the whole two-star grouped-count as ONE lazy
    plan, collected once at the join (the eager path pays a fixed collect cost per
    op; ~27 collects/exec dominated q5-q7 profiles). Value-identical to the eager
    lane -- same filters/semi-joins/aggregation, and the empty-match boundary
    reproduces the eager all-left-counts==1 shortcut's single n=0 row (openCypher
    count over no rows). Returns None to decline (untranslatable residual, missing
    group property) so the caller falls through to the eager path. Both frames must
    already be engine-converted polars frames.
    """
    import polars as pl
    from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_expr_by_dict_polars

    if isinstance(nodes, pl.LazyFrame) or isinstance(edges, pl.LazyFrame):
        return None  # eager lane owns LazyFrame inputs (schema probes on a LazyFrame warn/cost)
    frame_aliases = {shared_alias, first_end_alias, second_end_alias}
    node_schema = dict(nodes.schema)
    residual_exprs: Dict[str, List["pl.Expr"]] = {}
    for r_alias, r_exprs in residual_map.items():
        if r_alias not in frame_aliases:
            continue  # mirror the eager path: residuals for unbound aliases are not applied here
        r_translated = [_residual_polars_expr(e, r_alias, node_schema) for e in r_exprs]
        if any(t is None for t in r_translated):
            return None
        residual_exprs[r_alias] = [t for t in r_translated if t is not None]
    for _, prop in group_prop_refs:
        if prop not in nodes.columns:
            return None
    lf_nodes = nodes.lazy()
    lf_edges = edges.lazy()

    def _alias_nodes_lf(fds: List[Optional[dict]], r_alias: str) -> "pl.LazyFrame":
        lf = lf_nodes
        for fd in fds:
            fe = filter_expr_by_dict_polars(nodes, fd)
            if fe is not None:
                lf = lf.filter(fe)
        for rexpr in residual_exprs.get(r_alias, []):
            lf = lf.filter(rexpr)
        return lf

    shared_lf = _alias_nodes_lf([first_start_fd, second_start_fd], shared_alias)
    second_leaf_lf = _alias_nodes_lf([second_end_fd], second_end_alias)
    shared_ids_lf = shared_lf.select(node_col).unique()
    first_leaf_ids_lf = _alias_nodes_lf([first_end_fd], first_end_alias).select(node_col).unique()
    second_leaf_ids_lf = second_leaf_lf.select(node_col).unique()
    fe1 = filter_expr_by_dict_polars(edges, first_edge_match)
    fe2 = filter_expr_by_dict_polars(edges, second_edge_match)
    first_edges_lf = lf_edges.filter(fe1) if fe1 is not None else lf_edges
    second_edges_lf = lf_edges.filter(fe2) if fe2 is not None else lf_edges
    left_counts_lf = (
        first_edges_lf
        .join(shared_ids_lf, left_on=src_col, right_on=node_col, how="semi")
        .join(first_leaf_ids_lf, left_on=dst_col, right_on=node_col, how="semi")
        .group_by(src_col)
        .len("__left_count__")
        .rename({src_col: shared_alias})
    )
    right_base_lf = (
        second_edges_lf
        .join(shared_ids_lf, left_on=src_col, right_on=node_col, how="semi")
        .join(second_leaf_ids_lf, left_on=dst_col, right_on=node_col, how="semi")
    )
    if group_prop_refs:
        fused_lookup_key = "__gfql_fast_second_leaf_id__"
        lookup_lf = second_leaf_lf.select(
            [pl.col(node_col).alias(fused_lookup_key)]
            + [pl.col(prop).alias(out_col) for out_col, prop in group_prop_refs]
        ).unique(subset=[fused_lookup_key])
        right_base_lf = right_base_lf.join(lookup_lf, left_on=dst_col, right_on=fused_lookup_key, how="inner")
    right_rows_lf = right_base_lf.select(
        [pl.col(src_col).alias(shared_alias)] + [pl.col(key) for key in output_group_keys]
    )
    joined_lf = right_rows_lf.join(left_counts_lf, on=shared_alias, how="inner")
    # HOT PATH: one collect. left_counts is collected ONLY on the empty-match
    # boundary below (collect_all of both plans measured +2.5ms/query on the
    # 20k graphbench q5-q7 -- CSE does not absorb the left-arm recompute).
    joined = joined_lf.collect()
    if len(joined) == 0:
        # Eager-lane parity on the empty match: the eager all-left-counts==1
        # shortcut counts matched rows with pl.len(), emitting a single n=0 row
        # when the first arm is live but nothing joins (the openCypher-correct
        # count over zero rows). Every other empty shape returns the 0x0 frame,
        # exactly like the eager generic branch.
        left_counts_df = left_counts_lf.collect()
        if (
            not output_group_keys
            and len(left_counts_df) > 0
            and bool(left_counts_df.select((pl.col("__left_count__") == 1).all()).item())
        ):
            out_df = pl.DataFrame({agg_alias: [0]}).with_columns(pl.col(agg_alias).cast(pl.Int64))
        else:
            return cast(DataFrameT, joined.select([]))
    elif output_group_keys:
        out_df = joined.group_by(output_group_keys, maintain_order=True).agg(
            pl.col("__left_count__").sum().cast(pl.Int64).alias(agg_alias))
    else:
        out_df = joined.select(pl.col("__left_count__").sum().cast(pl.Int64).alias(agg_alias))
    if order_keys:
        out_df = out_df.sort(
            [key for key, _ in order_keys],
            descending=[desc for _, desc in order_keys],
            nulls_last=[not desc for _, desc in order_keys],
        )
    if limit_value is not None:
        out_df = out_df.head(limit_value)
    if select_items is not None:
        out_df = out_df.select([pl.col(s_col).alias(d_col) for s_col, d_col in select_items])
    return cast(DataFrameT, out_df)


def _connected_join_two_star_fast_grouped_count(
    base_graph: Plottable,
    plan: ConnectedMatchJoinPlan,
    *,
    engine: Engine,
    cache_store: Optional[Dict[str, Any]] = None,
) -> Optional[DataFrameT]:
    # Per-execution cache scope: a fresh store when none is threaded in, so intra-query reuse
    # is preserved without ever persisting caches on the caller's Plottable (BLOCKER 1).
    if cache_store is None:
        cache_store = {}
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

    if engine in POLARS_ENGINES:  # pragma: no cover - polars-only, covered by polars lane
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
                cache_store=cache_store,
            )
            first_leaf_ids = _connected_join_cached_node_ids(
                base_graph,
                nodes_source,
                [cast(Optional[dict], first_end.filter_dict)],
                node_col=node_col,
                engine=engine,
                cache_store=cache_store,
            )
            second_leaf_ids = _connected_join_cached_node_ids(
                base_graph,
                nodes_source,
                [cast(Optional[dict], second_end.filter_dict)],
                node_col=node_col,
                engine=engine,
                cache_store=cache_store,
            )
            first_leaf_nodes = _connected_join_cached_node_filter(base_graph, nodes_source, cast(Optional[dict], first_end.filter_dict), engine=engine, cache_store=cache_store)
            second_leaf_nodes = _connected_join_cached_node_filter(base_graph, nodes_source, cast(Optional[dict], second_end.filter_dict), engine=engine, cache_store=cache_store)
            if shared_ids is None or first_leaf_ids is None or second_leaf_ids is None:
                return None
        else:
            # FUSED lazy lane (#1755 lane-1): one lazy plan, one collect at the join.
            # Returns None to fall through to the eager path (untranslatable residual,
            # missing group property). Edges are engine-converted HERE because the
            # fused lane runs native polars ops directly on them (the eager path's
            # cached edge filter does its own conversion) -- pandas edges with
            # engine='polars' (incl. WITH..MATCH reentry frames) crash otherwise.
            fused_out = _connected_join_two_star_fused_polars(
                nodes,
                df_to_engine(edges, engine),
                node_col=node_col,
                src_col=src_col,
                dst_col=dst_col,
                residual_map=residual_map,
                shared_alias=shared_alias,
                first_end_alias=first_end_alias,
                second_end_alias=second_end_alias,
                first_start_fd=cast(Optional[dict], first_start.filter_dict),
                second_start_fd=cast(Optional[dict], second_start.filter_dict),
                first_end_fd=cast(Optional[dict], first_end.filter_dict),
                second_end_fd=cast(Optional[dict], second_end.filter_dict),
                first_edge_match=cast(Optional[dict], first_edge.edge_match),
                second_edge_match=cast(Optional[dict], second_edge.edge_match),
                group_prop_refs=group_prop_refs,
                output_group_keys=output_group_keys,
                agg_alias=agg_alias,
                order_keys=order_keys,
                limit_value=limit_value,
                select_items=select_items,
            )
            if fused_out is not None:
                return fused_out
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
        first_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], first_edge.edge_match), engine=engine, cache_store=cache_store)
        second_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], second_edge.edge_match), engine=engine, cache_store=cache_store)

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
                cache_store=cache_store,
            )
            # Unreachable under the current lowering: _connected_join_cached_first_arm_shared_counts
            # only returns None when a filter/edge/dst cache key is uncacheable, but every value
            # that reaches this cached polars path is cacheable (scalar equality pushes as scalars;
            # comparisons/toLower/ranges lower to residuals routed off the cached path). Kept as a
            # defensive fallback; excluded from coverage since no legitimate query reaches it.
            if cached_left_counts is None:  # pragma: no cover
                cached_left_counts = _connected_join_cached_singleton_dst_source_counts(
                    base_graph,
                    edges,
                    first_edges,
                    cast(Optional[dict], first_edge.edge_match),
                    first_leaf_singleton,
                    src_col=src_col,
                    dst_col=dst_col,
                    engine=engine,
                    cache_store=cache_store,
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
            cache_store=cache_store,
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
                # Dedup by node identity to match the pandas drop_duplicates(subset=[node_col]);
                # a duplicate node row would otherwise multiply the join and over-count.
                second_lookup = second_leaf_nodes.select(lookup_exprs).unique(subset=[lookup_key])
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
            # openCypher orders NULL as the largest value: ASC -> nulls last, DESC -> nulls
            # first. Polars defaults to nulls-first, which flips WHICH ROW an ORDER BY ... LIMIT
            # returns, so pin nulls_last per key.
            out_df = out_df.sort(
                [key for key, _ in order_keys],
                descending=[desc for _, desc in order_keys],
                nulls_last=[not desc for _, desc in order_keys],
            )
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
    first_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], first_edge.edge_match), engine=engine, cache_store=cache_store)
    second_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], second_edge.edge_match), engine=engine, cache_store=cache_store)

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
    cache_store: Optional[Dict[str, Any]] = None,
) -> Optional[DataFrameT]:
    """Fast rows for T1 two-star connected joins.

    Supported shape: ``(p)-[r1]->(i), (p)-[r2]->(c)`` where both arms are fixed
    forward one-hop patterns, share the same start node alias, have no residual
    per-arm row filters, and downstream projections need at most properties from
    the second leaf alias. The returned frame preserves row multiplicity, then
    the normal row pipeline handles RETURN/GROUP/ORDER/LIMIT.
    """
    # Per-execution cache scope (see _connected_join_two_star_fast_grouped_count); never
    # persisted on the caller's Plottable.
    if cache_store is None:
        cache_store = {}
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
    # A bare aggregate over the second leaf (count(c) / count(DISTINCT c)) lowers `c` to its
    # identity column c.__gfql_node_id__, which is NOT a materializable node property here --
    # fast_rows would drop it and post_join's count(c.__gfql_node_id__) would dereference a
    # missing column. Decline to the (correct) slow path in that case.
    if NODE_IDENTITY_COLUMN in second_props_needed:
        return None
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
        first_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], first_edge.edge_match), engine=engine, cache_store=cache_store)
        second_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], second_edge.edge_match), engine=engine, cache_store=cache_store)

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
            # Dedup by node identity (matches pandas drop_duplicates(subset=[node_col])); a
            # duplicate node row would otherwise multiply the join and inflate row multiplicity.
            second_lookup = second_leaf_nodes.select([node_col] + second_props).unique(subset=[node_col]).rename({node_col: second_lookup_key})
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
    first_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], first_edge.edge_match), engine=engine, cache_store=cache_store)
    second_edges = _connected_join_cached_edge_filter(base_graph, edges, cast(Optional[dict], second_edge.edge_match), engine=engine, cache_store=cache_store)

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


def _filter_nodes_for_fast_count(nodes: DataFrameT, filter_dict: Optional[dict], *, engine: Engine) -> DataFrameT:
    if engine in POLARS_ENGINES:
        from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars
        return cast(DataFrameT, filter_by_dict_polars(nodes, filter_dict))
    return filter_by_dict(nodes, filter_dict, engine=EngineAbstract(engine.value))


def _two_hop_count_alias(chain: Chain) -> Optional[str]:
    ops = list(chain.chain)
    if len(ops) != 4 or not all(isinstance(op, ASTCall) for op in ops):
        return None
    rows_call, with_call, group_call, select_call = cast(Tuple[ASTCall, ASTCall, ASTCall, ASTCall], tuple(ops))
    if rows_call.function != "rows" or with_call.function != "with_" or group_call.function != "group_by" or select_call.function != "select":
        return None
    if rows_call.params.get("table") != "nodes":
        return None
    if with_call.params.get("items") != [("__cypher_group__", 1)]:
        return None
    aggs = group_call.params.get("aggregations")
    if group_call.params.get("keys") != ["__cypher_group__"] or not isinstance(aggs, list) or len(aggs) != 1:
        return None
    agg = aggs[0]
    if not isinstance(agg, (tuple, list)) or len(agg) != 2 or str(agg[1]).lower() != "count":
        return None
    alias = str(agg[0])
    if select_call.params.get("items") != [(alias, alias)]:
        return None
    return alias


def _two_hop_count_binding_ops(chain: Chain) -> Optional[Tuple[ASTNode, ASTEdge, ASTNode, ASTEdge, ASTNode]]:
    if not chain.chain or not isinstance(chain.chain[0], ASTCall):
        return None
    rows_call = cast(ASTCall, chain.chain[0])
    binding_ops = rows_call.params.get("binding_ops")
    if not isinstance(binding_ops, list) or len(binding_ops) != 5:
        return None
    from graphistry.compute.ast import from_json as ast_from_json
    ops = [ast_from_json(op_json, validate=False) for op_json in binding_ops]
    n0, e0, n1, e1, n2 = ops
    if not isinstance(n0, ASTNode) or not isinstance(e0, ASTEdge) or not isinstance(n1, ASTNode) or not isinstance(e1, ASTEdge) or not isinstance(n2, ASTNode):
        return None
    if not _is_connected_fast_single_hop(e0) or not _is_connected_fast_single_hop(e1):
        return None
    return n0, e0, n1, e1, n2


def _property_ref(expr: Any, valid_aliases: Sequence[str]) -> Optional[Tuple[str, str]]:
    if not isinstance(expr, str) or "." not in expr:
        return None
    alias, prop = expr.split(".", 1)
    if alias not in valid_aliases or not prop:
        return None
    return alias, prop


def _execute_single_hop_grouped_aggregate_fast_path(
    base_graph: Plottable,
    chain: Chain,
    *,
    engine: Union[EngineAbstract, str],
) -> Optional[Plottable]:
    ops = list(chain.chain)
    if len(ops) not in (3, 4, 5) or not all(isinstance(op, ASTCall) for op in ops):
        return None
    rows_call = cast(ASTCall, ops[0])
    with_call = cast(ASTCall, ops[1])
    group_call = cast(ASTCall, ops[2])
    suffix = [cast(ASTCall, op) for op in ops[3:]]
    if rows_call.function != "rows" or with_call.function != "with_" or group_call.function != "group_by":
        return None
    if rows_call.params.get("table") != "nodes" or with_call.params.get("extend") is not True:
        return None

    order_call: Optional[ASTCall] = None
    limit_call: Optional[ASTCall] = None
    for op in suffix:
        if op.function == "order_by" and order_call is None and limit_call is None:
            order_call = op
        elif op.function == "limit" and limit_call is None:
            limit_call = op
        else:
            return None

    binding_ops = rows_call.params.get("binding_ops")
    if not isinstance(binding_ops, list) or len(binding_ops) != 3:
        return None
    from graphistry.compute.ast import from_json as ast_from_json
    parsed_ops = [ast_from_json(op_json, validate=False) for op_json in binding_ops]
    start_op, edge_op, end_op = parsed_ops
    if not isinstance(start_op, ASTNode) or not isinstance(edge_op, ASTEdge) or not isinstance(end_op, ASTNode):
        return None
    if not _is_connected_fast_single_hop(edge_op):
        return None
    start_alias = getattr(start_op, "_name", None)
    end_alias = getattr(end_op, "_name", None)
    if not isinstance(start_alias, str) or not isinstance(end_alias, str) or start_alias == end_alias:
        return None
    aliases = (start_alias, end_alias)

    with_items_raw = with_call.params.get("items")
    if not isinstance(with_items_raw, list):
        return None
    with_items: Dict[str, Tuple[str, Optional[str]]] = {}
    for item in with_items_raw:
        if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[0], str):
            return None
        prop_ref = _property_ref(item[1], aliases)
        ref: Tuple[str, Optional[str]]
        if prop_ref is None:
            if item[1] not in aliases:
                return None
            ref = (cast(str, item[1]), None)
        else:
            ref = prop_ref
        with_items[item[0]] = ref

    group_keys_raw = group_call.params.get("keys")
    if not isinstance(group_keys_raw, list) or not group_keys_raw or not all(isinstance(key, str) for key in group_keys_raw):
        return None
    group_keys = cast(List[str], group_keys_raw)
    if not all(key in with_items and with_items[key][1] is not None for key in group_keys):
        return None

    aggs_raw = group_call.params.get("aggregations")
    if not isinstance(aggs_raw, list) or not aggs_raw:
        return None
    aggregations: List[Tuple[str, str, Optional[str]]] = []
    for agg in aggs_raw:
        if not isinstance(agg, (tuple, list)) or len(agg) not in (2, 3):
            return None
        alias = str(agg[0])
        func = str(agg[1]).lower()
        expr_alias = agg[2] if len(agg) == 3 else None
        if func not in {"count", "avg", "sum", "min", "max"}:
            return None
        if expr_alias is not None and not isinstance(expr_alias, str):
            return None
        if expr_alias is not None and expr_alias not in with_items:
            return None
        if func != "count" and expr_alias is None:
            return None
        if expr_alias is not None and with_items[expr_alias][1] is None and func != "count":
            return None
        aggregations.append((alias, func, cast(Optional[str], expr_alias)))

    order_keys: List[Tuple[str, bool]] = []
    if order_call is not None:
        raw_order = order_call.params.get("keys")
        if not isinstance(raw_order, list):
            return None
        available = set(group_keys) | {alias for alias, _, _ in aggregations}
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

    requested_engine = resolve_engine(cast(Any, engine), base_graph)
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

    nodes = cast(DataFrameT, nodes_obj)
    start_nodes = _connected_join_cached_node_filter(base_graph, nodes, cast(Optional[dict], start_op.filter_dict), engine=requested_engine)
    end_nodes = _connected_join_cached_node_filter(base_graph, nodes, cast(Optional[dict], end_op.filter_dict), engine=requested_engine)
    edges = _connected_join_cached_edge_filter(base_graph, cast(DataFrameT, edges_obj), cast(Optional[dict], edge_op.edge_match), engine=requested_engine)

    needed_by_alias: Dict[str, List[Tuple[str, str]]] = {start_alias: [], end_alias: []}
    for out_col, ref in with_items.items():
        alias, prop = ref
        if prop is not None:
            needed_by_alias[alias].append((out_col, prop))

    if requested_engine in POLARS_ENGINES:  # pragma: no cover - polars-only, covered by polars lane
        import polars as pl
        start_ids = start_nodes.select(node_col).unique()
        end_ids = end_nodes.select(node_col).unique()
        work = (
            edges
            .join(start_ids, left_on=src_col, right_on=node_col, how="semi")
            .join(end_ids, left_on=dst_col, right_on=node_col, how="semi")
            .select([src_col, dst_col])
        )

        def join_props_polars(work_df: Any, alias: str, node_df: Any, edge_col: str) -> Any:
            props = needed_by_alias.get(alias, [])
            if not props:
                return work_df
            lookup_key = f"__gfql_t3_{alias}_id__"
            exprs = [pl.col(node_col).alias(lookup_key)]
            for out_col, prop in props:
                if prop not in node_df.columns:
                    return None
                exprs.append(pl.col(prop).alias(out_col))
            lookup = node_df.select(exprs)
            return work_df.join(lookup, left_on=edge_col, right_on=lookup_key, how="inner")

        work = join_props_polars(work, start_alias, start_nodes, src_col)
        if work is None:
            return None
        work = join_props_polars(work, end_alias, end_nodes, dst_col)
        if work is None:
            return None
        agg_exprs = []
        for alias, func, expr_alias in aggregations:
            if func == "count" and (expr_alias is None or with_items[expr_alias][1] is None):
                agg_exprs.append(pl.len().alias(alias))
            elif func == "count" and expr_alias is not None:
                agg_exprs.append(pl.col(expr_alias).count().alias(alias))
            elif func == "avg" and expr_alias is not None:
                agg_exprs.append(pl.col(expr_alias).mean().alias(alias))
            elif func == "sum" and expr_alias is not None:
                agg_exprs.append(pl.col(expr_alias).sum().alias(alias))
            elif func == "min" and expr_alias is not None:
                agg_exprs.append(pl.col(expr_alias).min().alias(alias))
            elif func == "max" and expr_alias is not None:
                agg_exprs.append(pl.col(expr_alias).max().alias(alias))
            else:
                return None
        out_nodes = work.group_by(group_keys, maintain_order=True).agg(agg_exprs)
        if order_keys:
            # openCypher orders NULL as the largest value (ASC -> nulls last, DESC -> nulls
            # first). Polars defaults nulls-first, which flips WHICH ROW ORDER BY ... LIMIT
            # returns, so pin nulls_last per key.
            out_nodes = out_nodes.sort(
                [key for key, _ in order_keys],
                descending=[desc for _, desc in order_keys],
                nulls_last=[not desc for _, desc in order_keys],
            )
        if limit_value is not None:
            out_nodes = out_nodes.head(limit_value)
        out_df = cast(DataFrameT, out_nodes)
    else:
        start_ids = start_nodes[node_col].drop_duplicates()
        end_ids = end_nodes[node_col].drop_duplicates()
        work = edges[edges[src_col].isin(start_ids) & edges[dst_col].isin(end_ids)][[src_col, dst_col]]

        def join_props_df(work_df: DataFrameT, alias: str, node_df: DataFrameT, edge_col: str) -> Optional[DataFrameT]:
            props = needed_by_alias.get(alias, [])
            if not props:
                return work_df
            prop_cols = []
            for _, prop in props:
                if prop not in node_df.columns:
                    return None
                if prop != node_col and prop not in prop_cols:
                    prop_cols.append(prop)
            lookup_key = f"__gfql_t3_{alias}_id__"
            lookup = node_df[[node_col] + prop_cols].drop_duplicates(subset=[node_col]).copy()
            for out_col, prop in props:
                lookup[out_col] = lookup[prop]
            lookup = lookup.rename(columns={node_col: lookup_key})
            return cast(DataFrameT, work_df.merge(lookup, left_on=edge_col, right_on=lookup_key, how="inner"))

        work = cast(DataFrameT, work)
        joined_start = join_props_df(work, start_alias, start_nodes, src_col)
        if joined_start is None:
            return None
        joined_end = join_props_df(joined_start, end_alias, end_nodes, dst_col)
        if joined_end is None:
            return None
        work = joined_end
        try:
            grouped = work.groupby(group_keys, sort=False, dropna=False)
        except TypeError:
            grouped = work.groupby(group_keys, sort=False)
        out_df = grouped.size().reset_index(name="__gfql_group_size__")[group_keys]
        for alias, func, expr_alias in aggregations:
            if func == "count" and (expr_alias is None or with_items[expr_alias][1] is None):
                agg_df = grouped.size().reset_index(name=alias)
            elif func == "count" and expr_alias is not None:
                agg_df = grouped[expr_alias].count().reset_index(name=alias)
            elif func == "avg" and expr_alias is not None:
                agg_df = grouped[expr_alias].mean().reset_index(name=alias)
            elif func == "sum" and expr_alias is not None:
                agg_df = grouped[expr_alias].sum().reset_index(name=alias)
            elif func == "min" and expr_alias is not None:
                agg_df = grouped[expr_alias].min().reset_index(name=alias)
            elif func == "max" and expr_alias is not None:
                agg_df = grouped[expr_alias].max().reset_index(name=alias)
            else:
                return None
            out_df = cast(DataFrameT, out_df.merge(agg_df, on=group_keys, how="left", sort=False))
        if order_keys:
            # openCypher orders NULL as the largest value (ASC -> nulls last, DESC -> nulls
            # first), matching the polars branch's per-key nulls_last above. pandas/cuDF
            # sort_values take a SCALAR na_position (can't be per-key), and cuDF has no stable
            # sort (kind='stable' silently falls back to quicksort), so a per-key multi-pass would
            # lose tie order on GPU. Instead express null placement with an explicit per-key
            # null-indicator column and sort in ONE pass: a key sorted ascending=(not desc) with
            # its indicator sorted the same way puts NULL at the correct (largest) end, no stable
            # sort required. The default single sort_values used pandas' na_position='last' for
            # every key, so DESC keys put NULL last, flipping which row an ORDER BY ... LIMIT keeps.
            sort_cols: List[str] = []
            ascending: List[bool] = []
            helper_cols: List[str] = []
            for i, (key, desc) in enumerate(order_keys):
                indicator = f"__gfql_nullrank_{i}__"
                out_df[indicator] = out_df[key].isna()
                sort_cols.extend([indicator, key])
                ascending.extend([not desc, not desc])
                helper_cols.append(indicator)
            out_df = out_df.sort_values(by=sort_cols, ascending=ascending)
            out_df = cast(DataFrameT, out_df.drop(columns=helper_cols))
        if limit_value is not None:
            out_df = cast(DataFrameT, out_df.head(limit_value))
        out_df = df_to_engine(out_df.reset_index(drop=True), requested_engine)

    out = base_graph.bind()
    out._nodes = out_df
    out._edges = df_cons(requested_engine)()
    return out


def _execute_two_hop_count_fast_path(
    base_graph: Plottable,
    chain: Chain,
    *,
    engine: Union[EngineAbstract, str],
) -> Optional[Plottable]:
    alias = _two_hop_count_alias(chain)
    if alias is None:
        return None
    ops = _two_hop_count_binding_ops(chain)
    if ops is None:
        return None
    start_op, first_edge, middle_op, second_edge, end_op = ops

    requested_engine = resolve_engine(cast(Any, engine), base_graph)
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

    nodes = cast(DataFrameT, nodes_obj)
    start_nodes = _connected_join_cached_node_filter(base_graph, nodes, cast(Optional[dict], start_op.filter_dict), engine=requested_engine)
    middle_nodes = (
        start_nodes
        if middle_op.filter_dict == start_op.filter_dict
        else _connected_join_cached_node_filter(base_graph, nodes, cast(Optional[dict], middle_op.filter_dict), engine=requested_engine)
    )
    end_nodes = (
        middle_nodes
        if end_op.filter_dict == middle_op.filter_dict
        else start_nodes
        if end_op.filter_dict == start_op.filter_dict
        else _connected_join_cached_node_filter(base_graph, nodes, cast(Optional[dict], end_op.filter_dict), engine=requested_engine)
    )
    first_edges = _connected_join_cached_edge_filter(base_graph, cast(DataFrameT, edges_obj), cast(Optional[dict], first_edge.edge_match), engine=requested_engine)
    reuse_single_edge_domain = (
        start_op.filter_dict == middle_op.filter_dict == end_op.filter_dict
        and first_edge.edge_match == second_edge.edge_match
    )
    second_edges = (
        first_edges
        if first_edge.edge_match == second_edge.edge_match
        else _connected_join_cached_edge_filter(base_graph, cast(DataFrameT, edges_obj), cast(Optional[dict], second_edge.edge_match), engine=requested_engine)
    )

    if requested_engine in POLARS_ENGINES:
        import polars as pl
        if reuse_single_edge_domain:
            cached_counts = _two_hop_cached_equal_domain_degree_counts(
                base_graph,
                nodes,
                cast(DataFrameT, edges_obj),
                start_nodes,
                first_edges,
                node_match=cast(Optional[dict], start_op.filter_dict),
                edge_match=cast(Optional[dict], first_edge.edge_match),
                node_col=node_col,
                src_col=src_col,
                dst_col=dst_col,
                engine=requested_engine,
            )
            if cached_counts is None:
                domain_ids = start_nodes.select(node_col).unique()
                domain_edges = (
                    first_edges
                    .join(domain_ids, left_on=src_col, right_on=node_col, how="semi")
                    .join(domain_ids, left_on=dst_col, right_on=node_col, how="semi")
                )
                in_counts = domain_edges.group_by(dst_col).len("__in_count__")
                out_counts = domain_edges.group_by(src_col).len("__out_count__")
            else:
                in_counts, out_counts = cached_counts
        else:
            start_ids = start_nodes.select(node_col).unique()
            middle_ids = middle_nodes.select(node_col).unique()
            end_ids = end_nodes.select(node_col).unique()
            in_counts = (
                first_edges
                .join(start_ids, left_on=src_col, right_on=node_col, how="semi")
                .join(middle_ids, left_on=dst_col, right_on=node_col, how="semi")
                .group_by(dst_col)
                .len("__in_count__")
            )
            out_counts = (
                second_edges
                .join(middle_ids, left_on=src_col, right_on=node_col, how="semi")
                .join(end_ids, left_on=dst_col, right_on=node_col, how="semi")
                .group_by(src_col)
                .len("__out_count__")
            )
        total_df = (
            in_counts
            .join(out_counts, left_on=dst_col, right_on=src_col, how="inner")
            .select((pl.col("__in_count__") * pl.col("__out_count__")).sum().fill_null(0).cast(pl.Int64).alias(alias))
        )
        out_nodes = cast(DataFrameT, total_df)
    else:
        if reuse_single_edge_domain:
            cached_counts = _two_hop_cached_equal_domain_degree_counts(
                base_graph,
                nodes,
                cast(DataFrameT, edges_obj),
                start_nodes,
                first_edges,
                node_match=cast(Optional[dict], start_op.filter_dict),
                edge_match=cast(Optional[dict], first_edge.edge_match),
                node_col=node_col,
                src_col=src_col,
                dst_col=dst_col,
                engine=requested_engine,
            )
            if cached_counts is None:
                domain_ids = start_nodes[node_col].drop_duplicates()
                domain_edges = first_edges[first_edges[src_col].isin(domain_ids) & first_edges[dst_col].isin(domain_ids)]
                in_counts = domain_edges.groupby(dst_col, sort=False).size().reset_index(name="__in_count__")
                out_counts = domain_edges.groupby(src_col, sort=False).size().reset_index(name="__out_count__")
            else:
                in_counts, out_counts = cached_counts
        else:
            start_ids = start_nodes[node_col].drop_duplicates()
            middle_ids = middle_nodes[node_col].drop_duplicates()
            end_ids = end_nodes[node_col].drop_duplicates()
            in_edges = first_edges[first_edges[src_col].isin(start_ids) & first_edges[dst_col].isin(middle_ids)]
            out_edges = second_edges[second_edges[src_col].isin(middle_ids) & second_edges[dst_col].isin(end_ids)]
            in_counts = in_edges.groupby(dst_col, sort=False).size().reset_index(name="__in_count__")
            out_counts = out_edges.groupby(src_col, sort=False).size().reset_index(name="__out_count__")
        joined = in_counts.merge(out_counts, left_on=dst_col, right_on=src_col, how="inner")
        total = int((joined["__in_count__"] * joined["__out_count__"]).sum()) if len(joined) else 0
        out_nodes = df_to_engine(pd.DataFrame({alias: [total]}), requested_engine)

    out = base_graph.bind()
    out._nodes = out_nodes
    out._edges = df_cons(requested_engine)()
    return out


def _execute_seeded_typed_hop_fast_path(
    base_graph: Plottable,
    compiled_query: CompiledCypherQuery,
    physical_plan: "PhysicalPlan",
    *,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
    start_nodes: Optional[DataFrameT] = None,
) -> Optional[Plottable]:
    """#1755 cypher-surface fast path: a seeded typed 1-hop with a whole-row node
    RETURN — ``MATCH (m {id})-[:T]->(p) RETURN p`` — reduces the graph to the seed's
    1-hop neighborhood with a few DataFrame filters (pandas/cuDF via the shared
    DataFrame API, or polars via polars filters), then applies the RETURN projection
    to just the destination rows (value-identical to the full path: same rows/
    columns/dtypes; row order and RangeIndex may differ; sub-ms because the frame
    is tiny). Returns None to fall through for anything outside this exact shape
    or carrying side-channels (policy, same-path WHERE, OPTIONAL null-row,
    carried reentry seeds)."""
    if start_nodes is not None:
        # A carried seed set (WITH..MATCH reentry) restricts n0 to those rows; the fast
        # path derives its seed from n0.filter_dict alone, so engaging here would
        # silently widen the seed back to the whole graph. Decline.
        return None
    if policy:
        # The full path fires prechain/postchain/postload policy hooks; the lean
        # projection below never enters chain(), so engaging would silently skip
        # them. Policy-gated queries always take the full path (native twin gates
        # identically in chain._try_chain_fast_path).
        return None
    if compiled_query.chain.where:
        # Same-path WHERE entries (e.g. WHERE p.id < m.id) are evaluated by the
        # full pipeline, not by the seed-first reduction — engaging would drop them.
        return None
    if compiled_query.empty_result_row is not None:
        # OPTIONAL MATCH null-row semantics: on no match the full path emits the
        # openCypher null row; the lean projection would return an empty frame.
        return None
    requested_engine = resolve_engine(cast(Any, engine), base_graph)
    if requested_engine not in (Engine.PANDAS, Engine.CUDF, Engine.POLARS, Engine.POLARS_GPU):
        return None
    projection = compiled_query.result_projection
    if projection is None or projection.table != "nodes":
        return None
    # Only a single whole-row node alias (RETURN p). Multi-alias returns (RETURN
    # m, p) combine aliases into one row per match — different shape — so bail.
    proj_cols = projection.columns
    if len(proj_cols) != 1 or proj_cols[0].kind != "whole_row":
        return None
    if compiled_query.execution_extras is not None and (
        compiled_query.execution_extras.connected_match_join is not None
        or compiled_query.execution_extras.connected_optional_match is not None
    ):
        return None
    ops = list(compiled_query.chain.chain)
    if len(ops) != 4:
        return None
    n0, e1, n2, call = ops
    if not (isinstance(n0, ASTNode) and isinstance(e1, ASTEdge)
            and isinstance(n2, ASTNode) and isinstance(call, ASTCall)):
        return None
    # Only a genuine SINGLE hop. A variable-length edge (-[*1..2]->) is still one
    # ASTEdge but expands to multiple hops, so the seeded 1-hop reduction below
    # would silently truncate it. Reuse the same canonical gate the native fast
    # path uses (also rejects hop labels / output slicing / fixed-point). See
    # test_engine_polars_chain::TestVarlenAliasHopGate.
    if not e1.is_simple_single_hop():
        return None
    if call.function != "rows":
        return None
    node, src, dst = base_graph._node, base_graph._source, base_graph._destination
    if node is None or src is None or dst is None:
        return None
    # RETURN alias must be the DESTINATION node (n2) and the seed must sit on the
    # source node (n0) — the forward seeded shape MATCH (m {id})-[:T]->(p) RETURN p.
    # Other alias/seed placements (e.g. reverse patterns where the seed is on the
    # RETURN node) fall back to the full path.
    if n2._name != projection.alias:
        return None
    if not (n0.filter_dict and any(not str(k).startswith("label__") for k in n0.filter_dict)):
        return None  # n0 must carry a selective (non-label) seed
    direction = e1.direction
    # Dispatch on the ACTUAL frame type, not the requested engine: the WITH..MATCH
    # reentry path can request engine=polars while handing us a pandas-materialized
    # intermediate graph, so trusting requested_engine would run polars ops on a
    # pandas frame (and vice versa). The pandas branch also covers cuDF (shared API).
    from graphistry.Engine import is_polars_df
    from graphistry.compute.chain_fast_paths import (
        _seeded_typed_return_dst_pandas_cudf, _seeded_typed_return_dst_polars,
    )
    nodes_frame = base_graph._nodes
    is_polars = is_polars_df(nodes_frame)
    if is_polars != is_polars_df(base_graph._edges):
        return None  # mixed-engine node/edge frames: decline, full path decides
    helper = _seeded_typed_return_dst_polars if is_polars else _seeded_typed_return_dst_pandas_cudf
    dst_res = helper(base_graph, n0, n2, e1, src, dst, node, direction)
    if dst_res is None:
        return None
    p_rows, _edges = dst_res
    # Lean projection: p_rows already IS the RETURN-alias (destination) node set.
    # Tag with the alias and reuse apply_result_projection for the exact
    # column-order/flatten semantics — all on a handful of rows, so seeded cypher
    # stays sub-ms (vs the ~25ms rows-pivot pipeline on the full graph).
    if is_polars:
        import polars as pl
        tagged = p_rows.with_columns(pl.lit(True).alias(projection.alias))
    else:
        tagged = p_rows.assign(**{projection.alias: True})
    result = base_graph.nodes(tagged)
    return apply_result_projection(result, projection)
