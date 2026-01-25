"""Post-pruning passes for same-path WHERE clause execution.

Contains the non-adjacent node and edge WHERE clause application logic.
These are applied after the initial backward prune to enforce constraints
that span multiple edges in the chain.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from graphistry.compute.ast import ASTEdge
from graphistry.compute.typing import DataFrameT
from graphistry.compute.gfql.same_path_types import PathState
from graphistry.otel import otel_detail_enabled
from .edge_semantics import EdgeSemantics
from .bfs import build_edge_pairs
from .df_utils import (
    evaluate_clause,
    series_values,
    concat_frames,
    df_cons,
    make_bool_series,
    domain_is_empty,
    domain_intersect,
    domain_to_frame,
    domain_empty,
)
from .multihop import filter_multihop_edges_by_endpoints, find_multihop_start_nodes

if TYPE_CHECKING:
    from graphistry.compute.gfql.df_executor import (
        DFSamePathExecutor,
        WhereComparison,
    )


def apply_non_adjacent_where_post_prune(
    executor: "DFSamePathExecutor",
    state: PathState,
    span: Optional[Any] = None,
) -> PathState:
    """Apply WHERE on non-adjacent node aliases by tracing paths.

    Args:
        executor: The executor instance with chain metadata and state
        state: Current PathState with allowed_nodes/allowed_edges

    Returns:
        New PathState with constraints applied
    """
    if not executor.inputs.where:
        return state

    # Experimental non-adjacent WHERE modes; default auto unless explicitly set.
    non_adj_mode = os.environ.get("GRAPHISTRY_NON_ADJ_WHERE_MODE", "auto").strip().lower()
    if not non_adj_mode:
        non_adj_mode = "auto"
    if not non_adj_mode:
        non_adj_mode = "auto"
    non_adj_strategy = os.environ.get("GRAPHISTRY_NON_ADJ_WHERE_STRATEGY", "").strip().lower()
    non_adj_order = os.environ.get("GRAPHISTRY_NON_ADJ_WHERE_ORDER", "").strip().lower()
    bounds_enabled = os.environ.get("GRAPHISTRY_NON_ADJ_WHERE_BOUNDS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    non_adj_value_card_max = os.environ.get("GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX", "").strip()
    non_adj_vector_max_hops = os.environ.get("GRAPHISTRY_NON_ADJ_WHERE_VECTOR_MAX_HOPS", "").strip()
    non_adj_vector_label_max = os.environ.get("GRAPHISTRY_NON_ADJ_WHERE_VECTOR_LABEL_MAX", "").strip()
    non_adj_vector_pair_max = os.environ.get("GRAPHISTRY_NON_ADJ_WHERE_VECTOR_PAIR_MAX", "").strip()
    non_adj_sip_ratio_raw = os.environ.get("GRAPHISTRY_NON_ADJ_WHERE_SIP_RATIO", "").strip()
    non_adj_domain_semijoin_raw = os.environ.get(
        "GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN", ""
    ).strip().lower()
    non_adj_domain_semijoin_auto_raw = os.environ.get(
        "GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_AUTO", ""
    ).strip().lower()
    non_adj_multi_eq_semijoin_raw = os.environ.get(
        "GRAPHISTRY_NON_ADJ_WHERE_MULTI_EQ_SEMIJOIN", ""
    ).strip().lower()
    non_adj_domain_semijoin_pair_max_raw = os.environ.get(
        "GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_PAIR_MAX", ""
    ).strip()
    non_adj_ineq_agg_raw = os.environ.get(
        "GRAPHISTRY_NON_ADJ_WHERE_INEQ_AGG", ""
    ).strip().lower()
    non_adj_value_ops_raw = os.environ.get("GRAPHISTRY_NON_ADJ_WHERE_VALUE_OPS", "").strip().lower()
    if non_adj_value_ops_raw:
        value_mode_ops = {
            op.strip()
            for op in non_adj_value_ops_raw.split(",")
            if op.strip()
        }
    else:
        if non_adj_mode in {"auto", "auto_prefilter"}:
            value_mode_ops = {"==", "!="}
        else:
            value_mode_ops = {"=="}
    value_mode_ops = {
        op for op in value_mode_ops
        if op in {"==", "!=", "<", "<=", ">", ">="}
    }
    if not value_mode_ops:
        value_mode_ops = {"=="}
    try:
        value_card_max = int(non_adj_value_card_max) if non_adj_value_card_max else None
    except ValueError:
        value_card_max = None
    if value_card_max is None and non_adj_mode in {"auto", "auto_prefilter"}:
        value_card_max = 300
    try:
        vector_max_hops = int(non_adj_vector_max_hops) if non_adj_vector_max_hops else 3
    except ValueError:
        vector_max_hops = 3
    try:
        vector_label_max = int(non_adj_vector_label_max) if non_adj_vector_label_max else None
    except ValueError:
        vector_label_max = None
    try:
        vector_pair_max = int(non_adj_vector_pair_max) if non_adj_vector_pair_max else 200000
    except ValueError:
        vector_pair_max = 200000
    if vector_pair_max is not None and vector_pair_max <= 0:
        vector_pair_max = None
    sip_ratio = 5.0
    if non_adj_sip_ratio_raw:
        try:
            sip_ratio = float(non_adj_sip_ratio_raw)
        except ValueError:
            sip_ratio = 5.0
    if sip_ratio <= 0:
        sip_ratio = None
    domain_semijoin_enabled = non_adj_domain_semijoin_raw in {"1", "true", "yes", "on"}
    domain_semijoin_auto = non_adj_domain_semijoin_auto_raw in {"1", "true", "yes", "on"}
    if (
        not non_adj_domain_semijoin_auto_raw
        and non_adj_mode in {"auto", "auto_prefilter"}
    ):
        domain_semijoin_auto = True
    multi_eq_semijoin_enabled = non_adj_multi_eq_semijoin_raw in {"1", "true", "yes", "on"}
    ineq_agg_enabled = non_adj_ineq_agg_raw in {"1", "true", "yes", "on"}
    try:
        domain_semijoin_pair_max = (
            int(non_adj_domain_semijoin_pair_max_raw)
            if non_adj_domain_semijoin_pair_max_raw
            else (vector_pair_max if vector_pair_max is not None else 200000)
        )
    except ValueError:
        domain_semijoin_pair_max = vector_pair_max if vector_pair_max is not None else 200000
    if domain_semijoin_pair_max is not None and domain_semijoin_pair_max <= 0:
        domain_semijoin_pair_max = None
    if vector_label_max is None:
        vector_label_max = value_card_max if value_card_max is not None else 1000

    non_adjacent_clauses = []
    for clause in executor.inputs.where:
        left_alias = clause.left.alias
        right_alias = clause.right.alias
        left_binding = executor.inputs.alias_bindings.get(left_alias)
        right_binding = executor.inputs.alias_bindings.get(right_alias)
        if left_binding and right_binding:
            if left_binding.kind == "node" and right_binding.kind == "node":
                # Non-adjacent = step indices differ by more than 2
                if not executor.meta.are_steps_adjacent_nodes(
                    left_binding.step_index, right_binding.step_index
                ):
                    non_adjacent_clauses.append(clause)

    if not non_adjacent_clauses:
        return state

    local_allowed_nodes: Dict[int, Any] = dict(state.allowed_nodes)
    local_allowed_edges: Dict[int, Any] = dict(state.allowed_edges)
    local_pruned_edges: Dict[int, Any] = dict(state.pruned_edges)

    edge_indices = executor.meta.edge_indices

    src_col = executor._source_column
    dst_col = executor._destination_column
    edge_id_col = executor._edge_column
    node_id_col = executor._node_column
    nodes_df = executor.inputs.graph._nodes

    if not src_col or not dst_col:
        return state

    if (
        non_adj_order in {"selectivity", "size"}
        and nodes_df is not None
        and node_id_col
        and node_id_col in nodes_df.columns
    ):
        def _clause_order_key(clause: "WhereComparison") -> tuple:
            left_alias = clause.left.alias
            right_alias = clause.right.alias
            left_binding = executor.inputs.alias_bindings.get(left_alias)
            right_binding = executor.inputs.alias_bindings.get(right_alias)
            if not left_binding or not right_binding:
                return (float("inf"), float("inf"))
            start_idx = left_binding.step_index
            end_idx = right_binding.step_index
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            start_nodes = local_allowed_nodes.get(start_idx)
            end_nodes = local_allowed_nodes.get(end_idx)
            if domain_is_empty(start_nodes) or domain_is_empty(end_nodes):
                return (float("inf"), float("inf"))
            left_col = clause.left.column
            right_col = clause.right.column
            if left_col not in nodes_df.columns or right_col not in nodes_df.columns:
                return (float("inf"), float("inf"))
            left_vals = nodes_df[nodes_df[node_id_col].isin(start_nodes)][left_col]
            right_vals = nodes_df[nodes_df[node_id_col].isin(end_nodes)][right_col]
            left_domain = series_values(left_vals)
            right_domain = series_values(right_vals)
            if clause.op == "==":
                inter = domain_intersect(left_domain, right_domain)
                score = len(inter) if not domain_is_empty(inter) else float("inf")
            else:
                score = max(len(left_domain), len(right_domain))
            return (score, end_idx - start_idx)

        non_adjacent_clauses = sorted(non_adjacent_clauses, key=_clause_order_key)

    def _filter_values_df_by_const(
        values_df: Any,
        value_col: str,
        op: str,
        const_value: Any,
        *,
        const_on_left: bool,
    ) -> Any:
        if values_df is None or len(values_df) == 0:
            return values_df
        if const_on_left:
            if op == "==":
                mask = values_df[value_col] == const_value
            elif op == "!=":
                mask = values_df[value_col] != const_value
            elif op == "<":
                mask = values_df[value_col] > const_value
            elif op == "<=":
                mask = values_df[value_col] >= const_value
            elif op == ">":
                mask = values_df[value_col] < const_value
            elif op == ">=":
                mask = values_df[value_col] <= const_value
            else:
                mask = values_df[value_col] == const_value
        else:
            if op == "==":
                mask = values_df[value_col] == const_value
            elif op == "!=":
                mask = values_df[value_col] != const_value
            elif op == "<":
                mask = values_df[value_col] < const_value
            elif op == "<=":
                mask = values_df[value_col] <= const_value
            elif op == ">":
                mask = values_df[value_col] > const_value
            elif op == ">=":
                mask = values_df[value_col] >= const_value
            else:
                mask = values_df[value_col] == const_value
        return values_df[mask]

    def _scalar_clause(left: Any, op: str, right: Any) -> bool:
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        return False

    clause_count = 0
    state_rows_max = 0
    pairs_rows_max = 0
    valid_pairs_max = 0
    last_state_rows = 0
    left_value_count_max = 0
    right_value_count_max = 0
    mid_intersect_rows_max = 0
    mid_label_intersect_rows_max = 0
    pairs_left_rows_max = 0
    pairs_right_rows_max = 0
    value_mode_used = False
    prefilter_used = False
    singleton_used = False
    bounds_used = False
    order_used = non_adj_order in {"selectivity", "size"}
    multi_eq_value_used = False
    multi_eq_label_card_max = 0
    domain_semijoin_used = False
    domain_semijoin_pairs_max = 0
    domain_semijoin_auto_used = False
    domain_semijoin_pair_est_max = 0
    value_pair_guard_used = False
    value_pair_guard_pair_est_max = 0
    value_pair_guard_edge_est_max = 0
    ineq_agg_used = False
    ineq_agg_pair_est_max = 0
    vector_used = False
    vector_label_card_max = 0
    vector_candidate_pairs_max = 0
    vector_path_pairs_max = 0
    vector_pair_est_max = 0
    composite_value_enabled = non_adj_mode in {
        "value",
        "value_prefilter",
        "auto",
        "auto_prefilter",
    }
    vector_enabled = non_adj_strategy == "vector"
    multi_eq_groups: Dict[tuple, List[tuple]] = {}
    multi_eq_order: List[tuple] = []
    processed_clause_ids: set = set()

    def _collect_multi_eq_groups(
        clauses: Sequence["WhereComparison"],
    ):
        groups: Dict[tuple, List[tuple]] = {}
        order: List[tuple] = []
        for clause in clauses:
            if clause.op != "==":
                continue
            left_binding = executor.inputs.alias_bindings.get(clause.left.alias)
            right_binding = executor.inputs.alias_bindings.get(clause.right.alias)
            if not left_binding or not right_binding:
                continue
            start_idx = left_binding.step_index
            end_idx = right_binding.step_index
            start_col = clause.left.column
            end_col = clause.right.column
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
                start_col, end_col = end_col, start_col
            key = (start_idx, end_idx)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((start_col, end_col, clause))
        groups = {
            key: entries for key, entries in groups.items()
            if len(entries) >= 2
        }
        return groups, order

    if composite_value_enabled or vector_enabled:
        multi_eq_groups, multi_eq_order = _collect_multi_eq_groups(non_adjacent_clauses)

    edge_pairs_cache: Dict[int, DataFrameT] = {}

    def _edge_pairs_cached(
        edge_idx: int,
        sem: EdgeSemantics,
        allowed_edges: Optional[Any],
    ) -> DataFrameT:
        edges_df = executor.forward_steps[edge_idx]._edges
        if edges_df is None or len(edges_df) == 0:
            template = nodes_df if nodes_df is not None else executor.inputs.graph._edges
            if template is None:
                import pandas as pd

                return pd.DataFrame({"__from__": [], "__to__": []})
            return df_cons(template, {"__from__": [], "__to__": []})

        if allowed_edges is None:
            cached = edge_pairs_cache.get(edge_idx)
            if cached is None:
                cached = build_edge_pairs(edges_df, src_col, dst_col, sem)
                edge_pairs_cache[edge_idx] = cached
            return cached

        if edge_id_col and edge_id_col in edges_df.columns:
            edges_df = edges_df[edges_df[edge_id_col].isin(allowed_edges)]
        return build_edge_pairs(edges_df, src_col, dst_col, sem)

    endpoint_clause_counts: Dict[Tuple[int, int], int] = {}
    endpoint_eq_clauses: Dict[Tuple[int, int], List[Tuple["WhereComparison", str, str]]] = {}
    for clause in non_adjacent_clauses:
        left_binding = executor.inputs.alias_bindings.get(clause.left.alias)
        right_binding = executor.inputs.alias_bindings.get(clause.right.alias)
        if not left_binding or not right_binding:
            continue
        if left_binding.kind != "node" or right_binding.kind != "node":
            continue
        start_idx = left_binding.step_index
        end_idx = right_binding.step_index
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        endpoint_clause_counts[(start_idx, end_idx)] = endpoint_clause_counts.get(
            (start_idx, end_idx), 0
        ) + 1
        if clause.op == "==":
            start_col = clause.left.column
            end_col = clause.right.column
            if left_binding.step_index > right_binding.step_index:
                start_col, end_col = end_col, start_col
            endpoint_eq_clauses.setdefault((start_idx, end_idx), []).append(
                (clause, start_col, end_col)
            )

    if vector_enabled and multi_eq_groups:
        for key in multi_eq_order:
            group_entries = multi_eq_groups.get(key)
            if not group_entries:
                continue
            if any(id(clause) in processed_clause_ids for _, _, clause in group_entries):
                continue
            start_node_idx, end_node_idx = key
            if nodes_df is None or not node_id_col or node_id_col not in nodes_df.columns:
                continue

            relevant_edge_indices = [
                idx for idx in edge_indices
                if start_node_idx < idx < end_node_idx
            ]

            start_nodes = local_allowed_nodes.get(start_node_idx)
            end_nodes = local_allowed_nodes.get(end_node_idx)

            if (
                non_adj_mode in {"auto", "auto_prefilter"}
                and domain_semijoin_pair_max is not None
            ):
                start_count = 0 if domain_is_empty(start_nodes) else len(start_nodes)
                end_count = 0 if domain_is_empty(end_nodes) else len(end_nodes)
                pair_est = start_count * end_count
                value_pair_guard_pair_est_max = max(value_pair_guard_pair_est_max, pair_est)
                guard = pair_est > domain_semijoin_pair_max
                if len(relevant_edge_indices) == 2:
                    edge_left = executor.forward_steps[relevant_edge_indices[0]]._edges
                    edge_right = executor.forward_steps[relevant_edge_indices[1]]._edges
                    edge_left_count = (
                        len(local_allowed_edges[relevant_edge_indices[0]])
                        if local_allowed_edges.get(relevant_edge_indices[0]) is not None
                        else (len(edge_left) if edge_left is not None else 0)
                    )
                    edge_right_count = (
                        len(local_allowed_edges[relevant_edge_indices[1]])
                        if local_allowed_edges.get(relevant_edge_indices[1]) is not None
                        else (len(edge_right) if edge_right is not None else 0)
                    )
                    edge_pair_est = edge_left_count * edge_right_count
                    value_pair_guard_edge_est_max = max(value_pair_guard_edge_est_max, edge_pair_est)
                    guard = guard or (edge_pair_est > domain_semijoin_pair_max)
                if guard:
                    value_pair_guard_used = True
                    continue
            if len(relevant_edge_indices) == 0 or len(relevant_edge_indices) > vector_max_hops:
                continue
            if domain_is_empty(start_nodes) or domain_is_empty(end_nodes):
                continue

            start_base = nodes_df[nodes_df[node_id_col].isin(start_nodes)]
            end_base = nodes_df[nodes_df[node_id_col].isin(end_nodes)]
            if len(start_base) == 0 or len(end_base) == 0:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                for _, _, clause in group_entries:
                    processed_clause_ids.add(id(clause))
                continue

            clause_specs: List[tuple] = []
            vector_applicable = True
            early_pruned = False
            for start_col, end_col, _ in group_entries:
                if start_col not in start_base.columns or end_col not in end_base.columns:
                    vector_applicable = False
                    break
                start_vals = start_base[[node_id_col, start_col]].rename(
                    columns={node_id_col: "__start__", start_col: "__value__"}
                )
                end_vals = end_base[[node_id_col, end_col]].rename(
                    columns={node_id_col: "__current__", end_col: "__value__"}
                )
                start_vals = start_vals[start_vals["__value__"].notna()]
                end_vals = end_vals[end_vals["__value__"].notna()]
                if len(start_vals) == 0 or len(end_vals) == 0:
                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                    for _, _, clause in group_entries:
                        processed_clause_ids.add(id(clause))
                    early_pruned = True
                    break
                start_vals = start_vals.drop_duplicates()
                end_vals = end_vals.drop_duplicates()

                start_counts = start_vals.groupby("__value__").size().reset_index()
                start_counts.columns = ["__value__", "__start_count__"]
                end_counts = end_vals.groupby("__value__").size().reset_index()
                end_counts.columns = ["__value__", "__end_count__"]
                pair_counts = start_counts.merge(end_counts, on="__value__", how="inner")
                label_cardinality = len(pair_counts)
                vector_label_card_max = max(vector_label_card_max, label_cardinality)
                if label_cardinality == 0:
                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                    for _, _, clause in group_entries:
                        processed_clause_ids.add(id(clause))
                    early_pruned = True
                    break
                if vector_label_max is not None and label_cardinality > vector_label_max:
                    vector_applicable = False
                    break

                pair_est = (pair_counts["__start_count__"] * pair_counts["__end_count__"]).sum()
                try:
                    pair_est_value = int(pair_est)
                except Exception:
                    pair_est_value = pair_est
                vector_pair_est_max = max(vector_pair_est_max, pair_est_value)
                if vector_pair_max is not None and pair_est_value > vector_pair_max:
                    vector_applicable = False
                    break

                allowed_values = pair_counts[["__value__"]]
                start_vals = start_vals.merge(allowed_values, on="__value__", how="inner")
                end_vals = end_vals.merge(allowed_values, on="__value__", how="inner")
                clause_specs.append((pair_est_value, start_vals, end_vals))

            if early_pruned:
                continue
            if not vector_applicable or not clause_specs:
                continue

            clause_specs.sort(key=lambda item: item[0])
            candidate_pairs = None
            for _, start_vals, end_vals in clause_specs:
                pairs = start_vals.merge(end_vals, on="__value__", how="inner")[
                    ["__start__", "__current__"]
                ].drop_duplicates()
                if candidate_pairs is None:
                    candidate_pairs = pairs
                else:
                    candidate_pairs = candidate_pairs.merge(
                        pairs, on=["__start__", "__current__"], how="inner"
                    ).drop_duplicates()
                if len(candidate_pairs) == 0:
                    break
                if vector_pair_max is not None and len(candidate_pairs) > vector_pair_max:
                    vector_applicable = False
                    break

            if not vector_applicable:
                continue
            if candidate_pairs is None or len(candidate_pairs) == 0:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                for _, _, clause in group_entries:
                    processed_clause_ids.add(id(clause))
                continue
            vector_candidate_pairs_max = max(vector_candidate_pairs_max, len(candidate_pairs))

            candidate_start_nodes = series_values(candidate_pairs["__start__"])
            candidate_end_nodes = series_values(candidate_pairs["__current__"])

            def _vector_edge_pairs(edge_idx: int):
                edges_df = executor.forward_steps[edge_idx]._edges
                if edges_df is None or len(edges_df) == 0:
                    return df_cons(nodes_df, {"__from__": [], "__to__": []}), True

                allowed_edges = local_allowed_edges.get(edge_idx)
                if allowed_edges is not None and edge_id_col and edge_id_col in edges_df.columns:
                    edges_df = edges_df[edges_df[edge_id_col].isin(allowed_edges)]

                edge_op = executor.inputs.chain[edge_idx]
                if not isinstance(edge_op, ASTEdge):
                    return None, False
                sem = EdgeSemantics.from_edge(edge_op)
                if sem.is_multihop:
                    return None, False

                pairs = build_edge_pairs(edges_df, src_col, dst_col, sem).drop_duplicates()
                from_nodes = local_allowed_nodes.get(edge_idx - 1)
                to_nodes = local_allowed_nodes.get(edge_idx + 1)
                if edge_idx - 1 == start_node_idx and not domain_is_empty(candidate_start_nodes):
                    if domain_is_empty(from_nodes):
                        from_nodes = candidate_start_nodes
                    else:
                        from_nodes = domain_intersect(from_nodes, candidate_start_nodes)
                if edge_idx + 1 == end_node_idx and not domain_is_empty(candidate_end_nodes):
                    if domain_is_empty(to_nodes):
                        to_nodes = candidate_end_nodes
                    else:
                        to_nodes = domain_intersect(to_nodes, candidate_end_nodes)
                if not domain_is_empty(from_nodes):
                    pairs = pairs[pairs["__from__"].isin(from_nodes)]
                if not domain_is_empty(to_nodes):
                    pairs = pairs[pairs["__to__"].isin(to_nodes)]
                return pairs, True

            def _bounded_product(values: Sequence[int], cap: Optional[int]) -> int:
                total = 1
                for value in values:
                    if value <= 0:
                        return 0
                    total *= int(value)
                    if cap is not None and total > cap:
                        return cap
                return total

            def _sip_prefilter(
                left_df: DataFrameT,
                left_key: str,
                right_df: DataFrameT,
                right_key: str,
            ) -> Tuple[DataFrameT, DataFrameT]:
                if sip_ratio is None:
                    return left_df, right_df
                left_len = len(left_df)
                right_len = len(right_df)
                if left_len == 0 or right_len == 0:
                    return left_df, right_df
                if left_len > sip_ratio * right_len:
                    right_keys = series_values(right_df[right_key])
                    left_df = left_df[left_df[left_key].isin(right_keys)]
                elif right_len > sip_ratio * left_len:
                    left_keys = series_values(left_df[left_key])
                    right_df = right_df[right_df[right_key].isin(left_keys)]
                return left_df, right_df

            def _join_edge_pairs(edge_pairs: Sequence[Any], start_label: str, end_label: str):
                path = None
                for pairs in edge_pairs:
                    if path is None:
                        path = pairs.rename(
                            columns={"__from__": start_label, "__to__": "__current__"}
                        )
                    else:
                        next_pairs = pairs.rename(
                            columns={"__from__": "__current__", "__to__": "__next__"}
                        )
                        path, next_pairs = _sip_prefilter(
                            path, "__current__", next_pairs, "__current__"
                        )
                        path = path.merge(next_pairs, on="__current__", how="inner")[
                            [start_label, "__next__"]
                        ].rename(columns={"__next__": "__current__"})
                    path = path.drop_duplicates()
                    if vector_pair_max is not None and len(path) > vector_pair_max:
                        return None
                    if len(path) == 0:
                        break
                if path is None:
                    return df_cons(nodes_df, {start_label: [], end_label: []})
                if end_label != "__current__":
                    path = path.rename(columns={"__current__": end_label})
                return path

            vector_applicable = True
            path_pairs = None
            if len(relevant_edge_indices) == 2:
                first_edge, second_edge = relevant_edge_indices
                first_pairs, ok = _vector_edge_pairs(first_edge)
                if not ok:
                    vector_applicable = False
                else:
                    second_pairs, ok = _vector_edge_pairs(second_edge)
                    if not ok:
                        vector_applicable = False
                    else:
                        if len(first_pairs) == 0 or len(second_pairs) == 0:
                            path_pairs = df_cons(nodes_df, {"__start__": [], "__current__": []})
                        else:
                            mid_candidates = domain_intersect(
                                series_values(first_pairs["__to__"]),
                                series_values(second_pairs["__from__"]),
                            )
                            if domain_is_empty(mid_candidates):
                                path_pairs = df_cons(
                                    nodes_df, {"__start__": [], "__current__": []}
                                )
                            else:
                                first_pairs = first_pairs[first_pairs["__to__"].isin(mid_candidates)]
                                second_pairs = second_pairs[second_pairs["__from__"].isin(mid_candidates)]
                                first_pairs = first_pairs.rename(
                                    columns={"__from__": "__start__", "__to__": "__mid__"}
                                )
                                second_pairs = second_pairs.rename(
                                    columns={"__from__": "__mid__", "__to__": "__current__"}
                                )
                                path_pairs = first_pairs.merge(
                                    second_pairs, on="__mid__", how="inner"
                                )[["__start__", "__current__"]].drop_duplicates()
            else:
                edge_pairs_list = []
                edge_pair_counts = []
                for edge_idx in relevant_edge_indices:
                    pairs, ok = _vector_edge_pairs(edge_idx)
                    if not ok:
                        vector_applicable = False
                        break
                    edge_pairs_list.append(pairs)
                    edge_pair_counts.append(len(pairs))
                if vector_applicable:
                    if len(edge_pairs_list) == 0:
                        path_pairs = df_cons(nodes_df, {"__start__": [], "__current__": []})
                    elif len(edge_pairs_list) == 1:
                        path_pairs = edge_pairs_list[0].rename(
                            columns={"__from__": "__start__", "__to__": "__current__"}
                        )
                    else:
                        best_split = 1
                        best_score = None
                        for split_idx in range(1, len(edge_pair_counts)):
                            prefix_est = _bounded_product(
                                edge_pair_counts[:split_idx], vector_pair_max
                            )
                            suffix_est = _bounded_product(
                                edge_pair_counts[split_idx:], vector_pair_max
                            )
                            score = max(prefix_est, suffix_est)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_split = split_idx
                        prefix_pairs = _join_edge_pairs(
                            edge_pairs_list[:best_split], "__start__", "__mid__"
                        )
                        if prefix_pairs is None:
                            vector_applicable = False
                        else:
                            suffix_pairs = _join_edge_pairs(
                                edge_pairs_list[best_split:], "__mid__", "__current__"
                            )
                            if suffix_pairs is None:
                                vector_applicable = False
                            else:
                                prefix_pairs, suffix_pairs = _sip_prefilter(
                                    prefix_pairs, "__mid__", suffix_pairs, "__mid__"
                                )
                                path_pairs = prefix_pairs.merge(
                                    suffix_pairs, on="__mid__", how="inner"
                                )[["__start__", "__current__"]].drop_duplicates()

            if not vector_applicable:
                continue

            vector_path_pairs_max = max(
                vector_path_pairs_max, len(path_pairs) if path_pairs is not None else 0
            )
            if vector_pair_max is not None and path_pairs is not None and len(path_pairs) > vector_pair_max:
                vector_applicable = False
                continue
            if path_pairs is None or len(path_pairs) == 0:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                for _, _, clause in group_entries:
                    processed_clause_ids.add(id(clause))
                continue

            valid_pairs = path_pairs.merge(
                candidate_pairs, on=["__start__", "__current__"], how="inner"
            )
            valid_pairs_max = max(valid_pairs_max, len(valid_pairs))
            if len(valid_pairs) == 0:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                for _, _, clause in group_entries:
                    processed_clause_ids.add(id(clause))
                continue

            valid_starts = series_values(valid_pairs["__start__"])
            valid_ends = series_values(valid_pairs["__current__"])
            if start_node_idx in local_allowed_nodes:
                local_allowed_nodes[start_node_idx] = domain_intersect(
                    local_allowed_nodes[start_node_idx], valid_starts
                )
            if end_node_idx in local_allowed_nodes:
                local_allowed_nodes[end_node_idx] = domain_intersect(
                    local_allowed_nodes[end_node_idx], valid_ends
                )

            vector_used = True
            clause_count += len(group_entries)
            for _, _, clause in group_entries:
                processed_clause_ids.add(id(clause))

            current_state = PathState.from_mutable(
                local_allowed_nodes, local_allowed_edges, local_pruned_edges
            )
            current_state = executor.backward_propagate_constraints(
                current_state, start_node_idx, end_node_idx
            )
            local_allowed_nodes, local_allowed_edges = current_state.to_mutable()
            local_pruned_edges.update(current_state.pruned_edges)

    if composite_value_enabled and multi_eq_groups:
        for key in multi_eq_order:
            group_entries = multi_eq_groups.get(key)
            if not group_entries:
                continue
            if any(id(clause) in processed_clause_ids for _, _, clause in group_entries):
                continue
            start_node_idx, end_node_idx = key

            start_nodes = local_allowed_nodes.get(start_node_idx)
            end_nodes = local_allowed_nodes.get(end_node_idx)
            if domain_is_empty(start_nodes) or domain_is_empty(end_nodes):
                continue
            if nodes_df is None or not node_id_col or node_id_col not in nodes_df.columns:
                continue

            relevant_edge_indices = [
                idx for idx in edge_indices
                if start_node_idx < idx < end_node_idx
            ]

            start_base = nodes_df[nodes_df[node_id_col].isin(start_nodes)]
            end_base = nodes_df[nodes_df[node_id_col].isin(end_nodes)]
            if len(start_base) == 0 or len(end_base) == 0:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                continue

            start_df = start_base[[node_id_col]].rename(columns={node_id_col: "__start__"}).copy()
            end_df = end_base[[node_id_col]].rename(columns={node_id_col: "__current__"}).copy()
            label_cols: List[str] = []
            can_build = True
            for idx, (start_col, end_col, _) in enumerate(group_entries):
                if start_col not in start_base.columns or end_col not in end_base.columns:
                    can_build = False
                    break
                label_col = f"__label{idx}__"
                label_cols.append(label_col)
                start_df[label_col] = start_base[start_col]
                end_df[label_col] = end_base[end_col]

            if not can_build or not label_cols:
                continue

            start_mask = start_df[label_cols[0]].notna()
            end_mask = end_df[label_cols[0]].notna()
            for label_col in label_cols[1:]:
                start_mask = start_mask & start_df[label_col].notna()
                end_mask = end_mask & end_df[label_col].notna()
            start_df = start_df[start_mask]
            end_df = end_df[end_mask]
            if len(start_df) == 0 or len(end_df) == 0:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                continue

            start_labels = start_df[label_cols].drop_duplicates()
            end_labels = end_df[label_cols].drop_duplicates()
            label_cardinality = max(len(start_labels), len(end_labels))
            multi_eq_label_card_max = max(multi_eq_label_card_max, label_cardinality)
            if value_card_max is not None and label_cardinality > value_card_max:
                continue

            if (
                multi_eq_semijoin_enabled
                and (domain_semijoin_enabled or domain_semijoin_auto)
                and len(relevant_edge_indices) == 2
                and nodes_df is not None
            ):
                edge_idx_left, edge_idx_right = relevant_edge_indices
                edges_left = executor.forward_steps[edge_idx_left]._edges
                edges_right = executor.forward_steps[edge_idx_right]._edges
                if edges_left is not None and edges_right is not None:
                    allowed_left = local_allowed_edges.get(edge_idx_left)
                    allowed_right = local_allowed_edges.get(edge_idx_right)

                    edge_left = executor.inputs.chain[edge_idx_left]
                    edge_right = executor.inputs.chain[edge_idx_right]
                    if isinstance(edge_left, ASTEdge) and isinstance(edge_right, ASTEdge):
                        sem_left = EdgeSemantics.from_edge(edge_left)
                        sem_right = EdgeSemantics.from_edge(edge_right)
                        if not sem_left.is_multihop and not sem_right.is_multihop:
                            pairs_left = _edge_pairs_cached(
                                edge_idx_left, sem_left, allowed_left
                            )
                            pairs_right = _edge_pairs_cached(
                                edge_idx_right, sem_right, allowed_right
                            )

                            if not domain_is_empty(start_nodes):
                                pairs_left = pairs_left[pairs_left["__from__"].isin(start_nodes)]
                            if not domain_is_empty(end_nodes):
                                pairs_right = pairs_right[pairs_right["__to__"].isin(end_nodes)]

                            start_vals = start_df[["__start__"] + label_cols].rename(
                                columns={"__start__": "__from__"}
                            ).drop_duplicates()
                            end_vals = end_df[["__current__"] + label_cols].rename(
                                columns={"__current__": "__to__"}
                            ).drop_duplicates()

                            left_pairs = pairs_left.merge(start_vals, on="__from__", how="inner")
                            right_pairs = pairs_right.merge(end_vals, on="__to__", how="inner")

                            left_pairs = left_pairs.rename(
                                columns={"__from__": "__start__", "__to__": "__mid__"}
                            )[["__start__", "__mid__"] + label_cols]
                            right_pairs = right_pairs.rename(
                                columns={"__from__": "__mid__", "__to__": "__current__"}
                            )[["__mid__", "__current__"] + label_cols]
                            pairs_left_rows_max = max(pairs_left_rows_max, len(left_pairs))
                            pairs_right_rows_max = max(pairs_right_rows_max, len(right_pairs))

                            if len(left_pairs) == 0 or len(right_pairs) == 0:
                                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                                continue

                            pair_est_value = len(left_pairs) * len(right_pairs)
                            domain_semijoin_pair_est_max = max(
                                domain_semijoin_pair_est_max, pair_est_value
                            )
                            semijoin_active = domain_semijoin_enabled
                            if not semijoin_active and domain_semijoin_auto:
                                if (
                                    domain_semijoin_pair_max is None
                                    or pair_est_value > domain_semijoin_pair_max
                                ):
                                    semijoin_active = True
                                    domain_semijoin_auto_used = True

                            if semijoin_active:
                                left_mid_labels = left_pairs[["__mid__"] + label_cols].drop_duplicates()
                                right_mid_labels = right_pairs[["__mid__"] + label_cols].drop_duplicates()
                                mid_values = left_mid_labels.merge(
                                    right_mid_labels, on=["__mid__"] + label_cols, how="inner"
                                )
                                mid_intersect_rows_max = max(
                                    mid_intersect_rows_max, len(mid_values)
                                )
                                if label_cols:
                                    mid_label_intersect_rows_max = max(
                                        mid_label_intersect_rows_max, len(mid_values)
                                    )
                                domain_semijoin_pairs_max = max(
                                    domain_semijoin_pairs_max, len(mid_values)
                                )
                                if len(mid_values) == 0:
                                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                                    continue

                                left_pairs = left_pairs.merge(
                                    mid_values, on=["__mid__"] + label_cols, how="inner"
                                )
                                right_pairs = right_pairs.merge(
                                    mid_values, on=["__mid__"] + label_cols, how="inner"
                                )

                                valid_starts = series_values(left_pairs["__start__"])
                                valid_ends = series_values(right_pairs["__current__"])

                                if start_node_idx in local_allowed_nodes:
                                    local_allowed_nodes[start_node_idx] = domain_intersect(
                                        local_allowed_nodes[start_node_idx],
                                        valid_starts,
                                    )
                                if end_node_idx in local_allowed_nodes:
                                    local_allowed_nodes[end_node_idx] = domain_intersect(
                                        local_allowed_nodes[end_node_idx],
                                        valid_ends,
                                    )

                                domain_semijoin_used = True
                                clause_count += len(group_entries)
                                for _, _, clause in group_entries:
                                    processed_clause_ids.add(id(clause))

                                current_state = PathState.from_mutable(
                                    local_allowed_nodes, local_allowed_edges, local_pruned_edges
                                )
                                current_state = executor.backward_propagate_constraints(
                                    current_state, start_node_idx, end_node_idx
                                )
                                local_allowed_nodes, local_allowed_edges = current_state.to_mutable()
                                local_pruned_edges.update(current_state.pruned_edges)
                                continue

            for _, _, clause in group_entries:
                processed_clause_ids.add(id(clause))

            state_df = start_df[["__start__"] + label_cols].rename(
                columns={"__start__": "__current__"}
            ).drop_duplicates()
            state_rows_max = max(state_rows_max, len(state_df))

            for edge_idx in relevant_edge_indices:
                edges_df = executor.forward_steps[edge_idx]._edges
                if edges_df is None or len(state_df) == 0:
                    break

                allowed_edges = local_allowed_edges.get(edge_idx)
                if allowed_edges is not None and edge_id_col and edge_id_col in edges_df.columns:
                    edges_df = edges_df[edges_df[edge_id_col].isin(allowed_edges)]

                edge_op = executor.inputs.chain[edge_idx]
                if not isinstance(edge_op, ASTEdge):
                    continue
                sem = EdgeSemantics.from_edge(edge_op)

                if sem.is_multihop:
                    edge_pairs = build_edge_pairs(edges_df, src_col, dst_col, sem)
                    all_reachable = [state_df.copy()]
                    current_state = state_df.copy()

                    for hop in range(1, sem.max_hops + 1):
                        next_state = edge_pairs.merge(
                            current_state, left_on="__from__", right_on="__current__", how="inner"
                        )[["__to__"] + label_cols].rename(columns={"__to__": "__current__"}).drop_duplicates()

                        if len(next_state) == 0:
                            break

                        if hop >= sem.min_hops:
                            all_reachable.append(next_state)
                        current_state = next_state
                        state_rows_max = max(state_rows_max, len(current_state))

                    if len(all_reachable) > 1:
                        state_df_concat = concat_frames(all_reachable[1:])
                        state_df = state_df_concat.drop_duplicates() if state_df_concat is not None else state_df.iloc[:0]
                    else:
                        state_df = state_df.iloc[:0]
                    state_rows_max = max(state_rows_max, len(state_df))
                else:
                    join_col, result_col = sem.join_cols(src_col, dst_col)
                    if sem.is_undirected:
                        next1 = edges_df.merge(
                            state_df, left_on=src_col, right_on="__current__", how="inner"
                        )[[dst_col] + label_cols].rename(columns={dst_col: "__current__"})
                        next2 = edges_df.merge(
                            state_df, left_on=dst_col, right_on="__current__", how="inner"
                        )[[src_col] + label_cols].rename(columns={src_col: "__current__"})
                        state_df_concat = concat_frames([next1, next2])
                        state_df = state_df_concat.drop_duplicates() if state_df_concat is not None else state_df.iloc[:0]
                    else:
                        state_df = edges_df.merge(
                            state_df, left_on=join_col, right_on="__current__", how="inner"
                        )[[result_col] + label_cols].rename(columns={result_col: "__current__"}).drop_duplicates()
                    state_rows_max = max(state_rows_max, len(state_df))

            state_df = state_df[state_df["__current__"].isin(end_nodes)]
            state_rows_max = max(state_rows_max, len(state_df))
            last_state_rows = len(state_df)

            if len(state_df) == 0:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                continue

            matches_df = state_df.merge(
                end_df, on=["__current__"] + label_cols, how="inner"
            )
            pairs_rows_max = max(pairs_rows_max, len(matches_df))
            if len(matches_df) == 0:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                continue

            valid_labels = matches_df[label_cols].drop_duplicates()
            valid_pairs_max = max(valid_pairs_max, len(valid_labels))
            valid_starts_df = start_df.merge(valid_labels, on=label_cols, how="inner")
            valid_ends_df = end_df.merge(valid_labels, on=label_cols, how="inner")
            if len(valid_starts_df) == 0 or len(valid_ends_df) == 0:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                continue

            valid_starts = series_values(valid_starts_df["__start__"])
            valid_ends = series_values(valid_ends_df["__current__"])

            if start_node_idx in local_allowed_nodes:
                local_allowed_nodes[start_node_idx] = domain_intersect(
                    local_allowed_nodes[start_node_idx], valid_starts
                )
            if end_node_idx in local_allowed_nodes:
                local_allowed_nodes[end_node_idx] = domain_intersect(
                    local_allowed_nodes[end_node_idx], valid_ends
                )

            value_mode_used = True
            multi_eq_value_used = True
            clause_count += len(group_entries)

            current_state = PathState.from_mutable(
                local_allowed_nodes, local_allowed_edges, local_pruned_edges
            )
            current_state = executor.backward_propagate_constraints(
                current_state, start_node_idx, end_node_idx
            )
            local_allowed_nodes, local_allowed_edges = current_state.to_mutable()
            local_pruned_edges.update(current_state.pruned_edges)

    remaining_clauses = [
        clause for clause in non_adjacent_clauses
        if id(clause) not in processed_clause_ids
    ]

    for clause in remaining_clauses:
        if id(clause) in processed_clause_ids:
            continue
        clause_count += 1
        left_alias = clause.left.alias
        right_alias = clause.right.alias
        left_binding = executor.inputs.alias_bindings[left_alias]
        right_binding = executor.inputs.alias_bindings[right_alias]

        if left_binding.step_index > right_binding.step_index:
            left_alias, right_alias = right_alias, left_alias
            left_binding, right_binding = right_binding, left_binding

        start_node_idx = left_binding.step_index
        end_node_idx = right_binding.step_index

        relevant_edge_indices = [
            idx for idx in edge_indices
            if start_node_idx < idx < end_node_idx
        ]
        endpoint_clause_count = endpoint_clause_counts.get((start_node_idx, end_node_idx), 1)

        start_nodes = local_allowed_nodes.get(start_node_idx)
        end_nodes = local_allowed_nodes.get(end_node_idx)
        if domain_is_empty(start_nodes) or domain_is_empty(end_nodes):
            continue

        left_col = clause.left.column
        right_col = clause.right.column
        if not node_id_col or nodes_df is None or node_id_col not in nodes_df.columns:
            continue

        left_values_df = None
        if left_col in nodes_df.columns:
            if node_id_col == left_col:
                left_values_df = nodes_df[nodes_df[node_id_col].isin(start_nodes)][[node_id_col]].drop_duplicates().copy()
                left_values_df.columns = ['__start__']
                left_values_df['__start_val__'] = left_values_df['__start__']
            else:
                left_values_df = nodes_df[nodes_df[node_id_col].isin(start_nodes)][[node_id_col, left_col]].drop_duplicates().rename(
                    columns={node_id_col: '__start__', left_col: '__start_val__'}
                )

        right_values_df = None
        if right_col in nodes_df.columns:
            if node_id_col == right_col:
                right_values_df = nodes_df[nodes_df[node_id_col].isin(end_nodes)][[node_id_col]].drop_duplicates().copy()
                right_values_df.columns = ['__current__']
                right_values_df['__end_val__'] = right_values_df['__current__']
            else:
                right_values_df = nodes_df[nodes_df[node_id_col].isin(end_nodes)][[node_id_col, right_col]].drop_duplicates().rename(
                    columns={node_id_col: '__current__', right_col: '__end_val__'}
                )

        left_values_domain = None
        right_values_domain = None
        if left_values_df is not None:
            left_values_df = left_values_df[left_values_df['__start_val__'].notna()]
        if right_values_df is not None:
            right_values_df = right_values_df[right_values_df['__end_val__'].notna()]

        if left_values_df is not None and len(left_values_df) > 0:
            left_values_domain = series_values(left_values_df['__start_val__'])
            left_value_count_max = max(left_value_count_max, len(left_values_domain))
        if right_values_df is not None and len(right_values_df) > 0:
            right_values_domain = series_values(right_values_df['__end_val__'])
            right_value_count_max = max(right_value_count_max, len(right_values_domain))

        auto_value_mode = non_adj_mode in {"auto", "auto_prefilter"}
        prefilter_enabled = non_adj_mode in {"prefilter", "value_prefilter", "auto_prefilter"}
        value_mode_requested = (
            non_adj_mode in {"value", "value_prefilter"} or auto_value_mode
        ) and clause.op in value_mode_ops

        if left_values_df is None or right_values_df is None:
            continue
        if len(left_values_df) == 0 or len(right_values_df) == 0:
            local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
            local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
            continue

        if prefilter_enabled and left_values_domain is not None and right_values_domain is not None:
            if clause.op == "==":
                allowed_values = domain_intersect(left_values_domain, right_values_domain)
                if domain_is_empty(allowed_values):
                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                    continue
                left_values_df = left_values_df[left_values_df['__start_val__'].isin(allowed_values)]
                right_values_df = right_values_df[right_values_df['__end_val__'].isin(allowed_values)]
                prefilter_used = True
            else:
                left_count = len(left_values_domain)
                right_count = len(right_values_domain)
                if left_count == 0 or right_count == 0:
                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                    continue
                if left_count == 1 and right_count == 1:
                    left_val = left_values_domain[0]
                    right_val = right_values_domain[0]
                    if not _scalar_clause(left_val, clause.op, right_val):
                        local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                        local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                        continue
                    prefilter_used = True
                    singleton_used = True
                elif left_count == 1:
                    left_val = left_values_domain[0]
                    right_values_df = _filter_values_df_by_const(
                        right_values_df, '__end_val__', clause.op, left_val, const_on_left=True
                    )
                    if len(right_values_df) == 0:
                        local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                        local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                        continue
                    prefilter_used = True
                    singleton_used = True
                elif right_count == 1:
                    right_val = right_values_domain[0]
                    left_values_df = _filter_values_df_by_const(
                        left_values_df, '__start_val__', clause.op, right_val, const_on_left=False
                    )
                    if len(left_values_df) == 0:
                        local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                        local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                        continue
                    prefilter_used = True
                    singleton_used = True

            if prefilter_used:
                start_nodes = series_values(left_values_df['__start__'])
                end_nodes = series_values(right_values_df['__current__'])
                cur_start_nodes = local_allowed_nodes.get(start_node_idx)
                cur_end_nodes = local_allowed_nodes.get(end_node_idx)
                local_allowed_nodes[start_node_idx] = (
                    domain_intersect(cur_start_nodes, start_nodes) if cur_start_nodes is not None else start_nodes
                )
                local_allowed_nodes[end_node_idx] = (
                    domain_intersect(cur_end_nodes, end_nodes) if cur_end_nodes is not None else end_nodes
                )
                left_values_domain = series_values(left_values_df['__start_val__']) if len(left_values_df) > 0 else left_values_domain
                right_values_domain = series_values(right_values_df['__end_val__']) if len(right_values_df) > 0 else right_values_domain

        if bounds_enabled and left_values_df is not None and right_values_df is not None and clause.op in {
            "<", "<=", ">", ">="
        }:
            left_vals = left_values_df['__start_val__']
            right_vals = right_values_df['__end_val__']
            if len(left_vals) > 0 and len(right_vals) > 0:
                left_min = left_vals.min()
                left_max = left_vals.max()
                right_min = right_vals.min()
                right_max = right_vals.max()
                if clause.op == "<":
                    left_mask = left_vals < right_max
                    right_mask = right_vals > left_min
                elif clause.op == "<=":
                    left_mask = left_vals <= right_max
                    right_mask = right_vals >= left_min
                elif clause.op == ">":
                    left_mask = left_vals > right_min
                    right_mask = right_vals < left_max
                else:  # ">="
                    left_mask = left_vals >= right_min
                    right_mask = right_vals <= left_max

                left_values_df = left_values_df[left_mask]
                right_values_df = right_values_df[right_mask]

                if len(left_values_df) == 0 or len(right_values_df) == 0:
                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                    continue

                start_nodes = series_values(left_values_df['__start__'])
                end_nodes = series_values(right_values_df['__current__'])
                cur_start_nodes = local_allowed_nodes.get(start_node_idx)
                cur_end_nodes = local_allowed_nodes.get(end_node_idx)
                local_allowed_nodes[start_node_idx] = (
                    domain_intersect(cur_start_nodes, start_nodes) if cur_start_nodes is not None else start_nodes
                )
                local_allowed_nodes[end_node_idx] = (
                    domain_intersect(cur_end_nodes, end_nodes) if cur_end_nodes is not None else end_nodes
                )
                left_values_domain = series_values(left_values_df['__start_val__']) if len(left_values_df) > 0 else left_values_domain
                right_values_domain = series_values(right_values_df['__end_val__']) if len(right_values_df) > 0 else right_values_domain
                bounds_used = True

        start_count = 0 if domain_is_empty(start_nodes) else len(start_nodes)
        end_count = 0 if domain_is_empty(end_nodes) else len(end_nodes)
        pair_est = start_count * end_count
        edge_pair_est = None
        if len(relevant_edge_indices) == 2:
            edge_left = executor.forward_steps[relevant_edge_indices[0]]._edges
            edge_right = executor.forward_steps[relevant_edge_indices[1]]._edges
            edge_left_count = (
                len(local_allowed_edges[relevant_edge_indices[0]])
                if local_allowed_edges.get(relevant_edge_indices[0]) is not None
                else (len(edge_left) if edge_left is not None else 0)
            )
            edge_right_count = (
                len(local_allowed_edges[relevant_edge_indices[1]])
                if local_allowed_edges.get(relevant_edge_indices[1]) is not None
                else (len(edge_right) if edge_right is not None else 0)
            )
            edge_pair_est = edge_left_count * edge_right_count

        if (
            auto_value_mode
            and value_mode_requested
            and domain_semijoin_pair_max is not None
            and endpoint_clause_count > 1
        ):
            value_pair_guard_pair_est_max = max(value_pair_guard_pair_est_max, pair_est)
            guard = pair_est > domain_semijoin_pair_max
            if edge_pair_est is not None:
                value_pair_guard_edge_est_max = max(value_pair_guard_edge_est_max, edge_pair_est)
                guard = guard or (edge_pair_est > domain_semijoin_pair_max)
            if guard:
                value_pair_guard_used = True
                value_mode_requested = False

        if (
            ineq_agg_enabled
            and auto_value_mode
            and clause.op in {"<", "<=", ">", ">="}
            and len(relevant_edge_indices) == 2
            and domain_semijoin_pair_max is not None
            and (pair_est > domain_semijoin_pair_max or (edge_pair_est is not None and edge_pair_est > domain_semijoin_pair_max))
        ):
            ineq_agg_pair_est_max = max(ineq_agg_pair_est_max, pair_est)
            edge_idx_left, edge_idx_right = relevant_edge_indices
            edges_left = executor.forward_steps[edge_idx_left]._edges
            edges_right = executor.forward_steps[edge_idx_right]._edges
            if edges_left is None or edges_right is None:
                continue

            allowed_left = local_allowed_edges.get(edge_idx_left)
            allowed_right = local_allowed_edges.get(edge_idx_right)

            edge_left = executor.inputs.chain[edge_idx_left]
            edge_right = executor.inputs.chain[edge_idx_right]
            if not isinstance(edge_left, ASTEdge) or not isinstance(edge_right, ASTEdge):
                continue
            sem_left = EdgeSemantics.from_edge(edge_left)
            sem_right = EdgeSemantics.from_edge(edge_right)
            if sem_left.is_multihop or sem_right.is_multihop:
                continue

            pairs_left = _edge_pairs_cached(
                edge_idx_left, sem_left, allowed_left
            ).drop_duplicates()
            pairs_right = _edge_pairs_cached(
                edge_idx_right, sem_right, allowed_right
            ).drop_duplicates()

            if not domain_is_empty(start_nodes):
                pairs_left = pairs_left[pairs_left["__from__"].isin(start_nodes)]
            if not domain_is_empty(end_nodes):
                pairs_right = pairs_right[pairs_right["__to__"].isin(end_nodes)]

            label_cols: List[str] = []
            eq_clause = None
            eq_entries = endpoint_eq_clauses.get((start_node_idx, end_node_idx), [])
            if len(eq_entries) == 1:
                eq_clause, eq_start_col, eq_end_col = eq_entries[0]
                if eq_start_col in nodes_df.columns and eq_end_col in nodes_df.columns:
                    label_cols = ["__label__"]
                else:
                    eq_clause = None
            if not label_cols:
                continue

            start_val_df = left_values_df.copy()
            end_val_df = right_values_df.copy()
            if label_cols:
                start_labels = nodes_df[nodes_df[node_id_col].isin(start_nodes)][
                    [node_id_col, eq_start_col]
                ].drop_duplicates()
                start_labels = start_labels.rename(
                    columns={node_id_col: "__start__", eq_start_col: "__label__"}
                )
                end_labels = nodes_df[nodes_df[node_id_col].isin(end_nodes)][
                    [node_id_col, eq_end_col]
                ].drop_duplicates()
                end_labels = end_labels.rename(
                    columns={node_id_col: "__current__", eq_end_col: "__label__"}
                )
                start_val_df = start_val_df.merge(start_labels, on="__start__", how="inner")
                end_val_df = end_val_df.merge(end_labels, on="__current__", how="inner")
                start_val_df = start_val_df[start_val_df["__label__"].notna()]
                end_val_df = end_val_df[end_val_df["__label__"].notna()]
                if len(start_val_df) == 0 or len(end_val_df) == 0:
                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                    continue

            left_edges = pairs_left.merge(
                start_val_df,
                left_on="__from__",
                right_on="__start__",
                how="inner",
            ).rename(columns={"__to__": "__mid__"})
            left_cols = ["__start__", "__mid__", "__start_val__"] + label_cols
            left_edges = left_edges[left_cols].drop_duplicates()

            right_edges = pairs_right.merge(
                end_val_df,
                left_on="__to__",
                right_on="__current__",
                how="inner",
            ).rename(columns={"__from__": "__mid__"})
            right_cols = ["__current__", "__mid__", "__end_val__"] + label_cols
            right_edges = right_edges[right_cols].drop_duplicates()

            if len(left_edges) == 0 or len(right_edges) == 0:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                continue

            group_cols = ["__mid__"] + label_cols
            if label_cols:
                left_labels = left_edges[["__mid__", "__label__"]].drop_duplicates()
                right_labels = right_edges[["__mid__", "__label__"]].drop_duplicates()
                allowed_labels = left_labels.merge(
                    right_labels, on=["__mid__", "__label__"], how="inner"
                )
                if len(allowed_labels) == 0:
                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                    continue
                left_edges = left_edges.merge(
                    allowed_labels, on=["__mid__", "__label__"], how="inner"
                )
                right_edges = right_edges.merge(
                    allowed_labels, on=["__mid__", "__label__"], how="inner"
                )
                if len(left_edges) == 0 or len(right_edges) == 0:
                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                    continue

            if clause.op in {"<", "<="}:
                left_bound = (
                    left_edges.groupby(group_cols)["__start_val__"]
                    .min()
                    .reset_index()
                    .rename(columns={"__start_val__": "__left_bound__"})
                )
                right_bound = (
                    right_edges.groupby(group_cols)["__end_val__"]
                    .max()
                    .reset_index()
                    .rename(columns={"__end_val__": "__right_bound__"})
                )
                allowed = left_bound.merge(right_bound, on=group_cols, how="inner")
                if clause.op == "<":
                    allowed = allowed[allowed["__left_bound__"] < allowed["__right_bound__"]]
                else:
                    allowed = allowed[allowed["__left_bound__"] <= allowed["__right_bound__"]]
                if len(allowed) == 0:
                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                    continue

                left_eval = left_edges.merge(
                    allowed[group_cols + ["__right_bound__"]], on=group_cols, how="inner"
                )
                if clause.op == "<":
                    left_eval = left_eval[left_eval["__start_val__"] < left_eval["__right_bound__"]]
                else:
                    left_eval = left_eval[left_eval["__start_val__"] <= left_eval["__right_bound__"]]

                right_eval = right_edges.merge(
                    allowed[group_cols + ["__left_bound__"]], on=group_cols, how="inner"
                )
                if clause.op == "<":
                    right_eval = right_eval[right_eval["__end_val__"] > right_eval["__left_bound__"]]
                else:
                    right_eval = right_eval[right_eval["__end_val__"] >= right_eval["__left_bound__"]]
            else:
                left_bound = (
                    left_edges.groupby(group_cols)["__start_val__"]
                    .max()
                    .reset_index()
                    .rename(columns={"__start_val__": "__left_bound__"})
                )
                right_bound = (
                    right_edges.groupby(group_cols)["__end_val__"]
                    .min()
                    .reset_index()
                    .rename(columns={"__end_val__": "__right_bound__"})
                )
                allowed = left_bound.merge(right_bound, on=group_cols, how="inner")
                if clause.op == ">":
                    allowed = allowed[allowed["__left_bound__"] > allowed["__right_bound__"]]
                else:
                    allowed = allowed[allowed["__left_bound__"] >= allowed["__right_bound__"]]
                if len(allowed) == 0:
                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                    continue

                left_eval = left_edges.merge(
                    allowed[group_cols + ["__right_bound__"]], on=group_cols, how="inner"
                )
                if clause.op == ">":
                    left_eval = left_eval[left_eval["__start_val__"] > left_eval["__right_bound__"]]
                else:
                    left_eval = left_eval[left_eval["__start_val__"] >= left_eval["__right_bound__"]]

                right_eval = right_edges.merge(
                    allowed[group_cols + ["__left_bound__"]], on=group_cols, how="inner"
                )
                if clause.op == ">":
                    right_eval = right_eval[right_eval["__end_val__"] < right_eval["__left_bound__"]]
                else:
                    right_eval = right_eval[right_eval["__end_val__"] <= right_eval["__left_bound__"]]

            if len(left_eval) == 0 or len(right_eval) == 0:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                continue

            valid_starts = series_values(left_eval["__start__"])
            valid_ends = series_values(right_eval["__current__"])
            cur_start_nodes = local_allowed_nodes.get(start_node_idx)
            cur_end_nodes = local_allowed_nodes.get(end_node_idx)
            local_allowed_nodes[start_node_idx] = (
                domain_intersect(cur_start_nodes, valid_starts)
                if cur_start_nodes is not None
                else valid_starts
            )
            local_allowed_nodes[end_node_idx] = (
                domain_intersect(cur_end_nodes, valid_ends)
                if cur_end_nodes is not None
                else valid_ends
            )

            ineq_agg_used = True
            if eq_clause is not None:
                processed_clause_ids.add(id(eq_clause))
            current_state = PathState.from_mutable(
                local_allowed_nodes, local_allowed_edges, local_pruned_edges
            )
            current_state = executor.backward_propagate_constraints(
                current_state, start_node_idx, end_node_idx
            )
            local_allowed_nodes, local_allowed_edges = current_state.to_mutable()
            local_pruned_edges.update(current_state.pruned_edges)
            continue

        value_cardinality = None
        if left_values_domain is not None or right_values_domain is not None:
            left_count = len(left_values_domain) if left_values_domain is not None else 0
            right_count = len(right_values_domain) if right_values_domain is not None else 0
            value_cardinality = max(left_count, right_count)
        value_mode_enabled = (
            value_mode_requested
            and left_values_df is not None
            and right_values_df is not None
            and len(left_values_df) > 0
            and len(right_values_df) > 0
            and (value_card_max is None or (value_cardinality is not None and value_cardinality <= value_card_max))
        )
        skip_value_auto_semijoin = (
            value_mode_enabled
            and domain_semijoin_auto
            and not domain_semijoin_enabled
            and endpoint_clause_count <= 1
        )

        if (
            (domain_semijoin_enabled or domain_semijoin_auto)
            and clause.op in {"==", "!=", "<", "<=", ">", ">="}
            and len(relevant_edge_indices) == 2
            and left_values_df is not None
            and right_values_df is not None
            and not skip_value_auto_semijoin
        ):
            edge_idx_left, edge_idx_right = relevant_edge_indices
            edges_left = executor.forward_steps[edge_idx_left]._edges
            edges_right = executor.forward_steps[edge_idx_right]._edges
            if edges_left is not None and edges_right is not None:
                allowed_left = local_allowed_edges.get(edge_idx_left)
                allowed_right = local_allowed_edges.get(edge_idx_right)

                edge_left = executor.inputs.chain[edge_idx_left]
                edge_right = executor.inputs.chain[edge_idx_right]
                if isinstance(edge_left, ASTEdge) and isinstance(edge_right, ASTEdge):
                    sem_left = EdgeSemantics.from_edge(edge_left)
                    sem_right = EdgeSemantics.from_edge(edge_right)
                    if not sem_left.is_multihop and not sem_right.is_multihop:
                        force_semijoin = (
                            (not domain_semijoin_enabled)
                            and domain_semijoin_auto
                            and non_adj_mode in {"auto", "auto_prefilter"}
                            and not value_mode_enabled
                            and clause.op in {"==", "!="}
                            and value_cardinality is not None
                            and value_card_max is not None
                            and value_cardinality > value_card_max
                        )
                        pair_est_approx = edge_pair_est if edge_pair_est is not None else pair_est
                        if pair_est_approx is not None:
                            domain_semijoin_pair_est_max = max(
                                domain_semijoin_pair_est_max, pair_est_approx
                            )

                        domain_semijoin_active = domain_semijoin_enabled
                        if not domain_semijoin_active and domain_semijoin_auto:
                            if (
                                force_semijoin
                                or domain_semijoin_pair_max is None
                                or (
                                    pair_est_approx is not None
                                    and pair_est_approx > domain_semijoin_pair_max
                                )
                            ):
                                domain_semijoin_active = True
                                domain_semijoin_auto_used = True

                        if not domain_semijoin_active:
                            pass
                        else:
                            pairs_left = _edge_pairs_cached(
                                edge_idx_left, sem_left, allowed_left
                            )
                            pairs_right = _edge_pairs_cached(
                                edge_idx_right, sem_right, allowed_right
                            )
                            if not domain_is_empty(start_nodes):
                                pairs_left = pairs_left[pairs_left["__from__"].isin(start_nodes)]
                            if not domain_is_empty(end_nodes):
                                pairs_right = pairs_right[pairs_right["__to__"].isin(end_nodes)]

                            start_vals = left_values_df[["__start__", "__start_val__"]].rename(
                                columns={"__start__": "__from__", "__start_val__": "__value__"}
                            ).drop_duplicates()
                            end_vals = right_values_df[["__current__", "__end_val__"]].rename(
                                columns={"__current__": "__to__", "__end_val__": "__value__"}
                            ).drop_duplicates()

                            left_pairs = pairs_left.merge(start_vals, on="__from__", how="inner")
                            right_pairs = pairs_right.merge(end_vals, on="__to__", how="inner")

                            left_pairs = left_pairs.rename(
                                columns={"__from__": "__start__", "__to__": "__mid__"}
                            )[["__start__", "__mid__", "__value__"]]
                            right_pairs = right_pairs.rename(
                                columns={"__from__": "__mid__", "__to__": "__current__"}
                            )[["__mid__", "__current__", "__value__"]]
                            pairs_left_rows_max = max(pairs_left_rows_max, len(left_pairs))
                            pairs_right_rows_max = max(pairs_right_rows_max, len(right_pairs))

                            if len(left_pairs) == 0 or len(right_pairs) == 0:
                                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                                continue

                            left_total = len(left_pairs)
                            right_total = len(right_pairs)
                            if clause.op in {"==", "!="}:
                                left_totals = left_pairs.groupby("__value__").size().reset_index()
                                left_totals.columns = ["__value__", "__left_count__"]
                                right_totals = right_pairs.groupby("__value__").size().reset_index()
                                right_totals.columns = ["__value__", "__right_count__"]
                                equal_counts = left_totals.merge(
                                    right_totals, on="__value__", how="inner"
                                )
                                equal_pairs = (
                                    equal_counts["__left_count__"] * equal_counts["__right_count__"]
                                ).sum()
                                try:
                                    equal_pairs_value = int(equal_pairs)
                                except Exception:
                                    equal_pairs_value = equal_pairs
                                if clause.op == "==":
                                    pair_est_value = equal_pairs_value
                                else:
                                    pair_est_value = left_total * right_total - equal_pairs_value
                            else:
                                pair_est_value = left_total * right_total
                            domain_semijoin_pair_est_max = max(
                                domain_semijoin_pair_est_max, pair_est_value
                            )

                            if clause.op == "==":
                                left_mid_values = left_pairs[["__mid__", "__value__"]].drop_duplicates()
                                right_mid_values = right_pairs[["__mid__", "__value__"]].drop_duplicates()
                                mid_values = left_mid_values.merge(
                                    right_mid_values, on=["__mid__", "__value__"], how="inner"
                                )
                                mid_intersect_rows_max = max(
                                    mid_intersect_rows_max, len(mid_values)
                                )
                                domain_semijoin_pairs_max = max(
                                    domain_semijoin_pairs_max, len(mid_values)
                                )
                                if len(mid_values) == 0:
                                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                                    continue

                                left_pairs = left_pairs.merge(
                                    mid_values, on=["__mid__", "__value__"], how="inner"
                                )
                                right_pairs = right_pairs.merge(
                                    mid_values, on=["__mid__", "__value__"], how="inner"
                                )

                                valid_starts = series_values(left_pairs["__start__"])
                                valid_ends = series_values(right_pairs["__current__"])
                            elif clause.op == "!=":
                                left_value_counts = (
                                    left_pairs[["__mid__", "__value__"]]
                                    .drop_duplicates()
                                    .groupby("__mid__")
                                    .size()
                                    .reset_index(name="__left_unique__")
                                )
                                right_value_counts = (
                                    right_pairs[["__mid__", "__value__"]]
                                    .drop_duplicates()
                                    .groupby("__mid__")
                                    .size()
                                    .reset_index(name="__right_unique__")
                                )

                                right_single = right_value_counts[
                                    right_value_counts["__right_unique__"] == 1
                                ]
                                right_only = right_pairs[["__mid__", "__value__"]].drop_duplicates()
                                right_only = right_only.merge(
                                    right_single, on="__mid__", how="inner"
                                )[["__mid__", "__value__"]].rename(
                                    columns={"__value__": "__right_only__"}
                                )

                                left_single = left_value_counts[
                                    left_value_counts["__left_unique__"] == 1
                                ]
                                left_only = left_pairs[["__mid__", "__value__"]].drop_duplicates()
                                left_only = left_only.merge(
                                    left_single, on="__mid__", how="inner"
                                )[["__mid__", "__value__"]].rename(
                                    columns={"__value__": "__left_only__"}
                                )

                                left_eval = left_pairs.merge(
                                    right_value_counts, on="__mid__", how="inner"
                                ).merge(
                                    right_only, on="__mid__", how="left"
                                )
                                left_mask = (
                                    (left_eval["__right_unique__"] > 1)
                                    | left_eval["__right_only__"].isna()
                                    | (left_eval["__right_only__"] != left_eval["__value__"])
                                )
                                left_eval = left_eval[left_mask]

                                right_eval = right_pairs.merge(
                                    left_value_counts, on="__mid__", how="inner"
                                ).merge(
                                    left_only, on="__mid__", how="left"
                                )
                                right_mask = (
                                    (right_eval["__left_unique__"] > 1)
                                    | right_eval["__left_only__"].isna()
                                    | (right_eval["__left_only__"] != right_eval["__value__"])
                                )
                                right_eval = right_eval[right_mask]

                                mid_intersect_rows_max = max(
                                    mid_intersect_rows_max,
                                    max(len(left_eval), len(right_eval)),
                                )
                                domain_semijoin_pairs_max = max(
                                    domain_semijoin_pairs_max,
                                    max(len(left_eval), len(right_eval)),
                                )
                                if len(left_eval) == 0 or len(right_eval) == 0:
                                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                                    continue

                                valid_starts = series_values(left_eval["__start__"])
                                valid_ends = series_values(right_eval["__current__"])
                            else:
                                left_min = (
                                    left_pairs.groupby("__mid__")["__value__"]
                                    .min()
                                    .reset_index()
                                    .rename(columns={"__value__": "__left_min__"})
                                )
                                left_max = (
                                    left_pairs.groupby("__mid__")["__value__"]
                                    .max()
                                    .reset_index()
                                    .rename(columns={"__value__": "__left_max__"})
                                )
                                right_min = (
                                    right_pairs.groupby("__mid__")["__value__"]
                                    .min()
                                    .reset_index()
                                    .rename(columns={"__value__": "__right_min__"})
                                )
                                right_max = (
                                    right_pairs.groupby("__mid__")["__value__"]
                                    .max()
                                    .reset_index()
                                    .rename(columns={"__value__": "__right_max__"})
                                )

                                if clause.op in {"<", "<="}:
                                    left_eval = left_pairs.merge(
                                        right_max, on="__mid__", how="inner"
                                    )
                                    if clause.op == "<":
                                        left_eval = left_eval[
                                            left_eval["__value__"] < left_eval["__right_max__"]
                                        ]
                                    else:
                                        left_eval = left_eval[
                                            left_eval["__value__"] <= left_eval["__right_max__"]
                                        ]
                                    right_eval = right_pairs.merge(
                                        left_min, on="__mid__", how="inner"
                                    )
                                    if clause.op == "<":
                                        right_eval = right_eval[
                                            right_eval["__value__"] > right_eval["__left_min__"]
                                        ]
                                    else:
                                        right_eval = right_eval[
                                            right_eval["__value__"] >= right_eval["__left_min__"]
                                        ]
                                else:
                                    left_eval = left_pairs.merge(
                                        right_min, on="__mid__", how="inner"
                                    )
                                    if clause.op == ">":
                                        left_eval = left_eval[
                                            left_eval["__value__"] > left_eval["__right_min__"]
                                        ]
                                    else:
                                        left_eval = left_eval[
                                            left_eval["__value__"] >= left_eval["__right_min__"]
                                        ]
                                    right_eval = right_pairs.merge(
                                        left_max, on="__mid__", how="inner"
                                    )
                                    if clause.op == ">":
                                        right_eval = right_eval[
                                            right_eval["__value__"] < right_eval["__left_max__"]
                                        ]
                                    else:
                                        right_eval = right_eval[
                                            right_eval["__value__"] <= right_eval["__left_max__"]
                                        ]

                                mid_intersect_rows_max = max(
                                    mid_intersect_rows_max,
                                    max(len(left_eval), len(right_eval)),
                                )
                                domain_semijoin_pairs_max = max(
                                    domain_semijoin_pairs_max,
                                    max(len(left_eval), len(right_eval)),
                                )
                                if len(left_eval) == 0 or len(right_eval) == 0:
                                    local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
                                    local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
                                    continue

                                valid_starts = series_values(left_eval["__start__"])
                                valid_ends = series_values(right_eval["__current__"])

                            if start_node_idx in local_allowed_nodes:
                                local_allowed_nodes[start_node_idx] = domain_intersect(
                                    local_allowed_nodes[start_node_idx],
                                    valid_starts,
                                )
                            if end_node_idx in local_allowed_nodes:
                                local_allowed_nodes[end_node_idx] = domain_intersect(
                                    local_allowed_nodes[end_node_idx],
                                    valid_ends,
                                )

                            domain_semijoin_used = True
                            current_state = PathState.from_mutable(
                                local_allowed_nodes, local_allowed_edges, local_pruned_edges
                            )
                            current_state = executor.backward_propagate_constraints(
                                current_state, start_node_idx, end_node_idx
                            )
                            local_allowed_nodes, local_allowed_edges = current_state.to_mutable()
                            local_pruned_edges.update(current_state.pruned_edges)
                            continue

        state_label_col = "__start_val__" if value_mode_enabled else "__start__"
        if value_mode_enabled:
            value_mode_used = True

        # State table propagation: (current_node, start_label) pairs
        if left_values_df is not None and len(left_values_df) > 0:
            if value_mode_enabled:
                state_df = left_values_df[['__start__', state_label_col]].rename(
                    columns={'__start__': '__current__'}
                ).drop_duplicates()
            else:
                state_df = left_values_df[['__start__']].copy()
                state_df['__current__'] = state_df['__start__']
        else:
            state_df = df_cons(nodes_df, {'__current__': [], state_label_col: []})
        state_rows_max = max(state_rows_max, len(state_df))

        for edge_idx in relevant_edge_indices:
            edges_df = executor.forward_steps[edge_idx]._edges
            if edges_df is None or len(state_df) == 0:
                break

            allowed_edges = local_allowed_edges.get(edge_idx)
            if allowed_edges is not None and edge_id_col and edge_id_col in edges_df.columns:
                edges_df = edges_df[edges_df[edge_id_col].isin(allowed_edges)]

            edge_op = executor.inputs.chain[edge_idx]
            if not isinstance(edge_op, ASTEdge):
                continue
            sem = EdgeSemantics.from_edge(edge_op)

            if sem.is_multihop:
                edge_pairs = build_edge_pairs(edges_df, src_col, dst_col, sem)
                all_reachable = [state_df.copy()]
                current_state = state_df.copy()

                for hop in range(1, sem.max_hops + 1):
                    next_state = edge_pairs.merge(
                        current_state, left_on='__from__', right_on='__current__', how='inner'
                    )[['__to__', state_label_col]].rename(columns={'__to__': '__current__'}).drop_duplicates()

                    if len(next_state) == 0:
                        break

                    if hop >= sem.min_hops:
                        all_reachable.append(next_state)
                    current_state = next_state
                    state_rows_max = max(state_rows_max, len(current_state))

                if len(all_reachable) > 1:
                    state_df_concat = concat_frames(all_reachable[1:])
                    state_df = state_df_concat.drop_duplicates() if state_df_concat is not None else state_df.iloc[:0]
                else:
                    state_df = state_df.iloc[:0]
                state_rows_max = max(state_rows_max, len(state_df))
            else:
                join_col, result_col = sem.join_cols(src_col, dst_col)
                if sem.is_undirected:
                    next1 = edges_df.merge(
                        state_df, left_on=src_col, right_on='__current__', how='inner'
                    )[[dst_col, state_label_col]].rename(columns={dst_col: '__current__'})
                    next2 = edges_df.merge(
                        state_df, left_on=dst_col, right_on='__current__', how='inner'
                    )[[src_col, state_label_col]].rename(columns={src_col: '__current__'})
                    state_df_concat = concat_frames([next1, next2])
                    state_df = state_df_concat.drop_duplicates() if state_df_concat is not None else state_df.iloc[:0]
                else:
                    state_df = edges_df.merge(
                        state_df, left_on=join_col, right_on='__current__', how='inner'
                    )[[result_col, state_label_col]].rename(columns={result_col: '__current__'}).drop_duplicates()
                state_rows_max = max(state_rows_max, len(state_df))

        state_df = state_df[state_df['__current__'].isin(end_nodes)]
        state_rows_max = max(state_rows_max, len(state_df))
        last_state_rows = len(state_df)

        if len(state_df) == 0:
            if start_node_idx in local_allowed_nodes:
                local_allowed_nodes[start_node_idx] = domain_empty(nodes_df)
            if end_node_idx in local_allowed_nodes:
                local_allowed_nodes[end_node_idx] = domain_empty(nodes_df)
            continue

        if left_values_df is None or right_values_df is None:
            continue

        if value_mode_enabled:
            pairs_df = state_df.merge(right_values_df, on='__current__', how='inner')
            pairs_rows_max = max(pairs_rows_max, len(pairs_df))
            mask = evaluate_clause(pairs_df[state_label_col], clause.op, pairs_df['__end_val__'], null_safe=True)
            valid_pairs = pairs_df[mask]
            valid_pairs_max = max(valid_pairs_max, len(valid_pairs))
            valid_start_values = series_values(valid_pairs[state_label_col])
            valid_starts = series_values(
                left_values_df[left_values_df['__start_val__'].isin(valid_start_values)]['__start__']
            )
            valid_ends = series_values(valid_pairs['__current__'])
        else:
            pairs_df = state_df.merge(left_values_df, on='__start__', how='inner')
            pairs_df = pairs_df.merge(right_values_df, on='__current__', how='inner')
            pairs_rows_max = max(pairs_rows_max, len(pairs_df))

            mask = evaluate_clause(pairs_df['__start_val__'], clause.op, pairs_df['__end_val__'], null_safe=True)
            valid_pairs = pairs_df[mask]
            valid_pairs_max = max(valid_pairs_max, len(valid_pairs))
            valid_starts = series_values(valid_pairs['__start__'])
            valid_ends = series_values(valid_pairs['__current__'])

        if start_node_idx in local_allowed_nodes:
            local_allowed_nodes[start_node_idx] = domain_intersect(
                local_allowed_nodes[start_node_idx],
                valid_starts,
            )
        if end_node_idx in local_allowed_nodes:
            local_allowed_nodes[end_node_idx] = domain_intersect(
                local_allowed_nodes[end_node_idx],
                valid_ends,
            )

        current_state = PathState.from_mutable(
            local_allowed_nodes, local_allowed_edges, local_pruned_edges
        )
        current_state = executor.backward_propagate_constraints(
            current_state, start_node_idx, end_node_idx
        )
        local_allowed_nodes, local_allowed_edges = current_state.to_mutable()
        local_pruned_edges.update(current_state.pruned_edges)

    if span is not None and otel_detail_enabled():
        span.set_attribute("gfql.non_adjacent.clause_count", clause_count)
        span.set_attribute("gfql.non_adjacent.state_rows_max", state_rows_max)
        span.set_attribute("gfql.non_adjacent.state_rows_final", last_state_rows)
        span.set_attribute("gfql.non_adjacent.pairs_rows_max", pairs_rows_max)
        span.set_attribute("gfql.non_adjacent.valid_pairs_max", valid_pairs_max)
        span.set_attribute("gfql.non_adjacent.value_mode_used", value_mode_used)
        span.set_attribute("gfql.non_adjacent.multi_eq_value_used", multi_eq_value_used)
        span.set_attribute("gfql.non_adjacent.multi_eq_label_card_max", multi_eq_label_card_max)
        span.set_attribute("gfql.non_adjacent.vector_used", vector_used)
        span.set_attribute("gfql.non_adjacent.vector_label_card_max", vector_label_card_max)
        span.set_attribute("gfql.non_adjacent.vector_candidate_pairs_max", vector_candidate_pairs_max)
        span.set_attribute("gfql.non_adjacent.vector_path_pairs_max", vector_path_pairs_max)
        span.set_attribute("gfql.non_adjacent.vector_pair_est_max", vector_pair_est_max)
        if vector_pair_max is not None:
            span.set_attribute("gfql.non_adjacent.vector_pair_max", vector_pair_max)
        span.set_attribute("gfql.non_adjacent.domain_semijoin_used", domain_semijoin_used)
        span.set_attribute("gfql.non_adjacent.domain_semijoin_pairs_max", domain_semijoin_pairs_max)
        span.set_attribute("gfql.non_adjacent.domain_semijoin_enabled", domain_semijoin_enabled)
        span.set_attribute("gfql.non_adjacent.domain_semijoin_auto_used", domain_semijoin_auto_used)
        span.set_attribute("gfql.non_adjacent.domain_semijoin_pair_est_max", domain_semijoin_pair_est_max)
        if domain_semijoin_pair_max is not None:
            span.set_attribute("gfql.non_adjacent.domain_semijoin_pair_max", domain_semijoin_pair_max)
        span.set_attribute("gfql.non_adjacent.domain_semijoin_auto", domain_semijoin_auto)
        span.set_attribute("gfql.non_adjacent.prefilter_used", prefilter_used)
        span.set_attribute("gfql.non_adjacent.singleton_used", singleton_used)
        span.set_attribute("gfql.non_adjacent.bounds_used", bounds_used)
        span.set_attribute("gfql.non_adjacent.order_used", order_used)
        span.set_attribute("gfql.non_adjacent.value_pair_guard_used", value_pair_guard_used)
        span.set_attribute("gfql.non_adjacent.value_pair_guard_pair_est_max", value_pair_guard_pair_est_max)
        span.set_attribute("gfql.non_adjacent.value_pair_guard_edge_est_max", value_pair_guard_edge_est_max)
        span.set_attribute("gfql.non_adjacent.ineq_agg_used", ineq_agg_used)
        span.set_attribute("gfql.non_adjacent.ineq_agg_pair_est_max", ineq_agg_pair_est_max)
        span.set_attribute("gfql.non_adjacent.left_values_max", left_value_count_max)
        span.set_attribute("gfql.non_adjacent.right_values_max", right_value_count_max)
        span.set_attribute("gfql.non_adjacent.mid_intersect_rows_max", mid_intersect_rows_max)
        span.set_attribute(
            "gfql.non_adjacent.mid_label_intersect_rows_max", mid_label_intersect_rows_max
        )
        span.set_attribute("gfql.non_adjacent.pairs_left_rows_max", pairs_left_rows_max)
        span.set_attribute("gfql.non_adjacent.pairs_right_rows_max", pairs_right_rows_max)
        if value_card_max is not None:
            span.set_attribute("gfql.non_adjacent.value_card_max", value_card_max)
        span.set_attribute("gfql.non_adjacent.value_ops", ",".join(sorted(value_mode_ops)))
        span.set_attribute("gfql.non_adjacent.mode", non_adj_mode)
        span.set_attribute("gfql.non_adjacent.order", non_adj_order or "none")
        span.set_attribute("gfql.non_adjacent.bounds_enabled", bounds_enabled)

    return PathState.from_mutable(local_allowed_nodes, local_allowed_edges, local_pruned_edges)


def apply_edge_where_post_prune(
    executor: "DFSamePathExecutor",
    state: PathState,
) -> PathState:
    """Apply WHERE on edge columns by enumerating paths.

    Args:
        executor: The executor instance with chain metadata and state
        state: Current PathState with allowed_nodes/allowed_edges

    Returns:
        New PathState with constraints applied
    """
    if not executor.inputs.where:
        return state

    edge_semijoin_raw = os.environ.get("GRAPHISTRY_EDGE_WHERE_SEMIJOIN", "").strip().lower()
    edge_semijoin_auto_raw = os.environ.get("GRAPHISTRY_EDGE_WHERE_SEMIJOIN_AUTO", "").strip().lower()
    non_adj_mode = os.environ.get("GRAPHISTRY_NON_ADJ_WHERE_MODE", "auto").strip().lower()
    edge_semijoin_pair_max_raw = os.environ.get("GRAPHISTRY_EDGE_WHERE_SEMIJOIN_PAIR_MAX", "").strip()
    edge_semijoin_enabled = edge_semijoin_raw in {"1", "true", "yes", "on"}
    edge_semijoin_auto = edge_semijoin_auto_raw in {"1", "true", "yes", "on"}
    if not edge_semijoin_auto_raw and non_adj_mode in {"auto", "auto_prefilter"}:
        edge_semijoin_auto = True
    try:
        edge_semijoin_pair_max = (
            int(edge_semijoin_pair_max_raw)
            if edge_semijoin_pair_max_raw
            else 200000
        )
    except ValueError:
        edge_semijoin_pair_max = 200000
    if edge_semijoin_pair_max is not None and edge_semijoin_pair_max <= 0:
        edge_semijoin_pair_max = None

    edge_clauses = [
        clause for clause in executor.inputs.where
        if (b1 := executor.inputs.alias_bindings.get(clause.left.alias))
        and (b2 := executor.inputs.alias_bindings.get(clause.right.alias))
        and (b1.kind == "edge" or b2.kind == "edge")
    ]
    if not edge_clauses:
        return state

    src_col = executor._source_column
    dst_col = executor._destination_column
    node_id_col = executor._node_column
    if not src_col or not dst_col or not node_id_col:
        return state

    node_indices = executor.meta.node_indices
    edge_indices = executor.meta.edge_indices

    # Work on local copies (internal immutability pattern)
    local_allowed_nodes: Dict[int, Any] = dict(state.allowed_nodes)
    # Preserve existing pruned_edges from input state
    pruned_edges: Dict[int, Any] = dict(state.pruned_edges)
    edge_overrides: Dict[int, DataFrameT] = {}

    seed_nodes = local_allowed_nodes.get(node_indices[0])
    if domain_is_empty(seed_nodes):
        return state

    nodes_df_template = executor.inputs.graph._nodes
    if nodes_df_template is None:
        return state

    edge_positions = {edge_idx: pos for pos, edge_idx in enumerate(edge_indices)}
    fast_path_possible = (
        (edge_semijoin_enabled or edge_semijoin_auto)
        and len(edge_indices) == 2
        and len(edge_clauses) == 1
    )
    fast_path_full_cover = fast_path_possible
    fast_path_left_pairs = None
    fast_path_right_pairs = None
    fast_path_left_edge_idx = None
    fast_path_right_edge_idx = None
    fast_path_sem_left = None
    fast_path_sem_right = None

    def _filter_edges_from_node_pairs(
        edges_df: DataFrameT,
        sem: EdgeSemantics,
        pairs_df: DataFrameT,
        left_label: str,
        right_label: str,
    ) -> DataFrameT:
        if sem.is_undirected:
            fwd = edges_df.merge(
                pairs_df.rename(columns={left_label: src_col, right_label: dst_col}),
                on=[src_col, dst_col],
                how="inner",
            )
            rev = edges_df.merge(
                pairs_df.rename(columns={left_label: dst_col, right_label: src_col}),
                on=[src_col, dst_col],
                how="inner",
            )
            edges_concat = concat_frames([fwd, rev])
            return (
                edges_concat.drop_duplicates(subset=[src_col, dst_col])
                if edges_concat is not None
                else edges_df.iloc[:0]
            )
        start_endpoint, end_endpoint = sem.endpoint_cols(src_col, dst_col)
        return edges_df.merge(
            pairs_df.rename(columns={left_label: start_endpoint, right_label: end_endpoint}),
            on=[src_col, dst_col],
            how="inner",
        )

    if edge_semijoin_enabled or edge_semijoin_auto:
        for clause in edge_clauses:
            left_binding = executor.inputs.alias_bindings.get(clause.left.alias)
            right_binding = executor.inputs.alias_bindings.get(clause.right.alias)
            if not left_binding or not right_binding:
                fast_path_full_cover = False
                continue
            if left_binding.kind != "edge" or right_binding.kind != "edge":
                fast_path_full_cover = False
                continue

            left_edge_idx = left_binding.step_index
            right_edge_idx = right_binding.step_index
            left_pos = edge_positions.get(left_edge_idx)
            right_pos = edge_positions.get(right_edge_idx)
            if left_pos is None or right_pos is None:
                fast_path_full_cover = False
                continue
            if abs(left_pos - right_pos) != 1:
                fast_path_full_cover = False
                continue

            op = clause.op
            if left_pos > right_pos:
                left_edge_idx, right_edge_idx = right_edge_idx, left_edge_idx
                left_pos, right_pos = right_pos, left_pos
                op = {
                    "<": ">",
                    "<=": ">=",
                    ">": "<",
                    ">=": "<=",
                    "==": "==",
                    "!=": "!=",
                }.get(op, op)

            if op not in {"==", "!=", "<", "<=", ">", ">="}:
                fast_path_full_cover = False
                continue

            left_node_idx = node_indices[left_pos]
            mid_node_idx = node_indices[left_pos + 1]
            right_node_idx = node_indices[left_pos + 2]

            left_value_col = clause.left.column
            right_value_col = clause.right.column

            left_edges = edge_overrides.get(left_edge_idx) or executor.edges_df_for_step(
                left_edge_idx, state
            )
            right_edges = edge_overrides.get(right_edge_idx) or executor.edges_df_for_step(
                right_edge_idx, state
            )
            if left_edges is None or right_edges is None or len(left_edges) == 0 or len(right_edges) == 0:
                fast_path_full_cover = False
                continue
            if left_value_col not in left_edges.columns or right_value_col not in right_edges.columns:
                fast_path_full_cover = False
                continue

            left_edge_op = executor.inputs.chain[left_edge_idx]
            right_edge_op = executor.inputs.chain[right_edge_idx]
            if not isinstance(left_edge_op, ASTEdge) or not isinstance(right_edge_op, ASTEdge):
                fast_path_full_cover = False
                continue
            sem_left = EdgeSemantics.from_edge(left_edge_op)
            sem_right = EdgeSemantics.from_edge(right_edge_op)
            if sem_left.is_multihop or sem_right.is_multihop:
                fast_path_full_cover = False
                continue

            def _edge_pairs_with_value(
                edges_df: DataFrameT,
                sem: EdgeSemantics,
                left_label: str,
                right_label: str,
                value_col: str,
                value_label: str,
            ) -> DataFrameT:
                if sem.is_undirected:
                    fwd = edges_df[[src_col, dst_col, value_col]].rename(
                        columns={src_col: left_label, dst_col: right_label, value_col: value_label}
                    )
                    rev = edges_df[[dst_col, src_col, value_col]].rename(
                        columns={dst_col: left_label, src_col: right_label, value_col: value_label}
                    )
                    pairs = concat_frames([fwd, rev])
                    return pairs.drop_duplicates() if pairs is not None else fwd.iloc[:0]
                join_col, result_col = sem.join_cols(src_col, dst_col)
                return edges_df[[join_col, result_col, value_col]].rename(
                    columns={join_col: left_label, result_col: right_label, value_col: value_label}
                )

            left_pairs = _edge_pairs_with_value(
                left_edges, sem_left, "__left__", "__mid__", left_value_col, "__left_val__"
            ).drop_duplicates()
            right_pairs = _edge_pairs_with_value(
                right_edges, sem_right, "__mid__", "__right__", right_value_col, "__right_val__"
            ).drop_duplicates()

            left_nodes = local_allowed_nodes.get(left_node_idx)
            mid_nodes = local_allowed_nodes.get(mid_node_idx)
            right_nodes = local_allowed_nodes.get(right_node_idx)
            if not domain_is_empty(left_nodes):
                left_pairs = left_pairs[left_pairs["__left__"].isin(left_nodes)]
            if not domain_is_empty(mid_nodes):
                left_pairs = left_pairs[left_pairs["__mid__"].isin(mid_nodes)]
                right_pairs = right_pairs[right_pairs["__mid__"].isin(mid_nodes)]
            if not domain_is_empty(right_nodes):
                right_pairs = right_pairs[right_pairs["__right__"].isin(right_nodes)]

            left_pairs = left_pairs[left_pairs["__left_val__"].notna()]
            right_pairs = right_pairs[right_pairs["__right_val__"].notna()]

            if len(left_pairs) == 0 or len(right_pairs) == 0:
                local_allowed_nodes[left_node_idx] = domain_empty(nodes_df_template)
                local_allowed_nodes[right_node_idx] = domain_empty(nodes_df_template)
                continue

            left_total = len(left_pairs)
            right_total = len(right_pairs)
            if op in {"==", "!="}:
                left_counts = left_pairs.groupby("__left_val__").size().reset_index()
                left_counts.columns = ["__value__", "__left_count__"]
                right_counts = right_pairs.groupby("__right_val__").size().reset_index()
                right_counts.columns = ["__value__", "__right_count__"]
                equal_counts = left_counts.merge(right_counts, on="__value__", how="inner")
                equal_pairs = (equal_counts["__left_count__"] * equal_counts["__right_count__"]).sum()
                try:
                    equal_pairs_value = int(equal_pairs)
                except Exception:
                    equal_pairs_value = equal_pairs
                if op == "==":
                    pair_est_value = equal_pairs_value
                else:
                    pair_est_value = left_total * right_total - equal_pairs_value
            else:
                pair_est_value = left_total * right_total

            semijoin_active = edge_semijoin_enabled
            if not semijoin_active and edge_semijoin_auto:
                if edge_semijoin_pair_max is None or pair_est_value > edge_semijoin_pair_max:
                    semijoin_active = True

            if not semijoin_active:
                fast_path_full_cover = False
                continue

            if op == "==":
                mid_values = left_pairs.rename(
                    columns={"__left_val__": "__value__"}
                )[["__mid__", "__value__"]].drop_duplicates()
                mid_values = mid_values.merge(
                    right_pairs.rename(columns={"__right_val__": "__value__"})[["__mid__", "__value__"]]
                    .drop_duplicates(),
                    on=["__mid__", "__value__"],
                    how="inner",
                )
                if len(mid_values) == 0:
                    local_allowed_nodes[left_node_idx] = domain_empty(nodes_df_template)
                    local_allowed_nodes[right_node_idx] = domain_empty(nodes_df_template)
                    continue
                left_pairs = left_pairs.merge(
                    mid_values.rename(columns={"__value__": "__left_val__"}),
                    on=["__mid__", "__left_val__"],
                    how="inner",
                )
                right_pairs = right_pairs.merge(
                    mid_values.rename(columns={"__value__": "__right_val__"}),
                    on=["__mid__", "__right_val__"],
                    how="inner",
                )
            elif op == "!=":
                left_unique = (
                    left_pairs[["__mid__", "__left_val__"]]
                    .drop_duplicates()
                    .groupby("__mid__")
                    .size()
                    .reset_index(name="__left_unique__")
                )
                right_unique = (
                    right_pairs[["__mid__", "__right_val__"]]
                    .drop_duplicates()
                    .groupby("__mid__")
                    .size()
                    .reset_index(name="__right_unique__")
                )

                right_single = right_unique[right_unique["__right_unique__"] == 1]
                right_only = right_pairs[["__mid__", "__right_val__"]].drop_duplicates()
                right_only = right_only.merge(
                    right_single, on="__mid__", how="inner"
                )[["__mid__", "__right_val__"]]

                left_single = left_unique[left_unique["__left_unique__"] == 1]
                left_only = left_pairs[["__mid__", "__left_val__"]].drop_duplicates()
                left_only = left_only.merge(
                    left_single, on="__mid__", how="inner"
                )[["__mid__", "__left_val__"]]

                left_eval = left_pairs.merge(
                    right_unique, on="__mid__", how="inner"
                ).merge(
                    right_only.rename(columns={"__right_val__": "__right_only__"}),
                    on="__mid__",
                    how="left",
                )
                left_mask = (
                    (left_eval["__right_unique__"] > 1)
                    | left_eval["__right_only__"].isna()
                    | (left_eval["__right_only__"] != left_eval["__left_val__"])
                )
                left_pairs = left_eval[left_mask][["__left__", "__mid__", "__left_val__"]]

                right_eval = right_pairs.merge(
                    left_unique, on="__mid__", how="inner"
                ).merge(
                    left_only.rename(columns={"__left_val__": "__left_only__"}),
                    on="__mid__",
                    how="left",
                )
                right_mask = (
                    (right_eval["__left_unique__"] > 1)
                    | right_eval["__left_only__"].isna()
                    | (right_eval["__left_only__"] != right_eval["__right_val__"])
                )
                right_pairs = right_eval[right_mask][["__mid__", "__right__", "__right_val__"]]
            else:
                try:
                    left_min = (
                        left_pairs.groupby("__mid__")["__left_val__"]
                        .min()
                        .reset_index(name="__left_min__")
                    )
                    left_max = (
                        left_pairs.groupby("__mid__")["__left_val__"]
                        .max()
                        .reset_index(name="__left_max__")
                    )
                    right_min = (
                        right_pairs.groupby("__mid__")["__right_val__"]
                        .min()
                        .reset_index(name="__right_min__")
                    )
                    right_max = (
                        right_pairs.groupby("__mid__")["__right_val__"]
                        .max()
                        .reset_index(name="__right_max__")
                    )
                except Exception:
                    continue

                if op in {"<", "<="}:
                    left_eval = left_pairs.merge(right_max, on="__mid__", how="inner")
                    if op == "<":
                        left_eval = left_eval[left_eval["__left_val__"] < left_eval["__right_max__"]]
                    else:
                        left_eval = left_eval[left_eval["__left_val__"] <= left_eval["__right_max__"]]
                    right_eval = right_pairs.merge(left_min, on="__mid__", how="inner")
                    if op == "<":
                        right_eval = right_eval[right_eval["__right_val__"] > right_eval["__left_min__"]]
                    else:
                        right_eval = right_eval[right_eval["__right_val__"] >= right_eval["__left_min__"]]
                else:
                    left_eval = left_pairs.merge(right_min, on="__mid__", how="inner")
                    if op == ">":
                        left_eval = left_eval[left_eval["__left_val__"] > left_eval["__right_min__"]]
                    else:
                        left_eval = left_eval[left_eval["__left_val__"] >= left_eval["__right_min__"]]
                    right_eval = right_pairs.merge(left_max, on="__mid__", how="inner")
                    if op == ">":
                        right_eval = right_eval[right_eval["__right_val__"] < right_eval["__left_max__"]]
                    else:
                        right_eval = right_eval[right_eval["__right_val__"] <= right_eval["__left_max__"]]

                left_pairs = left_eval[["__left__", "__mid__", "__left_val__"]]
                right_pairs = right_eval[["__mid__", "__right__", "__right_val__"]]

            if len(left_pairs) == 0 or len(right_pairs) == 0:
                local_allowed_nodes[left_node_idx] = domain_empty(nodes_df_template)
                local_allowed_nodes[right_node_idx] = domain_empty(nodes_df_template)
                continue

            if fast_path_possible:
                fast_path_left_pairs = left_pairs
                fast_path_right_pairs = right_pairs
                fast_path_left_edge_idx = left_edge_idx
                fast_path_right_edge_idx = right_edge_idx
                fast_path_sem_left = sem_left
                fast_path_sem_right = sem_right

            valid_left_nodes = series_values(left_pairs["__left__"])
            valid_mid_left = series_values(left_pairs["__mid__"])
            valid_right_nodes = series_values(right_pairs["__right__"])
            valid_mid_right = series_values(right_pairs["__mid__"])
            valid_mid_nodes = domain_intersect(valid_mid_left, valid_mid_right)

            if left_node_idx in local_allowed_nodes:
                local_allowed_nodes[left_node_idx] = domain_intersect(
                    local_allowed_nodes[left_node_idx], valid_left_nodes
                )
            if right_node_idx in local_allowed_nodes:
                local_allowed_nodes[right_node_idx] = domain_intersect(
                    local_allowed_nodes[right_node_idx], valid_right_nodes
                )
            if mid_node_idx in local_allowed_nodes:
                local_allowed_nodes[mid_node_idx] = domain_intersect(
                    local_allowed_nodes[mid_node_idx], valid_mid_nodes
                )

            def _filter_edges_from_pairs(
                edges_df: DataFrameT,
                sem: EdgeSemantics,
                pairs_df: DataFrameT,
                left_label: str,
                right_label: str,
                value_label: str,
                value_col: str,
            ) -> DataFrameT:
                if sem.is_undirected:
                    fwd = edges_df.merge(
                        pairs_df.rename(
                            columns={
                                left_label: src_col,
                                right_label: dst_col,
                                value_label: value_col,
                            }
                        ),
                        on=[src_col, dst_col, value_col],
                        how="inner",
                    )
                    rev = edges_df.merge(
                        pairs_df.rename(
                            columns={
                                left_label: dst_col,
                                right_label: src_col,
                                value_label: value_col,
                            }
                        ),
                        on=[src_col, dst_col, value_col],
                        how="inner",
                    )
                    edges_concat = concat_frames([fwd, rev])
                    return edges_concat.drop_duplicates() if edges_concat is not None else edges_df.iloc[:0]
                join_col, result_col = sem.join_cols(src_col, dst_col)
                return edges_df.merge(
                    pairs_df.rename(
                        columns={
                            left_label: join_col,
                            right_label: result_col,
                            value_label: value_col,
                        }
                    ),
                    on=[join_col, result_col, value_col],
                    how="inner",
                )

            left_edges_filtered = _filter_edges_from_pairs(
                left_edges, sem_left, left_pairs, "__left__", "__mid__", "__left_val__", left_value_col
            )
            right_edges_filtered = _filter_edges_from_pairs(
                right_edges, sem_right, right_pairs, "__mid__", "__right__", "__right_val__", right_value_col
            )
            edge_overrides[left_edge_idx] = left_edges_filtered
            edge_overrides[right_edge_idx] = right_edges_filtered

    if fast_path_full_cover:
        # Fast path: 2-hop single edge-edge clause, prune by endpoints (baseline semantics).
        if any(domain_is_empty(local_allowed_nodes.get(idx)) for idx in node_indices):
            for idx in node_indices:
                local_allowed_nodes[idx] = domain_empty(nodes_df_template)
            return PathState.from_mutable(local_allowed_nodes, {})
        if (
            fast_path_left_pairs is None
            or fast_path_right_pairs is None
            or fast_path_left_edge_idx is None
            or fast_path_right_edge_idx is None
            or fast_path_sem_left is None
            or fast_path_sem_right is None
        ):
            fast_path_full_cover = False
        else:
            left_pairs = fast_path_left_pairs[["__left__", "__mid__"]].drop_duplicates()
            right_pairs = fast_path_right_pairs[["__mid__", "__right__"]].drop_duplicates()
            left_edges_df = executor.edges_df_for_step(fast_path_left_edge_idx, state)
            right_edges_df = executor.edges_df_for_step(fast_path_right_edge_idx, state)
            if left_edges_df is not None:
                pruned_edges[fast_path_left_edge_idx] = _filter_edges_from_node_pairs(
                    left_edges_df, fast_path_sem_left, left_pairs, "__left__", "__mid__"
                )
            if right_edges_df is not None:
                pruned_edges[fast_path_right_edge_idx] = _filter_edges_from_node_pairs(
                    right_edges_df, fast_path_sem_right, right_pairs, "__mid__", "__right__"
                )
            return PathState.from_mutable(local_allowed_nodes, {}, pruned_edges)

    paths_df = domain_to_frame(nodes_df_template, seed_nodes, f'n{node_indices[0]}')

    for i, edge_idx in enumerate(edge_indices):
        left_node_idx = node_indices[i]
        right_node_idx = node_indices[i + 1]

        edges_df = edge_overrides.get(edge_idx)
        if edges_df is None:
            edges_df = executor.edges_df_for_step(edge_idx, state)
        if edges_df is None or len(edges_df) == 0:
            paths_df = paths_df.iloc[0:0]
            break

        edge_op = executor.inputs.chain[edge_idx]
        if not isinstance(edge_op, ASTEdge):
            continue
        sem = EdgeSemantics.from_edge(edge_op)

        edge_alias = executor.meta.alias_for_step(edge_idx)
        edge_cols_needed = {
            ref.column for clause in edge_clauses
            for ref in [clause.left, clause.right] if ref.alias == edge_alias
        }

        edge_cols = [src_col, dst_col] + [c for c in edge_cols_needed if c in edges_df.columns]
        edges_subset = edges_df[list(dict.fromkeys(edge_cols))].copy()

        rename_map = {
            col: f'e{edge_idx}_{col}' for col in edge_cols_needed
            if col in edges_subset.columns and col not in [src_col, dst_col]
        }
        edges_subset = edges_subset.rename(columns=rename_map)

        left_col = f'n{left_node_idx}'
        join_on, result_col = sem.join_cols(src_col, dst_col)
        if sem.is_undirected:
            join1 = paths_df.merge(
                edges_subset, left_on=left_col, right_on=src_col, how='inner'
            )
            join1[f'n{right_node_idx}'] = join1[dst_col]
            join2 = paths_df.merge(
                edges_subset, left_on=left_col, right_on=dst_col, how='inner'
            )
            join2[f'n{right_node_idx}'] = join2[src_col]
            paths_df_concat = concat_frames([join1, join2])
            if paths_df_concat is None:
                paths_df = paths_df.iloc[:0]
                break
            paths_df = paths_df_concat
        else:
            paths_df = paths_df.merge(
                edges_subset, left_on=left_col, right_on=join_on, how='inner'
            )
            paths_df[f'n{right_node_idx}'] = paths_df[result_col]

        right_allowed = local_allowed_nodes.get(right_node_idx)
        if right_allowed is not None and not domain_is_empty(right_allowed):
            paths_df = paths_df[paths_df[f'n{right_node_idx}'].isin(right_allowed)]

        paths_df = paths_df.drop(columns=[src_col, dst_col], errors='ignore')

    if len(paths_df) == 0:
        for idx in node_indices:
            local_allowed_nodes[idx] = domain_empty(nodes_df_template)
        return PathState.from_mutable(local_allowed_nodes, {})

    nodes_df = executor.inputs.graph._nodes
    if nodes_df is not None:
        for clause in edge_clauses:
            for ref in [clause.left, clause.right]:
                binding = executor.inputs.alias_bindings.get(ref.alias)
                if binding and binding.kind == "node" and ref.column != node_id_col:
                    step_idx = binding.step_index
                    col_name = f'n{step_idx}_{ref.column}'
                    if col_name not in paths_df.columns and ref.column in nodes_df.columns:
                        node_attr = nodes_df[[node_id_col, ref.column]].rename(
                            columns={node_id_col: f'n{step_idx}', ref.column: col_name}
                        )
                        paths_df = paths_df.merge(node_attr, on=f'n{step_idx}', how='left')

    mask = make_bool_series(paths_df, True)
    for clause in edge_clauses:
        left_binding = executor.inputs.alias_bindings[clause.left.alias]
        right_binding = executor.inputs.alias_bindings[clause.right.alias]

        if left_binding.kind == "edge":
            left_col_name = f'e{left_binding.step_index}_{clause.left.column}'
        else:
            if clause.left.column == node_id_col or clause.left.column == "id":
                left_col_name = f'n{left_binding.step_index}'
            else:
                left_col_name = f'n{left_binding.step_index}_{clause.left.column}'

        if right_binding.kind == "edge":
            right_col_name = f'e{right_binding.step_index}_{clause.right.column}'
        else:
            if clause.right.column == node_id_col or clause.right.column == "id":
                right_col_name = f'n{right_binding.step_index}'
            else:
                right_col_name = f'n{right_binding.step_index}_{clause.right.column}'

        if left_col_name not in paths_df.columns or right_col_name not in paths_df.columns:
            continue

        left_vals = paths_df[left_col_name]
        right_vals = paths_df[right_col_name]

        clause_mask = evaluate_clause(left_vals, clause.op, right_vals, null_safe=True)
        mask &= clause_mask.fillna(False)

    valid_paths = paths_df[mask]

    for node_idx in node_indices:
        col_name = f'n{node_idx}'
        if col_name in valid_paths.columns:
            valid_node_ids = series_values(valid_paths[col_name])
            current = local_allowed_nodes.get(node_idx)
            local_allowed_nodes[node_idx] = (
                domain_intersect(current, valid_node_ids)
                if current is not None
                else valid_node_ids
            )

    for i, edge_idx in enumerate(edge_indices):
        left_node_idx = node_indices[i]
        right_node_idx = node_indices[i + 1]
        left_col = f'n{left_node_idx}'
        right_col = f'n{right_node_idx}'

        if left_col in valid_paths.columns and right_col in valid_paths.columns:
            valid_pairs = valid_paths[[left_col, right_col]].drop_duplicates()
            edges_df = executor.edges_df_for_step(edge_idx, state)
            if edges_df is not None:
                edge_op = executor.inputs.chain[edge_idx]
                if not isinstance(edge_op, ASTEdge):
                    continue
                sem = EdgeSemantics.from_edge(edge_op)

                if sem.is_undirected:
                    fwd = edges_df.merge(
                        valid_pairs.rename(columns={left_col: src_col, right_col: dst_col}),
                        on=[src_col, dst_col], how='inner'
                    )
                    rev = edges_df.merge(
                        valid_pairs.rename(columns={left_col: dst_col, right_col: src_col}),
                        on=[src_col, dst_col], how='inner'
                    )
                    edges_concat = concat_frames([fwd, rev])
                    edges_df = edges_concat.drop_duplicates(subset=[src_col, dst_col]) if edges_concat is not None else edges_df.iloc[:0]
                else:
                    start_endpoint, end_endpoint = sem.endpoint_cols(src_col, dst_col)
                    edges_df = edges_df.merge(
                        valid_pairs.rename(columns={left_col: start_endpoint, right_col: end_endpoint}),
                        on=[src_col, dst_col], how='inner'
                    )
                pruned_edges[edge_idx] = edges_df

    return PathState.from_mutable(local_allowed_nodes, {}, pruned_edges)
