from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
from graphistry.compute.ast import ASTEdge
from graphistry.compute.typing import DataFrameT, DomainT
from graphistry.compute.gfql.same_path_types import (
    EQ_NEQ_WHERE_OPS,
    INEQ_WHERE_OPS,
    OP_FLIP,
    PathState,
    SUPPORTED_WHERE_OPS,
)
from .bfs import bfs_reachability, build_edge_pairs, walk_edge_state
from .edge_semantics import EdgeSemantics
from .df_utils import (
    df_cons,
    domain_empty,
    domain_from_values,
    domain_intersect,
    domain_is_empty,
    evaluate_clause,
    project_node_attrs,
    semijoin_eval_pairs,
    series_values,
)
from .env_utils import env_flag, env_lower, env_optional_int, normalize_limit

if TYPE_CHECKING:
    from graphistry.compute.gfql.df_executor import DFSamePathExecutor, WhereComparison
def filter_multihop_edges_by_endpoints(edges_df: DataFrameT, left_allowed: Optional[DomainT], right_allowed: Optional[DomainT], sem: EdgeSemantics, src_col: str, dst_col: str) -> DataFrameT:
    if not src_col or not dst_col or domain_is_empty(left_allowed) or domain_is_empty(right_allowed):
        return edges_df
    edge_pairs = build_edge_pairs(edges_df, src_col, dst_col, sem)
    left_domain = domain_from_values(left_allowed, edge_pairs)
    right_domain = domain_from_values(right_allowed, edge_pairs)
    fwd_df = bfs_reachability(edge_pairs, left_domain, sem.max_hops, "__fwd_hop__")
    rev_edge_pairs = edge_pairs.rename(columns={"__from__": "__to__", "__to__": "__from__"})
    bwd_df = bfs_reachability(rev_edge_pairs, right_domain, sem.max_hops, "__bwd_hop__")
    if len(fwd_df) == 0 or len(bwd_df) == 0:
        return edges_df.iloc[:0]
    fwd_col, bwd_col = sem.join_cols(src_col, dst_col)
    base_df = (
        sem.orient_edges(edges_df, src_col, dst_col, from_col=src_col, to_col=dst_col)
        if sem.is_undirected
        else edges_df
    )
    annotated = base_df.merge(fwd_df, left_on=fwd_col, right_on="__node__", how="inner").merge(
        bwd_df, left_on=bwd_col, right_on="__node__", how="inner", suffixes=("", "_bwd")
    )
    annotated["__total_hops__"] = annotated["__fwd_hop__"] + 1 + annotated["__bwd_hop__"]
    valid_edges = annotated[annotated["__total_hops__"] <= sem.max_hops][edges_df.columns]
    return valid_edges.drop_duplicates() if sem.is_undirected else valid_edges


def find_multihop_start_nodes(edges_df: DataFrameT, right_allowed: Optional[DomainT], sem: EdgeSemantics, src_col: str, dst_col: str) -> DomainT:
    if not src_col or not dst_col or domain_is_empty(right_allowed):
        return domain_empty(edges_df)
    inverted_sem = EdgeSemantics(
        is_reverse=not sem.is_reverse,
        is_undirected=sem.is_undirected,
        is_multihop=sem.is_multihop,
        min_hops=sem.min_hops,
        max_hops=sem.max_hops,
    )
    edge_pairs = build_edge_pairs(edges_df, src_col, dst_col, inverted_sem)
    right_domain = domain_from_values(right_allowed, edge_pairs)
    reachable = bfs_reachability(edge_pairs, right_domain, sem.max_hops, '__hop__')
    reachable = reachable[reachable['__hop__'] >= sem.min_hops]
    return series_values(reachable['__node__']) if len(reachable) else domain_empty(edge_pairs)


def apply_non_adjacent_where_post_prune(executor: "DFSamePathExecutor", state: PathState) -> PathState:
    if not executor.inputs.where:
        return state
    non_adj_mode = env_lower("GRAPHISTRY_NON_ADJ_WHERE_MODE", "auto") or "auto"
    auto_mode = non_adj_mode in {"auto", "auto_prefilter"}
    bounds_enabled = env_flag("GRAPHISTRY_NON_ADJ_WHERE_BOUNDS")
    value_card_max = env_optional_int("GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX")
    if value_card_max is None and auto_mode:
        value_card_max = 300
    domain_semijoin_enabled = env_flag("GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN")
    domain_semijoin_auto = env_flag("GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_AUTO", default=auto_mode)
    domain_semijoin_pair_max = normalize_limit(
        env_optional_int("GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_PAIR_MAX"),
        200000,
    )
    non_adj_value_ops_raw = env_lower("GRAPHISTRY_NON_ADJ_WHERE_VALUE_OPS")
    value_mode_ops = {
        op.strip() for op in non_adj_value_ops_raw.split(",") if op.strip()
    } if non_adj_value_ops_raw else (set(EQ_NEQ_WHERE_OPS) if auto_mode else {"=="})
    value_mode_ops = {op for op in value_mode_ops if op in SUPPORTED_WHERE_OPS} or {"=="}
    endpoint_clauses: Dict[Tuple[int, int], List[Tuple["WhereComparison", int, int, str, str]]] = defaultdict(list)
    endpoint_eq_clauses: Dict[Tuple[int, int], List[Tuple["WhereComparison", str, str]]] = defaultdict(list)
    for clause in executor.inputs.where:
        if not (left_binding := executor.inputs.alias_bindings.get(clause.left.alias)) or not (right_binding := executor.inputs.alias_bindings.get(clause.right.alias)) or left_binding.kind != "node" or right_binding.kind != "node" or executor.meta.are_steps_adjacent_nodes(left_binding.step_index, right_binding.step_index):
            continue
        start_idx, end_idx, start_col, end_col = (left_binding.step_index, right_binding.step_index, clause.left.column, clause.right.column) if left_binding.step_index <= right_binding.step_index else (right_binding.step_index, left_binding.step_index, clause.right.column, clause.left.column)
        key = (start_idx, end_idx)
        endpoint_clauses[key].append((clause, start_idx, end_idx, start_col, end_col))
        if clause.op == "==":
            endpoint_eq_clauses[key].append((clause, start_col, end_col))
    if not endpoint_clauses:
        return state
    local_allowed_nodes: Dict[int, DomainT] = dict(state.allowed_nodes)
    local_allowed_edges: Dict[int, DomainT] = dict(state.allowed_edges)
    local_pruned_edges: Dict[int, DataFrameT] = dict(state.pruned_edges)
    edge_indices = executor.meta.edge_indices
    src_col = executor._source_column
    dst_col = executor._destination_column
    edge_id_col = executor._edge_column
    node_id_col = executor._node_column
    nodes_df = executor.inputs.graph._nodes
    if src_col is None or dst_col is None or node_id_col is None or nodes_df is None or node_id_col not in nodes_df.columns:
        return state
    src = src_col
    dst = dst_col
    node_id = node_id_col

    def _attr_frame(node_domain: Optional[DomainT], cols: Sequence[str], id_label: str, attr_labels: Sequence[str]) -> Optional[DataFrameT]:
        if domain_is_empty(node_domain):
            return None
        if any(col not in nodes_df.columns for col in cols):
            return None
        return project_node_attrs(nodes_df, node_id, cols, id_label=id_label, labels=attr_labels, node_domain=node_domain, dedupe=True, drop_nulls=True)

    composite_value_enabled = non_adj_mode in {"value", "value_prefilter"} or auto_mode
    # NOTE: Prior multi-eq semijoin / ineq aggregation opt-ins regressed mixed clauses.
    # If restoring, gate to all-eq same-endpoint groups with large pair estimates.
    processed_clause_ids: set = set()
    empty_nodes = domain_empty(nodes_df)

    def _set_empty_nodes(*idxs: int) -> None:
        for idx in idxs:
            local_allowed_nodes[idx] = empty_nodes

    def _update_allowed(idx: int, values: DomainT) -> None:
        current = local_allowed_nodes.get(idx)
        local_allowed_nodes[idx] = domain_intersect(
            current, values) if current is not None else values

    def _apply_pairs_and_backprop(start_idx: int, end_idx: int, start_series: Any, end_series: Any, *, backprop: bool = True) -> None:
        _update_allowed(start_idx, series_values(start_series))
        _update_allowed(end_idx, series_values(end_series))
        if backprop:
            nonlocal local_allowed_nodes, local_allowed_edges
            current_state = PathState.from_mutable(
                local_allowed_nodes, local_allowed_edges, local_pruned_edges
            )
            current_state = executor.backward_propagate_constraints(
                current_state, start_idx, end_idx
            )
            local_allowed_nodes, local_allowed_edges = current_state.to_mutable()
            local_pruned_edges.update(current_state.pruned_edges)

    def _empty_pair(left_df: DataFrameT, right_df: DataFrameT, start_idx: int, end_idx: int) -> bool:
        if len(left_df) == 0 or len(right_df) == 0:
            _set_empty_nodes(start_idx, end_idx)
            return True
        return False

    edge_idxs_by_span = {key: [idx for idx in edge_indices if key[0] < idx < key[1]] for key in endpoint_clauses}
    edge_pairs_cache: Dict[int, DataFrameT] = {}

    def _edge_pairs_cached(edge_idx: int, sem: EdgeSemantics, allowed_edges: Optional[DomainT]) -> DataFrameT:
        edges_df = executor.forward_steps[edge_idx]._edges
        if edges_df is None or len(edges_df) == 0:
            return df_cons(nodes_df, {"__from__": [], "__to__": []})
        if allowed_edges is None:
            cached = edge_pairs_cache.get(edge_idx)
            if cached is None:
                cached = build_edge_pairs(edges_df, src, dst, sem)
                edge_pairs_cache[edge_idx] = cached
            return cached
        if edge_id_col and edge_id_col in edges_df.columns:
            edges_df = edges_df[edges_df[edge_id_col].isin(allowed_edges)]
        return build_edge_pairs(edges_df, src, dst, sem)

    def _pairs_from_endpoints(pairs_left: DataFrameT, pairs_right: DataFrameT, start_df: DataFrameT, end_df: DataFrameT, left_cols: Sequence[str], right_cols: Sequence[str], *, left_id: str = "__start__", right_id: str = "__current__") -> Tuple[DataFrameT, DataFrameT]:
        start_vals = start_df[[left_id] + list(left_cols)].rename(columns={left_id: "__from__"}).drop_duplicates()
        end_vals = end_df[[right_id] + list(right_cols)].rename(columns={right_id: "__to__"}).drop_duplicates()
        left_pairs = pairs_left.merge(start_vals, on="__from__", how="inner").rename(columns={"__from__": "__start__", "__to__": "__mid__"})[
            ["__start__", "__mid__"] + list(left_cols)
        ]
        right_pairs = pairs_right.merge(end_vals, on="__to__", how="inner").rename(columns={"__from__": "__mid__", "__to__": "__current__"})[
            ["__mid__", "__current__"] + list(right_cols)
        ]
        return left_pairs, right_pairs

    def _over_pair_limit(pair_est: float, edge_pair_est: Optional[float], limit: Optional[float]) -> bool:
        return limit is not None and (
            pair_est > limit or (edge_pair_est is not None and edge_pair_est > limit)
        )

    if composite_value_enabled:
        for key, eq_entries in endpoint_eq_clauses.items():
            if len(eq_entries) < 2:
                continue
            group_clause_ids = {id(clause) for clause, _, _ in eq_entries}
            if processed_clause_ids.intersection(group_clause_ids):
                continue
            start_node_idx, end_node_idx = key
            start_nodes = local_allowed_nodes.get(start_node_idx)
            end_nodes = local_allowed_nodes.get(end_node_idx)
            if domain_is_empty(start_nodes) or domain_is_empty(end_nodes):
                continue
            if start_nodes is None or end_nodes is None:
                continue
            relevant_edge_indices = edge_idxs_by_span.get((start_node_idx, end_node_idx), [])
            start_cols = [start_col for _, start_col, _ in eq_entries]
            end_cols = [end_col for _, end_col, _ in eq_entries]
            label_cols = [f"__label{idx}__" for idx in range(len(start_cols))]
            start_df = _attr_frame(start_nodes, start_cols, "__start__", label_cols)
            end_df = _attr_frame(end_nodes, end_cols, "__current__", label_cols)
            if start_df is None or end_df is None:
                continue
            if _empty_pair(start_df, end_df, start_node_idx, end_node_idx):
                continue
            label_cardinality = max(len(start_df[label_cols].drop_duplicates()), len(end_df[label_cols].drop_duplicates()))
            if value_card_max is None or label_cardinality <= value_card_max:
                processed_clause_ids.update(group_clause_ids)
                state_df = start_df[["__start__"] + label_cols].rename(columns={"__start__": "__current__"}).drop_duplicates()
                state_df = walk_edge_state(executor, relevant_edge_indices, state_df, label_cols, local_allowed_edges, edge_id_col, src_col, dst_col)
                state_df = state_df[state_df["__current__"].isin(end_nodes)]
                if _empty_pair(state_df, state_df, start_node_idx, end_node_idx):
                    continue
                valid_labels = state_df.merge(end_df, on=["__current__"] + label_cols, how="inner")[label_cols].drop_duplicates()
                if len(valid_labels) == 0:
                    _set_empty_nodes(start_node_idx, end_node_idx)
                    continue
                valid_starts_df = start_df.merge(valid_labels, on=label_cols, how="inner")
                valid_ends_df = end_df.merge(valid_labels, on=label_cols, how="inner")
                if _empty_pair(valid_starts_df, valid_ends_df, start_node_idx, end_node_idx):
                    continue
                _apply_pairs_and_backprop(start_node_idx, end_node_idx, valid_starts_df["__start__"], valid_ends_df["__current__"])
    for clause_entries in endpoint_clauses.values():
        endpoint_clause_count = len(clause_entries)
        for clause, start_idx, end_idx, _, _ in clause_entries:
            if id(clause) in processed_clause_ids:
                continue
            edge_idxs = edge_idxs_by_span.get((start_idx, end_idx), [])
            start_nodes = local_allowed_nodes.get(start_idx)
            end_nodes = local_allowed_nodes.get(end_idx)
            if domain_is_empty(start_nodes) or domain_is_empty(end_nodes):
                continue
            if start_nodes is None or end_nodes is None:
                continue
            left_vals = _attr_frame(start_nodes, [clause.left.column], "__start__", ["__start_val__"])
            right_vals = _attr_frame(end_nodes, [clause.right.column], "__current__", ["__end_val__"])
            if left_vals is None or right_vals is None:
                continue
            if _empty_pair(left_vals, right_vals, start_idx, end_idx):
                continue
            left_vals_df = left_vals
            right_vals_df = right_vals
            left_domain = series_values(left_vals_df["__start_val__"])
            right_domain = series_values(right_vals_df["__end_val__"])
            value_mode_requested = composite_value_enabled and clause.op in value_mode_ops
            if non_adj_mode in {"prefilter", "value_prefilter", "auto_prefilter"}:
                if clause.op == "==":
                    allowed_values = domain_intersect(left_domain, right_domain)
                    if domain_is_empty(allowed_values):
                        _set_empty_nodes(start_idx, end_idx)
                        continue
                    left_vals_df = left_vals_df[left_vals_df["__start_val__"].isin(allowed_values)]
                    right_vals_df = right_vals_df[right_vals_df["__end_val__"].isin(allowed_values)]
                else:
                    left_count = len(left_domain)
                    right_count = len(right_domain)
                    if left_count == 0 or right_count == 0:
                        _set_empty_nodes(start_idx, end_idx)
                        continue
                    if left_count == 1 and right_count == 1:
                        if not evaluate_clause(left_domain[0], clause.op, right_domain[0]):
                            _set_empty_nodes(start_idx, end_idx)
                            continue
                    elif left_count == 1:
                        left_val = left_domain[0]
                        right_vals_df = right_vals_df[evaluate_clause(right_vals_df["__end_val__"], OP_FLIP.get(clause.op, clause.op), left_val)]
                        if len(right_vals_df) == 0:
                            _set_empty_nodes(start_idx, end_idx)
                            continue
                    elif right_count == 1:
                        right_val = right_domain[0]
                        left_vals_df = left_vals_df[evaluate_clause(left_vals_df["__start_val__"], clause.op, right_val)]
                        if len(left_vals_df) == 0:
                            _set_empty_nodes(start_idx, end_idx)
                            continue
                _apply_pairs_and_backprop(start_idx, end_idx, left_vals_df["__start__"], right_vals_df["__current__"], backprop=False)
                left_domain = series_values(left_vals_df["__start_val__"])
                right_domain = series_values(right_vals_df["__end_val__"])
            if bounds_enabled and clause.op in INEQ_WHERE_OPS:
                left_values, right_values = left_vals_df["__start_val__"], right_vals_df["__end_val__"]
                if len(left_values) > 0 and len(right_values) > 0:
                    left_min = left_values.min()
                    right_max = right_values.max()
                    left_mask = evaluate_clause(left_values, clause.op, right_max)
                    right_mask = evaluate_clause(right_values, OP_FLIP.get(clause.op, clause.op), left_min)
                    left_vals_df = left_vals_df[left_mask]
                    right_vals_df = right_vals_df[right_mask]
                    if _empty_pair(left_vals_df, right_vals_df, start_idx, end_idx):
                        continue
                    start_nodes = series_values(left_vals_df["__start__"])
                    end_nodes = series_values(right_vals_df["__current__"])
                    _update_allowed(start_idx, start_nodes)
                    _update_allowed(end_idx, end_nodes)
                    left_domain = series_values(left_vals_df["__start_val__"])
                    right_domain = series_values(right_vals_df["__end_val__"])
            start_count = 0 if start_nodes is None else len(start_nodes)
            end_count = 0 if end_nodes is None else len(end_nodes)
            pair_est = start_count * end_count
            edge_pair_est = None
            if len(edge_idxs) == 2:
                edge_pair_est = 1
                for edge_idx in edge_idxs:
                    allowed = local_allowed_edges.get(edge_idx)
                    if allowed is not None:
                        edge_pair_est *= len(allowed)
                        continue
                    edges_df = executor.forward_steps[edge_idx]._edges
                    edge_pair_est *= len(edges_df) if edges_df is not None else 0
            if auto_mode and value_mode_requested and endpoint_clause_count > 1 and _over_pair_limit(pair_est, edge_pair_est, domain_semijoin_pair_max):
                value_mode_requested = False
            value_cardinality = max(len(left_domain), len(right_domain))
            value_mode_enabled = value_mode_requested and (
                value_card_max is None or value_cardinality <= value_card_max
            )
            if (domain_semijoin_enabled or domain_semijoin_auto) and clause.op in SUPPORTED_WHERE_OPS and len(edge_idxs) == 2 and not (value_mode_enabled and domain_semijoin_auto and not domain_semijoin_enabled and endpoint_clause_count <= 1):
                edge_idx_left, edge_idx_right = edge_idxs
                edge_left = executor.inputs.chain[edge_idx_left]
                edge_right = executor.inputs.chain[edge_idx_right]
                if isinstance(edge_left, ASTEdge) and isinstance(edge_right, ASTEdge):
                    sem_left = EdgeSemantics.from_edge(edge_left)
                    sem_right = EdgeSemantics.from_edge(edge_right)
                    if not (sem_left.is_multihop or sem_right.is_multihop):
                        pairs_left = _edge_pairs_cached(
                            edge_idx_left,
                            sem_left,
                            local_allowed_edges.get(edge_idx_left),
                        )
                        pairs_right = _edge_pairs_cached(
                            edge_idx_right,
                            sem_right,
                            local_allowed_edges.get(edge_idx_right),
                        )
                        if start_nodes is not None and not domain_is_empty(start_nodes):
                            pairs_left = pairs_left[
                                pairs_left["__from__"].isin(start_nodes)
                            ]
                        if end_nodes is not None and not domain_is_empty(end_nodes):
                            pairs_right = pairs_right[
                                pairs_right["__to__"].isin(end_nodes)
                            ]
                        force_semijoin = (
                            (not domain_semijoin_enabled)
                            and domain_semijoin_auto
                            and auto_mode
                            and not value_mode_enabled
                            and clause.op in EQ_NEQ_WHERE_OPS
                            and value_card_max is not None
                            and value_cardinality > value_card_max
                        )
                        if domain_semijoin_enabled or (domain_semijoin_auto and (force_semijoin or domain_semijoin_pair_max is None or ((edge_pair_est if edge_pair_est is not None else pair_est) > domain_semijoin_pair_max))):
                            start_val_df = left_vals.rename(columns={"__start_val__": "__value__"})
                            end_val_df = right_vals.rename(columns={"__end_val__": "__value__"})
                            left_pairs, right_pairs = _pairs_from_endpoints(pairs_left, pairs_right, start_val_df, end_val_df, ["__value__"], ["__value__"])
                            if _empty_pair(left_pairs, right_pairs, start_idx, end_idx):
                                continue
                            left_eval, right_eval, mid_values = semijoin_eval_pairs(left_pairs, right_pairs, clause.op, left_value="__value__", right_value="__value__", left_unique_col="__left_unique__", right_unique_col="__right_unique__", left_only_col="__left_only__", right_only_col="__right_only__", left_keep=["__start__"], right_keep=["__current__"])
                            if mid_values is not None:
                                if left_eval is None or right_eval is None:
                                    _set_empty_nodes(start_idx, end_idx)
                                    continue
                            else:
                                if left_eval is None or right_eval is None or _empty_pair(left_eval, right_eval, start_idx, end_idx):
                                    continue
                            _apply_pairs_and_backprop(start_idx, end_idx, left_eval["__start__"], right_eval["__current__"])
                            continue
            state_label_col = "__start_val__" if value_mode_enabled else "__start__"
            state_df = left_vals[["__start__", state_label_col]].rename(columns={"__start__": "__current__"}).drop_duplicates() if value_mode_enabled else left_vals[["__start__"]].assign(__current__=left_vals["__start__"])
            state_df = walk_edge_state(executor, edge_idxs, state_df, [state_label_col], local_allowed_edges, edge_id_col, src_col, dst_col)
            if end_nodes is None:
                continue
            state_df = state_df[state_df["__current__"].isin(end_nodes)]
            if len(state_df) == 0:
                _set_empty_nodes(start_idx, end_idx)
                continue
            pairs_df = state_df.merge(right_vals, on="__current__", how="inner") if value_mode_enabled else state_df.merge(left_vals, on="__start__", how="inner").merge(right_vals, on="__current__", how="inner")
            left_col = state_label_col if value_mode_enabled else "__start_val__"
            mask = evaluate_clause(pairs_df[left_col], clause.op, pairs_df["__end_val__"], null_safe=True)
            valid_pairs = pairs_df[mask]
            if value_mode_enabled:
                start_series = left_vals[left_vals["__start_val__"].isin(series_values(valid_pairs[left_col]))]["__start__"]
            else:
                start_series = valid_pairs["__start__"]
            end_series = valid_pairs["__current__"]
            _apply_pairs_and_backprop(start_idx, end_idx, start_series, end_series)
    return PathState.from_mutable(local_allowed_nodes, local_allowed_edges, local_pruned_edges)
