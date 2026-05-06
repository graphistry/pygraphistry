from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from graphistry.compute.ast import ASTEdge, ASTNode
from graphistry.compute.typing import DataFrameT, DomainT
from graphistry.compute.gfql.same_path_types import (
    ComparisonOp,
    EQ_NEQ_WHERE_OPS,
    OP_FLIP,
    PathState,
    SUPPORTED_WHERE_OPS,
)
from .edge_semantics import EdgeSemantics
from .df_utils import (
    concat_frames,
    domain_empty,
    domain_intersect,
    domain_is_empty,
    domain_to_frame,
    domain_union,
    evaluate_clause,
    project_node_attrs,
    semijoin_eval_pairs,
    series_values,
)
from .env_utils import env_flag, env_lower, env_optional_int, normalize_limit
from .multihop import filter_multihop_edges_by_endpoints

if TYPE_CHECKING:
    from graphistry.compute.gfql.df_executor import AliasBinding, DFSamePathExecutor, WhereComparison

def _apply_clause_filters(frame: DataFrameT, relevant: Sequence["WhereComparison"], *, left_alias: str, right_alias: str, left_prefix: str, right_prefix: str, node_col: Optional[str] = None, left_id_col: Optional[str] = None, right_id_col: Optional[str] = None) -> DataFrameT:
    for clause in relevant:
        left_col = clause.left.column if clause.left.alias == left_alias else clause.right.column
        right_col = clause.right.column if clause.right.alias == right_alias else clause.left.column
        col_left = left_id_col if node_col and left_id_col and left_col == node_col else f"{left_prefix}{left_col}"
        col_right = right_id_col if node_col and right_id_col and right_col == node_col else f"{right_prefix}{right_col}"
        if col_left in frame.columns and col_right in frame.columns:
            mask = evaluate_clause(frame[col_left], clause.op, frame[col_right], null_safe=True)
            frame = frame[mask]
    return frame


def filter_edges_by_where(executor: "DFSamePathExecutor", edges_df: DataFrameT, edge_op: ASTEdge, left_alias: str, right_alias: str, allowed_nodes: Dict[int, DomainT], sem: EdgeSemantics) -> DataFrameT:
    if len(edges_df) == 0:
        return edges_df
    relevant = [clause for clause in executor.inputs.where if {clause.left.alias, clause.right.alias} == {left_alias, right_alias}]
    src_col, dst_col, node_col = executor._source_column, executor._destination_column, executor._node_column
    if not relevant or not src_col or not dst_col or not node_col:
        return edges_df
    left_frame = executor.alias_frames.get(left_alias)
    right_frame = executor.alias_frames.get(right_alias)
    if left_frame is None or right_frame is None:
        return edges_df
    left_allowed = allowed_nodes.get(executor.inputs.alias_bindings[left_alias].step_index)
    right_allowed = allowed_nodes.get(executor.inputs.alias_bindings[right_alias].step_index)
    if sem.is_multihop:
        _, edge_label = executor._resolve_label_cols(edge_op)
        has_filtered_start = bool(executor.inputs.chain and isinstance(executor.inputs.chain[0], ASTNode) and executor.inputs.chain[0].filter_dict)
        if has_filtered_start and edge_label in edges_df.columns:
            hop_col = edges_df[edge_label]
            first_hop_edges = edges_df[hop_col == hop_col.min()]
            chain_min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
            valid_endpoint_edges = edges_df[hop_col >= chain_min_hops]
            if sem.is_undirected:
                start_nodes = domain_union(series_values(first_hop_edges[src_col]), series_values(first_hop_edges[dst_col]))
                end_nodes = domain_union(series_values(valid_endpoint_edges[src_col]), series_values(valid_endpoint_edges[dst_col]))
            else:
                start_col, end_col = sem.join_cols(src_col, dst_col)
                start_nodes = series_values(first_hop_edges[start_col])
                end_nodes = series_values(valid_endpoint_edges[end_col])
        else:
            start_nodes = series_values(left_frame[node_col])
            end_nodes = series_values(right_frame[node_col])
        if left_allowed is not None and not domain_is_empty(left_allowed):
            start_nodes = domain_intersect(start_nodes, left_allowed)
        if right_allowed is not None and not domain_is_empty(right_allowed):
            end_nodes = domain_intersect(end_nodes, right_allowed)
        if domain_is_empty(start_nodes) or domain_is_empty(end_nodes):
            return edges_df.iloc[:0]
        lf = project_node_attrs(left_frame, node_col, list(executor.inputs.column_requirements.get(left_alias, [])), id_label="__start_id__", prefix="__L_", node_domain=start_nodes)
        rf = project_node_attrs(right_frame, node_col, list(executor.inputs.column_requirements.get(right_alias, [])), id_label="__end_id__", prefix="__R_", node_domain=end_nodes)
        pairs_df = lf.assign(__cross_key__=1).merge(rf.assign(__cross_key__=1), on="__cross_key__").drop(columns=["__cross_key__"])
        pairs_df = _apply_clause_filters(pairs_df, relevant, left_alias=left_alias, right_alias=right_alias, left_prefix="__L_", right_prefix="__R_")
        if len(pairs_df) == 0:
            return edges_df.iloc[:0]
        return filter_multihop_edges_by_endpoints(
            edges_df,
            series_values(pairs_df["__start_id__"]),
            series_values(pairs_df["__end_id__"]),
            sem,
            src_col,
            dst_col,
        )
    lf = project_node_attrs(left_frame, node_col, list(executor.inputs.column_requirements.get(left_alias, [])), id_label="__left_id__", prefix="__L_", node_domain=left_allowed)
    rf = project_node_attrs(right_frame, node_col, list(executor.inputs.column_requirements.get(right_alias, [])), id_label="__right_id__", prefix="__R_", node_domain=right_allowed)
    merge_cols = [(src_col, dst_col), (dst_col, src_col)] if sem.is_undirected else [sem.join_cols(src_col, dst_col)]
    frames: List[DataFrameT] = [
        _apply_clause_filters(edges_df.merge(lf, left_on=left_merge_col, right_on="__left_id__", how="inner").merge(rf, left_on=right_merge_col, right_on="__right_id__", how="inner"), relevant, left_alias=left_alias, right_alias=right_alias, left_prefix="__L_", right_prefix="__R_", node_col=executor._node_column, left_id_col="__left_id__", right_id_col="__right_id__")
        for left_merge_col, right_merge_col in merge_cols
    ]
    out_df = concat_frames(frames)
    return frames[0] if out_df is None else out_df.drop_duplicates(subset=[src_col, dst_col])


def apply_edge_where_post_prune(executor: "DFSamePathExecutor", state: PathState) -> PathState:
    if not executor.inputs.where:
        return state
    edge_semijoin_enabled = env_flag("GRAPHISTRY_EDGE_WHERE_SEMIJOIN")
    auto_mode = (
        env_lower("GRAPHISTRY_NON_ADJ_WHERE_MODE", "auto") or "auto"
    ) in {"auto", "auto_prefilter"}
    edge_semijoin_auto = env_flag("GRAPHISTRY_EDGE_WHERE_SEMIJOIN_AUTO", default=auto_mode)
    edge_semijoin_pair_max = normalize_limit(env_optional_int("GRAPHISTRY_EDGE_WHERE_SEMIJOIN_PAIR_MAX"), 200000)
    edge_clauses: List[Tuple["WhereComparison", "AliasBinding", "AliasBinding"]] = []
    for clause in executor.inputs.where:
        left_binding = executor.inputs.alias_bindings.get(clause.left.alias)
        right_binding = executor.inputs.alias_bindings.get(clause.right.alias)
        if left_binding is None or right_binding is None:
            continue
        if left_binding.kind == "edge" or right_binding.kind == "edge":
            edge_clauses.append((clause, left_binding, right_binding))
    if not edge_clauses:
        return state
    src_col = executor._source_column
    dst_col = executor._destination_column
    node_id_col = executor._node_column
    if src_col is None or dst_col is None or node_id_col is None:
        return state
    src = src_col
    dst = dst_col
    node_indices, edge_indices = executor.meta.node_indices, executor.meta.edge_indices
    local_allowed_nodes: Dict[int, DomainT] = dict(state.allowed_nodes)
    pruned_edges: Dict[int, DataFrameT] = dict(state.pruned_edges)
    seed_nodes = local_allowed_nodes.get(node_indices[0])
    if domain_is_empty(seed_nodes):
        return state
    nodes_df_template = executor.inputs.graph._nodes
    if nodes_df_template is None:
        return state
    empty_nodes = domain_empty(nodes_df_template)

    def _set_empty_nodes(*idxs: int) -> None:
        for idx in idxs:
            local_allowed_nodes[idx] = empty_nodes

    def _intersect_allowed(idx: int, values: DomainT) -> None:
        if idx in local_allowed_nodes:
            local_allowed_nodes[idx] = domain_intersect(local_allowed_nodes[idx], values)
    edge_cols_by_step: Dict[int, set] = {}
    node_attrs: set = set()
    for clause, left_binding, right_binding in edge_clauses:
        for ref, binding in ((clause.left, left_binding), (clause.right, right_binding)):
            if binding.kind == "edge":
                edge_cols_by_step.setdefault(binding.step_index, set()).add(ref.column)
            elif ref.column != node_id_col:
                node_attrs.add((binding.step_index, ref.column))
    edge_positions = {edge_idx: pos for pos, edge_idx in enumerate(edge_indices)}

    def _merge_edges_with_pairs(edges_df: DataFrameT, sem: EdgeSemantics, pairs_df: DataFrameT, left_label: str, right_label: str, *, value_label: Optional[str] = None, value_col: Optional[str] = None) -> DataFrameT:
        edge_id_col = "__edge_row__"
        while edge_id_col in edges_df.columns:
            edge_id_col += "_x"
        edges_with_id = edges_df.reset_index(drop=True)
        edges_with_id[edge_id_col] = edges_with_id.index
        oriented = sem.orient_edges(edges_with_id, src, dst, dedupe=sem.is_undirected)
        rename_map = {left_label: "__from__", right_label: "__to__"}
        if value_label is not None and value_col is not None:
            rename_map[value_label] = value_col
            on_cols = ["__from__", "__to__", value_col]
        else:
            on_cols = ["__from__", "__to__"]
        merged = oriented.merge(pairs_df.rename(columns=rename_map), on=on_cols, how="inner")
        edge_ids = merged[edge_id_col].drop_duplicates()
        edges_out = edges_with_id[edges_with_id[edge_id_col].isin(edge_ids)].copy()
        return edges_out.drop(columns=[edge_id_col])

    def _edges_for_step(edge_idx: int) -> Optional[DataFrameT]:
        return pruned_edges.get(edge_idx, executor.edges_df_for_step(edge_idx, state))

    def _build_value_pairs(edges_df: DataFrameT, sem: EdgeSemantics, value_col: str, left_label: str, right_label: str, value_label: str) -> DataFrameT:
        pairs = sem.orient_edges(
            edges_df[[src, dst, value_col]], src, dst, dedupe=sem.is_undirected
        ).rename(columns={"__from__": left_label, "__to__": right_label, value_col: value_label}).drop_duplicates()
        return pairs[pairs[value_label].notna()]

    if edge_semijoin_enabled or edge_semijoin_auto:
        for clause, left_binding, right_binding in edge_clauses:
            if left_binding.kind != "edge" or right_binding.kind != "edge":
                continue

            left_edge_idx, right_edge_idx = left_binding.step_index, right_binding.step_index
            left_pos, right_pos = edge_positions.get(left_edge_idx), edge_positions.get(right_edge_idx)
            if left_pos is None or right_pos is None or abs(left_pos - right_pos) != 1:
                continue

            op: ComparisonOp = clause.op
            if left_pos > right_pos:
                left_edge_idx, right_edge_idx = right_edge_idx, left_edge_idx
                left_pos, right_pos = right_pos, left_pos
                op = OP_FLIP.get(op, op)

            if op not in SUPPORTED_WHERE_OPS:
                continue

            left_node_idx, mid_node_idx, right_node_idx = node_indices[left_pos], node_indices[left_pos + 1], node_indices[left_pos + 2]
            left_value_col, right_value_col = clause.left.column, clause.right.column

            left_edges = _edges_for_step(left_edge_idx)
            right_edges = _edges_for_step(right_edge_idx)
            if left_edges is None or right_edges is None or len(left_edges) == 0 or len(right_edges) == 0:
                continue
            if left_value_col not in left_edges.columns or right_value_col not in right_edges.columns:
                continue

            left_edge_op = executor.inputs.chain[left_edge_idx]
            right_edge_op = executor.inputs.chain[right_edge_idx]
            if not (isinstance(left_edge_op, ASTEdge) and isinstance(right_edge_op, ASTEdge)):
                continue
            sem_left, sem_right = EdgeSemantics.from_edge(left_edge_op), EdgeSemantics.from_edge(right_edge_op)
            if sem_left.is_multihop or sem_right.is_multihop:
                continue

            left_pairs, right_pairs = _build_value_pairs(left_edges, sem_left, left_value_col, "__left__", "__mid__", "__left_val__"), _build_value_pairs(right_edges, sem_right, right_value_col, "__mid__", "__right__", "__right_val__")

            left_nodes = local_allowed_nodes.get(left_node_idx)
            if left_nodes is not None and not domain_is_empty(left_nodes):
                left_pairs = left_pairs[left_pairs["__left__"].isin(left_nodes)]
            mid_nodes = local_allowed_nodes.get(mid_node_idx)
            if mid_nodes is not None and not domain_is_empty(mid_nodes):
                left_pairs = left_pairs[left_pairs["__mid__"].isin(mid_nodes)]
                right_pairs = right_pairs[right_pairs["__mid__"].isin(mid_nodes)]
            right_nodes = local_allowed_nodes.get(right_node_idx)
            if right_nodes is not None and not domain_is_empty(right_nodes):
                right_pairs = right_pairs[right_pairs["__right__"].isin(right_nodes)]

            if len(left_pairs) == 0 or len(right_pairs) == 0:
                _set_empty_nodes(left_node_idx, right_node_idx)
                continue

            pair_est_value = len(left_pairs) * len(right_pairs)
            if op in EQ_NEQ_WHERE_OPS:
                left_counts = (
                    left_pairs.groupby("__left_val__")
                    .size()
                    .reset_index()
                    .rename(columns={0: "__left_count__", "size": "__left_count__", "__left_val__": "__value__"})
                )
                right_counts = (
                    right_pairs.groupby("__right_val__")
                    .size()
                    .reset_index()
                    .rename(columns={0: "__right_count__", "size": "__right_count__", "__right_val__": "__value__"})
                )
                equal_counts = left_counts.merge(right_counts, on="__value__", how="inner")
                equal_pairs = (equal_counts["__left_count__"] * equal_counts["__right_count__"]).sum()
                pair_est_value = equal_pairs if op == "==" else pair_est_value - equal_pairs

            if not (edge_semijoin_enabled or (edge_semijoin_auto and (edge_semijoin_pair_max is None or pair_est_value > edge_semijoin_pair_max))):
                continue

            left_eval, right_eval, mid_values = semijoin_eval_pairs(left_pairs, right_pairs, op, left_value="__left_val__", right_value="__right_val__", left_unique_col="__left_unique__", right_unique_col="__right_unique__", left_only_col="__left_only__", right_only_col="__right_only__", left_keep=["__left__", "__mid__", "__left_val__"], right_keep=["__mid__", "__right__", "__right_val__"])
            if left_eval is None or right_eval is None:
                if mid_values is not None:
                    _set_empty_nodes(left_node_idx, right_node_idx)
                continue

            left_pairs, right_pairs = left_eval, right_eval

            if len(left_pairs) == 0 or len(right_pairs) == 0:
                _set_empty_nodes(left_node_idx, right_node_idx)
                continue

            _intersect_allowed(left_node_idx, series_values(left_pairs["__left__"]))
            _intersect_allowed(right_node_idx, series_values(right_pairs["__right__"]))
            _intersect_allowed(mid_node_idx, domain_intersect(series_values(left_pairs["__mid__"]), series_values(right_pairs["__mid__"])))

            for edge_idx, edges_df_pair, sem, pairs, labels, value_col in (
                (
                    left_edge_idx,
                    left_edges,
                    sem_left,
                    left_pairs,
                    ("__left__", "__mid__", "__left_val__"),
                    left_value_col,
                ),
                (
                    right_edge_idx,
                    right_edges,
                    sem_right,
                    right_pairs,
                    ("__mid__", "__right__", "__right_val__"),
                    right_value_col,
                ),
            ):
                pruned_edges[edge_idx] = _merge_edges_with_pairs(
                    edges_df_pair,
                    sem,
                    pairs,
                    labels[0],
                    labels[1],
                    value_label=labels[2],
                    value_col=value_col,
                )
            if len(edge_indices) == 2 and len(edge_clauses) == 1:
                if any(domain_is_empty(local_allowed_nodes.get(idx)) for idx in node_indices):
                    _set_empty_nodes(*node_indices)
                    return PathState.from_mutable(local_allowed_nodes, {})
                left_pairs = left_pairs[["__left__", "__mid__"]].drop_duplicates()
                right_pairs = right_pairs[["__mid__", "__right__"]].drop_duplicates()
                pruned_edges[left_edge_idx] = _merge_edges_with_pairs(left_edges, sem_left, left_pairs, "__left__", "__mid__")
                pruned_edges[right_edge_idx] = _merge_edges_with_pairs(right_edges, sem_right, right_pairs, "__mid__", "__right__")
                return PathState.from_mutable(local_allowed_nodes, {}, pruned_edges)
    paths_df = domain_to_frame(nodes_df_template, seed_nodes, f"n{node_indices[0]}")
    for edge_idx, left_node_idx, right_node_idx in zip(edge_indices, node_indices, node_indices[1:]):
        edges_df_step = _edges_for_step(edge_idx)
        if edges_df_step is None or len(edges_df_step) == 0:
            paths_df = paths_df.iloc[0:0]
            break
        edge_op = executor.inputs.chain[edge_idx]
        if not isinstance(edge_op, ASTEdge):
            continue
        sem = EdgeSemantics.from_edge(edge_op)
        edge_cols_needed = edge_cols_by_step.get(edge_idx, set())
        cols = [src_col, dst_col] + [col for col in edge_cols_needed if col in edges_df_step.columns and col not in {src_col, dst_col}]
        edges_subset = edges_df_step[cols].rename(columns={col: f"e{edge_idx}_{col}" for col in cols[2:]})
        left_col = f"n{left_node_idx}"
        edges_oriented = sem.orient_edges(edges_subset, src_col, dst_col)
        paths_df = paths_df.merge(edges_oriented, left_on=left_col, right_on="__from__", how="inner")
        paths_df[f"n{right_node_idx}"] = paths_df["__to__"]
        paths_df = paths_df.drop(columns=["__from__", "__to__", src_col, dst_col], errors="ignore")
        right_allowed = local_allowed_nodes.get(right_node_idx)
        if right_allowed is not None and not domain_is_empty(right_allowed):
            paths_df = paths_df[paths_df[f"n{right_node_idx}"].isin(right_allowed)]

    if len(paths_df) == 0:
        _set_empty_nodes(*node_indices)
        return PathState.from_mutable(local_allowed_nodes, {})
    nodes_df = executor.inputs.graph._nodes
    if nodes_df is not None:
        for step_idx, col in node_attrs:
            col_name = f"n{step_idx}_{col}"
            if col_name in paths_df.columns or col not in nodes_df.columns:
                continue
            node_attr = nodes_df[[node_id_col, col]].rename(columns={node_id_col: f"n{step_idx}", col: col_name})
            paths_df = paths_df.merge(node_attr, on=f"n{step_idx}", how="left")
    mask = paths_df.iloc[:, 0].notna() | True
    for clause, left_binding, right_binding in edge_clauses:
        left_col_name = f"e{left_binding.step_index}_{clause.left.column}" if left_binding.kind == "edge" else f"n{left_binding.step_index}" + ("" if clause.left.column in {node_id_col, "id"} else f"_{clause.left.column}")
        right_col_name = f"e{right_binding.step_index}_{clause.right.column}" if right_binding.kind == "edge" else f"n{right_binding.step_index}" + ("" if clause.right.column in {node_id_col, "id"} else f"_{clause.right.column}")
        if left_col_name not in paths_df.columns or right_col_name not in paths_df.columns:
            continue
        mask &= evaluate_clause(paths_df[left_col_name], clause.op, paths_df[right_col_name], null_safe=True).fillna(False)
    valid_paths = paths_df[mask]
    for node_idx in node_indices:
        col_name = f"n{node_idx}"
        if col_name in valid_paths.columns:
            _intersect_allowed(node_idx, series_values(valid_paths[col_name]))
    for edge_idx, left_node_idx, right_node_idx in zip(edge_indices, node_indices, node_indices[1:]):
        left_col = f"n{left_node_idx}"
        right_col = f"n{right_node_idx}"
        if left_col in valid_paths.columns and right_col in valid_paths.columns:
            valid_pairs = valid_paths[[left_col, right_col]].drop_duplicates()
            edges_df_step = executor.edges_df_for_step(edge_idx, state)
            if edges_df_step is not None:
                edge_op = executor.inputs.chain[edge_idx]
                if not isinstance(edge_op, ASTEdge):
                    continue
                sem = EdgeSemantics.from_edge(edge_op)
                edges_df_step = _merge_edges_with_pairs(edges_df_step, sem, valid_pairs, left_col, right_col)
                pruned_edges[edge_idx] = edges_df_step

    return PathState.from_mutable(local_allowed_nodes, {}, pruned_edges)
