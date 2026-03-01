from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTEdge
from graphistry.compute.gfql.same_path.edge_semantics import EdgeSemantics
from graphistry.compute.gfql.same_path.multihop import (
    filter_multihop_edges_by_endpoints,
    find_multihop_start_nodes,
)
from graphistry.compute.gfql.same_path.where_filter import filter_edges_by_where
from graphistry.compute.gfql.same_path_types import PathState
from graphistry.compute.typing import DataFrameT, DomainT

from .df_utils import (
    concat_frames,
    domain_from_values,
    domain_intersect,
    domain_is_empty,
    domain_union,
    domain_union_all,
    series_values,
)

if TYPE_CHECKING:
    from graphistry.compute.gfql.df_executor import DFSamePathExecutor


def backward_prune(executor: "DFSamePathExecutor", allowed_tags: Dict[str, object]) -> PathState:
    executor.meta.validate()  # Raises if chain structure is invalid
    node_indices, edge_indices = executor.meta.node_indices, executor.meta.edge_indices
    allowed_nodes: Dict[int, DomainT] = {}
    allowed_edges: Dict[int, DomainT] = {}
    pruned_edges: Dict[int, DataFrameT] = {}

    def _update_allowed(idx: int, values: DomainT) -> None:
        current = allowed_nodes.get(idx)
        allowed_nodes[idx] = domain_intersect(current, values) if current is not None else values

    node_col = executor._node_column
    for idx in node_indices:
        frame = executor.forward_steps[idx]._nodes
        if frame is None or node_col is None:
            continue
        alias = executor.meta.alias_for_step(idx)
        allowed = allowed_tags.get(alias) if alias is not None else None
        allowed_nodes[idx] = domain_from_values(allowed, frame) if allowed is not None else series_values(frame[node_col])

    for edge_idx, left_node_idx, right_node_idx in zip(
        reversed(edge_indices),
        reversed(node_indices[:-1]),
        reversed(node_indices[1:]),
    ):
        edges_df = executor.forward_steps[edge_idx]._edges
        if edges_df is None:
            continue

        filtered = edges_df
        edge_op = executor.inputs.chain[edge_idx]
        if not isinstance(edge_op, ASTEdge):
            continue
        sem = EdgeSemantics.from_edge(edge_op)
        if not sem.is_multihop:
            allowed_dst = allowed_nodes.get(right_node_idx)
            if allowed_dst is not None:
                filtered = executor._filter_edges_by_allowed_nodes(
                    filtered,
                    sem,
                    executor._source_column,
                    executor._destination_column,
                    right_allowed=allowed_dst,
                )

        left_alias = executor.meta.alias_for_step(left_node_idx)
        right_alias = executor.meta.alias_for_step(right_node_idx)
        if left_alias and right_alias:
            filtered = filter_edges_by_where(
                executor,
                filtered,
                edge_op,
                left_alias,
                right_alias,
                allowed_nodes,
                sem,
            )

        edge_alias = executor.meta.alias_for_step(edge_idx)
        if (
            edge_alias
            and edge_alias in allowed_tags
            and executor._edge_column
            and executor._edge_column in filtered.columns
        ):
            filtered = filtered[filtered[executor._edge_column].isin(allowed_tags[edge_alias])]

        if sem.is_undirected:
            if executor._source_column and executor._destination_column:
                all_nodes_in_edges = domain_union(
                    series_values(filtered[executor._source_column]),
                    series_values(filtered[executor._destination_column]),
                )
                _update_allowed(right_node_idx, all_nodes_in_edges)
                _update_allowed(left_node_idx, all_nodes_in_edges)
        else:
            start_col, end_col = sem.join_cols(
                executor._source_column or "",
                executor._destination_column or "",
            )
            if end_col and end_col in filtered.columns:
                _update_allowed(right_node_idx, series_values(filtered[end_col]))
            if start_col and start_col in filtered.columns:
                _update_allowed(left_node_idx, series_values(filtered[start_col]))

        if executor._edge_column and executor._edge_column in filtered.columns:
            allowed_edges[edge_idx] = series_values(filtered[executor._edge_column])

        if len(filtered) < len(edges_df):
            pruned_edges[edge_idx] = filtered

    return PathState.from_mutable(allowed_nodes, allowed_edges, pruned_edges)


def backward_propagate_constraints(
    executor: "DFSamePathExecutor",
    state: PathState,
    start_node_idx: int,
    end_node_idx: int,
) -> PathState:
    src_col, dst_col, edge_id_col = executor._source_column, executor._destination_column, executor._edge_column
    node_indices = executor.meta.node_indices
    edge_indices = executor.meta.edge_indices
    if not src_col or not dst_col:
        return state
    relevant_edge_indices = [idx for idx in edge_indices if start_node_idx < idx < end_node_idx]
    local_allowed_nodes: Dict[int, DomainT] = dict(state.allowed_nodes)
    local_allowed_edges: Dict[int, DomainT] = dict(state.allowed_edges)
    pruned_edges: Dict[int, DataFrameT] = dict(state.pruned_edges)

    for edge_idx in reversed(relevant_edge_indices):
        edge_pos = edge_indices.index(edge_idx)
        left_node_idx = node_indices[edge_pos]
        right_node_idx = node_indices[edge_pos + 1]

        edges_df = executor.edges_df_for_step(edge_idx, state)
        if edges_df is None:
            continue
        original_len = len(edges_df)
        allowed_edges = local_allowed_edges.get(edge_idx)
        if allowed_edges is not None and edge_id_col and edge_id_col in edges_df.columns:
            edges_df = edges_df[edges_df[edge_id_col].isin(allowed_edges)]

        edge_op = executor.inputs.chain[edge_idx]
        if not isinstance(edge_op, ASTEdge):
            continue
        sem = EdgeSemantics.from_edge(edge_op)
        left_allowed = local_allowed_nodes.get(left_node_idx)
        right_allowed = local_allowed_nodes.get(right_node_idx)

        if sem.is_multihop:
            edges_df = filter_multihop_edges_by_endpoints(
                edges_df,
                left_allowed,
                right_allowed,
                sem,
                src_col,
                dst_col,
            )
        else:
            edges_df = executor._filter_edges_by_allowed_nodes(
                edges_df,
                sem,
                src_col,
                dst_col,
                left_allowed,
                right_allowed,
            )

        if edge_id_col and edge_id_col in edges_df.columns:
            new_edge_ids = series_values(edges_df[edge_id_col])
            local_allowed_edges[edge_idx] = (
                domain_intersect(local_allowed_edges[edge_idx], new_edge_ids)
                if edge_idx in local_allowed_edges
                else new_edge_ids
            )

        if sem.is_multihop:
            new_src_nodes = find_multihop_start_nodes(edges_df, right_allowed, sem, src_col, dst_col)
        else:
            new_src_nodes = sem.start_nodes(edges_df, src_col, dst_col)
        local_allowed_nodes[left_node_idx] = (
            domain_intersect(local_allowed_nodes[left_node_idx], new_src_nodes)
            if left_node_idx in local_allowed_nodes
            else new_src_nodes
        )

        if len(edges_df) < original_len:
            pruned_edges[edge_idx] = edges_df

    return PathState.from_mutable(local_allowed_nodes, local_allowed_edges, pruned_edges)


def materialize_filtered(executor: "DFSamePathExecutor", state: PathState) -> Plottable:
    nodes_df = executor.inputs.graph._nodes
    node_id = executor._node_column
    edge_id = executor._edge_column
    src = executor._source_column
    dst = executor._destination_column
    edge_frames = [
        edges
        for idx, op in enumerate(executor.inputs.chain)
        if isinstance(op, ASTEdge) and (edges := executor.edges_df_for_step(idx, state)) is not None
    ]
    concatenated_edges = concat_frames(edge_frames)
    edges_df = concatenated_edges if concatenated_edges is not None else executor.inputs.graph._edges

    if nodes_df is None or edges_df is None or node_id is None or src is None or dst is None:
        raise ValueError("Graph bindings are incomplete for same-path execution")

    if any(domain_is_empty(node_set) for node_set in state.allowed_nodes.values()):
        return executor._materialize_from_oracle(nodes_df.iloc[0:0], edges_df.iloc[0:0])
    allowed_nodes_domain = domain_union_all(list(state.allowed_nodes.values())) if state.allowed_nodes else None
    if (
        any(isinstance(op, ASTEdge) and EdgeSemantics.from_edge(op).is_multihop for op in executor.inputs.chain)
        and src in edges_df.columns
        and dst in edges_df.columns
    ):
        endpoints = domain_union(series_values(edges_df[src]), series_values(edges_df[dst]))
        allowed_nodes_domain = endpoints if allowed_nodes_domain is None else domain_union(allowed_nodes_domain, endpoints)

    if allowed_nodes_domain is None or len(allowed_nodes_domain) == 0:
        filtered_nodes = nodes_df.iloc[0:0]
        filtered_edges = edges_df.iloc[0:0]
    else:
        filtered_nodes = nodes_df[nodes_df[node_id].isin(allowed_nodes_domain)]
        filtered_edges = edges_df[edges_df[src].isin(allowed_nodes_domain) & edges_df[dst].isin(allowed_nodes_domain)]

    if edge_id and edge_id in filtered_edges.columns:
        allowed_edges_domain = domain_union_all(list(state.allowed_edges.values()))
        if allowed_edges_domain is not None:
            filtered_edges = filtered_edges[filtered_edges[edge_id].isin(allowed_edges_domain)]
    filtered_nodes = executor._merge_label_frames(filtered_nodes, executor._collect_label_frames("node"), node_id)
    if edge_id is not None:
        filtered_edges = executor._merge_label_frames(filtered_edges, executor._collect_label_frames("edge"), edge_id)

    filtered_edges = executor._apply_output_slices(filtered_edges, "edge")
    if any(
        isinstance(op, ASTEdge) and (op.output_min_hops is not None or op.output_max_hops is not None)
        for op in executor.inputs.chain
    ) and len(filtered_edges) > 0:
        endpoint_ids_concat = concat_frames(
            [
                filtered_edges[[src]].rename(columns={src: "__node__"}),
                filtered_edges[[dst]].rename(columns={dst: "__node__"}),
            ]
        )
        if endpoint_ids_concat is not None:
            endpoint_ids_df = endpoint_ids_concat.drop_duplicates()
            filtered_nodes = filtered_nodes[filtered_nodes[node_id].isin(endpoint_ids_df["__node__"])]
    else:
        filtered_nodes = executor._apply_output_slices(filtered_nodes, "node")

    for alias, binding in executor.inputs.alias_bindings.items():
        frame = filtered_nodes if binding.kind == "node" else filtered_edges
        id_col = executor._node_column if binding.kind == "node" else executor._edge_column
        if id_col is None or id_col not in frame.columns:
            continue
        required_cols = list(executor.inputs.column_requirements.get(alias, ()))
        if id_col not in required_cols:
            required_cols.append(id_col)
        subset = frame[[c for c in frame.columns if c in required_cols]].copy()
        executor.alias_frames[alias] = subset

    return executor._materialize_from_oracle(filtered_nodes, filtered_edges)
