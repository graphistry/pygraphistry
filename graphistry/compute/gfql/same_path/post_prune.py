"""Post-pruning passes for same-path WHERE clause execution.

Contains the non-adjacent node and edge WHERE clause application logic.
These are applied after the initial backward prune to enforce constraints
that span multiple edges in the chain.
"""

from typing import Any, Dict, List, Optional, Set, Sequence, TYPE_CHECKING

import pandas as pd

from graphistry.compute.ast import ASTEdge
from graphistry.compute.typing import DataFrameT
from graphistry.compute.gfql.same_path_types import PathState
from .edge_semantics import EdgeSemantics
from .bfs import build_edge_pairs
from .df_utils import evaluate_clause, series_values, concat_frames, df_cons, make_bool_series
from .multihop import filter_multihop_edges_by_endpoints, find_multihop_start_nodes

if TYPE_CHECKING:
    from graphistry.compute.gfql.df_executor import (
        DFSamePathExecutor,
        WhereComparison,
    )


def apply_non_adjacent_where_post_prune(
    executor: "DFSamePathExecutor",
    state: PathState,
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

    local_allowed_nodes: Dict[int, Set[Any]] = {
        k: set(v) for k, v in state.allowed_nodes.items()
    }
    local_allowed_edges: Dict[int, Set[Any]] = {
        k: set(v) for k, v in state.allowed_edges.items()
    }
    local_pruned_edges: Dict[int, Any] = dict(state.pruned_edges)

    node_indices = executor.meta.node_indices
    edge_indices = executor.meta.edge_indices

    src_col = executor._source_column
    dst_col = executor._destination_column
    edge_id_col = executor._edge_column

    if not src_col or not dst_col:
        return state

    for clause in non_adjacent_clauses:
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

        start_nodes = local_allowed_nodes.get(start_node_idx, set())
        end_nodes = local_allowed_nodes.get(end_node_idx, set())
        if not start_nodes or not end_nodes:
            continue

        left_col = clause.left.column
        right_col = clause.right.column
        node_id_col = executor._node_column
        if not node_id_col:
            continue

        nodes_df = executor.inputs.graph._nodes
        if nodes_df is None or node_id_col not in nodes_df.columns:
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

        # State table propagation: (current_node, start_node) pairs
        if left_values_df is not None and len(left_values_df) > 0:
            state_df = left_values_df[['__start__']].copy()
            state_df['__current__'] = state_df['__start__']
        else:
            state_df = df_cons(nodes_df, {'__current__': [], '__start__': []})

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
                    )[['__to__', '__start__']].rename(columns={'__to__': '__current__'}).drop_duplicates()

                    if len(next_state) == 0:
                        break

                    if hop >= sem.min_hops:
                        all_reachable.append(next_state)
                    current_state = next_state

                if len(all_reachable) > 1:
                    state_df_concat = concat_frames(all_reachable[1:])
                    state_df = state_df_concat.drop_duplicates() if state_df_concat is not None else state_df.iloc[:0]
                else:
                    state_df = state_df.iloc[:0]
            else:
                join_col, result_col = sem.join_cols(src_col, dst_col)
                if sem.is_undirected:
                    next1 = edges_df.merge(
                        state_df, left_on=src_col, right_on='__current__', how='inner'
                    )[[dst_col, '__start__']].rename(columns={dst_col: '__current__'})
                    next2 = edges_df.merge(
                        state_df, left_on=dst_col, right_on='__current__', how='inner'
                    )[[src_col, '__start__']].rename(columns={src_col: '__current__'})
                    state_df_concat = concat_frames([next1, next2])
                    state_df = state_df_concat.drop_duplicates() if state_df_concat is not None else state_df.iloc[:0]
                else:
                    state_df = edges_df.merge(
                        state_df, left_on=join_col, right_on='__current__', how='inner'
                    )[[result_col, '__start__']].rename(columns={result_col: '__current__'}).drop_duplicates()

        state_df = state_df[state_df['__current__'].isin(end_nodes)]

        if len(state_df) == 0:
            if start_node_idx in local_allowed_nodes:
                local_allowed_nodes[start_node_idx] = set()
            if end_node_idx in local_allowed_nodes:
                local_allowed_nodes[end_node_idx] = set()
            continue

        if left_values_df is None or right_values_df is None:
            continue

        pairs_df = state_df.merge(left_values_df, on='__start__', how='inner')
        pairs_df = pairs_df.merge(right_values_df, on='__current__', how='inner')

        mask = evaluate_clause(pairs_df['__start_val__'], clause.op, pairs_df['__end_val__'])
        valid_pairs = pairs_df[mask]
        valid_starts = series_values(valid_pairs['__start__'])
        valid_ends = series_values(valid_pairs['__current__'])

        if start_node_idx in local_allowed_nodes:
            local_allowed_nodes[start_node_idx] = local_allowed_nodes[start_node_idx].intersection(valid_starts)
        if end_node_idx in local_allowed_nodes:
            local_allowed_nodes[end_node_idx] = local_allowed_nodes[end_node_idx].intersection(valid_ends)

        current_state = PathState.from_mutable(
            local_allowed_nodes, local_allowed_edges, local_pruned_edges
        )
        current_state = executor.backward_propagate_constraints(
            current_state, start_node_idx, end_node_idx
        )
        local_allowed_nodes, local_allowed_edges = current_state.to_mutable()
        local_pruned_edges.update(current_state.pruned_edges)

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
    local_allowed_nodes: Dict[int, Set[Any]] = {
        k: set(v) for k, v in state.allowed_nodes.items()
    }
    # Preserve existing pruned_edges from input state
    pruned_edges: Dict[int, Any] = dict(state.pruned_edges)

    seed_nodes = local_allowed_nodes.get(node_indices[0], set())
    if not seed_nodes:
        return state

    nodes_df_template = executor.inputs.graph._nodes
    if nodes_df_template is None:
        return state

    paths_df = df_cons(nodes_df_template, {f'n{node_indices[0]}': list(seed_nodes)})

    for i, edge_idx in enumerate(edge_indices):
        left_node_idx = node_indices[i]
        right_node_idx = node_indices[i + 1]

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
        edges_subset = edges_df[list(set(edge_cols))].copy()

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
        if right_allowed is not None and len(right_allowed) > 0:
            paths_df = paths_df[paths_df[f'n{right_node_idx}'].isin(right_allowed)]

        paths_df = paths_df.drop(columns=[src_col, dst_col], errors='ignore')

    if len(paths_df) == 0:
        for idx in node_indices:
            local_allowed_nodes[idx] = pd.Index([])
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
            local_allowed_nodes[node_idx] = current.intersection(valid_node_ids) if current is not None else valid_node_ids

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
