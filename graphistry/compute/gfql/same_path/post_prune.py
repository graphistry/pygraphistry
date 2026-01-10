"""Post-pruning passes for same-path WHERE clause execution.

Contains the non-adjacent node and edge WHERE clause application logic.
These are applied after the initial backward prune to enforce constraints
that span multiple edges in the chain.
"""

from typing import Any, Dict, List, Optional, Set, Sequence, TYPE_CHECKING

import pandas as pd

from graphistry.compute.ast import ASTEdge
from graphistry.compute.typing import DataFrameT
from .edge_semantics import EdgeSemantics
from .bfs import build_edge_pairs
from .df_utils import evaluate_clause, series_values
from .multihop import filter_multihop_edges_by_endpoints, find_multihop_start_nodes

if TYPE_CHECKING:
    from graphistry.compute.gfql.df_executor import (
        DFSamePathExecutor,
        WhereComparison,
    )


def apply_non_adjacent_where_post_prune(
    executor: "DFSamePathExecutor",
    path_state: Any,  # _PathState
) -> Any:
    """Apply WHERE on non-adjacent node aliases by tracing paths.

    Args:
        executor: The executor instance with chain metadata and state
        path_state: Current _PathState with allowed_nodes/allowed_edges

    Returns:
        Updated path_state
    """
    if not executor.inputs.where:
        return path_state

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
        return path_state

    node_indices = executor.meta.node_indices
    edge_indices = executor.meta.edge_indices

    src_col = executor._source_column
    dst_col = executor._destination_column
    edge_id_col = executor._edge_column

    if not src_col or not dst_col:
        return path_state

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

        start_nodes = path_state.allowed_nodes.get(start_node_idx, set())
        end_nodes = path_state.allowed_nodes.get(end_node_idx, set())
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
            state_df = pd.DataFrame(columns=['__current__', '__start__'])

        for edge_idx in relevant_edge_indices:
            edges_df = executor.forward_steps[edge_idx]._edges
            if edges_df is None or len(state_df) == 0:
                break

            allowed_edges = path_state.allowed_edges.get(edge_idx, None)
            if allowed_edges is not None and edge_id_col and edge_id_col in edges_df.columns:
                edges_df = edges_df[edges_df[edge_id_col].isin(list(allowed_edges))]

            edge_op = executor.inputs.chain[edge_idx]
            if not isinstance(edge_op, ASTEdge):
                continue
            sem = EdgeSemantics.from_edge(edge_op)

            if sem.is_multihop:
                # Build edge pairs based on direction
                edge_pairs = build_edge_pairs(edges_df, src_col, dst_col, sem)

                # Propagate state through hops
                all_reachable = [state_df.copy()]
                current_state = state_df.copy()

                for hop in range(1, sem.max_hops + 1):
                    # Propagate current_state through one hop
                    next_state = edge_pairs.merge(
                        current_state, left_on='__from__', right_on='__current__', how='inner'
                    )[['__to__', '__start__']].rename(columns={'__to__': '__current__'}).drop_duplicates()

                    if len(next_state) == 0:
                        break

                    if hop >= sem.min_hops:
                        all_reachable.append(next_state)
                    current_state = next_state

                # Combine all reachable states
                if len(all_reachable) > 1:
                    state_df = pd.concat(all_reachable[1:], ignore_index=True).drop_duplicates()
                else:
                    state_df = pd.DataFrame(columns=['__current__', '__start__'])
            else:
                # Single-hop: propagate state through one hop
                join_col, result_col = sem.join_cols(src_col, dst_col)
                if sem.is_undirected:
                    # Both directions
                    next1 = edges_df.merge(
                        state_df, left_on=src_col, right_on='__current__', how='inner'
                    )[[dst_col, '__start__']].rename(columns={dst_col: '__current__'})
                    next2 = edges_df.merge(
                        state_df, left_on=dst_col, right_on='__current__', how='inner'
                    )[[src_col, '__start__']].rename(columns={src_col: '__current__'})
                    state_df = pd.concat([next1, next2], ignore_index=True).drop_duplicates()
                else:
                    state_df = edges_df.merge(
                        state_df, left_on=join_col, right_on='__current__', how='inner'
                    )[[result_col, '__start__']].rename(columns={result_col: '__current__'}).drop_duplicates()

        # state_df now has (current_node=end_node, start_node) pairs
        # Filter to valid end nodes
        state_df = state_df[state_df['__current__'].isin(end_nodes)]

        if len(state_df) == 0:
            # No valid paths found
            if start_node_idx in path_state.allowed_nodes:
                path_state.allowed_nodes[start_node_idx] = set()
            if end_node_idx in path_state.allowed_nodes:
                path_state.allowed_nodes[end_node_idx] = set()
            continue

        # Join with start and end values to apply WHERE clause
        # left_values_df and right_values_df were built earlier (vectorized)
        if left_values_df is None or right_values_df is None:
            continue

        pairs_df = state_df.merge(left_values_df, on='__start__', how='inner')
        pairs_df = pairs_df.merge(right_values_df, on='__current__', how='inner')

        # Apply the comparison vectorized
        mask = evaluate_clause(pairs_df['__start_val__'], clause.op, pairs_df['__end_val__'])
        valid_pairs = pairs_df[mask]

        valid_starts = set(valid_pairs['__start__'].tolist())
        valid_ends = set(valid_pairs['__current__'].tolist())

        # Update allowed_nodes for start and end positions
        if start_node_idx in path_state.allowed_nodes:
            path_state.allowed_nodes[start_node_idx] &= valid_starts
        if end_node_idx in path_state.allowed_nodes:
            path_state.allowed_nodes[end_node_idx] &= valid_ends

        # Re-propagate constraints backward from the filtered ends
        # to update intermediate nodes and edges
        re_propagate_backward(
            executor, path_state, node_indices, edge_indices,
            start_node_idx, end_node_idx
        )

    return path_state


def apply_edge_where_post_prune(
    executor: "DFSamePathExecutor",
    path_state: Any,  # _PathState
) -> Any:
    """Apply WHERE on edge columns by enumerating paths.

    Args:
        executor: The executor instance with chain metadata and state
        path_state: Current _PathState with allowed_nodes/allowed_edges

    Returns:
        Updated path_state
    """
    if not executor.inputs.where:
        return path_state

    edge_clauses = [
        clause for clause in executor.inputs.where
        if (b1 := executor.inputs.alias_bindings.get(clause.left.alias))
        and (b2 := executor.inputs.alias_bindings.get(clause.right.alias))
        and (b1.kind == "edge" or b2.kind == "edge")
    ]
    if not edge_clauses:
        return path_state

    src_col = executor._source_column
    dst_col = executor._destination_column
    node_id_col = executor._node_column
    if not src_col or not dst_col or not node_id_col:
        return path_state

    node_indices = executor.meta.node_indices
    edge_indices = executor.meta.edge_indices

    seed_nodes = path_state.allowed_nodes.get(node_indices[0], set())
    if not seed_nodes:
        return path_state

    paths_df = pd.DataFrame({f'n{node_indices[0]}': list(seed_nodes)})

    for i, edge_idx in enumerate(edge_indices):
        left_node_idx = node_indices[i]
        right_node_idx = node_indices[i + 1]

        edges_df = executor.forward_steps[edge_idx]._edges
        if edges_df is None or len(edges_df) == 0:
            paths_df = paths_df.iloc[0:0]  # Empty paths
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
            paths_df = pd.concat([join1, join2], ignore_index=True)
        else:
            paths_df = paths_df.merge(
                edges_subset, left_on=left_col, right_on=join_on, how='inner'
            )
            paths_df[f'n{right_node_idx}'] = paths_df[result_col]

        right_allowed = path_state.allowed_nodes.get(right_node_idx, set())
        if right_allowed:
            paths_df = paths_df[paths_df[f'n{right_node_idx}'].isin(list(right_allowed))]

        paths_df = paths_df.drop(columns=[src_col, dst_col], errors='ignore')

    if len(paths_df) == 0:
        for idx in node_indices:
            path_state.allowed_nodes[idx] = set()
        return path_state

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

    mask = pd.Series(True, index=paths_df.index)
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

        # SQL NULL semantics: any comparison with NULL is NULL (treated as False)
        # We need to check for NULL before comparing, because pandas != returns True for X != NaN
        valid = left_vals.notna() & right_vals.notna()

        if clause.op == "==":
            clause_mask = valid & (left_vals == right_vals)
        elif clause.op == "!=":
            clause_mask = valid & (left_vals != right_vals)
        elif clause.op == "<":
            clause_mask = valid & (left_vals < right_vals)
        elif clause.op == "<=":
            clause_mask = valid & (left_vals <= right_vals)
        elif clause.op == ">":
            clause_mask = valid & (left_vals > right_vals)
        elif clause.op == ">=":
            clause_mask = valid & (left_vals >= right_vals)
        else:
            continue

        mask &= clause_mask.fillna(False)

    # Filter paths
    valid_paths = paths_df[mask]

    # Update allowed nodes based on valid paths
    for node_idx in node_indices:
        col_name = f'n{node_idx}'
        if col_name in valid_paths.columns:
            valid_node_ids = set(valid_paths[col_name].unique())
            current = path_state.allowed_nodes.get(node_idx, set())
            path_state.allowed_nodes[node_idx] = current & valid_node_ids if current else valid_node_ids

    for i, edge_idx in enumerate(edge_indices):
        left_node_idx = node_indices[i]
        right_node_idx = node_indices[i + 1]
        left_col = f'n{left_node_idx}'
        right_col = f'n{right_node_idx}'

        if left_col in valid_paths.columns and right_col in valid_paths.columns:
            valid_pairs = valid_paths[[left_col, right_col]].drop_duplicates()
            edges_df = executor.forward_steps[edge_idx]._edges
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
                    edges_df = pd.concat([fwd, rev], ignore_index=True).drop_duplicates(
                        subset=[src_col, dst_col]
                    )
                else:
                    # For directed edges, use endpoint_cols to get proper src/dst mapping
                    start_endpoint, end_endpoint = sem.endpoint_cols(src_col, dst_col)
                    edges_df = edges_df.merge(
                        valid_pairs.rename(columns={left_col: start_endpoint, right_col: end_endpoint}),
                        on=[src_col, dst_col], how='inner'
                    )
                executor.forward_steps[edge_idx]._edges = edges_df

    return path_state


def re_propagate_backward(
    executor: "DFSamePathExecutor",
    path_state: Any,  # _PathState
    node_indices: List[int],
    edge_indices: List[int],
    start_idx: int,
    end_idx: int,
) -> None:
    """Re-propagate constraints backward after filtering non-adjacent nodes.

    This function updates the path_state in-place by re-filtering edges and nodes
    between start_idx and end_idx to reflect new constraints from WHERE clauses.

    Args:
        executor: The executor instance with chain metadata and state
        path_state: Current _PathState with allowed_nodes/allowed_edges (modified in-place)
        node_indices: List of node step indices in the chain
        edge_indices: List of edge step indices in the chain
        start_idx: Start node index for re-propagation range
        end_idx: End node index for re-propagation range
    """
    src_col = executor._source_column
    dst_col = executor._destination_column
    edge_id_col = executor._edge_column

    if not src_col or not dst_col:
        return

    relevant_edge_indices = [idx for idx in edge_indices if start_idx < idx < end_idx]

    for edge_idx in reversed(relevant_edge_indices):
        edge_pos = edge_indices.index(edge_idx)
        left_node_idx = node_indices[edge_pos]
        right_node_idx = node_indices[edge_pos + 1]

        edges_df = executor.forward_steps[edge_idx]._edges
        if edges_df is None:
            continue

        original_len = len(edges_df)
        allowed_edges = path_state.allowed_edges.get(edge_idx, None)
        if allowed_edges is not None and edge_id_col and edge_id_col in edges_df.columns:
            edges_df = edges_df[edges_df[edge_id_col].isin(list(allowed_edges))]

        edge_op = executor.inputs.chain[edge_idx]
        if not isinstance(edge_op, ASTEdge):
            continue
        sem = EdgeSemantics.from_edge(edge_op)

        left_allowed = path_state.allowed_nodes.get(left_node_idx, set())
        right_allowed = path_state.allowed_nodes.get(right_node_idx, set())

        if sem.is_multihop:
            edges_df = filter_multihop_edges_by_endpoints(
                edges_df, edge_op, left_allowed, right_allowed, sem,
                src_col, dst_col
            )
        else:
            if sem.is_undirected:
                if left_allowed and right_allowed:
                    left_set = list(left_allowed)
                    right_set = list(right_allowed)
                    mask = (
                        (edges_df[src_col].isin(left_set) & edges_df[dst_col].isin(right_set))
                        | (edges_df[dst_col].isin(left_set) & edges_df[src_col].isin(right_set))
                    )
                    edges_df = edges_df[mask]
                elif left_allowed:
                    left_set = list(left_allowed)
                    edges_df = edges_df[
                        edges_df[src_col].isin(left_set) | edges_df[dst_col].isin(left_set)
                    ]
                elif right_allowed:
                    right_set = list(right_allowed)
                    edges_df = edges_df[
                        edges_df[src_col].isin(right_set) | edges_df[dst_col].isin(right_set)
                    ]
            else:
                # For directed edges, use endpoint_cols to determine filter columns
                start_col, end_col = sem.endpoint_cols(src_col, dst_col)
                if left_allowed:
                    edges_df = edges_df[edges_df[start_col].isin(list(left_allowed))]
                if right_allowed:
                    edges_df = edges_df[edges_df[end_col].isin(list(right_allowed))]

        if edge_id_col and edge_id_col in edges_df.columns:
            new_edge_ids = set(edges_df[edge_id_col].tolist())
            if edge_idx in path_state.allowed_edges:
                path_state.allowed_edges[edge_idx] &= new_edge_ids
            else:
                path_state.allowed_edges[edge_idx] = new_edge_ids

        if sem.is_multihop:
            new_src_nodes = find_multihop_start_nodes(
                edges_df, edge_op, right_allowed, sem,
                src_col, dst_col
            )
        else:
            new_src_nodes = sem.start_nodes(edges_df, src_col, dst_col)

        if left_node_idx in path_state.allowed_nodes:
            path_state.allowed_nodes[left_node_idx] &= new_src_nodes
        else:
            path_state.allowed_nodes[left_node_idx] = new_src_nodes

        # Persist filtered edges to forward_steps (important when no edge ID column)
        if len(edges_df) < original_len:
            executor.forward_steps[edge_idx]._edges = edges_df
