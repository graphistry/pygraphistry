"""WHERE clause filtering for edges in same-path execution.

Contains functions for filtering edges based on WHERE clause comparisons
between adjacent or multi-hop connected aliases.
"""

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import pandas as pd

from graphistry.compute.ast import ASTEdge, ASTNode
from graphistry.compute.typing import DataFrameT
from .edge_semantics import EdgeSemantics
from .df_utils import evaluate_clause, series_values
from .multihop import filter_multihop_edges_by_endpoints

if TYPE_CHECKING:
    from graphistry.compute.gfql.df_executor import (
        DFSamePathExecutor,
        WhereComparison,
    )


def filter_edges_by_clauses(
    executor: "DFSamePathExecutor",
    edges_df: DataFrameT,
    left_alias: str,
    right_alias: str,
    allowed_nodes: Dict[int, Set[Any]],
    sem: EdgeSemantics,
) -> DataFrameT:
    """Filter edges using WHERE clauses that connect adjacent aliases.

    For forward edges: left_alias matches src, right_alias matches dst.
    For reverse edges: left_alias matches dst, right_alias matches src.
    For undirected edges: try both orientations, keep edges matching either.

    Args:
        executor: The executor instance with inputs and alias_frames
        edges_df: DataFrame of edges to filter
        left_alias: Left node alias name
        right_alias: Right node alias name
        allowed_nodes: Dict mapping step indices to allowed node ID sets
        sem: EdgeSemantics for direction handling

    Returns:
        Filtered edges DataFrame
    """
    # Early return for empty edges - no filtering needed
    if len(edges_df) == 0:
        return edges_df

    relevant = [
        clause
        for clause in executor.inputs.where
        if {clause.left.alias, clause.right.alias} == {left_alias, right_alias}
    ]
    src_col = executor._source_column
    dst_col = executor._destination_column
    node_col = executor._node_column

    if not relevant or not src_col or not dst_col:
        return edges_df

    left_frame = executor.alias_frames.get(left_alias)
    right_frame = executor.alias_frames.get(right_alias)
    if left_frame is None or right_frame is None or node_col is None:
        return edges_df

    left_allowed = allowed_nodes.get(executor.inputs.alias_bindings[left_alias].step_index)
    right_allowed = allowed_nodes.get(executor.inputs.alias_bindings[right_alias].step_index)

    lf = left_frame
    rf = right_frame
    if left_allowed is not None:
        lf = lf[lf[node_col].isin(list(left_allowed))]
    if right_allowed is not None:
        rf = rf[rf[node_col].isin(list(right_allowed))]

    left_cols = list(executor.inputs.column_requirements.get(left_alias, []))
    right_cols = list(executor.inputs.column_requirements.get(right_alias, []))
    if node_col in left_cols:
        left_cols.remove(node_col)
    if node_col in right_cols:
        right_cols.remove(node_col)

    lf = lf[[node_col] + left_cols].rename(columns={node_col: "__left_id__"})
    rf = rf[[node_col] + right_cols].rename(columns={node_col: "__right_id__"})

    # For undirected edges, we need to try both orientations
    if sem.is_undirected:
        # Orientation 1: src=left, dst=right (forward)
        fwd_df = _merge_and_filter_edges(
            executor, edges_df, lf, rf, left_alias, right_alias, relevant,
            left_merge_col=src_col,
            right_merge_col=dst_col
        )
        # Orientation 2: dst=left, src=right (reverse)
        rev_df = _merge_and_filter_edges(
            executor, edges_df, lf, rf, left_alias, right_alias, relevant,
            left_merge_col=dst_col,
            right_merge_col=src_col
        )
        # Combine both orientations - keep edges that match either
        if len(fwd_df) == 0 and len(rev_df) == 0:
            return fwd_df  # Empty dataframe with correct schema
        elif len(fwd_df) == 0:
            out_df = rev_df
        elif len(rev_df) == 0:
            out_df = fwd_df
        else:
            from graphistry.Engine import safe_concat
            out_df = safe_concat([fwd_df, rev_df], ignore_index=True, sort=False)
            # Deduplicate by edge columns (src, dst) to avoid double-counting
            out_df = out_df.drop_duplicates(
                subset=[src_col, dst_col]
            )
        return out_df

    # For reverse edges, left_alias is reached via dst column, right_alias via src column
    # For forward edges, left_alias is reached via src column, right_alias via dst column
    if sem.is_reverse:
        left_merge_col = dst_col
        right_merge_col = src_col
    else:
        left_merge_col = src_col
        right_merge_col = dst_col

    out_df = _merge_and_filter_edges(
        executor, edges_df, lf, rf, left_alias, right_alias, relevant,
        left_merge_col=left_merge_col,
        right_merge_col=right_merge_col
    )

    return out_df


def _merge_and_filter_edges(
    executor: "DFSamePathExecutor",
    edges_df: DataFrameT,
    lf: DataFrameT,
    rf: DataFrameT,
    left_alias: str,
    right_alias: str,
    relevant: List["WhereComparison"],
    left_merge_col: str,
    right_merge_col: str,
) -> DataFrameT:
    """Helper to merge edges with alias frames and apply WHERE clauses.

    Args:
        executor: The executor instance for accessing minmax summaries
        edges_df: DataFrame of edges to filter
        lf: Left frame with __left_id__ column
        rf: Right frame with __right_id__ column
        left_alias: Left node alias name
        right_alias: Right node alias name
        relevant: List of WHERE clauses to apply
        left_merge_col: Column to merge left frame on
        right_merge_col: Column to merge right frame on

    Returns:
        Filtered edges DataFrame
    """
    out_df = edges_df.merge(
        lf,
        left_on=left_merge_col,
        right_on="__left_id__",
        how="inner",
    )
    out_df = out_df.merge(
        rf,
        left_on=right_merge_col,
        right_on="__right_id__",
        how="inner",
        suffixes=("", "__r"),
    )

    for clause in relevant:
        left_col = clause.left.column if clause.left.alias == left_alias else clause.right.column
        right_col = clause.right.column if clause.right.alias == right_alias else clause.left.column
        if clause.op in {">", ">=", "<", "<="}:
            out_df = _apply_inequality_clause(
                executor, out_df, clause, left_alias, right_alias, left_col, right_col
            )
        else:
            col_left_name = f"__val_left_{left_col}"
            col_right_name = f"__val_right_{right_col}"

            # When left_col == right_col, the right merge adds __r suffix
            # We need to rename them to distinct names for comparison
            rename_map = {}
            if left_col in out_df.columns:
                rename_map[left_col] = col_left_name
            # Handle right column: could be right_col or right_col__r depending on merge
            right_col_with_suffix = f"{right_col}__r"
            if right_col_with_suffix in out_df.columns:
                rename_map[right_col_with_suffix] = col_right_name
            elif right_col in out_df.columns and right_col != left_col:
                rename_map[right_col] = col_right_name

            if rename_map:
                out_df = out_df.rename(columns=rename_map)

            if col_left_name in out_df.columns and col_right_name in out_df.columns:
                mask = evaluate_clause(out_df[col_left_name], clause.op, out_df[col_right_name])
                out_df = out_df[mask]

    return out_df


def _apply_inequality_clause(
    executor: "DFSamePathExecutor",
    out_df: DataFrameT,
    clause: "WhereComparison",
    left_alias: str,
    right_alias: str,
    left_col: str,
    right_col: str,
) -> DataFrameT:
    """Apply inequality clause using minmax summaries if available.

    Args:
        executor: The executor instance for accessing minmax summaries
        out_df: DataFrame to filter
        clause: WHERE clause to apply
        left_alias: Left node alias name
        right_alias: Right node alias name
        left_col: Left column name
        right_col: Right column name

    Returns:
        Filtered DataFrame
    """
    left_summary = executor._minmax_summaries.get(left_alias, {}).get(left_col)
    right_summary = executor._minmax_summaries.get(right_alias, {}).get(right_col)

    # Fall back to raw values if summaries are missing
    lsum = None
    rsum = None
    if left_summary is not None:
        lsum = left_summary.rename(
            columns={
                left_summary.columns[0]: "__left_id__",
                "min": f"{left_col}__min",
                "max": f"{left_col}__max",
            }
        )
    if right_summary is not None:
        rsum = right_summary.rename(
            columns={
                right_summary.columns[0]: "__right_id__",
                "min": f"{right_col}__min",
                "max": f"{right_col}__max",
            }
        )

    if lsum is not None and rsum is not None:
        # Both summaries available - use min/max bounds
        merged = out_df.merge(lsum, on="__left_id__", how="left").merge(
            rsum, on="__right_id__", how="left"
        )

        left_min = merged[f"{left_col}__min"]
        left_max = merged[f"{left_col}__max"]
        right_min = merged[f"{right_col}__min"]
        right_max = merged[f"{right_col}__max"]

        if clause.op == ">":
            mask = left_max > right_min
        elif clause.op == ">=":
            mask = left_max >= right_min
        elif clause.op == "<":
            mask = left_min < right_max
        elif clause.op == "<=":
            mask = left_min <= right_max
        else:
            mask = merged.index == merged.index  # all True

        return merged[mask][out_df.columns]

    # Fall back to value-based comparison
    col_left_name = f"__val_left_{left_col}"
    col_right_name = f"__val_right_{right_col}"

    rename_map = {}
    if left_col in out_df.columns:
        rename_map[left_col] = col_left_name
    right_col_with_suffix = f"{right_col}__r"
    if right_col_with_suffix in out_df.columns:
        rename_map[right_col_with_suffix] = col_right_name
    elif right_col in out_df.columns and right_col != left_col:
        rename_map[right_col] = col_right_name

    if rename_map:
        out_df = out_df.rename(columns=rename_map)

    if col_left_name in out_df.columns and col_right_name in out_df.columns:
        mask = evaluate_clause(out_df[col_left_name], clause.op, out_df[col_right_name])
        return out_df[mask]

    return out_df


def filter_multihop_by_where(
    executor: "DFSamePathExecutor",
    edges_df: DataFrameT,
    edge_op: ASTEdge,
    left_alias: str,
    right_alias: str,
    allowed_nodes: Dict[int, Set[Any]],
) -> DataFrameT:
    """Filter multi-hop edges by WHERE clauses connecting start/end aliases.

    For multi-hop traversals, edges_df contains all edges in the path. The src/dst
    columns represent intermediate connections, not the start/end aliases directly.

    Strategy:
    1. Identify which (start, end) pairs satisfy WHERE clauses
    2. Trace paths to find valid edges: start nodes connect via hop 1, end nodes via last hop
    3. Keep only edges that participate in valid paths

    Args:
        executor: The executor instance with inputs and alias_frames
        edges_df: DataFrame of edges to filter
        edge_op: ASTEdge operation with hop constraints
        left_alias: Left node alias name
        right_alias: Right node alias name
        allowed_nodes: Dict mapping step indices to allowed node ID sets

    Returns:
        Filtered edges DataFrame
    """
    relevant = [
        clause
        for clause in executor.inputs.where
        if {clause.left.alias, clause.right.alias} == {left_alias, right_alias}
    ]
    src_col = executor._source_column
    dst_col = executor._destination_column
    node_col = executor._node_column

    if not relevant or not src_col or not dst_col:
        return edges_df

    left_frame = executor.alias_frames.get(left_alias)
    right_frame = executor.alias_frames.get(right_alias)
    if left_frame is None or right_frame is None or node_col is None:
        return edges_df

    # Get hop label column to identify first/last hop edges
    node_label, edge_label = executor._resolve_label_cols(edge_op)

    sem = EdgeSemantics.from_edge(edge_op)

    # Check if hop labels are usable (filtered start node gives unambiguous labels)
    # For unfiltered starts, all edges have hop_label=1, making them useless for identification
    first_node_step = executor.inputs.chain[0] if executor.inputs.chain else None
    has_filtered_start = (
        isinstance(first_node_step, ASTNode) and first_node_step.filter_dict
    )

    if edge_label and edge_label in edges_df.columns and has_filtered_start:
        # Use hop labels to identify start/end nodes (accurate when start is filtered)
        hop_col = edges_df[edge_label]
        min_hop = hop_col.min()
        first_hop_edges = edges_df[hop_col == min_hop]

        chain_min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
        valid_endpoint_edges = edges_df[hop_col >= chain_min_hops]

        if sem.is_undirected:
            start_nodes_df = pd.concat([
                first_hop_edges[[src_col]].rename(columns={src_col: '__node__'}),
                first_hop_edges[[dst_col]].rename(columns={dst_col: '__node__'})
            ], ignore_index=True).drop_duplicates()
            end_nodes_df = pd.concat([
                valid_endpoint_edges[[src_col]].rename(columns={src_col: '__node__'}),
                valid_endpoint_edges[[dst_col]].rename(columns={dst_col: '__node__'})
            ], ignore_index=True).drop_duplicates()
        else:
            # For directed edges, use endpoint_cols to get proper src/dst mapping
            start_col, end_col = sem.endpoint_cols(src_col, dst_col)
            start_nodes_df = first_hop_edges[[start_col]].rename(
                columns={start_col: '__node__'}
            ).drop_duplicates()
            end_nodes_df = valid_endpoint_edges[[end_col]].rename(
                columns={end_col: '__node__'}
            ).drop_duplicates()

        start_nodes = set(start_nodes_df['__node__'].tolist())
        end_nodes = set(end_nodes_df['__node__'].tolist())
    else:
        # Fallback: use alias frames directly when hop labels are ambiguous
        # (unfiltered start makes all edges "hop 1" from some start)
        start_nodes = series_values(left_frame[node_col])
        end_nodes = series_values(right_frame[node_col])

    # Filter to allowed nodes
    left_step_idx = executor.inputs.alias_bindings[left_alias].step_index
    right_step_idx = executor.inputs.alias_bindings[right_alias].step_index
    if left_step_idx in allowed_nodes and allowed_nodes[left_step_idx]:
        start_nodes &= allowed_nodes[left_step_idx]
    if right_step_idx in allowed_nodes and allowed_nodes[right_step_idx]:
        end_nodes &= allowed_nodes[right_step_idx]

    if not start_nodes or not end_nodes:
        return edges_df.iloc[:0]  # Empty dataframe

    # Build (start, end) pairs that satisfy WHERE
    lf = left_frame[left_frame[node_col].isin(list(start_nodes))]
    rf = right_frame[right_frame[node_col].isin(list(end_nodes))]

    left_cols = list(executor.inputs.column_requirements.get(left_alias, []))
    right_cols = list(executor.inputs.column_requirements.get(right_alias, []))
    if node_col in left_cols:
        left_cols.remove(node_col)
    if node_col in right_cols:
        right_cols.remove(node_col)

    lf = lf[[node_col] + left_cols].rename(columns={node_col: "__start_id__"})
    rf = rf[[node_col] + right_cols].rename(columns={node_col: "__end_id__"})

    # Cross join to get all (start, end) combinations
    lf = lf.assign(__cross_key__=1)
    rf = rf.assign(__cross_key__=1)
    pairs_df = lf.merge(rf, on="__cross_key__", suffixes=("", "__r")).drop(columns=["__cross_key__"])

    # Apply WHERE clauses to filter valid (start, end) pairs
    for clause in relevant:
        left_col = clause.left.column if clause.left.alias == left_alias else clause.right.column
        right_col = clause.right.column if clause.right.alias == right_alias else clause.left.column
        # Handle column name collision from merge - when left_col == right_col,
        # pandas adds __r suffix to the right side columns to avoid collision
        actual_right_col = right_col
        if left_col == right_col and f"{right_col}__r" in pairs_df.columns:
            actual_right_col = f"{right_col}__r"
        if left_col in pairs_df.columns and actual_right_col in pairs_df.columns:
            mask = evaluate_clause(pairs_df[left_col], clause.op, pairs_df[actual_right_col])
            pairs_df = pairs_df[mask]

    if len(pairs_df) == 0:
        return edges_df.iloc[:0]

    # Get valid start and end nodes
    valid_starts = set(pairs_df["__start_id__"].tolist())
    valid_ends = set(pairs_df["__end_id__"].tolist())

    # Use vectorized bidirectional reachability to filter edges
    return filter_multihop_edges_by_endpoints(
        edges_df, edge_op, valid_starts, valid_ends, sem,
        src_col, dst_col
    )
