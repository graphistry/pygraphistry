"""BFS traversal utilities for same-path execution.

Contains pure functions for building edge pairs and computing BFS reachability.
"""

from typing import Any, Sequence

from graphistry.compute.typing import DataFrameT
from .edge_semantics import EdgeSemantics
from .df_utils import (
    concat_frames,
    series_values,
    domain_from_values,
    domain_diff,
    domain_union,
    domain_is_empty,
    domain_to_frame,
)


def build_edge_pairs(
    edges_df: DataFrameT, src_col: str, dst_col: str, sem: EdgeSemantics
) -> DataFrameT:
    """Build normalized edge pairs for BFS traversal based on EdgeSemantics.

    Returns DataFrame with columns ['__from__', '__to__'] representing
    directed edges according to the edge semantics.

    For undirected edges, both directions are included.
    For directed edges, direction follows sem.join_cols().
    """
    if sem.is_undirected:
        fwd = edges_df[[src_col, dst_col]].copy()
        fwd.columns = ['__from__', '__to__']
        rev = edges_df[[dst_col, src_col]].copy()
        rev.columns = ['__from__', '__to__']
        result = concat_frames([fwd, rev])
        return result.drop_duplicates() if result is not None else fwd.iloc[:0]
    else:
        join_col, result_col = sem.join_cols(src_col, dst_col)
        pairs = edges_df[[join_col, result_col]].copy()
        pairs.columns = ['__from__', '__to__']
        return pairs


def bfs_reachability(
    edge_pairs: DataFrameT, start_nodes: Sequence[Any], max_hops: int, hop_col: str
) -> DataFrameT:
    """Compute BFS reachability with hop distance tracking.

    Returns DataFrame with columns ['__node__', hop_col] where hop_col
    contains the minimum hop distance from the start set to each node.

    Args:
        edge_pairs: DataFrame with ['__from__', '__to__'] columns
        start_nodes: Starting node domain (hop 0)
        max_hops: Maximum number of hops to traverse
        hop_col: Name for the hop distance column in output

    Returns:
        DataFrame with all reachable nodes and their hop distances
    """
    # Use same DataFrame type as input
    start_domain = domain_from_values(start_nodes, edge_pairs)
    result = domain_to_frame(edge_pairs, start_domain, '__node__')
    result[hop_col] = 0
    visited_idx = start_domain

    for hop in range(1, max_hops + 1):
        frontier = result[result[hop_col] == hop - 1][['__node__']].rename(columns={'__node__': '__from__'})
        if len(frontier) == 0:
            break
        next_df = edge_pairs.merge(frontier, on='__from__', how='inner')[['__to__']].drop_duplicates()
        next_df = next_df.rename(columns={'__to__': '__node__'})

        # Filter out already visited nodes using domain operations
        candidate_nodes = series_values(next_df['__node__'])
        new_node_ids = domain_diff(candidate_nodes, visited_idx)
        if domain_is_empty(new_node_ids):
            break

        new_nodes = domain_to_frame(edge_pairs, new_node_ids, '__node__')
        new_nodes[hop_col] = hop
        visited_idx = domain_union(visited_idx, new_node_ids)

        result_next = concat_frames([result, new_nodes])
        if result_next is None:
            break
        result = result_next
    return result
