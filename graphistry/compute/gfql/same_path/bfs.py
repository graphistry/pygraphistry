"""BFS traversal utilities for same-path execution.

Contains pure functions for building edge pairs and computing BFS reachability.
"""

from typing import Any, Set

import pandas as pd

from graphistry.compute.typing import DataFrameT
from .edge_semantics import EdgeSemantics


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
        fwd.columns = pd.Index(['__from__', '__to__'])
        rev = edges_df[[dst_col, src_col]].copy()
        rev.columns = pd.Index(['__from__', '__to__'])
        return pd.concat([fwd, rev], ignore_index=True).drop_duplicates()
    else:
        join_col, result_col = sem.join_cols(src_col, dst_col)
        pairs = edges_df[[join_col, result_col]].copy()
        pairs.columns = pd.Index(['__from__', '__to__'])
        return pairs


def bfs_reachability(
    edge_pairs: DataFrameT, start_nodes: Set[Any], max_hops: int, hop_col: str
) -> DataFrameT:
    """Compute BFS reachability with hop distance tracking.

    Returns DataFrame with columns ['__node__', hop_col] where hop_col
    contains the minimum hop distance from the start set to each node.

    Args:
        edge_pairs: DataFrame with ['__from__', '__to__'] columns
        start_nodes: Set of starting node IDs (hop 0)
        max_hops: Maximum number of hops to traverse
        hop_col: Name for the hop distance column in output

    Returns:
        DataFrame with all reachable nodes and their hop distances
    """
    result = pd.DataFrame({'__node__': list(start_nodes), hop_col: 0})
    all_visited = result.copy()
    for hop in range(1, max_hops + 1):
        frontier = result[result[hop_col] == hop - 1][['__node__']].rename(columns={'__node__': '__from__'})
        if len(frontier) == 0:
            break
        next_df = edge_pairs.merge(frontier, on='__from__', how='inner')[['__to__']].drop_duplicates()
        next_df = next_df.rename(columns={'__to__': '__node__'})
        next_df[hop_col] = hop
        merged = next_df.merge(all_visited[['__node__']], on='__node__', how='left', indicator=True)
        new_nodes = merged[merged['_merge'] == 'left_only'][['__node__', hop_col]]
        if len(new_nodes) == 0:
            break
        result = pd.concat([result, new_nodes], ignore_index=True)
        all_visited = pd.concat([all_visited, new_nodes], ignore_index=True)
    return result
