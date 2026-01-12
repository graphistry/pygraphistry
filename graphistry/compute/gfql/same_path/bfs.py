"""BFS traversal utilities for same-path execution.

Contains pure functions for building edge pairs and computing BFS reachability.
"""

from typing import Any, Set

import pandas as pd

from graphistry.compute.typing import DataFrameT
from .edge_semantics import EdgeSemantics
from .df_utils import concat_frames, df_cons


def build_edge_pairs(
    edges_df: DataFrameT, src_col: str, dst_col: str, sem: EdgeSemantics
) -> DataFrameT:
    """Build normalized edge pairs for BFS traversal based on EdgeSemantics.

    Returns DataFrame with columns ['__from__', '__to__'] representing
    directed edges according to the edge semantics.

    For undirected edges, both directions are included.
    For directed edges, direction follows sem.join_cols().
    """
    is_cudf = edges_df.__class__.__module__.startswith("cudf")
    if sem.is_undirected:
        fwd = edges_df[[src_col, dst_col]].copy()
        fwd.columns = pd.Index(['__from__', '__to__'])
        rev = edges_df[[dst_col, src_col]].copy()
        rev.columns = pd.Index(['__from__', '__to__'])
        result = concat_frames([fwd, rev])
        return result.drop_duplicates() if result is not None else fwd.iloc[:0]
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
    from .df_utils import series_values
    import pandas as pd

    # Use same DataFrame type as input
    result = df_cons(edge_pairs, {'__node__': list(start_nodes), hop_col: 0})
    visited_idx = pd.Index(start_nodes) if not isinstance(start_nodes, pd.Index) else start_nodes

    for hop in range(1, max_hops + 1):
        frontier = result[result[hop_col] == hop - 1][['__node__']].rename(columns={'__node__': '__from__'})
        if len(frontier) == 0:
            break
        next_df = edge_pairs.merge(frontier, on='__from__', how='inner')[['__to__']].drop_duplicates()
        next_df = next_df.rename(columns={'__to__': '__node__'})

        # Filter out already visited nodes using pd.Index operations
        candidate_nodes = series_values(next_df['__node__'])
        new_node_ids = candidate_nodes.difference(visited_idx)
        if len(new_node_ids) == 0:
            break

        new_nodes = df_cons(edge_pairs, {'__node__': list(new_node_ids), hop_col: hop})
        visited_idx = visited_idx.union(new_node_ids)

        result = concat_frames([result, new_nodes])
        if result is None:
            break
    return result
