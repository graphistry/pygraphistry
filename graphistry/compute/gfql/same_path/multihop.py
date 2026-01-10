"""Multi-hop edge traversal utilities for same-path execution.

Contains functions for filtering multi-hop edges and finding valid start nodes
using bidirectional reachability propagation.
"""

from typing import Any, List, Optional, Set

import pandas as pd

from graphistry.compute.ast import ASTEdge
from graphistry.compute.typing import DataFrameT
from .edge_semantics import EdgeSemantics
from .bfs import build_edge_pairs, bfs_reachability


def filter_multihop_edges_by_endpoints(
    edges_df: DataFrameT,
    edge_op: ASTEdge,
    left_allowed: Set[Any],
    right_allowed: Set[Any],
    sem: EdgeSemantics,
    src_col: str,
    dst_col: str,
) -> DataFrameT:
    """
    Filter multi-hop edges to only those participating in valid paths
    from left_allowed to right_allowed.

    Uses vectorized bidirectional reachability propagation:
    1. Forward: find nodes reachable from left_allowed at each hop
    2. Backward: find nodes that can reach right_allowed at each hop
    3. Keep edges connecting forward-reachable to backward-reachable nodes

    Args:
        edges_df: DataFrame of edges
        edge_op: ASTEdge operation with hop constraints
        left_allowed: Set of allowed start node IDs
        right_allowed: Set of allowed end node IDs
        sem: EdgeSemantics for direction handling
        src_col: Source column name
        dst_col: Destination column name

    Returns:
        Filtered edges DataFrame
    """
    if not src_col or not dst_col or not left_allowed or not right_allowed:
        return edges_df

    # Only max_hops needed here - min_hops is enforced at path level, not per-edge
    max_hops = edge_op.max_hops if edge_op.max_hops is not None else (
        edge_op.hops if edge_op.hops is not None else 1
    )

    # Build edge pairs and compute bidirectional reachability
    edge_pairs = build_edge_pairs(edges_df, src_col, dst_col, sem)
    fwd_df = bfs_reachability(edge_pairs, left_allowed, max_hops, '__fwd_hop__')
    rev_edge_pairs = edge_pairs.rename(columns={'__from__': '__to__', '__to__': '__from__'})
    bwd_df = bfs_reachability(rev_edge_pairs, right_allowed, max_hops, '__bwd_hop__')

    # An edge (u, v) is valid if:
    # - u is forward-reachable at hop h_fwd (path length from left_allowed to u)
    # - v is backward-reachable at hop h_bwd (path length from v to right_allowed)
    # - h_fwd + 1 + h_bwd is in [min_hops, max_hops]
    if len(fwd_df) == 0 or len(bwd_df) == 0:
        return edges_df.iloc[:0]

    # Yannakakis: min hop is correct here - edge validity uses shortest path through node
    fwd_df = fwd_df.groupby('__node__')['__fwd_hop__'].min().reset_index()
    bwd_df = bwd_df.groupby('__node__')['__bwd_hop__'].min().reset_index()

    # Join edges with hop distances
    if sem.is_undirected:
        # For undirected, check both directions
        # An edge is valid if it lies on ANY valid path from left_allowed to right_allowed.
        # This means: fwd_hop(u) + 1 + bwd_hop(v) <= max_hops
        # We also need at least one path through the edge to have length >= min_hops.

        # Direction 1: src is fwd, dst is bwd
        edges_annotated1 = edges_df.merge(
            fwd_df, left_on=src_col, right_on='__node__', how='inner'
        ).merge(
            bwd_df, left_on=dst_col, right_on='__node__', how='inner', suffixes=('', '_bwd')
        )
        edges_annotated1['__total_hops__'] = edges_annotated1['__fwd_hop__'] + 1 + edges_annotated1['__bwd_hop__']
        # Keep edges that can be part of a valid path (total <= max_hops)
        # The min_hops constraint is enforced at the path level, not per-edge
        valid1 = edges_annotated1[edges_annotated1['__total_hops__'] <= max_hops]

        # Direction 2: dst is fwd, src is bwd
        edges_annotated2 = edges_df.merge(
            fwd_df, left_on=dst_col, right_on='__node__', how='inner'
        ).merge(
            bwd_df, left_on=src_col, right_on='__node__', how='inner', suffixes=('', '_bwd')
        )
        edges_annotated2['__total_hops__'] = edges_annotated2['__fwd_hop__'] + 1 + edges_annotated2['__bwd_hop__']
        valid2 = edges_annotated2[edges_annotated2['__total_hops__'] <= max_hops]

        # Get original edge columns only
        orig_cols = list(edges_df.columns)
        valid_edges = pd.concat([valid1[orig_cols], valid2[orig_cols]], ignore_index=True).drop_duplicates()
        return valid_edges
    else:
        # Determine which column is "source" (fwd) and which is "dest" (bwd)
        fwd_col, bwd_col = sem.endpoint_cols(src_col, dst_col)

        edges_annotated = edges_df.merge(
            fwd_df, left_on=fwd_col, right_on='__node__', how='inner'
        ).merge(
            bwd_df, left_on=bwd_col, right_on='__node__', how='inner', suffixes=('', '_bwd')
        )
        edges_annotated['__total_hops__'] = edges_annotated['__fwd_hop__'] + 1 + edges_annotated['__bwd_hop__']

        # Keep edges that can be part of a valid path (total <= max_hops)
        # The min_hops constraint is enforced at the path level, not per-edge
        valid_edges = edges_annotated[edges_annotated['__total_hops__'] <= max_hops]

        # Return only original columns
        orig_cols = list(edges_df.columns)
        return valid_edges[orig_cols]


def find_multihop_start_nodes(
    edges_df: DataFrameT,
    edge_op: ASTEdge,
    right_allowed: Set[Any],
    sem: EdgeSemantics,
    src_col: str,
    dst_col: str,
) -> Set[Any]:
    """
    Find nodes that can start multi-hop paths reaching right_allowed.

    Uses vectorized hop-by-hop backward propagation via merge+groupby.

    Args:
        edges_df: DataFrame of edges
        edge_op: ASTEdge operation with hop constraints
        right_allowed: Set of allowed destination node IDs
        sem: EdgeSemantics for direction handling
        src_col: Source column name
        dst_col: Destination column name

    Returns:
        Set of valid start node IDs
    """
    if not src_col or not dst_col or not right_allowed:
        return set()

    min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
    max_hops = edge_op.max_hops if edge_op.max_hops is not None else (
        edge_op.hops if edge_op.hops is not None else 1
    )

    # Build edge pairs for backward traversal (inverted direction)
    # For forward edges, backward trace goes dst->src
    # Create inverted semantics for backward traversal
    inverted_sem = EdgeSemantics(
        is_reverse=not sem.is_reverse,
        is_undirected=sem.is_undirected,
        is_multihop=sem.is_multihop,
        min_hops=sem.min_hops,
        max_hops=sem.max_hops,
    )
    edge_pairs = build_edge_pairs(edges_df, src_col, dst_col, inverted_sem)

    # Vectorized backward BFS: propagate reachability hop by hop
    # Use DataFrame-based tracking throughout (no Python sets internally)
    # Start with right_allowed as target destinations (hop 0 means "at the destination")
    # We trace backward to find nodes that can REACH these destinations
    frontier = pd.DataFrame({'__node__': list(right_allowed)})
    all_visited = frontier.copy()
    valid_starts_frames: List[DataFrameT] = []

    # Collect nodes at each hop distance FROM the destination
    for hop in range(1, max_hops + 1):
        # Join with edges to find nodes one hop back from frontier
        # edge_pairs: __from__ = dst (target), __to__ = src (predecessor)
        # We want nodes (__to__) that can reach frontier nodes (__from__)
        new_frontier = edge_pairs.merge(
            frontier,
            left_on='__from__',
            right_on='__node__',
            how='inner'
        )[['__to__']].drop_duplicates()

        if len(new_frontier) == 0:
            break

        new_frontier = new_frontier.rename(columns={'__to__': '__node__'})

        # Collect valid starts (nodes at hop distance in [min_hops, max_hops])
        # These are nodes that can reach right_allowed in exactly `hop` hops
        if hop >= min_hops:
            valid_starts_frames.append(new_frontier[['__node__']])

        # Anti-join: filter out nodes already visited to avoid infinite loops
        # But still keep nodes for valid_starts even if visited before at different hop
        merged = new_frontier.merge(
            all_visited[['__node__']], on='__node__', how='left', indicator=True
        )
        unvisited = merged[merged['_merge'] == 'left_only'][['__node__']]

        if len(unvisited) == 0:
            break

        frontier = unvisited
        all_visited = pd.concat([all_visited, unvisited], ignore_index=True)

    # Combine all valid starts and convert to set (caller expects set)
    if valid_starts_frames:
        valid_starts_df = pd.concat(valid_starts_frames, ignore_index=True).drop_duplicates()
        return set(valid_starts_df['__node__'].tolist())
    return set()
