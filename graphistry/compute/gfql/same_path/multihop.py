"""Multi-hop edge traversal utilities for same-path execution."""

from typing import List, Optional

from graphistry.compute.ast import ASTEdge
from graphistry.compute.typing import DataFrameT, DomainT
from .edge_semantics import EdgeSemantics
from .bfs import build_edge_pairs, bfs_reachability
from .df_utils import (
    series_values,
    concat_frames,
    domain_is_empty,
    domain_from_values,
    domain_diff,
    domain_union,
    domain_to_frame,
    domain_empty,
)


def filter_multihop_edges_by_endpoints(
    edges_df: DataFrameT,
    edge_op: ASTEdge,
    left_allowed: Optional[DomainT],
    right_allowed: Optional[DomainT],
    sem: EdgeSemantics,
    src_col: str,
    dst_col: str,
) -> DataFrameT:
    if not src_col or not dst_col or domain_is_empty(left_allowed) or domain_is_empty(right_allowed):
        return edges_df

    max_hops = edge_op.max_hops if edge_op.max_hops is not None else (
        edge_op.hops if edge_op.hops is not None else 1
    )

    edge_pairs = build_edge_pairs(edges_df, src_col, dst_col, sem)
    left_domain = domain_from_values(left_allowed, edge_pairs)
    right_domain = domain_from_values(right_allowed, edge_pairs)
    fwd_df = bfs_reachability(edge_pairs, left_domain, max_hops, '__fwd_hop__')
    rev_edge_pairs = edge_pairs.rename(columns={'__from__': '__to__', '__to__': '__from__'})
    bwd_df = bfs_reachability(rev_edge_pairs, right_domain, max_hops, '__bwd_hop__')

    if len(fwd_df) == 0 or len(bwd_df) == 0:
        return edges_df.iloc[:0]

    fwd_df = fwd_df.groupby('__node__')['__fwd_hop__'].min().reset_index()
    bwd_df = bwd_df.groupby('__node__')['__bwd_hop__'].min().reset_index()

    if sem.is_undirected:
        edges_annotated1 = edges_df.merge(
            fwd_df, left_on=src_col, right_on='__node__', how='inner'
        ).merge(
            bwd_df, left_on=dst_col, right_on='__node__', how='inner', suffixes=('', '_bwd')
        )
        edges_annotated1['__total_hops__'] = edges_annotated1['__fwd_hop__'] + 1 + edges_annotated1['__bwd_hop__']
        valid1 = edges_annotated1[edges_annotated1['__total_hops__'] <= max_hops]

        edges_annotated2 = edges_df.merge(
            fwd_df, left_on=dst_col, right_on='__node__', how='inner'
        ).merge(
            bwd_df, left_on=src_col, right_on='__node__', how='inner', suffixes=('', '_bwd')
        )
        edges_annotated2['__total_hops__'] = edges_annotated2['__fwd_hop__'] + 1 + edges_annotated2['__bwd_hop__']
        valid2 = edges_annotated2[edges_annotated2['__total_hops__'] <= max_hops]

        orig_cols = list(edges_df.columns)
        valid_edges = concat_frames([valid1[orig_cols], valid2[orig_cols]])
        return valid_edges.drop_duplicates() if valid_edges is not None else edges_df.iloc[:0]
    else:
        fwd_col, bwd_col = sem.endpoint_cols(src_col, dst_col)

        edges_annotated = edges_df.merge(
            fwd_df, left_on=fwd_col, right_on='__node__', how='inner'
        ).merge(
            bwd_df, left_on=bwd_col, right_on='__node__', how='inner', suffixes=('', '_bwd')
        )
        edges_annotated['__total_hops__'] = edges_annotated['__fwd_hop__'] + 1 + edges_annotated['__bwd_hop__']

        valid_edges = edges_annotated[edges_annotated['__total_hops__'] <= max_hops]

        orig_cols = list(edges_df.columns)
        return valid_edges[orig_cols]


def find_multihop_start_nodes(
    edges_df: DataFrameT,
    edge_op: ASTEdge,
    right_allowed: Optional[DomainT],
    sem: EdgeSemantics,
    src_col: str,
    dst_col: str,
) -> DomainT:
    if not src_col or not dst_col or domain_is_empty(right_allowed):
        return domain_empty(edges_df)

    min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
    max_hops = edge_op.max_hops if edge_op.max_hops is not None else (
        edge_op.hops if edge_op.hops is not None else 1
    )

    inverted_sem = EdgeSemantics(
        is_reverse=not sem.is_reverse,
        is_undirected=sem.is_undirected,
        is_multihop=sem.is_multihop,
        min_hops=sem.min_hops,
        max_hops=sem.max_hops,
    )
    edge_pairs = build_edge_pairs(edges_df, src_col, dst_col, inverted_sem)

    right_domain = domain_from_values(right_allowed, edge_pairs)
    frontier = domain_to_frame(edge_pairs, right_domain, '__node__')
    visited_idx = right_domain
    valid_starts_frames: List[DataFrameT] = []

    for hop in range(1, max_hops + 1):
        new_frontier = edge_pairs.merge(
            frontier,
            left_on='__from__',
            right_on='__node__',
            how='inner'
        )[['__to__']].drop_duplicates()

        if len(new_frontier) == 0:
            break

        new_frontier = new_frontier.rename(columns={'__to__': '__node__'})

        if hop >= min_hops:
            valid_starts_frames.append(new_frontier[['__node__']])

        candidate_nodes = series_values(new_frontier['__node__'])
        new_node_ids = domain_diff(candidate_nodes, visited_idx)
        if domain_is_empty(new_node_ids):
            break

        unvisited = domain_to_frame(edge_pairs, new_node_ids, '__node__')
        visited_idx = domain_union(visited_idx, new_node_ids)

        frontier = unvisited

    if valid_starts_frames:
        valid_starts_df = concat_frames(valid_starts_frames)
        if valid_starts_df is not None:
            valid_starts_df = valid_starts_df.drop_duplicates()
            return series_values(valid_starts_df['__node__'])
    return domain_empty(edge_pairs)
