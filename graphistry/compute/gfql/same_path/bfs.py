from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING, Union
from graphistry.compute.ast import ASTEdge
from graphistry.compute.typing import DataFrameT, DomainT
from .edge_semantics import EdgeSemantics
from .df_utils import (
    concat_frames,
    domain_diff,
    domain_from_values,
    domain_is_empty,
    domain_to_frame,
    domain_union,
    series_values,
)
if TYPE_CHECKING:
    from graphistry.compute.gfql.df_executor import DFSamePathExecutor


def build_edge_pairs(edges_df: DataFrameT, src_col: str, dst_col: str, sem: EdgeSemantics) -> DataFrameT:
    return sem.orient_edges(edges_df[[src_col, dst_col]], src_col, dst_col, dedupe=sem.is_undirected)


def bfs_reachability(edge_pairs: DataFrameT, start_nodes: Union[Sequence[Any], DomainT], max_hops: int, hop_col: str) -> DataFrameT:
    start_domain = domain_from_values(start_nodes, edge_pairs)
    result = domain_to_frame(edge_pairs, start_domain, '__node__')
    result[hop_col] = 0
    visited_idx = start_domain
    frontier = result[['__node__']].rename(columns={'__node__': '__from__'})

    for hop in range(1, max_hops + 1):
        if len(frontier) == 0:
            break
        next_df = (
            edge_pairs.merge(frontier, on='__from__', how='inner')[['__to__']]
            .rename(columns={'__to__': '__node__'})
            .drop_duplicates()
        )
        candidate_nodes = series_values(next_df['__node__'])
        new_node_ids = domain_diff(candidate_nodes, visited_idx)
        if domain_is_empty(new_node_ids):
            break
        new_nodes = domain_to_frame(edge_pairs, new_node_ids, '__node__')
        new_nodes[hop_col] = hop
        visited_idx = domain_union(visited_idx, new_node_ids)
        frontier = new_nodes[['__node__']].rename(columns={'__node__': '__from__'})

        merged = concat_frames([result, new_nodes])
        if merged is None:
            break
        result = merged
    return result


def walk_edge_state(executor: "DFSamePathExecutor", edge_indices: Sequence[int], state_df: DataFrameT, label_cols: Sequence[str], local_allowed_edges: Dict[int, DomainT], edge_id_col: Optional[str], src_col: str, dst_col: str) -> DataFrameT:
    label_list = list(label_cols)
    for edge_idx in edge_indices:
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
            current_state = state_df.copy()
            reachable: list = []
            for hop in range(1, sem.max_hops + 1):
                next_state = (
                    edge_pairs.merge(
                        current_state,
                        left_on="__from__",
                        right_on="__current__",
                        how="inner",
                    )[["__to__"] + label_list]
                    .rename(columns={"__to__": "__current__"})
                    .drop_duplicates()
                )
                if len(next_state) == 0:
                    break
                if hop >= sem.min_hops:
                    reachable.append(next_state)
                current_state = next_state

            if not reachable:
                state_df = state_df.iloc[:0]
                continue
            merged = concat_frames(reachable)
            if merged is None:
                state_df = state_df.iloc[:0]
            else:
                state_df = merged.drop_duplicates()
            continue

        join_col, result_col = sem.join_cols(src_col, dst_col)
        if sem.is_undirected:
            next1 = (
                edges_df.merge(state_df, left_on=src_col, right_on="__current__", how="inner")
                [[dst_col] + label_list]
                .rename(columns={dst_col: "__current__"})
            )
            next2 = (
                edges_df.merge(state_df, left_on=dst_col, right_on="__current__", how="inner")
                [[src_col] + label_list]
                .rename(columns={src_col: "__current__"})
            )
            merged = concat_frames([next1, next2])
            if merged is None:
                state_df = state_df.iloc[:0]
            else:
                state_df = merged.drop_duplicates()
            continue

        state_df = (
            edges_df.merge(state_df, left_on=join_col, right_on="__current__", how="inner")
            [[result_col] + label_list]
            .rename(columns={result_col: "__current__"})
            .drop_duplicates()
        )
    return state_df
