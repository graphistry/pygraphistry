from dataclasses import dataclass
from typing import Tuple
from graphistry.compute.ast import ASTEdge
from graphistry.compute.typing import DataFrameT, DomainT
from .df_utils import concat_frames, series_values, domain_union


@dataclass(frozen=True)
class EdgeSemantics:
    is_reverse: bool
    is_undirected: bool
    is_multihop: bool
    min_hops: int
    max_hops: int

    @staticmethod
    def from_edge(edge_op: ASTEdge) -> "EdgeSemantics":
        min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
        max_hops = edge_op.max_hops if edge_op.max_hops is not None else (edge_op.hops if edge_op.hops is not None else 1)
        return EdgeSemantics(
            is_reverse=edge_op.direction == "reverse",
            is_undirected=edge_op.direction == "undirected",
            is_multihop=min_hops != 1 or max_hops != 1,
            min_hops=min_hops,
            max_hops=max_hops,
        )

    def join_cols(self, src_col: str, dst_col: str) -> Tuple[str, str]:
        return (dst_col, src_col) if self.is_reverse else (src_col, dst_col)

    def start_nodes(self, edges_df: DataFrameT, src_col: str, dst_col: str) -> DomainT:
        if self.is_undirected:
            return domain_union(series_values(edges_df[src_col]), series_values(edges_df[dst_col]))
        return series_values(edges_df[dst_col] if self.is_reverse else edges_df[src_col])

    def orient_edges(self, edges_df: DataFrameT, src_col: str, dst_col: str, *, from_col: str = "__from__", to_col: str = "__to__", dedupe: bool = False) -> DataFrameT:
        if self.is_undirected:
            fwd = edges_df.rename(columns={src_col: from_col, dst_col: to_col})
            rev = edges_df.rename(columns={dst_col: from_col, src_col: to_col})
            edges_concat = concat_frames([fwd, rev])
            if edges_concat is None:
                return edges_df.iloc[:0]
            return edges_concat.drop_duplicates() if dedupe else edges_concat
        join_col, result_col = self.join_cols(src_col, dst_col)
        return edges_df.rename(columns={join_col: from_col, result_col: to_col})
