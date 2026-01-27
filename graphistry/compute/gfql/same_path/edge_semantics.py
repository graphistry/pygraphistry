"""Edge semantics for direction handling in same-path execution."""

from dataclasses import dataclass
from typing import Tuple

from graphistry.compute.ast import ASTEdge
from graphistry.compute.typing import DataFrameT, DomainT
from .df_utils import series_values, domain_union

@dataclass(frozen=True)
class EdgeSemantics:
    is_reverse: bool
    is_undirected: bool
    is_multihop: bool
    min_hops: int
    max_hops: int

    @staticmethod
    def from_edge(edge_op: ASTEdge) -> "EdgeSemantics":
        is_reverse = edge_op.direction == "reverse"
        is_undirected = edge_op.direction == "undirected"

        min_hops = edge_op.min_hops if edge_op.min_hops is not None else 1
        if edge_op.max_hops is not None:
            max_hops = edge_op.max_hops
        elif edge_op.hops is not None:
            max_hops = edge_op.hops
        else:
            max_hops = 1

        is_multihop = min_hops != 1 or max_hops != 1

        return EdgeSemantics(
            is_reverse=is_reverse,
            is_undirected=is_undirected,
            is_multihop=is_multihop,
            min_hops=min_hops,
            max_hops=max_hops,
        )

    def join_cols(self, src_col: str, dst_col: str) -> Tuple[str, str]:
        if self.is_reverse:
            return (dst_col, src_col)
        else:
            return (src_col, dst_col)

    def endpoint_cols(self, src_col: str, dst_col: str) -> Tuple[str, str]:
        return self.join_cols(src_col, dst_col)

    def start_nodes(
        self, edges_df: DataFrameT, src_col: str, dst_col: str
    ) -> DomainT:
        if self.is_undirected:
            return domain_union(
                series_values(edges_df[src_col]),
                series_values(edges_df[dst_col]),
            )
        elif self.is_reverse:
            return series_values(edges_df[dst_col])
        else:
            return series_values(edges_df[src_col])
