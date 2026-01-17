"""Edge semantics for direction handling in same-path execution.

Centralizes direction detection and column mapping for edge traversal.
"""

from dataclasses import dataclass
from typing import Any, Tuple, TYPE_CHECKING

from graphistry.compute.ast import ASTEdge
from .df_utils import series_values, domain_union

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class EdgeSemantics:
    """Encapsulates edge direction semantics for traversal.

    Replaces repeated `is_reverse = op.direction == "reverse"` patterns
    with a single object that provides direction-aware column access.

    Attributes:
        is_reverse: True if edge traverses dst -> src
        is_undirected: True if edge traverses both directions
        is_multihop: True if edge allows multiple hops (min_hops/max_hops != 1)
        min_hops: Minimum number of hops (default 1)
        max_hops: Maximum number of hops (default 1)
    """
    is_reverse: bool
    is_undirected: bool
    is_multihop: bool
    min_hops: int
    max_hops: int

    @staticmethod
    def from_edge(edge_op: ASTEdge) -> "EdgeSemantics":
        """Create EdgeSemantics from an ASTEdge operation.

        Args:
            edge_op: The ASTEdge to analyze

        Returns:
            EdgeSemantics with direction and hop information
        """
        is_reverse = edge_op.direction == "reverse"
        is_undirected = edge_op.direction == "undirected"

        # Determine hop bounds
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
        """Get (left_on, result_col) for a forward join.

        For forward traversal: join on src, result is dst
        For reverse traversal: join on dst, result is src
        For undirected: caller must handle both directions

        Returns:
            (join_column, result_column) tuple
        """
        if self.is_reverse:
            return (dst_col, src_col)
        else:
            return (src_col, dst_col)

    def endpoint_cols(self, src_col: str, dst_col: str) -> Tuple[str, str]:
        """Get (start_endpoint, end_endpoint) columns based on direction.

        For forward: start=src, end=dst
        For reverse: start=dst, end=src

        Returns:
            (start_column, end_column) tuple
        """
        if self.is_reverse:
            return (dst_col, src_col)
        else:
            return (src_col, dst_col)

    def start_nodes(
        self, edges_df, src_col: str, dst_col: str
    ) -> Any:
        """Get starting nodes for edge traversal (for backward propagation).

        For forward: returns src nodes (where traversal starts)
        For reverse: returns dst nodes (where traversal starts when going reverse)
        For undirected: returns both

        Args:
            edges_df: DataFrame with edge data
            src_col: Source column name
            dst_col: Destination column name

        Returns:
            Index-like domain of node IDs where traversal starts
        """
        if self.is_undirected:
            return domain_union(
                series_values(edges_df[src_col]),
                series_values(edges_df[dst_col]),
            )
        elif self.is_reverse:
            return series_values(edges_df[dst_col])
        else:
            return series_values(edges_df[src_col])
