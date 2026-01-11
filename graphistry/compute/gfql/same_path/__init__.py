"""Same-path GFQL execution modules.

This package contains the Yannakakis-style semijoin executor for
GFQL chains with WHERE clause constraints.
"""

from .chain_meta import ChainMeta
from .edge_semantics import EdgeSemantics
from .df_utils import (
    to_pandas_series,
    series_values,
    evaluate_clause,
    concat_frames,
)
from .bfs import build_edge_pairs, bfs_reachability
from .post_prune import apply_non_adjacent_where_post_prune, apply_edge_where_post_prune
from .multihop import filter_multihop_edges_by_endpoints, find_multihop_start_nodes
from .where_filter import filter_edges_by_clauses, filter_multihop_by_where

__all__ = [
    "ChainMeta",
    "EdgeSemantics",
    "to_pandas_series",
    "series_values",
    "evaluate_clause",
    "concat_frames",
    "build_edge_pairs",
    "bfs_reachability",
    "apply_non_adjacent_where_post_prune",
    "apply_edge_where_post_prune",
    "filter_multihop_edges_by_endpoints",
    "find_multihop_start_nodes",
    "filter_edges_by_clauses",
    "filter_multihop_by_where",
]
