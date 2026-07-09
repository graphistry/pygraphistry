"""Shared internal types for GFQL physical indexes."""
from __future__ import annotations

from typing import List, Literal, Optional, TypedDict, Union

from graphistry.compute.typing import ArrayLike, ArrayNamespace, DataFrameT

IndexKind = Literal["edge_out_adj", "edge_in_adj", "node_id"]
AdjacencyIndexKind = Literal["edge_out_adj", "edge_in_adj"]
IndexBackend = Literal["numpy", "cupy"]
HopDirection = Literal["forward", "reverse", "undirected"]
EdgeIndexDirection = Literal["forward", "reverse", "both"]

FrameLike = DataFrameT


IndexPath = Literal["scan", "index"]


class IndexTraceStep(TypedDict, total=False):
    op: str
    direction: HopDirection
    hops: Optional[int]
    policy: str
    engine: str
    frontier_n: int
    path: IndexPath
    decision_reason: str
    n_keys: int
    seed_deg_sum: Optional[int]
    est_result_rows: Optional[int]
    threshold_frac: float


IndexTrace = List[IndexTraceStep]
