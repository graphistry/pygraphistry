"""Shared internal types for GFQL physical indexes."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Literal, Mapping, Optional, Set, Tuple, TypedDict, Union

import numpy as np
import pandas as pd

from graphistry.compute.typing import ArrayLike, ArrayNamespace, DataFrameT

if TYPE_CHECKING:
    from graphistry.compute.predicates.ASTPredicate import ASTPredicate

IndexKind = Literal["edge_out_adj", "edge_in_adj", "node_id", "node_prop"]
AdjacencyIndexKind = Literal["edge_out_adj", "edge_in_adj"]
IndexBackend = Literal["numpy", "cupy"]
HopDirection = Literal["forward", "reverse", "undirected"]
EdgeIndexDirection = Literal["forward", "reverse", "both"]

FrameLike = DataFrameT


IndexPath = Literal["scan", "index"]
IndexDecisionCode = Literal[
    "policy_off",
    "no_resident_index",
    "not_index_coverable",
    "missing_graph_columns",
    "index_build_declined",
    "scan_cost",
    "index_selected",
    "engine_mismatch",
    "index_path_unavailable",
]


# One column's constraint in an ``edge_match``/filter dict — exactly the runtime shapes
# ``filter_by_dict`` accepts: a plain scalar equality (python or numpy scalar), an
# ASTPredicate, a membership collection (isin), or a nested dict. ``EdgeMatch`` is the
# full dict; ``SimpleEqualityEdgeMatch`` is the scalar-equalities-only subset that the
# index path accelerates (see ``is_simple_equality_edge_match``, a TypeGuard for it).
ScalarMatchValue = Union[str, int, float, bool, None, np.generic]
EdgeMatchValue = Union[
    ScalarMatchValue, "ASTPredicate",
    List[Any], Tuple[Any, ...], Set[Any], FrozenSet[Any], Dict[str, Any],
    "pd.Index", "pd.Series",
]
EdgeMatch = Mapping[str, EdgeMatchValue]
SimpleEqualityEdgeMatch = Mapping[str, ScalarMatchValue]


class IndexTraceStep(TypedDict, total=False):
    op: str
    operation: str
    seam: str
    direction: HopDirection
    hops: Optional[int]
    hop_count: int
    policy: str
    engine: str
    served: bool
    reason: str
    public_seed_scan: bool
    hop_details: List[Dict[str, Any]]
    frontier_n: int
    path: IndexPath
    decision_reason: str
    decision_code: IndexDecisionCode
    n_keys: int
    seed_deg_sum: Optional[int]
    est_result_rows: Optional[int]
    threshold_frac: float


IndexTrace = List[IndexTraceStep]
