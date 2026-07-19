"""GFQL physical indexes — pay-as-you-go adjacency/node-id indexes for fast
seeded traversal.

Public surface (see api.py): ``create_index``, ``drop_index``, ``show_indexes``,
``gfql_index_edges``, ``gfql_index_all``, and the planner entry ``maybe_index_hop``.
These are wired onto Plottable via ComputeMixin.
"""
from .types import (
    AdjacencyIndexKind, ArrayLike, ArrayNamespace, EdgeIndexDirection,
    HopDirection, IndexBackend, IndexKind, IndexTraceStep,
)
from .registry import (
    GfqlIndexRegistry, EMPTY_REGISTRY,
    EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID, ADJ_KINDS, ALL_KINDS,
    AdjacencyIndex, NodeIdIndex,
)
from .api import (
    create_index, drop_index, show_indexes, gfql_index_edges, gfql_index_all,
    get_registry, set_registry, get_index_policy, maybe_index_hop, index_name, index_trace,
)
from .wire import (
    CreateIndex, DropIndex, ShowIndexes, IndexOp, apply_index_op, index_op_from_json,
    is_index_op, is_index_op_json,
)
from .cypher_ddl import parse_index_ddl, looks_like_index_ddl
from .cost import cost_gate_frac, reset_cost_gate_frac, set_cost_gate_frac
from .explain import GfqlExplainReport

__all__ = [
    "AdjacencyIndexKind", "ArrayLike", "ArrayNamespace", "EdgeIndexDirection",
    "HopDirection", "IndexBackend", "IndexKind", "IndexTraceStep",
    "GfqlIndexRegistry", "EMPTY_REGISTRY",
    "EDGE_OUT_ADJ", "EDGE_IN_ADJ", "NODE_ID", "ADJ_KINDS", "ALL_KINDS",
    "AdjacencyIndex", "NodeIdIndex",
    "create_index", "drop_index", "show_indexes", "gfql_index_edges",
    "gfql_index_all", "get_registry", "set_registry", "get_index_policy", "maybe_index_hop", "index_name",
    "index_trace",
    "CreateIndex", "DropIndex", "ShowIndexes", "IndexOp", "apply_index_op",
    "index_op_from_json", "is_index_op", "is_index_op_json",
    "parse_index_ddl", "looks_like_index_ddl",
    "cost_gate_frac", "reset_cost_gate_frac", "set_cost_gate_frac",
    "GfqlExplainReport",
]
