"""GFQL physical indexes — pay-as-you-go adjacency/node-id indexes for fast
seeded traversal.

Public surface (see api.py): ``create_index``, ``drop_index``, ``show_indexes``,
``gfql_index_edges``, ``gfql_index_all``, and the planner entry ``maybe_index_hop``.
These are wired onto Plottable via ComputeMixin.
"""
from .registry import (
    GfqlIndexRegistry, EMPTY_REGISTRY,
    EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID, ADJ_KINDS, ALL_KINDS,
    AdjacencyIndex, NodeIdIndex,
)
from .api import (
    create_index, drop_index, show_indexes, gfql_index_edges, gfql_index_all,
    get_registry, maybe_index_hop, index_name, REGISTRY_ATTR, index_trace,
)
from .wire import (
    CreateIndex, DropIndex, ShowIndexes, apply_index_op, index_op_from_json,
    is_index_op, is_index_op_json,
)
from .cypher_ddl import parse_index_ddl, looks_like_index_ddl

__all__ = [
    "GfqlIndexRegistry", "EMPTY_REGISTRY",
    "EDGE_OUT_ADJ", "EDGE_IN_ADJ", "NODE_ID", "ADJ_KINDS", "ALL_KINDS",
    "AdjacencyIndex", "NodeIdIndex",
    "create_index", "drop_index", "show_indexes", "gfql_index_edges",
    "gfql_index_all", "get_registry", "maybe_index_hop", "index_name",
    "REGISTRY_ATTR", "index_trace",
    "CreateIndex", "DropIndex", "ShowIndexes", "apply_index_op",
    "index_op_from_json", "is_index_op", "is_index_op_json",
    "parse_index_ddl", "looks_like_index_ddl",
]
