"""Indexed candidate union for a directed single-hop, cross-alias row OR.

The canonical binding materializer and row predicate still own the answer.
Candidates are original edge positions: overlap is removed without merging
different parallel edges or replacing their identities.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple, Union

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, from_json
from graphistry.compute.chain import Chain
from graphistry.compute.chain_fast_paths import _frame_engine, _ids_to_key_array, _resident_seed_indexes
from graphistry.compute.gfql.cypher.row_pushdown import _is_binding_ops
from graphistry.compute.gfql.expr_parser import (
    BinaryOp, GFQLExprParseError, Identifier, ListLiteral, Literal, PropertyAccessExpr, parse_expr,
)
from graphistry.compute.predicates.is_in import is_in
from graphistry.compute.typing import ArrayLike
from . import bindings
from .api import _record_indexed_traversal, get_index_policy
from .bindings import _filter_frame
from .cost import cost_gate_frac
from .engine_arrays import as_eager_polars_frame, col_to_array, take_rows
from .handoff import IndexedBindingsHandoff, attach_handoff
from .lookup import lookup_degree, lookup_edge_rows, lookup_node_rows
from .property_lookup import property_candidate_positions


Scalar = Union[str, int, float]


def _branch(expr: object) -> Optional[Tuple[str, str, List[Scalar], bool]]:
    if not isinstance(expr, BinaryOp) or expr.op not in ("=", "==", "in"):
        return None
    prop = expr.left
    if not isinstance(prop, PropertyAccessExpr) or not isinstance(prop.value, Identifier):
        return None
    if expr.op == "in" and isinstance(expr.right, ListLiteral):
        items: Sequence[object] = expr.right.items
    elif expr.op in ("=", "==") and isinstance(expr.right, Literal):
        items = (expr.right,)
    else:
        return None
    values: List[Scalar] = [item.value for item in items if isinstance(item, Literal)]
    if not values or len(values) != len(items) or any(
        isinstance(value, bool) or not isinstance(value, (str, int, float))
        or (isinstance(value, float) and math.isnan(value)) for value in values
    ):
        return None
    if any(isinstance(value, str) for value in values) and not all(isinstance(value, str) for value in values):
        return None
    return prop.value.name, prop.property, values, expr.op == "in"


def prepare_indexed_or_bindings(g: Plottable, chain: Chain, engine: Engine) -> Optional[Plottable]:
    """Attach a bounded indexed path bag, else leave the canonical execution alone."""
    if get_index_policy(g) == "off" or engine not in (Engine.PANDAS, Engine.CUDF, Engine.POLARS, Engine.POLARS_GPU):
        return None
    requested_engine = engine
    engine = Engine.POLARS if engine == Engine.POLARS_GPU else engine
    calls = chain.chain
    if (chain.where or len(calls) < 2 or not isinstance(calls[0], ASTCall)
            or calls[0].function != "rows" or not isinstance(calls[1], ASTCall)
            or calls[1].function != "where_rows" or calls[0].params.get("alias_prefilters")):
        return None
    plan, text = calls[0].params.get("binding_ops"), calls[1].params.get("expr")
    if not _is_binding_ops(plan) or len(plan) != 3 or not isinstance(text, str):
        return None
    ops = [from_json(item, validate=False) for item in plan]
    first, edge, last = ops
    if (not isinstance(first, ASTNode) or not isinstance(last, ASTNode)
            or not isinstance(edge, ASTEdge) or not edge.is_simple_single_hop()
            or edge.direction not in ("forward", "reverse")
            or first._name is None or last._name is None or first._name == last._name):
        return None
    try:
        expr = parse_expr(text)
    except GFQLExprParseError:
        return None
    if not isinstance(expr, BinaryOp) or expr.op != "or":
        return None
    left, right = _branch(expr.left), _branch(expr.right)
    if left is None or right is None or {left[0], right[0]} != {first._name, last._name}:
        return None
    nodes, edges = g._nodes, g._edges
    node, src, dst = g._node, g._source, g._destination
    if nodes is None or edges is None or node is None or src is None or dst is None:
        return None
    if _frame_engine(nodes) != engine or _frame_engine(edges) != engine:
        return None
    if engine == Engine.POLARS and (
        as_eager_polars_frame(nodes) is not nodes or as_eager_polars_frame(edges) is not edges
    ):
        return None
    from graphistry.compute.gfql.cypher.lowering import _connected_join_pushable_value
    dtypes = nodes.schema if engine == Engine.POLARS else dict(zip(nodes.columns, nodes.dtypes))
    parts: List[ArrayLike] = []
    for alias, column, values, membership in (left, right):
        if any(not _connected_join_pushable_value(
            "==", value, params=None, column=column, node_dtypes=dtypes,
        ) for value in values):
            return None
        direction = edge.direction if alias == first._name else (
            "reverse" if edge.direction == "forward" else "forward")
        ctx = _resident_seed_indexes(g, nodes, edges, node, src, dst, direction)
        if ctx is None:
            return None
        nid, adj, xp, index_engine = ctx
        if not bindings._integer_index(nid) or not bindings._integer_index(adj):
            return None
        filters = {column: is_in(list(values)) if membership else values[0]}
        positions: Optional[ArrayLike]
        if column == node:
            keys = _ids_to_key_array(values, nid.keys_sorted, xp)
            if keys is None:
                return None
            positions = lookup_node_rows(nid, keys, xp)
        else:
            positions = property_candidate_positions(g, "nodes", nodes, filters, index_engine)
        if positions is None:
            return None
        seed = _filter_frame(take_rows(nodes, positions, index_engine), filters, index_engine)
        ids = xp.unique(col_to_array(seed, node, index_engine))
        if get_index_policy(g) != "force" and (
            int(ids.shape[0]) >= cost_gate_frac(engine) * adj.n_keys
            or lookup_degree(adj, ids, xp) >= cost_gate_frac(engine) * len(edges)
        ):
            return None
        parts.append(lookup_edge_rows(adj, ids, xp)[0])
    candidates = xp.unique(xp.concatenate(parts))
    if get_index_policy(g) != "force" and int(candidates.shape[0]) >= cost_gate_frac(engine) * len(edges):
        return None
    state = bindings._try_indexed_connected_bindings_state(
        g, ops, engine=engine, candidate_edge_rows=candidates,
    )
    if state is None:
        return None
    state = replace(state, engine=requested_engine)
    _record_indexed_traversal(
        seam="cross_alias_or", engine=requested_engine, served=True, reason="served",
        hop_count=1, public_seed_scan=False,
        hop_details=[{"hop": 1, "estimated_rows": state.estimated_rows}],
    )
    return attach_handoff(g, IndexedBindingsHandoff(plan, state))
