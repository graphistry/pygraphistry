"""Bound wide single-hop binding payloads using canonical thin-row order."""
from __future__ import annotations

from typing import Optional, Sequence

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject
from graphistry.compute.chain_fast_paths import _frame_engine
from graphistry.compute.gfql.expr_parser import (
    GFQLExprParseError, Identifier, PropertyAccessExpr, parse_expr,
)
from graphistry.compute.gfql.index.api import drop_index, with_index_policy
from graphistry.compute.gfql.index.bindings import (
    IndexedBindingsState, _INTERNAL, _policy_is_active, _with_marker,
)
from graphistry.compute.gfql.index.engine_arrays import array_namespace, col_to_array, take_rows
from graphistry.compute.gfql.row.pipeline import RowPipelineMixin, _RowPipelineAdapter
from graphistry.compute.util import generate_safe_column_name_from


def try_limited_bindings_state(
    g: Plottable, ops: Sequence[ASTObject], suffix: Sequence[ASTObject], engine: Engine,
) -> Optional[IndexedBindingsState]:
    """Keep unsupported/filtered/ordered shapes on their existing execution path.

    Only keys and original edge positions participate in the full walk. The
    canonical binder owns its order; LIMIT bounds every later property gather.
    This still scans thin endpoint columns and does not claim indexed execution.
    """
    if (engine not in (Engine.PANDAS, Engine.CUDF) or len(ops) != 3
            or len(suffix) not in (2, 3) or _policy_is_active()):
        return None
    rows, limit = suffix[0], suffix[-1]
    if (not isinstance(rows, ASTCall) or rows.function != "rows"
            or not isinstance(limit, ASTCall) or limit.function != "limit"
            or rows.params.get("source") is not None or rows.params.get("alias_endpoints")
            or rows.params.get("alias_prefilters") or rows.params.get("table", "nodes") != "nodes"):
        return None
    count = limit.params.get("value")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    first, edge, last = ops
    if (not isinstance(first, ASTNode) or not isinstance(last, ASTNode)
            or not isinstance(edge, ASTEdge) or not edge.is_simple_single_hop()
            or edge.direction not in ("forward", "reverse")
            or first.filter_dict or last.filter_dict or first.query or last.query
            or edge.edge_match or edge.source_node_match or edge.destination_node_match
            or edge.source_node_query or edge.destination_node_query or edge.edge_query
            or edge.prune_to_endpoints or edge.include_zero_hop_seed):
        return None
    aliases = [op._name for op in ops if isinstance(op._name, str)]
    node_aliases = [first._name, last._name]
    if (any(alias is None for alias in node_aliases) or len(aliases) != len(set(aliases))
            or _INTERNAL.intersection(aliases)):
        return None
    nodes, edges = g._nodes, g._edges
    node, src, dst = g._node, g._source, g._destination
    if (nodes is None or edges is None or node is None or src is None or dst is None or src == dst
            or _frame_engine(nodes) != engine or _frame_engine(edges) != engine
            or count >= len(edges) or node not in nodes.columns
            or src not in edges.columns or dst not in edges.columns
            or set(aliases).intersection((node, src, dst))
            or _INTERNAL.intersection(nodes.columns) or _INTERNAL.intersection(edges.columns)
            or bool(nodes[node].isna().any()) or bool(nodes[node].duplicated().any())):
        return None
    if edge._name is None and set(edges.columns).difference((src, dst)).intersection(aliases):
        return None
    if engine == Engine.PANDAS and any(nodes[node].dtype.type is not edges[c].dtype.type for c in (src, dst)):
        return None  # canonical coercion/error handling owns mixed key representations
    if len(suffix) == 3:
        select = suffix[1]
        if not isinstance(select, ASTCall) or select.function != "select":
            return None
        items = select.params.get("items")
        if not isinstance(items, list) or not items:
            return None
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[1], str):
                return None
            try:
                expr = parse_expr(item[1])
            except GFQLExprParseError:
                return None
            if (not isinstance(expr, PropertyAccessExpr) or not isinstance(expr.value, Identifier)
                    or expr.value.name not in node_aliases or expr.property not in nodes.columns):
                return None

    position = generate_safe_column_name_from(
        "__gfql_limit_edge_position__", list(nodes.columns) + list(edges.columns) + aliases,
    )
    xp, _ = array_namespace(engine)
    thin_edges = edges[[src, dst]].assign(**{position: xp.arange(len(edges))})
    thin = drop_index(g.nodes(nodes[[node]]).edges(thin_edges, edge=position))
    state, _ = _RowPipelineAdapter(thin)._gfql_connected_bindings_state(ops)
    state = state.head(count)
    position_col = f"{edge._name}.{position}" if edge._name is not None else position
    if position_col not in state.columns:
        return None  # empty early walks retain the canonical schema builder
    gathered = take_rows(edges, col_to_array(state, position_col, engine), engine)
    payload = {
        (f"{edge._name}.{column}" if edge._name is not None else column): gathered[column]
        for column in edges.columns if column not in (src, dst)
    }
    # Canonical edge markers follow payload, including restored shadowed properties.
    marker = f"{edge._name}.{edge._name}" if edge._name is not None else None
    payload_order = [column for column in payload if column != marker]
    if marker is not None:
        payload_order.append(marker)
    columns = [
        name for column in state.columns
        for name in (payload_order if column == position_col else [column])
    ]
    state = RowPipelineMixin._gfql_assign_positional(state.drop(columns=[position_col]), **payload)
    state = state[list(dict.fromkeys(columns))]
    frames = {
        alias: _with_marker(nodes[nodes[node].isin(state[alias])], alias, engine)
        for alias in node_aliases if isinstance(alias, str)
    }
    from graphistry.compute.chain import _chain_impl
    empty = with_index_policy(drop_index(g.nodes(nodes.iloc[:0]).edges(edges.iloc[:0])), "off")
    empty_edges = _chain_impl(empty, list(ops), engine.value, False, None, None, None)._edges
    return IndexedBindingsState(state, frames, engine, 1, len(state), edge_template=empty_edges)


def record_binding_limit(engine: Engine) -> None:
    from graphistry.compute.gfql.index.api import _record, _trace_active

    if _trace_active():
        _record({
            "op": "binding_limit", "operation": "binding_limit", "seam": "unseeded_binding_limit",
            "engine": engine.value, "served": True, "path": "scan",
            "reason": "canonical thin bindings limited before property gathers",
        })
