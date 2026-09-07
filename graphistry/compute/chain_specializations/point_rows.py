"""Resident-index point queries that produce a row table."""
from typing import List, Literal, Optional, Tuple

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject
from graphistry.compute.chain_fast_paths import (
    _index_edge_rows, _index_node_rows, _record_native_seed_lane,
    _resident_node_id_index, _resident_seed_indexes, _seed_node_rows_from_index,
    _seeded_scalar_filters, _tag_fast_path_alias_frames, _verify_scalar_filters_on_hit,
)
from graphistry.compute.typing import DataFrameT
from .admission import point_rows_admits


def _point_hop_rows(
    g: Plottable, n0: ASTNode, edge: ASTEdge, n2: ASTNode,
    table: Literal["nodes", "edges"], source: str,
) -> Optional[Tuple[DataFrameT, DataFrameT]]:
    nodes, edges = g._nodes, g._edges
    node, src, dst = g._node, g._source, g._destination
    if nodes is None or edges is None or node is None or src is None or dst is None:
        return None
    seed_filter = _seeded_scalar_filters(n0.filter_dict, nodes)
    tail_filter = _seeded_scalar_filters(n2.filter_dict, nodes)
    edge_filter = _seeded_scalar_filters(edge.edge_match, edges)
    if not seed_filter or tail_filter is None or edge_filter is None:
        return None
    ctx = _resident_seed_indexes(g, nodes, edges, node, src, dst, edge.direction)
    if ctx is None:
        return None
    nid, adj, xp, engine = ctx
    indexed_seed = _seed_node_rows_from_index(g, nodes, seed_filter, node, (nid, xp, engine), n0.filter_dict)
    if indexed_seed is None:
        return None
    seed, _ = indexed_seed
    matched_edges = _index_edge_rows(adj, seed[node], xp, engine, edges)
    if matched_edges is None:
        return None
    if edge_filter:
        matched_edges = _verify_scalar_filters_on_hit(matched_edges, edge_filter, engine)
        if matched_edges is None:
            return None
    from_col, to_col = (src, dst) if edge.direction == "forward" else (dst, src)
    tail = _index_node_rows(nid, matched_edges[to_col], xp, engine, nodes)
    if tail is None:
        return None
    if tail_filter:
        tail = _verify_scalar_filters_on_hit(tail, tail_filter, engine)
        if tail is None:
            return None
    matched_edges = matched_edges[matched_edges[to_col].isin(tail[node].dropna())]
    if table == "nodes" and source == n0._name:
        selected = seed[seed[node].isin(matched_edges[from_col].dropna())]
    else:
        selected = tail
    original = matched_edges if table == "edges" else selected
    tagged_nodes, tagged_edges = _tag_fast_path_alias_frames(
        selected, matched_edges, n0._name, edge._name, n2._name, src, dst, node, edge.direction)
    if g._edge is not None and tagged_edges.columns[0] != g._edge:
        tagged_edges = tagged_edges[[g._edge, *[c for c in tagged_edges.columns if c != g._edge]]]
    result = tagged_edges if table == "edges" else tagged_nodes
    return _restore_point_source(result, original, source), tagged_edges


def _restore_point_source(result: DataFrameT, original: DataFrameT, source: str) -> DataFrameT:
    if source not in original.columns or str(original[source].dtype).startswith("bool"):
        return result
    from graphistry.compute.gfql.identifiers import shadow_restore_column
    out = result.reset_index(drop=True)
    out[shadow_restore_column(source)] = original[source].reset_index(drop=True)
    return out


def _try_point_rows(
    g: Plottable, ops: List[ASTObject], engine: Engine,
    start_nodes: Optional[DataFrameT] = None, validate_schema: bool = True,
) -> Optional[Plottable]:
    boundary = point_rows_admits(ops, engine, start_nodes)
    if boundary is None or g._nodes is None or g._node is None:
        return None
    from graphistry.compute.gfql.row.pipeline import _RowPipelineAdapter
    from graphistry.compute.gfql.row.frame_ops import row_table
    from graphistry.compute.gfql.exec_context import clear_row_exec_context
    from graphistry.compute.validate.validate_schema import validate_chain_schema

    n0, row = ops[0], ops[boundary]
    assert isinstance(n0, ASTNode) and isinstance(row, ASTCall)
    table, source = row.params["table"], row.params["source"]
    if validate_schema:
        from graphistry.compute.chain import Chain
        Chain(ops).validate(collect_all=False)
        validate_chain_schema(g, ops, collect_all=False)
    adapter = _RowPipelineAdapter(g)
    adapter._gfql_rows_base_graph = g
    if adapter._edges is not None and g._edge is not None and adapter._edges.columns[0] != g._edge:
        adapter._edges = adapter._edges.iloc[:0][[g._edge, *[c for c in adapter._edges.columns if c != g._edge]]]
    if boundary == 1:
        filters = _seeded_scalar_filters(n0.filter_dict, g._nodes)
        if not filters:
            return None
        ctx = _resident_node_id_index(g, g._nodes, g._node)
        indexed_seed = _seed_node_rows_from_index(g, g._nodes, filters, g._node, ctx, n0.filter_dict)
        if indexed_seed is None:
            return None
        selected, _ = indexed_seed
        original = selected
        collides = source in selected.columns
        if not collides and selected.columns[0] == g._node:
            selected = selected.reset_index(drop=True)
            selected.insert(1, source, True)
        else:
            selected = selected.drop(columns=[source]) if collides else selected
            selected = selected.assign(**{source: True})
            rest = [c for c in selected.columns if c not in (g._node, source)]
            selected = selected[[g._node, *rest, source] if collides else [g._node, source, *rest]]
        selected = _restore_point_source(selected, original, source)
    else:
        edge, n2 = ops[1:3]
        assert isinstance(edge, ASTEdge) and isinstance(n2, ASTNode)
        gathered = _point_hop_rows(g, n0, edge, n2, table, source)
        if gathered is None:
            return None
        selected, adapter._edges = gathered
    out = row_table(adapter, selected)
    if len(ops) > boundary + 1:
        projection = ops[-1]
        assert isinstance(projection, ASTCall)
        out = _RowPipelineAdapter(out).select(projection.params["items"])
    _record_native_seed_lane(g._nodes, seam="point_rows", reason="served", hop_count=boundary // 2,
                             public_seed_scan=g._node not in (n0.filter_dict or {}))
    return clear_row_exec_context(out)
