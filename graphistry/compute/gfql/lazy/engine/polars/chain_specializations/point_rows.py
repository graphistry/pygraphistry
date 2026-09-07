"""Resident-index source rows for native Polars point queries."""
from typing import List, Optional, Sequence

from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject
from graphistry.compute.chain_fast_paths import (
    _index_edge_rows, _index_node_rows, _record_native_seed_lane,
    _resident_node_id_index, _resident_seed_indexes, _seed_node_rows_from_index,
)
from .admission import polars_seeded_lane_admits


def polars_point_rows_admits(ops: Sequence[ASTObject]) -> Optional[int]:
    boundary = 1 if len(ops) in (2, 3) else 3 if len(ops) in (4, 5) else None
    if boundary is None:
        return None
    first = ops[0]
    if not isinstance(first, ASTNode) or first.query is not None or not first.filter_dict:
        return None
    if boundary == 3 and not polars_seeded_lane_admits(ops[:3]):
        return None
    for op in ops[:boundary]:
        filters = op.filter_dict if isinstance(op, ASTNode) else op.edge_match if isinstance(op, ASTEdge) else None
        if filters and any(not isinstance(value, (str, int, float, bool)) for value in filters.values()):
            return None
    row = ops[boundary]
    if (not isinstance(row, ASTCall) or row.function != "rows"
            or set(row.params) - {"table", "source"} or row.params.get("table") != "nodes"):
        return None
    source = row.params.get("source")
    if not isinstance(source, str) or not any(isinstance(op, ASTNode) and op._name == source for op in ops[:boundary]):
        return None
    if len(ops) > boundary + 1:
        projection = ops[-1]
        if not isinstance(projection, ASTCall) or projection.function != "select" or set(projection.params) != {"items"}:
            return None
    return boundary


def _try_point_rows_polars(g: Plottable, ops: List[ASTObject], start_nodes: Optional[object] = None) -> Optional[Plottable]:
    import polars as pl
    from graphistry.compute.gfql.index.bindings import _policy_is_active
    from graphistry.compute.gfql.lazy import active_target, ExecutionTarget
    from graphistry.compute.gfql.exec_context import clear_row_exec_context
    from graphistry.compute.gfql.row.frame_ops import row_table
    from graphistry.compute.gfql.row.pipeline import _RowPipelineAdapter
    from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import select_polars

    boundary = polars_point_rows_admits(ops)
    if boundary is None or start_nodes is not None or _policy_is_active() or active_target() == ExecutionTarget.GPU:
        return None
    nodes, edges = g._nodes, g._edges
    node, src, dst = g._node, g._source, g._destination
    if not isinstance(nodes, pl.DataFrame) or not isinstance(edges, pl.DataFrame) or node is None or src is None or dst is None:
        return None
    aliases = [op._name for op in ops[:boundary] if op._name is not None]
    if len(aliases) != len(set(aliases)) or any(alias in nodes.columns or alias in edges.columns for alias in aliases):
        return None
    n0, row = ops[0], ops[boundary]
    assert isinstance(n0, ASTNode) and isinstance(row, ASTCall) and n0.filter_dict
    source = row.params["source"]
    nid_ctx = _resident_node_id_index(g, nodes, node)
    if nid_ctx is None:
        return None
    nid, xp, engine = nid_ctx
    seed_result = _seed_node_rows_from_index(g, nodes, n0.filter_dict, node, nid_ctx, n0.filter_dict)
    if seed_result is None:
        return None
    seed, _ = seed_result
    if boundary == 1:
        selected = seed.with_columns(pl.lit(True).alias(source))
        kept_edges = edges.clear()
    else:
        edge, tail_op = ops[1:3]
        assert isinstance(edge, ASTEdge) and isinstance(tail_op, ASTNode)
        if nodes.schema[node] != edges.schema[src] or nodes.schema[node] != edges.schema[dst]:
            return None
        ctx = _resident_seed_indexes(g, nodes, edges, node, src, dst, edge.direction)
        if ctx is None:
            return None
        gathered_edges = _index_edge_rows(ctx[1], seed.get_column(node), xp, engine, edges, preserve_input_order=True)
        if not isinstance(gathered_edges, pl.DataFrame):
            return None
        kept_edges = filter_by_dict_polars(gathered_edges, edge.edge_match)
        from_col, to_col = (src, dst) if edge.direction == "forward" else (dst, src)
        tail = _index_node_rows(nid, kept_edges.get_column(to_col), xp, engine, nodes, preserve_input_order=True)
        if tail is None:
            return None
        tail = filter_by_dict_polars(tail, tail_op.filter_dict)
        kept_edges = kept_edges.filter(pl.col(to_col).is_in(tail.get_column(node).implode()))
        selected = tail if source == tail_op._name else seed.filter(pl.col(node).is_in(kept_edges.get_column(from_col).implode()))
        selected = selected.with_columns([
            pl.col(node).is_in(kept_edges.get_column(endpoint).implode()).fill_null(False).alias(alias)
            for alias, endpoint in ((n0._name, from_col), (tail_op._name, to_col)) if alias is not None
        ])
        if edge._name is not None:
            kept_edges = kept_edges.with_columns(pl.lit(True).alias(edge._name))
    out = row_table(_RowPipelineAdapter(g.edges(kept_edges)), selected)
    if len(ops) > boundary + 1:
        projection = ops[-1]
        assert isinstance(projection, ASTCall)
        projected = select_polars(out, projection.params["items"])
        if projected is None:
            return None
        out = projected
    _record_native_seed_lane(nodes, seam="point_rows", reason="served", hop_count=boundary // 2,
                             public_seed_scan=node not in n0.filter_dict)
    return clear_row_exec_context(out)
