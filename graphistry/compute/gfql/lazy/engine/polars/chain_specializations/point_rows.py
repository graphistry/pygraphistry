"""Resident-index source rows for native Polars point queries."""
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    import polars as pl

from graphistry.Plottable import Plottable
from graphistry.compute.typing import DataFrameT
from graphistry.compute.gfql.identifiers import is_bare_identifier
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
    joined = boundary == 3 and source is None and len(ops) == 5
    if joined:
        if not all(isinstance(op, ASTNode) and isinstance(op._name, str) for op in ops[:boundary:2]):
            return None
    elif not isinstance(source, str) or not any(isinstance(op, ASTNode) and op._name == source for op in ops[:boundary]):
        return None
    if len(ops) > boundary + 1:
        projection = ops[-1]
        if not isinstance(projection, ASTCall) or projection.function != "select" or set(projection.params) != {"items"}:
            return None
    return boundary


def _joined_projection(
    seed: "pl.DataFrame", edges: "pl.DataFrame", tail: "pl.DataFrame",
    node: str, from_col: str, to_col: str, seed_alias: str, tail_alias: str,
    edge_alias: Optional[str], projection: ASTCall,
) -> "Optional[pl.DataFrame]":
    import polars as pl
    from graphistry.compute.gfql.lazy import collect
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import _select_emits_temporal_constructor_text
    if ((seed.height > 1 and seed.get_column(node).n_unique() != seed.height)
            or (tail.height > 1 and tail.get_column(node).n_unique() != tail.height)):
        return None
    items = projection.params.get("items")
    if not isinstance(items, list) or not items:
        return None
    frames = {seed_alias: seed, tail_alias: tail}
    if edge_alias is not None:
        frames[edge_alias] = edges
    properties: Dict[str, List[pl.Expr]] = {alias: [] for alias in frames}
    outputs = []
    singleton = seed.height == 1 and tail.height == 1
    singleton_columns = []
    for index, item in enumerate(items):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        output, expr = item
        if not isinstance(output, str) or not isinstance(expr, str):
            return None
        parts = expr.split(".")
        if len(parts) != 2 or not all(is_bare_identifier(part) for part in parts):
            return None
        alias, column = parts
        if alias not in frames or column not in frames[alias].columns:
            return None
        if singleton:
            series = frames[alias].get_column(column)
            if alias != edge_alias:
                series = series.new_from_index(0, edges.height)
            singleton_columns.append(series.alias(output))
            continue
        temporary = f"_value_{index}"
        properties[alias].append(pl.col(column).alias(temporary))
        outputs.append(pl.col(temporary).alias(output))
    if len({item[0] for item in items}) != len(items):
        return None
    if singleton:
        result = pl.DataFrame(singleton_columns)
        return None if _select_emits_temporal_constructor_text(result) else result
    left = seed.lazy().select(pl.col(node).alias("_seed"), *properties[seed_alias]).with_row_index("_seed_order")
    step = edges.lazy().select(
        pl.col(from_col).alias("_seed"), pl.col(to_col).alias("_tail"),
        *([] if edge_alias is None else properties[edge_alias]),
    ).with_row_index("_edge_order")
    right = tail.lazy().select(pl.col(node).alias("_tail"), *properties[tail_alias])
    result = collect(left.join(step, on="_seed", how="inner").join(right, on="_tail", how="inner")
                     .sort(["_seed_order", "_edge_order"]).select(outputs))
    return None if _select_emits_temporal_constructor_text(result) else result


def _with_true_alias(frame: "pl.DataFrame", alias: str) -> "pl.DataFrame":
    """Append a Boolean alias without changing the input frame."""
    import polars as pl
    out = frame.clone()
    out.insert_column(out.width, pl.Series(alias, [True]).new_from_index(0, frame.height))
    return out


def _with_singleton_aliases(
    frame: "pl.DataFrame", seed: "pl.DataFrame", tail: "pl.DataFrame", node: str,
    first: Optional[str], last: Optional[str],
) -> "pl.DataFrame":
    """Attach alias flags to one selected node from single-row endpoints."""
    import polars as pl
    out = frame.clone()
    for alias, endpoint in ((first, seed), (last, tail)):
        if alias is not None:
            matches = frame.get_column(node).equals(endpoint.get_column(node), null_equal=False)
            out.insert_column(out.width, pl.Series(alias, [matches]))
    return out


def _try_point_rows_polars(g: Plottable, ops: List[ASTObject], start_nodes: Optional[DataFrameT] = None) -> Optional[Plottable]:
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
    source = row.params.get("source")
    nid_ctx = _resident_node_id_index(g, nodes, node)
    if nid_ctx is None:
        return None
    nid, xp, engine = nid_ctx
    seed_result = _seed_node_rows_from_index(g, nodes, n0.filter_dict, node, nid_ctx, n0.filter_dict)
    if seed_result is None:
        return None
    seed, _ = seed_result
    if boundary == 1:
        assert isinstance(source, str)
        selected = _with_true_alias(seed, source)
        kept_edges = edges.clear()
    else:
        edge, tail_op = ops[1:3]
        assert isinstance(edge, ASTEdge) and isinstance(tail_op, ASTNode)
        node_dtype = nodes.get_column(node).dtype
        if node_dtype != edges.get_column(src).dtype or node_dtype != edges.get_column(dst).dtype:
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
        # A single gathered tail is the endpoint of the single surviving edge.
        if not (kept_edges.height == 1 and tail.height == 1):
            kept_edges = kept_edges.filter(pl.col(to_col).is_in(tail.get_column(node).implode()))
        if source is None:
            projection = ops[-1]
            assert isinstance(projection, ASTCall) and isinstance(n0._name, str) and isinstance(tail_op._name, str)
            selected_joined = _joined_projection(seed, kept_edges, tail, node, from_col, to_col,
                                                 n0._name, tail_op._name, edge._name, projection)
            if selected_joined is None:
                return None
            if edge._name is not None:
                kept_edges = _with_true_alias(kept_edges, edge._name)
            adapter = _RowPipelineAdapter(g.edges(kept_edges))
            adapter._node = adapter._edge = adapter._source = adapter._destination = None
            _record_native_seed_lane(nodes, seam="point_rows", reason="served", hop_count=1,
                                     public_seed_scan=node not in n0.filter_dict)
            return clear_row_exec_context(row_table(adapter, selected_joined))
        if source == tail_op._name:
            selected = tail
        elif seed.height == 1 and kept_edges.height:
            selected = seed
        else:
            selected = seed.filter(pl.col(node).is_in(kept_edges.get_column(from_col).implode()))
        if seed.height == tail.height == selected.height == 1 and kept_edges.height:
            selected = _with_singleton_aliases(selected, seed, tail, node, n0._name, tail_op._name)
        else:
            selected = selected.with_columns([
                pl.col(node).is_in(kept_edges.get_column(endpoint).implode()).fill_null(False).alias(alias)
                for alias, endpoint in ((n0._name, from_col), (tail_op._name, to_col)) if alias is not None
            ])
        if edge._name is not None:
            kept_edges = _with_true_alias(kept_edges, edge._name)
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
