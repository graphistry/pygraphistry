"""Resident-index source rows for native Polars point queries."""
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from graphistry.compute.gfql.index.registry import AdjacencyIndex
    import polars as pl

from graphistry.Plottable import Plottable
from graphistry.Engine import Engine
from graphistry.compute.typing import ArrayNamespace, DataFrameT, FilterDict, SeriesT
from graphistry.compute.gfql.identifiers import is_bare_identifier
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject, Direction
from graphistry.compute.chain_fast_paths import (
    _index_edge_rows, _index_node_rows, _record_native_seed_lane,
    _resident_node_id_index, _resident_seed_indexes, _seed_node_rows_from_index,
)
from graphistry.compute.gfql.lazy.engine.polars.membership import id_set, is_in_ids
from .admission import polars_seeded_lane_admits


def polars_point_rows_admits(ops: Sequence[ASTObject]) -> Optional[int]:
    boundary = 1 if len(ops) >= 2 and isinstance(ops[1], ASTCall) else 3 if len(ops) >= 4 else None
    if boundary is None or len(ops) > boundary + 3:
        return None
    first = ops[0]
    if not isinstance(first, ASTNode) or first.query is not None or not first.filter_dict:
        return None
    if boundary == 3 and not polars_seeded_lane_admits(ops[:3], ("forward", "reverse", "undirected")):
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
    joined = boundary == 3 and source is None and len(ops) > 4
    if joined:
        if not all(isinstance(op, ASTNode) and isinstance(op._name, str) for op in ops[:boundary:2]):
            return None
    elif not isinstance(source, str) or not any(isinstance(op, ASTNode) and op._name == source for op in ops[:boundary]):
        return None
    tail_ops = ops[boundary + 1:]
    if tail_ops:
        projection = tail_ops[0]
        if not isinstance(projection, ASTCall) or projection.function != "select" or set(projection.params) != {"items"}:
            return None
    if len(tail_ops) == 2 and not _is_plain_order_by(tail_ops[1]):
        return None
    return boundary


def _has_duplicate_ids(frame: "pl.DataFrame", node: str) -> bool:
    return frame.height > 1 and frame.get_column(node).n_unique() != frame.height


def _is_plain_order_by(op: ASTObject) -> bool:
    return isinstance(op, ASTCall) and op.function == "order_by" and set(op.params) == {"keys"}


def _gathered_edges_matching(
    g: Plottable,
    adj: "AdjacencyIndex",
    seed_ids: SeriesT,
    xp: ArrayNamespace,
    engine: Engine,
    edges: "pl.DataFrame",
    edge_match: Optional[FilterDict],
) -> "Optional[pl.DataFrame]":
    """The seed's incident edges that satisfy ``edge_match``, gathering only survivors.

    Candidates are row positions, so a predicate the category index can answer narrows them
    before any frame exists; anything it cannot answer falls back to gather-then-filter.
    """
    import polars as pl

    from graphistry.compute.chain_fast_paths import _index_edge_positions
    from graphistry.compute.gfql.index.array_bindings import _positions_via_category_index
    from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars

    if edge_match:
        positions = _index_edge_positions(adj, seed_ids, xp, preserve_input_order=True)
        if positions is not None:
            narrowed = _positions_via_category_index(
                g, "edges", edges, positions, edge_match, engine, xp,
            )
            if narrowed is not None:
                from graphistry.compute.gfql.index.engine_arrays import take_rows

                gathered = take_rows(edges, narrowed, engine)
                return gathered if isinstance(gathered, pl.DataFrame) else None
    gathered_edges = _index_edge_rows(adj, seed_ids, xp, engine, edges, preserve_input_order=True)
    if not isinstance(gathered_edges, pl.DataFrame):
        return None
    return filter_by_dict_polars(gathered_edges, edge_match)


def _joined_projection(
    seed: "pl.DataFrame", edges: "pl.DataFrame", tail: "pl.DataFrame",
    node: str, from_col: str, to_col: str, seed_alias: str, tail_alias: str,
    edge_alias: Optional[str], projection: ASTCall, edge_properties: FrozenSet[str],
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
        if alias == edge_alias and column not in edge_properties:
            return None  # the general path exposes neither an edge's endpoint columns nor generated ones
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
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import order_by_polars, select_polars
    from graphistry.compute.gfql.index.api import index_policy_may_serve

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
    scan = nid_ctx is None
    if scan and index_policy_may_serve(g):
        return None
    if nid_ctx is None:
        seed = filter_by_dict_polars(nodes, n0.filter_dict)
    else:
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
        if scan and seed.height > max(_SCAN_SEEDS_FLOOR, nodes.height // _SCAN_SEEDS_DIVISOR):
            return None
        if edge.direction == "undirected" and (nid_ctx is not None or source is not None):
            return None
        if nid_ctx is None:
            from_col, to_col = _oriented_endpoint_names(edges, nodes, edge.direction, src, dst)
            kept_edges = _orient_toward_tail(edges, edge.direction, src, dst, from_col, to_col,
                                             seed.get_column(node))
            kept_edges = filter_by_dict_polars(kept_edges, edge.edge_match)
            payload_edges = kept_edges.select(edges.columns)
            tail = _rows_with_ids(nodes, (node,), kept_edges.get_column(to_col))
            # A repeated node id binds differently on the general path; check every row before a filter hides a copy.
            if (_has_duplicate_ids(tail, node)
                    or _rows_with_ids(nodes, (node,), seed.get_column(node)).height != seed.height):
                return None
        else:
            nid, xp, engine = nid_ctx
            ctx = _resident_seed_indexes(g, nodes, edges, node, src, dst, edge.direction)
            if ctx is None:
                return None
            matching = _gathered_edges_matching(
                g, ctx[1], seed.get_column(node), xp, engine, edges, edge.edge_match,
            )
            if matching is None:
                return None
            payload_edges = kept_edges = matching
            from_col, to_col = (src, dst) if edge.direction == "forward" else (dst, src)
            gathered_tail = _index_node_rows(nid, kept_edges.get_column(to_col), xp, engine, nodes,
                                             preserve_input_order=True)
            if gathered_tail is None:
                return None
            tail = gathered_tail
        tail = filter_by_dict_polars(tail, tail_op.filter_dict)
        # A single gathered tail is the endpoint of the single surviving edge.
        if not (kept_edges.height == 1 and tail.height == 1):
            kept_edges = kept_edges.filter(is_in_ids(pl.col(to_col), tail.get_column(node)))
        if source is None:
            projection = ops[boundary + 1]
            assert isinstance(projection, ASTCall) and isinstance(n0._name, str) and isinstance(tail_op._name, str)
            selected_joined = _joined_projection(seed, kept_edges, tail, node, from_col, to_col,
                                                 n0._name, tail_op._name, edge._name, projection,
                                                 frozenset(payload_edges.columns) - {src, dst})
            if selected_joined is None:
                return None
            if len(ops) == boundary + 3:
                order = ops[-1]
                assert isinstance(order, ASTCall)
                ordered = order_by_polars(row_table(_RowPipelineAdapter(g), selected_joined), order.params["keys"])
                if ordered is None or not isinstance(ordered._nodes, pl.DataFrame):
                    return None
                selected_joined = ordered._nodes
            result_edges = payload_edges.clear()
            if edge._name is not None:
                result_edges = _with_true_alias(result_edges, edge._name)
            adapter = _RowPipelineAdapter(g.edges(result_edges))
            adapter._node = adapter._edge = adapter._source = adapter._destination = None
            _record_point_rows(nodes, n0, node, scan)
            return clear_row_exec_context(row_table(adapter, selected_joined))
        if source == tail_op._name:
            selected = tail
        elif seed.height == 1 and kept_edges.height:
            selected = seed
        else:
            selected = seed.filter(is_in_ids(pl.col(node), kept_edges.get_column(from_col)))
        if seed.height == tail.height == selected.height == 1 and kept_edges.height:
            selected = _with_singleton_aliases(selected, seed, tail, node, n0._name, tail_op._name)
        else:
            selected = selected.with_columns([
                is_in_ids(pl.col(node), kept_edges.get_column(endpoint)).fill_null(False).alias(alias)
                for alias, endpoint in ((n0._name, from_col), (tail_op._name, to_col)) if alias is not None
            ])
        if edge._name is not None:
            kept_edges = _with_true_alias(kept_edges, edge._name)
    out = row_table(_RowPipelineAdapter(g.edges(kept_edges)), selected)
    for call in ops[boundary + 1:]:
        assert isinstance(call, ASTCall)
        stepped = (select_polars(out, call.params["items"]) if call.function == "select"
                   else order_by_polars(out, call.params["keys"]))
        if stepped is None:
            return None
        out = stepped
    _record_point_rows(nodes, n0, node, scan, hop_count=boundary // 2)
    return clear_row_exec_context(out)


def _record_point_rows(nodes: "pl.DataFrame", n0: ASTNode, node: str, scan: bool, hop_count: int = 1) -> None:
    if scan:
        from graphistry.compute.gfql.index.api import record_fast_path_decision
        record_fast_path_decision(path="point_rows_scan", engine=Engine.POLARS, served=True,
                                  reason="seed, edges and tail each resolved by one scan")
        return
    _record_native_seed_lane(nodes, seam="point_rows", reason="served", hop_count=hop_count,
                             public_seed_scan=node not in (n0.filter_dict or {}))


_EQUALITY_IDS_MAX = 8
_SCAN_SEEDS_FLOOR = 1024
_SCAN_SEEDS_DIVISOR = 20


def _rows_with_ids(frame: "pl.DataFrame", columns: Tuple[str, ...], ids: SeriesT) -> "pl.DataFrame":
    """Rows of ``frame`` where any of ``columns`` is in ``ids``, in frame order.

    Up to ``_EQUALITY_IDS_MAX`` integer, string or boolean ids compare by equality; every other id
    dtype (a Python value would lose nanoseconds or compare differently) and more ids semi-join per column.
    """
    import polars as pl
    from graphistry.compute.util import generate_safe_column_name_from
    unique = ids.drop_nulls().unique(maintain_order=True)
    if not unique.len():
        return frame.clear()
    exact_in_python = unique.dtype.is_integer() or unique.dtype in (pl.String, pl.Boolean)
    if exact_in_python and unique.len() <= _EQUALITY_IDS_MAX:
        values = unique.to_list()
        return frame.filter(pl.any_horizontal([pl.col(column) == value for column in columns for value in values]))
    if len(columns) == 1:
        return frame.join(pl.DataFrame({columns[0]: unique}), on=columns[0], how="semi", maintain_order="left")
    order = generate_safe_column_name_from("__gfql_point_rows_order__", list(frame.columns))
    numbered = frame.with_row_index(order)
    hits = [numbered.join(pl.DataFrame({column: unique}), on=column, how="semi") for column in columns]
    return pl.concat(hits).unique(subset=[order]).sort(order).drop(order)


def _oriented_endpoint_names(
    edges: "pl.DataFrame", nodes: "pl.DataFrame", direction: Direction, src: str, dst: str,
) -> Tuple[str, str]:
    if direction == "forward":
        return src, dst
    if direction == "reverse":
        return dst, src
    from graphistry.compute.util import generate_safe_column_name_from
    taken = list(edges.columns) + list(nodes.columns)
    from_col = generate_safe_column_name_from("__gfql_point_from__", taken)
    return from_col, generate_safe_column_name_from("__gfql_point_to__", taken + [from_col])


def _orient_toward_tail(
    edges: "pl.DataFrame", direction: Direction, src: str, dst: str, from_col: str, to_col: str, ids: SeriesT,
) -> "pl.DataFrame":
    """Edges leaving ``ids`` in edge order; an undirected edge once per seed-side orientation.

    A self-loop is one binding, so it keeps only its forward orientation.
    """
    import polars as pl
    if direction != "undirected":
        return _rows_with_ids(edges, (from_col,), ids)
    from graphistry.compute.gfql.lazy import collect
    from graphistry.compute.util import generate_safe_column_name_from
    taken = list(edges.columns) + [from_col, to_col]
    order = generate_safe_column_name_from("__gfql_point_order__", taken)
    side = generate_safe_column_name_from("__gfql_point_side__", taken + [order])
    touching = _rows_with_ids(edges, (src, dst), ids).lazy().with_row_index(order)
    seed_ids = id_set(ids)
    forward = touching.filter(pl.col(src).is_in(seed_ids)).with_columns(
        pl.col(src).alias(from_col), pl.col(dst).alias(to_col), pl.lit(0, pl.UInt8).alias(side))
    backward = touching.filter(pl.col(dst).is_in(seed_ids) & (pl.col(src) != pl.col(dst))).with_columns(
        pl.col(dst).alias(from_col), pl.col(src).alias(to_col), pl.lit(1, pl.UInt8).alias(side))
    return collect(pl.concat([forward, backward]).sort([order, side]).drop([order, side]))
