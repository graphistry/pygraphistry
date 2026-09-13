"""Resident-index point queries that produce a row table."""
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.api.extensions import ExtensionArray

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject
from graphistry.compute.chain_fast_paths import (
    _index_edge_rows, _index_node_rows, _record_native_seed_lane,
    _resident_node_id_index, _resident_seed_indexes, _seed_node_rows_from_index,
    _seeded_scalar_filters, _tag_fast_path_alias_frames, _verify_scalar_filters_on_hit,
)
from graphistry.compute.typing import DataFrameT, SeriesT
from graphistry.compute.util import generate_safe_column_name_from
from graphistry.compute.gfql.identifiers import is_bare_identifier
from graphistry.compute.gfql.expr_parser import (
    FunctionCall, GFQLExprParseError, Identifier, PropertyAccessExpr, parse_expr,
)
from .admission import point_rows_admits



def _point_hop_rows(
    g: Plottable, n0: ASTNode, edge: ASTEdge, n2: ASTNode,
) -> Optional[Tuple[DataFrameT, DataFrameT, DataFrameT]]:
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
    matched_edges = _index_edge_rows(adj, seed[node], xp, engine, edges, preserve_input_order=True)
    if matched_edges is None:
        return None
    if edge_filter:
        matched_edges = _verify_scalar_filters_on_hit(matched_edges, edge_filter, engine)
        if matched_edges is None:
            return None
    to_col = dst if edge.direction == "forward" else src
    tail = _index_node_rows(nid, matched_edges[to_col], xp, engine, nodes)
    if tail is None:
        return None
    if tail_filter:
        tail = _verify_scalar_filters_on_hit(tail, tail_filter, engine)
        if tail is None:
            return None
    if len(tail) == 0:
        matched_edges = matched_edges.iloc[:0]
    elif len(matched_edges) != 1:
        matched_edges = matched_edges[matched_edges[to_col].isin(tail[node].dropna())]
    return seed, tail, matched_edges


def _point_column(
    frame: DataFrameT, expr: str, source: str, aliases: Sequence[str],
) -> Optional[str]:
    if expr in frame.columns:
        return None if expr in aliases else expr
    if (expr.startswith(source + ".") and is_bare_identifier(source)
            and is_bare_identifier(expr[len(source) + 1:])
            and expr[len(source) + 1:] in frame.columns):
        column = expr[len(source) + 1:]
        if column in aliases and (column != source or str(frame[column].dtype).startswith("bool")):
            return None
        return column
    return None


def _project_point_columns(
    frame: DataFrameT, projection: ASTCall, source: str, aliases: Sequence[str],
) -> Optional[DataFrameT]:
    items = projection.params.get("items")
    if not isinstance(items, list):
        return None
    projected: Dict[str, SeriesT] = {}
    for item in items:
        if isinstance(item, str):
            alias, expr = item, item
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            alias, expr = item
        else:
            return None
        if not isinstance(alias, str) or not alias or not isinstance(expr, str):
            return None
        column = _point_column(frame, expr, source, aliases)
        if column is not None:
            projected[alias] = frame[column]
            continue
        # Reuse the evaluator's cached grammar; this route supports two bare properties.
        if not expr.lower().startswith("coalesce") or "`" in expr:
            return None
        try:
            coalesce = parse_expr(expr)
        except GFQLExprParseError:
            return None
        if (not isinstance(coalesce, FunctionCall) or coalesce.name.lower() != "coalesce"
                or coalesce.distinct or len(coalesce.args) != 2):
            return None
        terms: List[str] = []
        for argument in coalesce.args:
            if (not isinstance(argument, PropertyAccessExpr)
                    or not isinstance(argument.value, Identifier)
                    or not is_bare_identifier(argument.value.name)
                    or not is_bare_identifier(argument.property)):
                return None
            terms.append(f"{argument.value.name}.{argument.property}")
        left_col, right_col = (_point_column(frame, term, source, aliases) for term in terms)
        if left_col is None or right_col is None:
            return None
        left, right = frame[left_col], frame[right_col]
        projected[alias] = left.where(~left.isna(), right)
    if isinstance(frame, pd.DataFrame):
        return pd.DataFrame(projected, index=frame.index)
    return frame.assign(**projected)[list(projected)]


def _project_joined_point_columns(
    g: Plottable, seed: DataFrameT, tail: DataFrameT, edges: DataFrameT,
    n0: ASTNode, edge: ASTEdge, n2: ASTNode, projection: ASTCall,
) -> Optional[DataFrameT]:
    if g._node is None or g._source is None or g._destination is None:
        return None
    aliases = [op._name for op in (n0, edge, n2) if op._name is not None]
    if n0._name is None or n2._name is None or any(
        alias in seed.columns or alias in edges.columns for alias in aliases
    ):
        return None
    items = projection.params.get("items")
    if not isinstance(items, list) or not items:
        return None
    from_col, to_col = ((g._source, g._destination) if edge.direction == "forward"
                        else (g._destination, g._source))
    if len(seed) > 1 and len(edges) > 1:
        seed_order = generate_safe_column_name_from("__gfql_seed_order__", list(seed.columns) + list(edges.columns))
        edge_order = generate_safe_column_name_from("__gfql_edge_order__", list(edges.columns) + [seed_order])
        ranks = seed[[g._node]].assign(**{seed_order: range(len(seed))}).set_index(g._node)[seed_order]
        positions = ranks.reindex(edges[from_col]).reset_index(drop=True)
        ordered = edges.reset_index(drop=True).assign(**{seed_order: positions, edge_order: range(len(edges))})
        edges = ordered.sort_values([seed_order, edge_order]).drop(columns=[seed_order, edge_order])
    frames = {n0._name: (seed, from_col), n2._name: (tail, to_col)}
    aligned: Dict[str, DataFrameT] = {}
    pandas_positions: Dict[str, List[int]] = {}
    pandas_columns: Dict[str, ExtensionArray] = {}
    projected: Dict[str, SeriesT] = {}
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            return None
        output, expression = item
        if not isinstance(output, str) or not output or not isinstance(expression, str):
            return None
        parts = expression.split(".")
        if len(parts) != 2 or not all(is_bare_identifier(part) for part in parts):
            return None
        alias, column = parts
        if alias == edge._name and column in edges.columns:
            if isinstance(edges, pd.DataFrame):
                pandas_columns[output] = edges[column].array
                continue
            values = edges[column].reset_index(drop=True)
        elif alias in frames and column in frames[alias][0].columns:
            frame, endpoint = frames[alias]
            # Align properties to edge rows to retain repeated bindings.
            if isinstance(frame, pd.DataFrame):
                if alias not in pandas_positions:
                    if len(frame) == 1:
                        pandas_positions[alias] = [0] * len(edges)
                    else:
                        positions = pd.Index(frame[g._node]).get_indexer(edges[endpoint])
                        if (positions < 0).any():
                            return None
                        pandas_positions[alias] = positions.tolist()
                pandas_columns[output] = frame[column].array.take(pandas_positions[alias])
                continue
            if alias not in aligned:
                aligned[alias] = frame.set_index(g._node, drop=False).reindex(edges[endpoint]).reset_index(drop=True)
            values = aligned[alias][column]
        else:
            return None
        projected[output] = values
    if isinstance(seed, pd.DataFrame):
        return pd.DataFrame(pandas_columns)
    return edges.iloc[:, :0].reset_index(drop=True).assign(**projected)


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
    table, source = row.params["table"], row.params.get("source")
    if validate_schema:
        from graphistry.compute.chain import Chain
        Chain(ops)
        validate_chain_schema(g, ops, collect_all=False)
    adapter = _RowPipelineAdapter(g)
    adapter._gfql_rows_base_graph = g
    if adapter._edges is not None and g._edge is not None and adapter._edges.columns[0] != g._edge:
        adapter._edges = adapter._edges.iloc[:0][[g._edge, *[c for c in adapter._edges.columns if c != g._edge]]]
    projection = ops[-1] if len(ops) > boundary + 1 else None
    assert projection is None or isinstance(projection, ASTCall)
    aliases = [op._name for op in ops[:boundary] if op._name is not None]
    edge_alias = None
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
    else:
        edge, n2 = ops[1:3]
        assert isinstance(edge, ASTEdge) and isinstance(n2, ASTNode)
        gathered = _point_hop_rows(g, n0, edge, n2)
        if gathered is None:
            return None
        seed, tail, adapter._edges = gathered
        if source is None:
            assert projection is not None
            joined = _project_joined_point_columns(g, seed, tail, adapter._edges, n0, edge, n2, projection)
            if joined is None:
                return None
            assert g._source is not None and g._destination is not None
            _, adapter._edges = _tag_fast_path_alias_frames(
                g._nodes.iloc[:0], adapter._edges.iloc[:0], None, edge._name, None,
                g._source, g._destination, g._node, edge.direction)
            if g._edge is not None:
                adapter._edges = adapter._edges[[g._edge, *[c for c in adapter._edges.columns if c != g._edge]]]
            adapter._node = adapter._edge = adapter._source = adapter._destination = None
            _record_native_seed_lane(g._nodes, seam="point_rows", reason="served", hop_count=1,
                                     public_seed_scan=g._node not in (n0.filter_dict or {}))
            return clear_row_exec_context(row_table(adapter, joined))
        from_col = g._source if edge.direction == "forward" else g._destination
        selected = seed[seed[g._node].isin(adapter._edges[from_col].dropna())] if source == n0._name else tail
        edge_alias = edge._name
        original = adapter._edges if table == "edges" else selected
    assert isinstance(source, str)
    projected = _project_point_columns(original, projection, source, aliases) if projection is not None else None
    if projected is not None:
        adapter._node = g._node if g._node in original.columns else None
        adapter._edge = g._edge if g._edge in original.columns else None
        if edge_alias is not None:
            assert adapter._edges is not None and g._source is not None and g._destination is not None
            _, adapter._edges = _tag_fast_path_alias_frames(
                g._nodes.iloc[:0], adapter._edges.iloc[:0], None, edge_alias, None,
                g._source, g._destination, g._node, "forward")
        selected = projected
    elif boundary == 1:
        collides = source in selected.columns
        if not collides:
            selected = selected.reset_index(drop=True)
            if selected.columns[0] != g._node:
                selected.insert(0, g._node, selected.pop(g._node))
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
        assert adapter._edges is not None and g._source is not None and g._destination is not None
        tagged_nodes, adapter._edges = _tag_fast_path_alias_frames(
            selected, adapter._edges, n0._name, edge_alias, n2._name,
            g._source, g._destination, g._node, edge.direction)
        selected = adapter._edges if table == "edges" else tagged_nodes
        selected = _restore_point_source(selected, original, source)
    if adapter._edges is not None and g._edge is not None and adapter._edges.columns[0] != g._edge:
        adapter._edges = adapter._edges[[g._edge, *[c for c in adapter._edges.columns if c != g._edge]]]
        if projected is None and table == "edges":
            selected = selected[[g._edge, *[c for c in selected.columns if c != g._edge]]]
    out = row_table(adapter, selected)
    if projection is not None and projected is None:
        out = _RowPipelineAdapter(out).select(projection.params["items"])
    _record_native_seed_lane(g._nodes, seam="point_rows", reason="served", hop_count=boundary // 2,
                             public_seed_scan=g._node not in (n0.filter_dict or {}))
    return clear_row_exec_context(out)
