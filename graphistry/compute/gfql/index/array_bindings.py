"""Array-side execution of a seeded fixed-hop bindings pattern.

The frame-based builder in ``bindings.py`` materializes a node/edge frame at every
hop. For a seeded pattern the hop frames are small and the per-frame work is fixed
cost, so this module keeps the whole traversal in index arrays — a path bag is a
set of ROW POSITIONS per alias — and materializes one frame at the end, for the
projected columns only.

It serves the same patterns as ``_try_indexed_connected_bindings_state`` minus
undirected hops, on Polars only, and declines (``None``) everything else. Both
paths read the same indexes and apply the same unchanged filters, so a decline
costs correctness nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import polars as pl

    from graphistry.compute.ast import ASTObject

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.compute.typing import ArrayLike, ArrayNamespace, DataFrameT

from . import bindings as frame_bindings
from .api import get_index_policy, get_registry
from .cost import cost_gate_frac
from .engine_arrays import array_namespace, col_to_array, take_rows
from .lookup import lookup_degree, lookup_edge_rows, lookup_node_rows
from .registry import AdjacencyIndex, ColStatsRole, NodeIdIndex, NODE_ID


@dataclass(frozen=True)
class ArrayPathBag:
    """One row per matched path, as row positions into the node and edge frames."""

    node_rows: Dict[str, ArrayLike]
    edge_rows: Dict[str, ArrayLike]
    height: int
    hop_count: int
    estimated_rows: int


def _aligned_node_rows(
    index: NodeIdIndex, ids: ArrayLike, xp: ArrayNamespace,
) -> Optional[ArrayLike]:
    """Row position per id, positionally aligned with ``ids``; None if any id is absent."""
    keys = index.keys_sorted
    total = int(keys.shape[0])
    if total == 0:
        return None
    if ids.dtype != keys.dtype:
        common = xp.promote_types(ids.dtype, keys.dtype)  # promote, never narrow
        ids = ids.astype(common)
        keys = keys.astype(common)
    position = xp.searchsorted(keys, ids)
    clipped = xp.minimum(position, total - 1)
    if int(xp.count_nonzero(keys[clipped] == ids)) != int(ids.shape[0]):
        return None
    return index.row_positions[clipped]


def _code_for(value_codes: Mapping[object, int], expected: object) -> Optional[int]:
    """The code for ``expected``, matching on TYPE as well as value.

    ``True == 1`` in Python, so a plain dict lookup would let a boolean predicate match an
    integer column's code and vice versa; the canonical filter does not conflate them.
    """
    for value, code in value_codes.items():
        if type(value) is type(expected) and value == expected:
            return code
    return None


def _positions_via_category_index(
    base_graph: Plottable,
    role: "ColStatsRole",
    frame: DataFrameT,
    positions: ArrayLike,
    filter_dict: Mapping[str, object],
    engine: Engine,
    xp: ArrayNamespace,
) -> Optional[ArrayLike]:
    """Surviving ``positions`` from coded columns, or None when any predicate is not covered.

    Every predicate column must carry a live category index and the wanted value must be
    one the column actually holds; a value the column never holds matches nothing, which
    is answered here rather than deferred. Anything else returns None and the canonical
    filter answers, so the coverage of this path never changes a result.
    """
    registry = get_registry(base_graph)
    mask = None
    for column, expected in filter_dict.items():
        index = registry.get_category_valid(role, str(column), frame, engine)
        if index is None:
            return None
        code = _code_for(index.value_codes, expected)
        if code is None:
            return positions[:0]
        column_mask = index.codes[positions] == code
        mask = column_mask if mask is None else (mask & column_mask)
    if mask is None:
        return positions
    return positions[mask]


def _filtered_positions(
    frame: DataFrameT,
    positions: ArrayLike,
    filter_dict: Optional[dict],  # hygiene-ok: bare-generic -- the filter dict `_filter_frame` takes, passed through unchanged
    engine: Engine,
    xp: ArrayNamespace,
    base_graph: Optional[Plottable] = None,
    role: "ColStatsRole" = "nodes",
) -> Optional[ArrayLike]:
    """``positions`` whose rows satisfy ``filter_dict``, via the unchanged filter.

    The predicate runs on a frame of just the predicate columns, so an engine's
    3-valued and dtype behavior is the canonical one. None declines a filter the
    narrow form cannot answer (an absent column, whose verdict belongs to the
    canonical path).
    """
    if not filter_dict:
        return positions
    columns = set(map(str, frame.columns))
    if any(str(column) not in columns for column in filter_dict):
        return None
    marker = frame_bindings._ROW_POS
    if marker in columns:
        return None
    if engine == Engine.POLARS and base_graph is not None:
        coded = _positions_via_category_index(
            base_graph, role, frame, positions, filter_dict, engine, xp,
        )
        if coded is not None:
            return coded
    narrow = frame.select(list(filter_dict))  # type: ignore[operator]
    narrow = take_rows(narrow, positions, engine)
    narrow = frame_bindings._with_positions(narrow, marker, positions, engine)
    kept = frame_bindings._filter_frame(narrow, filter_dict, engine)
    if int(kept.shape[0]) == int(narrow.shape[0]):
        return positions
    return col_to_array(kept, marker, engine)


def _seed_rows(
    base_graph: Plottable,
    nodes: DataFrameT,
    node_id: str,
    node_index: NodeIdIndex,
    first_filter: Mapping[str, object],
    engine: Engine,
    xp: ArrayNamespace,
) -> Optional[ArrayLike]:
    """Seed node row positions: node-id index, else the property-index seam, else decline."""
    registry = get_registry(base_graph)
    rows: Optional[ArrayLike]
    if node_id in first_filter:
        value = first_filter[node_id]
        if not isinstance(value, Integral) or isinstance(value, bool):
            return None
        rows = xp.sort(lookup_node_rows(node_index, xp.asarray([value]), xp))
    else:
        # Module attribute, not a bound import: a bound import would not see the seam patch.
        rows = frame_bindings._seed_rows_via_property_index(
            registry, nodes, first_filter, engine, xp,
            policy=get_index_policy(base_graph),
        )
        if rows is None:
            return None  # a full seed scan belongs to the canonical path
    return _filtered_positions(nodes, rows, dict(first_filter), engine, xp, base_graph, "nodes")


def _admits(base_graph: Plottable, ops: Sequence["ASTObject"], engine: Engine) -> bool:
    """Whether this pattern is one the array traversal serves (Polars, directed, fixed hops)."""
    from graphistry.compute.ast import ASTEdge, ASTNode

    if engine != Engine.POLARS or len(ops) < 3 or len(ops) % 2 == 0:
        return False
    if frame_bindings._policy_is_active() or get_index_policy(base_graph) == "off":
        return False
    aliases = [
        op._name for op in ops
        if isinstance(op, (ASTNode, ASTEdge)) and isinstance(op._name, str)
    ]
    if len(aliases) != len(set(aliases)):
        return False
    for position, op in enumerate(ops):
        if position % 2 == 0:
            if (
                not isinstance(op, ASTNode)
                or op.query is not None
                or not frame_bindings._simple_filter_dict(op.filter_dict)
            ):
                return False
        else:
            if (
                not isinstance(op, ASTEdge)
                or not op.is_simple_single_hop()
                or op.direction not in ("forward", "reverse")
                or not frame_bindings._simple_filter_dict(op.edge_match)
                or any(value is not None for value in (
                    op.source_node_match, op.destination_node_match,
                    op.source_node_query, op.destination_node_query,
                    op.edge_query,
                ))
                or op.prune_to_endpoints
                or op.include_zero_hop_seed
            ):
                return False
    first = ops[0]
    return (
        isinstance(first, ASTNode)
        and frame_bindings._simple_filter_dict(first.filter_dict, allow_empty=False)
    )


def _integer_column(frame: DataFrameT, column: str) -> bool:
    series = frame.get_column(column)  # type: ignore[operator]
    return bool(series.dtype.is_integer()) and not series.null_count()


def try_array_path_bag(
    base_graph: Plottable, ops: Sequence["ASTObject"], *, engine: Engine,
) -> Optional[ArrayPathBag]:
    """Row positions per alias for every matched path, or None to decline."""
    from graphistry.compute.ast import ASTEdge, ASTNode

    if not _admits(base_graph, ops, engine):
        return None
    nodes, edges = base_graph._nodes, base_graph._edges
    node_id, src, dst = base_graph._node, base_graph._source, base_graph._destination
    if nodes is None or edges is None or node_id is None or src is None or dst is None:
        return None
    node_id, src, dst = str(node_id), str(src), str(dst)
    if frame_bindings._INTERNAL.intersection(set(map(str, nodes.columns))):
        return None
    if frame_bindings._INTERNAL.intersection(set(map(str, edges.columns))):
        return None
    if not (_integer_column(nodes, node_id) and _integer_column(edges, src) and _integer_column(edges, dst)):
        return None
    for node_op in ops[::2]:
        if not isinstance(node_op, ASTNode) or not frame_bindings._filter_compatible(nodes, node_op.filter_dict):
            return None
    for edge_op in ops[1::2]:
        if not isinstance(edge_op, ASTEdge) or not frame_bindings._filter_compatible(edges, edge_op.edge_match):
            return None

    registry = get_registry(base_graph)
    node_index = registry.get_valid(NODE_ID, nodes, (node_id,), engine)
    if not isinstance(node_index, NodeIdIndex) or not frame_bindings._integer_index(node_index):
        return None
    if nodes.schema[node_id] != edges.schema[src] or nodes.schema[node_id] != edges.schema[dst]:
        return None

    direction_indexes: Dict[int, Sequence[AdjacencyIndex]] = {}
    for edge_position in range(1, len(ops), 2):
        edge_step = ops[edge_position]
        assert isinstance(edge_step, ASTEdge)  # _admits checked every odd position
        indexes = frame_bindings._indices_for_direction(
            registry, edge_step.direction, edges, (src, dst), engine,
        )
        if indexes is None or not all(bool(frame_bindings._integer_index(index)) for index in indexes):
            return None
        direction_indexes[edge_position] = list(indexes)

    xp, _ = array_namespace(engine)
    first_op = ops[0]
    assert isinstance(first_op, ASTNode) and first_op.filter_dict is not None  # _admits
    seed_rows = _seed_rows(
        base_graph, nodes, node_id, node_index, first_op.filter_dict, engine, xp,
    )
    if seed_rows is None:
        return None

    node_id_values = col_to_array(nodes, node_id, engine)
    src_values = col_to_array(edges, src, engine)
    dst_values = col_to_array(edges, dst, engine)
    node_rows: Dict[str, ArrayLike] = {}
    edge_rows: Dict[str, ArrayLike] = {}
    if isinstance(first_op._name, str):
        node_rows[first_op._name] = seed_rows
    current_ids = node_id_values[seed_rows]
    policy = get_index_policy(base_graph)
    n_edges = int(edges.shape[0])
    estimated_rows = int(current_ids.shape[0])

    for edge_position in range(1, len(ops), 2):
        edge_op, next_op = ops[edge_position], ops[edge_position + 1]
        assert isinstance(edge_op, ASTEdge) and isinstance(next_op, ASTNode)  # _admits
        reverse = edge_op.direction == "reverse"
        hop_indexes = direction_indexes[edge_position]
        frontier = xp.unique(current_ids)
        if policy != "force":
            threshold = cost_gate_frac(engine) * min(index.n_keys for index in hop_indexes)
            if int(frontier.shape[0]) >= threshold:
                return None
            gathered = sum(lookup_degree(index, frontier, xp) for index in hop_indexes)
            if gathered >= cost_gate_frac(engine) * n_edges:
                return None

        candidates = xp.unique(lookup_edge_rows(hop_indexes[0], frontier, xp)[0])
        kept = _filtered_positions(edges, candidates, edge_op.edge_match, engine, xp, base_graph, "edges")
        if kept is None:
            return None
        from_ids = (dst_values if reverse else src_values)[kept]
        to_ids = (src_values if reverse else dst_values)[kept]

        endpoint_ids = xp.unique(to_ids)
        endpoint_rows = xp.sort(lookup_node_rows(node_index, endpoint_ids, xp))
        surviving = _filtered_positions(nodes, endpoint_rows, next_op.filter_dict, engine, xp, base_graph, "nodes")
        if surviving is None:
            return None
        if int(surviving.shape[0]) != int(endpoint_rows.shape[0]):
            keep = xp.isin(to_ids, node_id_values[surviving])
            kept, from_ids, to_ids = kept[keep], from_ids[keep], to_ids[keep]

        order = xp.lexsort((kept, from_ids))
        from_sorted, edge_sorted, to_sorted = from_ids[order], kept[order], to_ids[order]
        low = xp.searchsorted(from_sorted, current_ids, side="left")
        high = xp.searchsorted(from_sorted, current_ids, side="right")
        counts = high - low
        total = int(counts.sum())
        estimated_rows = total
        if policy != "force" and total > 0 and total >= n_edges:
            return None
        left = xp.repeat(xp.arange(int(current_ids.shape[0])), counts)
        starts = xp.cumsum(counts) - counts
        picked = low[left] + (xp.arange(total) - starts[left])
        node_rows = {alias: rows[left] for alias, rows in node_rows.items()}
        edge_rows = {alias: rows[left] for alias, rows in edge_rows.items()}
        current_ids = to_sorted[picked]
        current_rows = _aligned_node_rows(node_index, current_ids, xp)
        if current_rows is None:
            return None
        if isinstance(next_op._name, str):
            node_rows[next_op._name] = current_rows
        if isinstance(edge_op._name, str):
            edge_rows[edge_op._name] = edge_sorted[picked]

    return ArrayPathBag(
        node_rows=node_rows,
        edge_rows=edge_rows,
        height=int(current_ids.shape[0]),
        hop_count=(len(ops) - 1) // 2,
        estimated_rows=estimated_rows,
    )


ProjectionItem = Tuple[str, str, Optional[str], object]


def project_array_path_bag(
    bag: ArrayPathBag,
    nodes: DataFrameT,
    edges: DataFrameT,
    node_id: str,
    items: Sequence[ProjectionItem],
) -> "pl.DataFrame":
    """Materialize exactly the projected columns by gathering each alias's rows."""
    import polars as pl

    columns: List[pl.Series] = []
    for name, kind, alias, value in items:
        if kind == "literal":
            # `pl.lit`, not a Python list: `lit` types a Python int as Int32, a list as Int64.
            columns.append(pl.repeat(pl.lit(value), bag.height, eager=True).alias(name))
            continue
        assert isinstance(alias, str)  # every non-literal item names an alias
        if kind == "node_id":
            source, rows, column = nodes, bag.node_rows[alias], node_id
        else:
            assert isinstance(value, str)  # a column item names its column
            if kind == "node_column":
                source, rows, column = nodes, bag.node_rows[alias], value
            else:
                source, rows, column = edges, bag.edge_rows[alias], value
        columns.append(source.get_column(column).gather(rows).rename(name))  # type: ignore[operator]
    return pl.DataFrame(columns)
