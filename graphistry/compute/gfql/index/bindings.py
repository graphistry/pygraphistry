"""Index-driven fixed-hop connected binding rows.

This module is deliberately semantic rather than query-shaped: it recognizes a
small, exact subset of alternating node/edge ASTs and returns a same-engine path
bag, or ``None`` before visible mutation when the resident indexes, semantics,
or cost are unsuitable. Canonical row builders remain responsible for property
attachment and every suffix operation.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.compute.typing import DataFrameT

from .api import _record_indexed_traversal, get_index_policy, get_registry
from .cost import cost_gate_frac
from .engine_arrays import array_namespace, col_to_array, take_rows
from .lookup import lookup_edge_rows, lookup_node_rows
from .registry import EDGE_IN_ADJ, EDGE_OUT_ADJ, NODE_ID, AdjacencyIndex, NodeIdIndex
from .traverse import _indices_for_direction


_CURRENT = "__current__"
_FROM = "__gfql_ib_from__"
_TO = "__gfql_ib_to__"
_PATH_ORD = "__gfql_ib_path_ord__"
_EDGE_ORD = "__gfql_ib_edge_ord__"
_ORIENT_ORD = "__gfql_ib_orient_ord__"
_LEFT_N = "__gfql_ib_left_n__"
_RIGHT_N = "__gfql_ib_right_n__"
_INTERNAL = {
    _CURRENT, _FROM, _TO, _PATH_ORD, _EDGE_ORD, _ORIENT_ORD,
    _LEFT_N, _RIGHT_N,
}


@dataclass(frozen=True)
class IndexedBindingsState:
    """A complete fixed-hop path bag ready for canonical materialization."""

    state: DataFrameT
    alias_frames: Dict[str, DataFrameT]
    engine: Engine
    hop_count: int
    estimated_rows: int


def _plain_scalar_filter(value: Any) -> bool:
    from graphistry.compute.filter_by_dict import _is_membership_filter_value
    from graphistry.compute.predicates.ASTPredicate import ASTPredicate

    return (
        value is not None
        and not isinstance(value, (ASTPredicate, dict))
        and not _is_membership_filter_value(value)
    )


def _simple_filter_dict(value: Any, *, allow_empty: bool = True) -> bool:
    if value is None:
        return allow_empty
    return (
        isinstance(value, Mapping)
        and (allow_empty or bool(value))
        and all(isinstance(key, str) and _plain_scalar_filter(item)
                for key, item in value.items())
    )


def _integer_index(index: Any) -> bool:
    key_dtype = getattr(getattr(index, "keys_sorted", None), "dtype", None)
    other_dtype = getattr(getattr(index, "other_values", None), "dtype", None)
    key_ok = getattr(key_dtype, "kind", None) in ("i", "u")
    if isinstance(index, NodeIdIndex):
        return key_ok
    return key_ok and getattr(other_dtype, "kind", None) in ("i", "u")


def _filter_compatible(frame: DataFrameT, filter_dict: Optional[dict]) -> bool:
    """Cheap schema/type parity gate; decline so canonical filtering raises."""
    if not filter_dict:
        return True
    from graphistry.compute.filter_by_dict import (
        _is_numeric_dtype_safe,
        _is_string_dtype_safe,
        resolve_filter_column,
    )

    try:
        for col, value in filter_dict.items():
            resolved, resolved_value = resolve_filter_column(frame, col, value)
            series = (
                frame.get_column(resolved)  # type: ignore[operator]
                if "polars" in type(frame).__module__
                else frame[resolved]
            )
            dtype = series.dtype
            if _is_numeric_dtype_safe(dtype) and isinstance(resolved_value, str):
                return False
            if (
                _is_string_dtype_safe(dtype)
                and isinstance(resolved_value, (int, float))
                and not isinstance(resolved_value, bool)
            ):
                return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _filter_frame(
    frame: DataFrameT, filter_dict: Optional[dict], engine: Engine,
) -> DataFrameT:
    if engine == Engine.POLARS:
        from graphistry.compute.gfql.lazy.engine.polars.predicates import (
            filter_by_dict_polars,
        )

        return cast(DataFrameT, filter_by_dict_polars(frame, filter_dict))  # type: ignore[arg-type]
    from graphistry.compute.filter_by_dict import filter_by_dict

    return filter_by_dict(frame, filter_dict, engine)  # type: ignore[arg-type]


def _with_marker(frame: DataFrameT, name: Optional[str], engine: Engine) -> DataFrameT:
    if not isinstance(name, str):
        return frame
    if engine == Engine.POLARS:
        # Native Polars intentionally omits pandas' alias-marker residue.
        return frame
    out = frame.copy()
    out[name] = True
    return cast(DataFrameT, out)


def _concat(frames: Sequence[DataFrameT], engine: Engine) -> DataFrameT:
    if engine == Engine.POLARS:
        import polars as pl

        return cast(DataFrameT, pl.concat(list(frames), how="vertical"))  # type: ignore[type-var]
    if engine == Engine.CUDF:
        import cudf  # type: ignore

        return cast(DataFrameT, cudf.concat(list(frames), ignore_index=True))
    import pandas as pd

    return cast(DataFrameT, pd.concat(list(frames), ignore_index=True))


def _frame_with_positions(
    frame: DataFrameT, positions: Any, engine: Engine,
) -> DataFrameT:
    if engine == Engine.POLARS:
        import numpy as np
        import polars as pl

        return cast(
            DataFrameT,
            frame.with_columns(pl.Series(_EDGE_ORD, np.asarray(positions))),  # type: ignore[operator]
        )
    out = frame.copy()
    out[_EDGE_ORD] = positions
    return cast(DataFrameT, out)


def _orient_edges(
    gathered: DataFrameT,
    *,
    src: str,
    dst: str,
    direction: str,
    alias: Optional[str],
    engine: Engine,
) -> DataFrameT:
    payload = [
        col for col in gathered.columns
        if col not in (src, dst, _EDGE_ORD)
    ]
    if engine == Engine.POLARS:
        import polars as pl

        work = gathered
        if isinstance(alias, str):
            renames = {col: f"{alias}.{col}" for col in payload}
        else:
            work = work.select([src, dst, _EDGE_ORD])  # type: ignore[operator]
            renames = {}

        def one(from_col: str, to_col: str, orient: int) -> DataFrameT:
            out = work.rename({from_col: _FROM, to_col: _TO})
            if renames:
                out = out.rename(renames)
            return cast(
                DataFrameT,
                out.with_columns(pl.lit(orient).alias(_ORIENT_ORD)),  # type: ignore[operator]
            )

    else:
        work = gathered.copy()
        if isinstance(alias, str):
            work[alias] = True
            payload = payload + [alias]
            renames = {col: f"{alias}.{col}" for col in payload}
        else:
            renames = {}

        def one(from_col: str, to_col: str, orient: int) -> DataFrameT:
            out = work.rename(columns={from_col: _FROM, to_col: _TO})
            if renames:
                out = out.rename(columns=renames)
            out[_ORIENT_ORD] = orient
            return cast(DataFrameT, out)

    if direction == "undirected":
        forward = one(src, dst, 0)
        reverse = one(dst, src, 1)
        if engine == Engine.POLARS:
            reverse = reverse.select(forward.columns)  # type: ignore[operator]
        return _concat([forward, reverse], engine)
    if direction == "reverse":
        return one(dst, src, 0)
    return one(src, dst, 0)


def _estimate_join_rows(
    state: DataFrameT, oriented: DataFrameT, engine: Engine,
) -> int:
    if len(state) == 0 or len(oriented) == 0:
        return 0
    if engine == Engine.POLARS:
        import polars as pl

        left = state.group_by(_CURRENT).len().rename({"len": _LEFT_N})  # type: ignore[operator]
        right = oriented.group_by(_FROM).len().rename({"len": _RIGHT_N})  # type: ignore[operator]
        value = (
            left.join(right, left_on=_CURRENT, right_on=_FROM, how="inner")
            .select((pl.col(_LEFT_N) * pl.col(_RIGHT_N)).sum())
            .item()
        )
        return 0 if value is None else int(value)

    left = state.groupby(_CURRENT, sort=False).size().reset_index()
    left.columns = [_CURRENT, _LEFT_N]
    right = oriented.groupby(_FROM, sort=False).size().reset_index()
    right.columns = [_FROM, _RIGHT_N]
    counts = left.merge(
        right, left_on=_CURRENT, right_on=_FROM, how="inner", sort=False,
    )
    if len(counts) == 0:
        return 0
    return int((counts[_LEFT_N] * counts[_RIGHT_N]).sum())


def _join_state(
    state: DataFrameT,
    oriented: DataFrameT,
    *,
    node_alias: Optional[str],
    engine: Engine,
) -> DataFrameT:
    if engine == Engine.POLARS:
        import polars as pl

        joined = (
            state.with_row_index(_PATH_ORD)  # type: ignore[operator]
            .join(oriented, left_on=_CURRENT, right_on=_FROM, how="inner")
            .sort([_PATH_ORD, _ORIENT_ORD, _EDGE_ORD])
            .drop(_CURRENT)
            .rename({_TO: _CURRENT})
        )
        if isinstance(node_alias, str):
            joined = joined.with_columns(pl.col(_CURRENT).alias(node_alias))
        return cast(
            DataFrameT,
            joined.drop([
                col for col in (_FROM, _PATH_ORD, _EDGE_ORD, _ORIENT_ORD)
                if col in joined.columns
            ]),
        )

    out = state.copy()
    xp, _ = array_namespace(engine)
    out[_PATH_ORD] = xp.arange(len(out))  # type: ignore[call-overload]
    out = out.merge(
        oriented, left_on=_CURRENT, right_on=_FROM, how="inner", sort=False,
    )
    if len(out):
        if engine == Engine.PANDAS:
            out = out.sort_values(
                [_PATH_ORD, _ORIENT_ORD, _EDGE_ORD], kind="stable",
            )
        else:
            out = out.sort_values([_PATH_ORD, _ORIENT_ORD, _EDGE_ORD])
    out = out.drop(columns=[_CURRENT]).rename(columns={_TO: _CURRENT})
    if isinstance(node_alias, str):
        out[node_alias] = out[_CURRENT]
    return cast(
        DataFrameT,
        out.drop(
            columns=[
                col for col in (_FROM, _PATH_ORD, _EDGE_ORD, _ORIENT_ORD)
                if col in out.columns
            ],
        ),
    )


def _policy_is_active() -> bool:
    from graphistry.compute.gfql.call.executor import _thread_local

    return getattr(_thread_local, "policy", None) is not None


def _filter_oriented_endpoints(
    oriented: DataFrameT,
    next_nodes: DataFrameT,
    node_id: str,
    engine: Engine,
) -> DataFrameT:
    if engine == Engine.POLARS:
        return cast(
            DataFrameT,
            oriented.join(  # type: ignore[call-arg]
                next_nodes.select(node_id).unique(),  # type: ignore[operator]
                left_on=_TO,
                right_on=node_id,
                how="semi",  # type: ignore[arg-type]
            ),
        )
    return cast(
        DataFrameT,
        oriented[oriented[_TO].isin(next_nodes[node_id])].copy(),
    )



def _lookup_degree(index: AdjacencyIndex, frontier: Any, xp: Any) -> int:
    """Native O(frontier) degree estimate before CSR range expansion."""
    keys = index.keys_sorted
    if int(keys.shape[0]) == 0 or int(frontier.shape[0]) == 0:
        return 0
    values = frontier
    if values.dtype != keys.dtype:
        common = xp.promote_types(values.dtype, keys.dtype)
        values = values.astype(common)
        keys = keys.astype(common)
    positions = xp.searchsorted(keys, values)
    clipped = xp.where(
        positions < keys.shape[0], positions, keys.shape[0] - 1,
    )
    hits = clipped[keys[clipped] == values]
    if int(hits.shape[0]) == 0:
        return 0
    counts = index.group_offsets[hits + 1] - index.group_offsets[hits]
    return int(counts.sum())

def _try_indexed_connected_bindings_state(
    base_graph: Plottable,
    ops: Sequence[Any],
    *,
    engine: Engine,
    start_nodes: Optional[DataFrameT] = None,
    alias_prefilters: Optional[Any] = None,
) -> Optional[IndexedBindingsState]:
    """Return an exact indexed fixed-hop path bag, or safely decline with ``None``."""
    from graphistry.compute.ast import ASTEdge, ASTNode

    if (
        engine not in (Engine.PANDAS, Engine.CUDF, Engine.POLARS)
        or start_nodes is not None
        or alias_prefilters
        or _policy_is_active()
        or get_index_policy(base_graph) == "off"
        or len(ops) < 3
        or len(ops) % 2 == 0
    ):
        return None

    nodes = base_graph._nodes
    edges = base_graph._edges
    node_id = base_graph._node
    src = base_graph._source
    dst = base_graph._destination
    if nodes is None or edges is None or node_id is None or src is None or dst is None:
        return None
    node_id, src, dst = str(node_id), str(src), str(dst)

    aliases = [
        getattr(op, "_name", None)
        for op in ops
        if isinstance(getattr(op, "_name", None), str)
    ]
    if (
        len(aliases) != len(set(aliases))
        or _INTERNAL.intersection(set(map(str, nodes.columns)))
        or _INTERNAL.intersection(set(map(str, edges.columns)))
        or _INTERNAL.intersection(set(cast(Sequence[str], aliases)))
    ):
        return None

    for index, op in enumerate(ops):
        if index % 2 == 0:
            if (
                not isinstance(op, ASTNode)
                or op.query is not None
                or op._name == node_id
                or not _simple_filter_dict(op.filter_dict)
            ):
                return None
        else:
            if (
                not isinstance(op, ASTEdge)
                or not op.is_simple_single_hop()
                or op._name in (src, dst)
                or op.direction not in ("forward", "reverse", "undirected")
                or not _simple_filter_dict(op.edge_match)
                or any(value is not None for value in (
                    op.source_node_match, op.destination_node_match,
                    op.source_node_query, op.destination_node_query,
                    op.edge_query,
                ))
                or op.prune_to_endpoints
                or op.include_zero_hop_seed
                or not _filter_compatible(edges, op.edge_match)
            ):
                return None
    if any(
        isinstance(op, ASTEdge) and op.direction == "undirected" for op in ops
    ) and len(ops) != 3:
        return None

    if engine in (Engine.PANDAS, Engine.CUDF):
        unaliased_edges = sum(
            isinstance(op, ASTEdge) and not isinstance(op._name, str)
            for op in ops
        )
        edge_payload = set(map(str, edges.columns)).difference((src, dst))
        if unaliased_edges >= 4 and edge_payload:
            # Preserve the canonical pandas/cuDF fallback boundary: its fourth
            # repeated anonymous edge payload can collide with accumulated
            # merge suffixes. Aliased edges and payload-free frames are safe.
            return None

    first_op = ops[0]
    if (
        not isinstance(first_op, ASTNode)
        or not _simple_filter_dict(first_op.filter_dict, allow_empty=False)
    ):
        return None

    if any(
        isinstance(op, ASTNode)
        and not _filter_compatible(nodes, op.filter_dict)
        for op in ops[::2]
    ):
        return None

    registry = get_registry(base_graph)
    node_index = cast(
        Optional[NodeIdIndex],
        registry.get_valid(NODE_ID, nodes, (node_id,), engine),
    )
    if node_index is None or not _integer_index(node_index):
        return None

    if engine == Engine.POLARS and (
        nodes.schema[node_id] != edges.schema[src]
        or nodes.schema[node_id] != edges.schema[dst]
    ):
        return None

    direction_indexes: Dict[int, Sequence[AdjacencyIndex]] = {}
    for edge_index in range(1, len(ops), 2):
        edge_op = cast(ASTEdge, ops[edge_index])
        indexes = _indices_for_direction(
            registry, edge_op.direction, edges, (src, dst), engine,
        )
        if indexes is None or not all(_integer_index(index) for index in indexes):
            return None
        direction_indexes[edge_index] = indexes

    first_filter = cast(dict, first_op.filter_dict)
    xp, _ = array_namespace(engine)
    if node_id in first_filter:
        if (
            not isinstance(first_filter[node_id], Integral)
            or isinstance(first_filter[node_id], bool)
        ):
            return None
        seed_ids = xp.asarray([first_filter[node_id]])
        seed_rows = xp.sort(lookup_node_rows(node_index, seed_ids, xp))
        first_nodes = take_rows(nodes, seed_rows, engine)
        first_nodes = _filter_frame(first_nodes, first_filter, engine)
    else:
        hop_count = (len(ops) - 1) // 2
        if int(nodes.shape[0]) >= hop_count * int(edges.shape[0]):
            return None
        first_nodes = _filter_frame(nodes, first_filter, engine)

    first_alias = first_op._name
    alias_frames: Dict[str, DataFrameT] = {}
    first_alias_frame = _with_marker(first_nodes, first_alias, engine)
    if isinstance(first_alias, str):
        alias_frames[first_alias] = first_alias_frame

    if engine == Engine.POLARS:
        import polars as pl

        state = first_nodes.select(pl.col(node_id).alias(_CURRENT))  # type: ignore[operator]
        if isinstance(first_alias, str):
            state = state.with_columns(pl.col(_CURRENT).alias(first_alias))
    else:
        state = first_nodes[[node_id]].copy().rename(columns={node_id: _CURRENT})
        if isinstance(first_alias, str):
            state[first_alias] = state[_CURRENT]

    estimated_rows = int(state.shape[0])
    policy = get_index_policy(base_graph)
    n_edges = int(edges.shape[0])

    for edge_index in range(1, len(ops), 2):
        edge_op = cast(ASTEdge, ops[edge_index])
        frontier = xp.unique(col_to_array(state, _CURRENT, engine))
        edge_indexes = direction_indexes[edge_index]
        if policy != "force":
            threshold = cost_gate_frac(engine) * min(
                index.n_keys for index in edge_indexes
            )
            if int(frontier.shape[0]) >= threshold:
                return None
        gather_estimate = sum(
            _lookup_degree(index, frontier, xp) for index in edge_indexes
        )
        if (
            policy != "force"
            and gather_estimate >= cost_gate_frac(engine) * n_edges
        ):
            return None

        row_parts = [
            lookup_edge_rows(index, frontier, xp)[0] for index in edge_indexes
        ]
        rows = (
            row_parts[0] if len(row_parts) == 1
            else xp.concatenate(row_parts)
        )
        rows = xp.unique(rows)
        gathered = take_rows(edges, rows, engine)
        gathered = _frame_with_positions(gathered, rows, engine)
        gathered = _filter_frame(gathered, edge_op.edge_match, engine)
        oriented = _orient_edges(
            gathered,
            src=src,
            dst=dst,
            direction=edge_op.direction,
            alias=edge_op._name,
            engine=engine,
        )

        next_op = ops[edge_index + 1]
        if not isinstance(next_op, ASTNode):
            return None
        endpoint_ids = xp.unique(col_to_array(oriented, _TO, engine))
        node_rows = xp.sort(lookup_node_rows(node_index, endpoint_ids, xp))
        next_nodes = take_rows(nodes, node_rows, engine)
        next_nodes = _filter_frame(next_nodes, next_op.filter_dict, engine)
        next_alias_frame = _with_marker(next_nodes, next_op._name, engine)
        oriented = _filter_oriented_endpoints(
            oriented, next_nodes, node_id, engine,
        )

        estimated_rows = _estimate_join_rows(state, oriented, engine)
        if policy != "force" and estimated_rows > 0 and estimated_rows >= n_edges:
            return None
        state = _join_state(
            state,
            oriented,
            node_alias=next_op._name,
            engine=engine,
        )
        if isinstance(next_op._name, str):
            alias_frames[next_op._name] = next_alias_frame

    return IndexedBindingsState(
        state, alias_frames, engine, (len(ops) - 1) // 2, estimated_rows,
    )




def _connected_decline_reason(
    base_graph: Plottable,
    ops: Sequence[Any],
    *,
    engine: Engine,
    start_nodes: Optional[DataFrameT],
    alias_prefilters: Optional[Any],
) -> str:
    """Classify a completed safe decline without changing canonical behavior."""
    from graphistry.compute.ast import ASTEdge

    if engine not in (Engine.PANDAS, Engine.CUDF, Engine.POLARS):
        return "unsupported_engine"
    if _policy_is_active():
        return "policy_active"
    if get_index_policy(base_graph) == "off":
        return "index_policy_off"
    if start_nodes is not None or alias_prefilters:
        return "unsupported_shape"
    nodes, edges = base_graph._nodes, base_graph._edges
    node_id, src, dst = base_graph._node, base_graph._source, base_graph._destination
    if nodes is None or edges is None or node_id is None or src is None or dst is None:
        return "unsupported_shape"

    registry = get_registry(base_graph)
    required: List[Tuple[Any, Any, Tuple[str, ...]]] = [(NODE_ID, nodes, (str(node_id),))]
    for op in ops[1::2]:
        if not isinstance(op, ASTEdge):
            return "unsupported_shape"
        if op.direction in ("forward", "undirected"):
            required.append((EDGE_OUT_ADJ, edges, (str(src), str(dst))))
        if op.direction in ("reverse", "undirected"):
            required.append((EDGE_IN_ADJ, edges, (str(src), str(dst))))
    for kind, frame, columns in required:
        raw = registry.get(kind)
        if raw is None:
            return "index_missing"
        valid = registry.get_valid(kind, frame, columns, engine)
        if valid is None:
            return "index_stale"
        if not _integer_index(valid):
            return "unsupported_dtype"

    if get_index_policy(base_graph) != "force" and ops:
        first_filter = getattr(ops[0], "filter_dict", None)
        if isinstance(first_filter, Mapping) and first_filter:
            first_nodes = _filter_frame(nodes, cast(dict, first_filter), engine)
            frontier_n = int(first_nodes.shape[0])
            adjacency = [
                cast(AdjacencyIndex, valid)
                for kind, frame, columns in required
                for valid in [registry.get_valid(kind, frame, columns, engine)]
                if kind != NODE_ID and valid is not None
            ]
            if adjacency and frontier_n >= cost_gate_frac(engine) * min(
                index.n_keys for index in adjacency
            ):
                return "cost_frontier"
    return "unsupported_shape"


def try_indexed_connected_bindings_state(
    base_graph: Plottable,
    ops: Sequence[Any],
    *,
    engine: Engine,
    start_nodes: Optional[DataFrameT] = None,
    alias_prefilters: Optional[Any] = None,
) -> Optional[IndexedBindingsState]:
    """Attempt the indexed path; only explicit safe declines fall through."""
    hop_count = max(0, (len(ops) - 1) // 2)
    first_filter = getattr(ops[0], "filter_dict", None) if ops else None
    node_id = base_graph._node
    public_seed_scan = not (
        isinstance(first_filter, Mapping)
        and node_id is not None
        and str(node_id) in first_filter
    )
    result = _try_indexed_connected_bindings_state(
        base_graph,
        ops,
        engine=engine,
        start_nodes=start_nodes,
        alias_prefilters=alias_prefilters,
    )
    if result is None:
        reason = _connected_decline_reason(
            base_graph,
            ops,
            engine=engine,
            start_nodes=start_nodes,
            alias_prefilters=alias_prefilters,
        )
        _record_indexed_traversal(
            seam="connected_bindings",
            engine=engine,
            served=False,
            reason=reason,
            hop_count=hop_count,
            public_seed_scan=public_seed_scan,
        )
        return None
    _record_indexed_traversal(
        seam="connected_bindings",
        engine=engine,
        served=True,
        reason="served",
        hop_count=result.hop_count,
        public_seed_scan=public_seed_scan,
        hop_details=[
            {"hop": hop + 1, "estimated_rows": result.estimated_rows}
            for hop in range(result.hop_count)
        ],
    )
    return result
