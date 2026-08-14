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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

from graphistry.Engine import Engine, df_concat
from graphistry.Plottable import Plottable
from graphistry.compute.typing import DataFrameT

from .api import (
    _record_indexed_traversal,
    _trace_active,
    get_index_policy,
    get_registry,
    with_index_policy,
)
from graphistry.compute.dataframe.join import (
    estimate_inner_join_rows,
    path_ordered_expand_join,
    semijoin_by_column,
)

from .cost import cost_gate_frac
from .engine_arrays import array_namespace, col_to_array, take_rows
from .lookup import (
    lookup_degree,
    lookup_edge_rows,
    lookup_node_rows,
    lookup_prop_rows,
    prop_match_count,
)
from .registry import (
    EDGE_IN_ADJ,
    EDGE_OUT_ADJ,
    NODE_ID,
    AdjacencyIndex,
    GfqlIndexRegistry,
    NodeIdIndex,
    NodePropIndex,
)
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


def _integer_index(index: Union[AdjacencyIndex, NodeIdIndex, NodePropIndex]) -> bool:
    """Whether an index's keys (and, for adjacency, its neighbor ids) are integral.

    The vectorized gather promotes dtypes rather than narrowing, so non-integral
    keys are declined rather than risking a lossy compare.
    """
    key_ok = index.keys_sorted.dtype.kind in ("i", "u")
    if not isinstance(index, AdjacencyIndex):
        return key_ok
    return key_ok and index.other_values.dtype.kind in ("i", "u")


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

        # frame is DataFrameT (engine-neutral); on this branch it IS a polars frame,
        # which the helper's constrained TypeVar cannot see through the alias
        return cast(DataFrameT, filter_by_dict_polars(frame, filter_dict))  # type: ignore[type-var]
    from graphistry.compute.filter_by_dict import filter_by_dict

    return filter_by_dict(frame, filter_dict, engine)  # type: ignore[arg-type]


def _with_marker(frame: DataFrameT, name: Optional[str], engine: Engine) -> DataFrameT:
    if not isinstance(name, str):
        return frame
    if engine == Engine.POLARS:
        # Native Polars intentionally omits pandas' alias-marker residue.
        return frame
    return cast(DataFrameT, frame.assign(**{name: True}))


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
    return cast(DataFrameT, frame.assign(**{_EDGE_ORD: positions}))


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
        work = gathered
        if isinstance(alias, str):
            work = gathered.assign(**{alias: True})
            payload = payload + [alias]
            renames = {col: f"{alias}.{col}" for col in payload}
        else:
            renames = {}

        def one(from_col: str, to_col: str, orient: int) -> DataFrameT:
            out = work.rename(columns={from_col: _FROM, to_col: _TO})
            if renames:
                out = out.rename(columns=renames)
            return cast(DataFrameT, out.assign(**{_ORIENT_ORD: orient}))

    if direction == "undirected":
        forward = one(src, dst, 0)
        reverse = one(dst, src, 1)
        if engine == Engine.POLARS:
            reverse = reverse.select(forward.columns)  # type: ignore[operator]
        combined = cast(DataFrameT, df_concat(engine)([forward, reverse], ignore_index=True))
        # openCypher trail semantics (#1903 A-1): a self-loop's two undirected
        # orientations are the SAME binding -- dedupe the flip twin on the
        # per-edge ordinal, mirroring the standard builder.
        if _EDGE_ORD in combined.columns:
            if engine == Engine.POLARS:
                combined = cast(  # hygiene-ok: explicit-cast -- DataFrameT narrowing, module-wide idiom
                    DataFrameT,
                    combined.unique(  # type: ignore[operator]
                        subset=[_FROM, _TO, _EDGE_ORD], keep="first", maintain_order=True
                    ),
                )
            else:
                combined = cast(  # hygiene-ok: explicit-cast -- DataFrameT narrowing, module-wide idiom
                    DataFrameT,
                    combined.drop_duplicates(
                        subset=[_FROM, _TO, _EDGE_ORD], keep="first", ignore_index=True
                    ),
                )
        return combined
    if direction == "reverse":
        return one(dst, src, 0)
    return one(src, dst, 0)


def _policy_is_active() -> bool:
    from graphistry.compute.gfql.call.executor import _thread_local

    return getattr(_thread_local, "policy", None) is not None


def _seed_rows_via_property_index(
    registry: GfqlIndexRegistry,
    nodes: DataFrameT,
    first_filter: Mapping[str, Any],
    engine: Engine,
    xp: Any,
    *,
    policy: str,
) -> Optional[Any]:
    """Node row positions for the most selective indexed scalar seed predicate.

    The seed of a fixed-hop pattern is usually a high-selectivity equality on a
    business key (``{id: 42}``) that is NOT the graph's node-id binding, which
    otherwise costs a full node scan. When a resident, still-valid property index
    covers such a column, gather its candidates instead; the caller re-applies the
    WHOLE filter to them, so the result is identical to the scan either way.

    Returns None (keep scanning) when nothing is indexed, no predicate is a plain
    integer scalar, or the estimated candidate count is not selective enough to
    beat the scan (``force`` skips the cost gate).
    """
    if not first_filter:
        return None
    best_rows = None
    best_count: Optional[int] = None
    for column in registry.node_prop_cols():
        value = first_filter.get(column)
        if value is None or isinstance(value, bool) or not isinstance(value, Integral):
            continue
        index = registry.get_node_prop_valid(column, nodes, engine)
        if index is None:
            continue
        values = xp.asarray([value])
        count = prop_match_count(index, values, xp)
        if best_count is not None and count >= best_count:
            continue
        best_count = count
        best_rows = (index, values)
    if best_rows is None or best_count is None:
        return None
    if policy != "force":
        n_nodes = int(nodes.shape[0])
        if best_count >= cost_gate_frac(engine) * n_nodes:
            return None  # not selective enough to beat one vectorized scan
    index, values = best_rows
    return xp.sort(lookup_prop_rows(index, values, xp))


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
        op._name for op in ops
        if isinstance(op, (ASTNode, ASTEdge)) and isinstance(op._name, str)
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
        prop_rows = _seed_rows_via_property_index(
            registry, nodes, first_filter, engine, xp, policy=get_index_policy(base_graph),
        )
        if prop_rows is not None:
            # Secondary index hit: gather the candidates, then let the UNCHANGED
            # filter apply every remaining predicate to that small frame.
            first_nodes = _filter_frame(
                take_rows(nodes, prop_rows, engine), first_filter, engine,
            )
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
        state = first_nodes[[node_id]].rename(columns={node_id: _CURRENT})
        if isinstance(first_alias, str):
            state = state.assign(**{first_alias: state[_CURRENT]})

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
            lookup_degree(index, frontier, xp) for index in edge_indexes
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
        oriented = semijoin_by_column(
            oriented, next_nodes, left_on=_TO, right_on=node_id, engine=engine,
        )

        estimated_rows = estimate_inner_join_rows(
            state, oriented, left_on=_CURRENT, right_on=_FROM, engine=engine,
        )
        if policy != "force" and estimated_rows > 0 and estimated_rows >= n_edges:
            return None
        state = path_ordered_expand_join(
            state,
            oriented,
            current_col=_CURRENT,
            from_col=_FROM,
            to_col=_TO,
            path_order_col=_PATH_ORD,
            tiebreak_cols=(_ORIENT_ORD, _EDGE_ORD),
            alias=next_op._name,
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
    from graphistry.compute.ast import ASTEdge, ASTNode

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

    if get_index_policy(base_graph) != "force":
        # Was cost the ONLY thing in the way? Re-run the real gate with the cost
        # checks disabled rather than re-deriving them here: a second copy of the
        # thresholds drifts, and on engines with a tighter gate it mislabelled
        # structurally-unsupported shapes as "cost_frontier". Trace-only path.
        if _try_indexed_connected_bindings_state(
            with_index_policy(base_graph, "force"),
            ops,
            engine=engine,
            start_nodes=start_nodes,
            alias_prefilters=alias_prefilters,
        ) is not None:
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
    from graphistry.compute.ast import ASTNode

    hop_count = max(0, (len(ops) - 1) // 2)
    first_op_any = ops[0] if ops else None
    first_filter = (
        first_op_any.filter_dict if isinstance(first_op_any, ASTNode) else None
    )
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
        # Classifying WHY we declined re-validates index fingerprints and can even
        # filter the seed frame — diagnostic work whose only consumer is the trace.
        # Compute it strictly inside a trace context (api._trace_active contract:
        # diagnostic enrichment costs the hot path nothing).
        reason = (
            _connected_decline_reason(
                base_graph,
                ops,
                engine=engine,
                start_nodes=start_nodes,
                alias_prefilters=alias_prefilters,
            )
            if _trace_active()
            else "not_traced"
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
