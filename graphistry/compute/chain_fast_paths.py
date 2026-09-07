"""Seed-resolution and resident-index helpers shared by the chain specializations
(``chain_specializations/``, ``gfql/lazy/engine/polars/chain_specializations/``) and the Cypher
lanes in ``gfql_fast_paths.py``. This module imports only leaf modules (no back-edge into
``chain.py`` or the specialization packages)."""
# ruff: noqa: E501

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, TYPE_CHECKING, Union, cast

from graphistry.Plottable import Plottable
from .ast import Direction
from .typing import ArrayLike, ArrayNamespace, DataFrameT, SeriesT

if TYPE_CHECKING:
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index.registry import AdjacencyIndex, NodeIdIndex


def _tag_fast_path_aliases(
    res: Plottable,
    alias_n0: Optional[str], alias_e1: Optional[str], alias_n2: Optional[str],
    src: str, dst: str, node: str, direction: Direction,
) -> Plottable:
    """Attach the alias flag columns the full path's ``combine_steps`` would have merged in.

    The chain fast path's gate used to reject ANY named op, so a named
    `g.gfql([n(name=..), e(..), n(name=..)])` fell to the full two-pass machinery purely
    because the ops carried names — measured ~25.2 -> ~2.3 ms (medians of 5 paired runs) on
    a 200-node graph where data-proportional work is ~0. Naming is a PROJECTION concern,
    not a traversal one, so it should not change which engine path runs. NOTE the scope: this is the NATIVE chain
    surface. The Cypher `MATCH ... RETURN` shapes on the graph benchmark are served
    earlier by `gfql_fast_paths.py` and never reach here (measured, both engines).

    Why deriving the tags from the RETURNED EDGES matches the full path: `combine_steps`
    tags a node with an alias iff it matched that step in the BACKWARD-PRUNED frame, i.e.
    iff it still participates in a surviving edge. The edges this function receives are
    exactly the surviving ones — the fast path has already applied the node filters, the
    edge_match and the endpoint validation — so ``isin`` over their endpoint columns is the
    same predicate, computed without the join.

    A seed whose edges all fail the type filter yields an empty edge frame, so it is tagged
    False rather than True: that is the dead-end case, and it is why the tag keys on the
    edges rather than on the node filter.
    """
    if alias_n0 is None and alias_e1 is None and alias_n2 is None:
        return res
    nodes: Optional[DataFrameT] = res._nodes
    edges: Optional[DataFrameT] = res._edges
    if nodes is None or edges is None:
        return res
    from_col, to_col = (src, dst) if direction == "forward" else (dst, src)
    node_flags: Dict[str, SeriesT] = {}
    if alias_n0 is not None:
        node_flags[alias_n0] = nodes[node].isin(edges[from_col])
    if alias_n2 is not None:
        node_flags[alias_n2] = nodes[node].isin(edges[to_col])
    if node_flags:
        colliding_node_flags = [name for name in node_flags if name in nodes.columns]
        new_node_flags = [name for name in node_flags if name not in nodes.columns]
        if colliding_node_flags:
            nodes = nodes.drop(columns=colliding_node_flags)
        nodes = nodes.assign(**node_flags)
        rest = [c for c in nodes.columns if c != node and c not in node_flags]
        nodes = nodes[[node, *new_node_flags, *rest, *colliding_node_flags]].reset_index(drop=True)
    if alias_e1 is not None:
        alias_was_column = alias_e1 in edges.columns
        if alias_was_column:
            edges = edges.drop(columns=[alias_e1])
        edges = edges.assign(**{alias_e1: True})
        if not alias_was_column:
            edges = edges[[alias_e1, *[c for c in edges.columns if c != alias_e1]]]
        edges = edges.reset_index(drop=True)

    return res.nodes(nodes).edges(edges)


def _seeded_scalar_filters(fd: Optional[Dict[str, Any]], df: DataFrameT) -> Optional[Dict[str, Any]]:
    """Resolve a filter dict to plain scalar column==value pairs, or None to bail
    to the general path. Mirrors filter_by_dict.resolve_filter_column exactly for
    the shapes it accepts: the cypher ``label__X: True`` form maps to ``type``
    equality ONLY when no list-valued ``labels`` column exists (labels-containment
    is not scalar equality) and the frame is not edge-shaped — same precedence as
    the live resolver. Anything else (predicates, non-scalar values, absent
    columns) bails, so the full path keeps its exact semantics incl. E301."""
    from graphistry.compute.filter_by_dict import _looks_like_edge_dataframe
    if not fd:
        return {}
    cols = set(df.columns)
    out: Dict[str, Any] = {}
    for k, v in fd.items():
        if not isinstance(v, (int, float, str, bool)):
            return None  # predicate / non-scalar -> bail to the general path
        if k in cols:
            out[k] = v
        elif (isinstance(k, str) and k.startswith("label__") and v is True
              and "labels" not in cols and "type" in cols
              and not _looks_like_edge_dataframe(df)):
            out["type"] = k[len("label__"):]
        else:
            return None  # labels-list / unknown column -> bail
    return out


def _resident_seed_indexes(
    g: Plottable, nodes_df: DataFrameT, edges_df: DataFrameT,
    node: str, src: str, dst: str, direction: Direction,
) -> Optional[Tuple["NodeIdIndex", "AdjacencyIndex", ArrayNamespace, "Engine"]]:
    """(node_id_index, adjacency_index, xp, engine) when BOTH resident indexes
    validly cover this directed seeded hop on these EXACT frames (fingerprint +
    identity via get_valid), else None — callers keep the scan path, so a stale
    or absent index can never change results, only speed."""
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index import get_index_policy, get_registry
    from graphistry.compute.gfql.index.registry import EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID
    from graphistry.compute.gfql.index.engine_arrays import array_namespace
    if get_index_policy(g) == "off":
        return None
    registry = get_registry(g)
    if registry.is_empty():
        return None
    engine = _frame_engine(nodes_df)
    if engine is None:
        return None
    kind = EDGE_OUT_ADJ if direction == "forward" else EDGE_IN_ADJ
    engines = [engine]
    if engine == Engine.POLARS:
        # an index built with explicit engine='polars-gpu' serves the same eager
        # polars frames (same numpy sidecars + polars row-gather)
        engines.append(Engine.POLARS_GPU)
    adj = nid = None
    for eng_try in engines:
        adj = registry.get_valid(kind, edges_df, (src, dst), eng_try)
        nid = registry.get_valid(NODE_ID, nodes_df, (node,), eng_try)
        if adj is not None and nid is not None:
            break
    if adj is None or nid is None:
        return None
    xp, _ = array_namespace(engine)
    # get_valid returns the union type; kind selection above guarantees the concrete classes
    return cast("NodeIdIndex", nid), cast("AdjacencyIndex", adj), xp, engine


def _ids_to_key_array(
    vals: Union["SeriesT", Sequence[Any]], keys: ArrayLike, xp: ArrayNamespace,
) -> Optional[ArrayLike]:
    """Values (python list / Series / array) -> deduped backend array in the index
    key dtype, nulls dropped (null ids never link — matching the scan path's
    dropna semantics). None when the cast is not value-safe (mismatched families
    like str-vs-int decline to the scan path rather than risk false matches)."""
    try:
        if 'cudf' in str(type(vals).__module__):
            vals = vals.dropna()  # type: ignore[union-attr]  # cudf Series by module check
            raw = vals.values  # type: ignore[union-attr]  # device array; to_numpy() raises on nulls + round-trips host
        elif hasattr(vals, "to_numpy"):
            raw = vals.to_numpy()
        else:
            raw = vals
        arr = xp.asarray(raw)
        if arr.dtype.kind == "f":
            arr = arr[~xp.isnan(arr)]
        if arr.dtype.kind not in "iuf" or keys.dtype.kind not in "iuf":
            return None  # numeric id families only: object/str ids keep the scan path (null-object semantics)
        if arr.dtype != keys.dtype:
            common = xp.promote_types(arr.dtype, keys.dtype)
            if arr.dtype.kind in "iu" and keys.dtype.kind in "iu" and common.kind == "f":
                # int64<->uint64 promotes to float64, which collapses distinct ids
                # >= 2^53 into false matches; the scan path compares exactly -> decline.
                return None
            arr = arr.astype(common)
        return xp.unique(arr)
    except (TypeError, ValueError):
        return None


def _index_node_rows(
    nid: "NodeIdIndex", ids: Union["SeriesT", Sequence[Any]],
    xp: ArrayNamespace, engine: "Engine", nodes_df: DataFrameT,
    preserve_input_order: bool = False,
) -> Optional[DataFrameT]:
    """Node rows whose id is in ``ids`` via the resident node-id index (positional
    gather; row order is id-sorted, covered by the value-identical contract)."""
    from graphistry.compute.gfql.index.lookup import lookup_node_rows
    from graphistry.compute.gfql.index.engine_arrays import take_rows
    arr = _ids_to_key_array(ids, nid.keys_sorted, xp)
    if arr is None:
        return None
    positions = lookup_node_rows(nid, arr, xp)
    return take_rows(nodes_df, xp.sort(positions) if preserve_input_order else positions, engine)


def _frame_engine(df: DataFrameT) -> Optional["Engine"]:
    """The engine whose frame type ``df`` is, or None for anything the indexes do not cover."""
    from graphistry.Engine import Engine, is_polars_df
    mod = str(type(df).__module__)
    if is_polars_df(df):
        return Engine.POLARS
    if 'cudf' in mod:
        return Engine.CUDF
    if mod.startswith('pandas'):
        return Engine.PANDAS
    return None


def _resident_node_id_index(
    g: Plottable, nodes_df: DataFrameT, node: str,
) -> Optional[Tuple["NodeIdIndex", ArrayNamespace, "Engine"]]:
    """(node_id_index, xp, engine) when a valid resident node-id index covers this exact
    node frame, else None (no adjacency index required: a node-only lookup needs none)."""
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index import get_index_policy, get_registry
    from graphistry.compute.gfql.index.registry import NODE_ID, NodeIdIndex
    from graphistry.compute.gfql.index.engine_arrays import array_namespace
    if get_index_policy(g) == "off":
        return None
    registry = get_registry(g)
    if registry.is_empty():
        return None
    engine = _frame_engine(nodes_df)
    if engine is None:
        return None
    engines = [engine, Engine.POLARS_GPU] if engine == Engine.POLARS else [engine]
    for eng_try in engines:
        nid = registry.get_valid(NODE_ID, nodes_df, (node,), eng_try)
        if isinstance(nid, NodeIdIndex):
            xp, _ = array_namespace(engine)
            return nid, xp, engine
    return None


def _seed_rows_via_prop_index_frame(
    g: Plottable, nodes_df: DataFrameT, n0f: Dict[str, object], engine: "Engine",
) -> Optional[DataFrameT]:
    """Candidate seed rows through a resident node PROPERTY index covering one of the
    scalar predicates, else None (the caller re-applies the whole filter either way)."""
    from graphistry.compute.gfql.index import get_index_policy, get_registry
    from graphistry.compute.gfql.index.bindings import (
        _seed_rows_via_property_index as _prop_rows,
    )
    from graphistry.compute.gfql.index.engine_arrays import array_namespace, take_rows
    policy = get_index_policy(g)
    if policy == "off":
        return None
    registry = get_registry(g)
    if registry.is_empty() or not registry.node_prop_cols():
        return None
    xp, _ = array_namespace(engine)
    rows = _prop_rows(registry, nodes_df, n0f, engine, xp, policy=policy)
    if rows is None:
        return None
    return take_rows(nodes_df, rows, engine)


SeedRowsHow = Literal["node_id_index", "property_index", "scan"]
SeededReturn = Tuple[DataFrameT, DataFrameT, DataFrameT, bool]


def _seed_node_rows(
    g: Plottable, nodes_df: DataFrameT, n0f: Dict[str, object], node: str,
    nid_ctx: Optional[Tuple["NodeIdIndex", ArrayNamespace, "Engine"]],
    filter_dict: Optional[Dict[str, object]] = None,
) -> Tuple[DataFrameT, SeedRowsHow]:
    """Rows matching the scalar seed filter: node-id index when the predicate is on the
    binding column, else a resident property index, else a scan. The canonical filter
    (``filter_dict`` as written, or the resolved scalars) is re-applied to the candidates,
    so every branch keeps the full path's typed-error and comparison semantics."""
    from graphistry.compute.gfql.index.bindings import _filter_frame
    engine = _frame_engine(nodes_df)
    if engine is None:
        raise TypeError(f"unsupported node frame type {type(nodes_df).__name__}")
    seed: Optional[DataFrameT] = None
    how: SeedRowsHow = "scan"
    if nid_ctx is not None and node in n0f:
        nid, xp, idx_engine = nid_ctx
        seed = _index_node_rows(nid, [n0f[node]], xp, idx_engine, nodes_df)
        if seed is not None:
            how = "node_id_index"
    if seed is None:
        seed = _seed_rows_via_prop_index_frame(g, nodes_df, n0f, engine)
        if seed is not None:
            how = "property_index"
    if seed is None:
        seed = nodes_df
    return _filter_frame(seed, filter_dict if filter_dict is not None else n0f, engine), how


def _record_native_seed_lane(
    nodes_df: DataFrameT, *, seam: str, reason: str, hop_count: int, public_seed_scan: bool,
) -> None:
    """gfql_explain step for a native op-list lane that a resident index served."""
    from graphistry.compute.gfql.index.api import _record_indexed_traversal
    engine = _frame_engine(nodes_df)
    if engine is None:
        return
    _record_indexed_traversal(
        seam=seam, engine=engine, served=True, reason=reason, hop_count=hop_count,
        public_seed_scan=public_seed_scan,
        hop_details=[{"hop": 1}] if hop_count else None)


def _index_edge_rows(
    adj: "AdjacencyIndex", ids: Union["SeriesT", Sequence[Any]],
    xp: ArrayNamespace, engine: "Engine", edges_df: DataFrameT,
    preserve_input_order: bool = False,
) -> Optional[DataFrameT]:
    """Edge rows incident to ``ids`` on the indexed side via the CSR adjacency
    (searchsorted gather; replaces the O(E) isin scan)."""
    from graphistry.compute.gfql.index.lookup import lookup_edge_rows
    from graphistry.compute.gfql.index.engine_arrays import take_rows
    arr = _ids_to_key_array(ids, adj.keys_sorted, xp)
    if arr is None:
        return None
    rows, _ = lookup_edge_rows(adj, arr, xp)
    return take_rows(edges_df, xp.sort(rows) if preserve_input_order else rows, engine)
