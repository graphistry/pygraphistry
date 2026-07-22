"""Public-facing index lifecycle + planner entry, operating on a Plottable.

The registry rides on the Plottable as a private attribute (``_gfql_index_registry``)
and propagates through ``copy.copy``-based functional chaining. It is fingerprint-
validated at use time, so a rebind of ``.edges()``/``.nodes()`` safely invalidates
stale indexes (treated as absent, never a wrong answer).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Literal, Optional, cast

import pandas as pd

from graphistry.Engine import EngineAbstract, Engine, EngineAbstractType, resolve_engine
from graphistry.compute.typing import DataFrameT
from graphistry.Plottable import Plottable
from .registry import (
    AdjacencyIndex, GfqlIndexRegistry, EMPTY_REGISTRY, NodeIdIndex,
    EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID, ADJ_KINDS, ALL_KINDS,
)
from .build import build_adjacency_index, build_node_id_index
from .traverse import index_seeded_hop
from .cost import cost_gate_frac, seed_deg_sum, seed_id_array
from .policy import IndexPolicy, validate_index_policy
from .types import (
    AdjacencyIndexKind, EdgeIndexDirection, HopDirection, IndexKind,
    IndexTrace, IndexTraceStep,
)

# Private Plottable attachment keys. Keep access behind helpers.
POLICY_ATTR = "_gfql_index_policy"
REGISTRY_ATTR = "_gfql_index_registry"

# --- lightweight, thread-local index decision trace (for gfql_explain) -------
import threading as _threading
_TRACE = _threading.local()


class index_trace:
    """Context manager: capture the per-hop index-vs-scan decisions made inside."""
    def __enter__(self) -> IndexTrace:
        self.steps: IndexTrace = []
        self.prev = _get_trace_steps()
        _set_trace_steps(self.steps)
        return self.steps

    def __exit__(self, *exc: object) -> Literal[False]:
        _set_trace_steps(self.prev)
        return False


def _get_trace_steps() -> Optional[IndexTrace]:
    return cast(Optional[IndexTrace], getattr(_TRACE, "steps", None))


def _set_trace_steps(steps: Optional[IndexTrace]) -> None:
    _TRACE.steps = steps


def _record(decision: IndexTraceStep) -> None:
    steps = _get_trace_steps()
    if steps is not None:
        steps.append(decision)


def _trace_active() -> bool:
    """True only inside an ``index_trace()`` / ``gfql_explain`` context. Diagnostic
    enrichment (LP1) is computed only when this is True -> zero hot-path cost."""
    return _get_trace_steps() is not None


# Back-compat for existing private tests while helpers live in cost.py.
_seed_id_array = seed_id_array
_seed_deg_sum = seed_deg_sum

def get_registry(g: Plottable) -> GfqlIndexRegistry:
    return cast(GfqlIndexRegistry, getattr(g, REGISTRY_ATTR, EMPTY_REGISTRY))


def get_index_policy(g: Plottable) -> IndexPolicy:
    return cast(IndexPolicy, getattr(g, POLICY_ATTR, "use"))


def _attach(g: Plottable, registry: GfqlIndexRegistry) -> Plottable:
    res = copy.copy(g)
    setattr(res, REGISTRY_ATTR, registry)
    return res


def set_registry(g: Plottable, registry: GfqlIndexRegistry) -> Plottable:
    """Attach ``registry`` to a copy of ``g`` (public wrapper over ``_attach``).

    Used by the chain executor to re-point the resident index at an edge frame it
    augmented in place (see ``GfqlIndexRegistry.rebind_edges``)."""
    return _attach(g, registry)


def index_name(kind: IndexKind, column: Optional[str]) -> str:
    return f"{kind}:{column}" if column else kind


def _check_column(column: Optional[str], expected: str, kind: IndexKind) -> None:
    """A user-supplied ``column`` must match the binding the index keys on; a
    different column would be a silent no-op (registry is one-index-per-kind in
    v1), so reject it honestly rather than ignore it."""
    if column is not None and column != expected:
        raise NotImplementedError(
            f"GFQL index {kind!r} keys on the {expected!r} binding; a custom column "
            f"({column!r}) is not supported yet. Re-bind the graph or omit `column`."
        )


def _resolve_index_engine(engine: EngineAbstractType, g: Plottable) -> Engine:
    """Engine for index build/validation: AUTO keeps resident polars frames polars.

    ``resolve_engine(AUTO)`` maps polars input frames to PANDAS (the legacy
    input-format policy), which would make ``create_index`` coerce-and-REPLACE the
    user's polars frames with pandas copies — and then every later
    ``engine='polars'`` query pays a full-frame pandas->polars re-conversion, an
    O(E) tax per call that dwarfs the indexed hop itself. The index layer is
    engine-polymorphic (numpy sidecar arrays + polars row-gather), so when the
    caller said AUTO and the resident frames are polars, index them in place.
    Explicit engines are honored unchanged.
    """
    eng = resolve_engine(engine, g)
    abstract = EngineAbstract(engine) if isinstance(engine, str) else engine
    if abstract == EngineAbstract.AUTO and eng == Engine.PANDAS:
        df = g._edges if g._edges is not None else g._nodes
        if df is not None and 'polars' in str(type(df).__module__):
            return Engine.POLARS
    return eng


def _is_resident_index_valid(
    g: Plottable,
    kind: IndexKind,
    engine: EngineAbstractType = EngineAbstract.AUTO,
) -> bool:
    """True when a resident index still matches the current graph frames."""
    eng = _resolve_index_engine(engine, g)
    registry = get_registry(g)
    if kind in ADJ_KINDS:
        src, dst = g._source, g._destination
        if src is None or dst is None or g._edges is None:
            return False
        return registry.get_valid(kind, g._edges, (src, dst), eng) is not None
    if kind == NODE_ID:
        node_col = g._node
        if node_col is None or g._nodes is None:
            return False
        return registry.get_valid(NODE_ID, g._nodes, (node_col,), eng) is not None
    return False


def create_index(
    g: Plottable,
    kind: IndexKind,
    *,
    column: Optional[str] = None,
    name: Optional[str] = None,
    engine: EngineAbstractType = EngineAbstract.AUTO,
) -> Plottable:
    """Eagerly build a GFQL physical index and return a new Plottable carrying it.

    ``kind``: 'edge_out_adj' | 'edge_in_adj' | 'node_id'. ``column`` (if given) must
    match the index's natural binding (src/dst/node) — a mismatch raises rather than
    silently no-op. ``name`` overrides the display handle. Pay-as-you-go: cost is
    O(E log E) once, amortized over later seeded queries.
    """
    from dataclasses import replace
    eng = _resolve_index_engine(engine, g)
    # Build over frames already in the target engine so the index arrays land on
    # the right backend (cupy for cudf, numpy otherwise). No-op when already in-engine.
    from graphistry.compute.ComputeMixin import _coerce_input_formats
    g = _coerce_input_formats(g, eng)
    registry = get_registry(g)

    if kind in ADJ_KINDS:
        src, dst = g._source, g._destination
        if src is None or dst is None or g._edges is None:
            raise ValueError(
                "edge adjacency index requires bound edges with source/destination columns"
            )
        adj_kind = cast(AdjacencyIndexKind, kind)
        key_col = src if adj_kind == EDGE_OUT_ADJ else dst
        _check_column(column, key_col, adj_kind)
        other = dst if adj_kind == EDGE_OUT_ADJ else src
        idx = build_adjacency_index(g._edges, adj_kind, key_col, other, g._edge, eng, (src, dst))
        idx = replace(idx, name=name or index_name(adj_kind, key_col))
        registry = registry.with_index(adj_kind, idx)
        return _attach(g, registry)

    if kind == NODE_ID:
        g2 = g.materialize_nodes() if g._nodes is None else g
        node_col = g2._node
        assert node_col is not None and g2._nodes is not None
        _check_column(column, node_col, kind)
        node_idx = build_node_id_index(g2._nodes, node_col, eng)
        if node_idx is None:
            raise ValueError(
                f"Cannot build a {NODE_ID!r} index: node id column {node_col!r} has "
                f"duplicate values (a node-id index requires unique ids). Seeded "
                f"traversal still works via the un-indexed node materialization path."
            )
        node_idx = replace(node_idx, name=name or index_name(kind, node_col))
        registry = registry.with_index(NODE_ID, node_idx)
        return _attach(g2, registry)

    raise ValueError(f"Unknown GFQL index kind: {kind!r}. Expected one of {ALL_KINDS}.")


def drop_index(g: Plottable, kind: Optional[IndexKind] = None) -> Plottable:
    """Drop one index (by kind) or all indexes (kind=None). Idempotent."""
    registry = get_registry(g)
    if kind is None:
        return _attach(g, EMPTY_REGISTRY)
    return _attach(g, registry.without(kind))


def show_indexes(g: Plottable) -> pd.DataFrame:
    """Return a pandas DataFrame describing resident indexes (empty if none).

    ``valid`` reflects live fingerprint validity against the current frames — a
    stale index (after a ``.edges()``/``.nodes()`` rebind) shows ``valid=False`` and
    is auto-skipped (scan fallback) until rebuilt. ``nbytes`` is the resident
    sidecar-array footprint (the pay-as-you-go memory signal).
    """
    from .registry import index_nbytes

    registry = get_registry(g)
    rows: List[Dict[str, object]] = []
    for kind in registry.kinds():
        idx = registry.get(kind)
        assert idx is not None  # iterating registry.kinds() -> present
        if kind in ADJ_KINDS:
            assert g._source is not None and g._destination is not None
            adj = cast(AdjacencyIndex, idx)
            valid = registry.get_valid(kind, g._edges, (g._source, g._destination), adj.engine) is not None
            n_keys, n_rows = adj.n_keys, adj.n_edges
        else:  # NODE_ID
            node_idx = cast(NodeIdIndex, idx)
            valid = g._nodes is not None and registry.get_valid(
                NODE_ID, g._nodes, (node_idx.key_col,), node_idx.engine) is not None
            n_keys, n_rows = node_idx.n_nodes, 0
        rows.append({
            "name": idx.name or index_name(kind, idx.key_col),
            "kind": kind,
            "key_col": idx.key_col,
            "engine": idx.engine.value,
            "backend": idx.backend,
            "n_keys": n_keys,
            "n_rows": n_rows,
            "nbytes": index_nbytes(idx),
            "valid": valid,
        })
    cols = ["name", "kind", "key_col", "engine", "backend", "n_keys", "n_rows", "nbytes", "valid"]
    return pd.DataFrame(rows, columns=cols)


def gfql_index_edges(g: Plottable, direction: EdgeIndexDirection = "both",
                     engine: EngineAbstractType = EngineAbstract.AUTO) -> Plottable:
    """Convenience: build edge adjacency index(es). direction: 'forward'|'reverse'|'both'."""
    if direction in ("forward", "both"):
        g = create_index(g, EDGE_OUT_ADJ, engine=engine)
    if direction in ("reverse", "both"):
        g = create_index(g, EDGE_IN_ADJ, engine=engine)
    return g


def gfql_index_all(g: Plottable,
                   engine: EngineAbstractType = EngineAbstract.AUTO) -> Plottable:
    """Convenience: build out+in adjacency + (when ids are unique) node_id indexes.

    The node_id index is an optional materialization accelerator; if node ids aren't
    unique it can't be built (a unique-key CSR can't reproduce the scan's all-rows-
    per-id semantics), so this convenience SKIPS it rather than raising — seeded
    traversal stays correct via the un-indexed node materialization path. (Explicit
    ``create_index(NODE_ID)`` still raises, since the caller asked for it specifically.)"""
    g = gfql_index_edges(g, "both", engine=engine)
    try:
        g = create_index(g, NODE_ID, engine=engine)
    except ValueError:
        pass  # non-unique node ids -> skip the node_id accelerator (adjacency still built)
    return g


# ---- planner entry ---------------------------------------------------------

# Coverage: features the index fast path does NOT yet handle -> caller scans.
def _hop_is_index_coverable(
    *,
    nodes: Optional[DataFrameT],
    to_fixed_point: bool,
    hops: Optional[int],
    min_hops: Optional[int],
    max_hops: Optional[int],
    output_min_hops: Optional[int],
    output_max_hops: Optional[int],
    label_node_hops: Optional[str],
    label_edge_hops: Optional[str],
    label_seeds: bool,
    edge_match: Optional[object],
    source_node_match: Optional[object],
    destination_node_match: Optional[object],
    source_node_query: Optional[str],
    destination_node_query: Optional[str],
    edge_query: Optional[str],
    include_zero_hop_seed: bool,
    target_wave_front: Optional[DataFrameT],
    return_as_wave_front: bool = False,
) -> bool:
    if nodes is None:
        return False
    if any(x is not None for x in (
        min_hops if (min_hops is not None and min_hops > 1) else None,
        output_min_hops, output_max_hops, label_node_hops, label_edge_hops,
        source_node_match, destination_node_match,
        source_node_query, destination_node_query, edge_query, target_wave_front,
    )):
        return False
    # Typed-edge edge_match: coverable only as a simple scalar-equality filter on the
    # wavefront (chain/Cypher) path, where index_seeded_hop applies it per-hop parity-
    # exact with the scan's filter_edges_by_dict. Predicate/membership forms and the
    # direct-hop (non-wavefront) path stay on scan.
    if edge_match is not None:
        from .traverse import is_simple_equality_edge_match
        if not (
            return_as_wave_front
            and isinstance(edge_match, dict)
            and is_simple_equality_edge_match(edge_match)
        ):
            return False
    if label_seeds or include_zero_hop_seed:
        return False
    if not to_fixed_point and (not isinstance(hops, int) or hops < 1):
        return False
    return True


def _ensure_indexes(
    g: Plottable,
    registry: GfqlIndexRegistry,
    direction: HopDirection,
    engine: Engine,
    policy: IndexPolicy,
    nodes: DataFrameT,
    src: str,
    dst: str,
    node_col: str,
) -> GfqlIndexRegistry:
    """auto/force: build the indexes this seeded hop needs (opt-in pay-as-you-go).

    force => always build missing; auto => build only when the query looks
    selective (frontier small vs E), else leave registry as-is (scan).
    """
    needed: List[AdjacencyIndexKind] = []
    if direction in ("forward", "undirected"):
        needed.append(EDGE_OUT_ADJ)
    if direction in ("reverse", "undirected"):
        needed.append(EDGE_IN_ADJ)
    if policy == "auto":
        try:
            E = int(g._edges.shape[0])
            F = int(nodes.shape[0])
            if not (F <= max(1024, 0.001 * E)):
                return registry  # not selective enough to amortize a build
        except (AttributeError, TypeError, ValueError):
            return registry
    for kind in needed:
        if registry.get_valid(kind, g._edges, (src, dst), engine) is None:
            if kind == EDGE_OUT_ADJ:
                idx = build_adjacency_index(g._edges, kind, src, dst, g._edge, engine, (src, dst))
            else:
                idx = build_adjacency_index(g._edges, kind, dst, src, g._edge, engine, (src, dst))
            registry = registry.with_index(kind, idx)
    if registry.get_valid(NODE_ID, g._nodes, (node_col,), engine) is None:
        node_idx = build_node_id_index(g._nodes, node_col, engine)
        if node_idx is not None:  # None => non-unique ids; skip (scan materialization)
            registry = registry.with_index(NODE_ID, node_idx)
    return registry


def maybe_index_hop(
    g: Plottable, engine: Engine, *, nodes: Optional[DataFrameT], hops: Optional[int], direction: HopDirection, return_as_wave_front: bool,
    to_fixed_point: bool = False, policy: Optional[IndexPolicy] = "use", **rest: object,
) -> Optional[Plottable]:
    """Planner entry called from hop(). Returns an index-built subgraph, or None to
    fall back to the scan/join path.

    Cost gate: only route to the index when (a) a valid matching index is resident
    (or buildable under auto/force), (b) the query is covered, (c) the frontier is
    not so large that a full scan is cheaper. Correctness is identical either way.
    """
    resolved_policy: IndexPolicy = validate_index_policy(policy) or "use"

    # Diagnostic trace (LP1) is populated only inside an explain context — build the
    # base record + a `_bail` helper that logs *why* we fell back to scan. All of this
    # is skipped entirely when not tracing, so the hot path pays nothing.
    trace = _trace_active()
    diag: IndexTraceStep = {}
    if trace:
        diag = {
            "op": "hop", "direction": direction, "hops": hops,
            "policy": resolved_policy, "engine": engine.value,
        }
        try:
            if nodes is not None:
                diag["frontier_n"] = int(nodes.shape[0])
        except (AttributeError, TypeError, ValueError):
            pass

    def _bail(reason: str) -> Optional[Plottable]:
        if trace:
            _record(cast(IndexTraceStep, {**diag, "path": "scan", "decision_reason": reason}))
        return None

    if resolved_policy == "off":
        return _bail("policy=off")
    registry = get_registry(g)
    if registry.is_empty() and resolved_policy not in ("auto", "force"):
        return _bail("no resident index (policy=use)")

    min_hops = cast(Optional[int], rest.get("min_hops"))
    max_hops = cast(Optional[int], rest.get("max_hops"))
    output_min_hops = cast(Optional[int], rest.get("output_min_hops"))
    output_max_hops = cast(Optional[int], rest.get("output_max_hops"))
    label_node_hops = cast(Optional[str], rest.get("label_node_hops"))
    label_edge_hops = cast(Optional[str], rest.get("label_edge_hops"))
    label_seeds = cast(bool, rest.get("label_seeds", False))
    source_node_query = cast(Optional[str], rest.get("source_node_query"))
    destination_node_query = cast(Optional[str], rest.get("destination_node_query"))
    edge_query = cast(Optional[str], rest.get("edge_query"))
    include_zero_hop_seed = cast(bool, rest.get("include_zero_hop_seed", False))
    target_wave_front = cast(Optional[DataFrameT], rest.get("target_wave_front"))

    if not _hop_is_index_coverable(
        nodes=nodes, to_fixed_point=to_fixed_point, hops=hops,
        min_hops=min_hops, max_hops=max_hops,
        output_min_hops=output_min_hops,
        output_max_hops=output_max_hops,
        label_node_hops=label_node_hops,
        label_edge_hops=label_edge_hops,
        label_seeds=label_seeds,
        edge_match=rest.get("edge_match"),
        source_node_match=rest.get("source_node_match"),
        destination_node_match=rest.get("destination_node_match"),
        source_node_query=source_node_query,
        destination_node_query=destination_node_query,
        edge_query=edge_query,
        include_zero_hop_seed=include_zero_hop_seed,
        target_wave_front=target_wave_front,
        return_as_wave_front=return_as_wave_front,
    ):
        return _bail("query not index-coverable")
    assert nodes is not None

    node_col = g._node
    src, dst = g._source, g._destination
    if node_col is None or src is None or dst is None or g._edges is None or g._nodes is None:
        return _bail("graph missing node/edge columns")

    if resolved_policy in ("auto", "force"):
        registry = _ensure_indexes(g, registry, direction, engine, resolved_policy, nodes, src, dst, node_col)
    if registry.is_empty():
        return _bail("no index available (build declined)")

    # Cost gate: if the frontier covers a large fraction of distinct sources, the
    # scan path is competitive — fall back (avoids index overhead on bulk-ish hops).
    idx0 = cast(Optional[AdjacencyIndex], registry.get_valid(
        EDGE_OUT_ADJ if direction != "reverse" else EDGE_IN_ADJ, g._edges, (src, dst), engine
    ))
    frac = cost_gate_frac(engine)
    if trace and idx0 is not None:
        # Free fanout estimate (Σ seed degree) from the CSR offsets — the planner
        # signal the report wants EXPLAIN to surface (not just used-index yes/no).
        seed_ids = seed_id_array(nodes, node_col)
        deg_sum = seed_deg_sum(idx0, seed_ids) if seed_ids is not None else None
        diag["n_keys"] = int(idx0.n_keys)
        diag["seed_deg_sum"] = deg_sum
        diag["est_result_rows"] = deg_sum
        diag["threshold_frac"] = frac
    if idx0 is None:
        # required direction not resident (undirected needs both); let driver decide
        pass
    elif resolved_policy != "force":
        try:
            frontier_n = int(nodes.shape[0])
            if idx0.n_keys > 0 and frontier_n >= frac * idx0.n_keys:
                return _bail(
                    f"frontier {frontier_n} >= {frac}*n_keys "
                    f"({frac * idx0.n_keys:.0f}) -> scan cheaper"
                )
        except (AttributeError, TypeError, ValueError):
            pass

    # Honor max_hops: the scan resolves the hop count as ``max_hops or hops``
    # (compute/hop.py); the index must run the SAME number of accumulating hops.
    # (regression: max_hops was passed through *rest and silently ignored — the index ran
    # `hops` (default 1) while the scan ran max_hops → wrong answer.)
    eff_hops = max_hops if max_hops is not None else hops
    result = index_seeded_hop(
        g, registry, nodes=nodes, node_col=node_col, src=src, dst=dst, engine=engine,
        hops=eff_hops, to_fixed_point=to_fixed_point, direction=direction,
        return_as_wave_front=return_as_wave_front,
        edge_match=cast(Optional[dict], rest.get("edge_match")),
    )
    if trace:
        _record(cast(IndexTraceStep, {
            **diag, "hops": eff_hops,
            "path": "index" if result is not None else "scan",
            "decision_reason": (
                "frontier below cost gate -> index" if result is not None
                else "index path not applicable -> scan"
            ),
        }))
    return result
