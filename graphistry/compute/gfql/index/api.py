"""Public-facing index lifecycle + planner entry, operating on a Plottable.

The registry rides on the Plottable as a private attribute (``_gfql_index_registry``)
and propagates through ``copy.copy``-based functional chaining. It is fingerprint-
validated at use time, so a rebind of ``.edges()``/``.nodes()`` safely invalidates
stale indexes (treated as absent, never a wrong answer).
"""
from __future__ import annotations

import copy
from typing import Any, Optional, Union, cast

from graphistry.Engine import EngineAbstract, Engine, resolve_engine
from graphistry.Plottable import Plottable
from .registry import (
    GfqlIndexRegistry, EMPTY_REGISTRY,
    EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID, ADJ_KINDS, ALL_KINDS,
)
from .build import build_adjacency_index, build_node_id_index
from .traverse import index_seeded_hop

REGISTRY_ATTR = "_gfql_index_registry"

# Index-vs-scan crossover fraction (of distinct source keys): past this frontier a
# full scan is cheaper than per-seed index probes, so `use` falls back to scan and
# never loses to the un-indexed path. Engine-aware because vectorized-scan engines
# (polars/cudf/GPU) scan far faster than pandas, so their crossover is much smaller
# (see test_index_cost_gate_engine_aware). Measured N=1e5 deg8: pandas ~0.5, polars ~0.02. GPU values provisional
# (dgx-gated) — conservatively grouped with polars.
_COST_GATE_FRAC = {Engine.PANDAS: 0.5}
_COST_GATE_FRAC_DEFAULT = 0.02

# --- lightweight, thread-local index decision trace (for gfql_explain) -------
import threading as _threading
_TRACE = _threading.local()


class index_trace:
    """Context manager: capture the per-hop index-vs-scan decisions made inside."""
    def __enter__(self):
        self.steps = []
        self.prev = getattr(_TRACE, "steps", None)
        _TRACE.steps = self.steps
        return self.steps

    def __exit__(self, *exc):
        _TRACE.steps = self.prev
        return False


def _record(decision: dict) -> None:
    steps = getattr(_TRACE, "steps", None)
    if steps is not None:
        steps.append(decision)


def get_registry(g: Plottable) -> GfqlIndexRegistry:
    return getattr(g, REGISTRY_ATTR, EMPTY_REGISTRY)


def _attach(g: Plottable, registry: GfqlIndexRegistry) -> Plottable:
    res = copy.copy(g)
    setattr(res, REGISTRY_ATTR, registry)
    return res


def index_name(kind: str, column: Optional[str]) -> str:
    return f"{kind}:{column}" if column else kind


def _check_column(column: Optional[str], expected: str, kind: str) -> None:
    """A user-supplied ``column`` must match the binding the index keys on; a
    different column would be a silent no-op (registry is one-index-per-kind in
    v1), so reject it honestly rather than ignore it."""
    if column is not None and column != expected:
        raise NotImplementedError(
            f"GFQL index {kind!r} keys on the {expected!r} binding; a custom column "
            f"({column!r}) is not supported yet. Re-bind the graph or omit `column`."
        )


def create_index(
    g: Plottable,
    kind: str,
    *,
    column: Optional[str] = None,
    name: Optional[str] = None,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
) -> Plottable:
    """Eagerly build a GFQL physical index and return a new Plottable carrying it.

    ``kind``: 'edge_out_adj' | 'edge_in_adj' | 'node_id'. ``column`` (if given) must
    match the index's natural binding (src/dst/node) — a mismatch raises rather than
    silently no-op. ``name`` overrides the display handle. Pay-as-you-go: cost is
    O(E log E) once, amortized over later seeded queries.
    """
    from dataclasses import replace
    eng = resolve_engine(cast(Any, engine), g)
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
        key_col = src if kind == EDGE_OUT_ADJ else dst
        _check_column(column, key_col, kind)
        other = dst if kind == EDGE_OUT_ADJ else src
        idx = build_adjacency_index(g._edges, kind, key_col, other, g._edge, eng, (src, dst))
        idx = replace(idx, name=name or index_name(kind, key_col))
        registry = registry.with_index(kind, idx)
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


def drop_index(g: Plottable, kind: Optional[str] = None) -> Plottable:
    """Drop one index (by kind) or all indexes (kind=None). Idempotent."""
    registry = get_registry(g)
    if kind is None:
        return _attach(g, EMPTY_REGISTRY)
    return _attach(g, registry.without(kind))


def show_indexes(g: Plottable) -> Any:
    """Return a pandas DataFrame describing resident indexes (empty if none).

    ``valid`` reflects live fingerprint validity against the current frames — a
    stale index (after a ``.edges()``/``.nodes()`` rebind) shows ``valid=False`` and
    is auto-skipped (scan fallback) until rebuilt. ``nbytes`` is the resident
    sidecar-array footprint (the pay-as-you-go memory signal).
    """
    import pandas as pd
    from .registry import index_nbytes

    registry = get_registry(g)
    rows = []
    for kind in registry.kinds():
        idx = registry.get(kind)
        assert idx is not None  # iterating registry.kinds() -> present
        if kind in ADJ_KINDS:
            assert g._source is not None and g._destination is not None
            valid = registry.get_valid(kind, g._edges, (g._source, g._destination), idx.engine) is not None
            n_keys, n_rows = idx.n_keys, idx.n_edges
        else:  # NODE_ID
            valid = g._nodes is not None and registry.get_valid(
                NODE_ID, g._nodes, (idx.key_col,), idx.engine) is not None
            n_keys, n_rows = idx.n_nodes, 0
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


def gfql_index_edges(g: Plottable, direction: str = "both",
                     engine: Union[EngineAbstract, str] = EngineAbstract.AUTO) -> Plottable:
    """Convenience: build edge adjacency index(es). direction: 'forward'|'reverse'|'both'."""
    if direction in ("forward", "both"):
        g = create_index(g, EDGE_OUT_ADJ, engine=engine)
    if direction in ("reverse", "both"):
        g = create_index(g, EDGE_IN_ADJ, engine=engine)
    return g


def gfql_index_all(g: Plottable,
                   engine: Union[EngineAbstract, str] = EngineAbstract.AUTO) -> Plottable:
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
    *, nodes, to_fixed_point, hops, min_hops, max_hops,
    output_min_hops, output_max_hops, label_node_hops, label_edge_hops,
    label_seeds, edge_match, source_node_match, destination_node_match,
    source_node_query, destination_node_query, edge_query,
    include_zero_hop_seed, target_wave_front,
) -> bool:
    if nodes is None:
        return False
    if any(x is not None for x in (
        min_hops if (min_hops is not None and min_hops > 1) else None,
        output_min_hops, output_max_hops, label_node_hops, label_edge_hops,
        edge_match, source_node_match, destination_node_match,
        source_node_query, destination_node_query, edge_query, target_wave_front,
    )):
        return False
    if label_seeds or include_zero_hop_seed:
        return False
    if not to_fixed_point and (not isinstance(hops, int) or hops < 1):
        return False
    return True


def _ensure_indexes(g, registry, direction, engine, policy, nodes, src, dst, node_col):
    """auto/force: build the indexes this seeded hop needs (opt-in pay-as-you-go).

    force => always build missing; auto => build only when the query looks
    selective (frontier small vs E), else leave registry as-is (scan).
    """
    needed = []
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
        except Exception:
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
    g: Plottable, engine: Engine, *, nodes, hops, direction, return_as_wave_front,
    to_fixed_point=False, policy: str = "use", **rest,
) -> Optional[Plottable]:
    """Planner entry called from hop(). Returns an index-built subgraph, or None to
    fall back to the scan/join path.

    Cost gate: only route to the index when (a) a valid matching index is resident
    (or buildable under auto/force), (b) the query is covered, (c) the frontier is
    not so large that a full scan is cheaper. Correctness is identical either way.
    """
    if policy == "off":
        return None
    registry = get_registry(g)
    if registry.is_empty() and policy not in ("auto", "force"):
        return None
    if not _hop_is_index_coverable(
        nodes=nodes, to_fixed_point=to_fixed_point, hops=hops,
        min_hops=rest.get("min_hops"), max_hops=rest.get("max_hops"),
        output_min_hops=rest.get("output_min_hops"),
        output_max_hops=rest.get("output_max_hops"),
        label_node_hops=rest.get("label_node_hops"),
        label_edge_hops=rest.get("label_edge_hops"),
        label_seeds=rest.get("label_seeds", False),
        edge_match=rest.get("edge_match"),
        source_node_match=rest.get("source_node_match"),
        destination_node_match=rest.get("destination_node_match"),
        source_node_query=rest.get("source_node_query"),
        destination_node_query=rest.get("destination_node_query"),
        edge_query=rest.get("edge_query"),
        include_zero_hop_seed=rest.get("include_zero_hop_seed", False),
        target_wave_front=rest.get("target_wave_front"),
    ):
        return None

    node_col = g._node
    src, dst = g._source, g._destination
    if node_col is None or src is None or dst is None or g._edges is None or g._nodes is None:
        return None

    if policy in ("auto", "force"):
        registry = _ensure_indexes(g, registry, direction, engine, policy, nodes, src, dst, node_col)
    if registry.is_empty():
        return None

    # Cost gate: if the frontier covers a large fraction of distinct sources, the
    # scan path is competitive — fall back (avoids index overhead on bulk-ish hops).
    idx0 = registry.get_valid(
        EDGE_OUT_ADJ if direction != "reverse" else EDGE_IN_ADJ, g._edges, (src, dst), engine
    )
    if idx0 is None:
        # required direction not resident (undirected needs both); let driver decide
        pass
    elif policy != "force":
        try:
            frontier_n = int(nodes.shape[0])
            frac = _COST_GATE_FRAC.get(engine, _COST_GATE_FRAC_DEFAULT)
            if idx0.n_keys > 0 and frontier_n >= frac * idx0.n_keys:
                return None
        except Exception:
            pass

    # Honor max_hops: the scan resolves the hop count as ``max_hops or hops``
    # (compute/hop.py); the index must run the SAME number of accumulating hops.
    # (regression: max_hops was passed through *rest and silently ignored — the index ran
    # `hops` (default 1) while the scan ran max_hops → wrong answer.)
    _max_hops = rest.get("max_hops")
    eff_hops = _max_hops if _max_hops is not None else hops
    result = index_seeded_hop(
        g, registry, nodes=nodes, node_col=node_col, src=src, dst=dst, engine=engine,
        hops=eff_hops, to_fixed_point=to_fixed_point, direction=direction,
        return_as_wave_front=return_as_wave_front,
    )
    _record({
        "op": "hop", "direction": direction, "hops": eff_hops,
        "path": "index" if result is not None else "scan",
        "policy": policy, "engine": engine.value,
    })
    return result
