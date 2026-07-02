"""GFQL physical index registry — immutable, fingerprinted sidecars.

The registry holds typed indexes (adjacency CSR, node-id) keyed by ``kind``. It
is attached to a Plottable as a private attribute and travels with it. Because
PyGraphistry is pure-functional, an index is only valid for the exact frame it
was built over; a cheap structural fingerprint (object id + length + bindings +
engine) detects when ``.edges()``/``.nodes()`` rebinding has invalidated it, in
which case the planner treats the index as absent (a safe miss, never a wrong
answer).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

from graphistry.Engine import Engine

# Index kinds (v1). Property/label/type indexes share this registry shape later.
EDGE_OUT_ADJ = "edge_out_adj"
EDGE_IN_ADJ = "edge_in_adj"
NODE_ID = "node_id"

ADJ_KINDS = (EDGE_OUT_ADJ, EDGE_IN_ADJ)
ALL_KINDS = (EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID)


def frame_fingerprint(df: Any, cols: Tuple[str, ...], engine: Engine) -> Tuple:
    """Cheap O(1) structural fingerprint of the frame an index is built over: length
    + bound columns + engine. This is a SECONDARY guard; the primary validity check
    is object IDENTITY (``source_ref is df``, see ``get_valid``). We deliberately do
    NOT use ``id(df)`` here — a GC'd frame's id can be recycled by a new same-shape
    frame, which `id`-equality would accept (stale index → wrong answer). Holding
    a strong ref + identity is recycle-proof. (Pure-functional rebind via
    ``.edges()``/``.nodes()`` yields a new object → identity miss → safe scan.)"""
    try:
        n = int(df.shape[0]) if df is not None else -1
    except Exception:
        n = -1
    return (n, tuple(cols), engine.value)


@dataclass(frozen=True)
class AdjacencyIndex:
    """CSR adjacency over edge **row positions**, keyed by one endpoint.

    Lookup of a frontier of ids is O(F log U + result) via searchsorted +
    vectorized range expansion — sublinear in E, never a full edge scan.
    """
    kind: str                 # EDGE_OUT_ADJ | EDGE_IN_ADJ
    key_col: str              # endpoint we key on (src for out, dst for in)
    other_col: str            # opposite endpoint (the neighbor we emit)
    edge_id_col: Optional[str]  # edge-id binding if present (else row pos == id)
    keys_sorted: Any          # distinct key ids, ascending (len U)  [array]
    group_offsets: Any        # CSR offsets into row_positions (len U+1) [array]
    row_positions: Any        # edge row indices grouped by key (len E) [array]
    other_values: Any         # neighbor id per edge row, ORIGINAL order (len E) [array]
    backend: str              # 'numpy' | 'cupy'
    engine: Engine
    fingerprint: Tuple = field(compare=False, default=())
    source_ref: Any = field(compare=False, default=None)  # the indexed frame (identity guard)
    n_edges: int = 0
    n_keys: int = 0
    name: Optional[str] = None


@dataclass(frozen=True)
class NodeIdIndex:
    """Sorted node-id -> node row position (find seed/endpoint rows fast)."""
    key_col: str
    keys_sorted: Any
    row_positions: Any
    backend: str
    engine: Engine
    fingerprint: Tuple = field(compare=False, default=())
    source_ref: Any = field(compare=False, default=None)  # the indexed frame (identity guard, I5)
    n_nodes: int = 0
    name: Optional[str] = None


@dataclass(frozen=True)
class GfqlIndexRegistry:
    """Immutable kind -> index map. ``with_index`` / ``without`` return copies."""
    indexes: Dict[str, Any] = field(default_factory=dict)

    def with_index(self, kind: str, index: Any) -> "GfqlIndexRegistry":
        new = dict(self.indexes)
        new[kind] = index
        return GfqlIndexRegistry(new)

    def without(self, kind: str) -> "GfqlIndexRegistry":
        new = dict(self.indexes)
        new.pop(kind, None)
        return GfqlIndexRegistry(new)

    def get(self, kind: str) -> Optional[Any]:
        return self.indexes.get(kind)

    def has(self, kind: str) -> bool:
        return kind in self.indexes

    def kinds(self) -> Tuple[str, ...]:
        return tuple(sorted(self.indexes.keys()))

    def is_empty(self) -> bool:
        return not self.indexes

    def get_valid(self, kind: str, df: Any, cols: Tuple[str, ...], engine: Engine) -> Optional[Any]:
        """Return the index for ``kind`` only if its fingerprint still matches the
        live frame + engine; else None (treat as absent)."""
        idx = self.indexes.get(kind)
        if idx is None:
            return None
        if idx.engine != engine:
            return None
        # Primary: object IDENTITY — recycle-proof, since the index holds a strong
        # ref so the frame's id can't be reused while indexed. `is` on a rebound frame
        # is False → safe miss. (source_ref None only for legacy/hand-built indexes.)
        if idx.source_ref is not None and idx.source_ref is not df:
            return None
        if idx.fingerprint != frame_fingerprint(df, cols, engine):
            return None
        return idx


def index_nbytes(idx: Any) -> int:
    """Approximate resident memory of an index's sidecar arrays (bytes)."""
    total = 0
    for attr in ("keys_sorted", "group_offsets", "row_positions", "other_values"):
        arr = getattr(idx, attr, None)
        if arr is not None:
            total += int(getattr(arr, "nbytes", 0))
    return total


EMPTY_REGISTRY = GfqlIndexRegistry()
