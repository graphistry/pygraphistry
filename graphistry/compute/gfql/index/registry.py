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
from typing import Any, Dict, Literal, Optional, Tuple, Union, cast

from graphistry.Engine import Engine
from graphistry.compute.typing import DataFrameT
from .types import AdjacencyIndexKind, ArrayLike, IndexBackend, IndexKind

# Index kinds (v1). Property/label/type indexes share this registry shape later.
EDGE_OUT_ADJ: AdjacencyIndexKind = "edge_out_adj"
EDGE_IN_ADJ: AdjacencyIndexKind = "edge_in_adj"
NODE_ID: IndexKind = "node_id"
NODE_PROP: IndexKind = "node_prop"

ADJ_KINDS: Tuple[AdjacencyIndexKind, ...] = (EDGE_OUT_ADJ, EDGE_IN_ADJ)
ALL_KINDS: Tuple[IndexKind, ...] = (EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID, NODE_PROP)

FrameFingerprint = Tuple[int, Tuple[str, ...], str]


def frame_fingerprint(df: DataFrameT, cols: Tuple[str, ...], engine: Engine) -> FrameFingerprint:
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
    kind: AdjacencyIndexKind  # EDGE_OUT_ADJ | EDGE_IN_ADJ
    key_col: str              # endpoint we key on (src for out, dst for in)
    other_col: str            # opposite endpoint (the neighbor we emit)
    edge_id_col: Optional[str]  # edge-id binding if present (else row pos == id)
    keys_sorted: ArrayLike    # distinct key ids, ascending (len U)  [array]
    group_offsets: ArrayLike  # CSR offsets into row_positions (len U+1) [array]
    row_positions: ArrayLike  # edge row indices grouped by key (len E) [array]
    other_values: ArrayLike   # neighbor id per edge row, ORIGINAL order (len E) [array]
    backend: IndexBackend     # 'numpy' | 'cupy'
    engine: Engine
    fingerprint: FrameFingerprint = field(compare=False, default=(-1, (), ""))
    source_ref: Optional[DataFrameT] = field(compare=False, default=None)  # the indexed frame (identity guard)
    n_edges: int = 0
    n_keys: int = 0
    name: Optional[str] = None


@dataclass(frozen=True)
class NodeIdIndex:
    """Sorted node-id -> node row position (find seed/endpoint rows fast)."""
    key_col: str
    keys_sorted: ArrayLike
    row_positions: ArrayLike
    backend: IndexBackend
    engine: Engine
    fingerprint: FrameFingerprint = field(compare=False, default=(-1, (), ""))
    source_ref: Optional[DataFrameT] = field(compare=False, default=None)  # the indexed frame (identity guard, I5)
    n_nodes: int = 0
    name: Optional[str] = None


@dataclass(frozen=True)
class NodePropIndex:
    """Sorted node PROPERTY value -> node row positions (CSR, duplicates allowed).

    The secondary index: a seed predicate on a non-key column (``{id: 42}`` where
    the graph's node id is some other column) otherwise costs a full node scan.
    Unlike :class:`NodeIdIndex` this keeps ALL rows per key in CSR form, so
    non-unique properties are indexable — the caller applies any residual
    predicates to the gathered candidates, so results are identical either way.
    """
    key_col: str
    keys_sorted: ArrayLike    # distinct values, ascending (len U)
    group_offsets: ArrayLike  # CSR offsets into row_positions (len U+1)
    row_positions: ArrayLike  # node row indices grouped by value (len N)
    backend: IndexBackend
    engine: Engine
    fingerprint: FrameFingerprint = field(compare=False, default=(-1, (), ""))
    source_ref: Optional[DataFrameT] = field(compare=False, default=None)
    n_nodes: int = 0
    n_keys: int = 0
    name: Optional[str] = None


ColStatsRole = Literal["nodes", "edges"]

#: The value side of a type partition: the groupby key that the single scalar
#: equality of a typed pattern names -- a relationship type or label name
#: (``str``), a numeric type code (``int``), or a ``label__X`` flag (``bool``).
#: ``bool`` needs no separate member: Python types it as a subtype of ``int``.
#: It is admitted DELIBERATELY, since ``(a:Person)`` lowers to
#: ``{"label__Person": True}``. One consequence is load-bearing: ``True == 1``
#: and they hash alike, so a bool-keyed and an int-keyed partition of the SAME
#: column are the same registry key. That is unreachable in practice (a column
#: is bool-dtyped or int-dtyped, not both) and harmless where it is reachable
#: (a query asking ``flag == 1`` of a bool column does select the True rows).
PartitionValue = Union[str, int]


@dataclass(frozen=True)
class ColStatsFact:
    """VERIFIED per-column facts over the exact bound frame (min/max/null count,
    integer-dtype flag). Same identity+fingerprint validity contract as the
    indexes; consumers must use facts CONSERVATIVELY: a fact can prove a
    property of any row subset that upper-bounds it (subset bounds lie within
    full-frame bounds; zero nulls on the frame means zero nulls on any subset),
    and an insufficient fact means fall back to the scan -- never decline."""
    role: ColStatsRole
    column: str
    min_val: Optional[Union[int, float]]
    max_val: Optional[Union[int, float]]
    null_count: int
    is_integer: bool
    engine: Engine
    n_unique: Optional[int] = None  # computed for the nodes role only (interval proofs)
    # Per-type partition facts: (type_column, type_value) restricts the fact to the
    # rows where type_column == type_value; None/None = whole frame. A partition
    # fact upper-bounds any FURTHER-filtered subset of that partition, same
    # conservative direction as whole-frame facts.
    type_column: Optional[str] = None
    type_value: Optional[PartitionValue] = None
    fingerprint: FrameFingerprint = field(compare=False, default=(-1, (), ""))
    source_ref: Optional[DataFrameT] = field(compare=False, default=None)


@dataclass(frozen=True)
class DegreeFact:
    """Precomputed in/out degree over the exact bound edge frame, optionally
    restricted to one relationship type.

    Why this exists: the two-hop count kernel spends its time in an O(E)
    ``bincount`` + gather over every edge. With degrees precomputed the same
    answer is ``dot(indeg, outdeg)`` -- O(N). Measured at board scale (2.4M edges,
    107k nodes) that is 6.76ms of query work versus 0.046ms.

    STALENESS IS A WRONG ANSWER HERE, which is new. A stale min/max fact costs a
    scan; a stale DEGREE fact returns a confidently incorrect count. So the
    identity+fingerprint contract is not an optimization guard on this type -- it
    is the correctness guard, and ``get_degree_valid`` refuses on any mismatch.

    ``indeg``/``outdeg`` are indexed by node id MINUS ``lo``, so the arrays cover
    the dense interval [lo, hi] the kernel already proves; ids outside it cannot
    be represented, which is why a fact is only built for a dense domain.
    """
    src_col: str
    dst_col: str
    indeg: ArrayLike
    outdeg: ArrayLike
    lo: int
    hi: int
    backend: IndexBackend
    engine: Engine
    # Trail-illegal (r, r) pairs the degree product would count (#1905); None = unknown,
    # which the two-hop kernel treats as a decline rather than a zero correction.
    self_loops: Optional[int] = None
    type_column: Optional[str] = None
    type_value: Optional[PartitionValue] = None
    fingerprint: FrameFingerprint = field(compare=False, default=(-1, (), ""))
    source_ref: Optional[DataFrameT] = field(compare=False, default=None)


@dataclass(frozen=True)
class GfqlIndexRegistry:
    """Immutable kind -> index map. ``with_index`` / ``without`` return copies."""
    indexes: Dict[IndexKind, Union[AdjacencyIndex, NodeIdIndex]] = field(default_factory=dict)
    # Property indexes are keyed by COLUMN, not kind: a graph may carry several.
    node_props: Dict[str, NodePropIndex] = field(default_factory=dict)
    # Column-stat facts keyed by (role, column, type_column, type_value); the
    # whole-frame fact uses (role, column, None, None). See ColStatsFact.
    col_stats: Dict[Tuple[str, str, Optional[str], Optional[PartitionValue]], ColStatsFact] = field(default_factory=dict)
    # Degree facts keyed by (src, dst, type_column, type_value); see DegreeFact.
    degrees: Dict[Tuple[str, str, Optional[str], Optional[PartitionValue]], DegreeFact] = field(default_factory=dict)

    def with_index(self, kind: IndexKind, index: Union[AdjacencyIndex, NodeIdIndex]) -> "GfqlIndexRegistry":
        new = dict(self.indexes)
        new[kind] = index
        return replace(self, indexes=new)

    def with_node_prop(self, column: str, index: "NodePropIndex") -> "GfqlIndexRegistry":
        props = dict(self.node_props)
        props[column] = index
        return replace(self, node_props=props)

    def with_degrees(self, fact: DegreeFact) -> "GfqlIndexRegistry":
        d = dict(self.degrees)
        d[(fact.src_col, fact.dst_col, fact.type_column, fact.type_value)] = fact
        return replace(self, degrees=d)

    def get_degree_valid(
        self, src_col: str, dst_col: str, df: Optional[DataFrameT], engine: Engine,
        type_column: Optional[str] = None, type_value: Optional[PartitionValue] = None,
    ) -> Optional["DegreeFact"]:
        """The degree fact, only while it still matches the live frame + engine.

        Unlike the col-stat facts, a miss here is not merely a lost optimization
        and a stale hit is not merely slow -- it is a wrong count. Every guard is
        therefore refusal, never best-effort."""
        fact = self.degrees.get((src_col, dst_col, type_column, type_value))
        if fact is None or df is None or fact.engine != engine:
            return None
        if fact.source_ref is not None and fact.source_ref is not df:
            return None
        cols = tuple(sorted({src_col, dst_col} | ({type_column} if type_column else set())))
        if fact.fingerprint != frame_fingerprint(df, cols, engine):
            return None
        return fact

    def without_degrees(self) -> "GfqlIndexRegistry":
        return replace(self, degrees={})

    def with_col_stats(self, fact: ColStatsFact) -> "GfqlIndexRegistry":
        stats = dict(self.col_stats)
        stats[(fact.role, fact.column, fact.type_column, fact.type_value)] = fact
        return replace(self, col_stats=stats)

    def get_col_stats_valid(
        self, role: ColStatsRole, column: str, df: Optional[DataFrameT], engine: Engine,
        type_column: Optional[str] = None, type_value: Optional[PartitionValue] = None,
    ) -> Optional[ColStatsFact]:
        """The fact for (role, column[, type partition]), only while it still matches
        the live frame + engine (same identity/fingerprint contract as ``get_valid``).

        A partition fact's validity depends on the type column too -- editing it
        re-partitions the frame -- so its fingerprint spans both columns."""
        fact = self.col_stats.get((role, column, type_column, type_value))
        if fact is None or df is None or fact.engine != engine:
            return None
        if fact.source_ref is not None and fact.source_ref is not df:
            return None
        cols = (column,) if type_column is None else tuple(sorted({column, type_column}))
        if fact.fingerprint != frame_fingerprint(df, cols, engine):
            return None
        return fact

    def without_col_stats(self) -> "GfqlIndexRegistry":
        return replace(self, col_stats={})

    def node_prop_cols(self) -> Tuple[str, ...]:
        return tuple(sorted(self.node_props.keys()))

    def get_node_prop_valid(
        self, column: str, df: Optional[DataFrameT], engine: Engine
    ) -> Optional["NodePropIndex"]:
        """The property index for ``column``, only while it still matches the live
        frame + engine (same identity/fingerprint contract as ``get_valid``)."""
        idx = self.node_props.get(column)
        if idx is None or df is None or idx.engine != engine:
            return None
        if idx.source_ref is not None and idx.source_ref is not df:
            return None
        if idx.fingerprint != frame_fingerprint(df, (column,), engine):
            return None
        return idx

    def without(self, kind: IndexKind) -> "GfqlIndexRegistry":
        if kind == NODE_PROP:
            return replace(self, node_props={})
        new = dict(self.indexes)
        new.pop(kind, None)
        return replace(self, indexes=new)

    def without_node_prop(self, column: str) -> "GfqlIndexRegistry":
        props = dict(self.node_props)
        props.pop(column, None)
        return replace(self, node_props=props)

    def rebind_edges(self, new_edges: DataFrameT, old_edges: DataFrameT) -> "GfqlIndexRegistry":
        """Migrate the EDGE adjacency indexes' identity guard from ``old_edges`` to
        ``new_edges``.

        Caller contract: ``new_edges`` was derived FROM ``old_edges`` by a transform
        that preserves the indexed src/dst columns by value (same rows, same order) —
        e.g. a shallow copy that merely ADDS an unrelated column. The chain executor
        does exactly this when it attaches its synthetic per-edge id (chain.py), which
        otherwise breaks the ``source_ref is df`` identity guard and forces a full
        scan. The CSR arrays stay valid (row positions unchanged); we only swap the
        strong-ref so ``get_valid`` recognizes the live frame. NODE_ID is left
        untouched (node materialization may legitimately change node rows).

        ENFORCED (O(1), engine-portable), and BOTH halves are required:

        1. LINEAGE (#1913) — the index must still be VALID for ``old_edges``, the
           exact ``get_valid`` contract (identity + fingerprint + engine). Without
           this a caller launders a stale index onto a frame the index was never
           built over: after a user's ordinary ``g.edges(other_frame)`` the identity
           guard has already missed, yet a same-row-count frame re-passes the
           structural check below — silent wrong answers on both engines, including
           a plain ``df.sort_values`` permutation of the SAME edge set. An index that
           fails this is NOT ours to discard (it still describes its own frame — e.g.
           a foreign-engine index, or the user's pre-rebind frame): it is left in
           place, un-migrated, so ``get_valid`` misses it on the new frame (safe scan)
           while ``show_indexes``/``gfql_explain`` can still report it as resident and
           stale rather than absent.
        2. STRUCTURE — ``new_edges`` must still match the index's fingerprint (row
           count + bound cols + engine) and actually carry the indexed columns. Here
           the caller DID own the index and broke its own derivation contract, so the
           entry is DROPPED (safe miss -> scan) rather than re-pointed.

        Value-level preservation across the derivation remains the caller's promise —
        checking it would be the O(E) scan this path exists to avoid — but the caller
        can now only make that promise about a frame the index was demonstrably live
        for."""
        new = dict(self.indexes)
        for kind in (EDGE_OUT_ADJ, EDGE_IN_ADJ):
            idx = new.get(kind)
            if idx is None:
                continue
            if not isinstance(idx, AdjacencyIndex):  # defensive; also narrows for mypy
                new.pop(kind, None)
                continue
            cols = tuple(idx.fingerprint[1])
            # #1913: only an index that is LIVE for the frame being augmented may migrate.
            if not (
                idx.source_ref is not None
                and idx.source_ref is old_edges
                and self.get_valid(kind, old_edges, cols, idx.engine) is idx
            ):
                continue  # someone else's lineage -> leave untouched (get_valid will miss)
            ok = idx.fingerprint == frame_fingerprint(new_edges, cols, idx.engine)
            if ok:
                try:
                    colnames = set(new_edges.columns)
                    ok = idx.key_col in colnames and idx.other_col in colnames
                except Exception:
                    ok = False
            if ok:
                new[kind] = replace(idx, source_ref=new_edges)
            else:
                new.pop(kind, None)
        return replace(self, indexes=new)

    def get(self, kind: IndexKind) -> Optional[Union[AdjacencyIndex, NodeIdIndex]]:
        return self.indexes.get(kind)

    def has(self, kind: IndexKind) -> bool:
        return kind in self.indexes

    def kinds(self) -> Tuple[IndexKind, ...]:
        return cast(Tuple[IndexKind, ...], tuple(sorted(self.indexes.keys())))

    def is_empty(self) -> bool:
        return not self.indexes and not self.node_props and not self.col_stats

    def get_valid(self, kind: IndexKind, df: DataFrameT, cols: Tuple[str, ...], engine: Engine) -> Optional[Union[AdjacencyIndex, NodeIdIndex]]:
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


def index_nbytes(idx: Union[AdjacencyIndex, NodeIdIndex, "NodePropIndex"]) -> int:
    """Approximate resident memory of an index's sidecar arrays (bytes)."""
    total = 0
    for attr in ("keys_sorted", "group_offsets", "row_positions", "other_values"):
        arr = getattr(idx, attr, None)
        if arr is not None:
            total += int(getattr(arr, "nbytes", 0))
    return total


EMPTY_REGISTRY = GfqlIndexRegistry()
