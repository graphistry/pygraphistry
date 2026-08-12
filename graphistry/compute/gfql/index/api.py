"""Public-facing index lifecycle + planner entry, operating on a Plottable.

The registry rides on the Plottable as a private attribute (``_gfql_index_registry``)
and propagates through ``copy.copy``-based functional chaining. It is fingerprint-
validated at use time, so a rebind of ``.edges()``/``.nodes()`` safely invalidates
stale indexes (treated as absent, never a wrong answer).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple, cast

import pandas as pd

from graphistry.Engine import EngineAbstract, Engine, EngineAbstractType, POLARS_ENGINES, resolve_engine
from graphistry.compute.typing import DataFrameT
from graphistry.Plottable import Plottable
from .registry import (
    AdjacencyIndex, ColStatsFact, ColStatsRole, GfqlIndexRegistry, EMPTY_REGISTRY, NodeIdIndex,
    EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID, NODE_PROP, ADJ_KINDS, ALL_KINDS,
)
from .build import build_adjacency_index, build_node_id_index, build_node_prop_index
from .traverse import index_seeded_hop
from .cost import cost_gate_frac, seed_deg_sum, seed_id_array
from .policy import IndexPolicy, validate_index_policy
from .types import (
    AdjacencyIndexKind, EdgeIndexDirection, HopDirection, IndexKind,
    ColStatsOutcomeName, FastPathName, IndexDecisionCode, IndexTrace, IndexTraceStep,
    TraceEngine,
)

# Private Plottable attachment keys. Keep access behind helpers.
POLICY_ATTR = "_gfql_index_policy"
REGISTRY_ATTR = "_gfql_index_registry"


class GfqlIndexUnsupportedError(ValueError):
    """The DATA cannot support this index (duplicate node ids, an unindexable
    property dtype). Distinct from a caller mistake — a missing column, an unknown
    kind, unbound edges — which stays a plain ``ValueError`` and must propagate.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers keep
    working; the convenience builders catch only THIS type, so a real failure is
    never silently skipped.
    """

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


def _engine_mismatch_text(kinds: str, index_engines: str, engine: Engine) -> str:
    """Single wording for an engine-mismatched index decline, shared by the
    ``gfql_explain`` trace diagnostic and the ``show_indexes`` ``reason`` column."""
    return (
        f"resident {kinds} index engine={index_engines}, "
        f"requested engine={engine.value} -> scan"
    )


def _engine_mismatch_reason(
    registry: GfqlIndexRegistry, direction: HopDirection, engine: Engine
) -> Optional[str]:
    """Describe a trace-only adjacency-index engine mismatch, if present.

    Indexes intentionally remain engine-specific: falling back to the scan is the
    correct execution behavior. This helper makes that otherwise silent decline
    visible to ``gfql_explain`` without changing planner policy.
    """
    needed: Sequence[AdjacencyIndexKind]
    if direction == "forward":
        needed = (EDGE_OUT_ADJ,)
    elif direction == "reverse":
        needed = (EDGE_IN_ADJ,)
    else:
        needed = (EDGE_OUT_ADJ, EDGE_IN_ADJ)
    mismatches = [
        (kind, idx)
        for kind in needed
        for idx in (registry.get(kind),)
        if isinstance(idx, AdjacencyIndex) and idx.engine != engine
    ]
    if not mismatches:
        return None
    kinds = ", ".join(kind for kind, _ in mismatches)
    engines = ", ".join(sorted({idx.engine.value for _, idx in mismatches}))
    return _engine_mismatch_text(kinds, engines, engine)


def _index_usability(
    kind: IndexKind, index_engine: Engine, valid: bool, query_engine: Engine
) -> Tuple[bool, Optional[str]]:
    """Usability of ONE resident index under the resolved query engine.

    ``valid`` (fingerprint freshness) is necessary but not sufficient: indexes are
    engine-specific, so a fresh index built for another engine still declines to a
    scan at query time (#1767 disposition). Returns ``(usable, reason)`` where
    ``reason`` uses the same wording as the ``gfql_explain`` decline diagnostic.
    """
    reasons: List[str] = []
    if index_engine != query_engine:
        reasons.append(_engine_mismatch_text(kind, index_engine.value, query_engine))
    if not valid:
        reasons.append("stale fingerprint (frames rebound since build) -> rebuild")
    return (not reasons), ("; ".join(reasons) or None)


def _record_indexed_traversal(
    *,
    seam: str,
    engine: Engine,
    served: bool,
    reason: str,
    hop_count: int,
    public_seed_scan: bool,
    hop_details: Optional[List[Dict[str, object]]] = None,
) -> None:
    """Record one backward-compatible indexed traversal decision when tracing."""
    if not _trace_active():
        return
    path = "index" if served else "scan"
    _record(cast(IndexTraceStep, {
        "op": "indexed_traversal",
        "operation": "indexed_traversal",
        "seam": seam,
        "engine": engine.value if isinstance(engine, (Engine, EngineAbstract)) else str(engine),
        "served": served,
        "reason": reason,
        "hops": hop_count,
        "hop_count": hop_count,
        "public_seed_scan": public_seed_scan,
        "hop_details": [] if hop_details is None else hop_details,
        "path": path,
        "decision_reason": reason,
        "decision_code": "index_selected" if served else "index_path_unavailable",
    }))


ColStatsOutcome = ColStatsOutcomeName  # single definition, in types.py

# Explicit mapping instead of an f-string plus a cast: mypy then checks that every
# outcome has a real decision code, so adding one to either side without the other
# is a type error rather than a string that silently never matches.
_COL_STATS_CODE: Dict[ColStatsOutcome, IndexDecisionCode] = {
    "served": "col_stats_served",
    "absent": "col_stats_absent",
    "stale": "col_stats_stale",
    "insufficient": "col_stats_insufficient",
}


def record_fast_path_decision(
    *, path: FastPathName, served: bool, reason: str, engine: TraceEngine
) -> None:
    """Record whether a named fast path SERVED or declined, for ``gfql_explain``.

    Fast paths are contracted "same answer, faster": every one falls back, so a
    DEAD one is invisible -- the query is still correct and every value test still
    passes. Measured ratio in test_lowering.py when this was added: 665 value
    assertions vs 42 engagement ones. This is the surface that makes engagement
    assertable against a public API instead of by monkeypatching private callees,
    which fails open when another module imported the name directly.

    Free outside ``index_trace()`` / ``gfql_explain`` -- same ``_trace_active()``
    gate the adjacency and col-stats decisions use.
    """
    if not _trace_active():
        return
    step: IndexTraceStep = {
        "op": "fast_path",
        "operation": "fast_path",
        "seam": path,
        # Engagement is per-engine: a path can serve on one and decline on another.
        "engine": engine.value if isinstance(engine, (Engine, EngineAbstract)) else str(engine),
        "served": served,
        "reason": reason,
        "path": "index" if served else "scan",
        "decision_reason": reason,
        "decision_code": "index_selected" if served else "index_path_unavailable",
    }
    _record(step)


def record_col_stats_decision(
    *,
    role: str,
    column: str,
    type_column: Optional[str],
    type_value: Optional[object],
    outcome: "ColStatsOutcome",
    reason: str,
) -> None:
    """Record ONE column-stat fact consult, for ``gfql_explain``.

    Facts are a pure accelerator: a miss falls back to the scan and the answer is
    unchanged. That makes a dead fact INVISIBLE -- you pay the build and get
    nothing, with every value test still green. This is the surface that makes it
    visible, and it distinguishes the cases that need different fixes:

    - ``absent``       no fact for this key (never built, or built for another column)
    - ``stale``        a fact exists but the frame was rebound since it was built
    - ``insufficient`` the fact is live but cannot prove what the plan needs
    - ``served``       the fact was used and a scan was skipped

    Costs nothing outside ``index_trace()`` / ``gfql_explain`` -- the guard is the
    same ``_trace_active()`` gate the adjacency decisions use.
    """
    if not _trace_active():
        return
    step: IndexTraceStep = {
        "op": "col_stats",
        "operation": "col_stats",
        "role": role,
        "column": column,
        "type_column": type_column,
        "type_value": type_value,
        "served": outcome == "served",
        "reason": reason,
        "path": "facts" if outcome == "served" else "scan",
        "decision_reason": reason,
        "decision_code": _COL_STATS_CODE[outcome],
    }
    _record(step)


# Back-compat for existing private tests while helpers live in cost.py.
_seed_id_array = seed_id_array
_seed_deg_sum = seed_deg_sum

def get_registry(g: Plottable) -> GfqlIndexRegistry:
    registry = g._gfql_index_registry
    return registry if registry is not None else EMPTY_REGISTRY


def get_index_policy(g: Plottable) -> IndexPolicy:
    return g._gfql_index_policy


def with_index_policy(g: Plottable, policy: IndexPolicy) -> Plottable:
    """A copy of ``g`` carrying ``policy`` (never mutates ``g``)."""
    out = g.bind()
    out._gfql_index_policy = policy
    return out


def _attach(g: Plottable, registry: GfqlIndexRegistry) -> Plottable:
    res = copy.copy(g)
    res._gfql_index_registry = registry
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


def _is_resident_index_valid(
    g: Plottable,
    kind: IndexKind,
    engine: EngineAbstractType = EngineAbstract.AUTO,
) -> bool:
    """True when a resident index still matches the current graph frames."""
    eng = resolve_engine(engine, g)
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
    eng = resolve_engine(engine, g)
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
            raise GfqlIndexUnsupportedError(
                f"Cannot build a {NODE_ID!r} index: node id column {node_col!r} has "
                f"duplicate values (a node-id index requires unique ids). Seeded "
                f"traversal still works via the un-indexed node materialization path."
            )
        node_idx = replace(node_idx, name=name or index_name(kind, node_col))
        registry = registry.with_index(NODE_ID, node_idx)
        return _attach(g2, registry)

    if kind == NODE_PROP:
        if not column:
            raise ValueError(
                f"A {NODE_PROP!r} index indexes one node PROPERTY column; pass "
                f"column='<name>'."
            )
        g2 = g.materialize_nodes() if g._nodes is None else g
        assert g2._nodes is not None
        if column not in g2._nodes.columns:
            raise ValueError(
                f"Cannot build a {NODE_PROP!r} index: node column {column!r} not found."
            )
        prop_idx = build_node_prop_index(g2._nodes, column, eng)
        if prop_idx is None:
            raise GfqlIndexUnsupportedError(
                f"Cannot build a {NODE_PROP!r} index on {column!r}: only integer "
                f"columns without nulls are indexable today. Seeded queries still "
                f"work via the un-indexed scan path."
            )
        prop_idx = replace(prop_idx, name=name or index_name(kind, column))
        registry = registry.with_node_prop(column, prop_idx)
        return _attach(g2, registry)

    raise ValueError(f"Unknown GFQL index kind: {kind!r}. Expected one of {ALL_KINDS}.")


def drop_index(
    g: Plottable, kind: Optional[IndexKind] = None, *, column: Optional[str] = None
) -> Plottable:
    """Drop one index (by kind, or one property index by column) or all (kind=None).

    Idempotent."""
    registry = get_registry(g)
    if kind is None:
        return _attach(g, EMPTY_REGISTRY)
    if kind == NODE_PROP and column is not None:
        return _attach(g, registry.without_node_prop(column))
    return _attach(g, registry.without(kind))


def show_indexes(
    g: Plottable, engine: EngineAbstractType = EngineAbstract.AUTO
) -> pd.DataFrame:
    """Return a pandas DataFrame describing resident indexes (empty if none).

    ``valid`` reflects live fingerprint validity against the current frames — a
    stale index (after a ``.edges()``/``.nodes()`` rebind) shows ``valid=False`` and
    is auto-skipped (scan fallback) until rebuilt. ``nbytes`` is the resident
    sidecar-array footprint (the pay-as-you-go memory signal).

    ``valid`` alone is NOT "this index will serve your query": indexes are
    engine-specific, so a fresh index built for another engine silently declines to
    a scan at query time (#1767). ``query_engine`` is what ``engine`` resolves to
    for THIS graph (the same resolution a query makes — pass ``engine=`` to preview
    an explicit choice), ``usable`` is True only when the index is fresh AND
    engine-matched, and ``reason`` says why not, with the same wording as the
    ``gfql_explain`` decline diagnostic.
    """
    from .registry import index_nbytes

    query_engine = resolve_engine(engine, g)
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
        usable, reason = _index_usability(kind, idx.engine, valid, query_engine)
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
            "query_engine": query_engine.value,
            "usable": usable,
            "reason": reason,
        })
    for column in registry.node_prop_cols():
        prop = registry.node_props[column]
        prop_valid = registry.get_node_prop_valid(column, g._nodes, prop.engine) is not None
        usable, reason = _index_usability(NODE_PROP, prop.engine, prop_valid, query_engine)
        rows.append({
            "name": prop.name or index_name(NODE_PROP, column),
            "kind": NODE_PROP,
            "key_col": column,
            "engine": prop.engine.value,
            "backend": prop.backend,
            "n_keys": prop.n_keys,
            "n_rows": prop.n_nodes,
            "nbytes": index_nbytes(prop),
            "valid": prop_valid,
            "query_engine": query_engine.value,
            "usable": usable,
            "reason": reason,
        })
    cols = [
        "name", "kind", "key_col", "engine", "backend", "n_keys", "n_rows", "nbytes",
        "valid", "query_engine", "usable", "reason",
    ]
    return pd.DataFrame(rows, columns=cols)


def gfql_index_edges(g: Plottable, direction: EdgeIndexDirection = "both",
                     engine: EngineAbstractType = EngineAbstract.AUTO) -> Plottable:
    """Convenience: build edge adjacency index(es). direction: 'forward'|'reverse'|'both'."""
    if direction in ("forward", "both"):
        g = create_index(g, EDGE_OUT_ADJ, engine=engine)
    if direction in ("reverse", "both"):
        g = create_index(g, EDGE_IN_ADJ, engine=engine)
    return g


def gfql_index_node_props(g: Plottable, columns: Sequence[str],
                          engine: EngineAbstractType = EngineAbstract.AUTO) -> Plottable:
    """Convenience: build node property indexes for ``columns`` (skips unindexable).

    Skipping mirrors ``gfql_index_all``'s node_id behaviour — a column whose dtype
    this index cannot serve keeps the correct scan path. ONLY that case is skipped:
    a missing column, an unknown kind, or any unexpected failure propagates.
    ``create_index(NODE_PROP, column=...)`` still raises for everything, since the
    caller asked for that column specifically."""
    for column in columns:
        try:
            g = create_index(g, NODE_PROP, column=column, engine=engine)
        except GfqlIndexUnsupportedError:
            continue  # dtype this index cannot serve -> keep the correct scan path
    return g


def _schema_node_type_columns(g: Plottable) -> Tuple[str, ...]:
    """Node type columns a DECLARED schema names, restricted to ones the frame has.

    ``NodeType.labels`` maps to GFQL's ``label__<Label>`` convention (schema.py),
    which is the same column the Cypher lowering emits for ``(a:Label)``. Absent
    candidates are skipped, not raised: the schema declares a contract for the
    whole graph, and a frame legitimately carries only some of it.
    """
    schema = getattr(g, "_gfql_schema", None)
    if schema is None or g._nodes is None:
        return ()
    present = set(g._nodes.columns)
    out = [f"label__{label}"
           for node_type in getattr(schema, "node_types", ())
           for label in getattr(node_type, "labels", ())]
    return tuple(sorted({c for c in out if c in present}))


def _schema_edge_type_columns(g: Plottable) -> Tuple[str, ...]:
    """Edge type columns a DECLARED schema names, restricted to ones the frame has.

    The two conventions currently DISAGREE for edges: ``schema.py`` declares
    ``label__<Name>`` booleans while the Cypher lowering emits ``type ==
    '<Name>'``. Until that is reconciled, offer BOTH candidates and keep whichever
    the frame actually carries -- a fact on a column no query names is wasted
    build time, never a wrong answer, so covering both is the safe direction.
    """
    schema = getattr(g, "_gfql_schema", None)
    if schema is None or g._edges is None:
        return ()
    present = set(g._edges.columns)
    candidates = {f"label__{getattr(et, 'name', '')}" for et in getattr(schema, "edge_types", ())}
    if getattr(schema, "edge_types", ()):
        candidates.add("type")
    return tuple(sorted({c for c in candidates if c in present}))


def _add_degree_facts(
    registry: GfqlIndexRegistry,
    edges: DataFrameT,
    g: Plottable,
    type_column: str,
    partition_facts: Sequence[ColStatsFact],
    eng: Engine,
) -> GfqlIndexRegistry:
    """Degree facts for each type partition whose node ids form a DENSE interval.

    Only where that interval is provable: the arrays are indexed by ``id - lo``, so
    a gapped or unbounded domain has no valid indexing and we build nothing rather
    than build something the kernel could misread.
    """
    from .build import build_degree_fact
    if g._nodes is None or g._node is None or not g._source or not g._destination:
        return registry
    # Degrees are built over each edge partition's OWN endpoint span, not the node
    # space. Density is NOT required for the arrays -- ids absent from the span
    # contribute zero to the dot -- so gapped or interleaved node ids are fine;
    # only the span is bounded (memory). The kernel separately proves its DOMAIN
    # dense before consulting; demanding density here was strictly more
    # restrictive than the kernel it serves, and built nothing on real data.
    by_tv: Dict[object, Dict[str, ColStatsFact]] = {}
    for pf in partition_facts:
        by_tv.setdefault(pf.type_value, {})[pf.column] = pf
    for tv, cols in by_tv.items():
        sf, df_ = cols.get(g._source), cols.get(g._destination)
        if (sf is None or df_ is None or sf.min_val is None or sf.max_val is None
                or df_.min_val is None or df_.max_val is None):
            continue
        lo = int(min(sf.min_val, df_.min_val))
        hi = int(max(sf.max_val, df_.max_val))
        if eng in POLARS_ENGINES:
            import polars as pl
            sub = edges.filter(pl.col(type_column) == tv)  # type: ignore[union-attr]  # engine seam
        else:
            sub = edges[edges[type_column] == tv]
        d = build_degree_fact(sub, g._source, g._destination, lo, hi, eng,
                              type_column=type_column, type_value=tv)
        if d is not None:
            from dataclasses import replace as _replace
            from .registry import frame_fingerprint as _fp
            cols_fp = tuple(sorted({g._source, g._destination, type_column}))
            registry = registry.with_degrees(_replace(
                d, source_ref=edges, fingerprint=_fp(edges, cols_fp, eng)))
    return registry


def gfql_index_col_stats(g: Plottable,
                         node_columns: Optional[Sequence[str]] = None,
                         edge_columns: Optional[Sequence[str]] = None,
                         node_type_column: Optional[str] = None,
                         edge_type_column: Optional[str] = None,
                         col_stats_by_type: bool = False,
                         engine: EngineAbstractType = EngineAbstract.AUTO) -> Plottable:
    """Verified column-stat facts (min/max/null count) -- EAGER and TARGETED.

    Default target is the plan-relevant minimum: the node id binding and the edge
    src/dst bindings (what the count fast paths consult). Pass ``node_columns`` /
    ``edge_columns`` to fact additional columns: an EXPLICITLY requested column
    that is absent or unfactable (non-integer in v1) raises -- you asked for it
    by name -- while the binding defaults skip silently (convenience, consumers
    scan as before). Facts ride the same identity+fingerprint validity contract
    as the physical indexes, and consumers use them as UNDER-approximations of
    provability (see ColStatsFact): a missing/insufficient fact costs a scan,
    never an answer. Laziness (plan-driven fact building at query time) is
    deliberately out of scope -- that is the typed-ontology re-verification
    policy question; eager build here keeps fact cost a declared setup step,
    matching how the benchmark harness discloses index builds.

    ``col_stats_by_type`` additionally builds per-type facts for the types a BOUND
    SCHEMA declares. It defaults False because those facts cost a grouped pass per
    type column at build time -- and one pass PER LABEL under the ``label__X``
    convention -- while only typed count shapes can spend them. Turn it on for a
    long-lived resident graph serving typed queries, where the build amortizes;
    leave it off for one-shot work. Explicit ``*_type_column`` requests are
    unaffected by this flag. (Build/query costs are receipted in pyg-bench.)

    ``node_type_column`` / ``edge_type_column`` additionally build PER-TYPE facts
    over the bindings, one grouped pass each. Whole-frame facts prove nothing on a
    typed graph -- the id interval spans every label -- so these are what let a
    typed pattern reach the dense kernel. Like ``*_columns`` they were asked for
    by name, so an unusable request raises rather than skipping.
    """
    from .build import _MAX_COL_STATS_PARTITIONS, build_col_stats_fact, build_col_stats_facts_by_type
    eng = resolve_engine(engine, g)
    from graphistry.compute.ComputeMixin import _coerce_input_formats
    g = _coerce_input_formats(g, eng)
    registry = get_registry(g)
    if g._nodes is not None and g._node is not None:
        fact = build_col_stats_fact(g._nodes, g._node, "nodes", eng)
        if fact is not None:
            registry = registry.with_col_stats(fact)
    if g._edges is not None:
        for col in (g._source, g._destination):
            if col is None:
                continue
            fact = build_col_stats_fact(g._edges, col, "edges", eng)
            if fact is not None:
                registry = registry.with_col_stats(fact)
    targets: List[Tuple[Optional[Sequence[str]], Optional[DataFrameT], ColStatsRole]] = [
        (node_columns, g._nodes, "nodes"),
        (edge_columns, g._edges, "edges"),
    ]
    for requested, frame, role in targets:
        for col in (requested or ()):
            if frame is None:
                raise ValueError(f"col_stats requested for {role} column {col!r} but no {role} frame is bound")
            fact = build_col_stats_fact(frame, col, role, eng)
            if fact is None:
                raise ValueError(
                    f"Cannot build a col_stats fact on {role} column {col!r}: absent, "
                    f"empty, or non-integer (v1 facts integer columns only)")
            registry = registry.with_col_stats(fact)
    _requested_partitions: Set[Tuple[str, str]] = set()
    part_targets: List[Tuple[Optional[str], Optional[DataFrameT], ColStatsRole, Tuple[Optional[str], ...]]] = [
        (node_type_column, g._nodes, "nodes", (g._node,)),
        (edge_type_column, g._edges, "edges", (g._source, g._destination)),
    ]
    for type_column, frame, role, binding_cols in part_targets:
        if type_column is None:
            continue
        if frame is not None:
            _requested_partitions.add((role, type_column))
        if frame is None:
            raise ValueError(
                f"col_stats requested per {role} type column {type_column!r} but no {role} frame is bound")
        present = [c for c in binding_cols if c is not None]
        partition_facts = build_col_stats_facts_by_type(frame, present, role, type_column, eng)
        if present and not partition_facts:
            raise ValueError(
                f"Cannot build per-type col_stats facts on {role} columns {present!r} by "
                f"{type_column!r}: a column is absent, the frame is empty, the values are "
                f"non-integer or null-bearing, the type keys are float/null/list-valued, or "
                f"there are more than {_MAX_COL_STATS_PARTITIONS} distinct types")
        for fact in partition_facts:
            registry = registry.with_col_stats(fact)
        if role == "edges" and g._source and g._destination:
            registry = _add_degree_facts(registry, frame, g, type_column, partition_facts, eng)

    # A DECLARED schema names its own type partitions, so using it is not a
    # guess -- unlike sniffing column names. It is still OPT-IN because per-type
    # facts are NOT free (a grouped pass per type column; see col_stats_by_type).
    # Derived candidates SKIP when
    # unusable (they were not asked for by name, unlike the params above), and
    # an explicit param for the same role wins.
    derived_targets: List[Tuple[ColStatsRole, Optional[DataFrameT], Tuple[Optional[str], ...], Tuple[str, ...]]] = [
        ("nodes", g._nodes, (g._node,), _schema_node_type_columns(g)),
        ("edges", g._edges, (g._source, g._destination), _schema_edge_type_columns(g)),
    ] if col_stats_by_type else []
    derived_facts = 0
    for role, frame, binding_cols, candidates in derived_targets:
        if frame is None:
            continue
        for type_column in candidates:
            if (role, type_column) in _requested_partitions:
                continue
            present = [c for c in binding_cols if c is not None]
            for fact in build_col_stats_facts_by_type(frame, present, role, type_column, eng):
                registry = registry.with_col_stats(fact)
                derived_facts += 1
    if col_stats_by_type and not derived_facts and not _requested_partitions:
        # col_stats_by_type is an EXPLICIT request, so satisfying none of it
        # raises rather than no-oping -- the same contract as naming a type
        # column by hand. Partial coverage (a declared label the frame does not
        # carry) still skips: a schema is a contract for the whole graph.
        raise ValueError(
            "col_stats_by_type=True but no per-type facts could be built: "
            + ("no schema is bound -- call bind(schema=GraphSchema(...)) or name the "
               "columns with node_type_column=/edge_type_column="
               if getattr(g, "_gfql_schema", None) is None else
               "the bound schema declares no node label or edge type column that the "
               "bound frames carry, or the id/endpoint columns are unfactable"))
    return _attach(g, registry)


def gfql_index_all(g: Plottable,
                   col_stats_by_type: bool = False,
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
    except GfqlIndexUnsupportedError:
        pass  # non-unique node ids -> skip the node_id accelerator (adjacency still built)
    return gfql_index_col_stats(g, col_stats_by_type=col_stats_by_type, engine=engine)


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
    effective_hops = max_hops if max_hops is not None else hops
    if not to_fixed_point:
        if not isinstance(effective_hops, int) or effective_hops < 1:
            return False
        if hops is None and min_hops not in (None, 1):
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
    "force" bypasses only the cost gate for a covered query. It still falls back to
    scan when the index cannot serve the query.
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

    def _bail(reason: str, decision_code: IndexDecisionCode) -> Optional[Plottable]:
        if trace:
            _record(cast(IndexTraceStep, {
                **diag, "path": "scan", "decision_reason": reason, "decision_code": decision_code,
            }))
        return None

    if resolved_policy == "off":
        return _bail("policy=off", "policy_off")
    registry = get_registry(g)
    if registry.is_empty() and resolved_policy not in ("auto", "force"):
        return _bail("no resident index (policy=use)", "no_resident_index")

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
        return _bail("query not index-coverable", "not_index_coverable")
    assert nodes is not None

    node_col = g._node
    src, dst = g._source, g._destination
    if node_col is None or src is None or dst is None or g._edges is None or g._nodes is None:
        return _bail("graph missing node/edge columns", "missing_graph_columns")

    if resolved_policy in ("auto", "force"):
        registry = _ensure_indexes(g, registry, direction, engine, resolved_policy, nodes, src, dst, node_col)
    if registry.is_empty():
        return _bail("no index available (build declined)", "index_build_declined")

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
                    f"({frac * idx0.n_keys:.0f}) -> scan cheaper", "scan_cost"
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
        engine_mismatch_reason = (
            _engine_mismatch_reason(registry, direction, engine) if result is None else None
        )
        _record(cast(IndexTraceStep, {
            **diag, "hops": eff_hops,
            "path": "index" if result is not None else "scan",
            "decision_reason": (
                "frontier below cost gate -> index" if result is not None
                else engine_mismatch_reason or "index path not applicable -> scan"
            ),
            "decision_code": (
                "index_selected" if result is not None
                else "engine_mismatch" if engine_mismatch_reason is not None
                else "index_path_unavailable"
            ),
        }))
    return result
