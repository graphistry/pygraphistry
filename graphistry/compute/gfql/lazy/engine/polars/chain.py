"""Native Polars chain() — Phase 1.

Reimplements the chain forward/backward/combine orchestration in polars, reusing the polars hop
for edge steps. Set ops are semi/anti joins, alias tags are join-based flag
columns. Parity-or-NIE contract: differential
parity vs the pandas chain gates correctness; unsupported shapes raise NotImplementedError
(no silent pandas fallback). Deferred: variable-length/multi-hop edge sub-cases, some
undirected multi-edge combos, node query=.
"""
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Sequence, Tuple, Type, cast

from typing_extensions import TypedDict

# Runtime import (not TYPE_CHECKING): AggSpec is a pure typing Union of builtins (engine-
# neutral wire type), and it keeps _GroupByParams introspectable (get_type_hints) at runtime.
from graphistry.compute.gfql.call.support import AggSpec
from graphistry.compute.endpoint_utils import drop_null_endpoint_edges

from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTObject, ASTNode, ASTEdge
from graphistry.compute.chain_specializations.hotpaths import _single_node_rows_via_index_or_filter
from .chain_specializations.admission import polars_plain_single_hop_admits
from .chain_specializations.hotpaths import _plain_seeded_index_hop_polars, _plain_single_hop_polars, _try_seeded_chain_polars

if TYPE_CHECKING:
    import polars as pl
    from graphistry.compute.gfql.index.bindings import IndexedBindingsState
    from .dtypes import PolarsFrame, PolarsT
from .hop_eager import ensure_nodes_polars
from .dtypes import is_lazy, colnames, endpoint_ids
from .degrees import get_degrees_polars, get_indegrees_polars, get_outdegrees_polars
from .predicates import filter_by_dict_polars
from .reserved_columns import CHAIN_NODE_HOP
from graphistry.compute.gfql.identifiers import shadow_restore_column


def _polars_error_types() -> Tuple[Type[BaseException], ...]:
    """The polars exception hierarchy root, as an ``except`` target.

    A tuple (not the class) so the empty tuple is available as the fail-closed answer when an
    older polars has no ``PolarsError`` base: ``except ()`` matches nothing, which degrades to
    today's behaviour rather than swallowing something unrelated.
    """
    import polars as pl
    base = getattr(pl.exceptions, "PolarsError", None)
    if isinstance(base, type) and issubclass(base, BaseException):
        return (base,)
    return ()


def _semi(df: "PolarsT", ids_df: "PolarsT", df_col: str, id_col: str) -> "PolarsT":
    """Rows of df whose df_col is present in ids_df[id_col] (semi-join).

    Both frames share the ``PolarsT`` TypeVar because polars joins do not mix eagerness:
    ``DataFrame.join`` takes a ``DataFrame`` and ``LazyFrame.join`` takes a ``LazyFrame``, and a
    mixed pair raises at runtime. Same variable in, same variable out — the semi-join preserves
    the left frame's eagerness, so an eager caller keeps its ``.height``/``.columns`` and a lazy
    caller keeps a plan.

    The key frame is deliberately NOT deduplicated; ``test_semi_join_key_frame.py``
    pins why that is safe.
    """
    return df.join(ids_df.select(id_col), left_on=df_col, right_on=id_col, how="semi")


def _align_seed_dtype(seed, node_col, ref_nodes):
    """Cast the seed's node-id column to the node table's id dtype. start_nodes can arrive with a
    divergent dtype (e.g. empty crossfilter selection defaults float64 vs int64 ids); polars won't
    auto-cast, so the combine semi-join raises SchemaError where pandas joins fine. No-op when the
    column is absent or already matches."""
    import polars as pl
    if seed is None or node_col not in seed.columns:
        return seed
    ndt = ref_nodes.schema[node_col]
    if seed.schema[node_col] == ndt:
        return seed
    return seed.with_columns(pl.col(node_col).cast(ndt))


def _align_edge_endpoints(g, node_col, src, dst):
    """Cast edge endpoint columns to the node-id dtype so join keys match.

    Casts endpoint columns to the node-id dtype so the endpoint<->node-id joins match. Returns
    ``(aligned_g, restore)``: restore = original (src_dtype, dst_dtype) to put back on the OUTPUT
    edges (matching pandas' dtype), or None when already matched (common case — no table copy)."""
    import polars as pl
    ndt = g._nodes.schema[node_col]
    sdt, ddt = g._edges.schema[src], g._edges.schema[dst]
    if sdt == ndt and ddt == ndt:
        return g, None
    aligned = g.edges(g._edges.with_columns([pl.col(src).cast(ndt), pl.col(dst).cast(ndt)]), src, dst)
    return aligned, (sdt, ddt)


def _restore_edge_dtypes(edges, src, dst, restore):
    """Restore output edge endpoint dtypes recorded by :func:`_align_edge_endpoints`."""
    if restore is None:
        return edges
    import polars as pl
    sdt, ddt = restore
    return edges.with_columns([pl.col(src).cast(sdt), pl.col(dst).cast(ddt)])


# Auto-injected hop-distance label (#1741). pandas' chain asks the hop for node hop labels
# whenever an edge op is variable-length or output-sliced (ast.py:621-625 needs_auto_labels) and
# then gates node ALIASES by that distance (chain.py:456-501). Without the labels the polars chain
# had no distance to gate on, so `MATCH (a)-[*1..2]-(b)` over-flagged a backtracked-to seed.
#
# The concrete column name is resolved once per chain against the user's node columns via
# generate_safe_column_name (see _chain_traversal_polars), so a user column literally named
# `__gfql_chain_node_hop__` can't be clobbered (would otherwise crash on the int/str compare in
# the gate). This base string (declared in reserved_columns.py, the per-engine symbol registry)
# is only the seed + the fallback for callers that don't resolve.
_AUTO_NODE_HOP: str = CHAIN_NODE_HOP


def _auto_node_hop_col(op: ASTObject, name: str = _AUTO_NODE_HOP) -> Optional[str]:
    """The auto-label column for an edge op, or None when labels are unnecessary/unsupported."""
    if not isinstance(op, ASTEdge):
        return None
    if op.min_hops is not None and op.min_hops > 1:
        # min_hops>1 labels come from the layered backward walk, not yet ported -> asking for
        # them would turn a currently-native chain into an NIE. Gap tracked on #1741/#1748.
        return None
    needs = (
        (op.min_hops is not None and op.min_hops > 0)
        or op.output_min_hops is not None
        or op.output_max_hops is not None
        or op.prune_to_endpoints
    )
    return name if needs else None


def _alias_hop_bounds(op: ASTEdge) -> Tuple[int, Optional[int]]:
    """pandas' (min_hop, max_hop) alias window for a node op preceded by edge `op` (chain.py:477-499)."""
    min_hop = (
        op.output_min_hops if op.output_min_hops is not None
        else (op.min_hops if op.min_hops is not None else (op.hops if op.hops is not None else 1))
    )
    max_hop = (
        op.output_max_hops if op.output_max_hops is not None
        else (op.max_hops if op.max_hops is not None else op.hops)
    )
    if op.to_fixed_point:
        max_hop = None
    return min_hop, max_hop


def _step_edges_with_source_columns(g: Plottable, g_step: Plottable, edge_id: str) -> "pl.DataFrame":
    """The step's edge rows with the graph's ORIGINAL columns: an alias marker that shares a
    name with a column the step filters on must not be re-filtered as that column."""
    import polars as pl
    edges = g._edges
    assert edges is not None and g_step._edges is not None
    return edges.join(g_step._edges.select(pl.col(edge_id)), on=edge_id, how="semi")


def _with_source_column_instead_of_marker(frame: "PolarsT", g: Plottable, node_col: str, column: str) -> "PolarsT":
    """``frame`` with ``column`` re-read from the graph's node table: an alias marker stamped by
    an earlier pass must not be what the step's own filter on that column sees."""
    import polars as pl
    return frame.drop(column).join(g._nodes.select(pl.col(node_col), pl.col(column)), on=node_col, how="left")


def _exec(op: ASTObject, g: Plottable, prev_wf: Optional[Any], target_wf: Optional[Any],
          intermediate_universe: Optional[Any] = None,
          auto_hop_col: str = _AUTO_NODE_HOP) -> Plottable:
    # prev_wf/target_wf/intermediate_universe are polars wavefront frames (DataFrame|LazyFrame)
    # or None; typed Optional[Any] to match this module's frame-annotation convention.
    import polars as pl

    node_col = g._node
    assert node_col is not None
    if isinstance(op, ASTNode):
        if op.query is not None:
            raise NotImplementedError("polars chain engine does not yet support node query=")
        base = prev_wf if prev_wf is not None else g._nodes
        if op._name is not None and op.filter_dict and op._name in op.filter_dict and op._name in base.columns:
            base = _with_source_column_instead_of_marker(base, g, node_col, op._name)
        nodes = filter_by_dict_polars(base, op.filter_dict)
        if target_wf is not None:
            nodes = _semi(nodes, target_wf, node_col, node_col)
        if op._name is not None:
            nodes = nodes.with_columns(pl.lit(True).alias(op._name))
        return g.nodes(nodes, node_col).edges(g._edges.clear(), g._source, g._destination)

    if isinstance(op, ASTEdge):
        from graphistry.compute.gfql.lazy.engine.polars.hop import hop_lazy_or_eager
        g_step = hop_lazy_or_eager(
            g,
            nodes=prev_wf,
            hops=op.hops,
            min_hops=op.min_hops,
            max_hops=op.max_hops,
            to_fixed_point=op.to_fixed_point,
            direction=op.direction,
            edge_match=op.edge_match,
            source_node_match=op.source_node_match,
            destination_node_match=op.destination_node_match,
            source_node_query=op.source_node_query,
            destination_node_query=op.destination_node_query,
            edge_query=op.edge_query,
            label_node_hops=op.label_node_hops or _auto_node_hop_col(op, auto_hop_col),
            label_edge_hops=op.label_edge_hops,
            label_seeds=op.label_seeds,
            output_min_hops=op.output_min_hops,
            output_max_hops=op.output_max_hops,
            include_zero_hop_seed=op.include_zero_hop_seed,
            return_as_wave_front=True,
            target_wave_front=target_wf,
            intermediate_universe=intermediate_universe,
            # Chain wants pandas' LABELED min_hops node policy (null attrs on source-side
            # endpoints; mirrors ASTEdge.execute's auto hop-labels). Direct base.hop(
            # engine='polars', min_hops=...) leaves this False -> full attrs (pandas parity).
            min_hops_label_policy=True,
        )
        if op._name is not None:
            g_step = g_step.edges(
                g_step._edges.with_columns(pl.lit(True).alias(op._name)),
                g._source, g._destination,
            )
        return g_step

    raise NotImplementedError(f"polars chain engine does not support op {type(op).__name__}")


def _is_native_multihop(op: ASTObject) -> bool:
    """Multi-hop shapes the native combine supports: hops=N / max_hops (fwd/rev/undirected),
    to_fixed_point (fwd/rev), min_hops>1 (fwd/rev, finite max), optionally with match/name.
    Deferred combos (undirected to_fixed_point / undirected min_hops>1, output slicing, hop
    labels, *_query, include_zero_hop_seed, prune_to_endpoints) return False so the guard
    NIEs rather than risk silent divergence — inline comments give per-case reasons."""
    if not isinstance(op, ASTEdge):
        return False
    if op.is_simple_single_hop():
        return False
    if op.direction not in ("forward", "reverse", "undirected"):
        return False
    # to_fixed_point IS native fwd/rev: recompute re-runs the forward tfp hop over the
    # backward-pruned subgraph (hop's fixed-point detection guarantees termination), same
    # path-aware combine as hops=N. decline (NIE): UNDIRECTED tfp needs pandas' connected-
    # components + 2-core seed retention (hop.py:817-887), not reproducible in the polars hop.
    if op.direction == "undirected" and op.to_fixed_point:
        return False
    # min_hops>1 IS native fwd/rev with finite max_hops (NON-anti-joined BFS + layered
    # backward-tree walk). decline (NIE): undirected min_hops>1 and min_hops+to_fixed_point.
    if op.min_hops is not None and op.min_hops > 1 and (op.direction == "undirected" or op.to_fixed_point):
        return False
    if op.output_min_hops is not None or op.output_max_hops is not None:
        return False
    if op.label_node_hops is not None or op.label_edge_hops is not None or op.label_seeds:
        return False
    if op.source_node_query is not None or op.destination_node_query is not None or op.edge_query is not None:
        return False
    if op.include_zero_hop_seed or op.prune_to_endpoints:
        return False
    return True


class _LazyShim:
    """Track B collect-once shim: carries _nodes/_edges as LazyFrames (+ col names) so the eager
    combine helpers run lazily over already-materialized hop frames without Plottable rebinds.

    ``edges_empty`` records whether the step's edge frame was empty while it was still eager
    (tri-state: True/False, or None when unknown). ``.lazy()`` throws that fact away — a
    LazyFrame has no height without collecting — so the combine's cardinality shortcuts go dead
    without it; this is the only place in the lazy combine where the count is still available."""
    __slots__ = ("_nodes", "_edges", "_node", "_source", "_destination", "_edge", "edges_empty")

    # Bare annotations only — a class-level VALUE would collide with __slots__ at class
    # creation. These make the slots statically typed rather than inferred from __init__.
    # LazyFrame, not the PolarsFrame union: every construction site (`step` and the Track-B
    # entry) calls `.lazy()` first, which is the whole point of the shim, so the union would
    # be both less true and unusable — `pl.concat`'s TypeVar rejects a DataFrame|LazyFrame.
    _nodes: "Optional[pl.LazyFrame]"
    _edges: "Optional[pl.LazyFrame]"
    _node: Optional[str]
    _source: Optional[str]
    _destination: Optional[str]
    _edge: Optional[str]
    edges_empty: Optional[bool]

    def __init__(self, nodes_lf: "Optional[pl.LazyFrame]", edges_lf: "Optional[pl.LazyFrame]",
                 node: Optional[str], source: Optional[str], destination: Optional[str],
                 edge: Optional[str], edges_empty: Optional[bool] = None) -> None:
        self._nodes = nodes_lf
        self._edges = edges_lf
        self._node = node
        self._source = source
        self._destination = destination
        self._edge = edge
        self.edges_empty = edges_empty

    @staticmethod
    def step(p: Plottable) -> "_LazyShim":
        nd = p._nodes.lazy() if p._nodes is not None else None
        ed = p._edges.lazy() if p._edges is not None else None
        return _LazyShim(nd, ed, None, None, None, None, edges_empty=_known_empty(p._edges))


def _known_empty(frame: "Optional[PolarsFrame]") -> Optional[bool]:
    """Tri-state emptiness of an already-materialized frame: True/False, or None when unknown
    (frame absent, or already lazy so the height is not available without collecting)."""
    if frame is None or is_lazy(frame):
        return None
    # No cast: `is_lazy` is a TypeIs, so surviving this guard IS the proof that `frame` is the
    # eager member of the union and `.height` exists. A cast here would have asserted the same
    # fact the predicate already establishes, unchecked — and would have had to be repeated at
    # every other eager-side call site in the engine.
    return frame.height == 0


def _combine_edges(g: "_LazyShim",
                   steps: List[Tuple[ASTObject, "_LazyShim"]],
                   label_steps: List[Tuple[ASTObject, "_LazyShim"]],
                   has_multihop: bool = False) -> "pl.LazyFrame":
    """The output edge ROWS: the graph's edges restricted to the ids the traversal kept, plus a
    boolean flag column per named edge step.

    Takes the ``_LazyShim`` duck-type, NOT a ``Plottable``: the Track-B combine runs entirely on
    already-materialized-then-lazified hop frames, which is exactly what the shim carries (and
    why its frames are ``LazyFrame``, not the ``PolarsFrame`` union). ``steps`` are the edge
    steps (recomputed ones when ``has_multihop``); ``label_steps`` are the forward-pass steps
    used for the endpoint gates and the alias flags."""
    import polars as pl
    src, dst, node_col, edge_id = g._source, g._destination, g._node, g._edge
    assert src is not None and dst is not None and node_col is not None and edge_id is not None
    all_edges = g._edges
    assert all_edges is not None  # the shim's edge frame is the thing being combined

    frames: List["pl.LazyFrame"] = []
    for idx, (op, g_step) in enumerate(steps):
        edges_df = g_step._edges
        if edges_df is None:
            continue
        # A step with no edges contributes no ids to the union below, so drop it BEFORE the
        # endpoint gates. Height is read from the pre-lazy fact recorded by _LazyShim.step
        # because `.lazy()` erases it; `not is_lazy(...)` keeps the direct-eager-frame case working.
        if g_step.edges_empty is True or (not is_lazy(edges_df) and edges_df.height == 0):
            continue
        if has_multihop or (isinstance(op, ASTEdge) and not op.is_simple_single_hop()):
            # has_multihop: every edge step was already recomputed path-valid (forward re-exec over
            # its backward-pruned subgraph), so append ALL ids and skip the prev/next endpoint
            # semijoin below — it would drop intermediate-hop edges (and diverged by an edge/node
            # vs pandas on mixed single+multi chains). Port of pandas combine_steps has_multihop
            # branch: recompute all steps, concat as-is, no semijoin.
            frames.append(edges_df.select(pl.col(edge_id)))
            continue
        prev_nodes = label_steps[idx - 1][1]._nodes if idx > 0 else g._nodes
        next_nodes = label_steps[idx + 1][1]._nodes if idx + 1 < len(label_steps) else None
        direction = op.direction if isinstance(op, ASTEdge) else "forward"

        if direction == "undirected" and prev_nodes is not None and next_nodes is not None:
            fwd = _semi(_semi(edges_df, prev_nodes, src, node_col), next_nodes, dst, node_col)
            rev = _semi(_semi(edges_df, prev_nodes, dst, node_col), next_nodes, src, node_col)
            edges_df = pl.concat([fwd, rev], how="vertical_relaxed").unique(subset=[edge_id])
        else:
            prev_col, next_col = (dst, src) if direction == "reverse" else (src, dst)
            if prev_nodes is not None:
                edges_df = _semi(edges_df, prev_nodes, prev_col, node_col)
            if next_nodes is not None:
                edges_df = _semi(edges_df, next_nodes, next_col, node_col)
        frames.append(edges_df.select(pl.col(edge_id)))

    if not frames:
        out_ids = all_edges.select(pl.col(edge_id)).limit(0)
    else:
        out_ids = pl.concat(frames, how="vertical_relaxed").unique(subset=[edge_id])

    out = all_edges.join(out_ids, on=edge_id, how="semi")

    for op, g_step in label_steps:
        if op._name is not None and isinstance(op, ASTEdge) and g_step._edges is not None and op._name in colnames(g_step._edges):
            named = g_step._edges.filter(pl.col(op._name)).select(pl.col(edge_id)).with_columns(pl.lit(True).alias(op._name))
            if op._name in colnames(out):  # the alias marker shadows the user column; its values stay under the restore name for Cypher scoping
                restore = shadow_restore_column(op._name)
                out = out.drop([c for c in (restore,) if c in colnames(out)]).rename({op._name: restore})
            out = out.join(named, on=edge_id, how="left").with_columns(pl.col(op._name).fill_null(False))
    return out


def _combine_node_ids(g: "_LazyShim",
                      steps: List[Tuple[ASTObject, "_LazyShim"]]) -> "pl.LazyFrame":
    """One-column frame of the node ids the traversal kept, unioned over the pruned steps.

    IDS ONLY, not the node rows: the caller still has to fold in the surviving edges' endpoints
    before materializing rows.

    Not deduplicated: the single consumer is a ``how="semi"`` key side, where duplicate keys can
    neither change which rows come back nor multiply them (see the module note on semi-join key
    frames). The caller's own ``.unique()`` on the materialized node rows is a DIFFERENT dedup
    (by node id, over rows) and is still required."""
    import polars as pl
    node_col = g._node
    assert node_col is not None
    all_nodes = g._nodes
    assert all_nodes is not None
    frames = [
        g_step._nodes.select(pl.col(node_col))
        for _, g_step in steps
        if g_step._nodes is not None and node_col in colnames(g_step._nodes)
    ]
    if not frames:
        return all_nodes.select(pl.col(node_col)).limit(0)
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical_relaxed")


def _materialize_node_rows(all_nodes: "pl.LazyFrame", step_ids: "pl.LazyFrame",
                           endpoint_ids_frame: "pl.LazyFrame", node_col: str) -> "pl.LazyFrame":
    """The output node ROWS: every node the steps kept, plus every endpoint of a surviving edge.

    Union the two ID sides FIRST, then read the node table ONCE. Row identity is unchanged:
    semi-joining the UNION of two key sets selects exactly the rows the two semi-joins selected
    between them.

    Neither id side is deduplicated — both feed a ``how="semi"`` key side, where duplicates
    cannot change or multiply the rows that come back. The trailing ``unique`` is a DIFFERENT
    dedup and is REQUIRED: it is over the node ROWS, and these rows go on to feed ``how="left"``
    alias joins where a node table carrying the same id twice would multiply every matching row.

    Row ORDER out of here is arbitrary — a polars semi-join does not preserve left-frame order —
    and the caller restores input-frame order with an explicit sort. WHICH duplicate row survives
    ``maintain_order`` is stable only under the default in-memory collect; under streaming collect
    it can differ, so anything needing a specific survivor must order explicitly."""
    import polars as pl
    ids = pl.concat([step_ids, endpoint_ids_frame], how="vertical_relaxed")
    return all_nodes.join(ids, on=node_col, how="semi").unique(
        subset=[node_col], maintain_order=True)


def _apply_node_names(out: "pl.LazyFrame", g: "_LazyShim",
                      steps: List[Tuple[ASTObject, "_LazyShim"]],
                      auto_hop_col: str = _AUTO_NODE_HOP) -> "pl.LazyFrame":
    """Tag node aliases on the FINAL node frame (after endpoint materialization). A node carries
    the alias iff it matched the named step in the backward-PRUNED frame (dead-end matches
    excluded) AND, when followed by an edge step, participates in that edge's PRUNED edges.
    Pruned ``steps`` (not forward frames) are essential — forward frames over-include and would
    tag nodes absent from the final graph."""
    import polars as pl
    node_col, src, dst = g._node, g._source, g._destination
    assert node_col is not None and src is not None and dst is not None
    step_list: List[Tuple[ASTObject, "_LazyShim"]] = list(steps)
    for idx, (op, g_step) in enumerate(step_list):
        if op._name is None or not isinstance(op, ASTNode) or g_step._nodes is None:
            continue
        if op._name not in colnames(g_step._nodes):
            continue
        named = g_step._nodes.filter(pl.col(op._name)).select(pl.col(node_col)).unique()
        # #1741 hop-distance gate: a node named AFTER a variable-length edge carries the alias
        # only if its hop distance lands inside that edge's [min_hop, max_hop] window (pandas
        # chain.py:456-501). An unlabeled node (null distance) always fails — which is how the
        # seed of an undirected `*1..2` walk loses the alias when the walk backtracks into it.
        if idx > 0:
            prev_op, _prev_step = step_list[idx - 1]
            # The distance travels WITH the wavefront, so it lands on this node step's own frame.
            if isinstance(prev_op, ASTEdge) and auto_hop_col in colnames(g_step._nodes):
                min_hop, max_hop = _alias_hop_bounds(prev_op)
                hop = pl.col(auto_hop_col)
                in_window = hop.is_not_null() & (hop >= min_hop)
                if max_hop is not None:
                    in_window = in_window & (hop <= max_hop)
                named = named.join(
                    g_step._nodes.filter(in_window).select(pl.col(node_col)),
                    on=node_col, how="semi")
        if idx + 1 < len(step_list):
            next_op, next_step = step_list[idx + 1]
            # Cardinality guard, restated against a fact that SURVIVES lazification. The old
            # spelling was `is_lazy(df) or df.height > 0`, and `_apply_node_names` is always
            # called with lazified steps — so `is_lazy` short-circuited True and the height
            # test was unreachable. That is the identical silent death this commit fixes one
            # function above; leaving a second copy of it here is how the bug recurs.
            # Unlike the edges combine this one is SEMANTIC, not a cost guard: an empty next
            # edge step must not empty `named` via the gate below.
            # Plain attribute read, not getattr: `next_step` is a `_LazyShim`, which declares
            # `edges_empty` in __slots__ with a real `Optional[bool]` annotation, so the
            # tri-state is part of the type and a typo here is a checker error rather than a
            # silent None (which would have re-armed the very gate this guard disarms).
            next_edges_empty = next_step.edges_empty
            if (isinstance(next_op, ASTEdge) and next_step._edges is not None
                    and next_edges_empty is not True):
                e = next_step._edges
                if next_op.direction == "forward":
                    part = e.select(pl.col(src).alias(node_col))
                elif next_op.direction == "reverse":
                    part = e.select(pl.col(dst).alias(node_col))
                else:
                    part = endpoint_ids(e, src, dst, node_col)
                # `part` is a semi-join key side (dupes cannot change it); `named` above keeps
                # its .unique() because it feeds a how="left" join, where they WOULD multiply.
                named = named.join(part, on=node_col, how="semi")
        flag = named.with_columns(pl.lit(True).alias(op._name))
        if op._name in colnames(out):
            out = out.drop(op._name)  # the marker replaces a colliding column, as pandas' combine does
        out = out.join(flag, on=node_col, how="left").with_columns(pl.col(op._name).fill_null(False))
    return out


def _call_native_on_polars(op) -> bool:
    """Whether a row-pipeline call has a native polars implementation (no bridge)."""
    from graphistry.compute.ast import ASTCall
    from graphistry.compute.gfql.row.pipeline import _POLARS_NATIVE_ROW_PIPELINE_CALLS
    if not isinstance(op, ASTCall):
        return False
    if op.function not in _POLARS_NATIVE_ROW_PIPELINE_CALLS:
        return False
    if op.function == "rows" and (
        op.params.get("binding_ops") is not None
        or op.params.get("alias_endpoints") is not None
    ):
        return False
    return True


def _run_calls_polars(g_cur, calls, start_nodes, base_graph, middle):
    """Execute a boundary run of ASTCall ops on a polars graph.

    Mirrors ``chain._handle_boundary_calls``: threads the row-pipeline context attrs and applies
    the named-middle → ``rows(binding_ops=...)`` rewrite. Each call runs natively via
    ``_try_native_row_op``; a row op with no native impl raises NotImplementedError (no pandas
    fallback) rather than secretly running the pandas row pipeline.
    """
    from graphistry.compute.ast import ASTCall, ASTNode as _ASTNode, ASTEdge as _ASTEdge, rows as rows_fn
    from graphistry.compute.chain import serialize_binding_ops
    from graphistry.compute.gfql.exec_context import attach_row_exec_context, clear_row_exec_context

    calls = list(calls)
    if not calls:
        return g_cur

    # #1786: twin of the generic chain -- per-execution state on an INTERNAL COPY.
    # `g_cur` is the CALLER's graph on the all-calls boundary run, so assigning here
    # left the WITH re-entry seed behind for the next, unrelated query.
    g_cur = attach_row_exec_context(g_cur, start_nodes=start_nodes, rows_base_graph=base_graph)

    if (
        middle
        and any(op._name is not None for op in middle)
        and isinstance(calls[0], ASTCall)
        and calls[0].function == "rows"
        and calls[0].params.get("binding_ops") is None
        and calls[0].params.get("source") is None
        and calls[0].params.get("alias_endpoints") is None
        # See the twin guard in compute/chain.py: a NON-DEFAULT `table` names the table the
        # caller wants, so the bindings rewrite must not override it. Both surfaces need
        # the check — this one is the native polars chain, that one the generic chain.
        # `== "nodes"`, not `is None`: `rows()` defaults table to "nodes" and always emits
        # it, so an `is None` test disables the rewrite entirely.
        and calls[0].params.get("table", "nodes") == "nodes"
        and all(isinstance(op, (_ASTNode, _ASTEdge)) for op in middle)
    ):
        # See the twin in compute/chain.py: ADD binding_ops to the caller's call rather
        # than building a fresh one, so the params the rewrite has no opinion about
        # (`attach_prop_aliases`, `alias_prefilters`) reach the binding_ops builder.
        prev_params = calls[0].params
        calls = [rows_fn(
            binding_ops=serialize_binding_ops(middle),
            alias_prefilters=prev_params.get("alias_prefilters"),
            attach_prop_aliases=prev_params.get("attach_prop_aliases"),
        )] + list(calls[1:])

    # Per-op NATIVE-OR-DEFER. Ops that don't lower:
    #  - non-native ROW op (correlated-subquery semi_apply/anti_semi_apply/join_apply):
    #    parity-or-NIE — it SHOULD be polars-native; a bridge would hide that.
    #  - ANALYTIC (compute_cugraph/compute_igraph/layout_*/collapse/get_topological_levels/...),
    #    no native impl by nature: route via execute_call, which applies the off-engine modality
    #    policy (call_mode='auto' bridges to pandas/cuDF + coerces back; 'strict' declines) — the
    #    SAME path as the DAG/let() surface, keeping surfaces consistent. Row-vs-analytic split
    #    is MECHANICAL (is_row_pipeline_call), not curated. (umap/hypergraph never reach here —
    #    the generic chain routes schema-changers straight to execute_call.)
    from graphistry.compute.ast import ASTCall
    from graphistry.compute.gfql.row.pipeline import is_row_pipeline_call
    from graphistry.compute.exceptions import ErrorCode, GFQLTypeError, GFQLValidationError
    for op in calls:
        if not isinstance(op, ASTCall):
            raise NotImplementedError(
                f"polars engine does not yet natively support cypher row op "
                f"{op!r}; use engine='pandas' or engine='cudf' for this "
                f"query (no silent fallback; parity-or-error by design)"
            )
        try:
            native = _try_native_row_op(g_cur, op)
        except GFQLTypeError:
            raise
        except GFQLValidationError as validation_error:
            # Same wrapping `execute_call` applies (gfql/call/executor.py): a kernel that
            # raises a validation error surfaces as GFQLTypeError(E303) with this message
            # shape. The native attempt runs BEFORE execute_call, so without this the SAME
            # query carries a different class AND a different `.code` per engine — and this
            # repo's control flow keys on `.code`. Scoped to GFQLValidationError, the one
            # divergence actually observed (an E108 from the var-length cycle guard);
            # other exception classes are left alone rather than blanket-normalized.
            raise GFQLTypeError(
                ErrorCode.E303,
                f"Error executing '{op.function}': {validation_error}",
                field="function",
                value=op.function,
            ) from validation_error
        except _polars_error_types() as polars_error:
            # A THIRD-PARTY exception must never be the GFQL surface. On the pandas/cuDF side
            # `execute_call` already wraps any kernel exception as GFQLTypeError(E303) — the
            # native polars path runs BEFORE execute_call and so skipped that wrapper entirely,
            # letting e.g. `polars.exceptions.InvalidOperationError: \`sum\` operation not
            # supported for dtype \`str\`` reach the caller verbatim. Same code, same message
            # shape as the pandas surface; the polars text is preserved as the cause.
            raise GFQLTypeError(
                ErrorCode.E303,
                f"Error executing '{op.function}': {polars_error}",
                field="function",
                value=op.function,
            ) from polars_error
        if native is not None:
            g_cur = native
            continue
        if not is_row_pipeline_call(op.function):
            from graphistry.compute.gfql.call.executor import execute_call
            from graphistry.compute.gfql.lazy import active_target, ExecutionTarget
            from graphistry.Engine import Engine as _Engine
            _eng = _Engine.POLARS_GPU if active_target() == ExecutionTarget.GPU else _Engine.POLARS
            g_cur = execute_call(g_cur, op.function, op.params or {}, _eng)
            continue
        raise NotImplementedError(
            f"polars engine does not yet natively support cypher row op "
            f"{op.function!r}; use engine='pandas' or engine='cudf' for this "
            f"query (no silent fallback; parity-or-error by design)"
        )
    # Attach/detach pair: the boundary run is done, so the context is spent and must not
    # ride out on the result the caller sees (see the twin in compute/chain.py).
    return clear_row_exec_context(g_cur)



class _GroupByParams(TypedDict, total=False):
    """group_by's slice of ASTCall.params (wire JSON): keys + (out_col, agg, in_col)
    aggregation triples + optional per-key entity prefixes ("node." / edge alias dots)."""
    keys: List[str]
    aggregations: List[AggSpec]
    key_prefixes: Optional[List[str]]

def _try_native_row_op(g_cur, op):
    """Run a row-pipeline call natively on polars, or return None to defer (NIE)."""
    from graphistry.Engine import Engine
    from .row_pipeline import (
        select_polars, with_columns_polars, order_by_polars, group_by_polars,
        unwind_polars, where_rows_polars, binding_rows_polars,
    )
    from .pattern_apply import (
        rows_binding_ops_polars, semi_apply_mark_polars, anti_semi_apply_polars,
    )
    from .search import search_any_polars

    fn = op.function
    if fn == "rows" and op.params.get("binding_ops") is not None:
        # #1731: single-entity boundary rows (MATCH (n) / EXISTS seeds) are handled by
        # the pattern-apply helper; try that narrow shape first. alias_prefilters
        # thread through — a helper that ignored them would silently drop the filter.
        if op.params.get("source") is None:
            out = rows_binding_ops_polars(
                g_cur, op.params["binding_ops"],
                alias_prefilters=op.params.get("alias_prefilters"),
            )
            if out is not None:
                return out
        # #1730 gate: only take the multi-alias bindings table when alias_endpoints is
        # absent (the alias-endpoints shape is handled elsewhere and must fall through).
        if op.params.get("alias_endpoints") is None:
            # Multi-alias bindings table (#1709): native for fixed-length connected
            # patterns. A decline must fall through to the pre-existing correlated
            # pattern handler below (EXISTS/searchAny); returning None here would turn
            # those already-native shapes into an NIE.
            bindings_result = binding_rows_polars(
                g_cur, op.params["binding_ops"], op.params.get("attach_prop_aliases"),
                alias_prefilters=op.params.get("alias_prefilters"),
            )
            if bindings_result is not None:
                return bindings_result
    if _call_native_on_polars(op):
        # frame ops (rows/limit/skip/distinct/drop_cols) — engine-polymorphic
        return op.execute(g=g_cur, prev_node_wavefront=None, target_wave_front=None, engine=Engine.POLARS)
    if fn == "semi_apply_mark":
        # required params are safelist-validated — direct indexing (an or-default
        # here could only mask an unvalidated call); neq is the optional one.
        return semi_apply_mark_polars(
            g_cur, op.params["binding_ops"], op.params["join_aliases"],
            op.params["out_col"], neq=op.params.get("neq"),
        )
    if fn == "search_any":
        return search_any_polars(
            g_cur, op.params["alias"], op.params["term"], op.params["out_col"],
            case_sensitive=op.params.get("case_sensitive", False),
            regex=op.params.get("regex", False),
            columns=op.params.get("columns"),
        )
    if fn == "anti_semi_apply":
        return anti_semi_apply_polars(
            g_cur, op.params["binding_ops"], op.params["join_aliases"],
            neq=op.params.get("neq"),
        )
    if fn in ("select", "return_"):
        return select_polars(g_cur, op.params.get("items", []))
    if fn == "with_":
        # extend=True (WITH keeping existing columns) -> with_columns; extend=False (full
        # re-projection) -> select. Both decline (NIE) on an unlowerable item.
        if op.params.get("extend", False):
            return with_columns_polars(g_cur, op.params.get("items", []))
        return select_polars(g_cur, op.params.get("items", []))
    if fn == "where_rows":
        return where_rows_polars(g_cur, op.params.get("filter_dict"), op.params.get("expr"))
    if fn == "fill_empty_row":
        from .row_pipeline import fill_empty_row_polars
        return fill_empty_row_polars(g_cur, op.params["row"])
    if fn == "order_by":
        return order_by_polars(g_cur, op.params.get("keys", []))
    if fn == "group_by":
        # ASTCall.params is a wire-format Dict[str, Any] whose schema is keyed by the
        # RUNTIME value of op.function, so precise typing lives in per-function
        # TypedDicts narrowed at dispatch (full tagged-union ASTCall is a larger AST
        # refactor). The cast is sound: validate() has already checked these shapes.
        gb = cast(_GroupByParams, op.params)
        return group_by_polars(
            g_cur,
            gb.get("keys", []),
            gb.get("aggregations", []),
            key_prefixes=gb.get("key_prefixes"),
        )
    if fn == "unwind":
        return unwind_polars(g_cur, op.params.get("expr", ""), op.params.get("as_", "value"))
    if fn == "get_degrees":
        return get_degrees_polars(
            g_cur,
            col=op.params.get("col", "degree"),
            degree_in=op.params.get("degree_in", "degree_in"),
            degree_out=op.params.get("degree_out", "degree_out"),
        )
    if fn == "get_indegrees":
        return get_indegrees_polars(g_cur, col=op.params.get("col", "degree_in"))
    if fn == "get_outdegrees":
        return get_outdegrees_polars(g_cur, col=op.params.get("col", "degree_out"))
    return None


def chain_polars(self: Plottable, ops, start_nodes: Optional[Any] = None) -> Plottable:
    from graphistry.compute.ast import ASTCall
    from graphistry.compute.chain import Chain, _get_boundary_calls

    # Normalize input eagerness ONCE: polars joins do not mix eagerness, and the
    # traversal joins user frames against eager wavefronts throughout.
    if self._nodes is not None and is_lazy(self._nodes):
        self = self.nodes(self._nodes.collect(), self._node)
    if self._edges is not None and is_lazy(self._edges):
        self = self.edges(self._edges.collect(), self._source, self._destination)

    if isinstance(ops, Chain):
        ops = ops.chain
    ops = list(ops)

    if len(ops) == 0:
        return self

    # Reject duplicate alias names (node/edge aliases scoped separately; mirrors the pandas
    # combine_steps E201 guard). A reused name like [n(name='a'), e(), n(name='a')] would
    # produce a malformed schema (colliding a/a_right join cols) — wrong where pandas declines.
    for _alias_type in (ASTNode, ASTEdge):
        _seen: dict = {}
        for _idx, _op in enumerate(ops):
            _name = getattr(_op, "_name", None)
            if _name is not None and isinstance(_op, _alias_type):
                if _name in _seen:
                    from graphistry.compute.exceptions import GFQLValidationError, ErrorCode
                    raise GFQLValidationError(
                        code=ErrorCode.E201,
                        message=f"Duplicate alias name '{_name}' in chain (steps {_seen[_name]} and {_idx})",
                        suggestion="Use distinct alias names for each step in the chain",
                    )
                _seen[_name] = _idx

    has_call = any(isinstance(op, ASTCall) for op in ops)
    has_traversal = any(isinstance(op, (ASTNode, ASTEdge)) for op in ops)

    if not has_call:
        return _chain_traversal_polars(self, ops, start_nodes)

    if not has_traversal:
        # Pure call chain (e.g. let() bodies): no traversal, just run the calls.
        return _run_calls_polars(self, ops, start_nodes, base_graph=self, middle=[])

    prefix, middle, suffix = _get_boundary_calls(ops)

    # has_traversal is True here, so middle is non-empty.
    has_call_in_middle = any(isinstance(op, ASTCall) for op in middle)
    has_traversal_in_middle = any(isinstance(op, (ASTNode, ASTEdge)) for op in middle)
    if has_call_in_middle and has_traversal_in_middle:
        from graphistry.compute.exceptions import GFQLValidationError, ErrorCode
        raise GFQLValidationError(
            code=ErrorCode.E201,
            message="Cannot mix call() operations with n()/e() traversals in interior of chain",
            suggestion="call() operations are only allowed at chain boundaries (start/end).",
        )

    if prefix:
        # decline (NIE): leading call() yields a row table the following traversal would have to
        # re-enter as a graph. pandas/cuDF cascade via _chain_impl, but it's not a cypher shape
        # (MATCH comes first) and the polars traversal doesn't yet consume a row-table input.
        raise NotImplementedError(
            "polars chain engine does not yet support call() before a traversal; "
            "use engine='pandas' or engine='cudf' for this chain."
        )

    from graphistry.compute.chain import serialize_binding_ops
    from graphistry.compute.gfql.index.handoff import (
        IndexedBindingsHandoff, attach_handoff,
    )

    indexed_state, indexed_attempted = _try_indexed_middle_polars(self, middle, suffix, start_nodes)
    if indexed_state is not None:
        # Skip the canonical traversal: the compact indexed path bag already IS the
        # binding rows the suffix asks for. Hand it to the unchanged native rows
        # materializer through an internal graph copy.
        g_cur = attach_handoff(self, IndexedBindingsHandoff(
            binding_ops=serialize_binding_ops(middle),
            state=indexed_state,
            edge_aliases=tuple(
                op._name for op in middle
                if isinstance(op, ASTEdge) and isinstance(op._name, str)
            ),
        ))
    else:
        g_cur = _chain_traversal_polars(self, middle, start_nodes)
        if indexed_attempted:
            # Record the exact declined plan so the rows materializer does not
            # re-attempt (and re-record) the same decision. Attach, never mutate:
            # the pandas twin does the same, so neither boundary can write onto a
            # graph it does not own.
            g_cur = attach_handoff(g_cur, IndexedBindingsHandoff(
                binding_ops=serialize_binding_ops(middle),
            ))
    if suffix:
        g_cur = _run_calls_polars(g_cur, suffix, start_nodes, base_graph=self, middle=middle)
    return g_cur


def _try_indexed_middle_polars(
    g: Plottable,
    middle: List[ASTObject],
    suffix: List[ASTObject],
    start_nodes: Optional[Any],
) -> Tuple[Optional["IndexedBindingsState"], bool]:
    """Attempt the indexed fixed-hop path BEFORE the canonical polars traversal.

    Returns ``(state_or_None, attempted)``. The shape gate mirrors the pandas
    boundary gate and ``_run_calls_polars``' named-middle rewrite: the bypass is
    only sound when the leading rows call consumes exactly this middle as binding
    ops. Everything else (prefiltered/seeded/aliased-endpoint rows, unnamed middle
    without binding ops, non-traversal middle) keeps the canonical path.
    """
    from graphistry.compute.ast import ASTCall
    from graphistry.compute.chain import serialize_binding_ops

    if (
        not middle
        or not suffix
        or start_nodes is not None
        or not isinstance(suffix[0], ASTCall)
        or suffix[0].function != "rows"
        or suffix[0].params.get("source") is not None
        or suffix[0].params.get("alias_endpoints") is not None
        or suffix[0].params.get("alias_prefilters")
        # See the twin guard in chain._plan_indexed_middle: serving the bypass skips the
        # canonical traversal, so a non-default `table` would read the PRE-traversal edge
        # table (the whole graph) instead of the narrowed one. "nodes", not None — `rows()`
        # always emits `table`.
        or suffix[0].params.get("table", "nodes") != "nodes"
        or not all(isinstance(op, (ASTNode, ASTEdge)) for op in middle)
    ):
        return None, False
    binding_ops = suffix[0].params.get("binding_ops")
    if not (
        binding_ops == serialize_binding_ops(middle)
        or (
            binding_ops is None
            and any(op._name is not None for op in middle)
        )
    ):
        return None, False

    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index.bindings import try_indexed_connected_bindings_state
    from graphistry.compute.gfql.lazy import active_target, ExecutionTarget

    engine = Engine.POLARS_GPU if active_target() == ExecutionTarget.GPU else Engine.POLARS
    return try_indexed_connected_bindings_state(g, middle, engine=engine), True


def _bound_edge_endpoints(g: Plottable) -> Tuple[str, str]:
    """The graph's (source, destination) columns; a row-pipeline call() result leaves them unbound."""
    if g._source is None or g._destination is None:
        raise NotImplementedError(
            "polars chain engine does not yet support traversing a graph with unbound edge "
            "endpoints (e.g. a call() row-pipeline result); use engine='pandas' or "
            "engine='cudf' for this chain."
        )
    return g._source, g._destination


def _chain_traversal_polars(self: Plottable, ops, start_nodes: Optional[Any] = None) -> Plottable:
    import polars as pl
    from graphistry.compute.chain import Chain

    if isinstance(ops, Chain):
        ops = ops.chain
    ops = list(ops)

    if len(ops) == 0:
        return self

    edge_src, edge_dst = _bound_edge_endpoints(self)

    # Node-only shape: single MATCH (n). Result is just the filtered node table + empty edges,
    # so skip forward/backward/combine. Byte-identical: the one-node-step combine yields filtered
    # g._nodes in order + empty edges + the alias flag on every matched node.
    if len(ops) == 1 and isinstance(ops[0], ASTNode) and ops[0].query is None:
        op0 = ops[0]
        g0 = ensure_nodes_polars(self)
        nc = g0._node
        assert nc is not None
        from graphistry.Engine import EngineAbstract
        nodes = _single_node_rows_via_index_or_filter(g0, op0, EngineAbstract.POLARS)
        if start_nodes is not None:
            from graphistry.Engine import Engine as _E, df_to_engine as _d2e
            seed = _align_seed_dtype(_d2e(start_nodes, _E.POLARS), nc, g0._nodes)
            nodes = _semi(nodes, seed, nc, nc)
        if op0._name is not None:
            nodes = nodes.with_columns(pl.lit(True).alias(op0._name))
        return g0.nodes(nodes, nc).edges(g0._edges.clear(), edge_src, edge_dst)

    if isinstance(ops[0], ASTEdge):
        ops = [ASTNode()] + ops
    if isinstance(ops[-1], ASTEdge):
        ops = ops + [ASTNode()]

    if any(isinstance(op, ASTEdge) and op.prune_to_endpoints and op.is_simple_single_hop() for op in ops):
        raise NotImplementedError(
            "polars chain engine: prune_to_endpoints on a single-hop edge (arrival-side pruning "
            "by hop label) is not implemented; use engine='pandas' or engine='cudf'"
        )
    if any(
        isinstance(op, ASTEdge) and not op.is_simple_single_hop() and not _is_native_multihop(op)
        for op in ops
    ):
        raise NotImplementedError(
            "polars chain engine supports single-hop and multi-hop edges "
            "(hops=N / max_hops fwd/rev/undirected; to_fixed_point and min_hops>1 "
            "fwd/rev); deferred features (undirected to_fixed_point / undirected "
            "min_hops>1, output slicing, hop labels, *_query, "
            "include_zero_hop_seed, prune_to_endpoints) require engine='pandas'."
        )

    # issue #1748: forward/reverse min_hops>1 gets its node hop-labels from pandas' layered
    # backward walk, which is not ported — so a node named AFTER such an edge cannot be hop-gated
    # (_auto_node_hop_col returns None for min_hops>1), and its alias would include nodes OUTSIDE
    # the [min_hop, max_hop] window. The node/edge SETS stay correct; only the projected alias is
    # wrong. Decline that specific shape (honest NIE) instead of emitting silently-wrong rows —
    # min_hops>1 chains WITHOUT a following node alias keep running. Adapted from the retired
    # #1742 decline pattern (which did the same for undirected, now natively gated by #1741).
    for _i in range(1, len(ops)):
        _prev, _cur = ops[_i - 1], ops[_i]
        if (isinstance(_prev, ASTEdge) and _prev.direction in ("forward", "reverse")
                and _prev.min_hops is not None and _prev.min_hops > 1
                and isinstance(_cur, ASTNode) and _cur._name is not None):
            raise NotImplementedError(
                "polars chain engine: a node alias after a forward/reverse variable-length edge "
                "with min_hops>1 is not yet hop-gated (would tag nodes outside the hop window — "
                "issue #1748); use engine='pandas' or engine='cudf' for this query"
            )

    edge_ops = [op for op in ops if isinstance(op, ASTEdge)]
    # Undirected edges in multi-edge chains: NATIVE for single-hop (backward pass threads BOTH
    # endpoints — see override below) and fixed-length multi-hop (generic backward hop +
    # path-aware recompute, same as directed). decline (NIE): undirected carrying
    # include_zero_hop_seed / *_query (unverified combine semantics); undirected to_fixed_point
    # is NIE'd upstream by _is_native_multihop (needs components/2-core).
    if len(edge_ops) > 1:
        _undirected = [op for op in edge_ops if op.direction == "undirected"]
        _undirected_unsupported = any(
            op.include_zero_hop_seed
            or op.source_node_query is not None
            or op.destination_node_query is not None
            or op.edge_query is not None
            for op in _undirected
        )
        if _undirected and _undirected_unsupported:
            raise NotImplementedError(
                "polars chain engine supports single-hop and fixed-length multi-hop "
                "undirected edges in multi-edge chains; deferred undirected sub-cases — "
                "include_zero_hop_seed or *_query — require engine='pandas'."
            )
    plain_shape = polars_plain_single_hop_admits(ops, start_nodes)
    if plain_shape == "seeded-index":
        indexed = _plain_seeded_index_hop_polars(self, ops)
        if indexed is not None:
            return indexed
    if start_nodes is None:
        seeded = _try_seeded_chain_polars(self, ops)
        if seeded is not None:
            return seeded
    if plain_shape is not None:
        return _plain_single_hop_polars(self, ops)

    if start_nodes is not None:
        from graphistry.Engine import Engine, df_to_engine
        start_nodes = df_to_engine(start_nodes, Engine.POLARS)

    g = ensure_nodes_polars(self)
    assert g._node is not None and g._source is not None and g._destination is not None
    start_nodes = _align_seed_dtype(start_nodes, g._node, g._nodes)
    g, _endpoint_restore = _align_edge_endpoints(g, g._node, g._source, g._destination)
    if g._edge is None:
        EID = "__gfql_edge_index__"
        pre_index_edges = g._edges
        g = g.edges(g._edges.with_row_index(EID), g._source, g._destination, edge=EID)
        added_edge_index = True
        from graphistry.compute.gfql.index import get_registry, set_registry
        _reg = get_registry(g)
        if not _reg.is_empty():
            g = set_registry(g, _reg.rebind_edges(g._edges, pre_index_edges))
    else:
        EID = g._edge
        added_edge_index = False

    node_col, src, dst = g._node, g._source, g._destination
    assert node_col is not None and src is not None and dst is not None

    # Resolve the #1741 auto hop-label column against the user's node columns ONCE, so it can't
    # collide with a real column (a clash would clobber user data and crash the int/str gate).
    # Same helper the endpoint block below uses for NORD/EORD.
    from graphistry.compute.util import generate_safe_column_name as _gsafe
    auto_hop_col = _gsafe(_AUTO_NODE_HOP, g._nodes, prefix="__gfql_", suffix="__")

    # Forward pass.
    g_stack: List[Plottable] = []
    for i, op in enumerate(ops):
        prev = start_nodes if i == 0 else g_stack[-1]._nodes
        g_stack.append(_exec(op, g, prev, None, auto_hop_col=auto_hop_col))

    # Backward pass.
    g_rev: List[Plottable] = []
    for op, g_step in zip(reversed(ops), reversed(g_stack)):
        prev_loop = g_stack[-1] if len(g_rev) == 0 else g_rev[-1]
        if len(g_rev) == len(g_stack) - 1:
            prev_orig = None
        else:
            prev_orig = g_stack[-(len(g_rev) + 2)]
        prev_wf = prev_loop._nodes
        target_wf = prev_orig._nodes if prev_orig is not None else None
        # OUTPUT universe = FULL g._nodes (reverse hop output not truncated / keeps node columns);
        # the INTERMEDIATE-hop gate universe is decoupled. Multi-hop reverse: pass the forward
        # wavefront (g_step._nodes, pre-widening) so intermediate hops can't wander outside
        # forward-reached nodes (pandas chain.py:1104 feeds g_step as the multi-hop reverse node
        # table). Single-hop reverse: None -> gate = all_nodes (vacuous), matching pandas
        # use_fast_backward (full g._nodes).
        _iu = g_step._nodes if (isinstance(op, ASTEdge) and not op.is_simple_single_hop()) else None
        g_step_full = g_step.nodes(g._nodes, g._node).edges(
            _step_edges_with_source_columns(g, g_step, EID), src, dst, edge=EID)
        rev = _exec(op.reverse(), g_step_full, prev_wf, target_wf, intermediate_universe=_iu,
                    auto_hop_col=auto_hop_col)
        # Undirected single-hop backward threading: the generic hop returns a ONE-SIDED
        # (TO-side) wavefront; pandas' fast backward branch (chain.py:1090-1098) threads BOTH
        # endpoints of surviving edges. One-sided drops an intermediate node reachable only as
        # the frontier-side endpoint of a sibling edge, and the combine then drops its incident
        # edge (fuzz seed-18: >1 undirected edge + intermediate node filters loses a node vs
        # pandas). Edge set already matches (both orientations joined), so override ONLY the
        # node frame — with FULL node columns so the next backward node step can still filter.
        if (isinstance(op, ASTEdge) and op.direction == "undirected"
                and op.is_simple_single_hop() and rev._edges is not None):
            _both = endpoint_ids(rev._edges, src, dst, node_col).unique(subset=[node_col])
            rev = rev.nodes(g._nodes.join(_both, on=node_col, how="semi"), node_col)
        g_rev.append(rev)

    steps: List[Tuple[ASTObject, Plottable]] = list(zip(ops, list(reversed(g_rev))))
    label_steps: List[Tuple[ASTObject, Plottable]] = list(zip(ops, g_stack))

    # Native multi-hop: port of pandas combine_steps has_multihop branch (chain.py:201-209).
    # The per-step single-hop endpoint semijoin in _combine_edges would drop intermediate-hop
    # edges (observed e=3 vs pandas e=10). Instead RE-EXECUTE the FORWARD hop over each step's
    # backward-pruned edge subgraph, seeded from the forward-pass entry wavefront: backward
    # prune = "reaches a valid endpoint", forward re-exec = "reachable from seed", and their
    # composition along the BFS frontier is exactly the valid-path edges (NOT a flat
    # forward∩backward intersection — the reverted shortcut). Only the EDGE combine sees the recomputed
    # frames (pandas gates on kind=='edges'); node combine + name tagging keep the original
    # backward-pruned steps.
    edge_steps: List[Tuple[ASTObject, Plottable]] = steps
    has_multihop = any(isinstance(op, ASTEdge) and not op.is_simple_single_hop() for op, _ in steps)
    if has_multihop:
        # pandas recomputes EVERY step when any is multi-hop (chain.py:201-209), then concats
        # with NO per-step semijoin. Mirror exactly: recompute ALL ASTEdge steps and let
        # _combine_edges append-all (has_multihop=True). Recomputing only the multi-hop step +
        # semijoining the single-hop steps diverged by an edge/node on mixed chains
        # (fuzz seeds 24/48).
        edge_steps = []
        for idx, (op, g_step) in enumerate(steps):
            if isinstance(op, ASTEdge):
                prev_src = label_steps[idx - 1][1]._nodes if idx > 0 else g_step._nodes
                prev_wf = (
                    _semi(g._nodes, prev_src, node_col, node_col)
                    if prev_src is not None else None
                )
                g_sub = g.edges(_step_edges_with_source_columns(g, g_step, EID), src, dst, edge=g._edge)
                edge_steps.append((op, _exec(op, g_sub, prev_wf, None, auto_hop_col=auto_hop_col)))
            else:
                edge_steps.append((op, g_step))

    # Track B: build the WHOLE combine (combine_nodes/edges + endpoint + names) as ONE deferred
    # plan over already-materialized hop frames and collect ONCE. NO recompute (inputs already
    # materialized). Stable order columns restore the eager g._nodes/g._edges order (lazy joins
    # don't preserve it) so a trailing row pipeline's LIMIT/SKIP is unaffected.
    from graphistry.compute.util import generate_safe_column_name
    from graphistry.compute.gfql.lazy import collect_all
    NORD = generate_safe_column_name("__gfql_norder__", g._nodes, prefix="__gfql_", suffix="__")
    if added_edge_index:
        # EID was attached above as `with_row_index` over THIS frame in THIS order, so it
        # already IS the stable edge order; reuse it rather than adding a second 0..n-1 column.
        EORD = EID
        edges_lz = g._edges.lazy()
    else:
        EORD = generate_safe_column_name("__gfql_eorder__", g._edges, prefix="__gfql_", suffix="__")
        edges_lz = g._edges.with_row_index(EORD).lazy()
    g_lz = _LazyShim(g._nodes.with_row_index(NORD).lazy(), edges_lz,
                     node_col, src, dst, g._edge)
    steps_lz = [(op, _LazyShim.step(p)) for op, p in steps]
    edge_steps_lz = [(op, _LazyShim.step(p)) for op, p in edge_steps]
    label_lz = [(op, _LazyShim.step(p)) for op, p in label_steps]

    node_ids = _combine_node_ids(g_lz, steps_lz)
    final_edges = _combine_edges(g_lz, edge_steps_lz, label_lz, has_multihop)
    all_nodes_lz = g_lz._nodes
    assert all_nodes_lz is not None  # constructed from g._nodes two statements above
    final_nodes = _materialize_node_rows(
        all_nodes_lz, node_ids, endpoint_ids(final_edges, src, dst, node_col), node_col)
    final_nodes = _apply_node_names(final_nodes, g_lz, steps_lz, auto_hop_col=auto_hop_col)

    final_nodes = final_nodes.sort(NORD).drop(NORD)
    # EORD IS EID whenever we synthesized the id (that is the point of this change), so the
    # single drop above removes it. There is no `added_edge_index and EID != EORD` case left
    # to handle: the only branch that sets added_edge_index also sets EORD = EID.
    final_edges = final_edges.sort(EORD).drop(EORD)
    # Distinct names on the eager side: `final_nodes` is statically a LazyFrame all the way
    # down this block, and `collect_all` hands back DataFrames — rebinding would be a type
    # error, and silencing it would cost the lazy/eager distinction the shim exists to keep.
    final_edges_eager, final_nodes_eager = collect_all([final_edges, final_nodes])
    final_edges_eager = _restore_edge_dtypes(final_edges_eager, src, dst, _endpoint_restore)
    return self.nodes(final_nodes_eager, node_col).edges(final_edges_eager, src, dst)
