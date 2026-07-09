"""Native Polars chain() — Phase 1, vectorized.

Reimplements the chain forward/backward/combine orchestration in polars, reusing the polars hop
for edge steps. Vectorization-first: set ops are semi/anti joins, alias tags are join-based flag
columns — no Python id lists or ``is_in(python_list)``. Parity-or-NIE contract: differential
parity vs the pandas chain gates correctness; unsupported shapes raise NotImplementedError
(no silent pandas fallback). Deferred: variable-length/multi-hop edge sub-cases, some
undirected multi-edge combos, node query=.
"""
from typing import Any, List, Optional, Tuple

from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTObject, ASTNode, ASTEdge
from .hop_eager import ensure_nodes_polars
from .dtypes import is_lazy, colnames, endpoint_ids
from .degrees import get_degrees_polars, get_indegrees_polars, get_outdegrees_polars
from .predicates import filter_by_dict_polars


def _semi(df, ids_df, df_col, id_col):
    """Rows of df whose df_col is present in ids_df[id_col] (vectorized semi-join)."""
    return df.join(ids_df.select(id_col).unique(), left_on=df_col, right_on=id_col, how="semi")


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

    polars won't auto-cast int↔float join keys, and a null endpoint promotes its column to float
    while int node ids stay int — endpoint↔node-id joins then raise SchemaError where pandas joins
    fine. The hop casts internally; the chain (fast paths + combine) did not. Returns
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


def _exec(op: ASTObject, g: Plottable, prev_wf, target_wf, intermediate_universe=None) -> Plottable:
    import polars as pl

    node_col = g._node
    assert node_col is not None
    if isinstance(op, ASTNode):
        if op.query is not None:
            raise NotImplementedError("polars chain engine does not yet support node query=")
        base = prev_wf if prev_wf is not None else g._nodes
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
            label_node_hops=op.label_node_hops,
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
    # components + 2-core seed retention (hop.py:817-887), not reproducible in the vectorized hop.
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
    combine helpers run lazily over already-materialized hop frames without Plottable rebinds."""
    __slots__ = ("_nodes", "_edges", "_node", "_source", "_destination", "_edge")

    def __init__(self, nodes_lf, edges_lf, node, source, destination, edge):
        self._nodes = nodes_lf
        self._edges = edges_lf
        self._node = node
        self._source = source
        self._destination = destination
        self._edge = edge

    @staticmethod
    def step(p):
        nd = p._nodes.lazy() if p._nodes is not None else None
        ed = p._edges.lazy() if p._edges is not None else None
        return _LazyShim(nd, ed, None, None, None, None)


def _combine_edges(g, steps, label_steps, has_multihop=False):
    import polars as pl
    src, dst, node_col, edge_id = g._source, g._destination, g._node, g._edge
    assert src is not None and dst is not None and node_col is not None and edge_id is not None

    frames = []
    for idx, (op, g_step) in enumerate(steps):
        edges_df = g_step._edges
        if edges_df is None:
            continue
        if not is_lazy(edges_df) and edges_df.height == 0:
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
        out_ids = g._edges.select(pl.col(edge_id)).limit(0)
    else:
        out_ids = pl.concat(frames, how="vertical_relaxed").unique(subset=[edge_id])

    out = g._edges.join(out_ids, on=edge_id, how="semi")

    for op, g_step in label_steps:
        if op._name is not None and isinstance(op, ASTEdge) and g_step._edges is not None and op._name in colnames(g_step._edges):
            named = g_step._edges.filter(pl.col(op._name)).select(pl.col(edge_id)).with_columns(pl.lit(True).alias(op._name))
            out = out.join(named, on=edge_id, how="left").with_columns(pl.col(op._name).fill_null(False))
    return out


def _combine_nodes(g, steps):
    import polars as pl
    node_col = g._node
    assert node_col is not None
    frames = [
        g_step._nodes.select(pl.col(node_col))
        for _, g_step in steps
        if g_step._nodes is not None and node_col in colnames(g_step._nodes)
    ]
    if frames:
        ids = pl.concat(frames, how="vertical_relaxed").unique(subset=[node_col])
    else:
        ids = g._nodes.select(pl.col(node_col)).limit(0)
    return g._nodes.join(ids, on=node_col, how="semi")


def _apply_node_names(out, g, steps):
    """Tag node aliases on the FINAL node frame (after endpoint materialization). A node carries
    the alias iff it matched the named step in the backward-PRUNED frame (dead-end matches
    excluded) AND, when followed by an edge step, participates in that edge's PRUNED edges.
    Pruned ``steps`` (not forward frames) are essential — forward frames over-include and would
    tag nodes absent from the final graph."""
    import polars as pl
    node_col, src, dst = g._node, g._source, g._destination
    assert node_col is not None and src is not None and dst is not None
    step_list = list(steps)
    for idx, (op, g_step) in enumerate(step_list):
        if op._name is None or not isinstance(op, ASTNode) or g_step._nodes is None:
            continue
        if op._name not in colnames(g_step._nodes):
            continue
        named = g_step._nodes.filter(pl.col(op._name)).select(pl.col(node_col)).unique()
        if idx + 1 < len(step_list):
            next_op, next_step = step_list[idx + 1]
            if isinstance(next_op, ASTEdge) and next_step._edges is not None and (is_lazy(next_step._edges) or next_step._edges.height > 0):
                e = next_step._edges
                if next_op.direction == "forward":
                    part = e.select(pl.col(src).alias(node_col))
                elif next_op.direction == "reverse":
                    part = e.select(pl.col(dst).alias(node_col))
                else:
                    part = endpoint_ids(e, src, dst, node_col)
                named = named.join(part.unique(), on=node_col, how="semi")
        flag = named.with_columns(pl.lit(True).alias(op._name))
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

    calls = list(calls)
    if not calls:
        return g_cur

    if start_nodes is not None:
        setattr(g_cur, "_gfql_start_nodes", start_nodes)
    setattr(g_cur, "_gfql_rows_base_graph", base_graph)
    setattr(g_cur, "_gfql_shortest_path_backend", getattr(g_cur, "_gfql_shortest_path_backend", "auto"))

    if (
        middle
        and any(getattr(op, "_name", None) is not None for op in middle)
        and isinstance(calls[0], ASTCall)
        and calls[0].function == "rows"
        and calls[0].params.get("binding_ops") is None
        and calls[0].params.get("source") is None
        and calls[0].params.get("alias_endpoints") is None
        and all(isinstance(op, (_ASTNode, _ASTEdge)) for op in middle)
    ):
        calls = [rows_fn(binding_ops=serialize_binding_ops(middle))] + list(calls[1:])

    # Per-op NATIVE-OR-DEFER. Ops that don't lower:
    #  - non-native ROW op (correlated-subquery semi_apply/anti_semi_apply/join_apply):
    #    parity-or-NIE — it SHOULD be polars-native; a bridge would hide that.
    #  - ANALYTIC (compute_cugraph/compute_igraph/layout_*/collapse/get_topological_levels/...),
    #    no native impl by nature: route via execute_call, which applies the off-engine modality
    #    policy (call_mode='auto' bridges to pandas/cuDF + coerces back; 'strict' declines) — the
    #    SAME path as the DAG/let() surface, keeping surfaces consistent. Row-vs-analytic split
    #    is MECHANICAL (is_row_pipeline_call), not curated. (umap/hypergraph never reach here —
    #    the generic chain routes schema-changers straight to execute_call.)
    from graphistry.compute.gfql.row.pipeline import is_row_pipeline_call
    for op in calls:
        native = _try_native_row_op(g_cur, op)
        if native is not None:
            g_cur = native
            continue
        fn = getattr(op, "function", None)
        if fn is not None and not is_row_pipeline_call(fn):
            from graphistry.compute.gfql.call.executor import execute_call
            from graphistry.compute.gfql.lazy import active_target, ExecutionTarget
            from graphistry.Engine import Engine as _Engine
            _eng = _Engine.POLARS_GPU if active_target() == ExecutionTarget.GPU else _Engine.POLARS
            g_cur = execute_call(g_cur, fn, getattr(op, "params", {}) or {}, _eng)
            continue
        raise NotImplementedError(
            f"polars engine does not yet natively support cypher row op "
            f"{getattr(op, 'function', op)!r}; use engine='pandas' for this query "
            f"(no pandas fallback; parity-or-error by design)"
        )
    return g_cur


def _try_native_row_op(g_cur, op):
    """Run a row-pipeline call natively on polars, or return None to defer (NIE)."""
    from graphistry.Engine import Engine
    from .row_pipeline import (
        select_polars, with_columns_polars, order_by_polars, group_by_polars,
        unwind_polars, where_rows_polars,
    )
    from .pattern_apply import (
        rows_binding_ops_polars, semi_apply_mark_polars, anti_semi_apply_polars,
    )
    from .search import search_any_polars

    fn = getattr(op, "function", None)
    if _call_native_on_polars(op):
        # frame ops (rows/limit/skip/distinct/drop_cols) — engine-polymorphic
        return op.execute(g=g_cur, prev_node_wavefront=None, target_wave_front=None, engine=Engine.POLARS)
    # correlated pattern-existence family (EXISTS { } / pattern predicates): native
    # via chain_polars-computed key sets; unsupported shapes return None -> honest NIE.
    if (
        fn == "rows"
        and op.params.get("binding_ops") is not None
        and op.params.get("alias_endpoints") is None
        and op.params.get("source") is None
    ):
        return rows_binding_ops_polars(g_cur, op.params["binding_ops"])
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
    if fn == "order_by":
        return order_by_polars(g_cur, op.params.get("keys", []))
    if fn == "group_by":
        return group_by_polars(g_cur, op.params.get("keys", []), op.params.get("aggregations", []))
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
        # re-enter as a graph. pandas cascades via _chain_impl, but it's not a cypher shape
        # (MATCH comes first) and the polars traversal doesn't yet consume a row-table input.
        raise NotImplementedError(
            "polars chain engine does not yet support call() before a traversal; "
            "use engine='pandas' for this chain."
        )

    g_cur = _chain_traversal_polars(self, middle, start_nodes)
    if suffix:
        g_cur = _run_calls_polars(g_cur, suffix, start_nodes, base_graph=self, middle=middle)
    return g_cur


def _chain_traversal_polars(self: Plottable, ops, start_nodes: Optional[Any] = None) -> Plottable:
    import polars as pl
    from graphistry.compute.chain import Chain

    if isinstance(ops, Chain):
        ops = ops.chain
    ops = list(ops)

    if len(ops) == 0:
        return self

    # Node-only fast path: single MATCH (n) — the dominant tabular/viz/crossfilter shape.
    # Result is just the filtered node table + empty edges, so skip forward/backward/combine +
    # collect_all (~2.5 ms fixed cost at small sizes — the interactive crossfilter regime).
    # Byte-identical: the one-node-step combine yields filtered g._nodes in order + empty edges
    # + the alias flag on every matched node.
    if len(ops) == 1 and isinstance(ops[0], ASTNode) and ops[0].query is None:
        op0 = ops[0]
        g0 = ensure_nodes_polars(self)
        nc = g0._node
        assert nc is not None and g0._source is not None and g0._destination is not None
        nodes = filter_by_dict_polars(g0._nodes, op0.filter_dict)
        if start_nodes is not None:
            from graphistry.Engine import Engine as _E, df_to_engine as _d2e
            seed = _align_seed_dtype(_d2e(start_nodes, _E.POLARS), nc, g0._nodes)
            nodes = _semi(nodes, seed, nc, nc)
        if op0._name is not None:
            nodes = nodes.with_columns(pl.lit(True).alias(op0._name))
        return g0.nodes(nodes, nc).edges(g0._edges.clear(), g0._source, g0._destination)

    if isinstance(ops[0], ASTEdge):
        ops = [ASTNode()] + ops
    if isinstance(ops[-1], ASTEdge):
        ops = ops + [ASTNode()]

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

    # Single-hop fast path: [n(), e, n()] with no names/queries/matches — the basic
    # "filter then expand" crossfilter (`MATCH (a {f})-[e]->(b)`). Result = edges whose
    # endpoints pass the node filters + those endpoint nodes (isolated/dead-ends excluded);
    # one hop means the backward pass prunes nothing more, so skip forward/backward/combine.
    # Byte-identical vs pandas (verified: src/dst/both filters, reverse, dup/self-loop/cycle/
    # isolated). Undirected fast-paths only when UNCONSTRAINED; filtered-undirected (OR of
    # both directions) falls through to the full path.
    def _fp_node(op):
        return isinstance(op, ASTNode) and op._name is None and op.query is None

    def _plain_edge(op):
        return (isinstance(op, ASTEdge) and op.is_simple_single_hop()
                and op.edge_match is None and op.source_node_match is None
                and op.destination_node_match is None and op._name is None
                and op.source_node_query is None and op.destination_node_query is None
                and op.edge_query is None and not op.include_zero_hop_seed)

    # GFQL physical index fast path for the seeded single-hop shape
    # `MATCH (a {id-filter})-[e]->(b)` (forward/reverse, no destination filter) —
    # the canonical seeded query. This native chain fast path does its own O(E)
    # semi-join, so it must consult the index here too (not just compute/hop.py).
    _idx_pol = getattr(self, "_gfql_index_policy", "use")
    if (start_nodes is None and len(ops) == 3 and _fp_node(ops[0]) and _plain_edge(ops[1])
            and _fp_node(ops[2]) and ops[0].filter_dict and not ops[2].filter_dict
            and ops[1].direction in ("forward", "reverse")):
        from graphistry.compute.gfql.index import get_registry, maybe_index_hop
        if (not get_registry(self).is_empty()) or _idx_pol in ("auto", "force"):
            gf0 = ensure_nodes_polars(self)
            seed0 = filter_by_dict_polars(gf0._nodes, ops[0].filter_dict)
            from graphistry.Engine import Engine
            from graphistry.compute.gfql.lazy import active_target, ExecutionTarget
            _eng0 = Engine.POLARS_GPU if active_target() == ExecutionTarget.GPU else Engine.POLARS
            _idxed0 = maybe_index_hop(
                gf0, _eng0, nodes=seed0, hops=1, direction=ops[1].direction,
                return_as_wave_front=False, to_fixed_point=False, policy=_idx_pol,
            )
            if _idxed0 is not None:
                return _idxed0

    if start_nodes is None and len(ops) == 3 and _fp_node(ops[0]) and _plain_edge(ops[1]) and _fp_node(ops[2]):
        n0, e1, n2 = ops
        unconstrained = not n0.filter_dict and not n2.filter_dict
        if unconstrained or e1.direction in ("forward", "reverse"):
            gf = ensure_nodes_polars(self)
            ncol, scol, dcol = gf._node, gf._source, gf._destination
            assert ncol is not None and scol is not None and dcol is not None
            gf, restore = _align_edge_endpoints(gf, ncol, scol, dcol)
            edges = gf._edges
            if not unconstrained:
                from_col, to_col = (scol, dcol) if e1.direction == "forward" else (dcol, scol)
                if n0.filter_dict:
                    from_ids = filter_by_dict_polars(gf._nodes, n0.filter_dict).select(pl.col(ncol))
                    edges = edges.join(from_ids, left_on=from_col, right_on=ncol, how="semi")
                if n2.filter_dict:
                    to_ids = filter_by_dict_polars(gf._nodes, n2.filter_dict).select(pl.col(ncol))
                    edges = edges.join(to_ids, left_on=to_col, right_on=ncol, how="semi")
            endpoints = endpoint_ids(edges, scol, dcol, ncol)
            nodes = gf._nodes.join(endpoints.unique(), on=ncol, how="semi")
            return gf.nodes(nodes, ncol).edges(_restore_edge_dtypes(edges, scol, dcol, restore), scol, dcol)

    if start_nodes is not None:
        from graphistry.Engine import Engine, df_to_engine
        start_nodes = df_to_engine(start_nodes, Engine.POLARS)

    g = ensure_nodes_polars(self)
    assert g._node is not None and g._source is not None and g._destination is not None
    start_nodes = _align_seed_dtype(start_nodes, g._node, g._nodes)
    g, _endpoint_restore = _align_edge_endpoints(g, g._node, g._source, g._destination)
    if g._edge is None:
        EID = "__gfql_edge_index__"
        g = g.edges(g._edges.with_row_index(EID), g._source, g._destination, edge=EID)
        added_edge_index = True
    else:
        EID = g._edge
        added_edge_index = False

    node_col, src, dst = g._node, g._source, g._destination
    assert node_col is not None and src is not None and dst is not None

    # Forward pass.
    g_stack: List[Plottable] = []
    for i, op in enumerate(ops):
        prev = start_nodes if i == 0 else g_stack[-1]._nodes
        g_stack.append(_exec(op, g, prev, None))

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
        g_step_full = g_step.nodes(g._nodes, g._node)
        rev = _exec(op.reverse(), g_step_full, prev_wf, target_wf, intermediate_universe=_iu)
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
                g_sub = g.edges(g_step._edges, src, dst, edge=g._edge)
                edge_steps.append((op, _exec(op, g_sub, prev_wf, None)))
            else:
                edge_steps.append((op, g_step))

    # Track B: build the WHOLE combine (combine_nodes/edges + endpoint + names) as ONE deferred
    # plan over already-materialized hop frames and collect ONCE — collapsing the ~dozen eager
    # combine ops (each internally lazy().op().collect()) into a single fused pass on the active
    # target. NO recompute (inputs materialized; distinct from the disproven full-chain fusion).
    # Stable order columns restore the eager g._nodes/g._edges order (lazy joins don't preserve
    # it) so a trailing row pipeline's LIMIT/SKIP is unaffected — byte-identical to eager combine.
    from graphistry.compute.util import generate_safe_column_name
    from graphistry.compute.gfql.lazy import collect_all
    NORD = generate_safe_column_name("__gfql_norder__", g._nodes, prefix="__gfql_", suffix="__")
    EORD = generate_safe_column_name("__gfql_eorder__", g._edges, prefix="__gfql_", suffix="__")
    g_lz = _LazyShim(g._nodes.with_row_index(NORD).lazy(), g._edges.with_row_index(EORD).lazy(),
                     node_col, src, dst, g._edge)
    steps_lz = [(op, _LazyShim.step(p)) for op, p in steps]
    edge_steps_lz = [(op, _LazyShim.step(p)) for op, p in edge_steps]
    label_lz = [(op, _LazyShim.step(p)) for op, p in label_steps]

    final_nodes = _combine_nodes(g_lz, steps_lz)
    final_edges = _combine_edges(g_lz, edge_steps_lz, label_lz, has_multihop)
    # Endpoint (lazy: always compute; maintain_order keeps the semi-join order).
    endpoints = endpoint_ids(final_edges, src, dst, node_col).unique(subset=[node_col])
    missing = endpoints.join(final_nodes.select(pl.col(node_col)), on=node_col, how="anti")
    extra = g_lz._nodes.join(missing, on=node_col, how="semi")
    final_nodes = pl.concat([final_nodes, extra], how="diagonal_relaxed").unique(
        subset=[node_col], maintain_order=True)
    final_nodes = _apply_node_names(final_nodes, g_lz, steps_lz)

    final_nodes = final_nodes.sort(NORD).drop(NORD)
    final_edges = final_edges.sort(EORD).drop(EORD)
    if added_edge_index:
        final_edges = final_edges.drop(EID)
    final_edges, final_nodes = collect_all([final_edges, final_nodes])
    final_edges = _restore_edge_dtypes(final_edges, src, dst, _endpoint_restore)
    return self.nodes(final_nodes, node_col).edges(final_edges, src, dst)
