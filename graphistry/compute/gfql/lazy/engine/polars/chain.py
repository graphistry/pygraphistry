"""Native Polars chain() — Phase 1, vectorized.

Reimplements the chain forward/backward/combine orchestration in polars,
reusing the polars hop for edge steps. Vectorization-first: node/edge set
operations are semi/anti joins, alias tags are join-based flag columns — no
Python-level id lists or ``is_in(python_list)``. Correctness gated by
differential parity vs the pandas chain.

Deferred (explicit NotImplementedError): variable-length/multi-hop edges,
undirected edges in multi-edge chains, node query=.
"""
from typing import Any, List, Optional, Tuple

from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTObject, ASTNode, ASTEdge
from .hop_eager import ensure_nodes_polars
from .predicates import filter_by_dict_polars


def _semi(df, ids_df, df_col, id_col):
    """Rows of df whose df_col is present in ids_df[id_col] (vectorized semi-join)."""
    return df.join(ids_df.select(id_col).unique(), left_on=df_col, right_on=id_col, how="semi")


def _align_seed_dtype(seed, node_col, ref_nodes):
    """Cast the seed's node-id column to the node table's id dtype so the seed↔node
    semi-join keys match. A ``start_nodes`` frame can arrive with a divergent id dtype
    (e.g. an empty crossfilter selection defaults to float64 while node ids are int64),
    which polars won't auto-cast — the combine semi-join then raises ``SchemaError``
    where pandas joins fine. No-op when the column is absent or already matches."""
    import polars as pl
    if seed is None or node_col not in seed.columns:
        return seed
    ndt = ref_nodes.schema[node_col]
    if seed.schema[node_col] == ndt:
        return seed
    return seed.with_columns(pl.col(node_col).cast(ndt))


def _align_edge_endpoints(g, node_col, src, dst):
    """Cast edge endpoint columns to the node-id dtype so polars join keys match.

    polars won't auto-cast int↔float join keys, and a null in an edge endpoint
    promotes that column to float while integer node ids stay int — which makes the
    chain's endpoint↔node-id joins raise ``SchemaError`` where pandas joins fine. The
    hop already casts internally; the chain (fast paths + combine) did not. Returns
    ``(aligned_g, restore)`` where ``restore`` is the original ``(src_dtype, dst_dtype)``
    to put back on the OUTPUT edges (so the result matches pandas' dtype), or None when
    the dtypes already matched (the common case — a no-op, no table copy)."""
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
            # The chain wants pandas' LABELED min_hops node policy (null attrs for source-side
            # endpoints) — mirrors pandas' ASTEdge.execute passing auto hop-labels. A direct
            # base.hop(engine='polars', min_hops=...) leaves this False -> full attrs (pandas parity).
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
    """Whether a non-simple-single-hop edge op is a multi-hop the native polars
    combine supports: ``hops=N`` / ``max_hops`` (fwd/rev/undirected),
    ``to_fixed_point`` (fwd/rev), and ``min_hops>1`` (fwd/rev, finite max),
    optionally with edge/endpoint match or a name. Deferred combos — undirected
    ``to_fixed_point`` / undirected ``min_hops>1``, output slicing, hop labels,
    ``*_query`` strings, ``include_zero_hop_seed``, ``prune_to_endpoints`` —
    return False so the guard defers (NotImplementedError) rather than risk a
    silent divergence (see the inline comments for why each stays deferred)."""
    if not isinstance(op, ASTEdge):
        return False
    if op.is_simple_single_hop():
        return False
    if op.direction not in ("forward", "reverse", "undirected"):
        return False
    # to_fixed_point (unbounded variable-length) IS native for forward/reverse: the recompute
    # re-runs the forward to_fixed_point hop over the backward-pruned subgraph (the hop's own
    # fixed-point detection guarantees termination), same path-aware combine as hops=N. But
    # UNDIRECTED to_fixed_point needs pandas' connected-components + 2-core seed retention
    # (graphistry/compute/hop.py:817-887) which the vectorized polars hop cannot reproduce -> NIE.
    if op.direction == "undirected" and op.to_fixed_point:
        return False
    # min_hops>1 (variable-length LOWER bound) IS native for forward/reverse with a finite
    # max_hops (the hop runs a NON-anti-joined BFS + layered backward-tree walk). UNDIRECTED
    # min_hops>1 and min_hops+to_fixed_point stay NIE.
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


def _is_lazy(df) -> bool:
    import polars as pl
    return isinstance(df, pl.LazyFrame)


def _colnames(df):
    return df.collect_schema().names() if _is_lazy(df) else df.columns


class _LazyShim:
    """Minimal lazy graph/step shim for the Track B collect-once combine: carries
    ``_nodes``/``_edges`` as LazyFrames (+ col names) so the eager combine helpers
    run lazily over the already-materialized hop frames without Plottable rebinds."""
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
        if not _is_lazy(edges_df) and edges_df.height == 0:
            continue
        if has_multihop or (isinstance(op, ASTEdge) and not op.is_simple_single_hop()):
            # has_multihop: EVERY edge step was recomputed path-valid over its backward-pruned
            # subgraph (forward re-exec; see the chain recompute), so append ALL ids and skip the
            # single-hop prev/next endpoint semijoin below — that semijoin keeps only edges whose
            # src/dst sit in the prev/next wavefront and would drop intermediate-hop edges (and,
            # on mixed single+multi chains, diverged by an edge/node vs pandas). Port of the pandas
            # combine_steps has_multihop branch (recompute all steps, then concat as-is, no semijoin).
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
        if op._name is not None and isinstance(op, ASTEdge) and g_step._edges is not None and op._name in _colnames(g_step._edges):
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
        if g_step._nodes is not None and node_col in _colnames(g_step._nodes)
    ]
    if frames:
        ids = pl.concat(frames, how="vertical_relaxed").unique(subset=[node_col])
    else:
        ids = g._nodes.select(pl.col(node_col)).limit(0)
    return g._nodes.join(ids, on=node_col, how="semi")


def _apply_node_names(out, g, steps):
    """Tag node aliases on the FINAL node frame (after endpoint materialization).

    A node carries the alias if it matched the named node step (in the
    backward-PRUNED step, so dead-end matches are excluded) AND, when that step
    is followed by an edge step, participates in that edge's PRUNED edges.
    Using the pruned ``steps`` (not the forward-pass frames) is essential — the
    forward frames over-include and would tag nodes absent from the final graph.
    """
    import polars as pl
    node_col, src, dst = g._node, g._source, g._destination
    assert node_col is not None and src is not None and dst is not None
    step_list = list(steps)
    for idx, (op, g_step) in enumerate(step_list):
        if op._name is None or not isinstance(op, ASTNode) or g_step._nodes is None:
            continue
        if op._name not in _colnames(g_step._nodes):
            continue
        named = g_step._nodes.filter(pl.col(op._name)).select(pl.col(node_col)).unique()
        if idx + 1 < len(step_list):
            next_op, next_step = step_list[idx + 1]
            if isinstance(next_op, ASTEdge) and next_step._edges is not None and (_is_lazy(next_step._edges) or next_step._edges.height > 0):
                e = next_step._edges
                if next_op.direction == "forward":
                    part = e.select(pl.col(src).alias(node_col))
                elif next_op.direction == "reverse":
                    part = e.select(pl.col(dst).alias(node_col))
                else:
                    part = pl.concat(
                        [e.select(pl.col(src).alias(node_col)), e.select(pl.col(dst).alias(node_col))],
                        how="vertical_relaxed",
                    )
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

    Mirrors the suffix/prefix handling in ``chain._handle_boundary_calls``:
    threads the row-pipeline context attrs and applies the named-middle →
    ``rows(binding_ops=...)`` rewrite. Each call runs natively on
    ``Engine.POLARS`` via ``_try_native_row_op``; an op with no native polars
    implementation raises ``NotImplementedError`` (NO pandas fallback — see
    plan.md NO-CHEATING) rather than secretly running the pandas row pipeline.
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

    # Per-op NATIVE-OR-DEFER: run each call natively on polars; an op we can't
    # lower natively raises NotImplementedError (NO pandas fallback — see plan.md
    # NO-CHEATING). The honest signal tells the caller to use engine='pandas'.
    for op in calls:
        native = _try_native_row_op(g_cur, op)
        if native is None:
            raise NotImplementedError(
                f"polars engine does not yet natively support cypher row op "
                f"{getattr(op, 'function', op)!r}; use engine='pandas' for this query "
                f"(no pandas fallback — see plans/gfql-polars-engine NO-CHEATING)"
            )
        g_cur = native
    return g_cur


def get_degrees_polars(
    g: Plottable,
    col: str = "degree",
    degree_in: str = "degree_in",
    degree_out: str = "degree_out",
    engine: Optional[str] = None,
) -> Plottable:
    """Native polars ``get_degrees`` — parity with ComputeMixin.get_degrees.

    Per-node in/out degree via a group_by-count on the edge endpoint columns,
    left-joined onto the node table (materialized from edges when absent). Pure
    polars — NO pandas bridge (see plan.md NO-CHEATING). Parity contract vs the
    pandas oracle: isolated nodes and src-only / dst-only nodes get 0; self-loops
    are double-counted (one in + one out); all three columns are Int32. Empty
    edges need no special case — the left-join + fill_null(0) yields all-zero
    degrees, matching the pandas empty-edges branch.

    The ``engine`` safelist sub-param is accepted and ignored: execution is already
    committed to polars and the result stays parity-equal to the pandas oracle.
    """
    import polars as pl

    g = ensure_nodes_polars(g)
    node_col, src, dst = g._node, g._source, g._destination
    assert node_col is not None and src is not None and dst is not None
    assert g._nodes is not None and g._edges is not None
    nodes, edges = g._nodes, g._edges

    # Align the count keys to the node-id dtype so the left-join keys match even when
    # a user node table's id dtype diverges from the edge endpoint dtype (polars will
    # not auto-cast int<->float join keys, where pandas merges fine).
    node_dt = (nodes.collect_schema() if _is_lazy(nodes) else nodes.schema)[node_col]

    in_counts = edges.group_by(pl.col(dst).cast(node_dt).alias(node_col)).agg(
        pl.len().alias(degree_in)
    )
    out_counts = edges.group_by(pl.col(src).cast(node_dt).alias(node_col)).agg(
        pl.len().alias(degree_out)
    )

    # Drop any pre-existing degree columns, mirroring the pandas keep-subset, so a
    # re-run overwrites rather than producing ``*_right`` join collisions.
    drop_cols = [c for c in _colnames(nodes) if c in (degree_in, degree_out, col)]
    base = nodes.drop(drop_cols) if drop_cols else nodes

    out = (
        base.join(in_counts, on=node_col, how="left")
        .join(out_counts, on=node_col, how="left")
        .with_columns(
            pl.col(degree_in).fill_null(0).cast(pl.Int32),
            pl.col(degree_out).fill_null(0).cast(pl.Int32),
        )
        .with_columns(
            (pl.col(degree_in) + pl.col(degree_out)).cast(pl.Int32).alias(col)
        )
    )
    return g.nodes(out, node_col)


def _single_direction_degree_polars(g: Plottable, key_col: str, col: str) -> Plottable:
    """Native shared body for get_indegrees / get_outdegrees — parity with
    ComputeMixin._single_direction_degree. A group_by-count on ONE edge endpoint
    (``destination`` for in-degree, ``source`` for out-degree), left-joined onto the
    node table. Isolated / opposite-endpoint-only nodes get 0; a self-loop counts once
    for the relevant direction (single-direction is NOT double-counted, unlike the
    get_degrees total); the degree column is Int32. The count key is cast to the node-id
    dtype so the join keys match even when the edge endpoint dtype diverges (polars won't
    auto-cast int<->float join keys where pandas merges fine).

    The empty-edges case mirrors the pandas oracle EXACTLY, which differs from
    get_degrees: when there are no edges AND the target column already exists, the node
    table is returned UNCHANGED (pandas keeps the pre-existing values); otherwise the
    column is materialized as all-zero Int32. Pure polars — NO pandas bridge (see
    NO-CHEATING).
    """
    import polars as pl

    g = ensure_nodes_polars(g)
    node_col = g._node
    assert node_col is not None
    assert g._nodes is not None and g._edges is not None
    nodes, edges = g._nodes, g._edges

    if _is_lazy(edges):
        edges_empty = edges.select(pl.len()).collect().item() == 0
    else:
        edges_empty = edges.height == 0
    if edges_empty:
        if col in _colnames(nodes):
            return g.nodes(nodes, node_col)
        return g.nodes(nodes.with_columns(pl.lit(0).cast(pl.Int32).alias(col)), node_col)

    node_dt = (nodes.collect_schema() if _is_lazy(nodes) else nodes.schema)[node_col]
    counts = edges.group_by(pl.col(key_col).cast(node_dt).alias(node_col)).agg(
        pl.len().alias(col)
    )
    # Drop a pre-existing degree column so a re-run overwrites rather than producing a
    # ``*_right`` join collision (mirrors the pandas keep-subset).
    base = nodes.drop(col) if col in _colnames(nodes) else nodes
    out = base.join(counts, on=node_col, how="left").with_columns(
        pl.col(col).fill_null(0).cast(pl.Int32)
    )
    return g.nodes(out, node_col)


def get_indegrees_polars(g: Plottable, col: str = "degree_in") -> Plottable:
    """Native polars ``get_indegrees`` — parity with ComputeMixin.get_indegrees.

    In-degree = count of edges whose DESTINATION endpoint is the node. Pure polars —
    NO pandas bridge (see NO-CHEATING).
    """
    assert g._destination is not None, "Missing destination binding; set via .bind() or .edges()"
    return _single_direction_degree_polars(g, g._destination, col)


def get_outdegrees_polars(g: Plottable, col: str = "degree_out") -> Plottable:
    """Native polars ``get_outdegrees`` — parity with ComputeMixin.get_outdegrees.

    Out-degree = count of edges whose SOURCE endpoint is the node. Pure polars —
    NO pandas bridge (see NO-CHEATING).
    """
    assert g._source is not None, "Missing source binding; set via .bind() or .edges()"
    return _single_direction_degree_polars(g, g._source, col)


def _try_native_row_op(g_cur, op):
    """Run a row-pipeline call natively on polars, or return None to defer (NIE)."""
    from graphistry.Engine import Engine
    from .row_pipeline import select_polars, with_columns_polars, order_by_polars, group_by_polars, unwind_polars, where_rows_polars

    fn = getattr(op, "function", None)
    if _call_native_on_polars(op):
        # frame ops (rows/limit/skip/distinct/drop_cols) — engine-polymorphic
        return op.execute(g=g_cur, prev_node_wavefront=None, target_wave_front=None, engine=Engine.POLARS)
    if fn in ("select", "return_"):
        return select_polars(g_cur, op.params.get("items", []))
    if fn == "with_":
        # extend=True (WITH ... that KEEPS existing columns) -> with_columns; extend=False
        # (full re-projection) -> select. Both decline (NIE) on an unlowerable item.
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

    # Reject duplicate alias names (node aliases and edge aliases scoped separately,
    # mirroring the pandas ``combine_steps`` E201 guard). Without this, a reused name
    # like ``[n(name='a'), e(), n(name='a')]`` produces a malformed schema (colliding
    # ``a``/``a_right`` join columns) — a wrong answer where pandas declines.
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
        # Leading call() ops produce a row table that a following traversal would
        # have to re-enter as a graph; the pandas path handles this via cascading
        # _chain_impl, but it is not a cypher shape (MATCH always comes first) and
        # the polars traversal does not yet consume a row-table input. Defer.
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

    # Node-only fast path: a single MATCH (n) (no edge traversal) — the dominant
    # tabular/viz/crossfilter shape (MATCH (n) WHERE/RETURN ...). The result is
    # just the filtered node table + empty edges, so skip the whole
    # forward/backward/combine + collect_all (a ~2.5 ms fixed cost at small sizes
    # — exactly the interactive crossfilter regime). Byte-identical: the combine
    # for one node step yields g._nodes (filtered) in order + empty edges + the
    # alias flag on every matched node.
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
    # Undirected edges in multi-edge chains: NATIVE for single-hop AND fixed-length multi-hop
    # undirected (the backward pass threads BOTH endpoints for single-hop — see the override
    # below; fixed multi-hop undirected uses the generic backward hop + the path-aware
    # recompute, same as directed). STILL deferred (NIE): undirected single-hop/multi-hop
    # carrying include_zero_hop_seed / *_query (unverified combine semantics). Undirected
    # to_fixed_point is NIE'd upstream by _is_native_multihop (needs components/2-core).
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

    # Single-hop fast path: [n(), e, n()] where both nodes have no name/query and
    # the edge has no match/name/query — the basic graph query + "filter then
    # expand" viz crossfilter (`MATCH (a {f})-[e]->(b)`). The single-hop result is
    # exactly the edges whose endpoints pass the node filters + those endpoint
    # nodes (isolated/dead-end nodes excluded). For one hop the backward pass
    # prunes nothing beyond this, so we skip forward/backward/combine entirely.
    # Byte-identical (verified vs pandas incl src/dst/both filters, reverse,
    # dup/self-loop/cycle/isolated). Undirected is only fast-pathed when
    # UNCONSTRAINED (all edges); filtered-undirected (OR of both directions) falls
    # through to the full path.
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
            endpoints = pl.concat(
                [edges.select(pl.col(scol).alias(ncol)), edges.select(pl.col(dcol).alias(ncol))],
                how="vertical_relaxed",
            )
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
        # OUTPUT universe = FULL g._nodes (so the reverse hop's output isn't truncated / loses node
        # columns). The INTERMEDIATE-hop gate universe is decoupled: for a MULTI-HOP reverse step pass
        # the forward wavefront (g_step._nodes, before widening) so intermediate hops can't wander
        # outside the forward-reached nodes (pandas chain.py:1104 feeds g_step as the multi-hop reverse
        # node table). Single-hop reverse: None -> gate universe = all_nodes (vacuous), matching the
        # pandas use_fast_backward path (full g._nodes).
        _iu = g_step._nodes if (isinstance(op, ASTEdge) and not op.is_simple_single_hop()) else None
        g_step_full = g_step.nodes(g._nodes, g._node)
        rev = _exec(op.reverse(), g_step_full, prev_wf, target_wf, intermediate_universe=_iu)
        # Undirected single-hop backward threading: the generic hop returns a
        # ONE-SIDED wavefront (the TO/target-side endpoints only). Pandas' fast
        # backward branch (graphistry/compute/chain.py:1090-1098) instead threads
        # BOTH endpoints of the surviving edges as the step's node frame. For an
        # undirected edge the one-sided wavefront drops an intermediate node that is
        # only reachable as the frontier-side endpoint of a sibling edge, which then
        # drops that node's incident edge in the combine (repo fuzz seed-18: >1
        # undirected edge + intermediate node filters loses a node vs pandas). The
        # edge set already matches pandas (both edge orientations are joined), so we
        # override ONLY the node frame, carrying FULL node columns so the next
        # backward node step can still filter on them.
        if (isinstance(op, ASTEdge) and op.direction == "undirected"
                and op.is_simple_single_hop() and rev._edges is not None):
            _both = pl.concat(
                [rev._edges.select(pl.col(src).alias(node_col)),
                 rev._edges.select(pl.col(dst).alias(node_col))],
                how="vertical_relaxed",
            ).unique(subset=[node_col])
            rev = rev.nodes(g._nodes.join(_both, on=node_col, how="semi"), node_col)
        g_rev.append(rev)

    steps: List[Tuple[ASTObject, Plottable]] = list(zip(ops, list(reversed(g_rev))))
    label_steps: List[Tuple[ASTObject, Plottable]] = list(zip(ops, g_stack))

    # Native multi-hop edges: port of the pandas combine_steps ``has_multihop`` branch
    # (graphistry/compute/chain.py:201-209). For a fixed-length multi-hop edge step the
    # per-step single-hop endpoint semijoin in _combine_edges keeps only edges whose
    # src/dst sit in the prev/next wavefront, dropping intermediate-hop edges (observed
    # e=3 vs pandas e=10). Instead recompute the path-valid edge set by RE-EXECUTING the
    # FORWARD hop over this step's backward-pruned edge subgraph, seeded from the
    # forward-pass entry wavefront: the backward prune enforces "reaches a valid
    # endpoint", the forward re-exec enforces "reachable from the seed", and their
    # composition along the BFS frontier is exactly the valid-path edges (NOT a flat
    # forward∩backward set intersection — that was the reverted shortcut). Only the EDGE
    # combine sees these recomputed frames (pandas gates the recompute on kind=='edges');
    # the node combine + name tagging keep the original backward-pruned steps.
    edge_steps: List[Tuple[ASTObject, Plottable]] = steps
    has_multihop = any(isinstance(op, ASTEdge) and not op.is_simple_single_hop() for op, _ in steps)
    if has_multihop:
        # pandas combine_steps recomputes EVERY step (not only the multi-hop one) when any step is
        # multi-hop (chain.py:201-209), and the edge combine then concatenates each recomputed
        # step's edges with NO per-step prev/next semijoin. Mirror that exactly: recompute ALL
        # ASTEdge steps (forward re-exec over their backward-pruned subgraph, seeded from the
        # forward-pass entry wavefront) and let _combine_edges append-all (has_multihop=True).
        # Recomputing only the multi-hop step + semijoining the single-hop steps diverged by an
        # edge/node on chains mixing single + multi-hop (fuzz seeds 24/48).
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

    # Track B: build the WHOLE combine (combine_nodes/edges + endpoint + names)
    # as ONE deferred plan over the already-materialized hop frames and collect
    # ONCE — collapsing the ~dozen eager combine ops (each internally a
    # lazy().op().collect()) into a single fused pass on the active target. NO
    # recompute (inputs are materialized; distinct from the disproven full chain
    # fusion). Stable order columns restore the eager g._nodes / g._edges order
    # (lazy joins don't preserve it) so a trailing row pipeline's LIMIT/SKIP is
    # unaffected — byte-identical to the eager combine.
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
    endpoints = pl.concat(
        [final_edges.select(pl.col(src).alias(node_col)),
         final_edges.select(pl.col(dst).alias(node_col))],
        how="vertical_relaxed",
    ).unique(subset=[node_col])
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
