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
from .hop import hop_polars, ensure_nodes_polars
from .predicates import filter_by_dict_polars


def _semi(df, ids_df, df_col, id_col):
    """Rows of df whose df_col is present in ids_df[id_col] (vectorized semi-join)."""
    return df.join(ids_df.select(id_col).unique(), left_on=df_col, right_on=id_col, how="semi")


def _exec(op: ASTObject, g: Plottable, prev_wf, target_wf) -> Plottable:
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
        )
        if op._name is not None:
            g_step = g_step.edges(
                g_step._edges.with_columns(pl.lit(True).alias(op._name)),
                g._source, g._destination,
            )
        return g_step

    raise NotImplementedError(f"polars chain engine does not support op {type(op).__name__}")


def _combine_edges(g, steps, label_steps):
    import polars as pl
    src, dst, node_col, edge_id = g._source, g._destination, g._node, g._edge
    assert src is not None and dst is not None and node_col is not None and edge_id is not None

    frames = []
    for idx, (op, g_step) in enumerate(steps):
        edges_df = g_step._edges
        if edges_df is None or edges_df.height == 0:
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
        out_ids = g._edges.select(pl.col(edge_id)).clear()
    else:
        out_ids = pl.concat(frames, how="vertical_relaxed").unique(subset=[edge_id])

    out = g._edges.join(out_ids, on=edge_id, how="semi")

    for op, g_step in label_steps:
        if op._name is not None and isinstance(op, ASTEdge) and g_step._edges is not None and op._name in g_step._edges.columns:
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
        if g_step._nodes is not None and node_col in g_step._nodes.columns
    ]
    if frames:
        ids = pl.concat(frames, how="vertical_relaxed").unique(subset=[node_col])
    else:
        ids = g._nodes.select(pl.col(node_col)).clear()
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
        if op._name not in g_step._nodes.columns:
            continue
        named = g_step._nodes.filter(pl.col(op._name)).select(pl.col(node_col)).unique()
        if idx + 1 < len(step_list):
            next_op, next_step = step_list[idx + 1]
            if isinstance(next_op, ASTEdge) and next_step._edges is not None and next_step._edges.height > 0:
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
    from graphistry.Engine import Engine
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


def _try_native_row_op(g_cur, op):
    """Run a row-pipeline call natively on polars, or return None to defer (NIE)."""
    from graphistry.Engine import Engine
    from .row_pipeline import select_polars, order_by_polars, group_by_polars, unwind_polars, where_rows_polars

    fn = getattr(op, "function", None)
    if _call_native_on_polars(op):
        # frame ops (rows/limit/skip/distinct/drop_cols) — engine-polymorphic
        return op.execute(g=g_cur, prev_node_wavefront=None, target_wave_front=None, engine=Engine.POLARS)
    if fn in ("select", "return_"):
        return select_polars(g_cur, op.params.get("items", []))
    if fn == "with_" and not op.params.get("extend", False):
        return select_polars(g_cur, op.params.get("items", []))
    if fn == "where_rows":
        return where_rows_polars(g_cur, op.params.get("filter_dict"), op.params.get("expr"))
    if fn == "order_by":
        return order_by_polars(g_cur, op.params.get("keys", []))
    if fn == "group_by":
        return group_by_polars(g_cur, op.params.get("keys", []), op.params.get("aggregations", []))
    if fn == "unwind":
        return unwind_polars(g_cur, op.params.get("expr", ""), op.params.get("as_", "value"))
    return None


def chain_polars(self: Plottable, ops, start_nodes: Optional[Any] = None) -> Plottable:
    from graphistry.compute.ast import ASTCall
    from graphistry.compute.chain import Chain, _get_boundary_calls

    if isinstance(ops, Chain):
        ops = ops.chain
    ops = list(ops)

    if len(ops) == 0:
        return self

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

    if isinstance(ops[0], ASTEdge):
        ops = [ASTNode()] + ops
    if isinstance(ops[-1], ASTEdge):
        ops = ops + [ASTNode()]

    if any(isinstance(op, ASTEdge) and not op.is_simple_single_hop() for op in ops):
        raise NotImplementedError(
            "polars chain engine (Phase 1) supports single-hop edges only; "
            "variable-length/multi-hop chains are deferred. Use engine='pandas'."
        )

    edge_ops = [op for op in ops if isinstance(op, ASTEdge)]
    if len(edge_ops) > 1 and any(op.direction == "undirected" for op in edge_ops):
        raise NotImplementedError(
            "polars chain engine (Phase 1) does not yet support undirected edges "
            "in multi-edge chains; deferred. Use engine='pandas'."
        )

    if start_nodes is not None:
        from graphistry.Engine import Engine, df_to_engine
        start_nodes = df_to_engine(start_nodes, Engine.POLARS)

    g = ensure_nodes_polars(self)
    assert g._node is not None and g._source is not None and g._destination is not None
    if g._edge is None:
        EID = "__gfql_edge_index__"
        g = g.edges(g._edges.with_row_index(EID), g._source, g._destination, edge=EID)
        added_edge_index = True
    else:
        EID = g._edge
        added_edge_index = False

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
        # Give the reverse hop the FULL node universe (g_step._nodes is only the
        # forward wavefront; filtering reached ids against it would truncate the
        # reverse wavefront and break threading).
        g_step_full = g_step.nodes(g._nodes, g._node)
        g_rev.append(_exec(op.reverse(), g_step_full, prev_wf, target_wf))

    steps: List[Tuple[ASTObject, Plottable]] = list(zip(ops, list(reversed(g_rev))))
    label_steps: List[Tuple[ASTObject, Plottable]] = list(zip(ops, g_stack))

    node_col, src, dst = g._node, g._source, g._destination
    assert node_col is not None and src is not None and dst is not None

    final_nodes = _combine_nodes(g, steps)
    final_edges = _combine_edges(g, steps, label_steps)

    # Endpoint materialization (vectorized: anti-join missing endpoints).
    if final_edges.height > 0:
        endpoints = pl.concat(
            [final_edges.select(pl.col(src).alias(node_col)),
             final_edges.select(pl.col(dst).alias(node_col))],
            how="vertical_relaxed",
        ).unique(subset=[node_col])
        missing = endpoints.join(final_nodes.select(pl.col(node_col)), on=node_col, how="anti")
        if missing.height > 0:
            extra = g._nodes.join(missing, on=node_col, how="semi")
            final_nodes = pl.concat([final_nodes, extra], how="diagonal_relaxed").unique(subset=[node_col])

    final_nodes = _apply_node_names(final_nodes, g, steps)

    if added_edge_index:
        final_edges = final_edges.drop(EID)
        return self.nodes(final_nodes, node_col).edges(final_edges, src, dst)
    return g.nodes(final_nodes, node_col).edges(final_edges, src, dst)
