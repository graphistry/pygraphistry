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
from .hop import hop_polars
from .predicates import filter_by_dict_polars


def _semi(df, ids_df, df_col, id_col):
    """Rows of df whose df_col is present in ids_df[id_col] (vectorized semi-join)."""
    return df.join(ids_df.select(id_col).unique(), left_on=df_col, right_on=id_col, how="semi")


def _exec(op: ASTObject, g: Plottable, prev_wf, target_wf) -> Plottable:
    import polars as pl

    node_col = g._node
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
        g_step = hop_polars(
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


def _apply_node_names(out, g, label_steps):
    """Tag node aliases on the FINAL node frame (after endpoint materialization).

    A node carries the alias if it matched the named node step AND (when that
    step is followed by an edge step) participates in that edge. Join-based.
    """
    import polars as pl
    node_col, src, dst = g._node, g._source, g._destination
    label_list = list(label_steps)
    for idx, (op, g_step) in enumerate(label_list):
        if op._name is None or not isinstance(op, ASTNode) or g_step._nodes is None:
            continue
        if op._name not in g_step._nodes.columns:
            continue
        named = g_step._nodes.filter(pl.col(op._name)).select(pl.col(node_col)).unique()
        if idx + 1 < len(label_list):
            next_op, next_step = label_list[idx + 1]
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


def chain_polars(self: Plottable, ops, start_nodes: Optional[Any] = None) -> Plottable:
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

    g = self.materialize_nodes(engine="polars")
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

    final_nodes = _combine_nodes(g, steps)
    final_edges = _combine_edges(g, steps, label_steps)

    # Endpoint materialization (vectorized: anti-join missing endpoints).
    if final_edges.height > 0:
        endpoints = pl.concat(
            [final_edges.select(pl.col(g._source).alias(g._node)),
             final_edges.select(pl.col(g._destination).alias(g._node))],
            how="vertical_relaxed",
        ).unique(subset=[g._node])
        missing = endpoints.join(final_nodes.select(pl.col(g._node)), on=g._node, how="anti")
        if missing.height > 0:
            extra = g._nodes.join(missing, on=g._node, how="semi")
            final_nodes = pl.concat([final_nodes, extra], how="diagonal_relaxed").unique(subset=[g._node])

    final_nodes = _apply_node_names(final_nodes, g, label_steps)

    if added_edge_index:
        final_edges = final_edges.drop(EID)
        return self.nodes(final_nodes, g._node).edges(final_edges, g._source, g._destination)
    return g.nodes(final_nodes, g._node).edges(final_edges, g._source, g._destination)
