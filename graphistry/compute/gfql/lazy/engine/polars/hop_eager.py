"""Native Polars hop() — Phase 1, vectorized.

Supports forward/reverse/undirected, integer hops + to_fixed_point, default and
return_as_wave_front seed semantics, edge/source/destination match predicates, and
target_wave_front (chain reverse pass). Vectorization-first: BFS frontier/visited/allowed sets
are polars frames advanced by semi/anti joins — no per-element Python work, no
``.to_list()``/``is_in(python_list)`` ping-pong; each hop is one big join. Parity-or-NIE:
pandas is the oracle; not-yet-ported features (hop labeling, min_hops>1 outside the chain
policy, output_min/max slicing, *_query, prune_to_endpoints, include_zero_hop_seed) raise
NotImplementedError.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from graphistry.Plottable import Plottable
from graphistry.compute.util import generate_safe_column_name
from .dtypes import endpoint_ids
from .predicates import filter_by_dict_polars

if TYPE_CHECKING:
    import polars as pl
    from .dtypes import PolarsT


def _unsupported(**kwargs: Any) -> None:
    unsupported = [k for k, v in kwargs.items() if v is not None and v is not False]
    if unsupported:
        raise NotImplementedError(
            "polars hop engine (Phase 1) does not yet support: "
            + ", ".join(sorted(unsupported))
            + ". Use engine='pandas' or extend graphistry/compute/gfql/lazy/engine/polars/hop_eager.py."
        )


def ensure_nodes_polars(g: Plottable) -> Plottable:
    """Materialize a polars node table from edges when absent (native — avoids the
    pandas-idiom ``materialize_nodes`` path, which uses drop_duplicates/reset_index)."""
    import polars as pl

    if g._nodes is not None:
        return g
    src, dst = g._source, g._destination
    assert src is not None and dst is not None and g._edges is not None
    node_id = g._node if g._node is not None else "id"
    ids = endpoint_ids(g._edges, src, dst, node_id).unique()
    return g.nodes(ids, node_id)


def _hop_setup_columns(
    edges: "pl.DataFrame", all_nodes: "pl.DataFrame", node_col: str, edge_binding: Optional[str],
) -> Tuple[str, str, str, str, "pl.DataFrame", bool, "pl.DataType"]:
    """Shared eager+lazy hop setup: safe (FROM, TO, NID, EID) names, the edge-id frame (reuse
    `edge_binding` like the chain's __gfql_edge_index__, else synthesize a row index), and the
    node-id join dtype. Identical on pl.DataFrame and pl.LazyFrame (previously duplicated
    verbatim in hop.py; now de-duplicated)."""
    FROM = generate_safe_column_name("__gfql_from__", edges, prefix="__gfql_", suffix="__")
    TO = generate_safe_column_name("__gfql_to__", edges, prefix="__gfql_", suffix="__")
    NID = generate_safe_column_name("__gfql_nid__", all_nodes, prefix="__gfql_", suffix="__")
    if edge_binding is not None and edge_binding in edges.columns:
        EID, edges_idx, synth_eid = edge_binding, edges, False
    else:
        EID = generate_safe_column_name("__gfql_eid__", edges, prefix="__gfql_", suffix="__")
        edges_idx, synth_eid = edges.with_row_index(EID), True
    return FROM, TO, NID, EID, edges_idx, synth_eid, all_nodes.schema[node_col]


def _build_hop_pairs(
    frame: "PolarsT", direction: str, src: str, dst: str,
    node_dtype: "pl.DataType", FROM: str, TO: str, EID: str,
) -> "PolarsT":
    """Directed-(FROM,TO,EID) builder with join-key dtype aligned (polars won't coerce int/float
    join keys like pandas). `frame` = edge-id frame, eager or lazy; select/concat identical on both."""
    import polars as pl

    def _p(s: str, d: str) -> "PolarsT":
        return frame.select(pl.col(s).cast(node_dtype).alias(FROM),
                            pl.col(d).cast(node_dtype).alias(TO), pl.col(EID))
    if direction == "forward":
        return _p(src, dst)
    if direction == "reverse":
        return _p(dst, src)
    return pl.concat([_p(src, dst), _p(dst, src)], how="vertical_relaxed")


def _min_hops_labeled_node_output(
    all_nodes: "pl.DataFrame", needed: "pl.DataFrame", labeled: "pl.DataFrame",
    node_col: str, NID: str,
) -> "pl.DataFrame":
    """min_hops CHAIN wavefront node frame, mirroring pandas' labeled (track_node_hops) hop:
    FULL-attribute rows for hop-LABELED nodes (retained-path destinations; rich_nodes inner-merge,
    hop.py:944) + id-only NULL-attribute stubs for remaining retained-edge endpoints (source-side
    nodes pandas adds via the edge-endpoint concat, hop.py:994-1011, non-id cols NaN) — so a
    downstream attribute filter rejects those source-side endpoints (fuzz seed-48: reverse n5/n7,
    kind->NaN, dropped by `kind=y`). Used ONLY under min_hops_label_policy; a direct hop takes the
    plain full-attr join, matching pandas' un-labeled direct hop (hop.py:978-983)."""
    import polars as pl
    full = all_nodes.join(
        needed.join(labeled, on=NID, how="semi").rename({NID: node_col}), on=node_col, how="semi")
    stub_ids = needed.join(labeled, on=NID, how="anti").rename({NID: node_col})
    if stub_ids.height == 0:
        return full
    stub = stub_ids.with_columns(
        [pl.lit(None, dtype=all_nodes.schema[c]).alias(c) for c in all_nodes.columns if c != node_col]
    ).select(all_nodes.columns)
    return pl.concat([full, stub], how="vertical_relaxed")


def hop_polars(
    self: Plottable,
    nodes: Optional[Any] = None,
    hops: Optional[int] = 1,
    *,
    min_hops: Optional[int] = None,
    max_hops: Optional[int] = None,
    output_min_hops: Optional[int] = None,
    output_max_hops: Optional[int] = None,
    label_node_hops: Optional[str] = None,
    label_edge_hops: Optional[str] = None,
    label_seeds: bool = False,
    to_fixed_point: bool = False,
    direction: str = "forward",
    edge_match: Optional[dict] = None,
    source_node_match: Optional[dict] = None,
    destination_node_match: Optional[dict] = None,
    source_node_query: Optional[str] = None,
    destination_node_query: Optional[str] = None,
    edge_query: Optional[str] = None,
    return_as_wave_front: bool = False,
    include_zero_hop_seed: bool = False,
    target_wave_front: Optional[Any] = None,
    intermediate_universe: Optional[Any] = None,
    min_hops_label_policy: bool = False,
) -> Plottable:
    import polars as pl
    from graphistry.Engine import Engine, df_to_engine

    if direction not in ("forward", "reverse", "undirected"):
        raise ValueError(
            f'Invalid direction: "{direction}", must be one of: '
            '"forward" (default), "reverse", "undirected"'
        )

    _unsupported(
        # min_hops>1 is NATIVE fwd/rev with finite max_hops, but ONLY in the CHAIN context
        # (min_hops_label_policy=True): the layered walk + NON-anti-joined BFS port pandas'
        # CHAIN/labeled min_hops policy (hop.py:509-776 + track_node_hops node-output rule).
        # decline (NIE): a DIRECT base.hop(engine='polars', min_hops>1) needs pandas' UN-labeled
        # direct-hop node-output (hop.py:978-983) + the chain's target_wave_front threading —
        # without them it silently drops genuinely-reachable nodes; use chain()/gfql() or
        # engine='pandas'. UNDIRECTED min_hops>1 (2-core/components, hop.py:817-887) and
        # min_hops+to_fixed_point also stay deferred.
        min_hops=min_hops if (min_hops is not None and min_hops > 1
                              and (direction == "undirected" or to_fixed_point
                                   or not min_hops_label_policy)) else None,
        output_min_hops=output_min_hops,
        output_max_hops=output_max_hops,
        label_node_hops=label_node_hops,
        label_edge_hops=label_edge_hops,
        label_seeds=label_seeds or None,
        source_node_query=source_node_query,
        destination_node_query=destination_node_query,
        edge_query=edge_query,
        include_zero_hop_seed=include_zero_hop_seed or None,
    )

    if target_wave_front is not None and nodes is None:
        raise ValueError("target_wave_front requires nodes to target against (for intermediate hops)")

    if nodes is not None:
        nodes = df_to_engine(nodes, Engine.POLARS)
    if target_wave_front is not None:
        target_wave_front = df_to_engine(target_wave_front, Engine.POLARS)

    g = ensure_nodes_polars(self)

    node_col = g._node
    src = g._source
    dst = g._destination
    assert node_col is not None and src is not None and dst is not None
    assert g._edges is not None and g._nodes is not None
    edges = g._edges
    all_nodes = g._nodes

    if edge_match is not None:
        edges = filter_by_dict_polars(edges, edge_match)

    resolved_max_hops = max_hops if max_hops is not None else hops
    if to_fixed_point:
        resolved_max_hops = None
    elif not isinstance(resolved_max_hops, int):
        raise ValueError(
            f"Must provide integer hops when to_fixed_point is False, received: {resolved_max_hops}"
        )

    FROM, TO, NID, EID, edges_idx, synth_eid, node_dtype = _hop_setup_columns(
        edges, all_nodes, node_col, g._edge)
    pairs = _build_hop_pairs(edges_idx, direction, src, dst, node_dtype, FROM, TO, EID)

    def _idframe(df: "pl.DataFrame", col: str) -> "pl.DataFrame":
        return df.select(pl.col(col).cast(node_dtype).alias(NID)).unique()

    # --- SINGLE BOUNDED HOP (the dominant case — every chain edge): ONE lazy plan, ONE
    # collect_all on the active target (GPU: edge table read/transferred once — the 2.84x
    # collect-once path, formerly the separate hop.py twin). Placed BEFORE the eager gate
    # construction: the seed/gate/target id-frame .unique()s must stay INSIDE the lazy plan
    # (eagerly materializing them over chain wavefronts cost +5-14% at 1M-10M edges — A/B
    # measured, twice). Multi-hop/min_hops/to_fixed_point use the eager loop below: the
    # early-break + revisit bookkeeping need per-hop materialization, and for hops>=2 an
    # unrolled lazy plan recomputes the big edge-join per hop (polars CSE doesn't dedup it).
    if (not to_fixed_point and resolved_max_hops == 1
            and not (min_hops is not None and min_hops > 1 and direction in ("forward", "reverse"))):
        from graphistry.compute.gfql.lazy import collect_all
        edges_lf = edges_idx.lazy()
        pairs_lf = _build_hop_pairs(edges_lf, direction, src, dst, node_dtype, FROM, TO, EID)

        def _idframe_lf(lf: "pl.LazyFrame", col: str) -> "pl.LazyFrame":
            return lf.select(pl.col(col).cast(node_dtype).alias(NID)).unique()

        allowed_source_lf = (
            _idframe_lf(filter_by_dict_polars(nodes if nodes is not None else all_nodes,
                                              source_node_match).lazy(), node_col)
            if source_node_match is not None else None
        )
        allowed_dest_lf = (
            _idframe_lf(filter_by_dict_polars(all_nodes, destination_node_match).lazy(), node_col)
            if destination_node_match is not None else None
        )
        target_final_lf = (_idframe_lf(target_wave_front.lazy(), node_col)
                           if target_wave_front is not None else None)
        frontier_lf = _idframe_lf((nodes if nodes is not None else all_nodes).lazy(), node_col)
        if allowed_source_lf is not None:
            frontier_lf = frontier_lf.join(allowed_source_lf, on=NID, how="semi")
        hop_edges_lf = pairs_lf.join(frontier_lf.rename({NID: FROM}), on=FROM, how="semi")
        if target_final_lf is not None:   # single hop IS the last hop -> final gate only
            hop_edges_lf = hop_edges_lf.join(target_final_lf.rename({NID: TO}), on=TO, how="semi")
        if allowed_dest_lf is not None:
            hop_edges_lf = hop_edges_lf.join(allowed_dest_lf.rename({NID: TO}), on=TO, how="semi")
        visited_edges_lf = hop_edges_lf.select(pl.col(EID)).unique(subset=[EID])
        out_edges_lf = edges_lf.join(visited_edges_lf, on=EID, how="semi")
        if synth_eid:
            out_edges_lf = out_edges_lf.drop(EID)
        # Node set == one eager iteration: DESTINATIONS ∪ (FROM side unless wavefront); then
        # ∪ retained-edge endpoints unless wavefront-with-seeds (the wave front IS the
        # destinations). Empty out_edges yields empty endpoints — no height guard needed.
        dest_lf = hop_edges_lf.select(pl.col(TO).alias(NID)).unique()
        needed_lf = (dest_lf if return_as_wave_front else pl.concat(
            [hop_edges_lf.select(pl.col(FROM).alias(NID)), dest_lf],
            how="vertical_relaxed").unique(subset=[NID]))
        if not (return_as_wave_front and nodes is not None):
            endpoints_lf = endpoint_ids(out_edges_lf, src, dst, NID, node_dtype).unique(subset=[NID])
            needed_lf = pl.concat([needed_lf, endpoints_lf], how="vertical_relaxed").unique(subset=[NID])
        out_nodes_lf = all_nodes.lazy().join(needed_lf.rename({NID: node_col}), on=node_col, how="semi")
        out_edges_c, out_nodes_c = collect_all([out_edges_lf, out_nodes_lf])
        return g.nodes(out_nodes_c, node_col).edges(out_edges_c, src, dst)

    allowed_source = (
        _idframe(filter_by_dict_polars(nodes if (nodes is not None and not to_fixed_point and resolved_max_hops == 1) else all_nodes, source_node_match), node_col)
        if source_node_match is not None else None
    )
    allowed_dest = (
        _idframe(filter_by_dict_polars(all_nodes, destination_node_match), node_col)
        if destination_node_match is not None else None
    )
    target_final = _idframe(target_wave_front, node_col) if target_wave_front is not None else None
    # Intermediate-hop target gate (pandas hop.py:319-349,529-533): a NON-final hop's TO-node must
    # land in target_wave_front UNION the intermediate universe. The chain passes the multi-hop
    # reverse step's FORWARD WAVEFRONT as intermediate_universe (the only valid intermediate path
    # nodes); standalone hops pass None -> all_nodes, gate vacuous (a standalone hop may pass
    # through any node). Decoupled from the OUTPUT universe (all_nodes at materialization) so a
    # reduced gate never truncates returned nodes. Only the FINAL hop is gated by target alone.
    if target_final is not None:
        _univ = _idframe(intermediate_universe, node_col) if intermediate_universe is not None else _idframe(all_nodes, node_col)
        target_intermediate = pl.concat([target_final, _univ], how="vertical_relaxed").unique(subset=[NID])
    else:
        target_intermediate = None

    # min_hops>1 (fwd/rev, finite max): pandas runs a NON-anti-joined BFS (wavefront carries
    # REVISITS, hop.py:535,620) so a cycle keeps bumping max_reached_hop until max_hops — what
    # lets the lower bound be satisfied on cyclic graphs. The anti-joined (shortest-path)
    # frontier used for plain hops stops after one pass and under-counts the bound.
    min_hops_active = (
        min_hops is not None and min_hops > 1 and direction in ("forward", "reverse") and not to_fixed_point
    )
    # Per-hop label column (populated only when min_hops_active; computed unconditionally so it
    # is a plain str — every use is guarded by `if min_hops_active`).
    HOP = generate_safe_column_name("__gfql_hop__", edges, prefix="__gfql_", suffix="__")
    max_reached_hop = 0
    reached_for_attrs = None  # set in the min_hops gate: the reached-destination set (full attrs)

    empty_ids = all_nodes.select(pl.col(node_col).cast(node_dtype).alias(NID)).clear()

    seed = _idframe(nodes if nodes is not None else all_nodes, node_col)
    frontier = seed                      # DataFrame[NID]
    visited_nodes = empty_ids            # DataFrame[NID]
    visited_edge_frames = []             # collect per-hop EID frames; concat once at end
    current_hop = 0
    first = True

    while True:
        if not to_fixed_point and resolved_max_hops is not None and current_hop >= resolved_max_hops:
            break
        if frontier.height == 0:
            break
        current_hop += 1

        frontier_iter = frontier if allowed_source is None else frontier.join(allowed_source, on=NID, how="semi")
        hop_edges = pairs.join(frontier_iter.rename({NID: FROM}), on=FROM, how="semi")

        is_last = not to_fixed_point and resolved_max_hops is not None and current_hop >= resolved_max_hops
        if target_final is not None:
            assert target_intermediate is not None  # set together with target_final
            gate = target_final if is_last else target_intermediate
            hop_edges = hop_edges.join(gate.rename({NID: TO}), on=TO, how="semi")
        if allowed_dest is not None:
            hop_edges = hop_edges.join(allowed_dest.rename({NID: TO}), on=TO, how="semi")

        if first and not return_as_wave_front:
            visited_nodes = hop_edges.select(pl.col(FROM).alias(NID)).unique()
        first = False

        if min_hops_active:
            if hop_edges.height > 0:
                max_reached_hop = current_hop
            visited_edge_frames.append(
                hop_edges.select(pl.col(EID)).with_columns(pl.lit(current_hop, dtype=pl.Int64).alias(HOP))
            )
        else:
            visited_edge_frames.append(hop_edges.select(pl.col(EID)))

        cand = hop_edges.select(pl.col(TO).alias(NID)).unique()
        new_frontier = cand.join(visited_nodes, on=NID, how="anti")
        visited_nodes = pl.concat([visited_nodes, new_frontier], how="vertical_relaxed").unique(subset=[NID])
        if min_hops_active:
            # Advance with ALL destinations (revisits, hop.py:620) so a cycle-re-entered node
            # re-traverses its edges, but terminate once the cumulative reachable set stops
            # growing (hop.py:617). max_reached_hop (set above when the hop had any edge) is then
            # the closure hop the 3-case gate compares to min_hops.
            if new_frontier.height == 0:
                break
            frontier = cand
        else:
            frontier = new_frontier

    if visited_edge_frames:
        visited_edges = pl.concat(visited_edge_frames, how="vertical_relaxed").unique(subset=[EID])
    else:
        visited_edges = edges_idx.select(pl.col(EID)).clear()

    if min_hops_active:
        assert min_hops is not None
        # visited_nodes here = the loop's cumulative reached-DEST accumulation = pandas
        # matches_nodes (hop.py:609-621). `reached` drives the seed-strip; `labeled` the attr carry.
        reached = visited_nodes
        labeled = empty_ids   # nodes that get a retained-path HOP LABEL (-> full attributes)
        # Port of the pandas min_hops gate (hop.py:623-776). Three cases on max_reached_hop / goal:
        if max_reached_hop < min_hops:
            # genuinely can't reach the lower bound (hop.py:623-629) -> empty.
            visited_edges = edges_idx.select(pl.col(EID)).clear()
            reached = empty_ids
        else:
            edge_hops = pl.concat(visited_edge_frames, how="vertical_relaxed").group_by(EID).agg(pl.col(HOP).min())
            edge_rec = pairs.join(edge_hops, on=EID, how="inner")   # FROM, TO, EID, HOP (first-traversal)
            goal = edge_rec.filter(pl.col(HOP) >= min_hops)
            if goal.height == 0:
                # max_reached_hop>=min_hops but NO edge labeled >=min_hops: a cyclic REVISIT
                # satisfied the bound (no NEW node at that depth). pandas skips the prune
                # (hop.py:660 else) -> return the UNPRUNED ball; reached = full loop accumulation.
                # With no layered prune, track_node_hops keeps ALL BFS-reached node rows (seed
                # labels may reset to NA but the ROW remains -> full attrs via the rich-node
                # merge), so every reached node is "labeled". edge_hops is already grouped by
                # EID, so its key column IS the distinct retained set.
                visited_edges = edge_hops.select(pl.col(EID))
                labeled = reached
            else:
                # layered backward-TREE walk (hop.py:688-724): descend level-by-level by edge
                # label, keep edges whose TO is a current target, reset targets = their FROM.
                current_targets = goal.select(pl.col(TO).alias(NID)).unique()
                valid_node = current_targets
                # node hop-labels (hop.py:676-723): goal DESTINATIONS labeled, plus the FROM of a
                # retained edge at level>=2 (label=level-1). A FROM only at level 1 (seed/source
                # side, e.g. seed-48 reverse n5/n7) gets NO label -> null attrs later.
                labeled = current_targets
                max_edge_hop = int(edge_hops.select(pl.col(HOP).max()).item())
                valid_edge_frames = []
                for level in range(max_edge_hop, 0, -1):
                    lvl = edge_rec.filter(pl.col(HOP) == level)
                    reaching = lvl.join(current_targets.rename({NID: TO}), on=TO, how="semi")
                    valid_edge_frames.append(reaching.select(pl.col(EID)))
                    current_targets = reaching.select(pl.col(FROM).alias(NID)).unique()
                    valid_node = pl.concat([valid_node, current_targets], how="vertical_relaxed").unique(subset=[NID])
                    if level >= 2:
                        labeled = pl.concat([labeled, current_targets], how="vertical_relaxed").unique(subset=[NID])
                visited_edges = (
                    pl.concat(valid_edge_frames, how="vertical_relaxed").unique(subset=[EID])
                    if valid_edge_frames else edges_idx.select(pl.col(EID)).clear()
                )
                # matches_nodes ∩ valid_node_series (hop.py:783) — restrict reached to the pruned tree.
                reached = reached.join(valid_node, on=NID, how="semi")

        # min_hops node output = endpoints (src ∪ dst) of RETAINED edges, MINUS seeds not
        # genuinely re-reached at >=min_hops. Pandas' labeled hop keeps a node iff its
        # retained-path label is in [output_min..output_max] OR it's a retained-edge endpoint
        # (hop.py:1117-1140 output-slice mask; no slicing collapses to exactly the endpoints),
        # THEN strips SEEDS absent from matches_nodes (hop.py:1144-1170). So an endpoint seed
        # never re-reached on a >=min path drops (fuzz seed-404 reverse n1), a seed re-reached
        # at >=min keeps (seed-24 forward n2/n5), a reached non-endpoint seed drops (seed-24 n0).
        # The polars chain runs this hop WITHOUT labels, so replicate the labeled+stripped result.
        ep = pairs.join(visited_edges, on=EID, how="semi")
        visited_nodes = endpoint_ids(ep, FROM, TO, NID).unique(subset=[NID])
        if nodes is not None:  # seeds provided (chain wavefront) -> strip unreached seeds
            unreached_seeds = seed.join(reached, on=NID, how="anti")
            visited_nodes = visited_nodes.join(unreached_seeds, on=NID, how="anti")
        reached_for_attrs = labeled

    out_edges = edges_idx.join(visited_edges, on=EID, how="semi")
    if synth_eid:
        out_edges = out_edges.drop(EID)

    # Final node set: reached ∪ (edge endpoints, unless wavefront-with-seeds).
    needed = visited_nodes
    materialize_endpoints = not (return_as_wave_front and nodes is not None)
    if out_edges.height > 0 and materialize_endpoints:
        endpoints = endpoint_ids(out_edges, src, dst, NID, node_dtype).unique(subset=[NID])
        needed = pl.concat([needed, endpoints], how="vertical_relaxed").unique(subset=[NID])

    if min_hops_active and reached_for_attrs is not None and nodes is not None and min_hops_label_policy:
        out_nodes = _min_hops_labeled_node_output(all_nodes, needed, reached_for_attrs, node_col, NID)
    else:
        out_nodes = all_nodes.join(needed.rename({NID: node_col}), on=node_col, how="semi")

    return g.nodes(out_nodes, node_col).edges(out_edges, src, dst)
