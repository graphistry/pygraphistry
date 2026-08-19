"""LDBC-Graphalytics-style graph kernels as engine-agnostic dataframe code.

One implementation runs on both pandas and cudf; the engine is inferred from the
frame you pass in. Every kernel takes a dense-renumbered int32 edge frame (see
`dfops.dense_renumber`) and returns a dense per-vertex result Series.

Three formulation decisions carry the entire memory budget at 1B edges:

1. The symmetrized (2E) edge list is NEVER materialized. Undirected kernels
   symmetrize at aggregation time via two direction-keyed groupbys over the same
   E-row frame.
2. Vertex ids are dense int32 and per-vertex state is positional, so frontier
   expansion is `gather(vec, edges[src])` + a boolean mask rather than a hash
   join over the edge list.
3. CDLP's "most frequent label, ties to smallest" is a single groupby-max over a
   packed key, not a global sort.

Four of the five kernels are hand-written rather than delegated because cuGraph
and igraph do not implement the LDBC semantics: PageRank there runs to a
tolerance while LDBC fixes the iteration count, cuGraph has no label propagation
at all, and neither has a usable MIS. cuGraph is used as an *oracle* for WCC and
SSSP, where it is exact.
"""
from __future__ import annotations

from typing import Any, Optional

from ._dfops import (
    SHIFT32,
    align,
    arange,
    chunk_bounds,
    concat_frames,
    df_cons,
    emin,
    full,
    gather,
    is_cudf,
    mask_rows,
    rows,
    slice_by_key,
    splitmix64,
    to_host_int,
    u64,
    vertex_ranges,
)


class ConvergenceError(RuntimeError):
    """A kernel hit its iteration cap without converging."""


def _sorted_copies(edges: Any, src: str, dst: str) -> tuple[Any, Any]:
    """Two sorted copies of the edge frame: by src and by dst.

    Costs 2x the edge frame resident (16.8 GB at E=1.05B, int32) and buys free
    contiguous-slice chunking by vertex range, key-disjoint partial aggregates,
    and a stable deterministic row order. Sorting happens once, never in a loop.
    """
    return (
        edges.sort_values(src).reset_index(drop=True),
        edges.sort_values(dst).reset_index(drop=True),
    )


def _sym_min(
    by_src: Any, by_dst: Any, src: str, dst: str, vec: Any, v_count: int, chunks: int
) -> Any:
    """min over the undirected neighborhood, without materializing 2E rows.

    For a vertex range [a,b) every out-edge is a contiguous slice of the
    src-sorted copy and every in-edge a contiguous slice of the dst-sorted copy,
    so the chunk sees each vertex's COMPLETE neighbor set and the per-chunk
    results are key-disjoint -- the combine is a concat, not a second reduce.
    """
    outs = []
    for a, b in vertex_ranges(v_count, chunks):
        ea = slice_by_key(by_src, src, a, b)
        eb = slice_by_key(by_dst, dst, a, b)
        if len(ea):
            outs.append(
                df_cons(ea, {"v": ea[src], "p": gather(vec, ea[dst])})
                .groupby("v", sort=False)["p"]
                .min()
                .reset_index()
            )
        if len(eb):
            outs.append(
                df_cons(eb, {"v": eb[dst], "p": gather(vec, eb[src])})
                .groupby("v", sort=False)["p"]
                .min()
                .reset_index()
            )
    res = concat_frames(outs)
    if res is None:
        return None
    # Chunks are key-disjoint, but the two DIRECTIONS within a chunk are not.
    return res.groupby("v", sort=False)["p"].min().reset_index()


def wcc(edges: Any, src: str, dst: str, v_count: int, chunks: int = 1, max_iter: int = 1000) -> Any:
    """Weakly connected components, LDBC label = min original vertex id.

    Shiloach-Vishkin min-label propagation with pointer jumping. Because
    `dense_renumber` is monotone, the dense min-label IS the min original id, so
    the LDBC label semantics can be asserted directly with no reference output.

    Pointer jumping is what makes this O(log V) rather than O(diameter) -- it is
    the difference between ~6 iterations and hundreds on a long-path graph.
    """
    by_src, by_dst = _sorted_copies(edges, src, dst)
    lbl = arange(edges, v_count, "int32")

    for it in range(max_iter):
        cand = _sym_min(by_src, by_dst, src, dst, lbl, v_count, chunks)
        new = align(edges, v_count, cand, "v", "p", lbl)
        new = emin(new, lbl).astype("int32")
        # Path compression: label[v] <- label[label[v]].
        for _ in range(2):
            new = gather(new, new).astype("int32")
            new = emin(new, lbl).astype("int32")
        if to_host_int((new != lbl).sum()) == 0:
            return lbl
        lbl = new

    raise ConvergenceError(f"wcc did not converge in {max_iter} iterations")


def pagerank(
    edges: Any,
    src: str,
    dst: str,
    v_count: int,
    iterations: int = 10,
    damping: float = 0.85,
    chunks: int = 1,
) -> Any:
    """LDBC PageRank: FIXED iteration count, dangling mass redistributed uniformly.

    There is deliberately no early exit. With d=0.85 and K=10, d^K ~ 0.197, so a
    fixed-K vector differs materially from a converged one -- which is exactly
    why `cugraph.pagerank` and `igraph.pagerank` cannot serve as oracles here.

    Chunking is by DST so partial results are key-disjoint and the combine is a
    concat rather than a second groupby.
    """
    _, by_dst = _sorted_copies(edges, src, dst)

    deg = edges.groupby(src, sort=False).size().reset_index(name="__deg")
    outdeg = align(edges, v_count, deg, src, "__deg", full(edges, v_count, 0, "int64"))
    dangling = outdeg == 0

    pr = full(edges, v_count, 1.0 / v_count, "float64")
    d = float(damping)

    for _ in range(iterations):
        safe_deg = outdeg.where(~dangling, 1)
        contrib = (pr / safe_deg).where(~dangling, 0.0)
        dang = float(pr.where(dangling, 0.0).sum())

        outs = []
        for lo, hi in chunk_bounds(len(by_dst), chunks):
            e = rows(by_dst, lo, hi)
            msg = gather(contrib, e[src])
            outs.append(
                df_cons(e, {"v": e[dst], "m": msg})
                .groupby("v", sort=False)["m"]
                .sum()
                .reset_index()
            )
        res = concat_frames(outs)
        if res is not None and chunks > 1:
            # Row-chunking by dst can split a dst group across a boundary.
            res = res.groupby("v", sort=False)["m"].sum().reset_index()
        inflow = align(edges, v_count, res, "v", "m", full(edges, v_count, 0.0, "float64"))

        pr = (1.0 - d) / v_count + d * (inflow + dang / v_count)

        # Mass conservation is a free bug detector for the dangling term.
        total = float(pr.sum())
        if abs(total - 1.0) > 1e-9:
            raise AssertionError(f"pagerank mass not conserved: sum={total!r}")

    return pr


def cdlp(
    edges: Any,
    src: str,
    dst: str,
    v_count: int,
    iterations: int = 10,
    chunks: int = 1,
) -> Any:
    """LDBC community detection by label propagation.

    Synchronous, fixed iteration count, undirected, MULTISET semantics (parallel
    edges count multiple times), most frequent label wins, ties break to the
    smallest label.

    The tie-break is done by maximizing a packed key `count*2^32 + (LMAX-label)`
    in a single groupby-max. That maximizes count and, among equal counts,
    minimizes label -- replacing a global `sort_values` that costs ~17 GB at
    graph500-26 (~34 GB with cuDF's sort overhead) with one O(V)-output pass.
    Being order-independent, it is also bitwise deterministic across engines.

    No early exit: synchronous label propagation can oscillate with period 2, so
    "no change" may never occur. The spec is K iterations regardless.
    """
    by_src, by_dst = _sorted_copies(edges, src, dst)
    lmax = v_count - 1
    lbl = arange(edges, v_count, "int32")

    for _ in range(iterations):
        outs = []
        for a, b in vertex_ranges(v_count, chunks):
            ea = slice_by_key(by_src, src, a, b)
            eb = slice_by_key(by_dst, dst, a, b)
            parts = []
            if len(ea):
                parts.append(df_cons(ea, {"v": ea[src], "l": gather(lbl, ea[dst])}))
            if len(eb):
                parts.append(df_cons(eb, {"v": eb[dst], "l": gather(lbl, eb[src])}))
            pair = concat_frames(parts)
            if pair is None:
                continue
            cnt = pair.groupby(["v", "l"], sort=False).size().reset_index(name="c")
            key = cnt["c"].astype("int64") * SHIFT32 + (lmax - cnt["l"].astype("int64"))
            cnt = df_cons(cnt, {"v": cnt["v"], "key": key})
            outs.append(cnt.groupby("v", sort=False)["key"].max().reset_index())
            del pair, cnt

        win = concat_frames(outs)
        if win is None:
            break
        win = win.groupby("v", sort=False)["key"].max().reset_index()
        best = df_cons(win, {"v": win["v"], "l": (lmax - (win["key"] % SHIFT32)).astype("int32")})
        # Isolated vertices are absent from `best` and keep their own label.
        lbl = align(edges, v_count, best, "v", "l", lbl).astype("int32")

    return lbl


def sssp(
    edges: Any,
    src: str,
    dst: str,
    weight: str,
    v_count: int,
    source: int,
    chunks: int = 1,
    max_iter: Optional[int] = None,
) -> Any:
    """Weighted single-source shortest path, frontier Bellman-Ford.

    Relaxation is gather + boolean mask, not a per-hop hash join against the
    frontier. With small integer weights (see `make_weights`) all distances stay
    integral and well under 2^24, so float32 sums are EXACT -- which is what
    makes bitwise equality against `cugraph.sssp` an achievable validation gate.
    """
    by_src, _ = _sorted_copies(edges, src, dst)
    # V-1 relaxation rounds suffice, plus one more to observe no-change and exit.
    cap = max_iter if max_iter is not None else max(v_count, 2)

    inf = float("inf")
    dist = full(edges, v_count, inf, "float32")
    dist.iloc[source] = 0.0
    frontier = full(edges, v_count, False, "bool")
    frontier.iloc[source] = True

    for _ in range(cap):
        if not bool(frontier.any()):
            return dist
        outs = []
        for lo, hi in chunk_bounds(len(by_src), chunks):
            e = rows(by_src, lo, hi)
            mask = gather(frontier, e[src])
            e = mask_rows(e, mask)
            if len(e) == 0:
                continue
            nd = gather(dist, e[src]) + e[weight].reset_index(drop=True).astype("float32")
            outs.append(
                df_cons(e, {"v": e[dst], "d": nd}).groupby("v", sort=False)["d"].min().reset_index()
            )
        res = concat_frames(outs)
        if res is None:
            return dist
        res = res.groupby("v", sort=False)["d"].min().reset_index()

        cur = gather(dist, res["v"])
        improved = res[(res["d"].reset_index(drop=True) < cur).values if not is_cudf(res) else (res["d"].reset_index(drop=True) < cur)]
        if len(improved) == 0:
            return dist
        dist = align(edges, v_count, improved, "v", "d", dist).astype("float32")
        frontier = full(edges, v_count, False, "bool")
        frontier.iloc[improved["v"]] = True

    raise ConvergenceError(f"sssp did not settle in {cap} iterations")


def make_weights(edges: Any, src: str, dst: str) -> Any:
    """Deterministic integer edge weights in [1, 255] as float32.

    Pure integer arithmetic in uint64 so both engines produce bitwise identical
    weights. Small integer weights keep every path distance exactly representable
    in float32, which is what lets SSSP be validated by exact equality.
    """
    u = edges[src].astype("uint64")
    v = edges[dst].astype("uint64")
    h = splitmix64(u * u64(0x9E3779B97F4A7C15) + v * u64(0xC2B2AE3D27D4EB4F))
    return (h % u64(255) + u64(1)).astype("float32")


def mis(
    edges: Any, src: str, dst: str, v_count: int, seed: int = 0x5EED, chunks: int = 1,
    max_rounds: int = 200,
) -> Any:
    """Maximal independent set, Luby's algorithm.

    NOTE: MIS is not one of the six official LDBC Graphalytics kernels
    (BFS/PR/WCC/CDLP/LCC/SSSP), so there is no reference output and no canonical
    variant. This is Luby's random-priority form; a greedy-by-id MIS would give a
    different set at a very different cost.

    Priority is a packed `(hash32, vertex_id)` giving a STRICT TOTAL ORDER, so
    there are no ties, so at least one vertex joins per active component per
    round -- termination is guaranteed, in O(log V) rounds expected. The uint64
    hash wraps identically on both engines, making the result bitwise
    reproducible; that is what substitutes for the missing reference output.

    Self-loops MUST be dropped before calling (a self-looped vertex trivially
    violates independence) and isolated vertices MUST start in the set (else
    maximality fails).
    """
    by_src, by_dst = _sorted_copies(edges, src, dst)
    vid = arange(edges, v_count, "int64")

    deg_s = edges.groupby(src, sort=False).size().reset_index(name="__deg")
    deg_d = edges.groupby(dst, sort=False).size().reset_index(name="__deg")
    zero = full(edges, v_count, 0, "int64")
    deg = align(edges, v_count, deg_s, src, "__deg", zero) + align(edges, v_count, deg_d, dst, "__deg", zero)

    in_set = deg == 0          # isolated vertices are in the set
    active = ~in_set
    uint_max = u64((1 << 64) - 1)

    for r in range(max_rounds):
        if not bool(active.any()):
            return in_set

        h = splitmix64(vid.astype("uint64") ^ u64(seed + r * 0x9E3779B97F4A7C15))
        prio = (h % u64(SHIFT32)) * u64(SHIFT32) + vid.astype("uint64")
        prio = prio.where(active, uint_max)

        minnbr = _sym_min_active(by_src, by_dst, src, dst, prio, active, v_count, chunks)
        minnbr = align(edges, v_count, minnbr, "v", "p", full(edges, v_count, uint_max, "uint64"))

        joins = active & (prio < minnbr)
        if not bool(joins.any()):
            raise ConvergenceError("mis round produced no joiners; priority order is not total")
        in_set = in_set | joins

        nbr_joined = _sym_any(by_src, by_dst, src, dst, joins, v_count, chunks)
        nbr_joined = align(edges, v_count, nbr_joined, "v", "p", full(edges, v_count, 0, "int8"))
        active = active & ~joins & (nbr_joined == 0)

    raise ConvergenceError(f"mis did not terminate in {max_rounds} rounds")


def _sym_min_active(
    by_src: Any, by_dst: Any, src: str, dst: str, vec: Any, active: Any, v_count: int, chunks: int
) -> Any:
    """min of `vec` over ACTIVE undirected neighbours, chunked by vertex range."""
    outs = []
    for a, b in vertex_ranges(v_count, chunks):
        for frame, key, other in ((by_src, src, dst), (by_dst, dst, src)):
            e = slice_by_key(frame, key, a, b)
            if len(e) == 0:
                continue
            keep = gather(active, e[key]) & gather(active, e[other])
            e = mask_rows(e, keep)
            if len(e) == 0:
                continue
            outs.append(
                df_cons(e, {"v": e[key], "p": gather(vec, e[other])})
                .groupby("v", sort=False)["p"]
                .min()
                .reset_index()
            )
    res = concat_frames(outs)
    return None if res is None else res.groupby("v", sort=False)["p"].min().reset_index()


def _sym_any(
    by_src: Any, by_dst: Any, src: str, dst: str, flag: Any, v_count: int, chunks: int
) -> Any:
    """1 where any undirected neighbour has `flag` set."""
    outs = []
    for a, b in vertex_ranges(v_count, chunks):
        for frame, key, other in ((by_src, src, dst), (by_dst, dst, src)):
            e = slice_by_key(frame, key, a, b)
            if len(e) == 0:
                continue
            outs.append(
                df_cons(e, {"v": e[key], "p": gather(flag, e[other]).astype("int8")})
                .groupby("v", sort=False)["p"]
                .max()
                .reset_index()
            )
    res = concat_frames(outs)
    return None if res is None else res.groupby("v", sort=False)["p"].max().reset_index()
