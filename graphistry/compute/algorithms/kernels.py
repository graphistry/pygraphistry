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

The kernels are hand-written so one implementation can run on pandas and cuDF.
PageRank supports both cuGraph-compatible convergence and the fixed-iteration
LDBC workload; the other algorithms retain their documented standard-library
semantics without requiring an optional graph backend.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Union

from graphistry.compute.typing import ArrayLike, DataFrameT, SeriesT

from ._dfops import (
    SHIFT32,
    align,
    arange,
    array_namespace,
    chunk_bounds,
    concat_frames,
    df_cons,
    emin,
    full,
    gather,
    is_cudf,
    mask_rows,
    rows,
    series_from_array,
    series_to_array,
    slice_by_key,
    splitmix64,
    to_host_floats,
    to_host_int,
    u64,
    vertex_ranges,
)


class ConvergenceError(RuntimeError):
    """A kernel hit its iteration cap without converging."""


def _sorted_copies(edges: DataFrameT, src: str, dst: str) -> tuple[DataFrameT, DataFrameT]:
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
    by_src: DataFrameT, by_dst: DataFrameT, src: str, dst: str, vec: SeriesT, v_count: int, chunks: int
) -> Optional[DataFrameT]:
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
                df_cons(ea, {"v": ea[src], "p": gather(vec, ea[dst])}).groupby("v", sort=False)["p"].min().reset_index()
            )
        if len(eb):
            outs.append(
                df_cons(eb, {"v": eb[dst], "p": gather(vec, eb[src])}).groupby("v", sort=False)["p"].min().reset_index()
            )
    res = concat_frames(outs)
    if res is None:
        return None
    # Chunks are key-disjoint, but the two DIRECTIONS within a chunk are not.
    return res.groupby("v", sort=False)["p"].min().reset_index()


def wcc(edges: DataFrameT, src: str, dst: str, v_count: int, chunks: int = 1, max_iter: int = 1000) -> SeriesT:
    """Weakly connected components, LDBC label = min original vertex id.

    Shiloach-Vishkin min-label propagation with pointer jumping. Because
    `dense_renumber` is monotone, the dense min-label IS the min original id, so
    the LDBC label semantics can be asserted directly with no reference output.

    Pointer jumping reduces dependence on path diameter; this is
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


def _pagerank_iterations(
    initial: ArrayLike,
    step: Callable[[ArrayLike], tuple[ArrayLike, float, float]],
    max_iter: int,
    tol: float,
    stopping: Literal["convergence", "fixed_iterations"],
) -> tuple[ArrayLike, bool]:
    """Run the common PageRank convergence and mass policy for either backend."""
    current = initial
    converged = False
    for _iteration in range(max_iter):
        new, total, delta = step(current)
        if abs(total - 1.0) > 1e-9:
            raise AssertionError(f"pagerank mass not conserved: sum={total!r}")
        converged = delta < float(tol)
        current = new
        if stopping == "convergence" and converged:
            break
    return current, converged


def _pagerank_cudf_fast(
    edges: DataFrameT,
    src: str,
    dst: str,
    v_count: int,
    weight: Optional[str],
    out_weight: SeriesT,
    is_dangling: SeriesT,
    teleport: SeriesT,
    initial: SeriesT,
    damping: float,
    max_iter: int,
    tol: float,
    stopping: Literal["convergence", "fixed_iterations"],
) -> tuple[SeriesT, bool]:
    """CuPy fast path with reusable dense buffers and fused atomic scatter."""
    xp = array_namespace(edges)
    src_values = series_to_array(edges[src])
    dst_values = series_to_array(edges[dst])
    out_weight_values = series_to_array(out_weight)
    teleport_values = series_to_array(teleport)
    initial_values = series_to_array(initial)
    dangling_weights = series_to_array(is_dangling).astype(xp.float64, copy=False)
    weight_values = series_to_array(edges[weight].reset_index(drop=True).astype("float64")) if weight is not None else None
    weighted = weight_values is not None
    weight_parameter = "const double* edge_weight," if weighted else ""
    weight_factor = " * edge_weight[edge]" if weighted else ""
    kernel_name = "graphistry_pagerank_scatter_weighted" if weighted else "graphistry_pagerank_scatter_unweighted"
    scatter = xp.RawKernel(
        f"""
        extern "C" __global__
        void {kernel_name}(
            const int* src,
            const int* dst,
            const double* ranks,
            const double* out_weight,
            {weight_parameter}
            double* inflow,
            const unsigned long long edge_count
        ) {{
            unsigned long long edge =
                (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long stride =
                (unsigned long long) blockDim.x * gridDim.x;
            for (; edge < edge_count; edge += stride) {{
                const int source = src[edge];
                const double denominator = out_weight[source];
                if (denominator > 0.0) {{
                    atomicAdd(
                        &inflow[dst[edge]],
                        (ranks[source] / denominator){weight_factor}
                    );
                }}
            }}
        }}
        """,
        kernel_name,
    )
    rank_update = xp.ElementwiseKernel(
        "float64 inflow, float64 teleport, float64 dangling_mass, float64 alpha",
        "float64 rank",
        "rank = (1.0 - alpha) * teleport + alpha * (inflow + dangling_mass * teleport)",
        "graphistry_pagerank_rank_update",
    )
    inflow = xp.empty(v_count, dtype=xp.float64)
    next_values = xp.empty(v_count, dtype=xp.float64)
    edge_count = int(src_values.size)
    edge_count_arg = xp.uint64(edge_count)
    threads = 256
    blocks = min(65535, max(1, (edge_count + threads - 1) // threads))

    def step(current: ArrayLike) -> tuple[ArrayLike, float, float]:
        nonlocal next_values
        inflow.fill(0.0)
        args = (
            (
                src_values,
                dst_values,
                current,
                out_weight_values,
                weight_values,
                inflow,
                edge_count_arg,
            )
            if weighted
            else (
                src_values,
                dst_values,
                current,
                out_weight_values,
                inflow,
                edge_count_arg,
            )
        )
        scatter((blocks,), (threads,), args)
        dangling_mass = xp.dot(current, dangling_weights)
        rank_update(
            inflow,
            teleport_values,
            dangling_mass,
            damping,
            next_values,
        )
        new = next_values
        total, delta = to_host_floats((new.sum(), xp.absolute(new - current).sum()))
        next_values = current
        return new, total, delta

    result, converged = _pagerank_iterations(initial_values, step, max_iter, tol, stopping)
    return series_from_array(edges, result), converged


def _pagerank_fast(
    edges: DataFrameT,
    src: str,
    dst: str,
    v_count: int,
    weight: Optional[str],
    out_weight: SeriesT,
    is_dangling: SeriesT,
    teleport: SeriesT,
    initial: SeriesT,
    damping: float,
    max_iter: int,
    tol: float,
    stopping: Literal["convergence", "fixed_iterations"],
) -> tuple[SeriesT, bool]:
    """Backend-native dense reduction without dataframe materialization."""
    if is_cudf(edges):
        return _pagerank_cudf_fast(
            edges,
            src,
            dst,
            v_count,
            weight,
            out_weight,
            is_dangling,
            teleport,
            initial,
            damping,
            max_iter,
            tol,
            stopping,
        )
    xp = array_namespace(edges)
    src_values = series_to_array(edges[src])
    dst_values = series_to_array(edges[dst])
    weight_values = (
        series_to_array(edges[weight].reset_index(drop=True).astype("float64"))
        if weight is not None
        else None
    )
    dangling_mask = series_to_array(is_dangling)
    safe_out = series_to_array(out_weight.where(~is_dangling, 1.0))
    teleport_values = series_to_array(teleport)
    initial_values = series_to_array(initial)

    def step(current: ArrayLike) -> tuple[ArrayLike, float, float]:
        contribution = xp.where(
            dangling_mask, 0.0, xp.divide(current, safe_out)
        )
        messages = contribution[src_values]
        if weight_values is not None:
            messages = xp.multiply(messages, weight_values)
        inflow = xp.bincount(dst_values, weights=messages, minlength=v_count)
        dangling_mass = xp.multiply(current, dangling_mask).sum()
        new = xp.multiply(teleport_values, 1.0 - damping) + xp.multiply(
            inflow + xp.multiply(teleport_values, dangling_mass), damping
        )
        total, delta = to_host_floats(
            (new.sum(), xp.absolute(new - current).sum())
        )
        return new, total, delta

    result, converged = _pagerank_iterations(
        initial_values, step, max_iter, tol, stopping
    )
    return series_from_array(edges, result), converged


def _pagerank_bounded(
    edges: DataFrameT,
    src: str,
    dst: str,
    v_count: int,
    weight: Optional[str],
    chunks: int,
    out_weight: SeriesT,
    is_dangling: SeriesT,
    teleport: SeriesT,
    initial: SeriesT,
    damping: float,
    max_iter: int,
    tol: float,
    stopping: Literal["convergence", "fixed_iterations"],
) -> tuple[SeriesT, bool]:
    """Chunkable dataframe path retained for explicit bounded-memory use."""
    by_dst = edges.sort_values(dst).reset_index(drop=True)
    safe_out = out_weight.where(~is_dangling, 1.0)

    def step(current: SeriesT) -> tuple[SeriesT, float, float]:
        contribution = (current / safe_out).where(~is_dangling, 0.0)
        dangling_mass = float(current.where(is_dangling, 0.0).sum())

        outs = []
        for lo, hi in chunk_bounds(len(by_dst), chunks):
            edge_chunk = rows(by_dst, lo, hi)
            messages = gather(contribution, edge_chunk[src])
            if weight is not None:
                messages = messages * edge_chunk[weight].reset_index(drop=True)
            outs.append(
                df_cons(edge_chunk, {"v": edge_chunk[dst], "m": messages})
                .groupby("v", sort=False)["m"]
                .sum()
                .reset_index()
            )
        reduced = concat_frames(outs)
        if reduced is not None and chunks > 1:
            reduced = (
                reduced.groupby("v", sort=False)["m"].sum().reset_index()
            )
        inflow = align(
            edges,
            v_count,
            reduced,
            "v",
            "m",
            full(edges, v_count, 0.0, "float64"),
        )
        new = (1.0 - damping) * teleport + damping * (
            inflow + dangling_mass * teleport
        )
        return new, float(new.sum()), float(abs(new - current).sum())

    return _pagerank_iterations(initial, step, max_iter, tol, stopping)


def pagerank(
    edges: DataFrameT,
    src: str,
    dst: str,
    v_count: int,
    alpha: float = 0.85,
    personalization: Optional[SeriesT] = None,
    precomputed_vertex_out_weight: Optional[SeriesT] = None,
    max_iter: int = 100,
    tol: float = 1.0e-5,
    nstart: Optional[SeriesT] = None,
    dangling: object = None,
    fail_on_nonconvergence: bool = True,
    *,
    weight: Optional[str] = None,
    chunks: int = 1,
    stopping: Literal["convergence", "fixed_iterations"] = "convergence",
    iterations: Optional[int] = None,
    damping: Optional[float] = None,
    method: Literal["auto", "fast", "bounded"] = "auto",
) -> Union[SeriesT, tuple[SeriesT, bool]]:
    """PageRank with cuGraph-compatible controls and fast/bounded extras."""
    if v_count == 0:
        empty = full(edges, 0, 0.0, "float64")
        return (empty, True) if not fail_on_nonconvergence else empty
    if damping is not None:
        if alpha != 0.85 and alpha != damping:
            raise ValueError("pagerank alpha and damping aliases disagree")
        alpha = damping
    if iterations is not None:
        if max_iter != 100 and max_iter != iterations:
            raise ValueError("pagerank max_iter and iterations aliases disagree")
        max_iter = iterations
        stopping = "fixed_iterations"
    if stopping not in ("convergence", "fixed_iterations"):
        raise ValueError("pagerank stopping must be 'convergence' or 'fixed_iterations'")
    if method not in ("auto", "fast", "bounded"):
        raise ValueError("pagerank method must be 'auto', 'fast', or 'bounded'")
    if method == "fast" and chunks != 1:
        raise ValueError("pagerank method='fast' requires chunks=1")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("pagerank alpha must be greater than 0 and less than 1")
    if stopping == "convergence" and max_iter <= 0:
        max_iter = 100
    elif max_iter < 0:
        raise ValueError("pagerank max_iter must be non-negative in fixed mode")
    if tol == 0.0:
        tol = 1.0e-5
    if tol < 0.0:
        raise ValueError("pagerank tol must be non-negative")

    def normalized(values: Optional[SeriesT], name: str) -> SeriesT:
        if values is None:
            return full(edges, v_count, 1.0 / v_count, "float64")
        out = values.reset_index(drop=True).astype("float64")
        if len(out) != v_count:
            raise ValueError(f"pagerank {name} must have one value per vertex")
        invalid = (out != out) | (out < 0.0) | (out == float("inf"))
        if to_host_int(invalid.sum()) != 0:
            raise ValueError(f"pagerank {name} values must be finite and non-negative")
        total = float(out.sum())
        if total <= 0.0:
            raise ValueError(f"pagerank {name} values must have a positive sum")
        return out / total

    use_fast = method == "fast" or (method == "auto" and chunks == 1)
    use_cudf_fast = use_fast and is_cudf(edges)

    if weight is None:
        if use_cudf_fast:
            xp = array_namespace(edges)
            computed_out = series_from_array(
                edges,
                xp.bincount(series_to_array(edges[src]), minlength=v_count).astype(xp.float64, copy=False),
            )
        else:
            sums = edges.groupby(src, sort=False).size().reset_index(name="__out")
            computed_out = align(
                edges,
                v_count,
                sums,
                src,
                "__out",
                full(edges, v_count, 0.0, "float64"),
            )
    else:
        if weight not in edges.columns:
            raise ValueError(f"pagerank weight column {weight!r} is missing")
        edge_weight = edges[weight].reset_index(drop=True).astype("float64")
        invalid_weight = (
            (edge_weight != edge_weight)
            | (edge_weight < 0.0)
            | (edge_weight == float("inf"))
        )
        if to_host_int(invalid_weight.sum()) != 0:
            raise ValueError("pagerank edge weights must be finite and non-negative")
        if use_cudf_fast:
            xp = array_namespace(edges)
            computed_out = series_from_array(
                edges,
                xp.bincount(
                    series_to_array(edges[src]),
                    weights=series_to_array(edge_weight),
                    minlength=v_count,
                ).astype(xp.float64, copy=False),
            )
        else:
            sums = (
                df_cons(edges, {src: edges[src], "__out": edge_weight})
                .groupby(src, sort=False)["__out"]
                .sum()
                .reset_index()
            )
            computed_out = align(
                edges,
                v_count,
                sums,
                src,
                "__out",
                full(edges, v_count, 0.0, "float64"),
            )

    out_weight = computed_out
    if precomputed_vertex_out_weight is not None:
        supplied = precomputed_vertex_out_weight.reset_index(drop=True).astype("float64")
        if len(supplied) != v_count:
            raise ValueError(
                "pagerank precomputed_vertex_out_weight must have one value per vertex"
            )
        present = supplied == supplied
        invalid_out = present & ((supplied < 0.0) | (supplied == float("inf")))
        if to_host_int(invalid_out.sum()) != 0:
            raise ValueError(
                "pagerank precomputed out weights must be finite and non-negative"
            )
        out_weight = supplied.where(present, computed_out)

    is_dangling = out_weight == 0.0
    if to_host_int(((computed_out > 0.0) & is_dangling).sum()) != 0:
        raise ValueError(
            "pagerank precomputed out weight is zero for a vertex with outgoing weight"
        )

    teleport = normalized(personalization, "personalization")
    pr = normalized(nstart, "nstart")
    d = float(alpha)
    _ = dangling

    if use_fast:
        pr, converged = _pagerank_fast(
            edges,
            src,
            dst,
            v_count,
            weight,
            out_weight,
            is_dangling,
            teleport,
            pr,
            d,
            max_iter,
            tol,
            stopping,
        )
    else:
        pr, converged = _pagerank_bounded(
            edges,
            src,
            dst,
            v_count,
            weight,
            chunks,
            out_weight,
            is_dangling,
            teleport,
            pr,
            d,
            max_iter,
            tol,
            stopping,
        )

    if stopping == "fixed_iterations" or converged:
        return (pr, converged) if not fail_on_nonconvergence else pr
    if fail_on_nonconvergence:
        raise ConvergenceError(f"pagerank did not converge in {max_iter} iterations")
    return pr, False


def cdlp(
    edges: DataFrameT,
    src: str,
    dst: str,
    v_count: int,
    iterations: int = 10,
    chunks: int = 1,
) -> SeriesT:
    """LDBC community detection by label propagation.

    Synchronous, fixed iteration count, undirected, MULTISET semantics (parallel
    edges count multiple times), most frequent label wins, ties break to the
    smallest label.

    The tie-break is done by maximizing a packed key `count*2^32 + (LMAX-label)`
    in a single groupby-max. That maximizes count and, among equal counts,
    minimizes label -- replacing a global `sort_values` that costs ~17 GB at
    graph500-26 (~34 GB with cuDF's sort overhead) with a vertex-sized output pass.
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
    edges: DataFrameT,
    src: str,
    dst: str,
    weight: str,
    v_count: int,
    source: int,
    chunks: int = 1,
    max_iter: Optional[int] = None,
) -> SeriesT:
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
            outs.append(df_cons(e, {"v": e[dst], "d": nd}).groupby("v", sort=False)["d"].min().reset_index())
        res = concat_frames(outs)
        if res is None:
            return dist
        res = res.groupby("v", sort=False)["d"].min().reset_index()

        cur = gather(dist, res["v"])
        improved = res[
            (res["d"].reset_index(drop=True) < cur).values if not is_cudf(res) else (res["d"].reset_index(drop=True) < cur)
        ]
        if len(improved) == 0:
            return dist
        dist = align(edges, v_count, improved, "v", "d", dist).astype("float32")
        frontier = full(edges, v_count, False, "bool")
        frontier.iloc[improved["v"]] = True

    raise ConvergenceError(f"sssp did not settle in {cap} iterations")


def make_weights(edges: DataFrameT, src: str, dst: str) -> SeriesT:
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
    edges: DataFrameT,
    src: str,
    dst: str,
    v_count: int,
    seed: int = 0x5EED,
    chunks: int = 1,
    max_rounds: int = 200,
) -> SeriesT:
    """Maximal independent set, Luby's algorithm.

    NOTE: MIS is not one of the six official LDBC Graphalytics kernels
    (BFS/PR/WCC/CDLP/LCC/SSSP), so there is no reference output and no canonical
    variant. This is Luby's random-priority form; a greedy-by-id MIS would give a
    different set at a very different cost.

    Priority is a packed `(hash32, vertex_id)` giving a STRICT TOTAL ORDER, so
    there are no ties, so at least one vertex joins per active component per
    round. The uint64
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

    in_set = deg == 0  # isolated vertices are in the set
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
    by_src: DataFrameT, by_dst: DataFrameT, src: str, dst: str, vec: SeriesT, active: SeriesT, v_count: int, chunks: int
) -> Optional[DataFrameT]:
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
                df_cons(e, {"v": e[key], "p": gather(vec, e[other])}).groupby("v", sort=False)["p"].min().reset_index()
            )
    res = concat_frames(outs)
    return None if res is None else res.groupby("v", sort=False)["p"].min().reset_index()


def _sym_any(
    by_src: DataFrameT, by_dst: DataFrameT, src: str, dst: str, flag: SeriesT, v_count: int, chunks: int
) -> Optional[DataFrameT]:
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
