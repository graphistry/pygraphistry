"""
Native graph-library backends for shortest-path distance computation.

Called from _gfql_multihop_binding_rows when in shortest_path_mode and a
native library is available. Falls back to BFS when not available.

Contract
--------
- step_pairs : DataFrame with columns __from__, __to__  (filtered edge set)
- sources    : 1-D array-like of source node IDs
- targets    : 1-D array-like of target node IDs  (one per source row)
- max_hops   : Optional[int] — cap on search depth; None means unbounded
- directed   : bool — whether to treat edges as directed

Returns a DataFrame with columns:
  __sp_source__  — source node ID
  __sp_target__  — target node ID
  __sp_hops__    — int hop distance, or None/NaN if unreachable

Backend selection (try_native_shortest_path):
  1. cugraph  — if engine == Engine.CUDF
  2. igraph   — if engine == Engine.PANDAS and igraph is importable
  3. None     — caller falls back to BFS
"""

import logging
from typing import Any, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ShortestPathBackend = Literal["auto", "igraph", "cugraph", "bfs"]


def igraph_shortest_path_distances(
    step_pairs: Any,
    sources: Any,
    targets: Any,
    *,
    min_hops: int = 1,
    max_hops: Optional[int],
    directed: bool,
) -> pd.DataFrame:
    """
    Compute pairwise shortest-path distances using igraph.distances().

    Batches all targets per unique source to avoid redundant BFS.
    Returns a pandas DataFrame with __sp_source__, __sp_target__, __sp_hops__.
    Unreachable, under-min-hops, or over-max-hops pairs get __sp_hops__ = None.
    """
    import igraph as ig  # type: ignore[import]

    sources_list = list(sources)
    targets_list = list(targets)

    sp_frm = list(step_pairs["__from__"])
    sp_to = list(step_pairs["__to__"])
    all_nodes = list(dict.fromkeys(sp_frm + sp_to + sources_list + targets_list))
    node_index = {n: i for i, n in enumerate(all_nodes)}

    edges = [(node_index[f], node_index[t]) for f, t in zip(sp_frm, sp_to)]
    g = ig.Graph(n=len(all_nodes), edges=edges, directed=directed)
    mode = "out" if directed else "all"

    # Self-loop nodes: nodes with at least one self-referencing edge
    self_loop_nodes = {f for f, t in zip(sp_frm, sp_to) if f == t}

    # Group targets per source to batch igraph.distances() calls
    source_to_targets: dict = {}
    for src, tgt in zip(sources_list, targets_list):
        source_to_targets.setdefault(src, []).append(tgt)

    rows: list = []
    for src, tgts in source_to_targets.items():
        if src not in node_index:
            for tgt in tgts:
                rows.append((src, tgt, None))
            continue

        src_idx = node_index[src]
        tgt_idxs = [node_index.get(t, -1) for t in tgts]
        valid_idxs = [i for i in tgt_idxs if i >= 0]

        if valid_idxs:
            # distances() returns [[d(src,t0), d(src,t1), ...]]
            dist_row = g.distances(source=src_idx, target=valid_idxs, mode=mode)[0]
        else:
            dist_row = []

        valid_iter = iter(dist_row)
        for tgt, tgt_idx in zip(tgts, tgt_idxs):
            if tgt_idx < 0:
                rows.append((src, tgt, None))
            else:
                d = next(valid_iter)
                # igraph returns float('inf') for unreachable
                if d == float("inf") or d >= 2**31:
                    hops: Optional[int] = None
                elif d == 0 and min_hops >= 1 and src == tgt:
                    # igraph trivially returns 0 for src==target; when min_hops>=1
                    # the caller requires at least one edge traversal, so use the
                    # self-loop distance (1) if a self-loop exists, else unreachable.
                    if src in self_loop_nodes and (max_hops is None or max_hops >= 1):
                        hops = 1
                    else:
                        hops = None
                elif d < min_hops:
                    hops = None
                elif max_hops is not None and d > max_hops:
                    hops = None
                else:
                    hops = int(d)
                rows.append((src, tgt, hops))

    return pd.DataFrame(rows, columns=["__sp_source__", "__sp_target__", "__sp_hops__"])


def cugraph_shortest_path_distances(
    step_pairs: Any,
    sources: Any,
    targets: Any,
    *,
    min_hops: int = 1,
    max_hops: Optional[int],
    directed: bool,
) -> Any:
    """
    Compute pairwise shortest-path distances using cugraph.bfs().

    One BFS per unique source node; target lookup is a vectorized cuDF join.
    Returns a cuDF DataFrame with __sp_source__, __sp_target__, __sp_hops__.
    Unreachable, under-min-hops, or over-max-hops pairs get __sp_hops__ = None.
    """
    import cudf  # type: ignore[import]
    import cugraph  # type: ignore[import]

    # Build the graph from the filtered edge set
    edges_gdf = cudf.DataFrame({
        "src": step_pairs["__from__"],
        "dst": step_pairs["__to__"],
    })
    G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(edges_gdf, source="src", destination="dst")

    # Pair table: one row per (source, target) request — stays on GPU
    pair_df = cudf.DataFrame({"__sp_source__": sources, "__sp_target__": targets})

    # BFS once per unique source; collect distance slabs via cuDF joins
    unique_sources = pair_df["__sp_source__"].unique()
    slabs: list = []
    for src in unique_sources.to_arrow().tolist():
        try:
            # cugraph.bfs() does not accept 'directed' with a Graph object;
            # directionality is encoded in Graph(directed=...)
            bfs_result = cugraph.bfs(G, start=src)
        except Exception:
            # Source not in graph — all targets unreachable
            tgts = pair_df.loc[pair_df["__sp_source__"] == src, "__sp_target__"]
            slabs.append(cudf.DataFrame({
                "__sp_source__": cudf.Series([src] * len(tgts), dtype=pair_df["__sp_source__"].dtype),
                "__sp_target__": tgts.reset_index(drop=True),
                "__sp_hops__": cudf.Series([None] * len(tgts), dtype="Int64"),
            }))
            continue

        # bfs_result columns: vertex, distance, predecessor
        # Join to keep only the targets requested for this source
        tgts_for_src = pair_df.loc[pair_df["__sp_source__"] == src, ["__sp_target__"]]
        merged = tgts_for_src.merge(
            bfs_result[["vertex", "distance"]].rename(columns={"vertex": "__sp_target__", "distance": "__sp_hops__"}),
            on="__sp_target__",
            how="left",
        )
        merged["__sp_source__"] = src
        slabs.append(merged[["__sp_source__", "__sp_target__", "__sp_hops__"]])

    if not slabs:
        return cudf.DataFrame({"__sp_source__": [], "__sp_target__": [], "__sp_hops__": []})

    result = cudf.concat(slabs, ignore_index=True)

    # Apply hop bounds: unreachable (sentinel 2**31-1 or NA), under min, over max → None
    sentinel = 2 ** 31
    hops = result["__sp_hops__"]
    out_of_range = hops.isna() | (hops >= sentinel)
    if min_hops > 0:
        out_of_range = out_of_range | (hops < min_hops)
    if max_hops is not None:
        out_of_range = out_of_range | (hops > max_hops)
    result["__sp_hops__"] = hops.where(~out_of_range, other=None)

    return result


def try_native_shortest_path(
    step_pairs: Any,
    sources: Any,
    targets: Any,
    *,
    min_hops: int = 1,
    max_hops: Optional[int],
    directed: bool,
    engine: Any,
    backend: ShortestPathBackend = "auto",
) -> Optional[Any]:
    """
    Compute shortest-path distances using a native graph library.

    backend controls selection:
    - "auto"    : try cugraph on CUDF engine, igraph on PANDAS engine, return
                  None on failure so caller can fall back to BFS
    - "igraph"  : require igraph; raise ImportError if not available
    - "cugraph" : require cugraph; raise ImportError if not available
    - "bfs"     : skip native backends; return None so caller uses BFS

    Always logs at DEBUG which backend ran (or why it was skipped).
    """
    from graphistry.Engine import Engine as _Engine

    if backend == "bfs":
        logger.debug("shortestPath: backend=bfs, skipping native dispatch")
        return None

    if backend == "cugraph" or (backend == "auto" and engine == _Engine.CUDF):
        try:
            result = cugraph_shortest_path_distances(
                step_pairs, sources, targets,
                min_hops=min_hops, max_hops=max_hops, directed=directed,
            )
            logger.debug("shortestPath: backend=cugraph")
            return result
        except ImportError:
            if backend == "cugraph":
                raise
            logger.debug("shortestPath: cugraph not available, falling back to BFS")
            return None
        except Exception as e:
            if backend == "cugraph":
                raise
            logger.debug("shortestPath: cugraph failed (%s), falling back to BFS", e)
            return None

    if backend == "igraph" or backend == "auto":
        try:
            result = igraph_shortest_path_distances(
                step_pairs, sources, targets,
                min_hops=min_hops, max_hops=max_hops, directed=directed,
            )
            logger.debug("shortestPath: backend=igraph")
            return result
        except ImportError:
            if backend == "igraph":
                raise
            logger.debug("shortestPath: igraph not available, falling back to BFS")
            return None
        except Exception as e:
            if backend == "igraph":
                raise
            logger.debug("shortestPath: igraph failed (%s), falling back to BFS", e)
            return None

    return None
