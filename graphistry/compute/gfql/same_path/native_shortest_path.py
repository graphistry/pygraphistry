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

from typing import Any, Optional

import pandas as pd


def igraph_shortest_path_distances(
    step_pairs: Any,
    sources: Any,
    targets: Any,
    *,
    max_hops: Optional[int],
    directed: bool,
) -> pd.DataFrame:
    """
    Compute pairwise shortest-path distances using igraph.distances().

    Batches all targets per unique source to avoid redundant BFS.
    Returns a pandas DataFrame with __sp_source__, __sp_target__, __sp_hops__.
    Unreachable or over-max-hops pairs get __sp_hops__ = None.
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

    # Group targets per source to batch igraph.distances() calls
    source_to_targets: dict = {}
    for src, tgt in zip(sources_list, targets_list):
        source_to_targets.setdefault(src, []).append(tgt)

    rows = []
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
    max_hops: Optional[int],
    directed: bool,
) -> Any:
    """
    Compute pairwise shortest-path distances using cugraph.bfs().

    One BFS per unique source node. Results are filtered to requested targets.
    Returns a cuDF DataFrame with __sp_source__, __sp_target__, __sp_hops__.
    Unreachable or over-max-hops pairs get __sp_hops__ = None.
    """
    import cudf  # type: ignore[import]
    import cugraph  # type: ignore[import]

    sources_list = list(sources)
    targets_list = list(targets)

    edges_gdf = cudf.DataFrame({
        "src": step_pairs["__from__"],
        "dst": step_pairs["__to__"],
    })
    G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(edges_gdf, source="src", destination="dst")

    source_to_targets: dict = {}
    for src, tgt in zip(sources_list, targets_list):
        source_to_targets.setdefault(src, []).append(tgt)

    rows_src: list = []
    rows_tgt: list = []
    rows_hops: list = []

    for src, tgts in source_to_targets.items():
        try:
            bfs_result = cugraph.bfs(G, start=src, directed=directed)
        except Exception:
            rows_src.extend([src] * len(tgts))
            rows_tgt.extend(tgts)
            rows_hops.extend([None] * len(tgts))
            continue

        # bfs_result columns: vertex, distance, predecessor
        dist_df = bfs_result[bfs_result["vertex"].isin(tgts)][["vertex", "distance"]].to_pandas()
        dist_map = dict(zip(dist_df["vertex"], dist_df["distance"]))

        for tgt in tgts:
            d = dist_map.get(tgt)
            if d is None or d >= 2**31:
                hops: Optional[int] = None
            elif max_hops is not None and d > max_hops:
                hops = None
            else:
                hops = int(d)
            rows_src.append(src)
            rows_tgt.append(tgt)
            rows_hops.append(hops)

    return cudf.DataFrame({
        "__sp_source__": rows_src,
        "__sp_target__": rows_tgt,
        "__sp_hops__": rows_hops,
    })


def try_native_shortest_path(
    step_pairs: Any,
    sources: Any,
    targets: Any,
    *,
    max_hops: Optional[int],
    directed: bool,
    engine: Any,
) -> Optional[Any]:
    """
    Attempt to compute shortest-path distances using a native graph library.

    Returns a DataFrame with __sp_source__, __sp_target__, __sp_hops__ on
    success, or None if no native backend is available (caller falls back to BFS).

    Dispatch:
    - Engine.CUDF   → cugraph BFS
    - Engine.PANDAS → igraph distances
    """
    from graphistry.Engine import Engine as _Engine

    if engine == _Engine.CUDF:
        try:
            return cugraph_shortest_path_distances(
                step_pairs, sources, targets,
                max_hops=max_hops, directed=directed,
            )
        except Exception:
            return None

    try:
        return igraph_shortest_path_distances(
            step_pairs, sources, targets,
            max_hops=max_hops, directed=directed,
        )
    except Exception:
        return None
