"""Memory preflight estimator for dgx GB10 (unified memory, ~119 GB, shared).

peak_gb(edges) estimates the worst-case resident memory for a cudf+cugraph run:
the edge frame (edges * 16 B: two int64 columns) plus ~2x for the cugraph CSR +
pagerank vectors. safe_run refuses to launch if peak exceeds the budget, so we
never repeat the 1.8B-edge crash (28.9 GB frame -> ~87 GB peak -> host OOM).
"""
HOST_GB = 119
DEFAULT_FLOOR_GB = 39  # leave this much for the OS + non-RMM (pandas/pinned) allocs


def peak_gb(edges: int, cugraph_factor: float = 3.0) -> float:
    """Worst-case GB: edge frame (edges*16B) scaled for the cugraph CSR build."""
    return edges * 16 * cugraph_factor / 1e9


def is_safe(edges: int, host_gb: int = HOST_GB, floor_gb: int = DEFAULT_FLOOR_GB) -> bool:
    return peak_gb(edges) <= (host_gb - floor_gb)


if __name__ == "__main__":
    import sys
    e = int(sys.argv[1])
    p = peak_gb(e)
    ok = is_safe(e)
    print(f"edges={e:,} peak~{p:.1f}GB budget={HOST_GB-DEFAULT_FLOOR_GB}GB -> {'SAFE' if ok else 'REFUSE'}")
    sys.exit(0 if ok else 3)
