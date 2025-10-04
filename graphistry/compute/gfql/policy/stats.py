"""Graph statistics extraction for policy decisions."""

from typing import TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable


class GraphStats(TypedDict, total=False):
    """Statistics about a graph for policy decisions.

    All fields are optional as extraction may fail for various DataFrame types.
    """
    nodes: int  # Number of nodes
    edges: int  # Number of edges
    node_bytes: int  # Memory usage of node DataFrame
    edge_bytes: int  # Memory usage of edge DataFrame


def extract_graph_stats(g: 'Plottable') -> GraphStats:
    """Extract statistics from a Plottable safely across all DataFrame types.

    Handles pandas, cudf, dask, and dask-cudf DataFrames gracefully.

    Args:
        g: Plottable instance to extract stats from

    Returns:
        GraphStats with node/edge counts and memory estimates.
        Returns empty dict on any failure to avoid breaking queries.
    """
    stats: GraphStats = {}

    def safe_len(df) -> int:
        """Get length safely for any DataFrame type."""
        if df is None:
            return 0
        try:
            # Works for pandas, cudf, and most DataFrames
            return len(df)
        except Exception:
            # Fallback for dask - use persist() then compute()
            try:
                if hasattr(df, 'persist'):
                    # For dask DataFrames - persist() caches the computation
                    persisted = df.persist()
                    return len(persisted.compute())
                elif hasattr(df, 'compute'):
                    # Fallback if no persist method
                    return len(df.compute())
            except Exception:
                pass
        return 0

    def safe_memory(df) -> int:
        """Get memory usage safely, with estimates for dask."""
        if df is None:
            return 0

        try:
            # pandas/cudf have memory_usage
            if hasattr(df, 'memory_usage'):
                mem = df.memory_usage(deep=True)
                # memory_usage returns Series, sum it
                if hasattr(mem, 'sum'):
                    return int(mem.sum())
                return int(mem)
        except Exception:
            pass

        # For dask, estimate from dtypes and length
        try:
            if hasattr(df, 'dtypes'):
                # Rough estimate: 8 bytes per numeric, 50 per object
                bytes_per_row = sum(
                    8 if str(dtype).startswith(('int', 'float', 'uint', 'bool'))
                    else 50  # String/object columns
                    for dtype in df.dtypes
                )
                return safe_len(df) * bytes_per_row
        except Exception:
            pass

        # Last resort: just return row count * 100 as rough estimate
        return safe_len(df) * 100

    # Extract node stats
    try:
        if hasattr(g, '_nodes'):
            stats['nodes'] = safe_len(g._nodes)
            stats['node_bytes'] = safe_memory(g._nodes)
    except Exception:
        # Best effort - don't break on stats extraction
        pass

    # Extract edge stats
    try:
        if hasattr(g, '_edges'):
            stats['edges'] = safe_len(g._edges)
            stats['edge_bytes'] = safe_memory(g._edges)
    except Exception:
        # Best effort - don't break on stats extraction
        pass

    return stats
