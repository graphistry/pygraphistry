"""Shared test fixtures and helpers for GFQL ref tests."""

import os
import pandas as pd
import pytest

from graphistry.tests.test_compute import CGFull

# Environment variable to enable cudf parity testing (set in CI GPU tests)
TEST_CUDF = "TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"


def has_working_gpu() -> bool:
    """Check if cuDF is available AND GPU memory allocation works."""
    try:
        import cudf
        # Try to actually allocate GPU memory
        test_df = cudf.DataFrame({"x": [1, 2, 3]})
        _ = test_df["x"].sum()  # Force computation
        return True
    except Exception:
        return False


# Cache the result at module load time
_HAS_WORKING_GPU = None


def requires_gpu(func):
    """Decorator to skip tests if GPU is not available."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _HAS_WORKING_GPU
        if _HAS_WORKING_GPU is None:
            _HAS_WORKING_GPU = has_working_gpu()
        if not _HAS_WORKING_GPU:
            pytest.skip("GPU not available or cuDF cannot allocate memory")
        return func(*args, **kwargs)

    return wrapper


def make_simple_graph():
    """Create a simple account->user graph for basic tests."""
    nodes = pd.DataFrame(
        [
            {"id": "acct1", "type": "account", "owner_id": "user1", "score": 5},
            {"id": "acct2", "type": "account", "owner_id": "user2", "score": 9},
            {"id": "user1", "type": "user", "score": 7},
            {"id": "user2", "type": "user", "score": 3},
        ]
    )
    edges = pd.DataFrame(
        [
            {"src": "acct1", "dst": "user1"},
            {"src": "acct2", "dst": "user2"},
        ]
    )
    return CGFull().nodes(nodes, "id").edges(edges, "src", "dst")


def make_hop_graph():
    """Create a multi-hop graph for traversal tests."""
    nodes = pd.DataFrame(
        [
            {"id": "acct1", "type": "account", "owner_id": "u1", "score": 1},
            {"id": "user1", "type": "user", "owner_id": "u1", "score": 5},
            {"id": "user2", "type": "user", "owner_id": "u1", "score": 7},
            {"id": "acct2", "type": "account", "owner_id": "u1", "score": 9},
            {"id": "user3", "type": "user", "owner_id": "u3", "score": 2},
        ]
    )
    edges = pd.DataFrame(
        [
            {"src": "acct1", "dst": "user1"},
            {"src": "user1", "dst": "user2"},
            {"src": "user2", "dst": "acct2"},
            {"src": "acct1", "dst": "user3"},
        ]
    )
    return CGFull().nodes(nodes, "id").edges(edges, "src", "dst")


# Backwards compatibility aliases
_make_graph = make_simple_graph
_make_hop_graph = make_hop_graph


# =============================================================================
# Generic cuDF Parity Testing Infrastructure
# =============================================================================

def graph_to_cudf(g):
    """Convert a Plottable's DataFrames to cuDF. Returns new Plottable."""
    import cudf  # type: ignore
    cudf_nodes = cudf.DataFrame(g._nodes) if g._nodes is not None else None
    cudf_edges = cudf.DataFrame(g._edges) if g._edges is not None else None
    result = CGFull()
    if cudf_nodes is not None:
        result = result.nodes(cudf_nodes, g._node)
    if cudf_edges is not None:
        result = result.edges(cudf_edges, g._source, g._destination, edge=g._edge)
    return result


def to_node_set(df, col='id'):
    """Extract node IDs as a set, handling both pandas and cuDF."""
    if hasattr(df, 'to_pandas'):
        return set(df[col].to_pandas())
    return set(df[col])


def to_edge_set(df, src='src', dst='dst'):
    """Extract edges as set of tuples, handling both pandas and cuDF."""
    if hasattr(df, 'to_pandas'):
        df = df.to_pandas()
    return set(zip(df[src], df[dst]))


def _to_python(series_or_df_col):
    """
    Convert Series to Python-native for test assertions.

    Test-only helper - production code should use engine-agnostic DataFrame ops.
    """
    if hasattr(series_or_df_col, 'to_pandas'):
        return series_or_df_col.to_pandas()
    return series_or_df_col


def to_list(series_or_df_col):
    """Convert Series/column to list for test assertions."""
    return _to_python(series_or_df_col).tolist()


def to_set(series_or_df_col):
    """Convert Series/column to set for test assertions."""
    return set(_to_python(series_or_df_col))


# Determine which engines to test based on TEST_CUDF environment variable
_ENGINE_MODES = ['pandas']
if TEST_CUDF:
    _ENGINE_MODES.append('cudf')


@pytest.fixture(params=_ENGINE_MODES)
def engine_mode(request):
    """Parametrized fixture for engine mode: 'pandas' or 'cudf' (if TEST_CUDF=1)."""
    mode = request.param
    if mode == 'cudf':
        global _HAS_WORKING_GPU
        if _HAS_WORKING_GPU is None:
            _HAS_WORKING_GPU = has_working_gpu()
        if not _HAS_WORKING_GPU:
            pytest.skip("GPU not available or cuDF cannot allocate memory")
    return mode


def maybe_cudf(g, engine_mode):
    """Convert graph to cuDF if engine_mode is 'cudf', otherwise return as-is."""
    if engine_mode == 'cudf':
        return graph_to_cudf(g)
    return g
