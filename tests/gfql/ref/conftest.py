"""Shared test fixtures and helpers for GFQL ref tests."""

import os
import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute.ast import ASTEdge
from graphistry.compute.gfql.df_executor import (
    build_same_path_inputs,
    DFSamePathExecutor,
)
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
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


def _has_multihop(chain) -> bool:
    """Check if chain has any multi-hop edges (oracle doesn't support multi-hop + WHERE)."""
    for op in chain:
        if isinstance(op, ASTEdge):
            min_h = op.min_hops if op.min_hops is not None else (op.hops if isinstance(op.hops, int) else 1)
            max_h = op.max_hops if op.max_hops is not None else (op.hops if isinstance(op.hops, int) else min_h)
            if min_h != 1 or max_h != 1:
                return True
    return False


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


def assert_executor_parity(graph, chain, where):
    """Assert executor parity with oracle. Tests pandas, and cudf if TEST_CUDF=1.

    For multi-hop + WHERE, oracle comparison is skipped (oracle doesn't support it).
    We just verify the executor runs and produces valid output.
    """
    inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
    executor = DFSamePathExecutor(inputs)
    executor._forward()
    result = executor._run_native()

    assert result._nodes is not None and result._edges is not None

    # Oracle doesn't support multi-hop + WHERE, skip comparison
    if where and _has_multihop(chain):
        # Just verify executor produced valid output
        assert "id" in result._nodes.columns
        assert "src" in result._edges.columns
        assert "dst" in result._edges.columns
        return

    oracle = enumerate_chain(
        graph,
        chain,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=50, max_edges=50),
    )
    assert set(result._nodes["id"]) == set(oracle.nodes["id"]), \
        f"pandas nodes mismatch: got {set(result._nodes['id'])}, expected {set(oracle.nodes['id'])}"
    assert set(result._edges["src"]) == set(oracle.edges["src"])
    assert set(result._edges["dst"]) == set(oracle.edges["dst"])

    if not TEST_CUDF:
        return

    import cudf  # type: ignore

    cudf_nodes = cudf.DataFrame(graph._nodes)
    cudf_edges = cudf.DataFrame(graph._edges)
    cudf_graph = CGFull().nodes(cudf_nodes, graph._node).edges(cudf_edges, graph._source, graph._destination)

    cudf_inputs = build_same_path_inputs(cudf_graph, chain, where, Engine.CUDF)
    cudf_executor = DFSamePathExecutor(cudf_inputs)
    cudf_executor._forward()
    cudf_result = cudf_executor._run_native()

    assert cudf_result._nodes is not None and cudf_result._edges is not None
    assert set(cudf_result._nodes["id"].to_pandas()) == set(oracle.nodes["id"]), \
        f"cudf nodes mismatch: got {set(cudf_result._nodes['id'].to_pandas())}, expected {set(oracle.nodes['id'])}"
    assert set(cudf_result._edges["src"].to_pandas()) == set(oracle.edges["src"])
    assert set(cudf_result._edges["dst"].to_pandas()) == set(oracle.edges["dst"])


# Backwards compatibility aliases
_make_graph = make_simple_graph
_make_hop_graph = make_hop_graph
_assert_parity = assert_executor_parity


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
