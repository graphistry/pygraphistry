"""Comprehensive tests for call() engine coercion.

Tests verify that call() honors the parent operation's engine parameter
when used in chain or let context, even when the called method (like UMAP)
changes DataFrame types mid-execution.

These tests complement test_chain_concat.py (which has comprehensive UMAP tests)
by focusing specifically on call() engine coercion integration.
"""

import os
import pandas as pd
import pytest
from graphistry.compute.ast import ASTLet, ASTRef, n, call
from graphistry.tests.test_compute import CGFull
from graphistry.utils.lazy_import import lazy_umap_import
import graphistry

# Skip GPU tests if TEST_CUDF not set
skip_gpu = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1"
)

# Check if UMAP is available
has_umap, _, _ = lazy_umap_import()

# Conditionally import cuDF for GPU tests
try:
    import cudf
    has_cudf = True
except ImportError:
    has_cudf = False


def make_test_graph_pandas(node_count=15):
    """Create pandas test graph with sufficient nodes for UMAP (needs 10+)."""
    nodes_df = pd.DataFrame({
        'id': range(node_count),
        'x': [i * 0.1 for i in range(node_count)],
        'y': [i * 0.2 for i in range(node_count)],
        'category': [f'cat_{i % 3}' for i in range(node_count)]
    })
    edges_df = pd.DataFrame({
        'src': [i for i in range(node_count - 1)],
        'dst': [i + 1 for i in range(node_count - 1)],
        'weight': [1.0] * (node_count - 1)
    })
    return CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')


def make_test_graph_cudf(node_count=15):
    """Create cuDF test graph with sufficient nodes for UMAP (needs 10+)."""
    nodes_df = cudf.DataFrame({
        'id': range(node_count),
        'x': [i * 0.1 for i in range(node_count)],
        'y': [i * 0.2 for i in range(node_count)],
        'category': [f'cat_{i % 3}' for i in range(node_count)]
    })
    edges_df = cudf.DataFrame({
        'src': [i for i in range(node_count - 1)],
        'dst': [i + 1 for i in range(node_count - 1)],
        'weight': [1.0] * (node_count - 1)
    })
    return CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')


class TestCallEngineCoercionInChain:
    """Test suite for call() engine coercion in chain context."""

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_in_chain_pandas_to_pandas(self):
        """Test call(umap) in chain with pandas→pandas (no coercion needed)."""
        g = make_test_graph_pandas()

        # Chain with UMAP call - UMAP picks umap-learn on CPU → pandas
        result = g.gfql([
            n({}),
            call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'}),
            n({})
        ], engine='pandas')

        # Verify types match requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be pandas DataFrame"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_in_chain_pandas_to_cudf(self):
        """Test call(umap) in chain with pandas→cuDF coercion."""
        g = make_test_graph_pandas()

        # Chain with UMAP call - UMAP picks umap-learn → pandas, but chain wants cuDF
        result = g.gfql([
            n({}),
            call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'}),
            n({})
        ], engine='cudf')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be coerced to cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be coerced to cuDF DataFrame"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_in_chain_cudf_to_pandas(self):
        """Test call(umap) in chain with cuDF→pandas coercion."""
        g = make_test_graph_cudf()

        # Chain with UMAP call - UMAP picks cuML → cuDF, but chain wants pandas
        result = g.gfql([
            n({}),
            call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'}),
            n({})
        ], engine='pandas')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be coerced to pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be coerced to pandas DataFrame"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_in_chain_cudf_to_cudf(self):
        """Test call(umap) in chain with cuDF→cuDF (no coercion needed)."""
        g = make_test_graph_cudf()

        # Chain with UMAP call - UMAP picks cuML → cuDF
        result = g.gfql([
            n({}),
            call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'}),
            n({})
        ], engine='cudf')

        # Verify types match requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be cuDF DataFrame"


class TestCallEngineCoercionInLet:
    """Test suite for call() engine coercion in let context."""

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_in_let_pandas_to_pandas(self):
        """Test call(umap) in let with pandas→pandas (no coercion needed)."""
        g = make_test_graph_pandas()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})])
        })

        result = g.gfql(dag, engine='pandas', output='umap')

        # Verify types match requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be pandas DataFrame"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_in_let_pandas_to_cudf(self):
        """Test call(umap) in let with pandas→cuDF coercion."""
        g = make_test_graph_pandas()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})])
        })

        result = g.gfql(dag, engine='cudf', output='umap')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be coerced to cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be coerced to cuDF DataFrame"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_in_let_cudf_to_pandas(self):
        """Test call(umap) in let with cuDF→pandas coercion."""
        g = make_test_graph_cudf()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})])
        })

        result = g.gfql(dag, engine='pandas', output='umap')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be coerced to pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be coerced to pandas DataFrame"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_in_let_cudf_to_cudf(self):
        """Test call(umap) in let with cuDF→cuDF (no coercion needed)."""
        g = make_test_graph_cudf()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})])
        })

        result = g.gfql(dag, engine='cudf', output='umap')

        # Verify types match requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be cuDF DataFrame"


class TestCallEngineCoercionExplicitEngines:
    """Test call() engine coercion with explicit UMAP engine parameters."""

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_cuml_with_pandas_request(self):
        """Test call(umap, engine='cuml') with pandas chain request (coercion)."""
        g = make_test_graph_pandas()

        # UMAP with engine='cuml' → cuDF, but chain wants pandas
        result = g.gfql([
            n({}),
            call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'cuml'}),
            n({})
        ], engine='pandas')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be coerced to pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be coerced to pandas DataFrame"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_cuml_with_cudf_request(self):
        """Test call(umap, engine='cuml') with cuDF chain request (no coercion)."""
        g = make_test_graph_pandas()

        # UMAP with engine='cuml' → cuDF, chain wants cuDF
        result = g.gfql([
            n({}),
            call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'cuml'}),
            n({})
        ], engine='cudf')

        # Verify types match requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be cuDF DataFrame"

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_umap_learn_with_pandas_request(self):
        """Test call(umap, engine='umap_learn') with pandas chain request (no coercion)."""
        g = make_test_graph_pandas()

        # UMAP with engine='umap_learn' → pandas
        result = g.gfql([
            n({}),
            call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'umap_learn'}),
            n({})
        ], engine='pandas')

        # Verify types match requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be pandas DataFrame"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_call_umap_learn_with_cudf_request(self):
        """Test call(umap, engine='umap_learn') with cuDF chain request (coercion)."""
        g = make_test_graph_pandas()

        # UMAP with engine='umap_learn' → pandas, but chain wants cuDF
        result = g.gfql([
            n({}),
            call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'umap_learn'}),
            n({})
        ], engine='cudf')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be coerced to cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be coerced to cuDF DataFrame"


class TestCallEngineCoercionMultipleCalls:
    """Test call() engine coercion with multiple calls in sequence."""

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_multiple_calls_in_chain_pandas(self):
        """Test multiple call() operations in chain with pandas."""
        g = make_test_graph_pandas()

        # Multiple UMAP calls in sequence
        result = g.gfql([
            n({}),
            call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'}),
            n({}),
            call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'}),
            n({})
        ], engine='pandas')

        # Verify types match requested engine after both calls
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas DataFrame after multiple calls"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be pandas DataFrame after multiple calls"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_multiple_calls_in_let_cudf(self):
        """Test multiple call() operations in let with cuDF."""
        g = make_test_graph_cudf()

        dag = ASTLet({
            'start': n({}),
            'umap1': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})]),
            'umap2': ASTRef('umap1', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})])
        })

        result = g.gfql(dag, engine='cudf', output='umap2')

        # Verify types match requested engine after both calls
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be cuDF DataFrame after multiple calls"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be cuDF DataFrame after multiple calls"
