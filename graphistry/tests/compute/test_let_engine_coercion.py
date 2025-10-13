"""Comprehensive tests for let() engine coercion.

Tests verify that gfql(ASTLet) honors the engine parameter even when
nested operations (like UMAP) change DataFrame types mid-execution.

Test Matrix:
- Input DataFrame Type: pandas/cuDF
- UMAP Engine: auto/cuml/umap_learn
- Let Engine Request: pandas/cudf

Each test verifies that BOTH nodes AND edges match the requested engine.
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


class TestLetEngineCoercion:
    """Test suite for chain_let() engine coercion."""

    def test_basic_let_pandas_to_pandas(self):
        """Test pandas→pandas let with no schema-changing operations."""
        g = make_test_graph_pandas()

        # Simple let with node filter
        dag = ASTLet({
            'filtered': n({'category': 'cat_0'})
        })

        result = g.gfql(dag, engine='pandas')

        # Verify types match requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be pandas DataFrame"

        # Verify data integrity
        assert len(result._nodes) > 0, "Should have filtered nodes"

    @skip_gpu
    def test_basic_let_cudf_to_cudf(self):
        """Test cuDF→cuDF let with no schema-changing operations."""
        g = make_test_graph_cudf()

        # Simple let with node filter
        dag = ASTLet({
            'filtered': n({'category': 'cat_0'})
        })

        result = g.gfql(dag, engine='cudf')

        # Verify types match requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be cuDF DataFrame"

        # Verify data integrity
        assert len(result._nodes) > 0, "Should have filtered nodes"

    @skip_gpu
    def test_basic_let_pandas_to_cudf(self):
        """Test pandas→cuDF coercion in let()."""
        g = make_test_graph_pandas()

        # Simple let with node filter
        dag = ASTLet({
            'filtered': n({'category': 'cat_0'})
        })

        result = g.gfql(dag, engine='cudf')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be coerced to cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be coerced to cuDF DataFrame"

        # Verify data integrity
        assert len(result._nodes) > 0, "Should have filtered nodes"

    @skip_gpu
    def test_basic_let_cudf_to_pandas(self):
        """Test cuDF→pandas coercion in let()."""
        g = make_test_graph_cudf()

        # Simple let with node filter
        dag = ASTLet({
            'filtered': n({'category': 'cat_0'})
        })

        result = g.gfql(dag, engine='pandas')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be coerced to pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be coerced to pandas DataFrame"

        # Verify data integrity
        assert len(result._nodes) > 0, "Should have filtered nodes"

    # ============================================================================
    # UMAP Tests: These are the critical tests for engine coercion
    # UMAP's engine parameter controls WHICH library (cuML vs umap-learn)
    # Let's engine parameter controls OUTPUT type (pandas vs cuDF)
    # ============================================================================

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_pandas_auto_request_pandas(self):
        """Test: pandas input, UMAP engine='auto', request pandas output.

        UMAP with engine='auto' picks umap-learn on CPU → returns pandas.
        Let engine='pandas' should preserve pandas (no conversion needed).
        """
        g = make_test_graph_pandas()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})])
        })

        result = g.gfql(dag, engine='pandas', output='umap')

        # Verify types match requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be pandas DataFrame"

        # Verify UMAP columns added
        assert 'x' in result._nodes.columns or '_x' in result._nodes.columns, "UMAP should add positioning columns"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_pandas_auto_request_cudf(self):
        """Test: pandas input, UMAP engine='auto', request cuDF output (COERCION).

        UMAP with engine='auto' picks umap-learn on CPU → returns pandas.
        Let engine='cudf' should convert pandas→cuDF.
        """
        g = make_test_graph_pandas()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})])
        })

        result = g.gfql(dag, engine='cudf', output='umap')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be coerced to cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be coerced to cuDF DataFrame"

        # Verify UMAP columns added
        node_cols = result._nodes.columns.to_list()
        assert 'x' in node_cols or '_x' in node_cols, "UMAP should add positioning columns"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_cudf_auto_request_pandas(self):
        """Test: cuDF input, UMAP engine='auto', request pandas output (COERCION).

        UMAP with engine='auto' on GPU picks cuML → returns cuDF.
        Let engine='pandas' should convert cuDF→pandas.
        """
        g = make_test_graph_cudf()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})])
        })

        result = g.gfql(dag, engine='pandas', output='umap')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be coerced to pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be coerced to pandas DataFrame"

        # Verify UMAP columns added
        assert 'x' in result._nodes.columns or '_x' in result._nodes.columns, "UMAP should add positioning columns"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_cudf_auto_request_cudf(self):
        """Test: cuDF input, UMAP engine='auto', request cuDF output.

        UMAP with engine='auto' on GPU picks cuML → returns cuDF.
        Let engine='cudf' should preserve cuDF (no conversion needed).
        """
        g = make_test_graph_cudf()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})])
        })

        result = g.gfql(dag, engine='cudf', output='umap')

        # Verify types match requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be cuDF DataFrame"

        # Verify UMAP columns added
        node_cols = result._nodes.columns.to_list()
        assert 'x' in node_cols or '_x' in node_cols, "UMAP should add positioning columns"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_pandas_cuml_request_pandas(self):
        """Test: pandas input, UMAP engine='cuml' (explicit GPU), request pandas (COERCION).

        UMAP with engine='cuml' converts to cuDF internally, computes on GPU → returns cuDF.
        Let engine='pandas' should convert cuDF→pandas.
        """
        g = make_test_graph_pandas()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'cuml'})])
        })

        result = g.gfql(dag, engine='pandas', output='umap')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be coerced to pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be coerced to pandas DataFrame"

        # Verify UMAP columns added
        assert 'x' in result._nodes.columns or '_x' in result._nodes.columns, "UMAP should add positioning columns"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_pandas_cuml_request_cudf(self):
        """Test: pandas input, UMAP engine='cuml' (explicit GPU), request cuDF.

        UMAP with engine='cuml' converts to cuDF internally, computes on GPU → returns cuDF.
        Let engine='cudf' should preserve cuDF (no conversion needed).
        """
        g = make_test_graph_pandas()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'cuml'})])
        })

        result = g.gfql(dag, engine='cudf', output='umap')

        # Verify types match requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be cuDF DataFrame"

        # Verify UMAP columns added
        node_cols = result._nodes.columns.to_list()
        assert 'x' in node_cols or '_x' in node_cols, "UMAP should add positioning columns"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_cudf_cuml_request_pandas(self):
        """Test: cuDF input, UMAP engine='cuml' (explicit GPU), request pandas (COERCION).

        UMAP with engine='cuml' on cuDF input → returns cuDF.
        Let engine='pandas' should convert cuDF→pandas.
        """
        g = make_test_graph_cudf()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'cuml'})])
        })

        result = g.gfql(dag, engine='pandas', output='umap')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be coerced to pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be coerced to pandas DataFrame"

        # Verify UMAP columns added
        assert 'x' in result._nodes.columns or '_x' in result._nodes.columns, "UMAP should add positioning columns"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_cudf_cuml_request_cudf(self):
        """Test: cuDF input, UMAP engine='cuml' (explicit GPU), request cuDF.

        UMAP with engine='cuml' on cuDF input → returns cuDF.
        Let engine='cudf' should preserve cuDF (no conversion needed).
        """
        g = make_test_graph_cudf()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'cuml'})])
        })

        result = g.gfql(dag, engine='cudf', output='umap')

        # Verify types match requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be cuDF DataFrame"

        # Verify UMAP columns added
        node_cols = result._nodes.columns.to_list()
        assert 'x' in node_cols or '_x' in node_cols, "UMAP should add positioning columns"

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_pandas_umap_learn_request_pandas(self):
        """Test: pandas input, UMAP engine='umap_learn' (explicit CPU), request pandas.

        UMAP with engine='umap_learn' → returns pandas.
        Let engine='pandas' should preserve pandas (no conversion needed).
        """
        g = make_test_graph_pandas()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'umap_learn'})])
        })

        result = g.gfql(dag, engine='pandas', output='umap')

        # Verify types match requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be pandas DataFrame"

        # Verify UMAP columns added
        assert 'x' in result._nodes.columns or '_x' in result._nodes.columns, "UMAP should add positioning columns"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_pandas_umap_learn_request_cudf(self):
        """Test: pandas input, UMAP engine='umap_learn' (explicit CPU), request cuDF (COERCION).

        UMAP with engine='umap_learn' → returns pandas.
        Let engine='cudf' should convert pandas→cuDF.
        """
        g = make_test_graph_pandas()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'umap_learn'})])
        })

        result = g.gfql(dag, engine='cudf', output='umap')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be coerced to cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be coerced to cuDF DataFrame"

        # Verify UMAP columns added
        node_cols = result._nodes.columns.to_list()
        assert 'x' in node_cols or '_x' in node_cols, "UMAP should add positioning columns"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_cudf_umap_learn_request_pandas(self):
        """Test: cuDF input, UMAP engine='umap_learn' (explicit CPU), request pandas.

        UMAP with engine='umap_learn' converts cuDF→pandas internally → returns pandas.
        Let engine='pandas' should preserve pandas (no conversion needed).
        """
        g = make_test_graph_cudf()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'umap_learn'})])
        })

        result = g.gfql(dag, engine='pandas', output='umap')

        # Verify types match requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be pandas DataFrame"

        # Verify UMAP columns added
        assert 'x' in result._nodes.columns or '_x' in result._nodes.columns, "UMAP should add positioning columns"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_umap_cudf_umap_learn_request_cudf(self):
        """Test: cuDF input, UMAP engine='umap_learn' (explicit CPU), request cuDF (COERCION).

        UMAP with engine='umap_learn' converts cuDF→pandas internally → returns pandas.
        Let engine='cudf' should convert pandas→cuDF.
        """
        g = make_test_graph_cudf()

        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'umap_learn'})])
        })

        result = g.gfql(dag, engine='cudf', output='umap')

        # Verify types coerced to requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be coerced to cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be coerced to cuDF DataFrame"

        # Verify UMAP columns added
        node_cols = result._nodes.columns.to_list()
        assert 'x' in node_cols or '_x' in node_cols, "UMAP should add positioning columns"

    # ============================================================================
    # Complex DAG Tests: Multiple bindings with dependencies
    # ============================================================================

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_complex_dag_with_umap_pandas(self):
        """Test complex DAG with multiple bindings and UMAP, request pandas."""
        g = make_test_graph_pandas()

        dag = ASTLet({
            'filtered': n({'category': 'cat_0'}),
            'positioned': ASTRef('filtered', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})]),
            'final': ASTRef('positioned', [n({})])
        })

        result = g.gfql(dag, engine='pandas', output='final')

        # Verify types match requested engine
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be pandas DataFrame"

        # Verify data integrity
        assert len(result._nodes) > 0, "Should have nodes after complex DAG"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_let_complex_dag_with_umap_cudf(self):
        """Test complex DAG with multiple bindings and UMAP, request cuDF."""
        g = make_test_graph_cudf()

        dag = ASTLet({
            'filtered': n({'category': 'cat_0'}),
            'positioned': ASTRef('filtered', [call('umap', {'n_components': 2, 'n_neighbors': 3, 'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 'engine': 'auto'})]),
            'final': ASTRef('positioned', [n({})])
        })

        result = g.gfql(dag, engine='cudf', output='final')

        # Verify types match requested engine
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be cuDF DataFrame"

        # Verify data integrity
        assert len(result._nodes) > 0, "Should have nodes after complex DAG"
