"""
Tests for chain.py combine_steps() concat operations.

This test module ensures proper DataFrame engine handling in chain operations,
particularly when chaining with schema-changing operations like UMAP.

Related issues: #777
"""

import os
import pytest
import pandas as pd
import graphistry

from graphistry.compute.ast import n, e, call
from graphistry.tests.test_compute import CGFull
from graphistry.utils.lazy_import import lazy_umap_import

# Skip GPU tests if TEST_CUDF not set
skip_gpu = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1"
)

# Check if UMAP is available (AI dependency, not minimal)
has_umap, _, _ = lazy_umap_import()


class TestChainCombineSteps:
    """Test combine_steps() function with different DataFrame engines"""

    def test_chain_node_filter_pandas(self):
        """Test basic node filter chain with pandas (CPU regression test)"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'type': ['person', 'person', 'org', 'org', 'person'],
            'score': [10, 20, 30, 40, 50]
        })
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e'],
            'rel': ['knows', 'works_for', 'partners', 'knows']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Chain with node filters - exercises combine_steps()
        result = g.chain([
            n({}),  # Identity filter
            e(),    # Edge filter
            n({})   # Another identity filter
        ], engine='pandas')

        # Should complete without error
        assert result._nodes is not None
        assert isinstance(result._nodes, pd.DataFrame)
        assert result._edges is not None
        assert isinstance(result._edges, pd.DataFrame)

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_chain_umap_with_node_filters_pandas(self):
        """Test UMAP with node filters in pandas mode (CPU baseline)"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        # Use graphistry.Plotter (not CGFull) since UMAP requires UMAPMixin
        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Chain: Node filter → UMAP → Node filter
        # This is the pattern from issue #777
        result = g.chain([
            n({}),  # Identity node filter
            call('umap', {
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            }),
            n({})  # Another identity filter
        ], engine='pandas')

        # Should complete successfully
        assert result._nodes is not None
        assert isinstance(result._nodes, pd.DataFrame)
        # UMAP creates new node embeddings, may have 0 edges
        assert result._edges is not None

    @skip_gpu
    def test_chain_node_filter_cudf(self):
        """Test basic node filter chain with cuDF (GPU test)"""
        import cudf

        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'type': ['person', 'person', 'org', 'org', 'person'],
            'score': [10, 20, 30, 40, 50]
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e'],
            'rel': ['knows', 'works_for', 'partners', 'knows']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Verify we're starting with cuDF
        assert isinstance(g._nodes, cudf.DataFrame)
        assert isinstance(g._edges, cudf.DataFrame)

        # Chain with node filters - exercises combine_steps() with cuDF
        result = g.chain([
            n({}),  # Identity filter
            e(),    # Edge filter
            n({})   # Another identity filter
        ], engine='cudf')

        # Should complete without error and preserve cuDF
        assert result._nodes is not None
        assert isinstance(result._nodes, cudf.DataFrame), \
            f"Expected cudf.DataFrame for nodes, got {type(result._nodes)}"
        assert result._edges is not None
        assert isinstance(result._edges, cudf.DataFrame), \
            f"Expected cudf.DataFrame for edges, got {type(result._edges)}"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_chain_umap_with_node_filters_cudf(self):
        """Test UMAP with node filters in cuDF mode (GPU test for issue #777)"""
        import cudf

        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        # Use graphistry.Plotter (not CGFull) since UMAP requires UMAPMixin
        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Verify we're starting with cuDF
        assert isinstance(g._nodes, cudf.DataFrame)
        assert isinstance(g._edges, cudf.DataFrame)

        # Chain: Node filter → UMAP → Node filter
        # This is the exact pattern from issue #777
        result = g.chain([
            n({}),  # Identity node filter
            call('umap', {
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            }),
            n({})  # Another identity filter
        ], engine='cudf')

        # Should complete without TypeError
        # This was failing with: "cannot concatenate object of type '<class 'cudf.core.dataframe.DataFrame'>'"
        assert result._nodes is not None
        assert isinstance(result._nodes, cudf.DataFrame), \
            f"Expected cudf.DataFrame for nodes, got {type(result._nodes)}"
        assert result._edges is not None
        # After UMAP schema change, edges should still be cuDF

    @skip_gpu
    def test_combine_steps_with_cudf_dataframes(self):
        """Direct test of combine_steps path with cuDF DataFrames"""
        import cudf

        # Create a graph with multiple operations that will go through combine_steps
        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'category': ['X', 'Y', 'X', 'Y']
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c'],
            'dst': ['b', 'c', 'd']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Chain that exercises combine_steps with multiple steps
        result = g.chain([
            n({'category': 'X'}, name='step1'),
            e(),
            n({}, name='step2'),
            e(),
            n({}, name='step3')
        ], engine='cudf')

        # combine_steps should handle cuDF concat correctly
        assert result._nodes is not None
        assert isinstance(result._nodes, cudf.DataFrame)
        # Should have boolean columns for named steps
        assert 'step1' in result._nodes.columns
        assert 'step2' in result._nodes.columns
        assert 'step3' in result._nodes.columns

    def test_combine_steps_edge_concatenation_pandas(self):
        """Test combine_steps edge concatenation with pandas"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e']
        })
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Chain with multiple edge operations
        result = g.chain([
            n(),
            e(name='e1'),
            n(),
            e(name='e2')
        ], engine='pandas')

        # Should concatenate edges correctly
        assert result._edges is not None
        assert isinstance(result._edges, pd.DataFrame)
        assert 'e1' in result._edges.columns
        assert 'e2' in result._edges.columns

    @skip_gpu
    def test_combine_steps_edge_concatenation_cudf(self):
        """Test combine_steps edge concatenation with cuDF"""
        import cudf

        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e']
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Chain with multiple edge operations
        result = g.chain([
            n(),
            e(name='e1'),
            n(),
            e(name='e2')
        ], engine='cudf')

        # Should concatenate edges correctly with cuDF
        assert result._edges is not None
        assert isinstance(result._edges, cudf.DataFrame)
        assert 'e1' in result._edges.columns
        assert 'e2' in result._edges.columns

    @skip_gpu
    def test_engine_honors_cudf_request_with_cudf_input(self):
        """Test that engine='cudf' request returns cuDF when inputs are cuDF"""
        import cudf

        nodes_df = cudf.DataFrame({'id': ['a', 'b', 'c']})
        edges_df = cudf.DataFrame({'src': ['a', 'b'], 'dst': ['b', 'c']})

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        result = g.chain([n(), e(), n()], engine='cudf')

        # Output should be cuDF (honoring the request)
        assert isinstance(result._nodes, cudf.DataFrame), \
            f"engine='cudf' should return cuDF nodes, got {type(result._nodes)}"
        assert isinstance(result._edges, cudf.DataFrame), \
            f"engine='cudf' should return cuDF edges, got {type(result._edges)}"

    def test_engine_honors_pandas_request_with_pandas_input(self):
        """Test that engine='pandas' request returns pandas when inputs are pandas"""
        nodes_df = pd.DataFrame({'id': ['a', 'b', 'c']})
        edges_df = pd.DataFrame({'src': ['a', 'b'], 'dst': ['b', 'c']})

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        result = g.chain([n(), e(), n()], engine='pandas')

        # Output should be pandas (honoring the request)
        assert isinstance(result._nodes, pd.DataFrame), \
            f"engine='pandas' should return pandas nodes, got {type(result._nodes)}"
        assert isinstance(result._edges, pd.DataFrame), \
            f"engine='pandas' should return pandas edges, got {type(result._edges)}"

    @skip_gpu
    def test_engine_coerces_cudf_to_pandas(self):
        """Test that engine='pandas' converts cuDF inputs to pandas (cross-engine coercion)"""
        import cudf

        nodes_df = cudf.DataFrame({'id': ['a', 'b', 'c']})
        edges_df = cudf.DataFrame({'src': ['a', 'b'], 'dst': ['b', 'c']})

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Verify we're starting with cuDF
        assert isinstance(g._nodes, cudf.DataFrame)
        assert isinstance(g._edges, cudf.DataFrame)

        # Request pandas engine - should coerce cuDF to pandas
        result = g.chain([n(), e(), n()], engine='pandas')

        # Output should be pandas (coerced from cuDF)
        assert isinstance(result._nodes, pd.DataFrame), \
            f"engine='pandas' with cuDF input should return pandas nodes, got {type(result._nodes)}"
        assert isinstance(result._edges, pd.DataFrame), \
            f"engine='pandas' with cuDF input should return pandas edges, got {type(result._edges)}"

    @skip_gpu
    def test_engine_coerces_pandas_to_cudf(self):
        """Test that engine='cudf' converts pandas inputs to cuDF (cross-engine coercion)"""
        import cudf

        nodes_df = pd.DataFrame({'id': ['a', 'b', 'c']})
        edges_df = pd.DataFrame({'src': ['a', 'b'], 'dst': ['b', 'c']})

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Verify we're starting with pandas
        assert isinstance(g._nodes, pd.DataFrame)
        assert isinstance(g._edges, pd.DataFrame)

        # Request cudf engine - should coerce pandas to cuDF
        result = g.chain([n(), e(), n()], engine='cudf')

        # Output should be cuDF (coerced from pandas)
        assert isinstance(result._nodes, cudf.DataFrame), \
            f"engine='cudf' with pandas input should return cuDF nodes, got {type(result._nodes)}"
        assert isinstance(result._edges, cudf.DataFrame), \
            f"engine='cudf' with pandas input should return cuDF edges, got {type(result._edges)}"

    # UMAP Cross-Engine Coercion Tests - Testing 3D matrix: input type x UMAP engine x chain engine

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_auto_coerces_pandas_to_cudf(self):
        """UMAP(engine='auto') with pandas input → chain(engine='cudf') → cuDF output"""
        import cudf

        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
        assert isinstance(g._nodes, pd.DataFrame)

        # UMAP with engine='auto' may pick cuML/umap_learn internally
        # But chain engine='cudf' should force cuDF output
        result = g.chain([
            n({}),
            call('umap', {
                'engine': 'auto',
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            }),
            n({})
        ], engine='cudf')

        assert isinstance(result._nodes, cudf.DataFrame), \
            f"chain(engine='cudf') should return cuDF nodes, got {type(result._nodes)}"
        assert isinstance(result._edges, cudf.DataFrame), \
            f"chain(engine='cudf') should return cuDF edges, got {type(result._edges)}"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_cuml_coerces_pandas_to_cudf(self):
        """UMAP(engine='cuml') with pandas input → chain(engine='cudf') → cuDF output"""
        import cudf

        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # UMAP with explicit engine='cuml' produces cuDF internally
        # Chain engine='cudf' should preserve cuDF output
        result = g.chain([
            n({}),
            call('umap', {
                'engine': 'cuml',
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            }),
            n({})
        ], engine='cudf')

        assert isinstance(result._nodes, cudf.DataFrame), \
            f"chain(engine='cudf') should return cuDF nodes, got {type(result._nodes)}"
        assert isinstance(result._edges, cudf.DataFrame), \
            f"chain(engine='cudf') should return cuDF edges, got {type(result._edges)}"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_umap_learn_coerces_pandas_to_cudf(self):
        """UMAP(engine='umap_learn') with pandas input → chain(engine='cudf') → cuDF output"""
        import cudf

        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # UMAP with engine='umap_learn' produces pandas internally
        # But chain engine='cudf' should convert to cuDF output
        result = g.chain([
            n({}),
            call('umap', {
                'engine': 'umap_learn',
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            }),
            n({})
        ], engine='cudf')

        assert isinstance(result._nodes, cudf.DataFrame), \
            f"chain(engine='cudf') should return cuDF nodes, got {type(result._nodes)}"
        assert isinstance(result._edges, cudf.DataFrame), \
            f"chain(engine='cudf') should return cuDF edges, got {type(result._edges)}"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_auto_coerces_cudf_to_pandas(self):
        """UMAP(engine='auto') with cuDF input → chain(engine='pandas') → pandas output"""
        import cudf

        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
        assert isinstance(g._nodes, cudf.DataFrame)

        # UMAP with engine='auto' may pick cuML/umap_learn internally
        # But chain engine='pandas' should force pandas output
        result = g.chain([
            n({}),
            call('umap', {
                'engine': 'auto',
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            }),
            n({})
        ], engine='pandas')

        assert isinstance(result._nodes, pd.DataFrame), \
            f"chain(engine='pandas') should return pandas nodes, got {type(result._nodes)}"
        assert isinstance(result._edges, pd.DataFrame), \
            f"chain(engine='pandas') should return pandas edges, got {type(result._edges)}"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_cuml_coerces_cudf_to_pandas(self):
        """UMAP(engine='cuml') with cuDF input → chain(engine='pandas') → pandas output"""
        import cudf

        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # UMAP with engine='cuml' produces cuDF internally
        # But chain engine='pandas' should convert to pandas output
        result = g.chain([
            n({}),
            call('umap', {
                'engine': 'cuml',
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            }),
            n({})
        ], engine='pandas')

        assert isinstance(result._nodes, pd.DataFrame), \
            f"chain(engine='pandas') should return pandas nodes, got {type(result._nodes)}"
        assert isinstance(result._edges, pd.DataFrame), \
            f"chain(engine='pandas') should return pandas edges, got {type(result._edges)}"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_umap_learn_coerces_cudf_to_pandas(self):
        """UMAP(engine='umap_learn') with cuDF input → chain(engine='pandas') → pandas output"""
        import cudf

        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # UMAP with engine='umap_learn' produces pandas internally
        # Chain engine='pandas' should preserve pandas output
        result = g.chain([
            n({}),
            call('umap', {
                'engine': 'umap_learn',
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            }),
            n({})
        ], engine='pandas')

        assert isinstance(result._nodes, pd.DataFrame), \
            f"chain(engine='pandas') should return pandas nodes, got {type(result._nodes)}"
        assert isinstance(result._edges, pd.DataFrame), \
            f"chain(engine='pandas') should return pandas edges, got {type(result._edges)}"
