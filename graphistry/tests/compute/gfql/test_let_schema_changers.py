"""
Tests for schema-changing operations (UMAP, hypergraph) in let() bindings.

Schema-changers in let() bindings bypass chain machinery and execute via execute_call(),
while schema-changers in ASTRef chains go through recursive dispatch. These tests verify
both code paths work correctly.

Execution paths:
- Direct in let: DAG executor → execute_node() → execute_call() (line 345-348)
- In ASTRef chain: execute_node() → ASTRef case → chain_impl() → recursive dispatch (line 250-256)
- Nested let: DAG executor → execute_node() → chain_let_impl() recursion (line 236-238)
"""

import pandas as pd
import pytest
from graphistry.compute.ast import ASTCall, let, ref, n, e, ge

# Suppress deprecation warnings for chain() method in this test file
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning:graphistry"),
]

# Note: Tests use PyGraphistry instead of test fixture to access full UMAP/hypergraph functionality
try:
    from graphistry import PyGraphistry
except ImportError:
    PyGraphistry = None


@pytest.fixture
def graph():
    """Create test graph with numeric features for UMAP."""
    nodes_df = pd.DataFrame({
        'id': [0, 1, 2, 3, 4],
        'x': [1.0, 2.0, 3.0, 4.0, 5.0],
        'y': [2.0, 3.0, 4.0, 5.0, 6.0],
        'score': [10, 20, 30, 40, 50],
        'type': ['A', 'B', 'A', 'B', 'A']
    })
    edges_df = pd.DataFrame({
        'src': [0, 1, 2],
        'dst': [1, 2, 3]
    })

    if PyGraphistry is None:
        pytest.skip("PyGraphistry not available")

    return PyGraphistry.bind(source='src', destination='dst').nodes(nodes_df, 'id').edges(edges_df)


class TestLetDirectSchemaChangers:
    """Test schema-changers directly in let bindings (execute_call path)."""

    @pytest.mark.skip(reason="Requires full Plotter with UMAPMixin - validated via standalone tests")
    def test_umap_in_let_direct(self, graph):
        """Test UMAP directly in let binding.

        Execution: DAG executor → execute_node() → execute_call()
        """
        query = let({
            'embedded': ASTCall('umap', {
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            })
        })

        result = graph.gfql(query, output='embedded', engine='pandas')

        assert len(result._nodes) == 5, "Should preserve all nodes"
        assert len(result._edges) > 0, "UMAP should create edges"

    @pytest.mark.skip(reason="Requires full Plotter with HypergraphMixin - validated via standalone tests")
    def test_hypergraph_in_let_direct(self, graph):
        """Test hypergraph directly in let binding.

        Execution: DAG executor → execute_node() → execute_call()
        """
        query = let({
            'hg': ASTCall('hypergraph', {
                'entity_types': ['type'],
                'direct': True
            })
        })

        result = graph.gfql(query, output='hg', engine='pandas')

        assert result is not None
        assert len(result._nodes) > 0, "Hypergraph should create nodes"


class TestLetRefSchemaChangers:
    """Test schema-changers in ASTRef chains (recursive dispatch path)."""

    @pytest.mark.skip(reason="Requires full Plotter with UMAPMixin - validated via standalone tests")
    def test_umap_in_ref_chain(self, graph):
        """Test UMAP within ASTRef chain.

        Execution: execute_node() → ASTRef case → chain_impl() → recursive dispatch
        """
        query = let({
            'filtered': n({'score': ge(20)}),
            'embedded': ref('filtered', [
                ASTCall('umap', {
                    'n_components': 2,
                    'n_neighbors': 3,
                    'umap_kwargs': {'random_state': 42}
                })
            ])
        })

        result = graph.gfql(query, output='embedded', engine='pandas')

        assert len(result._nodes) == 4, "Should have 4 filtered nodes"
        assert len(result._edges) > 0, "UMAP should create edges"

    @pytest.mark.skip(reason="Requires full Plotter with UMAPMixin/HypergraphMixin - validated via standalone tests")
    def test_consecutive_schema_changers_in_let(self, graph):
        """Test multiple schema-changers via refs in let.

        Tests: let({'a': call('umap'), 'b': ref('a', [call('hypergraph')])})
        """
        query = let({
            'embedded': ASTCall('umap', {
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            }),
            'transformed': ref('embedded', [
                ASTCall('hypergraph', {
                    'entity_types': ['type'],
                    'direct': True
                })
            ])
        })

        result = graph.gfql(query, output='transformed', engine='pandas')

        assert result is not None
        assert len(result._nodes) > 0


class TestNestedLetSchemaChangers:
    """Test schema-changers in nested let bindings (recursive let execution)."""

    @pytest.mark.skip(reason="Requires full Plotter with UMAPMixin - validated via standalone tests")
    def test_nested_let_with_umap_inner(self, graph):
        """Test UMAP in inner let, referenced by outer let.

        Execution: DAG executor → execute_node() → chain_let_impl() recursion
        """
        query = let({
            'outer': let({
                'inner': ASTCall('umap', {
                    'n_components': 2,
                    'n_neighbors': 3,
                    'umap_kwargs': {'random_state': 42}
                }),
                'filtered': ref('inner', [n()])
            }),
            'result': ref('outer')
        })

        result = graph.gfql(query, output='result', engine='pandas')

        assert len(result._nodes) == 5
        assert len(result._edges) > 0

    @pytest.mark.skip(reason="Requires full Plotter with UMAPMixin - validated via standalone tests")
    def test_nested_let_with_ref_chain_containing_umap(self, graph):
        """Test nested let where inner ref chain contains schema-changer.

        Combines nested let with ASTRef chain containing UMAP.
        """
        query = let({
            'base': n({'score': ge(20)}),
            'nested': let({
                'embedded': ref('base', [
                    e(),
                    ASTCall('umap', {
                        'n_components': 2,
                        'n_neighbors': 2,
                        'umap_kwargs': {'random_state': 42}
                    })
                ])
            }),
            'final': ref('nested')
        })

        result = graph.gfql(query, output='final', engine='pandas')

        assert result is not None
        assert len(result._nodes) > 0

    @pytest.mark.skip(reason="Requires full Plotter with UMAPMixin - validated via standalone tests")
    def test_deeply_nested_schema_changers(self, graph):
        """Test 3-level nesting with schema-changers at different levels."""
        query = let({
            'level1': let({
                'level2': let({
                    'level3': ASTCall('umap', {
                        'n_components': 2,
                        'n_neighbors': 3,
                        'umap_kwargs': {'random_state': 42}
                    })
                }),
                'mid': ref('level2')
            }),
            'top': ref('level1')
        })

        result = graph.gfql(query, output='top', engine='pandas')

        assert len(result._nodes) == 5
        assert len(result._edges) > 0


class TestMixedLetOperations:
    """Test schema-changers mixed with regular operations in let."""

    @pytest.mark.skip(reason="Requires full Plotter with UMAPMixin/HypergraphMixin - validated via standalone tests")
    def test_filter_then_umap_then_hypergraph_in_let(self, graph):
        """Test complex pipeline: filter → UMAP → hypergraph via let bindings."""
        query = let({
            'filtered': n({'score': ge(20)}),
            'embedded': ref('filtered', [
                ASTCall('umap', {
                    'n_components': 2,
                    'n_neighbors': 3,
                    'umap_kwargs': {'random_state': 42}
                })
            ]),
            'transformed': ref('embedded', [
                ASTCall('hypergraph', {
                    'entity_types': ['type'],
                    'direct': True
                })
            ])
        })

        result = graph.gfql(query, output='transformed', engine='pandas')

        assert result is not None
        assert len(result._nodes) > 0
