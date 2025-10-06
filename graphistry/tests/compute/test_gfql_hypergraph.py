"""Tests for GFQL hypergraph support."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from graphistry.compute.ast import call, n, e, let, ref
from graphistry.compute.chain import Chain
from graphistry.compute.calls import hypergraph
from graphistry.tests.test_compute import CGFull


class TestGFQLHypergraph:
    """Test hypergraph operations through GFQL."""

    def test_hypergraph_typed_builder(self):
        """Test using the typed hypergraph() builder."""
        # Create events data
        events_df = pd.DataFrame({
            'user': ['alice', 'bob', 'alice', 'carol'],
            'product': ['laptop', 'phone', 'tablet', 'laptop'],
            'action': ['view', 'buy', 'view', 'buy'],
            'amount': [1000, 500, 300, 1000]
        })

        # Create node-only graph
        g = CGFull().nodes(events_df)

        # Use typed builder instead of call()
        result = g.gfql(
            hypergraph(
                entity_types=['user', 'product'],
                direct=True,
                engine='pandas'
            )
        )

        # Verify result has both nodes and edges
        assert result._nodes is not None, "Hypergraph should have nodes"
        assert result._edges is not None, "Hypergraph should have edges"
        assert len(result._nodes) > 0, "Hypergraph should have non-empty nodes"
        assert len(result._edges) > 0, "Hypergraph should have non-empty edges"

    def test_hypergraph_typed_builder_with_let(self):
        """Test typed builder in let/DAG context."""
        events_df = pd.DataFrame({
            'user': ['alice', 'bob', 'alice', 'carol'],
            'product': ['laptop', 'phone', 'tablet', 'laptop'],
            'type': ['person', 'person', 'person', 'person']
        })

        g = CGFull().nodes(events_df)

        # Use typed builder in DAG
        result = g.gfql(
            let({
                'hg': hypergraph(
                    entity_types=['user', 'product'],
                    direct=False,  # Keep hypernodes
                    drop_na=True
                ),
                'filtered': ref('hg', [n({'type': 'person'})])
            }),
            output='filtered'
        )

        assert result._nodes is not None

    def test_hypergraph_typed_builder_defaults(self):
        """Test typed builder with default parameters."""
        events_df = pd.DataFrame({
            'a': ['x', 'y'],
            'b': ['1', '2']
        })

        g = CGFull().nodes(events_df)

        # Minimal call with defaults
        result = g.gfql(hypergraph())

        assert result._nodes is not None
        assert result._edges is not None

    def test_hypergraph_top_level_call(self):
        """Test hypergraph as top-level call() - simple single operation."""
        # Create events data
        events_df = pd.DataFrame({
            'user': ['alice', 'bob', 'alice', 'carol'],
            'product': ['laptop', 'phone', 'tablet', 'laptop'],
            'action': ['view', 'buy', 'view', 'buy'],
            'amount': [1000, 500, 300, 1000]
        })

        # Create node-only graph
        g = CGFull().nodes(events_df)

        # Transform to hypergraph via top-level call
        result = g.gfql(
            call('hypergraph', {
                'entity_types': ['user', 'product'],
                'direct': True,
                'engine': 'pandas'
            })
        )

        # Verify result has both nodes and edges
        assert result._nodes is not None, "Hypergraph should have nodes"
        assert result._edges is not None, "Hypergraph should have edges"
        assert len(result._nodes) > 0, "Hypergraph should have non-empty nodes"
        assert len(result._edges) > 0, "Hypergraph should have non-empty edges"

    def test_hypergraph_basic_with_let(self):
        """Test basic hypergraph call through GFQL using let/DAG approach."""
        # Create events data
        events_df = pd.DataFrame({
            'user': ['alice', 'bob', 'alice', 'carol'],
            'product': ['laptop', 'phone', 'tablet', 'laptop'],
            'action': ['view', 'buy', 'view', 'buy'],
            'amount': [1000, 500, 300, 1000]
        })

        # Create node-only graph
        g = CGFull().nodes(events_df)

        # Transform to hypergraph via GFQL using let
        result = g.gfql(
            let({
                'hg': call('hypergraph', {
                    'entity_types': ['user', 'product'],
                    'direct': True,
                    'engine': 'pandas'
                })
            }),
            output='hg'  # Return the hypergraph result
        )

        # Verify result has both nodes and edges
        assert result._nodes is not None, "Hypergraph should have nodes"
        assert result._edges is not None, "Hypergraph should have edges"
        assert len(result._nodes) > 0, "Hypergraph should have non-empty nodes"
        assert len(result._edges) > 0, "Hypergraph should have non-empty edges"

    def test_hypergraph_rejected_when_mixed(self):
        """Test that hypergraph cannot be mixed with other operations in chains."""
        events_df = pd.DataFrame({
            'user': ['alice', 'bob'],
            'product': ['laptop', 'phone'],
            'type': ['person', 'person']
        })

        g = CGFull().nodes(events_df)

        # Mixing hypergraph with other operations should raise an error
        with pytest.raises(ValueError) as exc_info:
            g.gfql([
                n({'type': 'person'}),  # Filter operation
                call('hypergraph', {    # Transform operation
                    'entity_types': ['user', 'product'],
                    'direct': True
                })
            ])

        assert "cannot be mixed with other operations" in str(exc_info.value)
        assert "use hypergraph alone" in str(exc_info.value)

    def test_hypergraph_with_chaining_in_let(self):
        """Test chaining operations after hypergraph transformation using let."""
        events_df = pd.DataFrame({
            'user': ['alice', 'bob', 'alice', 'carol'],
            'product': ['laptop', 'phone', 'tablet', 'laptop'],
            'type': ['person', 'person', 'person', 'person'],
            'amount': [1000, 500, 300, 1000]
        })

        g = CGFull().nodes(events_df)

        # Use let to transform then filter
        result = g.gfql(
            let({
                'transformed': call('hypergraph', {
                    'entity_types': ['user', 'product'],
                    'direct': True,
                    'engine': 'pandas'
                }),
                'filtered': ref('transformed', [n({'type': 'person'})])
            }),
            output='filtered'
        )

        assert result._nodes is not None
        # The filtered result should have nodes

    def test_hypergraph_all_params(self):
        """Test hypergraph with all parameters using let."""
        events_df = pd.DataFrame({
            'entity1': ['a', 'b', 'c', None],
            'entity2': ['x', 'y', 'z', 'w'],
            'value': [1, 2, 3, 4]
        })

        g = CGFull().nodes(events_df)

        result = g.gfql(
            let({
                'hg': call('hypergraph', {
                    'entity_types': ['entity1', 'entity2'],
                    'opts': {},
                    'drop_na': True,
                    'drop_edge_attrs': False,
                    'verbose': False,
                    'direct': False,
                    'engine': 'pandas',
                    'npartitions': None,
                    'chunksize': None
                })
            }),
            output='hg'
        )

        assert result._nodes is not None
        assert result._edges is not None

    def test_hypergraph_no_entity_types(self):
        """Test hypergraph with no entity_types specified (use all columns)."""
        events_df = pd.DataFrame({
            'a': ['1', '2'],
            'b': ['x', 'y'],
            'c': [10, 20]
        })

        g = CGFull().nodes(events_df)

        result = g.gfql(
            let({
                'hg': call('hypergraph', {
                    'entity_types': None,  # Use all columns
                    'direct': True
                })
            }),
            output='hg'
        )

        assert result._nodes is not None
        assert result._edges is not None

    def test_hypergraph_error_no_nodes(self):
        """Test hypergraph fails gracefully with no nodes."""
        g = CGFull()  # No nodes

        with pytest.raises(Exception) as exc_info:
            g.gfql(
                let({
                    'hg': call('hypergraph', {
                        'entity_types': ['user', 'product']
                    })
                }),
                output='hg'
            )

        # Error could mention "nodes" or "edges" depending on the state
        error_msg = str(exc_info.value).lower()
        assert "nodes" in error_msg or "edges" in error_msg

    def test_hypergraph_error_invalid_entity_types(self):
        """Test hypergraph with invalid entity_types."""
        events_df = pd.DataFrame({
            'user': ['alice', 'bob'],
            'product': ['laptop', 'phone']
        })

        g = CGFull().nodes(events_df)

        # This should fail during hypergraph execution
        with pytest.raises(Exception):
            g.gfql(
                let({
                    'hg': call('hypergraph', {
                        'entity_types': ['nonexistent_column']
                    })
                }),
                output='hg'
            )

    def test_hypergraph_indirect_mode(self):
        """Test hypergraph with direct=False (keep hypernodes)."""
        events_df = pd.DataFrame({
            'user': ['alice', 'bob', 'alice'],
            'product': ['laptop', 'phone', 'tablet'],
            'session': ['s1', 's1', 's2']
        })

        g = CGFull().nodes(events_df)

        result = g.gfql(
            let({
                'hg': call('hypergraph', {
                    'entity_types': ['user', 'product', 'session'],
                    'direct': False,  # Keep hypernodes
                    'engine': 'pandas'
                })
            }),
            output='hg'
        )

        assert result._nodes is not None
        assert result._edges is not None
        # With direct=False, there should be hypernode entries

    def test_hypergraph_drop_na(self):
        """Test hypergraph with drop_na parameter."""
        events_df = pd.DataFrame({
            'user': ['alice', None, 'carol'],
            'product': ['laptop', 'phone', None],
            'value': [1, 2, 3]
        })

        g = CGFull().nodes(events_df)

        # With drop_na=True
        result_drop = g.gfql(
            let({
                'hg_drop': call('hypergraph', {
                    'entity_types': ['user', 'product'],
                    'drop_na': True,
                    'direct': True
                })
            }),
            output='hg_drop'
        )

        # With drop_na=False
        result_keep = g.gfql(
            let({
                'hg_keep': call('hypergraph', {
                    'entity_types': ['user', 'product'],
                    'drop_na': False,
                    'direct': True
                })
            }),
            output='hg_keep'
        )

        # Both should work, potentially with different sizes
        assert result_drop._nodes is not None
        assert result_keep._nodes is not None


class TestGFQLHypergraphRemote:
    """Test hypergraph operations through GFQL remote mode (mocked)."""

    @patch('graphistry.compute.chain_remote.chain_remote_generic')
    def test_hypergraph_remote_top_level_call(self, mock_chain_remote):
        """Test remote execution of top-level hypergraph call."""
        # Create mock response
        mock_result = CGFull().edges(
            pd.DataFrame({
                'src': ['alice', 'bob', 'alice'],
                'dst': ['laptop', 'phone', 'tablet']
            }),
            'src', 'dst'
        ).nodes(
            pd.DataFrame({
                'id': ['alice', 'bob', 'laptop', 'phone', 'tablet'],
                'type': ['user', 'user', 'product', 'product', 'product']
            }),
            'id'
        )
        mock_chain_remote.return_value = mock_result

        # Create test data
        events_df = pd.DataFrame({
            'user': ['alice', 'bob', 'alice'],
            'product': ['laptop', 'phone', 'tablet']
        })
        g = CGFull().nodes(events_df)

        # Execute remote hypergraph
        result = g.gfql_remote(
            call('hypergraph', {
                'entity_types': ['user', 'product'],
                'direct': True,
                'engine': 'pandas'
            })
        )

        # Verify result
        assert result is mock_result
        mock_chain_remote.assert_called_once()

        # Check that hypergraph call was passed correctly
        call_args = mock_chain_remote.call_args
        chain_arg = call_args[0][1]  # Second positional arg is chain

        # Debug what we actually received
        from graphistry.compute.ast import ASTCall

        # It should be an ASTCall object for single operations
        assert isinstance(chain_arg, ASTCall)
        assert chain_arg.function == 'hypergraph'
        assert chain_arg.params['entity_types'] == ['user', 'product']

    @patch('graphistry.compute.chain_remote.chain_remote_generic')
    def test_hypergraph_remote_with_let(self, mock_chain_remote):
        """Test remote execution of hypergraph within let/DAG."""
        # Create mock response
        mock_result = CGFull().edges(
            pd.DataFrame({
                'src': ['alice', 'bob'],
                'dst': ['laptop', 'phone']
            }),
            'src', 'dst'
        ).nodes(
            pd.DataFrame({
                'id': ['alice', 'laptop'],
                'type': ['user', 'product']
            }),
            'id'
        )
        mock_chain_remote.return_value = mock_result

        # Create test data
        events_df = pd.DataFrame({
            'user': ['alice', 'bob', 'alice', 'carol'],
            'product': ['laptop', 'phone', 'tablet', 'laptop'],
            'action': ['view', 'buy', 'view', 'buy']
        })
        g = CGFull().nodes(events_df)

        # Execute remote DAG with hypergraph
        # Note: gfql_remote doesn't support output param yet, returns last binding
        result = g.gfql_remote(
            let({
                'hg': call('hypergraph', {
                    'entity_types': ['user', 'product'],
                    'direct': True
                }),
                'filtered': ref('hg', [n({'type': 'user'})])
            })
        )

        # Verify result
        assert result is mock_result
        mock_chain_remote.assert_called_once()

    @patch('graphistry.compute.chain_remote.chain_remote_generic')
    def test_hypergraph_remote_error_handling(self, mock_chain_remote):
        """Test remote error handling for hypergraph."""
        # Simulate remote error
        mock_chain_remote.side_effect = Exception("Remote server error: Invalid entity_types")

        events_df = pd.DataFrame({
            'user': ['alice', 'bob'],
            'product': ['laptop', 'phone']
        })
        g = CGFull().nodes(events_df)

        # Should propagate the error
        with pytest.raises(Exception) as exc_info:
            g.gfql_remote(
                call('hypergraph', {
                    'entity_types': ['nonexistent_column'],
                    'direct': True
                })
            )

        assert "Remote server error" in str(exc_info.value)

    @patch('graphistry.compute.chain_remote.chain_remote_generic')
    def test_hypergraph_remote_with_all_params(self, mock_chain_remote):
        """Test remote hypergraph with all parameters."""
        # Create mock response
        mock_result = CGFull().edges(
            pd.DataFrame({'src': ['a'], 'dst': ['x']}),
            'src', 'dst'
        )
        mock_chain_remote.return_value = mock_result

        events_df = pd.DataFrame({
            'entity1': ['a', 'b', None],
            'entity2': ['x', 'y', 'z']
        })
        g = CGFull().nodes(events_df)

        # Execute with all params
        result = g.gfql_remote(
            call('hypergraph', {
                'entity_types': ['entity1', 'entity2'],
                'opts': {'some': 'option'},
                'drop_na': True,
                'drop_edge_attrs': False,
                'verbose': False,
                'direct': False,
                'engine': 'cudf',
                'npartitions': 4,
                'chunksize': 1000
            })
        )

        assert result is mock_result
        mock_chain_remote.assert_called_once()

        # Verify all params were passed
        call_args = mock_chain_remote.call_args
        chain_arg = call_args[0][1]

        from graphistry.compute.ast import ASTCall
        assert isinstance(chain_arg, ASTCall)
        assert chain_arg.params['engine'] == 'cudf'
        assert chain_arg.params['npartitions'] == 4
        assert chain_arg.params['chunksize'] == 1000

    @patch('graphistry.compute.chain_remote.chain_remote_generic')
    def test_hypergraph_remote_server_rejection(self, mock_chain_remote):
        """Test that server would reject mixed operations."""
        # Simulate server-side rejection
        mock_chain_remote.side_effect = ValueError(
            "Server error: Hypergraph cannot be mixed with other operations"
        )

        events_df = pd.DataFrame({
            'user': ['alice', 'bob'],
            'product': ['laptop', 'phone'],
            'type': ['person', 'person']
        })
        g = CGFull().nodes(events_df)

        # Server should reject mixed operations
        with pytest.raises(ValueError) as exc_info:
            g.gfql_remote([
                n({'type': 'person'}),
                call('hypergraph', {
                    'entity_types': ['user', 'product'],
                    'direct': True
                })
            ])

        assert "cannot be mixed" in str(exc_info.value)
        mock_chain_remote.assert_called_once()

    @patch('graphistry.compute.chain_remote.chain_remote_generic')
    def test_hypergraph_remote_mock_server_implementation(self, mock_chain_remote):
        """Test mocked remote server that uses actual hypergraph internally."""
        # Mock server implementation that actually calls hypergraph
        def mock_server_hypergraph(g, chain, api_token=None, dataset_id=None,
                                  output_type='all', format=None, df_export_args=None,
                                  node_col_subset=None, edge_col_subset=None,
                                  engine=None, validate=True, persist=False):
            """Mock server that executes hypergraph locally."""
            from graphistry.compute.ast import ASTCall

            # For single hypergraph call, chain is the ASTCall directly
            if isinstance(chain, ASTCall) and chain.function == 'hypergraph':
                params = chain.params
                # Server would use nodes as raw_events
                raw_events = g._nodes
                # Call actual hypergraph function
                result = g.hypergraph(raw_events, **params)
                # Return the graph from the result
                if isinstance(result, dict) and 'graph' in result:
                    return result['graph']
                return result
            # For other operations, could implement chain execution
            raise NotImplementedError(f"Mock server doesn't handle: {chain}")

        mock_chain_remote.side_effect = mock_server_hypergraph

        # Create test data
        events_df = pd.DataFrame({
            'user': ['alice', 'bob', 'alice', 'carol'],
            'product': ['laptop', 'phone', 'tablet', 'laptop'],
            'amount': [1000, 500, 300, 1000]
        })
        g = CGFull().nodes(events_df)

        # Execute remote hypergraph - mock server will use actual hypergraph
        result = g.gfql_remote(
            call('hypergraph', {
                'entity_types': ['user', 'product'],
                'direct': True,
                'engine': 'pandas'
            })
        )

        # Verify we got a proper hypergraph result
        assert result._nodes is not None
        assert result._edges is not None
        assert len(result._nodes) > 0
        assert len(result._edges) > 0
        # Check that it actually created entity relationships
        assert set(result._nodes.columns) != set(events_df.columns)  # Should have transformed columns
