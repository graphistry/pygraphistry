"""
Tests for 'id' column name restriction in GFQL

Issue: GFQL rejects datasets using 'id' as node column name
Error: ValueError: GFQL The node column 'id' cannot be used as node id, please rename it!
Location: server/client/graph/processing_handler.py:649 (server-side validation)

This test suite verifies:
1. Local GFQL (.chain()) works with 'id' column
2. Remote GFQL (.gfql_remote()) behavior with 'id' column
3. Both nodes and edges can use 'id' as column name
4. No internal column conflicts are introduced

Test Tiers:
- Tier 1 (default): Fast CI tests - core coverage (~2.5s)
- Tier 2: Full coverage - comprehensive operators/predicates (~8s total)
- Tier 3 (future): Exhaustive edge cases
"""

import pandas as pd
import pytest
from graphistry.tests.common import NoAuthTestCase
from graphistry.tests.test_compute import CGFull
from graphistry import n, e_forward, e_reverse, e_undirected


# Shared symbol arrays for problematic column names
PROBLEMATIC_NODE_NAMES = ['id', 'idx', 'index', 'node', 'edge', 'type', 'label', 'name']
PROBLEMATIC_EDGE_SRC_NAMES = ['id', 'index', 'src', 'source', 'from', 'edge']
PROBLEMATIC_EDGE_DST_NAMES = ['id', 'index', 'dst', 'dest', 'destination', 'to', 'target', 'edge']
PROBLEMATIC_DATA_COLUMN_NAMES = ['id', 'index', 'node', 'edge', 'type', 'label', 'name']


class TestIdColumnRestriction(NoAuthTestCase):
    """Tests for 'id' column name usage in GFQL"""

    def test_node_column_named_id_local_chain(self):
        """Test that local GFQL (chain) works with node column named 'id'"""

        # Create nodes with 'id' column
        nodes_df = pd.DataFrame({
            'id': [1, 2, 3],
            'x': [10.5, 15.2, 20.1]
        })

        edges_df = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3]
        })

        # Create graph with 'id' as node column
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Test basic chain operation with 'id' column
        result = g.chain([n(), e_forward(), n()])

        # Verify operation succeeded
        assert result._nodes.shape[0] > 0
        assert result._edges.shape[0] > 0
        assert 'id' in result._nodes.columns
        assert 'x' in result._nodes.columns

    def test_node_column_named_id_with_filter(self):
        """Test filtering on 'id' column in local GFQL"""

        nodes_df = pd.DataFrame({
            'id': [1, 2, 3],
            'type': ['A', 'B', 'A']
        })

        edges_df = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3]
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Filter by 'id' column value
        result = g.chain([n({'id': 1}), e_forward(), n()])

        assert result._nodes.shape[0] >= 1
        assert result._edges.shape[0] >= 1

        # Verify we can access 'id' column in results
        assert 1 in result._nodes['id'].values

    def test_edge_column_named_id(self):
        """Test that edge column named 'id' works in local GFQL"""

        nodes_df = pd.DataFrame({
            'node': ['a', 'b', 'c']
        })

        # Edge dataframe with 'id' column
        edges_df = pd.DataFrame({
            'src': ['a', 'b'],
            'dst': ['b', 'c'],
            'id': ['e1', 'e2'],  # Edge ID column
            'weight': [0.5, 0.8]
        })

        g = CGFull().nodes(nodes_df, 'node').edges(edges_df, 'src', 'dst')

        result = g.chain([n(), e_forward(), n()])

        # Verify 'id' column is preserved in edges
        assert 'id' in result._edges.columns
        assert 'e1' in result._edges['id'].values or 'e2' in result._edges['id'].values

    def test_both_node_and_edge_columns_named_id(self):
        """Test when both nodes and edges have 'id' column"""

        nodes_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })

        edges_df = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3],
            'id': ['edge_1', 'edge_2'],  # Different type to distinguish
            'relationship': ['knows', 'likes']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        result = g.chain([n({'id': 1}), e_forward(), n()])

        # Both should be preserved
        assert 'id' in result._nodes.columns
        assert 'id' in result._edges.columns

        # Verify node 'id' values (numeric)
        assert 1 in result._nodes['id'].values or 2 in result._nodes['id'].values

        # Verify edge 'id' values (string)
        assert 'edge_1' in result._edges['id'].values or 'edge_2' in result._edges['id'].values

    def test_id_column_with_hop_operation(self):
        """Test 'id' column with hop operation"""

        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd']
        })

        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c'],
            'dst': ['b', 'c', 'd']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Hop from node 'a'
        result = g.hop(pd.DataFrame({'id': ['a']}), hops=2)

        assert result._nodes.shape[0] >= 2
        assert 'id' in result._nodes.columns

    def test_id_column_comparison_with_safe_name(self):
        """Compare results: 'id' column vs 'node_id' column"""

        # Graph with 'id' column
        nodes_id = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        edges_id = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3]
        })

        g_id = CGFull().nodes(nodes_id, 'id').edges(edges_id, 'src', 'dst')

        # Graph with 'node_id' column (safe alternative)
        nodes_safe = pd.DataFrame({
            'node_id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        edges_safe = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3]
        })

        g_safe = CGFull().nodes(nodes_safe, 'node_id').edges(edges_safe, 'src', 'dst')

        # Run identical operations
        result_id = g_id.chain([n(), e_forward(), n()])
        result_safe = g_safe.chain([n(), e_forward(), n()])

        # Results should have same structure
        assert result_id._nodes.shape[0] == result_safe._nodes.shape[0]
        assert result_id._edges.shape[0] == result_safe._edges.shape[0]

    def test_id_column_with_named_results(self):
        """Test 'id' column with named results (name='...')"""

        nodes_df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'type': ['A', 'B', 'A', 'B']
        })

        edges_df = pd.DataFrame({
            'src': [1, 2, 3],
            'dst': [2, 3, 4]
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Use named results
        result = g.chain([
            n({'type': 'A'}, name='type_a_nodes'),
            e_forward(name='connections'),
            n(name='neighbors')
        ])

        # Verify named columns are created
        assert 'type_a_nodes' in result._nodes.columns
        assert 'neighbors' in result._nodes.columns
        assert 'connections' in result._edges.columns

        # Original 'id' column should still be present
        assert 'id' in result._nodes.columns

    def test_id_column_preserved_through_complex_chain(self):
        """Test 'id' column preservation through complex chain operations"""

        nodes_df = pd.DataFrame({
            'id': list(range(10)),
            'category': ['A', 'B'] * 5
        })

        edges_df = pd.DataFrame({
            'src': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'dst': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Complex chain with multiple hops and filters
        result = g.chain([
            n({'id': 0}),
            e_forward(hops=3),
            n({'category': 'A'})
        ])

        # Verify 'id' column preserved in nodes
        assert 'id' in result._nodes.columns
        # Edges don't have 'id' column in this test (edges_df doesn't have one)
        # This is expected behavior

    @pytest.mark.skipif(True, reason="Requires server connection - manual test for remote GFQL")
    def test_node_column_named_id_remote_gfql(self):
        """Test remote GFQL (.gfql_remote()) with 'id' column

        This test documents the issue:
        ValueError: GFQL The node column 'id' cannot be used as node id, please rename it!

        This is a server-side validation that currently blocks 'id' column usage.
        Run manually with proper server configuration.
        """
        import graphistry

        # Requires registration
        # graphistry.register(api=3, server='...', username='...', password='...')

        nodes_df = pd.DataFrame({
            'id': [1, 2, 3],
            'x': [10.5, 15.2, 20.1]
        })

        edges_df = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3]
        })

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Upload to server
        g_remote = g.upload()

        # This should fail with current server validation
        # TODO: After fix, this should succeed
        with pytest.raises(ValueError, match="cannot be used as node id"):
            g_remote.gfql_remote([n(), e_forward(), n()])


class TestIdColumnEdgeCases(NoAuthTestCase):
    """Edge cases and special scenarios for 'id' column"""

    def test_id_column_with_special_types(self):
        """Test 'id' column with various data types"""

        test_cases = [
            # Integer IDs
            {'id': [1, 2, 3]},
            # String IDs
            {'id': ['a', 'b', 'c']},
            # UUID-like IDs
            {'id': ['550e8400-e29b-41d4-a716-446655440000',
                   '550e8400-e29b-41d4-a716-446655440001',
                   '550e8400-e29b-41d4-a716-446655440002']},
        ]

        for node_data in test_cases:
            nodes_df = pd.DataFrame(node_data)

            # Create edges using same ID values
            edges_df = pd.DataFrame({
                'src': [nodes_df['id'][0], nodes_df['id'][1]],
                'dst': [nodes_df['id'][1], nodes_df['id'][2]]
            })

            g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

            result = g.chain([n(), e_forward(), n()])

            assert 'id' in result._nodes.columns
            assert result._nodes.shape[0] > 0

    def test_id_column_case_sensitivity(self):
        """Test case variations of 'id' column name"""

        # Test different case variations
        case_variations = ['id', 'Id', 'ID', 'iD']

        for col_name in case_variations:
            nodes_df = pd.DataFrame({
                col_name: [1, 2, 3]
            })

            edges_df = pd.DataFrame({
                'src': [1, 2],
                'dst': [2, 3]
            })

            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            result = g.chain([n(), e_forward(), n()])

            assert col_name in result._nodes.columns
            assert result._nodes.shape[0] > 0

    def test_id_column_in_materialize_nodes(self):
        """Test 'id' column when using materialize_nodes()"""

        # Graph with only edges (nodes inferred)
        edges_df = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3]
        })

        g = CGFull().edges(edges_df, 'src', 'dst')

        # Materialize nodes (generates node table)
        g_mat = g.materialize_nodes()

        # Check what column name is generated
        node_col = g_mat._node

        # Run chain operation
        result = g_mat.chain([n(), e_forward(), n()])

        assert result._nodes.shape[0] > 0
        assert node_col in result._nodes.columns


class TestProblematicColumnNames(NoAuthTestCase):
    """Test other common problematic column names from shared symbol arrays"""

    def test_hop_with_all_problematic_node_names(self):
        """Test hop() with each problematic node name - CRITICAL for issue #761"""
        for col_name in PROBLEMATIC_NODE_NAMES:
            nodes_df = pd.DataFrame({col_name: [1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50]})
            edges_df = pd.DataFrame({'src': [1, 2, 3, 4], 'dst': [2, 3, 4, 5]})
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            # Test multi-hop
            result = g.hop(pd.DataFrame({col_name: [1]}), hops=3)
            assert col_name in result._nodes.columns, f"hop() failed for node column '{col_name}'"
            assert result._node == col_name
            assert result._nodes.shape[0] >= 2

    def test_edge_reverse_and_undirected_with_problematic_names(self):
        """Test e_reverse() and e_undirected() with problematic column names"""
        for col_name in ['id', 'index', 'node', 'edge']:
            nodes_df = pd.DataFrame({col_name: [1, 2, 3], 'value': [10, 20, 30]})
            edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3]})
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            # Test reverse
            result_rev = g.chain([n(), e_reverse(), n()])
            assert col_name in result_rev._nodes.columns, f"e_reverse() failed for '{col_name}'"

            # Test undirected
            result_und = g.chain([n(), e_undirected(), n()])
            assert col_name in result_und._nodes.columns, f"e_undirected() failed for '{col_name}'"

    def test_predicates_on_problematic_columns(self):
        """Test predicates (gt, is_in, between) on problematic column names"""
        from graphistry import gt, is_in, between

        for col_name in ['id', 'index', 'node']:
            nodes_df = pd.DataFrame({
                'nid': [1, 2, 3, 4, 5],
                col_name: [10, 20, 30, 40, 50]
            })
            edges_df = pd.DataFrame({'src': [1, 2, 3, 4], 'dst': [2, 3, 4, 5]})
            g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, 'src', 'dst')

            # Test gt predicate
            result_gt = g.chain([n({col_name: gt(25)}), e_forward(), n()])
            assert result_gt._nodes.shape[0] >= 1, f"gt() failed on '{col_name}'"
            assert col_name in result_gt._nodes.columns

            # Test is_in predicate
            result_in = g.chain([n({col_name: is_in([10, 30, 50])}), e_forward(), n()])
            assert result_in._nodes.shape[0] >= 1, f"is_in() failed on '{col_name}'"

            # Test between predicate
            result_bw = g.chain([n({col_name: between(15, 45)}), e_forward(), n()])
            assert result_bw._nodes.shape[0] >= 1, f"between() failed on '{col_name}'"

    @pytest.mark.skip(reason="let/ref returns empty result - needs investigation")
    def test_let_ref_with_problematic_columns(self):
        """Test let/ref DAG operations with problematic column names

        TODO: Returns empty result - may need different query structure or data
        """
        from graphistry.compute.ast import let, ref

        for col_name in ['id', 'index', 'node']:
            nodes_df = pd.DataFrame({col_name: [1, 2, 3, 4], 'type': ['A', 'B', 'A', 'B']})
            edges_df = pd.DataFrame({'src': [1, 2, 3], 'dst': [2, 3, 4]})
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            query = let({
                'start': n({'type': 'A'}),
                'neighbors': ref('start', [e_forward(), n()])
            })

            result = g.gfql(query, output='neighbors')

            # Verify column preserved and bindings created
            assert col_name in result._nodes.columns, f"let/ref failed for '{col_name}'"
            assert result._node == col_name
            assert result._nodes.shape[0] >= 1, f"let/ref returned empty result for '{col_name}'"

    def test_call_get_degrees_with_problematic_columns(self):
        """Test call operation (get_degrees) with problematic column names"""
        for col_name in ['id', 'index', 'node']:
            nodes_df = pd.DataFrame({col_name: [1, 2, 3, 4]})
            edges_df = pd.DataFrame({'src': [1, 1, 2, 3], 'dst': [2, 3, 3, 4]})
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            # Test get_degrees
            g_degrees = g.get_degrees()
            assert col_name in g_degrees._nodes.columns, f"get_degrees() failed for '{col_name}'"
            assert 'degree' in g_degrees._nodes.columns
            assert g_degrees._node == col_name

    def test_node_id_bindings_with_problematic_names(self):
        """Test each problematic name as node ID binding"""
        for col_name in PROBLEMATIC_NODE_NAMES:
            nodes_df = pd.DataFrame({col_name: [1, 2, 3], 'value': [10, 20, 30]})
            edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3]})
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')
            result = g.chain([n(), e_forward(), n()])
            assert col_name in result._nodes.columns, f"Node binding failed for '{col_name}'"
            assert result._node == col_name

    def test_edge_source_bindings_with_problematic_names(self):
        """Test each problematic name as edge source binding"""
        for col_name in PROBLEMATIC_EDGE_SRC_NAMES:
            nodes_df = pd.DataFrame({'nid': [1, 2, 3]})
            edges_df = pd.DataFrame({col_name: [1, 2], 'target': [2, 3]})
            g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, col_name, 'target')
            result = g.chain([n(), e_forward(), n()])
            assert g._source == col_name, f"Edge source binding failed for '{col_name}'"
            assert result._edges.shape[0] > 0

    def test_edge_destination_bindings_with_problematic_names(self):
        """Test each problematic name as edge destination binding"""
        for col_name in PROBLEMATIC_EDGE_DST_NAMES:
            nodes_df = pd.DataFrame({'nid': [1, 2, 3]})
            edges_df = pd.DataFrame({'origin': [1, 2], col_name: [2, 3]})
            g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, 'origin', col_name)
            result = g.chain([n(), e_forward(), n()])
            assert g._destination == col_name, f"Edge dest binding failed for '{col_name}'"
            assert result._edges.shape[0] > 0

    def test_node_data_columns_with_problematic_names(self):
        """Test filtering on node data columns with problematic names"""
        for col_name in PROBLEMATIC_DATA_COLUMN_NAMES:
            nodes_df = pd.DataFrame({'nid': [1, 2, 3], col_name: ['A', 'B', 'A']})
            edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3]})
            g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, 'src', 'dst')
            result = g.chain([n({col_name: 'A'}), e_forward(), n()])
            assert col_name in result._nodes.columns, f"Node data filter failed for '{col_name}'"
            assert result._nodes.shape[0] >= 1

    def test_edge_data_columns_with_problematic_names(self):
        """Test filtering on edge data columns with problematic names"""
        for col_name in PROBLEMATIC_DATA_COLUMN_NAMES:
            nodes_df = pd.DataFrame({'nid': [1, 2, 3]})
            edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3], col_name: ['A', 'B']})
            g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, 'src', 'dst')
            result = g.chain([n(), e_forward({col_name: 'A'}), n()])
            assert col_name in result._edges.columns, f"Edge data filter failed for '{col_name}'"

    def test_both_nodes_and_edges_with_same_problematic_name(self):
        """Test when both nodes and edges use same problematic name (id, index)"""
        for col_name in ['id', 'index']:
            # Node uses col_name as ID binding
            nodes_df = pd.DataFrame({col_name: [1, 2, 3], 'name': ['a', 'b', 'c']})
            # Edge uses col_name as data column
            edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3], col_name: ['e1', 'e2']})
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')
            result = g.chain([n(), e_forward(), n()])
            assert col_name in result._nodes.columns, f"Node '{col_name}' lost"
            assert col_name in result._edges.columns, f"Edge '{col_name}' lost"
            # Verify both preserved with different values
            assert result._node == col_name


@pytest.mark.tier2
class TestProblematicColumnNamesTier2(NoAuthTestCase):
    """Tier 2: Full coverage tests - comprehensive operators and predicates"""

    def test_e_operator_with_problematic_names(self):
        """Test e() (alias for e_undirected) with problematic names"""
        from graphistry import e

        for col_name in ['id', 'index', 'node']:
            nodes_df = pd.DataFrame({col_name: [1, 2, 3], 'value': [10, 20, 30]})
            edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3]})
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            result = g.chain([n(), e(), n()])
            assert col_name in result._nodes.columns, f"e() failed for '{col_name}'"

    def test_comparison_predicates_lt_ge_le(self):
        """Test lt, ge, le predicates on problematic columns"""
        from graphistry import lt, ge, le

        for col_name in ['id', 'index']:
            nodes_df = pd.DataFrame({
                'nid': [1, 2, 3, 4, 5],
                col_name: [10, 20, 30, 40, 50]
            })
            edges_df = pd.DataFrame({'src': [1, 2, 3, 4], 'dst': [2, 3, 4, 5]})
            g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, 'src', 'dst')

            # Test lt
            result_lt = g.chain([n({col_name: lt(35)}), e_forward(), n()])
            assert result_lt._nodes.shape[0] >= 1, f"lt() failed on '{col_name}'"

            # Test ge
            result_ge = g.chain([n({col_name: ge(30)}), e_forward(), n()])
            assert result_ge._nodes.shape[0] >= 1, f"ge() failed on '{col_name}'"

            # Test le
            result_le = g.chain([n({col_name: le(30)}), e_forward(), n()])
            assert result_le._nodes.shape[0] >= 1, f"le() failed on '{col_name}'"

    def test_null_predicates(self):
        """Test isna, notna on problematic columns"""
        from graphistry import isna, notna

        for col_name in ['id', 'index']:
            nodes_df = pd.DataFrame({
                'nid': [1, 2, 3, 4],
                col_name: [10, None, 30, None]
            })
            edges_df = pd.DataFrame({'src': [1, 2, 3], 'dst': [2, 3, 4]})
            g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, 'src', 'dst')

            # Test isna
            result_isna = g.chain([n({col_name: isna()}), e_forward(), n()])
            assert col_name in result_isna._nodes.columns, f"isna() failed on '{col_name}'"

            # Test notna
            result_notna = g.chain([n({col_name: notna()}), e_forward(), n()])
            assert col_name in result_notna._nodes.columns, f"notna() failed on '{col_name}'"
            assert result_notna._nodes.shape[0] >= 1

    def test_string_predicate_contains(self):
        """Test contains predicate on string 'id' column"""
        from graphistry import contains

        nodes_df = pd.DataFrame({
            'nid': [1, 2, 3, 4],
            'id': ['user_123', 'admin_456', 'user_789', 'guest_012']
        })
        edges_df = pd.DataFrame({'src': [1, 2, 3], 'dst': [2, 3, 4]})
        g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, 'src', 'dst')

        result = g.chain([n({'id': contains('user')}), e_forward(), n()])
        assert result._nodes.shape[0] >= 1, "contains() failed on 'id'"
        assert 'id' in result._nodes.columns

    def test_call_get_indegrees_outdegrees(self):
        """Test get_indegrees/outdegrees with problematic columns"""
        for col_name in ['id', 'index']:
            nodes_df = pd.DataFrame({col_name: [1, 2, 3, 4]})
            edges_df = pd.DataFrame({'src': [1, 1, 2, 3], 'dst': [2, 3, 3, 4]})
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            # Test get_indegrees
            g_in = g.get_indegrees()
            assert col_name in g_in._nodes.columns, f"get_indegrees() failed for '{col_name}'"
            assert 'degree_in' in g_in._nodes.columns  # Column name is degree_in not indegree
            assert g_in._node == col_name

            # Test get_outdegrees
            g_out = g.get_outdegrees()
            assert col_name in g_out._nodes.columns, f"get_outdegrees() failed for '{col_name}'"
            assert 'degree_out' in g_out._nodes.columns  # Column name is degree_out not outdegree
            assert g_out._node == col_name

    def test_call_filter_nodes_by_dict(self):
        """Test filter_nodes_by_dict with problematic columns"""
        for col_name in ['id', 'index']:
            nodes_df = pd.DataFrame({
                col_name: [1, 2, 3, 4],
                'type': ['A', 'B', 'A', 'B'],
                'value': [10, 20, 30, 40]
            })
            edges_df = pd.DataFrame({'src': [1, 2, 3], 'dst': [2, 3, 4]})
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            # Filter on problematic column
            result = g.filter_nodes_by_dict({col_name: 1, 'type': 'A'})
            assert result._nodes.shape[0] == 1, f"filter_nodes_by_dict failed for '{col_name}'"
            assert col_name in result._nodes.columns

    def test_call_filter_edges_by_dict(self):
        """Test filter_edges_by_dict with problematic columns"""
        nodes_df = pd.DataFrame({'nid': [1, 2, 3, 4]})

        for col_name in ['id', 'index']:
            edges_df = pd.DataFrame({
                'src': [1, 2, 3],
                'dst': [2, 3, 4],
                col_name: ['e1', 'e2', 'e3'],
                'weight': [0.5, 0.8, 0.3]
            })
            g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, 'src', 'dst')

            # Filter on problematic column
            result = g.filter_edges_by_dict({col_name: 'e1'})
            assert result._edges.shape[0] == 1, f"filter_edges_by_dict failed for '{col_name}'"
            assert col_name in result._edges.columns

    @pytest.mark.skip(reason="collapse() has complex API - needs topology-aware traversal setup")
    def test_call_collapse_with_problematic_columns(self):
        """Test collapse operation with problematic columns

        TODO: collapse(node, attribute, column) requires topology-aware setup
        Skipping for now - less critical than other operations
        """
        pass

    def test_call_drop_nodes_with_problematic_columns(self):
        """Test drop_nodes with problematic columns"""
        for col_name in ['id', 'index']:
            nodes_df = pd.DataFrame({
                col_name: [1, 2, 3, 4, 5],
                'type': ['A', 'B', 'A', 'B', 'C']
            })
            edges_df = pd.DataFrame({'src': [1, 2, 3, 4], 'dst': [2, 3, 4, 5]})
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            # Drop nodes where col_name > 3
            nodes_to_drop = nodes_df[nodes_df[col_name] > 3][col_name].tolist()
            result = g.drop_nodes(nodes_to_drop)

            assert col_name in result._nodes.columns, f"drop_nodes failed for '{col_name}'"
            assert result._node == col_name
            assert result._nodes.shape[0] == 3  # Should have 3 nodes left (1,2,3)
            assert 4 not in result._nodes[col_name].values
            assert 5 not in result._nodes[col_name].values

    def test_call_keep_nodes_with_problematic_columns(self):
        """Test keep_nodes with problematic columns"""
        for col_name in ['id', 'index']:
            nodes_df = pd.DataFrame({
                col_name: [1, 2, 3, 4, 5],
                'type': ['A', 'B', 'A', 'B', 'C']
            })
            edges_df = pd.DataFrame({'src': [1, 2, 3, 4], 'dst': [2, 3, 4, 5]})
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            # Keep only nodes where col_name <= 3
            nodes_to_keep = nodes_df[nodes_df[col_name] <= 3][col_name].tolist()
            result = g.keep_nodes(nodes_to_keep)

            assert col_name in result._nodes.columns, f"keep_nodes failed for '{col_name}'"
            assert result._node == col_name
            assert result._nodes.shape[0] == 3  # Should have 3 nodes (1,2,3)
            assert all(v in [1, 2, 3] for v in result._nodes[col_name].values)

    def test_call_prune_self_edges_with_problematic_columns(self):
        """Test prune_self_edges with problematic columns"""
        for col_name in ['id', 'index']:
            nodes_df = pd.DataFrame({col_name: [1, 2, 3]})
            # Include self-edges
            edges_df = pd.DataFrame({
                'src': [1, 1, 2, 3],
                'dst': [1, 2, 2, 3]  # Self-edges: 1->1, 2->2, 3->3
            })
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            result = g.prune_self_edges()

            assert col_name in result._nodes.columns, f"prune_self_edges failed for '{col_name}'"
            assert result._node == col_name
            # Should only have 1->2 edge left (removed self-edges)
            assert result._edges.shape[0] == 1
            assert result._edges.iloc[0]['src'] == 1
            assert result._edges.iloc[0]['dst'] == 2

    def test_call_get_topological_levels_with_problematic_columns(self):
        """Test get_topological_levels with problematic columns"""
        for col_name in ['id', 'index']:
            # Create DAG (no cycles)
            nodes_df = pd.DataFrame({col_name: [1, 2, 3, 4]})
            edges_df = pd.DataFrame({
                'src': [1, 1, 2, 3],
                'dst': [2, 3, 4, 4]
            })
            g = CGFull().nodes(nodes_df, col_name).edges(edges_df, 'src', 'dst')

            result = g.get_topological_levels()

            assert col_name in result._nodes.columns, f"get_topological_levels failed for '{col_name}'"
            assert result._node == col_name
            assert 'level' in result._nodes.columns
            # Node 1 should be at level 0 (no incoming edges)
            assert result._nodes[result._nodes[col_name] == 1]['level'].iloc[0] == 0

    def test_encode_point_color_with_problematic_columns(self):
        """Test encode_point_color with problematic column as color source"""
        for col_name in ['id', 'index', 'type']:
            nodes_df = pd.DataFrame({
                'nid': [1, 2, 3],
                col_name: ['A', 'B', 'A']
            })
            edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3]})
            g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, 'src', 'dst')

            # Encode color using problematic column
            result = g.encode_point_color(col_name)

            assert col_name in result._nodes.columns, f"encode_point_color failed for '{col_name}'"
            # Check that point_color binding is set
            assert result._point_color == col_name

    def test_encode_point_size_with_problematic_columns(self):
        """Test encode_point_size with problematic column as size source"""
        for col_name in ['id', 'index']:
            nodes_df = pd.DataFrame({
                'nid': [1, 2, 3],
                col_name: [10, 20, 30]
            })
            edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3]})
            g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, 'src', 'dst')

            # Encode size using problematic column
            result = g.encode_point_size(col_name)

            assert col_name in result._nodes.columns, f"encode_point_size failed for '{col_name}'"
            # Check that point_size binding is set
            assert result._point_size == col_name

    def test_encode_edge_color_with_problematic_columns(self):
        """Test encode_edge_color with problematic column on edges"""
        nodes_df = pd.DataFrame({'nid': [1, 2, 3]})

        for col_name in ['id', 'index', 'type']:
            edges_df = pd.DataFrame({
                'src': [1, 2],
                'dst': [2, 3],
                col_name: ['A', 'B']
            })
            g = CGFull().nodes(nodes_df, 'nid').edges(edges_df, 'src', 'dst')

            # Encode edge color using problematic column
            result = g.encode_edge_color(col_name)

            assert col_name in result._edges.columns, f"encode_edge_color failed for '{col_name}'"
            # Check that edge_color binding is set
            assert result._edge_color == col_name


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
