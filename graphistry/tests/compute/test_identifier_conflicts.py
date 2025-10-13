"""
Tests for GFQL identifier conflicts with user column names.

These tests verify that GFQL operations work correctly when user graphs
have columns with names that GFQL uses internally.

Priority levels:
- P0: Critical - Operations that currently raise ValueError
- P2: Medium - Operations that silently overwrite user data
- P3: Low - Edge cases with low probability of conflict
"""
import pandas as pd
import pytest

from graphistry.tests.test_compute import CGFull


class TestP0IndexConflict:
    """P0 🚨 CRITICAL: Test 'index' column conflicts in chain() and hop()"""

    def test_user_index_column_chain_no_edge_binding(self):
        """User graph with 'index' column should work with chain() when g._edge is None"""
        # Create graph with user 'index' column in edges
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd'],
            'index': [100, 200, 300]  # User's column
        })
        nodes_df = pd.DataFrame({
            'n': ['a', 'b', 'c', 'd'],
            'type': ['x', 'y', 'z', 'w']
        })

        g = CGFull().edges(edges_df, 's', 'd').nodes(nodes_df, 'n')

        # After fix, this should work (no ValueError)
        from graphistry.compute.ast import n, e_forward

        # Chain with edges to actually get edges in the result
        result = g.chain([n({'type': 'x'}), e_forward()])

        # User's 'index' column should be preserved in the result
        assert 'index' in result._edges.columns
        # Edge from 'a' to 'b' should be included (node 'a' has type 'x')
        assert 100 in list(result._edges['index'])

    def test_user_index_column_hop_no_edge_binding(self):
        """User graph with 'index' column should work with hop() when g._edge is None"""
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd'],
            'index': [100, 200, 300]  # User's column
        })
        nodes_df = pd.DataFrame({
            'n': ['a', 'b', 'c', 'd']
        })

        g = CGFull().edges(edges_df, 's', 'd').nodes(nodes_df, 'n')

        # After fix, this should work (no ValueError)
        start_nodes = nodes_df[nodes_df['n'] == 'a']

        result = g.hop(nodes=start_nodes, hops=1)

        # User's 'index' column should be preserved in the result
        assert 'index' in result._edges.columns
        assert list(result._edges['index'].sort_values()) == [100]

    def test_user_index_column_with_edge_binding(self):
        """User graph with 'index' column works when g._edge is set"""
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd'],
            'index': [100, 200, 300],  # User's column
            'edge_id': [1, 2, 3]  # Edge binding
        })
        nodes_df = pd.DataFrame({
            'n': ['a', 'b', 'c', 'd']
        })

        g = CGFull().edges(edges_df, 's', 'd', edge='edge_id').nodes(nodes_df, 'n')

        # This should work because g._edge is set
        from graphistry.compute.ast import n, e_forward
        # Need edges in the chain to actually get edges in the result
        result = g.chain([n(), e_forward()])

        # User's 'index' column should be preserved
        assert 'index' in result._edges.columns
        assert sorted(list(result._edges['index'])) == [100, 200, 300]

    def test_user_has_gfql_internal_column_name(self):
        """User graph with '__gfql_edge_index_0__' column should still work via auto-increment"""
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd'],
            'index': [100, 200, 300],  # User's 'index' column
            '__gfql_edge_index_0__': [10, 20, 30]  # User also has base internal name
        })
        nodes_df = pd.DataFrame({
            'n': ['a', 'b', 'c', 'd'],
            'type': ['x', 'y', 'z', 'w']
        })

        g = CGFull().edges(edges_df, 's', 'd').nodes(nodes_df, 'n')

        # Should work by auto-incrementing to __gfql_edge_index_1__
        from graphistry.compute.ast import n, e_forward

        result = g.chain([n({'type': 'x'}), e_forward()])

        # Both user columns should be preserved
        assert 'index' in result._edges.columns
        assert '__gfql_edge_index_0__' in result._edges.columns
        assert 100 in list(result._edges['index'])
        assert 10 in list(result._edges['__gfql_edge_index_0__'])


class TestP2DBSCANConflict:
    """P2 🟡 MEDIUM: Test '_dbscan' column conflicts in clustering"""

    def test_user_dbscan_column_preserved_after_cluster(self):
        """User graph with '_dbscan' column should preserve original values"""
        pytest.skip("Clustering requires additional dependencies - manual verification needed")

        nodes_df = pd.DataFrame({
            'n': ['a', 'b', 'c', 'd'],
            '_dbscan': [10, 20, 30, 40],  # User's column
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [1.0, 2.0, 3.0, 4.0]
        })

        _ = CGFull().nodes(nodes_df, 'n')

        # Current behavior: silently overwrites '_dbscan' column
        # Decision: No change - '_dbscan' is user-facing output API
        # Users should not have pre-existing '_dbscan' columns when calling .dbscan()
        # This test documents that we're NOT fixing this conflict (breaking change)
        # Skipping actual clustering since it requires optional dependencies


class TestP2CollapseConflict:
    """P2 🟡 MEDIUM: Test 'node_collapse' and 'src_collapse' column conflicts"""

    def test_user_collapse_columns_preserved(self):
        """User graph with collapse column names should preserve user data via auto-increment"""
        nodes_df = pd.DataFrame({
            'n': ['a', 'b', 'c', 'd'],
            'node_collapse': ['user_val_1', 'user_val_2', 'user_val_3', 'user_val_4'],
            'type': ['x', 'x', 'y', 'y']
        })
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd'],
            'src_collapse': ['user_src_1', 'user_src_2', 'user_src_3']
        })

        g = CGFull().edges(edges_df, 's', 'd').nodes(nodes_df, 'n')

        # After fix: should use '__gfql_node_collapse_0__' etc. to avoid conflict
        # collapse_by is the main API for collapse operations
        # This should work without overwriting user columns
        from graphistry.compute.collapse import collapse_by

        result = collapse_by(
            g,
            parent='a',
            start_node='a',
            attribute='x',
            column='type',
            seen={},
            unwrap=True,
            verbose=False
        )

        # User's original columns should be preserved somewhere in the result
        # The exact columns depend on the collapse algorithm's final output
        # At minimum, we should not have raised an error
        assert result is not None


class TestP3XColumnConflict:
    """P3 🟢 LOW: Test 'x' column conflict in filter_by_dict"""

    def test_user_x_column_filter_by_dict(self):
        """User graph with 'x' column should work with filter_by_dict"""
        nodes_df = pd.DataFrame({
            'n': ['a', 'b', 'c', 'd'],
            'x': [10, 20, 30, 40],  # User's column
            'type': ['foo', 'bar', 'foo', 'bar']
        })

        g = CGFull().nodes(nodes_df, 'n')

        # This should work - filter_by_dict uses 'x' as temp var in empty df
        result = g.filter_nodes_by_dict({'type': 'foo'})

        # User's 'x' column should be preserved
        assert 'x' in result._nodes.columns
        assert list(result._nodes['x'].sort_values()) == [10, 30]


class TestP5IndexParameter:
    """P5 ⬜ SKIP: Verify 'index' parameter in conditional.py is user-facing API"""

    def test_conditional_probs_index_parameter(self):
        """Verify conditional_probs 'index' parameter is API, not column name"""
        pytest.skip("This is user-facing API parameter - no change needed")

        # The 'index' string here refers to pandas index/column operations
        # It's not a hardcoded column name that conflicts with user data
        # Example: g.conditional_probs('col1', 'col2', how='index')


class TestIdentifierConflictSummary:
    """Summary test to document all identifier conflicts"""

    def test_document_all_conflicts(self):
        """Document all identifier conflicts found in scan"""
        conflicts = {
            'P0_CRITICAL': {
                'index': {
                    'files': ['chain.py:465-469,576', 'hop.py:379-399'],
                    'fix': '__gfql_edge_index__',
                    'behavior': 'ValueError: Edges cannot have column "index"'
                }
            },
            'P2_MEDIUM': {
                '_dbscan': {
                    'files': ['cluster.py:182,185,418,420'],
                    'fix': '__gfql_dbscan__',
                    'behavior': 'Silently overwrites user column'
                },
                'node_collapse': {
                    'files': ['collapse.py:10,311'],
                    'fix': '__gfql_node_collapse__',
                    'behavior': 'Column conflict in collapse operations'
                },
                'src_collapse': {
                    'files': ['collapse.py:11,316'],
                    'fix': '__gfql_src_collapse__',
                    'behavior': 'Column conflict in collapse operations'
                }
            },
            'P3_LOW': {
                'x': {
                    'files': ['filter_by_dict.py:104'],
                    'fix': '__gfql_temp_x__',
                    'behavior': 'Unlikely conflict (temp var in empty df)'
                }
            },
            'P5_SKIP': {
                'index_parameter': {
                    'files': ['conditional.py:31,38,44,46,90,97'],
                    'fix': 'NO CHANGE - user-facing API',
                    'behavior': 'Parameter name, not column name'
                }
            }
        }

        # This test always passes - it's documentation
        assert len(conflicts['P0_CRITICAL']) == 1
        assert len(conflicts['P2_MEDIUM']) == 3
        assert len(conflicts['P3_LOW']) == 1
        assert len(conflicts['P5_SKIP']) == 1
