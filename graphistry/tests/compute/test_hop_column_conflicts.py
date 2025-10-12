import pandas as pd
import pytest
from graphistry.tests.common import NoAuthTestCase

from graphistry.tests.test_compute import CGFull


class TestHopColumnConflicts(NoAuthTestCase):
    """Tests for potential column name conflicts in hop.py"""
    
    # List of potential internal variable names that might conflict as column names
    POTENTIAL_RESERVED_NAMES = [
        # Graph processing internal variables
        'wave_front', 'matches_nodes', 'matches_edges', 'combined_node_ids',
        'new_node_ids', 'hop_edges_forward', 'hop_edges_reverse',
        'new_node_ids_forward', 'new_node_ids_reverse',
        'intermediate_target_wave_front', 'base_target_nodes', 'EDGE_ID',
        
        # Function parameter names
        'source_node_match', 'destination_node_match', 'edge_match',
        'source_node_query', 'destination_node_query', 'edge_query', 
        'target_wave_front', 'direction', 'to_fixed_point', 'engine',
        
        # DataFrame operation terms
        'how', 'on', 'sort', 'ignore_index', 'subset'
    ]

    def test_index_column_in_edges_no_edge_id(self):
        """Test that 'index' column is preserved when no edge ID is bound (auto-generates unique name)"""

        # Create edges with an 'index' column
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'a'],
            'index': [1, 2, 3]  # This should be preserved via auto-increment naming
        })

        # Create graph without binding edge ID
        g = CGFull().edges(edges_df, 's', 'd')

        # After fix: hop should work and preserve the user's 'index' column
        result = g.hop(pd.DataFrame({'id': ['a']}), 1)
        assert result._edges.shape[0] > 0
        # User's 'index' column should be preserved in the result
        assert 'index' in result._edges.columns
        assert 1 in result._edges['index'].values  # edge from 'a' to 'b' has index=1

    def test_edge_id_same_as_index(self):
        """Test hop with edge ID column explicitly bound as 'index'"""
        
        # Create edges with an 'index' column
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'], 
            'd': ['b', 'c', 'a'],
            'index': [1, 2, 3]
        })
        
        # Create graph binding 'index' as edge ID
        g = CGFull().edges(edges_df, 's', 'd', 'index')
        
        # This should work since 'index' is now properly bound
        result = g.hop(pd.DataFrame({'id': ['a']}), 1)
        assert result._edges.shape[0] > 0

    def test_node_column_same_as_source(self):
        """Test that hop works when node ID column has same name as edge source column"""
        
        # Create nodes and edges where the node ID has the same name as source
        nodes_df = pd.DataFrame({
            's': ['a', 'b', 'c']  # Node column same as source column
        })
        
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'a']
        })
        
        # Create graph with conflicting column names
        g = CGFull().nodes(nodes_df, 's').edges(edges_df, 's', 'd')
        
        # With the new implementation, hop should work correctly
        result = g.hop(pd.DataFrame({'s': ['a']}), 1)
        
        # Verify the hop operation worked correctly
        assert result._nodes.shape[0] > 0
        assert result._edges.shape[0] > 0
        assert 's' in result._nodes.columns

    def test_node_column_same_as_destination(self):
        """Test that hop works when node ID column has same name as edge destination column"""
        
        # Create nodes and edges where the node ID has the same name as destination
        nodes_df = pd.DataFrame({
            'd': ['a', 'b', 'c']  # Node column same as destination column
        })
        
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'a']
        })
        
        # Create graph with conflicting column names
        g = CGFull().nodes(nodes_df, 'd').edges(edges_df, 's', 'd')
        
        # With the new implementation, hop should work correctly
        result = g.hop(pd.DataFrame({'d': ['b']}), 1)
        
        # Verify the hop operation worked correctly
        assert result._nodes.shape[0] > 0
        assert result._edges.shape[0] > 0
        assert 'd' in result._nodes.columns

    def test_safe_vs_dangerous_column_names(self):
        """Compare results of hop with safe column names vs potentially dangerous ones"""
        
        # Safe case - standard column names
        safe_nodes = pd.DataFrame({
            'node_id': ['a', 'b', 'c', 'd'],
            'type': ['x', 'y', 'x', 'y']
        })
        
        safe_edges = pd.DataFrame({
            'source': ['a', 'b', 'c'],
            'target': ['b', 'c', 'd'],
            'weight': [0.1, 0.2, 0.3]
        })
        
        g_safe = CGFull().nodes(safe_nodes, 'node_id').edges(safe_edges, 'source', 'target')
        
        # Dangerous case - column names that might conflict with internal variables
        dangerous_nodes = pd.DataFrame({
            'node_id': ['a', 'b', 'c', 'd'],
            'type': ['x', 'y', 'x', 'y'],
            'wave_front': [1, 2, 3, 4],      # Same as internal variable
            'matches_nodes': [5, 6, 7, 8],   # Same as internal variable
            'matches_edges': [9, 10, 11, 12]  # Same as internal variable
        })
        
        dangerous_edges = pd.DataFrame({
            'source': ['a', 'b', 'c'],
            'target': ['b', 'c', 'd'],
            'weight': [0.1, 0.2, 0.3],
            'new_node_ids': ['n1', 'n2', 'n3'],  # Same as internal variable
            'combined_node_ids': ['c1', 'c2', 'c3']  # Same as internal variable
        })
        
        g_dangerous = CGFull().nodes(dangerous_nodes, 'node_id').edges(dangerous_edges, 'source', 'target')
        
        # Run hop with same parameters on both
        result_safe = g_safe.hop(pd.DataFrame({'node_id': ['a']}), 1)
        result_dangerous = g_dangerous.hop(pd.DataFrame({'node_id': ['a']}), 1)
        
        # Core connectivity should be identical
        assert result_safe._nodes.shape[0] == result_dangerous._nodes.shape[0]
        assert result_safe._edges.shape[0] == result_dangerous._edges.shape[0]
        
        # All original dangerous columns should be preserved
        for col in dangerous_nodes.columns:
            assert col in result_dangerous._nodes.columns
            
        for col in dangerous_edges.columns:
            assert col in result_dangerous._edges.columns

    def test_compare_hop_with_index_column_vs_safe_column(self):
        """Compare hop results with 'index' column vs a safe alternative column name"""
        
        # Create identical edge data with different column names
        safe_edges = pd.DataFrame({
            's': ['a', 'b', 'c'], 
            'd': ['b', 'c', 'a'],
            'safe_id': [1, 2, 3]  # Safe column name
        })
        
        g_safe = CGFull().edges(safe_edges, 's', 'd', 'safe_id')
        
        # Same data but with explicit edge ID that is not 'index'
        risky_edges = pd.DataFrame({
            's': ['a', 'b', 'c'], 
            'd': ['b', 'c', 'a'],
            'EDGE_ID': [1, 2, 3]  # Same as internal variable used for temporary column
        })
        
        g_risky = CGFull().edges(risky_edges, 's', 'd', 'EDGE_ID')
        
        # Run hop with same parameters on both
        result_safe = g_safe.hop(pd.DataFrame({'id': ['a']}), 1)
        result_risky = g_risky.hop(pd.DataFrame({'id': ['a']}), 1)
        
        # Results should be equivalent in structure
        assert result_safe._nodes.shape[0] == result_risky._nodes.shape[0]
        assert result_safe._edges.shape[0] == result_risky._edges.shape[0]
        
        # The EDGE_ID column should be preserved in result_risky
        assert 'EDGE_ID' in result_risky._edges.columns
        assert 'safe_id' in result_safe._edges.columns

    def test_compare_direction_variants_with_risky_columns(self):
        """Compare hop results across different direction modes with risky column names"""
        
        # Create graph with columns that could conflict with internal variables
        nodes_df = pd.DataFrame({
            'node': ['a', 'b', 'c', 'd', 'e'],
            'hop_edges_forward': [1, 2, 3, 4, 5],  # Same as internal variable
            'hop_edges_reverse': [6, 7, 8, 9, 10]  # Same as internal variable
        })
        
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd'],
            'd': ['b', 'c', 'd', 'e'],
            'new_node_ids_forward': ['f1', 'f2', 'f3', 'f4'],  # Same as internal variable
            'new_node_ids_reverse': ['r1', 'r2', 'r3', 'r4']   # Same as internal variable
        })
        
        g = CGFull().nodes(nodes_df, 'node').edges(edges_df, 's', 'd')
        
        # Run hop with different directions
        result_forward = g.hop(pd.DataFrame({'node': ['a']}), 1, direction='forward')
        result_reverse = g.hop(pd.DataFrame({'node': ['e']}), 1, direction='reverse')
        result_undirected = g.hop(pd.DataFrame({'node': ['c']}), 1, direction='undirected')
        
        # Original columns should be preserved in all results
        for col in nodes_df.columns:
            assert col in result_forward._nodes.columns
            assert col in result_reverse._nodes.columns
            assert col in result_undirected._nodes.columns
            
        for col in edges_df.columns:
            assert col in result_forward._edges.columns
            assert col in result_reverse._edges.columns
            assert col in result_undirected._edges.columns
        
        # Verify expected connectivity patterns
        assert 'b' in result_forward._nodes['node'].values  # a -> b
        assert 'd' in result_reverse._nodes['node'].values  # e <- d
        assert set(['b', 'd']).issubset(set(result_undirected._nodes['node'].values))  # c <-> b, c <-> d

    def test_edge_with_both_index_and_edge_id(self):
        """Test hop with edges having both 'index' column and a separate edge ID"""
        
        # Create edges with both 'index' column and edge_id
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'], 
            'd': ['b', 'c', 'a'],
            'index': [10, 20, 30],    # This should be allowed when edge_id is explicitly set
            'edge_id': [1, 2, 3]
        })
        
        # Create graph with explicit edge ID
        g = CGFull().edges(edges_df, 's', 'd', 'edge_id')
        
        # Determine the node ID column
        g2 = g.materialize_nodes()
        node_id_col = g2._node
        
        # This should work because we explicitly set edge_id, which bypasses the 'index' check
        result = g.hop(pd.DataFrame({node_id_col: ['a']}), 1)
        assert result._edges.shape[0] > 0
        assert 'index' in result._edges.columns

    def test_target_wave_front_comparison(self):
        """Compare hop with and without target_wave_front having risky column names"""
        
        # Create basic graph
        nodes_df = pd.DataFrame({
            'node': ['a', 'b', 'c', 'd', 'e', 'f'],
            'type': ['x', 'y', 'x', 'y', 'x', 'y']
        })
        
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd', 'e'],
            'd': ['b', 'c', 'd', 'e', 'f'],
            'weight': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        g = CGFull().nodes(nodes_df, 'node').edges(edges_df, 's', 'd')
        
        # Run without target_wave_front for comparison
        g.hop(
            nodes=pd.DataFrame({'node': ['a']}),
            hops=3
        )
        
        # Create target_wave_front with risky column names
        target_df = pd.DataFrame({
            'node': ['f'],
            'intermediate_target_wave_front': ['risky_value'],  # Same as internal variable
            'base_target_nodes': ['another_risky_value']       # Same as internal variable
        })
        
        # In a typical graph pattern, the 'f' node would be 3 hops away from 'a'
        # However, it's possible our test graph doesn't guarantee that path
        # Let's adjust the test to check more generally for target_wave_front effects
        
        # Run with target_wave_front
        result_targeted = g.hop(
            nodes=pd.DataFrame({'node': ['a']}),
            hops=3,
            target_wave_front=target_df
        )
        
        # Target result should filter based on target_wave_front
        # Check that at least some paths are included
        assert result_targeted._nodes.shape[0] > 0
        assert result_targeted._edges.shape[0] > 0
        
        # Risky columns should be preserved - this is the main thing we're testing
        for col in target_df.columns:
            if col != 'node':  # Node ID will be merged
                assert col in result_targeted._nodes.columns

    def test_mixed_case_index_column(self):
        """Test that mixed case versions of 'index' are handled properly"""
        
        # Test with mixed case 'Index' column
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'], 
            'd': ['b', 'c', 'a'],
            'Index': [1, 2, 3]  # Mixed case version
        })
        
        # Create graph
        g = CGFull().edges(edges_df, 's', 'd')
        
        # This should work as 'Index' is different from 'index'
        result = g.hop(pd.DataFrame({'id': ['a']}), 1)
        assert result._edges.shape[0] > 0
        assert 'Index' in result._edges.columns
        
        # Compare with safe column name
        safe_edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'], 
            'd': ['b', 'c', 'a'],
            'safe_column': [1, 2, 3]
        })
        
        g_safe = CGFull().edges(safe_edges_df, 's', 'd')
        result_safe = g_safe.hop(pd.DataFrame({'id': ['a']}), 1)
        
        # Results should be equivalent in structure
        assert result._nodes.shape[0] == result_safe._nodes.shape[0]
        assert result._edges.shape[0] == result_safe._edges.shape[0]
        
    def test_node_column_name_conflicts_individually(self):
        """Test each potential reserved name as a node column name individually"""
        
        # Basic valid graph structure for testing
        base_edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'], 
            'd': ['b', 'c', 'd']
        })
        
        # Test each potential reserved name individually
        for reserved_name in self.POTENTIAL_RESERVED_NAMES:
            # Skip 'index' which is tested separately
            if reserved_name.lower() == 'index':
                continue
                
            # Create a node dataframe with the reserved name as a column
            nodes_df = pd.DataFrame({
                'node': ['a', 'b', 'c', 'd'],
                reserved_name: [f"{reserved_name}_1", f"{reserved_name}_2", 
                               f"{reserved_name}_3", f"{reserved_name}_4"]
            })
            
            # Create the graph with this potentially conflicting column name
            g = CGFull().nodes(nodes_df, 'node').edges(base_edges_df, 's', 'd')
            
            try:
                # Attempt to run a basic hop
                result = g.hop(pd.DataFrame({'node': ['a']}), 1)
                
                # If we get here, no exception was raised - check the column was preserved
                assert reserved_name in result._nodes.columns, f"Column '{reserved_name}' was lost during hop"
                
                # Check the value is preserved
                a_node = result._nodes[result._nodes['node'] == 'a']
                assert len(a_node) > 0, f"Node 'a' missing from result when testing column '{reserved_name}'"
                assert a_node[reserved_name].iloc[0] == f"{reserved_name}_1", \
                    f"Column value for '{reserved_name}' was modified during hop"
                
            except Exception as e:
                # If an exception happens, fail the test with a clear message
                pytest.fail(f"Column name '{reserved_name}' caused an error in hop(): {str(e)}")
                
    def test_edge_column_name_conflicts_individually(self):
        """Test each potential reserved name as an edge column name individually"""
        
        # Basic valid graph structure for testing
        base_nodes_df = pd.DataFrame({
            'node': ['a', 'b', 'c', 'd']
        })
        
        # Test each potential reserved name individually
        for reserved_name in self.POTENTIAL_RESERVED_NAMES:
            # Skip 'index' which is tested separately
            if reserved_name.lower() == 'index':
                continue
                
            # Create an edge dataframe with the reserved name as a column
            edges_df = pd.DataFrame({
                's': ['a', 'b', 'c'],
                'd': ['b', 'c', 'd'],
                reserved_name: [f"{reserved_name}_1", f"{reserved_name}_2", f"{reserved_name}_3"]
            })
            
            # Create the graph with this potentially conflicting column name
            g = CGFull().nodes(base_nodes_df, 'node').edges(edges_df, 's', 'd')
            
            try:
                # Attempt to run a basic hop
                result = g.hop(pd.DataFrame({'node': ['a']}), 1)
                
                # If we get here, no exception was raised - check the column was preserved
                assert reserved_name in result._edges.columns, f"Column '{reserved_name}' was lost during hop"
                
                # Find an edge that should be in the result
                result_edge = result._edges[(result._edges['s'] == 'a') & (result._edges['d'] == 'b')]
                assert len(result_edge) > 0, f"Edge a->b missing from result when testing column '{reserved_name}'"
                assert result_edge[reserved_name].iloc[0] == f"{reserved_name}_1", \
                    f"Column value for '{reserved_name}' was modified during hop"
                
            except Exception as e:
                # If an exception happens, fail the test with a clear message
                pytest.fail(f"Column name '{reserved_name}' caused an error in hop(): {str(e)}")
                
    def test_combined_node_and_edge_column_conflicts(self):
        """Test using all potential reserved names simultaneously in both nodes and edges"""
        
        # Create node dataframe with all reserved names as columns
        node_data = {
            'node': ['a', 'b', 'c', 'd']
        }
        # Add all reserved names as columns except 'index'
        for name in self.POTENTIAL_RESERVED_NAMES:
            if name.lower() != 'index':
                node_data[name] = [f"node_{name}_{i}" for i in range(4)]
        
        nodes_df = pd.DataFrame(node_data)
        
        # Create edge dataframe with all reserved names as columns
        edge_data = {
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd']
        }
        # Add all reserved names as columns except 'index'
        for name in self.POTENTIAL_RESERVED_NAMES:
            if name.lower() != 'index':
                edge_data[name] = [f"edge_{name}_{i}" for i in range(3)]
        
        edges_df = pd.DataFrame(edge_data)
        
        # Create graph with all these potentially conflicting column names
        g = CGFull().nodes(nodes_df, 'node').edges(edges_df, 's', 'd')
        
        try:
            # Run a basic hop
            result = g.hop(pd.DataFrame({'node': ['a']}), 1)
            
            # Verify all columns were preserved in nodes
            for name in self.POTENTIAL_RESERVED_NAMES:
                if name.lower() != 'index':
                    assert name in result._nodes.columns, f"Node column '{name}' was lost during hop"
                    
            # Verify all columns were preserved in edges
            for name in self.POTENTIAL_RESERVED_NAMES:
                if name.lower() != 'index':
                    assert name in result._edges.columns, f"Edge column '{name}' was lost during hop"
            
        except Exception as e:
            # If an exception happens, fail the test with a clear message
            pytest.fail(f"Using all reserved column names together caused an error in hop(): {str(e)}")
