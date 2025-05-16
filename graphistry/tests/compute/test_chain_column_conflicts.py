import pandas as pd
import pytest
from graphistry.tests.common import NoAuthTestCase

from graphistry.tests.test_compute import CGFull
from graphistry.compute.ast import n, e, e_forward, e_reverse, e_undirected


class TestChainColumnConflicts(NoAuthTestCase):
    """Tests for potential column name conflicts in chain.py"""
    
    # List of potential internal variable names that might conflict as column names
    POTENTIAL_RESERVED_NAMES = [
        # Chain processing internal variables
        'wave_front', 'matches_nodes', 'matches_edges', 'combined_node_ids',
        'id', 'prev_node_wavefront', 'target_wave_front', 'engine',
        
        # DataFrame operation terms
        'how', 'on', 'sort', 'ignore_index', 'subset'
    ]
    
    def test_node_column_same_as_source(self):
        """Test chain with node ID column having same name as edge source column"""
        
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
        
        # Chain should work with node/source name conflict
        result = g.chain([n(), e(), n()])
        
        # Verify the chain operation worked correctly
        assert result._nodes.shape[0] > 0
        assert result._edges.shape[0] > 0
        assert 's' in result._nodes.columns
    
    def test_node_column_same_as_destination(self):
        """Test chain with node ID column having same name as edge destination column"""
        
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
        
        # Chain should work with node/destination name conflict
        result = g.chain([n(), e(), n()])
        
        # Verify the chain operation worked correctly
        assert result._nodes.shape[0] > 0
        assert result._edges.shape[0] > 0
        assert 'd' in result._nodes.columns
    
    def test_direction_variants_simpler(self):
        """Test direction variants with column conflicts using a simpler test case"""
        
        # Create a basic graph with node column that matches source or destination
        nodes_s_df = pd.DataFrame({'s': ['a', 'b', 'c']})
        nodes_d_df = pd.DataFrame({'d': ['a', 'b', 'c']})
        
        edges_df = pd.DataFrame({
            's': ['a', 'b'],
            'd': ['b', 'c']
        })
        
        # Create graphs with conflicting column names
        g_source_conflict = CGFull().nodes(nodes_s_df, 's').edges(edges_df, 's', 'd')
        g_dest_conflict = CGFull().nodes(nodes_d_df, 'd').edges(edges_df, 's', 'd')
        
        # Basic tests with different directions - node/source conflict
        result1 = g_source_conflict.chain([n({'s': 'a'}), e_forward(), n()])
        assert result1._nodes.shape[0] > 0
        assert 's' in result1._nodes.columns
        
        # Basic tests with different directions - node/destination conflict
        result2 = g_dest_conflict.chain([n({'d': 'b'}), e(), n()])
        assert result2._nodes.shape[0] > 0
        assert 'd' in result2._nodes.columns
    
    def test_chain_operations_with_risky_column_names(self):
        """Test chain operations with column names that might conflict with internal variables"""
        
        # Create test data with potentially risky column names
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'wave_front': [1, 2, 3, 4],
            'target_wave_front': [5, 6, 7, 8],
            'prev_node_wavefront': [9, 10, 11, 12]
        })
        
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd'],
            'matches_nodes': ['x', 'y', 'z'],
            'matches_edges': ['p', 'q', 'r'],
            'combined_node_ids': ['m', 'n', 'o']
        })
        
        # Create graph with potentially conflicting column names
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Run chain operations
        result = g.chain([n({'id': 'a'}), e_forward(), n()])
        
        # Verify operation worked correctly and preserved all columns
        assert result._nodes.shape[0] > 0
        assert result._edges.shape[0] > 0
        
        # Check that all original columns are preserved
        for col in nodes_df.columns:
            assert col in result._nodes.columns
            
        for col in edges_df.columns:
            assert col in result._edges.columns
    
    def test_multiple_hops_with_conflicts(self):
        """Test chain with multiple hops and column name conflicts"""
        
        # Create test data with column name conflicts
        nodes_df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd', 'e', 'f']  # Same as source column
        })
        
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd', 'e'],
            'd': ['b', 'c', 'd', 'e', 'f']
        })
        
        # Create graph with conflicting column names
        g = CGFull().nodes(nodes_df, 's').edges(edges_df, 's', 'd')
        
        # Run a simpler chain with filter on start node
        result = g.chain([
            n({'s': 'a'}),
            e(),
            n()
        ])
        
        # Verify operation worked correctly
        assert result._nodes.shape[0] > 0
        assert 's' in result._nodes.columns
        
        # Check that we found the expected node
        node_ids = result._nodes['s'].tolist()
        assert 'b' in node_ids
    
    def test_named_nodes_and_edges_with_conflicts(self):
        """Test chain with named nodes/edges and column name conflicts"""
        
        # Create test data with column name conflicts
        nodes_df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd']  # Same as source column
        })
        
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd']
        })
        
        # Create graph with conflicting column names
        g = CGFull().nodes(nodes_df, 's').edges(edges_df, 's', 'd')
        
        # Run chain with a simpler named pattern
        result = g.chain([
            n({'s': 'a'}, name='start'), 
            e(name='hop'), 
            n(name='end')
        ])
        
        # Verify operation worked correctly
        assert result._nodes.shape[0] > 0
        
        # Check that named columns exist
        assert 'start' in result._nodes.columns
        assert 'end' in result._nodes.columns
        assert 'hop' in result._edges.columns
        
        # Check original column is preserved
        assert 's' in result._nodes.columns
    
    def test_chain_with_column_name_matches_node_edge_keywords(self):
        """Test chain with column names that match node/edge keywords"""
        
        # Create test data with column names that match keywords in n(), e() functions
        nodes_df = pd.DataFrame({
            'node': ['a', 'b', 'c'],  # Matches n() keyword
            'filter_dict': [1, 2, 3],  # Matches internal parameter
            'name': ['x', 'y', 'z']   # Matches name parameter
        })
        
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'a'],
            'direction': ['forward', 'reverse', 'undirected'],  # Matches edge direction
            'hops': [1, 2, 3],  # Matches hops parameter
            'to_fixed_point': [True, False, True]  # Matches to_fixed_point parameter
        })
        
        # Create graph
        g = CGFull().nodes(nodes_df, 'node').edges(edges_df, 's', 'd')
        
        # Run chain
        result = g.chain([n(), e(), n()])
        
        # Verify operation worked correctly and preserved all columns
        assert result._nodes.shape[0] > 0
        assert result._edges.shape[0] > 0
        
        # Check that all original columns are preserved
        for col in nodes_df.columns:
            assert col in result._nodes.columns
            
        for col in edges_df.columns:
            assert col in result._edges.columns
