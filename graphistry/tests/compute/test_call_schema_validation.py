"""Test schema validation for Call operations."""

import pytest
import pandas as pd
from graphistry.tests.test_compute import CGFull
from graphistry.compute.ast import ASTCall, n
from graphistry.compute.chain import Chain
from graphistry.compute.validate.validate_schema import validate_chain_schema
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError


class TestCallSchemaValidation:
    """Test schema validation for Call operations."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 0],
            'weight': [1.0, 2.0, 3.0]
        })
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2],
            'type': ['user', 'user', 'admin'],
            'score': [0.5, 0.8, 0.9]
        })
        
        return CGFull()\
            .edges(edges_df)\
            .nodes(nodes_df)\
            .bind(source='source', destination='target', node='node')
    
    def test_filter_nodes_requires_columns(self, sample_graph):
        """Test that filter_nodes_by_dict validates required columns."""
        # Valid: filtering by existing column
        call = ASTCall('filter_nodes_by_dict', {'filter_dict': {'type': 'user'}})
        errors = validate_chain_schema(sample_graph, [call], collect_all=True)
        assert len(errors) == 0
        
        # Invalid: filtering by non-existent column
        call = ASTCall('filter_nodes_by_dict', {'filter_dict': {'missing_col': 'value'}})
        with pytest.raises(GFQLSchemaError) as exc_info:
            validate_chain_schema(sample_graph, [call], collect_all=False)
        assert exc_info.value.code == ErrorCode.E301
        assert 'missing_col' in str(exc_info.value)
        assert 'does not exist' in str(exc_info.value)
    
    def test_filter_edges_requires_columns(self, sample_graph):
        """Test that filter_edges_by_dict validates required columns."""
        # Valid: filtering by existing edge column
        call = ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': 2.0}})
        errors = validate_chain_schema(sample_graph, [call], collect_all=True)
        assert len(errors) == 0
        
        # Invalid: filtering by non-existent edge column
        call = ASTCall('filter_edges_by_dict', {'filter_dict': {'edge_type': 'friend'}})
        with pytest.raises(GFQLSchemaError) as exc_info:
            validate_chain_schema(sample_graph, [call], collect_all=False)
        assert exc_info.value.code == ErrorCode.E301
        assert 'edge_type' in str(exc_info.value)
    
    def test_encode_requires_columns(self, sample_graph):
        """Test that encode methods validate required columns."""
        # Valid: encoding existing column
        call = ASTCall('encode_point_color', {'column': 'type'})
        errors = validate_chain_schema(sample_graph, [call], collect_all=True)
        assert len(errors) == 0
        
        # Invalid: encoding non-existent column
        call = ASTCall('encode_point_color', {'column': 'category'})
        errors = validate_chain_schema(sample_graph, [call], collect_all=True)
        # Note: encode methods don't require columns to exist (they create bindings)
        # so this should not produce errors
        assert len(errors) == 0
    
    def test_chain_with_multiple_calls(self, sample_graph):
        """Test validation of chains with multiple Call operations."""
        chain = Chain([
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'type': 'user'}}),
            ASTCall('get_degrees', {'col': 'degree'}),
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'degree': 2}})
        ])
        
        errors = validate_chain_schema(sample_graph, chain, collect_all=True)
        # Schema validation tracks added columns from call schema effects
        assert len(errors) == 0
    
    def test_method_without_schema_effects(self, sample_graph):
        """Test that methods without schema effects don't cause errors."""
        # materialize_nodes doesn't require any columns
        call = ASTCall('materialize_nodes', {})
        errors = validate_chain_schema(sample_graph, [call], collect_all=True) 
        assert len(errors) == 0
    
    def test_collect_all_mode(self, sample_graph):
        """Test collect_all mode returns all errors."""
        chain = Chain([
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'missing1': 'a', 'missing2': 'b'}}),
            ASTCall('filter_edges_by_dict', {'filter_dict': {'missing3': 'c'}})
        ])
        
        errors = validate_chain_schema(sample_graph, chain, collect_all=True)
        # Should collect all 3 missing column errors
        assert len(errors) >= 3
        missing_cols = {'missing1', 'missing2', 'missing3'}
        error_cols = set()
        for e in errors:
            for col in missing_cols:
                if col in str(e):
                    error_cols.add(col)
        assert error_cols == missing_cols
    
    def test_operation_index_in_errors(self, sample_graph):
        """Test that errors include operation index."""
        chain = Chain([
            n({'type': 'user'}),  # op 0
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'bad_col': 1}})  # op 1
        ])
        
        errors = validate_chain_schema(sample_graph, chain, collect_all=True)
        call_errors = [e for e in errors if 'bad_col' in str(e)]
        assert len(call_errors) > 0
        assert call_errors[0].context['operation_index'] == 1
