"""Tests for pre-execution schema validation."""

import pytest
import pandas as pd
from graphistry import edges, nodes
from graphistry.compute.chain import Chain
from graphistry.compute.ast import n, e_forward
from graphistry.compute.validate_schema import validate_chain_schema
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError
from graphistry.compute.predicates.numeric import gt
from graphistry.compute.predicates.str import contains


class TestChainSchemaPreValidation:
    """Test schema validation without execution."""
    
    def setup_method(self):
        """Set up test data."""
        self.edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c'],
            'dst': ['b', 'c', 'd'],
            'edge_type': ['friend', 'friend', 'enemy'],
            'weight': [1.0, 2.0, 3.0]
        })
        
        self.nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company'],
            'age': [25, 30, None, None],
            'name': ['Alice', 'Bob', 'Corp1', 'Corp2']
        })
        
        self.g = edges(self.edges_df, 'src', 'dst').nodes(self.nodes_df, 'id')
    
    def test_valid_operations_pass(self):
        """Valid operations pass pre-validation."""
        ops = [
            n({'type': 'person'}),
            e_forward({'edge_type': 'friend'}),
            n({'type': 'company'})
        ]
        
        # Should not raise
        result = validate_chain_schema(self.g, ops)
        assert result is None
    
    def test_missing_node_column_caught(self):
        """Missing node column is caught before execution."""
        ops = [n({'missing_col': 'value'})]
        
        with pytest.raises(GFQLSchemaError) as exc_info:
            validate_chain_schema(self.g, ops)
        
        assert exc_info.value.code == ErrorCode.E301
        assert 'missing_col' in str(exc_info.value)
        assert 'does not exist in node' in str(exc_info.value)
    
    def test_missing_edge_column_caught(self):
        """Missing edge column is caught before execution."""
        ops = [
            n(),
            e_forward({'missing_edge': 'value'})
        ]
        
        with pytest.raises(GFQLSchemaError) as exc_info:
            validate_chain_schema(self.g, ops)
        
        assert exc_info.value.code == ErrorCode.E301
        assert 'missing_edge' in str(exc_info.value)
        assert 'does not exist in edge' in str(exc_info.value)
    
    def test_type_mismatch_caught(self):
        """Type mismatches are caught before execution."""
        # String value on numeric column
        ops = [n({'age': 'twenty-five'})]
        
        with pytest.raises(GFQLSchemaError) as exc_info:
            validate_chain_schema(self.g, ops)
        
        assert exc_info.value.code == ErrorCode.E302
        assert 'numeric but filter value is string' in str(exc_info.value)
    
    def test_predicate_mismatch_caught(self):
        """Predicate type mismatches are caught."""
        # Numeric predicate on string column
        ops = [n({'type': gt(5)})]
        
        with pytest.raises(GFQLSchemaError) as exc_info:
            validate_chain_schema(self.g, ops)
        
        assert exc_info.value.code == ErrorCode.E302
        assert 'numeric predicate used on non-numeric' in str(exc_info.value)
        
        # String predicate on numeric column
        ops = [n({'age': contains('old')})]
        
        with pytest.raises(GFQLSchemaError) as exc_info:
            validate_chain_schema(self.g, ops)
        
        assert exc_info.value.code == ErrorCode.E302
        assert 'string predicate used on non-string' in str(exc_info.value)
    
    def test_collect_all_errors(self):
        """Can collect multiple schema errors."""
        ops = [
            n({'missing1': 'value', 'age': 'string'}),  # 2 errors
            e_forward({'missing2': 'value'}),  # 1 error
            n({'type': gt(5)})  # 1 error
        ]
        
        errors = validate_chain_schema(self.g, ops, collect_all=True)
        assert len(errors) >= 4  # At least 4 errors
        
        # Check operation indices are set
        assert all('operation_index' in e.context for e in errors)
    
    def test_chain_method(self):
        """Chain.validate_schema method works."""
        chain = Chain([
            n({'missing': 'value'})
        ])
        
        # Should raise
        with pytest.raises(GFQLSchemaError):
            chain.validate_schema(self.g)
        
        # Collect all mode
        errors = chain.validate_schema(self.g, collect_all=True)
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E301
    
    def test_edge_source_dest_validation(self):
        """Edge source/destination node filters are validated."""
        # Invalid source node filter
        ops = [
            n(),
            e_forward(source_node_match={'bad_col': 'value'})
        ]
        
        with pytest.raises(GFQLSchemaError) as exc_info:
            validate_chain_schema(self.g, ops)
        
        assert 'does not exist in source node' in str(exc_info.value)
        
        # Invalid destination node filter
        ops = [
            n(),
            e_forward(destination_node_match={'bad_col': 'value'})
        ]
        
        with pytest.raises(GFQLSchemaError) as exc_info:
            validate_chain_schema(self.g, ops)
        
        assert 'does not exist in destination node' in str(exc_info.value)
    
    def test_no_execution_happens(self):
        """Verify that validation doesn't execute the chain."""
        # Create a chain that would fail if executed
        # (since there's no path from person->company via 'missing' edges)
        ops = [
            n({'type': 'person'}),
            e_forward({'edge_type': 'missing'}),  # No such edges
            n({'type': 'company'})
        ]
        
        # Schema validation should pass (columns exist)
        result = validate_chain_schema(self.g, ops)
        assert result is None  # Valid schema
        
        # But execution would return empty results
        # (not testing execution here, just noting the difference)