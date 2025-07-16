"""Tests for Chain schema validation (data-aware validation)."""

import pytest
import pandas as pd
from graphistry import edges, nodes
from graphistry.compute.chain import Chain
from graphistry.compute.ast import n, e_forward
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError


class TestChainSchemaValidation:
    """Test schema-aware validation in chain() function."""
    
    def setup_method(self):
        """Set up test data."""
        # Simple test graph with valid paths
        self.edges_df = pd.DataFrame({
            'src': ['a', 'b', 'b'],
            'dst': ['b', 'c', 'd'],
            'edge_type': ['friend', 'friend', 'enemy']
        })
        
        self.nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company'],
            'age': [25, 30, None, None]
        })
        
        self.g = edges(self.edges_df, 'src', 'dst').nodes(self.nodes_df, 'id')
    
    def test_valid_schema_operations(self):
        """Valid operations pass schema validation."""
        # These should work - columns exist
        result = self.g.chain([
            n({'type': 'person'}),
            e_forward({'edge_type': 'friend'}),
            n({'type': 'company'})
        ])
        
        # Should have some results
        assert len(result._nodes) > 0
    
    def test_nonexistent_node_column(self):
        """Reference to non-existent node column fails."""
        with pytest.raises(GFQLSchemaError) as exc_info:
            self.g.chain([
                n({'missing_column': 'value'})
            ])
        
        assert exc_info.value.code == ErrorCode.E301
        assert 'missing_column' in str(exc_info.value)
        assert 'does not exist' in str(exc_info.value)
    
    def test_nonexistent_edge_column(self):
        """Reference to non-existent edge column fails."""
        with pytest.raises(GFQLSchemaError) as exc_info:
            self.g.chain([
                n(),
                e_forward({'missing_edge_col': 'value'})
            ])
        
        assert exc_info.value.code == ErrorCode.E301
        assert 'missing_edge_col' in str(exc_info.value)
    
    def test_type_mismatch_numeric_filter(self):
        """Type mismatch in filter fails."""
        # 'type' column contains strings, not numbers
        with pytest.raises(GFQLSchemaError) as exc_info:
            self.g.chain([
                n({'type': 123})  # Wrong type
            ])
        
        assert exc_info.value.code == ErrorCode.E302
        assert 'type mismatch' in str(exc_info.value).lower()
    
    def test_empty_graph_warning(self):
        """Empty graph should provide helpful error."""
        empty_edges = pd.DataFrame({'s': [], 'd': []})
        empty_nodes = pd.DataFrame({'id': []})
        empty_g = edges(empty_edges, 's', 'd').nodes(empty_nodes, 'id')
        
        # Should succeed but return empty result
        result = empty_g.chain([n()])
        assert len(result._nodes) == 0
        
        # But filtering on non-existent column should still fail
        with pytest.raises(GFQLSchemaError) as exc_info:
            empty_g.chain([n({'any_col': 'value'})])
        
        assert exc_info.value.code == ErrorCode.E301
    
    def test_collect_all_schema_errors(self):
        """Can collect multiple schema errors."""
        chain_obj = Chain([
            n({'missing1': 'value'}),
            e_forward({'missing2': 'value'}),
            n({'type': 123})  # Wrong type
        ])
        
        # Note: chain() function would need collect_all parameter
        # For now, it will fail fast on first error
        with pytest.raises(GFQLSchemaError):
            self.g.chain(chain_obj)
    
    def test_schema_validation_with_predicates(self):
        """Schema validation works with predicates."""
        from graphistry.compute.predicates.numeric import gt
        
        # Valid predicate on numeric column
        result = self.g.chain([
            n({'age': gt(20)})
        ])
        assert len(result._nodes) == 2  # Only persons have age
        
        # Invalid predicate on non-numeric column
        with pytest.raises(GFQLSchemaError) as exc_info:
            self.g.chain([
                n({'type': gt(20)})  # String column
            ])
        
        assert exc_info.value.code == ErrorCode.E302
    
    def test_schema_validation_disabled(self):
        """Schema validation can be disabled."""
        # This would need a parameter like validate_schema=False
        # For now, document expected behavior
        
        # Future API:
        # result = self.g.chain([n({'missing': 'value'})], validate_schema=False)
        # Would return empty result instead of raising
        pass