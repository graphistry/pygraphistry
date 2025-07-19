"""Test error handling and messages in AST serialization"""

import pytest
from graphistry.compute.ast import from_json


class TestSerializationErrors:
    """Test error handling in JSON serialization/deserialization"""
    
    def test_from_json_non_dict_input(self):
        """Test clear error when input is not a dict"""
        with pytest.raises(AssertionError) as exc_info:
            from_json("not a dict")
        
        assert "Expected dict for JSON deserialization, got str" in str(exc_info.value)
    
    def test_from_json_none_input(self):
        """Test clear error when input is None"""
        with pytest.raises(AssertionError) as exc_info:
            from_json(None)
        
        assert "Expected dict for JSON deserialization, got NoneType" in str(exc_info.value)
    
    def test_from_json_missing_type(self):
        """Test clear error when 'type' field is missing"""
        with pytest.raises(AssertionError) as exc_info:
            from_json({"no_type": "value"})
        
        assert "JSON object missing required 'type' field" in str(exc_info.value)
    
    def test_from_json_unknown_type(self):
        """Test clear error for unknown type"""
        with pytest.raises(ValueError) as exc_info:
            from_json({"type": "UnknownType"})
        
        assert "Unknown type UnknownType" in str(exc_info.value)
    
    def test_edge_missing_direction(self):
        """Test clear error when Edge missing direction"""
        with pytest.raises(ValueError) as exc_info:
            from_json({"type": "Edge"})
        
        assert "Edge missing direction" in str(exc_info.value)
    
    def test_edge_invalid_direction(self):
        """Test clear error for invalid Edge direction"""
        with pytest.raises(ValueError) as exc_info:
            from_json({"type": "Edge", "direction": "invalid"})
        
        assert "Edge has unknown direction invalid" in str(exc_info.value)
    
    def test_querydag_missing_bindings(self):
        """Test clear error when QueryDAG missing bindings"""
        with pytest.raises(AssertionError) as exc_info:
            from_json({"type": "QueryDAG"})
        
        assert "QueryDAG missing bindings" in str(exc_info.value)
    
    def test_remotegraph_missing_dataset_id(self):
        """Test clear error when RemoteGraph missing dataset_id"""
        with pytest.raises(AssertionError) as exc_info:
            from_json({"type": "RemoteGraph"})
        
        assert "RemoteGraph missing dataset_id" in str(exc_info.value)
    
    def test_chainref_missing_ref(self):
        """Test clear error when ChainRef missing ref"""
        with pytest.raises(AssertionError) as exc_info:
            from_json({"type": "ChainRef"})
        
        assert "ChainRef missing ref" in str(exc_info.value)
    
    def test_chainref_missing_chain(self):
        """Test clear error when ChainRef missing chain"""
        with pytest.raises(AssertionError) as exc_info:
            from_json({"type": "ChainRef", "ref": "test"})
        
        assert "ChainRef missing chain" in str(exc_info.value)