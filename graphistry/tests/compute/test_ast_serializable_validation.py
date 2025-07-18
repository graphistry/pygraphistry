"""Tests for ASTSerializable validation functionality."""

import pytest
from typing import List

from graphistry.compute.ASTSerializable import ASTSerializable
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.utils.json import JSONVal


class MockValidAST(ASTSerializable):
    """Mock AST that validates successfully."""
    
    def __init__(self, value: str = "valid"):
        self.value = value
    
    def _validate_fields(self):
        # Valid - no errors
        pass


class MockInvalidAST(ASTSerializable):
    """Mock AST that fails validation."""
    
    def __init__(self, value: str = "invalid"):
        self.value = value
    
    def _validate_fields(self):
        raise GFQLValidationError(
            ErrorCode.E101,
            "Mock validation error",
            field="value",
            value=self.value
        )


class MockParentAST(ASTSerializable):
    """Mock parent with children."""
    
    def __init__(self, children: List[ASTSerializable]):
        self.children = children
    
    def _get_child_validators(self):
        return self.children


class TestASTSerializableValidation:
    """Test base validation functionality."""
    
    def test_valid_object_validates(self):
        """Valid object passes validation."""
        obj = MockValidAST()
        result = obj.validate()
        assert result is None  # No errors
        
        # Also works with collect_all
        errors = obj.validate(collect_all=True)
        assert errors == []
    
    def test_invalid_object_fails_fast(self):
        """Invalid object raises on first error by default."""
        obj = MockInvalidAST()
        
        with pytest.raises(GFQLValidationError) as exc_info:
            obj.validate()
        
        assert exc_info.value.code == ErrorCode.E101
        assert "Mock validation error" in str(exc_info.value)
    
    def test_invalid_object_collect_all(self):
        """Invalid object returns errors when collect_all=True."""
        obj = MockInvalidAST()
        
        errors = obj.validate(collect_all=True)
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E101
        assert errors[0].message == "Mock validation error"
    
    def test_parent_validates_children_fail_fast(self):
        """Parent validates children in fail-fast mode."""
        child1 = MockValidAST()
        child2 = MockInvalidAST()
        parent = MockParentAST([child1, child2])
        
        with pytest.raises(GFQLValidationError) as exc_info:
            parent.validate()
        
        assert exc_info.value.code == ErrorCode.E101
    
    def test_parent_collects_child_errors(self):
        """Parent collects all child errors."""
        child1 = MockInvalidAST("error1")
        child2 = MockInvalidAST("error2")
        parent = MockParentAST([child1, child2])
        
        errors = parent.validate(collect_all=True)
        assert len(errors) == 2
        assert all(e.code == ErrorCode.E101 for e in errors)
        assert errors[0].context['value'] == "error1"
        assert errors[1].context['value'] == "error2"
    
    def test_nested_validation(self):
        """Nested structures validate correctly."""
        # grandchild -> child -> parent
        grandchild = MockInvalidAST("nested_error")
        child = MockParentAST([grandchild])
        parent = MockParentAST([child])
        
        # Fail fast mode
        with pytest.raises(GFQLValidationError) as exc_info:
            parent.validate()
        assert "nested_error" in str(exc_info.value)
        
        # Collect all mode
        errors = parent.validate(collect_all=True)
        assert len(errors) == 1
        assert errors[0].context['value'] == "nested_error"
    
    def test_to_json_validates_by_default(self):
        """to_json validates by default."""
        obj = MockInvalidAST()
        
        with pytest.raises(GFQLValidationError):
            obj.to_json()
    
    def test_to_json_skip_validation(self):
        """to_json can skip validation."""
        obj = MockInvalidAST()
        
        # Should not raise
        result = obj.to_json(validate=False)
        assert result['type'] == 'MockInvalidAST'
        assert result['value'] == 'invalid'
    
    def test_from_json_validates_by_default(self):
        """from_json validates by default."""
        # Need a class that implements from_json properly
        class TestAST(ASTSerializable):
            def __init__(self, value: int):
                self.value = value
            
            def _validate_fields(self):
                if self.value < 0:
                    raise GFQLValidationError(
                        ErrorCode.E103,
                        "Value must be positive",
                        field="value",
                        value=self.value
                    )
        
        # Valid case
        obj = TestAST.from_json({'type': 'TestAST', 'value': 5})
        assert obj.value == 5
        
        # Invalid case
        with pytest.raises(GFQLValidationError) as exc_info:
            TestAST.from_json({'type': 'TestAST', 'value': -1})
        assert exc_info.value.code == ErrorCode.E103
    
    def test_from_json_skip_validation(self):
        """from_json can skip validation."""
        class TestAST(ASTSerializable):
            def __init__(self, value: int):
                self.value = value
            
            def _validate_fields(self):
                if self.value < 0:
                    raise GFQLValidationError(ErrorCode.E103, "Negative value")
        
        # Should not raise even with invalid data
        obj = TestAST.from_json({'type': 'TestAST', 'value': -1}, validate=False)
        assert obj.value == -1
        
        # But manual validation would fail
        with pytest.raises(GFQLValidationError):
            obj.validate()
    
    def test_non_validation_errors_propagate(self):
        """Non-validation errors are not caught."""
        class BrokenAST(ASTSerializable):
            def _validate_fields(self):
                raise ValueError("Not a validation error")
        
        obj = BrokenAST()
        
        # Should propagate in both modes
        with pytest.raises(ValueError, match="Not a validation error"):
            obj.validate()
        
        with pytest.raises(ValueError, match="Not a validation error"):
            obj.validate(collect_all=True)
