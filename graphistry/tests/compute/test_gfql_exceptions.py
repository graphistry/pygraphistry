"""Tests for GFQL validation exceptions."""

import pytest
from graphistry.compute.exceptions import (
    ErrorCode, GFQLValidationError, GFQLSyntaxError, 
    GFQLTypeError, GFQLSchemaError
)


class TestErrorCode:
    """Test error code constants."""
    
    def test_error_codes_exist(self):
        """Error codes are defined."""
        assert ErrorCode.E101 == "invalid-chain-type"
        assert ErrorCode.E102 == "invalid-filter-key"
        assert ErrorCode.E201 == "type-mismatch"
        assert ErrorCode.E301 == "column-not-found"
    
    def test_error_code_ranges(self):
        """Error codes follow range convention."""
        # E1xx for syntax
        assert ErrorCode.E101.startswith("invalid")
        assert ErrorCode.E106 == "empty-chain"
        
        # E2xx for types
        assert ErrorCode.E201 == "type-mismatch"
        assert ErrorCode.E202 == "predicate-type-mismatch"
        
        # E3xx for schema
        assert ErrorCode.E301 == "column-not-found"
        assert ErrorCode.E302 == "incompatible-column-type"


class TestGFQLValidationError:
    """Test base validation error."""
    
    def test_basic_error_creation(self):
        """Can create basic error."""
        error = GFQLValidationError(ErrorCode.E101, "Test message")
        assert error.code == ErrorCode.E101
        assert error.message == "Test message"
        assert str(error).startswith("[invalid-chain-type]")
        assert "Test message" in str(error)
    
    def test_error_with_context(self):
        """Error includes context fields."""
        error = GFQLValidationError(
            ErrorCode.E102,
            "Invalid filter key",
            field="filter_dict.123",
            value=123,
            suggestion="Use string keys"
        )
        
        formatted = str(error)
        assert "[invalid-filter-key]" in formatted
        assert "field: filter_dict.123" in formatted
        assert "value: 123" in formatted
        assert "suggestion: Use string keys" in formatted
    
    def test_error_with_operation_index(self):
        """Error includes operation index."""
        error = GFQLValidationError(
            ErrorCode.E101,
            "Bad operation",
            operation_index=2
        )
        assert "at operation 2" in str(error)
    
    def test_error_truncates_long_values(self):
        """Long values are truncated in string representation."""
        long_value = "x" * 100
        error = GFQLValidationError(
            ErrorCode.E201,
            "Value too long",
            value=long_value
        )
        formatted = str(error)
        assert "..." in formatted
        assert len(formatted) < 200  # Reasonable length
    
    def test_error_to_dict(self):
        """Error converts to dictionary."""
        error = GFQLValidationError(
            ErrorCode.E102,
            "Test error",
            field="test_field",
            value="test_value",
            custom_field="custom"
        )
        
        d = error.to_dict()
        assert d['code'] == ErrorCode.E102
        assert d['message'] == "Test error"
        assert d['field'] == "test_field"
        assert d['value'] == "test_value"
        assert d['custom_field'] == "custom"
    
    def test_error_filters_none_context(self):
        """None values are filtered from context."""
        error = GFQLValidationError(
            ErrorCode.E101,
            "Test",
            field=None,
            value="something",
            suggestion=None
        )
        
        assert 'field' not in error.context
        assert 'suggestion' not in error.context
        assert error.context['value'] == "something"


class TestErrorSubclasses:
    """Test error subclasses."""
    
    def test_syntax_error(self):
        """GFQLSyntaxError works correctly."""
        error = GFQLSyntaxError(
            ErrorCode.E101,
            "Invalid syntax",
            field="chain"
        )
        assert isinstance(error, GFQLValidationError)
        assert error.code == ErrorCode.E101
        assert "Invalid syntax" in str(error)
    
    def test_type_error(self):
        """GFQLTypeError works correctly."""
        error = GFQLTypeError(
            ErrorCode.E201,
            "Type mismatch",
            field="hops",
            value="not_a_number"
        )
        assert isinstance(error, GFQLValidationError)
        assert error.code == ErrorCode.E201
        assert "Type mismatch" in str(error)
        assert "not_a_number" in str(error)
    
    def test_schema_error(self):
        """GFQLSchemaError works correctly."""
        error = GFQLSchemaError(
            ErrorCode.E301,
            "Column not found",
            field="user_id",
            suggestion="Available columns: id, name, email"
        )
        assert isinstance(error, GFQLValidationError)
        assert error.code == ErrorCode.E301
        assert "Column not found" in str(error)
        assert "Available columns" in str(error)
    
    def test_error_inheritance(self):
        """All errors inherit from base class."""
        errors = [
            GFQLSyntaxError(ErrorCode.E101, "test"),
            GFQLTypeError(ErrorCode.E201, "test"),
            GFQLSchemaError(ErrorCode.E301, "test")
        ]
        
        for error in errors:
            assert isinstance(error, GFQLValidationError)
            assert isinstance(error, Exception)
            assert hasattr(error, 'code')
            assert hasattr(error, 'to_dict')
