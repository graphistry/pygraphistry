"""GFQL-specific exceptions for validation and error handling."""

from typing import Optional, Dict, Any


class GFQLException(Exception):
    """Base exception for all GFQL-related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{super().__str__()} ({context_str})"
        return super().__str__()


class GFQLValidationError(GFQLException, ValueError):
    """Base validation error. Inherits from ValueError for backwards compatibility."""
    pass


class GFQLSyntaxError(GFQLValidationError):
    """Raised when GFQL query has invalid syntax."""
    pass


class GFQLSchemaError(GFQLValidationError):
    """Raised when GFQL query references non-existent columns or has type mismatches."""
    pass


class GFQLSemanticError(GFQLValidationError):
    """Raised when GFQL query is syntactically valid but semantically incorrect."""
    pass


class GFQLTypeError(GFQLSchemaError):
    """Raised when a predicate is applied to incompatible column type."""
    
    def __init__(self, column: str, column_type: str, predicate: str, expected_type: str):
        message = f"Column '{column}' has type '{column_type}' but predicate '{predicate}' expects '{expected_type}'"
        super().__init__(message, {
            'column': column,
            'column_type': column_type,
            'predicate': predicate,
            'expected_type': expected_type
        })


class GFQLColumnNotFoundError(GFQLSchemaError):
    """Raised when a referenced column doesn't exist in the schema."""
    
    def __init__(self, column: str, table: str, available_columns: list):
        message = f"Column '{column}' not found in {table} data"
        super().__init__(message, {
            'column': column,
            'table': table,
            'available_columns': available_columns
        })
