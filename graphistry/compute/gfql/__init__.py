"""GFQL validation and related utilities."""

import warnings

# Suppress deprecation warning from validate module - we intentionally re-export
# these symbols for backwards compatibility
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="graphistry.compute.gfql.validate")
    from graphistry.compute.gfql.validate import (
        ValidationIssue,
        Schema,
        validate_syntax,
        validate_schema,
        validate_query,
        extract_schema,
        extract_schema_from_dataframes,
        format_validation_errors,
        suggest_fixes
    )

from graphistry.compute.gfql.exceptions import (
    GFQLException,
    GFQLValidationError,
    GFQLSyntaxError,
    GFQLSchemaError,
    GFQLTypeError,
    GFQLColumnNotFoundError
)

__all__ = [
    # Validation classes
    'ValidationIssue',
    'Schema',
    
    # Validation functions
    'validate_syntax',
    'validate_schema', 
    'validate_query',
    'extract_schema',
    'extract_schema_from_dataframes',
    'format_validation_errors',
    'suggest_fixes',
    
    # Exceptions
    'GFQLException',
    'GFQLValidationError',
    'GFQLSyntaxError',
    'GFQLSchemaError',
    'GFQLTypeError',
    'GFQLColumnNotFoundError'
]
