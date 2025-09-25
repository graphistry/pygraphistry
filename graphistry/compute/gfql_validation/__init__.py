"""
DEPRECATED: This module is deprecated and will be removed in a future version.

All functionality has been moved to graphistry.compute.gfql.
Please update your imports:
  FROM: graphistry.compute.gfql_validation
  TO:   graphistry.compute.gfql

This duplicate module was created accidentally during code extraction and
provides no additional functionality.
"""

import warnings
from typing import TYPE_CHECKING

# Import everything from the real location
from graphistry.compute.gfql.validate import (  # noqa: E402
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

from graphistry.compute.gfql.exceptions import (  # noqa: E402
    GFQLException,
    GFQLValidationError,
    GFQLSyntaxError,
    GFQLSchemaError,
    GFQLTypeError,
    GFQLColumnNotFoundError
)

# Issue deprecation warning on import
warnings.warn(
    "graphistry.compute.gfql_validation is deprecated and will be removed in a future version. "
    "Please use graphistry.compute.gfql instead. "
    "All functionality is identical - this was a duplicate created during code extraction.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything to maintain backwards compatibility
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
