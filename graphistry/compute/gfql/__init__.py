"""GFQL validation and related utilities."""

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

from graphistry.compute.gfql.rollout import (
    STRICT_SCHEMA_ENV,
    env_bool,
    resolve_strict_schema,
    strict_schema_env_default,
)

from graphistry.compute.gfql.layout import (
    LAYOUT_FUNCTION_NAMES,
    LAYOUT_KINDS,
    RADIAL_LAYOUT_FUNCTION_NAMES,
    RADIAL_LAYOUT_KINDS,
    is_layout_chain,
    is_layout_kind,
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
    'GFQLColumnNotFoundError',

    # Rollout gates (T5 #1311)
    'STRICT_SCHEMA_ENV',
    'env_bool',
    'resolve_strict_schema',
    'strict_schema_env_default',

    # Layout chain helpers
    'LAYOUT_FUNCTION_NAMES',
    'LAYOUT_KINDS',
    'RADIAL_LAYOUT_FUNCTION_NAMES',
    'RADIAL_LAYOUT_KINDS',
    'is_layout_chain',
    'is_layout_kind',
]
