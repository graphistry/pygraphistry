"""GFQL validation exceptions with structured error codes."""

from typing import Optional, Any, Dict


class ErrorCode:
    """Error codes for GFQL validation errors.

    Error code ranges:
    - E1xx: Syntax errors (structural issues)
    - E2xx: Type errors (type mismatches)
    - E3xx: Schema errors (data-related issues)
    """

    # Syntax errors (E1xx)
    E101 = "invalid-chain-type"
    E102 = "invalid-filter-key"
    E103 = "invalid-hops-value"
    E104 = "invalid-direction"
    E105 = "missing-required-field"
    E106 = "empty-chain"

    # Type errors (E2xx)
    E201 = "type-mismatch"
    E202 = "predicate-type-mismatch"
    E203 = "invalid-predicate-value"
    E204 = "invalid-name-type"
    E205 = "invalid-query-type"

    # Schema errors (E3xx) - for future use
    E301 = "column-not-found"
    E302 = "incompatible-column-type"
    E303 = "invalid-node-reference"
    E304 = "invalid-edge-reference"


class GFQLValidationError(Exception):
    """Base class for GFQL validation errors with structured information."""

    def __init__(self,
                 code: str,
                 message: str,
                 field: Optional[str] = None,
                 value: Optional[Any] = None,
                 suggestion: Optional[str] = None,
                 operation_index: Optional[int] = None,
                 **extra_context):
        """Initialize validation error with structured information.

        Args:
            code: Error code from ErrorCode class
            message: Human-readable error message
            field: Field that caused the error (e.g., "filter_dict.user_id")
            value: The invalid value that caused the error
            suggestion: Helpful suggestion for fixing the error
            operation_index: Index in chain where error occurred
            **extra_context: Additional context information
        """
        self.code = code
        self.message = message
        self.context = {
            'field': field,
            'value': value,
            'suggestion': suggestion,
            'operation_index': operation_index,
            **extra_context
        }
        # Remove None values from context
        self.context = {k: v for k, v in self.context.items() if v is not None}

        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format error message with code and context."""
        parts = [f"[{self.code}] {self.message}"]

        if 'field' in self.context:
            parts.append(f"field: {self.context['field']}")

        if 'value' in self.context:
            # Truncate long values
            val_str = repr(self.context['value'])
            if len(val_str) > 50:
                val_str = val_str[:47] + "..."
            parts.append(f"value: {val_str}")

        if 'operation_index' in self.context:
            parts.append(f"at operation {self.context['operation_index']}")

        if 'suggestion' in self.context:
            parts.append(f"suggestion: {self.context['suggestion']}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for structured output."""
        return {
            'code': self.code,
            'message': self.message,
            **self.context
        }


class GFQLSyntaxError(GFQLValidationError):
    """Syntax errors in GFQL query structure."""
    pass


class GFQLTypeError(GFQLValidationError):
    """Type mismatches in GFQL queries."""
    pass


class GFQLSchemaError(GFQLValidationError):
    """Schema validation errors (column existence, type compatibility)."""
    pass
