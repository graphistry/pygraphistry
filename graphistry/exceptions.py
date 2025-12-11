class SsoException(Exception):
    """
    Koa, 15 Sep 2022  Custom Base Exception to handle Sso exception scenario
    """
    pass


class SsoRetrieveTokenTimeoutException(SsoException):
    """
    Koa, 15 Sep 2022  Custom Exception to Sso retrieve token time out exception scenario
    """
    pass

class SsoStateInvalidException(SsoException):
    """
    30 Jun 2025 Raised when the SSO state is missing, invalid, or not initialized during token retrieval.
    """
    pass



class TokenExpireException(Exception):
    """
    Koa, 15 Mar 2024  Custom Exception for JWT Token expiry when refresh
    """
    pass


# =============================================================================
# Validation Exceptions (Issue #867)
# =============================================================================

class ValidationException(Exception):
    """
    Base exception for data validation errors in strict mode.

    Raised when validate='strict' or validate='strict-fast' encounters
    data that cannot be processed without auto-fixing.
    """

    def __init__(self, message: str, context: dict = None):  # type: ignore[assignment]
        self.context = context or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{super().__str__()} ({context_str})"
        return super().__str__()


class ArrowConversionError(ValidationException):
    """
    Raised in strict mode when Arrow conversion fails due to mixed types.

    When validate='strict' or validate='strict-fast', this exception is raised
    instead of auto-coercing mixed-type columns to strings.

    **Example:**
        ::

            import graphistry
            import pandas as pd

            df = pd.DataFrame({
                'src': [1, 2],
                'dst': [2, 1],
                'mixed': [b'bytes', 1.5]  # Mixed types
            })

            # This will raise ArrowConversionError in strict mode
            g = graphistry.edges(df, 'src', 'dst').plot(validate='strict')

            # Use autofix mode to auto-coerce (default behavior)
            g = graphistry.edges(df, 'src', 'dst').plot(validate='autofix')
    """

    def __init__(self, columns: list, original_error: Exception):
        message = (
            f"Arrow conversion failed for columns {columns}. "
            f"Mixed types detected. Use validate='autofix' to auto-coerce to strings, "
            f"or convert these columns explicitly before calling plot()."
        )
        super().__init__(message, {
            'columns': columns,
            'original_error': str(original_error)
        })
