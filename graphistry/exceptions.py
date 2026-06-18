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


class ArrowConversionError(Exception):
    """Raised in strict mode when Arrow conversion fails due to mixed types."""
    def __init__(self, columns: list, original_error: Exception):
        self.columns = columns
        self.original_error = original_error
        msg = (f"Arrow conversion failed for columns {columns}. "
               f"Use validate='autofix' to auto-coerce to strings.")
        super().__init__(msg)


class SchemaValidationError(Exception):
    """Raised when opt-in schema boundary validation fails."""

    def __init__(
        self,
        *,
        table: str,
        column: str,
        reason: str,
        expected: object = None,
        actual: object = None,
    ):
        self.table = table
        self.column = column
        self.reason = reason
        self.expected = expected
        self.actual = actual
        details = f"Schema validation failed for {table}.{column}: {reason}."
        if expected is not None:
            details += f" Expected {expected!r}."
        if actual is not None:
            details += f" Actual {actual!r}."
        super().__init__(details)
