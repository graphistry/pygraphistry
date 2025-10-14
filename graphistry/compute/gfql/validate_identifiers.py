"""
Validation utilities for GFQL identifiers and column references.
"""

from typing import Optional, Dict, Any
from graphistry.compute.reserved_identifiers import (
    is_internal_column,
    INTERNAL_COLUMN_PATTERN
)


def validate_column_references(
    filter_dict: Optional[Dict[str, Any]],
    context: str = "Operation"
) -> None:
    """
    Validate that column references don't use internal GFQL identifiers.

    Internal columns using pattern '__gfql_*__' are temporary and only exist
    during certain operations. Filtering on them would be unpredictable and
    likely fail when using gfql_remote().

    Parameters
    ----------
    filter_dict : Optional[Dict[str, Any]]
        Dictionary with column names as keys and filter values
    context : str
        Context string for error message (e.g., "n()", "e_forward()")

    Raises
    ------
    ValueError
        If any key matches the internal column pattern '__gfql_*__'

    Examples
    --------
    >>> validate_column_references({'col': 1, 'val': 2})  # OK
    >>> validate_column_references({'__gfql_temp__': 1})  # Raises ValueError
    """
    if not filter_dict:
        return

    for key in filter_dict.keys():
        if is_internal_column(key):
            raise ValueError(
                f"{context} cannot filter on column '{key}'. "
                f"Column names matching pattern '{INTERNAL_COLUMN_PATTERN}' are reserved for internal use "
                f"and cannot be used in filter predicates. These columns are temporary "
                f"and would not work with gfql_remote()."
            )
