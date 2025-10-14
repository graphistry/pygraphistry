"""Centralized registry of reserved identifiers in GFQL.

This module provides a single source of truth for reserved column name patterns
and validation logic, following DRY principles.
"""

from typing import Set

# Internal column pattern (temporary columns created during GFQL operations)
INTERNAL_COLUMN_PATTERN: str = '__gfql_*__'
INTERNAL_COLUMN_PREFIX: str = '__gfql_'
INTERNAL_COLUMN_SUFFIX: str = '__'


def is_internal_column(name: str) -> bool:
    """Check if a column name matches the internal column pattern.

    Internal columns use the pattern '__gfql_*__' and are created temporarily
    during GFQL operations. They should not be referenced in user predicates
    as they may not exist in all contexts, especially with gfql_remote().

    Parameters
    ----------
    name : str
        Column name to check

    Returns
    -------
    bool
        True if name matches __gfql_*__ pattern

    Examples
    --------
    >>> is_internal_column('__gfql_edge_index_0__')
    True
    >>> is_internal_column('my_column')
    False
    >>> is_internal_column('__gfql_')  # Missing suffix
    False
    >>> is_internal_column('__gfql_foo')  # Missing suffix
    False
    >>> is_internal_column('gfql_foo__')  # Missing prefix
    False
    """
    return (
        isinstance(name, str) and
        name.startswith(INTERNAL_COLUMN_PREFIX) and
        name.endswith(INTERNAL_COLUMN_SUFFIX) and
        len(name) > len(INTERNAL_COLUMN_PREFIX) + len(INTERNAL_COLUMN_SUFFIX)
    )


# Reserved column names (legacy - for documentation only)
# Note: These are NOT enforced client-side, only documented
# Server-side enforcement at server/client/graph/processing_handler.py:649
RESERVED_COLUMN_NAMES_LEGACY: Set[str] = {
    'id',  # Server-side restriction (processing_handler.py:649)
}


# Column naming guidelines (for documentation)
# These names work fine in PyGraphistry client but may cause confusion
# or have special meaning in some contexts
RECOMMENDED_AVOID: Set[str] = {
    # Core graph concepts
    'node', 'edge', 'graph',
    # Common identifiers
    'id', 'idx', 'index',
    # Source/destination variants
    'src', 'source', 'from',
    'dst', 'dest', 'destination', 'to', 'target',
    # Type/classification
    'type', 'label', 'name'
}
