"""GFQL reserved identifiers and validation."""

from typing import Optional, Dict, Any, Set

# Internal column pattern for temporary GFQL columns
INTERNAL_COLUMN_PATTERN: str = '__gfql_*__'
INTERNAL_COLUMN_PREFIX: str = '__gfql_'
INTERNAL_COLUMN_SUFFIX: str = '__'


def is_internal_column(name: str) -> bool:
    """Check if name matches internal column pattern __gfql_*__."""
    return (isinstance(name, str)
            and name.startswith(INTERNAL_COLUMN_PREFIX)
            and name.endswith(INTERNAL_COLUMN_SUFFIX)
            and len(name) > len(INTERNAL_COLUMN_PREFIX) + len(INTERNAL_COLUMN_SUFFIX))


def validate_column_name(name: str, context: str = "Column") -> None:
    """Validate output column name doesn't use internal pattern.

    Used for operation output parameters like get_degrees(col='...').
    """
    if is_internal_column(name):
        raise ValueError(
            f"{context} cannot use column name '{name}'. "
            f"Pattern '{INTERNAL_COLUMN_PATTERN}' is reserved for internal use. "
            f"Choose a different name."
        )


def validate_column_references(
    col_dict: Optional[Dict[str, Any]],
    context: str = "Operation"
) -> None:
    """Validate dict keys don't reference internal columns.

    Internal columns are temporary and won't work with gfql_remote().
    """
    if not col_dict:
        return

    for key in col_dict.keys():
        if is_internal_column(key):
            raise ValueError(
                f"{context} cannot use column '{key}'. "
                f"Pattern '{INTERNAL_COLUMN_PATTERN}' is reserved for internal use. "
                f"These columns are temporary and won't work with gfql_remote()."
            )


# Legacy reserved names (server-side only, not enforced client-side)
RESERVED_COLUMN_NAMES_LEGACY: Set[str] = {'id'}

# Recommended to avoid (may cause confusion in some contexts)
RECOMMENDED_AVOID: Set[str] = {
    'node', 'edge', 'graph',
    'id', 'idx', 'index',
    'src', 'source', 'from',
    'dst', 'dest', 'destination', 'to', 'target',
    'type', 'label', 'name'
}
