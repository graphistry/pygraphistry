"""Naming conventions for hidden reentry carry columns."""
from __future__ import annotations

from graphistry.compute.gfql.identifiers import (
    HIDDEN_ALIAS_COLUMN_PREFIX,
    INTERNAL_COLUMN_SUFFIX,
)

__all__ = [
    "REENTRY_HIDDEN_COLUMN_PREFIX",
    "REENTRY_PROPERTY_CARRY_PREFIX",
    "is_reentry_hidden_column_reference",
    "_reentry_hidden_column_name",
    "_reentry_property_carry_name",
    "_secondary_reentry_hidden_column_name",
    "_is_hidden_reentry_property",
]

#: Prefix of every hidden reentry carry column.
REENTRY_HIDDEN_COLUMN_PREFIX: str = "__cypher_reentry_"

#: Prefix of the intermediate carry alias for a non-source whole-row alias's property.
REENTRY_PROPERTY_CARRY_PREFIX: str = "__carry_"

_REENTRY_COLUMN_SUFFIX: str = INTERNAL_COLUMN_SUFFIX


def _reentry_hidden_column_name(output_name: str) -> str:
    return f"{REENTRY_HIDDEN_COLUMN_PREFIX}{output_name}{_REENTRY_COLUMN_SUFFIX}"


def _reentry_property_carry_name(alias: str, prop: str) -> str:
    """Intermediate carry alias for a non-source whole-row alias's property."""
    return f"{REENTRY_PROPERTY_CARRY_PREFIX}{alias}__{prop}{_REENTRY_COLUMN_SUFFIX}"


def _secondary_reentry_hidden_column_name(alias: str, prop: str) -> str:
    """Hidden carry column for a secondary whole-row alias's property."""
    return f"{REENTRY_HIDDEN_COLUMN_PREFIX}{alias}_{prop}{_REENTRY_COLUMN_SUFFIX}"


def _is_hidden_reentry_property(property_name: str) -> bool:
    return (property_name.startswith(REENTRY_HIDDEN_COLUMN_PREFIX)
            or property_name.startswith(HIDDEN_ALIAS_COLUMN_PREFIX))


def is_reentry_hidden_column_reference(expression: str) -> bool:
    """A projection/aggregate source that reaches into a hidden reentry carry column."""
    return REENTRY_HIDDEN_COLUMN_PREFIX in expression
