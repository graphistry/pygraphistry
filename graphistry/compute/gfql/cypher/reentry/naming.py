"""Naming conventions for hidden reentry carry columns."""
from __future__ import annotations

__all__ = [
    "_reentry_hidden_column_name",
    "_reentry_property_carry_name",
    "_secondary_reentry_hidden_column_name",
    "_is_hidden_reentry_property",
]


def _reentry_hidden_column_name(output_name: str) -> str:
    return f"__cypher_reentry_{output_name}__"


def _reentry_property_carry_name(alias: str, prop: str) -> str:
    """Intermediate carry alias for a non-source whole-row alias's property."""
    return f"__carry_{alias}__{prop}__"


def _secondary_reentry_hidden_column_name(alias: str, prop: str) -> str:
    """Hidden carry column for a secondary whole-row alias's property."""
    return f"__cypher_reentry_{alias}_{prop}__"


def _is_hidden_reentry_property(property_name: str) -> bool:
    return property_name.startswith("__cypher_reentry_") or property_name.startswith("__gfql_hidden_")
