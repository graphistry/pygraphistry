"""Naming conventions for hidden reentry carry columns.

Pure string helpers — no other dependencies. Wrapping carry-column names in a
recognizable namespace (``__cypher_reentry_*__`` and ``__carry_*__*__``) keeps
them away from any user-visible identifier surface and lets the reentry
rewriters detect them via ``_is_hidden_reentry_property``.

Extracted from ``cypher.lowering`` (#1295, #1260 S2).
"""
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
    """Intermediate carry alias for a non-source whole-row alias's property.

    Used as the prefix WITH alias (`WITH a, x.id AS <carry_name>`); the existing
    scalar-carry plumbing wraps this with ``_reentry_hidden_column_name`` to
    produce the final hidden column on the reentry-alias's row table. Trailing
    `<alias>.<prop>` references rewrite to a property access on the reentry
    alias whose property name is that wrapped form, so both sides resolve to
    the same column.

    The leading ``__carry_`` namespace plus surrounding underscores keep the
    intermediate alias away from any user identifier that might survive in the
    same prefix projection.
    """
    return f"__carry_{alias}__{prop}__"


def _secondary_reentry_hidden_column_name(alias: str, prop: str) -> str:
    """Hidden carry column name for a secondary whole-row alias's property access.

    Distinct from `_reentry_hidden_column_name` (which is keyed only by the
    output name) so secondary `<S>.<X>` carries cannot collide with user-named
    scalar carries on the primary alias (#1071).
    """
    return f"__cypher_reentry_{alias}_{prop}__"


def _is_hidden_reentry_property(property_name: str) -> bool:
    return property_name.startswith("__cypher_reentry_") or property_name.startswith("__gfql_hidden_")
