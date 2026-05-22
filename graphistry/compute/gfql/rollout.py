"""GFQL rollout gates: env-driven canary toggles for staged adoption.

Stable contract surface for T5 (#1311) under #1262 / #1046.

This module retains the historical strict-schema resolver surface. As of
#1420, the Cypher binder is strict by construction, so false resolver outputs
are compatibility values for callers that still inspect the helper and do not
restore loose binder behavior.

Precedence (most specific wins):
    1. Explicit caller parameter (e.g. ``FrontendBinder.bind(strict_name_resolution=True)``)
    2. Catalog-level metadata flag (e.g. ``GraphSchemaCatalog.metadata['strict']``)
    3. Process-wide env default (e.g. ``GRAPHISTRY_GFQL_STRICT_SCHEMA``)
    4. Historical loose default for helper-only consumers

The env tier remains default-off at the helper layer; binder execution no
longer treats that default as permission for loose compatibility paths.

Production binder callers no longer use this helper as a loose/strict gate.
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = [
    "STRICT_SCHEMA_ENV",
    "ENV_TRUTHY",
    "ENV_FALSY",
    "env_bool",
    "strict_schema_env_default",
    "resolve_strict_schema",
]

STRICT_SCHEMA_ENV: str = "GRAPHISTRY_GFQL_STRICT_SCHEMA"

ENV_TRUTHY: "frozenset[str]" = frozenset({"1", "true", "yes", "on"})
ENV_FALSY: "frozenset[str]" = frozenset({"0", "false", "no", "off"})


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean env var. Unset / empty / unrecognized -> ``default``."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ENV_TRUTHY:
        return True
    if raw in ENV_FALSY:
        return False
    return default


def strict_schema_env_default() -> bool:
    """Return the env-default for strict schema mode (default off)."""
    return env_bool(STRICT_SCHEMA_ENV, default=False)


def resolve_strict_schema(
    *,
    explicit: bool,
    catalog_strict: Optional[bool],
) -> bool:
    """Apply strict-schema precedence for helper consumers.

    A monotonic widening: any tier that asks for strict wins. Explicit
    ``False`` does not force loose binder behavior.
    """
    if explicit:
        return True
    if catalog_strict:
        return True
    return strict_schema_env_default()
