"""GFQL rollout gates: env-driven canary toggles for staged adoption.

Stable contract surface for T5 (#1311) under #1262 / #1046.

Today this module gates one knob — strict schema validation in the Cypher
binder (T2 #1302). Helpers and the resolver are written so additional
rollout gates can be added without re-shaping callers.

Precedence (most specific wins):
    1. Explicit caller parameter (e.g. ``FrontendBinder.bind(strict_name_resolution=True)``)
    2. Catalog-level metadata flag (e.g. ``GraphSchemaCatalog.metadata['strict']``)
    3. Process-wide env default (e.g. ``GRAPHISTRY_GFQL_STRICT_SCHEMA``)
    4. Loose default

The env tier is for canary / gradual rollout; default-off so existing
loose-mode callers see no behavior change.

Production callers:
    ``graphistry/compute/gfql/frontends/cypher/binder.py:_strict_schema_mode``
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
    """Apply strict-schema precedence: explicit > catalog > env > loose.

    A monotonic widening: any tier that asks for strict wins. Explicit
    ``False`` does not force loose mode (callers passing the default cannot
    override a catalog/env opt-in).
    """
    if explicit:
        return True
    if catalog_strict:
        return True
    return strict_schema_env_default()
