"""Explicit ReentryPlan contract for bounded MATCH-after-WITH compilation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from typing_extensions import Literal


@dataclass(frozen=True)
class CarriedAlias:
    """A whole-row alias surviving from the prefix WITH into the trailing MATCH."""

    output_name: str
    table: Literal["nodes", "edges"]
    is_reentry_alias: bool
    carried_properties: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReentryPlan:
    """Compile-time contract between a prefix WITH stage and trailing MATCH."""

    reentry_alias_name: str
    aliases: Tuple[CarriedAlias, ...] = ()
    scalar_columns: Tuple[str, ...] = ()
    scalar_only: bool = False
    free_form: bool = False

    @property
    def reentry_alias(self) -> Optional[CarriedAlias]:
        for alias in self.aliases:
            if alias.is_reentry_alias:
                return alias
        return None

    @property
    def non_source_aliases(self) -> Tuple[CarriedAlias, ...]:
        return tuple(alias for alias in self.aliases if not alias.is_reentry_alias)
