"""Explicit `ReentryPlan` contract for bounded MATCH-after-WITH compilation.

Replaces the implicit handshake previously spread across:
- tuple returns from `_bounded_reentry_carry_columns` (lowering.py)
- `scalar_reentry_alias` / `scalar_reentry_columns` fields on
  `CompiledCypherExecutionExtras` (lowering.py)
- runtime contract re-extraction in `_compiled_query_reentry_contract`
  (gfql_unified.py)

See `plans/989-reentryplan-multi-alias-carry/design/reentry-plan.md` for the
full design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from typing_extensions import Literal


@dataclass(frozen=True)
class CarriedAlias:
    """A whole-row alias surviving from the prefix WITH into the trailing MATCH.

    Exactly one carried alias per `ReentryPlan` is the trailing-MATCH source
    (`is_reentry_alias=True`). Other carried aliases are decomposed at runtime
    into hidden scalar columns attached to the reentry-alias's row table.
    """

    output_name: str
    table: Literal["nodes", "edges"]
    is_reentry_alias: bool
    id_column: Optional[str] = None
    carried_properties: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReentryPlan:
    """Compile-time contract between a prefix `WITH` stage and the trailing `MATCH`.

    Two shapes:

    * Whole-row prefix: ``aliases`` is non-empty with exactly one
      ``is_reentry_alias=True``. ``scalar_only`` is False. Top-level scalar
      carries (e.g. ``WITH a, b.id AS bid``) live in ``scalar_columns``.

    * Scalar-only prefix: ``aliases`` is empty. ``scalar_only`` is True.
      ``reentry_alias_name`` is the trailing MATCH's first node alias (it is
      not projected by the prefix; the runtime injects hidden scalar columns
      onto its node table directly). ``scalar_columns`` lists the carried
      scalar names.
    """

    reentry_alias_name: str
    aliases: Tuple[CarriedAlias, ...] = ()
    scalar_columns: Tuple[str, ...] = ()
    scalar_only: bool = False

    @property
    def reentry_alias(self) -> Optional[CarriedAlias]:
        for alias in self.aliases:
            if alias.is_reentry_alias:
                return alias
        return None

    @property
    def non_source_aliases(self) -> Tuple[CarriedAlias, ...]:
        return tuple(alias for alias in self.aliases if not alias.is_reentry_alias)
