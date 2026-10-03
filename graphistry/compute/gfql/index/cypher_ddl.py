"""Targeted recognizer for GFQL index DDL Cypher statements.

    CREATE GFQL INDEX [<name>] FOR <kind> [ON <column>]
    DROP   GFQL INDEX [IF EXISTS] <name>
    DROP   GFQL INDEX [IF EXISTS] FOR <kind> [ON <column>]
    SHOW   GFQL INDEXES

The mandatory ``GFQL`` token disambiguates from standard property ``CREATE INDEX``
(which the GFQL grammar does not implement), so this fixed-form recognizer is
unambiguous and additive — it runs before the Earley parser and returns a wire op
or None (not-a-DDL -> normal query path).
"""
from __future__ import annotations

from graphistry.compute.gfql.cache_registry import register_process_singleton

from functools import lru_cache
import re
from typing import Optional, Pattern, Tuple, cast

from .types import IndexKind
from .wire import CreateIndex, DropIndex, ShowIndexes, IndexOp

_KIND = r"(?P<kind>edge_out_adj|edge_in_adj|node_id|node_prop)"

# Optional parts follow the Cypher spellings: `[name] [IF NOT EXISTS]`, `ON (col)`, `DROP ... name [IF EXISTS]`.
_ON_COL = r"(?:\s+ON\s+(?:\(\s*(?P<col>[A-Za-z_]\w*)\s*\)|(?P<col2>[A-Za-z_]\w*)))?"
_CREATE_PATTERN = (
    r"^\s*CREATE\s+GFQL\s+INDEX\s+(?:(?P<name>(?!IF\b|FOR\b)[A-Za-z_]\w*)\s+)?(?:IF\s+NOT\s+EXISTS\s+)?FOR\s+"
    + _KIND + _ON_COL + r"\s*;?\s*$"
)
_DROP_FOR_PATTERN = (
    r"^\s*DROP\s+GFQL\s+INDEX\s+(?P<ifexists>IF\s+EXISTS\s+)?FOR\s+" + _KIND + _ON_COL + r"\s*;?\s*$"
)
_DROP_NAME_PATTERN = (
    r"^\s*DROP\s+GFQL\s+INDEX\s+(?:(?P<ifexists>IF\s+EXISTS\s+)(?P<name>[A-Za-z_][\w:]*)"
    r"|(?P<name2>(?!IF\b)[A-Za-z_][\w:]*)(?P<ifexists2>\s+IF\s+EXISTS)?)\s*;?\s*$"
)
_SHOW_PATTERN = r"^\s*SHOW\s+GFQL\s+INDEXES\s*;?\s*$"
_DDL_PREFIX_PATTERN = r"^\s*(CREATE|DROP|SHOW)\s+GFQL\s+INDEX"


@lru_cache(maxsize=1)
def _ddl_prefix_re() -> Pattern[str]:
    return re.compile(_DDL_PREFIX_PATTERN, re.IGNORECASE)


register_process_singleton(_ddl_prefix_re, "a compiled regex over a module-level pattern constant; function of the code alone")


@lru_cache(maxsize=1)
def _ddl_res() -> Tuple[Pattern[str], Pattern[str], Pattern[str], Pattern[str]]:
    return (
        re.compile(_SHOW_PATTERN, re.IGNORECASE),
        re.compile(_CREATE_PATTERN, re.IGNORECASE),
        re.compile(_DROP_FOR_PATTERN, re.IGNORECASE),
        re.compile(_DROP_NAME_PATTERN, re.IGNORECASE),
    )


register_process_singleton(_ddl_res, "compiled regexes over module-level pattern constants; function of the code alone")


def looks_like_index_ddl(query: str) -> bool:
    return bool(isinstance(query, str) and _ddl_prefix_re().match(query))


def parse_index_ddl(query: str) -> Optional[IndexOp]:
    """Return a typed wire op (CreateIndex/DropIndex/ShowIndexes) or None."""
    if not isinstance(query, str):
        return None
    show_re, create_re, drop_for_re, drop_name_re = _ddl_res()
    if show_re.match(query):
        return ShowIndexes()
    m = create_re.match(query)
    if m:
        return CreateIndex(kind=cast(IndexKind, m.group("kind").lower()), column=m.group("col") or m.group("col2"),
                           name=m.group("name"))
    m = drop_for_re.match(query)
    if m:
        return DropIndex(kind=cast(IndexKind, m.group("kind").lower()), column=m.group("col") or m.group("col2"),
                         missing_ok=bool(m.group("ifexists")))
    m = drop_name_re.match(query)
    if m:
        return DropIndex(name=m.group("name") or m.group("name2"),
                         missing_ok=bool(m.group("ifexists") or m.group("ifexists2")))
    if looks_like_index_ddl(query):
        raise ValueError(
            f"Malformed GFQL INDEX DDL: {query!r}. Expected e.g. "
            "'CREATE GFQL INDEX FOR edge_out_adj', 'DROP GFQL INDEX FOR edge_in_adj', "
            "'SHOW GFQL INDEXES'."
        )
    return None
