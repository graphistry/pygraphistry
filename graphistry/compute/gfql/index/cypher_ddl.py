"""Targeted recognizer for GFQL index DDL Cypher statements.

    CREATE GFQL INDEX [<name>] FOR <kind> [ON <column>]
    DROP   GFQL INDEX <name>
    DROP   GFQL INDEX FOR <kind> [ON <column>]
    SHOW   GFQL INDEXES

The mandatory ``GFQL`` token disambiguates from standard property ``CREATE INDEX``
(which the GFQL grammar does not implement), so this fixed-form recognizer is
unambiguous and additive — it runs before the Earley parser and returns a wire op
or None (not-a-DDL -> normal query path).
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .wire import CreateIndex, DropIndex, ShowIndexes

_KIND = r"(?P<kind>edge_out_adj|edge_in_adj|node_id)"

_CREATE_RE = re.compile(
    r"^\s*CREATE\s+GFQL\s+INDEX\s+(?:(?P<name>[A-Za-z_]\w*)\s+)?FOR\s+" + _KIND
    + r"(?:\s+ON\s+(?P<col>[A-Za-z_]\w*))?\s*;?\s*$",
    re.IGNORECASE,
)
_DROP_FOR_RE = re.compile(
    r"^\s*DROP\s+GFQL\s+INDEX\s+FOR\s+" + _KIND
    + r"(?:\s+ON\s+(?P<col>[A-Za-z_]\w*))?\s*;?\s*$",
    re.IGNORECASE,
)
_DROP_NAME_RE = re.compile(
    r"^\s*DROP\s+GFQL\s+INDEX\s+(?P<name>[A-Za-z_][\w:]*)\s*;?\s*$",
    re.IGNORECASE,
)
_SHOW_RE = re.compile(r"^\s*SHOW\s+GFQL\s+INDEXES\s*;?\s*$", re.IGNORECASE)

_DDL_PREFIX = re.compile(r"^\s*(CREATE|DROP|SHOW)\s+GFQL\s+INDEX", re.IGNORECASE)


def looks_like_index_ddl(query: str) -> bool:
    return bool(isinstance(query, str) and _DDL_PREFIX.match(query))


def parse_index_ddl(query: str) -> Optional[Any]:
    """Return a wire op (CreateIndex/DropIndex/ShowIndexes) or None."""
    if not isinstance(query, str):
        return None
    if _SHOW_RE.match(query):
        return ShowIndexes()
    m = _CREATE_RE.match(query)
    if m:
        return CreateIndex(kind=m.group("kind").lower(), column=m.group("col"),
                           name=m.group("name"))
    m = _DROP_FOR_RE.match(query)
    if m:
        return DropIndex(kind=m.group("kind").lower(), column=m.group("col"))
    m = _DROP_NAME_RE.match(query)
    if m:
        return DropIndex(name=m.group("name"))
    if looks_like_index_ddl(query):
        raise ValueError(
            f"Malformed GFQL INDEX DDL: {query!r}. Expected e.g. "
            "'CREATE GFQL INDEX FOR edge_out_adj', 'DROP GFQL INDEX FOR edge_in_adj', "
            "'SHOW GFQL INDEXES'."
        )
    return None
