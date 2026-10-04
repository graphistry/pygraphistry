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
from typing import List, Optional, Pattern, Tuple, cast

from .types import IndexKind
from .wire import CreateIndex, DropIndex, ShowIndexes, IndexOp

_KIND = r"(?P<kind>edge_out_adj|edge_in_adj|node_id|node_prop)"

_CREATE_PATTERN = (
    r"^\s*CREATE\s+GFQL\s+INDEX\s+(?:(?P<name>[A-Za-z_]\w*)\s+)?FOR\s+" + _KIND
    + r"(?:\s+ON\s+(?P<col>[A-Za-z_]\w*))?\s*;?\s*$"
)
_DROP_FOR_PATTERN = (
    r"^\s*DROP\s+GFQL\s+INDEX\s+(?P<ifexists>IF\s+EXISTS\s+)?FOR\s+" + _KIND
    + r"(?:\s+ON\s+(?P<col>[A-Za-z_]\w*))?\s*;?\s*$"
)
_DROP_NAME_PATTERN = (
    r"^\s*DROP\s+GFQL\s+INDEX\s+(?P<ifexists>IF\s+EXISTS\s+)?(?P<name>[A-Za-z_][\w:]*)\s*;?\s*$"
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
        return CreateIndex(kind=cast(IndexKind, m.group("kind").lower()), column=m.group("col"),
                           name=m.group("name"))
    m = drop_for_re.match(query)
    if m:
        return DropIndex(kind=cast(IndexKind, m.group("kind").lower()), column=m.group("col"),
                         missing_ok=bool(m.group("ifexists")))
    m = drop_name_re.match(query)
    if m:
        return DropIndex(name=m.group("name"), missing_ok=bool(m.group("ifexists")))
    if looks_like_index_ddl(query):
        raise ValueError(
            f"Malformed GFQL INDEX DDL: {query!r}. Expected e.g. "
            "'CREATE GFQL INDEX FOR edge_out_adj', 'DROP GFQL INDEX FOR edge_in_adj', "
            "'SHOW GFQL INDEXES'."
        )
    return None


_DDL_STATEMENT_PATTERN = r"^\s*(?:CREATE|DROP|SHOW)\s+GFQL\s+INDEX[^;]*;"


@lru_cache(maxsize=1)
def _ddl_statement_re() -> Pattern[str]:
    return re.compile(_DDL_STATEMENT_PATTERN, re.IGNORECASE)


register_process_singleton(_ddl_statement_re, "a compiled regex over a module-level pattern constant; function of the code alone")


def split_leading_index_ddl(query: str) -> Tuple[List[IndexOp], str]:
    """``CREATE GFQL INDEX ...; CREATE GFQL INDEX ...; MATCH ...`` -> the DDL ops in order, and the rest.

    Only statements at the FRONT are DDL; the remainder is handed to the ordinary query path
    untouched, so a ``;`` inside it (a string literal, say) is never split. A malformed leading
    statement raises the same error the single-statement form does.
    """
    ops: List[IndexOp] = []
    rest = query
    while True:
        m = _ddl_statement_re().match(rest)
        if m is None:
            break
        statement = m.group(0)
        op = parse_index_ddl(statement)
        if op is None:  # cannot happen: the prefix matched; parse_index_ddl raises on malformed DDL
            break
        ops.append(op)
        rest = rest[m.end():]
    rest = rest.strip()
    if ops and rest and looks_like_index_ddl(rest):
        trailing = parse_index_ddl(rest)  # the last statement needs no ';'; malformed DDL raises here
        if trailing is not None:
            ops.append(trailing)
            rest = ""
    return ops, rest
