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
_DDL_WORDS = r"(CREATE|DROP|SHOW)\s+GFQL\s+INDEX"
_DDL_PREFIX_PATTERN = r"^\s*" + _DDL_WORDS


@lru_cache(maxsize=1)
def _ddl_anywhere_re() -> Pattern[str]:
    return re.compile(r"\b" + _DDL_WORDS, re.IGNORECASE)


register_process_singleton(_ddl_anywhere_re, "a compiled regex over a module-level pattern constant; function of the code alone")


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


def split_top_level_statements(query: str) -> List[str]:
    """Split on ``;`` outside quotes and brackets; empty statements are dropped."""
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    quote: Optional[str] = None
    i = 0
    while i < len(query):
        ch = query[i]
        if quote is not None:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(query):
                buf.append(query[i + 1])
                i += 1
            elif ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == ";" and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    out.append("".join(buf))
    return [stmt.strip() for stmt in out if stmt.strip()]


def parse_index_ddl_prefix(query: str) -> Optional[Tuple[List[IndexOp], Optional[str]]]:
    """``CREATE/DROP GFQL INDEX ...; <query>``: the leading DDL ops and the remaining query
    (None when the list is DDL only). None when ``query`` is not a statement list that
    starts with index DDL; a lone DDL statement keeps the whole-string path."""
    if not isinstance(query, str) or not _ddl_anywhere_re().search(query):
        return None
    statements = split_top_level_statements(query)
    if len(statements) <= 1:
        return None
    ops: List[IndexOp] = []
    i = 0
    while i < len(statements) and looks_like_index_ddl(statements[i]):
        op = parse_index_ddl(statements[i])
        if op is None or isinstance(op, ShowIndexes):
            raise ValueError(
                f"GFQL INDEX statement {i + 1} cannot be part of a statement list: {statements[i]!r}. "
                "SHOW GFQL INDEXES returns a table; run it on its own."
            )
        ops.append(op)
        i += 1
    rest = statements[i:]
    trailing = [stmt for stmt in rest if looks_like_index_ddl(stmt)]
    if trailing:
        raise ValueError(
            f"GFQL INDEX DDL must lead the statement list, found after the query: {trailing[0]!r}. "
            "Write 'CREATE GFQL INDEX FOR ...; <query>'."
        )
    if not ops:
        return None
    return ops, ("; ".join(rest) if rest else None)
