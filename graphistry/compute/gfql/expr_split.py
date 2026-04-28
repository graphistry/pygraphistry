"""Shared utilities for splitting expression text on top-level AND.

Used by both the Cypher parser (``generic_where_clause`` lifting label
predicates, and WHERE-pattern canonicalization) and the GFQL predicate
pushdown pass (splitting conjunctive WHERE bodies into independent
pushable predicates).

The splitter is quote-aware (single, double, backtick), bracket-depth-
aware (parens, square, curly), and handles simple backslash escape
sequences inside quoted strings.
"""

from __future__ import annotations

from typing import Optional, Tuple

__all__ = ("split_top_level_and",)


def split_top_level_and(expr: str) -> Tuple[str, ...]:
    """Split *expr* on whitespace-bounded top-level ``AND`` (case-insensitive).

    Quoted string contents and any ``( ... )``, ``[ ... ]``, ``{ ... }``,
    or ``` `...` `` fragments are treated opaquely; ``AND`` tokens inside
    them do not split.  Leading and trailing whitespace on each term is
    stripped.

    **AND-only by design.**  Do NOT add a sibling ``split_top_level_or``
    that pushes individual OR branches independently — predicate
    pushdown's correctness depends on OR conjuncts staying intact past
    a join.  Specifically, distributing ``a.x = 1 OR b.y = 2`` into two
    pre-join filters and unioning produces the same answer ONLY for
    1:1 join topologies; for cross-product / fan-out topologies it
    produces wrong rows.  The current pushdown leaves OR/XOR/NOT trees
    as opaque single conjuncts (referenced by all aliases they touch),
    which is correct under any join topology but conservative.  Lifting
    OR-awareness is a separate design problem (#1219 territory) that
    must be paired with topology-aware pushdown safety logic.

    :param expr: The expression text to split (typically a WHERE body).
    :returns: A tuple of non-empty terms.  ``()`` when *expr* is empty,
        whitespace-only, has a leading/trailing top-level ``AND``, or
        contains consecutive ANDs with an empty term between them —
        the function collapses empty-input and malformed-input into a
        single sentinel; callers should treat ``()`` as "do not split".
        A single-element tuple means *expr* has no top-level ``AND``.
    """
    terms: list[str] = []
    term_start = 0
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    string_quote: Optional[str] = None
    in_backtick = False
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if string_quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == string_quote:
                string_quote = None
            i += 1
            continue
        if in_backtick:
            if ch == "`":
                in_backtick = False
            i += 1
            continue
        if ch in {"'", '"'}:
            string_quote = ch
            i += 1
            continue
        if ch == "`":
            in_backtick = True
            i += 1
            continue
        if ch == "(":
            paren_depth += 1
            i += 1
            continue
        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue
        if ch == "[":
            bracket_depth += 1
            i += 1
            continue
        if ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
            i += 1
            continue
        if ch == "{":
            brace_depth += 1
            i += 1
            continue
        if ch == "}":
            brace_depth = max(0, brace_depth - 1)
            i += 1
            continue
        if (
            paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
            and expr[i:i + 3].upper() == "AND"
            and (i == 0 or expr[i - 1].isspace())
            and (i + 3 == n or expr[i + 3].isspace())
        ):
            term = expr[term_start:i].strip()
            if term == "":
                return ()
            terms.append(term)
            i += 3
            while i < n and expr[i].isspace():
                i += 1
            term_start = i
            continue
        i += 1
    tail = expr[term_start:].strip()
    if tail == "":
        # Empty input, or a trailing ``AND`` with no right-hand term — the
        # caller should treat either as "do not split".
        return ()
    terms.append(tail)
    return tuple(terms)
