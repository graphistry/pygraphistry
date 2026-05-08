"""Flatten safe carried-endpoint rebind shapes into a single MATCH (#1341).

When a query carries whole-row aliases through a single ``WITH`` and the
trailing ``MATCH`` re-binds only carried aliases as node variables (e.g. LDBC
SNB IC1 ``WITH p, friend MATCH path = shortestPath((p)-[:KNOWS*]-(friend))``),
the WITH stage is semantically a no-op: the same patterns can run as
comma-separated patterns in a single MATCH clause. This module recognizes
that narrow shape and returns an equivalent reentry-free query that the
existing single-MATCH lowering paths (including two-endpoint
``shortestPath``) can compile directly.

The transformation is intentionally narrow: any aggregation, alias rename,
DISTINCT, ORDER BY, SKIP, LIMIT, WHERE on the WITH stage, multiple WITH /
trailing MATCH stages, UNWINDs, OPTIONAL on the trailing MATCH, fresh
trailing aliases, or non-bare projection items disqualify the pattern.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Optional, Set, Tuple

from graphistry.compute.gfql.cypher.ast import (
    CypherQuery,
    MatchClause,
    NodePattern,
    PathPatternKind,
    PatternElement,
    ProjectionStage,
    RelationshipPattern,
)


_BARE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _pure_carry_aliases(stage: ProjectionStage) -> Optional[Set[str]]:
    """Return the carried alias set if the WITH stage is a pure bare carry, else None."""
    if stage.where is not None:
        return None
    if stage.order_by is not None or stage.skip is not None or stage.limit is not None:
        return None
    clause = stage.clause
    if clause.distinct:
        return None
    aliases: Set[str] = set()
    for item in clause.items:
        if item.alias is not None:
            return None
        text = item.expression.text.strip()
        if not _BARE_IDENT.fullmatch(text):
            return None
        aliases.add(text)
    return aliases


def _node_aliases(pattern: Tuple[PatternElement, ...]) -> Set[str]:
    out: Set[str] = set()
    for el in pattern:
        if isinstance(el, NodePattern) and el.variable is not None:
            out.add(el.variable)
    return out


def _all_pattern_aliases(pattern: Tuple[PatternElement, ...]) -> Set[str]:
    """Return every variable bound by the pattern: node and relationship aliases."""
    out: Set[str] = set()
    for el in pattern:
        if isinstance(el, (NodePattern, RelationshipPattern)) and el.variable is not None:
            out.add(el.variable)
    return out


def _normalized_aliases(
    aliases: Tuple[Optional[str], ...],
    patterns: Tuple[Tuple[PatternElement, ...], ...],
) -> Tuple[Optional[str], ...]:
    if aliases:
        return aliases
    return tuple(None for _ in patterns)


def _normalized_kinds(
    kinds: Tuple[PathPatternKind, ...],
    patterns: Tuple[Tuple[PatternElement, ...], ...],
) -> Tuple[PathPatternKind, ...]:
    # ``MatchClause.pattern_alias_kinds`` defaults to ``()`` per ``ast.py``;
    # the parser populates it for every pattern when present. When absent we
    # back-fill ``"pattern"`` since that is the implicit kind for unaliased
    # comma-separated patterns.
    if kinds:
        return kinds
    default: PathPatternKind = "pattern"
    return tuple(default for _ in patterns)


def flatten_carried_endpoint_rebind(query: CypherQuery) -> Optional[CypherQuery]:
    """Return a flattened equivalent if the query matches the narrow shape.

    Narrow shape:
    - exactly one prefix MATCH and one trailing MATCH
    - exactly one WITH stage that is a pure bare-alias carry (no DISTINCT,
      no aggregation, no rename, no WHERE/ORDER BY/SKIP/LIMIT)
    - no UNWIND (prefix or trailing), no CALL, no row sequence,
      no OPTIONAL on the trailing MATCH, no reentry WHEREs
    - every node alias bound by trailing patterns is among the carried set
    - every carried alias was bound by the prefix MATCH
    """
    if not query.reentry_matches or len(query.reentry_matches) != 1:
        return None
    if len(query.with_stages) != 1:
        return None
    if len(query.matches) != 1:
        return None
    if query.unwinds or query.reentry_unwinds:
        return None
    if query.call is not None or query.row_sequence:
        return None
    if query.reentry_wheres and any(w is not None for w in query.reentry_wheres):
        return None

    prefix_match = query.matches[0]
    trailing_match = query.reentry_matches[0]
    if prefix_match.optional or trailing_match.optional:
        return None

    carried = _pure_carry_aliases(query.with_stages[0])
    if carried is None or not carried:
        return None

    # Each trailing pattern must:
    # - bind only carried aliases (no fresh aliases)
    # - add structural constraints (at least one relationship pattern); a pure
    #   single-node re-reference like ``MATCH (a) RETURN a`` after ``WITH a``
    #   would create a redundant alias binding that downstream lowering
    #   rejects, and the existing reentry path handles such no-op trailing
    #   patterns natively.
    for pattern in trailing_match.patterns:
        if not _node_aliases(pattern).issubset(carried):
            return None
        if not any(isinstance(el, RelationshipPattern) for el in pattern):
            return None

    prefix_aliases: Set[str] = set()
    for pattern in prefix_match.patterns:
        prefix_aliases.update(_all_pattern_aliases(pattern))
    # Require equality across both node AND relationship aliases. When WITH
    # drops a prefix-bound alias of either kind, post-WITH references to it
    # must surface as the existing reentry path's scope-rejection rather than
    # silently re-admitting it through the merged single MATCH (e.g.
    # ``MATCH (a)-[r:R]->(b) WITH a, b MATCH (b)-[:S]->(a) RETURN r.weight``
    # would leak ``r`` back into RETURN scope).
    if carried != prefix_aliases:
        return None

    # Per ``parser.py`` (the top-level WHERE between MATCH and WITH is
    # mirrored onto ``match_clauses[-1].where``), checking ``prefix_match.where``
    # covers both ``query.where`` and an inline MATCH WHERE for the prefix.
    # The trailing-MATCH branch below is defensive: post-WITH WHEREs are
    # routed by the parser to ``reentry_wheres`` (already disqualified above),
    # so ``trailing_match.where`` is None for parser-produced queries; the
    # check guards AST-built inputs.
    if prefix_match.where is not None and trailing_match.where is not None:
        return None
    inline_where = prefix_match.where if prefix_match.where is not None else trailing_match.where

    new_patterns = prefix_match.patterns + trailing_match.patterns
    new_pattern_aliases = (
        _normalized_aliases(prefix_match.pattern_aliases, prefix_match.patterns)
        + _normalized_aliases(trailing_match.pattern_aliases, trailing_match.patterns)
    )
    new_pattern_alias_kinds = (
        _normalized_kinds(prefix_match.pattern_alias_kinds, prefix_match.patterns)
        + _normalized_kinds(trailing_match.pattern_alias_kinds, trailing_match.patterns)
    )
    merged = MatchClause(
        patterns=new_patterns,
        span=prefix_match.span,
        optional=prefix_match.optional,
        pattern_aliases=new_pattern_aliases,
        where=inline_where,
        pattern_alias_kinds=new_pattern_alias_kinds,
    )
    return replace(
        query,
        matches=(merged,),
        with_stages=(),
        reentry_matches=(),
        reentry_wheres=(),
        reentry_unwinds=(),
    )
