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
trailing MATCH stages, UNWINDs, OPTIONAL on either MATCH, a fresh node
alias on the trailing pattern, a trailing pattern with no
``RelationshipPattern``, or non-bare projection items disqualify the
pattern. (Fresh relationship and path aliases on the trailing MATCH are
admitted — they are bound after the WITH and legitimately in scope.)
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, Optional, Set, Tuple

from graphistry.compute.gfql.cypher.ast import (
    CypherQuery,
    ExpressionText,
    MatchClause,
    NodePattern,
    PathPatternKind,
    PatternElement,
    ProjectionStage,
    RelationshipPattern,
    ReturnClause,
    ReturnItem,
)


_BARE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IDENT_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_AGGREGATE_CALL = re.compile(
    r"\b(count|sum|avg|min|max|collect|stdev|percentile\w*)\s*\(", re.IGNORECASE
)


def _bare_carry_aliases(clause: ReturnClause) -> Optional[Set[str]]:
    """Alias set when every projection item is an unaliased bare identifier, else None."""
    aliases: Set[str] = set()
    for item in clause.items:
        if item.alias is not None:
            return None
        text = item.expression.text.strip()
        if not _BARE_IDENT.fullmatch(text):
            return None
        aliases.add(text)
    return aliases


def _pure_carry_aliases(stage: ProjectionStage) -> Optional[Set[str]]:
    """Return the carried alias set if the WITH stage is a pure bare carry, else None."""
    if stage.where is not None:
        return None
    if _stage_reshapes_rows(stage):
        return None
    return _bare_carry_aliases(stage.clause)


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


def flatten_pure_carry_optional_reentry(query: CypherQuery) -> Optional[CypherQuery]:
    """Flatten ``MATCH ... WITH <pure carry> OPTIONAL MATCH ...`` (#1891).

    When the single WITH stage is a pure bare-alias carry of EVERY
    prefix-bound alias (a scope-preserving no-op) and every trailing match is
    OPTIONAL, the query is equivalent to the WITH-less
    ``MATCH ... OPTIONAL MATCH ...`` form, which the connected optional-match
    left-join lowering serves with correct null-extension semantics. The
    reentry row assembly, by contrast, loses carried seed properties for
    unmatched rows, so routing onto the join mechanism fixes the semantics
    rather than patching the reentry concat.

    Trailing OPTIONAL MATCH clauses keep their position; a per-clause reentry
    WHERE is attached to its clause (WHERE inside an optional clause filters
    matches but keeps rows null-extended, which the join lowering honors).
    Each trailing pattern must share at least one node alias with the aliases
    known before it, mirroring the connected lowering's join requirement, so
    disconnected shapes keep their current reentry behavior.
    """
    if not query.reentry_matches:
        return None
    if len(query.with_stages) != 1:
        return None
    if len(query.matches) != 1:
        return None
    if query.unwinds or query.reentry_unwinds:
        return None
    if query.call is not None or query.row_sequence:
        return None

    prefix_match = query.matches[0]
    if prefix_match.optional:
        return None
    if not all(m.optional for m in query.reentry_matches):
        return None

    carried = _pure_carry_aliases(query.with_stages[0])
    if carried is None or not carried:
        return None

    prefix_aliases: Set[str] = set()
    for pattern in prefix_match.patterns:
        prefix_aliases.update(_all_pattern_aliases(pattern))
    if prefix_match.pattern_aliases:
        prefix_aliases.update(
            alias for alias in prefix_match.pattern_aliases if alias is not None
        )
    # Scope preservation: the WITH must carry every prefix-bound alias, so
    # dropping it cannot re-admit an out-of-scope reference.
    if carried != prefix_aliases:
        return None

    reentry_wheres = query.reentry_wheres or tuple(None for _ in query.reentry_matches)
    if len(reentry_wheres) != len(query.reentry_matches):
        return None

    known_aliases: Set[str] = set(prefix_aliases)
    trailing_matches = []
    for trailing_match, reentry_where in zip(query.reentry_matches, reentry_wheres):
        trailing_node_aliases: Set[str] = set()
        for pattern in trailing_match.patterns:
            trailing_node_aliases.update(_node_aliases(pattern))
        if not (trailing_node_aliases & known_aliases):
            return None
        if reentry_where is not None:
            if trailing_match.where is not None:
                return None
            trailing_match = replace(trailing_match, where=reentry_where)
        trailing_matches.append(trailing_match)
        for pattern in trailing_match.patterns:
            known_aliases.update(_all_pattern_aliases(pattern))

    return replace(
        query,
        matches=(prefix_match, *trailing_matches),
        with_stages=(),
        reentry_matches=(),
        reentry_wheres=(),
        reentry_unwinds=(),
    )


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
    # Path aliases (``MATCH path = (a)-->(b)``) live on
    # ``MatchClause.pattern_aliases`` — also a prefix-bound alias that WITH
    # can drop, so include them too for the equality boundary.
    if prefix_match.pattern_aliases:
        prefix_aliases.update(
            alias for alias in prefix_match.pattern_aliases if alias is not None
        )
    # Require equality across node, relationship, AND path aliases. When WITH
    # drops a prefix-bound alias of any kind, post-WITH references must
    # surface as the existing reentry path's scope-rejection rather than
    # silently re-admitting through the merged single MATCH (e.g.
    # ``MATCH (a)-[r:R]->(b) WITH a, b MATCH (b)-[:S]->(a) RETURN r.weight``
    # would leak ``r``; ``MATCH path = (a)-->(b) WITH a, b MATCH ... RETURN
    # length(path)`` would leak ``path``).
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


def flatten_terminal_with_over_optional(
    query: CypherQuery,
) -> Optional[Tuple[CypherQuery, Optional[ExpressionText]]]:
    """Rewrite terminal ``WITH`` over OPTIONAL MATCH onto the left-join lowering (#1896).

    Returns the WITH-free query plus the stage ``WHERE`` for the caller to apply as a
    post-join binding-row filter, or None to leave the query on its typed-decline path.
    """
    if not _is_terminal_with_over_optional_match(query):
        return None
    stage = query.with_stages[0]
    if _stage_reshapes_rows(stage) or query.return_.distinct:
        return None
    match_aliases = _match_clause_aliases(query)
    carried = _bare_carry_aliases(stage.clause)
    if carried is not None and carried and carried <= match_aliases:
        return _query_without_pure_carry_stage(query, stage, carried, match_aliases)
    return _query_with_terminal_stage_folded_into_return(query, stage, match_aliases)


def _is_terminal_with_over_optional_match(query: CypherQuery) -> bool:
    """One trailing WITH stage over a non-optional lead MATCH plus at least one OPTIONAL MATCH."""
    if query.reentry_matches or query.unwinds or query.call is not None or query.row_sequence:
        return False
    if len(query.with_stages) != 1:
        return False
    if not query.matches or query.matches[0].optional:
        return False
    return any(m.optional for m in query.matches)


def _stage_reshapes_rows(stage: ProjectionStage) -> bool:
    return (
        stage.order_by is not None
        or stage.skip is not None
        or stage.limit is not None
        or stage.clause.distinct
    )


def _match_clause_aliases(query: CypherQuery) -> Set[str]:
    aliases: Set[str] = set()
    for m in query.matches:
        for pattern in m.patterns:
            aliases.update(_all_pattern_aliases(pattern))
    return aliases


def _referenced_aliases(text: str, candidates: Set[str]) -> Set[str]:
    return {tok for tok in _IDENT_TOKEN.findall(text) if tok in candidates}


def _query_without_pure_carry_stage(
    query: CypherQuery,
    stage: ProjectionStage,
    carried: Set[str],
    match_aliases: Set[str],
) -> Optional[Tuple[CypherQuery, Optional[ExpressionText]]]:
    """Drop a bare-alias carry stage, handing its WHERE back as a post-join row filter."""
    downstream_texts = [item.expression.text for item in query.return_.items]
    if query.order_by is not None:
        downstream_texts.extend(item.expression.text for item in query.order_by.items)
    if stage.where is not None:
        downstream_texts.append(stage.where.text)
    for text in downstream_texts:
        if not _referenced_aliases(text, match_aliases) <= carried:
            return None
    return replace(query, with_stages=()), stage.where


def _query_with_terminal_stage_folded_into_return(
    query: CypherQuery,
    stage: ProjectionStage,
    match_aliases: Set[str],
) -> Optional[Tuple[CypherQuery, Optional[ExpressionText]]]:
    """Fold a stage that RETURN passes through unchanged into the RETURN clause itself."""
    if stage.where is not None:
        return None
    stage_items: Dict[str, ReturnItem] = {}
    bare_carries: Set[str] = set()
    for item in stage.clause.items:
        name = item.alias or item.expression.text
        text = item.expression.text.strip()
        if name in stage_items:
            return None
        stage_items[name] = item
        if item.alias is None and _BARE_IDENT.fullmatch(text) and text in match_aliases:
            bare_carries.add(name)
    stage_aggregates = _stage_has_aggregates(stage)
    new_items = []
    for ret_item in query.return_.items:
        text = ret_item.expression.text.strip()
        if text in stage_items:
            if text in bare_carries and stage_aggregates:
                return None
            src = stage_items[text]
            new_items.append(replace(src, alias=ret_item.alias if ret_item.alias is not None else src.alias))
            continue
        parts = text.split(".")
        if len(parts) == 2 and parts[0] in bare_carries and _BARE_IDENT.fullmatch(parts[1]):
            new_items.append(ret_item)
            continue
        return None
    return replace(
        query,
        with_stages=(),
        return_=replace(stage.clause, kind="return", items=tuple(new_items)),
    ), None


def _stage_has_aggregates(stage: ProjectionStage) -> bool:
    return any(_AGGREGATE_CALL.search(item.expression.text) for item in stage.clause.items)
