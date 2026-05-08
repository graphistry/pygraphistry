"""Direct unit tests for the carried-endpoint rebind flattener (#1341).

Anchors each disqualification branch in ``flatten_carried_endpoint_rebind``
so future edits to the narrow shape boundary are caught by tests rather
than indirectly via reentry-path regressions.
"""
from __future__ import annotations

import dataclasses
import typing
from typing import Any, Dict, Literal

import pytest

from graphistry.compute.gfql.cypher.ast import (
    CypherQuery,
    MatchClause,
    PatternElement,
    SourceSpan,
    WhereClause,
)
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.cypher.reentry.flatten import (
    flatten_carried_endpoint_rebind,
    _all_pattern_aliases,
)


def _parse(query: str) -> CypherQuery:
    parsed = parse_cypher(query)
    assert isinstance(parsed, CypherQuery)
    return parsed


def test_flatten_admits_ic1_shortest_path_shape() -> None:
    q = _parse(
        "MATCH (p:Person {id: 'p1'}), (friend:Person) "
        "WHERE NOT p = friend "
        "WITH p, friend "
        "MATCH path = shortestPath((p)-[:KNOWS*1..3]-(friend)) "
        "RETURN friend.id AS friendId"
    )
    flattened = flatten_carried_endpoint_rebind(q)
    assert flattened is not None
    assert flattened.with_stages == ()
    assert flattened.reentry_matches == ()
    # 3 patterns: (p), (friend), shortestPath
    assert len(flattened.matches) == 1
    assert len(flattened.matches[0].patterns) == 3


def test_flatten_admits_simple_rebind_with_edge() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id AS bid"
    )
    flattened = flatten_carried_endpoint_rebind(q)
    assert flattened is not None
    assert flattened.with_stages == ()


def test_flatten_preserves_prefix_match_where_on_merged_match() -> None:
    """Prefix-MATCH inline WHERE survives flatten on the merged MatchClause
    (parser also mirrors top-level WHERE between MATCH and WITH onto
    ``match_clauses[-1].where``, so this covers both shapes)."""
    q = _parse(
        "MATCH (p:Person {id: 'p1'}), (friend:Person) "
        "WHERE NOT p = friend "
        "WITH p, friend "
        "MATCH path = shortestPath((p)-[:KNOWS*1..3]-(friend)) "
        "RETURN friend.id AS friendId"
    )
    flattened = flatten_carried_endpoint_rebind(q)
    assert flattened is not None
    # prefix WHERE rides on the merged single MatchClause; original prefix's
    # WhereClause object reference must be preserved (not re-synthesized).
    assert flattened.matches[0].where is q.matches[0].where


def test_flatten_disqualifies_no_relationship_in_trailing_pattern() -> None:
    """``WITH a MATCH (a) RETURN a`` — no edge, would create redundant alias binding."""
    q = _parse("MATCH (a:A) WITH a MATCH (a) RETURN a")
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_distinct_with() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH DISTINCT a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_alias_rename_in_with() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b AS bb "
        "MATCH (bb)-[:S]->(a) "
        "RETURN bb.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_with_order_by() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b ORDER BY b.id "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_with_limit() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b LIMIT 5 "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_fresh_trailing_alias() -> None:
    """Trailing pattern introduces a fresh node alias (``c``); not a rebind."""
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN c.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_optional_trailing_match() -> None:
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "OPTIONAL MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_optional_prefix_match() -> None:
    q = _parse(
        "OPTIONAL MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_when_query_has_no_reentry() -> None:
    q = _parse("MATCH (a:A {id: 'a1'})-[:R]->(b:B) RETURN b.id")
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_multiple_with_stages() -> None:
    """Two-stage WITH chain → flatten only handles single-WITH narrow shape."""
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH a, b, c "
        "MATCH (c)-[:T]->(a) "
        "RETURN a.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_reentry_where_present() -> None:
    """A WHERE between trailing MATCH and RETURN is a reentry_where → disqualify."""
    q = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "WHERE b.score > 5 "
        "RETURN b.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_drops_relationship_variable() -> None:
    """Prefix binds a named relationship var (``r``); WITH drops it. The merged
    single MATCH would re-introduce ``r`` into RETURN scope, so flatten must
    disqualify (#1341 wave-4 review)."""
    q = _parse(
        "MATCH (a:A)-[r:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN r.weight"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_disqualifies_drops_prefix_path_alias() -> None:
    """Prefix binds a path alias (``path``); WITH drops it. Defense-in-depth:
    even though current row-pipeline rejects ``length(path)`` projections
    downstream, including path aliases in the equality check prevents
    future row-pipeline path-function support from silently re-admitting
    a dropped path alias into post-WITH scope (#1341 wave-5 review)."""
    q = _parse(
        "MATCH path = (a:A)-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN length(path) AS n"
    )
    assert flatten_carried_endpoint_rebind(q) is None


def test_flatten_admits_when_relationship_variable_is_carried() -> None:
    """A carried relationship var keeps the carry-set equal to prefix aliases."""
    q = _parse(
        "MATCH (a:A)-[r:R]->(b:B) "
        "WITH a, b, r "
        "MATCH (b)-[:S]->(a) "
        "RETURN r.weight"
    )
    flattened = flatten_carried_endpoint_rebind(q)
    assert flattened is not None
    assert flattened.with_stages == ()


def test_flatten_disqualifies_partial_carry_drops_prefix_alias() -> None:
    """WITH drops a prefix-bound alias; the existing reentry path emits a clean
    scope error if RETURN references the dropped alias. Flatten must not
    silently admit the merged form, since merging would re-introduce the
    dropped alias into RETURN scope (#1341 wave-2 review)."""
    q = _parse(
        "MATCH (a:A)-[:R]->(c:C) "
        "WITH a "
        "MATCH (a)-[:S]->(a) "
        "RETURN c.id"
    )
    assert flatten_carried_endpoint_rebind(q) is None


# --- Round-001 amplification: lock-in / matrix / defensive tests (#1341 followup)


def test_flatten_alias_kind_enumeration_locks_in_with_ast() -> None:
    """Lock-in: the prefix-aliases computation must observe every variable-bearing
    AST surface. If a future AST change adds a new ``PatternElement`` Union
    member or a new variable-bearing tuple-of-aliases on ``MatchClause``, this
    test forces ``_all_pattern_aliases`` (or the prefix-alias union step in
    ``flatten_carried_endpoint_rebind``) to be updated. Anchors the wave-4 /
    wave-5 bug class permanently against AST evolution."""
    # 1) Every PatternElement subtype with a ``variable`` attribute must be
    #    reflected by ``_all_pattern_aliases``. Build a minimal instance for
    #    each subtype by filling required fields with type-appropriate empty
    #    values (empty tuple, "forward" direction string). Future
    #    PatternElement members with new required fields will fail this loop's
    #    constructor call until the fillers are extended — which forces a
    #    deliberate review of the helper.
    pattern_subtypes = typing.get_args(PatternElement)
    assert pattern_subtypes, "PatternElement Union must have at least one member"
    span = SourceSpan(line=1, column=1, end_line=1, end_column=1, start_pos=0, end_pos=0)
    asserted_subtype = False
    for subtype in pattern_subtypes:
        type_hints = typing.get_type_hints(subtype)
        if "variable" not in type_hints:
            continue
        kwargs: Dict[str, Any] = {"variable": "probe", "span": span}
        for fld in dataclasses.fields(subtype):
            if fld.name in kwargs:
                continue
            if fld.default is not dataclasses.MISSING:
                continue
            if fld.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                continue
            ftype = type_hints.get(fld.name)
            origin = typing.get_origin(ftype)
            if origin is tuple:
                # Tuple[...]: empty tuple satisfies every parameterization.
                kwargs[fld.name] = ()
                continue
            if origin is Literal:
                # Pick the first allowed literal value.
                literal_args = typing.get_args(ftype)
                if literal_args:
                    kwargs[fld.name] = literal_args[0]
                    continue
            raise AssertionError(
                f"Lock-in test cannot synthesize required field {fld.name!r} of "
                f"type {ftype!r} on {subtype.__name__}; extend the filler logic "
                f"and verify ``_all_pattern_aliases`` handles the new shape."
            )
        instance = subtype(**kwargs)
        observed = _all_pattern_aliases((instance,))
        assert "probe" in observed, (
            f"_all_pattern_aliases must collect variable from {subtype.__name__} "
            f"(got {observed!r})"
        )
        asserted_subtype = True
    assert asserted_subtype, (
        "PatternElement enumeration produced no variable-bearing subtype probe; "
        "the lock-in test must assert against at least one concrete subtype."
    )

    # 2) MatchClause carries variable-bearing tuple fields (today only
    #    ``pattern_aliases``). The flatten's prefix-aliases union must observe
    #    them. Encode as a query-level admit/disqualify probe: a query with a
    #    named path alias on the prefix that is dropped at WITH must be
    #    disqualified.
    matchclause_field_names = {f.name for f in dataclasses.fields(MatchClause)}
    assert "pattern_aliases" in matchclause_field_names, (
        "MatchClause.pattern_aliases removed/renamed — flatten prefix-alias "
        "union step must be updated to track the new field."
    )
    q = _parse(
        "MATCH path = (a:A)-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN length(path) AS n"
    )
    assert flatten_carried_endpoint_rebind(q) is None, (
        "Path alias dropped at WITH must disqualify flatten — prefix-alias "
        "union step must observe MatchClause.pattern_aliases."
    )


@pytest.mark.parametrize(
    "kind,prefix,carry,trailing,return_clause",
    [
        # node alias dropped
        (
            "node",
            "MATCH (a:A)-[:R]->(b:B), (c:C)",
            "WITH a, b",
            "MATCH (b)-[:S]->(a)",
            "RETURN c.id",
        ),
        # relationship variable dropped
        (
            "rel",
            "MATCH (a:A)-[r:R]->(b:B)",
            "WITH a, b",
            "MATCH (b)-[:S]->(a)",
            "RETURN r.weight",
        ),
        # path alias dropped
        (
            "path",
            "MATCH path = (a:A)-[:R]->(b:B)",
            "WITH a, b",
            "MATCH (b)-[:S]->(a)",
            "RETURN length(path) AS n",
        ),
    ],
)
def test_flatten_disqualifies_partial_carry_for_each_alias_kind(
    kind: str, prefix: str, carry: str, trailing: str, return_clause: str
) -> None:
    """Parametrized partial-carry sweep: dropping a prefix-bound alias of any
    kind (node / relationship / path) must disqualify flatten."""
    q = _parse(f"{prefix} {carry} {trailing} {return_clause}")
    assert flatten_carried_endpoint_rebind(q) is None, (
        f"Dropping a prefix-bound {kind} alias must disqualify flatten"
    )


@pytest.mark.parametrize(
    "prefix_optional,trailing_optional,expected_admit",
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (True, True, False),
    ],
)
def test_flatten_disqualifies_optional_match_combinations(
    prefix_optional: bool, trailing_optional: bool, expected_admit: bool
) -> None:
    """Parametrized OPTIONAL-MATCH matrix. Flatten must admit only when both
    prefix and trailing MATCH are non-OPTIONAL; any OPTIONAL changes the
    semantics under merge (NULL-row propagation)."""
    p = "OPTIONAL " if prefix_optional else ""
    t = "OPTIONAL " if trailing_optional else ""
    q = _parse(
        f"{p}MATCH (a:A {{id: 'a1'}})-[:R]->(b:B) "
        f"WITH a, b "
        f"{t}MATCH (b)-[:S]->(a) "
        f"RETURN b.id"
    )
    flattened = flatten_carried_endpoint_rebind(q)
    if expected_admit:
        assert flattened is not None, (
            f"Expected admit for prefix.optional={prefix_optional}, "
            f"trailing.optional={trailing_optional}"
        )
    else:
        assert flattened is None, (
            f"Expected disqualify for prefix.optional={prefix_optional}, "
            f"trailing.optional={trailing_optional}"
        )


def test_flatten_disqualifies_both_inline_wheres_via_built_ast() -> None:
    """Defensive: parser routes post-WITH WHEREs to ``reentry_wheres``, so
    setting ``trailing_match.where`` requires hand-building the AST. This
    test reaches the otherwise-unreachable ``return None`` branch when both
    prefix and trailing MATCHes carry inline WHEREs."""
    base = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    # Synthesize a WhereClause-bearing trailing match from the parsed query.
    # Reuse the prefix's WhereClause shape (any WhereClause works for the
    # branch reachability test).
    parsed_with_where = _parse(
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WHERE a.score > 5 "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id"
    )
    prefix_where: WhereClause = parsed_with_where.matches[0].where  # type: ignore[assignment]
    assert prefix_where is not None
    trailing = base.reentry_matches[0]
    forged_trailing = dataclasses.replace(trailing, where=prefix_where)
    forged_query = dataclasses.replace(
        base,
        matches=(dataclasses.replace(base.matches[0], where=prefix_where),),
        reentry_matches=(forged_trailing,),
    )
    # Both prefix.where and trailing_match.where set → defensive branch fires.
    assert flatten_carried_endpoint_rebind(forged_query) is None
