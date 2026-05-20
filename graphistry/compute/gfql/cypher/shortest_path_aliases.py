from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, cast

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.ast import (
    CypherQuery,
    MatchClause,
    NodePattern,
    PathPatternKind,
    PatternElement,
    RelationshipPattern,
)


@dataclass(frozen=True)
class _ShortestPathAliasSpec:
    alias: str
    hop_column: str
    pattern: Tuple[PatternElement, ...]
    start_alias: Optional[str]
    end_alias: Optional[str]


def _unsupported_shortest_path(message: str, *, field: str, value: object, clause: MatchClause) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        message,
        field=field,
        value=value,
        suggestion="Use a subset currently supported by the local Cypher compiler.",
        line=clause.span.line,
        column=clause.span.column,
        language="cypher",
    )


def _is_variable_length_relationship_pattern(relationship: RelationshipPattern) -> bool:
    return (
        relationship.min_hops is not None
        or relationship.max_hops is not None
        or relationship.to_fixed_point
    )


def _match_pattern_alias_kinds(clause: MatchClause) -> Tuple[PathPatternKind, ...]:
    if clause.pattern_alias_kinds:
        return cast(Tuple[PathPatternKind, ...], clause.pattern_alias_kinds)
    return tuple("pattern" for _ in clause.patterns)


def _shortest_path_alias_specs(query: CypherQuery) -> Dict[str, _ShortestPathAliasSpec]:
    out: Dict[str, _ShortestPathAliasSpec] = {}
    for clause in query.matches + query.reentry_matches:
        pattern_aliases = clause.pattern_aliases or tuple(None for _ in clause.patterns)
        pattern_kinds = _match_pattern_alias_kinds(clause)
        for alias, pattern, kind in zip(pattern_aliases, clause.patterns, pattern_kinds):
            if alias is None or kind != "shortestPath":
                continue
            relationships = [element for element in pattern if isinstance(element, RelationshipPattern)]
            if len(relationships) != 1 or len(pattern) != 3:
                raise _unsupported_shortest_path(
                    "Cypher shortestPath() currently supports only single-relationship path patterns in the local compiler",
                    field="match",
                    value=alias,
                    clause=clause,
                )
            relationship = relationships[0]
            if not _is_variable_length_relationship_pattern(relationship):
                raise _unsupported_shortest_path(
                    "Cypher shortestPath() requires a variable-length relationship pattern in the local compiler",
                    field="match",
                    value=alias,
                    clause=clause,
                )
            start_alias = pattern[0].variable if isinstance(pattern[0], NodePattern) else None
            end_alias = pattern[-1].variable if isinstance(pattern[-1], NodePattern) else None
            out[alias] = _ShortestPathAliasSpec(
                alias=alias,
                hop_column=f"__cypher_shortest_path_hops__{alias}",
                pattern=pattern,
                start_alias=start_alias,
                end_alias=end_alias,
            )
    return out
