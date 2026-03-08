from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union


@dataclass(frozen=True)
class SourceSpan:
    line: int
    column: int
    end_line: int
    end_column: int
    start_pos: int
    end_pos: int


@dataclass(frozen=True)
class ExpressionText:
    text: str
    span: SourceSpan


@dataclass(frozen=True)
class ParameterRef:
    name: str
    span: SourceSpan


CypherLiteral = Union[None, bool, int, float, str, ParameterRef]


@dataclass(frozen=True)
class PropertyEntry:
    key: str
    value: CypherLiteral
    span: SourceSpan


@dataclass(frozen=True)
class NodePattern:
    variable: Optional[str]
    labels: Tuple[str, ...]
    properties: Tuple[PropertyEntry, ...]
    span: SourceSpan


@dataclass(frozen=True)
class RelationshipPattern:
    direction: Literal["forward", "reverse", "undirected"]
    variable: Optional[str]
    types: Tuple[str, ...]
    properties: Tuple[PropertyEntry, ...]
    span: SourceSpan


PatternElement = Union[NodePattern, RelationshipPattern]


@dataclass(frozen=True)
class MatchClause:
    pattern: Tuple[PatternElement, ...]
    span: SourceSpan


@dataclass(frozen=True)
class ReturnItem:
    expression: ExpressionText
    alias: Optional[str]
    span: SourceSpan


@dataclass(frozen=True)
class ReturnClause:
    items: Tuple[ReturnItem, ...]
    distinct: bool
    span: SourceSpan


@dataclass(frozen=True)
class CypherQuery:
    match: MatchClause
    return_: ReturnClause
    trailing_semicolon: bool
    span: SourceSpan
