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
CypherPageValue = Union[int, ParameterRef]


@dataclass(frozen=True)
class PropertyRef:
    alias: str
    property: str
    span: SourceSpan


@dataclass(frozen=True)
class LabelRef:
    alias: str
    labels: Tuple[str, ...]
    span: SourceSpan


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
    patterns: Tuple[Tuple[PatternElement, ...], ...]
    span: SourceSpan

    @property
    def pattern(self) -> Tuple[PatternElement, ...]:
        if len(self.patterns) != 1:
            raise ValueError("MATCH clause contains multiple patterns; use .patterns instead of .pattern")
        return self.patterns[0]


@dataclass(frozen=True)
class UnwindClause:
    expression: ExpressionText
    alias: str
    span: SourceSpan


WhereOp = Literal["==", "!=", "<", "<=", ">", ">=", "is_null", "is_not_null", "has_labels"]


@dataclass(frozen=True)
class WherePredicate:
    left: Union[PropertyRef, LabelRef]
    op: WhereOp
    right: Optional[Union[PropertyRef, CypherLiteral]]
    span: SourceSpan


@dataclass(frozen=True)
class WhereClause:
    predicates: Tuple[WherePredicate, ...]
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
    kind: Literal["return", "with"]
    span: SourceSpan


@dataclass(frozen=True)
class ProjectionStage:
    clause: ReturnClause
    where: Optional[ExpressionText]
    order_by: Optional["OrderByClause"]
    skip: Optional["SkipClause"]
    limit: Optional["LimitClause"]
    span: SourceSpan


@dataclass(frozen=True)
class OrderItem:
    expression: ExpressionText
    direction: Literal["asc", "desc"]
    span: SourceSpan


@dataclass(frozen=True)
class OrderByClause:
    items: Tuple[OrderItem, ...]
    span: SourceSpan


@dataclass(frozen=True)
class SkipClause:
    value: CypherPageValue
    span: SourceSpan


@dataclass(frozen=True)
class LimitClause:
    value: CypherPageValue
    span: SourceSpan


@dataclass(frozen=True)
class CypherQuery:
    matches: Tuple[MatchClause, ...]
    where: Optional[WhereClause]
    unwinds: Tuple[UnwindClause, ...]
    with_stages: Tuple[ProjectionStage, ...]
    return_: ReturnClause
    order_by: Optional[OrderByClause]
    skip: Optional[SkipClause]
    limit: Optional[LimitClause]
    trailing_semicolon: bool
    span: SourceSpan

    @property
    def match(self) -> Optional[MatchClause]:
        if not self.matches:
            return None
        return self.matches[-1]
