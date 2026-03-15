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
CypherPageValue = Union[int, ParameterRef, ExpressionText]


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
    min_hops: Optional[int] = None
    max_hops: Optional[int] = None
    to_fixed_point: bool = False


PatternElement = Union[NodePattern, RelationshipPattern]


@dataclass(frozen=True)
class MatchClause:
    patterns: Tuple[Tuple[PatternElement, ...], ...]
    span: SourceSpan
    optional: bool = False
    pattern_aliases: Tuple[Optional[str], ...] = ()

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


@dataclass(frozen=True)
class CypherYieldItem:
    name: str
    alias: Optional[str]
    span: SourceSpan


@dataclass(frozen=True)
class CallClause:
    procedure: str
    args: Tuple[ExpressionText, ...]
    yield_items: Tuple[CypherYieldItem, ...]
    span: SourceSpan


WhereOp = Literal[
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "is_null",
    "is_not_null",
    "has_labels",
    "contains",
    "starts_with",
    "ends_with",
]


@dataclass(frozen=True)
class WherePredicate:
    left: Union[PropertyRef, LabelRef]
    op: WhereOp
    right: Optional[Union[PropertyRef, CypherLiteral]]
    span: SourceSpan


@dataclass(frozen=True)
class WherePatternPredicate:
    pattern: Tuple[PatternElement, ...]
    span: SourceSpan


WhereTerm = Union[WherePredicate, WherePatternPredicate]


@dataclass(frozen=True)
class WhereClause:
    predicates: Tuple[WhereTerm, ...]
    span: SourceSpan
    expr: Optional[ExpressionText] = None


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
    call: Optional[CallClause]
    unwinds: Tuple[UnwindClause, ...]
    with_stages: Tuple[ProjectionStage, ...]
    return_: ReturnClause
    order_by: Optional[OrderByClause]
    skip: Optional[SkipClause]
    limit: Optional[LimitClause]
    row_sequence: Tuple[Union[ProjectionStage, UnwindClause], ...]
    trailing_semicolon: bool
    span: SourceSpan
    reentry_matches: Tuple[MatchClause, ...] = ()
    reentry_where: Optional[WhereClause] = None

    @property
    def match(self) -> Optional[MatchClause]:
        if not self.matches:
            return None
        return self.matches[-1]


@dataclass(frozen=True)
class CypherUnionQuery:
    branches: Tuple[CypherQuery, ...]
    union_kind: Literal["distinct", "all"]
    trailing_semicolon: bool
    span: SourceSpan
