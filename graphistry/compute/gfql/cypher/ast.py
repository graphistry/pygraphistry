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
CypherPropertyValue = Union[CypherLiteral, ExpressionText]


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
    value: CypherPropertyValue
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
PathPatternKind = Literal["pattern", "shortestPath", "allShortestPaths"]


@dataclass(frozen=True)
class MatchClause:
    patterns: Tuple[Tuple[PatternElement, ...], ...]
    span: SourceSpan
    optional: bool = False
    pattern_aliases: Tuple[Optional[str], ...] = ()
    where: Optional["WhereClause"] = None
    pattern_alias_kinds: Tuple[PathPatternKind, ...] = ()

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
    # #1031 slice 2: True when lifted from a `WHERE NOT (...)` shape, signaling
    # anti-semi-join lowering instead of intersect-MATCH.  Default False keeps
    # all existing single-positive / multi-positive callers unchanged.
    negated: bool = False


WhereTerm = Union[WherePredicate, WherePatternPredicate]


BooleanOp = Literal["and", "or", "xor", "not", "atom", "pattern"]


@dataclass(frozen=True)
class BooleanExpr:
    """Structural representation of a parsed boolean expression tree.

    Mirrors Lark's ``and_op`` / ``or_op`` / ``xor_op`` / ``not_op`` grammar
    rules so downstream consumers (binder serialization, predicate
    pushdown) can walk structure instead of re-parsing expression text.

    Leaf nodes:
    - ``op == "atom"`` — atomic predicate; carries ``atom_text`` (exact
      source slice) and ``atom_span``.
    - ``op == "pattern"`` — WHERE pattern predicate (e.g. ``(n)-[:R]->()``);
      carries ``pattern`` (parsed pattern elements) and ``atom_text`` /
      ``atom_span`` for source slice.  Used as a leaf inside boolean
      expressions starting in #1031 slice 1.

    Branch nodes have ``op`` in ``{"and", "or", "xor"}`` with both ``left``
    and ``right`` set, or ``op == "not"`` with only ``left`` set.  ``span``
    covers the full subexpression in every case.
    """

    op: BooleanOp
    span: SourceSpan
    left: Optional["BooleanExpr"] = None
    right: Optional["BooleanExpr"] = None
    atom_text: Optional[str] = None
    atom_span: Optional[SourceSpan] = None
    pattern: Optional[Tuple[PatternElement, ...]] = None


@dataclass(frozen=True)
class WhereClause:
    """Parsed WHERE clause.

    Field coexistence rules (post-#1213).  Three observable shapes, all
    populated by the parser; consumers MUST handle all three:

    - **Structured path**: ``predicates`` populated, ``expr_tree is None``.
      ``predicates`` carries either ``WherePredicate`` entries (pure AND of
      comparable / has-labels predicates routed via the ``where_predicates``
      grammar rule, or AND-joined bare label predicates lifted by
      ``generic_where_clause`` via label narrowing) or a single
      ``WherePatternPredicate`` (pattern-only WHERE: ``WHERE (n)-[]->(m)``).
    - **Tree path**: ``predicates == ()``, ``expr_tree`` populated.  Fires
      when ``generic_where_clause`` cannot lift to structured predicates
      (OR / XOR / NOT / parenthesized boolean / non-label atoms); consumers
      walk ``expr_tree`` (boolean structure) and reconstruct surface text
      via ``boolean_expr_to_text`` when needed.
    - **Mixed path**: BOTH ``predicates`` (a single ``WherePatternPredicate``)
      AND ``expr_tree`` populated.  Fires for ``WHERE pattern AND expr`` and
      ``WHERE expr AND pattern`` (``_mixed_where_clause`` in parser.py).
      ``expr_tree`` carries a single-atom ``BooleanExpr`` whose ``atom_text``
      is the expr-side fragment; consumers handle both halves.
    """

    predicates: Tuple[WhereTerm, ...]
    span: SourceSpan
    expr_tree: Optional[BooleanExpr] = None


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
class UseClause:
    ref: str
    span: SourceSpan


@dataclass(frozen=True)
class GraphConstructor:
    matches: Tuple[MatchClause, ...]
    where: Optional[WhereClause]
    use: Optional[UseClause]
    span: SourceSpan
    call: Optional[CallClause] = None


@dataclass(frozen=True)
class GraphBinding:
    name: str
    constructor: GraphConstructor
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
    reentry_wheres: Tuple[Optional[WhereClause], ...] = ()
    reentry_unwinds: Tuple[UnwindClause, ...] = ()
    graph_bindings: Tuple[GraphBinding, ...] = ()
    use: Optional[UseClause] = None

    @property
    def match(self) -> Optional[MatchClause]:
        if not self.matches:
            return None
        return self.matches[-1]

    @property
    def reentry_where(self) -> Optional[WhereClause]:
        if not self.reentry_wheres:
            return None
        return self.reentry_wheres[0]


@dataclass(frozen=True)
class CypherGraphQuery:
    """A query whose final result is a graph (from a standalone GRAPH { } constructor)."""
    graph_bindings: Tuple[GraphBinding, ...]
    constructor: GraphConstructor
    trailing_semicolon: bool
    span: SourceSpan


@dataclass(frozen=True)
class CypherUnionQuery:
    branches: Tuple[CypherQuery, ...]
    union_kind: Literal["distinct", "all"]
    trailing_semicolon: bool
    span: SourceSpan
