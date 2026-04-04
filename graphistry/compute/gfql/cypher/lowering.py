from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import re
from typing import AbstractSet, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
from typing_extensions import Literal

from graphistry.compute.ast import (
    ASTEdge,
    ASTObject,
    ASTNode,
    distinct,
    e_forward,
    e_reverse,
    e_undirected,
    group_by,
    limit,
    order_by,
    return_,
    rows,
    serialize_binding_ops,
    select,
    skip,
    unwind,
    where_rows,
    with_,
)
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.predicates.ASTPredicate import ASTPredicate
from graphistry.compute.predicates.comparison import eq, ge, gt, isna, le, lt, ne, notna
from graphistry.compute.predicates.is_in import is_in
from graphistry.compute.predicates.logical import all_of
from graphistry.compute.predicates.str import contains as str_contains, endswith, never_match, startswith
from graphistry.compute.gfql.language_defs import GFQL_AGGREGATION_FUNCTIONS
from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    CaseWhen,
    ExprNode,
    FunctionCall,
    GFQLExprParseError,
    Identifier,
    IsNullOp,
    ListComprehension,
    ListLiteral,
    Literal as ExprLiteral,
    MapLiteral,
    PropertyAccessExpr,
    QuantifierExpr,
    SliceExpr,
    SubscriptExpr,
    UnaryOp,
    Wildcard,
    collect_identifiers,
    parse_expr,
    walk_expr_nodes,
)
from graphistry.compute.gfql.cypher.ast import (
    CallClause,
    CypherGraphQuery,
    CypherLiteral,
    CypherQuery,
    CypherUnionQuery,
    ExpressionText,
    GraphBinding,
    GraphConstructor,
    LabelRef,
    LimitClause,
    MatchClause,
    NodePattern,
    ParameterRef,
    OrderByClause,
    PatternElement,
    ProjectionStage,
    PropertyRef,
    PropertyEntry,
    RelationshipPattern,
    ReturnClause,
    ReturnItem,
    SourceSpan,
    UnwindClause,
    WhereClause,
    WherePredicate,
    WherePatternPredicate,
)
from graphistry.compute.gfql.cypher.call_procedures import (
    CompiledCypherProcedureCall,
    compile_cypher_call,
)
from graphistry.compute.gfql.temporal_text import (
    fold_temporal_constructor_ast,
    resolve_duration_text_property,
    rewrite_temporal_constructors_in_expr,
)
from graphistry.compute.gfql.same_path_types import WhereComparison, col, compare


@dataclass(frozen=True)
class LoweredCypherMatch:
    query: List[ASTObject]
    where: List[WhereComparison]
    row_where: Optional[ExpressionText] = None


_CYPHER_INT64_MIN = -(2**63)
_CYPHER_INT64_MAX = (2**63) - 1


@dataclass(frozen=True)
class CompiledCypherQuery:
    chain: Chain
    seed_rows: bool = False
    procedure_call: Optional[CompiledCypherProcedureCall] = None
    result_projection: Optional["ResultProjectionPlan"] = None
    empty_result_row: Optional[Dict[str, Any]] = None
    optional_null_fill: Optional["OptionalNullFillPlan"] = None
    optional_projection_row_guard: Optional["OptionalProjectionRowGuardPlan"] = None
    connected_optional_match: Optional["ConnectedOptionalMatchPlan"] = None
    connected_match_join: Optional["ConnectedMatchJoinPlan"] = None
    start_nodes_query: Optional["CompiledCypherQuery"] = None
    scalar_reentry_alias: Optional[str] = None
    scalar_reentry_columns: Tuple[str, ...] = ()
    graph_bindings: Tuple["CompiledGraphBinding", ...] = ()
    use_ref: Optional[str] = None


@dataclass(frozen=True)
class CompiledCypherUnionQuery:
    branches: Tuple[CompiledCypherQuery, ...]
    union_kind: Literal["distinct", "all"]


@dataclass(frozen=True)
class CompiledGraphBinding:
    """A named graph binding: GRAPH g = GRAPH { ... } compiled to a chain or procedure call."""
    name: str
    chain: Chain
    procedure_call: Optional[CompiledCypherProcedureCall] = None
    use_ref: Optional[str] = None


@dataclass(frozen=True)
class CompiledCypherGraphQuery:
    """A query whose final result is a graph (from standalone GRAPH { })."""
    graph_bindings: Tuple[CompiledGraphBinding, ...]
    chain: Chain
    procedure_call: Optional[CompiledCypherProcedureCall] = None
    use_ref: Optional[str] = None


@dataclass(frozen=True)
class OptionalNullFillPlan:
    base_chain: Chain
    null_row: Dict[str, Any]
    alignment_chain: Chain
    alignment_projection: "ResultProjectionPlan"
    alignment_output_name: str


@dataclass(frozen=True)
class OptionalProjectionRowGuardPlan:
    base_chains: Tuple[Chain, ...]


@dataclass(frozen=True)
class ResultProjectionColumn:
    output_name: str
    kind: Literal["whole_row", "property", "expr"]
    source_name: Optional[str] = None


@dataclass(frozen=True)
class ResultProjectionPlan:
    alias: str
    table: Literal["nodes", "edges"]
    columns: Tuple[ResultProjectionColumn, ...]
    exclude_columns: Tuple[str, ...] = ()


@dataclass(frozen=True)
class _ProjectionPlan:
    source_alias: str
    table: str
    whole_row_output_names: List[str]
    whole_row_sources: Dict[str, str]
    clause_kind: str
    projection_items: List[Tuple[str, Any]]
    projection_columns: List[ResultProjectionColumn]
    available_columns: Set[str]
    projected_property_outputs: Dict[str, str]
    output_to_source_property: Dict[str, str]
    output_to_expr_source: Dict[str, str]
    all_source_aliases: Optional[Set[str]] = None


@dataclass(frozen=True)
class _AggregateSpec:
    source_text: str
    output_name: str
    func: str
    expr_text: Optional[str]
    distinct: bool
    span_line: int
    span_column: int


@dataclass(frozen=True)
class _PostAggregateExprPlan:
    output_name: str
    expr: ExpressionText
    span_line: int
    span_column: int


@dataclass(frozen=True)
class _StageColumnBinding:
    kind: Literal["property", "expr"]
    source_name: str


@dataclass(frozen=True)
class _StageScope:
    mode: Literal["match_alias", "row_columns"]
    alias_targets: Dict[str, ASTObject]
    active_alias: Optional[str]
    row_columns: Set[str]
    projected_columns: Dict[str, _StageColumnBinding]
    table: Optional[Literal["nodes", "edges"]]
    seed_rows: bool
    relationship_count: int
    allowed_match_aliases: Set[str] = field(default_factory=set)


_CYPHER_PARAM_RE = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")
_CYPHER_PARAM_TOKEN_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_CYPHER_ENTITY_KEYS_RE = re.compile(r"\bkeys\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
_CYPHER_LABEL_PREDICATE_RE = re.compile(
    r"^\(\s*(?P<alias>[A-Za-z_][A-Za-z0-9_]*)((?::[A-Za-z_][A-Za-z0-9_]*)+)\s*\)$"
)
_CYPHER_BARE_LABEL_PREDICATE_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<alias>[A-Za-z_][A-Za-z0-9_]*)((?::[A-Za-z_][A-Za-z0-9_]*)+)(?![A-Za-z0-9_])"
)
_CYPHER_CHAINED_COMPARISON_RE = re.compile(
    r"^\s*(?P<left>.+?)\s*(?P<op1><=|>=|<>|!=|=|<|>)\s*(?P<middle>.+?)\s*(?P<op2><=|>=|<>|!=|=|<|>)\s*(?P<right>.+?)\s*$",
    re.DOTALL,
)
_CYPHER_AGGREGATES = frozenset({"count", "sum", "min", "max", "avg", "collect"})
_CYPHER_BARE_WHERE_GROUPED_ALIAS_RE = re.compile(r"^\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)$")


def _rewrite_label_predicate_expr(
    expr_text: str,
    *,
    alias_targets: Optional[Mapping[str, ASTObject]] = None,
) -> str:
    def _render(alias: str, labels_text: str) -> str:
        labels = [label for label in labels_text.split(":") if label]
        if alias_targets is not None:
            target = alias_targets.get(alias)
            if isinstance(target, ASTEdge):
                return " and ".join(f"{alias}.type = '{label}'" for label in labels)
        return " and ".join(f"{alias}.label__{label} = true" for label in labels)

    stripped = expr_text.strip()
    full_match = _CYPHER_LABEL_PREDICATE_RE.fullmatch(stripped)
    if full_match is not None:
        alias = full_match.group("alias")
        if alias_targets is None or alias in alias_targets:
            return _render(alias, full_match.group(2))

    def _rewrite_segment(segment: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            alias = match.group("alias")
            if alias_targets is not None and alias not in alias_targets:
                return match.group(0)
            return _render(alias, match.group(2))

        return _CYPHER_BARE_LABEL_PREDICATE_RE.sub(_replace, segment)

    return _rewrite_unquoted_expr_segments(expr_text, rewrite=_rewrite_segment)


def _rewrite_chained_comparison_expr(expr_text: str) -> str:
    match = _CYPHER_CHAINED_COMPARISON_RE.fullmatch(expr_text)
    if match is None:
        return expr_text
    left = match.group("left").strip()
    middle = match.group("middle").strip()
    right = match.group("right").strip()
    if any(token in segment.upper() for token in {" AND", " OR", " XOR"} for segment in (left, middle, right)):
        return expr_text
    return f"({left} {match.group('op1')} {middle}) AND ({middle} {match.group('op2')} {right})"


def _unsupported(message: str, *, field: str, value: Any, line: int, column: int) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        message,
        field=field,
        value=value,
        suggestion="Use a subset currently supported by the local Cypher compiler.",
        line=line,
        column=column,
        language="cypher",
    )


def _unsupported_at_span(message: str, *, field: str, value: Any, span: SourceSpan) -> GFQLValidationError:
    return _unsupported(
        message,
        field=field,
        value=value,
        line=span.line,
        column=span.column,
    )


def _rewrite_unquoted_expr_segments(
    expr_text: str,
    *,
    rewrite: Callable[[str], str],
) -> str:
    out: List[str] = []
    segment_start = 0
    idx = 0
    in_single = False
    in_double = False

    while idx < len(expr_text):
        ch = expr_text[idx]
        if ch == "'" and not in_double and (idx == 0 or expr_text[idx - 1] != "\\"):
            if not in_single:
                out.append(rewrite(expr_text[segment_start:idx]))
                segment_start = idx
            in_single = not in_single
            idx += 1
            if not in_single:
                out.append(expr_text[segment_start:idx])
                segment_start = idx
            continue
        if ch == '"' and not in_single and (idx == 0 or expr_text[idx - 1] != "\\"):
            if not in_double:
                out.append(rewrite(expr_text[segment_start:idx]))
                segment_start = idx
            in_double = not in_double
            idx += 1
            if not in_double:
                out.append(expr_text[segment_start:idx])
                segment_start = idx
            continue
        idx += 1

    trailing = expr_text[segment_start:]
    out.append(trailing if in_single or in_double else rewrite(trailing))
    return "".join(out)


def _parse_validated_row_expr_node(
    expr_text: str,
    *,
    alias_targets: Optional[Mapping[str, ASTObject]],
    field: str,
    line: int,
    column: int,
) -> ExprNode:
    node = fold_temporal_constructor_ast(parse_expr(expr_text))
    _validate_cypher_boolean_literal_constraints(
        node,
        field=field,
        value=expr_text,
        line=line,
        column=column,
    )
    if alias_targets:
        node = _rewrite_collection_alias_entities(node, alias_targets=alias_targets)
    _validate_cypher_expr_constraints(
        node,
        alias_targets=alias_targets,
        field=field,
        value=expr_text,
        line=line,
        column=column,
    )
    return node


def _parse_row_expr(
    expr_text: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    alias_targets: Optional[Mapping[str, ASTObject]] = None,
    allow_missing_params: bool = False,
    field: str,
    line: int,
    column: int,
) -> Any:
    prepared = _rewrite_param_expr(
        expr_text,
        params=params,
        field=field,
        line=line,
        column=column,
        allow_missing=allow_missing_params,
    )
    prepared = _rewrite_entity_keys_expr(prepared, alias_targets=alias_targets)
    prepared = _rewrite_label_predicate_expr(prepared, alias_targets=alias_targets)
    prepared = rewrite_temporal_constructors_in_expr(prepared)
    try:
        return _parse_validated_row_expr_node(
            prepared,
            alias_targets=alias_targets,
            field=field,
            line=line,
            column=column,
        )
    except (GFQLExprParseError, ImportError) as exc:
        rewritten = _rewrite_chained_comparison_expr(prepared)
        if rewritten != prepared:
            try:
                return _parse_validated_row_expr_node(
                    rewritten,
                    alias_targets=alias_targets,
                    field=field,
                    line=line,
                    column=column,
                )
            except (GFQLExprParseError, ImportError):
                pass
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher expression is outside the currently supported local GFQL subset",
            field=field,
            value=prepared,
            suggestion="Use column references, supported GFQL scalar expressions, or supported aggregate functions.",
            line=line,
            column=column,
            language="cypher",
        ) from exc


def _is_obviously_non_boolean_expr(node: ExprNode) -> bool:
    if isinstance(node, ExprLiteral):
        return node.value is not None and not isinstance(node.value, bool)
    return isinstance(node, (ListLiteral, ListComprehension, MapLiteral))


def _validate_cypher_boolean_literal_constraints(
    node: ExprNode,
    *,
    field: str,
    value: str,
    line: int,
    column: int,
) -> None:
    def _raise_invalid(op_name: str) -> None:
        raise GFQLValidationError(
            ErrorCode.E108,
            f"Cypher {op_name} requires boolean or null operands in the local compiler subset",
            field=field,
            value=value,
            suggestion="Use boolean/null operands for NOT/AND/OR/XOR, or rewrite the expression before compiling.",
            line=line,
            column=column,
            language="cypher",
        )

    if isinstance(node, UnaryOp):
        if node.op == "not" and _is_obviously_non_boolean_expr(node.operand):
            _raise_invalid("NOT")
        _validate_cypher_boolean_literal_constraints(
            node.operand,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        return
    if isinstance(node, BinaryOp):
        if node.op in {"and", "or", "xor"}:
            if _is_obviously_non_boolean_expr(node.left) or _is_obviously_non_boolean_expr(node.right):
                _raise_invalid(node.op.upper())
        _validate_cypher_boolean_literal_constraints(
            node.left,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        _validate_cypher_boolean_literal_constraints(
            node.right,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        return
    if isinstance(node, IsNullOp):
        _validate_cypher_boolean_literal_constraints(
            node.value,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        return
    if isinstance(node, FunctionCall):
        for arg in node.args:
            _validate_cypher_boolean_literal_constraints(
                arg,
                field=field,
                value=value,
                line=line,
                column=column,
            )
        return
    if isinstance(node, CaseWhen):
        _validate_cypher_boolean_literal_constraints(
            node.condition,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        _validate_cypher_boolean_literal_constraints(
            node.when_true,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        _validate_cypher_boolean_literal_constraints(
            node.when_false,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        return
    if isinstance(node, QuantifierExpr):
        _validate_cypher_boolean_literal_constraints(
            node.source,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        _validate_cypher_boolean_literal_constraints(
            node.predicate,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        return
    if isinstance(node, ListComprehension):
        _validate_cypher_boolean_literal_constraints(
            node.source,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        if node.predicate is not None:
            _validate_cypher_boolean_literal_constraints(
                node.predicate,
                field=field,
                value=value,
                line=line,
                column=column,
            )
        if node.projection is not None:
            _validate_cypher_boolean_literal_constraints(
                node.projection,
                field=field,
                value=value,
                line=line,
                column=column,
            )
        return
    if isinstance(node, ListLiteral):
        for item in node.items:
            _validate_cypher_boolean_literal_constraints(
                item,
                field=field,
                value=value,
                line=line,
                column=column,
            )
        return
    if isinstance(node, MapLiteral):
        for _, item in node.items:
            _validate_cypher_boolean_literal_constraints(
                item,
                field=field,
                value=value,
                line=line,
                column=column,
            )
        return
    if isinstance(node, SubscriptExpr):
        _validate_cypher_boolean_literal_constraints(
            node.value,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        _validate_cypher_boolean_literal_constraints(
            node.key,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        return
    if isinstance(node, SliceExpr):
        _validate_cypher_boolean_literal_constraints(
            node.value,
            field=field,
            value=value,
            line=line,
            column=column,
        )
        if node.start is not None:
            _validate_cypher_boolean_literal_constraints(
                node.start,
                field=field,
                value=value,
                line=line,
                column=column,
            )
        if node.stop is not None:
            _validate_cypher_boolean_literal_constraints(
                node.stop,
                field=field,
                value=value,
                line=line,
                column=column,
            )
        return
    if isinstance(node, PropertyAccessExpr):
        _validate_cypher_boolean_literal_constraints(
            node.value,
            field=field,
            value=value,
            line=line,
            column=column,
        )


def _list_item_invalid_for_tofloat(
    item: ExprNode,
    *,
    alias_targets: Optional[Mapping[str, ASTObject]],
) -> bool:
    if isinstance(item, ExprLiteral):
        return isinstance(item.value, bool) or isinstance(item.value, (list, dict))
    if isinstance(item, (ListLiteral, MapLiteral)):
        return True
    if isinstance(item, Identifier):
        return alias_targets is not None and item.name in alias_targets
    if isinstance(item, FunctionCall):
        return item.name in {"__node_entity__", "__edge_entity__"}
    return False


def _list_item_invalid_for_tostring(
    item: ExprNode,
    *,
    alias_targets: Optional[Mapping[str, ASTObject]],
) -> bool:
    if isinstance(item, Identifier):
        return alias_targets is not None and item.name in alias_targets
    if isinstance(item, FunctionCall):
        return item.name in {"__node_entity__", "__edge_entity__"}
    return False


def _contains_aggregate_call(node: ExprNode) -> bool:
    found = False

    def _enter(current: ExprNode) -> None:
        nonlocal found
        if isinstance(current, FunctionCall) and current.name in GFQL_AGGREGATION_FUNCTIONS:
            found = True

    walk_expr_nodes(node, enter=_enter)
    return found


def _validate_cypher_expr_constraints(
    node: ExprNode,
    *,
    alias_targets: Optional[Mapping[str, ASTObject]],
    field: str,
    value: str,
    line: int,
    column: int,
) -> None:
    def _raise(message: str) -> None:
        raise GFQLValidationError(
            ErrorCode.E108,
            message,
            field=field,
            value=value,
            suggestion="Use the supported local Cypher subset or rewrite the expression before compiling.",
            line=line,
            column=column,
            language="cypher",
        )

    def _enter(current: ExprNode) -> None:
        if isinstance(current, ExprLiteral) and isinstance(current.value, int) and not isinstance(current.value, bool):
            if current.value < _CYPHER_INT64_MIN or current.value > _CYPHER_INT64_MAX:
                _raise("Cypher integer literal is out of the supported 64-bit range")
        if isinstance(current, BinaryOp) and current.op.lower() == "in":
            if (
                isinstance(current.right, ExprLiteral)
                and current.right.value is not None
            ) or isinstance(current.right, MapLiteral):
                _raise("Cypher IN requires a list-valued right-hand side in the local compiler")
        if isinstance(current, SubscriptExpr) and isinstance(current.key, ExprLiteral):
            key_value = current.key.value
            if isinstance(key_value, bool) or isinstance(key_value, float):
                _raise("Cypher list indexing requires integer keys in the local compiler")
        if isinstance(current, ListComprehension):
            if current.predicate is not None and _contains_aggregate_call(current.predicate):
                _raise("Cypher list comprehensions cannot contain aggregate functions in the local compiler")
            if current.projection is not None and _contains_aggregate_call(current.projection):
                _raise("Cypher list comprehensions cannot contain aggregate functions in the local compiler")
            if (
                isinstance(current.projection, FunctionCall)
                and len(current.projection.args) == 1
                and isinstance(current.projection.args[0], Identifier)
                and current.projection.args[0].name == current.var
                and isinstance(current.source, ListLiteral)
            ):
                if current.projection.name == "tofloat":
                    if any(
                        _list_item_invalid_for_tofloat(item, alias_targets=alias_targets)
                        for item in current.source.items
                    ):
                        _raise("Cypher toFloat() cannot consume the current list-comprehension item types in the local compiler")
                if current.projection.name == "tostring":
                    if any(
                        _list_item_invalid_for_tostring(item, alias_targets=alias_targets)
                        for item in current.source.items
                    ):
                        _raise("Cypher toString() cannot consume the current list-comprehension item types in the local compiler")

    walk_expr_nodes(node, enter=_enter)


def _validate_with_projection_aliasing(stage: ProjectionStage) -> None:
    if stage.clause.kind != "with":
        return
    for item in stage.clause.items:
        if item.alias is not None:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item.expression.text):
            continue
        raise _unsupported(
            "Cypher WITH requires aliasing non-variable projection expressions in the local compiler",
            field="with",
            value=item.expression.text,
            line=item.span.line,
            column=item.span.column,
        )


def _cypher_literal_expr_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "null"
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_cypher_literal_expr_text(item) for item in value) + "]"
    if isinstance(value, dict):
        parts: List[str] = []
        for key, item in value.items():
            key_txt = str(key)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_txt):
                rendered_key = key_txt
            else:
                rendered_key = "'" + key_txt.replace("\\", "\\\\").replace("'", "\\'") + "'"
            parts.append(f"{rendered_key}: {_cypher_literal_expr_text(item)}")
        return "{" + ", ".join(parts) + "}"
    raise GFQLValidationError(
        ErrorCode.E108,
        "Cypher parameter value is outside the currently supported literal subset",
        field="params",
        value=type(value).__name__,
        suggestion="Use null, booleans, numbers, strings, lists, or maps as parameter values.",
        language="cypher",
    )


def _render_expr_node(node: ExprNode) -> str:
    if isinstance(node, Identifier):
        return node.name
    if isinstance(node, ExprLiteral):
        return _cypher_literal_expr_text(node.value)
    if isinstance(node, UnaryOp):
        operand = _render_expr_node(node.operand)
        if node.op == "not":
            return f"(NOT {operand})"
        return f"({node.op}{operand})"
    if isinstance(node, BinaryOp):
        left = _render_expr_node(node.left)
        right = _render_expr_node(node.right)
        if node.op in {"and", "or", "xor", "in"}:
            op_txt = node.op.upper()
        elif node.op == "starts_with":
            op_txt = "STARTS WITH"
        elif node.op == "ends_with":
            op_txt = "ENDS WITH"
        elif node.op == "contains":
            op_txt = "CONTAINS"
        else:
            op_txt = node.op
        return f"({left} {op_txt} {right})"
    if isinstance(node, IsNullOp):
        suffix = "IS NOT NULL" if node.negated else "IS NULL"
        return f"({_render_expr_node(node.value)} {suffix})"
    if isinstance(node, FunctionCall):
        args = ", ".join(_render_expr_node(arg) for arg in node.args)
        if node.distinct:
            args = f"DISTINCT {args}"
        return f"{node.name}({args})"
    if isinstance(node, Wildcard):
        return "*"
    if isinstance(node, CaseWhen):
        return (
            "CASE WHEN "
            f"{_render_expr_node(node.condition)} THEN {_render_expr_node(node.when_true)} "
            f"ELSE {_render_expr_node(node.when_false)} END"
        )
    if isinstance(node, QuantifierExpr):
        return (
            f"{node.fn.upper()}({node.var} IN {_render_expr_node(node.source)} "
            f"WHERE {_render_expr_node(node.predicate)})"
        )
    if isinstance(node, ListComprehension):
        rendered = f"[{node.var} IN {_render_expr_node(node.source)}"
        if node.predicate is not None:
            rendered += f" WHERE {_render_expr_node(node.predicate)}"
        if node.projection is not None:
            rendered += f" | {_render_expr_node(node.projection)}"
        return rendered + "]"
    if isinstance(node, ListLiteral):
        return "[" + ", ".join(_render_expr_node(item) for item in node.items) + "]"
    if isinstance(node, MapLiteral):
        parts: List[str] = []
        for key, value in node.items:
            rendered_key = key if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) else _cypher_literal_expr_text(key)
            parts.append(f"{rendered_key}: {_render_expr_node(value)}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(node, SubscriptExpr):
        return f"{_render_expr_node(node.value)}[{_render_expr_node(node.key)}]"
    if isinstance(node, SliceExpr):
        start = "" if node.start is None else _render_expr_node(node.start)
        stop = "" if node.stop is None else _render_expr_node(node.stop)
        return f"{_render_expr_node(node.value)}[{start}..{stop}]"
    if isinstance(node, PropertyAccessExpr):
        return f"{_render_expr_node(node.value)}.{node.property}"
    raise TypeError(f"Unsupported expression node type for rendering: {type(node).__name__}")


def _rewrite_expr_identifiers(node: ExprNode, replacements: Mapping[str, str]) -> ExprNode:
    if isinstance(node, Identifier):
        return Identifier(replacements.get(node.name, node.name))
    if isinstance(node, CaseWhen):
        return CaseWhen(
            _rewrite_expr_identifiers(node.condition, replacements),
            _rewrite_expr_identifiers(node.when_true, replacements),
            _rewrite_expr_identifiers(node.when_false, replacements),
        )
    if isinstance(node, QuantifierExpr):
        shadowed = {
            name: replacement
            for name, replacement in replacements.items()
            if name != node.var and not name.startswith(f"{node.var}.")
        }
        return QuantifierExpr(
            node.fn,
            node.var,
            _rewrite_expr_identifiers(node.source, shadowed),
            _rewrite_expr_identifiers(node.predicate, shadowed),
        )
    if isinstance(node, ListComprehension):
        shadowed = {
            name: replacement
            for name, replacement in replacements.items()
            if name != node.var and not name.startswith(f"{node.var}.")
        }
        return ListComprehension(
            node.var,
            _rewrite_expr_identifiers(node.source, shadowed),
            predicate=None if node.predicate is None else _rewrite_expr_identifiers(node.predicate, shadowed),
            projection=None if node.projection is None else _rewrite_expr_identifiers(node.projection, shadowed),
        )
    return _rebuild_expr_node(
        node,
        rewrite=lambda child: _rewrite_expr_identifiers(child, replacements),
        error_context="identifier rewrite",
    )


def _rebuild_expr_node(
    node: ExprNode,
    *,
    rewrite: Callable[[ExprNode], ExprNode],
    error_context: str,
) -> ExprNode:
    if isinstance(node, (Identifier, ExprLiteral, Wildcard)):
        return node
    if isinstance(node, UnaryOp):
        return UnaryOp(node.op, rewrite(node.operand))
    if isinstance(node, BinaryOp):
        return BinaryOp(node.op, rewrite(node.left), rewrite(node.right))
    if isinstance(node, IsNullOp):
        return IsNullOp(rewrite(node.value), negated=node.negated)
    if isinstance(node, FunctionCall):
        return FunctionCall(node.name, tuple(rewrite(arg) for arg in node.args), distinct=node.distinct)
    if isinstance(node, CaseWhen):
        return CaseWhen(rewrite(node.condition), rewrite(node.when_true), rewrite(node.when_false))
    if isinstance(node, QuantifierExpr):
        return QuantifierExpr(node.fn, node.var, rewrite(node.source), rewrite(node.predicate))
    if isinstance(node, ListComprehension):
        return ListComprehension(
            node.var,
            rewrite(node.source),
            predicate=None if node.predicate is None else rewrite(node.predicate),
            projection=None if node.projection is None else rewrite(node.projection),
        )
    if isinstance(node, ListLiteral):
        return ListLiteral(tuple(rewrite(item) for item in node.items))
    if isinstance(node, MapLiteral):
        return MapLiteral(tuple((key, rewrite(value)) for key, value in node.items))
    if isinstance(node, SubscriptExpr):
        return SubscriptExpr(rewrite(node.value), rewrite(node.key))
    if isinstance(node, SliceExpr):
        return SliceExpr(
            rewrite(node.value),
            None if node.start is None else rewrite(node.start),
            None if node.stop is None else rewrite(node.stop),
        )
    if isinstance(node, PropertyAccessExpr):
        return PropertyAccessExpr(rewrite(node.value), node.property)
    raise TypeError(f"Unsupported expression node type for {error_context}: {type(node).__name__}")


def _expr_is_cypher_integer_like(node: ExprNode, *, integer_identifiers: AbstractSet[str]) -> bool:
    if isinstance(node, Identifier):
        return node.name in integer_identifiers
    if isinstance(node, ExprLiteral):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, UnaryOp):
        return node.op in {"+", "-"} and _expr_is_cypher_integer_like(node.operand, integer_identifiers=integer_identifiers)
    if isinstance(node, BinaryOp):
        return node.op in {"+", "-", "*", "%", "/"} and _expr_is_cypher_integer_like(
            node.left,
            integer_identifiers=integer_identifiers,
        ) and _expr_is_cypher_integer_like(
            node.right,
            integer_identifiers=integer_identifiers,
        )
    if isinstance(node, FunctionCall):
        return node.name.lower() == "tointeger" and len(node.args) == 1
    return False


def _rewrite_cypher_integer_division_ast(
    node: ExprNode,
    *,
    integer_identifiers: AbstractSet[str],
) -> ExprNode:
    if isinstance(node, BinaryOp):
        left = _rewrite_cypher_integer_division_ast(node.left, integer_identifiers=integer_identifiers)
        right = _rewrite_cypher_integer_division_ast(node.right, integer_identifiers=integer_identifiers)
        rewritten = BinaryOp(node.op, left, right)
        if node.op == "/" and _expr_is_cypher_integer_like(left, integer_identifiers=integer_identifiers) and _expr_is_cypher_integer_like(
            right,
            integer_identifiers=integer_identifiers,
        ):
            return FunctionCall("toInteger", (rewritten,))
        return rewritten
    return _rebuild_expr_node(
        node,
        rewrite=lambda child: _rewrite_cypher_integer_division_ast(child, integer_identifiers=integer_identifiers),
        error_context="cypher integer division rewrite",
    )


def _rewrite_alias_properties_to_outputs(
    expr_text: str,
    *,
    source_alias: str,
    property_outputs: Mapping[str, str],
    params: Optional[Mapping[str, Any]],
    alias_targets: Mapping[str, ASTObject],
    field: str,
    line: int,
    column: int,
) -> str:
    node = _parse_row_expr(
        expr_text,
        params=params,
        alias_targets=alias_targets,
        field=field,
        line=line,
        column=column,
    )

    def _rewrite(node_in: ExprNode) -> ExprNode:
        if isinstance(node_in, PropertyAccessExpr) and isinstance(node_in.value, Identifier):
            alias_name, prop = _qualified_ref_from_node(
                node_in,
                field=field,
                value=expr_text,
                line=line,
                column=column,
            )
            if alias_name == source_alias and prop is not None:
                output_name = property_outputs.get(prop)
                if output_name is not None:
                    return Identifier(output_name)
            return PropertyAccessExpr(_rewrite(node_in.value), node_in.property)
        return _rebuild_expr_node(node_in, rewrite=_rewrite, error_context="alias property rewrite")

    return _render_expr_node(_rewrite(node))


def _rewrite_expr_to_output_names(
    expr_text: str,
    *,
    replacements: Mapping[str, str],
    params: Optional[Mapping[str, Any]],
    field: str,
    line: int,
    column: int,
) -> str:
    node = _parse_row_expr(
        expr_text,
        params=params,
        field=field,
        line=line,
        column=column,
    )

    def _rewrite(node_in: ExprNode) -> ExprNode:
        rendered = _render_expr_node(node_in)
        replacement = replacements.get(rendered)
        if replacement is not None:
            return Identifier(replacement)
        if isinstance(node_in, Identifier):
            return Identifier(replacements.get(node_in.name, node_in.name))
        return _rebuild_expr_node(node_in, rewrite=_rewrite, error_context="output rewrite")

    return _render_expr_node(_rewrite(node))


def _connected_join_alias_identity_expr(
    expr_text: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
    field: str,
    line: int,
    column: int,
) -> str:
    node = _parse_row_expr(
        expr_text,
        params=params,
        alias_targets=alias_targets,
        field=field,
        line=line,
        column=column,
    )

    def _rewrite(node_in: ExprNode) -> ExprNode:
        if isinstance(node_in, PropertyAccessExpr) and isinstance(node_in.value, Identifier):
            if "." not in node_in.value.name and node_in.value.name in alias_targets:
                return node_in
            return PropertyAccessExpr(cast(ExprNode, _rewrite(node_in.value)), node_in.property)
        if isinstance(node_in, Identifier) and "." not in node_in.name and node_in.name in alias_targets:
            target = alias_targets[node_in.name]
            prop = "id" if isinstance(target, ASTNode) else "__gfql_edge_index_0__"
            return PropertyAccessExpr(Identifier(node_in.name), prop)
        return _rebuild_expr_node(node_in, rewrite=_rewrite, error_context="connected join identity rewrite")

    return _render_expr_node(_rewrite(node))


def _rewrite_connected_join_return_clause(
    clause: ReturnClause,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> ReturnClause:
    return replace(
        clause,
        items=tuple(
            replace(
                item,
                expression=ExpressionText(
                    text=_connected_join_alias_identity_expr(
                        item.expression.text,
                        alias_targets=alias_targets,
                        params=params,
                        field=clause.kind,
                        line=item.span.line,
                        column=item.span.column,
                    ),
                    span=item.expression.span,
                ),
            )
            for item in clause.items
        ),
    )


def _rewrite_connected_join_order_by_clause(
    clause: Optional[OrderByClause],
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> Optional[OrderByClause]:
    if clause is None:
        return None
    return replace(
        clause,
        items=tuple(
            replace(
                item,
                expression=ExpressionText(
                    text=_connected_join_alias_identity_expr(
                        item.expression.text,
                        alias_targets=alias_targets,
                        params=params,
                        field="order_by",
                        line=item.span.line,
                        column=item.span.column,
                    ),
                    span=item.expression.span,
                ),
            )
            for item in clause.items
        ),
    )


def _rewrite_connected_join_expr(
    expr: Optional[ExpressionText],
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
    field: str,
    line: int,
    column: int,
) -> Optional[ExpressionText]:
    if expr is None:
        return None
    return ExpressionText(
        text=_connected_join_alias_identity_expr(
            expr.text,
            alias_targets=alias_targets,
            params=params,
            field=field,
            line=line,
            column=column,
        ),
        span=expr.span,
    )


def _reject_unsupported_connected_join_clause_shapes(
    clause: ReturnClause,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> None:
    for item in clause.items:
        if item.expression.text == "*":
            raise _unsupported(
                f"Cypher {clause.kind.upper()} * is not yet supported for connected comma-pattern join lowering",
                field=clause.kind,
                value="*",
                line=item.span.line,
                column=item.span.column,
            )
        agg_spec = _aggregate_spec(item, params=params, alias_targets=alias_targets)
        if agg_spec is not None and agg_spec.func == "collect" and agg_spec.expr_text in alias_targets:
            raise _unsupported(
                "Cypher collect(alias) is not yet supported for connected comma-pattern join lowering",
                field=clause.kind,
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )
        if agg_spec is None and item.expression.text in alias_targets:
            raise _unsupported(
                "Cypher whole-row alias projection is not yet supported for connected comma-pattern join lowering",
                field=clause.kind,
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )

def _rewrite_param_expr(
    expr_text: str,
    *,
    params: Optional[Mapping[str, Any]],
    field: str,
    line: int,
    column: int,
    allow_missing: bool,
) -> str:
    out: List[str] = []
    idx = 0
    in_single = False
    in_double = False
    while idx < len(expr_text):
        ch = expr_text[idx]
        if ch == "'" and not in_double and (idx == 0 or expr_text[idx - 1] != "\\"):
            in_single = not in_single
            out.append(ch)
            idx += 1
            continue
        if ch == '"' and not in_single and (idx == 0 or expr_text[idx - 1] != "\\"):
            in_double = not in_double
            out.append(ch)
            idx += 1
            continue
        if not in_single and not in_double and ch == "$":
            match = _CYPHER_PARAM_TOKEN_RE.match(expr_text, idx)
            if match is not None:
                name = match.group(1)
                if params is None or name not in params:
                    if allow_missing:
                        out.append("null")
                    else:
                        raise GFQLValidationError(
                            ErrorCode.E105,
                            f"Missing Cypher parameter '${name}'",
                            field=field,
                            value=name,
                            suggestion=f"Pass params={{'{name}': ...}} when compiling or executing the query.",
                            line=line,
                            column=column,
                            language="cypher",
                        )
                else:
                    out.append(_cypher_literal_expr_text(params[name]))
                idx = match.end()
                continue
        out.append(ch)
        idx += 1
    return "".join(out)


def _rewrite_entity_keys_expr(
    expr_text: str,
    *,
    alias_targets: Optional[Mapping[str, ASTObject]],
) -> str:
    if not alias_targets:
        return expr_text

    alias_names = list(alias_targets.keys())

    def _rewrite_segment(segment: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            alias = match.group(1)
            target = alias_targets.get(alias)
            if target is None:
                return match.group(0)
            fn = "__node_keys__" if isinstance(target, ASTNode) else "__edge_keys__"
            ordered_aliases = [alias] + [name for name in alias_names if name != alias]
            return f"{fn}({', '.join(ordered_aliases)})"

        return _CYPHER_ENTITY_KEYS_RE.sub(_replace, segment)

    return _rewrite_unquoted_expr_segments(expr_text, rewrite=_rewrite_segment)


def _entity_wrapper_call(
    alias_name: str,
    *,
    alias_targets: Mapping[str, ASTObject],
) -> ExprNode:
    target = alias_targets.get(alias_name)
    if isinstance(target, ASTNode):
        fn = "__node_entity__"
    elif isinstance(target, ASTEdge):
        fn = "__edge_entity__"
    else:
        return Identifier(alias_name)
    extra_aliases = tuple(
        ExprLiteral(str(name))
        for name in alias_targets.keys()
        if str(name) != alias_name
    )
    return FunctionCall(fn, (Identifier(alias_name),) + extra_aliases)


def _rewrite_collection_alias_entities(
    node: ExprNode,
    *,
    alias_targets: Mapping[str, ASTObject],
    inside_collection: bool = False,
) -> ExprNode:
    if isinstance(node, Identifier):
        if inside_collection and "." not in node.name and node.name in alias_targets:
            return _entity_wrapper_call(node.name, alias_targets=alias_targets)
        return node
    if isinstance(node, ExprLiteral):
        return node
    if isinstance(node, UnaryOp):
        return UnaryOp(node.op, _rewrite_collection_alias_entities(node.operand, alias_targets=alias_targets))
    if isinstance(node, BinaryOp):
        return BinaryOp(
            node.op,
            _rewrite_collection_alias_entities(node.left, alias_targets=alias_targets),
            _rewrite_collection_alias_entities(node.right, alias_targets=alias_targets),
        )
    if isinstance(node, IsNullOp):
        return IsNullOp(_rewrite_collection_alias_entities(node.value, alias_targets=alias_targets), negated=node.negated)
    if isinstance(node, FunctionCall):
        return FunctionCall(
            node.name,
            tuple(_rewrite_collection_alias_entities(arg, alias_targets=alias_targets) for arg in node.args),
            distinct=node.distinct,
        )
    if isinstance(node, Wildcard):
        return node
    if isinstance(node, CaseWhen):
        return CaseWhen(
            _rewrite_collection_alias_entities(node.condition, alias_targets=alias_targets),
            _rewrite_collection_alias_entities(node.when_true, alias_targets=alias_targets),
            _rewrite_collection_alias_entities(node.when_false, alias_targets=alias_targets),
        )
    if isinstance(node, QuantifierExpr):
        return QuantifierExpr(
            node.fn,
            node.var,
            _rewrite_collection_alias_entities(node.source, alias_targets=alias_targets),
            _rewrite_collection_alias_entities(node.predicate, alias_targets=alias_targets),
        )
    if isinstance(node, ListComprehension):
        return ListComprehension(
            node.var,
            _rewrite_collection_alias_entities(node.source, alias_targets=alias_targets),
            predicate=None
            if node.predicate is None
            else _rewrite_collection_alias_entities(node.predicate, alias_targets=alias_targets),
            projection=None
            if node.projection is None
            else _rewrite_collection_alias_entities(node.projection, alias_targets=alias_targets),
        )
    if isinstance(node, ListLiteral):
        return ListLiteral(
            tuple(
                _rewrite_collection_alias_entities(item, alias_targets=alias_targets, inside_collection=True)
                for item in node.items
            )
        )
    if isinstance(node, MapLiteral):
        return MapLiteral(
            tuple(
                (
                    key,
                    _rewrite_collection_alias_entities(value, alias_targets=alias_targets, inside_collection=True),
                )
                for key, value in node.items
            )
        )
    return _rebuild_expr_node(
        node,
        rewrite=lambda child: _rewrite_collection_alias_entities(child, alias_targets=alias_targets),
        error_context="collection alias rewrite",
    )


def _expr_match_alias_usage(
    expr_text: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]] = None,
    field: str,
    line: int,
    column: int,
) -> Tuple[Set[str], Set[str]]:
    node = _parse_row_expr(
        expr_text,
        params=params,
        alias_targets=alias_targets,
        allow_missing_params=True,
        field=field,
        line=line,
        column=column,
    )
    non_aggregate_aliases: Set[str] = set()
    aggregate_aliases: Set[str] = set()

    def _visit(node_in: ExprNode, *, inside_aggregate: bool = False) -> None:
        if isinstance(node_in, Identifier):
            root = node_in.name.split(".", 1)[0]
            if root in alias_targets:
                if inside_aggregate:
                    aggregate_aliases.add(root)
                else:
                    non_aggregate_aliases.add(root)
            return
        if isinstance(node_in, ExprLiteral):
            return
        if isinstance(node_in, UnaryOp):
            _visit(node_in.operand, inside_aggregate=inside_aggregate)
            return
        if isinstance(node_in, BinaryOp):
            _visit(node_in.left, inside_aggregate=inside_aggregate)
            _visit(node_in.right, inside_aggregate=inside_aggregate)
            return
        if isinstance(node_in, IsNullOp):
            _visit(node_in.value, inside_aggregate=inside_aggregate)
            return
        if isinstance(node_in, FunctionCall):
            child_aggregate = inside_aggregate or node_in.name in _CYPHER_AGGREGATES
            for arg in node_in.args:
                _visit(arg, inside_aggregate=child_aggregate)
            return
        if isinstance(node_in, Wildcard):
            return
        if isinstance(node_in, CaseWhen):
            _visit(node_in.condition, inside_aggregate=inside_aggregate)
            _visit(node_in.when_true, inside_aggregate=inside_aggregate)
            _visit(node_in.when_false, inside_aggregate=inside_aggregate)
            return
        if isinstance(node_in, QuantifierExpr):
            _visit(node_in.source, inside_aggregate=inside_aggregate)
            _visit(node_in.predicate, inside_aggregate=inside_aggregate)
            return
        if isinstance(node_in, ListComprehension):
            _visit(node_in.source, inside_aggregate=inside_aggregate)
            if node_in.predicate is not None:
                _visit(node_in.predicate, inside_aggregate=inside_aggregate)
            if node_in.projection is not None:
                _visit(node_in.projection, inside_aggregate=inside_aggregate)
            return
        if isinstance(node_in, ListLiteral):
            for item in node_in.items:
                _visit(item, inside_aggregate=inside_aggregate)
            return
        if isinstance(node_in, MapLiteral):
            for _key, value in node_in.items:
                _visit(value, inside_aggregate=inside_aggregate)
            return
        if isinstance(node_in, SubscriptExpr):
            _visit(node_in.value, inside_aggregate=inside_aggregate)
            _visit(node_in.key, inside_aggregate=inside_aggregate)
            return
        if isinstance(node_in, SliceExpr):
            _visit(node_in.value, inside_aggregate=inside_aggregate)
            if node_in.start is not None:
                _visit(node_in.start, inside_aggregate=inside_aggregate)
            if node_in.stop is not None:
                _visit(node_in.stop, inside_aggregate=inside_aggregate)
            return
        if isinstance(node_in, PropertyAccessExpr):
            _visit(node_in.value, inside_aggregate=inside_aggregate)
            return

    _visit(node)
    return non_aggregate_aliases, aggregate_aliases


def _expr_match_aliases(
    expr_text: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]] = None,
    field: str,
    line: int,
    column: int,
) -> Set[str]:
    non_aggregate_aliases, aggregate_aliases = _expr_match_alias_usage(
        expr_text,
        alias_targets=alias_targets,
        params=params,
        field=field,
        line=line,
        column=column,
    )
    return non_aggregate_aliases | aggregate_aliases


def _expr_non_aggregate_match_aliases(
    expr_text: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]] = None,
    field: str,
    line: int,
    column: int,
) -> Set[str]:
    non_aggregate_aliases, _aggregate_aliases = _expr_match_alias_usage(
        expr_text,
        alias_targets=alias_targets,
        params=params,
        field=field,
        line=line,
        column=column,
    )
    return non_aggregate_aliases


def _expr_aggregate_match_aliases(
    expr_text: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]] = None,
    field: str,
    line: int,
    column: int,
) -> Set[str]:
    _non_aggregate_aliases, aggregate_aliases = _expr_match_alias_usage(
        expr_text,
        alias_targets=alias_targets,
        params=params,
        field=field,
        line=line,
        column=column,
    )
    return aggregate_aliases


def _validate_row_expr_scope(
    expr_text: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    active_match_alias: Optional[str],
    allowed_match_aliases: Optional[AbstractSet[str]] = None,
    unwind_aliases: Iterable[str],
    params: Optional[Mapping[str, Any]] = None,
    field: str,
    line: int,
    column: int,
) -> None:
    allowed_roots = set(unwind_aliases)
    if allowed_match_aliases is not None:
        allowed_roots.update(allowed_match_aliases)
    for root in _expr_match_aliases(
        expr_text,
        alias_targets=alias_targets,
        params=params,
        field=field,
        line=line,
        column=column,
    ):
        if active_match_alias is None and root not in allowed_roots:
            raise _unsupported(
                "Cypher row expressions cannot reference MATCH aliases when no row source is active",
                field=field,
                value=expr_text,
                line=line,
                column=column,
            )
        if active_match_alias is not None and root != active_match_alias and root not in allowed_roots:
            raise _unsupported(
                "Cypher row lowering currently supports one MATCH source alias at a time",
                field=field,
                value=expr_text,
                line=line,
                column=column,
            )


def _whole_param_name(expr_text: str) -> Optional[str]:
    match = _CYPHER_PARAM_RE.match(expr_text.strip())
    if match is None:
        return None
    return match.group(1)


def _row_expr_arg(
    expr: ExpressionText,
    *,
    params: Optional[Mapping[str, Any]],
    alias_targets: Optional[Mapping[str, ASTObject]] = None,
    field: str,
) -> Any:
    param_name = _whole_param_name(expr.text)
    if param_name is not None:
        return _resolve_literal(
            ParameterRef(name=param_name, span=expr.span),
            params=params,
            field=field,
        )
    node = _parse_row_expr(
        expr.text,
        params=params,
        alias_targets=alias_targets,
        field=field,
        line=expr.span.line,
        column=expr.span.column,
    )
    return _render_expr_node(node)


def _projected_source_replacement(binding: _StageColumnBinding) -> str:
    source = binding.source_name
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", source):
        return source
    if source in {"null", "true", "false"}:
        return source
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", source):
        return source
    if source.startswith(("'", "[", "{")):
        return source
    return f"({source})"


def _rewrite_expr_to_projected_sources(
    expr: ExpressionText,
    *,
    projected_columns: Optional[Mapping[str, _StageColumnBinding]],
    params: Optional[Mapping[str, Any]],
    alias_targets: Mapping[str, ASTObject],
    field: str,
) -> ExpressionText:
    if not projected_columns:
        return expr
    prepared = _rewrite_param_expr(
        expr.text,
        params=params,
        field=field,
        line=expr.span.line,
        column=expr.span.column,
        allow_missing=False,
    )
    prepared = _rewrite_entity_keys_expr(prepared, alias_targets=alias_targets)
    try:
        node = parse_expr(prepared)
    except (GFQLExprParseError, ImportError) as exc:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher expression is outside the currently supported local GFQL subset",
            field=field,
            value=prepared,
            suggestion="Use column references, supported GFQL scalar expressions, or supported aggregate functions.",
            line=expr.span.line,
            column=expr.span.column,
            language="cypher",
        ) from exc
    replacements: Dict[str, str] = {}
    for ident in collect_identifiers(node):
        binding = projected_columns.get(ident)
        if binding is not None:
            replacements[ident] = _projected_source_replacement(binding)
            continue
        alias_name, prop = _split_qualified_name(ident, line=expr.span.line, column=expr.span.column)
        if prop is None:
            continue
        base_binding = projected_columns.get(alias_name)
        if base_binding is None or base_binding.kind != "expr":
            continue
        try:
            source_node = fold_temporal_constructor_ast(parse_expr(base_binding.source_name))
        except (GFQLExprParseError, ImportError):
            continue
        if not isinstance(source_node, ExprLiteral) or not isinstance(source_node.value, str):
            continue
        replacement = resolve_duration_text_property(source_node.value, prop)
        if replacement is None:
            continue
        replacements[ident] = replacement
    if not replacements:
        return ExpressionText(text=prepared, span=expr.span)
    return ExpressionText(
        text=_render_expr_node(_rewrite_expr_identifiers(node, replacements)),
        span=expr.span,
    )


def _aggregate_spec(
    item: ReturnItem,
    *,
    params: Optional[Mapping[str, Any]] = None,
    alias_targets: Optional[Mapping[str, ASTObject]] = None,
) -> Optional[_AggregateSpec]:
    if item.expression.text == "*":
        return None
    node = _parse_row_expr(
        item.expression.text,
        params=params,
        alias_targets=alias_targets,
        field="return.item",
        line=item.span.line,
        column=item.span.column,
    )
    if not isinstance(node, FunctionCall) or node.name not in _CYPHER_AGGREGATES:
        return None

    if len(node.args) == 0:
        raise _unsupported(
            "Cypher aggregate functions require an argument or '*'",
            field="return.item",
            value=item.expression.text,
            line=item.span.line,
            column=item.span.column,
        )
    if len(node.args) != 1:
        raise _unsupported(
            "Cypher local aggregate lowering currently supports single-argument aggregate functions only",
            field="return.item",
            value=item.expression.text,
            line=item.span.line,
            column=item.span.column,
        )

    arg = node.args[0]
    if isinstance(arg, Wildcard):
        if node.name != "count":
            raise _unsupported(
                "Only count(*) supports '*' in the local Cypher aggregate subset",
                field="return.item",
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )
        if node.distinct:
            raise _unsupported(
                "count(DISTINCT *) is not supported in the local Cypher aggregate subset",
                field="return.item",
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )
        expr_text: Optional[str] = None
    else:
        open_paren = item.expression.text.find("(")
        close_paren = item.expression.text.rfind(")")
        if open_paren < 0 or close_paren <= open_paren:
            raise _unsupported(
                "Invalid Cypher aggregate function syntax",
                field="return.item",
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )
        expr_text = item.expression.text[open_paren + 1:close_paren].strip()
        if node.distinct:
            distinct_match = re.match(r"(?is)^distinct\s+(.+)$", expr_text)
            if distinct_match is None:
                raise _unsupported(
                    "Cypher DISTINCT aggregate syntax requires a non-empty argument",
                    field="return.item",
                    value=item.expression.text,
                    line=item.span.line,
                    column=item.span.column,
                )
            expr_text = distinct_match.group(1).strip()
        if expr_text == "":
            raise _unsupported(
                "Cypher aggregate functions require a non-empty argument",
                field="return.item",
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )

    return _AggregateSpec(
        source_text=item.expression.text,
        output_name=item.alias or item.expression.text,
        func=node.name,
        expr_text=expr_text,
        distinct=node.distinct,
        span_line=item.span.line,
        span_column=item.span.column,
    )


def _aggregate_spec_from_function_call(
    node: FunctionCall,
    *,
    source_text: str,
    output_name: str,
    span_line: int,
    span_column: int,
) -> _AggregateSpec:
    if node.name not in _CYPHER_AGGREGATES:
        raise _unsupported(
            "Only supported Cypher aggregate functions may appear in aggregate expressions",
            field="return.item",
            value=source_text,
            line=span_line,
            column=span_column,
        )
    if len(node.args) == 0:
        raise _unsupported(
            "Cypher aggregate functions require an argument or '*'",
            field="return.item",
            value=source_text,
            line=span_line,
            column=span_column,
        )
    if len(node.args) != 1:
        raise _unsupported(
            "Cypher local aggregate lowering currently supports single-argument aggregate functions only",
            field="return.item",
            value=source_text,
            line=span_line,
            column=span_column,
        )
    arg = node.args[0]
    if isinstance(arg, Wildcard):
        if node.name != "count":
            raise _unsupported(
                "Only count(*) supports '*' in the local Cypher aggregate subset",
                field="return.item",
                value=source_text,
                line=span_line,
                column=span_column,
            )
        if node.distinct:
            raise _unsupported(
                "count(DISTINCT *) is not supported in the local Cypher aggregate subset",
                field="return.item",
                value=source_text,
                line=span_line,
                column=span_column,
            )
        expr_text: Optional[str] = None
    else:
        expr_text = _render_expr_node(arg)
        if expr_text == "":
            raise _unsupported(
                "Cypher aggregate functions require a non-empty argument",
                field="return.item",
                value=source_text,
                line=span_line,
                column=span_column,
            )

    return _AggregateSpec(
        source_text=source_text,
        output_name=output_name,
        func=node.name,
        expr_text=expr_text,
        distinct=node.distinct,
        span_line=span_line,
        span_column=span_column,
    )


def _post_aggregate_expr_plan(
    item: ReturnItem,
    *,
    params: Optional[Mapping[str, Any]] = None,
    alias_targets: Optional[Mapping[str, ASTObject]] = None,
) -> Optional[Tuple[List[_AggregateSpec], _PostAggregateExprPlan]]:
    if item.expression.text == "*":
        return None

    node = _parse_row_expr(
        item.expression.text,
        params=params,
        alias_targets=alias_targets,
        field="return.item",
        line=item.span.line,
        column=item.span.column,
    )

    temp_names: Set[str] = set()
    aggregate_specs: List[_AggregateSpec] = []
    aggregate_temp_by_source: Dict[str, str] = {}

    def _rewrite(node_in: ExprNode) -> ExprNode:
        if isinstance(node_in, FunctionCall) and node_in.name in _CYPHER_AGGREGATES:
            source_text = _render_expr_node(node_in)
            temp_name = aggregate_temp_by_source.get(source_text)
            if temp_name is None:
                temp_name = _fresh_temp_name(temp_names, "__cypher_postagg__")
                aggregate_temp_by_source[source_text] = temp_name
                aggregate_specs.append(
                    _aggregate_spec_from_function_call(
                        node_in,
                        source_text=source_text,
                        output_name=temp_name,
                        span_line=item.span.line,
                        span_column=item.span.column,
                    )
                )
            return Identifier(temp_name)
        return _rebuild_expr_node(node_in, rewrite=_rewrite, error_context="aggregate rewrite")

    rewritten = _rewrite(node)
    if not aggregate_specs:
        return None
    integer_aggregate_names = {
        spec.output_name
        for spec in aggregate_specs
        if spec.func == "count"
    }
    rewritten = _rewrite_cypher_integer_division_ast(
        rewritten,
        integer_identifiers=integer_aggregate_names,
    )

    return aggregate_specs, _PostAggregateExprPlan(
        output_name=item.alias or item.expression.text,
        expr=ExpressionText(text=_render_expr_node(rewritten), span=item.expression.span),
        span_line=item.span.line,
        span_column=item.span.column,
    )


def _empty_aggregate_row(aggregate_specs: Sequence[_AggregateSpec]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for agg_spec in aggregate_specs:
        if agg_spec.func in {"count"}:
            out[agg_spec.output_name] = 0
        elif agg_spec.func == "collect":
            out[agg_spec.output_name] = []
        else:
            out[agg_spec.output_name] = None
    return out


def _active_match_alias_for_stage(
    *,
    unwinds: Sequence[UnwindClause],
    clause: ReturnClause,
    order_by_clause: Optional[OrderByClause],
    alias_targets: Mapping[str, ASTObject],
    allowed_match_aliases: Optional[AbstractSet[str]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    if not alias_targets:
        return None

    expr_texts: List[Tuple[str, int, int, str]] = []
    for unwind_clause in unwinds:
        expr_texts.append(
            (
                unwind_clause.expression.text,
                unwind_clause.span.line,
                unwind_clause.span.column,
                "unwind",
            )
        )
    for return_item in clause.items:
        expr_texts.append(
            (
                return_item.expression.text,
                return_item.span.line,
                return_item.span.column,
                clause.kind,
            )
        )
    if order_by_clause is not None:
        for order_item in order_by_clause.items:
            expr_texts.append(
                (
                    order_item.expression.text,
                    order_item.span.line,
                    order_item.span.column,
                    "order_by",
                )
            )

    referenced: Set[str] = set()
    aggregate_only_referenced: Set[str] = set()
    for expr_text, line, column, field_name in expr_texts:
        if expr_text == "*":
            continue
        non_aggregate_aliases, aggregate_aliases = _expr_match_alias_usage(
            expr_text,
            alias_targets=alias_targets,
            params=params,
            field=field_name,
            line=line,
            column=column,
        )
        referenced.update(non_aggregate_aliases)
        aggregate_only_referenced.update(aggregate_aliases - non_aggregate_aliases)

    if len(referenced) > 1:
        if allowed_match_aliases is not None and referenced <= allowed_match_aliases:
            return _first_allowed_alias(alias_targets, allowed_match_aliases, referenced)
        raise _unsupported(
            "Cypher row lowering currently supports one MATCH source alias at a time",
            field="return",
            value=sorted(referenced),
            line=clause.span.line,
            column=clause.span.column,
        )
    if len(referenced) == 1:
        return next(iter(referenced))
    if len(aggregate_only_referenced) > 1:
        if allowed_match_aliases is not None and aggregate_only_referenced <= allowed_match_aliases:
            return _first_allowed_alias(alias_targets, allowed_match_aliases, aggregate_only_referenced)
        raise _unsupported(
            "Cypher row lowering currently supports one MATCH source alias at a time",
            field="return",
            value=sorted(aggregate_only_referenced),
            line=clause.span.line,
            column=clause.span.column,
        )
    if len(aggregate_only_referenced) == 1:
        return next(iter(aggregate_only_referenced))
    if allowed_match_aliases is not None:
        allowed_alias = _first_allowed_alias(alias_targets, allowed_match_aliases, set())
        if allowed_alias is not None:
            return allowed_alias
    return next(iter(alias_targets))


def _validate_aggregate_expr_scope(
    agg_spec: _AggregateSpec,
    *,
    alias_targets: Mapping[str, ASTObject],
    active_match_alias: Optional[str],
    allowed_match_aliases: Optional[AbstractSet[str]] = None,
    unwind_aliases: Iterable[str],
    params: Optional[Mapping[str, Any]],
    field: str,
) -> None:
    if agg_spec.expr_text is None:
        return
    _validate_row_expr_scope(
        agg_spec.expr_text,
        alias_targets=alias_targets,
        active_match_alias=active_match_alias,
        allowed_match_aliases=allowed_match_aliases,
        unwind_aliases=unwind_aliases,
        params=params,
        field=field,
        line=agg_spec.span_line,
        column=agg_spec.span_column,
    )


def _is_multiplicity_sensitive_aggregate(agg_spec: _AggregateSpec) -> bool:
    if agg_spec.func in {"sum", "avg"}:
        return True
    if agg_spec.func in {"count", "collect"}:
        return not agg_spec.distinct
    return False


def _match_relationship_count(clause: MatchClause) -> int:
    return sum(1 for element in _match_pattern_elements(clause) if isinstance(element, RelationshipPattern))


def _reject_unsound_relationship_multiplicity_aggregates_common(
    *,
    aggregate_specs: Sequence[_AggregateSpec],
    alias_targets: Mapping[str, ASTObject],
    active_match_alias: Optional[str],
    relationship_count: int,
    field: str,
    value: Any,
    line: int,
    column: int,
) -> None:
    if not any(_is_multiplicity_sensitive_aggregate(spec) for spec in aggregate_specs):
        return
    if relationship_count == 0:
        return
    if active_match_alias is not None:
        active_target = alias_targets.get(active_match_alias)
        if isinstance(active_target, ASTEdge) and relationship_count == 1:
            return
    raise _unsupported(
        "This Cypher aggregate would need repeated MATCH rows from a relationship pattern, but the current runtime collapses those rows before aggregation. Queries like MATCH (a)-[r]->(b) RETURN a, count(*) are not supported yet.",
        field=field,
        value=value,
        line=line,
        column=column,
    )


def _reject_unsound_relationship_multiplicity_aggregates(
    query: CypherQuery,
    *,
    aggregate_specs: Sequence[_AggregateSpec],
    alias_targets: Mapping[str, ASTObject],
    active_match_alias: Optional[str],
) -> None:
    merged_match = _merged_match_clause(query)
    if merged_match is None:
        return
    _reject_unsound_relationship_multiplicity_aggregates_common(
        aggregate_specs=aggregate_specs,
        alias_targets=alias_targets,
        active_match_alias=active_match_alias,
        relationship_count=_match_relationship_count(merged_match),
        field=query.return_.kind,
        value=[item.expression.text for item in query.return_.items],
        line=query.return_.span.line,
        column=query.return_.span.column,
    )


def _return_references_optional_only_alias(
    query: CypherQuery,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]] = None,
) -> bool:
    if not any(clause.optional for clause in query.matches) or not any(not clause.optional for clause in query.matches):
        return False
    bound_aliases = {
        alias
        for clause in query.matches
        if not clause.optional
        for alias in _match_clause_aliases_raw(clause)
    }
    optional_only_aliases = {
        alias
        for clause in query.matches
        if clause.optional
        for alias in _match_clause_aliases_raw(clause)
        if alias not in bound_aliases
    }
    if not optional_only_aliases:
        return False
    for item in query.return_.items:
        if item.expression.text == "*":
            return True
        referenced = _expr_match_aliases(
            item.expression.text,
            alias_targets=alias_targets,
            params=params,
            field=query.return_.kind,
            line=item.span.line,
            column=item.span.column,
        )
        if referenced & optional_only_aliases:
            return True
    return False


def _where_uses_optional_only_label_predicate(query: CypherQuery) -> bool:
    if query.where is None or not query.where.predicates:
        return False
    bound_aliases = {
        alias
        for clause in query.matches
        if not clause.optional
        for alias in _match_clause_aliases_raw(clause)
    }
    optional_only_aliases = {
        alias
        for clause in query.matches
        if clause.optional
        for alias in _match_clause_aliases_raw(clause)
        if alias not in bound_aliases
    }
    if not optional_only_aliases:
        return False
    for predicate in query.where.predicates:
        if not isinstance(predicate, WherePredicate):
            continue
        left = predicate.left
        if isinstance(left, LabelRef) and left.alias in optional_only_aliases:
            return True
    return False


def _active_match_alias(
    query: CypherQuery,
    *,
    alias_targets: Mapping[str, ASTObject],
    allowed_match_aliases: Optional[AbstractSet[str]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    return _active_match_alias_for_stage(
        unwinds=query.unwinds,
        clause=query.return_,
        order_by_clause=query.order_by,
        alias_targets=alias_targets,
        allowed_match_aliases=allowed_match_aliases,
        params=params,
    )

def _resolve_literal(
    value: CypherLiteral,
    *,
    params: Optional[Mapping[str, Any]],
    field: str,
) -> Any:
    if isinstance(value, ParameterRef):
        if params is None or value.name not in params:
            raise GFQLValidationError(
                ErrorCode.E105,
                f"Missing Cypher parameter '${value.name}'",
                field=field,
                value=value.name,
                suggestion=f"Pass params={{'{value.name}': ...}} when compiling or executing the query.",
                line=value.span.line,
                column=value.span.column,
                language="cypher",
            )
        return params[value.name]
    return value


def _resolve_page_value(
    value: Any,
    *,
    params: Optional[Mapping[str, Any]],
    field: str,
    line: int,
    column: int,
) -> int:
    if isinstance(value, ParameterRef):
        resolved = _resolve_literal(value, params=params, field=field)
    elif isinstance(value, ExpressionText):
        resolved = _eval_constant_int_expr(value, params=params, field=field, line=line, column=column)
    else:
        resolved = value
    if isinstance(resolved, bool) or not isinstance(resolved, int):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher SKIP/LIMIT values must be integers",
            field=field,
            value=resolved,
            suggestion="Use a non-negative integer literal or parameter.",
            line=line,
            column=column,
            language="cypher",
        )
    if resolved < 0:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher SKIP/LIMIT values must be non-negative",
            field=field,
            value=resolved,
            suggestion="Use 0 or a positive integer.",
            line=line,
            column=column,
            language="cypher",
        )
    return resolved


def _eval_constant_int_expr(
    expr: ExpressionText,
    *,
    params: Optional[Mapping[str, Any]],
    field: str,
    line: int,
    column: int,
) -> int:
    node = _parse_row_expr(
        expr.text,
        params=params,
        field=field,
        line=line,
        column=column,
    )
    if collect_identifiers(node):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher SKIP/LIMIT expressions cannot depend on row variables in the local compiler",
            field=field,
            value=expr.text,
            suggestion="Use a constant expression or parameter for SKIP/LIMIT.",
            line=line,
            column=column,
            language="cypher",
        )

    def _eval(node: ExprNode) -> Any:
        if isinstance(node, ExprLiteral):
            return node.value
        if isinstance(node, UnaryOp):
            value = _eval(node.operand)
            if node.op == "+":
                return +value
            if node.op == "-":
                return -value
            if node.op == "not":
                return not value
            raise GFQLValidationError(
                ErrorCode.E108,
                "Unsupported unary function in Cypher SKIP/LIMIT expression",
                field=field,
                value=expr.text,
                line=line,
                column=column,
                language="cypher",
            )
        if isinstance(node, BinaryOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "*":
                return left * right
            if node.op == "/":
                if (
                    isinstance(left, int)
                    and not isinstance(left, bool)
                    and isinstance(right, int)
                    and not isinstance(right, bool)
                ):
                    return int(left / right)
                return left / right
            if node.op == "%":
                return left % right
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher SKIP/LIMIT expressions currently support arithmetic constants only",
                field=field,
                value=expr.text,
                line=line,
                column=column,
                language="cypher",
            )
        if isinstance(node, FunctionCall):
            args = tuple(_eval(arg) for arg in node.args)
            if node.name == "ceil" and len(args) == 1:
                return math.ceil(args[0])
            if node.name == "floor" and len(args) == 1:
                return math.floor(args[0])
            if node.name == "abs" and len(args) == 1:
                return abs(args[0])
            if node.name == "tointeger" and len(args) == 1:
                return int(args[0])
            if node.name == "tofloat" and len(args) == 1:
                return float(args[0])
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher SKIP/LIMIT expression uses an unsupported function",
                field=field,
                value=expr.text,
                suggestion="Use arithmetic constants, parameters, ceil/floor/abs, or toInteger/toFloat.",
                line=line,
                column=column,
                language="cypher",
            )
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher SKIP/LIMIT expression uses an unsupported construct",
            field=field,
            value=expr.text,
            suggestion="Use arithmetic constants or supported numeric functions.",
            line=line,
            column=column,
            language="cypher",
        )

    resolved = _eval(node)
    if isinstance(resolved, bool) or not isinstance(resolved, int):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher SKIP/LIMIT values must resolve to integers",
            field=field,
            value=resolved,
            suggestion="Use a constant integer expression or parameter.",
            line=line,
            column=column,
            language="cypher",
        )
    return resolved


def _filter_dict_from_entries(
    *,
    discriminator_key: Optional[str],
    discriminator_values: Sequence[str],
    properties: Sequence[PropertyEntry],
    params: Optional[Mapping[str, Any]],
    field_prefix: str,
    line: int,
    column: int,
) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    if discriminator_key is not None and len(discriminator_values) == 1:
        out[discriminator_key] = discriminator_values[0]
    elif discriminator_key is not None and len(discriminator_values) > 1:
        out[discriminator_key] = is_in(list(discriminator_values))
    for entry in properties:
        if isinstance(entry.value, ExpressionText):
            continue
        out[entry.key] = _resolve_literal(
            entry.value,
            params=params,
            field=f"{field_prefix}.{entry.key}",
        )
    return out or None


def _render_dynamic_property_entry_predicate(
    *,
    alias: str,
    key: str,
    expr: ExpressionText,
) -> str:
    return f"{alias}.{key} = ({expr.text})"


def _is_hidden_reentry_property(property_name: str) -> bool:
    return property_name.startswith("__cypher_reentry_") or property_name.startswith("__gfql_hidden_")


def _dynamic_property_entry_constraints(
    clause: MatchClause,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> Tuple[List[WhereComparison], List[str]]:
    where_out: List[WhereComparison] = []
    row_predicates: List[str] = []
    force_row_predicates = _cartesian_node_only_patterns(clause) is not None
    for element in _match_pattern_elements(clause):
        alias = getattr(element, "variable", None)
        properties = getattr(element, "properties", ())
        for entry in properties:
            if not isinstance(entry.value, ExpressionText):
                continue
            if alias is None:
                raise _unsupported(
                    "Cypher expression-valued pattern properties currently require an explicit alias",
                    field="match",
                    value=entry.key,
                    line=entry.span.line,
                    column=entry.span.column,
                )
            node = _parse_row_expr(
                entry.value.text,
                params=params,
                alias_targets=alias_targets,
                field=f"match.{alias}.{entry.key}",
                line=entry.value.span.line,
                column=entry.value.span.column,
            )
            if isinstance(node, PropertyAccessExpr) and isinstance(node.value, Identifier):
                source_alias = node.value.name
                if (
                    source_alias in alias_targets
                    and not _is_hidden_reentry_property(node.property)
                    and not force_row_predicates
                ):
                    where_out.append(compare(col(alias, entry.key), "==", col(source_alias, node.property)))
                    continue
            row_predicates.append(
                _render_dynamic_property_entry_predicate(alias=alias, key=entry.key, expr=entry.value)
            )
    return where_out, row_predicates


def _lower_node(node: NodePattern, *, params: Optional[Mapping[str, Any]]) -> ASTNode:
    filter_dict = _filter_dict_from_entries(
        discriminator_key=None,
        discriminator_values=(),
        properties=node.properties,
        params=params,
        field_prefix=f"node.{node.variable or '_'}",
        line=node.span.line,
        column=node.span.column,
    )
    if node.labels:
        filter_dict = dict(filter_dict or {})
        for label in node.labels:
            filter_dict[f"label__{label}"] = True
    return ASTNode(filter_dict=filter_dict, name=node.variable)


def _lower_relationship(
    relationship: RelationshipPattern,
    *,
    params: Optional[Mapping[str, Any]],
    prune_to_endpoints: bool = False,
) -> ASTObject:
    edge_match = _filter_dict_from_entries(
        discriminator_key="type",
        discriminator_values=relationship.types,
        properties=relationship.properties,
        params=params,
        field_prefix=f"edge.{relationship.variable or '_'}",
        line=relationship.span.line,
        column=relationship.span.column,
    )
    hops = (
        None
        if (
            relationship.min_hops is not None
            or relationship.max_hops is not None
            or relationship.to_fixed_point
        )
        else 1
    )
    if relationship.direction == "forward":
        return cast(
            ASTObject,
            e_forward(
                edge_match=edge_match,
                hops=hops,
                min_hops=relationship.min_hops,
                max_hops=relationship.max_hops,
                to_fixed_point=relationship.to_fixed_point,
                name=relationship.variable,
                prune_to_endpoints=prune_to_endpoints,
            ),
        )
    if relationship.direction == "reverse":
        return cast(
            ASTObject,
            e_reverse(
                edge_match=edge_match,
                hops=hops,
                min_hops=relationship.min_hops,
                max_hops=relationship.max_hops,
                to_fixed_point=relationship.to_fixed_point,
                name=relationship.variable,
                prune_to_endpoints=prune_to_endpoints,
            ),
        )
    return cast(
        ASTObject,
        e_undirected(
            edge_match=edge_match,
            hops=hops,
            min_hops=relationship.min_hops,
            max_hops=relationship.max_hops,
            to_fixed_point=relationship.to_fixed_point,
            name=relationship.variable,
            prune_to_endpoints=prune_to_endpoints,
        ),
    )


def _pattern_line_column(pattern: Sequence[PatternElement], clause: MatchClause) -> Tuple[int, int]:
    if len(pattern) == 0:
        return clause.span.line, clause.span.column
    return pattern[0].span.line, pattern[0].span.column


def _reverse_relationship_pattern(relationship: RelationshipPattern) -> RelationshipPattern:
    direction: Literal["forward", "reverse", "undirected"]
    if relationship.direction == "forward":
        direction = "reverse"
    elif relationship.direction == "reverse":
        direction = "forward"
    else:
        direction = "undirected"
    return RelationshipPattern(
        direction=direction,
        variable=relationship.variable,
        types=relationship.types,
        properties=relationship.properties,
        span=relationship.span,
        min_hops=relationship.min_hops,
        max_hops=relationship.max_hops,
        to_fixed_point=relationship.to_fixed_point,
    )


def _reverse_pattern(pattern: Sequence[PatternElement]) -> Tuple[PatternElement, ...]:
    out: List[PatternElement] = []
    for element in reversed(pattern):
        if isinstance(element, NodePattern):
            out.append(element)
        else:
            out.append(_reverse_relationship_pattern(element))
    return tuple(out)


def _merge_property_entries(
    left: Sequence[PropertyEntry],
    right: Sequence[PropertyEntry],
    *,
    line: int,
    column: int,
) -> Tuple[PropertyEntry, ...]:
    out: Dict[str, PropertyEntry] = {entry.key: entry for entry in left}
    for entry in right:
        if entry.key in out and out[entry.key].value != entry.value:
            raise _unsupported(
                "Comma-connected Cypher MATCH patterns have conflicting node filters",
                field="match",
                value=entry.key,
                line=line,
                column=column,
            )
        out[entry.key] = entry
    return tuple(out.values())


def _merge_node_patterns(
    left: NodePattern,
    right: NodePattern,
) -> NodePattern:
    line = right.span.line
    column = right.span.column
    if left.variable is None or right.variable is None or left.variable != right.variable:
        raise _unsupported(
            "Comma-connected Cypher MATCH patterns must share an explicit endpoint alias",
            field="match",
            value={"left": left.variable, "right": right.variable},
            line=line,
            column=column,
        )
    labels = tuple(dict.fromkeys(left.labels + right.labels))
    properties = _merge_property_entries(left.properties, right.properties, line=line, column=column)
    return NodePattern(
        variable=left.variable,
        labels=labels,
        properties=properties,
        span=left.span,
    )


def _node_join_alias(node: NodePattern) -> Optional[str]:
    return node.variable


def _node_can_join(left: NodePattern, right: NodePattern) -> bool:
    left_alias = _node_join_alias(left)
    right_alias = _node_join_alias(right)
    return left_alias is not None and right_alias is not None and left_alias == right_alias


def _stitch_patterns(
    left: Sequence[PatternElement],
    right: Sequence[PatternElement],
    *,
    clause: MatchClause,
) -> Tuple[PatternElement, ...]:
    if len(left) == 0:
        return tuple(right)
    if len(right) == 0:
        return tuple(left)
    left_start = cast(NodePattern, left[0])
    left_end = cast(NodePattern, left[-1])
    right_start = cast(NodePattern, right[0])
    right_end = cast(NodePattern, right[-1])

    if _node_can_join(left_end, right_start):
        merged = _merge_node_patterns(left_end, right_start)
        return tuple(left[:-1]) + (merged,) + tuple(right[1:])
    if _node_can_join(left_end, right_end):
        reversed_right = _reverse_pattern(right)
        merged = _merge_node_patterns(left_end, cast(NodePattern, reversed_right[0]))
        return tuple(left[:-1]) + (merged,) + tuple(reversed_right[1:])
    if _node_can_join(left_start, right_end):
        merged = _merge_node_patterns(cast(NodePattern, right_end), left_start)
        return tuple(right[:-1]) + (merged,) + tuple(left[1:])
    if _node_can_join(left_start, right_start):
        reversed_right = _reverse_pattern(right)
        merged = _merge_node_patterns(cast(NodePattern, reversed_right[-1]), left_start)
        return tuple(reversed_right[:-1]) + (merged,) + tuple(left[1:])

    line, column = _pattern_line_column(right, clause)
    raise _unsupported(
        "Comma-separated Cypher MATCH patterns are only supported for a single linear connected path with shared endpoint aliases",
        field="match",
        value=None,
        line=line,
        column=column,
    )


def _normalized_match_pattern(clause: MatchClause) -> Tuple[PatternElement, ...]:
    if len(clause.patterns) == 0:
        return ()
    pattern = clause.patterns[0]
    for next_pattern in clause.patterns[1:]:
        pattern = _stitch_patterns(pattern, next_pattern, clause=clause)
    return tuple(pattern)


def _cartesian_node_only_patterns(clause: MatchClause) -> Optional[Tuple[NodePattern, ...]]:
    if len(clause.patterns) <= 1:
        return None
    out: List[NodePattern] = []
    for pattern in clause.patterns:
        if len(pattern) != 1 or not isinstance(pattern[0], NodePattern):
            return None
        out.append(pattern[0])
    return tuple(out)


def _match_pattern_elements(clause: MatchClause) -> Tuple[PatternElement, ...]:
    cartesian_nodes = _cartesian_node_only_patterns(clause)
    if cartesian_nodes is not None:
        return cast(Tuple[PatternElement, ...], cartesian_nodes)
    return _normalized_match_pattern(clause)


def _pattern_node_aliases(pattern: Sequence[PatternElement]) -> Set[str]:
    return {
        cast(str, element.variable)
        for element in pattern
        if isinstance(element, NodePattern) and element.variable is not None
    }


def _is_node_connected_multi_pattern_clause(clause: MatchClause) -> bool:
    if len(clause.patterns) <= 1:
        return False
    if _cartesian_node_only_patterns(clause) is not None:
        return False
    pattern_aliases = [_pattern_node_aliases(pattern) for pattern in clause.patterns]
    if any(len(alias_set) == 0 for alias_set in pattern_aliases):
        return False
    seen = {0}
    frontier = [0]
    while frontier:
        idx = frontier.pop()
        for other_idx, other_aliases in enumerate(pattern_aliases):
            if other_idx in seen:
                continue
            if pattern_aliases[idx] & other_aliases:
                seen.add(other_idx)
                frontier.append(other_idx)
    return len(seen) == len(pattern_aliases)


def _connected_join_alias_targets(
    clause: MatchClause,
    *,
    params: Optional[Mapping[str, Any]],
) -> Mapping[str, ASTObject]:
    combined_alias_targets: Dict[str, ASTObject] = {}
    for pattern in clause.patterns:
        single_clause = replace(clause, patterns=(pattern,), pattern_aliases=(None,))
        ops, _ = _lower_match_clause_with_alias_equalities(single_clause, params=params)
        for alias, target in _alias_target(ops).items():
            if alias not in combined_alias_targets:
                combined_alias_targets[alias] = target
    return combined_alias_targets


def _clause_has_connected_join_whole_row_alias_passthrough(
    clause: ReturnClause,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> bool:
    for item in clause.items:
        agg_spec = _aggregate_spec(item, params=params, alias_targets=alias_targets)
        if agg_spec is None and item.expression.text in alias_targets:
            return True
    return False

def _is_connected_multi_pattern_clause(clause: MatchClause) -> bool:
    if clause.optional or len(clause.patterns) <= 1:
        return False
    if _cartesian_node_only_patterns(clause) is not None:
        return False
    try:
        _normalized_match_pattern(clause)
        return False
    except GFQLValidationError as exc:
        if "shared endpoint aliases" not in str(exc):
            return False
    pattern_aliases = [_pattern_node_aliases(pattern) for pattern in clause.patterns]
    if any(len(alias_set) == 0 for alias_set in pattern_aliases):
        return False
    seen = {0}
    frontier = [0]
    while frontier:
        idx = frontier.pop()
        for other_idx, other_aliases in enumerate(pattern_aliases):
            if other_idx in seen:
                continue
            if pattern_aliases[idx] & other_aliases:
                seen.add(other_idx)
                frontier.append(other_idx)
    return len(seen) == len(pattern_aliases)


def _query_requires_general_lowering_for_connected_join(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]],
) -> bool:
    if len(query.matches) != 1 or not _is_node_connected_multi_pattern_clause(query.matches[0]):
        return False
    alias_targets = _connected_join_alias_targets(query.matches[0], params=params)
    return any(
        _clause_has_connected_join_whole_row_alias_passthrough(
            stage.clause,
            alias_targets=alias_targets,
            params=params,
        )
        for stage in query.with_stages
    ) or _clause_has_connected_join_whole_row_alias_passthrough(
        query.return_,
        alias_targets=alias_targets,
        params=params,
    )


def _query_has_aggregate_stage(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]],
) -> bool:
    clauses = [stage.clause for stage in query.with_stages] + [query.return_]
    for clause in clauses:
        for item in clause.items:
            if _aggregate_spec(item, params=params, alias_targets={}) is not None:
                return True
            if _post_aggregate_expr_plan(item, params=params, alias_targets={}) is not None:
                return True
    return False


def _binding_row_aliases_for_match(
    clause: Optional[MatchClause],
    *,
    alias_targets: Mapping[str, ASTObject],
) -> Set[str]:
    if clause is None or _cartesian_node_only_patterns(clause) is None:
        return set()
    if len(alias_targets) <= 1:
        return set()
    if not all(isinstance(target, ASTNode) for target in alias_targets.values()):
        return set()
    return set(alias_targets.keys())


def _binding_row_aliases_for_row_where(
    row_where: Optional[ExpressionText],
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> Set[str]:
    if row_where is None:
        return set()
    referenced = _expr_match_aliases(
        row_where.text,
        alias_targets=alias_targets,
        params=params,
        field="where",
        line=row_where.span.line,
        column=row_where.span.column,
    )
    if len(referenced) <= 1:
        return set()
    return set(alias_targets.keys())


def _first_allowed_alias(
    alias_targets: Mapping[str, ASTObject],
    allowed_match_aliases: AbstractSet[str],
    fallback_aliases: AbstractSet[str],
) -> Optional[str]:
    candidates = fallback_aliases & allowed_match_aliases
    if not candidates:
        candidates = set(allowed_match_aliases)
    for alias_name in alias_targets:
        if alias_name in candidates:
            return alias_name
    return next(iter(candidates)) if candidates else None


def _lower_match_clause_with_alias_equalities(
    clause: MatchClause,
    *,
    params: Optional[Mapping[str, Any]],
) -> Tuple[List[ASTObject], List[WhereComparison]]:
    out: List[ASTObject] = []
    where_out: List[WhereComparison] = []
    seen_alias_kinds: Dict[str, Literal["node", "edge"]] = {}
    seen_alias_ops: Dict[str, ASTObject] = {}
    pattern = list(_match_pattern_elements(clause))
    existing_aliases: Set[str] = {
        cast(str, element.variable)
        for element in pattern
        if getattr(element, "variable", None) is not None
    }

    # Pre-compute last relationship index for prune_to_endpoints
    rel_indices = [i for i, el in enumerate(pattern) if isinstance(el, RelationshipPattern)]
    last_rel_idx = rel_indices[-1] if rel_indices else -1

    for idx, element in enumerate(pattern):
        if isinstance(element, NodePattern):
            alias = element.variable
            if alias is None or alias not in seen_alias_kinds:
                lowered = _lower_node(element, params=params)
                if alias is not None:
                    seen_alias_kinds[alias] = "node"
                    seen_alias_ops[alias] = lowered
                out.append(lowered)
                continue
            if seen_alias_kinds[alias] != "node":
                raise _unsupported(
                    "Cypher duplicate aliases across node/relationship kinds are not supported",
                    field="alias",
                    value=alias,
                    line=element.span.line,
                    column=element.span.column,
                )
            previous = seen_alias_ops[alias]
            if not isinstance(previous, ASTNode):
                raise _unsupported(
                    "Cypher duplicate aliases across node/relationship kinds are not supported",
                    field="alias",
                    value=alias,
                    line=element.span.line,
                    column=element.span.column,
                )
            rewritten_alias = _fresh_temp_name(existing_aliases, f"__cypher_aliasdup_{alias}")
            previous._name = rewritten_alias
            seen_alias_kinds[rewritten_alias] = "node"
            seen_alias_ops[rewritten_alias] = previous
            lowered = _lower_node(element, params=params)
            seen_alias_ops[alias] = lowered
            out.append(lowered)
            where_out.append(
                compare(col(alias, "id"), "==", col(rewritten_alias, "id"))
            )
            continue

        alias = element.variable
        if alias is not None and alias in seen_alias_kinds:
            raise _unsupported(
                "Cypher duplicate relationship aliases are not yet supported",
                field="alias",
                value=alias,
                line=element.span.line,
                column=element.span.column,
            )
        if alias is not None:
            seen_alias_kinds[alias] = "edge"
        is_varlen = _is_variable_length_relationship_pattern(element)
        lowered_edge = _lower_relationship(
            element,
            params=params,
            prune_to_endpoints=is_varlen and idx < last_rel_idx,
        )
        if alias is not None:
            seen_alias_ops[alias] = lowered_edge
        out.append(lowered_edge)

    return out, where_out


def _duplicate_node_aliases(match: Optional[MatchClause]) -> Set[str]:
    if match is None:
        return set()
    counts: Dict[str, int] = {}
    for element in _match_pattern_elements(match):
        if isinstance(element, NodePattern) and element.variable is not None:
            counts[element.variable] = counts.get(element.variable, 0) + 1
    return {alias for alias, count in counts.items() if count > 1}


def _seed_node_bindings(matches: Sequence[MatchClause]) -> Dict[str, NodePattern]:
    out: Dict[str, NodePattern] = {}
    for clause in matches:
        for pattern in clause.patterns:
            if len(pattern) != 1 or not isinstance(pattern[0], NodePattern):
                raise _unsupported(
                    "Only node-only pre-binding MATCH clauses are supported before the final connected MATCH in this phase",
                    field="match",
                    value=None,
                    line=clause.span.line,
                    column=clause.span.column,
                )
            node = pattern[0]
            if node.variable is None:
                raise _unsupported(
                    "Pre-binding MATCH clauses currently require explicit node aliases",
                    field="match",
                    value=None,
                    line=node.span.line,
                    column=node.span.column,
                )
            existing = out.get(node.variable)
            out[node.variable] = node if existing is None else _merge_node_patterns(existing, node)
    return out


def _apply_seed_node_bindings(
    clause: MatchClause,
    *,
    seed_bindings: Mapping[str, NodePattern],
) -> MatchClause:
    if not seed_bindings:
        return clause
    pattern = list(_normalized_match_pattern(clause))
    seen: Set[str] = set()
    updated: List[PatternElement] = []
    for element in pattern:
        alias = element.variable if isinstance(element, NodePattern) else None
        if isinstance(element, NodePattern) and alias is not None and alias in seed_bindings:
            updated.append(_merge_node_patterns(seed_bindings[alias], element))
            seen.add(alias)
        else:
            updated.append(element)
    unused = set(seed_bindings) - seen
    if unused:
        raise _unsupported(
            "Earlier MATCH-bound aliases must participate in the final connected MATCH pattern in this phase",
            field="match",
            value=sorted(unused),
            line=clause.span.line,
            column=clause.span.column,
        )
    return MatchClause(
        patterns=(tuple(updated),),
        span=clause.span,
        optional=clause.optional,
        pattern_aliases=clause.pattern_aliases[-1:] if clause.pattern_aliases else (),
    )


def _merged_match_clause(query: CypherQuery) -> Optional[MatchClause]:
    if not query.matches:
        return None
    if len(query.matches) == 1:
        return query.matches[0]
    seed_bindings = _seed_node_bindings(query.matches[:-1])
    return _apply_seed_node_bindings(query.matches[-1], seed_bindings=seed_bindings)


def _match_clause_aliases(clause: MatchClause) -> Set[str]:
    return {
        cast(str, element.variable)
        for element in _match_pattern_elements(clause)
        if getattr(element, "variable", None) is not None
    }


def _match_clause_aliases_raw(clause: MatchClause) -> Set[str]:
    return {
        cast(str, element.variable)
        for pattern in clause.patterns
        for element in pattern
        if getattr(element, "variable", None) is not None
    }


def _single_node_seed_alias(clause: MatchClause) -> Optional[str]:
    if clause.optional or len(clause.patterns) != 1:
        return None
    pattern = clause.patterns[0]
    if len(pattern) != 1 or not isinstance(pattern[0], NodePattern):
        return None
    return pattern[0].variable


def _alias_target(ops: Sequence[ASTObject]) -> Dict[str, ASTObject]:
    targets: Dict[str, ASTObject] = {}
    for op in ops:
        alias = getattr(op, "_name", None)
        if alias is None:
            continue
        if alias in targets:
            raise GFQLValidationError(
                ErrorCode.E108,
                f"Duplicate Cypher alias '{alias}' is not yet supported",
                field="alias",
                value=alias,
                suggestion="Use unique aliases per node/relationship pattern.",
                language="cypher",
            )
        targets[alias] = op
    return targets


def _target_filter_dict(target: ASTObject) -> Optional[Dict[str, Any]]:
    if isinstance(target, ASTNode):
        return cast(Optional[Dict[str, Any]], target.filter_dict)
    if isinstance(target, ASTEdge):
        return cast(Optional[Dict[str, Any]], target.edge_match)
    raise _unsupported(
        "Only node and edge aliases are supported in Cypher MATCH lowering",
        field="alias",
        value=type(target).__name__,
        line=1,
        column=1,
    )


def _set_target_filter_dict(target: ASTObject, filter_dict: Dict[str, Any]) -> None:
    if isinstance(target, ASTNode):
        target.filter_dict = filter_dict
        return
    if isinstance(target, ASTEdge):
        target.edge_match = filter_dict
        return
    raise _unsupported(
        "Only node and edge aliases are supported in Cypher MATCH lowering",
        field="alias",
        value=type(target).__name__,
        line=1,
        column=1,
    )


def _predicate_value(op: str, value: Any) -> Any:
    if op == "==":
        return value
    if op == "!=":
        return ne(value)
    if op == "<":
        return lt(value)
    if op == "<=":
        return le(value)
    if op == ">":
        return gt(value)
    if op == ">=":
        return ge(value)
    if op == "is_null":
        return isna()
    if op == "is_not_null":
        return notna()
    if op == "contains":
        return never_match() if value is None else str_contains(str(value), regex=False, na=False)
    if op == "starts_with":
        return never_match() if value is None else startswith(str(value), na=False)
    if op == "ends_with":
        return never_match() if value is None else endswith(str(value), na=False)
    raise ValueError(f"Unsupported predicate op: {op}")


def _alias_table(target: ASTObject, *, alias: str, line: int, column: int) -> str:
    if isinstance(target, ASTNode):
        return "nodes"
    if isinstance(target, ASTEdge):
        return "edges"
    raise _unsupported(
        "Only node and edge aliases are supported in Cypher row projections",
        field="return.alias",
        value=alias,
        line=line,
        column=column,
    )


def _split_qualified_name(expr: str, *, line: int, column: int) -> Tuple[str, Optional[str]]:
    parts = expr.split(".")
    if len(parts) == 1:
        return parts[0], None
    if len(parts) == 2:
        return parts[0], parts[1]
    raise _unsupported(
        "Only simple aliases and alias.property expressions are supported in Cypher RETURN/ORDER BY",
        field="expression",
        value=expr,
        line=line,
        column=column,
    )


def _qualified_ref_from_node(
    node: ExprNode,
    *,
    field: str,
    value: str,
    line: int,
    column: int,
) -> Tuple[str, Optional[str]]:
    if isinstance(node, Identifier):
        return _split_qualified_name(node.name, line=line, column=column)
    if isinstance(node, PropertyAccessExpr) and isinstance(node.value, Identifier):
        alias_name, prop = _split_qualified_name(node.value.name, line=line, column=column)
        if prop is not None:
            raise _unsupported(
                "Only simple aliases and alias.property expressions are supported in Cypher RETURN/ORDER BY",
                field=field,
                value=value,
                line=line,
                column=column,
            )
        return alias_name, node.property
    raise _unsupported(
        "Only simple aliases and alias.property expressions are supported in Cypher RETURN/ORDER BY",
        field=field,
        value=value,
        line=line,
        column=column,
    )


def _projection_ref_from_expr(
    expr: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]] = None,
    field: str,
    line: int,
    column: int,
) -> Tuple[str, Optional[str]]:
    node = _parse_row_expr(expr, params=params, field=field, line=line, column=column)
    if isinstance(node, (Identifier, PropertyAccessExpr)):
        return _qualified_ref_from_node(
            node,
            field=field,
            value=expr,
            line=line,
            column=column,
        )
    if isinstance(node, FunctionCall) and node.name == "type":
        if len(node.args) != 1 or not isinstance(node.args[0], Identifier):
            raise _unsupported(
                "type(...) is only supported with a single relationship alias argument in this phase",
                field=field,
                value=expr,
                line=line,
                column=column,
            )
        alias_name, prop = _split_qualified_name(node.args[0].name, line=line, column=column)
        if prop is not None:
            raise _unsupported(
                "type(...) only supports bare relationship aliases in this phase",
                field=field,
                value=expr,
                line=line,
                column=column,
            )
        target = alias_targets.get(alias_name)
        if not isinstance(target, ASTEdge):
            raise _unsupported(
                "type(...) is only supported for relationship aliases in this phase",
                field=field,
                value=expr,
                line=line,
                column=column,
            )
        return alias_name, "type"
    raise _unsupported(
        "Only simple aliases, alias.property, and type(rel_alias) expressions are supported in Cypher RETURN/ORDER BY",
        field=field,
        value=expr,
        line=line,
        column=column,
    )


def _raise_if_invalid_graph_projection_expr(
    expr: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]] = None,
    field: str,
    line: int,
    column: int,
) -> None:
    node = _parse_row_expr(expr, params=params, field=field, line=line, column=column)
    if not isinstance(node, FunctionCall) or node.name != "type" or len(node.args) != 1:
        return
    arg = node.args[0]
    if not isinstance(arg, Identifier):
        return
    alias_name, prop = _split_qualified_name(arg.name, line=line, column=column)
    if prop is not None:
        return
    target = alias_targets.get(alias_name)
    if target is None:
        return
    if not isinstance(target, ASTEdge):
        raise _unsupported(
            "type(...) is only supported for relationship aliases in this phase",
            field=field,
            value=expr,
            line=line,
            column=column,
        )


def _reject_duplicate_alias_row_refs(
    query: CypherQuery,
    *,
    alias_targets: Mapping[str, ASTObject],
    duplicated_aliases: Set[str],
    params: Optional[Mapping[str, Any]],
) -> None:
    if not duplicated_aliases:
        return

    def _check(expr_text: str, *, field: str, line: int, column: int) -> None:
        refs = _expr_match_aliases(
            expr_text,
            alias_targets=alias_targets,
            params=params,
            field=field,
            line=line,
            column=column,
        )
        if refs & duplicated_aliases:
            raise _unsupported(
                "Cypher row projection from repeated MATCH aliases is not yet supported in the local compiler",
                field=field,
                value=expr_text,
                line=line,
                column=column,
            )

    for unwind_clause in query.unwinds:
        _check(
            unwind_clause.expression.text,
            field="unwind",
            line=unwind_clause.span.line,
            column=unwind_clause.span.column,
        )
    for item in query.return_.items:
        _check(
            item.expression.text,
            field=query.return_.kind,
            line=item.span.line,
            column=item.span.column,
        )
    if query.order_by is not None:
        for order_item in query.order_by.items:
            _check(
                order_item.expression.text,
                field="order_by",
                line=order_item.span.line,
                column=order_item.span.column,
            )


def _build_projection_plan(
    clause: ReturnClause,
    *,
    alias_targets: Dict[str, ASTObject],
    active_alias: Optional[str] = None,
    projected_columns: Optional[Mapping[str, _StageColumnBinding]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> _ProjectionPlan:
    source_alias: Optional[str] = None
    all_source_aliases: Optional[Set[str]] = None
    whole_row_output_names: List[str] = []
    whole_row_sources: Dict[str, str] = {}
    projection_items: List[Tuple[str, Any]] = []
    projection_columns: List[ResultProjectionColumn] = []
    available_columns: Set[str] = set()
    projected_property_outputs: Dict[str, str] = {}
    output_to_source_property: Dict[str, str] = {}
    output_to_expr_source: Dict[str, str] = {}

    for item in clause.items:
        binding: Optional[_StageColumnBinding] = None
        projected_expr_binding = False
        simple_ref = True
        if item.expression.text == "*":
            if len(alias_targets) > 1:
                raise _unsupported(
                    "Cypher RETURN * currently requires a single MATCH alias in the local compiler",
                    field=f"{clause.kind}.items",
                    value=item.expression.text,
                    line=item.span.line,
                    column=item.span.column,
                )
            if active_alias is not None:
                alias_name = active_alias
            elif len(alias_targets) == 1:
                alias_name = next(iter(alias_targets.keys()))
            else:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Cypher RETURN/WITH * currently requires a single active alias in the local compiler",
                    field=f"{clause.kind}.items",
                    value=item.expression.text,
                    suggestion="Use a single MATCH alias or project explicit columns.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
            prop = None
            binding = None
            projected_expr_binding = False
        else:
            try:
                alias_name, prop = _projection_ref_from_expr(
                    item.expression.text,
                    alias_targets=alias_targets,
                    params=params,
                    field=f"{clause.kind}.items",
                    line=item.span.line,
                    column=item.span.column,
                )
            except GFQLValidationError:
                _raise_if_invalid_graph_projection_expr(
                    item.expression.text,
                    alias_targets=alias_targets,
                    params=params,
                    field=f"{clause.kind}.items",
                    line=item.span.line,
                    column=item.span.column,
                )
                simple_ref = False
                aliases = sorted(
                    _expr_match_aliases(
                        item.expression.text,
                        alias_targets=alias_targets,
                        params=params,
                        field=f"{clause.kind}.items",
                        line=item.span.line,
                        column=item.span.column,
                    )
                )
                if len(aliases) == 1:
                    alias_name = aliases[0]
                    prop = None
                elif len(aliases) == 0:
                    if active_alias is not None:
                        alias_name = active_alias
                        prop = None
                    elif len(alias_targets) == 1:
                        alias_name = next(iter(alias_targets.keys()))
                        prop = None
                    else:
                        raise
                else:
                    raise
        if alias_name not in alias_targets and projected_columns is not None:
            binding = projected_columns.get(alias_name)
            if binding is not None:
                if prop is None and binding.kind == "property":
                    if active_alias is None:
                        raise _unsupported(
                            "Projected Cypher column references require an active MATCH alias in this phase",
                            field=f"{clause.kind}.items",
                            value=item.expression.text,
                            line=item.span.line,
                            column=item.span.column,
                        )
                    alias_name = active_alias
                    simple_ref = True
                    prop = binding.source_name
                else:
                    simple_ref = False
                    projected_expr_binding = True
        if alias_name not in alias_targets:
            if projected_expr_binding:
                alias_name = active_alias or alias_name
            else:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    f"Unknown Cypher alias '{alias_name}' in {clause.kind.upper()} clause",
                    field=f"{clause.kind}.alias",
                    value=alias_name,
                    suggestion="Reference an alias declared in the MATCH pattern.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
        if source_alias is None:
            source_alias = alias_name
        elif source_alias != alias_name:
            if all_source_aliases is None:
                all_source_aliases = {source_alias}
            all_source_aliases.add(alias_name)
        if item.expression.text == "*":
            output_name = item.alias or alias_name
            if output_name in available_columns or output_name in whole_row_output_names:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Duplicate Cypher projection names are not yet supported in local lowering",
                    field=f"{clause.kind}.items",
                    value=output_name,
                    suggestion="Use distinct output names in RETURN/WITH.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
            whole_row_output_names.append(output_name)
            whole_row_sources[output_name] = alias_name
            continue
        if simple_ref and prop is None:
            output_name = item.alias or alias_name
            if output_name in available_columns or output_name in whole_row_output_names:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Duplicate Cypher projection names are not yet supported in local lowering",
                    field=f"{clause.kind}.items",
                    value=output_name,
                    suggestion="Use distinct output names in RETURN/WITH.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
            whole_row_output_names.append(output_name)
            whole_row_sources[output_name] = alias_name
            continue
        output_name = item.alias or item.expression.text
        if output_name in available_columns or output_name in whole_row_output_names:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Duplicate Cypher projection names are not yet supported in local lowering",
                field=f"{clause.kind}.items",
                value=output_name,
                suggestion="Use distinct output names in RETURN/WITH.",
                line=item.span.line,
                column=item.span.column,
                language="cypher",
            )
        row_expr = _rewrite_expr_to_projected_sources(
            item.expression,
            projected_columns=projected_columns,
            params=params,
            alias_targets=alias_targets,
            field=f"{clause.kind}.items",
        )
        runtime_expr = (
            f"{alias_name}.{prop}"
            if simple_ref and prop is not None
            else (
                binding.source_name
                if binding is not None and binding.kind == "expr" and prop is None
                else _row_expr_arg(
                    row_expr,
                    params=params,
                    alias_targets=alias_targets,
                    field=f"{clause.kind}.items",
                )
            )
        )
        projection_items.append((output_name, runtime_expr))
        available_columns.add(output_name)
        if simple_ref and prop is not None:
            projected_property_outputs.setdefault(prop, output_name)
            source_property_name = prop if alias_name == source_alias else f"{alias_name}.{prop}"
            output_to_source_property[output_name] = source_property_name
            projection_columns.append(
                ResultProjectionColumn(
                    output_name=output_name,
                    kind="property",
                    source_name=source_property_name,
                )
            )
        else:
            if isinstance(runtime_expr, str):
                output_to_expr_source[output_name] = runtime_expr
            projection_columns.append(
                ResultProjectionColumn(
                    output_name=output_name,
                    kind="expr",
                    source_name=runtime_expr if isinstance(runtime_expr, str) else item.expression.text,
                )
            )

    if source_alias is None:
        raise GFQLValidationError(
            ErrorCode.E108,
            f"Cypher {clause.kind.upper()} clause must project at least one supported expression",
            field=clause.kind,
            value=None,
            suggestion="Project a match alias or alias.property expression.",
            line=clause.span.line,
            column=clause.span.column,
            language="cypher",
        )
    if whole_row_output_names:
        available_columns = set()
    table = _alias_table(
        alias_targets[source_alias],
        alias=source_alias,
        line=clause.span.line,
        column=clause.span.column,
    )
    return _ProjectionPlan(
        source_alias=source_alias,
        table=table,
        whole_row_output_names=whole_row_output_names,
        whole_row_sources=whole_row_sources,
        clause_kind=clause.kind,
        projection_items=projection_items,
        projection_columns=projection_columns,
        available_columns=available_columns,
        projected_property_outputs=projected_property_outputs,
        output_to_source_property=output_to_source_property,
        output_to_expr_source=output_to_expr_source,
        all_source_aliases=all_source_aliases,
    )


def _can_lower_multi_alias_projection_bindings(
    plan: _ProjectionPlan,
    *,
    alias_targets: Mapping[str, ASTObject],
) -> bool:
    all_refs = (plan.all_source_aliases or set()) | {plan.source_alias}
    all_are_edges = all(isinstance(alias_targets.get(alias_name), ASTEdge) for alias_name in all_refs)
    has_non_scalar = bool(plan.whole_row_output_names) or bool(plan.output_to_expr_source)
    if not has_non_scalar:
        return not all_are_edges
    if len(plan.whole_row_output_names) != 1:
        return False
    if isinstance(alias_targets.get(plan.source_alias), ASTEdge):
        return False
    simple_qualified_ref = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
    for source_name in plan.output_to_expr_source.values():
        match = simple_qualified_ref.fullmatch(source_name)
        if match is None:
            return False
        alias_name = source_name.split(".", 1)[0]
        if isinstance(alias_targets.get(alias_name), ASTEdge):
            return False
    return True


def _result_projection_plan(
    plan: _ProjectionPlan,
    *,
    alias_targets: Mapping[str, ASTObject],
) -> Optional[ResultProjectionPlan]:
    if not plan.whole_row_output_names:
        return None
    columns: List[ResultProjectionColumn] = []
    for output_name in plan.whole_row_output_names:
        columns.append(
            ResultProjectionColumn(
                output_name=output_name,
                kind="whole_row",
                source_name=plan.whole_row_sources.get(output_name, plan.source_alias),
            )
        )
    columns.extend(plan.projection_columns)
    return ResultProjectionPlan(
        alias=plan.source_alias,
        table=cast(Literal["nodes", "edges"], plan.table),
        columns=tuple(columns),
        exclude_columns=tuple(sorted(alias_targets.keys())),
    )


def _empty_optional_projection_row(plan: _ProjectionPlan) -> Dict[str, Any]:
    out: Dict[str, Any] = {name: None for name in plan.whole_row_output_names}
    for column in plan.projection_columns:
        out[column.output_name] = None
    return out


def _optional_null_fill_plan(
    query: CypherQuery,
    *,
    lowered: LoweredCypherMatch,
    alias_targets: Mapping[str, ASTObject],
    plan: _ProjectionPlan,
    params: Optional[Mapping[str, Any]],
) -> Optional[OptionalNullFillPlan]:
    if (
        len(query.matches) != 2
        or query.matches[0].optional
        or not query.matches[1].optional
        or query.where is not None
        or query.with_stages
        or query.unwinds
        or query.order_by is not None
        or query.skip is not None
        or query.limit is not None
    ):
        return None

    seed_alias = _single_node_seed_alias(query.matches[0])
    if seed_alias is None:
        return None

    optional_aliases = _match_clause_aliases(query.matches[1]) - {seed_alias}
    if not optional_aliases:
        return None

    referenced: Set[str] = set()
    for item in query.return_.items:
        referenced.update(
            _expr_match_aliases(
                item.expression.text,
                alias_targets=alias_targets,
                params=params,
                field=query.return_.kind,
                line=item.span.line,
                column=item.span.column,
            )
        )
    if not referenced or not referenced <= optional_aliases:
        return None

    alignment_output_name = "__cypher_optional_seed__"
    alignment_table = cast(
        Literal["nodes", "edges"],
        _alias_table(
            alias_targets[seed_alias],
            alias=seed_alias,
            line=query.return_.span.line,
            column=query.return_.span.column,
        ),
    )
    alignment_plan = _ProjectionPlan(
        source_alias=seed_alias,
        table=alignment_table,
        whole_row_output_names=[alignment_output_name],
        whole_row_sources={alignment_output_name: seed_alias},
        clause_kind="return",
        projection_items=[],
        projection_columns=[],
        available_columns=set(),
        projected_property_outputs={},
        output_to_source_property={},
        output_to_expr_source={},
    )
    alignment_projection = _result_projection_plan(alignment_plan, alias_targets=alias_targets)
    if alignment_projection is None:
        raise _unsupported(
            "Cypher OPTIONAL MATCH null-row alignment could not construct a seed-row projection",
            field="return",
            value=[item.expression.text for item in query.return_.items],
            line=query.return_.span.line,
            column=query.return_.span.column,
        )

    return OptionalNullFillPlan(
        base_chain=Chain(lower_match_clause(query.matches[0], params=params)),
        null_row=_empty_optional_projection_row(plan),
        alignment_chain=Chain(_lower_projection_chain(query, lowered, params=params, plan=alignment_plan)),
        alignment_projection=alignment_projection,
        alignment_output_name=alignment_output_name,
    )


def _optional_projection_row_guard_plan(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]],
) -> Optional[OptionalProjectionRowGuardPlan]:
    if (
        len(query.matches) != 2
        or query.matches[0].optional
        or not query.matches[1].optional
        or query.where is not None
        or query.with_stages
        or query.unwinds
        or query.order_by is not None
        or query.skip is not None
        or query.limit is not None
    ):
        return None
    base_clause = query.matches[0]
    if len(base_clause.patterns) == 1:
        return OptionalProjectionRowGuardPlan(
            base_chains=(Chain(lower_match_clause(base_clause, params=params)),)
        )
    base_chains: List[Chain] = []
    for idx, pattern in enumerate(base_clause.patterns):
        if len(pattern) != 1 or not isinstance(pattern[0], NodePattern):
            return None
        pattern_clause = MatchClause(
            patterns=(pattern,),
            span=base_clause.span,
            optional=False,
            pattern_aliases=((base_clause.pattern_aliases[idx] if idx < len(base_clause.pattern_aliases) else None),),
        )
        base_chains.append(Chain(lower_match_clause(pattern_clause, params=params)))
    return OptionalProjectionRowGuardPlan(base_chains=tuple(base_chains))


def _plan_with_visible_projected_columns(
    plan: _ProjectionPlan,
    projected_columns: Mapping[str, _StageColumnBinding],
) -> _ProjectionPlan:
    if not projected_columns:
        return plan
    output_to_source_property = dict(plan.output_to_source_property)
    output_to_expr_source = dict(plan.output_to_expr_source)
    available_columns = set(plan.available_columns)
    for name, binding in projected_columns.items():
        available_columns.add(name)
        if name in output_to_source_property or name in output_to_expr_source:
            continue
        if binding.kind == "property":
            output_to_source_property[name] = binding.source_name
        else:
            output_to_expr_source[name] = binding.source_name
    return replace(
        plan,
        available_columns=available_columns,
        output_to_source_property=output_to_source_property,
        output_to_expr_source=output_to_expr_source,
    )


def _projection_output_names(plan: _ProjectionPlan) -> Set[str]:
    return {name for name, _ in plan.projection_items} | set(plan.whole_row_output_names)


def _lower_order_by_clause(
    clause: OrderByClause,
    *,
    plan: _ProjectionPlan,
    alias_targets: Optional[Mapping[str, ASTObject]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> ASTObject:
    keys: List[Tuple[str, str]] = []
    projection_output_names = _projection_output_names(plan)
    for item in clause.items:
        try:
            alias_name, prop = _projection_ref_from_expr(
                item.expression.text,
                alias_targets=alias_targets or {},
                field="order_by",
                line=item.span.line,
                column=item.span.column,
            )
            simple_ref = True
        except GFQLValidationError:
            simple_ref = False
            alias_name = None
            prop = None

        if simple_ref:
            assert alias_name is not None
            if prop is None:
                if alias_name in plan.output_to_source_property:
                    order_key = (
                        plan.output_to_source_property[alias_name]
                        if plan.whole_row_output_names or alias_name not in projection_output_names
                        else alias_name
                    )
                elif alias_name in plan.output_to_expr_source:
                    order_key = (
                        plan.output_to_expr_source[alias_name]
                        if plan.whole_row_output_names or alias_name not in projection_output_names
                        else alias_name
                    )
                elif alias_name in plan.whole_row_output_names or alias_name == plan.source_alias:
                    raise GFQLValidationError(
                        ErrorCode.E108,
                        "ORDER BY whole-row entity outputs is not yet supported in local Cypher lowering",
                        field="order_by",
                        value=item.expression.text,
                        suggestion="Order by a projected property or source property instead.",
                        line=item.span.line,
                        column=item.span.column,
                        language="cypher",
                    )
                else:
                    order_key = alias_name
            else:
                source_alias = (
                    plan.whole_row_sources.get(alias_name, plan.source_alias)
                    if alias_name in plan.whole_row_output_names
                    else alias_name
                )
                if source_alias != plan.source_alias and alias_name not in plan.whole_row_output_names:
                    raise GFQLValidationError(
                        ErrorCode.E108,
                        "ORDER BY expressions must reference the active RETURN/WITH source alias",
                        field="order_by",
                        value=item.expression.text,
                        suggestion=f"Use columns from alias '{plan.source_alias}' only in this phase.",
                        line=item.span.line,
                        column=item.span.column,
                        language="cypher",
                    )
                order_key = (
                    f"{source_alias}.{prop}"
                    if plan.whole_row_output_names
                    else plan.projected_property_outputs.get(prop, prop)
                )
        else:
            if alias_targets is None:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "ORDER BY expressions must reference projected columns or simple source properties in this phase",
                    field="order_by",
                    value=item.expression.text,
                    suggestion="Order by a projected output column, source property, or supported expression on the active alias.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
            referenced = _expr_match_aliases(
                item.expression.text,
                alias_targets=alias_targets,
                params=params,
                field="order_by",
                line=item.span.line,
                column=item.span.column,
            )
            if any(root != plan.source_alias for root in referenced):
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "ORDER BY expressions must reference the active RETURN/WITH source alias",
                    field="order_by",
                    value=item.expression.text,
                    suggestion=f"Use columns from alias '{plan.source_alias}' only in this phase.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
            order_key = (
                cast(
                    str,
                    _row_expr_arg(
                        item.expression,
                        params=params,
                        alias_targets=alias_targets,
                        field="order_by",
                    ),
                )
                if plan.whole_row_output_names
                else _rewrite_alias_properties_to_outputs(
                    item.expression.text,
                    source_alias=plan.source_alias,
                    property_outputs=plan.projected_property_outputs,
                    params=params,
                    alias_targets=alias_targets,
                    field="order_by",
                    line=item.span.line,
                    column=item.span.column,
                )
            )
        if not plan.whole_row_output_names and order_key not in plan.available_columns:
            node = _parse_row_expr(
                order_key,
                params=params,
                field="order_by",
                line=item.span.line,
                column=item.span.column,
            )
            missing = sorted(ident for ident in collect_identifiers(node) if ident not in plan.available_columns)
            if missing:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "ORDER BY column must exist after RETURN/WITH projection in this phase",
                    field="order_by",
                    value=item.expression.text,
                    suggestion="Order by a projected output column or alias.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
        keys.append((order_key, item.direction))
    return order_by(keys)


def _lower_order_by_outputs(
    clause: OrderByClause,
    *,
    available_columns: Set[str],
    expr_to_output: Mapping[str, str],
    params: Optional[Mapping[str, Any]] = None,
) -> ASTObject:
    keys: List[Tuple[str, str]] = []
    for item in clause.items:
        order_key = expr_to_output.get(item.expression.text)
        if order_key is None:
            order_key = _rewrite_expr_to_output_names(
                item.expression.text,
                replacements=expr_to_output,
                params=params,
                field="order_by",
                line=item.span.line,
                column=item.span.column,
            )
        if order_key not in available_columns:
            node = _parse_row_expr(
                order_key,
                params=params,
                field="order_by",
                line=item.span.line,
                column=item.span.column,
            )
            if any(ident not in available_columns for ident in collect_identifiers(node)):
                raise _unsupported(
                    "ORDER BY column must exist after RETURN/WITH projection in this phase",
                    field="order_by",
                    value=item.expression.text,
                    line=item.span.line,
                    column=item.span.column,
                )
        keys.append((order_key, item.direction))
    return order_by(keys)


def _append_page_ops_values(
    row_steps: List[ASTObject],
    *,
    skip_clause: Optional[Any],
    limit_clause: Optional[Any],
    params: Optional[Mapping[str, Any]],
) -> None:
    if skip_clause is not None:
        row_steps.append(
            skip(
                _resolve_page_value(
                    skip_clause.value,
                    params=params,
                    field="skip",
                    line=skip_clause.span.line,
                    column=skip_clause.span.column,
                )
            )
        )
    if limit_clause is not None:
        row_steps.append(
            limit(
                _resolve_page_value(
                    limit_clause.value,
                    params=params,
                    field="limit",
                    line=limit_clause.span.line,
                    column=limit_clause.span.column,
                )
            )
        )


def _append_page_ops(
    row_steps: List[ASTObject],
    *,
    query: CypherQuery,
    params: Optional[Mapping[str, Any]],
) -> None:
    _append_page_ops_values(
        row_steps,
        skip_clause=query.skip,
        limit_clause=query.limit,
        params=params,
    )


def _append_match_row_where(
    row_steps: List[ASTObject],
    *,
    lowered: LoweredCypherMatch,
    alias_targets: Mapping[str, ASTObject],
    active_alias: Optional[str],
    allowed_match_aliases: Optional[AbstractSet[str]],
    params: Optional[Mapping[str, Any]],
) -> None:
    expr = lowered.row_where
    if expr is None:
        return
    _validate_row_expr_scope(
        expr.text,
        alias_targets=alias_targets,
        active_match_alias=active_alias,
        allowed_match_aliases=allowed_match_aliases,
        unwind_aliases=set(),
        params=params,
        field="where",
        line=expr.span.line,
        column=expr.span.column,
    )
    row_steps.append(
        where_rows(
            expr=_row_expr_arg(
                expr,
                params=params,
                alias_targets=alias_targets,
                field="where",
            )
        )
    )


def _lower_projection_chain(
    query: CypherQuery,
    lowered: LoweredCypherMatch,
    *,
    params: Optional[Mapping[str, Any]],
    plan: Optional[_ProjectionPlan] = None,
) -> List[ASTObject]:
    alias_targets = _alias_target(lowered.query)
    binding_row_aliases = _binding_row_aliases_for_match(query.match, alias_targets=alias_targets)
    binding_row_aliases.update(
        _binding_row_aliases_for_row_where(
            lowered.row_where,
            alias_targets=alias_targets,
            params=params,
        )
    )
    if plan is None:
        try:
            active = _active_match_alias(
                query,
                alias_targets=alias_targets,
                allowed_match_aliases=binding_row_aliases or None,
                params=params,
            )
        except GFQLValidationError as exc:
            active = next(iter(alias_targets)) if alias_targets else None
            _multi_alias_exc: Optional[GFQLValidationError] = exc
        else:
            _multi_alias_exc = None
        plan = _build_projection_plan(
            query.return_,
            alias_targets=alias_targets,
            active_alias=active,
            params=params,
        )
        if _multi_alias_exc is not None:
            if not _can_lower_multi_alias_projection_bindings(plan, alias_targets=alias_targets):
                raise _multi_alias_exc

    allowed_match_aliases = (
        ({plan.source_alias} | plan.all_source_aliases | binding_row_aliases)
        if plan.all_source_aliases is not None
        else binding_row_aliases
    )
    if plan.all_source_aliases is not None or binding_row_aliases:
        row_steps: List[ASTObject] = [rows(binding_ops=serialize_binding_ops(lowered.query))]
    else:
        row_steps = [rows(table=plan.table, source=plan.source_alias)]
    _append_match_row_where(
        row_steps,
        lowered=lowered,
        alias_targets=alias_targets,
        active_alias=plan.source_alias,
        allowed_match_aliases=allowed_match_aliases or None,
        params=params,
    )

    if not plan.whole_row_output_names:
        projection_fn = with_ if plan.clause_kind == "with" else return_
        row_steps.append(projection_fn(plan.projection_items))

    if query.return_.distinct:
        row_steps.append(distinct())
    if query.order_by is not None:
        row_steps.append(
            _lower_order_by_clause(
                query.order_by,
                plan=plan,
                alias_targets=alias_targets,
                params=params,
            )
        )
    _append_page_ops(row_steps, query=query, params=params)
    if binding_row_aliases:
        return row_steps
    return lowered.query + row_steps


def _build_initial_row_scope(
    query: CypherQuery,
    lowered: LoweredCypherMatch,
    *,
    stage_clause: ReturnClause,
    stage_order_by: Optional[OrderByClause],
    params: Optional[Mapping[str, Any]],
) -> Tuple[List[ASTObject], _StageScope]:
    alias_targets = _alias_target(lowered.query) if query.match is not None else {}
    merged_match = _merged_match_clause(query)
    binding_row_aliases = _binding_row_aliases_for_match(query.match, alias_targets=alias_targets)
    # For connected multi-pattern MATCH (not cartesian), enable binding-row
    # aliases when the first WITH/RETURN stage projects multiple whole-row
    # match aliases and all targets are nodes.  This allows multi-alias WITH
    # projections to flow through the bindings-row path (#880).
    if (
        not binding_row_aliases
        and query.match is not None
        and len(alias_targets) > 1
        and all(isinstance(t, ASTNode) for t in alias_targets.values())
    ):
        # Check if the stage clause references 2+ whole-row match aliases
        whole_row_refs = {
            item.expression.text
            for item in stage_clause.items
            if item.expression.text in alias_targets
        }
        if len(whole_row_refs) > 1:
            binding_row_aliases = set(alias_targets.keys())
    binding_row_aliases.update(
        _binding_row_aliases_for_row_where(
            lowered.row_where,
            alias_targets=alias_targets,
            params=params,
        )
    )
    active_match_alias = _active_match_alias_for_stage(
        unwinds=query.unwinds,
        clause=stage_clause,
        order_by_clause=stage_order_by,
        alias_targets=alias_targets,
        allowed_match_aliases=binding_row_aliases or None,
        params=params,
    )
    seed_rows = query.match is None

    if active_match_alias is None:
        row_steps: List[ASTObject] = [rows(table="nodes")]
        table: Optional[Literal["nodes", "edges"]] = None
        scope_mode: Literal["match_alias", "row_columns"] = "row_columns"
    elif binding_row_aliases:
        row_steps = [rows(binding_ops=serialize_binding_ops(lowered.query))]
        table = None
        scope_mode = "match_alias"
    else:
        table = cast(
            Literal["nodes", "edges"],
            _alias_table(
                alias_targets[active_match_alias],
                alias=active_match_alias,
                line=stage_clause.span.line,
                column=stage_clause.span.column,
            ),
        )
        row_steps = [rows(table=table, source=active_match_alias)]
        scope_mode = "match_alias"
    _append_match_row_where(
        row_steps,
        lowered=lowered,
        alias_targets=alias_targets,
        active_alias=active_match_alias,
        allowed_match_aliases=binding_row_aliases or None,
        params=params,
    )

    unwind_aliases: Set[str] = set()
    for unwind_clause in query.unwinds:
        if unwind_clause.alias in alias_targets or unwind_clause.alias in unwind_aliases:
            raise _unsupported(
                "Cypher UNWIND alias collides with an existing alias in this local subset",
                field="unwind.alias",
                value=unwind_clause.alias,
                line=unwind_clause.span.line,
                column=unwind_clause.span.column,
            )
        _validate_row_expr_scope(
            unwind_clause.expression.text,
            alias_targets=alias_targets,
            active_match_alias=active_match_alias,
            allowed_match_aliases=binding_row_aliases or None,
            unwind_aliases=unwind_aliases,
            params=params,
            field="unwind",
            line=unwind_clause.span.line,
            column=unwind_clause.span.column,
        )
        row_steps.append(
            unwind(
                _row_expr_arg(
                    unwind_clause.expression,
                    params=params,
                    alias_targets=alias_targets,
                    field="unwind",
                ),
                as_=unwind_clause.alias,
            )
        )
        unwind_aliases.add(unwind_clause.alias)

    return row_steps, _StageScope(
            mode=scope_mode,
            alias_targets=dict(alias_targets),
            active_alias=active_match_alias,
            allowed_match_aliases=set(binding_row_aliases),
            row_columns=set(unwind_aliases),
            projected_columns={},
            table=table,
            seed_rows=seed_rows,
            relationship_count=_match_relationship_count(merged_match) if merged_match is not None else 0,
        )


def _lower_match_alias_stage(
    stage: ProjectionStage,
    *,
    scope: _StageScope,
    params: Optional[Mapping[str, Any]],
    final_stage: bool,
) -> Tuple[List[ASTObject], _StageScope, Optional[ResultProjectionPlan]]:
    _validate_with_projection_aliasing(stage)
    if scope.active_alias is None:
        raise _unsupported(
            "Cypher row expressions cannot reference MATCH aliases when no row source is active",
            field=stage.clause.kind,
            value=[item.expression.text for item in stage.clause.items],
            line=stage.clause.span.line,
            column=stage.clause.span.column,
        )
    if scope.row_columns:
        raise _unsupported(
            "Cypher WITH pipelines that mix MATCH aliases with UNWIND-produced row columns are not yet supported",
            field=stage.clause.kind,
            value=sorted(scope.row_columns),
            line=stage.clause.span.line,
            column=stage.clause.span.column,
        )

    aggregate_specs: List[_AggregateSpec] = []
    non_aggregate_items: List[ReturnItem] = []
    for item in stage.clause.items:
        agg_spec = _aggregate_spec(item, params=params, alias_targets=scope.alias_targets)
        if agg_spec is None:
            non_aggregate_items.append(item)
        else:
            aggregate_specs.append(agg_spec)

    if aggregate_specs:
        return _lower_match_alias_aggregate_stage(
            stage,
            scope=scope,
            params=params,
            aggregate_specs=aggregate_specs,
            non_aggregate_items=non_aggregate_items,
            final_stage=final_stage,
        )

    plan = _build_projection_plan(
        stage.clause,
        alias_targets=scope.alias_targets,
        active_alias=scope.active_alias,
        projected_columns=scope.projected_columns,
        params=params,
    )
    row_steps: List[ASTObject] = []
    if not plan.whole_row_output_names:
        projection_fn = with_ if stage.clause.kind == "with" else return_
        row_steps.append(projection_fn(plan.projection_items))
    elif scope.allowed_match_aliases and plan.projection_items:
        # Mixed case: whole-row aliases + scalar items on a bindings-row table.
        # Use extend mode to add scalar columns without dropping the existing
        # alias-prefixed bindings columns (#880).
        row_steps.append(with_(plan.projection_items, extend=True))
    if stage.clause.distinct:
        row_steps.append(distinct())
    if stage.where is not None:
        where_expr = stage.where
        if not plan.whole_row_output_names:
            where_expr = ExpressionText(
                text=_rewrite_alias_properties_to_outputs(
                    stage.where.text,
                    source_alias=plan.source_alias,
                    property_outputs=plan.projected_property_outputs,
                    params=params,
                    alias_targets=scope.alias_targets,
                    field="with.where",
                    line=stage.where.span.line,
                    column=stage.where.span.column,
                ),
                span=stage.where.span,
            )
        _validate_row_expr_scope(
            stage.where.text,
            alias_targets=scope.alias_targets,
            active_match_alias=scope.active_alias,
            allowed_match_aliases=scope.allowed_match_aliases or None,
            unwind_aliases=set(),
            params=params,
            field="with.where",
            line=stage.where.span.line,
            column=stage.where.span.column,
        )
        row_steps.append(
            where_rows(
                expr=_row_expr_arg(
                    where_expr,
                    params=params,
                    alias_targets=scope.alias_targets,
                    field="with.where",
                )
            )
        )
    if stage.order_by is not None:
        order_plan = _plan_with_visible_projected_columns(plan, scope.projected_columns)
        row_steps.append(
            _lower_order_by_clause(
                stage.order_by,
                plan=order_plan,
                alias_targets=scope.alias_targets,
                params=params,
            )
        )
    _append_page_ops_values(
        row_steps,
        skip_clause=stage.skip,
        limit_clause=stage.limit,
        params=params,
    )

    if plan.whole_row_output_names:
        next_projected_columns = {
            column.output_name: _StageColumnBinding(
                kind=cast(Literal["property", "expr"], column.kind),
                source_name=cast(str, column.source_name),
            )
            for column in plan.projection_columns
            if column.source_name is not None and column.kind in {"property", "expr"}
        }
        next_scope = _StageScope(
            mode="match_alias",
            alias_targets=scope.alias_targets,
            active_alias=plan.source_alias,
            allowed_match_aliases=set(scope.allowed_match_aliases),
            row_columns=set(),
            projected_columns=next_projected_columns,
            table=cast(Optional[Literal["nodes", "edges"]], plan.table),
            seed_rows=scope.seed_rows,
            relationship_count=scope.relationship_count,
        )
    else:
        next_scope = _StageScope(
            mode="row_columns",
            alias_targets={},
            active_alias=None,
            allowed_match_aliases=set(),
            row_columns=set(plan.available_columns),
            projected_columns={},
            table=None,
            seed_rows=scope.seed_rows,
            relationship_count=scope.relationship_count,
        )

    result_projection = _result_projection_plan(plan, alias_targets=scope.alias_targets) if final_stage else None
    return row_steps, next_scope, result_projection


def _lower_match_alias_aggregate_stage(
    stage: ProjectionStage,
    *,
    scope: _StageScope,
    params: Optional[Mapping[str, Any]],
    aggregate_specs: Sequence[_AggregateSpec],
    non_aggregate_items: Sequence[ReturnItem],
    final_stage: bool,
) -> Tuple[List[ASTObject], _StageScope, Optional[ResultProjectionPlan]]:
    active_alias = scope.active_alias
    if active_alias is None:
        raise _unsupported(
            "Cypher aggregate row lowering requires an active MATCH alias",
            field=stage.clause.kind,
            value=[item.expression.text for item in stage.clause.items],
            line=stage.clause.span.line,
            column=stage.clause.span.column,
        )

    projection_fn = with_ if stage.clause.kind == "with" else return_
    # On the bindings-row path (allowed_match_aliases populated), the row
    # table preserves per-row multiplicity from the MATCH, so relationship-
    # count aggregation guards do not apply (#880).
    if not scope.allowed_match_aliases:
        _reject_unsound_relationship_multiplicity_aggregates_common(
            aggregate_specs=aggregate_specs,
            alias_targets=scope.alias_targets,
            active_match_alias=active_alias,
            relationship_count=scope.relationship_count,
            field=stage.clause.kind,
            value=[item.expression.text for item in stage.clause.items],
            line=stage.clause.span.line,
            column=stage.clause.span.column,
        )
    pre_items: List[Tuple[str, Any]] = []
    key_names: List[str] = []
    temp_names: Set[str] = set()
    available_columns: Set[str] = set()
    expr_to_output: Dict[str, str] = {}
    projected_property_outputs: Dict[str, str] = {}
    output_to_source_property: Dict[str, str] = {}
    hidden_group_key_names: Set[str] = set()

    for item in non_aggregate_items:
        output_name = item.alias or item.expression.text
        if output_name in available_columns:
            raise _unsupported(
                "Duplicate Cypher projection names are not yet supported in local lowering",
                field=stage.clause.kind,
                value=output_name,
                line=item.span.line,
                column=item.span.column,
            )
        if item.expression.text in scope.alias_targets:
            alias_name = item.expression.text
            if alias_name != active_alias:
                raise _unsupported(
                    "Cypher aggregate whole-row grouping currently supports the active MATCH alias only",
                    field=stage.clause.kind,
                    value=item.expression.text,
                    line=item.span.line,
                    column=item.span.column,
                )
            hidden_key_name = _fresh_temp_name(temp_names, "__cypher_group_key__")
            pre_items.append(
                (
                    hidden_key_name,
                    _whole_row_group_key_expr(
                        alias_name,
                        alias_targets=scope.alias_targets,
                        field=stage.clause.kind,
                        line=item.span.line,
                        column=item.span.column,
                    ),
                )
            )
            pre_items.append(
                (
                    output_name,
                    _whole_row_group_entity_expr(
                        alias_name,
                        alias_targets=scope.alias_targets,
                        field=stage.clause.kind,
                        line=item.span.line,
                        column=item.span.column,
                    ),
                )
            )
            key_names.extend([hidden_key_name, output_name])
            hidden_group_key_names.add(hidden_key_name)
            available_columns.add(output_name)
            _add_output_mapping(
                expr_to_output,
                source_expr=item.expression.text,
                output_name=output_name,
                alias_name=item.alias,
            )
            continue
        _validate_row_expr_scope(
            item.expression.text,
            alias_targets=scope.alias_targets,
            active_match_alias=active_alias,
            allowed_match_aliases=scope.allowed_match_aliases or None,
            unwind_aliases=set(),
            params=params,
            field=stage.clause.kind,
            line=item.span.line,
            column=item.span.column,
        )
        pre_items.append(
            (
                output_name,
                _row_expr_arg(
                    item.expression,
                    params=params,
                    alias_targets=scope.alias_targets,
                    field=stage.clause.kind,
                ),
            )
        )
        key_names.append(output_name)
        available_columns.add(output_name)
        _add_output_mapping(
            expr_to_output,
            source_expr=item.expression.text,
            output_name=output_name,
            alias_name=item.alias,
        )
        try:
            alias_name, prop = _projection_ref_from_expr(
                item.expression.text,
                alias_targets=scope.alias_targets,
                field=stage.clause.kind,
                line=item.span.line,
                column=item.span.column,
            )
        except GFQLValidationError:
            continue
        if alias_name == active_alias and prop is not None:
            projected_property_outputs.setdefault(prop, output_name)
            output_to_source_property[output_name] = prop

    aggregations: List[Sequence[Any]] = []
    for agg_spec in aggregate_specs:
        if agg_spec.output_name in available_columns:
            raise _unsupported(
                "Duplicate Cypher projection names are not yet supported in local lowering",
                field=stage.clause.kind,
                value=agg_spec.output_name,
                line=agg_spec.span_line,
                column=agg_spec.span_column,
            )
        _validate_aggregate_expr_scope(
            agg_spec,
            alias_targets=scope.alias_targets,
            active_match_alias=active_alias,
            allowed_match_aliases=scope.allowed_match_aliases or None,
            unwind_aliases=set(),
            params=params,
            field=stage.clause.kind,
        )
        runtime_func, runtime_expr = _aggregate_runtime_spec(
            agg_spec,
            alias_targets=scope.alias_targets,
        )
        if runtime_expr is None:
            aggregations.append((agg_spec.output_name, runtime_func))
        else:
            expr_text_obj = ExpressionText(
                text=runtime_expr,
                span=SourceSpan(
                    line=agg_spec.span_line,
                    column=agg_spec.span_column,
                    end_line=agg_spec.span_line,
                    end_column=agg_spec.span_column,
                    start_pos=0,
                    end_pos=0,
                ),
            )
            temp_name = _fresh_temp_name(temp_names, "__cypher_agg__")
            pre_items.append(
                (
                    temp_name,
                    _row_expr_arg(
                        expr_text_obj,
                        params=params,
                        alias_targets=scope.alias_targets,
                        field=stage.clause.kind,
                    ),
                )
            )
            aggregations.append((agg_spec.output_name, runtime_func, temp_name))
        available_columns.add(agg_spec.output_name)
        _add_output_mapping(
            expr_to_output,
            source_expr=agg_spec.source_text,
            output_name=agg_spec.output_name,
            alias_name=agg_spec.output_name,
        )

    row_steps: List[ASTObject] = []
    if key_names:
        if pre_items:
            row_steps.append(with_(pre_items))
        row_steps.append(group_by(key_names, aggregations))
    else:
        global_key = _fresh_temp_name(temp_names, "__cypher_group__")
        row_steps.append(with_([(global_key, 1)] + pre_items))
        row_steps.append(group_by([global_key], aggregations))
        row_steps.append(projection_fn([(agg.output_name, agg.output_name) for agg in aggregate_specs]))
        available_columns = {agg.output_name for agg in aggregate_specs}
        expr_to_output = {agg.source_text: agg.output_name for agg in aggregate_specs}
        projected_property_outputs = {}
        output_to_source_property = {}

    if hidden_group_key_names:
        visible_projection_items = [(item.alias or item.expression.text, item.alias or item.expression.text) for item in non_aggregate_items]
        visible_projection_items.extend((agg.output_name, agg.output_name) for agg in aggregate_specs)
        row_steps.append(projection_fn(visible_projection_items))
        available_columns = {name for name, _ in visible_projection_items}

    if stage.where is not None:
        _validate_row_expr_scope(
            stage.where.text,
            alias_targets={},
            active_match_alias=None,
            unwind_aliases=available_columns,
            params=params,
            field="with.where",
            line=stage.where.span.line,
            column=stage.where.span.column,
        )
        row_steps.append(
            where_rows(
                expr=_row_expr_arg(
                    stage.where,
                    params=params,
                    alias_targets={},
                    field="with.where",
                )
            )
        )
    if stage.order_by is not None:
        row_steps.append(
            _lower_order_by_outputs(
                stage.order_by,
                available_columns=available_columns,
                expr_to_output=expr_to_output,
                params=params,
            )
        )
    _append_page_ops_values(
        row_steps,
        skip_clause=stage.skip,
        limit_clause=stage.limit,
        params=params,
    )

    return row_steps, _StageScope(
        mode="row_columns",
        alias_targets={},
        active_alias=None,
        allowed_match_aliases=set(),
        row_columns=set(available_columns),
        projected_columns={},
        table=None,
        seed_rows=scope.seed_rows,
        relationship_count=scope.relationship_count,
    ), None if final_stage else None


def _expand_row_column_star_items(
    items: Sequence[ReturnItem],
    *,
    available_columns: Set[str],
    clause_kind: str,
) -> List[ReturnItem]:
    saw_star = any(item.expression.text == "*" for item in items)
    if not saw_star:
        return list(items)
    if len(items) != 1:
        first = items[0]
        raise _unsupported(
            f"Cypher {clause_kind.upper()} currently supports * only as the sole row-column projection item",
            field=f"{clause_kind}.items",
            value=[item.expression.text for item in items],
            line=first.span.line,
            column=first.span.column,
        )
    star_item = items[0]
    if star_item.alias is not None:
        raise _unsupported(
            f"Cypher {clause_kind.upper()} * does not support aliasing in the local compiler",
            field=f"{clause_kind}.items",
            value=star_item.alias,
            line=star_item.span.line,
            column=star_item.span.column,
        )
    if len(available_columns) == 0:
        raise _unsupported(
            f"Cypher {clause_kind.upper()} * requires projected row columns in this local phase",
            field=f"{clause_kind}.items",
            value="*",
            line=star_item.span.line,
            column=star_item.span.column,
        )
    return [
        ReturnItem(
            expression=ExpressionText(text=column, span=star_item.expression.span),
            alias=None,
            span=star_item.span,
        )
        for column in sorted(available_columns)
    ]


def _lower_row_column_stage(
    stage: ProjectionStage,
    *,
    scope: _StageScope,
    params: Optional[Mapping[str, Any]],
) -> Tuple[List[ASTObject], _StageScope]:
    _validate_with_projection_aliasing(stage)
    aggregate_specs: List[_AggregateSpec] = []
    non_aggregate_items: List[ReturnItem] = []
    post_aggregate_items: List[_PostAggregateExprPlan] = []
    clause_items = _expand_row_column_star_items(
        stage.clause.items,
        available_columns=scope.row_columns,
        clause_kind=stage.clause.kind,
    )
    for item in clause_items:
        agg_spec = _aggregate_spec(item, params=params, alias_targets={})
        if agg_spec is None:
            post_agg_plan = _post_aggregate_expr_plan(item, params=params, alias_targets={})
            if post_agg_plan is not None:
                nested_aggregate_specs, post_agg_item = post_agg_plan
                aggregate_specs.extend(nested_aggregate_specs)
                post_aggregate_items.append(post_agg_item)
            else:
                non_aggregate_items.append(item)
        else:
            aggregate_specs.append(agg_spec)

    if aggregate_specs:
        return _lower_row_column_aggregate_stage(
            stage,
            scope=scope,
            params=params,
            aggregate_specs=aggregate_specs,
            non_aggregate_items=non_aggregate_items,
            post_aggregate_items=post_aggregate_items,
        )

    projection_fn = with_ if stage.clause.kind == "with" else return_
    projection_items: List[Tuple[str, Any]] = []
    available_columns: Set[str] = set()
    expr_to_output: Dict[str, str] = {}
    next_projected_columns: Dict[str, _StageColumnBinding] = {}
    hidden_order_columns: List[str] = []
    temp_names: Set[str] = set()

    for item in clause_items:
        output_name = item.alias or item.expression.text
        if output_name in available_columns:
            raise _unsupported(
                "Duplicate Cypher projection names are not yet supported in local lowering",
                field=stage.clause.kind,
                value=output_name,
                line=item.span.line,
                column=item.span.column,
            )
        _validate_row_expr_scope(
            item.expression.text,
            alias_targets={},
            active_match_alias=None,
            unwind_aliases=scope.row_columns,
            params=params,
            field=stage.clause.kind,
                line=item.span.line,
                column=item.span.column,
            )
        rewritten_item = _rewrite_expr_to_projected_sources(
            item.expression,
            projected_columns=scope.projected_columns,
            params=params,
            alias_targets={},
            field=stage.clause.kind,
        )
        runtime_expr = _row_expr_arg(
            rewritten_item,
            params=params,
            alias_targets={},
            field=stage.clause.kind,
        )
        projection_items.append(
            (
                output_name,
                runtime_expr,
            )
        )
        available_columns.add(output_name)
        if isinstance(runtime_expr, str):
            next_projected_columns[output_name] = _StageColumnBinding(kind="expr", source_name=runtime_expr)
        _add_output_mapping(
            expr_to_output,
            source_expr=item.expression.text,
            output_name=output_name,
            alias_name=item.alias,
        )

    if stage.order_by is not None and stage.clause.kind == "return":
        if stage.clause.distinct:
            for order_item in stage.order_by.items:
                if order_item.expression.text in available_columns or order_item.expression.text in expr_to_output:
                    continue
                raise _unsupported(
                    "Cypher RETURN DISTINCT with ORDER BY on non-returned columns is not yet supported in local lowering",
                    field="order_by",
                    value=order_item.expression.text,
                    line=order_item.span.line,
                    column=order_item.span.column,
                )
        for order_item in stage.order_by.items:
            if order_item.expression.text in available_columns or order_item.expression.text in expr_to_output:
                continue
            rewritten_item = _rewrite_expr_to_projected_sources(
                order_item.expression,
                projected_columns=scope.projected_columns,
                params=params,
                alias_targets={},
                field="order_by",
            )
            temp_name = _fresh_temp_name(temp_names, "__cypher_order__")
            runtime_expr = _row_expr_arg(
                rewritten_item,
                params=params,
                alias_targets={},
                field="order_by",
            )
            projection_items.append((temp_name, runtime_expr))
            available_columns.add(temp_name)
            hidden_order_columns.append(temp_name)
            if isinstance(runtime_expr, str):
                next_projected_columns[temp_name] = _StageColumnBinding(kind="expr", source_name=runtime_expr)
            expr_to_output[order_item.expression.text] = temp_name

    row_steps: List[ASTObject] = [projection_fn(projection_items)]
    if stage.clause.distinct:
        row_steps.append(distinct())
    if stage.where is not None:
        _validate_row_expr_scope(
            stage.where.text,
            alias_targets={},
            active_match_alias=None,
            unwind_aliases=available_columns,
            params=params,
            field="with.where",
            line=stage.where.span.line,
            column=stage.where.span.column,
        )
        row_steps.append(
            where_rows(
                expr=_row_expr_arg(
                    _rewrite_expr_to_projected_sources(
                        stage.where,
                        projected_columns=next_projected_columns,
                        params=params,
                        alias_targets={},
                        field="with.where",
                    ),
                    params=params,
                    alias_targets={},
                    field="with.where",
                )
            )
        )
    if stage.order_by is not None:
        row_steps.append(
            _lower_order_by_outputs(
                stage.order_by,
                available_columns=available_columns,
                expr_to_output=expr_to_output,
                params=params,
            )
        )
    _append_page_ops_values(
        row_steps,
        skip_clause=stage.skip,
        limit_clause=stage.limit,
        params=params,
    )

    if hidden_order_columns:
        visible_items = [
            (item.alias or item.expression.text, item.alias or item.expression.text)
            for item in clause_items
        ]
        row_steps.append(select(visible_items))
        for hidden_column in hidden_order_columns:
            available_columns.discard(hidden_column)
            next_projected_columns.pop(hidden_column, None)

    return row_steps, _StageScope(
        mode="row_columns",
        alias_targets={},
        active_alias=None,
        row_columns=set(available_columns),
        projected_columns=next_projected_columns,
        table=None,
        seed_rows=scope.seed_rows,
        relationship_count=scope.relationship_count,
        allowed_match_aliases=set(),
    )


def _lower_row_only_sequence_with_scope(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]],
    initial_row_columns: AbstractSet[str],
    seed_rows: bool,
    procedure_call: Optional[CompiledCypherProcedureCall] = None,
) -> CompiledCypherQuery:
    row_steps: List[ASTObject] = [rows(table="nodes")]
    scope = _StageScope(
        mode="row_columns",
        alias_targets={},
        active_alias=None,
        row_columns=set(initial_row_columns),
        projected_columns={},
        table=None,
        seed_rows=seed_rows,
        relationship_count=0,
        allowed_match_aliases=set(),
    )

    for item in query.row_sequence:
        if isinstance(item, UnwindClause):
            if item.alias in scope.row_columns or item.alias in scope.projected_columns:
                raise _unsupported(
                    "Cypher UNWIND alias collides with an existing row column in this local subset",
                    field="unwind.alias",
                    value=item.alias,
                    line=item.span.line,
                    column=item.span.column,
                )
            rewritten_expr = _rewrite_expr_to_projected_sources(
                item.expression,
                projected_columns=scope.projected_columns,
                params=params,
                alias_targets={},
                field="unwind",
            )
            row_steps.append(
                unwind(
                    _row_expr_arg(
                        rewritten_expr,
                        params=params,
                        alias_targets={},
                        field="unwind",
                    ),
                    as_=item.alias,
                )
            )
            scope = _StageScope(
                mode="row_columns",
                alias_targets={},
                active_alias=None,
                row_columns=set(scope.row_columns) | {item.alias},
                projected_columns=dict(scope.projected_columns),
                table=None,
                seed_rows=scope.seed_rows,
                relationship_count=scope.relationship_count,
                allowed_match_aliases=set(),
            )
            continue

        stage_steps, scope = _lower_row_column_stage(item, scope=scope, params=params)
        row_steps.extend(stage_steps)

    return CompiledCypherQuery(Chain(row_steps), seed_rows=seed_rows, procedure_call=procedure_call)


def _lower_row_only_sequence(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]],
) -> CompiledCypherQuery:
    return _lower_row_only_sequence_with_scope(
        query,
        params=params,
        initial_row_columns=set(),
        seed_rows=True,
    )


def _lower_row_column_aggregate_stage(
    stage: ProjectionStage,
    *,
    scope: _StageScope,
    params: Optional[Mapping[str, Any]],
    aggregate_specs: Sequence[_AggregateSpec],
    non_aggregate_items: Sequence[ReturnItem],
    post_aggregate_items: Sequence[_PostAggregateExprPlan],
) -> Tuple[List[ASTObject], _StageScope]:
    non_aggregate_items = _expand_row_column_star_items(
        non_aggregate_items,
        available_columns=scope.row_columns,
        clause_kind=stage.clause.kind,
    )
    projection_fn = with_ if stage.clause.kind == "with" else return_
    pre_items: List[Tuple[str, Any]] = []
    key_names: List[str] = []
    temp_names: Set[str] = set()
    available_columns: Set[str] = set()
    expr_to_output: Dict[str, str] = {}

    for item in non_aggregate_items:
        output_name = item.alias or item.expression.text
        if output_name in available_columns:
            raise _unsupported(
                "Duplicate Cypher projection names are not yet supported in local lowering",
                field=stage.clause.kind,
                value=output_name,
                line=item.span.line,
                column=item.span.column,
            )
        _validate_row_expr_scope(
            item.expression.text,
            alias_targets={},
            active_match_alias=None,
            unwind_aliases=scope.row_columns,
            params=params,
            field=stage.clause.kind,
            line=item.span.line,
            column=item.span.column,
        )
        pre_items.append(
            (
                output_name,
                _row_expr_arg(
                    item.expression,
                    params=params,
                    alias_targets={},
                    field=stage.clause.kind,
                ),
            )
        )
        key_names.append(output_name)
        available_columns.add(output_name)
        _add_output_mapping(
            expr_to_output,
            source_expr=item.expression.text,
            output_name=output_name,
            alias_name=item.alias,
        )

    aggregations: List[Sequence[Any]] = []
    for agg_spec in aggregate_specs:
        if agg_spec.output_name in available_columns:
            raise _unsupported(
                "Duplicate Cypher projection names are not yet supported in local lowering",
                field=stage.clause.kind,
                value=agg_spec.output_name,
                line=agg_spec.span_line,
                column=agg_spec.span_column,
            )
        runtime_func, runtime_expr = _aggregate_runtime_spec(agg_spec, alias_targets={})
        if runtime_expr is None:
            aggregations.append((agg_spec.output_name, runtime_func))
        else:
            expr_text_obj = ExpressionText(
                text=runtime_expr,
                span=SourceSpan(
                    line=agg_spec.span_line,
                    column=agg_spec.span_column,
                    end_line=agg_spec.span_line,
                    end_column=agg_spec.span_column,
                    start_pos=0,
                    end_pos=0,
                ),
            )
            _validate_row_expr_scope(
                runtime_expr,
                alias_targets={},
                active_match_alias=None,
                unwind_aliases=scope.row_columns,
                params=params,
                field=stage.clause.kind,
                line=agg_spec.span_line,
                column=agg_spec.span_column,
            )
            temp_name = _fresh_temp_name(temp_names, "__cypher_agg__")
            pre_items.append(
                (
                    temp_name,
                    _row_expr_arg(
                        expr_text_obj,
                        params=params,
                        alias_targets={},
                        field=stage.clause.kind,
                    ),
                )
            )
            aggregations.append((agg_spec.output_name, runtime_func, temp_name))
        available_columns.add(agg_spec.output_name)
        _add_output_mapping(
            expr_to_output,
            source_expr=agg_spec.source_text,
            output_name=agg_spec.output_name,
            alias_name=agg_spec.output_name,
        )

    row_steps: List[ASTObject] = []
    if key_names:
        if pre_items:
            row_steps.append(with_(pre_items))
        row_steps.append(group_by(key_names, aggregations))
    else:
        global_key = _fresh_temp_name(temp_names, "__cypher_group__")
        row_steps.append(with_([(global_key, 1)] + pre_items))
        row_steps.append(group_by([global_key], aggregations))
        available_columns = {agg.output_name for agg in aggregate_specs}
        expr_to_output = {agg.source_text: agg.output_name for agg in aggregate_specs}

    if post_aggregate_items:
        post_projection_items: List[Tuple[str, Any]] = []
        projected_columns: Set[str] = set()
        for item in non_aggregate_items:
            output_name = item.alias or item.expression.text
            post_projection_items.append((output_name, output_name))
            projected_columns.add(output_name)
        for agg_spec in aggregate_specs:
            if agg_spec.output_name.startswith("__cypher_postagg__"):
                continue
            if agg_spec.output_name in projected_columns:
                raise _unsupported(
                    "Duplicate Cypher projection names are not yet supported in local lowering",
                    field=stage.clause.kind,
                    value=agg_spec.output_name,
                    line=agg_spec.span_line,
                    column=agg_spec.span_column,
                )
            post_projection_items.append((agg_spec.output_name, agg_spec.output_name))
            projected_columns.add(agg_spec.output_name)
        for plan in post_aggregate_items:
            if plan.output_name in projected_columns:
                raise _unsupported(
                    "Duplicate Cypher projection names are not yet supported in local lowering",
                    field=stage.clause.kind,
                    value=plan.output_name,
                    line=plan.span_line,
                    column=plan.span_column,
                )
            post_projection_items.append(
                (
                    plan.output_name,
                    _row_expr_arg(
                        plan.expr,
                        params=params,
                        alias_targets={},
                        field=stage.clause.kind,
                    ),
                )
            )
            projected_columns.add(plan.output_name)
        row_steps.append(projection_fn(post_projection_items))
        available_columns = projected_columns
    elif not key_names:
        row_steps.append(projection_fn([(agg.output_name, agg.output_name) for agg in aggregate_specs]))
        available_columns = {agg.output_name for agg in aggregate_specs}

    if stage.where is not None:
        _validate_row_expr_scope(
            stage.where.text,
            alias_targets={},
            active_match_alias=None,
            unwind_aliases=available_columns,
            params=params,
            field="with.where",
            line=stage.where.span.line,
            column=stage.where.span.column,
        )
        row_steps.append(
            where_rows(
                expr=_row_expr_arg(
                    stage.where,
                    params=params,
                    alias_targets={},
                    field="with.where",
                )
            )
        )

    if stage.order_by is not None:
        row_steps.append(
            _lower_order_by_outputs(
                stage.order_by,
                available_columns=available_columns,
                expr_to_output=expr_to_output,
                params=params,
            )
        )
    _append_page_ops_values(
        row_steps,
        skip_clause=stage.skip,
        limit_clause=stage.limit,
        params=params,
    )

    return row_steps, _StageScope(
        mode="row_columns",
        alias_targets={},
        active_alias=None,
        row_columns=set(available_columns),
        projected_columns={},
        table=None,
        seed_rows=scope.seed_rows,
        relationship_count=scope.relationship_count,
        allowed_match_aliases=set(),
    )


def _apply_literal_where(
    targets: Dict[str, ASTObject],
    *,
    left: PropertyRef,
    op: str,
    right: Optional[CypherLiteral],
    params: Optional[Mapping[str, Any]],
    ) -> None:
    if left.alias not in targets:
        raise GFQLValidationError(
            ErrorCode.E108,
            f"Unknown Cypher alias '{left.alias}' in WHERE clause",
            field="where.left.alias",
            value=left.alias,
            suggestion="Reference an alias declared in the MATCH pattern.",
            line=left.span.line,
            column=left.span.column,
            language="cypher",
        )
    target = targets[left.alias]
    filter_dict = dict(_target_filter_dict(target) or {})
    resolved = None if right is None else _resolve_literal(
        right,
        params=params,
        field=f"where.{left.alias}.{left.property}",
    )
    new_filter = _predicate_value(op, resolved)
    existing_filter = filter_dict.get(left.property)
    if existing_filter is None or left.property not in filter_dict:
        filter_dict[left.property] = new_filter
    else:
        filter_dict[left.property] = _merge_filter_predicates(
            existing_filter,
            new_filter,
            field=f"where.{left.alias}.{left.property}",
            line=left.span.line,
            column=left.span.column,
        )
    _set_target_filter_dict(target, filter_dict)


def _as_ast_predicate(value: Any) -> Any:
    if isinstance(value, ASTPredicate):
        return value
    if value is None:
        return None
    return eq(value)


def _merge_filter_predicates(existing: Any, new: Any, *, field: str, line: int, column: int) -> Any:
    existing_predicate = _as_ast_predicate(existing)
    new_predicate = _as_ast_predicate(new)
    if existing_predicate is None or new_predicate is None:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Duplicate null-equality filtering on the same property is not supported in the local Cypher compiler",
            field=field,
            value=None,
            suggestion="Rewrite the filter using IS NULL / IS NOT NULL or avoid repeated null equality on the same property.",
            line=line,
            column=column,
            language="cypher",
        )
    return all_of(existing_predicate, new_predicate)


def _apply_label_where(
    targets: Dict[str, ASTObject],
    *,
    left: LabelRef,
) -> None:
    if left.alias not in targets:
        raise GFQLValidationError(
            ErrorCode.E108,
            f"Unknown Cypher alias '{left.alias}' in WHERE clause",
            field="where.left.alias",
            value=left.alias,
            suggestion="Reference an alias declared in the MATCH pattern.",
            line=left.span.line,
            column=left.span.column,
            language="cypher",
        )

    target = targets[left.alias]
    if not isinstance(target, ASTNode):
        raise _unsupported(
            "Cypher label predicates in WHERE currently support node aliases only",
            field="where",
            value=left.alias,
            line=left.span.line,
            column=left.span.column,
        )

    filter_dict = dict(target.filter_dict or {})
    for label in left.labels:
        filter_dict[f"label__{label}"] = True
    target.filter_dict = filter_dict


def _extract_relationship_type_where(
    expr: ExpressionText,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> Optional[Tuple[PropertyRef, Literal["==", "!="], CypherLiteral]]:
    node = _parse_row_expr(
        expr.text,
        params=params,
        alias_targets=alias_targets,
        field="where",
        line=expr.span.line,
        column=expr.span.column,
    )
    if not isinstance(node, BinaryOp) or node.op not in {"=", "!="}:
        return None

    def _relationship_alias(node_in: ExprNode) -> Optional[str]:
        if not isinstance(node_in, FunctionCall) or node_in.name != "type" or len(node_in.args) != 1:
            return None
        arg = node_in.args[0]
        if not isinstance(arg, Identifier):
            return None
        alias_name, prop = _split_qualified_name(arg.name, line=expr.span.line, column=expr.span.column)
        if prop is not None:
            return None
        if not isinstance(alias_targets.get(alias_name), ASTEdge):
            return None
        return alias_name

    left_alias = _relationship_alias(node.left)
    right_alias = _relationship_alias(node.right)
    op = cast(Literal["==", "!="], "==" if node.op == "=" else "!=")
    if left_alias is not None and isinstance(node.right, ExprLiteral) and isinstance(node.right.value, str):
        return PropertyRef(alias=left_alias, property="type", span=expr.span), op, node.right.value
    if right_alias is not None and isinstance(node.left, ExprLiteral) and isinstance(node.left.value, str):
        return PropertyRef(alias=right_alias, property="type", span=expr.span), op, node.left.value
    return None


def lower_match_clause(
    clause: MatchClause,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> List[ASTObject]:
    out: List[ASTObject] = []
    elements = list(_match_pattern_elements(clause))
    relationships = [
        (i, el) for i, el in enumerate(elements) if isinstance(el, RelationshipPattern)
    ]
    last_rel_idx = relationships[-1][0] if relationships else -1
    for i, element in enumerate(elements):
        if isinstance(element, NodePattern):
            out.append(_lower_node(element, params=params))
        else:
            # Prune intermediate graph after variable-length hops that are
            # not the last relationship in a connected pattern, so the next
            # hop only expands from the wavefront endpoints.
            is_varlen = _is_variable_length_relationship_pattern(element)
            is_nonterminal = i < last_rel_idx
            out.append(_lower_relationship(
                element,
                params=params,
                prune_to_endpoints=is_varlen and is_nonterminal,
            ))
    return out


def _rewrite_where_pattern_predicates_to_matches(query: CypherQuery) -> CypherQuery:
    if query.where is None or not query.where.predicates:
        return query
    pattern_preds = [predicate for predicate in query.where.predicates if isinstance(predicate, WherePatternPredicate)]
    if not pattern_preds:
        return query
    first = pattern_preds[0]
    if len(pattern_preds) > 1:
        raise _unsupported(
            "Cypher WHERE currently supports one positive pattern predicate at a time",
            field="where",
            value=len(pattern_preds),
            line=first.span.line,
            column=first.span.column,
        )
    if len(first.pattern) < 3:
        raise _unsupported(
            "Cypher WHERE pattern predicates must include a relationship",
            field="where",
            value=None,
            line=first.span.line,
            column=first.span.column,
        )
    bound_aliases = {
        cast(str, element.variable)
        for clause in query.matches
        for pattern in clause.patterns
        for element in pattern
        if getattr(element, "variable", None) is not None
    }
    introduced_aliases = sorted(
        cast(str, element.variable)
        for element in first.pattern
        if getattr(element, "variable", None) is not None and cast(str, element.variable) not in bound_aliases
    )
    if introduced_aliases:
        raise _unsupported(
            "Cypher WHERE pattern predicates cannot introduce new aliases in this phase",
            field="where",
            value=introduced_aliases,
            line=first.span.line,
            column=first.span.column,
        )

    remaining = tuple(predicate for predicate in query.where.predicates if not isinstance(predicate, WherePatternPredicate))
    remaining_where = None
    if remaining or query.where.expr is not None:
        remaining_where = WhereClause(
            predicates=cast(Any, remaining),
            expr=query.where.expr,
            span=query.where.span,
        )
    extra_match = MatchClause(patterns=(first.pattern,), span=first.span, optional=False, pattern_aliases=(None,))
    return replace(query, matches=query.matches + (extra_match,), where=remaining_where)


def _reject_unsupported_where_expr_forms(query: CypherQuery) -> None:
    if query.where is None or query.where.expr is None:
        return
    expr_text = query.where.expr.text.strip()
    if _CYPHER_BARE_WHERE_GROUPED_ALIAS_RE.fullmatch(expr_text) is not None:
        raise _unsupported(
            "Cypher WHERE pattern predicates must include a relationship",
            field="where",
            value=expr_text,
            line=query.where.span.line,
            column=query.where.span.column,
        )


def _is_variable_length_relationship_pattern(relationship: RelationshipPattern) -> bool:
    return (
        relationship.min_hops is not None
        or relationship.max_hops is not None
        or relationship.to_fixed_point
    )


def _reject_unsupported_variable_length_where_pattern_predicates(query: CypherQuery) -> None:
    if query.where is None:
        return
    for predicate in query.where.predicates:
        if not isinstance(predicate, WherePatternPredicate):
            continue
        relationships = [
            element
            for element in predicate.pattern
            if isinstance(element, RelationshipPattern)
        ]
        for relationship in relationships:
            if not _is_variable_length_relationship_pattern(relationship):
                continue
            if relationship.min_hops is None and relationship.max_hops is None and relationship.to_fixed_point:
                continue
            raise _unsupported(
                "Cypher WHERE pattern predicates currently support only bare variable-length fixed-point relationships, not exact or bounded hop counts",
                field="where",
                value=query.where.expr.text if query.where.expr is not None else None,
                line=predicate.span.line,
                column=predicate.span.column,
            )


def _reject_nonterminal_variable_length_relationship_patterns(query: CypherQuery) -> None:  # noqa: ARG001
    """No-op: variable-length rels in connected patterns are now supported.

    The lowering sets ``prune_to_endpoints=True`` on non-terminal
    variable-length edges so the next hop starts from the correct
    wavefront endpoints only.  See #1001 for reentry-match follow-up.
    """


def _variable_length_relationship_aliases(
    alias_targets: Mapping[str, ASTObject],
) -> Set[str]:
    out: Set[str] = set()
    for alias, target in alias_targets.items():
        if not isinstance(target, ASTEdge):
            continue
        if target.min_hops is not None or target.max_hops is not None or target.to_fixed_point:
            out.add(alias)
    return out


def _variable_length_path_aliases(query: CypherQuery) -> Set[str]:
    out: Set[str] = set()
    for clause in query.matches + query.reentry_matches:
        pattern_aliases = clause.pattern_aliases or tuple(None for _ in clause.patterns)
        for alias, pattern in zip(pattern_aliases, clause.patterns):
            if alias is None:
                continue
            if any(
                isinstance(element, RelationshipPattern)
                and _is_variable_length_relationship_pattern(element)
                for element in pattern
            ):
                out.add(alias)
    return out


def _reject_variable_length_path_alias_references(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]],
) -> None:
    variable_length_path_aliases = _variable_length_path_aliases(query)
    if not variable_length_path_aliases:
        return

    alias_targets = {
        alias: cast(ASTObject, ASTNode(name=alias))
        for alias in variable_length_path_aliases
    }

    def _check_expr(expr_text: str, *, field: str, line: int, column: int) -> None:
        if not (_expr_match_aliases(
            expr_text,
            alias_targets=alias_targets,
            params=params,
            field=field,
            line=line,
            column=column,
        ) & variable_length_path_aliases):
            return
        raise _unsupported(
            "Cypher variable-length named path aliases cannot yet be projected or used in row expressions in the local compiler",
            field=field,
            value=expr_text,
            line=line,
            column=column,
        )

    if query.where is not None and query.where.expr is not None:
        _check_expr(
            query.where.expr.text,
            field="where",
            line=query.where.span.line,
            column=query.where.span.column,
        )
    for reentry_where in query.reentry_wheres:
        if reentry_where is not None and reentry_where.expr is not None:
            _check_expr(
                reentry_where.expr.text,
                field="where",
                line=reentry_where.span.line,
                column=reentry_where.span.column,
            )

    def _check_projection_clause(clause: ReturnClause) -> None:
        for item in clause.items:
            if item.expression.text == "*":
                continue
            _check_expr(
                item.expression.text,
                field=clause.kind,
                line=item.span.line,
                column=item.span.column,
            )

    _check_projection_clause(query.return_)

    if query.order_by is not None:
        for item in query.order_by.items:
            _check_expr(
                item.expression.text,
                field="order_by",
                line=item.span.line,
                column=item.span.column,
            )

    for stage in query.with_stages:
        _check_projection_clause(stage.clause)
        if stage.where is not None:
            _check_expr(
                stage.where.text,
                field="with.where",
                line=stage.span.line,
                column=stage.span.column,
            )
        if stage.order_by is not None:
            for item in stage.order_by.items:
                _check_expr(
                    item.expression.text,
                    field="order_by",
                    line=item.span.line,
                    column=item.span.column,
                )


def _reject_variable_length_relationship_alias_path_carriers(
    query: CypherQuery,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> None:
    variable_length_aliases = _variable_length_relationship_aliases(alias_targets)
    if not variable_length_aliases:
        return

    def _check_expr(expr_text: str, *, field: str, line: int, column: int) -> None:
        if not (_expr_match_aliases(
            expr_text,
            alias_targets=alias_targets,
            params=params,
            field=field,
            line=line,
            column=column,
        ) & variable_length_aliases):
            return
        raise _unsupported(
            "Cypher variable-length relationship aliases cannot yet be projected or aggregated as path/list carriers in the local compiler",
            field=field,
            value=expr_text,
            line=line,
            column=column,
        )

    if query.where is not None and query.where.expr is not None:
        _check_expr(
            query.where.expr.text,
            field="where",
            line=query.where.span.line,
            column=query.where.span.column,
        )

    def _check_projection_clause(clause: ReturnClause) -> None:
        for item in clause.items:
            if item.expression.text == "*":
                continue
            _check_expr(
                item.expression.text,
                field=clause.kind,
                line=item.span.line,
                column=item.span.column,
            )

    _check_projection_clause(query.return_)

    if query.order_by is not None:
        for item in query.order_by.items:
            _check_expr(
                item.expression.text,
                field="order_by",
                line=item.span.line,
                column=item.span.column,
            )

    for stage in query.with_stages:
        _check_projection_clause(stage.clause)
        if stage.where is not None:
            _check_expr(
                stage.where.text,
                field="with.where",
                line=stage.span.line,
                column=stage.span.column,
            )
        if stage.order_by is not None:
            for item in stage.order_by.items:
                _check_expr(
                    item.expression.text,
                    field="order_by",
                    line=item.span.line,
                    column=item.span.column,
                )


def lower_match_query(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> LoweredCypherMatch:
    query = _rewrite_where_pattern_predicates_to_matches(query)
    _reject_unsupported_where_expr_forms(query)
    _reject_variable_length_path_alias_references(query, params=params)
    merged_match = _merged_match_clause(query)
    if merged_match is None:
        raise _unsupported(
            "Cypher MATCH lowering requires a MATCH clause",
            field="match",
            value=None,
            line=query.return_.span.line,
            column=query.return_.span.column,
        )
    ops, duplicate_alias_where = _lower_match_clause_with_alias_equalities(merged_match, params=params)
    alias_targets = _alias_target(ops)
    binding_row_aliases = _binding_row_aliases_for_match(query.match, alias_targets=alias_targets)
    where_out: List[WhereComparison] = list(duplicate_alias_where)
    dynamic_where_out, dynamic_row_where_predicates = _dynamic_property_entry_constraints(
        merged_match,
        alias_targets=alias_targets,
        params=params,
    )
    where_out.extend(dynamic_where_out)

    row_where: Optional[ExpressionText] = None
    row_where_predicates: List[str] = list(dynamic_row_where_predicates)
    if query.where is not None:
        if query.where.expr is not None:
            type_where = _extract_relationship_type_where(
                query.where.expr,
                alias_targets=alias_targets,
                params=params,
            )
            if type_where is None:
                row_where = query.where.expr
            else:
                left, op, right = type_where
                _apply_literal_where(
                    alias_targets,
                    left=left,
                    op=op,
                    right=right,
                    params=params,
                )
        for predicate in query.where.predicates:
            if isinstance(predicate, WherePatternPredicate):
                raise _unsupported(
                    "Cypher WHERE pattern predicates must be rewritten before lowering",
                    field="where",
                    value=None,
                    line=predicate.span.line,
                    column=predicate.span.column,
                )
            if isinstance(predicate.left, LabelRef):
                _apply_label_where(alias_targets, left=predicate.left)
                continue
            if binding_row_aliases:
                row_predicate_expr = _row_where_predicate_text(predicate)
                if row_predicate_expr is not None:
                    row_where_predicates.append(row_predicate_expr)
                    continue
            if isinstance(predicate.right, PropertyRef):
                assert isinstance(predicate.left, PropertyRef)
                where_out.append(
                    compare(
                        col(predicate.left.alias, predicate.left.property),
                        cast(Any, predicate.op),
                        col(predicate.right.alias, predicate.right.property),
                    )
                )
                continue
            _apply_literal_where(
                alias_targets,
                left=cast(PropertyRef, predicate.left),
                op=predicate.op,
                right=cast(Optional[CypherLiteral], predicate.right),
                params=params,
            )
    if row_where_predicates:
        combined_expr = " and ".join(row_where_predicates)
        row_where = ExpressionText(
            text=combined_expr if row_where is None else f"({row_where.text}) and ({combined_expr})",
            span=query.where.span if query.where is not None else merged_match.span,
        )

    return LoweredCypherMatch(query=ops, where=where_out, row_where=row_where)


def _fresh_temp_name(existing: Set[str], prefix: str) -> str:
    candidate = prefix
    counter = 0
    while candidate in existing:
        counter += 1
        candidate = f"{prefix}{counter}"
    existing.add(candidate)
    return candidate


def _render_row_where_operand_text(value: Union[PropertyRef, CypherLiteral]) -> str:
    if isinstance(value, PropertyRef):
        return f"{value.alias}.{value.property}"
    if isinstance(value, ParameterRef):
        return f"${value.name}"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    return str(value)


def _row_where_predicate_text(predicate: WherePredicate) -> Optional[str]:
    if isinstance(predicate.left, LabelRef) or predicate.right is None:
        return None
    op_map = {"==": "=", "<>": "!="}
    rendered_op = op_map.get(predicate.op, predicate.op)
    if rendered_op not in {"=", "!=", "<", "<=", ">", ">="}:
        return None
    return (
        f"{_render_row_where_operand_text(predicate.left)} "
        f"{rendered_op} "
        f"{_render_row_where_operand_text(predicate.right)}"
    )


def _distinct_aggregate_expr_text(
    agg_spec: _AggregateSpec,
    *,
    alias_targets: Mapping[str, ASTObject],
) -> Optional[str]:
    expr_text = agg_spec.expr_text
    if expr_text is None:
        return None
    target = alias_targets.get(expr_text)
    if isinstance(target, ASTNode):
        return "id"
    if isinstance(target, ASTEdge):
        if agg_spec.func == "collect":
            raise _unsupported(
                "collect(DISTINCT rel_alias) is not yet supported in local Cypher lowering",
                field="return.item",
                value=agg_spec.source_text,
                line=agg_spec.span_line,
                column=agg_spec.span_column,
            )
        return "__gfql_edge_index_0__"
    return expr_text


def _aggregate_runtime_spec(
    agg_spec: _AggregateSpec,
    *,
    alias_targets: Mapping[str, ASTObject],
) -> Tuple[str, Optional[str]]:
    func = agg_spec.func
    expr_text = agg_spec.expr_text
    if expr_text is not None:
        target = alias_targets.get(expr_text)
        if isinstance(target, ASTNode) and func in {"collect", "collect_distinct"}:
            expr_text = f"__node_entity__({expr_text})"
        elif isinstance(target, ASTEdge) and func in {"collect", "collect_distinct"}:
            expr_text = f"__edge_entity__({expr_text})"
    if not agg_spec.distinct:
        return func, expr_text
    if func == "count":
        return "count_distinct", _distinct_aggregate_expr_text(agg_spec, alias_targets=alias_targets)
    if func == "collect":
        return "collect_distinct", _distinct_aggregate_expr_text(agg_spec, alias_targets=alias_targets)
    raise _unsupported(
        "Cypher DISTINCT aggregates are currently supported for count() and collect() only",
        field="return.item",
        value=agg_spec.source_text,
        line=agg_spec.span_line,
        column=agg_spec.span_column,
    )


def _whole_row_group_entity_expr(
    alias_name: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    field: str,
    line: int,
    column: int,
) -> str:
    target = alias_targets.get(alias_name)
    alias_args = ", ".join(dict.fromkeys([alias_name] + [str(name) for name in alias_targets.keys()]))
    if isinstance(target, ASTNode):
        return f"__node_entity__({alias_args})"
    if isinstance(target, ASTEdge):
        return f"__edge_entity__({alias_args})"
    raise _unsupported(
        "Cypher aggregate whole-row grouping requires a node or edge alias",
        field=field,
        value=alias_name,
        line=line,
        column=column,
    )


def _whole_row_group_key_expr(
    alias_name: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    field: str,
    line: int,
    column: int,
) -> str:
    target = alias_targets.get(alias_name)
    if isinstance(target, ASTNode):
        return "id"
    if isinstance(target, ASTEdge):
        return "__gfql_edge_index_0__"
    raise _unsupported(
        "Cypher aggregate whole-row grouping requires a node or edge alias",
        field=field,
        value=alias_name,
        line=line,
        column=column,
    )


def _add_output_mapping(
    expr_to_output: Dict[str, str],
    *,
    source_expr: str,
    output_name: str,
    alias_name: Optional[str],
) -> None:
    expr_to_output[source_expr] = output_name
    if alias_name is not None:
        expr_to_output[alias_name] = output_name


def _lower_general_row_projection(
    query: CypherQuery,
    lowered: LoweredCypherMatch,
    *,
    params: Optional[Mapping[str, Any]],
) -> CompiledCypherQuery:
    alias_targets = _alias_target(lowered.query) if query.match is not None else {}
    binding_row_aliases = _binding_row_aliases_for_match(query.match, alias_targets=alias_targets)
    active_match_alias = _active_match_alias(
        query,
        alias_targets=alias_targets,
        allowed_match_aliases=binding_row_aliases or None,
        params=params,
    )
    seed_rows = query.match is None

    if active_match_alias is None:
        row_steps: List[ASTObject] = [rows(table="nodes")]
    elif binding_row_aliases:
        row_steps = [rows(binding_ops=serialize_binding_ops(lowered.query))]
    else:
        row_steps = [
            rows(
                table=_alias_table(
                    alias_targets[active_match_alias],
                    alias=active_match_alias,
                    line=query.return_.span.line,
                    column=query.return_.span.column,
                ),
                source=active_match_alias,
            )
        ]

    _append_match_row_where(
        row_steps,
        lowered=lowered,
        alias_targets=alias_targets,
        active_alias=active_match_alias,
        allowed_match_aliases=binding_row_aliases or None,
        params=params,
    )

    unwind_aliases: Set[str] = set()
    for unwind_clause in query.unwinds:
        if unwind_clause.alias in alias_targets or unwind_clause.alias in unwind_aliases:
            raise _unsupported(
                "Cypher UNWIND alias collides with an existing alias in this local subset",
                field="unwind.alias",
                value=unwind_clause.alias,
                line=unwind_clause.span.line,
                column=unwind_clause.span.column,
            )
        _validate_row_expr_scope(
            unwind_clause.expression.text,
            alias_targets=alias_targets,
            active_match_alias=active_match_alias,
            allowed_match_aliases=binding_row_aliases or None,
            unwind_aliases=unwind_aliases,
            params=params,
            field="unwind",
            line=unwind_clause.span.line,
            column=unwind_clause.span.column,
        )
        row_steps.append(
            unwind(
                _row_expr_arg(
                    unwind_clause.expression,
                    params=params,
                    alias_targets=alias_targets,
                    field="unwind",
                ),
                as_=unwind_clause.alias,
            )
        )
        unwind_aliases.add(unwind_clause.alias)

    aggregate_specs: List[_AggregateSpec] = []
    non_aggregate_items: List[ReturnItem] = []
    post_aggregate_items: List[_PostAggregateExprPlan] = []
    empty_result_row: Optional[Dict[str, Any]] = None
    for item in query.return_.items:
        agg_spec = _aggregate_spec(item, params=params, alias_targets=alias_targets)
        if agg_spec is not None:
            aggregate_specs.append(agg_spec)
            continue
        post_agg_plan = _post_aggregate_expr_plan(item, params=params, alias_targets=alias_targets)
        if post_agg_plan is not None:
            nested_aggregate_specs, post_agg_item = post_agg_plan
            aggregate_specs.extend(nested_aggregate_specs)
            post_aggregate_items.append(post_agg_item)
            continue
        non_aggregate_items.append(item)

    expr_to_output: Dict[str, str] = {}
    available_columns: Set[str] = set()

    if aggregate_specs:
        projection_fn = with_ if query.return_.kind == "with" else return_
        pre_items: List[Tuple[str, Any]] = []
        key_names: List[str] = []
        temp_names: Set[str] = set()
        hidden_group_key_names: Set[str] = set()

        for item in non_aggregate_items:
            output_name = item.alias or item.expression.text
            if output_name in available_columns:
                raise _unsupported(
                    "Duplicate Cypher projection names are not yet supported in local lowering",
                    field=query.return_.kind,
                    value=output_name,
                    line=item.span.line,
                    column=item.span.column,
                )
            if item.expression.text in alias_targets:
                alias_name = item.expression.text
                if alias_name != active_match_alias:
                    raise _unsupported(
                        "Cypher aggregate whole-row grouping currently supports the active MATCH alias only",
                        field=query.return_.kind,
                        value=item.expression.text,
                        line=item.span.line,
                        column=item.span.column,
                    )
                hidden_key_name = _fresh_temp_name(temp_names, "__cypher_group_key__")
                pre_items.append(
                    (
                        hidden_key_name,
                        _whole_row_group_key_expr(
                            alias_name,
                            alias_targets=alias_targets,
                            field=query.return_.kind,
                            line=item.span.line,
                            column=item.span.column,
                        ),
                    )
                )
                pre_items.append(
                    (
                        output_name,
                        _whole_row_group_entity_expr(
                            alias_name,
                            alias_targets=alias_targets,
                            field=query.return_.kind,
                            line=item.span.line,
                            column=item.span.column,
                        ),
                    )
                )
                key_names.extend([hidden_key_name, output_name])
                hidden_group_key_names.add(hidden_key_name)
                available_columns.add(output_name)
                _add_output_mapping(
                    expr_to_output,
                    source_expr=item.expression.text,
                    output_name=output_name,
                    alias_name=item.alias,
                )
                continue
            _validate_row_expr_scope(
                item.expression.text,
                alias_targets=alias_targets,
                active_match_alias=active_match_alias,
                allowed_match_aliases=binding_row_aliases or None,
                unwind_aliases=unwind_aliases,
                params=params,
                field=query.return_.kind,
                line=item.span.line,
                column=item.span.column,
            )
            pre_items.append(
                (
                    output_name,
                    _row_expr_arg(
                        item.expression,
                        params=params,
                        alias_targets=alias_targets,
                        field=query.return_.kind,
                    ),
                )
            )
            key_names.append(output_name)
            available_columns.add(output_name)
            _add_output_mapping(
                expr_to_output,
                source_expr=item.expression.text,
                output_name=output_name,
                alias_name=item.alias,
            )

        aggregations: List[Sequence[Any]] = []
        for agg_spec in aggregate_specs:
            if agg_spec.output_name in available_columns:
                raise _unsupported(
                    "Duplicate Cypher projection names are not yet supported in local lowering",
                    field=query.return_.kind,
                    value=agg_spec.output_name,
                    line=agg_spec.span_line,
                    column=agg_spec.span_column,
                )
            _validate_aggregate_expr_scope(
                agg_spec,
                alias_targets=alias_targets,
                active_match_alias=active_match_alias,
                allowed_match_aliases=binding_row_aliases or None,
                unwind_aliases=unwind_aliases,
                params=params,
                field=query.return_.kind,
            )
            runtime_func, runtime_expr = _aggregate_runtime_spec(
                agg_spec,
                alias_targets=alias_targets,
            )
            if runtime_expr is None:
                aggregations.append((agg_spec.output_name, runtime_func))
            else:
                expr_text_obj = ExpressionText(text=runtime_expr, span=SourceSpan(
                    line=agg_spec.span_line,
                    column=agg_spec.span_column,
                    end_line=agg_spec.span_line,
                    end_column=agg_spec.span_column,
                    start_pos=0,
                    end_pos=0,
                ))
                temp_name = _fresh_temp_name(temp_names, "__cypher_agg__")
                pre_items.append(
                    (
                        temp_name,
                        _row_expr_arg(
                            expr_text_obj,
                            params=params,
                            alias_targets=alias_targets,
                            field=query.return_.kind,
                        ),
                    )
                )
                aggregations.append((agg_spec.output_name, runtime_func, temp_name))
            available_columns.add(agg_spec.output_name)
            _add_output_mapping(
                expr_to_output,
                source_expr=agg_spec.source_text,
                output_name=agg_spec.output_name,
                alias_name=agg_spec.output_name,
            )

        _reject_unsound_relationship_multiplicity_aggregates(
            query,
            aggregate_specs=aggregate_specs,
            alias_targets=alias_targets,
            active_match_alias=active_match_alias,
        )
        if key_names:
            if len(pre_items) > 0:
                row_steps.append(with_(pre_items))
            row_steps.append(group_by(key_names, aggregations))
        else:
            global_key = _fresh_temp_name(temp_names, "__cypher_group__")
            row_steps.append(with_([(global_key, 1)] + pre_items))
            row_steps.append(group_by([global_key], aggregations))
            available_columns = {agg.output_name for agg in aggregate_specs}
            empty_result_row = _empty_aggregate_row(aggregate_specs)

        if post_aggregate_items or hidden_group_key_names:
            post_projection_items: List[Tuple[str, Any]] = []
            projected_columns: Set[str] = set()
            for item in non_aggregate_items:
                output_name = item.alias or item.expression.text
                post_projection_items.append((output_name, output_name))
                projected_columns.add(output_name)
            for agg_spec in aggregate_specs:
                if agg_spec.output_name in projected_columns:
                    raise _unsupported(
                        "Duplicate Cypher projection names are not yet supported in local lowering",
                        field=query.return_.kind,
                        value=agg_spec.output_name,
                        line=agg_spec.span_line,
                        column=agg_spec.span_column,
                    )
                if agg_spec.output_name.startswith("__cypher_postagg__"):
                    continue
                post_projection_items.append((agg_spec.output_name, agg_spec.output_name))
                projected_columns.add(agg_spec.output_name)
            for plan in post_aggregate_items:
                if plan.output_name in projected_columns:
                    raise _unsupported(
                        "Duplicate Cypher projection names are not yet supported in local lowering",
                        field=query.return_.kind,
                        value=plan.output_name,
                        line=plan.span_line,
                        column=plan.span_column,
                    )
                post_projection_items.append(
                    (
                        plan.output_name,
                        _row_expr_arg(
                            plan.expr,
                            params=params,
                            alias_targets={},
                            field=query.return_.kind,
                        ),
                    )
                )
                projected_columns.add(plan.output_name)
            row_steps.append(projection_fn(post_projection_items))
            available_columns = projected_columns
        elif not key_names:
            row_steps.append(projection_fn([(agg.output_name, agg.output_name) for agg in aggregate_specs]))
    else:
        if query.match is not None and not query.unwinds and not binding_row_aliases:
            return CompiledCypherQuery(
                Chain(_lower_projection_chain(query, lowered, params=params), where=lowered.where),
                seed_rows=False,
            )
        projection_fn = with_ if query.return_.kind == "with" else return_
        projection_items: List[Tuple[str, Any]] = []
        for item in query.return_.items:
            if item.expression.text in alias_targets:
                raise _unsupported(
                    "Whole-row alias RETURN/WITH is not supported once UNWIND/top-level row execution is active",
                    field=query.return_.kind,
                    value=item.expression.text,
                    line=item.span.line,
                    column=item.span.column,
                )
            _validate_row_expr_scope(
                item.expression.text,
                alias_targets=alias_targets,
                active_match_alias=active_match_alias,
                allowed_match_aliases=binding_row_aliases or None,
                unwind_aliases=unwind_aliases,
                params=params,
                field=query.return_.kind,
                line=item.span.line,
                column=item.span.column,
            )
            output_name = item.alias or item.expression.text
            if output_name in available_columns:
                raise _unsupported(
                    "Duplicate Cypher projection names are not yet supported in local lowering",
                    field=query.return_.kind,
                    value=output_name,
                    line=item.span.line,
                    column=item.span.column,
                )
            projection_items.append(
                (
                    output_name,
                    _row_expr_arg(
                        item.expression,
                        params=params,
                        alias_targets=alias_targets,
                        field=query.return_.kind,
                    ),
                )
            )
            available_columns.add(output_name)
            _add_output_mapping(
                expr_to_output,
                source_expr=item.expression.text,
                output_name=output_name,
                alias_name=item.alias,
            )
        row_steps.append(projection_fn(projection_items))
        if query.return_.distinct:
            row_steps.append(distinct())

    if query.order_by is not None:
        row_steps.append(
            _lower_order_by_outputs(
                query.order_by,
                available_columns=available_columns,
                expr_to_output=expr_to_output,
                params=params,
            )
        )
    _append_page_ops(row_steps, query=query, params=params)
    exec_steps = row_steps if binding_row_aliases else lowered.query + row_steps
    return CompiledCypherQuery(
        Chain(exec_steps, where=lowered.where),
        seed_rows=seed_rows,
        empty_result_row=empty_result_row,
    )


def _cypher_return_output_names(clause: ReturnClause) -> Tuple[str, ...]:
    names: List[str] = []
    for item in clause.items:
        if item.expression.text == "*":
            raise _unsupported(
                "Cypher UNION does not yet support RETURN * branches in the local compiler",
                field="return",
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )
        names.append(item.alias or item.expression.text)
    return tuple(names)



def lower_cypher_query(
    query: Union[CypherQuery, CypherUnionQuery],
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Chain:
    compiled = compile_cypher_query(query, params=params)
    if isinstance(compiled, CompiledCypherUnionQuery):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher UNION cannot be represented as a single GFQL Chain",
            field="union",
            value=compiled.union_kind,
            suggestion="Execute the Cypher query through g.gfql(\"...\", language=\"cypher\") instead of cypher_to_gfql().",
            language="cypher",
        )
    if compiled.procedure_call is not None:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher CALL cannot be represented as a single GFQL Chain",
            field="call",
            value=compiled.procedure_call.procedure,
            suggestion="Execute the Cypher query through g.gfql(\"...\", language=\"cypher\") instead of cypher_to_gfql().",
            language="cypher",
        )
    return compiled.chain


def _reentry_hidden_column_name(output_name: str) -> str:
    return f"__cypher_reentry_{output_name}__"


def _rewrite_reentry_expr_to_hidden_properties(
    expr: ExpressionText,
    *,
    carried_alias: str,
    carried_columns: Sequence[str],
    field: str,
) -> ExpressionText:
    if not carried_columns:
        return expr
    normalized_text = expr.text
    for output_name in carried_columns:
        hidden_name = _reentry_hidden_column_name(output_name)
        normalized_text = re.sub(
            rf"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*\.{re.escape(hidden_name)}(?![A-Za-z0-9_])",
            f"{carried_alias}.{hidden_name}",
            normalized_text,
        )
    try:
        node = parse_expr(normalized_text)
    except (GFQLExprParseError, ImportError) as exc:
        raise _unsupported(
            "Cypher MATCH after WITH carried-column rewrite requires a locally supported scalar expression",
            field=field,
            value=normalized_text,
            line=expr.span.line,
            column=expr.span.column,
        ) from exc
    replacements = {
        output_name: f"{carried_alias}.{_reentry_hidden_column_name(output_name)}"
        for output_name in carried_columns
    }
    identifiers = collect_identifiers(node)
    if not any(identifier in replacements for identifier in identifiers):
        if normalized_text == expr.text:
            return expr
        return ExpressionText(text=normalized_text, span=expr.span)
    return ExpressionText(
        text=_render_expr_node(_rewrite_expr_identifiers(node, replacements)),
        span=expr.span,
    )


def _bounded_reentry_carry_columns(
    prefix_projection: ResultProjectionPlan,
    *,
    projection_items: Sequence[str],
    query: CypherQuery,
    prefix_stage: ProjectionStage,
) -> Tuple[str, Tuple[str, ...]]:
    whole_row_columns = tuple(column.output_name for column in prefix_projection.columns if column.kind == "whole_row")
    if len(whole_row_columns) != 1:
        raise _unsupported_at_span(
            "Cypher MATCH after WITH currently requires the prefix WITH stage to project exactly one whole-row alias",
            field="with",
            value=projection_items,
            span=prefix_stage.span,
        )
    carried_columns = tuple(column.output_name for column in prefix_projection.columns if column.kind != "whole_row")
    if not carried_columns:
        return whole_row_columns[0], ()
    invalid_output = next((name for name in carried_columns if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)), None)
    if invalid_output is not None:
        raise _unsupported_at_span(
            "Cypher MATCH after WITH carried scalar columns currently require identifier-style WITH aliases",
            field="with",
            value=invalid_output,
            span=prefix_stage.span,
        )
    if len(set(carried_columns)) != len(carried_columns):
        raise _unsupported_at_span(
            "Cypher MATCH after WITH carried scalar columns currently require distinct WITH aliases",
            field="with",
            value=carried_columns,
            span=prefix_stage.span,
        )
    return whole_row_columns[0], carried_columns


def _bounded_reentry_scalar_prefix_columns(
    prefix_stage: ProjectionStage,
    *,
    projection_items: Sequence[str],
) -> Tuple[str, ...]:
    if prefix_stage.order_by is not None or prefix_stage.skip is not None or prefix_stage.limit is not None:
        raise _unsupported_at_span(
            "Cypher MATCH after WITH scalar-only prefix stages do not yet support ORDER BY, SKIP, or LIMIT",
            field="with",
            value=projection_items,
            span=prefix_stage.span,
        )
    carried_columns = tuple(
        item.alias or item.expression.text
        for item in prefix_stage.clause.items
    )
    if not carried_columns:
        raise _unsupported_at_span(
            "Cypher MATCH after WITH scalar-only prefix stages require at least one scalar output",
            field="with",
            value=projection_items,
            span=prefix_stage.span,
        )
    invalid_output = next((name for name in carried_columns if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)), None)
    if invalid_output is not None:
        raise _unsupported_at_span(
            "Cypher MATCH after WITH scalar-only prefix stages currently require identifier-style WITH aliases",
            field="with",
            value=invalid_output,
            span=prefix_stage.span,
        )
    if len(set(carried_columns)) != len(carried_columns):
        raise _unsupported_at_span(
            "Cypher MATCH after WITH scalar-only prefix stages currently require distinct WITH aliases",
            field="with",
            value=carried_columns,
            span=prefix_stage.span,
        )
    return carried_columns


def _literal_limit_value(limit_clause: Optional[LimitClause]) -> Optional[int]:
    if limit_clause is None:
        return None
    value = limit_clause.value
    if isinstance(value, int):
        return value
    if isinstance(value, ParameterRef):
        return None
    text = value.text.strip()
    if not re.fullmatch(r"\d+", text):
        return None
    return int(text)


def _bounded_reentry_prefix_order_is_safe(
    *,
    prefix_stage: ProjectionStage,
    query: CypherQuery,
) -> bool:
    if prefix_stage.order_by is None:
        return True
    if query.order_by is not None:
        return True
    return prefix_stage.skip is None and _literal_limit_value(prefix_stage.limit) == 1


def _first_pattern_node_alias(clause: MatchClause) -> Optional[str]:
    if clause.patterns:
        first_pattern = clause.patterns[0]
        if first_pattern and isinstance(first_pattern[0], NodePattern):
            return first_pattern[0].variable
    pattern = _match_pattern_elements(clause)
    if not pattern or not isinstance(pattern[0], NodePattern):
        return None
    return pattern[0].variable


def _map_terminal_reentry_query(
    compiled_query: CompiledCypherQuery,
    *,
    transform: Callable[[CompiledCypherQuery], CompiledCypherQuery],
) -> CompiledCypherQuery:
    if compiled_query.start_nodes_query is None:
        return transform(compiled_query)
    return replace(
        compiled_query,
        start_nodes_query=_map_terminal_reentry_query(
            compiled_query.start_nodes_query,
            transform=transform,
        ),
    )


def _rewrite_reentry_projection_clause(
    clause: ReturnClause,
    *,
    rewrite_expr: Callable[[ExpressionText, str], ExpressionText],
) -> ReturnClause:
    return replace(
        clause,
        items=tuple(
            replace(
                item,
                expression=rewritten_expr,
                alias=item.alias or (item.expression.text if rewritten_expr.text != item.expression.text else None),
            )
            for item in clause.items
            for rewritten_expr in (rewrite_expr(item.expression, clause.kind),)
        ),
    )


def _rewrite_reentry_property_entry(
    entry: PropertyEntry,
    *,
    rewrite_expr: Callable[[ExpressionText, str], ExpressionText],
) -> PropertyEntry:
    if not isinstance(entry.value, ExpressionText):
        return entry
    return replace(
        entry,
        value=rewrite_expr(entry.value, "match.property"),
    )


def _rewrite_reentry_pattern_element(
    element: PatternElement,
    *,
    rewrite_expr: Callable[[ExpressionText, str], ExpressionText],
) -> PatternElement:
    rewritten_properties = tuple(
        _rewrite_reentry_property_entry(entry, rewrite_expr=rewrite_expr)
        for entry in element.properties
    )
    return replace(element, properties=rewritten_properties)


def _rewrite_reentry_match_clause(
    clause: MatchClause,
    *,
    rewrite_expr: Callable[[ExpressionText, str], ExpressionText],
) -> MatchClause:
    return replace(
        clause,
        patterns=tuple(
            tuple(
                _rewrite_reentry_pattern_element(element, rewrite_expr=rewrite_expr)
                for element in pattern
            )
            for pattern in clause.patterns
        ),
    )


def _rewrite_reentry_projection_stage(
    stage: ProjectionStage,
    *,
    rewrite_expr: Callable[[ExpressionText, str], ExpressionText],
) -> ProjectionStage:
    rewritten_order_by = None
    if stage.order_by is not None:
        rewritten_order_by = replace(
            stage.order_by,
            items=tuple(
                replace(
                    item,
                    expression=rewrite_expr(item.expression, "order_by"),
                )
                for item in stage.order_by.items
            ),
        )
    return replace(
        stage,
        clause=_rewrite_reentry_projection_clause(stage.clause, rewrite_expr=rewrite_expr),
        where=None if stage.where is None else rewrite_expr(stage.where, "where"),
        order_by=rewritten_order_by,
    )


def _rewrite_collect_unwind_reentry_query(query: CypherQuery) -> Optional[CypherQuery]:
    if not query.with_stages or len(query.unwinds) != 1 or len(query.reentry_matches) != 1:
        return None
    prefix_stage = query.with_stages[0]
    remaining_with_stages = query.with_stages[1:]
    if (
        prefix_stage.where is not None
        or prefix_stage.order_by is not None
        or prefix_stage.skip is not None
        or prefix_stage.limit is not None
        or len(prefix_stage.clause.items) < 1
    ):
        return None
    unwind_clause = query.unwinds[0]
    # Find the collect(...) item that feeds the UNWIND
    collected_idx: Optional[int] = None
    collected_match_result: Optional[re.Match[str]] = None
    for idx, item in enumerate(prefix_stage.clause.items):
        output_name = item.alias or item.expression.text
        if output_name != unwind_clause.expression.text:
            continue
        m = re.fullmatch(
            r"collect\(\s*(distinct\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\)",
            item.expression.text,
            flags=re.IGNORECASE,
        )
        if m is not None:
            collected_idx = idx
            collected_match_result = m
            break
    if collected_idx is None or collected_match_result is None:
        return None
    reentry_alias = _first_pattern_node_alias(query.reentry_matches[0])
    if reentry_alias is None or reentry_alias != unwind_clause.alias:
        return None
    collected_item = prefix_stage.clause.items[collected_idx]
    source_alias = collected_match_result.group(2)
    rewritten_item = replace(
        collected_item,
        expression=ExpressionText(text=source_alias, span=collected_item.expression.span),
        alias=unwind_clause.alias,
    )
    # Rebuild items: put the whole-row alias first, then carried scalars.
    # The reentry machinery expects the whole-row alias to be the primary
    # projection source, so it must come first.
    other_items = tuple(
        item for i, item in enumerate(prefix_stage.clause.items) if i != collected_idx
    )
    rewritten_items = (rewritten_item,) + other_items
    rewritten_prefix_stage = replace(
        prefix_stage,
        clause=replace(
            prefix_stage.clause,
            items=rewritten_items,
            distinct=bool(collected_match_result.group(1)),
        ),
    )
    return replace(
        query,
        with_stages=(rewritten_prefix_stage,) + remaining_with_stages,
        unwinds=(),
    )


def _compile_bounded_reentry_query(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> CompiledCypherQuery:
    if query.unwinds:
        rewritten_query = _rewrite_collect_unwind_reentry_query(query)
        if rewritten_query is None:
            first_unwind = query.unwinds[0]
            raise _unsupported_at_span(
                "Cypher UNWIND after WITH/RETURN currently supports only a single WITH collect([distinct] alias) AS list UNWIND list AS alias MATCH ... RETURN shape",
                field="unwind",
                value=first_unwind.expression.text,
                span=first_unwind.span,
            )
        query = rewritten_query
    if not query.reentry_matches or len(query.with_stages) not in {
        len(query.reentry_matches),
        len(query.reentry_matches) + 1,
    }:
        raise _unsupported_at_span(
            "Cypher MATCH after WITH is only supported for alternating MATCH ... WITH ... MATCH ... [WITH ... MATCH ...] ... [WITH] RETURN read shapes in the local compiler",
            field="match",
            value=len(query.reentry_matches),
            span=query.return_.span,
        )
    prefix_stage = query.with_stages[0]
    projection_items = [item.expression.text for item in prefix_stage.clause.items]
    if prefix_stage.where is not None:
        raise _unsupported_at_span(
            "Cypher MATCH after WITH does not yet support WITH ... WHERE in the prefix stage",
            field="with.where",
            value=prefix_stage.where.text,
            span=prefix_stage.span,
        )
    prefix_query = replace(
        query,
        call=None,
        row_sequence=(),
        reentry_matches=(),
        reentry_wheres=(),
        graph_bindings=(),
        use=None,
        with_stages=(),
        return_=replace(prefix_stage.clause, kind="return"),
        order_by=prefix_stage.order_by,
        skip=prefix_stage.skip,
        limit=prefix_stage.limit,
        trailing_semicolon=False,
        reentry_unwinds=(),
    )
    prefix_compiled = compile_cypher_query(prefix_query, params=params)
    if not isinstance(prefix_compiled, CompiledCypherQuery):
        raise _unsupported_at_span(
            "Cypher MATCH after WITH prefix compilation produced an unexpected UNION program",
            field="with",
            value="union",
            span=prefix_stage.span,
        )
    reentry_match = query.reentry_matches[0]
    remaining_with_stages = query.with_stages[1:]
    remaining_reentry_matches = query.reentry_matches[1:]
    first_alias = _first_pattern_node_alias(reentry_match)
    prefix_projection = prefix_compiled.result_projection
    scalar_only_prefix = prefix_projection is None
    prefix_projection_table: Optional[Literal["nodes", "edges"]] = None
    if scalar_only_prefix:
        scalar_prefix_aliases = {
            item.alias
            for item in prefix_stage.clause.items
            if item.alias is not None
        }
        reused_scalar_aliases = sorted(
            scalar_prefix_aliases
            & set().union(*(_pattern_node_aliases(pattern) for pattern in reentry_match.patterns))
        )
        if reused_scalar_aliases:
            raise _unsupported_at_span(
                "Cypher MATCH after WITH scalar-only prefix aliases cannot be reused as node variables in the trailing MATCH",
                field="match",
                value=reused_scalar_aliases,
                span=reentry_match.span,
            )
        if first_alias is None:
            raise _unsupported_at_span(
                "Cypher MATCH after WITH currently requires the trailing MATCH to start from a named node alias",
                field="match",
                value=first_alias,
                span=reentry_match.span,
            )
        reentry_alias = first_alias
        carry_columns = _bounded_reentry_scalar_prefix_columns(
            prefix_stage,
            projection_items=projection_items,
        )
    else:
        assert prefix_projection is not None
        prefix_projection_table = prefix_projection.table
        reentry_alias, carry_columns = _bounded_reentry_carry_columns(
            prefix_projection,
            projection_items=projection_items,
            query=query,
            prefix_stage=prefix_stage,
        )
    if not _bounded_reentry_prefix_order_is_safe(prefix_stage=prefix_stage, query=query):
        raise _unsupported(
            "Cypher MATCH after WITH does not yet preserve prefix WITH row ordering across MATCH re-entry for multi-row result shapes",
            field="with.order_by",
            value=(
                [item.expression.text for item in prefix_stage.order_by.items]
                if prefix_stage.order_by is not None
                else None
            ),
            line=prefix_stage.order_by.span.line if prefix_stage.order_by is not None else prefix_stage.span.line,
            column=prefix_stage.order_by.span.column if prefix_stage.order_by is not None else prefix_stage.span.column,
        )
    if prefix_projection_table is not None and prefix_projection_table != "nodes":
        raise _unsupported_at_span(
            "Cypher MATCH after WITH currently supports node re-entry only",
            field="with",
            value=prefix_projection_table,
            span=prefix_stage.span,
        )
    if len(query.return_.items) == 1 and query.return_.items[0].expression.text == "*":
        raise _unsupported_at_span(
            "Cypher MATCH after WITH does not yet support RETURN * from the trailing MATCH re-entry stage",
            field=query.return_.kind,
            value="*",
            span=query.return_.span,
        )
    if first_alias is None or first_alias != reentry_alias:
        raise _unsupported_at_span(
            "Cypher MATCH after WITH currently requires the trailing MATCH to start from the same carried node alias",
            field="match",
            value=first_alias,
            span=reentry_match.span,
        )

    hidden_columns = tuple(_reentry_hidden_column_name(output_name) for output_name in carry_columns)

    def rewrite_expr(expr: ExpressionText, field: str) -> ExpressionText:
        return _rewrite_reentry_expr_to_hidden_properties(
            expr,
            carried_alias=reentry_alias,
            carried_columns=carry_columns,
            field=field,
        )

    reentry_where = query.reentry_where
    reentry_return = query.return_
    reentry_order_by = query.order_by
    rewritten_with_stages = remaining_with_stages
    rewritten_reentry_unwinds = query.reentry_unwinds
    remaining_reentry_wheres = query.reentry_wheres[1:] if query.reentry_wheres else ()
    rewritten_remaining_reentry_wheres = remaining_reentry_wheres
    rewritten_reentry_match = reentry_match
    rewritten_remaining_reentry_matches = remaining_reentry_matches
    if hidden_columns:
        rewritten_reentry_match = _rewrite_reentry_match_clause(reentry_match, rewrite_expr=rewrite_expr)
        rewritten_remaining_reentry_matches = tuple(
            _rewrite_reentry_match_clause(match_clause, rewrite_expr=rewrite_expr)
            for match_clause in remaining_reentry_matches
        )
        rewritten_remaining_reentry_wheres = tuple(
            None
            if where_clause is None or where_clause.expr is None
            else replace(
                where_clause,
                expr=rewrite_expr(where_clause.expr, "where"),
            )
            for where_clause in remaining_reentry_wheres
        )
        rewritten_with_stages = tuple(
            _rewrite_reentry_projection_stage(stage, rewrite_expr=rewrite_expr)
            for stage in remaining_with_stages
        )
        rewritten_reentry_unwinds = tuple(
            replace(
                unwind_clause,
                expression=rewrite_expr(unwind_clause.expression, "unwind"),
            )
            for unwind_clause in query.reentry_unwinds
        )
        if query.reentry_where is not None and query.reentry_where.expr is not None:
            reentry_where = replace(
                query.reentry_where,
                expr=rewrite_expr(query.reentry_where.expr, "where"),
            )
        if not remaining_reentry_matches:
            reentry_return = _rewrite_reentry_projection_clause(query.return_, rewrite_expr=rewrite_expr)
            if reentry_order_by is not None:
                reentry_order_by = replace(
                    reentry_order_by,
                    items=tuple(
                        replace(
                            item,
                            expression=rewrite_expr(item.expression, "order_by"),
                        )
                        for item in reentry_order_by.items
                    ),
                )
    if rewritten_reentry_unwinds and rewritten_with_stages and not rewritten_remaining_reentry_matches:
        first_unwind = rewritten_reentry_unwinds[0]
        raise _unsupported_at_span(
            "Cypher UNWIND after WITH/RETURN is not yet supported once MATCH has introduced graph aliases",
            field="unwind",
            value=first_unwind.expression.text,
            span=first_unwind.span,
        )
    suffix_query = replace(
        query,
        call=None,
        row_sequence=(),
        graph_bindings=(),
        use=None,
        matches=(rewritten_reentry_match,),
        where=reentry_where,
        unwinds=rewritten_reentry_unwinds,
        with_stages=rewritten_with_stages,
        return_=reentry_return,
        order_by=reentry_order_by,
        reentry_matches=rewritten_remaining_reentry_matches,
        reentry_wheres=rewritten_remaining_reentry_wheres,
        reentry_unwinds=(),
    )
    suffix_compiled = compile_cypher_query(suffix_query, params=params)
    if not isinstance(suffix_compiled, CompiledCypherQuery):
        raise _unsupported_at_span(
            "Cypher MATCH after WITH suffix compilation produced an unexpected UNION program",
            field="match",
            value="union",
            span=reentry_match.span,
        )
    def attach_current_reentry(target: CompiledCypherQuery) -> CompiledCypherQuery:
        target_projection = target.result_projection
        if target_projection is not None and target_projection.alias == reentry_alias and hidden_columns:
            target_projection = replace(
                target_projection,
                exclude_columns=tuple(
                    dict.fromkeys(target_projection.exclude_columns + hidden_columns)
                ),
            )
        return replace(
            target,
            start_nodes_query=prefix_compiled,
            result_projection=target_projection,
            scalar_reentry_alias=reentry_alias if scalar_only_prefix else target.scalar_reentry_alias,
            scalar_reentry_columns=carry_columns if scalar_only_prefix else target.scalar_reentry_columns,
        )

    return _map_terminal_reentry_query(
        suffix_compiled,
        transform=attach_current_reentry,
    )


def _compile_call_query(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> CompiledCypherQuery:
    if query.call is None:
        raise ValueError("Expected query.call for Cypher CALL compilation")
    if query.matches or query.where is not None or query.reentry_matches or any(
        where_clause is not None for where_clause in query.reentry_wheres
    ):
        raise _unsupported(
            "Cypher CALL is only supported in standalone or row-only local queries",
            field="call",
            value=query.call.procedure,
            line=query.call.span.line,
            column=query.call.span.column,
        )
    compiled_call = compile_cypher_call(query.call, params=params)
    is_bare_call = (
        not query.unwinds
        and not query.with_stages
        and query.order_by is None
        and query.skip is None
        and query.limit is None
        and len(query.return_.items) == 1
        and query.return_.items[0].expression.text == "*"
    )
    if compiled_call.result_kind == "graph":
        if not is_bare_call:
            raise _unsupported(
                "Graph-preserving Cypher CALL procedures are only supported as standalone local queries",
                field="call",
                value=query.call.procedure,
                line=query.call.span.line,
                column=query.call.span.column,
            )
        return CompiledCypherQuery(
            Chain([]),
            seed_rows=False,
            procedure_call=compiled_call,
        )
    if is_bare_call:
        return CompiledCypherQuery(
            Chain([]),
            seed_rows=False,
            procedure_call=compiled_call,
        )
    return _lower_row_only_sequence_with_scope(
        query,
        params=params,
        initial_row_columns={output.output_name for output in compiled_call.output_columns},
        seed_rows=False,
        procedure_call=compiled_call,
    )


def _compile_graph_constructor(
    constructor: GraphConstructor,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> CompiledCypherQuery:
    """Compile a GRAPH { ... } constructor body into a CompiledCypherQuery.

    Supports both MATCH-based constructors (subgraph extraction) and
    CALL-based constructors (graph-preserving procedure execution).
    """
    if not constructor.matches and constructor.call is None:
        raise _unsupported(
            "Graph constructor must contain at least one MATCH or CALL clause",
            field="graph_constructor",
            value="GRAPH { }",
            line=constructor.span.line,
            column=constructor.span.column,
        )

    if constructor.call is not None:
        # CALL-based constructor: compile the procedure call
        compiled_call = compile_cypher_call(constructor.call)
        if compiled_call.result_kind != "graph":
            raise _unsupported(
                "Only graph-preserving CALL procedures (with .write()) are allowed inside a graph constructor",
                field="graph_constructor",
                value=constructor.call.procedure,
                line=constructor.call.span.line,
                column=constructor.call.span.column,
            )
        return CompiledCypherQuery(
            Chain([]),
            seed_rows=False,
            procedure_call=compiled_call,
        )

    # MATCH-based constructor: lower match + where into a chain
    synthetic = CypherQuery(
        matches=constructor.matches,
        where=constructor.where,
        call=None,
        unwinds=(),
        with_stages=(),
        return_=ReturnClause(
            items=(),
            distinct=False,
            kind="return",
            span=constructor.span,
        ),
        order_by=None,
        skip=None,
        limit=None,
        row_sequence=(),
        trailing_semicolon=False,
        span=constructor.span,
        reentry_unwinds=(),
    )
    lowered = lower_match_query(synthetic, params=params)
    return CompiledCypherQuery(
        Chain(lowered.query, where=lowered.where),
        seed_rows=False,
    )


def _compile_graph_bindings(
    bindings: Tuple[GraphBinding, ...],
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Tuple[CompiledGraphBinding, ...]:
    """Compile a sequence of GRAPH g = GRAPH { ... } bindings."""
    compiled: List[CompiledGraphBinding] = []
    for binding in bindings:
        compiled_query = _compile_graph_constructor(binding.constructor, params=params)
        use_ref = binding.constructor.use.ref if binding.constructor.use is not None else None
        compiled.append(CompiledGraphBinding(
            name=binding.name,
            chain=compiled_query.chain,
            procedure_call=compiled_query.procedure_call,
            use_ref=use_ref,
        ))
    return tuple(compiled)


def _is_connected_optional_match_query(query: CypherQuery) -> bool:
    """Detect: exactly 2 MATCH clauses, first non-optional connected, second optional."""
    if len(query.matches) != 2:
        return False
    if query.matches[0].optional or not query.matches[1].optional:
        return False
    if query.where is not None or query.with_stages or query.unwinds or query.call is not None or query.row_sequence:
        return False
    first = query.matches[0]
    # Check for RelationshipPattern directly in the raw patterns to avoid
    # calling _normalized_match_pattern which may fail on comma-separated
    # patterns that cannot be stitched.
    has_relationship = any(
        isinstance(el, RelationshipPattern)
        for pat in first.patterns
        for el in pat
    )
    if not has_relationship:
        return False
    # The existing _merged_match_clause path would fail here because the
    # connected first MATCH cannot be seed-extracted as a single node pattern.
    return True


@dataclass(frozen=True)
class ConnectedOptionalMatchPlan:
    """Plan for a connected MATCH + OPTIONAL MATCH left-outer-join.

    Execution: run base_chain and opt_chain to produce binding-row tables,
    left-outer-join on shared_node_aliases, then dispatch post_join_chain
    (which uses the standard row pipeline) for RETURN / ORDER BY / etc.
    """
    base_chain: Chain
    opt_chain: Chain
    shared_node_aliases: Tuple[str, ...]
    opt_only_aliases: Tuple[str, ...]
    post_join_chain: Chain


@dataclass(frozen=True)
class ConnectedMatchJoinPlan:
    """Plan for a connected non-linear comma MATCH lowered through joined rows."""

    pattern_chains: Tuple[Chain, ...]
    pattern_shared_node_aliases: Tuple[Tuple[str, ...], ...]
    post_join_chain: Chain


def _compile_connected_match_join(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> CompiledCypherQuery:
    clause = query.matches[0]
    pattern_chains: List[Chain] = []
    pattern_node_aliases: List[Set[str]] = []
    combined_alias_targets: Dict[str, ASTObject] = {}
    pre_join_filters: List[ExpressionText] = []

    for pattern in clause.patterns:
        single_clause = MatchClause(
            patterns=(pattern,),
            optional=clause.optional,
            span=clause.span,
            pattern_aliases=(None,),
        )
        ops, duplicate_alias_where = _lower_match_clause_with_alias_equalities(single_clause, params=params)
        alias_targets = _alias_target(ops)
        dynamic_where_out, dynamic_row_preds = _dynamic_property_entry_constraints(
            single_clause,
            alias_targets=alias_targets,
            params=params,
        )
        pattern_chains.append(Chain(ops, where=duplicate_alias_where + dynamic_where_out))
        pattern_node_aliases.append({alias for alias, target in alias_targets.items() if isinstance(target, ASTNode)})
        for alias, target in alias_targets.items():
            if alias not in combined_alias_targets:
                combined_alias_targets[alias] = target
        pre_join_filters.extend(
            ExpressionText(text=expr, span=single_clause.span)
            for expr in dynamic_row_preds
        )

    shared_aliases_per_pattern: List[Tuple[str, ...]] = []
    accumulated_aliases = set(pattern_node_aliases[0])
    for node_aliases in pattern_node_aliases[1:]:
        shared = tuple(sorted(accumulated_aliases & node_aliases))
        if not shared:
            raise _unsupported(
                "Cypher connected comma-pattern join lowering requires each additional pattern to share a node alias with the accumulated join graph",
                field="match",
                value=None,
                line=clause.span.line,
                column=clause.span.column,
            )
        shared_aliases_per_pattern.append(shared)
        accumulated_aliases.update(node_aliases)

    if query.where is not None and query.where.expr is not None:
        pre_join_filters.append(query.where.expr)

    for projection_clause in [stage.clause for stage in query.with_stages] + [query.return_]:
        _reject_unsupported_connected_join_clause_shapes(
            projection_clause,
            alias_targets=combined_alias_targets,
            params=params,
        )

    row_steps: List[ASTObject] = []
    for expr in pre_join_filters:
        rewritten = _rewrite_connected_join_expr(
            expr,
            alias_targets=combined_alias_targets,
            params=params,
            field="where",
            line=expr.span.line,
            column=expr.span.column,
        )
        if rewritten is None:
            continue
        row_steps.append(
            where_rows(
                expr=_row_expr_arg(
                    rewritten,
                    params=params,
                    alias_targets={},
                    field="where",
                )
            )
        )

    scope = _StageScope(
        mode="row_columns",
        alias_targets={},
        active_alias=None,
        row_columns=set(),
        projected_columns={},
        table=None,
        seed_rows=False,
        relationship_count=0,
        allowed_match_aliases=set(),
    )

    for stage in query.with_stages:
        rewritten_stage = replace(
            stage,
            clause=_rewrite_connected_join_return_clause(
                stage.clause,
                alias_targets=combined_alias_targets,
                params=params,
            ),
            where=_rewrite_connected_join_expr(
                stage.where,
                alias_targets=combined_alias_targets,
                params=params,
                field="with.where",
                line=stage.span.line,
                column=stage.span.column,
            ),
            order_by=_rewrite_connected_join_order_by_clause(
                stage.order_by,
                alias_targets=combined_alias_targets,
                params=params,
            ),
        )
        stage_steps, scope = _lower_row_column_stage(rewritten_stage, scope=scope, params=params)
        row_steps.extend(stage_steps)

    final_stage = ProjectionStage(
        clause=_rewrite_connected_join_return_clause(
            query.return_,
            alias_targets=combined_alias_targets,
            params=params,
        ),
        where=None,
        order_by=_rewrite_connected_join_order_by_clause(
            query.order_by,
            alias_targets=combined_alias_targets,
            params=params,
        ),
        skip=query.skip,
        limit=query.limit,
        span=query.return_.span,
    )
    final_steps, _ = _lower_row_column_stage(final_stage, scope=scope, params=params)
    row_steps.extend(final_steps)

    return CompiledCypherQuery(
        chain=Chain([]),
        seed_rows=False,
        connected_match_join=ConnectedMatchJoinPlan(
            pattern_chains=tuple(pattern_chains),
            pattern_shared_node_aliases=tuple(shared_aliases_per_pattern),
            post_join_chain=Chain(row_steps),
        ),
    )


def _compile_connected_optional_match(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> CompiledCypherQuery:
    """Compile a connected MATCH + OPTIONAL MATCH query.

    Lowers the non-optional and optional MATCH clauses as independent chains,
    builds a post-join row-pipeline chain for RETURN / ORDER BY / SKIP / LIMIT,
    and emits a ``ConnectedOptionalMatchPlan`` so the executor can left-outer-join
    the results at runtime then delegate projection to the standard row pipeline.
    """
    base_clause = query.matches[0]
    opt_clause = query.matches[1]

    base_aliases = _match_clause_aliases(base_clause)
    opt_aliases = _match_clause_aliases(opt_clause)
    shared_aliases = base_aliases & opt_aliases
    opt_only_aliases = opt_aliases - base_aliases

    if not shared_aliases:
        raise _unsupported(
            "Cypher connected MATCH + OPTIONAL MATCH requires at least one shared alias between the two patterns",
            field="match",
            value=None,
            line=opt_clause.span.line,
            column=opt_clause.span.column,
        )

    base_ops = lower_match_clause(base_clause, params=params)
    opt_ops = lower_match_clause(opt_clause, params=params)
    base_alias_targets = _alias_target(base_ops)
    opt_alias_targets = _alias_target(opt_ops)
    shared_node_aliases = sorted(
        alias for alias in shared_aliases
        if isinstance(base_alias_targets.get(alias), ASTNode)
    )
    if not shared_node_aliases:
        raise _unsupported(
            "Cypher connected MATCH + OPTIONAL MATCH requires at least one shared node alias for the join",
            field="match",
            value=sorted(shared_aliases),
            line=opt_clause.span.line,
            column=opt_clause.span.column,
        )

    base_chain = Chain(base_ops)
    opt_chain = Chain(opt_ops)

    # Build the combined alias_targets so projection can reference any alias.
    combined_alias_targets: Dict[str, ASTObject] = {}
    combined_alias_targets.update(base_alias_targets)
    for alias, target in opt_alias_targets.items():
        if alias not in combined_alias_targets:
            combined_alias_targets[alias] = target

    # Build the row-pipeline ops that will run on the joined DataFrame.
    # This reuses the same projection infrastructure as the normal path.
    try:
        active = _active_match_alias(query, alias_targets=combined_alias_targets, params=params)
    except GFQLValidationError:
        active = next(iter(combined_alias_targets)) if combined_alias_targets else None

    plan = _build_projection_plan(
        query.return_,
        alias_targets=combined_alias_targets,
        active_alias=active,
        params=params,
    )

    # The post-join chain starts with rows() (the joined DataFrame is
    # already in binding-row format), then standard projection ops.
    post_join_ops: List[ASTObject] = []
    if not plan.whole_row_output_names:
        post_join_ops.append(return_(plan.projection_items))
    if query.return_.distinct:
        post_join_ops.append(distinct())
    if query.order_by is not None:
        post_join_ops.append(
            _lower_order_by_clause(
                query.order_by,
                plan=plan,
                alias_targets=combined_alias_targets,
                params=params,
            )
        )
    _append_page_ops(post_join_ops, query=query, params=params)
    post_join_chain = Chain(post_join_ops)

    return CompiledCypherQuery(
        chain=base_chain,
        seed_rows=False,
        connected_optional_match=ConnectedOptionalMatchPlan(
            base_chain=base_chain,
            opt_chain=opt_chain,
            shared_node_aliases=tuple(shared_node_aliases),
            opt_only_aliases=tuple(sorted(opt_only_aliases)),
            post_join_chain=post_join_chain,
        ),
    )


def compile_cypher_query(
    query: Union[CypherQuery, CypherUnionQuery, CypherGraphQuery],
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Union[CompiledCypherQuery, CompiledCypherUnionQuery, CompiledCypherGraphQuery]:
    if isinstance(query, CypherGraphQuery):
        compiled_bindings = _compile_graph_bindings(query.graph_bindings, params=params)
        compiled_constructor = _compile_graph_constructor(query.constructor, params=params)
        use_ref = query.constructor.use.ref if query.constructor.use is not None else None
        return CompiledCypherGraphQuery(
            graph_bindings=compiled_bindings,
            chain=compiled_constructor.chain,
            procedure_call=compiled_constructor.procedure_call,
            use_ref=use_ref,
        )
    if isinstance(query, CypherUnionQuery):
        branch_output_names: Optional[Tuple[str, ...]] = None
        compiled_branches: List[CompiledCypherQuery] = []
        for branch in query.branches:
            output_names = _cypher_return_output_names(branch.return_)
            if branch_output_names is None:
                branch_output_names = output_names
            elif output_names != branch_output_names:
                raise _unsupported(
                    "Cypher UNION branches must project the same output names in the same order",
                    field="union",
                    value={"expected": branch_output_names, "actual": output_names},
                    line=branch.return_.span.line,
                    column=branch.return_.span.column,
                )
            compiled_branch = compile_cypher_query(branch, params=params)
            if not isinstance(compiled_branch, CompiledCypherQuery):
                raise _unsupported(
                    "Nested Cypher UNION branches are not supported in the local compiler",
                    field="union",
                    value=branch.return_.span.line,
                    line=branch.return_.span.line,
                    column=branch.return_.span.column,
                )
            compiled_branches.append(compiled_branch)
        return CompiledCypherUnionQuery(
            branches=tuple(compiled_branches),
            union_kind=query.union_kind,
        )

    # Capture graph bindings/use before lowering the regular query
    _graph_bindings = query.graph_bindings
    _use_ref = query.use.ref if query.use is not None else None
    if _graph_bindings:
        compiled_bindings = _compile_graph_bindings(_graph_bindings, params=params)
    else:
        compiled_bindings = ()

    def _attach_graph_context(result: CompiledCypherQuery) -> CompiledCypherQuery:
        if not compiled_bindings and _use_ref is None:
            return result
        return replace(result, graph_bindings=compiled_bindings, use_ref=_use_ref)

    _reject_unsupported_variable_length_where_pattern_predicates(query)
    _reject_variable_length_path_alias_references(query, params=params)
    query = _rewrite_where_pattern_predicates_to_matches(query)
    _reject_unsupported_where_expr_forms(query)
    _reject_nonterminal_variable_length_relationship_patterns(query)
    if query.reentry_matches:
        return _attach_graph_context(_compile_bounded_reentry_query(query, params=params))
    if query.call is not None:
        return _attach_graph_context(_compile_call_query(query, params=params))
    if query.row_sequence:
        return _attach_graph_context(_lower_row_only_sequence(query, params=params))
    if (
        len(query.matches) == 1
        and _is_node_connected_multi_pattern_clause(query.matches[0])
        and (
            _is_connected_multi_pattern_clause(query.matches[0])
            or _query_has_aggregate_stage(query, params=params)
        )
        and not _query_requires_general_lowering_for_connected_join(query, params=params)
    ):
        return _attach_graph_context(_compile_connected_match_join(query, params=params))
    if _is_connected_optional_match_query(query):
        return _attach_graph_context(_compile_connected_optional_match(query, params=params))

    merged_match = _merged_match_clause(query)
    lowered = (
        lower_match_query(query, params=params)
        if merged_match is not None
        else LoweredCypherMatch(query=[], where=[])
    )

    if query.with_stages:
        alias_targets = _alias_target(lowered.query)
        binding_row_aliases = _binding_row_aliases_for_match(query.match, alias_targets=alias_targets)
        _reject_variable_length_relationship_alias_path_carriers(
            query,
            alias_targets=alias_targets,
            params=params,
        )
        first_stage = query.with_stages[0]
        row_steps, scope = _build_initial_row_scope(
            query,
            lowered,
            stage_clause=first_stage.clause,
            stage_order_by=first_stage.order_by,
            params=params,
        )

        for stage in query.with_stages:
            if scope.mode == "match_alias":
                stage_steps, scope, _ = _lower_match_alias_stage(
                    stage,
                    scope=scope,
                    params=params,
                    final_stage=False,
                )
            else:
                stage_steps, scope = _lower_row_column_stage(stage, scope=scope, params=params)
            row_steps.extend(stage_steps)

        final_stage = ProjectionStage(
            clause=query.return_,
            where=None,
            order_by=query.order_by,
            skip=query.skip,
            limit=query.limit,
            span=query.return_.span,
        )
        result_projection: Optional[ResultProjectionPlan] = None
        if scope.mode == "match_alias":
            stage_steps, _next_scope, result_projection = _lower_match_alias_stage(
                final_stage,
                scope=scope,
                params=params,
                final_stage=True,
            )
            row_steps.extend(stage_steps)
        else:
            stage_steps, _next_scope = _lower_row_column_stage(final_stage, scope=scope, params=params)
            row_steps.extend(stage_steps)

        return _attach_graph_context(CompiledCypherQuery(
            Chain(row_steps if binding_row_aliases else lowered.query + row_steps, where=lowered.where),
            seed_rows=scope.seed_rows,
            result_projection=result_projection,
        ))

    if merged_match is not None and not query.unwinds:
        alias_targets = _alias_target(lowered.query)
        binding_row_aliases = _binding_row_aliases_for_match(query.match, alias_targets=alias_targets)
        _reject_variable_length_relationship_alias_path_carriers(
            query,
            alias_targets=alias_targets,
            params=params,
        )
        duplicated_aliases = _duplicate_node_aliases(merged_match)
        _reject_duplicate_alias_row_refs(
            query,
            alias_targets=alias_targets,
            duplicated_aliases=duplicated_aliases,
            params=params,
        )
        has_aggregates = any(
            _aggregate_spec(item, params=params, alias_targets=alias_targets) is not None
            or _post_aggregate_expr_plan(item, params=params, alias_targets=alias_targets) is not None
            for item in query.return_.items
        )
        if not has_aggregates:
            try:
                active = _active_match_alias(
                    query,
                    alias_targets=alias_targets,
                    allowed_match_aliases=binding_row_aliases or None,
                    params=params,
                )
            except GFQLValidationError as exc:
                active = next(iter(alias_targets)) if alias_targets else None
                _multi_alias_exc2: Optional[GFQLValidationError] = exc
            else:
                _multi_alias_exc2 = None
            try:
                plan = _build_projection_plan(
                    query.return_,
                    alias_targets=alias_targets,
                    active_alias=active,
                    params=params,
                )
            except GFQLValidationError:
                if binding_row_aliases:
                    plan = None
                else:
                    raise
            if plan is None:
                return _attach_graph_context(_lower_general_row_projection(query, lowered, params=params))
            if _multi_alias_exc2 is not None:
                if not _can_lower_multi_alias_projection_bindings(plan, alias_targets=alias_targets):
                    if binding_row_aliases:
                        return _attach_graph_context(_lower_general_row_projection(query, lowered, params=params))
                    raise _multi_alias_exc2
            seed_alias = _single_node_seed_alias(query.matches[0]) if len(query.matches) == 2 else None
            if (
                seed_alias is not None
                and query.matches[0].optional is False
                and query.matches[1].optional is True
                and plan.source_alias == seed_alias
            ):
                raise _unsupported(
                    "Cypher MATCH ... OPTIONAL MATCH projections that return only the bound seed alias are not yet supported in the local compiler",
                    field=query.return_.kind,
                    value=[item.expression.text for item in query.return_.items],
                    line=query.return_.span.line,
                    column=query.return_.span.column,
                )
            empty_result_row = (
                _empty_optional_projection_row(plan)
                if len(query.matches) == 1 and query.matches[0].optional
                else None
            )
            optional_null_fill = _optional_null_fill_plan(
                query,
                lowered=lowered,
                alias_targets=alias_targets,
                plan=plan,
                params=params,
            )
            optional_only_projection = _return_references_optional_only_alias(
                query,
                alias_targets=alias_targets,
                params=params,
            )
            optional_projection_row_guard = None
            if optional_null_fill is None and optional_only_projection:
                if _where_uses_optional_only_label_predicate(query):
                    raise _unsupported(
                        "Cypher OPTIONAL MATCH label filters over optional-only aliases are not yet supported when projecting optional aliases",
                        field=query.return_.kind,
                        value=[item.expression.text for item in query.return_.items],
                        line=query.return_.span.line,
                        column=query.return_.span.column,
                    )
                optional_projection_row_guard = _optional_projection_row_guard_plan(
                    query,
                    params=params,
                )
                if seed_alias is None and optional_projection_row_guard is None:
                    raise _unsupported(
                        "Cypher MATCH ... OPTIONAL MATCH projections that require null-extension for optional aliases are only supported from a single-node seed MATCH",
                        field=query.return_.kind,
                        value=[item.expression.text for item in query.return_.items],
                        line=query.return_.span.line,
                        column=query.return_.span.column,
                    )
            return _attach_graph_context(CompiledCypherQuery(
                Chain(_lower_projection_chain(query, lowered, params=params, plan=plan), where=lowered.where),
                seed_rows=False,
                result_projection=_result_projection_plan(plan, alias_targets=alias_targets),
                empty_result_row=empty_result_row,
                optional_null_fill=optional_null_fill,
                optional_projection_row_guard=optional_projection_row_guard,
            ))

    alias_targets = _alias_target(lowered.query)
    _reject_variable_length_relationship_alias_path_carriers(
        query,
        alias_targets=alias_targets,
        params=params,
    )
    return _attach_graph_context(_lower_general_row_projection(query, lowered, params=params))
