from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import re
from typing import AbstractSet, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
from typing_extensions import Literal
import pandas as pd

from graphistry.compute.ast import (
    ASTCall,
    ASTEdge,
    ASTObject,
    ASTNode,
    anti_semi_apply,
    count_table,
    distinct,
    drop_cols,
    e_forward,
    e_reverse,
    e_undirected,
    group_by,
    limit,
    order_by,
    return_,
    rows,
    search_any,
    semi_apply_mark,
    serialize_binding_ops,
    select,
    skip,
    unwind,
    where_rows,
    with_,
)
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.defer_codes import LOGICAL_PLAN_DEFER_OPTIONAL_MATCH_REENTRY
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.bound_ir import BoundIR, ScopeFrame
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import (
    Join as LogicalJoin,
    LogicalPlan,
    PatternMatch,
    ProcedureCall as LogicalProcedureCall,
    ProcedureOutputColumn as LogicalProcedureOutputColumn,
    Project as LogicalProject,
    RowSchema as LogicalRowSchema,
)
from graphistry.compute.gfql.ir.query_graph import ConnectedComponent, OptionalArm, QueryGraph
from graphistry.compute.gfql.ir.types import ScalarType
from graphistry.compute.gfql.ir.verifier import verify as verify_logical_plan
from graphistry.compute.gfql.logical_planner import LogicalPlanner
from graphistry.compute.predicates.ASTPredicate import ASTPredicate
from graphistry.compute.predicates.comparison import eq, ge, gt, isna, le, lt, ne, notna
from graphistry.compute.predicates.is_in import is_in
from graphistry.compute.predicates.logical import all_of
from graphistry.compute.predicates.str import contains as str_contains, endswith, fullmatch, never_match, startswith
from graphistry.compute.gfql.cypher.parser import _mask_quoted_backticked_and_commented_for_scan
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
    _rebuild_expr_node,
    collect_identifiers,
    parse_expr,
    walk_expr_nodes,
)
from graphistry.compute.gfql.cypher.reentry_plan import ReentryPlan
from graphistry.compute.gfql.cypher.ast import (
    BooleanExpr,
    CypherGraphQuery,
    CypherLiteral,
    CypherQuery,
    CypherUnionQuery,
    ExpressionText,
    GraphBinding,
    GraphConstructor,
    LabelRef,
    MatchClause,
    NodePattern,
    ParameterRef,
    OrderByClause,
    PatternElement,
    PathPatternKind,
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
from graphistry.compute.gfql.cypher._boolean_expr_text import boolean_expr_to_text
from graphistry.compute.gfql.cypher.expression_text import (
    cypher_literal_expr_text as _cypher_literal_expr_text,
    render_expr_node as _render_expr_node,
)
from graphistry.compute.gfql.cypher.call_procedures import (
    CompiledCypherProcedureCall,
    ProcedureOutputColumn as CompiledProcedureOutputColumn,
    compile_cypher_call,
)
from graphistry.compute.gfql.cypher.ast_normalizer import ASTNormalizer
from graphistry.compute.gfql.cypher.shortest_path_aliases import (
    _ShortestPathAliasSpec,
    _is_variable_length_relationship_pattern,
    _match_pattern_alias_kinds,
    _shortest_path_alias_specs,
)
from graphistry.compute.gfql.cypher.shortest_path_guards import (
    reject_shortest_path_alias_references_after_follow_on_match,
)
from graphistry.compute.gfql.string_literals import render_cypher_string_literal
from graphistry.compute.gfql.temporal.durations import resolve_duration_text_property
from graphistry.compute.gfql.temporal.folding import (
    fold_temporal_constructor_ast,
    rewrite_temporal_constructors_in_expr,
)
from graphistry.compute.gfql.same_path_types import NODE_IDENTITY_COLUMN, WhereComparison, col, compare, where_to_row_expr
from graphistry.compute.gfql.cypher.reentry import naming as _reentry_naming
from graphistry.compute.gfql.cypher.reentry import scope as _reentry_scope


@dataclass(frozen=True)
class LoweredCypherMatch:
    query: List[ASTObject]
    where: List[WhereComparison]
    row_where: Optional[ExpressionText] = None
    row_pre_filters: Tuple[ASTCall, ...] = ()


_CYPHER_INT64_MIN = -(2**63)
_CYPHER_INT64_MAX = (2**63) - 1


@dataclass(frozen=True)
class CompiledCypherPostProcessing:
    result_projection: Optional["ResultProjectionPlan"] = None
    empty_result_row: Optional[Dict[str, Any]] = None
    optional_null_fill: Optional["OptionalNullFillPlan"] = None
    optional_projection_row_guard: Optional["OptionalProjectionRowGuardPlan"] = None


@dataclass(frozen=True)
class CompiledCypherExecutionExtras:
    connected_optional_match: Optional["ConnectedOptionalMatchPlan"] = None
    connected_match_join: Optional["ConnectedMatchJoinPlan"] = None
    query_graph: Optional[QueryGraph] = None
    start_nodes_query: Optional["CompiledCypherQuery"] = None
    optional_reentry: bool = False
    reentry_plan: Optional[ReentryPlan] = None
    scope_stack: Tuple[ScopeFrame, ...] = ()
    logical_plan: Optional[LogicalPlan] = None
    logical_plan_defer_reason: Optional[str] = None
    logical_plan_defer_code: Optional[str] = None


@dataclass(frozen=True)
class CompiledGraphResidualFilter:
    alias: str
    kind: Literal["node", "edge"]
    expr: str
    pre_filters: Tuple[ASTCall, ...] = ()


@dataclass(frozen=True)
class CompiledCypherQuery:
    chain: Chain
    seed_rows: bool = False
    procedure_call: Optional[CompiledCypherProcedureCall] = None
    post_processing: Optional[CompiledCypherPostProcessing] = None
    execution_extras: Optional[CompiledCypherExecutionExtras] = None
    graph_bindings: Tuple["CompiledGraphBinding", ...] = ()
    use_ref: Optional[str] = None
    graph_residual_filters: Tuple[CompiledGraphResidualFilter, ...] = ()

    @property
    def result_projection(self) -> Optional["ResultProjectionPlan"]:
        return None if self.post_processing is None else self.post_processing.result_projection

    @property
    def empty_result_row(self) -> Optional[Dict[str, Any]]:
        return None if self.post_processing is None else self.post_processing.empty_result_row

    @property
    def optional_null_fill(self) -> Optional["OptionalNullFillPlan"]:
        return None if self.post_processing is None else self.post_processing.optional_null_fill

    @property
    def optional_projection_row_guard(self) -> Optional["OptionalProjectionRowGuardPlan"]:
        return None if self.post_processing is None else self.post_processing.optional_projection_row_guard

    @property
    def start_nodes_query(self) -> Optional["CompiledCypherQuery"]:
        return None if self.execution_extras is None else self.execution_extras.start_nodes_query

    @property
    def optional_reentry(self) -> bool:
        return False if self.execution_extras is None else self.execution_extras.optional_reentry

    @property
    def reentry_plan(self) -> Optional[ReentryPlan]:
        return None if self.execution_extras is None else self.execution_extras.reentry_plan

    @property
    def logical_plan(self) -> Optional[LogicalPlan]:
        return None if self.execution_extras is None else self.execution_extras.logical_plan

    @property
    def logical_plan_defer_reason(self) -> Optional[str]:
        return None if self.execution_extras is None else self.execution_extras.logical_plan_defer_reason

    @property
    def logical_plan_defer_code(self) -> Optional[str]:
        return None if self.execution_extras is None else self.execution_extras.logical_plan_defer_code


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
    logical_plan: Optional[LogicalPlan] = None
    logical_plan_defer_reason: Optional[str] = None
    graph_residual_filters: Tuple[CompiledGraphResidualFilter, ...] = ()

@dataclass(frozen=True)
class CompiledCypherGraphQuery:
    """A query whose final result is a graph (from standalone GRAPH { })."""
    graph_bindings: Tuple[CompiledGraphBinding, ...]
    chain: Chain
    procedure_call: Optional[CompiledCypherProcedureCall] = None
    use_ref: Optional[str] = None
    logical_plan: Optional[LogicalPlan] = None
    logical_plan_defer_reason: Optional[str] = None
    graph_residual_filters: Tuple[CompiledGraphResidualFilter, ...] = ()

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


def _normalize_post_processing(
    post_processing: CompiledCypherPostProcessing,
) -> Optional[CompiledCypherPostProcessing]:
    if (
        post_processing.result_projection is None
        and post_processing.empty_result_row is None
        and post_processing.optional_null_fill is None
        and post_processing.optional_projection_row_guard is None
    ):
        return None
    return post_processing


def _normalize_execution_extras(
    execution_extras: CompiledCypherExecutionExtras,
) -> Optional[CompiledCypherExecutionExtras]:
    if (
        execution_extras.connected_optional_match is None
        and execution_extras.connected_match_join is None
        and execution_extras.query_graph is None
        and execution_extras.start_nodes_query is None
        and execution_extras.optional_reentry is False
        and execution_extras.reentry_plan is None
        and execution_extras.scope_stack == ()
        and execution_extras.logical_plan is None
        and execution_extras.logical_plan_defer_reason is None
        and execution_extras.logical_plan_defer_code is None
    ):
        return None
    return execution_extras


def _execution_extras_with(
    compiled_query: CompiledCypherQuery,
    *,
    connected_optional_match: Optional["ConnectedOptionalMatchPlan"] = None,
    connected_match_join: Optional["ConnectedMatchJoinPlan"] = None,
    query_graph: Optional[QueryGraph] = None,
    start_nodes_query: Optional[CompiledCypherQuery] = None,
    optional_reentry: bool = False,
    reentry_plan: Optional[ReentryPlan] = None,
    scope_stack: Tuple[ScopeFrame, ...] = (),
    logical_plan: Optional[LogicalPlan] = None,
    logical_plan_defer_reason: Optional[str] = None,
    logical_plan_defer_code: Optional[str] = None,
) -> Optional[CompiledCypherExecutionExtras]:
    base = compiled_query.execution_extras or CompiledCypherExecutionExtras()
    return _normalize_execution_extras(
        replace(
            base,
            connected_optional_match=connected_optional_match,
            connected_match_join=connected_match_join,
            query_graph=query_graph,
            start_nodes_query=start_nodes_query,
            optional_reentry=optional_reentry,
            reentry_plan=reentry_plan,
            scope_stack=scope_stack,
            logical_plan=logical_plan,
            logical_plan_defer_reason=logical_plan_defer_reason,
            logical_plan_defer_code=logical_plan_defer_code,
        )
    )


def _logical_schema_for_call_outputs(
    output_columns: Sequence[CompiledProcedureOutputColumn],
) -> LogicalRowSchema:
    return LogicalRowSchema(
        columns={
            output.output_name: ScalarType(kind="unknown", nullable=True)
            for output in output_columns
        }
    )


def _logical_plan_from_compiled_call(compiled_call: CompiledCypherProcedureCall) -> LogicalPlan:
    return LogicalProcedureCall(
        op_id=1,
        procedure=compiled_call.procedure,
        backend=compiled_call.backend,
        algorithm=compiled_call.algorithm,
        call_function=compiled_call.call_function,
        result_kind=compiled_call.result_kind,
        row_kind=compiled_call.row_kind,
        output_columns=tuple(
            LogicalProcedureOutputColumn(
                source_name=output.source_name,
                output_name=output.output_name,
            )
            for output in compiled_call.output_columns
        ),
        call_params=dict(compiled_call.call_params),
        output_schema=_logical_schema_for_call_outputs(compiled_call.output_columns),
    )


def _verify_selected_logical_plan(logical_plan: LogicalPlan) -> None:
    diagnostics = verify_logical_plan(logical_plan)
    if diagnostics:
        summary = "; ".join(diagnostic.message for diagnostic in diagnostics[:3])
        raise GFQLValidationError(
            ErrorCode.E108,
            "LogicalPlan verification failed for a query shape selected for LogicalPlan routing",
            field="logical_plan",
            value=summary,
            suggestion="Keep using covered MATCH/WHERE/WITH/RETURN/UNWIND/CALL shapes until richer plan lowering lands.",
            language="cypher",
        )


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
    seed_rows: bool
    relationship_count: int
    allowed_match_aliases: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _BoundLoweringContext:
    params: Optional[Mapping[str, Any]]
    visible_aliases: AbstractSet[str]
    nullable_aliases: AbstractSet[str]
    entity_kinds: Mapping[str, Literal["node", "edge", "scalar"]]


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


def _merge_bound_params(
    *,
    params: Optional[Mapping[str, Any]],
    bound_params: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    if not bound_params:
        return params
    # Binder metadata keys are not user runtime params.
    merged: Dict[str, Any] = {key: value for key, value in bound_params.items() if not key.startswith("_binder_")}
    if params:
        merged.update(params)
    return merged


def _bound_visible_aliases(bound_ir: BoundIR) -> AbstractSet[str]:
    if not bound_ir.scope_stack:
        return frozenset()
    # Scope narrowing must respect active scope boundaries.
    return frozenset(bound_ir.scope_stack[-1].visible_vars)


def _bound_nullable_aliases(bound_ir: BoundIR) -> AbstractSet[str]:
    return frozenset(
        alias
        for alias, variable in bound_ir.semantic_table.variables.items()
        if variable.nullable
    )


def _bound_entity_kinds(
    bound_ir: BoundIR,
) -> Mapping[str, Literal["node", "edge", "scalar"]]:
    return {alias: variable.entity_kind for alias, variable in bound_ir.semantic_table.variables.items()}


def _build_bound_lowering_context(
    *,
    bound_ir: BoundIR,
    params: Optional[Mapping[str, Any]],
) -> _BoundLoweringContext:
    return _BoundLoweringContext(
        params=_merge_bound_params(params=params, bound_params=bound_ir.params),
        visible_aliases=_bound_visible_aliases(bound_ir),
        nullable_aliases=_bound_nullable_aliases(bound_ir),
        entity_kinds=_bound_entity_kinds(bound_ir),
    )


def _apply_bound_scope_membership(
    binding_row_aliases: Set[str],
    *,
    alias_targets: Mapping[str, ASTObject],
    bound_visible_aliases: AbstractSet[str],
) -> Set[str]:
    if not bound_visible_aliases:
        return set(binding_row_aliases)
    visible = set(alias_targets.keys()) & set(bound_visible_aliases)
    # Scope membership narrowing must not promote plain source/table projections.
    if not binding_row_aliases:
        return set()
    narrowed = set(binding_row_aliases) & visible
    return narrowed if narrowed else set(binding_row_aliases)


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
    def _rewrite_match_text(text: str) -> str:
        match = _CYPHER_CHAINED_COMPARISON_RE.fullmatch(text)
        if match is None:
            return text
        left = match.group("left").strip()
        middle = match.group("middle").strip()
        right = match.group("right").strip()
        if any(token in segment.upper() for token in {" AND", " OR", " XOR"} for segment in (left, middle, right)):
            return text
        return f"({left} {match.group('op1')} {middle}) AND ({middle} {match.group('op2')} {right})"

    def _replace_case_condition(match: re.Match[str]) -> str:
        condition = match.group("condition")
        return match.group(0) if (rewritten_condition := _rewrite_match_text(condition)) == condition else f"{match.group('prefix')}{rewritten_condition}{match.group('suffix')}"

    case_rewritten = _rewrite_unquoted_expr_segments(expr_text, rewrite=lambda segment: re.sub(r"(?P<prefix>\bWHEN\s+)(?P<condition>.*?)(?P<suffix>\s+THEN\b)", _replace_case_condition, segment, flags=re.IGNORECASE | re.DOTALL))
    return case_rewritten if case_rewritten != expr_text or expr_text.lstrip().upper().startswith("CASE ") else _rewrite_match_text(expr_text)

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
    if isinstance(item, ExprLiteral):
        return isinstance(item.value, (list, dict))
    if isinstance(item, (ListLiteral, MapLiteral)):
        return True
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
        if isinstance(current, MapLiteral) and _contains_aggregate_call(current):
            _raise("Cypher aggregate expressions inside map literals are not supported in the local compiler yet")
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
    from graphistry.compute.gfql.cypher import projection_planning as _projection

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
            alias_name, prop = _projection._qualified_ref_from_node(
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
            return PropertyAccessExpr(_rewrite(node_in.value), node_in.property)
        if isinstance(node_in, Identifier) and "." not in node_in.name and node_in.name in alias_targets:
            target = alias_targets[node_in.name]
            prop = NODE_IDENTITY_COLUMN if isinstance(target, ASTNode) else "__gfql_edge_index_0__"
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
        if (
            node.op == "in"
            and isinstance(node.left, Identifier)
            and node.left.name in alias_targets
        ):
            return BinaryOp(
                node.op,
                _entity_wrapper_call(node.left.name, alias_targets=alias_targets),
                _rewrite_collection_alias_entities(node.right, alias_targets=alias_targets),
            )
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
            if _reentry_naming._is_hidden_reentry_property(node_in.property):
                return
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
                "Cypher row lowering currently supports one MATCH source alias at a time; "
                "for remaining multi-source residuals see issue #1273",
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
    if _contains_aggregate_call(node):
        raise _unsupported(
            "Cypher aggregate functions must be top-level RETURN/WITH projections or valid post-aggregate expressions",
            field=field,
            value=expr.text,
            line=expr.span.line,
            column=expr.span.column,
        )
    # openCypher integer division truncates when both operands are integer-like.
    # Runtime row expressions do not carry stable identifier type metadata here,
    # so this pass currently rewrites literal-only integer divisions.
    rewritten = _rewrite_cypher_integer_division_ast(
        node,
        integer_identifiers=frozenset(),
    )
    return _render_expr_node(rewritten)


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
    from graphistry.compute.gfql.cypher import projection_planning as _projection

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
        alias_name, prop = _projection._split_qualified_name(ident, line=expr.span.line, column=expr.span.column)
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
    reserved_temp_names: Optional[Set[str]] = None,
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

    temp_names: Set[str] = reserved_temp_names if reserved_temp_names is not None else set()
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


class _SyntheticRowGraph:
    def __init__(self, table_df: Any) -> None:
        self._nodes = table_df
        self._edges = table_df.iloc[0:0].copy()
        self._node = None
        self._source = None
        self._destination = None
        self._edge = None
        self._g = self
        self._gfql_start_nodes = None
        self._gfql_rows_base_graph = None
        self._gfql_shortest_path_backend = "auto"

    def bind(self) -> "_SyntheticRowGraph":
        return _SyntheticRowGraph(self._nodes.copy())


def _evaluate_empty_projection_row(
    row: Mapping[str, Any],
    *,
    projection_items: Sequence[Tuple[str, Any]],
) -> Optional[Dict[str, Any]]:
    from graphistry.compute.gfql.row.pipeline import _RowPipelineAdapter

    adapter = _RowPipelineAdapter(cast(Any, _SyntheticRowGraph(pd.DataFrame([dict(row)]))))
    projection = adapter.select(items=list(projection_items))
    if projection._nodes is None or len(projection._nodes) == 0:
        return None
    return projection._nodes.iloc[0].to_dict()


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
            "Cypher row lowering currently supports one MATCH source alias at a time; "
            "for remaining multi-source residuals see issue #1273",
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
            "Cypher row lowering currently supports one MATCH source alias at a time; "
            "for remaining multi-source residuals see issue #1273",
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


def _is_multi_source_match_alias_boundary_error(
    exc: GFQLValidationError,
    *,
    alias_targets: Mapping[str, ASTObject],
) -> bool:
    """Detect the #1273 one-source boundary using structured error data."""
    if exc.code != ErrorCode.E108:
        return False
    if exc.context.get("field") != "return":
        return False
    value = exc.context.get("value")
    if not isinstance(value, (list, tuple, set, frozenset)):
        return False
    alias_refs = {name for name in value if isinstance(name, str)}
    if len(alias_refs) < 2 or len(alias_refs) != len(value):
        return False
    return alias_refs <= set(alias_targets.keys())


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
    if agg_spec.func == "collect":
        return True
    if agg_spec.func == "count":
        return not agg_spec.distinct
    return False


def _collect_aggregate_specs_for_clause(
    clause: ReturnClause,
    *,
    params: Optional[Mapping[str, Any]],
    alias_targets: Mapping[str, ASTObject],
) -> List[_AggregateSpec]:
    aggregate_specs: List[_AggregateSpec] = []
    post_aggregate_temp_names: Set[str] = set()
    for item in clause.items:
        agg_spec = _aggregate_spec(item, params=params, alias_targets=alias_targets)
        if agg_spec is not None:
            aggregate_specs.append(agg_spec)
            continue
        post_agg_plan = _post_aggregate_expr_plan(
            item,
            params=params,
            alias_targets=alias_targets,
            reserved_temp_names=post_aggregate_temp_names,
        )
        if post_agg_plan is not None:
            nested_aggregate_specs, _ = post_agg_plan
            aggregate_specs.extend(nested_aggregate_specs)
    return aggregate_specs


def _requires_relationship_multiplicity_bindings(
    *,
    aggregate_specs: Sequence[_AggregateSpec],
    relationship_count: int,
) -> bool:
    return relationship_count > 0 and any(
        _is_multiplicity_sensitive_aggregate(spec)
        for spec in aggregate_specs
    )


def _match_relationship_count(clause: MatchClause) -> int:
    return sum(1 for element in _match_pattern_elements(clause) if isinstance(element, RelationshipPattern))


def _is_pure_count_star_shortcircuit(
    *,
    aggregate_specs: Sequence[_AggregateSpec],
    pre_items: Sequence[Tuple[str, Any]],
    row_steps: Sequence[ASTObject],
    query: CypherQuery,
    binding_row_aliases: AbstractSet[str],
    relationship_count: int,
    active_match_alias: Optional[str],
    alias_targets: Mapping[str, ASTObject],
) -> bool:
    """True when a RETURN is exactly ``count(*)`` over a single node/edge scan.

    Guards the count_table fast path (skip the full-frame materialize + constant-
    key group_by): the count then equals the height (or source-mask sum) of the
    active table. Requires a lone non-DISTINCT ``count(*)`` with no group keys,
    row-level WHERE, UNWIND, or multi-relationship binding, and a plain
    ``rows(table=nodes|edges[, source])`` as the only prior step. Post-aggregate
    exprs (``count(*) + 1``) compose fine: the count lands in a temp column and
    the trailing ``select`` applies the expr, same as the group_by path.
    Sound only for a pure node scan (``relationship_count == 0``) or a single
    relationship counted on its edge alias (``relationship_count == 1`` with an
    ``ASTEdge`` active alias) — exactly the cases
    ``_reject_unsound_relationship_multiplicity_aggregates`` permits; any other
    shape (node-alias-over-relationship, multi-hop paths) falls through to the
    general aggregate path (which counts bindings or rejects as unsound).
    """
    if len(aggregate_specs) != 1:
        return False
    agg = aggregate_specs[0]
    if agg.func != "count" or agg.expr_text is not None or agg.distinct:
        return False
    if pre_items or binding_row_aliases or query.unwinds:
        return False
    # Exactly the initial rows() step: any row-level WHERE or UNWIND would have
    # appended further steps, so len == 1 proves the count is over the raw scan.
    if len(row_steps) != 1:
        return False
    base = row_steps[0]
    if not (isinstance(base, ASTCall) and base.function == "rows"):
        return False
    if base.params.get("table") not in ("nodes", "edges"):
        return False
    if base.params.get("binding_ops") is not None or base.params.get("alias_endpoints") is not None:
        return False
    if relationship_count == 0:
        return True
    if (
        relationship_count == 1
        and active_match_alias is not None
        and isinstance(alias_targets.get(active_match_alias), ASTEdge)
    ):
        return True
    return False


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
    bound_nullable_aliases: Optional[AbstractSet[str]] = None,
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
    if bound_nullable_aliases:
        optional_only_aliases.update(alias for alias in bound_nullable_aliases if alias in alias_targets)
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


def _where_uses_optional_only_label_predicate(
    query: CypherQuery,
    *,
    bound_nullable_aliases: Optional[AbstractSet[str]] = None,
) -> bool:
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
    if bound_nullable_aliases:
        optional_only_aliases.update(bound_nullable_aliases)
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
                    and not _reentry_naming._is_hidden_reentry_property(node.property)
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
    hop_column: Optional[str] = None,
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
    min_hops = relationship.min_hops
    max_hops = relationship.max_hops
    # shortestPath(...*...) should admit the valid zero-hop path when the
    # endpoints are the same node, unlike generic carrier-style varlen refs.
    if hop_column is not None and relationship.to_fixed_point and min_hops is None and max_hops is None:
        min_hops = 0
    include_zero_hop_seed = (
        not relationship.to_fixed_point
        and min_hops == 0
        and max_hops == 0
    )
    hops = (
        None
        if (
            min_hops is not None
            or max_hops is not None
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
                min_hops=min_hops,
                max_hops=max_hops,
                to_fixed_point=relationship.to_fixed_point,
                label_node_hops=hop_column,
                name=relationship.variable,
                prune_to_endpoints=prune_to_endpoints,
                include_zero_hop_seed=include_zero_hop_seed,
            ),
        )
    if relationship.direction == "reverse":
        return cast(
            ASTObject,
            e_reverse(
                edge_match=edge_match,
                hops=hops,
                min_hops=min_hops,
                max_hops=max_hops,
                to_fixed_point=relationship.to_fixed_point,
                label_node_hops=hop_column,
                name=relationship.variable,
                prune_to_endpoints=prune_to_endpoints,
                include_zero_hop_seed=include_zero_hop_seed,
            ),
        )
    return cast(
        ASTObject,
        e_undirected(
            edge_match=edge_match,
            hops=hops,
            min_hops=min_hops,
            max_hops=max_hops,
            to_fixed_point=relationship.to_fixed_point,
            label_node_hops=hop_column,
            name=relationship.variable,
            prune_to_endpoints=prune_to_endpoints,
            include_zero_hop_seed=include_zero_hop_seed,
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
        merged = _merge_node_patterns(right_end, left_start)
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


def _query_has_shortest_path_patterns(query: CypherQuery) -> bool:
    return any(
        kind == "shortestPath"
        for clause in query.matches + query.reentry_matches
        for kind in _match_pattern_alias_kinds(clause)
    )


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
    for idx, pattern in enumerate(clause.patterns):
        single_clause = replace(
            clause,
            patterns=(pattern,),
            pattern_aliases=(None,),
            pattern_alias_kinds=(_match_pattern_alias_kinds(clause)[idx],),
        )
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
    if clause is not None and any(kind == "shortestPath" for kind in _match_pattern_alias_kinds(clause)):
        return {
            alias
            for alias, target in alias_targets.items()
            if isinstance(target, ASTNode)
        }
    if clause is None or _cartesian_node_only_patterns(clause) is None:
        return set()
    if len(alias_targets) <= 1:
        return set()
    if not all(isinstance(target, ASTNode) for target in alias_targets.values()):
        return set()
    return set(alias_targets.keys())


def _binding_row_aliases_for_multi_alias_whole_row_node_projection(
    query: CypherQuery,
    *,
    clause: ReturnClause,
    alias_targets: Mapping[str, ASTObject],
) -> Set[str]:
    if query.match is None or len(query.matches) > 1 or len(alias_targets) <= 1:
        return set()
    whole_row_refs = {
        item.expression.text
        for item in clause.items
        if item.expression.text in alias_targets
    }
    if len(whole_row_refs) <= 1:
        return set()
    if not all(isinstance(alias_targets.get(alias), ASTNode) for alias in whole_row_refs):
        return set()
    return set(whole_row_refs)


def _binding_row_aliases_for_row_where(
    row_where: Optional[ExpressionText],
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> Set[str]:
    if row_where is None:
        return set()
    hidden_refs = _reentry_scope._expr_hidden_reentry_aliases(
        row_where.text,
        alias_targets=alias_targets,
        params=params,
        field="where",
        line=row_where.span.line,
        column=row_where.span.column,
    )
    if hidden_refs:
        return set(alias_targets.keys())
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
    shortest_path_hop_columns = _shortest_path_relationship_hop_columns(clause)

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
                compare(col(alias, NODE_IDENTITY_COLUMN), "==", col(rewritten_alias, NODE_IDENTITY_COLUMN))
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
        hop_col = shortest_path_hop_columns.get(
            (
                element.span.line,
                element.span.column,
                element.span.end_line,
                element.span.end_column,
                element.span.start_pos,
                element.span.end_pos,
            )
        )
        is_varlen = _is_variable_length_relationship_pattern(element)
        lowered_edge = _lower_relationship(
            element,
            params=params,
            prune_to_endpoints=is_varlen and idx < last_rel_idx,
            hop_column=hop_col,
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
        pattern_alias_kinds=_match_pattern_alias_kinds(clause)[-1:] if clause.patterns else (),
    )


def _merge_non_optional_match_clauses(matches: Sequence[MatchClause]) -> MatchClause:
    if not matches:
        raise ValueError("Expected at least one MATCH clause to merge")
    merged_patterns: List[Tuple[PatternElement, ...]] = []
    merged_aliases: List[Optional[str]] = []
    merged_alias_kinds: List[PathPatternKind] = []
    merged_where: Optional[WhereClause] = None
    last_idx = len(matches) - 1
    for idx, clause in enumerate(matches):
        if clause.optional:
            raise _unsupported(
                "Cypher sequential MATCH merge currently supports only non-optional MATCH clauses",
                field="match",
                value=None,
                line=clause.span.line,
                column=clause.span.column,
            )
        if clause.where is not None:
            if idx != last_idx:
                raise _unsupported(
                    "Cypher WHERE on intermediate MATCH clauses is not yet supported for sequential MATCH merge",
                    field="where",
                    value=None,
                    line=clause.where.span.line,
                    column=clause.where.span.column,
                )
            merged_where = clause.where
        merged_patterns.extend(clause.patterns)
        if clause.pattern_aliases and len(clause.pattern_aliases) == len(clause.patterns):
            merged_aliases.extend(clause.pattern_aliases)
        else:
            merged_aliases.extend([None] * len(clause.patterns))
        merged_alias_kinds.extend(_match_pattern_alias_kinds(clause))
    return MatchClause(
        patterns=tuple(merged_patterns),
        span=matches[-1].span,
        optional=False,
        pattern_aliases=tuple(merged_aliases),
        where=merged_where,
        pattern_alias_kinds=tuple(merged_alias_kinds),
    )


def _merged_match_clause(query: CypherQuery) -> Optional[MatchClause]:
    if not query.matches:
        return None
    if len(query.matches) == 1:
        return query.matches[0]
    try:
        seed_bindings = _seed_node_bindings(query.matches[:-1])
    except GFQLValidationError as exc:
        if "Only node-only pre-binding MATCH clauses are supported before the final connected MATCH in this phase" not in str(exc):
            raise
        return _merge_non_optional_match_clauses(query.matches)
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
    if op == "regex":
        # openCypher/neo4j `=~`: Java-regex, full-string/anchored match → fullmatch.
        # Inline flags in the pattern (e.g. `(?i)`) are honored by the regex engine.
        return never_match() if value is None else fullmatch(str(value), na=False)
    raise ValueError(f"Unsupported predicate op: {op}")


def _alias_table(
    target: ASTObject,
    *,
    alias: str,
    line: int,
    column: int,
    semantic_entity_kinds: Optional[Mapping[str, Literal["node", "edge", "scalar"]]] = None,
) -> str:
    if semantic_entity_kinds is not None:
        semantic_kind = semantic_entity_kinds.get(alias)
        if semantic_kind == "node":
            return "nodes"
        if semantic_kind == "edge":
            return "edges"
        # A projection alias can shadow a MATCH alias name (for example,
        # `RETURN a.id IS NOT NULL AS a`) while the source alias `a` is still a
        # node/edge target in the current match context. In that case, defer to
        # the structural AST target below instead of hard-failing on the
        # projected scalar alias kind.
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


def _lower_order_by_clause(
    clause: OrderByClause,
    *,
    plan: _ProjectionPlan,
    alias_targets: Optional[Mapping[str, ASTObject]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> ASTObject:
    from graphistry.compute.gfql.cypher import projection_planning as _projection

    keys: List[Tuple[str, str]] = []
    projection_output_names = _projection._projection_output_names(plan)
    for item in clause.items:
        try:
            alias_name, prop = _projection._projection_ref_from_expr(
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

_SEARCH_ANY_CALL_RE = re.compile(
    r"\bsearchAny\s*\(\s*([A-Za-z_]\w*)\s*,\s*"
    r"('(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
    r"\s*(?:,\s*\{([^{}]*)\})?\s*\)",
    re.IGNORECASE,
)
_SEARCH_ANY_OPT_KEYS = {"casesensitive": "case_sensitive", "regex": "regex", "columns": "columns"}
_SEARCH_ANY_COLUMNS_RE = re.compile(
    r"^\[\s*(?:'[^']*'|\"[^\"]*\")(?:\s*,\s*(?:'[^']*'|\"[^\"]*\"))*\s*\]$")
_SEARCH_ANY_COL_ITEM_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def _parse_search_any_opts(opts_text: str, *, line: int, column: int) -> Dict[str, Any]:
    """Parse searchAny's option map literal — strict: unknown keys error listing the
    valid ones (persona pass: predictable options beat silent typo-tolerance)."""
    out: Dict[str, Any] = {}
    if not opts_text.strip():
        return out
    depth = 0
    parts: List[str] = []
    cur = ""
    for ch in opts_text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    for part in parts:
        m = re.match(r"\s*([A-Za-z_]\w*)\s*:\s*(.+?)\s*$", part)
        key = m.group(1) if m else part.strip()
        canon = _SEARCH_ANY_OPT_KEYS.get(key.lower()) if m else None
        if m is None or canon is None:
            raise _unsupported(
                f"searchAny got an unsupported option {key!r}",
                field="where",
                value=opts_text,
                line=line,
                column=column,
            )
        val_text = m.group(2)
        if canon in ("case_sensitive", "regex"):
            low = val_text.lower()
            if low not in ("true", "false"):
                raise _unsupported(
                    f"searchAny option {key!r} must be true or false",
                    field="where", value=val_text, line=line, column=column,
                )
            out[canon] = low == "true"
        else:
            if not _SEARCH_ANY_COLUMNS_RE.match(val_text):
                raise _unsupported(
                    "searchAny option 'columns' must be a list of string literals",
                    field="where", value=val_text, line=line, column=column,
                )
            out[canon] = [a or b for a, b in _SEARCH_ANY_COL_ITEM_RE.findall(val_text)]
    return out


def _lift_search_any_from_row_where(
    expr: ExpressionText,
    *,
    alias_targets: Mapping[str, ASTObject],
    existing_cols: AbstractSet[str],
) -> Tuple[str, List[ASTCall]]:
    """viz-filter L2-b: rewrite each ``searchAny(alias, 'term'[, {opts}])`` in the
    WHERE row-expression into a fresh boolean MARKER column + a ``search_any`` row
    pre-filter (exactly the pattern-predicate marker mechanism), so the remaining
    boolean expression composes through AND/OR/NOT unchanged.

    Matches are found against a literal/comment-masked copy of the text so a
    ``searchAny(...)`` occurring INSIDE a string literal is data, not a call
    (wave-1 B1: the bare ``.sub`` rewrote literal contents — silent wrong answer)."""
    calls: List[ASTCall] = []
    used: set = set(existing_cols)

    def _sub(m: "re.Match[str]") -> str:
        alias, term_lit, opts_text = m.group(1), m.group(2), m.group(3) or ""
        if alias not in alias_targets:
            raise _unsupported(
                f"searchAny references alias {alias!r} not bound in the active MATCH scope",
                field="where", value=alias,
                line=expr.span.line, column=expr.span.column,
            )
        term = term_lit[1:-1].replace("\\'", "'").replace('\\"', '"')
        opts = _parse_search_any_opts(
            opts_text, line=expr.span.line, column=expr.span.column)
        base = f"__gfql_search_any_{expr.span.line}_{expr.span.column}_{len(calls)}__"
        out_col = _fresh_temp_name(used, base)
        used.add(out_col)
        calls.append(search_any(alias=alias, term=term, out_col=out_col, **opts))
        return out_col

    masked = _mask_quoted_backticked_and_commented_for_scan(expr.text)
    parts: List[str] = []
    last = 0
    pos = 0
    while True:
        m = _SEARCH_ANY_CALL_RE.search(expr.text, pos)
        if m is None:
            break
        if masked[m.start()] != expr.text[m.start()]:
            # starts inside a string literal / comment: data, not a call — resume
            # just past the START (not the end: an in-literal pseudo-match can span
            # across quotes and swallow a real call inside its span; wave-2 S1)
            pos = m.start() + 1
            continue
        parts.append(expr.text[last:m.start()])
        parts.append(_sub(m))
        last = m.end()
        pos = m.end()
    parts.append(expr.text[last:])
    new_text = "".join(parts)
    # Anything still spelled searchAny( outside literals is a form the lift can't
    # parse (e.g. a $param term — the persona-pass signature is literal-only): fail
    # loudly here instead of leaking an unknown function downstream (wave-1 I5).
    masked_new = _mask_quoted_backticked_and_commented_for_scan(new_text)
    if re.search(r"\bsearchAny\s*\(", masked_new, re.IGNORECASE):
        raise _unsupported(
            "searchAny requires a string-literal term and literal options, e.g. "
            "searchAny(n, 'term', {caseSensitive: false}) — parameters ($p) and "
            "expressions are not supported",
            field="where", value=expr.text,
            line=expr.span.line, column=expr.span.column,
        )
    return new_text, calls


def _append_match_row_where(
    row_steps: List[ASTObject],
    *,
    lowered: LoweredCypherMatch,
    alias_targets: Mapping[str, ASTObject],
    active_alias: Optional[str],
    allowed_match_aliases: Optional[AbstractSet[str]],
    params: Optional[Mapping[str, Any]],
) -> None:
    if lowered.row_pre_filters:
        row_steps.extend(lowered.row_pre_filters)
    if allowed_match_aliases is not None:
        row_steps.extend(where_rows(expr=where_to_row_expr(clause)) for clause in lowered.where)
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
    bound_visible_aliases: AbstractSet[str] = frozenset(),
    semantic_entity_kinds: Optional[Mapping[str, Literal["node", "edge", "scalar"]]] = None,
) -> List[ASTObject]:
    from graphistry.compute.gfql.cypher import projection_planning as _projection

    alias_targets = _alias_target(lowered.query)
    binding_row_aliases = _binding_row_aliases_for_match(query.match, alias_targets=alias_targets)
    binding_row_aliases.update(
        _binding_row_aliases_for_row_where(
            lowered.row_where,
            alias_targets=alias_targets,
            params=params,
        )
    )
    binding_row_aliases.update(
        _reentry_scope._binding_row_aliases_for_hidden_reentry_refs(
            unwinds=query.unwinds,
            clause=query.return_,
            order_by_clause=query.order_by,
            alias_targets=alias_targets,
            params=params,
        )
    )
    pre_scope_binding_row_aliases = set(binding_row_aliases)
    binding_row_aliases = _apply_bound_scope_membership(
        binding_row_aliases,
        alias_targets=alias_targets,
        bound_visible_aliases=bound_visible_aliases,
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
        plan = _projection._build_projection_plan(
            query.return_,
            alias_targets=alias_targets,
            active_alias=active,
            params=params,
            semantic_entity_kinds=semantic_entity_kinds,
        )
        if _multi_alias_exc is not None:
            if not _projection._can_lower_multi_alias_projection_bindings(plan, alias_targets=alias_targets):
                raise _multi_alias_exc

    allowed_match_aliases = ({plan.source_alias} | plan.all_source_aliases | binding_row_aliases) if plan.all_source_aliases is not None else binding_row_aliases
    if plan.all_source_aliases is not None or pre_scope_binding_row_aliases or lowered.row_pre_filters:
        row_steps: List[ASTObject] = [rows(binding_ops=serialize_binding_ops(lowered.query))]
    else:
        row_steps = [rows(table=plan.table, source=plan.source_alias)]
    _append_match_row_where(
        row_steps,
        lowered=lowered,
        alias_targets=alias_targets,
        active_alias=plan.source_alias,
        allowed_match_aliases=(allowed_match_aliases | pre_scope_binding_row_aliases) or None,
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
    if pre_scope_binding_row_aliases or lowered.row_pre_filters:
        return row_steps
    return lowered.query + row_steps


def _build_initial_row_scope(
    query: CypherQuery,
    lowered: LoweredCypherMatch,
    *,
    stage_clause: ReturnClause,
    stage_order_by: Optional[OrderByClause],
    params: Optional[Mapping[str, Any]],
    bound_visible_aliases: AbstractSet[str] = frozenset(),
    semantic_entity_kinds: Optional[Mapping[str, Literal["node", "edge", "scalar"]]] = None,
) -> Tuple[List[ASTObject], _StageScope]:
    from graphistry.compute.gfql.cypher import projection_planning as _projection

    alias_targets = _alias_target(lowered.query) if query.match is not None else {}
    merged_match = _merged_match_clause(query)
    relationship_count = _match_relationship_count(merged_match) if merged_match is not None else 0
    binding_row_aliases = _binding_row_aliases_for_match(query.match, alias_targets=alias_targets)
    stage_aggregate_specs = _collect_aggregate_specs_for_clause(
        stage_clause,
        params=params,
        alias_targets=alias_targets,
    )
    # Admit first-stage multi-alias non-aggregate WITH projections (shape A, #1273)
    # by routing through the bindings-row path when multiple MATCH node aliases are
    # referenced together in scalar expressions.
    stage_has_aggregates = bool(stage_aggregate_specs)
    if (
        not stage_has_aggregates
        and len(alias_targets) > 1
    ):
        stage_non_aggregate_refs: Set[str] = set()
        for item in stage_clause.items:
            expr_text = item.expression.text
            if expr_text == "*":
                continue
            # Keep whole-row alias pipelines on the existing conservative path.
            # This admission is scoped to multi-alias scalar/property projections.
            if expr_text in alias_targets:
                continue
            stage_non_aggregate_refs.update(
                _expr_non_aggregate_match_aliases(
                    expr_text,
                    alias_targets=alias_targets,
                    params=params,
                    field=stage_clause.kind,
                    line=item.span.line,
                    column=item.span.column,
                )
            )
        if len(stage_non_aggregate_refs) > 1 and all(
            isinstance(alias_targets.get(alias), ASTNode)
            for alias in stage_non_aggregate_refs
        ):
            binding_row_aliases.update(stage_non_aggregate_refs)
    # For connected non-cartesian MATCH, allow first-stage multi-whole-row node
    # projections to use bindings rows (#880 / #1393).
    if not binding_row_aliases:
        binding_row_aliases.update(
            _binding_row_aliases_for_multi_alias_whole_row_node_projection(
                query,
                clause=stage_clause,
                alias_targets=alias_targets,
            )
        )
    if _requires_relationship_multiplicity_bindings(
        aggregate_specs=stage_aggregate_specs,
        relationship_count=relationship_count,
    ):
        return_has_whole_row_alias = any(
            item.expression.text in alias_targets
            for item in query.return_.items
        )
        if len(query.with_stages) == 1 and not return_has_whole_row_alias:
            binding_row_aliases = set(alias_targets.keys())
    binding_row_aliases.update(
        _binding_row_aliases_for_row_where(
            lowered.row_where,
            alias_targets=alias_targets,
            params=params,
        )
    )
    # Hidden reentry property references (for example `c2.__cypher_reentry_bid__`)
    # must run on bindings rows so carry columns survive trailing WITH narrowing.
    binding_row_aliases.update(
        _reentry_scope._binding_row_aliases_for_hidden_reentry_refs(
            unwinds=query.unwinds,
            clause=stage_clause,
            order_by_clause=stage_order_by,
            alias_targets=alias_targets,
            params=params,
        )
    )
    binding_row_aliases = _apply_bound_scope_membership(
        binding_row_aliases,
        alias_targets=alias_targets,
        bound_visible_aliases=bound_visible_aliases,
    )
    try:
        active_match_alias = _active_match_alias_for_stage(
            unwinds=query.unwinds,
            clause=stage_clause,
            order_by_clause=stage_order_by,
            alias_targets=alias_targets,
            allowed_match_aliases=binding_row_aliases or None,
            params=params,
        )
    except GFQLValidationError as exc:
        if not _is_multi_source_match_alias_boundary_error(exc, alias_targets=alias_targets):
            raise
        whole_row_alias_refs = {
            item.expression.text for item in stage_clause.items if item.expression.text in alias_targets
        }
        if len(whole_row_alias_refs) < 2:
            raise
        fallback_alias = next(iter(alias_targets)) if alias_targets else None
        try:
            fallback_plan = _projection._build_projection_plan(
                stage_clause,
                alias_targets=alias_targets,
                active_alias=fallback_alias,
                params=params,
                semantic_entity_kinds=semantic_entity_kinds,
            )
        except GFQLValidationError:
            raise exc
        if not _projection._can_lower_multi_alias_projection_bindings(fallback_plan, alias_targets=alias_targets):
            raise exc
        active_match_alias = fallback_plan.source_alias
    seed_rows = query.match is None

    if active_match_alias is None:
        row_steps: List[ASTObject] = [rows(table="nodes")]
        scope_mode: Literal["match_alias", "row_columns"] = "row_columns"
    elif binding_row_aliases or lowered.row_pre_filters:
        row_steps = [rows(binding_ops=serialize_binding_ops(lowered.query))]
        scope_mode = "match_alias"
    else:
        table = cast(
            Literal["nodes", "edges"],
            _alias_table(
                alias_targets[active_match_alias],
                alias=active_match_alias,
                line=stage_clause.span.line,
                column=stage_clause.span.column,
                semantic_entity_kinds=semantic_entity_kinds,
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
            seed_rows=seed_rows,
            relationship_count=relationship_count,
        )


def _lower_match_alias_stage(
    stage: ProjectionStage,
    *,
    scope: _StageScope,
    params: Optional[Mapping[str, Any]],
    final_stage: bool,
    semantic_entity_kinds: Optional[Mapping[str, Literal["node", "edge", "scalar"]]] = None,
) -> Tuple[List[ASTObject], _StageScope, Optional[ResultProjectionPlan]]:
    from graphistry.compute.gfql.cypher import projection_planning as _projection

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

    plan = _projection._build_projection_plan(
        stage.clause,
        alias_targets=scope.alias_targets,
        active_alias=scope.active_alias,
        projected_columns=scope.projected_columns,
        params=params,
        semantic_entity_kinds=semantic_entity_kinds,
    )
    row_steps: List[ASTObject] = []
    defer_return_projection = False
    if not plan.whole_row_output_names:
        projection_fn = with_ if stage.clause.kind == "with" else return_
        if (
            stage.clause.kind == "return"
            and stage.order_by is not None
            and not stage.clause.distinct
        ):
            # Keep pre-existing columns alive for ORDER BY references that are
            # intentionally not part of the final RETURN projection.
            row_steps.append(with_(plan.projection_items, extend=True))
            defer_return_projection = True
        else:
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
        order_plan = _projection._plan_with_visible_projected_columns(plan, scope.projected_columns)
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
    if defer_return_projection:
        row_steps.append(return_([(name, name) for name, _ in plan.projection_items]))

    if plan.whole_row_output_names:
        # In extend mode (bindings-row path), scalar columns land in the DataFrame under
        # their output_name after with_(extend=True).  Using kind="property" would cause the
        # next stage to form f"{active_alias}.{source_name}" as the runtime expression, which
        # double-qualifies the column (e.g. "tag.cd" instead of "cd").  Use kind="expr" with
        # source_name=output_name so the next stage resolves it as a direct DataFrame column.
        extend_mode = bool(scope.allowed_match_aliases)
        next_projected_columns = {
            column.output_name: _StageColumnBinding(
                kind="expr" if extend_mode else cast(Literal["property", "expr"], column.kind),
                source_name=column.output_name if extend_mode else cast(str, column.source_name),
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
            seed_rows=scope.seed_rows,
            relationship_count=scope.relationship_count,
        )

    result_projection = _projection._result_projection_plan(plan, alias_targets=scope.alias_targets) if final_stage else None
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
    from graphistry.compute.gfql.cypher import projection_planning as _projection

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
    whole_row_group_aliases: List[str] = []

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
                if not scope.allowed_match_aliases:
                    raise _unsupported(
                        "Cypher aggregate whole-row grouping currently supports the active MATCH alias only",
                        field=stage.clause.kind,
                        value=item.expression.text,
                        line=item.span.line,
                        column=item.span.column,
                    )
                if alias_name not in scope.allowed_match_aliases:
                    raise _unsupported(
                        "Cypher aggregate whole-row grouping alias is not available in the active bindings-row scope",
                        field=stage.clause.kind,
                        value=item.expression.text,
                        line=item.span.line,
                        column=item.span.column,
                    )
            hidden_key_name = _fresh_temp_name(temp_names, "__cypher_group_key__")
            raw_key_expr = _whole_row_group_key_expr(
                alias_name,
                alias_targets=scope.alias_targets,
                field=stage.clause.kind,
                line=item.span.line,
                column=item.span.column,
            )
            if scope.allowed_match_aliases:
                key_expr = f"{alias_name}.{raw_key_expr}"
                pre_items.append((hidden_key_name, key_expr))
                key_names.append(hidden_key_name)
                hidden_group_key_names.add(hidden_key_name)
                if alias_name not in whole_row_group_aliases:
                    whole_row_group_aliases.append(alias_name)
            else:
                key_expr = raw_key_expr
                pre_items.append((hidden_key_name, key_expr))
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
            alias_name, prop = _projection._projection_ref_from_expr(
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

    bindings_row_path = bool(scope.allowed_match_aliases)
    alias_key_prefixes = [f"{alias_name}." for alias_name in whole_row_group_aliases] if whole_row_group_aliases else None

    row_steps: List[ASTObject] = []
    if key_names:
        if pre_items:
            row_steps.append(with_(pre_items, extend=bindings_row_path))
        row_steps.append(group_by(key_names, aggregations, key_prefixes=alias_key_prefixes))
    else:
        global_key = _fresh_temp_name(temp_names, "__cypher_group__")
        row_steps.append(with_([(global_key, 1)] + pre_items))
        row_steps.append(group_by([global_key], aggregations, key_prefixes=alias_key_prefixes))
        row_steps.append(projection_fn([(agg.output_name, agg.output_name) for agg in aggregate_specs]))
        available_columns = {agg.output_name for agg in aggregate_specs}
        expr_to_output = {agg.source_text: agg.output_name for agg in aggregate_specs}
        projected_property_outputs = {}
        output_to_source_property = {}

    if hidden_group_key_names:
        # Drop the temporary group-key columns and the entity blob columns that were only needed
        # for grouping.  On the bindings-row path we keep alias-prefixed property columns
        # (e.g. "tag.name") by using drop_cols instead of an explicit select projection so those
        # columns remain available for subsequent RETURN/WITH stages.
        if bindings_row_path:
            # Drop the temp group-key column(s).  Entity blob columns were never created
            # on this path; alias.* property columns survive as group-by keys.
            cols_to_drop = list(hidden_group_key_names)
            row_steps.append(drop_cols(cols_to_drop))
            agg_output_names = {agg.output_name for agg in aggregate_specs}
            if final_stage:
                # Final stage: project explicitly to only the named output columns.
                visible_projection_items = [(item.alias or item.expression.text, item.alias or item.expression.text) for item in non_aggregate_items if item.expression.text not in scope.alias_targets]
                visible_projection_items.extend((agg.output_name, agg.output_name) for agg in aggregate_specs)
                row_steps.append(projection_fn(visible_projection_items))
                available_columns = {name for name, _ in visible_projection_items}
            else:
                available_columns = agg_output_names  # alias.* cols available at runtime
        else:
            entity_output_names = [
                item.alias or item.expression.text
                for item in non_aggregate_items
                if item.expression.text in scope.alias_targets
            ]
            cols_to_drop = list(hidden_group_key_names) + entity_output_names
            visible_projection_items = [(item.alias or item.expression.text, item.alias or item.expression.text) for item in non_aggregate_items]
            visible_projection_items.extend((agg.output_name, agg.output_name) for agg in aggregate_specs)
            row_steps.append(projection_fn(visible_projection_items))
            agg_output_names = {agg.output_name for agg in aggregate_specs}
            non_agg_output_names = {item.alias or item.expression.text for item in non_aggregate_items}
            available_columns = (agg_output_names | non_agg_output_names) - set(cols_to_drop)

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

    # On the bindings-row path, propagate allowed_match_aliases so the next stage's
    # _validate_row_expr_scope accepts "tag.name"-style references.
    next_allowed_match_aliases = scope.allowed_match_aliases if bindings_row_path else set()

    return row_steps, _StageScope(
        mode="row_columns",
        alias_targets={},
        active_alias=None,
        allowed_match_aliases=next_allowed_match_aliases,
        row_columns=set(available_columns),
        projected_columns={},
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
    post_aggregate_temp_names: Set[str] = set()
    clause_items = _expand_row_column_star_items(
        stage.clause.items,
        available_columns=scope.row_columns,
        clause_kind=stage.clause.kind,
    )
    for item in clause_items:
        agg_spec = _aggregate_spec(item, params=params, alias_targets={})
        if agg_spec is None:
            post_agg_plan = _post_aggregate_expr_plan(
                item,
                params=params,
                alias_targets={},
                reserved_temp_names=post_aggregate_temp_names,
            )
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
        # On the bindings-row path, alias-prefixed column names (e.g. "tag.name") are literal
        # DataFrame column names.  _row_expr_arg would try to parse "tag" as an identifier and
        # fail.  Short-circuit: if the expression matches an allowed alias prefix, use the
        # expression text directly as a column reference string.
        alias_prefixed_shortcircuit = scope.allowed_match_aliases and any(
            item.expression.text.startswith(f"{a}.") for a in scope.allowed_match_aliases
        )
        if alias_prefixed_shortcircuit:
            runtime_expr: Any = item.expression.text
        else:
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
        if isinstance(runtime_expr, str) and not alias_prefixed_shortcircuit:
            # alias-prefixed short-circuit expressions are literal DataFrame column names
            # that should not be re-resolved via projected_columns in subsequent stages.
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


def _where_clause_expr_text(where: WhereClause) -> Optional[ExpressionText]:
    """Synthesize an ``ExpressionText`` from ``where.expr_tree`` for callers
    that historically read ``where.expr``.

    Used by the #1213 sub-PR C reader migration so that ExpressionText-shaped
    consumers (rewrite helpers, error reporters, downstream filter pipelines)
    can continue to receive an ``ExpressionText`` while the source-of-truth
    moves to ``WhereClause.expr_tree``.  Uses ``where.span`` (which equals
    the original ``where.expr.span`` for parser-produced WhereClauses, both
    derived from the same ``_span_from_meta(meta)`` of the
    ``generic_where_clause`` rule) to preserve error-position semantics.
    """
    if where.expr_tree is not None:
        return ExpressionText(text=boolean_expr_to_text(where.expr_tree), span=where.span)

    # Structured-predicate WHERE clauses can still require downstream rewrite
    # (for example, multi-alias re-entry secondary carry demotion).  Synthesize
    # equivalent row-expression text when all predicates are row-renderable.
    if where.predicates:
        predicate_texts: List[str] = []
        for predicate in where.predicates:
            if not isinstance(predicate, WherePredicate):
                return None
            row_text = _row_where_predicate_text(predicate)
            if row_text is None:
                return None
            predicate_texts.append(row_text)
        if predicate_texts:
            return ExpressionText(text=" and ".join(predicate_texts), span=where.span)
    return None


def _rewrite_where_clause_and_resync(
    where: WhereClause,
    rewrite: Callable[[ExpressionText, str], ExpressionText],
    field: str = "where",
) -> WhereClause:
    """Rewrite the WHERE expression and resynchronize ``expr_tree`` to match
    the rewritten text via single-atom synthesis (#1213 sub-PR C, Option B).

    Caveat: collapses any boolean structure in ``expr_tree`` to a single atom
    carrying the rewritten text.  Acceptable for current consumers (the
    binder's ``boolean_expr_to_text`` round-trips a single-atom tree to its
    ``atom_text``, identical to the legacy text path).

    Fixes the latent staleness bug present on master post-#1214: the prior
    pattern ``replace(where, expr=rewrite(where.expr, field))`` left
    ``expr_tree`` pointing at the pre-rewrite text.  Sub-PR D+E dropped the
    ``expr`` field; only ``expr_tree`` survives.
    """
    synthesized = _where_clause_expr_text(where)
    if synthesized is None:
        return where
    rewritten = rewrite(synthesized, field)
    new_tree = BooleanExpr(
        op="atom",
        span=rewritten.span,
        atom_text=rewritten.text,
        atom_span=rewritten.span,
    )
    rewritten_where = replace(where, expr_tree=new_tree)
    if where.expr_tree is None and where.predicates:
        rewritten_where = replace(rewritten_where, predicates=())
    return rewritten_where


def _extract_relationship_type_where(
    expr: ExpressionText,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> Optional[Tuple[PropertyRef, Literal["==", "!="], CypherLiteral]]:
    from graphistry.compute.gfql.cypher import projection_planning as _projection

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
        alias_name, prop = _projection._split_qualified_name(arg.name, line=expr.span.line, column=expr.span.column)
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


def _reject_unsupported_where_expr_forms(query: CypherQuery) -> None:
    if query.where is None or query.where.expr_tree is None:
        return
    expr_text = boolean_expr_to_text(query.where.expr_tree).strip()
    if _CYPHER_BARE_WHERE_GROUPED_ALIAS_RE.fullmatch(expr_text) is not None:
        raise _unsupported(
            "Cypher WHERE pattern predicates must include a relationship",
            field="where",
            value=expr_text,
            line=query.where.span.line,
            column=query.where.span.column,
        )


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
        pattern_kinds = _match_pattern_alias_kinds(clause)
        for alias, pattern, kind in zip(pattern_aliases, clause.patterns, pattern_kinds):
            if alias is None:
                continue
            if kind != "pattern":
                continue
            if any(
                isinstance(element, RelationshipPattern)
                and _is_variable_length_relationship_pattern(element)
                for element in pattern
            ):
                out.add(alias)
    return out


def _shortest_path_empty_result_seed_df(
    *,
    specs: Mapping[str, _ShortestPathAliasSpec],
    alias_targets: Mapping[str, ASTObject],
) -> pd.DataFrame:
    row: Dict[str, Any] = {}
    for spec in specs.values():
        row[spec.hop_column] = pd.NA
        if spec.end_alias is not None:
            row[f"{spec.end_alias}.{spec.hop_column}"] = pd.NA
    for alias, target in alias_targets.items():
        if not isinstance(target, ASTNode):
            continue
        if target.filter_dict is None:
            continue
        for key, value in target.filter_dict.items():
            row[f"{alias}.{key}"] = value
    return pd.DataFrame([row])


def _shortest_path_empty_result_row_for_row_steps(
    *,
    row_steps: Sequence[ASTObject],
    specs: Mapping[str, _ShortestPathAliasSpec],
    alias_targets: Mapping[str, ASTObject],
) -> Optional[Dict[str, Any]]:
    from graphistry.compute.gfql.row.pipeline import execute_row_pipeline_call

    seed_df = _shortest_path_empty_result_seed_df(
        specs=specs,
        alias_targets=alias_targets,
    )
    graph: Any = _SyntheticRowGraph(seed_df)
    start_idx = 0
    if (
        row_steps
        and isinstance(row_steps[0], ASTCall)
        and row_steps[0].function == "rows"
        and "binding_ops" in row_steps[0].params
    ):
        start_idx = 1
    for step in row_steps[start_idx:]:
        if not isinstance(step, ASTCall):
            return None
        graph = execute_row_pipeline_call(graph, step.function, step.params)
    if graph._nodes is None or len(graph._nodes) == 0:
        return None
    return graph._nodes.iloc[0].to_dict()


def _shortest_path_relationship_hop_columns(clause: MatchClause) -> Dict[Tuple[int, int, int, int, int, int], str]:
    out: Dict[Tuple[int, int, int, int, int, int], str] = {}
    pattern_aliases = clause.pattern_aliases or tuple(None for _ in clause.patterns)
    pattern_kinds = _match_pattern_alias_kinds(clause)
    for alias, pattern, kind in zip(pattern_aliases, clause.patterns, pattern_kinds):
        if alias is None or kind != "shortestPath":
            continue
        for element in pattern:
            if isinstance(element, RelationshipPattern):
                span = element.span
                out[(span.line, span.column, span.end_line, span.end_column, span.start_pos, span.end_pos)] = (
                    f"__cypher_shortest_path_hops__{alias}"
                )
    return out


def _check_query_projection_exprs(
    query: CypherQuery,
    *,
    check_expr: Callable[[str, str, int, int], None],
    include_reentry_wheres: bool = False,
) -> None:
    if query.where is not None and query.where.expr_tree is not None:
        check_expr(
            boolean_expr_to_text(query.where.expr_tree),
            "where",
            query.where.span.line,
            query.where.span.column,
        )
    if include_reentry_wheres:
        for reentry_where in query.reentry_wheres:
            if reentry_where is not None and reentry_where.expr_tree is not None:
                check_expr(
                    boolean_expr_to_text(reentry_where.expr_tree),
                    "where",
                    reentry_where.span.line,
                    reentry_where.span.column,
                )

    def _check_projection_clause(clause: ReturnClause) -> None:
        for item in clause.items:
            if item.expression.text == "*":
                continue
            check_expr(
                item.expression.text,
                clause.kind,
                item.span.line,
                item.span.column,
            )

    _check_projection_clause(query.return_)

    if query.order_by is not None:
        for item in query.order_by.items:
            check_expr(
                item.expression.text,
                "order_by",
                item.span.line,
                item.span.column,
            )

    for stage in query.with_stages:
        _check_projection_clause(stage.clause)
        if stage.where is not None:
            check_expr(
                stage.where.text,
                "with.where",
                stage.span.line,
                stage.span.column,
            )
        if stage.order_by is not None:
            for item in stage.order_by.items:
                check_expr(
                    item.expression.text,
                    "order_by",
                    item.span.line,
                    item.span.column,
                )


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

    _check_query_projection_exprs(
        query,
        check_expr=lambda expr_text, field, line, column: _check_expr(
            expr_text,
            field=field,
            line=line,
            column=column,
        ),
        include_reentry_wheres=True,
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

    _check_query_projection_exprs(
        query,
        check_expr=lambda expr_text, field, line, column: _check_expr(
            expr_text,
            field=field,
            line=line,
            column=column,
        ),
    )


def _predicate_pattern_aliases(predicate: WherePatternPredicate) -> List[str]:
    aliases: List[str] = []
    seen: Set[str] = set()
    for element in predicate.pattern:
        alias = getattr(element, "variable", None)
        if alias is None:
            continue
        alias_name = cast(str, alias)
        if alias_name in seen:
            continue
        seen.add(alias_name)
        aliases.append(alias_name)
    return aliases


def _where_expr_tree_pattern_predicates(expr: BooleanExpr) -> List[WherePatternPredicate]:
    out: List[WherePatternPredicate] = []
    stack: List[BooleanExpr] = [expr]
    while stack:
        cur = stack.pop()
        if cur.op == "pattern":
            if cur.pattern is None:
                raise _unsupported(
                    "Cypher WHERE pattern predicates must include a relationship",
                    field="where",
                    value=cur.atom_text,
                    line=cur.span.line,
                    column=cur.span.column,
                )
            out.append(WherePatternPredicate(
                pattern=cur.pattern, span=cur.span, negated=False,
                pattern_origin=cur.pattern_origin, pattern_neq=cur.pattern_neq))
            continue
        if cur.left is not None:
            stack.append(cur.left)
        if cur.right is not None:
            stack.append(cur.right)
    return out


def _lower_pattern_predicate_to_row_marker(
    predicate: WherePatternPredicate,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
    out_col: str,
) -> ASTCall:
    if len(predicate.pattern) < 3:
        raise _unsupported(
            "Cypher WHERE pattern predicates must include a relationship",
            field="where",
            value=None,
            line=predicate.span.line,
            column=predicate.span.column,
        )

    predicate_aliases = _predicate_pattern_aliases(predicate)
    if not predicate_aliases:
        raise _unsupported(
            "Cypher WHERE pattern predicates currently require at least one shared bound alias",
            field="where",
            value=None,
            line=predicate.span.line,
            column=predicate.span.column,
        )

    introduced_aliases = sorted(alias for alias in predicate_aliases if alias not in alias_targets)
    if introduced_aliases and predicate.pattern_origin != "exists":
        # EXISTS { } subquery aliases are EXISTENTIALLY quantified (locals) — the
        # bindings table projects them away; bare pattern predicates keep the
        # conservative guard (viz-filter L1).
        raise _unsupported(
            "Cypher WHERE pattern predicates cannot introduce new aliases in this phase",
            field="where",
            value=introduced_aliases,
            line=predicate.span.line,
            column=predicate.span.column,
        )

    shared_aliases = [alias for alias in predicate_aliases if alias in alias_targets]
    if not shared_aliases:
        raise _unsupported(
            "Cypher WHERE pattern predicates currently require at least one shared bound alias",
            field="where",
            value=predicate_aliases,
            line=predicate.span.line,
            column=predicate.span.column,
        )

    pattern_clause = MatchClause(
        patterns=(predicate.pattern,),
        span=predicate.span,
        optional=False,
        pattern_aliases=(None,),
        pattern_alias_kinds=("pattern",),
    )
    pattern_ops = lower_match_clause(pattern_clause, params=params)
    return semi_apply_mark(
        binding_ops=serialize_binding_ops(pattern_ops),
        join_aliases=shared_aliases,
        out_col=out_col,
        neq=predicate.pattern_neq,
    )


def _rewrite_where_expr_patterns_to_markers(
    *,
    where: WhereClause,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> Tuple[Optional[ExpressionText], List[ASTCall]]:
    if where.expr_tree is None:
        return None, []

    pattern_preds = _where_expr_tree_pattern_predicates(where.expr_tree)
    if not pattern_preds:
        return _where_clause_expr_text(where), []

    marker_ops: List[ASTCall] = []
    marker_counter = 0

    def _fresh_marker_col(span: SourceSpan) -> str:
        nonlocal marker_counter
        marker_counter += 1
        return (
            "__gfql_where_pattern_"
            f"{span.line}_{span.column}_{span.end_line}_{span.end_column}_{marker_counter}__"
        )

    def _rewrite(expr: BooleanExpr) -> BooleanExpr:
        if expr.op == "pattern":
            if expr.pattern is None:
                raise _unsupported(
                    "Cypher WHERE pattern predicates must include a relationship",
                    field="where",
                    value=expr.atom_text,
                    line=expr.span.line,
                    column=expr.span.column,
                )
            marker_col = _fresh_marker_col(expr.span)
            marker_ops.append(
                _lower_pattern_predicate_to_row_marker(
                    WherePatternPredicate(
                        pattern=expr.pattern, span=expr.span, negated=False,
                        pattern_origin=expr.pattern_origin, pattern_neq=expr.pattern_neq),
                    alias_targets=alias_targets,
                    params=params,
                    out_col=marker_col,
                )
            )
            return BooleanExpr(
                op="atom",
                span=expr.span,
                atom_text=marker_col,
                atom_span=expr.atom_span or expr.span,
            )
        if expr.op in {"atom"}:
            return expr
        if expr.op == "not":
            return replace(expr, left=_rewrite(cast(BooleanExpr, expr.left)))
        if expr.op in {"and", "or", "xor"}:
            return replace(
                expr,
                left=_rewrite(cast(BooleanExpr, expr.left)),
                right=_rewrite(cast(BooleanExpr, expr.right)),
            )
        return expr

    rewritten = _rewrite(where.expr_tree)
    return ExpressionText(text=boolean_expr_to_text(rewritten), span=where.span), marker_ops


def _where_pattern_predicate_marker_col(
    span: SourceSpan,
    *,
    existing: Set[str],
) -> str:
    base = (
        "__gfql_where_pattern_"
        f"{span.line}_{span.column}_{span.end_line}_{span.end_column}__"
    )
    return _fresh_temp_name(existing, base)


def _lower_negated_pattern_predicate_to_row_filter(
    predicate: WherePatternPredicate,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
) -> ASTCall:
    if len(predicate.pattern) < 3:
        raise _unsupported(
            "Cypher WHERE pattern predicates must include a relationship",
            field="where",
            value=None,
            line=predicate.span.line,
            column=predicate.span.column,
        )

    predicate_aliases = _predicate_pattern_aliases(predicate)
    if not predicate_aliases:
        raise _unsupported(
            "Cypher WHERE NOT (pattern) currently requires at least one shared bound alias",
            field="where",
            value=None,
            line=predicate.span.line,
            column=predicate.span.column,
        )

    introduced_aliases = sorted(alias for alias in predicate_aliases if alias not in alias_targets)
    if introduced_aliases and predicate.pattern_origin != "exists":
        # EXISTS { } subquery aliases are EXISTENTIALLY quantified (locals) — the
        # bindings table projects them away; bare pattern predicates keep the
        # conservative guard (viz-filter L1).
        raise _unsupported(
            "Cypher WHERE pattern predicates cannot introduce new aliases in this phase",
            field="where",
            value=introduced_aliases,
            line=predicate.span.line,
            column=predicate.span.column,
        )

    shared_aliases = [alias for alias in predicate_aliases if alias in alias_targets]
    if not shared_aliases:
        raise _unsupported(
            "Cypher WHERE NOT (pattern) currently requires at least one shared bound alias",
            field="where",
            value=predicate_aliases,
            line=predicate.span.line,
            column=predicate.span.column,
        )

    pattern_clause = MatchClause(
        patterns=(predicate.pattern,),
        span=predicate.span,
        optional=False,
        pattern_aliases=(None,),
        pattern_alias_kinds=("pattern",),
    )
    pattern_ops = lower_match_clause(pattern_clause, params=params)
    return anti_semi_apply(
        binding_ops=serialize_binding_ops(pattern_ops),
        join_aliases=shared_aliases,
        neq=predicate.pattern_neq,
    )


def lower_match_query(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> LoweredCypherMatch:
    reject_shortest_path_alias_references_after_follow_on_match(query, params=params)

    normalizer = ASTNormalizer()
    query = normalizer.rewrite_shortest_path(query)
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
    row_pre_filters: List[ASTCall] = []

    row_where: Optional[ExpressionText] = None
    row_where_predicates: List[str] = list(dynamic_row_where_predicates)
    if query.where is not None:
        if _cartesian_node_only_patterns(merged_match) is not None and query.where.expr_tree is not None:
            where_expr_upper = boolean_expr_to_text(query.where.expr_tree).upper()
            stack: List[BooleanExpr] = [query.where.expr_tree]
            while stack:
                cur = stack.pop()
                expr_left = cur.left
                expr_right = cur.right
                if cur.op in {"or", "xor"} and ((expr_left is not None and _where_expr_tree_pattern_predicates(expr_left)) or (expr_right is not None and _where_expr_tree_pattern_predicates(expr_right))):
                    raise _unsupported_at_span("Cypher WHERE pattern predicates mixed with OR/XOR are not yet supported for cartesian MATCH patterns", field="where", value=where_expr_upper, span=query.where.span)
                if expr_left is not None:
                    stack.append(expr_left)
                if expr_right is not None:
                    stack.append(expr_right)
        where_expr, where_pattern_row_filters = _rewrite_where_expr_patterns_to_markers(
            where=query.where,
            alias_targets=alias_targets,
            params=params,
        )
        row_pre_filters.extend(where_pattern_row_filters)
        marker_cols_in_use: Set[str] = {
            cast(str, op.params.get("out_col"))
            for op in row_pre_filters
            if isinstance(op.params.get("out_col"), str)
        }
        if where_expr is not None:
            type_where = _extract_relationship_type_where(
                where_expr,
                alias_targets=alias_targets,
                params=params,
            )
            if type_where is None:
                row_where = where_expr
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
                if predicate.negated:
                    row_pre_filters.append(
                        _lower_negated_pattern_predicate_to_row_filter(
                            predicate,
                            alias_targets=alias_targets,
                            params=params,
                        )
                    )
                    continue
                marker_col = _where_pattern_predicate_marker_col(
                    predicate.span,
                    existing=marker_cols_in_use,
                )
                row_pre_filters.append(
                    _lower_pattern_predicate_to_row_marker(
                        predicate,
                        alias_targets=alias_targets,
                        params=params,
                        out_col=marker_col,
                    )
                )
                row_where_predicates.append(marker_col)
                continue
            if isinstance(predicate.left, LabelRef):
                _apply_label_where(alias_targets, left=predicate.left)
                continue
            if isinstance(predicate.right, PropertyRef):
                if binding_row_aliases:
                    row_predicate_expr = _row_where_predicate_text(predicate)
                    if row_predicate_expr is not None:
                        row_where_predicates.append(row_predicate_expr)
                        continue
                where_out.append(
                    compare(
                        col(cast(PropertyRef, predicate.left).alias, cast(PropertyRef, predicate.left).property),
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

    if row_where is not None:
        # viz-filter L2-b: lift searchAny(...) calls into search_any pre-filters +
        # marker columns HERE (assembly level, like the pattern-predicate markers),
        # so every LoweredCypherMatch consumer sees the lifted form.
        lifted_text, search_calls = _lift_search_any_from_row_where(
            row_where, alias_targets=alias_targets, existing_cols=frozenset())
        if search_calls:
            row_pre_filters.extend(search_calls)
            row_where = ExpressionText(text=lifted_text, span=row_where.span)

    return LoweredCypherMatch(
        query=ops,
        where=where_out,
        row_where=row_where,
        row_pre_filters=tuple(row_pre_filters),
    )


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
        return render_cypher_string_literal(value)
    return str(value)


def _row_where_predicate_text(predicate: WherePredicate) -> Optional[str]:
    if isinstance(predicate.left, LabelRef):
        return None
    if predicate.op in {"is_null", "is_not_null"}:
        if predicate.right is not None:
            return None
        rendered_op = "IS NOT NULL" if predicate.op == "is_not_null" else "IS NULL"
        return f"{_render_row_where_operand_text(predicate.left)} {rendered_op}"
    if predicate.right is None:
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
        return NODE_IDENTITY_COLUMN
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
    if isinstance(target, ASTNode):
        return f"__node_entity__({alias_name})"
    if isinstance(target, ASTEdge):
        return f"__edge_entity__({alias_name})"
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
        return NODE_IDENTITY_COLUMN
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
    bound_visible_aliases: AbstractSet[str] = frozenset(),
    semantic_entity_kinds: Optional[Mapping[str, Literal["node", "edge", "scalar"]]] = None,
) -> CompiledCypherQuery:
    alias_targets = _alias_target(lowered.query) if query.match is not None else {}
    merged_match = _merged_match_clause(query)
    relationship_count = _match_relationship_count(merged_match) if merged_match is not None else 0
    aggregate_specs: List[_AggregateSpec] = []
    non_aggregate_items: List[ReturnItem] = []
    post_aggregate_items: List[_PostAggregateExprPlan] = []
    post_aggregate_temp_names: Set[str] = set()
    empty_result_row: Optional[Dict[str, Any]] = None
    empty_aggregate_row: Optional[Dict[str, Any]] = None
    for item in query.return_.items:
        agg_spec = _aggregate_spec(item, params=params, alias_targets=alias_targets)
        if agg_spec is not None:
            aggregate_specs.append(agg_spec)
            continue
        post_agg_plan = _post_aggregate_expr_plan(
            item,
            params=params,
            alias_targets=alias_targets,
            reserved_temp_names=post_aggregate_temp_names,
        )
        if post_agg_plan is not None:
            nested_aggregate_specs, post_agg_item = post_agg_plan
            aggregate_specs.extend(nested_aggregate_specs)
            post_aggregate_items.append(post_agg_item)
            continue
        non_aggregate_items.append(item)

    binding_row_aliases = _binding_row_aliases_for_match(query.match, alias_targets=alias_targets)
    forced_binding_row_aliases = False
    if _requires_relationship_multiplicity_bindings(
        aggregate_specs=aggregate_specs,
        relationship_count=relationship_count,
    ):
        base_active_alias: Optional[str] = None
        can_force_bindings = True
        whole_row_group_alias_refs = {
            item.expression.text
            for item in non_aggregate_items
            if item.expression.text in alias_targets
        }
        aggregate_alias_refs: Set[str] = set()
        for agg_spec in aggregate_specs:
            if agg_spec.expr_text is None:
                continue
            try:
                aggregate_alias_refs.update(
                    _expr_match_aliases(
                        agg_spec.expr_text,
                        alias_targets=alias_targets,
                        params=params,
                        field=query.return_.kind,
                        line=agg_spec.span_line,
                        column=agg_spec.span_column,
                    )
                )
            except GFQLValidationError:
                can_force_bindings = False
                break
        allow_whole_row_binding_grouping = (
            bool(whole_row_group_alias_refs)
            and can_force_bindings and not any(clause.optional for clause in query.matches)
            and bool(aggregate_alias_refs)
            and aggregate_alias_refs <= set(alias_targets.keys())
            and all(isinstance(alias_targets.get(alias_name), ASTNode) for alias_name in whole_row_group_alias_refs)
        )
        if whole_row_group_alias_refs and not allow_whole_row_binding_grouping:
            # Keep whole-row grouping on the existing conservative path.
            # This preserves the current fail-fast boundary for relationship-
            # pattern grouped aggregates such as `RETURN a, count(*)`.
            can_force_bindings = False
        if allow_whole_row_binding_grouping:
            base_active_alias = next(iter(whole_row_group_alias_refs))
        else:
            try:
                base_active_alias = _active_match_alias(
                    query,
                    alias_targets=alias_targets,
                    allowed_match_aliases=None,
                    params=params,
                )
            except GFQLValidationError:
                can_force_bindings = False
        if can_force_bindings and base_active_alias is not None:
            if isinstance(alias_targets.get(base_active_alias), ASTEdge):
                can_force_bindings = False
        if can_force_bindings and base_active_alias is not None:
            for agg_spec in aggregate_specs:
                if agg_spec.expr_text is None:
                    continue
                try:
                    refs = _expr_match_aliases(
                        agg_spec.expr_text,
                        alias_targets=alias_targets,
                        params=params,
                        field=query.return_.kind,
                        line=agg_spec.span_line,
                        column=agg_spec.span_column,
                    )
                except GFQLValidationError:
                    can_force_bindings = False
                    break
                if allow_whole_row_binding_grouping:
                    if not refs <= set(alias_targets.keys()):
                        can_force_bindings = False
                        break
                    continue
                if len(refs) > 1 or (len(refs) == 1 and base_active_alias not in refs):
                    can_force_bindings = False
                    break
        if can_force_bindings:
            binding_row_aliases = set(alias_targets.keys())
            forced_binding_row_aliases = True
    if not forced_binding_row_aliases:
        binding_row_aliases = _apply_bound_scope_membership(
            binding_row_aliases,
            alias_targets=alias_targets,
            bound_visible_aliases=bound_visible_aliases,
        )
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
                    semantic_entity_kinds=semantic_entity_kinds,
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

    expr_to_output: Dict[str, str] = {}
    available_columns: Set[str] = set()
    result_projection: Optional[ResultProjectionPlan] = None

    if aggregate_specs:
        projection_fn = with_ if query.return_.kind == "with" else return_
        pre_items: List[Tuple[str, Any]] = []
        key_names: List[str] = []
        temp_names: Set[str] = set()
        hidden_group_key_names: Set[str] = set()
        whole_row_group_aliases: List[str] = []

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
                    if not binding_row_aliases:
                        raise _unsupported(
                            "Cypher aggregate whole-row grouping currently supports the active MATCH alias only",
                            field=query.return_.kind,
                            value=item.expression.text,
                            line=item.span.line,
                            column=item.span.column,
                        )
                    if alias_name not in binding_row_aliases:
                        raise _unsupported(
                            "Cypher aggregate whole-row grouping alias is not available in the active bindings-row scope",
                            field=query.return_.kind,
                            value=item.expression.text,
                            line=item.span.line,
                            column=item.span.column,
                        )
                hidden_key_name = _fresh_temp_name(temp_names, "__cypher_group_key__")
                raw_key_expr = _whole_row_group_key_expr(
                    alias_name,
                    alias_targets=alias_targets,
                    field=query.return_.kind,
                    line=item.span.line,
                    column=item.span.column,
                )
                if binding_row_aliases:
                    pre_items.append((hidden_key_name, f"{alias_name}.{raw_key_expr}"))
                    key_names.append(hidden_key_name)
                    if alias_name not in whole_row_group_aliases:
                        whole_row_group_aliases.append(alias_name)
                else:
                    pre_items.append((hidden_key_name, raw_key_expr))
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

        if not binding_row_aliases:
            _reject_unsound_relationship_multiplicity_aggregates(
                query,
                aggregate_specs=aggregate_specs,
                alias_targets=alias_targets,
                active_match_alias=active_match_alias,
            )
        bindings_row_path = bool(binding_row_aliases)
        alias_key_prefixes = [f"{alias_name}." for alias_name in whole_row_group_aliases] if whole_row_group_aliases else None
        if key_names:
            if len(pre_items) > 0:
                row_steps.append(with_(pre_items, extend=bindings_row_path))
            row_steps.append(group_by(key_names, aggregations, key_prefixes=alias_key_prefixes))
        elif _is_pure_count_star_shortcircuit(
            aggregate_specs=aggregate_specs,
            pre_items=pre_items,
            row_steps=row_steps,
            query=query,
            binding_row_aliases=binding_row_aliases,
            relationship_count=relationship_count,
            active_match_alias=active_match_alias,
            alias_targets=alias_targets,
        ):
            # Fast path: count_table reads the scanned table's height (or the
            # source-alias mask sum) with one reduction — no full-frame
            # materialize + constant-key group_by. It replaces the sole rows()
            # step and produces the same ``count(*)`` column the group_by would,
            # so the trailing identity projection (elif not key_names, below) is
            # unchanged. empty_result_row is belt-and-suspenders here (count_table
            # always emits a 1-row result), kept for parity with the group_by path.
            base_rows = cast(ASTCall, row_steps[0])
            count_alias = aggregate_specs[0].output_name
            row_steps = [
                count_table(
                    table=cast(str, base_rows.params.get("table", "nodes")),
                    source=cast(Optional[str], base_rows.params.get("source")),
                    alias=count_alias,
                )
            ]
            available_columns = {count_alias}
            empty_aggregate_row = _empty_aggregate_row(aggregate_specs)
            empty_result_row = empty_aggregate_row
        else:
            global_key = _fresh_temp_name(temp_names, "__cypher_group__")
            row_steps.append(with_([(global_key, 1)] + pre_items))
            row_steps.append(group_by([global_key], aggregations))
            available_columns = {agg.output_name for agg in aggregate_specs}
            empty_aggregate_row = _empty_aggregate_row(aggregate_specs)
            empty_result_row = empty_aggregate_row

        if bindings_row_path and whole_row_group_aliases:
            projection_alias = whole_row_group_aliases[0]
            result_projection = ResultProjectionPlan(
                alias=projection_alias,
                table=cast(
                    Literal["nodes", "edges"],
                    _alias_table(
                        alias_targets[projection_alias],
                        alias=projection_alias,
                        line=query.return_.span.line,
                        column=query.return_.span.column,
                        semantic_entity_kinds=semantic_entity_kinds,
                    ),
                ),
                columns=tuple(
                    ResultProjectionColumn(
                        output_name=item.alias or item.expression.text,
                        kind="whole_row",
                        source_name=item.expression.text,
                    )
                    for item in non_aggregate_items
                    if item.expression.text in alias_targets
                )
                + tuple(
                    ResultProjectionColumn(
                        output_name=agg.output_name,
                        kind="expr",
                        source_name=agg.output_name,
                    )
                    for agg in aggregate_specs
                    if not agg.output_name.startswith("__cypher_postagg__")
                ),
            )

        if post_aggregate_items or hidden_group_key_names:
            if bindings_row_path and whole_row_group_aliases:
                row_steps.append(drop_cols(list(hidden_group_key_names)))
                available_columns = set(aggregate.output_name for aggregate in aggregate_specs)
            else:
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
                if empty_result_row is not None:
                    empty_result_row = _evaluate_empty_projection_row(
                        empty_result_row,
                        projection_items=post_projection_items,
                    )
        elif not key_names:
            row_steps.append(projection_fn([(agg.output_name, agg.output_name) for agg in aggregate_specs]))
    else:
        if query.match is not None and not query.unwinds and not binding_row_aliases:
            return CompiledCypherQuery(
                Chain(
                    _lower_projection_chain(
                        query,
                        lowered,
                        params=params,
                        bound_visible_aliases=bound_visible_aliases,
                        semantic_entity_kinds=semantic_entity_kinds,
                    ),
                    where=lowered.where,
                ),
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

    if empty_result_row is None and binding_row_aliases and _query_has_shortest_path_patterns(query):
        from graphistry.compute.gfql.row.pipeline import _RowPipelineAdapter

        shortest_specs = _shortest_path_alias_specs(query)
        seed_df = _shortest_path_empty_result_seed_df(
            specs=shortest_specs,
            alias_targets=alias_targets,
        )
        adapter = _RowPipelineAdapter(cast(Any, _SyntheticRowGraph(seed_df)))
        empty_projection = adapter.select(items=projection_items)
        empty_result_row = (
            empty_projection._nodes.iloc[0].to_dict()
            if empty_projection._nodes is not None and len(empty_projection._nodes) > 0
            else None
        )

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
        Chain(exec_steps, where=[] if binding_row_aliases else lowered.where),
        seed_rows=seed_rows,
        post_processing=_normalize_post_processing(
            CompiledCypherPostProcessing(
                result_projection=result_projection,
                empty_result_row=empty_result_row,
            )
        ),
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


def _compile_graph_residual_filters(
    lowered: LoweredCypherMatch,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
    line: int,
    column: int,
) -> Tuple[CompiledGraphResidualFilter, ...]:
    if lowered.row_where is None and not lowered.row_pre_filters:
        return ()

    unsupported_pre_filters = [op.function for op in lowered.row_pre_filters if op.function != "search_any"]
    if unsupported_pre_filters or lowered.row_where is None:
        raise _unsupported(
            "Cypher GRAPH constructors do not yet support pattern-predicate residuals inside GRAPH { }",
            field="graph_constructor",
            value=unsupported_pre_filters or "row_pre_filters",
            line=line,
            column=column,
        )

    expr = lowered.row_where
    expr_aliases = _expr_non_aggregate_match_aliases(
        expr.text,
        alias_targets=alias_targets,
        params=params,
        field="graph_constructor",
        line=expr.span.line,
        column=expr.span.column,
    )
    pre_filter_aliases = {
        cast(str, op.params.get("alias"))
        for op in lowered.row_pre_filters
        if isinstance(op.params.get("alias"), str)
    }
    aliases = expr_aliases | pre_filter_aliases
    if len(aliases) != 1:
        raise _unsupported(
            "Cypher GRAPH residual predicates must reference exactly one node or edge alias to be applied as graph masks",
            field="graph_constructor",
            value=expr.text,
            line=expr.span.line,
            column=expr.span.column,
        )

    alias = next(iter(aliases))
    target = alias_targets.get(alias)
    if isinstance(target, ASTNode):
        kind: Literal["node", "edge"] = "node"
        if not (
            len(lowered.query) == 1
            and isinstance(lowered.query[0], ASTNode)
            and lowered.query[0]._name == alias
        ):
            raise _unsupported(
                "Cypher GRAPH node residual predicates are only supported for single-node GRAPH MATCH masks",
                field="graph_constructor",
                value=expr.text,
                line=expr.span.line,
                column=expr.span.column,
            )
    elif isinstance(target, ASTEdge):
        kind = "edge"
    else:
        raise _unsupported(
            "Cypher GRAPH residual predicates must target a node or edge alias",
            field="graph_constructor",
            value=alias,
            line=expr.span.line,
            column=expr.span.column,
        )

    _validate_row_expr_scope(
        expr.text,
        alias_targets=alias_targets,
        active_match_alias=alias,
        allowed_match_aliases={alias},
        unwind_aliases=(),
        params=params,
        field="graph_constructor",
        line=expr.span.line,
        column=expr.span.column,
    )
    return (
        CompiledGraphResidualFilter(
            alias=alias,
            kind=kind,
            expr=_row_expr_arg(expr, params=params, alias_targets=alias_targets, field="graph_constructor"),
            pre_filters=tuple(lowered.row_pre_filters),
        ),
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
        compiled_call = compile_cypher_call(constructor.call, params=params)
        if compiled_call.result_kind != "graph":
            raise _unsupported(
                "Only graph-preserving CALL procedures (with .write()) are allowed inside a graph constructor",
                field="graph_constructor",
                value=constructor.call.procedure,
                line=constructor.call.span.line,
                column=constructor.call.span.column,
            )
        call_logical_plan = _logical_plan_from_compiled_call(compiled_call)
        _verify_selected_logical_plan(call_logical_plan)
        return CompiledCypherQuery(
            Chain([]),
            seed_rows=False,
            procedure_call=compiled_call,
            execution_extras=_normalize_execution_extras(
                CompiledCypherExecutionExtras(logical_plan=call_logical_plan)
            ),
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
    alias_targets = _alias_target(lowered.query)
    graph_residual_filters = _compile_graph_residual_filters(
        lowered,
        alias_targets=alias_targets,
        params=params,
        line=constructor.span.line,
        column=constructor.span.column,
    )
    constructor_bound_ir = FrontendBinder().bind(synthetic, PlanContext(), strict_name_resolution=True)
    (
        constructor_logical_plan,
        constructor_logical_plan_defer_reason,
        constructor_logical_plan_defer_code,
    ) = _logical_plan_route_for_query(
        synthetic,
        bound_ir=constructor_bound_ir,
        params=params,
        allow_unknown_match_aliases=True,
    )
    return CompiledCypherQuery(
        Chain(lowered.query, where=lowered.where),
        seed_rows=False,
        execution_extras=_normalize_execution_extras(
            CompiledCypherExecutionExtras(
                logical_plan=constructor_logical_plan,
                logical_plan_defer_reason=constructor_logical_plan_defer_reason,
                logical_plan_defer_code=constructor_logical_plan_defer_code,
            )
        ),
        graph_residual_filters=graph_residual_filters,
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
            logical_plan=compiled_query.logical_plan,
            logical_plan_defer_reason=compiled_query.logical_plan_defer_reason,
            graph_residual_filters=compiled_query.graph_residual_filters,
        ))
    return tuple(compiled)


def _is_connected_optional_match_query(query: CypherQuery) -> bool:
    """Detect: 1 non-optional MATCH + 1-or-more OPTIONAL MATCH clauses.

    Returns True when the first clause is non-optional and every subsequent
    clause is optional, and the query has no WITH/UNWIND/CALL stages.
    The first MATCH may be a single-node or connected pattern.
    """
    if len(query.matches) < 2:
        return False
    if query.matches[0].optional:
        return False
    if not all(m.optional for m in query.matches[1:]):
        return False
    if query.with_stages or query.unwinds or query.call is not None or query.row_sequence:
        return False
    first = query.matches[0]
    has_relationship = any(
        isinstance(el, RelationshipPattern)
        for pat in first.patterns
        for el in pat
    )
    # For single-node first MATCH, only take this path when there are 3+
    # matches (multiple optionals) because projection_planning already handles
    # the 2-match single-node optional-null-fill path.
    if not has_relationship and len(query.matches) == 2:
        return False
    # Reject comma-separated base MATCH patterns (e.g., (a:A), (b:B)) — the
    # binding_ops mechanism requires a single connected path.
    if len(first.patterns) > 1:
        return False
    # Reject optional clauses with variable-length relationships — binding_ops
    # can handle them for the base chain but the join semantics are untested.
    for m in query.matches[1:]:
        for pat in m.patterns:
            for el in pat:
                if isinstance(el, RelationshipPattern) and (
                    el.min_hops is not None or el.max_hops is not None
                    or (el.to_fixed_point if hasattr(el, "to_fixed_point") else False)
                ):
                    return False
    return True


@dataclass(frozen=True)
class _OptionalMatchArm:
    """One OPTIONAL MATCH arm: its chain, join keys, and new aliases."""
    chain: Chain
    shared_node_aliases: Tuple[str, ...]
    opt_only_aliases: Tuple[str, ...]


@dataclass(frozen=True)
class ConnectedOptionalMatchPlan:
    """Plan for 1 non-optional MATCH + N OPTIONAL MATCH left-outer-joins.

    Execution: run base_chain, then for each arm run its chain, left-outer-join
    onto the accumulated result, and finally dispatch post_join_chain for
    RETURN / ORDER BY / etc.
    """
    base_chain: Chain
    arms: Tuple[_OptionalMatchArm, ...]
    post_join_chain: Chain


@dataclass(frozen=True)
class ConnectedMatchJoinPlan:
    """Plan for a connected non-linear comma MATCH lowered through joined rows."""

    pattern_chains: Tuple[Chain, ...]
    pattern_shared_node_aliases: Tuple[Tuple[str, ...], ...]
    post_join_chain: Chain


def _pattern_alias_lists(pattern: Sequence[PatternElement]) -> Tuple[List[str], List[str]]:
    node_aliases: List[str] = []
    edge_aliases: List[str] = []
    for element in pattern:
        alias = getattr(element, "variable", None)
        if alias is None:
            continue
        if isinstance(element, NodePattern):
            node_aliases.append(alias)
        elif isinstance(element, RelationshipPattern):
            edge_aliases.append(alias)
    return node_aliases, edge_aliases


def _connected_component_from_pattern(
    pattern: Sequence[PatternElement],
    *,
    entry_points: Sequence[str],
) -> ConnectedComponent:
    node_aliases, edge_aliases = _pattern_alias_lists(pattern)
    return ConnectedComponent(
        node_aliases=node_aliases,
        edge_aliases=edge_aliases,
        entry_points=list(entry_points),
        hop_order=list(range(len(edge_aliases))),
    )


def _logical_plan_from_query_graph(
    query_graph: QueryGraph,
    *,
    optional: bool,
) -> LogicalPlan:
    next_op_id = 1

    def _next_id() -> int:
        nonlocal next_op_id
        op_id = next_op_id
        next_op_id += 1
        return op_id

    if not query_graph.components:
        return LogicalProject(
            op_id=_next_id(),
            input=None,
            expressions=[],
            output_schema=LogicalRowSchema(columns={}),
        )

    base_component = query_graph.components[0]
    current: LogicalPlan = PatternMatch(
        op_id=_next_id(),
        pattern={
            "node_aliases": tuple(base_component.node_aliases),
            "edge_aliases": tuple(base_component.edge_aliases),
            "entry_points": tuple(base_component.entry_points),
        },
        optional=False,
        arm_id=None,
        output_schema=LogicalRowSchema(columns={}),
    )

    for idx, component in enumerate(query_graph.components[1:]):
        arm = query_graph.optional_arms[idx] if idx < len(query_graph.optional_arms) else None
        right = PatternMatch(
            op_id=_next_id(),
            pattern={
                "node_aliases": tuple(component.node_aliases),
                "edge_aliases": tuple(component.edge_aliases),
                "entry_points": tuple(component.entry_points),
            },
            optional=optional and arm is not None,
            arm_id=None if arm is None else arm.arm_id,
            output_schema=LogicalRowSchema(columns={}),
        )
        join_aliases = component.entry_points if arm is None else sorted(arm.join_aliases)
        current = LogicalJoin(
            op_id=_next_id(),
            left=current,
            right=right,
            condition={"join_aliases": tuple(join_aliases)},
            join_type="left" if optional and arm is not None else "inner",
            output_schema=LogicalRowSchema(columns={}),
        )

    return LogicalProject(
        op_id=_next_id(),
        input=current,
        expressions=["connected_optional_match" if optional else "connected_match_join"],
        output_schema=LogicalRowSchema(columns={}),
    )


def _compile_connected_match_join(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
    semantic_entity_kinds: Optional[Mapping[str, Literal["node", "edge", "scalar"]]] = None,
) -> CompiledCypherQuery:
    clause = query.matches[0]
    pattern_chains: List[Chain] = []
    pattern_node_aliases: List[Set[str]] = []
    combined_alias_targets: Dict[str, ASTObject] = {}
    pre_join_filters: List[ExpressionText] = []

    for idx, pattern in enumerate(clause.patterns):
        single_clause = MatchClause(
            patterns=(pattern,),
            optional=clause.optional,
            span=clause.span,
            pattern_aliases=(None,),
            pattern_alias_kinds=(_match_pattern_alias_kinds(clause)[idx],),
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

    if query.where is not None and query.where.expr_tree is not None:
        synthesized = _where_clause_expr_text(query.where)
        assert synthesized is not None  # gated by expr_tree is not None
        pre_join_filters.append(synthesized)

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
    query_graph = QueryGraph(
        components=[
            _connected_component_from_pattern(
                pattern,
                entry_points=() if idx == 0 else shared_aliases_per_pattern[idx - 1],
            )
            for idx, pattern in enumerate(clause.patterns)
        ],
        boundary_aliases={},
        optional_arms=[],
    )
    logical_plan = _logical_plan_from_query_graph(query_graph, optional=False)

    return CompiledCypherQuery(
        chain=Chain([]),
        seed_rows=False,
        execution_extras=_normalize_execution_extras(
            CompiledCypherExecutionExtras(
                connected_match_join=ConnectedMatchJoinPlan(
                    pattern_chains=tuple(pattern_chains),
                    pattern_shared_node_aliases=tuple(shared_aliases_per_pattern),
                    post_join_chain=Chain(row_steps),
                ),
                query_graph=query_graph,
                logical_plan=logical_plan,
            )
        ),
    )


def _apply_where_to_ops(
    where: Optional[WhereClause],
    alias_targets: Dict[str, ASTObject],
    *,
    params: Optional[Mapping[str, Any]],
) -> Tuple[List[WhereComparison], List[ExpressionText], List[ASTCall]]:
    """Apply a WHERE clause's predicates to already-lowered ops.

    Label predicates mutate the ASTNode filter in *alias_targets* (in-place).
    Property-comparison predicates are returned as ``WhereComparison`` entries
    for the chain's ``where`` list. Expressions that cannot be lowered to
    node/edge filter dicts are returned as row-expression filters so connected
    OPTIONAL MATCH lowering can apply them via ``where_rows(expr=...)``.
    """
    where_out: List[WhereComparison] = []
    row_expr_filters: List[ExpressionText] = []
    row_pre_filters: List[ASTCall] = []
    if where is None:
        return where_out, row_expr_filters, row_pre_filters
    where_expr, where_pattern_row_filters = _rewrite_where_expr_patterns_to_markers(
        where=where,
        alias_targets=alias_targets,
        params=params,
    )
    row_pre_filters.extend(where_pattern_row_filters)
    marker_cols_in_use: Set[str] = {
        cast(str, op.params.get("out_col"))
        for op in row_pre_filters
        if isinstance(op.params.get("out_col"), str)
    }
    if where_expr is not None:
        type_where = _extract_relationship_type_where(
            where_expr,
            alias_targets=alias_targets,
            params=params,
        )
        if type_where is not None:
            left, op, right = type_where
            _apply_literal_where(alias_targets, left=left, op=op, right=right, params=params)
        else:
            rewritten = _rewrite_connected_join_expr(
                where_expr,
                alias_targets=alias_targets,
                params=params,
                field="where",
                line=where_expr.span.line,
                column=where_expr.span.column,
            )
            if rewritten is not None:
                row_expr_filters.append(rewritten)
    for predicate in where.predicates:
        if isinstance(predicate, WherePatternPredicate):
            if predicate.negated:
                row_pre_filters.append(
                    _lower_negated_pattern_predicate_to_row_filter(
                        predicate,
                        alias_targets=alias_targets,
                        params=params,
                    )
                )
                continue
            marker_col = _where_pattern_predicate_marker_col(
                predicate.span,
                existing=marker_cols_in_use,
            )
            row_pre_filters.append(
                _lower_pattern_predicate_to_row_marker(
                    predicate,
                    alias_targets=alias_targets,
                    params=params,
                    out_col=marker_col,
                )
            )
            row_expr_filters.append(ExpressionText(text=marker_col, span=predicate.span))
            continue
        if isinstance(predicate.left, LabelRef):
            _apply_label_where(alias_targets, left=predicate.left)
            continue
        if isinstance(predicate.right, PropertyRef):
            assert isinstance(predicate.left, PropertyRef)
            # Reject cross-clause alias references: both sides must be in
            # this clause's alias_targets to avoid runtime ValueError.
            for ref in (predicate.left, predicate.right):
                if ref.alias not in alias_targets:
                    raise _unsupported(
                        f"Cypher WHERE property comparison references alias '{ref.alias}' which is not bound in this MATCH clause",
                        field="where",
                        value=f"{predicate.left.alias}.{predicate.left.property} {predicate.op} {predicate.right.alias}.{predicate.right.property}",
                        line=predicate.span.line,
                        column=predicate.span.column,
                    )
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
    return where_out, row_expr_filters, row_pre_filters


def _compile_connected_optional_match(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
    semantic_entity_kinds: Optional[Mapping[str, Literal["node", "edge", "scalar"]]] = None,
) -> CompiledCypherQuery:
    """Compile a MATCH + N OPTIONAL MATCH query.

    Lowers each clause independently, builds one arm per OPTIONAL MATCH for
    chained left-outer-joins at runtime, and delegates RETURN / ORDER BY /
    SKIP / LIMIT to the standard row pipeline.
    """
    from graphistry.compute.gfql.cypher import projection_planning as _projection

    base_clause = query.matches[0]
    base_ops = lower_match_clause(base_clause, params=params)
    base_alias_targets = _alias_target(base_ops)
    base_aliases = _match_clause_aliases(base_clause)
    base_where, base_row_expr_filters, base_row_pre_filters = _apply_where_to_ops(
        base_clause.where,
        base_alias_targets,
        params=params,
    )
    base_chain_ops: List[ASTObject] = list(base_ops)
    base_chain_ops.extend(base_row_pre_filters)
    for expr in base_row_expr_filters:
        base_chain_ops.append(
            where_rows(
                expr=_row_expr_arg(
                    expr,
                    params=params,
                    alias_targets={},
                    field="where",
                )
            )
        )
    base_chain = Chain(base_chain_ops, where=base_where)

    arms: List[_OptionalMatchArm] = []
    optional_components: List[ConnectedComponent] = []
    optional_arms_meta: List[OptionalArm] = []
    all_known_aliases = set(base_aliases)
    combined_alias_targets: Dict[str, ASTObject] = dict(base_alias_targets)

    for opt_clause in query.matches[1:]:
        opt_ops = lower_match_clause(opt_clause, params=params)
        opt_alias_targets = _alias_target(opt_ops)
        opt_aliases = _match_clause_aliases(opt_clause)
        shared_aliases = all_known_aliases & opt_aliases
        opt_only_aliases = opt_aliases - all_known_aliases

        if not shared_aliases:
            raise _unsupported(
                "Cypher connected MATCH + OPTIONAL MATCH requires at least one shared alias between the two patterns",
                field="match",
                value=None,
                line=opt_clause.span.line,
                column=opt_clause.span.column,
            )

        shared_node_aliases = sorted(
            alias for alias in shared_aliases
            if isinstance(combined_alias_targets.get(alias), ASTNode)
            or isinstance(opt_alias_targets.get(alias), ASTNode)
        )
        if not shared_node_aliases:
            raise _unsupported(
                "Cypher connected MATCH + OPTIONAL MATCH requires at least one shared node alias for the join",
                field="match",
                value=sorted(shared_aliases),
                line=opt_clause.span.line,
                column=opt_clause.span.column,
            )

        opt_where, opt_row_expr_filters, opt_row_pre_filters = _apply_where_to_ops(
            opt_clause.where,
            opt_alias_targets,
            params=params,
        )
        opt_chain_ops: List[ASTObject] = list(opt_ops)
        opt_chain_ops.extend(opt_row_pre_filters)
        for expr in opt_row_expr_filters:
            opt_chain_ops.append(
                where_rows(
                    expr=_row_expr_arg(
                        expr,
                        params=params,
                        alias_targets={},
                        field="where",
                    )
                )
            )
        arms.append(_OptionalMatchArm(
            chain=Chain(opt_chain_ops, where=opt_where),
            shared_node_aliases=tuple(shared_node_aliases),
            opt_only_aliases=tuple(sorted(opt_only_aliases)),
        ))
        optional_components.append(
            _connected_component_from_pattern(
                _match_pattern_elements(opt_clause),
                entry_points=shared_node_aliases,
            )
        )
        optional_arms_meta.append(
            OptionalArm(
                arm_id=f"optional_arm_{len(optional_arms_meta) + 1}",
                join_aliases=frozenset(shared_node_aliases),
                nullable_aliases=frozenset(opt_only_aliases),
            )
        )

        all_known_aliases.update(opt_aliases)
        for alias, target in opt_alias_targets.items():
            if alias not in combined_alias_targets:
                combined_alias_targets[alias] = target

    # Build the row-pipeline ops that will run on the joined DataFrame.
    try:
        active = _active_match_alias(query, alias_targets=combined_alias_targets, params=params)
    except GFQLValidationError:
        active = next(iter(combined_alias_targets)) if combined_alias_targets else None

    plan = _projection._build_projection_plan(
        query.return_,
        alias_targets=combined_alias_targets,
        active_alias=active,
        params=params,
        semantic_entity_kinds=semantic_entity_kinds,
    )

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
    query_graph = QueryGraph(
        components=[
            _connected_component_from_pattern(
                _match_pattern_elements(base_clause),
                entry_points=(),
            ),
            *optional_components,
        ],
        boundary_aliases={},
        optional_arms=optional_arms_meta,
    )
    logical_plan = _logical_plan_from_query_graph(query_graph, optional=True)

    return CompiledCypherQuery(
        chain=base_chain,
        seed_rows=False,
        execution_extras=_normalize_execution_extras(
            CompiledCypherExecutionExtras(
                connected_optional_match=ConnectedOptionalMatchPlan(
                    base_chain=base_chain,
                    arms=tuple(arms),
                    post_join_chain=Chain(post_join_ops),
                ),
                query_graph=query_graph,
                logical_plan=logical_plan,
            )
        ),
    )


def _logical_plan_route_for_query(
    query: CypherQuery,
    *,
    bound_ir: BoundIR,
    params: Optional[Mapping[str, Any]] = None,
    allow_unknown_match_aliases: bool = False,
) -> Tuple[Optional[LogicalPlan], Optional[str], Optional[str]]:
    ctx = PlanContext()
    if query.call is not None:
        compiled_call = compile_cypher_call(query.call, params=params)
        logical_plan = _logical_plan_from_compiled_call(compiled_call)
        _verify_selected_logical_plan(logical_plan)
        return logical_plan, None, None
    try:
        logical_plan = LogicalPlanner(
            allow_unknown_match_aliases=allow_unknown_match_aliases
        ).plan(bound_ir, ctx)
    except GFQLValidationError as exc:
        context = getattr(exc, "context", None)
        defer_code = (
            context.get("logical_plan_defer_code")
            if isinstance(context, dict)
            else None
        )
        return None, str(exc.message), cast(Optional[str], defer_code)
    _verify_selected_logical_plan(logical_plan)
    return logical_plan, None, None


def _attach_logical_plan_route(
    result: CompiledCypherQuery,
    *,
    logical_plan: Optional[LogicalPlan],
    logical_plan_defer_reason: Optional[str],
    logical_plan_defer_code: Optional[str],
) -> CompiledCypherQuery:
    result_extras = result.execution_extras or CompiledCypherExecutionExtras()
    if result.optional_reentry:
        if result.logical_plan is not None:
            effective_logical_plan = result.logical_plan
            effective_defer_reason = None
            effective_defer_code = None
        else:
            effective_logical_plan = None
            effective_defer_reason = (
                logical_plan_defer_reason
                or result.logical_plan_defer_reason
                or "LogicalPlanner skeleton does not yet support OPTIONAL MATCH planning for reentry"
            )
            effective_defer_code = (
                logical_plan_defer_code
                or result.logical_plan_defer_code
                or LOGICAL_PLAN_DEFER_OPTIONAL_MATCH_REENTRY
            )
    else:
        effective_logical_plan = result.logical_plan if result.logical_plan is not None else logical_plan
        if effective_logical_plan is not None:
            effective_defer_reason = None
            effective_defer_code = None
        elif logical_plan_defer_reason is not None:
            effective_defer_reason = logical_plan_defer_reason
            effective_defer_code = logical_plan_defer_code
        else:
            effective_defer_reason = result.logical_plan_defer_reason
            effective_defer_code = result.logical_plan_defer_code
    return replace(
        result,
        execution_extras=_execution_extras_with(
            result,
            connected_optional_match=result_extras.connected_optional_match,
            connected_match_join=result_extras.connected_match_join,
            query_graph=result_extras.query_graph,
            start_nodes_query=result.start_nodes_query,
            optional_reentry=result.optional_reentry,
            reentry_plan=result.reentry_plan,
            scope_stack=result_extras.scope_stack,
            logical_plan=effective_logical_plan,
            logical_plan_defer_reason=effective_defer_reason,
            logical_plan_defer_code=effective_defer_code,
        ),
    )


def compile_cypher_query(
    query: Union[CypherQuery, CypherUnionQuery, CypherGraphQuery],
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> Union[CompiledCypherQuery, CompiledCypherUnionQuery, CompiledCypherGraphQuery]:
    from graphistry.compute.gfql.cypher import projection_planning as _projection

    prepass_bound_ir = FrontendBinder().bind(query, PlanContext(), strict_name_resolution=True)
    prepass_context = _build_bound_lowering_context(bound_ir=prepass_bound_ir, params=params)
    params = prepass_context.params

    if isinstance(query, CypherGraphQuery):
        compiled_bindings = _compile_graph_bindings(query.graph_bindings, params=params)
        compiled_constructor = _compile_graph_constructor(query.constructor, params=params)
        use_ref = query.constructor.use.ref if query.constructor.use is not None else None
        return CompiledCypherGraphQuery(
            graph_bindings=compiled_bindings,
            chain=compiled_constructor.chain,
            procedure_call=compiled_constructor.procedure_call,
            use_ref=use_ref,
            logical_plan=compiled_constructor.logical_plan,
            logical_plan_defer_reason=compiled_constructor.logical_plan_defer_reason,
            graph_residual_filters=compiled_constructor.graph_residual_filters,
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
    logical_plan: Optional[LogicalPlan] = None
    logical_plan_defer_reason: Optional[str] = None
    logical_plan_defer_code: Optional[str] = None
    _bound_scope_stack: Tuple[ScopeFrame, ...] = ()

    def _attach_graph_context(result: CompiledCypherQuery) -> CompiledCypherQuery:
        result_extras = result.execution_extras or CompiledCypherExecutionExtras()
        out = result
        if not compiled_bindings and _use_ref is None:
            out = result
        else:
            out = replace(result, graph_bindings=compiled_bindings, use_ref=_use_ref)
        out = replace(
            out,
            execution_extras=_execution_extras_with(
                out,
                connected_optional_match=result_extras.connected_optional_match,
                connected_match_join=result_extras.connected_match_join,
                query_graph=result_extras.query_graph,
                start_nodes_query=out.start_nodes_query,
                optional_reentry=out.optional_reentry,
                reentry_plan=out.reentry_plan,
                scope_stack=_bound_scope_stack,
                logical_plan=out.logical_plan,
                logical_plan_defer_reason=out.logical_plan_defer_reason,
                logical_plan_defer_code=out.logical_plan_defer_code,
            ),
        )
        return _attach_logical_plan_route(
            out,
            logical_plan=logical_plan,
            logical_plan_defer_reason=logical_plan_defer_reason,
            logical_plan_defer_code=logical_plan_defer_code,
        )

    reject_shortest_path_alias_references_after_follow_on_match(query, params=params)

    normalizer = ASTNormalizer()
    query = normalizer.rewrite_shortest_path(query)
    _reject_variable_length_path_alias_references(query, params=params)

    # Re-bind after normalization so scope and semantic metadata reflect the
    # lowered query shape consumed by downstream lowering decisions.
    # #1357: strict alias/name-resolution is now the runtime default for the
    # post-normalize bind pass so alias-scope enforcement is centralized at
    # binder time (validator/runtime parity).
    bound_ir = FrontendBinder().bind(query, PlanContext(), strict_name_resolution=True)
    _bound_scope_stack = tuple(bound_ir.scope_stack)
    bound_context = _build_bound_lowering_context(bound_ir=bound_ir, params=params)
    params = bound_context.params
    _reject_unsupported_where_expr_forms(query)
    logical_plan, logical_plan_defer_reason, logical_plan_defer_code = _logical_plan_route_for_query(
        query,
        bound_ir=bound_ir,
        params=params,
    )
    if query.reentry_matches:
        # #1341: when the trailing MATCH only re-binds carried whole-row aliases
        # (e.g. LDBC SNB IC1 ``shortestPath((p)-[:KNOWS*]-(friend))``), the WITH
        # stage is a no-op. Flatten the reentry into a single MATCH so the
        # supported single-MATCH paths (including two-endpoint shortestPath)
        # handle it directly.
        from graphistry.compute.gfql.cypher.reentry.flatten import (
            flatten_carried_endpoint_rebind,
        )

        flattened = flatten_carried_endpoint_rebind(query)
        if flattened is not None:
            # ``flatten_carried_endpoint_rebind`` returns a query with
            # ``reentry_matches=()``, so the recursive call cannot re-enter
            # this branch — recursion terminates after one step.
            return compile_cypher_query(flattened, params=params)
        from graphistry.compute.gfql.cypher.reentry import compiletime as _reentry_compiletime

        return _attach_graph_context(_reentry_compiletime._compile_bounded_reentry_query(query, params=params))
    if query.call is not None:
        return _attach_graph_context(_compile_call_query(query, params=params))
    if query.row_sequence:
        return _attach_graph_context(_lower_row_only_sequence(query, params=params))
    if (
        len(query.matches) == 1
        and _is_node_connected_multi_pattern_clause(query.matches[0])
        and not _query_has_shortest_path_patterns(query)
        and (
            _is_connected_multi_pattern_clause(query.matches[0])
            or _query_has_aggregate_stage(query, params=params)
        )
        and not _query_requires_general_lowering_for_connected_join(query, params=params)
    ):
        return _attach_graph_context(
            _compile_connected_match_join(
                query,
                params=params,
                semantic_entity_kinds=bound_context.entity_kinds,
            )
        )
    if _is_connected_optional_match_query(query):
        return _attach_graph_context(
            _compile_connected_optional_match(
                query,
                params=params,
                semantic_entity_kinds=bound_context.entity_kinds,
            )
        )

    merged_match = _merged_match_clause(query)
    lowered = (
        lower_match_query(query, params=params)
        if merged_match is not None
        else LoweredCypherMatch(query=[], where=[])
    )
    alias_targets = _alias_target(lowered.query) if query.match is not None else {}

    def _lower_general() -> CompiledCypherQuery:
        return _lower_general_row_projection(
            query,
            lowered,
            params=params,
            bound_visible_aliases=bound_context.visible_aliases,
            semantic_entity_kinds=bound_context.entity_kinds,
        )

    if query.with_stages:
        binding_row_aliases = _binding_row_aliases_for_match(query.match, alias_targets=alias_targets)
        binding_row_aliases = _apply_bound_scope_membership(
            binding_row_aliases,
            alias_targets=alias_targets,
            bound_visible_aliases=bound_context.visible_aliases,
        )
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
            bound_visible_aliases=bound_context.visible_aliases,
            semantic_entity_kinds=bound_context.entity_kinds,
        )

        for stage in query.with_stages:
            if scope.mode == "match_alias":
                stage_steps, scope, _ = _lower_match_alias_stage(
                    stage,
                    scope=scope,
                    params=params,
                    final_stage=False,
                    semantic_entity_kinds=bound_context.entity_kinds,
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
                semantic_entity_kinds=bound_context.entity_kinds,
            )
            row_steps.extend(stage_steps)
        else:
            stage_steps, _next_scope = _lower_row_column_stage(final_stage, scope=scope, params=params)
            row_steps.extend(stage_steps)

        empty_result_row: Optional[Dict[str, Any]] = None
        if binding_row_aliases and _query_has_shortest_path_patterns(query) and result_projection is None:
            empty_result_row = _shortest_path_empty_result_row_for_row_steps(
                row_steps=row_steps,
                specs=_shortest_path_alias_specs(query),
                alias_targets=alias_targets,
            )

        return _attach_graph_context(CompiledCypherQuery(
            Chain(row_steps if binding_row_aliases else lowered.query + row_steps, where=lowered.where),
            seed_rows=scope.seed_rows,
            post_processing=_normalize_post_processing(
                CompiledCypherPostProcessing(
                    result_projection=result_projection,
                    empty_result_row=empty_result_row,
                )
            ),
        ))

    if merged_match is not None and not query.unwinds:
        binding_row_aliases = _binding_row_aliases_for_match(query.match, alias_targets=alias_targets)
        if not binding_row_aliases:
            binding_row_aliases.update(
                _binding_row_aliases_for_multi_alias_whole_row_node_projection(
                    query,
                    clause=query.return_,
                    alias_targets=alias_targets,
                )
            )
        binding_row_aliases = _apply_bound_scope_membership(
            binding_row_aliases,
            alias_targets=alias_targets,
            bound_visible_aliases=bound_context.visible_aliases,
        )
        _reject_variable_length_relationship_alias_path_carriers(
            query,
            alias_targets=alias_targets,
            params=params,
        )
        if _query_has_shortest_path_patterns(query):
            return _attach_graph_context(_lower_general())
        duplicated_aliases = _duplicate_node_aliases(merged_match)
        _projection._reject_duplicate_alias_row_refs(
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
                plan = _projection._build_projection_plan(
                    query.return_,
                    alias_targets=alias_targets,
                    active_alias=active,
                    params=params,
                    semantic_entity_kinds=bound_context.entity_kinds,
                )
            except GFQLValidationError:
                if binding_row_aliases:
                    plan = None
                else:
                    raise
            if plan is None:
                return _attach_graph_context(_lower_general())
            if _multi_alias_exc2 is not None:
                if not _projection._can_lower_multi_alias_projection_bindings(plan, alias_targets=alias_targets):
                    if binding_row_aliases:
                        return _attach_graph_context(_lower_general())
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
                _projection._empty_optional_projection_row(
                    plan,
                    query=query,
                    optional_aliases=_match_clause_aliases(query.matches[0]),
                    alias_targets=alias_targets,
                    params=params,
                )
                if len(query.matches) == 1 and query.matches[0].optional
                else None
            )
            optional_null_fill = _projection._optional_null_fill_plan(
                query,
                lowered=lowered,
                alias_targets=alias_targets,
                plan=plan,
                params=params,
                bound_visible_aliases=bound_context.visible_aliases,
                semantic_entity_kinds=bound_context.entity_kinds,
            )
            optional_only_projection = _return_references_optional_only_alias(
                query,
                alias_targets=alias_targets,
                params=params,
                bound_nullable_aliases=bound_context.nullable_aliases,
            )
            optional_projection_row_guard = None
            if optional_null_fill is None and optional_only_projection:
                if _where_uses_optional_only_label_predicate(
                    query,
                    bound_nullable_aliases=bound_context.nullable_aliases,
                ):
                    raise _unsupported(
                        "Cypher OPTIONAL MATCH label filters over optional-only aliases are not yet supported when projecting optional aliases",
                        field=query.return_.kind,
                        value=[item.expression.text for item in query.return_.items],
                        line=query.return_.span.line,
                        column=query.return_.span.column,
                    )
                optional_projection_row_guard = _projection._optional_projection_row_guard_plan(
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
                Chain(
                    _lower_projection_chain(
                        query,
                        lowered,
                        params=params,
                        plan=plan,
                        bound_visible_aliases=bound_context.visible_aliases,
                        semantic_entity_kinds=bound_context.entity_kinds,
                    ),
                    where=lowered.where,
                ),
                seed_rows=False,
                post_processing=_normalize_post_processing(
                    CompiledCypherPostProcessing(
                        result_projection=_projection._result_projection_plan(plan, alias_targets=alias_targets),
                        empty_result_row=empty_result_row,
                        optional_null_fill=optional_null_fill,
                        optional_projection_row_guard=optional_projection_row_guard,
                    )
                ),
            ))

    alias_targets = _alias_target(lowered.query)
    _reject_variable_length_relationship_alias_path_carriers(
        query,
        alias_targets=alias_targets,
        params=params,
    )
    return _attach_graph_context(_lower_general())
