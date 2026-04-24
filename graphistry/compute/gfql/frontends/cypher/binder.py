"""Cypher frontend binder interface.

This pass builds a frontend-neutral semantic view (BoundIR) from Cypher ASTs.
The lowering path still owns execution behavior; binder output is used for
semantic contracts and staged migration work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union, cast

from graphistry.compute.gfql.cypher.ast import (
    BooleanExpr,
    CallClause,
    CypherGraphQuery,
    CypherQuery,
    CypherUnionQuery,
    ExpressionText,
    GraphConstructor,
    LabelRef,
    MatchClause,
    NodePattern,
    ParameterRef,
    PathPatternKind,
    PatternElement,
    ProjectionStage,
    RelationshipPattern,
    ReturnClause,
    ReturnItem,
    UnwindClause,
    WhereClause,
    WherePredicate,
)
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundQueryPart, BoundVariable, ScopeFrame, SemanticTable
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import BoundPredicate, EdgeRef, ListType, LogicalType, NodeRef, PathType, ScalarType

CypherAST = Union[CypherQuery, CypherUnionQuery, CypherGraphQuery]
SchemaConfidence = Literal["declared", "propagated", "inferred"]

_KEYWORDS: FrozenSet[str] = frozenset(
    {
        "AND",
        "AS",
        "ASC",
        "ASCENDING",
        "BY",
        "CALL",
        "CASE",
        "CONTAINS",
        "DESC",
        "DESCENDING",
        "DISTINCT",
        "ELSE",
        "END",
        "ENDS",
        "FALSE",
        "IN",
        "IS",
        "LIMIT",
        "MATCH",
        "NOT",
        "NULL",
        "OPTIONAL",
        "OR",
        "ORDER",
        "RETURN",
        "SKIP",
        "STARTS",
        "THEN",
        "TRUE",
        "UNION",
        "UNWIND",
        "USE",
        "WHEN",
        "WHERE",
        "WITH",
        "XOR",
    }
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PROPERTY_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")
_PARAMETER_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_COUNT_CALL_RE = re.compile(r"(?i)^count\s*\(")
_STRING_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")


@dataclass
class _BindState:
    """Mutable binder state while constructing immutable BoundIR payloads."""

    scope: Dict[str, BoundVariable] = field(default_factory=dict)
    scope_confidence: Dict[str, SchemaConfidence] = field(default_factory=dict)
    query_parts: List[BoundQueryPart] = field(default_factory=list)
    scope_stack: List[ScopeFrame] = field(default_factory=list)
    parameter_names: Set[str] = field(default_factory=set)
    next_scope_id: int = 1
    optional_arm_count: int = 0
    strict_name_resolution: bool = False


class FrontendBinder:
    """Typed binder interface for Cypher frontend ASTs."""

    def bind(self, ast: CypherAST, ctx: PlanContext, *, strict_name_resolution: bool = False) -> BoundIR:
        """Bind frontend AST into frontend-neutral IR.

        Binder semantics are intentionally conservative in this stage: when a
        type or nullability cannot be derived exactly, binder keeps "unknown"
        scalar types and nullable=True.
        """
        if isinstance(ast, CypherUnionQuery):
            return self._bind_union_query(ast=ast, ctx=ctx, strict_name_resolution=strict_name_resolution)
        if isinstance(ast, CypherGraphQuery):
            return self._bind_graph_query(ast=ast, ctx=ctx, strict_name_resolution=strict_name_resolution)
        return self._bind_query(ast=ast, ctx=ctx, strict_name_resolution=strict_name_resolution)

    def _bind_query(self, ast: CypherQuery, ctx: PlanContext, *, strict_name_resolution: bool) -> BoundIR:
        state = _BindState(strict_name_resolution=strict_name_resolution)
        _collect_parameter_names(ast, out=state.parameter_names)

        if ast.row_sequence and not ast.matches and not ast.reentry_matches:
            self._bind_row_sequence(state=state, row_sequence=ast.row_sequence)
        else:
            self._bind_graph_sequence(state=state, ast=ast)

        if not any(part.clause == "RETURN" for part in state.query_parts):
            # Query shape can still carry a final RETURN clause outside row_sequence.
            self._bind_return_clause(state=state, clause=ast.return_)

        return _finalize_bound_ir(state=state, ctx=ctx)

    def _bind_union_query(self, ast: CypherUnionQuery, ctx: PlanContext, *, strict_name_resolution: bool) -> BoundIR:
        branch_irs = [self._bind_query(branch, ctx=ctx, strict_name_resolution=strict_name_resolution) for branch in ast.branches]
        union_scope: Dict[str, BoundVariable] = {}
        union_conf: Dict[str, SchemaConfidence] = {}

        for branch_ir in branch_irs:
            branch_conf = _schema_confidence_map(branch_ir)
            for name, branch_var in branch_ir.semantic_table.variables.items():
                existing = union_scope.get(name)
                if existing is None:
                    union_scope[name] = branch_var
                    union_conf[name] = branch_conf.get(name, "inferred")
                    continue
                union_scope[name] = BoundVariable(
                    name=name,
                    logical_type=_merge_logical_types(existing.logical_type, branch_var.logical_type),
                    nullable=existing.nullable or branch_var.nullable,
                    null_extended_from=existing.null_extended_from | branch_var.null_extended_from,
                    entity_kind=_merge_entity_kind(existing.entity_kind, branch_var.entity_kind),
                    scope_id=max(existing.scope_id, branch_var.scope_id),
                )
                union_conf[name] = _merge_confidence(union_conf[name], branch_conf.get(name, "inferred"))

        merged_parts: List[BoundQueryPart] = []
        merged_scopes: List[ScopeFrame] = []
        merged_params: Dict[str, object] = {
            "_binder_schema_confidence": {},
            "_binder_parameter_names": tuple(),
        }
        param_names: Set[str] = set()

        for idx, branch_ir in enumerate(branch_irs):
            merged_parts.extend(branch_ir.query_parts)
            merged_scopes.extend(branch_ir.scope_stack)
            branch_meta = _schema_confidence_map(branch_ir)
            merged_parts.append(
                BoundQueryPart(
                    clause="UNION_BRANCH",
                    inputs=frozenset(branch_ir.semantic_table.variables.keys()),
                    outputs=frozenset(branch_ir.semantic_table.variables.keys()),
                    predicates=[],
                    metadata={
                        "branch_index": idx,
                        "schema_confidence": dict(sorted(branch_meta.items())),
                    },
                )
            )
            param_names.update(_parameter_name_set(branch_ir))

        merged_parts.append(
            BoundQueryPart(
                clause="UNION",
                inputs=frozenset(union_scope.keys()),
                outputs=frozenset(union_scope.keys()),
                predicates=[],
                metadata={
                    "union_kind": ast.union_kind,
                    "schema_confidence": dict(sorted(union_conf.items())),
                },
            )
        )
        merged_scopes.append(
            ScopeFrame(
                visible_vars=frozenset(union_scope.keys()),
                schema=RowSchema(columns={name: var.logical_type for name, var in union_scope.items()}),
                origin_clause="UNION",
            )
        )
        merged_params["_binder_schema_confidence"] = dict(sorted(union_conf.items()))
        merged_params["_binder_parameter_names"] = tuple(sorted(param_names))

        return BoundIR(
            query_parts=merged_parts,
            semantic_table=SemanticTable(variables=union_scope),
            scope_stack=merged_scopes,
            params=merged_params,
        )

    def _bind_graph_query(self, ast: CypherGraphQuery, ctx: PlanContext, *, strict_name_resolution: bool) -> BoundIR:
        graph_binding_parts: List[BoundQueryPart] = []
        graph_binding_params: Set[str] = set()

        for binding in ast.graph_bindings:
            bound = self._bind_graph_constructor(
                binding.constructor,
                ctx=ctx,
                strict_name_resolution=strict_name_resolution,
            )
            graph_binding_parts.extend(bound.query_parts)
            graph_binding_parts.append(
                BoundQueryPart(
                    clause="GRAPH_BINDING",
                    inputs=frozenset(bound.semantic_table.variables.keys()),
                    outputs=frozenset(bound.semantic_table.variables.keys()),
                    predicates=[],
                    metadata={
                        "binding_name": binding.name,
                        "schema_confidence": _schema_confidence_map(bound),
                    },
                )
            )
            graph_binding_params.update(_parameter_name_set(bound))

        constructor_bound = self._bind_graph_constructor(
            ast.constructor,
            ctx=ctx,
            strict_name_resolution=strict_name_resolution,
        )
        final_params = dict(constructor_bound.params)
        param_names = set(_parameter_name_set(constructor_bound))
        param_names.update(graph_binding_params)
        final_params["_binder_parameter_names"] = tuple(sorted(param_names))

        return BoundIR(
            query_parts=graph_binding_parts + constructor_bound.query_parts,
            semantic_table=constructor_bound.semantic_table,
            scope_stack=constructor_bound.scope_stack,
            params=final_params,
        )

    def _bind_graph_constructor(
        self,
        constructor: GraphConstructor,
        ctx: PlanContext,
        *,
        strict_name_resolution: bool,
    ) -> BoundIR:
        state = _BindState(strict_name_resolution=strict_name_resolution)
        _collect_parameter_names(constructor, out=state.parameter_names)

        for clause in constructor.matches:
            self._bind_match_clause(state=state, clause=clause)
        if constructor.where is not None:
            self._append_where_part(state=state, clause_name="WHERE", where=constructor.where)
        if constructor.call is not None:
            self._bind_call_clause(state=state, clause=constructor.call)

        return _finalize_bound_ir(state=state, ctx=ctx)

    def _bind_row_sequence(self, state: _BindState, row_sequence: Sequence[Union[ProjectionStage, UnwindClause]]) -> None:
        for item in row_sequence:
            if isinstance(item, UnwindClause):
                self._bind_unwind_clause(state=state, clause=item)
                continue
            if item.clause.kind == "with":
                self._bind_projection_stage(state=state, stage=item, origin="WITH")
            else:
                self._bind_return_clause(state=state, clause=item.clause, stage=item)

    def _bind_graph_sequence(self, state: _BindState, ast: CypherQuery) -> None:
        for match_clause in ast.matches:
            self._bind_match_clause(state=state, clause=match_clause)

        for unwind_clause in ast.unwinds:
            self._bind_unwind_clause(state=state, clause=unwind_clause)

        if ast.reentry_matches:
            for idx, reentry_match in enumerate(ast.reentry_matches):
                if idx < len(ast.with_stages):
                    self._bind_projection_stage(state=state, stage=ast.with_stages[idx], origin="WITH")
                self._bind_match_clause(state=state, clause=reentry_match)
                if idx < len(ast.reentry_wheres) and ast.reentry_wheres[idx] is not None:
                    self._append_where_part(state=state, clause_name="WHERE", where=cast(WhereClause, ast.reentry_wheres[idx]))
                if idx < len(ast.reentry_unwinds):
                    self._bind_unwind_clause(state=state, clause=ast.reentry_unwinds[idx])
            for stage in ast.with_stages[len(ast.reentry_matches) :]:
                self._bind_projection_stage(state=state, stage=stage, origin="WITH")
            return

        for stage in ast.with_stages:
            self._bind_projection_stage(state=state, stage=stage, origin="WITH")

    def _bind_match_clause(self, state: _BindState, clause: MatchClause) -> None:
        inputs = frozenset(state.scope.keys())
        clause_scope_id = _next_scope_id(state)
        clause_kind = "OPTIONAL MATCH" if clause.optional else "MATCH"
        arm_id: Optional[str] = None
        changed_aliases: Set[str] = set()

        if clause.optional:
            state.optional_arm_count += 1
            arm_id = "optional_arm_%d" % state.optional_arm_count

        for pattern_idx, pattern in enumerate(clause.patterns):
            pattern_alias = clause.pattern_aliases[pattern_idx] if pattern_idx < len(clause.pattern_aliases) else None
            pattern_kind = clause.pattern_alias_kinds[pattern_idx] if pattern_idx < len(clause.pattern_alias_kinds) else "pattern"

            path_hops = _path_hops(pattern)

            for element_idx, element in enumerate(pattern):
                if isinstance(element, NodePattern):
                    if element.variable is not None:
                        changed_aliases.add(
                            self._bind_node_pattern(
                                state=state,
                                alias=element.variable,
                                labels=frozenset(element.labels),
                                clause_scope_id=clause_scope_id,
                                optional_arm_id=arm_id,
                            )
                        )
                    continue

                if element.variable is not None:
                    src_labels = _node_labels_at(pattern=pattern, idx=element_idx - 1)
                    dst_labels = _node_labels_at(pattern=pattern, idx=element_idx + 1)
                    changed_aliases.add(
                        self._bind_relationship_pattern(
                            state=state,
                            alias=element.variable,
                            relationship=element,
                            src_labels=src_labels,
                            dst_labels=dst_labels,
                            clause_scope_id=clause_scope_id,
                            optional_arm_id=arm_id,
                        )
                    )

            if pattern_alias is not None:
                changed_aliases.add(
                    self._bind_path_alias(
                        state=state,
                        alias=pattern_alias,
                        kind=pattern_kind,
                        path_hops=path_hops,
                        clause_scope_id=clause_scope_id,
                        optional_arm_id=arm_id,
                    )
                )

        predicates: List[BoundPredicate] = []
        if clause.where is not None:
            predicates.extend(_where_predicates(clause.where))
            _collect_parameter_names(clause.where, out=state.parameter_names)
            changed_aliases.update(_apply_where_label_narrowing(state=state, where=clause.where))

        outputs = frozenset(state.scope.keys())
        schema_update = {name: state.scope_confidence[name] for name in sorted(changed_aliases) if name in state.scope_confidence}

        state.query_parts.append(
            BoundQueryPart(
                clause=clause_kind,
                inputs=inputs,
                outputs=outputs,
                predicates=predicates,
                metadata={
                    "arm_id": arm_id,
                    "schema_confidence": schema_update,
                },
            )
        )
        _append_scope_frame(state=state, origin_clause=clause_kind)

    def _bind_projection_stage(self, state: _BindState, stage: ProjectionStage, origin: str) -> None:
        inputs = frozenset(state.scope.keys())
        clause_scope_id = _next_scope_id(state)
        next_scope, next_confidence = self._project_items(
            state=state,
            items=stage.clause.items,
            clause_scope_id=clause_scope_id,
        )

        predicates: List[BoundPredicate] = []
        if stage.where is not None:
            predicates.append(BoundPredicate(expression=stage.where.text))
            _collect_parameter_names(stage.where, out=state.parameter_names)

        state.scope = next_scope
        state.scope_confidence = next_confidence
        state.query_parts.append(
            BoundQueryPart(
                clause=origin,
                inputs=inputs,
                outputs=frozenset(next_scope.keys()),
                predicates=predicates,
                metadata={
                    "distinct": stage.clause.distinct,
                    "schema_confidence": dict(sorted(next_confidence.items())),
                },
            )
        )
        _append_scope_frame(state=state, origin_clause=origin)

    def _bind_unwind_clause(self, state: _BindState, clause: UnwindClause) -> None:
        inputs = frozenset(state.scope.keys())
        clause_scope_id = _next_scope_id(state)
        binding = _infer_unwind_binding(
            expression=clause.expression,
            scope=state.scope,
            confidence=state.scope_confidence,
            strict_name_resolution=state.strict_name_resolution,
        )
        _collect_parameter_names(clause.expression, out=state.parameter_names)

        next_scope = dict(state.scope)
        next_conf = dict(state.scope_confidence)
        next_scope[clause.alias] = BoundVariable(
            name=clause.alias,
            logical_type=binding.logical_type,
            nullable=binding.nullable,
            null_extended_from=binding.null_extended_from,
            entity_kind="scalar",
            scope_id=clause_scope_id,
        )
        next_conf[clause.alias] = binding.schema_confidence

        state.scope = next_scope
        state.scope_confidence = next_conf
        state.query_parts.append(
            BoundQueryPart(
                clause="UNWIND",
                inputs=inputs,
                outputs=frozenset(next_scope.keys()),
                predicates=[BoundPredicate(expression=clause.expression.text)],
                metadata={
                    "alias": clause.alias,
                    "schema_confidence": {clause.alias: binding.schema_confidence},
                },
            )
        )
        _append_scope_frame(state=state, origin_clause="UNWIND")

    def _bind_return_clause(self, state: _BindState, clause: ReturnClause, stage: Optional[ProjectionStage] = None) -> None:
        inputs = frozenset(state.scope.keys())
        clause_scope_id = _next_scope_id(state)
        next_scope, next_confidence = self._project_items(
            state=state,
            items=clause.items,
            clause_scope_id=clause_scope_id,
        )

        predicates: List[BoundPredicate] = []
        if stage is not None and stage.where is not None:
            predicates.append(BoundPredicate(expression=stage.where.text))
            _collect_parameter_names(stage.where, out=state.parameter_names)

        state.scope = next_scope
        state.scope_confidence = next_confidence
        state.query_parts.append(
            BoundQueryPart(
                clause="RETURN",
                inputs=inputs,
                outputs=frozenset(next_scope.keys()),
                predicates=predicates,
                metadata={
                    "distinct": clause.distinct,
                    "schema_confidence": dict(sorted(next_confidence.items())),
                },
            )
        )
        _append_scope_frame(state=state, origin_clause="RETURN")

    def _project_items(
        self,
        *,
        state: _BindState,
        items: Sequence[ReturnItem],
        clause_scope_id: int,
    ) -> Tuple[Dict[str, BoundVariable], Dict[str, SchemaConfidence]]:
        next_scope: Dict[str, BoundVariable] = {}
        next_confidence: Dict[str, SchemaConfidence] = {}
        for item in items:
            out_name = item.alias or item.expression.text
            binding = _infer_expression_binding(
                expression=item.expression,
                scope=state.scope,
                confidence=state.scope_confidence,
                strict_name_resolution=state.strict_name_resolution,
            )
            _collect_parameter_names(item.expression, out=state.parameter_names)
            next_scope[out_name] = BoundVariable(
                name=out_name,
                logical_type=binding.logical_type,
                nullable=binding.nullable,
                null_extended_from=binding.null_extended_from,
                entity_kind=binding.entity_kind,
                scope_id=clause_scope_id,
            )
            next_confidence[out_name] = binding.schema_confidence
        return next_scope, next_confidence

    def _append_where_part(self, state: _BindState, clause_name: str, where: WhereClause) -> None:
        state.query_parts.append(
            BoundQueryPart(
                clause=clause_name,
                inputs=frozenset(state.scope.keys()),
                outputs=frozenset(state.scope.keys()),
                predicates=_where_predicates(where),
                metadata={
                    "schema_confidence": dict(sorted(state.scope_confidence.items())),
                },
            )
        )

    def _bind_call_clause(self, state: _BindState, clause: CallClause) -> None:
        inputs = frozenset(state.scope.keys())
        clause_scope_id = _next_scope_id(state)
        next_scope = dict(state.scope)
        next_conf = dict(state.scope_confidence)

        if clause.yield_items:
            for item in clause.yield_items:
                output_name = item.alias or item.name
                next_scope[output_name] = BoundVariable(
                    name=output_name,
                    logical_type=ScalarType(kind="unknown", nullable=True),
                    nullable=True,
                    null_extended_from=frozenset(),
                    entity_kind="scalar",
                    scope_id=clause_scope_id,
                )
                next_conf[output_name] = "inferred"

        state.scope = next_scope
        state.scope_confidence = next_conf
        state.query_parts.append(
            BoundQueryPart(
                clause="CALL",
                inputs=inputs,
                outputs=frozenset(next_scope.keys()),
                predicates=[BoundPredicate(expression=arg.text) for arg in clause.args],
                metadata={
                    "procedure": clause.procedure,
                    "schema_confidence": dict(sorted(next_conf.items())),
                },
            )
        )
        _append_scope_frame(state=state, origin_clause="CALL")

    def _bind_node_pattern(
        self,
        *,
        state: _BindState,
        alias: str,
        labels: FrozenSet[str],
        clause_scope_id: int,
        optional_arm_id: Optional[str],
    ) -> str:
        logical_type = NodeRef(labels=labels)
        existing = state.scope.get(alias)
        if existing is None:
            state.scope[alias] = BoundVariable(
                name=alias,
                logical_type=logical_type,
                nullable=optional_arm_id is not None,
                null_extended_from=frozenset({optional_arm_id}) if optional_arm_id is not None else frozenset(),
                entity_kind="node",
                scope_id=clause_scope_id,
            )
            state.scope_confidence[alias] = "declared"
            return alias

        merged = BoundVariable(
            name=alias,
            logical_type=_merge_logical_types(existing.logical_type, logical_type),
            nullable=existing.nullable,
            null_extended_from=existing.null_extended_from,
            entity_kind="node",
            scope_id=existing.scope_id,
        )
        state.scope[alias] = merged
        state.scope_confidence[alias] = _merge_confidence(state.scope_confidence.get(alias, "declared"), "declared")
        return alias

    def _bind_relationship_pattern(
        self,
        *,
        state: _BindState,
        alias: str,
        relationship: RelationshipPattern,
        src_labels: FrozenSet[str],
        dst_labels: FrozenSet[str],
        clause_scope_id: int,
        optional_arm_id: Optional[str],
    ) -> str:
        rel_type = relationship.types[0] if len(relationship.types) == 1 else None
        logical_type = EdgeRef(
            type=rel_type,
            src_label=_first_or_none(src_labels),
            dst_label=_first_or_none(dst_labels),
        )
        existing = state.scope.get(alias)
        if existing is None:
            state.scope[alias] = BoundVariable(
                name=alias,
                logical_type=logical_type,
                nullable=optional_arm_id is not None,
                null_extended_from=frozenset({optional_arm_id}) if optional_arm_id is not None else frozenset(),
                entity_kind="edge",
                scope_id=clause_scope_id,
            )
            state.scope_confidence[alias] = "declared"
            return alias

        merged = BoundVariable(
            name=alias,
            logical_type=_merge_logical_types(existing.logical_type, logical_type),
            nullable=existing.nullable,
            null_extended_from=existing.null_extended_from,
            entity_kind="edge",
            scope_id=existing.scope_id,
        )
        state.scope[alias] = merged
        state.scope_confidence[alias] = _merge_confidence(state.scope_confidence.get(alias, "declared"), "declared")
        return alias

    def _bind_path_alias(
        self,
        *,
        state: _BindState,
        alias: str,
        kind: PathPatternKind,
        path_hops: Tuple[int, int],
        clause_scope_id: int,
        optional_arm_id: Optional[str],
    ) -> str:
        if kind == "allShortestPaths":
            logical_type: LogicalType = ListType(element_type=PathType(min_hops=path_hops[0], max_hops=path_hops[1]))
        else:
            logical_type = PathType(min_hops=path_hops[0], max_hops=path_hops[1])

        existing = state.scope.get(alias)
        if existing is None:
            state.scope[alias] = BoundVariable(
                name=alias,
                logical_type=logical_type,
                nullable=optional_arm_id is not None,
                null_extended_from=frozenset({optional_arm_id}) if optional_arm_id is not None else frozenset(),
                entity_kind="scalar",
                scope_id=clause_scope_id,
            )
            state.scope_confidence[alias] = "declared"
            return alias

        state.scope[alias] = BoundVariable(
            name=alias,
            logical_type=_merge_logical_types(existing.logical_type, logical_type),
            nullable=existing.nullable,
            null_extended_from=existing.null_extended_from,
            entity_kind="scalar",
            scope_id=existing.scope_id,
        )
        state.scope_confidence[alias] = _merge_confidence(state.scope_confidence.get(alias, "declared"), "declared")
        return alias


@dataclass(frozen=True)
class _ExpressionBinding:
    logical_type: LogicalType
    entity_kind: Literal["node", "edge", "scalar"]
    nullable: bool
    null_extended_from: FrozenSet[str]
    schema_confidence: SchemaConfidence


def _infer_expression_binding(
    *,
    expression: ExpressionText,
    scope: Mapping[str, BoundVariable],
    confidence: Mapping[str, SchemaConfidence],
    strict_name_resolution: bool,
) -> _ExpressionBinding:
    text = expression.text.strip()
    if _COUNT_CALL_RE.match(text):
        return _ExpressionBinding(
            logical_type=ScalarType(kind="int64", nullable=False),
            entity_kind="scalar",
            nullable=False,
            null_extended_from=frozenset(),
            schema_confidence="declared",
        )

    direct = scope.get(text)
    if direct is not None:
        return _ExpressionBinding(
            logical_type=direct.logical_type,
            entity_kind=direct.entity_kind,
            nullable=direct.nullable,
            null_extended_from=direct.null_extended_from,
            schema_confidence=confidence.get(text, "propagated"),
        )

    prop_match = _PROPERTY_RE.fullmatch(text)
    if prop_match is not None:
        alias = prop_match.group(1)
        source = scope.get(alias)
        if source is None and strict_name_resolution:
            raise _unresolved_name_error(identifier=alias, visible_scope=scope)
        if source is None:
            return _ExpressionBinding(
                logical_type=ScalarType(kind="unknown", nullable=True),
                entity_kind="scalar",
                nullable=True,
                null_extended_from=frozenset(),
                schema_confidence="inferred",
            )
        return _ExpressionBinding(
            logical_type=ScalarType(kind="unknown", nullable=True),
            entity_kind="scalar",
            nullable=True,
            null_extended_from=source.null_extended_from,
            schema_confidence=_demote_confidence(confidence.get(alias, "propagated")),
        )

    if strict_name_resolution:
        unresolved = _unresolved_identifiers(text=text, scope=scope)
        if unresolved:
            raise _unresolved_name_error(identifier=sorted(unresolved)[0], visible_scope=scope)

    if _is_bool_literal(text):
        return _ExpressionBinding(
            logical_type=ScalarType(kind="bool", nullable=False),
            entity_kind="scalar",
            nullable=False,
            null_extended_from=frozenset(),
            schema_confidence="declared",
        )
    if _is_int_literal(text):
        return _ExpressionBinding(
            logical_type=ScalarType(kind="int64", nullable=False),
            entity_kind="scalar",
            nullable=False,
            null_extended_from=frozenset(),
            schema_confidence="declared",
        )
    if _is_float_literal(text):
        return _ExpressionBinding(
            logical_type=ScalarType(kind="float64", nullable=False),
            entity_kind="scalar",
            nullable=False,
            null_extended_from=frozenset(),
            schema_confidence="declared",
        )
    if _is_null_literal(text):
        return _ExpressionBinding(
            logical_type=ScalarType(kind="null", nullable=True),
            entity_kind="scalar",
            nullable=True,
            null_extended_from=frozenset(),
            schema_confidence="declared",
        )
    if _is_string_literal(text):
        return _ExpressionBinding(
            logical_type=ScalarType(kind="string", nullable=False),
            entity_kind="scalar",
            nullable=False,
            null_extended_from=frozenset(),
            schema_confidence="declared",
        )
    if _looks_like_list_literal(text):
        return _ExpressionBinding(
            logical_type=ListType(element_type=ScalarType(kind="unknown", nullable=True)),
            entity_kind="scalar",
            nullable=False,
            null_extended_from=frozenset(),
            schema_confidence="inferred",
        )
    if text.startswith("shortestPath(") or text.startswith("allShortestPaths("):
        path_type = PathType(min_hops=1, max_hops=1)
        if text.startswith("allShortestPaths("):
            logical: LogicalType = ListType(element_type=path_type)
        else:
            logical = path_type
        return _ExpressionBinding(
            logical_type=logical,
            entity_kind="scalar",
            nullable=True,
            null_extended_from=frozenset(),
            schema_confidence="inferred",
        )

    refs = _referenced_aliases(text=text, scope=scope)
    null_extended_from: FrozenSet[str] = frozenset()
    nullable = True
    schema_conf: SchemaConfidence = "declared"

    if refs:
        nullable = any(scope[ref].nullable for ref in refs)
        for ref in refs:
            null_extended_from |= scope[ref].null_extended_from
            schema_conf = _min_rule_confidence(schema_conf, confidence.get(ref, "inferred"))
    else:
        schema_conf = "inferred"

    return _ExpressionBinding(
        logical_type=ScalarType(kind="unknown", nullable=nullable),
        entity_kind="scalar",
        nullable=nullable,
        null_extended_from=null_extended_from,
        schema_confidence=schema_conf,
    )


def _infer_unwind_binding(
    *,
    expression: ExpressionText,
    scope: Mapping[str, BoundVariable],
    confidence: Mapping[str, SchemaConfidence],
    strict_name_resolution: bool,
) -> _ExpressionBinding:
    expr_binding = _infer_expression_binding(
        expression=expression,
        scope=scope,
        confidence=confidence,
        strict_name_resolution=strict_name_resolution,
    )
    expr_type = expr_binding.logical_type
    if isinstance(expr_type, ListType):
        return _ExpressionBinding(
            logical_type=expr_type.element_type,
            entity_kind="scalar",
            nullable=True,
            null_extended_from=expr_binding.null_extended_from,
            schema_confidence=expr_binding.schema_confidence,
        )
    return _ExpressionBinding(
        logical_type=ScalarType(kind="unknown", nullable=True),
        entity_kind="scalar",
        nullable=True,
        null_extended_from=expr_binding.null_extended_from,
        schema_confidence="inferred",
    )


def _collect_parameter_names(obj: object, *, out: Set[str]) -> None:
    names = out
    stack: List[object] = [obj]
    visited: Set[int] = set()

    while stack:
        current = stack.pop()
        marker = id(current)
        if marker in visited:
            continue
        visited.add(marker)
        if isinstance(current, ParameterRef):
            names.add(current.name)
            continue
        if isinstance(current, ExpressionText):
            for match in _PARAMETER_RE.finditer(current.text):
                names.add(match.group(1))
            continue
        if isinstance(current, (str, int, float, bool)) or current is None:
            continue
        if isinstance(current, tuple) or isinstance(current, list):
            stack.extend(current)
            continue
        if isinstance(current, dict):
            stack.extend(current.keys())
            stack.extend(current.values())
            continue
        if hasattr(current, "__dict__"):
            stack.extend(vars(current).values())


def _append_scope_frame(state: _BindState, *, origin_clause: str) -> None:
    state.scope_stack.append(
        ScopeFrame(
            visible_vars=frozenset(state.scope.keys()),
            schema=RowSchema(columns={name: value.logical_type for name, value in state.scope.items()}),
            origin_clause=origin_clause,
        )
    )


def _finalize_bound_ir(state: _BindState, ctx: PlanContext) -> BoundIR:
    params = dict(ctx.catalog.metadata)
    params["_binder_schema_confidence"] = dict(sorted(state.scope_confidence.items()))
    params["_binder_parameter_names"] = tuple(sorted(state.parameter_names))
    return BoundIR(
        query_parts=list(state.query_parts),
        semantic_table=SemanticTable(variables=dict(state.scope)),
        scope_stack=list(state.scope_stack),
        params=params,
    )


def _schema_confidence_map(bound_ir: BoundIR) -> Dict[str, SchemaConfidence]:
    raw = bound_ir.params.get("_binder_schema_confidence", {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, SchemaConfidence] = {}
    for key, value in raw.items():
        if isinstance(key, str) and value in {"declared", "propagated", "inferred"}:
            out[key] = cast(SchemaConfidence, value)
    return out


def _parameter_name_set(bound_ir: BoundIR) -> Set[str]:
    raw = bound_ir.params.get("_binder_parameter_names", ())
    if not isinstance(raw, (tuple, list)):
        return set()
    names: Set[str] = set()
    for value in raw:
        if isinstance(value, str):
            names.add(value)
    return names


def _flatten_top_level_ands(expr: BooleanExpr) -> List[BooleanExpr]:
    """Flatten left-associated AND chains into a flat list of conjuncts.

    ``and(and(a, b), c)`` → ``[a, b, c]``.  Non-AND nodes stop the
    recursion: ``or(a, b)`` returns ``[or(a, b)]`` (one conjunct),
    ``not(a)`` returns ``[not(a)]``, atoms return ``[atom]``.

    Used by :func:`_where_predicates` to emit one ``BoundPredicate``
    per top-level AND conjunct so downstream passes (predicate pushdown)
    don't have to re-parse compound expression text.
    """
    if expr.op != "and":
        return [expr]
    conjuncts: List[BooleanExpr] = []
    if expr.left is not None:
        conjuncts.extend(_flatten_top_level_ands(expr.left))
    if expr.right is not None:
        conjuncts.extend(_flatten_top_level_ands(expr.right))
    return conjuncts


_BOOLEAN_OP_KEYWORD: Dict[str, str] = {
    "and": "AND",
    "or": "OR",
    "xor": "XOR",
}


def _boolean_expr_to_text(expr: BooleanExpr) -> str:
    """Reconstruct surface text for a boolean-expression subtree.

    Atoms emit ``atom_text`` directly.  Branches stringify recursively
    with parentheses around any branch operand to keep operator
    precedence unambiguous when the result is later parsed back as a
    single conjunct.  ``NOT`` prefixes its operand; binary ops produce
    ``"L OP R"``.

    Inherits the slice-1 known limitation for primitive literal atoms
    (``str(True) == "True"`` rather than Cypher ``"true"``); that is a
    follow-up under #1200 to be addressed when literal transformers
    gain span-carrying wrappers.
    """
    if expr.op == "atom":
        return expr.atom_text or ""
    if expr.op == "not":
        operand = _boolean_expr_to_text(expr.left) if expr.left is not None else ""
        if expr.left is not None and expr.left.op != "atom":
            operand = f"({operand})"
        return f"NOT {operand}"
    keyword = _BOOLEAN_OP_KEYWORD.get(expr.op)
    if keyword is None or expr.left is None or expr.right is None:
        return expr.atom_text or ""
    left = _boolean_expr_to_text(expr.left)
    right = _boolean_expr_to_text(expr.right)
    if expr.left.op != "atom":
        left = f"({left})"
    if expr.right.op != "atom":
        right = f"({right})"
    return f"{left} {keyword} {right}"


def _where_predicates(where: WhereClause) -> List[BoundPredicate]:
    # Routing invariant (parser.py): ``where.predicates`` and
    # ``where.expr`` / ``where.expr_tree`` are mutually exclusive at the
    # parse layer.  Structured ``where_predicates`` populates the former
    # and returns; the generic ``expr`` route populates the latter (and
    # ``expr_tree`` only fires on it).  We accept all three populated
    # defensively in case a future parser refactor introduces overlap,
    # but no current path exercises that combination.
    predicates = [BoundPredicate(expression=str(term)) for term in where.predicates]
    if where.expr_tree is not None:
        # Slice 2 of #1200: emit one BoundPredicate per top-level AND conjunct
        # so downstream passes (e.g. predicate pushdown) walk pre-split
        # conjuncts rather than re-parsing a compound expression string.
        for conjunct in _flatten_top_level_ands(where.expr_tree):
            predicates.append(BoundPredicate(expression=_boolean_expr_to_text(conjunct)))
    elif where.expr is not None:
        predicates.append(BoundPredicate(expression=where.expr.text))
    return predicates


def _apply_where_label_narrowing(state: _BindState, where: WhereClause) -> Set[str]:
    narrowed: Dict[str, Set[str]] = {}

    for term in where.predicates:
        if isinstance(term, WherePredicate) and term.op == "has_labels" and isinstance(term.left, LabelRef):
            labels = narrowed.setdefault(term.left.alias, set())
            labels.update(term.left.labels)

    changed: Set[str] = set()
    for alias, labels in narrowed.items():
        existing = state.scope.get(alias)
        if existing is None:
            continue
        if existing.entity_kind != "node" or not isinstance(existing.logical_type, NodeRef):
            continue
        updated_labels = existing.logical_type.labels | frozenset(labels)
        state.scope[alias] = BoundVariable(
            name=existing.name,
            logical_type=NodeRef(labels=updated_labels),
            nullable=existing.nullable,
            null_extended_from=existing.null_extended_from,
            entity_kind=existing.entity_kind,
            scope_id=existing.scope_id,
        )
        state.scope_confidence[alias] = _min_rule_confidence(state.scope_confidence.get(alias, "declared"), "declared")
        changed.add(alias)

    return changed


def _referenced_aliases(text: str, scope: Mapping[str, BoundVariable]) -> Set[str]:
    refs: Set[str] = set()

    for match in _PROPERTY_RE.finditer(text):
        alias = match.group(1)
        if alias in scope:
            refs.add(alias)

    for match in _IDENTIFIER_RE.finditer(text):
        token = match.group(0)
        if token.upper() in _KEYWORDS:
            continue
        if token in scope:
            refs.add(token)

    return refs


def _unresolved_identifiers(*, text: str, scope: Mapping[str, BoundVariable]) -> Set[str]:
    unresolved: Set[str] = set()
    string_spans = _string_literal_spans(text)

    def in_string(index: int) -> bool:
        for start, end in string_spans:
            if start <= index < end:
                return True
        return False

    for match in _PROPERTY_RE.finditer(text):
        if in_string(match.start()):
            continue
        alias = match.group(1)
        if alias not in scope:
            unresolved.add(alias)

    for match in _IDENTIFIER_RE.finditer(text):
        token = match.group(0)
        if token.upper() in _KEYWORDS:
            continue
        start, end = match.span()
        if in_string(start):
            continue

        prev = start - 1
        while prev >= 0 and text[prev].isspace():
            prev -= 1
        prev_char = text[prev] if prev >= 0 else ""

        nxt = end
        while nxt < len(text) and text[nxt].isspace():
            nxt += 1
        next_char = text[nxt] if nxt < len(text) else ""

        if prev_char in {"$", ".", ":"}:
            continue
        if next_char == "(":
            continue
        if next_char == ":" and (prev_char == "" or prev_char in "{,"):
            continue

        if token not in scope:
            unresolved.add(token)

    return unresolved


def _string_literal_spans(text: str) -> List[Tuple[int, int]]:
    return [(match.start(), match.end()) for match in _STRING_LITERAL_RE.finditer(text)]


def _path_hops(pattern: Sequence[PatternElement]) -> Tuple[int, int]:
    min_hops = 0
    max_hops = 0
    for element in pattern:
        if not isinstance(element, RelationshipPattern):
            continue
        rel_min = element.min_hops if element.min_hops is not None else 1
        rel_max = element.max_hops if element.max_hops is not None else rel_min
        min_hops += rel_min
        max_hops += rel_max
    return max(min_hops, 1), max(max_hops, 1)


def _node_labels_at(pattern: Sequence[PatternElement], idx: int) -> FrozenSet[str]:
    if idx < 0 or idx >= len(pattern):
        return frozenset()
    element = pattern[idx]
    if isinstance(element, NodePattern):
        return frozenset(element.labels)
    return frozenset()


def _merge_logical_types(left: LogicalType, right: LogicalType) -> LogicalType:
    if isinstance(left, NodeRef) and isinstance(right, NodeRef):
        return NodeRef(labels=left.labels | right.labels)
    if isinstance(left, EdgeRef) and isinstance(right, EdgeRef):
        return EdgeRef(
            type=left.type if left.type == right.type else None,
            src_label=left.src_label if left.src_label == right.src_label else None,
            dst_label=left.dst_label if left.dst_label == right.dst_label else None,
        )
    if isinstance(left, PathType) and isinstance(right, PathType):
        return PathType(min_hops=min(left.min_hops, right.min_hops), max_hops=max(left.max_hops, right.max_hops))
    if isinstance(left, ListType) and isinstance(right, ListType):
        return ListType(element_type=_merge_logical_types(left.element_type, right.element_type))
    return right


def _merge_entity_kind(
    left: Literal["node", "edge", "scalar"],
    right: Literal["node", "edge", "scalar"],
) -> Literal["node", "edge", "scalar"]:
    if left == right:
        return left
    return "scalar"


def _merge_confidence(left: SchemaConfidence, right: SchemaConfidence) -> SchemaConfidence:
    return _min_rule_confidence(left, right)


def _min_rule_confidence(left: SchemaConfidence, right: SchemaConfidence) -> SchemaConfidence:
    order = {
        "declared": 0,
        "propagated": 1,
        "inferred": 2,
    }
    return left if order[left] >= order[right] else right


def _demote_confidence(value: SchemaConfidence) -> SchemaConfidence:
    if value == "declared":
        return "propagated"
    if value == "propagated":
        return "inferred"
    return "inferred"


def _first_or_none(values: Iterable[str]) -> Optional[str]:
    for value in values:
        return value
    return None


def _next_scope_id(state: _BindState) -> int:
    out = state.next_scope_id
    state.next_scope_id += 1
    return out


def _is_int_literal(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+", text))


def _is_float_literal(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+\.\d+", text))


def _is_bool_literal(text: str) -> bool:
    return text in {"true", "false", "TRUE", "FALSE"}


def _is_null_literal(text: str) -> bool:
    return text in {"null", "NULL"}


def _is_string_literal(text: str) -> bool:
    return len(text) >= 2 and text[0] == "'" and text[-1] == "'"


def _looks_like_list_literal(text: str) -> bool:
    return len(text) >= 2 and text[0] == "[" and text[-1] == "]"


def _unresolved_name_error(identifier: str, visible_scope: Mapping[str, BoundVariable]) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E204,
        "Unresolved identifier in binder scope",
        field="identifier",
        value=identifier,
        suggestion="Introduce the alias in MATCH/WITH before referencing it.",
        visible_scope=sorted(visible_scope.keys()),
        language="cypher",
    )
