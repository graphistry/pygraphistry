"""Initial M2 logical planner skeleton.

This module provides a minimal, pure planning contract from BoundIR to
LogicalPlan while assigning stable operator IDs.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Literal, Mapping, Optional

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundQueryPart, BoundVariable
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import (
    Distinct,
    Filter,
    LogicalPlan,
    NodeScan,
    PatternMatch,
    Project,
    RowSchema,
    Unwind,
)
from graphistry.compute.gfql.ir.types import EdgeRef, LogicalType, NodeRef, PathType, ScalarType


class IdGen:
    """Monotonic plan-local operator id generator."""

    def __init__(self, start: int = 1) -> None:
        self._next = start

    def next(self) -> int:
        value = self._next
        self._next += 1
        return value


class LogicalPlanner:
    """Initial planner skeleton from BoundIR to LogicalPlan."""

    def __init__(self, *, allow_unknown_match_aliases: bool = False) -> None:
        self._allow_unknown_match_aliases = allow_unknown_match_aliases

    def plan(self, bound_ir: BoundIR, ctx: PlanContext) -> LogicalPlan:
        """Build a minimal logical plan root for supported M2 skeleton shapes."""
        _ = ctx
        id_gen = IdGen()
        current: Optional[LogicalPlan] = None
        vars_by_name = bound_ir.semantic_table.variables
        seen_match = False

        for part_index, part in enumerate(bound_ir.query_parts):
            clause = part.clause.upper()
            part_vars = self._vars_for_part(bound_ir, part_index=part_index, fallback=vars_by_name)
            if clause == "OPTIONAL MATCH":
                scope_visible_aliases = (
                    bound_ir.scope_stack[part_index].visible_vars
                    if part_index < len(bound_ir.scope_stack)
                    else frozenset()
                )
                self._reject_unsupported_match_shape(
                    part=part,
                    vars_by_name=part_vars,
                    scope_visible_aliases=scope_visible_aliases,
                )
                metadata_arm_id = part.metadata.get("arm_id")
                is_top_level_optional = current is None and not seen_match and part_index == 0
                arm_id = (
                    "top_level_optional_0"
                    if is_top_level_optional
                    else metadata_arm_id if isinstance(metadata_arm_id, str) and metadata_arm_id else f"optional_arm_{part_index}"
                )
                current = self._plan_match(
                    part=part,
                    vars_by_name=part_vars,
                    id_gen=id_gen,
                    optional=True,
                    input=None if is_top_level_optional else current,
                    arm_id=arm_id,
                )
                current = self._apply_predicates(part=part, current=current, vars_by_name=part_vars, id_gen=id_gen)
                seen_match = True
                continue
            if clause == "MATCH":
                scope_visible_aliases = (
                    bound_ir.scope_stack[part_index].visible_vars
                    if part_index < len(bound_ir.scope_stack)
                    else frozenset()
                )
                scope_schema = (
                    bound_ir.scope_stack[part_index].schema
                    if part_index < len(bound_ir.scope_stack)
                    else RowSchema()
                )
                if seen_match and self._match_part_has_path_alias(part=part, scope_schema=scope_schema):
                    raise GFQLValidationError(
                        ErrorCode.E108,
                        "LogicalPlanner skeleton does not yet support multiple MATCH stages containing path aliases",
                        field="clause",
                        value=part.clause,
                        suggestion="Split shortestPath/path-alias multi-MATCH shapes until chained path planning is implemented.",
                        logical_plan_defer_code="multiple_match_stages",
                    )
                self._reject_unsupported_match_shape(
                    part=part,
                    vars_by_name=part_vars,
                    scope_visible_aliases=scope_visible_aliases,
                )
                current = self._plan_match(
                    part=part,
                    vars_by_name=part_vars,
                    id_gen=id_gen,
                    input=current if seen_match else None,
                )
                current = self._apply_predicates(part=part, current=current, vars_by_name=part_vars, id_gen=id_gen)
                seen_match = True
                continue
            if clause == "WHERE":
                current = self._plan_where(part=part, current=current, vars_by_name=part_vars, id_gen=id_gen)
                continue
            if clause in {"WITH", "RETURN"}:
                current = self._plan_projection(part=part, current=current, vars_by_name=part_vars, id_gen=id_gen)
                if part.metadata.get("distinct", False):
                    current = Distinct(
                        op_id=id_gen.next(),
                        input=current,
                        output_schema=current.output_schema,
                    )
                current = self._apply_predicates(part=part, current=current, vars_by_name=part_vars, id_gen=id_gen)
                continue
            if clause == "UNWIND":
                current = self._plan_unwind(part=part, current=current, vars_by_name=part_vars, id_gen=id_gen)
                continue
            raise GFQLValidationError(
                ErrorCode.E108,
                "LogicalPlanner skeleton does not support this clause type",
                field="clause",
                value=part.clause,
                suggestion="Use covered clause shapes only in M2-PR1 (MATCH/WHERE/WITH/RETURN/UNWIND).",
            )

        if current is None:
            # Keep fallback deterministic for empty / unsupported skeleton inputs.
            current = Project(op_id=id_gen.next(), expressions=[], output_schema=RowSchema(columns={}))

        return current

    def _plan_match(
        self,
        *,
        part: BoundQueryPart,
        vars_by_name: Mapping[str, BoundVariable],
        id_gen: IdGen,
        optional: bool = False,
        input: Optional[LogicalPlan] = None,
        arm_id: Optional[str] = None,
    ) -> LogicalPlan:
        aliases = sorted(self._match_aliases_for_part(part, vars_by_name=vars_by_name))
        schema = self._schema_for_aliases(alias_names=aliases, vars_by_name=vars_by_name)
        if input is None and not optional and len(aliases) == 1:
            variable = vars_by_name.get(aliases[0])
            if variable is not None and variable.entity_kind == "node":
                return NodeScan(
                    op_id=id_gen.next(),
                    label=self._first_node_label(var_names=aliases, vars_by_name=vars_by_name),
                    output_schema=schema,
                )
        return PatternMatch(
            op_id=id_gen.next(),
            pattern={"aliases": tuple(aliases)},
            input=input,
            optional=optional,
            arm_id=arm_id,
            output_schema=schema,
        )

    def _reject_unsupported_match_shape(
        self,
        *,
        part: BoundQueryPart,
        vars_by_name: Mapping[str, BoundVariable],
        scope_visible_aliases: FrozenSet[str] = frozenset(),
    ) -> None:
        alias_names = part.outputs or part.inputs
        if not alias_names:
            return
        has_known_alias = False
        for alias in alias_names:
            variable = vars_by_name.get(alias)
            if variable is not None:
                if variable.entity_kind not in {"node", "edge"}:
                    if isinstance(variable.logical_type, PathType):
                        continue
                    raise GFQLValidationError(
                        ErrorCode.E108,
                        "LogicalPlanner skeleton only supports MATCH outputs bound to node/edge aliases",
                        field="clause",
                        value=part.clause,
                        suggestion="Use MATCH with node/edge aliases only until richer pattern planning is implemented.",
                        logical_plan_defer_code="scalar_projection_alias_match",
                    )
                has_known_alias = True
                continue
            if alias in scope_visible_aliases:
                has_known_alias = True
                continue
        if not has_known_alias:
            if self._allow_unknown_match_aliases:
                # Some synthetic compile paths (for example, graph constructors
                # lowered to MATCH + empty RETURN) may not materialize alias
                # entries in SemanticTable. Keep this as an explicit opt-in.
                return
            raise GFQLValidationError(
                ErrorCode.E108,
                "LogicalPlanner skeleton requires MATCH aliases to be present in semantic scope",
                field="clause",
                value=part.clause,
                suggestion="Use query shapes whose MATCH aliases are bound in semantic scope.",
            )

    def _plan_where(
        self,
        *,
        part: BoundQueryPart,
        current: Optional[LogicalPlan],
        vars_by_name: Mapping[str, BoundVariable],
        id_gen: IdGen,
    ) -> LogicalPlan:
        return self._apply_predicates(part=part, current=current, vars_by_name=vars_by_name, id_gen=id_gen)

    def _apply_predicates(
        self,
        *,
        part: BoundQueryPart,
        current: Optional[LogicalPlan],
        vars_by_name: Mapping[str, BoundVariable],
        id_gen: IdGen,
    ) -> LogicalPlan:
        aliases = (
            self._match_aliases_for_part(part, vars_by_name=vars_by_name)
            if part.clause.upper() in {"MATCH", "OPTIONAL MATCH"}
            else self._aliases_for_part(part)
        )
        node = (
            current
            if current is not None
            else NodeScan(op_id=id_gen.next(), label="", output_schema=RowSchema(columns={}))
        )
        for predicate in part.predicates:
            node = Filter(
                op_id=id_gen.next(),
                input=node,
                predicate=predicate,
                output_schema=self._schema_for_aliases(alias_names=aliases, vars_by_name=vars_by_name),
            )
        return node

    def _plan_projection(
        self,
        *,
        part: BoundQueryPart,
        current: Optional[LogicalPlan],
        vars_by_name: Mapping[str, BoundVariable],
        id_gen: IdGen,
    ) -> LogicalPlan:
        expressions = sorted(part.outputs)
        return Project(
            op_id=id_gen.next(),
            input=current,
            expressions=expressions,
            output_schema=self._projection_output_schema(part=part, current=current, vars_by_name=vars_by_name),
        )

    def _plan_unwind(
        self,
        *,
        part: BoundQueryPart,
        current: Optional[LogicalPlan],
        vars_by_name: Mapping[str, BoundVariable],
        id_gen: IdGen,
    ) -> LogicalPlan:
        unwind_var = self._unwind_alias(part)
        list_expr = part.metadata.get("expression", "")
        if not list_expr and part.predicates:
            list_expr = part.predicates[0].expression
        if not list_expr:
            raise GFQLValidationError(
                ErrorCode.E108,
                "LogicalPlanner skeleton requires UNWIND list expression metadata",
                field="clause",
                value=part.clause,
                suggestion="Use binder-emitted UNWIND parts with expression metadata or list predicate expression.",
            )
        return Unwind(
            op_id=id_gen.next(),
            input=current,
            list_expr=list_expr,
            variable=unwind_var,
            output_schema=self._schema_for_aliases(alias_names=part.outputs, vars_by_name=vars_by_name),
        )

    @staticmethod
    def _unwind_alias(part: BoundQueryPart) -> str:
        new_aliases = sorted(part.outputs - part.inputs)
        if len(new_aliases) != 1:
            raise GFQLValidationError(
                ErrorCode.E108,
                "LogicalPlanner skeleton requires UNWIND to introduce exactly one output alias",
                field="clause",
                value=part.clause,
                suggestion="Use UNWIND with one new alias (e.g., UNWIND expr AS x) until richer planning is implemented.",
            )
        return new_aliases[0]

    @staticmethod
    def _schema_for_aliases(*, alias_names: Iterable[str], vars_by_name: Mapping[str, BoundVariable]) -> RowSchema:
        return RowSchema(
            columns={
                alias: var.logical_type
                for alias in sorted(alias_names)
                for var in [vars_by_name.get(alias)]
                if var is not None
            }
        )

    @classmethod
    def _projection_output_schema(
        cls,
        *,
        part: BoundQueryPart,
        current: Optional[LogicalPlan],
        vars_by_name: Mapping[str, BoundVariable],
    ) -> RowSchema:
        schema = cls._schema_for_aliases(alias_names=part.outputs, vars_by_name=vars_by_name)
        if current is None:
            return schema
        input_columns = current.output_schema.columns
        if not input_columns:
            return schema
        return RowSchema(
            columns={
                alias: logical_type
                for alias, logical_type in schema.columns.items()
                if alias not in input_columns or type(input_columns[alias]) is type(logical_type)
            }
        )

    @staticmethod
    def _first_node_label(*, var_names: Iterable[str], vars_by_name: Mapping[str, BoundVariable]) -> str:
        for name in sorted(var_names):
            var = vars_by_name.get(name)
            if var is None or var.entity_kind != "node":
                continue
            logical_type = var.logical_type
            labels: FrozenSet[str] = logical_type.labels if isinstance(logical_type, NodeRef) else frozenset()
            if labels:
                return sorted(labels)[0]
        return ""

    @staticmethod
    def _aliases_for_part(part: BoundQueryPart) -> frozenset[str]:
        # Binder guarantees one of these sets captures the active alias scope.
        return part.outputs or part.inputs

    @staticmethod
    def _match_aliases_for_part(
        part: BoundQueryPart,
        *,
        vars_by_name: Mapping[str, BoundVariable],
    ) -> frozenset[str]:
        aliases = part.outputs or part.inputs
        return frozenset(
            alias
            for alias in aliases
            if (
                vars_by_name.get(alias) is None
                or vars_by_name[alias].entity_kind in {"node", "edge"}
            )
        )

    @staticmethod
    def _entity_kind_for_type(logical_type: LogicalType) -> Literal["node", "edge", "scalar"]:
        if isinstance(logical_type, NodeRef):
            return "node"
        if isinstance(logical_type, EdgeRef):
            return "edge"
        return "scalar"

    @classmethod
    def _vars_for_part(
        cls,
        bound_ir: BoundIR,
        *,
        part_index: int,
        fallback: Mapping[str, BoundVariable],
    ) -> Mapping[str, BoundVariable]:
        if part_index >= len(bound_ir.scope_stack):
            return fallback
        frame = bound_ir.scope_stack[part_index]
        if not frame.schema.columns:
            return fallback
        scoped: Dict[str, BoundVariable] = dict(fallback)
        for alias, logical_type in frame.schema.columns.items():
            fallback_var = fallback.get(alias)
            scoped[alias] = BoundVariable(
                name=alias,
                logical_type=logical_type,
                nullable=(
                    logical_type.nullable
                    if isinstance(logical_type, ScalarType)
                    else (fallback_var.nullable if fallback_var is not None else False)
                ),
                null_extended_from=fallback_var.null_extended_from if fallback_var is not None else frozenset(),
                entity_kind=cls._entity_kind_for_type(logical_type),
                scope_id=fallback_var.scope_id if fallback_var is not None else part_index + 1,
            )
        return scoped

    @staticmethod
    def _match_part_has_path_alias(*, part: BoundQueryPart, scope_schema: RowSchema) -> bool:
        aliases = part.inputs
        return any(isinstance(scope_schema.columns.get(alias), PathType) for alias in aliases)
