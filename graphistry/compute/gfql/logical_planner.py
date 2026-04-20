"""Initial M2 logical planner skeleton.

This module provides a minimal, pure planning contract from BoundIR to
LogicalPlan while assigning stable operator IDs.
"""
from __future__ import annotations

from typing import FrozenSet, Iterable, Mapping, Optional

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundQueryPart, BoundVariable
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import (
    Filter,
    LogicalPlan,
    NodeScan,
    PatternMatch,
    Project,
    RowSchema,
    Unwind,
)
from graphistry.compute.gfql.ir.types import NodeRef


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

    def plan(self, bound_ir: BoundIR, ctx: PlanContext) -> LogicalPlan:
        """Build a minimal logical plan root for supported M2 skeleton shapes."""
        _ = ctx
        id_gen = IdGen()
        current: Optional[LogicalPlan] = None
        vars_by_name = bound_ir.semantic_table.variables
        seen_match = False

        for part in bound_ir.query_parts:
            clause = part.clause.upper()
            if clause == "OPTIONAL MATCH":
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "LogicalPlanner skeleton does not yet support OPTIONAL MATCH planning",
                    field="clause",
                    value=part.clause,
                    suggestion="Use non-optional MATCH shapes until optional planning is implemented.",
                )
            if clause == "MATCH":
                if seen_match:
                    raise GFQLValidationError(
                        ErrorCode.E108,
                        "LogicalPlanner skeleton does not yet support multiple MATCH stages",
                        field="clause",
                        value=part.clause,
                        suggestion="Use a single MATCH stage until chained pattern planning is implemented.",
                    )
                self._reject_unsupported_match_shape(part=part, vars_by_name=vars_by_name)
                current = self._plan_match(part=part, vars_by_name=vars_by_name, id_gen=id_gen)
                current = self._apply_predicates(part=part, current=current, vars_by_name=vars_by_name, id_gen=id_gen)
                seen_match = True
                continue
            if clause == "WHERE":
                current = self._plan_where(part=part, current=current, vars_by_name=vars_by_name, id_gen=id_gen)
                continue
            if clause in {"WITH", "RETURN"}:
                self._reject_distinct_projection(part=part)
                current = self._plan_projection(part=part, current=current, vars_by_name=vars_by_name, id_gen=id_gen)
                current = self._apply_predicates(part=part, current=current, vars_by_name=vars_by_name, id_gen=id_gen)
                continue
            if clause == "UNWIND":
                current = self._plan_unwind(part=part, current=current, vars_by_name=vars_by_name, id_gen=id_gen)
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
    ) -> LogicalPlan:
        aliases = sorted(self._aliases_for_part(part))
        schema = self._schema_for_aliases(alias_names=aliases, vars_by_name=vars_by_name)
        if len(aliases) == 1:
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
            optional=False,
            arm_id=None,
            output_schema=schema,
        )

    def _reject_unsupported_match_shape(
        self,
        *,
        part: BoundQueryPart,
        vars_by_name: Mapping[str, BoundVariable],
    ) -> None:
        alias_names = part.outputs or part.inputs
        if not alias_names:
            raise GFQLValidationError(
                ErrorCode.E108,
                "LogicalPlanner skeleton requires at least one MATCH alias",
                field="clause",
                value=part.clause,
                suggestion="Use MATCH with at least one alias in scope.",
            )
        has_known_alias = False
        for alias in alias_names:
            variable = vars_by_name.get(alias)
            if variable is None:
                continue
            has_known_alias = True
            if variable.entity_kind not in {"node", "edge"}:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "LogicalPlanner skeleton only supports MATCH outputs bound to node/edge aliases",
                    field="clause",
                    value=part.clause,
                    suggestion="Use MATCH with node/edge aliases only until richer pattern planning is implemented.",
                )
        if not has_known_alias:
            # Some synthetic compile paths (for example, graph constructors
            # lowered to MATCH + empty RETURN) may not materialize alias
            # entries in SemanticTable. Allow these through as skeleton plans.
            return

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
        aliases = self._aliases_for_part(part)
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
            output_schema=self._schema_for_aliases(alias_names=part.outputs, vars_by_name=vars_by_name),
        )

    def _reject_distinct_projection(self, *, part: BoundQueryPart) -> None:
        if part.metadata.get("distinct", False):
            raise GFQLValidationError(
                ErrorCode.E108,
                "LogicalPlanner skeleton does not yet support DISTINCT projections",
                field="clause",
                value=part.clause,
                suggestion="Use non-DISTINCT WITH/RETURN shapes until DISTINCT planning is implemented.",
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
