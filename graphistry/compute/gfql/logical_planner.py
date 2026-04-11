"""Initial M2 logical planner skeleton.

This module provides a minimal, pure planning contract from BoundIR to
LogicalPlan while assigning stable operator IDs.
"""
from __future__ import annotations

from typing import FrozenSet, Iterable, Mapping, Optional

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundQueryPart, BoundVariable
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import Filter, LogicalPlan, NodeScan, Project, RowSchema, Unwind
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
                current = self._plan_match(part=part, vars_by_name=vars_by_name, id_gen=id_gen)
                seen_match = True
                continue
            if clause == "WHERE":
                current = self._plan_where(part=part, current=current, vars_by_name=vars_by_name, id_gen=id_gen)
                continue
            if clause in {"WITH", "RETURN"}:
                current = self._plan_projection(part=part, current=current, vars_by_name=vars_by_name, id_gen=id_gen)
                continue
            if clause == "UNWIND":
                current = self._plan_unwind(part=part, current=current, vars_by_name=vars_by_name, id_gen=id_gen)
                continue

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
        return NodeScan(
            op_id=id_gen.next(),
            label=self._first_node_label(var_names=part.outputs or part.inputs, vars_by_name=vars_by_name),
            output_schema=self._schema_for_aliases(alias_names=part.outputs or part.inputs, vars_by_name=vars_by_name),
        )

    def _plan_where(
        self,
        *,
        part: BoundQueryPart,
        current: Optional[LogicalPlan],
        vars_by_name: Mapping[str, BoundVariable],
        id_gen: IdGen,
    ) -> LogicalPlan:
        node = current if current is not None else NodeScan(op_id=id_gen.next(), label="", output_schema=RowSchema(columns={}))
        for predicate in part.predicates:
            node = Filter(
                op_id=id_gen.next(),
                input=node,
                predicate=predicate,
                output_schema=self._schema_for_aliases(alias_names=part.outputs or part.inputs, vars_by_name=vars_by_name),
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

    def _plan_unwind(
        self,
        *,
        part: BoundQueryPart,
        current: Optional[LogicalPlan],
        vars_by_name: Mapping[str, BoundVariable],
        id_gen: IdGen,
    ) -> LogicalPlan:
        unwind_var = sorted(part.outputs - part.inputs)[0] if part.outputs - part.inputs else ""
        list_expr = part.metadata.get("expression", "")
        if not list_expr and part.predicates:
            list_expr = part.predicates[0].expression
        return Unwind(
            op_id=id_gen.next(),
            input=current,
            list_expr=list_expr,
            variable=unwind_var,
            output_schema=self._schema_for_aliases(alias_names=part.outputs, vars_by_name=vars_by_name),
        )

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
