"""M3 physical planner skeleton mapping LogicalPlan trees to executor wrappers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Literal, Mapping, Sequence, Tuple, Union

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.ir.compilation import PhysicalPlan, PlanContext
from graphistry.compute.gfql.ir.logical_plan import (
    Aggregate,
    AntiSemiApply,
    Apply,
    Distinct,
    EdgeScan,
    Filter,
    GraphToRows,
    IndexScan,
    Join,
    Limit,
    LogicalPlan,
    NodeScan,
    OrderBy,
    PathProjection,
    PatternMatch,
    Project,
    RowsToGraph,
    SemiApply,
    Skip,
    Union as LogicalUnion,
    Unwind,
)

PhysicalRoute = Literal["same_path", "wavefront", "row_pipeline"]


@dataclass(frozen=True)
class SamePathExecutorWrapper:
    """Descriptor for dispatching to the same-path executor."""

    route: Literal["same_path"] = "same_path"
    executor: Literal["execute_same_path_chain"] = "execute_same_path_chain"
    logical_op_ids: Tuple[int, ...] = field(default_factory=tuple)
    logical_operator_types: Tuple[str, ...] = field(default_factory=tuple)
    include_paths: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class WavefrontExecutorWrapper:
    """Descriptor for dispatching to the connected-match wavefront join runtime."""

    route: Literal["wavefront"] = "wavefront"
    executor: Literal["_apply_connected_match_join"] = "_apply_connected_match_join"
    logical_op_ids: Tuple[int, ...] = field(default_factory=tuple)
    logical_operator_types: Tuple[str, ...] = field(default_factory=tuple)
    join_types: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RowPipelineExecutorWrapper:
    """Descriptor for dispatching to row-pipeline execution helpers."""

    route: Literal["row_pipeline"] = "row_pipeline"
    executor: Literal["execute_row_pipeline_call"] = "execute_row_pipeline_call"
    logical_op_ids: Tuple[int, ...] = field(default_factory=tuple)
    logical_operator_types: Tuple[str, ...] = field(default_factory=tuple)
    row_stage_ops: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, object] = field(default_factory=dict)


PhysicalOperator = Union[
    SamePathExecutorWrapper,
    WavefrontExecutorWrapper,
    RowPipelineExecutorWrapper,
]


_SAME_PATH_LOGICAL_OPS = (NodeScan, EdgeScan, IndexScan, PatternMatch, PathProjection)
_WAVEFRONT_LOGICAL_OPS = (Join, Apply, SemiApply, AntiSemiApply, LogicalUnion)
_ROW_PIPELINE_LOGICAL_OPS = (Filter, Project, Aggregate, Distinct, OrderBy, Limit, Skip, Unwind, GraphToRows, RowsToGraph)
_SUPPORTED_LOGICAL_OPS = _SAME_PATH_LOGICAL_OPS + _WAVEFRONT_LOGICAL_OPS + _ROW_PIPELINE_LOGICAL_OPS


class PhysicalPlanner:
    """Map logical plan trees to stable physical wrapper descriptors."""

    def plan(self, logical_plan: LogicalPlan, ctx: PlanContext) -> PhysicalPlan:
        _ = ctx
        nodes = tuple(self._walk_logical_plan(logical_plan))
        if not nodes:
            raise GFQLValidationError(
                ErrorCode.E108,
                "PhysicalPlanner requires a non-empty logical plan tree",
                field="logical_plan",
                value=None,
                suggestion="Compile a covered logical plan shape before physical planning.",
            )

        route = self._classify_route(nodes)
        logical_op_ids = tuple(node.op_id for node in nodes)
        logical_types = tuple(type(node).__name__ for node in nodes)
        operator = self._build_operator(
            route,
            nodes=nodes,
            logical_op_ids=logical_op_ids,
            logical_types=logical_types,
        )

        return PhysicalPlan(
            route=route,
            operators=(operator,),
            logical_op_ids=logical_op_ids,
            metadata={"logical_operator_types": logical_types},
        )

    def _classify_route(self, nodes: Sequence[LogicalPlan]) -> PhysicalRoute:
        unsupported = sorted({type(node).__name__ for node in nodes if not isinstance(node, _SUPPORTED_LOGICAL_OPS)})
        if unsupported:
            raise GFQLValidationError(
                ErrorCode.E108,
                "PhysicalPlanner skeleton does not yet support one or more logical operators",
                field="logical_plan",
                value=unsupported,
                suggestion="Use covered M3 logical operator shapes (same-path/wavefront/row-pipeline lane).",
            )

        if any(isinstance(node, _WAVEFRONT_LOGICAL_OPS) for node in nodes):
            return "wavefront"
        if any(isinstance(node, _SAME_PATH_LOGICAL_OPS) for node in nodes):
            return "same_path"
        if all(isinstance(node, _ROW_PIPELINE_LOGICAL_OPS) for node in nodes):
            return "row_pipeline"

        raise GFQLValidationError(
            ErrorCode.E108,
            "PhysicalPlanner could not classify this logical plan into a covered executor route",
            field="logical_plan",
            value=[type(node).__name__ for node in nodes],
            suggestion="Use a covered route shape or extend planner classification rules.",
        )

    def _build_operator(
        self,
        route: PhysicalRoute,
        *,
        nodes: Sequence[LogicalPlan],
        logical_op_ids: Tuple[int, ...],
        logical_types: Tuple[str, ...],
    ) -> PhysicalOperator:
        if route == "same_path":
            return SamePathExecutorWrapper(
                logical_op_ids=logical_op_ids,
                logical_operator_types=logical_types,
            )

        if route == "wavefront":
            join_types = tuple(sorted({node.join_type for node in nodes if isinstance(node, Join)}))
            return WavefrontExecutorWrapper(
                logical_op_ids=logical_op_ids,
                logical_operator_types=logical_types,
                join_types=join_types,
            )

        row_stage_ops = tuple(
            sorted({type(node).__name__.lower() for node in nodes if isinstance(node, _ROW_PIPELINE_LOGICAL_OPS)})
        )
        return RowPipelineExecutorWrapper(
            logical_op_ids=logical_op_ids,
            logical_operator_types=logical_types,
            row_stage_ops=row_stage_ops,
        )

    def _walk_logical_plan(self, root: LogicalPlan) -> Iterator[LogicalPlan]:
        stack: list[LogicalPlan] = [root]
        seen: set[int] = set()

        while stack:
            node = stack.pop()
            marker = id(node)
            if marker in seen:
                continue
            seen.add(marker)
            yield node

            for child in reversed(tuple(self._children(node))):
                stack.append(child)

    @staticmethod
    def _children(node: LogicalPlan) -> Iterable[LogicalPlan]:
        for attr in ("input", "left", "right", "subquery"):
            child = getattr(node, attr, None)
            if isinstance(child, LogicalPlan):
                yield child


__all__ = [
    "PhysicalOperator",
    "PhysicalPlanner",
    "RowPipelineExecutorWrapper",
    "SamePathExecutorWrapper",
    "WavefrontExecutorWrapper",
]
