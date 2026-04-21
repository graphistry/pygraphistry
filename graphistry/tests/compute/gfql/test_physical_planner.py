from __future__ import annotations

import pytest

from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import (
    Filter,
    Join,
    Limit,
    OrderBy,
    PatternMatch,
    ProcedureCall,
    Project,
    RowSchema,
)
from graphistry.compute.gfql.ir.types import ScalarType
from graphistry.compute.gfql.physical_planner import (
    PhysicalPlanner,
    RowPipelineExecutorWrapper,
    SamePathExecutorWrapper,
    WavefrontExecutorWrapper,
)


def _schema(*columns: str) -> RowSchema:
    return RowSchema(columns={column: ScalarType(kind="string", nullable=True) for column in columns})


def test_physical_planner_classifies_same_path_wrapper_for_linear_match_shape() -> None:
    logical_plan = Project(
        op_id=3,
        input=PatternMatch(
            op_id=2,
            pattern={"aliases": ("a", "b")},
            output_schema=_schema("a", "b"),
        ),
        expressions=["a"],
        output_schema=_schema("a"),
    )

    physical_plan = PhysicalPlanner().plan(logical_plan, PlanContext())

    assert physical_plan.route == "same_path"
    assert physical_plan.logical_op_ids == (3, 2)
    assert len(physical_plan.operators) == 1
    assert isinstance(physical_plan.operators[0], SamePathExecutorWrapper)
    assert physical_plan.operators[0].executor == "execute_same_path_chain"


def test_physical_planner_classifies_wavefront_wrapper_for_join_shape() -> None:
    logical_plan = Project(
        op_id=4,
        input=Join(
            op_id=3,
            left=PatternMatch(op_id=1, pattern={"aliases": ("a", "b")}, output_schema=_schema("a", "b")),
            right=PatternMatch(op_id=2, pattern={"aliases": ("a", "c")}, output_schema=_schema("a", "c")),
            condition={"join_aliases": ("a",)},
            join_type="inner",
            output_schema=_schema("a", "b", "c"),
        ),
        expressions=["a", "b", "c"],
        output_schema=_schema("a", "b", "c"),
    )

    physical_plan = PhysicalPlanner().plan(logical_plan, PlanContext())

    assert physical_plan.route == "wavefront"
    assert physical_plan.logical_op_ids == (4, 3, 1, 2)
    assert len(physical_plan.operators) == 1
    assert isinstance(physical_plan.operators[0], WavefrontExecutorWrapper)
    assert physical_plan.operators[0].executor == "_apply_connected_match_join"
    assert physical_plan.operators[0].join_types == ("inner",)


def test_physical_planner_classifies_row_pipeline_wrapper_for_row_only_shape() -> None:
    logical_plan = Limit(
        op_id=4,
        input=OrderBy(
            op_id=3,
            input=Project(
                op_id=2,
                input=Filter(op_id=1, input=None, output_schema=_schema("x")),
                expressions=["x"],
                output_schema=_schema("x"),
            ),
            sort_keys=["x"],
            output_schema=_schema("x"),
        ),
        count=10,
        output_schema=_schema("x"),
    )

    physical_plan = PhysicalPlanner().plan(logical_plan, PlanContext())

    assert physical_plan.route == "row_pipeline"
    assert len(physical_plan.operators) == 1
    assert isinstance(physical_plan.operators[0], RowPipelineExecutorWrapper)
    assert physical_plan.operators[0].executor == "execute_row_pipeline_call"
    assert set(physical_plan.operators[0].row_stage_ops) == {"filter", "limit", "orderby", "project"}


def test_physical_planner_rejects_unsupported_operator_shapes_with_explicit_error() -> None:
    logical_plan = ProcedureCall(
        op_id=1,
        procedure="graphistry.degree",
        output_schema=_schema("node_id"),
    )

    with pytest.raises(GFQLValidationError, match="does not yet support"):
        PhysicalPlanner().plan(logical_plan, PlanContext())
