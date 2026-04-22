from dataclasses import replace

import pytest

from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import Filter, NodeScan
from graphistry.compute.gfql.ir.types import BoundPredicate
from graphistry.compute.gfql.passes import PassManager, PassResult


class _BumpPass:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self._events = events

    def run(self, plan, ctx):  # noqa: ANN001, ANN201
        _ = ctx
        self._events.append(self.name)
        return PassResult(
            plan=replace(plan, op_id=plan.op_id + 1),
            metadata={"op_id": plan.op_id + 1},
        )


class _InvalidPass:
    name = "invalid-pass"

    def run(self, plan, ctx):  # noqa: ANN001, ANN201
        _ = (plan, ctx)
        return PassResult(
            plan=Filter(
                op_id=5,
                input=NodeScan(op_id=1),
                predicate=BoundPredicate(expression=""),
            )
        )


def test_pass_manager_runs_in_order_and_returns_final_plan():
    events: list[str] = []
    manager = PassManager((_BumpPass("p1", events), _BumpPass("p2", events)))
    start = NodeScan(op_id=1)

    result = manager.run(start, PlanContext())

    assert events == ["p1", "p2"]
    assert result.plan.op_id == 3
    assert result.metadata["p1"]["op_id"] == 2
    assert result.metadata["p2"]["op_id"] == 3


def test_pass_manager_raises_when_pass_output_fails_verifier():
    manager = PassManager((_InvalidPass(),))

    with pytest.raises(GFQLValidationError, match="invalid plan rejected by verifier"):
        manager.run(NodeScan(op_id=1), PlanContext())
