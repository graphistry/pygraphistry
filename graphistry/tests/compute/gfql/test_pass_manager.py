from __future__ import annotations

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


class _CountingPass:
    """Pass that changes the plan for the first N calls, then stops."""

    def __init__(self, name: str, events: list[str], change_times: int) -> None:
        self.name = name
        self._events = events
        self._remaining = change_times

    def run(self, plan, ctx):  # noqa: ANN001, ANN201
        _ = ctx
        self._events.append(self.name)
        if self._remaining > 0:
            self._remaining -= 1
            return PassResult(
                plan=replace(plan, op_id=plan.op_id + 1),
                changed=True,
            )
        return PassResult(plan=plan, changed=False)


class _NoOpPass:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self._events = events

    def run(self, plan, ctx):  # noqa: ANN001, ANN201
        _ = ctx
        self._events.append(self.name)
        return PassResult(plan=plan, changed=False)


# ---------------------------------------------------------------------------
# Original tier-1 tests (backward compat)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tier 1 single-execution contract
# ---------------------------------------------------------------------------

def test_tier1_each_pass_runs_exactly_once():
    events: list[str] = []
    manager = PassManager(
        tier1_passes=(_BumpPass("a", events), _BumpPass("b", events), _BumpPass("c", events))
    )
    manager.run(NodeScan(op_id=0), PlanContext())
    assert events == ["a", "b", "c"]


def test_tier1_no_passes_is_identity():
    plan = NodeScan(op_id=7)
    result = PassManager(()).run(plan, PlanContext())
    assert result.plan is plan


# ---------------------------------------------------------------------------
# Tier 2 fixed-point convergence
# ---------------------------------------------------------------------------

def test_tier2_runs_until_no_changes():
    events: list[str] = []
    # changes for 2 iterations then stops
    p = _CountingPass("r1", events, change_times=2)
    manager = PassManager(tier1_passes=(), tier2_passes=(p,))
    result = manager.run(NodeScan(op_id=0), PlanContext())

    # 3 sweeps: iterations 0+1 produce changes, iteration 2 produces none → stops
    assert events == ["r1", "r1", "r1"]
    assert result.plan.op_id == 2


def test_tier2_no_change_on_first_sweep_exits_immediately():
    events: list[str] = []
    p = _NoOpPass("noop", events)
    manager = PassManager(tier1_passes=(), tier2_passes=(p,))
    manager.run(NodeScan(op_id=0), PlanContext())
    # Only one sweep — no changes → stops immediately
    assert events == ["noop"]


def test_tier2_multiple_passes_all_run_each_iteration():
    events: list[str] = []
    p1 = _CountingPass("x", events, change_times=1)
    p2 = _NoOpPass("y", events)
    manager = PassManager(tier1_passes=(), tier2_passes=(p1, p2))
    manager.run(NodeScan(op_id=0), PlanContext())
    # Iteration 0: x changes, y noop → any_changed=True
    # Iteration 1: x noop, y noop → any_changed=False → stop
    assert events == ["x", "y", "x", "y"]


def test_tier2_raises_on_verifier_failure():
    manager = PassManager(tier1_passes=(), tier2_passes=(_InvalidPass(),))
    with pytest.raises(GFQLValidationError, match="invalid plan rejected by verifier"):
        manager.run(NodeScan(op_id=1), PlanContext())


def test_tier2_raises_on_max_iterations_exceeded():
    # Pass always reports changed — loop never converges
    events: list[str] = []
    always_changing = _BumpPass("inf", events)
    manager = PassManager(tier1_passes=(), tier2_passes=(always_changing,), max_iterations=5)
    with pytest.raises(GFQLValidationError, match="did not converge"):
        manager.run(NodeScan(op_id=0), PlanContext())
    assert len(events) == 5


# ---------------------------------------------------------------------------
# Tier 1 + Tier 2 combined
# ---------------------------------------------------------------------------

def test_tier1_runs_before_tier2():
    events: list[str] = []
    t1 = _BumpPass("t1", events)
    t2 = _NoOpPass("t2", events)
    manager = PassManager(tier1_passes=(t1,), tier2_passes=(t2,))
    manager.run(NodeScan(op_id=0), PlanContext())
    assert events[0] == "t1"
    assert events[1] == "t2"


def test_backward_compat_single_positional_arg():
    # Old call site: PassManager(passes) — still works as tier1
    events: list[str] = []
    manager = PassManager((_BumpPass("legacy", events),))
    result = manager.run(NodeScan(op_id=0), PlanContext())
    assert events == ["legacy"]
    assert result.plan.op_id == 1


# ---------------------------------------------------------------------------
# Tier 2 metadata accumulation
# ---------------------------------------------------------------------------

class _MetadataPass:
    """Pass that emits integer metadata and changes the plan once."""

    def __init__(self, name: str, change_times: int) -> None:
        self.name = name
        self._remaining = change_times

    def run(self, plan, ctx):  # noqa: ANN001, ANN201
        _ = ctx
        if self._remaining > 0:
            self._remaining -= 1
            return PassResult(
                plan=replace(plan, op_id=plan.op_id + 1),
                metadata={"count": 3},
                changed=True,
            )
        return PassResult(plan=plan, metadata={"count": 0}, changed=False)


def test_tier2_metadata_accumulates_integer_fields():
    # Pass changes plan twice, emitting count=3 each time, then converges with count=0.
    # Accumulated metadata should total 6 (3+3+0).
    p = _MetadataPass("acc", change_times=2)
    manager = PassManager(tier1_passes=(), tier2_passes=(p,))
    result = manager.run(NodeScan(op_id=0), PlanContext())
    assert result.metadata["acc"]["count"] == 6
