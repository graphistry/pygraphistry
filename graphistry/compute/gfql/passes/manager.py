"""Logical pass manager and pass-result contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Sequence, Tuple

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.ir.compilation import CompilerError, PlanContext
from graphistry.compute.gfql.ir.logical_plan import LogicalPlan
from graphistry.compute.gfql.ir.verifier import verify


@dataclass(frozen=True)
class PassResult:
    """Result payload returned by a logical pass."""

    plan: LogicalPlan
    metadata: Dict[str, object] = field(default_factory=dict)


class LogicalPass(Protocol):
    """Protocol for deterministic logical-plan passes."""

    name: str

    def run(self, plan: LogicalPlan, ctx: PlanContext) -> PassResult:
        """Transform a logical plan and return a PassResult."""


DEFAULT_LOGICAL_PASSES: Tuple[LogicalPass, ...] = ()


class PassManager:
    """Sequential pass runner with verifier guards after each pass."""

    def __init__(self, passes: Sequence[LogicalPass] = DEFAULT_LOGICAL_PASSES) -> None:
        self._passes: Tuple[LogicalPass, ...] = tuple(passes)

    def run(self, logical_plan: LogicalPlan, ctx: PlanContext) -> PassResult:
        current = logical_plan
        merged_metadata: Dict[str, object] = {}

        for logical_pass in self._passes:
            result = logical_pass.run(current, ctx)
            current = result.plan
            if result.metadata:
                merged_metadata[logical_pass.name] = dict(result.metadata)
            diagnostics = verify(current)
            if diagnostics:
                raise _verification_error(logical_pass.name, diagnostics)

        return PassResult(plan=current, metadata=merged_metadata)


def _verification_error(pass_name: str, diagnostics: Sequence[CompilerError]) -> GFQLValidationError:
    message = "; ".join(error.message for error in diagnostics[:3])
    if len(diagnostics) > 3:
        message = f"{message}; ... (+{len(diagnostics) - 3} more)"
    return GFQLValidationError(
        ErrorCode.E108,
        "Logical pass produced an invalid plan rejected by verifier",
        field="pass",
        value=pass_name,
        suggestion=message or "Ensure pass output satisfies LogicalPlan verifier invariants.",
        language="cypher",
    )
