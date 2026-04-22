"""Logical pass manager and pass-result contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, Sequence, Tuple

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.ir.compilation import CompilerError, PlanContext
from graphistry.compute.gfql.ir.logical_plan import LogicalPlan
from graphistry.compute.gfql.ir.verifier import verify


@dataclass(frozen=True)
class PassResult:
    """Result payload returned by a logical pass."""

    plan: LogicalPlan
    metadata: Dict[str, object] = field(default_factory=dict)
    # Tier 2 convergence signal: set False when the pass made no changes.
    # Defaults to True so passes that predate two-tier semantics are conservative.
    changed: bool = True


class LogicalPass(Protocol):
    """Protocol for deterministic logical-plan passes."""

    name: str

    def run(self, plan: LogicalPlan, ctx: PlanContext) -> PassResult:
        """Transform a logical plan and return a PassResult."""
        ...


DEFAULT_LOGICAL_PASSES: Tuple[LogicalPass, ...] = ()
DEFAULT_TIER2_PASSES: Tuple[LogicalPass, ...] = ()

_DEFAULT_MAX_ITERATIONS = 100


class PassManager:
    """Two-tier pass runner with verifier guards after each pass.

    Tier 1 (*tier1_passes*): each pass runs exactly once in configured order.
    Tier 2 (*tier2_passes*): all passes run repeatedly in a fixed-point loop
    until a full sweep produces no changes (every pass returns
    ``PassResult.changed=False``).  Bounded by *max_iterations* to guarantee
    termination.
    """

    def __init__(
        self,
        tier1_passes: Sequence[LogicalPass] = DEFAULT_LOGICAL_PASSES,
        tier2_passes: Sequence[LogicalPass] = DEFAULT_TIER2_PASSES,
        *,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self._tier1: Tuple[LogicalPass, ...] = tuple(tier1_passes)
        self._tier2: Tuple[LogicalPass, ...] = tuple(tier2_passes)
        self._max_iterations = max_iterations

    def run(self, logical_plan: LogicalPlan, ctx: PlanContext) -> PassResult:
        current = logical_plan
        merged_metadata: Dict[str, object] = {}

        # --- Tier 1: structural passes, each runs exactly once ---
        for logical_pass in self._tier1:
            result = logical_pass.run(current, ctx)
            current = result.plan
            if result.metadata:
                merged_metadata[logical_pass.name] = dict(result.metadata)
            diagnostics = verify(current)
            if diagnostics:
                raise _verification_error(logical_pass.name, diagnostics)

        # --- Tier 2: rewrite rules, fixed-point loop ---
        if self._tier2:
            for _ in range(self._max_iterations):
                any_changed = False
                for logical_pass in self._tier2:
                    result = logical_pass.run(current, ctx)
                    current = result.plan
                    if result.metadata:
                        merged_metadata[logical_pass.name] = dict(result.metadata)
                    if result.changed:
                        any_changed = True
                    diagnostics = verify(current)
                    if diagnostics:
                        raise _verification_error(logical_pass.name, diagnostics)
                if not any_changed:
                    break
            else:
                raise _convergence_error(self._max_iterations)

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


def _convergence_error(max_iterations: int) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        f"Tier 2 pass loop did not converge after {max_iterations} iterations",
        field="pass",
        value="tier2",
        suggestion="Check that Tier 2 passes converge and set PassResult.changed=False when unchanged.",
        language="cypher",
    )
