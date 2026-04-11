"""Structural verifier for LogicalPlan (M2-PR3 / issue #1127).

verify(plan) walks the operator tree and checks five invariants:
  1. op_id uniqueness  — non-zero op_ids are distinct across the whole tree
  2. Dangling refs     — child slots (input/left/right/subquery) hold LogicalPlan | None
  3. Predicate scope   — PatternMatch predicates must have non-empty expression strings
  4. Schema validity   — RowSchema column values must be LogicalType instances
  5. Optional-arm      — PatternMatch(optional=True) must carry a non-empty arm_id
"""
from __future__ import annotations

from typing import Iterator

from graphistry.compute.gfql.ir.compilation import CompilerError
from graphistry.compute.gfql.ir.logical_plan import LogicalPlan, PatternMatch, RowSchema
from graphistry.compute.gfql.ir.types import EdgeRef, ListType, LogicalType, NodeRef, PathType, ScalarType


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LOGICAL_TYPES = (NodeRef, EdgeRef, ScalarType, PathType, ListType)


def _children(op: LogicalPlan) -> list[object]:
    """Return all child-slot values for ``op`` (may include None)."""
    kids: list[object] = []
    for attr in ("input", "left", "right", "subquery"):
        val = getattr(op, attr, _MISSING)
        if val is not _MISSING:
            kids.append(val)
    return kids


_MISSING = object()


def _walk(op: LogicalPlan) -> Iterator[LogicalPlan]:
    """Pre-order traversal of all LogicalPlan nodes reachable from *op*."""
    yield op
    for child in _children(op):
        if isinstance(child, LogicalPlan):
            yield from _walk(child)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify(plan: LogicalPlan) -> list[CompilerError]:
    """Return all structural invariant violations in *plan*.

    Accumulates every error found rather than short-circuiting.
    """
    errors: list[CompilerError] = []
    seen_ids: set[int] = set()

    for op in _walk(plan):
        # ------------------------------------------------------------------
        # Invariant 1: op_id uniqueness (0 = unassigned, exempt)
        # ------------------------------------------------------------------
        if op.op_id != 0:
            if op.op_id in seen_ids:
                errors.append(CompilerError(
                    message=f"Duplicate op_id {op.op_id} on {type(op).__name__}"
                ))
            else:
                seen_ids.add(op.op_id)

        # ------------------------------------------------------------------
        # Invariant 2: Dangling references
        # ------------------------------------------------------------------
        for attr in ("input", "left", "right", "subquery"):
            val = getattr(op, attr, _MISSING)
            if val is _MISSING:
                continue
            if val is not None and not isinstance(val, LogicalPlan):
                errors.append(CompilerError(
                    message=(
                        f"Dangling reference: {type(op).__name__}.{attr} "
                        f"is {type(val).__name__!r}, expected LogicalPlan or None"
                    )
                ))

        # ------------------------------------------------------------------
        # Invariant 3: Predicate scope (PatternMatch only)
        # ------------------------------------------------------------------
        if isinstance(op, PatternMatch):
            for i, pred in enumerate(op.predicates):
                if not pred.expression:
                    errors.append(CompilerError(
                        message=(
                            f"PatternMatch op_id={op.op_id}: predicate[{i}] "
                            "has empty expression"
                        )
                    ))

        # ------------------------------------------------------------------
        # Invariant 4: Output schema consistency
        # ------------------------------------------------------------------
        errors.extend(_check_schema(op, op.output_schema))

        # ------------------------------------------------------------------
        # Invariant 5: Optional-arm nullability contract
        # ------------------------------------------------------------------
        if isinstance(op, PatternMatch) and op.optional:
            if not op.arm_id:
                errors.append(CompilerError(
                    message=(
                        f"PatternMatch op_id={op.op_id}: optional=True "
                        "but arm_id is missing or empty"
                    )
                ))

    return errors


# ---------------------------------------------------------------------------
# Schema checker (invariant 4)
# ---------------------------------------------------------------------------

def _check_schema(op: LogicalPlan, schema: RowSchema) -> list[CompilerError]:
    errors: list[CompilerError] = []
    for col, typ in schema.columns.items():
        if not isinstance(typ, _LOGICAL_TYPES):
            errors.append(CompilerError(
                message=(
                    f"{type(op).__name__} op_id={op.op_id}: "
                    f"output_schema column {col!r} has invalid type "
                    f"{type(typ).__name__!r}, expected LogicalType"
                )
            ))
    return errors
