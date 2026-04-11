"""Structural verifier for LogicalPlan (M2-PR3 / issue #1127).

verify(plan) walks the operator tree and checks five invariants:
  1. op_id uniqueness  — non-zero op_ids are distinct across the whole tree
  2. Dangling refs     — child slots (input/left/right/subquery) hold LogicalPlan | None
  3. Predicate scope   — BoundPredicate.expression must be non-empty on all predicate-bearing ops
  4. Schema validity   — RowSchema column values must be LogicalType instances
  5. Optional-arm      — PatternMatch(optional=True) must carry a non-empty arm_id
"""
from __future__ import annotations

from typing import Iterator

from graphistry.compute.gfql.ir.compilation import CompilerError
from graphistry.compute.gfql.ir.logical_plan import (
    Filter,
    IndexScan,
    LogicalPlan,
    PatternMatch,
    RowSchema,
)
from graphistry.compute.gfql.ir.types import BoundPredicate, EdgeRef, ListType, NodeRef, PathType, ScalarType


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Child-slot attribute names that may hold a child LogicalPlan (or None).
_CHILD_SLOTS = ("input", "left", "right", "subquery")

# All concrete LogicalType subtypes for isinstance checks (Union alias can't be used).
_LOGICAL_TYPES = (NodeRef, EdgeRef, ScalarType, PathType, ListType)

# Sentinel for getattr default when the attribute doesn't exist on an operator.
_MISSING = object()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _children(op: LogicalPlan) -> list[object]:
    """Return all child-slot values for *op* (may include None)."""
    kids: list[object] = []
    for attr in _CHILD_SLOTS:
        val = getattr(op, attr, _MISSING)
        if val is not _MISSING:
            kids.append(val)
    return kids


def _walk(
    op: LogicalPlan,
    visited: set,
    path: set,
    cycle_pairs: list,
) -> Iterator[LogicalPlan]:
    """Pre-order traversal of all LogicalPlan nodes reachable from *op*.

    *visited* prevents re-visiting shared subtrees (DAG diamonds are fine).
    *path* is the current ancestor set; a back-edge into *path* is a true cycle.
    *cycle_pairs* accumulates (parent_type, child_type) for every cycle edge
    found; callers convert these to CompilerErrors.
    """
    node_id = id(op)
    if node_id in visited:
        return
    visited.add(node_id)
    path.add(node_id)
    yield op
    for child in _children(op):
        if isinstance(child, LogicalPlan):
            if id(child) in path:
                cycle_pairs.append((type(op).__name__, type(child).__name__))
            else:
                yield from _walk(child, visited, path, cycle_pairs)
    path.discard(node_id)


def _check_predicate(op: LogicalPlan, pred: BoundPredicate, label: str) -> list[CompilerError]:
    """Return an error if *pred* has an empty expression string."""
    if not pred.expression:
        return [CompilerError(
            message=f"{type(op).__name__} op_id={op.op_id}: {label} has empty expression"
        )]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify(plan: LogicalPlan) -> list[CompilerError]:
    """Return all structural invariant violations in *plan*.

    Accumulates every error found rather than short-circuiting.
    """
    errors: list[CompilerError] = []
    seen_ids: set[int] = set()
    visited: set[int] = set()
    path: set[int] = set()
    cycle_pairs: list[tuple[str, str]] = []

    for op in _walk(plan, visited, path, cycle_pairs):
        # ------------------------------------------------------------------
        # Invariant 1: op_id uniqueness (0 = unassigned sentinel, exempt)
        # op_id=0 is the dataclass default meaning "not yet assigned by the
        # planner".  Hand-built test fixtures routinely use it.  Multiple
        # zeros in one tree are intentional and are not a uniqueness violation.
        # Planner-emitted plans should always carry non-zero IDs; that
        # contract is enforced by the planner, not here.
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
        for attr in _CHILD_SLOTS:
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
        # Invariant 3: Predicate scope
        # Applies to all operators that carry BoundPredicate fields.
        # ------------------------------------------------------------------
        if isinstance(op, PatternMatch):
            for i, pred in enumerate(op.predicates):
                errors.extend(_check_predicate(op, pred, f"predicate[{i}]"))
        elif isinstance(op, Filter):
            errors.extend(_check_predicate(op, op.predicate, "predicate"))
        elif isinstance(op, IndexScan):
            errors.extend(_check_predicate(op, op.predicate, "predicate"))
            for i, pred in enumerate(op.residual_predicates):
                errors.extend(_check_predicate(op, pred, f"residual_predicates[{i}]"))

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

    # ------------------------------------------------------------------
    # Structural cycle diagnostic (reported after traversal completes)
    # ------------------------------------------------------------------
    for parent_type, child_type in cycle_pairs:
        errors.append(CompilerError(
            message=f"Cycle detected: {parent_type} has a child edge back to already-visited {child_type}"
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
