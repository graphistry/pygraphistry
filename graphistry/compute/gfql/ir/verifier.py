"""Structural verifier for LogicalPlan (M2-PR3 / issue #1127).

verify(plan) walks the operator tree and checks five invariants:
  1. op_id uniqueness  — non-zero op_ids are distinct across the whole tree
  2. Dangling refs     — child slots (input/left/right/subquery) hold LogicalPlan | None
  3. Predicate scope   — BoundPredicate.expression must be non-empty on all predicate-bearing ops
  4. Schema validity   — RowSchema column values must be LogicalType instances (ListType.element_type validated recursively)
  5. Optional-arm      — PatternMatch(optional=True) must carry a non-empty arm_id
"""
from __future__ import annotations

from typing import Iterator, List, Set, Tuple

from graphistry.compute.gfql.ir.compilation import CompilerError
from graphistry.compute.gfql.ir.logical_plan import (
    CHILD_SLOTS,
    Filter,
    IndexScan,
    LogicalPlan,
    PatternMatch,
    RowSchema,
    iter_children,
)
from graphistry.compute.gfql.ir.types import BoundPredicate, EdgeRef, ListType, NodeRef, PathType, ScalarType


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# All concrete LogicalType subtypes for isinstance checks (Union alias can't be used).
_LOGICAL_TYPES = (NodeRef, EdgeRef, ScalarType, PathType, ListType)

# Sentinel for getattr default when the attribute doesn't exist on an operator.
_MISSING = object()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _walk(
    op: LogicalPlan,
    visited: Set[int],
    path: Set[int],
    cycle_pairs: List[Tuple[str, str]],
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
    for _slot, child in iter_children(op):
        if id(child) in path:
            cycle_pairs.append((type(op).__name__, type(child).__name__))
        else:
            yield from _walk(child, visited, path, cycle_pairs)
    path.discard(node_id)


def _check_predicate(op: LogicalPlan, pred: BoundPredicate, label: str) -> list[CompilerError]:
    """Return errors for *pred*: empty expression or out-of-scope references."""
    errors: list[CompilerError] = []
    if not pred.expression:
        errors.append(CompilerError(
            message=f"{type(op).__name__} op_id={op.op_id}: {label} has empty expression"
        ))
    if pred.references:
        visible = frozenset(op.output_schema.columns.keys())
        unknown = pred.references - visible
        if unknown:
            errors.append(CompilerError(
                message=(
                    f"{type(op).__name__} op_id={op.op_id}: {label} references "
                    f"aliases not in output_schema: {sorted(unknown)}"
                )
            ))
    return errors


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
        for attr in CHILD_SLOTS:
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
            # Optional arms may produce NULL rows when the pattern is absent.
            # ScalarType columns in the output_schema must be nullable to
            # accommodate those NULL rows.
            for col, typ in op.output_schema.columns.items():
                if isinstance(typ, ScalarType) and not typ.nullable:
                    errors.append(CompilerError(
                        message=(
                            f"PatternMatch op_id={op.op_id}: optional=True "
                            f"but output_schema column {col!r} is ScalarType(nullable=False); "
                            "optional arms must produce nullable outputs"
                        )
                    ))

    # ------------------------------------------------------------------
    # Structural cycle diagnostic (reported after traversal completes)
    # ------------------------------------------------------------------
    for parent_type, child_type in cycle_pairs:
        errors.append(CompilerError(
            message=f"Cycle detected: {parent_type} has a child edge back to ancestor {child_type}"
        ))

    return errors


# ---------------------------------------------------------------------------
# Schema checker (invariant 4)
# ---------------------------------------------------------------------------

def _check_logical_type(typ: object, label: str, op: LogicalPlan) -> list[CompilerError]:
    """Recursively validate *typ* is a valid LogicalType, including ListType.element_type."""
    errors: list[CompilerError] = []
    if not isinstance(typ, _LOGICAL_TYPES):
        errors.append(CompilerError(
            message=(
                f"{type(op).__name__} op_id={op.op_id}: "
                f"{label} has invalid type {type(typ).__name__!r}, expected LogicalType"
            )
        ))
        return errors  # can't recurse into a non-LogicalType value
    if isinstance(typ, ListType):
        errors.extend(_check_logical_type(typ.element_type, f"{label}.element_type", op))
    return errors


def _check_schema(op: LogicalPlan, schema: RowSchema) -> list[CompilerError]:
    errors: list[CompilerError] = []
    for col, typ in schema.columns.items():
        errors.extend(_check_logical_type(typ, f"output_schema column {col!r}", op))
    return errors
