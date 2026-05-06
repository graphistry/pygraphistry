"""Structural verifier for LogicalPlan (M2-PR3 / issue #1127).

Production callers: ``passes/manager.py`` runs ``verify(...)`` after every
tier-1 structural pass and every tier-2 rewrite-rule fixed-point step
(treats any returned diagnostic as fatal); ``cypher/lowering.py`` runs it
through ``_verify_selected_logical_plan`` to gate LogicalPlan routing for
covered Cypher shapes. So the invariants below are real safety nets on the
paths that exercise them.

Scope caveat for invariant 6 (#1300, T3): the kind/nullability check is
short-circuited when either ``input.output_schema.columns`` or
``op.output_schema.columns`` is empty. Most planner-emitted ``Project``
and ``Aggregate`` nodes today initialise ``output_schema=RowSchema()``
empty — until the schema-population slice lands, invariant 6 mainly
catches hand-built fixtures and pass-emitted plans that explicitly
populate schemas. This is intentional — populating projection schemas is
T4-adjacent work, not T3.

verify(plan) walks the operator tree and checks six invariants:
  1. op_id uniqueness  — non-zero op_ids are distinct across the whole tree
  2. Dangling refs     — child slots (input/left/right/subquery) hold LogicalPlan | None
  3. Predicate scope   — BoundPredicate.expression must be non-empty on all predicate-bearing ops
  4. Schema validity   — RowSchema column values must be LogicalType instances (ListType.element_type validated recursively)
  5. Optional-arm      — PatternMatch(optional=True) must carry a non-empty arm_id
  6. Type continuity   — for unary ops, columns shared between input.output_schema and
                         op.output_schema must agree on kind; ScalarType nullability is
                         monotone-widening except where the operator is allowed to drop NULL
                         rows (currently: Filter, PatternMatch, SemiApply, AntiSemiApply).
                         Skipped when either schema is empty. (#1300, T3 under #1262.)
"""
from __future__ import annotations

from typing import Iterator, List, Set, Tuple

from graphistry.compute.gfql.ir.compilation import CompilerError
from graphistry.compute.gfql.ir.logical_plan import (
    AntiSemiApply,
    CHILD_SLOTS,
    Filter,
    IndexScan,
    LogicalPlan,
    PatternMatch,
    RowSchema,
    SemiApply,
    iter_children,
)
from graphistry.compute.gfql.ir.metadata import is_nullable
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
        # Invariant 6: Type propagation continuity (#1300 / T3)
        # Walk the unary `input` slot (if any) and compare shared columns'
        # kinds + ScalarType nullability against the parent op's
        # output_schema.  Skipped when either schema is empty so legacy
        # planner-emitted plans that initialise `output_schema=RowSchema()`
        # remain valid until callers populate the schema.
        # ------------------------------------------------------------------
        errors.extend(_check_propagation_continuity(op))

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
                if isinstance(typ, ScalarType) and not is_nullable(typ):
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


# ---------------------------------------------------------------------------
# Propagation continuity (invariant 6) — #1300 T3
# ---------------------------------------------------------------------------

# Operators that may legitimately *narrow* ScalarType nullability — they can
# drop NULL rows, so a non-nullable output is consistent with a nullable
# input.  Other unary operators must preserve or widen nullability.
#
# `Filter` predicates can drop NULL rows directly. `PatternMatch` with
# `where` and per-pattern `predicates` carries the same row-dropping
# semantics — non-optional patterns also drop rows where the pattern fails
# to match — so it sits in the same carve-out. `SemiApply` /
# `AntiSemiApply` (Cypher EXISTS / NOT EXISTS subquery filters) are
# filter-shaped row-droppers as well. The narrowing here is *not*
# extended to `optional=True` PatternMatch arms: invariant 5 already pins
# their outputs to nullable=True, so a narrowing output there is still a
# correctness violation but is reported by invariant 5 rather than 6.
_NULLABILITY_NARROWING_OPS: tuple[type, ...] = (Filter, PatternMatch, SemiApply, AntiSemiApply)


def _check_propagation_continuity(op: LogicalPlan) -> list[CompilerError]:
    """Validate type/nullability continuity across a unary op's input edge.

    Walks the single ``input`` slot when present.  For each column name
    appearing in BOTH ``input.output_schema.columns`` and
    ``op.output_schema.columns``, the kinds must agree
    (NodeRef/EdgeRef/ScalarType/PathType/ListType) and ScalarType nullability
    must monotonically widen (input.nullable=True → op.nullable=True), with a
    carve-out for operators in ``_NULLABILITY_NARROWING_OPS``.

    Skipped entirely when either schema has no columns — most planner-emitted
    Project/Aggregate nodes today initialise ``output_schema=RowSchema()`` and
    we don't want to retro-break those plans before the schema-population
    slice lands.  See ``_check_schema`` for the always-on structural check.
    """
    parent_input = getattr(op, "input", None)
    if not isinstance(parent_input, LogicalPlan):
        return []
    parent_cols = parent_input.output_schema.columns
    child_cols = op.output_schema.columns
    if not parent_cols or not child_cols:
        return []
    errors: list[CompilerError] = []
    allow_narrow = isinstance(op, _NULLABILITY_NARROWING_OPS)
    for name, child_typ in child_cols.items():
        parent_typ = parent_cols.get(name)
        if parent_typ is None:
            continue  # newly-introduced column — nothing to compare against
        # Kind continuity: must agree on the LogicalType family.
        if type(parent_typ) is not type(child_typ):
            errors.append(CompilerError(
                message=(
                    f"{type(op).__name__} op_id={op.op_id}: column {name!r} "
                    f"changed kind across input edge: "
                    f"{type(parent_typ).__name__} → {type(child_typ).__name__}"
                )
            ))
            continue
        # Nullability monotonicity: only meaningful for ScalarType today.
        if (
            isinstance(parent_typ, ScalarType)
            and isinstance(child_typ, ScalarType)
            and not allow_narrow
            and is_nullable(parent_typ)
            and not is_nullable(child_typ)
        ):
            narrowing_op_names = ", ".join(t.__name__ for t in _NULLABILITY_NARROWING_OPS)
            errors.append(CompilerError(
                message=(
                    f"{type(op).__name__} op_id={op.op_id}: column {name!r} "
                    "narrowed nullability across input edge "
                    "(input nullable=True, output nullable=False); only "
                    f"row-dropping operators may narrow nullability "
                    f"({narrowing_op_names})"
                )
            ))
    return errors
