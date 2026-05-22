"""Structural verifier for LogicalPlan.

Production callers run this after structural rewrites and before routed
LogicalPlan execution, so these checks are safety gates rather than fixture-only
assertions. The continuity check is intentionally skipped when either side has
an empty schema: several planner-emitted nodes still use ``RowSchema()`` until
the schema-population slices fill them in.

The verifier checks:
- non-zero ``op_id`` uniqueness
- valid child slots
- predicate scope
- recursive schema type validity
- optional-arm ``arm_id`` and nullable outputs
- shared-column type/nullability continuity across unary input edges
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
from graphistry.compute.gfql.ir.types import BoundPredicate, EdgeRef, ListType, NodeRef, PathType, ScalarType

_LOGICAL_TYPES = (NodeRef, EdgeRef, ScalarType, PathType, ListType)
_MISSING = object()


def _walk(
    op: LogicalPlan,
    visited: Set[int],
    path: Set[int],
    cycle_pairs: List[Tuple[str, str]],
) -> Iterator[LogicalPlan]:
    """Walk a LogicalPlan DAG once while recording true ancestor cycles."""
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


def verify(plan: LogicalPlan) -> list[CompilerError]:
    """Return all structural invariant violations in *plan*."""
    errors: list[CompilerError] = []
    seen_ids: set[int] = set()
    visited: set[int] = set()
    path: set[int] = set()
    cycle_pairs: list[tuple[str, str]] = []

    for op in _walk(plan, visited, path, cycle_pairs):
        # op_id=0 is the dataclass "unassigned" sentinel used by hand-built
        # fixtures, so only non-zero planner IDs participate in uniqueness.
        if op.op_id != 0:
            if op.op_id in seen_ids:
                errors.append(CompilerError(
                    message=f"Duplicate op_id {op.op_id} on {type(op).__name__}"
                ))
            else:
                seen_ids.add(op.op_id)

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

        if isinstance(op, PatternMatch):
            for i, pred in enumerate(op.predicates):
                errors.extend(_check_predicate(op, pred, f"predicate[{i}]"))
        elif isinstance(op, Filter):
            errors.extend(_check_predicate(op, op.predicate, "predicate"))
        elif isinstance(op, IndexScan):
            errors.extend(_check_predicate(op, op.predicate, "predicate"))
            for i, pred in enumerate(op.residual_predicates):
                errors.extend(_check_predicate(op, pred, f"residual_predicates[{i}]"))

        errors.extend(_check_schema(op, op.output_schema))
        errors.extend(_check_propagation_continuity(op))

        if isinstance(op, PatternMatch) and op.optional:
            if not op.arm_id:
                errors.append(CompilerError(
                    message=(
                        f"PatternMatch op_id={op.op_id}: optional=True "
                        "but arm_id is missing or empty"
                    )
                ))
            # Optional arms can synthesize NULL rows when the pattern is
            # absent, so every scalar output must advertise that possibility.
            for col, typ in op.output_schema.columns.items():
                if isinstance(typ, ScalarType) and not typ.nullable:
                    errors.append(CompilerError(
                        message=(
                            f"PatternMatch op_id={op.op_id}: optional=True "
                            f"but output_schema column {col!r} is ScalarType(nullable=False); "
                            "optional arms must produce nullable outputs"
                        )
                    ))

    for parent_type, child_type in cycle_pairs:
        errors.append(CompilerError(
            message=f"Cycle detected: {parent_type} has a child edge back to ancestor {child_type}"
        ))

    return errors


def _check_logical_type(typ: object, label: str, op: LogicalPlan) -> list[CompilerError]:
    errors: list[CompilerError] = []
    if not isinstance(typ, _LOGICAL_TYPES):
        errors.append(CompilerError(
            message=(
                f"{type(op).__name__} op_id={op.op_id}: "
                f"{label} has invalid type {type(typ).__name__!r}, expected LogicalType"
            )
        ))
        return errors
    if isinstance(typ, ListType):
        errors.extend(_check_logical_type(typ.element_type, f"{label}.element_type", op))
    return errors


def _check_schema(op: LogicalPlan, schema: RowSchema) -> list[CompilerError]:
    errors: list[CompilerError] = []
    for col, typ in schema.columns.items():
        errors.extend(_check_logical_type(typ, f"output_schema column {col!r}", op))
    return errors


# Row-dropping unary operators may prove away NULL rows. Optional PatternMatch
# remains guarded by the optional-arm invariant above.
_NULLABILITY_NARROWING_OPS: tuple[type, ...] = (Filter, PatternMatch, SemiApply, AntiSemiApply)


def _check_propagation_continuity(op: LogicalPlan) -> list[CompilerError]:
    """Validate shared-column type/nullability continuity across input edges."""
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
            continue
        if type(parent_typ) is not type(child_typ):
            errors.append(CompilerError(
                message=(
                    f"{type(op).__name__} op_id={op.op_id}: column {name!r} "
                    f"changed kind across input edge: "
                    f"{type(parent_typ).__name__} → {type(child_typ).__name__}"
                )
            ))
            continue
        if (
            isinstance(parent_typ, ScalarType)
            and isinstance(child_typ, ScalarType)
            and not allow_narrow
            and parent_typ.nullable
            and not child_typ.nullable
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
