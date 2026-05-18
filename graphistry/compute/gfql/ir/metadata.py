"""Stable type/nullability metadata helpers for BoundIR / LogicalPlan seams.

Thin contract surface over the type metadata the IR already carries
(``BoundVariable.nullable``, ``RowSchema.columns``, ``ScalarType.nullable``).
Callers in T2 strict-validation, T4 arrow-bridge, and downstream rewrite passes
should consult these helpers rather than reach into dataclass internals — the
indirection lets later slices evolve type representations without re-touching
every call site.

Convention: nullability is currently tracked only on ``ScalarType``.
``NodeRef``, ``EdgeRef``, ``PathType``, ``ListType`` are treated as
non-null structural containers — :func:`is_nullable` returns ``False`` for
them. If a future slice adds an ``Optional[T]`` wrapper, only this module
needs to learn about it.

Issue: #1300 (T3 under #1262, umbrella #1046).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from graphistry.compute.gfql.ir.bound_ir import BoundVariable
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import (
    EdgeRef,
    ListType,
    LogicalType,
    NodeRef,
    PathType,
    ScalarType,
)

__all__ = [
    "is_nullable",
    "with_nullable",
    "widen_to_nullable",
    "column_logical_type",
    "column_is_nullable",
    "merge_types",
    "bound_variable_type",
    "bound_variable_is_nullable",
]


def is_nullable(typ: LogicalType) -> bool:
    """Return whether *typ* admits NULL.

    Only ScalarType carries an explicit ``nullable`` bit; the structural
    types are not nullable on their own.
    """
    return isinstance(typ, ScalarType) and typ.nullable


def with_nullable(typ: LogicalType, nullable: bool) -> LogicalType:
    """Return a copy of *typ* with the requested ``nullable`` flag.

    For non-ScalarType inputs this is a pass-through — there is no
    nullability dimension to set on NodeRef/EdgeRef/PathType/ListType today.
    """
    if isinstance(typ, ScalarType):
        if typ.nullable == nullable:
            return typ
        return replace(typ, nullable=nullable)
    return typ


def widen_to_nullable(typ: LogicalType) -> LogicalType:
    """Force ScalarType nullability to True; structural types pass through.

    Used at optional-arm boundaries and at union/outer-join merges where the
    column may be NULL on one side.
    """
    return with_nullable(typ, True)


def column_logical_type(schema: RowSchema, name: str) -> Optional[LogicalType]:
    """Return the LogicalType for column *name*, or None if absent."""
    return schema.columns.get(name)


def column_is_nullable(schema: RowSchema, name: str) -> Optional[bool]:
    """Return whether column *name* is nullable.

    None when the column is absent OR when the column is non-scalar — callers
    expecting a tri-state can distinguish "unknown" from "yes/no". Use
    :func:`column_logical_type` first when the kind matters.
    """
    typ = schema.columns.get(name)
    if typ is None:
        return None
    if not isinstance(typ, ScalarType):
        return None
    return typ.nullable


def merge_types(left: LogicalType, right: LogicalType) -> Optional[LogicalType]:
    """Least-upper-bound for two LogicalTypes meeting at a column boundary.

    Used by union/outer-join column merges where the same logical column
    appears on both branches. Returns None when the two sides disagree on
    kind (no implicit cross-kind coercion at the IR layer).

    Rules:
    - ScalarType ⊔ ScalarType: same kind only; nullable is OR'd.
      Differing kinds (e.g. ``int64`` vs ``string``) → None.
      ``unknown`` is treated as a wildcard that takes the other side's kind.
    - NodeRef ⊔ NodeRef: union of label sets; both must be NodeRef.
    - EdgeRef ⊔ EdgeRef: type/src_label/dst_label kept only when equal,
      otherwise widened to None on that field.
    - ListType ⊔ ListType: recurse into element_type; None on incompatible
      elements.
    - PathType ⊔ PathType: min_hops=min(left,right), max_hops=max(left,right).
    - All other cross-kind pairings → None.
    """
    if isinstance(left, ScalarType) and isinstance(right, ScalarType):
        if left.kind == right.kind:
            kind = left.kind
        elif left.kind == "unknown":
            kind = right.kind
        elif right.kind == "unknown":
            kind = left.kind
        else:
            return None
        return ScalarType(kind=kind, nullable=left.nullable or right.nullable)
    if isinstance(left, NodeRef) and isinstance(right, NodeRef):
        return NodeRef(labels=left.labels | right.labels)
    if isinstance(left, EdgeRef) and isinstance(right, EdgeRef):
        return EdgeRef(
            type=left.type if left.type == right.type else None,
            src_label=left.src_label if left.src_label == right.src_label else None,
            dst_label=left.dst_label if left.dst_label == right.dst_label else None,
        )
    if isinstance(left, ListType) and isinstance(right, ListType):
        merged_elem = merge_types(left.element_type, right.element_type)
        if merged_elem is None:
            return None
        return ListType(element_type=merged_elem)
    if isinstance(left, PathType) and isinstance(right, PathType):
        return PathType(
            min_hops=min(left.min_hops, right.min_hops),
            max_hops=max(left.max_hops, right.max_hops),
        )
    return None


def bound_variable_type(bv: BoundVariable) -> LogicalType:
    """Return the BoundVariable's LogicalType with its `nullable` flag honored.

    For scalar variables the binder may set ``BoundVariable.nullable``
    independently of the ``ScalarType.nullable`` field embedded in
    ``logical_type``; this helper returns a single LogicalType where the two
    are reconciled (BoundVariable.nullable wins for scalars).

    For structural types (NodeRef/EdgeRef/PathType/ListType) the LogicalType
    is returned unchanged — those families do not currently carry a
    nullability dimension on the LogicalType itself, so callers asking
    "is this *variable* nullable" should consult :func:`bound_variable_is_nullable`
    directly. ``bv.nullable=True`` on a structural variable (e.g. an
    optional-arm whole-row alias) is *not* propagated onto the returned
    LogicalType; this is the documented contract.
    """
    if isinstance(bv.logical_type, ScalarType):
        return with_nullable(bv.logical_type, bv.nullable)
    return bv.logical_type


def bound_variable_is_nullable(bv: BoundVariable) -> bool:
    """Return whether *bv* admits NULL.

    Source of truth is ``BoundVariable.nullable``: the binder records
    nullability on every variable regardless of LogicalType family
    (e.g. optional-arm whole-row aliases get ``nullable=True`` even though
    their NodeRef/EdgeRef LogicalType has no nullable dimension). Use this
    helper instead of ``is_nullable(bound_variable_type(bv))``, which
    would always return ``False`` for structural variables and silently
    drop the binder-recorded nullable bit.
    """
    return bv.nullable
