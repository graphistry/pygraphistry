"""Predicate-pushdown safety primitives for the M4 pass framework.

Two independent utilities consumed by PredicatePushdownPass (M4-PR3) and
future verifier extensions:

  is_null_rejecting() / is_null_safe()
      Classify whether a BoundPredicate would filter out NULL-extended rows
      produced by OPTIONAL MATCH.  Conservative syntactic heuristic — unknown
      expression forms are treated as null-rejecting.

  with_barrier_blocks_pushdown()
      Detect whether a WITH scope boundary in the accumulated ScopeFrame stack
      prevents backward predicate movement for a given reference set.
"""
from __future__ import annotations

from typing import FrozenSet, List

from graphistry.compute.gfql.ir.bound_ir import ScopeFrame
from graphistry.compute.gfql.ir.types import BoundPredicate


# ---------------------------------------------------------------------------
# Null-safety classifier
# ---------------------------------------------------------------------------

# Expression substrings that indicate a predicate handles NULL values
# explicitly.  Presence of any of these means the predicate is null-safe
# regardless of which aliases it references.
_NULL_SAFE_FORMS = (
    " is null",
    " is not null",
    "coalesce(",
    "nullif(",
)


def is_null_rejecting(
    predicate: BoundPredicate,
    null_extended_aliases: FrozenSet[str],
) -> bool:
    """Return True if *predicate* would filter out NULL-extended rows.

    A predicate is null-rejecting when it references at least one alias from
    *null_extended_aliases* and does not use a null-safe expression form
    (IS NULL, IS NOT NULL, COALESCE, NULLIF).  Such a predicate must not be
    pushed into an OPTIONAL MATCH arm because it would silently turn the
    null-extension rows into filtered-out rows, breaking OPTIONAL MATCH
    semantics.

    Conservative default: predicates with empty expressions are treated as
    null-rejecting (caller should populate the expression field).

    :param predicate: The bound predicate to classify.
    :param null_extended_aliases: Aliases that may be NULL from OPTIONAL MATCH.
    :returns: True if the predicate cannot be safely pushed into an optional arm.
    """
    if not predicate.references & null_extended_aliases:
        return False
    if not predicate.expression:
        return True
    expr_lower = predicate.expression.lower()
    for form in _NULL_SAFE_FORMS:
        if form in expr_lower:
            return False
    return True


def is_null_safe(
    predicate: BoundPredicate,
    null_extended_aliases: FrozenSet[str],
) -> bool:
    """Return True if *predicate* safely handles NULL in *null_extended_aliases*.

    Inverse of :func:`is_null_rejecting`.  A predicate is null-safe either
    because it does not reference any null-extended alias, or because it uses
    an explicit null-handling form (IS NULL, IS NOT NULL, COALESCE, NULLIF).
    """
    return not is_null_rejecting(predicate, null_extended_aliases)


# ---------------------------------------------------------------------------
# WITH barrier detection
# ---------------------------------------------------------------------------

def with_barrier_blocks_pushdown(
    scope_stack: List[ScopeFrame],
    predicate_refs: FrozenSet[str],
) -> bool:
    """Return True if a WITH boundary in *scope_stack* blocks backward pushdown.

    Pushdown is blocked when there is at least one ScopeFrame whose
    ``origin_clause`` is ``"WITH"`` (case-insensitive) and the predicate
    references an alias that is **not** visible before that boundary
    (i.e. not in ``frame.visible_vars``).

    A WITH clause limits the variables it forwards: any alias not listed in
    the WITH projection cannot appear in a predicate that is pushed backward
    through it.

    :param scope_stack: Accumulated scope frames from the binder, outermost
        first (index 0) to innermost (last).
    :param predicate_refs: Alias names the predicate reads.
    :returns: True if backward pushdown is blocked by a WITH boundary.
    """
    for frame in scope_stack:
        if frame.origin_clause.upper() == "WITH":
            if predicate_refs - frame.visible_vars:
                return True
    return False
