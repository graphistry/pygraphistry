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
# explicitly when used as the sole (non-compound) expression.
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

    Conservative defaults:
    - Empty expressions are treated as null-rejecting.
    - Compound AND expressions are always treated as null-rejecting, even when
      one conjunct contains a null-safe form.  With alias=NULL, ``True AND NULL``
      equals NULL (falsy), so the AND makes the whole expression null-rejecting
      regardless of the left side.  Example: ``n.name IS NULL AND n.type = 'x'``
      contains IS NULL but is null-rejecting overall.

    Compound OR is not analyzed — a null-safe form anywhere in an OR chain
    correctly triggers the null-safe classification because ``True OR <anything>``
    is True when the null-safe conjunct evaluates to True for NULL inputs.

    :param predicate: The bound predicate to classify.
    :param null_extended_aliases: Aliases that may be NULL from OPTIONAL MATCH.
    :returns: True if the predicate cannot be safely pushed into an optional arm.
    """
    if not predicate.references & null_extended_aliases:
        return False
    if not predicate.expression:
        return True
    expr_lower = predicate.expression.lower()
    # AND compounds are always null-rejecting: even if one conjunct is null-safe,
    # the other may not be, and True AND NULL = NULL (row filtered).
    if " and " in expr_lower:
        return True
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

    .. note::
        This function operates on ``BoundIR.scope_stack`` (binder output),
        not on a ``LogicalPlan`` tree.  ``PredicatePushdownPass`` (M4-PR3)
        will need either a plan-level ``WithBarrier`` operator or companion
        ``scope_stack`` threading to use this check during plan-tree walks.
    """
    for frame in scope_stack:
        if frame.origin_clause.upper() == "WITH":
            if predicate_refs - frame.visible_vars:
                return True
    return False
