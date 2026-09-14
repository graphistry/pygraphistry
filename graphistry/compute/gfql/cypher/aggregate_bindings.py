"""Bindings-row routing gates for Cypher aggregates over relationship patterns.

A relationship MATCH can bind the same node into many rows; the per-alias node
table collapses that multiplicity, so an aggregate whose value depends on row
multiplicity (sum/avg/collect/plain count) must run on binding rows instead.
``requires_aggregate_bindings`` is the one gate the lowering asks: it covers the
multiplicity-sensitive family AND the cross-alias shape -- a sole
multiplicity-INSENSITIVE aggregate (min/max/count DISTINCT) whose clause spans
two MATCH aliases has no single source table to lower onto at all, and binding
rows are sound for it because its value is multiplicity-invariant. The lowering's
force-bindings block still vets each spec before actually engaging.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Set

from graphistry.compute.ast import ASTObject
from graphistry.compute.exceptions import GFQLValidationError

if TYPE_CHECKING:
    from graphistry.compute.gfql.cypher.ast import ReturnClause
    from graphistry.compute.gfql.cypher.lowering import _AggregateSpec


def is_multiplicity_sensitive_aggregate(agg_spec: "_AggregateSpec") -> bool:
    if agg_spec.func in {"sum", "avg"}:
        return True
    if agg_spec.func == "collect":
        return True
    if agg_spec.func == "count":
        return not agg_spec.distinct
    return False


def requires_relationship_multiplicity_bindings(
    *,
    aggregate_specs: Sequence["_AggregateSpec"],
    relationship_count: int,
) -> bool:
    return relationship_count > 0 and any(
        is_multiplicity_sensitive_aggregate(spec)
        for spec in aggregate_specs
    )


def _clause_spans_multiple_match_aliases(
    clause: "ReturnClause",
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],  # hygiene-ok: explicit-any -- untyped Cypher query-params mapping
) -> bool:
    from graphistry.compute.gfql.cypher.lowering import _expr_match_aliases

    referenced: Set[str] = set()
    for item in clause.items:
        try:
            referenced |= _expr_match_aliases(
                item.expression.text,
                alias_targets=alias_targets,
                params=params,
                field=clause.kind,
                line=item.span.line,
                column=item.span.column,
            )
        except GFQLValidationError:
            return False  # unanalyzable expression -> keep the conservative path
        if len(referenced) >= 2:
            return True
    return False


def requires_aggregate_bindings(
    *,
    aggregate_specs: Sequence["_AggregateSpec"],
    relationship_count: int,
    clause: "ReturnClause",
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],  # hygiene-ok: explicit-any -- untyped Cypher query-params mapping
) -> bool:
    """True when this clause's aggregates must route to the binding-rows table."""
    if requires_relationship_multiplicity_bindings(
        aggregate_specs=aggregate_specs,
        relationship_count=relationship_count,
    ):
        return True
    if relationship_count <= 0 or not aggregate_specs or len(alias_targets) < 2:
        return False
    return _clause_spans_multiple_match_aliases(
        clause,
        alias_targets=alias_targets,
        params=params,
    )
