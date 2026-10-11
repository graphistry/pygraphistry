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

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Set, Tuple

from graphistry.compute.ast import ASTEdge, ASTNode, ASTObject
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.same_path_types import EDGE_IDENTITY_COLUMN, NODE_IDENTITY_COLUMN

if TYPE_CHECKING:
    from graphistry.compute.gfql.cypher.ast import CypherQuery, ReturnClause, ReturnItem
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


def per_path_aggregate_bindings_apply(
    query: "CypherQuery",
    *,
    aggregate_specs: Sequence["_AggregateSpec"],
    non_aggregate_items: Sequence["ReturnItem"],
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],  # hygiene-ok: explicit-any -- untyped Cypher query-params mapping
) -> bool:
    """True when a single required MATCH's aggregates can run on per-path binding rows.

    Binding rows hold one row per matched path with every alias materialized, so
    grouping keys and aggregate arguments may reference any node or edge alias and
    multiplicity is exactly the path count. Shapes outside this check keep the
    lowering's narrower carve-outs: OPTIONAL MATCH, several MATCH clauses, UNWIND,
    an item mixing aggregate and non-aggregate references, a whole-row group on an
    edge alias, or a non-count aggregate of a whole entity such as ``collect(b)``.
    """
    from graphistry.compute.gfql.cypher.lowering import _expr_match_aliases

    if len(query.matches) != 1 or query.matches[0].optional or query.unwinds or query.reentry_matches:
        return False
    try:
        for item in non_aggregate_items:
            text = item.expression.text
            if text in alias_targets:
                if not isinstance(alias_targets[text], ASTNode):
                    return False
                continue
            _expr_match_aliases(  # raises for an unanalyzable expression
                text, alias_targets=alias_targets, params=params,
                field=query.return_.kind, line=item.span.line, column=item.span.column,
            )
        for spec in aggregate_specs:
            if spec.expr_text is None:
                continue
            arg = spec.expr_text.strip()
            if arg in alias_targets:
                if spec.func != "count":
                    return False
                continue
            _expr_match_aliases(  # raises for an unanalyzable expression
                arg, alias_targets=alias_targets, params=params,
                field=query.return_.kind, line=spec.span_line, column=spec.span_column,
            )
    except GFQLValidationError:
        return False
    return not _clause_mixes_group_and_aggregate_refs(query, alias_targets=alias_targets, params=params)


def _aliases_outside_aggregates(node: object, aliases: Set[str]) -> Tuple[bool, Set[str]]:
    """(has an aggregate call, MATCH aliases referenced outside every aggregate call)."""
    from graphistry.compute.gfql.cypher.lowering import _CYPHER_AGGREGATES
    from graphistry.compute.gfql.expr_parser import FunctionCall, Identifier, is_expr_node

    if isinstance(node, FunctionCall) and node.name.lower() in _CYPHER_AGGREGATES:
        return True, set()
    if isinstance(node, Identifier):
        head = node.name.split(".", 1)[0]
        return False, {head} & aliases
    has_agg: bool = False
    refs: Set[str] = set()
    for child in vars(node).values() if hasattr(node, "__dict__") else ():
        for c in child if isinstance(child, (list, tuple)) else (child,):
            if is_expr_node(c):
                child_agg, child_refs = _aliases_outside_aggregates(c, aliases)
                has_agg, refs = has_agg or child_agg, refs | child_refs
    return has_agg, refs


def _clause_mixes_group_and_aggregate_refs(
    query: "CypherQuery",
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],  # hygiene-ok: explicit-any -- untyped Cypher query-params mapping
) -> bool:
    """True if a RETURN or ORDER BY item combines an aggregate with a MATCH alias outside it.

    ``a.name + count(b)`` has no single value per group; ``count(b) + 1`` and
    ``sum(r.w) * 2`` do. An aggregate inside ORDER BY, and an unparseable item, also
    count as mixed.
    """
    from graphistry.compute.gfql.cypher.lowering import _parse_row_expr

    items = [(i.expression.text, i.span.line, i.span.column, False) for i in query.return_.items]
    if query.order_by is not None:
        items += [(i.expression.text, i.span.line, i.span.column, True) for i in query.order_by.items]
    aliases = set(alias_targets.keys())
    for text, line, column, in_order_by in items:
        try:
            node = _parse_row_expr(
                text, params=params, alias_targets=alias_targets,
                allow_missing_params=True, field="return", line=line, column=column,
            )
        except GFQLValidationError:
            return True
        has_agg, outside = _aliases_outside_aggregates(node, aliases)
        if has_agg and (outside or in_order_by):
            return True  # ORDER BY aggregates keep the existing lowering and its errors
    return False


def distinct_aggregate_expr_text(
    agg_spec: "_AggregateSpec",
    *,
    alias_targets: Mapping[str, ASTObject],
    binding_rows: bool = False,
) -> Optional[str]:
    """Column a DISTINCT aggregate compares; on binding rows each alias holds its own identity."""
    from graphistry.compute.gfql.cypher.lowering import _unsupported

    expr_text = agg_spec.expr_text
    if expr_text is None:
        return None
    target = alias_targets.get(expr_text)
    if binding_rows and agg_spec.func == "count" and isinstance(target, (ASTNode, ASTEdge)):
        return expr_text
    if isinstance(target, ASTNode):
        return NODE_IDENTITY_COLUMN
    if isinstance(target, ASTEdge):
        if agg_spec.func == "collect":
            raise _unsupported(
                "collect(DISTINCT rel_alias) is not yet supported in local Cypher lowering",
                field="return.item",
                value=agg_spec.source_text,
                line=agg_spec.span_line,
                column=agg_spec.span_column,
            )
        return EDGE_IDENTITY_COLUMN
    return expr_text


def aggregate_runtime_spec(
    agg_spec: "_AggregateSpec",
    *,
    alias_targets: Mapping[str, ASTObject],
    binding_rows: bool = False,
) -> Tuple[str, Optional[str]]:
    """(runtime aggregation function, argument expression) for one Cypher aggregate."""
    from graphistry.compute.gfql.cypher.lowering import _unsupported

    func = agg_spec.func
    expr_text = agg_spec.expr_text
    if expr_text is not None:
        target = alias_targets.get(expr_text)
        if isinstance(target, ASTNode) and func in {"collect", "collect_distinct"}:
            expr_text = f"__node_entity__({expr_text})"
        elif isinstance(target, ASTEdge) and func in {"collect", "collect_distinct"}:
            expr_text = f"__edge_entity__({expr_text})"
    if not agg_spec.distinct:
        return func, expr_text
    if func in ("count", "collect"):
        return f"{func}_distinct", distinct_aggregate_expr_text(
            agg_spec, alias_targets=alias_targets, binding_rows=binding_rows
        )
    raise _unsupported(
        "Cypher DISTINCT aggregates are currently supported for count() and collect() only",
        field="return.item",
        value=agg_spec.source_text,
        line=agg_spec.span_line,
        column=agg_spec.span_column,
    )
