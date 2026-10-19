"""``WHERE alias.prop IN [scalar literals]`` seeds the MATCH pattern.

A literal equality on a MATCH alias lowers onto the pattern's filter dict; a literal list did
not, so the seed reached the executor as a post-join row filter. Each top-level AND conjunct of
that shape becomes ``is_in([...])`` on the node or edge alias and leaves the row WHERE. Lists
holding null keep their three-valued evaluation; OR, NOT and nested lists stay in the WHERE.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple, Union

from graphistry.compute.ast import ASTEdge, ASTNode, ASTObject
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.expr_parser import BinaryOp, Identifier, ListLiteral, Literal as ExprLiteral, PropertyAccessExpr
from graphistry.compute.predicates.is_in import is_in

if TYPE_CHECKING:
    from graphistry.compute.gfql.cypher.ast import ExpressionText, PropertyRef, SourceSpan

Scalar = Union[str, int, float]


def _literal_membership_seed(
    text: str,
    *,
    span: "SourceSpan",
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],  # hygiene-ok: explicit-any -- Cypher params are heterogeneous JSON scalars
) -> Optional[Tuple["PropertyRef", List[Scalar]]]:
    """``alias.prop IN [scalar literals]`` as (property, values), else None."""
    from graphistry.compute.gfql.cypher.ast import PropertyRef
    from graphistry.compute.gfql.cypher.lowering import _ZONED_ISO_TEMPORAL_TEXT_RE, _parse_row_expr

    try:
        node = _parse_row_expr(
            text, params=params, alias_targets=alias_targets, field="where", line=span.line, column=span.column,
        )
    except GFQLValidationError:
        return None  # the row evaluator reports it, with its own wording
    if not isinstance(node, BinaryOp) or node.op != "in":
        return None
    if not (isinstance(node.left, PropertyAccessExpr) and isinstance(node.left.value, Identifier)):
        return None
    alias, prop = node.left.value.name, node.left.property
    if not isinstance(node.right, ListLiteral):
        return None  # a list held in a property or a parameter row is not a constant
    values: List[Scalar] = []
    for item in node.right.items:  # ExprLiteral.value is untyped; this loop is the type check
        value = item.value if isinstance(item, ExprLiteral) else None
        if isinstance(value, bool) or not isinstance(value, (str, int, float)) or (isinstance(value, float) and math.isnan(value)):
            return None  # null and NaN carry three-valued verdicts; a nested list is structural; `true == 1` is not every engine's isin
        if isinstance(value, str) and _ZONED_ISO_TEMPORAL_TEXT_RE.match(value) is not None:
            return None  # datetime('...') lowers to zoned ISO text; the row path compares it as an instant, like `=` keeps it
        values.append(value)
    if not values:
        return None  # `x IN []` is false for every row; NeverMatch has no wire form for the bindings op, so the WHERE keeps it
    if any(isinstance(v, str) for v in values) and not all(isinstance(v, str) for v in values):
        return None  # a list mixing text and numbers is compared element-wise by the row path; isin would coerce
    if not isinstance(alias_targets.get(alias), (ASTNode, ASTEdge)):
        return None
    return PropertyRef(alias=alias, property=prop, span=span), values


def _apply_membership_where(targets: Mapping[str, ASTObject], *, left: "PropertyRef", values: List[Scalar]) -> None:
    from graphistry.compute.gfql.cypher.lowering import (
        _merge_filter_predicates, _set_target_filter_dict, _target_filter_dict,
    )

    target = targets[left.alias]
    filter_dict = dict(_target_filter_dict(target) or {})
    new_filter = is_in(list(values))
    existing_filter = filter_dict.get(left.property)
    if existing_filter is None or left.property not in filter_dict:
        filter_dict[left.property] = new_filter
    else:
        filter_dict[left.property] = _merge_filter_predicates(
            existing_filter,
            new_filter,
            field=f"where.{left.alias}.{left.property}",
            line=left.span.line,
            column=left.span.column,
        )
    _set_target_filter_dict(target, filter_dict)


def peel_literal_membership_where(
    expr: "ExpressionText",
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],  # hygiene-ok: explicit-any -- Cypher params are heterogeneous JSON scalars
) -> Optional["ExpressionText"]:
    """Move each ``alias.prop IN [scalar literals]`` conjunct onto the pattern as an ``is_in``
    filter and return the WHERE that is left (None when nothing is). The pattern filter and the
    row WHERE agree on these: a row whose property is in the list is kept, every other row is
    dropped.
    """
    from graphistry.compute.gfql.cypher.ast import ExpressionText
    from graphistry.compute.gfql.cypher.row_pushdown import _flatten_and_conjuncts

    conjuncts = _flatten_and_conjuncts(expr.text)
    kept: List[str] = []
    for text in conjuncts:
        seed = _literal_membership_seed(text, span=expr.span, alias_targets=alias_targets, params=params)
        if seed is None:
            kept.append(text)
            continue
        left, values = seed
        _apply_membership_where(alias_targets, left=left, values=values)
    if len(kept) == len(conjuncts):
        return expr
    if not kept:
        return None
    return ExpressionText(text=" and ".join(f"({text})" for text in kept), span=expr.span)
