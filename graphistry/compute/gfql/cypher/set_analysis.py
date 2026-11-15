"""Static, fail-closed analysis of SET assignments (parameter substitution, value checks, read-after-write).

Every SET right-hand side is evaluated once, on the graph before any SET, so the compiler must prove that a value
is a scalar computed only from the matched node's properties. Anything it cannot analyze is declined, never skipped.
The compiler-private helpers are reached lazily because ``lowering`` imports this module.
"""
from __future__ import annotations

import math
from typing import Mapping, MutableSet, Optional

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.ast import CypherParams, ExpressionText, SetProperty, SourceSpan
from graphistry.compute.gfql.cypher.expression_text import cypher_literal_expr_text
from graphistry.compute.gfql.expr_parser import (
    CaseWhen,
    ExprNode,
    FunctionCall,
    Identifier,
    ListComprehension,
    ListLiteral,
    MapLiteral,
    PropertyAccessExpr,
    QuantifierExpr,
    iter_expr_children,
)


def _decline(message: str, value: str, span: SourceSpan) -> GFQLValidationError:
    from graphistry.compute.gfql.cypher.lowering import _unsupported_at_span

    return _unsupported_at_span(message, field="set", value=value, span=span)


def _non_finite_kind(value: object) -> Optional[str]:
    """``nan_property`` / ``non_finite_parameter`` for a NaN or infinite float anywhere inside a parameter value."""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan_property"
        return "non_finite_parameter" if math.isinf(value) else None
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple, set)):
        for item in value:
            kind = _non_finite_kind(item)
            if kind is not None:
                return kind
    return None


def _is_negative_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return math.copysign(1.0, value) < 0
    return isinstance(value, int) and value < 0


def prepare_param_text(text: str, params: Optional[CypherParams], span: SourceSpan) -> str:
    """Quote-aware pre-pass: decline NaN/inf parameters (also inside lists/maps), parenthesize negative numbers.

    Parenthesizing keeps ``n.a -$v`` with ``v = -5`` from rendering as ``n.a --5``.
    """
    from graphistry.compute.gfql.cypher.lowering import _CYPHER_PARAM_TOKEN_RE

    out: list[str] = []
    idx = 0
    in_single = in_double = False
    while idx < len(text):
        ch = text[idx]
        if ch == "'" and not in_double and (idx == 0 or text[idx - 1] != "\\"):
            in_single = not in_single
        elif ch == '"' and not in_single and (idx == 0 or text[idx - 1] != "\\"):
            in_double = not in_double
        elif ch == "$" and not in_single and not in_double:
            token = _CYPHER_PARAM_TOKEN_RE.match(text, idx)
            if token is not None:
                supplied = (params or {}).get(token.group(1))
                kind = _non_finite_kind(supplied)
                if kind is not None:
                    raise _decline("GFQL SET cannot store a NaN or infinite parameter value", kind, span)
                if _is_negative_number(supplied):
                    out.append(f"({cypher_literal_expr_text(supplied)})")
                    idx = token.end()
                    continue
        out.append(ch)
        idx += 1
    return "".join(out)


def alias_uses(
    expression: ExprNode, aliases: Mapping[str, object]
) -> tuple[set[tuple[str, str]], bool, set[str], set[str]]:
    """Property reads of the SET aliases, whole-entity use, unbound names, and aliases shadowed by a local variable.

    The alias is allowed only as ``alias.property``; a comprehension or quantifier variable is local to its scope;
    any other identifier is unbound.
    """
    reads: set[tuple[str, str]] = set()
    unbound: set[str] = set()
    shadowed: set[str] = set()
    whole_entity = False

    def visit(node: ExprNode, bound: frozenset[str]) -> None:
        nonlocal whole_entity
        if isinstance(node, PropertyAccessExpr) and isinstance(node.value, Identifier) and node.value.name in aliases:
            reads.add((node.value.name, node.property))
            return
        if isinstance(node, Identifier):
            root, dot, remainder = node.name.partition(".")
            if root in bound:
                return
            if dot and root in aliases:
                reads.add((root, remainder.split(".")[0]))
            elif node.name in aliases:
                whole_entity = True
            else:
                unbound.add(node.name)
            return
        if isinstance(node, (ListComprehension, QuantifierExpr)):
            if node.var in aliases:
                shadowed.add(node.var)
            visit(node.source, bound)
            inner = bound | {node.var}
            for child in (node.predicate, getattr(node, "projection", None)):
                if child is not None:
                    visit(child, inner)
            return
        for child in iter_expr_children(node):
            visit(child, bound)

    visit(expression, frozenset())
    return reads, whole_entity, unbound, shadowed


_MAP_RETURNING_FUNCTIONS = frozenset({"properties"})


def _may_return_map(node: ExprNode) -> bool:
    """Whether the value can be a map (the read path would render it as text), through CASE branches and coalesce."""
    if isinstance(node, FunctionCall):
        name = node.name.lower()
        if name in _MAP_RETURNING_FUNCTIONS:
            return True
        return name in {"coalesce", "nullif"} and any(_may_return_map(arg) for arg in node.args)
    if isinstance(node, CaseWhen):
        return _may_return_map(node.when_true) or _may_return_map(node.when_false)
    return False


def analyze_set_assignment(
    assignment: SetProperty,
    *,
    params: Optional[CypherParams],
    aliases: Mapping[str, object],
    written: MutableSet[tuple[str, str]],
) -> str:
    """Declines unsafe SET values; returns the parameter-substituted value text and records the written property."""
    from graphistry.compute.gfql.cypher import lowering

    value = assignment.value
    if isinstance(value, ExpressionText):
        text = value.text
    elif value is None:
        text = "null"
    else:
        text = cypher_literal_expr_text(value)
    span = assignment.span
    text = prepare_param_text(text, params, span)
    prepared = lowering._rewrite_param_expr(
        text, params=params, field="set", line=span.line, column=span.column, allow_missing=False,
    )
    try:
        expression = lowering._parse_row_expr(prepared, field="set", line=span.line, column=span.column)
    except GFQLValidationError as exc:
        raise _decline("GFQL SET cannot analyze this value expression", "unanalyzable_value", span) from exc
    if isinstance(expression, (ListLiteral, ListComprehension, MapLiteral)) or _may_return_map(expression):
        raise _decline("GFQL SET supports ordinary scalar properties only", "non_scalar_value", span)
    if lowering._contains_aggregate_call(expression):
        raise _decline("GFQL SET values cannot use aggregation", "aggregate_value", span)
    reads, whole_entity, unbound, shadowed = alias_uses(expression, aliases)
    if shadowed:
        raise _decline(
            "GFQL SET values cannot bind a local variable with the name of the matched node", "shadowed_alias", span,
        )
    if unbound:
        raise _decline(
            "GFQL SET values may use only the matched node's properties and locally bound names",
            "unbound_variable", span,
        )
    if whole_entity:
        raise _decline(
            "GFQL SET may read the matched entity only through direct property access (alias.property)",
            "whole_entity_read", span,
        )
    if reads & set(written):
        raise _decline(
            "GFQL SET evaluates every right-hand side on the graph before the SET; "
            "it cannot read a property assigned earlier in the same query",
            "read_after_write", span,
        )
    target = (assignment.target.alias, assignment.target.property)
    if target in written:
        raise GFQLValidationError(
            ErrorCode.E105,
            f"GFQL SET assigns {target[0]}.{target[1]} more than once",
            field="set", value="repeated_property",
            suggestion="Assign each property at most once per query.",
            line=span.line, column=span.column, language="cypher",
        )
    written.add(target)
    return prepared
