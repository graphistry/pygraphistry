from __future__ import annotations

import re
from typing import Any, List

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    CaseWhen,
    ExprNode,
    FunctionCall,
    Identifier,
    IsNullOp,
    ListComprehension,
    ListLiteral,
    Literal as ExprLiteral,
    MapLiteral,
    PropertyAccessExpr,
    QuantifierExpr,
    SliceExpr,
    SubscriptExpr,
    UnaryOp,
    Wildcard,
)
from graphistry.compute.gfql.string_literals import render_cypher_string_literal


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def cypher_literal_expr_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "null"
        return repr(value)
    if isinstance(value, str):
        return render_cypher_string_literal(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(cypher_literal_expr_text(item) for item in value) + "]"
    if isinstance(value, dict):
        parts: List[str] = []
        for key, item in value.items():
            key_txt = str(key)
            rendered_key = key_txt if _IDENTIFIER_RE.fullmatch(key_txt) else render_cypher_string_literal(key_txt)
            parts.append(f"{rendered_key}: {cypher_literal_expr_text(item)}")
        return "{" + ", ".join(parts) + "}"
    raise GFQLValidationError(
        ErrorCode.E108,
        "Cypher parameter value is outside the currently supported literal subset",
        field="params",
        value=type(value).__name__,
        suggestion="Use null, booleans, numbers, strings, lists, or maps as parameter values.",
        language="cypher",
    )


def render_expr_node(node: ExprNode) -> str:
    if isinstance(node, Identifier):
        return node.name
    if isinstance(node, ExprLiteral):
        return cypher_literal_expr_text(node.value)
    if isinstance(node, UnaryOp):
        operand = render_expr_node(node.operand)
        if node.op == "not":
            return f"(NOT {operand})"
        return f"({node.op}{operand})"
    if isinstance(node, BinaryOp):
        left = render_expr_node(node.left)
        right = render_expr_node(node.right)
        if node.op in {"and", "or", "xor", "in"}:
            op_txt = node.op.upper()
        elif node.op == "starts_with":
            op_txt = "STARTS WITH"
        elif node.op == "ends_with":
            op_txt = "ENDS WITH"
        elif node.op == "contains":
            op_txt = "CONTAINS"
        elif node.op == "regex":
            op_txt = "=~"
        else:
            op_txt = node.op
        return f"({left} {op_txt} {right})"
    if isinstance(node, IsNullOp):
        suffix = "IS NOT NULL" if node.negated else "IS NULL"
        return f"({render_expr_node(node.value)} {suffix})"
    if isinstance(node, FunctionCall):
        args = ", ".join(render_expr_node(arg) for arg in node.args)
        if node.distinct:
            args = f"DISTINCT {args}"
        return f"{node.name}({args})"
    if isinstance(node, Wildcard):
        return "*"
    if isinstance(node, CaseWhen):
        return (
            "CASE WHEN "
            f"{render_expr_node(node.condition)} THEN {render_expr_node(node.when_true)} "
            f"ELSE {render_expr_node(node.when_false)} END"
        )
    if isinstance(node, QuantifierExpr):
        return (
            f"{node.fn.upper()}({node.var} IN {render_expr_node(node.source)} "
            f"WHERE {render_expr_node(node.predicate)})"
        )
    if isinstance(node, ListComprehension):
        rendered = f"[{node.var} IN {render_expr_node(node.source)}"
        if node.predicate is not None:
            rendered += f" WHERE {render_expr_node(node.predicate)}"
        if node.projection is not None:
            rendered += f" | {render_expr_node(node.projection)}"
        return rendered + "]"
    if isinstance(node, ListLiteral):
        return "[" + ", ".join(render_expr_node(item) for item in node.items) + "]"
    if isinstance(node, MapLiteral):
        parts: List[str] = []
        for key, value in node.items:
            rendered_key = key if _IDENTIFIER_RE.fullmatch(key) else cypher_literal_expr_text(key)
            parts.append(f"{rendered_key}: {render_expr_node(value)}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(node, SubscriptExpr):
        return f"{render_expr_node(node.value)}[{render_expr_node(node.key)}]"
    if isinstance(node, SliceExpr):
        start = "" if node.start is None else render_expr_node(node.start)
        stop = "" if node.stop is None else render_expr_node(node.stop)
        return f"{render_expr_node(node.value)}[{start}..{stop}]"
    if isinstance(node, PropertyAccessExpr):
        return f"{render_expr_node(node.value)}.{node.property}"
    raise TypeError(f"Unsupported expression node type for rendering: {type(node).__name__}")
