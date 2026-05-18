from __future__ import annotations

from typing import Optional

from graphistry.compute.gfql.expr_parser import ExprNode, ListLiteral, Literal, MapLiteral

def _render_temporal_arg(node: ExprNode) -> Optional[str]:
    if isinstance(node, Literal):
        value = node.value
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace("'", "\\'")
            return f"'{escaped}'"
        return None
    if isinstance(node, MapLiteral):
        parts: list[str] = []
        for key, value in node.items:
            rendered = _render_temporal_arg(value)
            if rendered is None:
                return None
            parts.append(f"{key}: {rendered}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(node, ListLiteral):
        rendered_items: list[str] = []
        for item in node.items:
            rendered = _render_temporal_arg(item)
            if rendered is None:
                return None
            rendered_items.append(rendered)
        return "[" + ", ".join(rendered_items) + "]"
    return None
