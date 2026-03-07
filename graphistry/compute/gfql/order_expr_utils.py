import re


_ORDER_AGG_ALIAS_RE = re.compile(
    r"(?is)\s*(?:count|sum|min|max|avg|mean|collect)\s*\(\s*(?:\*|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*\)\s*"
)
_ORDER_LABEL_DISALLOWED_TOKENS = (
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "+",
    "-",
    "*",
    "/",
    "%",
    "<",
    ">",
    "=",
    "!",
)
_ORDER_UNSUPPORTED_NODE_NAMES = frozenset(
    {
        "QuantifierExpr",
        "ListComprehension",
        "ListLiteral",
        "MapLiteral",
        "SubscriptExpr",
        "SliceExpr",
    }
)


def is_aggregate_alias_expr(text: str) -> bool:
    return _ORDER_AGG_ALIAS_RE.fullmatch(text) is not None


def is_plain_order_label(text: str) -> bool:
    txt = text.strip()
    if txt == "":
        return False
    for token in _ORDER_LABEL_DISALLOWED_TOKENS:
        if token in txt:
            return False
    return True


def order_expr_ast_static_supported(node: object) -> bool:
    node_type = type(node)
    if node_type.__name__ in _ORDER_UNSUPPORTED_NODE_NAMES:
        return False
    if node_type.__module__ != "graphistry.compute.gfql.expr_parser":
        return True
    try:
        fields = vars(node).values()
    except Exception:
        return True
    for value in fields:
        if isinstance(value, (list, tuple)):
            for item in value:
                if not order_expr_ast_static_supported(item):
                    return False
            continue
        if isinstance(value, dict):
            for item in value.values():
                if not order_expr_ast_static_supported(item):
                    return False
            continue
        if not order_expr_ast_static_supported(value):
            return False
    return True
