_EXPR_PARSER_MODULE = "graphistry.compute.gfql.expr_parser"
_ORDER_AGG_ALIAS_FUNCS = frozenset({"count", "sum", "min", "max", "avg", "mean", "collect"})
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


def is_order_aggregate_alias_ast(node: object) -> bool:
    node_type = type(node)
    if node_type.__name__ != "FunctionCall" or node_type.__module__ != _EXPR_PARSER_MODULE:
        return False

    fn = getattr(node, "name", None)
    args = getattr(node, "args", None)
    if not isinstance(fn, str) or fn.lower() not in _ORDER_AGG_ALIAS_FUNCS:
        return False
    if not isinstance(args, tuple) or len(args) != 1:
        return False

    arg = args[0]
    arg_type = type(arg)
    if arg_type.__module__ != _EXPR_PARSER_MODULE:
        return False
    if arg_type.__name__ == "Wildcard":
        return True
    if arg_type.__name__ != "Identifier":
        return False

    name = getattr(arg, "name", None)
    return isinstance(name, str) and name != ""


def order_expr_ast_static_supported(node: object) -> bool:
    node_type = type(node)
    if node_type.__name__ in _ORDER_UNSUPPORTED_NODE_NAMES:
        return False
    if node_type.__module__ != _EXPR_PARSER_MODULE:
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
