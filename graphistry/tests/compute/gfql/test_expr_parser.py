import pytest

from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    CaseWhen,
    FunctionCall,
    Identifier,
    Literal,
    ListComprehension,
    PropertyAccessExpr,
    QuantifierExpr,
    SubscriptExpr,
    Wildcard,
    collect_identifiers,
    find_unsupported_functions,
    parse_expr,
    validate_expr_capabilities,
)
from graphistry.compute.gfql.language_defs import GFQL_COMPARISON_BINARY_OP_NAMES
from graphistry.compute.gfql.string_literals import parse_cypher_string_token, render_cypher_string_literal


def _has_lark() -> bool:
    try:
        import lark  # noqa: F401
        return True
    except Exception:
        return False


requires_lark = pytest.mark.skipif(not _has_lark(), reason="lark dependency unavailable in local env")


def test_parse_expr_requires_lark_dependency() -> None:
    if _has_lark():
        node = parse_expr("score > 1")
        assert isinstance(node, BinaryOp)
        return

    with pytest.raises(ImportError):
        parse_expr("score > 1")


def test_validate_expr_capabilities_rejects_unknown_function() -> None:
    node = BinaryOp(
        op=">",
        left=FunctionCall(name="unknown_fn", args=(Identifier("score"),)),
        right=Literal(1),
    )
    errors = validate_expr_capabilities(node)
    assert "unsupported function: unknown_fn" in errors


def test_validate_expr_capabilities_rejects_unknown_operator() -> None:
    node = BinaryOp(op="pow", left=Identifier("a"), right=Identifier("b"))
    errors = validate_expr_capabilities(node)
    assert "unsupported binary op: pow" in errors


def test_validate_expr_capabilities_accepts_supported_tree() -> None:
    node = BinaryOp(
        op="and",
        left=BinaryOp(op=">", left=Identifier("score"), right=Literal(1)),
        right=FunctionCall(name="size", args=(Identifier("vals"),)),
    )
    errors = validate_expr_capabilities(node)
    assert errors == []


def test_validate_expr_capabilities_accepts_properties_on_identifier_map_and_null() -> None:
    assert validate_expr_capabilities(FunctionCall(name="properties", args=(Identifier("n"),))) == []
    assert validate_expr_capabilities(FunctionCall(name="properties", args=(Literal(None),))) == []

    map_node = parse_expr("{name: 'Popeye', level: 9001}") if _has_lark() else None
    if map_node is not None:
        assert validate_expr_capabilities(FunctionCall(name="properties", args=(map_node,))) == []


def test_validate_expr_capabilities_rejects_properties_on_scalar_literals() -> None:
    for bad in (Literal(1), Literal("Cypher"), parse_expr("[true, false]") if _has_lark() else Literal(True)):
        errors = validate_expr_capabilities(FunctionCall(name="properties", args=(bad,)))
        assert "properties() requires a node, relationship, map, or null argument" in errors


@pytest.mark.parametrize(
    "token,expected",
    [
        ("'plain'", "plain"),
        ('"double"', "double"),
        (r"'\''", "'"),
        (r'"\""', '"'),
        (r"'\u01FF'", "\u01ff"),
        (r"'\U000001FF'", "\u01ff"),
        (r"'a\\bcn5t'", "a\\\\bcn5t"),
        (r"'\n\t\r\b\f'", "\n\t\r\b\f"),
        (r"'\\'", "\\\\"),
        (r"'\"'", '"'),
        (r'"\'"', "'"),
    ],
)
def test_parse_cypher_string_token_accepts_unicode_quote_and_preserved_escapes(
    token: str,
    expected: str,
) -> None:
    assert parse_cypher_string_token(token) == expected


@pytest.mark.parametrize(
    "token",
    [
        "plain",
        "'unterminated",
        r"'trailing\'",
        r"'\q'",
        r"'\/'",
        r"'\uH'",
        r"'\U0001FF'",
        "123",
        "None",
    ],
)
def test_parse_cypher_string_token_rejects_invalid_literals(token: str) -> None:
    with pytest.raises(ValueError):
        parse_cypher_string_token(token)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "'",
        '"',
        "\\",
        "\\'",
        "\\\\'",
        "a\\b",
        "line\nbreak",
        "tab\tseparated",
        "carriage\rreturn",
        "backspace\bseparated",
        "formfeed\fseparated",
        "\u01ff",
    ],
)
def test_cypher_string_literal_renderer_round_trips(value: str) -> None:
    assert parse_cypher_string_token(render_cypher_string_literal(value)) == value


def test_cypher_string_literal_renderer_uses_unambiguous_backslash_and_quote_escapes() -> None:
    assert render_cypher_string_literal("\\'\n") == r"'\u005C\u0027\u000A'"


@requires_lark
@pytest.mark.parametrize(
    "value",
    [
        "\\",
        "\\'",
        "'",
        '"',
        "a\\b",
        "line\nbreak",
        "tab\tseparated",
        "\u01ff",
    ],
)
def test_parse_expr_round_trips_rendered_cypher_string_literals(value: str) -> None:
    assert parse_expr(render_cypher_string_literal(value)) == Literal(value)


@requires_lark
@pytest.mark.parametrize(
    "expr,expected",
    [
        (r"'\n\t\r\b\f'", "\n\t\r\b\f"),
        (r"'\\'", "\\\\"),
        (r"'\"'", '"'),
        (r'"\'"', "'"),
    ],
)
def test_parse_expr_handles_user_written_cypher_string_escapes(expr: str, expected: str) -> None:
    assert parse_expr(expr) == Literal(expected)


@requires_lark
@pytest.mark.parametrize("expr", ["'unterminated", r"'\uH'"])
def test_parse_expr_rejects_invalid_string_literals(expr: str) -> None:
    with pytest.raises(Exception):
        parse_expr(expr)


@requires_lark
def test_parse_expr_precedence_tree() -> None:
    node = parse_expr("NOT a = 1 AND b = 2 OR c = 3")
    assert isinstance(node, BinaryOp)
    assert node.op == "or"
    assert isinstance(node.left, BinaryOp)
    assert node.left.op == "and"


@requires_lark
def test_parse_expr_xor_precedence_tree() -> None:
    node = parse_expr("a OR b XOR c AND d")
    assert isinstance(node, BinaryOp)
    assert node.op == "or"
    assert isinstance(node.right, BinaryOp)
    assert node.right.op == "xor"
    assert isinstance(node.right.right, BinaryOp)
    assert node.right.right.op == "and"


@requires_lark
@pytest.mark.parametrize(
    "expr",
    [
        "score > 1",
        "__node_keys__(n, n, r)",
        "NOT (score > 1 AND score < 3)",
        "flag XOR other_flag",
        "(flag XOR other_flag) IS NULL = (other_flag XOR flag) IS NULL",
        "CASE WHEN score > 1 THEN true ELSE false END",
        "CASE score WHEN 1 THEN 'one' ELSE 'other' END",
        "any(x IN vals WHERE x = 2)",
        "[x IN vals WHERE x > 1 | x + 1]",
        "name CONTAINS 'a'",
        "meta['k'] = 'v'",
        "vals[1..3]",
        "date.truncate('year', date({year: 1984, month: 10, day: 11}), {})",
    ],
)
def test_parse_expr_accepts_supported_samples(expr: str) -> None:
    node = parse_expr(expr)
    assert node is not None


@requires_lark
@pytest.mark.parametrize("op", sorted(GFQL_COMPARISON_BINARY_OP_NAMES))
def test_parse_expr_accepts_shared_comparison_vocab(op: str) -> None:
    node = parse_expr(f"score {op} 1")
    assert isinstance(node, BinaryOp)
    assert node.op == op


@requires_lark
@pytest.mark.parametrize(
    "expr",
    [
        "score == 1",
        "id = 'a' -- comment",
        "any(x vals WHERE x = 2)",
        "any(x IN vals WHERE x = 2))",
        "size() > 0",
        "name = 'unterminated",
        "meta = {k: }",
    ],
)
def test_parse_expr_rejects_malformed_samples(expr: str) -> None:
    with pytest.raises(Exception):
        parse_expr(expr)


@requires_lark
def test_parse_expr_case_and_quantifier_nodes() -> None:
    case_node = parse_expr("CASE WHEN score > 1 THEN true ELSE false END")
    assert isinstance(case_node, CaseWhen)
    simple_case_node = parse_expr("CASE score WHEN 1 THEN true WHEN 2 THEN false ELSE null END")
    assert isinstance(simple_case_node, CaseWhen)
    assert isinstance(simple_case_node.condition, FunctionCall)
    assert simple_case_node.condition.name == "__cypher_case_eq__"
    no_else_case = parse_expr("CASE WHEN score > 1 THEN true END")
    assert isinstance(no_else_case, CaseWhen)
    assert isinstance(no_else_case.when_false, Literal)
    assert no_else_case.when_false.value is None

    quant = parse_expr("any(x IN vals WHERE x = 2)")
    assert isinstance(quant, QuantifierExpr)
    assert quant.fn == "any"
    assert quant.var == "x"


@requires_lark
def test_parse_expr_list_comprehension_node() -> None:
    node = parse_expr("[x IN vals WHERE x > 1 | x + 1]")
    assert isinstance(node, ListComprehension)
    assert node.var == "x"
    assert node.predicate is not None
    assert node.projection is not None


@requires_lark
def test_parse_expr_postfix_property_access_node() -> None:
    node = parse_expr("(list[1]).missing")
    assert isinstance(node, PropertyAccessExpr)
    assert node.property == "missing"
    assert isinstance(node.value, SubscriptExpr)


@requires_lark
def test_parse_expr_identifier_property_access_node() -> None:
    node = parse_expr("m.missing")
    assert isinstance(node, PropertyAccessExpr)
    assert node.property == "missing"
    assert isinstance(node.value, Identifier)
    assert node.value.name == "m"


@requires_lark
def test_collect_identifiers_excludes_bound_vars() -> None:
    node = parse_expr("any(x IN vals WHERE x > threshold)")
    names = collect_identifiers(node)
    assert "vals" in names
    assert "threshold" in names
    assert "x" not in names


@requires_lark
def test_find_unsupported_functions_reports_unknown() -> None:
    node = parse_expr("unknown_fn(score) > 1")
    bad = find_unsupported_functions(node)
    assert "unknown_fn" in bad


@requires_lark
def test_find_unsupported_functions_accepts_known() -> None:
    node = parse_expr("size(vals) > 1 AND toString(name) = 'x'")
    bad = find_unsupported_functions(node)
    assert bad == set()


@requires_lark
def test_parse_expr_aggregate_wildcard_function_node() -> None:
    node = parse_expr("count(*)")
    assert isinstance(node, FunctionCall)
    assert len(node.args) == 1
    assert isinstance(node.args[0], Wildcard)


@requires_lark
def test_parse_expr_distinct_function_node() -> None:
    node = parse_expr("count(DISTINCT score)")
    assert isinstance(node, FunctionCall)
    assert node.name == "count"
    assert node.distinct is True
    assert node.args == (Identifier("score"),)


@requires_lark
def test_parse_expr_dotted_function_node() -> None:
    node = parse_expr("date.truncate('year', date({year: 1984, month: 10, day: 11}), {})")
    assert isinstance(node, FunctionCall)
    assert node.name == "date.truncate"
    assert len(node.args) == 3


@requires_lark
def test_validate_expr_capabilities_rejects_wildcard_function_arg() -> None:
    node = parse_expr("count(*)")
    errors = validate_expr_capabilities(node)
    assert "unsupported function: count" in errors
    assert "unsupported wildcard: *" in errors


def test_parse_expr_memoizes_identical_expressions() -> None:
    from graphistry.compute.gfql import expr_parser as _ep

    expr = "a.val > 50 AND a.kind = 'x'"
    first = _ep.parse_expr(expr)
    second = _ep.parse_expr(expr)
    # Same object identity => cache hit (no re-parse / transformer rebuild).
    assert first is second


def test_parse_expr_cache_distinguishes_and_registers_hits() -> None:
    from graphistry.compute.gfql import expr_parser as _ep

    a = _ep.parse_expr("zzz_expr_probe.k + 1")
    b = _ep.parse_expr("zzz_expr_probe.k + 2")
    assert a is not b
    before = _ep._parse_expr_cached.cache_info().hits
    _ep.parse_expr("zzz_expr_probe.k + 1")
    after = _ep._parse_expr_cached.cache_info().hits
    assert after == before + 1


def test_parse_expr_does_not_cache_invalid_expressions() -> None:
    from graphistry.compute.gfql.expr_parser import GFQLExprParseError, parse_expr

    for _ in range(2):
        with pytest.raises(GFQLExprParseError):
            parse_expr("")
