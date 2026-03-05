import pytest

from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    CaseWhen,
    FunctionCall,
    Identifier,
    Literal,
    ListComprehension,
    QuantifierExpr,
    collect_identifiers,
    find_unsupported_functions,
    parse_expr,
    validate_expr_capabilities,
)


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
    node = BinaryOp(op="xor", left=Identifier("a"), right=Identifier("b"))
    errors = validate_expr_capabilities(node)
    assert "unsupported binary op: xor" in errors


def test_validate_expr_capabilities_accepts_supported_tree() -> None:
    node = BinaryOp(
        op="and",
        left=BinaryOp(op=">", left=Identifier("score"), right=Literal(1)),
        right=FunctionCall(name="size", args=(Identifier("vals"),)),
    )
    errors = validate_expr_capabilities(node)
    assert errors == []


@requires_lark
def test_parse_expr_precedence_tree() -> None:
    node = parse_expr("NOT a = 1 AND b = 2 OR c = 3")
    assert isinstance(node, BinaryOp)
    assert node.op == "or"
    assert isinstance(node.left, BinaryOp)
    assert node.left.op == "and"


@requires_lark
@pytest.mark.parametrize(
    "expr",
    [
        "score > 1",
        "NOT (score > 1 AND score < 3)",
        "CASE WHEN score > 1 THEN true ELSE false END",
        "any(x IN vals WHERE x = 2)",
        "[x IN vals WHERE x > 1 | x + 1]",
        "name CONTAINS 'a'",
        "meta['k'] = 'v'",
        "vals[1..3]",
    ],
)
def test_parse_expr_accepts_supported_samples(expr: str) -> None:
    node = parse_expr(expr)
    assert node is not None


@requires_lark
@pytest.mark.parametrize(
    "expr",
    [
        "score == 1",
        "id = 'a' -- comment",
        "CASE WHEN score > 1 THEN true END",
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
