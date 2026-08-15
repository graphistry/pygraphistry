"""Unary-operator surface pins (#1902 review round).

The GFQL grammar admits EXACTLY three unary operators -- `+`, `-`, `not`. `~` and `!`
are NOT in the language, so every dispatch over `UnaryOp.op` is total once it covers
those three; `UnaryOpName` makes that checkable instead of leaving an if/elif chain
over a bare `str` to fall through silently.

Every expected value is HAND-COMPUTED from openCypher (integer `/` truncates toward
zero; `%` is the truncated remainder carrying the sign of the DIVIDEND; an integer
zero divisor is an error). Cross-engine agreement is NOT used as an oracle -- pandas,
polars and cudf have been wrong together before on this stack.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLTypeError
from graphistry.compute.gfql.cypher.expression_text import render_expr_node
from graphistry.compute.gfql.expr_parser import (
    GFQLExprParseError,
    UnaryOp,
    parse_expr,
)
from graphistry.compute.gfql.language_defs import GFQL_ALLOWED_UNARY_OPS

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

ENGINES = [
    "pandas",
    pytest.param("polars", marks=pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")),
    pytest.param("cudf", marks=pytest.mark.skipif(not HAS_CUDF, reason="cudf not installed")),
]

# neg spans both signs so a floored `//` (-7 // 2 == -4) cannot pass as truncation (-3).
_NODES = pd.DataFrame({
    "id": ["a", "b", "c", "d"],
    "neg": [-7, -8, 7, 8],
    "flag": [True, False, True, False],
    "score": [1.5, -2.0, 0.0, 2.5],
})
_EDGES = pd.DataFrame({"src": ["a"], "dst": ["b"]})


def _graph(engine, nodes=None, edges=None):
    nodes = _NODES if nodes is None else nodes
    edges = _EDGES if edges is None else edges
    if engine == "polars":
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def _values(engine, expr, column="v", nodes=None):
    """Run `RETURN n.id AS id, <expr> AS v` and return v ordered by id."""
    out = _graph(engine, nodes=nodes).gfql(
        f"MATCH (n) RETURN n.id AS id, {expr} AS {column}", engine=engine
    )._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return [None if pd.isna(v) else v for v in out.sort_values("id")[column].tolist()]


# ---------------------------------------------------------------- the op set itself

def test_grammar_admits_exactly_plus_minus_not():
    """The authoritative unary-op set; a new operator must update it deliberately."""
    assert set(GFQL_ALLOWED_UNARY_OPS) == {"+", "-", "not"}


@pytest.mark.parametrize("expr", ["~x", "!x", "~ 1", "!true"])
def test_tilde_and_bang_are_not_in_the_language(expr):
    """`~`/`!` are UNREACHABLE: the grammar rejects them, so no evaluator can see them."""
    with pytest.raises(GFQLExprParseError):
        parse_expr(expr)


def test_unary_op_ast_only_ever_carries_an_admitted_op():
    for expr, expected in [("+x", "+"), ("-x", "-"), ("NOT x", "not"), ("not x", "not")]:
        node = parse_expr(expr)
        assert isinstance(node, UnaryOp)
        assert node.op == expected
        assert node.op in GFQL_ALLOWED_UNARY_OPS


def test_unknown_unary_op_raises_typed_error_naming_the_op():
    """A programmatically-built (grammar-impossible) op must raise a TYPED error naming
    it -- not fall through the if/elif chain into a generic 'AST evaluator unsupported'."""
    from graphistry.compute.gfql.row import pipeline as row_pipeline_mixin

    g = _graph("pandas")
    ctx = row_pipeline_mixin._RowPipelineAdapter(g)
    with pytest.raises(GFQLTypeError) as exc:
        ctx._gfql_eval_expr_ast(g._nodes, UnaryOp(op="~", operand=parse_expr("neg")))  # type: ignore[arg-type]
    assert "~" in str(exc.value)
    assert "unary" in str(exc.value).lower()


# ------------------------------------------------- expression_text round-trip (`--`)

@pytest.mark.parametrize("expr", ["- -2", "- - -2", "-(-2)", "- -x", "+ +2", "-x", "+x", "NOT x"])
def test_rendered_unary_reparses(expr):
    """render_expr_node output must be re-parseable: the parser folds a signed literal
    into the Literal, so `- -2` renders via a naive concat as the unparseable `(--2)`."""
    rendered = render_expr_node(parse_expr(expr))
    parse_expr(rendered)  # must not raise


def test_nested_unary_minus_renders_without_fusing_signs():
    assert render_expr_node(parse_expr("- -2")) == "(- -2)"


# ------------------------------------------------------------ end-to-end, per engine

@pytest.mark.parametrize("engine", ENGINES)
def test_unary_plus_is_identity(engine):
    """+n.neg leaves values untouched (openCypher unary plus)."""
    assert _values(engine, "+n.neg") == [-7, -8, 7, 8]


@pytest.mark.parametrize("engine", ENGINES)
def test_unary_minus_negates(engine):
    assert _values(engine, "-n.neg") == [7, 8, -7, -8]


@pytest.mark.parametrize("engine", ENGINES)
def test_int_division_truncates_toward_zero(engine):
    """-7/2 = -3 (truncate), NOT -4 (floor). Sign pairs cover all four quadrants."""
    assert _values(engine, "n.neg / 2") == [-3, -4, 3, 4]
    assert _values(engine, "n.neg / -2") == [3, 4, -3, -4]


@pytest.mark.parametrize("engine", ENGINES)
def test_int_modulo_is_truncated_remainder(engine):
    """Sign of the DIVIDEND (Java/openCypher): -7 % 3 = -1, not Python's floored 2."""
    assert _values(engine, "n.neg % 3") == [-1, -2, 1, 2]
    assert _values(engine, "n.neg % -3") == [-1, -2, 1, 2]


@pytest.mark.parametrize("engine", ENGINES)
def test_nested_unary_minus_divisor_serves_truncated_division(engine):
    """`- -2` is the literal 2 after double negation -> same answer as `/ 2`."""
    assert _values(engine, "n.neg / - -2") == [-3, -4, 3, 4]


@pytest.mark.parametrize("engine", ENGINES)
def test_unary_minus_zero_divisor_still_errors(engine):
    """`-(0)` is still an integer zero divisor -- the unary wrapper must not smuggle it
    past the gate into a silent polars `// 0` null."""
    for divisor in ["0", "-(0)", "+(0)"]:
        with pytest.raises((GFQLTypeError, NotImplementedError)) as exc:
            _values(engine, f"n.neg / {divisor}")
        if isinstance(exc.value, GFQLTypeError):
            assert "zero" in str(exc.value).lower()


# --------------------------------------------------------- bool-vs-number x-engine

@pytest.mark.parametrize("engine", ENGINES)
def test_bool_column_ordered_against_number_never_matches(engine):
    """openCypher: ordering a BOOLEAN against a NUMBER is incomparable -> null -> the row
    drops. Pinned on every engine (the polars filter_by_dict lowering has its own guard)."""
    out = _graph(engine).gfql("MATCH (n) WHERE n.flag > 0 RETURN n.id AS id", engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    assert len(out) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_bool_column_equality_against_bool_still_served(engine):
    """Equality is NOT ordering -- the incomparability guard must not swallow it."""
    out = _graph(engine).gfql("MATCH (n) WHERE n.flag = true RETURN n.id AS id", engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    assert sorted(out["id"].tolist()) == ["a", "c"]


# ------------------------------------------------------------------- row multiplicity

@pytest.mark.parametrize("engine", ENGINES)
def test_duplicate_rows_each_get_their_own_result(engine):
    """Row multiplicity: 3 identical rows -> 3 identical results, not a collapsed 1."""
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "neg": [-7, -7, -7], "flag": [True] * 3,
                          "score": [1.0] * 3})
    assert _values(engine, "n.neg / 2", nodes=nodes) == [-3, -3, -3]


@pytest.mark.parametrize("engine", ENGINES)
def test_single_row_frame(engine):
    nodes = pd.DataFrame({"id": ["a"], "neg": [-7], "flag": [True], "score": [1.0]})
    assert _values(engine, "n.neg / 2", nodes=nodes) == [-3]


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_frame_yields_no_rows_not_an_error(engine):
    nodes = pd.DataFrame({"id": pd.Series([], dtype="object"), "neg": pd.Series([], dtype="int64"),
                          "flag": pd.Series([], dtype="bool"), "score": pd.Series([], dtype="float64")})
    edges = pd.DataFrame({"src": pd.Series([], dtype="object"), "dst": pd.Series([], dtype="object")})
    out = _graph(engine, nodes=nodes, edges=edges).gfql(
        "MATCH (n) RETURN n.id AS id, n.neg / 2 AS v", engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    assert len(out) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_null_cell_propagates_through_truncated_division(engine):
    """null / 2 is null (Cypher 3VL) -- and must not become 0 via a fillna in the
    truncation path."""
    nodes = pd.DataFrame({"id": ["a", "b"], "neg": pd.Series([-7, None], dtype="Int64"),
                          "flag": [True, False], "score": [1.0, 2.0]})
    assert _values(engine, "n.neg / 2", nodes=nodes) == [-3, None]
