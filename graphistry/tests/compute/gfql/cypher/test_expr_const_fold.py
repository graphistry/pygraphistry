"""Plan-time constant folding of GFQL row expressions (``expr_const_fold``).

Engine-free: everything here is AST-in / AST-out or text-in / text-out.  The
cross-engine value parity that the folding CRITERION rests on lives in
``graphistry/tests/compute/gfql/test_const_fold_engine_parity.py`` (pandas / polars /
cuDF / polars-gpu), because that is the only place the divergence in #1802 is visible.

THE POINT OF THE PASS: collapse equivalent spellings of the same predicate into ONE
canonical text, so downstream matchers learn one shape and every other foldable
function comes along for free.  THE TRAP: for exactly the functions this targets
(``toLower``/``toUpper``) the engines do NOT agree outside ASCII — pandas>=3's
Arrow-backed ``str`` uses SIMPLE case mappings where polars/Python use FULL ones
(#1802) — so folding outside the provable region silently changes answers.  The
non-ASCII decline tests below are the load-bearing ones.
"""
from typing import Mapping, Optional

import pytest

from graphistry.compute.gfql.expr_const_fold import (
    FOLDABLE_FUNCTIONS,
    NON_FOLDABLE_REASONS,
    ConstantFolder,
    FoldedValue,
    LiteralArgs,
    fold_constants,
)
from graphistry.compute.gfql.expr_parser import (
    FunctionCall,
    Identifier,
    Literal,
    PropertyAccessExpr,
    parse_expr,
)
from graphistry.compute.gfql.cypher.ast import ExpressionText, SourceSpan
from graphistry.compute.gfql.cypher.expression_text import render_expr_node
from graphistry.compute.gfql.cypher.lowering import _row_expr_arg
from graphistry.compute.gfql.language_defs import (
    GFQL_AGGREGATION_FUNCTIONS,
    GFQL_ALLOWED_FUNCTIONS,
)


def _fold_text(text: str) -> str:
    """Parse -> fold -> render.  The exact transform the lowering applies."""
    return render_expr_node(fold_constants(parse_expr(text)))


def _lowered_text(text: str) -> str:
    """The predicate text the lowering actually serializes into ``where_rows``."""
    span = SourceSpan(1, 0, 1, len(text), 0, len(text))
    out = _row_expr_arg(ExpressionText(text=text, span=span), params=None, field="where")
    assert isinstance(out, str)
    return out


# --------------------------------------------------------------------------------
# The two spellings collapse into one -- the whole reason the pass exists
# --------------------------------------------------------------------------------

class TestCanonicalization:
    @pytest.mark.parametrize("fn,rendered", [("toLower", "tolower"), ("lower", "lower")])
    @pytest.mark.parametrize("lit", ["fine dining", "Fine Dining", "FINE DINING", "MALE", "Male", "male"])
    def test_two_sided_collapses_into_one_sided(self, fn, rendered, lit):
        """`f(a.c) = f('LIT')` and `f(a.c) = 'lit'` become the SAME text."""
        two_sided = _lowered_text(f"{fn}(a.c) = {fn}('{lit}')")
        one_sided = _lowered_text(f"{fn}(a.c) = '{lit.lower()}'")
        assert two_sided == one_sided == f"({rendered}(a.c) = '{lit.lower()}')"

    @pytest.mark.parametrize("fn,expected", [
        ("toUpper", "(toupper(a.c) = 'MALE')"),
        ("upper", "(upper(a.c) = 'MALE')"),
        ("toLower", "(tolower(a.c) = 'male')"),
        ("lower", "(lower(a.c) = 'male')"),
    ])
    def test_canonical_texts(self, fn, expected):
        assert _lowered_text(f"{fn}(a.c) = {fn}('Male')") == expected

    def test_one_sided_literal_is_left_alone(self):
        """A bare literal is NOT case-folded: `toLower(x) = 'MALE'` keeps 'MALE',
        because the where_rows evaluator compares it as written."""
        assert _lowered_text("toLower(a.c) = 'MALE'") == "(tolower(a.c) = 'MALE')"

    def test_column_side_is_never_folded(self):
        assert _lowered_text("toLower(a.c) = toLower(b.d)") == "(tolower(a.c) = tolower(b.d))"

    @pytest.mark.parametrize("text,expected", [
        ("size('abcd') / 2", "(4 / 2)"),
        ("size('abcdef') / 4", "(6 / 4)"),
        ("size('abcd') / size('ab')", "(4 / 2)"),
        ("n.a = size('abcd') / 2", "(n.a = (4 / 2))"),
    ])
    def test_folding_runs_AFTER_the_integer_division_rewrite(self, text, expected):
        """PASS ORDER IS LOAD-BEARING, and this is not obvious.

        `_rewrite_cypher_integer_division_ast` wraps a division in `toInteger(...)`
        only when BOTH operands are already integer literals.  If folding ran first,
        `size('abcd') / 2` would become `4 / 2` in time for that rewrite to fire and
        the expression would start truncating -- a DIFFERENT ANSWER from master, from
        a pass whose whole contract is to preserve answers.  Folding therefore runs
        after, and this pins it.

        (Separately and pre-existing: openCypher says `size(s) / 2` IS integer
        division, because `size` returns an Integer.  GFQL's rewrite only recognizes
        literal operands, so it does not truncate here -- on master or on this branch.
        Fixing that is a real change to division semantics and does not belong in a
        constant-folding PR.)"""
        assert _lowered_text(text) == expected

    def test_projection_output_name_is_unaffected(self):
        """The RETURN column name comes from the SOURCE text, not the folded text.
        Folding must not rename a user's output column."""
        from graphistry.compute.gfql.cypher.parser import parse_cypher
        query = parse_cypher("MATCH (a) RETURN toLower('AB')")
        assert query.return_.items[0].expression.text == "toLower('AB')"


# --------------------------------------------------------------------------------
# Criterion (E): the ASCII region, and what happens outside it
# --------------------------------------------------------------------------------

# Non-ASCII strings whose case mapping is where implementations diverge:
#   'straße'  -> German sharp s; FULL upper is 'STRASSE', SIMPLE upper is 'STRAßE'
#   'İSTANBUL'-> Turkish dotted capital I; FULL lower adds U+0307
#   'ı'       -> Turkish dotless i; upper is 'I'
#   'CAFÉ'    -> plain Latin-1 accent (agrees in practice, still declined: the
#                criterion is a provable region, not a survey of what happens to work)
NON_ASCII_LITERALS = ["straße", "STRASSE ß", "İSTANBUL", "ıstanbul", "CAFÉ", "ΣΊΣΥΦΟΣ"]


class TestAsciiGate:
    @pytest.mark.parametrize("lit", ["male", "MALE", "Male", "mAlE", "", "fine dining", "a1_-.!"])
    def test_ascii_string_literals_fold(self, lit):
        assert _fold_text(f"toLower('{lit}')") == render_expr_node(Literal(lit.lower()))
        assert _fold_text(f"toUpper('{lit}')") == render_expr_node(Literal(lit.upper()))

    @pytest.mark.parametrize("lit", NON_ASCII_LITERALS)
    @pytest.mark.parametrize("fn", ["toLower", "toUpper", "lower", "upper", "size"])
    def test_non_ascii_declines_unchanged(self, lit, fn):
        """DECLINE, not a guess.  #1802 is live on master: for these inputs pandas>=3
        and polars genuinely disagree, so there is no engine-invariant value to fold to.
        The node must come back byte-identical."""
        text = f"{fn}('{lit}')"
        assert _fold_text(text) == render_expr_node(parse_expr(text))

    @pytest.mark.parametrize("lit", NON_ASCII_LITERALS)
    def test_non_ascii_two_sided_predicate_stays_two_sided(self, lit):
        assert _lowered_text(f"toLower(a.c) = toLower('{lit}')") \
            == f"(tolower(a.c) = tolower('{lit}'))"

    def test_ascii_arg_with_non_ascii_result_is_impossible_but_gated(self):
        """ASCII in => ASCII out for case mapping (the invariant that makes (E) provable).
        Pinned so a future folder that breaks it is caught."""
        for lit in ["male", "MALE", "Male", "i", "I"]:
            assert lit.lower().isascii() and lit.upper().isascii()


# --------------------------------------------------------------------------------
# Criterion (A): argument-closed, and nesting
# --------------------------------------------------------------------------------

class TestArgumentClosure:
    def test_column_argument_declines(self):
        assert _fold_text("toLower(a.c)") == "tolower(a.c)"

    def test_list_literal_argument_declines(self):
        assert _fold_text("size(['a', 'b'])") == "size(['a', 'b'])"

    def test_nested_fold_bottom_up(self):
        """`toLower(substring('ABCDEF', 0, 3))` -> `toLower('ABC')` -> `'abc'`."""
        assert _fold_text("toLower(substring('ABCDEF', 0, 3))") == "'abc'"

    def test_nested_fold_inside_a_predicate(self):
        assert _lowered_text("a.c = toUpper(substring('abcdef', 1, 2))") == "(a.c = 'BC')"

    def test_partial_nesting_declines_at_the_outer_level_only(self):
        """Inner folds, outer cannot (non-ASCII result region) -> inner substitution
        is still visible and the outer call is untouched."""
        assert _fold_text("toLower(size('abc'))") == "tolower(3)"


# --------------------------------------------------------------------------------
# NULL POLICY + booleans
# --------------------------------------------------------------------------------

class TestNullAndBooleanPolicy:
    @pytest.mark.parametrize("text", [
        "toLower(null)", "toUpper(null)", "size(null)",
        "substring(null, 0, 1)", "substring('abc', null)", "substring('abc', 0, null)",
    ])
    def test_null_arguments_never_fold(self, text):
        assert _fold_text(text) == render_expr_node(parse_expr(text))

    def test_pass_never_synthesizes_null(self):
        for folder in FOLDABLE_FUNCTIONS.values():
            for args in [(None,), (None, 0), ("abc", None), ()]:
                assert folder(args) is None

    @pytest.mark.parametrize("text", [
        "toLower(true)", "size(false)", "substring('abc', true)", "substring('abc', 0, true)",
    ])
    def test_boolean_arguments_never_fold(self, text):
        """`isinstance(True, int)` is True in Python; Cypher does not coerce."""
        assert _fold_text(text) == render_expr_node(parse_expr(text))

    @pytest.mark.parametrize("text", ["toLower(5)", "toUpper(1.5)", "size(5)"])
    def test_non_string_arguments_decline(self, text):
        assert _fold_text(text) == render_expr_node(parse_expr(text))


# --------------------------------------------------------------------------------
# Criterion (T): totality, out-of-range, and folders that raise
# --------------------------------------------------------------------------------

class TestTotality:
    @pytest.mark.parametrize("text,expected", [
        ("substring('abcdef', 0)", "'abcdef'"),
        ("substring('abcdef', 6)", "''"),
        ("substring('abcdef', 2, 3)", "'cde'"),
        ("substring('abcdef', 0, 0)", "''"),
        ("substring('abcdef', 6, 0)", "''"),
    ])
    def test_in_range_substring_folds(self, text, expected):
        assert _fold_text(text) == expected

    @pytest.mark.parametrize("text", [
        "substring('abc', 99)",       # start past the end: Python '', neo4j raises, polars clamps
        "substring('abc', 1, 99)",    # length past the end
        "substring('abc', 4)",
        "substring('abc', -1)",       # negative parses as UnaryOp -> not argument-closed
        "substring('abc', 0, -1)",
        "substring('abc')",           # wrong arity
        "substring('abc', 0, 1, 2)",
    ])
    def test_out_of_range_or_bad_arity_declines(self, text):
        assert _fold_text(text) == render_expr_node(parse_expr(text))

    def test_a_folder_that_raises_is_a_decline_not_a_crash(self):
        def _boom(args: LiteralArgs) -> Optional[FoldedValue]:
            raise RuntimeError("folder blew up")

        registry: Mapping[str, ConstantFolder] = {"tolower": _boom}
        node = parse_expr("toLower('MALE')")
        assert render_expr_node(fold_constants(node, registry=registry)) == "tolower('MALE')"

    @pytest.mark.parametrize("bad", [1.5, True, b"x", ("a",)])
    def test_out_of_contract_folder_return_is_ignored(self, bad):
        """A folder returning something outside ``FoldedValue`` is a bug in the folder,
        not a licence to rewrite the plan."""
        def _bad(args: LiteralArgs) -> Optional[FoldedValue]:
            return bad  # type: ignore[return-value]  # deliberately out of contract

        registry: Mapping[str, ConstantFolder] = {"tolower": _bad}
        node = parse_expr("toLower('MALE')")
        assert render_expr_node(fold_constants(node, registry=registry)) == "tolower('MALE')"

    def test_unregistered_function_is_untouched(self):
        registry: Mapping[str, ConstantFolder] = {}
        node = parse_expr("toLower('MALE')")
        assert render_expr_node(fold_constants(node, registry=registry)) == "tolower('MALE')"

    def test_distinct_calls_are_never_folded(self):
        folded = fold_constants(FunctionCall("tolower", (Literal("MALE"),), distinct=True))
        assert isinstance(folded, FunctionCall) and folded.distinct

    def test_escaped_result_renders_escaped(self):
        """A folded value containing a quote/backslash must round-trip through the
        renderer's escaping (the residual translator then declines on it, by design)."""
        assert _fold_text("toLower('IT\\u0027S')") == "'it\\u0027s'"


# --------------------------------------------------------------------------------
# THE CLASSIFICATION IS A PARTITION, NOT A LIST
# --------------------------------------------------------------------------------

class TestClassification:
    def test_every_surface_function_is_classified_exactly_once(self):
        surface = set(GFQL_ALLOWED_FUNCTIONS) | set(GFQL_AGGREGATION_FUNCTIONS)
        foldable = set(FOLDABLE_FUNCTIONS)
        declined = set(NON_FOLDABLE_REASONS)
        assert not (foldable & declined), f"classified twice: {sorted(foldable & declined)}"
        assert foldable | declined == surface, (
            "UNCLASSIFIED (add to FOLDABLE_FUNCTIONS or NON_FOLDABLE_REASONS with a "
            f"criterion): {sorted(surface - (foldable | declined))}; "
            f"stale (no longer on the surface): {sorted((foldable | declined) - surface)}"
        )

    def test_every_decline_cites_a_criterion(self):
        for name, reason in NON_FOLDABLE_REASONS.items():
            assert reason.startswith(("(P)", "(A)", "(E)", "(T)")), (name, reason)

    @pytest.mark.parametrize("name", ["rand", "randomuuid", "timestamp", "now", "date", "datetime"])
    def test_nondeterministic_functions_are_not_on_the_surface(self, name):
        """Criterion (P) has nothing to reject today because the parser accepts none of
        these.  If one is ever added, the partition test above fails until it is
        classified — which is the point of making the classification a partition."""
        assert name not in set(GFQL_ALLOWED_FUNCTIONS) | set(GFQL_AGGREGATION_FUNCTIONS)

    def test_aggregates_are_never_folded(self):
        for name in GFQL_AGGREGATION_FUNCTIONS:
            assert name not in FOLDABLE_FUNCTIONS

    def test_foldable_set_is_exactly_the_documented_six(self):
        """A guard against silently widening the pass: adding a function here must be a
        deliberate edit with its own (E) argument and parity tests."""
        assert set(FOLDABLE_FUNCTIONS) == {
            "tolower", "lower", "toupper", "upper", "size", "substring"
        }


# --------------------------------------------------------------------------------
# Structure preservation: folding must not disturb anything else
# --------------------------------------------------------------------------------

class TestStructurePreservation:
    @pytest.mark.parametrize("text", [
        "a.c = 5",
        "(a.c > 1) AND (b.d < 2)",
        "a.c IS NULL",
        "NOT (a.c = 'x')",
        "a.c IN ['x', 'y']",
        "a.c STARTS WITH 'x'",
        "CASE WHEN a.c = 1 THEN 'y' ELSE 'n' END",
        "coalesce(a.c, 'x')",
        "round(a.c, 2)",
        "abs(a.c)",
        "any(x IN a.c WHERE x = 1)",
        "a.c[0]",
        "a.c =~ '^x'",
    ])
    def test_non_foldable_expressions_round_trip_unchanged(self, text):
        assert _fold_text(text) == render_expr_node(parse_expr(text))

    def test_literal_only_arithmetic_is_not_folded(self):
        """Numeric folding is DECLINED by classification (engine-pinned kernels), so
        arithmetic over literals must survive to the runtime unchanged."""
        assert _fold_text("abs(0 - 3)") == render_expr_node(parse_expr("abs(0 - 3)"))

    def test_fold_is_idempotent(self):
        once = fold_constants(parse_expr("toLower(a.c) = toLower('Fine Dining')"))
        assert fold_constants(once) == once

    def test_leaf_nodes_are_preserved(self) -> None:
        leaf = PropertyAccessExpr(Identifier("a"), "c")
        assert fold_constants(leaf) == leaf
