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

THE BAR FOR A DECLINE: every disqualifier must have a violating existence proof -- a
concrete expression where folding would change the answer.  A disqualifier with no
constructible witness is a guess, and the function either folds or its reason is stated
as POLICY (perf) rather than CORRECTNESS.  ``TestArgumentClosureWitness`` and the engine
witnesses in ``test_const_fold_engine_parity.py`` are that proof; ``TestClassification``
proves only COVERAGE (nothing forgotten), which is worth having but is not meaning.
"""
from typing import Mapping, Optional

import pytest

from graphistry.compute.gfql.expr_const_fold import (
    DECLINED_FUNCTIONS,
    DENIED_AGGREGATE,
    DENIED_BY_POLICY,
    DENIED_NOT_ARGUMENT_CLOSED,
    DENIED_RESULT_TYPE,
    DENIED_UNVERIFIED,
    FOLDABLE_FUNCTIONS,
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
    QuantifierExpr,
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


def _lowered_text(text: str, params: Optional[Mapping[str, object]] = None) -> str:
    """The predicate text the lowering actually serializes into ``where_rows``."""
    span = SourceSpan(1, 0, 1, len(text), 0, len(text))
    out = _row_expr_arg(ExpressionText(text=text, span=span), params=params, field="where")
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
        """PASS ORDER PIN -- a STRUCTURAL assertion, deliberately not an answer one.

        `_rewrite_cypher_integer_division_ast` wraps a division in `toInteger(...)`
        only when BOTH operands are already integer literals.  Folding first would make
        `size('abcd') / 2` into `4 / 2` in time for that rewrite to fire, so the plan
        text a downstream matcher sees changes: `(4 / 2)` becomes `toInteger((4 / 2))`.

        HONEST LIMIT, MEASURED RATHER THAN ASSUMED: that text change did NOT change any
        answer on the nine division shapes tried (`size(...)/int`, `/float`,
        `/size(...)`, negated, and substring-derived), because the row evaluators
        already floor-divide two integer VALUES -- `6 / 4` answers `1` with or without
        the wrapper, on master and on this branch.  So this pins the ORDER and the plan
        TEXT; it does not claim a wrong answer.  It is still worth pinning: the wrapper
        is semantically load-bearing wherever the evaluator would not integer-divide on
        its own, and a plan-time pass must not silently move which expressions get it.

        (Separately and pre-existing on master: openCypher says `size(col) / 2` IS
        integer division, because `size` returns an Integer.  GFQL answers `1.5` for
        `size(n.s) / 4` and `1` for `size('abcdef') / 4` -- an internal inconsistency
        this PR neither creates nor fixes.)"""
        assert _lowered_text(text) == expected

    @pytest.mark.parametrize("value,expected", [
        ("FINE DINING", "(tolower(a.c) = 'fine dining')"),
        ("Fine Dining", "(tolower(a.c) = 'fine dining')"),
        ("fine dining", "(tolower(a.c) = 'fine dining')"),
    ])
    def test_parameter_values_fold_too(self, value, expected):
        """A `$param` is substituted before this pass runs, so a parameterized
        `toLower($p)` canonicalizes exactly like a written literal. Safe because the
        compiled-plan cache keys on the params (`_compile_string_query`), so a plan
        folded for one parameter value is never reused for another -- pinned end to
        end in test_const_fold_engine_parity.py."""
        assert _lowered_text("toLower(a.c) = toLower($p)", params={"p": value}) == expected

    def test_non_ascii_parameter_declines_like_a_written_literal(self):
        assert _lowered_text("toLower(a.c) = toLower($p)", params={"p": "STRAßE"}) \
            == "(tolower(a.c) = tolower('STRAßE'))"

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
# EVERY FOLDABLE ENTRY GETS THREE TESTS
#
# (a) a literal-only call folds to the EXPECTED literal (the value is pinned, not just
#     "something changed"), (b) the same function with a non-literal argument comes back
#     untouched.  (c) VALUE IDENTITY -- folded and unfolded plans answer the same on the
#     same data, per engine -- is the one that actually protects the change, and it needs
#     real frames, so it lives in test_const_fold_engine_parity.py.
# --------------------------------------------------------------------------------

#: name -> (literal-only call, the literal it must fold to, a non-literal spelling)
FOLD_CASES = [
    ("tolower", "toLower('MALE')", "'male'", "toLower(a.c)"),
    ("lower", "lower('MALE')", "'male'", "lower(a.c)"),
    ("toupper", "toUpper('male')", "'MALE'", "toUpper(a.c)"),
    ("upper", "upper('male')", "'MALE'", "upper(a.c)"),
    ("size", "size('abcde')", "5", "size(a.c)"),
    ("substring", "substring('abcdef', 1, 3)", "'bcd'", "substring(a.c, 1, 3)"),
    ("head", "head('abc')", "'a'", "head(a.c)"),
    ("tail", "tail('abc')", "'bc'", "tail(a.c)"),
    ("reverse", "reverse('abc')", "'cba'", "reverse(a.c)"),
]


class TestFoldableEntries:
    def test_every_foldable_function_has_a_pinned_case(self):
        """A new folder cannot ship without its three tests."""
        assert {case[0] for case in FOLD_CASES} == set(FOLDABLE_FUNCTIONS)

    @pytest.mark.parametrize("name,call,expected,_column", FOLD_CASES, ids=[c[0] for c in FOLD_CASES])
    def test_literal_only_call_folds_to_the_expected_literal(self, name, call, expected, _column):
        assert _fold_text(call) == expected

    @pytest.mark.parametrize("name,_call,_expected,column", FOLD_CASES, ids=[c[0] for c in FOLD_CASES])
    def test_non_literal_argument_is_returned_untouched(self, name, _call, _expected, column):
        assert _fold_text(column) == render_expr_node(parse_expr(column))


# --------------------------------------------------------------------------------
# head / tail / reverse: the STRING overload, and every decline around it
# --------------------------------------------------------------------------------

class TestSequenceStringFolds:
    """On GFQL's surface these are STRING operations, not list operations.

    ``row/dispatch.py`` implements them as ``.str.get(0)`` / ``.str.slice(start=1)`` /
    ``.str[::-1]`` on a Series, and ``eval_sequence_fn_scalar`` checks
    ``isinstance(value, str)`` FIRST for ``reverse``.  They were previously declined
    with "(A) sequence op; list literals parse to ListLiteral, not Literal" -- a reason
    that describes only the LIST overload, which the driver's per-call argument guard
    already declines on its own.  For the string overload the reason had no witness:
    ``head('abc')`` is argument-closed and answers ``'a'``, a value the contract guard
    accepts, identically on every engine for ASCII input.  So they fold.
    """

    @pytest.mark.parametrize("text,expected", [
        ("head('a')", "'a'"),
        ("head('abc')", "'a'"),
        ("tail('a')", "''"),
        ("tail('')", "''"),
        ("tail('abc')", "'bc'"),
        ("reverse('')", "''"),
        ("reverse('a')", "'a'"),
        ("reverse('abc')", "'cba'"),
        ("reverse(reverse('abc'))", "'abc'"),
        ("toUpper(tail('xabc'))", "'ABC'"),
        ("size(reverse('abcd'))", "4"),
    ])
    def test_string_overload_folds(self, text, expected):
        assert _fold_text(text) == expected

    def test_head_of_the_empty_string_declines(self):
        """``eval_sequence_fn_scalar`` answers ``value[0] if len(value) > 0 else None``,
        so ``head('')`` is null at runtime -- and NULL POLICY forbids synthesizing one.
        ``tail('')`` is ``''`` rather than null, so it does NOT need this carve-out."""
        assert _fold_text("head('')") == "head('')"

    @pytest.mark.parametrize("text", [
        "head([1, 2, 3])", "tail([1, 2, 3])", "reverse([1, 2, 3])", "reverse(['a', 'b'])",
    ])
    def test_list_overload_declines_on_the_argument_guard(self, text):
        """A ``ListLiteral`` is not a ``Literal``, so the driver declines before any
        folder runs -- which is exactly what the old (A) reason described, and all it
        described."""
        node = parse_expr(text)
        assert isinstance(node, FunctionCall)
        assert not all(isinstance(arg, Literal) for arg in node.args)
        assert _fold_text(text) == render_expr_node(node)

    @pytest.mark.parametrize("fn", ["head", "tail", "reverse"])
    @pytest.mark.parametrize("lit", NON_ASCII_LITERALS)
    def test_non_ascii_declines_like_the_case_folds(self, fn, lit):
        """Same gate, same reason: ``[::-1]`` and ``[0]`` are CODEPOINT operations, which
        only coincide with character operations where there are no combining sequences or
        surrogate pairs."""
        text = f"{fn}('{lit}')"
        assert _fold_text(text) == render_expr_node(parse_expr(text))

    @pytest.mark.parametrize("text", [
        "head(5)", "tail(1.5)", "reverse(true)", "head(null)", "tail(null)", "reverse(null)",
        "head('a', 'b')", "reverse('a', 'b')",
    ])
    def test_non_string_null_or_bad_arity_declines(self, text):
        assert _fold_text(text) == render_expr_node(parse_expr(text))


# --------------------------------------------------------------------------------
# THE DECLINE TAXONOMY IS A SET OF WITNESSES, NOT A LIST OF OPINIONS
#
# A decline is cheap to write and expensive to check.  The previous
# ``NON_FOLDABLE_REASONS`` was 40 free-text criteria that NOTHING read -- the driver's
# only name-keyed gate is ``table.get(name) is None`` -- and the obvious test for it is
# vacuous: ``fold_constants(parse_expr(q)) == parse_expr(q)`` passes for all 40 names
# trivially, because nothing outside FOLDABLE_FUNCTIONS can fold at all.  So each bucket
# is now tested by THE MECHANISM ITS NAME CLAIMS, and the two buckets with no available
# mechanism say so instead.
# --------------------------------------------------------------------------------

class TestClassification:
    """Coverage, not meaning: this proves nothing can be forgotten, not that anything
    below is right.  The witness tests are what carry the meaning."""

    def test_every_surface_function_is_classified_exactly_once(self):
        surface = set(GFQL_ALLOWED_FUNCTIONS) | set(GFQL_AGGREGATION_FUNCTIONS)
        buckets = {
            "FOLDABLE_FUNCTIONS": set(FOLDABLE_FUNCTIONS),
            "DENIED_AGGREGATE": set(DENIED_AGGREGATE),
            "DENIED_NOT_ARGUMENT_CLOSED": set(DENIED_NOT_ARGUMENT_CLOSED),
            "DENIED_RESULT_TYPE": set(DENIED_RESULT_TYPE),
            "DENIED_BY_POLICY": set(DENIED_BY_POLICY),
            "DENIED_UNVERIFIED": set(DENIED_UNVERIFIED),
        }
        seen: set = set()
        for label, names in buckets.items():
            twice = seen & names
            assert not twice, f"{label}: classified twice: {sorted(twice)}"
            seen |= names
        assert seen == surface, (
            "UNCLASSIFIED (put it in FOLDABLE_FUNCTIONS with three tests, or in the "
            "bucket naming the mechanism that stops it -- with that mechanism's "
            f"witness): {sorted(surface - seen)}; "
            f"stale (no longer on the surface): {sorted(seen - surface)}"
        )

    def test_declined_functions_is_the_union_of_the_decline_buckets(self):
        assert DECLINED_FUNCTIONS == (
            set(DENIED_AGGREGATE) | set(DENIED_NOT_ARGUMENT_CLOSED)
            | set(DENIED_RESULT_TYPE) | set(DENIED_BY_POLICY) | set(DENIED_UNVERIFIED)
        )
        assert not (DECLINED_FUNCTIONS & set(FOLDABLE_FUNCTIONS))

    def test_the_only_name_keyed_gate_is_the_registry_lookup(self):
        """THE STRUCTURAL INVARIANT the old reason list was gesturing at, stated so it can
        be false.  ``fold_constants`` reads NO decline table: a name declines iff it is
        absent from the registry it is handed.  Prove it by handing the driver a registry
        that contains a declined name -- if a decline table were consulted anywhere, this
        would still decline."""
        recorded = []

        def _spy(args: LiteralArgs) -> Optional[FoldedValue]:
            recorded.append(args)
            return "folded"

        for name in ["count", "abs", "range", "tostring", "keys"]:
            registry: Mapping[str, ConstantFolder] = {name: _spy}
            assert render_expr_node(
                fold_constants(FunctionCall(name, (Literal(1),)), registry=registry)
            ) == "'folded'", f"{name}: something other than the name lookup declined it"
        assert len(recorded) == 5

    @pytest.mark.parametrize("name", ["rand", "randomuuid", "timestamp", "now", "date", "datetime"])
    def test_nondeterministic_functions_are_not_on_the_surface(self, name):
        """Criterion (P) has nothing to reject today because the parser accepts none of
        these.  If one is ever added, the partition test above fails until it is
        classified — which is the point of making the classification a partition."""
        assert name not in set(GFQL_ALLOWED_FUNCTIONS) | set(GFQL_AGGREGATION_FUNCTIONS)

    def test_aggregates_are_never_folded(self):
        for name in GFQL_AGGREGATION_FUNCTIONS:
            assert name not in FOLDABLE_FUNCTIONS
        assert set(GFQL_AGGREGATION_FUNCTIONS) == set(DENIED_AGGREGATE)

    def test_foldable_set_is_exactly_the_documented_nine(self):
        """A guard against silently widening the pass: adding a function here must be a
        deliberate edit with its own (E) argument and parity tests."""
        assert set(FOLDABLE_FUNCTIONS) == {
            "tolower", "lower", "toupper", "upper", "size", "substring",
            "head", "tail", "reverse",
        }


class TestArgumentClosureWitness:
    """(A): the witness is the parsed node of the shape the lowering actually emits."""

    @pytest.mark.parametrize(
        "name,shape",
        sorted(DENIED_NOT_ARGUMENT_CLOSED.items()),
        ids=sorted(DENIED_NOT_ARGUMENT_CLOSED),
    )
    def test_the_emitted_shape_is_not_argument_closed(self, name, shape):
        node = parse_expr(shape)
        if not isinstance(node, FunctionCall):
            # The quantifiers are the strongest form of this: they parse to a
            # QuantifierExpr, so they cannot reach the driver's name lookup at all.
            assert isinstance(node, QuantifierExpr), (name, shape, type(node).__name__)
            return
        assert node.name.lower() == name, (name, shape, node.name)
        assert any(not isinstance(arg, Literal) for arg in node.args), (
            f"{name}: `{shape}` IS argument-closed, so criterion (A) does not stop it -- "
            "either move it to FOLDABLE_FUNCTIONS or refile it under the mechanism that "
            "actually does"
        )
        assert _fold_text(shape) == render_expr_node(node)

    @pytest.mark.parametrize("text", [
        "head('abc')", "tail('abc')", "reverse('abc')", "range(1, 5)",
    ])
    def test_criterion_A_is_NOT_available_for_these(self, text):
        """THE TEST THAT FOUND THE MISCLASSIFICATION, kept as a regression pin.

        All four were filed under (A).  All four are argument-closed ``FunctionCall``s
        with ``Literal`` arguments, so the argument guard never sees them: (A) was never
        the mechanism.  ``head``/``tail``/``reverse`` moved into FOLDABLE_FUNCTIONS;
        ``range`` moved to DENIED_RESULT_TYPE, where the witness is that the engine
        answers it with a ``list``."""
        node = parse_expr(text)
        assert isinstance(node, FunctionCall)
        assert all(isinstance(arg, Literal) for arg in node.args)

    @pytest.mark.parametrize("name", [
        "keys", "labels", "type", "properties", "nodes", "relationships",
    ])
    def test_the_grammar_accepts_a_literal_spelling_the_lowering_never_emits(self, name):
        """HONEST SCOPE OF THE (A) CLAIM.  The grammar is looser than the language:
        ``keys('x')`` parses, and it IS argument-closed.  So for the entity functions (A)
        is a statement about the shape the LOWERING emits, and what declines the
        hand-written literal spelling is the name lookup alone.  Pinned so the qualified
        claim cannot quietly rot back into an unqualified one."""
        node = parse_expr(f"{name}('x')")
        assert isinstance(node, FunctionCall)
        assert all(isinstance(arg, Literal) for arg in node.args)
        assert name not in FOLDABLE_FUNCTIONS
        assert _fold_text(f"{name}('x')") == render_expr_node(node)


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
        """Numeric folding is declined, so arithmetic over literals must survive to the
        runtime unchanged.  Note the reason: ``abs`` is DENIED_BY_POLICY, not by a
        correctness criterion -- ``abs(-3)`` is ``3`` on every engine and folding it
        could not change an answer.  What it also could not do is speed anything up, and
        this pass exists to canonicalize predicate spellings."""
        assert _fold_text("abs(0 - 3)") == render_expr_node(parse_expr("abs(0 - 3)"))

    def test_fold_is_idempotent(self):
        once = fold_constants(parse_expr("toLower(a.c) = toLower('Fine Dining')"))
        assert fold_constants(once) == once

    def test_leaf_nodes_are_preserved(self) -> None:
        leaf = PropertyAccessExpr(Identifier("a"), "c")
        assert fold_constants(leaf) == leaf
