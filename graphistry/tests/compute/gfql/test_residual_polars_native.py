"""#1729/#1755/#1806: native polars translation of connected-join residuals.

Covers `_residual_polars_expr` (the string→pl.Expr translator) and the fast-lane /
chain-fallback split in `_connected_join_apply_node_residuals`:
- positive: every covered shape translates and filters byte-identically to the
  chain fallback (the previous behavior), including nulls and case folding
- negative: unsupported shapes, alias mismatches, and absent columns decline
  (translator returns None); a group with ANY untranslatable expr falls back whole
- cross-engine: pandas frames never enter the fast lane (chain fallback only)

#1806 widened the translator from two hand-written regex shapes to the full
single-alias vocabulary by delegating to the SAME `lower_expr` the where_rows
fallback uses (`row_pipeline.lower_single_alias_predicate`). The differential
classes below are the gate: for every shape, `frame.filter(translated)` must equal
the forced chain fallback on the same frame, and every decline must be a shape the
fallback declines too (so the designed NotImplementedError still surfaces).
"""
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.compute import gfql_fast_paths as fp

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

requires_polars = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")


def _pl_nodes():
    return pl.DataFrame({
        "node_id": [1, 2, 3, 4, 5, 6],
        "name": ["Alice", "alice", "BOB", None, "Chloé", "bob"],
        "age": [30, 25, None, 40, 35, 25],
        "score": [1.5, 2.5, 3.5, None, 0.5, 2.5],
    })


def _pl_graph(nodes):
    edges = pl.DataFrame({"src": [1, 2], "dst": [2, 3]})
    return graphistry.nodes(nodes, "node_id").edges(edges, "src", "dst")


def COLS():
    """Schema of the _pl_nodes fixture (the translator now dtype-gates)."""
    return dict(_pl_nodes().schema)


def _canon(df):
    """Normalize either frame type to a sorted pandas frame for exact comparison."""
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    return pdf.sort_values("node_id").reset_index(drop=True)


class TestResidualTranslator:
    @requires_polars
    def test_tolower_eq_casefold(self):
        expr = fp._residual_polars_expr("(tolower(a.name) = 'alice')", "a", COLS())
        assert expr is not None
        out = _pl_nodes().filter(expr)
        assert sorted(out["node_id"].to_list()) == [1, 2]

    @requires_polars
    def test_tolower_eq_null_dropped(self):
        expr = fp._residual_polars_expr("(tolower(a.name) = 'bob')", "a", COLS())
        out = _pl_nodes().filter(expr)
        assert sorted(out["node_id"].to_list()) == [3, 6]  # null name row 4 dropped

    @requires_polars
    @pytest.mark.parametrize("op,lit,expected", [
        ("=", "25", [2, 6]),
        (">=", "30", [1, 4, 5]),
        ("<=", "25", [2, 6]),
        (">", "30", [4, 5]),
        ("<", "30", [2, 6]),
    ])
    def test_scalar_int_cmp(self, op, lit, expected):
        expr = fp._residual_polars_expr(f"(a.age {op} {lit})", "a", COLS())
        assert expr is not None
        out = _pl_nodes().filter(expr)
        # null age (row 3) always dropped: null comparison -> null -> filtered
        assert sorted(out["node_id"].to_list()) == expected

    @requires_polars
    def test_scalar_float_cmp(self):
        expr = fp._residual_polars_expr("(a.score >= 2.5)", "a", COLS())
        out = _pl_nodes().filter(expr)
        assert sorted(out["node_id"].to_list()) == [2, 3, 6]

    @requires_polars
    def test_scalar_string_eq(self):
        expr = fp._residual_polars_expr("(a.name = 'BOB')", "a", COLS())
        out = _pl_nodes().filter(expr)
        assert out["node_id"].to_list() == [3]  # exact case, unlike tolower

    @requires_polars
    def test_negative_int_literal(self):
        nodes = pl.DataFrame({"node_id": [1, 2], "delta": [-5, 5]})
        expr = fp._residual_polars_expr("(a.delta < -1)", "a", dict(nodes.schema))
        assert expr is not None
        assert nodes.filter(expr)["node_id"].to_list() == [1]

    @requires_polars
    @pytest.mark.parametrize("bad", [
        # --- string predicates: the row lowering has no native kernel for them, AND the
        # connected-join WHERE renderer cannot emit them at all (pinned in
        # TestStringPredicatesAreUnreachable), so covering them here would be dead code
        "(a.name CONTAINS 'x')",
        "(a.name STARTS WITH 'x')",
        "(a.name ENDS WITH 'x')",
        "(a.name =~ '.*x.*')",
        # --- another alias's column: not in THIS frame, and the fallback's prefixed row
        # table cannot resolve it either (-> its designed NotImplementedError)
        "(tolower(b.name) = 'alice')",  # alias mismatch (checked with alias='a')
        "(toupper(b.name) = 'ALICE')",  # alias mismatch, other case fn
        "(upper(zz.name) = 'ALICE')",   # alias mismatch, GQL alias spelling
        "(tolower(a.name) = b.name)",   # rhs is another alias's column
        "(b.age = 25)",                 # alias mismatch (checked with alias='a')
        "((a.age = 25) AND (b.age = 30))",  # one conjunct on a foreign alias
        "(NOT (b.age = 25))",           # foreign alias under NOT
        "(b.age IS NULL)",              # foreign alias under IS NULL
        "(a.name IN ['x', b.name])",    # foreign alias inside the IN list
        "(CASE WHEN b.age > 1 THEN 1 ELSE 0 END = 1)",  # foreign alias inside CASE
        "(a.age + b.age = 2)",          # foreign alias inside arithmetic
        # --- bare identifiers: no bare name is a column of the fallback's prefixed table
        "(age = 25)",
        "(__gfql_node_id__ = 1)",
        "(a.__gfql_node_id__ = 1)",     # sentinel has no column on a bare-name frame
        # --- absent column / dtype mismatch: must stay residual (designed NIE)
        "(a.missing = 25)",
        "(tolower(a.name) = 25)",       # case fn (String) vs number: cross-type
        # --- unparseable / outside the row lowering's node whitelist
        "not-an-expression @@",
        "(a.age = 25) AND",
        "(a.name[0] = 'A')",            # subscript: lower_expr declines it
    ])
    def test_unsupported_shapes_decline(self, bad):
        assert fp._residual_polars_expr(bad, "a", COLS()) is None

    @requires_polars
    def test_declines_when_the_row_expr_parser_is_unavailable(self, monkeypatch):
        """No parser bundle => decline, never a half-translated filter."""
        from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as rp
        monkeypatch.setattr(rp, "_parser", lambda: None)
        assert fp._residual_polars_expr("(a.age = 25)", "a", COLS()) is None


class TestResidualApplyFastLane:
    @requires_polars
    def test_fast_lane_matches_chain_fallback(self, monkeypatch):
        """The fast lane and the where_rows chain fallback agree byte-for-byte."""
        nodes = _pl_nodes()
        g = _pl_graph(nodes)
        exprs = ["(tolower(a.name) = 'alice')", "(a.age >= 25)"]
        fast = fp._connected_join_apply_node_residuals(
            g, nodes, "a", exprs, "node_id", engine=Engine.POLARS)
        # force the fallback by declining every translation
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        slow = fp._connected_join_apply_node_residuals(
            g, nodes, "a", exprs, "node_id", engine=Engine.POLARS)
        assert _canon(fast).equals(_canon(slow))
        assert sorted(fast["node_id"].to_list()) == [1, 2]

    @requires_polars
    def test_mixed_group_falls_back_whole(self, monkeypatch):
        """One untranslatable expr => the ENTIRE group uses the chain fallback.

        Simulates a translator gap on an expr the chain fallback DOES support
        (declining one of two supported exprs), and asserts no partial native
        filtering is mixed in: the result matches the pure chain fallback.
        """
        nodes = _pl_nodes()
        g = _pl_graph(nodes)
        exprs = ["(a.age >= 25)", "(tolower(a.name) = 'alice')"]
        real = fp._residual_polars_expr
        calls = []

        def gappy(expr, alias, columns):
            # decline the second expr only -> group must fall back WHOLE
            r = None if "tolower" in expr else real(expr, alias, columns)
            calls.append((expr, r is not None))
            return r
        monkeypatch.setattr(fp, "_residual_polars_expr", gappy)
        out = fp._connected_join_apply_node_residuals(
            g, nodes, "a", exprs, "node_id", engine=Engine.POLARS)
        assert any(ok for _, ok in calls) and not all(ok for _, ok in calls)
        # pure chain fallback as the oracle
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        expected = fp._connected_join_apply_node_residuals(
            g, nodes, "a", exprs, "node_id", engine=Engine.POLARS)
        assert _canon(out).equals(_canon(expected))
        assert sorted(_canon(out)["node_id"].tolist()) == [1, 2]

    def test_pandas_frames_never_fast_lane(self, monkeypatch):
        """pandas node frames must take the chain fallback, not polars exprs."""
        nodes = pd.DataFrame({
            "node_id": [1, 2, 3],
            "name": ["Alice", "alice", None],
            "age": [30, 25, 40],
        })
        edges = pd.DataFrame({"src": [1], "dst": [2]})
        g = graphistry.nodes(nodes, "node_id").edges(edges, "src", "dst")

        def boom(*a, **k):
            raise AssertionError("fast lane must not engage on pandas frames")
        monkeypatch.setattr(fp, "_residual_polars_expr", boom)
        out = fp._connected_join_apply_node_residuals(
            g, nodes, "a", ["(tolower(a.name) = 'alice')"], "node_id",
            engine=Engine.PANDAS)
        assert sorted(out["node_id"].tolist()) == [1, 2]


class TestResidualDtypeAndEscapeGates:
    """Review-skill wave (#1763): dtype mismatches must DECLINE so the chain fallback
    keeps the evaluator's designed parity-or-error NotImplementedError instead of raw
    polars behavior.

    #1806 RETIRED the escaped-literal decline: the residual text is now parsed by the
    evaluator's OWN parser (which unescapes ``\\uXXXX`` exactly as the fallback does)
    rather than compared as raw regex text, so there is nothing left to mismatch. The
    replacement is a non-vacuous differential (`TestEscapedLiteralParity`) over rows that
    really contain a quote and a backslash.
    """

    @requires_polars
    @pytest.mark.parametrize("expr", [
        "(a.age = 'thirty')",           # string literal vs numeric column
        "(tolower(a.age) = 'x')",       # tolower on numeric column
        "(a.name >= 25)",               # numeric literal vs string column
    ])
    def test_dtype_mismatch_declines(self, expr):
        assert fp._residual_polars_expr(expr, "a", COLS()) is None

    @requires_polars
    def test_case_fn_on_categorical_declines(self):
        """`.str` on Categorical raises in polars only, so the row lowering gates
        toLower/toUpper on a String OUTPUT dtype -- Categorical stays residual."""
        nodes = pl.DataFrame({"node_id": [1], "cat": ["x"]}).with_columns(
            pl.col("cat").cast(pl.Categorical))
        assert fp._residual_polars_expr(
            "(tolower(a.cat) = 'x')", "a", dict(nodes.schema)) is None

    @requires_polars
    def test_dtype_mismatch_group_reaches_designed_error(self):
        """End-to-end at the apply level: the group falls back whole and the chain
        evaluator raises its designed parity-or-error NotImplementedError (never a
        raw polars ComputeError)."""
        nodes = _pl_nodes()
        g = _pl_graph(nodes)
        with pytest.raises(NotImplementedError):
            fp._connected_join_apply_node_residuals(
                g, nodes, "a", ["(a.name >= 25)"], "node_id", engine=Engine.POLARS)


# ---------------------------------------------------------------------------------------
# #1806: the widened vocabulary.
#
# THE GATE IS DIFFERENTIAL, NOT DECLARATIVE. For every shape below the native filter must
# equal the forced where_rows chain fallback on the SAME frame, and the expected row ids are
# pinned as well so a differential that both sides get wrong is not mistaken for parity.
# ---------------------------------------------------------------------------------------

def _tricky_nodes():
    """Nulls, the empty string, non-ASCII, regex metacharacters, an embedded quote and an
    embedded backslash -- the values a raw-text/regex translator silently gets wrong."""
    return pl.DataFrame({
        "node_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "name": ["Alice", "alice", "BOB", None, "Chloé", "", "a.b*c[d]", "it's", "C:\\x"],
        "age": [30, 25, None, 40, 35, 0, -5, 7, 8],
        "flag": [True, False, None, True, False, True, None, False, True],
    })


def _float_nodes():
    """A FLOAT column (``score``) plus a second float for in-query math, an int and a string --
    the dtype spread the NaN-mask scoping turns on."""
    return pl.DataFrame({
        "node_id": [1, 2, 3, 4],
        "score": [0.5, 1.5, 2.5, None],
        # num/other is 0.0/0.0 on row 2 -> a GENUINE in-query NaN, and an ordinary number
        # elsewhere, so a mask on the computed operand is discriminating rather than vacuous
        "num": [1.0, 0.0, 3.0, 0.0],
        "other": [1.0, 0.0, 2.0, 4.0],
        "label": ["x", "x", "y", "x"],
    })


def _fast_lane_ids_matching_fallback(expr, nodes=None, alias="a"):
    """Assert the native path is TAKEN for ``expr`` and answers exactly like the fallback.

    Returns the surviving node ids so the caller also pins the ANSWER. Forcing the
    translator to decline is what makes the right-hand side the pre-#1806 chain evaluator.
    """
    nodes = _pl_nodes() if nodes is None else nodes
    translated = fp._residual_polars_expr(expr, alias, dict(nodes.schema))
    assert translated is not None, f"{expr!r} declined -- the native path was NOT taken"
    fast = nodes.filter(translated)
    real = fp._residual_polars_expr
    try:
        fp._residual_polars_expr = lambda *a, **k: None
        slow = fp._connected_join_apply_node_residuals(
            _pl_graph(nodes), nodes, alias, [expr], "node_id", engine=Engine.POLARS)
    finally:
        fp._residual_polars_expr = real
    assert _canon(fast).equals(_canon(slow)), f"{expr!r}: fast lane != where_rows fallback"
    return sorted(fast["node_id"].to_list())


class TestWidenedShapesTakeTheNativePath:
    """(a) POSITIVE: each shape the pre-#1806 regex pair declined now translates, and the
    native answer is the chain evaluator's answer.

    Null handling is the sharp edge and is pinned per shape: Cypher WHERE is 3-valued, so a
    NULL operand yields NULL and ``filter`` drops the row for ``!=``/``<>`` just as it does
    for ``=`` -- the fast lane must NOT keep those rows (pandas' object-dtype ``!=`` on a
    missing value would).
    """

    @requires_polars
    @pytest.mark.parametrize("expr,expected", [
        ("(a.name != 'BOB')", [1, 2, 5, 6]),          # null name (4) DROPPED, not kept
        ("(a.name <> 'BOB')", [1, 2, 5, 6]),          # `<>` spelling of the same op
        ("(a.age != 25)", [1, 4, 5]),                 # null age (3) dropped
        ("(a.name IS NULL)", [4]),
        ("(a.name IS NOT NULL)", [1, 2, 3, 5, 6]),
        ("(a.age IS NULL)", [3]),
        ("(a.name IN ['BOB', 'alice'])", [2, 3]),
        ("(a.age IN [25, 30])", [1, 2, 6]),
        ("((a.name = 'BOB') OR (a.age = 25))", [2, 3, 6]),
        ("(NOT (a.name = 'BOB'))", [1, 2, 5, 6]),     # NOT null = null -> dropped
        ("((a.age >= 25) AND (a.age <= 30))", [1, 2, 6]),
        ("(tolower(a.name) != 'alice')", [3, 5, 6]),
        ("(toupper(a.name) <> 'BOB')", [1, 2, 5]),
        # shapes the regex pair rejected purely for LAYOUT, not semantics
        ("a.age = 25", [2, 6]),                       # no outer parens
        ("('alice' = tolower(a.name))", [1, 2]),      # reversed operand order
        ("(substring(a.name, 0, 2) = 'al')", [2]),    # fn on the column side
        ("(CASE WHEN a.age > 30 THEN 1 ELSE 0 END = 1)", [4, 5]),  # ternary
        ("(a.age + 1 = 26)", [2, 6]),                 # arithmetic
    ])
    def test_shape_translates_and_matches_fallback(self, expr, expected):
        assert _fast_lane_ids_matching_fallback(expr) == expected


class TestEscapedLiteralParity:
    """The renderer escapes ``'`` and ``\\`` to ``\\uXXXX`` text. The pre-#1806 translator
    DECLINED any literal containing a backslash because it compared the raw regex capture;
    the widened one hands the text to the evaluator's own parser, which unescapes it the
    same way. Rows that really contain a quote and a backslash keep this non-vacuous."""

    @requires_polars
    @pytest.mark.parametrize("expr,expected", [
        ("(a.name = 'it\\u0027s')", [8]),
        ("(a.name != 'it\\u0027s')", [1, 2, 3, 5, 6, 7, 9]),
        ("(a.name = 'C:\\u005Cx')", [9]),
        ("(a.name != 'C:\\u005Cx')", [1, 2, 3, 5, 6, 7, 8]),
        ("(a.name IN ['it\\u0027s', 'C:\\u005Cx'])", [8, 9]),
        ("(tolower(a.name) = 'it\\u0027s')", [8]),
    ])
    def test_escaped_literal_matches_fallback(self, expr, expected):
        assert _fast_lane_ids_matching_fallback(expr, _tricky_nodes()) == expected


class TestLiteralTextSemanticsParity:
    """(c) DIFFERENTIAL over the values that separate a literal comparison from a regex or
    a byte comparison: metacharacters, the empty string, non-ASCII, and NULL."""

    @requires_polars
    @pytest.mark.parametrize("expr,expected", [
        # `.` `*` `[` are DATA here, never a pattern
        ("(a.name = 'a.b*c[d]')", [7]),
        ("(a.name != 'a.b*c[d]')", [1, 2, 3, 5, 6, 8, 9]),
        ("(a.name IN ['a.b*c[d]'])", [7]),
        # a metacharacter-only literal must match NOTHING, not everything
        ("(a.name = '.*')", []),
        # empty string is a value, not a null
        ("(a.name = '')", [6]),
        ("(a.name != '')", [1, 2, 3, 5, 7, 8, 9]),
        # non-ASCII round-trips through the escape + the case kernels
        ("(a.name = 'Chlo\\u00e9')", [5]),
        ("(tolower(a.name) = 'chlo\\u00e9')", [5]),
        ("(a.name != 'Chlo\\u00e9')", [1, 2, 3, 6, 7, 8, 9]),
        # 3-valued booleans
        ("(a.flag IS NULL)", [3, 7]),
        ("(a.flag IS NOT NULL)", [1, 2, 4, 5, 6, 8, 9]),
        ("(NOT (a.flag = true))", [2, 5, 8]),
        ("((a.name IS NULL) OR (a.age > 30))", [4, 5]),
    ])
    def test_value_semantics_match_fallback(self, expr, expected):
        assert _fast_lane_ids_matching_fallback(expr, _tricky_nodes()) == expected


class TestWrongDtypeStaysResidual:
    """(b) NEGATIVE: a dtype-incompatible variant of every widened shape must DECLINE, and
    the fallback must then raise the row op's designed NotImplementedError. Declining is
    load-bearing here: translating would hand back a raw polars error (or, for ``IN``, a
    silently different membership answer) instead of the designed one."""

    @requires_polars
    @pytest.mark.parametrize("expr", [
        "(a.name != 25)",              # numeric literal vs String column
        "(a.age != 'thirty')",         # string literal vs Int column
        "(a.age IN ['x'])",            # cross-category IN
        "(toupper(a.age) != 'X')",     # case fn on a non-String column
    ])
    def test_declines_and_fallback_raises_designed_error(self, expr):
        nodes = _tricky_nodes()
        assert fp._residual_polars_expr(expr, "a", dict(nodes.schema)) is None
        with pytest.raises(NotImplementedError):
            fp._connected_join_apply_node_residuals(
                _pl_graph(nodes), nodes, "a", [expr], "node_id", engine=Engine.POLARS)


class TestCategoricalNonStrOpsParity:
    """A Categorical column is only a problem for ``.str`` kernels. ``=``/``!=``/``IS NULL``/
    ``IN`` are not, and the where_rows evaluator answers them, so the fast lane must too --
    the pre-#1806 dtype gate declined them wholesale (correct but needlessly)."""

    @staticmethod
    def _cat_nodes():
        return pl.DataFrame({
            "node_id": [1, 2, 3, 4],
            "cat": ["Alice", "BOB", None, "bob"],
        }).with_columns(pl.col("cat").cast(pl.Categorical))

    @requires_polars
    @pytest.mark.parametrize("expr,expected", [
        ("(a.cat = 'BOB')", [2]),
        ("(a.cat != 'BOB')", [1, 4]),
        ("(a.cat IS NULL)", [3]),
        ("(a.cat IN ['BOB', 'bob'])", [2, 4]),
    ])
    def test_categorical_non_str_ops_match_fallback(self, expr, expected):
        assert _fast_lane_ids_matching_fallback(expr, self._cat_nodes()) == expected


class TestStringPredicatesAreUnreachable:
    """DECLINE WITH EVIDENCE (contradicting the obvious guess).

    ``STARTS WITH`` / ``ENDS WITH`` / ``CONTAINS`` / ``=~`` are declined not because parity
    is hard but because NO such residual can reach this translator: on the polars engine
    ``_pushdown_connected_join_where_filters`` cannot render them to a row filter, so the
    comma-pattern query is rejected upstream and the translator is never called. Teaching
    it those shapes would be unreachable code; the gap is in the WHERE renderer.
    """

    Q = ("MATCH (p {node_type:'Person'})-[]->(i), (p)-[]->(c) WHERE %s "
         "RETURN count(p) AS n")

    @staticmethod
    def _g():
        nodes = pl.DataFrame({
            "node_id": [1, 2, 3],
            "node_type": ["Person", "X", "Y"],
            "s": ["ab", "cd", None],
        })
        edges = pl.DataFrame({"src": [1, 1], "dst": [2, 3]})
        return graphistry.nodes(nodes, "node_id").edges(edges, "src", "dst")

    @requires_polars
    @pytest.mark.parametrize("pred", [
        "i.s STARTS WITH 'a'", "i.s ENDS WITH 'b'", "i.s CONTAINS 'b'", "i.s =~ '.*b'",
    ])
    def test_no_such_residual_ever_reaches_the_translator(self, pred, monkeypatch):
        from graphistry.compute.exceptions import GFQLValidationError
        seen = []
        real = fp._residual_polars_expr

        def spy(expr, alias, columns):
            seen.append(expr)
            return real(expr, alias, columns)

        monkeypatch.setattr(fp, "_residual_polars_expr", spy)
        with pytest.raises(GFQLValidationError):
            self._g().gfql(self.Q % pred, engine="polars")._nodes
        assert seen == [], f"{pred!r} unexpectedly reached the residual translator"

    @requires_polars
    @pytest.mark.parametrize("expr", [
        "(a.name STARTS WITH 'A')", "(a.name ENDS WITH 'e')",
        "(a.name CONTAINS 'l')", "(a.name =~ '.*l.*')",
    ])
    def test_translator_declines_them_and_so_does_the_evaluator(self, expr):
        """Belt-and-braces: even if such a residual were synthesized by hand, the fast lane
        declines it and the fallback raises -- no engine answers it natively, so there is
        no correct answer for the fast lane to guess at."""
        nodes = _tricky_nodes()
        assert fp._residual_polars_expr(expr, "a", dict(nodes.schema)) is None
        with pytest.raises(NotImplementedError):
            fp._connected_join_apply_node_residuals(
                _pl_graph(nodes), nodes, "a", [expr], "node_id", engine=Engine.POLARS)


class TestFusedTwoStarLane:
    """#1755 lane-1: the fused single-collect two-star plan must be value-identical
    to the eager path (which it replaces when residuals translate natively).
    Every fused-arm test ASSERTS lane engagement via a spy on the extracted
    _connected_join_two_star_fused_polars helper -- the original tests silently
    compared slow-path vs slow-path because count(*) lowers to a 2-tuple agg that
    declines the whole two-star fast path before either lane."""

    def _star_graph(self):
        pl2 = pytest.importorskip("polars")
        ndf = pl2.DataFrame({
            "node_id": list(range(1, 11)),
            "node_type": ["Person"] * 4 + ["Interest"] * 3 + ["City"] * 3,
            "interest": [None] * 4 + ["Fine Dining", "fine dining", "tennis"] + [None] * 3,
            "city": [None] * 7 + ["London", "london", "Paris"],
            "gender": ["male", "female", "male", "female"] + [None] * 6,
        })
        edf = pl2.DataFrame({
            "src": [1, 1, 2, 2, 3, 4, 1, 2, 3, 4],
            "dst": [5, 6, 5, 7, 6, 5, 8, 8, 9, 10],
            "rel": ["HAS_INTEREST"] * 6 + ["LIVES_IN"] * 4,
        })
        return graphistry.nodes(ndf, "node_id").edges(edf, "src", "dst")

    # count(p) -- count(*) lowers to a 2-tuple agg and declines the two-star fast
    # path entirely (pinned below), so it can never reach the fused lane.
    Q = ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
         "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
         "WHERE toLower(i.interest) = toLower('FINE DINING') AND p.gender = 'male' "
         "RETURN c.city AS city, count(p) AS n ORDER BY n DESC, city LIMIT 5")

    def _spy_fused(self, monkeypatch):
        calls = []
        orig = fp._connected_join_two_star_fused_polars

        def spy(*a, **k):
            out = orig(*a, **k)
            calls.append(out is not None)
            return out

        monkeypatch.setattr(fp, "_connected_join_two_star_fused_polars", spy)
        return calls

    @staticmethod
    def _rows(res):
        df = res._nodes
        df = df.to_pandas() if hasattr(df, "to_pandas") else df
        return df.to_dict("records")

    @requires_polars
    def test_fused_matches_eager_chain_path(self, monkeypatch):
        g = self._star_graph()
        calls = self._spy_fused(monkeypatch)
        fused = g.gfql(self.Q, engine="polars")
        assert calls and calls[-1], "fused lane did not engage (vacuous comparison)"
        # forcing every translation to decline disables the fused lane AND the
        # residual fast lane -> full eager path + where_rows chain fallback
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        eager = g.gfql(self.Q, engine="polars")
        assert self._rows(fused) == self._rows(eager)
        assert self._rows(fused)  # non-empty: ORDER BY pinned, exact row order compared

    @requires_polars
    def test_fused_empty_result(self, monkeypatch):
        g = self._star_graph()
        q = self.Q.replace("FINE DINING", "no such interest")
        calls = self._spy_fused(monkeypatch)
        fused = g.gfql(q, engine="polars")
        assert calls and calls[-1], "fused lane did not engage"
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        eager = g.gfql(q, engine="polars")

        def shape(res):
            df = res._nodes
            df = df.to_pandas() if hasattr(df, "to_pandas") else df
            return (len(df), sorted(map(str, df.columns)))
        assert shape(fused) == shape(eager)

    @requires_polars
    def test_fused_matches_pandas_oracle(self, monkeypatch):
        g = self._star_graph()
        gpd = graphistry.nodes(g._nodes.to_pandas(), "node_id").edges(g._edges.to_pandas(), "src", "dst")
        calls = self._spy_fused(monkeypatch)
        got = g.gfql(self.Q, engine="polars")._nodes
        assert calls and calls[-1], "fused lane did not engage"
        got = (got.to_pandas() if hasattr(got, "to_pandas") else got).to_dict("records")
        oracle = gpd.gfql(self.Q, engine="pandas")._nodes.to_dict("records")
        assert got == oracle

    @requires_polars
    def test_pandas_frames_polars_engine_no_crash(self, monkeypatch):
        """BLOCKER-1 pin: pandas frames + engine='polars' (the WITH..MATCH reentry
        shape) must run the residual two-star query, not AttributeError on
        edges.lazy() -- the fused lane converts edges before going lazy."""
        g = self._star_graph()
        gpd = graphistry.nodes(g._nodes.to_pandas(), "node_id").edges(g._edges.to_pandas(), "src", "dst")
        res = gpd.gfql(self.Q, engine="polars")
        assert self._rows(res) == self._rows(g.gfql(self.Q, engine="polars"))

    @requires_polars
    def test_fused_ungrouped_empty_match_returns_zero_row(self, monkeypatch):
        """BLOCKER-2 pin: ungrouped count with a live first arm but empty join must
        return the single n=0 row (the eager all-left-counts==1 shortcut / the
        openCypher count over no rows), not a 0x0 frame."""
        g = self._star_graph()
        # tennis -> only person 2, one HAS_INTEREST edge (left counts all == 1, non-empty);
        # NoSuchCity -> right arm empty -> empty join
        q = ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
             "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
             "WHERE toLower(i.interest) = toLower('TENNIS') AND c.city = 'NoSuchCity' "
             "RETURN count(p) AS n")
        calls = self._spy_fused(monkeypatch)
        fused = g.gfql(q, engine="polars")
        assert calls and calls[-1], "fused lane did not engage"
        assert self._rows(fused) == [{"n": 0}]
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        eager = g.gfql(q, engine="polars")
        assert self._rows(fused) == self._rows(eager)

    @requires_polars
    def test_count_star_declines_two_star_fast_path(self, monkeypatch):
        """Decline-shape pin: count(*) lowers to a 2-tuple agg, so the two-star fast
        path (fused AND eager) declines and the general path answers -- and the
        fused lane must NOT engage."""
        g = self._star_graph()
        q = self.Q.replace("count(p)", "count(*)")
        calls = self._spy_fused(monkeypatch)
        res = g.gfql(q, engine="polars")
        assert not any(calls), "count(*) unexpectedly reached the fused lane"
        assert self._rows(res)  # still answered (general path)

    # --- CONSTANT FOLDING: one canonical residual shape reaches the translator ------

    #: The BOARD's own spelling (benchmarks/graphbench/matched_q1_q9/gb_queries.py,
    #: md5 6e7ae268a5a41742587fcb87854b6e27): a ONE-SIDED toLower with an already
    #: lowercase literal. Master declines this and drops the whole fused lane.
    Q_ONE_SIDED = ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
                   "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
                   "WHERE toLower(i.interest) = 'fine dining' AND p.gender = 'male' "
                   "RETURN c.city AS city, count(p) AS n ORDER BY n DESC, city LIMIT 5")

    def _spy_residual_texts(self, monkeypatch):
        """Record the residual STRINGS the translator is asked to handle."""
        seen = []
        real = fp._residual_polars_expr

        def spy(expr, alias, columns):
            out = real(expr, alias, columns)
            seen.append((expr, out is not None))
            return out

        monkeypatch.setattr(fp, "_residual_polars_expr", spy)
        return seen

    @requires_polars
    def test_two_sided_query_reaches_the_translator_already_folded(self, monkeypatch):
        """CANONICALIZATION, observed at the fast-path boundary: the user writes the
        TWO-SIDED form, and what arrives here is the ONE-SIDED text. This is what
        makes a single matcher shape sufficient."""
        g = self._star_graph()
        seen = self._spy_residual_texts(monkeypatch)
        g.gfql(self.Q, engine="polars")
        tolower_exprs = [e for e, _ in seen if "tolower" in e]
        assert tolower_exprs, "no toLower residual reached the translator"
        assert all(e == "(tolower(i.interest) = 'fine dining')" for e in tolower_exprs), \
            f"expected the folded one-sided text, got {tolower_exprs}"

    @requires_polars
    def test_one_sided_residual_engages_fused_lane(self, monkeypatch):
        """STRUCTURAL LOCK-IN (not a timing gate): a single untranslatable residual
        declines the ENTIRE fused lane, so `served == 1` is the regression signal.
        A scaling-ladder gate is the wrong shape here -- the removed cost is a
        per-op constant, so the residual O(N) term dominates any growth ratio."""
        g = self._star_graph()
        calls = self._spy_fused(monkeypatch)
        g.gfql(self.Q_ONE_SIDED, engine="polars")
        assert calls.count(True) == 1, (
            f"fused lane served {calls.count(True)} times, expected 1 "
            "(0 => the one-sided toLower residual stopped translating)")

    @requires_polars
    def test_one_sided_fused_matches_eager_chain_path(self, monkeypatch):
        g = self._star_graph()
        calls = self._spy_fused(monkeypatch)
        fused = g.gfql(self.Q_ONE_SIDED, engine="polars")
        assert calls and calls[-1], "fused lane did not engage (vacuous comparison)"
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        eager = g.gfql(self.Q_ONE_SIDED, engine="polars")
        assert self._rows(fused) == self._rows(eager)
        # `Fine Dining` + `fine dining` both fold on the COLUMN side -> persons 1, 2, 4
        assert self._rows(fused) == [{"city": "London", "n": 2}, {"city": "london", "n": 1}]

    @requires_polars
    @pytest.mark.parametrize("lit", ["FINE DINING", "Fine Dining", "fine Dining"])
    def test_one_sided_mixed_case_literal_matches_nothing_end_to_end(self, lit, monkeypatch):
        """THE TRAP, end to end. A mixed-case ONE-SIDED literal must return the SAME
        (empty) answer through the fused lane as through the chain evaluator: the
        evaluator does NOT case-fold a bare literal, and neither may the translator.
        The two-sided form of the same query returns rows -- pinned below, so this is
        not a vacuous 'everything is empty' assertion. Every board literal is already
        lowercase, which is exactly why a wrong rule here would ship green."""
        g = self._star_graph()
        q = self.Q_ONE_SIDED.replace("'fine dining'", f"'{lit}'")
        calls = self._spy_fused(monkeypatch)
        fused = g.gfql(q, engine="polars")
        assert calls and calls[-1], "fused lane did not engage"
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        eager = g.gfql(q, engine="polars")
        assert self._rows(fused) == self._rows(eager) == []
        # control: folding the literal (two-sided) DOES match -> the empty answer above
        # is a real semantic difference, not an inert query
        assert self._rows(g.gfql(q.replace(f"'{lit}'", f"toLower('{lit}')"), engine="polars"))

    @requires_polars
    def test_one_sided_matches_pandas_oracle(self, monkeypatch):
        g = self._star_graph()
        gpd = graphistry.nodes(g._nodes.to_pandas(), "node_id").edges(
            g._edges.to_pandas(), "src", "dst")
        calls = self._spy_fused(monkeypatch)
        got = g.gfql(self.Q_ONE_SIDED, engine="polars")._nodes
        assert calls and calls[-1], "fused lane did not engage"
        got = (got.to_pandas() if hasattr(got, "to_pandas") else got).to_dict("records")
        assert got == gpd.gfql(self.Q_ONE_SIDED, engine="pandas")._nodes.to_dict("records")

    @requires_polars
    @pytest.mark.parametrize("fn,lit", [
        ("toUpper", "FINE DINING"), ("upper", "FINE DINING"), ("lower", "fine dining"),
    ])
    def test_other_case_functions_engage_and_match_the_evaluator(self, fn, lit, monkeypatch):
        """The generalization is not toLower-shaped: every case function the row
        evaluator supports takes the same lane, on the same canonical text."""
        g = self._star_graph()
        q = self.Q_ONE_SIDED.replace("toLower(i.interest) = 'fine dining'",
                                     f"{fn}(i.interest) = '{lit}'")
        calls = self._spy_fused(monkeypatch)
        fused = g.gfql(q, engine="polars")
        assert calls and calls[-1], f"{fn}: fused lane did not engage"
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        assert self._rows(fused) == self._rows(g.gfql(q, engine="polars"))
        assert self._rows(fused), f"{fn}: vacuous (empty) comparison"

    @requires_polars
    def test_non_ascii_two_sided_stays_unfolded_but_now_serves_the_lane(self, monkeypatch):
        """DISCLOSED NARROWING, restated for #1806. A non-ASCII two-sided literal is
        outside the region where the engines provably agree, so the plan-time CONSTANT FOLD
        still declines and the residual text arrives UNFOLDED -- that invariant is asserted
        directly, and it is the one that protects the Python-vs-Rust case table.

        What changed: the fused lane no longer declines with it. The pre-#1806 regex pair
        could not match a two-sided text, and that incidental decline bought nothing --
        the where_rows fallback it deferred to lowers BOTH sides with the same polars
        ``to_lowercase`` kernel anyway, so the answer was already the Rust-cased one. The
        widened translator lowers the same two sides with the same kernel; the assertion
        that matters (answer == chain fallback) is unchanged and still made."""
        g = self._star_graph()
        q = self.Q_ONE_SIDED.replace("toLower(i.interest) = 'fine dining'",
                                     "toLower(i.interest) = toLower('FINE DINİNG')")
        seen = self._spy_residual_texts(monkeypatch)
        calls = self._spy_fused(monkeypatch)
        served = g.gfql(q, engine="polars")
        assert any("= tolower(" in e for e, _ in seen), \
            f"constant fold unexpectedly collapsed the two-sided literal: {seen}"
        assert calls and calls[-1], "two-sided residual no longer serves the fused lane"
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        assert self._rows(served) == self._rows(g.gfql(q, engine="polars"))

    # --- #1806 widened vocabulary, END TO END through gfql() -----------------------------

    #: `!=`/`<>`, IS [NOT] NULL, IN, OR, NOT: every one of these was a whole-lane decline
    #: before #1806 (one untranslatable residual drops the fused plan entirely).
    WIDENED_PREDICATES = [
        "i.interest <> 'tennis'",
        "i.interest != 'tennis'",
        "i.interest IS NOT NULL",
        "i.interest IN ['tennis', 'fine dining']",
        "(i.interest = 'tennis' OR i.interest = 'fine dining')",
        "NOT (i.interest = 'tennis')",
        "toUpper(i.interest) <> 'TENNIS'",
        "i.interest <> 'tennis' AND p.gender IS NOT NULL",
    ]

    def _widened_query(self, pred):
        return ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
                "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
                f"WHERE {pred} "
                "RETURN c.city AS city, count(p) AS n ORDER BY n DESC, city LIMIT 5")

    @requires_polars
    @pytest.mark.parametrize("pred", WIDENED_PREDICATES)
    def test_widened_residual_engages_fused_lane_and_matches_the_evaluator(self, pred, monkeypatch):
        """STRUCTURAL LOCK-IN: `served == 1` is the regression signal (a single
        untranslatable residual declines the ENTIRE fused lane), and the answer must equal
        the forced where_rows chain path on the same graph."""
        g = self._star_graph()
        q = self._widened_query(pred)
        calls = self._spy_fused(monkeypatch)
        fused = g.gfql(q, engine="polars")
        assert calls.count(True) == 1, (
            f"{pred}: fused lane served {calls.count(True)} times, expected 1 "
            "(0 => the widened residual stopped translating)")
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        eager = g.gfql(q, engine="polars")
        assert self._rows(fused) == self._rows(eager)
        assert self._rows(fused), f"{pred}: vacuous (empty) comparison"

    @requires_polars
    @pytest.mark.parametrize("pred", [
        p for p in WIDENED_PREDICATES if " IN [" not in p
    ])
    def test_widened_residual_matches_pandas_oracle(self, pred, monkeypatch):
        """CROSS-ENGINE: the widened polars fast lane answers what pandas answers.

        ``IN [...]`` is excluded because the PANDAS connected-join route raises
        ``GFQLTypeError: Unalignable boolean Series`` on an ``x IN [...]`` residual -- a
        pre-existing pandas-side defect on master, unrelated to and untouched by the polars
        translator (the polars fast lane and the polars where_rows fallback agree on it, as
        pinned above)."""
        g = self._star_graph()
        gpd = graphistry.nodes(g._nodes.to_pandas(), "node_id").edges(
            g._edges.to_pandas(), "src", "dst")
        q = self._widened_query(pred)
        calls = self._spy_fused(monkeypatch)
        got = g.gfql(q, engine="polars")._nodes
        assert calls and calls[-1], f"{pred}: fused lane did not engage"
        got = (got.to_pandas() if hasattr(got, "to_pandas") else got).to_dict("records")
        assert got == gpd.gfql(q, engine="pandas")._nodes.to_dict("records")
        assert got, f"{pred}: vacuous (empty) comparison"

    # --- MINIMAL-JOIN fused plan: dropped-semi-join proofs, end to end ------------------

    #: The graph-benchmark q7 anatomy verbatim: range filter on the shared alias, an
    #: equality pushdown on the second leaf, a one-sided toLower residual on the first
    #: leaf, TWO group keys from the second leaf, ORDER BY count DESC + key ASC, LIMIT 1.
    Q_Q7_SHAPE = ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
                  "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
                  "WHERE toLower(i.interest) = 'fine dining' "
                  "AND p.age >= 20 AND p.age <= 40 AND c.country = 'United Kingdom' "
                  "RETURN c.city AS city, c.country AS country, count(p) AS n "
                  "ORDER BY n DESC, city ASC LIMIT 1")

    def _q7_shape_graph(self):
        pl2 = pytest.importorskip("polars")
        ndf = pl2.DataFrame({
            "node_id": [1, 2, 3, 4, 5, 6, 7],
            "node_type": ["Person", "Person", "Person", "Interest", "Interest", "City", "City"],
            "age": [25, 30, 55, None, None, None, None],
            "interest": [None, None, None, "Fine Dining", "tennis", None, None],
            "city": [None, None, None, None, None, "London", "Paris"],
            "country": [None, None, None, None, None, "United Kingdom", "France"],
        })
        edf = pl2.DataFrame({
            # Fine dining: p1 holds TWO edges (left count 2), p2 and p3 one each --
            # p3 is OUTSIDE the age domain, so its group EXISTS in the unrestricted
            # left counts and must be dropped by the final inner join, not by a
            # pre-restriction. Tennis: p2 (in-domain) once, p3 (out-of-domain)
            # TWICE -- the boundary-probe discriminator below.
            "src": [1, 1, 2, 3, 2, 3, 3, 1, 2, 3],
            "dst": [4, 4, 4, 4, 5, 5, 5, 6, 6, 7],
            "rel": ["HAS_INTEREST"] * 7 + ["LIVES_IN"] * 3,
        })
        return graphistry.nodes(ndf, "node_id").edges(edf, "src", "dst")

    @requires_polars
    def test_minimal_join_q7_shape_serves_and_matches_both_twins(self, monkeypatch):
        """Lane pin + value parity for the exact board-cell anatomy. The minimal-join
        plan drops the left-arm shared semi-join AND the right-arm leaf semi-join
        (the unique-keyed lookup join subsumes it); the answer must still be the
        eager twin's and the pandas oracle's, with the out-of-domain shared node
        (p3) and the un-matched leaf (Paris/tennis) both excluded."""
        g = self._q7_shape_graph()
        calls = self._spy_fused(monkeypatch)
        fused = g.gfql(self.Q_Q7_SHAPE, engine="polars")
        assert calls and calls[-1], "fused lane did not engage"
        assert self._rows(fused) == [{"city": "London", "country": "United Kingdom", "n": 3}]
        gpd = graphistry.nodes(g._nodes.to_pandas(), "node_id").edges(
            g._edges.to_pandas(), "src", "dst")
        assert self._rows(fused) == gpd.gfql(self.Q_Q7_SHAPE, engine="pandas")._nodes.to_dict("records")
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        eager = g.gfql(self.Q_Q7_SHAPE, engine="polars")
        assert self._rows(fused) == self._rows(eager)

    @requires_polars
    def test_empty_match_boundary_probe_keeps_the_shared_domain_restriction(self, monkeypatch):
        """THE dropped-semi-join hazard, pinned. The n=0 shortcut probes whether the
        LEFT counts are all 1 -- and that probe must see the EAGER lane's counts,
        i.e. WITH the shared-domain restriction the hot plan proved redundant.
        Here the out-of-domain person (age 55) holds a left count of 2; if the
        boundary probe ever reads the UNRESTRICTED hot-plan counts, all-==-1 flips
        false and the answer degrades from the openCypher n=0 row to a 0x0 frame."""
        g = self._q7_shape_graph()
        q = ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
             "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
             "WHERE toLower(i.interest) = 'tennis' "
             "AND p.age >= 20 AND p.age <= 40 AND c.city = 'NoSuchCity' "
             "RETURN count(p) AS n")
        # tennis: p2 (in-domain) once -> restricted left counts are all 1 and
        # non-empty; p3 (age 55, OUT of domain) twice -> the unrestricted counts
        # are NOT all 1. The two probe readings give different answers, so this
        # test fails if the boundary ever reads the hot-plan counts.
        calls = self._spy_fused(monkeypatch)
        fused = g.gfql(q, engine="polars")
        assert calls and calls[-1], "fused lane did not engage"
        assert self._rows(fused) == [{"n": 0}]
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        eager = g.gfql(q, engine="polars")
        assert self._rows(fused) == self._rows(eager)

    @requires_polars
    def test_grouped_limit_zero_keeps_columns(self, monkeypatch):
        """LIMIT 0 must yield the 0-row WITH-columns frame, not the 0x0 empty-match
        boundary frame -- the lazy grouped tail cannot tell those apart from its
        own (empty) output, so LIMIT 0 is pinned to the eager tail."""
        g = self._q7_shape_graph()
        q = self.Q_Q7_SHAPE.replace("LIMIT 1", "LIMIT 0")
        calls = self._spy_fused(monkeypatch)
        fused = g.gfql(q, engine="polars")

        def shape(res):
            df = res._nodes
            df = df.to_pandas() if hasattr(df, "to_pandas") else df
            return (len(df), sorted(map(str, df.columns)))

        assert calls and calls[-1], "fused lane did not engage"
        assert shape(fused) == (0, ["city", "country", "n"])
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        eager = g.gfql(q, engine="polars")
        assert shape(fused) == shape(eager)

    @requires_polars
    def test_grouped_tail_rides_the_single_collect(self, monkeypatch):
        """STRUCTURAL LOCK-IN of the fused lane's one-collect contract, now including
        the grouped tail: a served non-empty grouped query must reach polars exactly
        once (the eager tail's group_by/sort each pay their own engine dispatch --
        the cost this plan shape removed)."""
        pl2 = pytest.importorskip("polars")
        g = self._q7_shape_graph()
        collects_inside = []
        orig_fused = fp._connected_join_two_star_fused_polars
        orig_collect = pl2.LazyFrame.collect

        def counting_fused(*a, **k):
            count = [0]

            def counting_collect(self_lf, *ca, **ck):
                count[0] += 1
                return orig_collect(self_lf, *ca, **ck)

            monkeypatch.setattr(pl2.LazyFrame, "collect", counting_collect)
            try:
                out = orig_fused(*a, **k)
            finally:
                monkeypatch.setattr(pl2.LazyFrame, "collect", orig_collect)
            collects_inside.append((count[0], out is not None))
            return out

        monkeypatch.setattr(fp, "_connected_join_two_star_fused_polars", counting_fused)
        res = g.gfql(self.Q_Q7_SHAPE, engine="polars")
        assert self._rows(res) == [{"city": "London", "country": "United Kingdom", "n": 3}]
        assert collects_inside and collects_inside[-1][1], "fused lane did not serve"
        assert collects_inside[-1][0] == 1, (
            f"fused lane issued {collects_inside[-1][0]} collects, expected exactly 1 "
            "(the grouped tail must ride the hot-path collect)")


# ---------------------------------------------------------------------------------------
# #1846 minimal-join: BOTH SIDES of every algebraic removal, pinned.
#
# The fused plan drops (a) the left-arm shared-domain semi-join and (b) the right-arm
# second-leaf semi-join, and (c) folds the grouped tail into the single collect -- each
# under a proof-carrying guard. The tests below pin the SERVED side (the removal really
# happens, and the answer is still the eager twin's / the pandas oracle's) AND the KEPT
# side (the guard branch that must NOT take the removal still emits the old op).
# Plan shapes are observed by instrumenting polars INSIDE the fused helper only, so a
# later refactor cannot silently swap which branch a query takes.
# ---------------------------------------------------------------------------------------

_LOOKUP_KEY = "__gfql_fast_second_leaf_id__"


def _run_with_fused_plan_spies(g, q):
    """Run ``q`` on the polars engine with the fused helper instrumented.

    Returns ``(nodes, served, joins, eager_group_bys, collects)`` where ``joins`` is the
    list of ``(how, left_on, right_on, on)`` for every LAZY join built inside the fused
    helper, ``eager_group_bys`` the DataFrame-level (eager) group_by calls, and
    ``collects`` the number of LazyFrame.collect calls it issued. Instrumentation is
    scoped to the helper's dynamic extent and restored unconditionally.
    """
    pl2 = pl
    joins: list = []
    eager_group_bys: list = []
    served: list = []
    collect_count = [0]
    orig_fused = fp._connected_join_two_star_fused_polars
    orig_join = pl2.LazyFrame.join
    orig_gb = pl2.DataFrame.group_by
    orig_collect = pl2.LazyFrame.collect

    def spy_join(self, other, *a, **k):
        joins.append((k.get("how", "inner"), k.get("left_on"), k.get("right_on"), k.get("on")))
        return orig_join(self, other, *a, **k)

    def spy_gb(self, *a, **k):
        eager_group_bys.append((a, k))
        return orig_gb(self, *a, **k)

    def spy_collect(self, *a, **k):
        collect_count[0] += 1
        return orig_collect(self, *a, **k)

    def spy_fused(*a, **k):
        pl2.LazyFrame.join = spy_join
        pl2.DataFrame.group_by = spy_gb
        pl2.LazyFrame.collect = spy_collect
        try:
            out = orig_fused(*a, **k)
        finally:
            pl2.LazyFrame.join = orig_join
            pl2.DataFrame.group_by = orig_gb
            pl2.LazyFrame.collect = orig_collect
        served.append(out is not None)
        return out

    fp._connected_join_two_star_fused_polars = spy_fused
    try:
        nodes = g.gfql(q, engine="polars")._nodes
    finally:
        fp._connected_join_two_star_fused_polars = orig_fused
    return nodes, served, joins, eager_group_bys, collect_count[0]


def _eager_twin_nodes(g, q):
    """The forced-decline eager twin: the fused helper declines, everything else is live
    (residual fast lane included), so this is exactly the plan the removals rewrote."""
    orig = fp._connected_join_two_star_fused_polars
    fp._connected_join_two_star_fused_polars = lambda *a, **k: None
    try:
        return g.gfql(q, engine="polars")._nodes
    finally:
        fp._connected_join_two_star_fused_polars = orig


def _to_records(df):
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    return pdf.to_dict("records")


def _col_names(df):
    return [str(c) for c in df.columns]


class TestMinimalJoinBothSides:
    """Served-side AND kept-side pins for each of the three #1846 removals."""

    #: the q7 anatomy (grouped, group properties) -- removal (b)'s SERVED side
    Q_GROUPED = TestFusedTwoStarLane.Q_Q7_SHAPE

    #: same match, ungrouped count -- NO group properties, removal (b)'s KEPT side
    Q_UNGROUPED = ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
                   "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
                   "WHERE toLower(i.interest) = 'fine dining' "
                   "AND p.age >= 20 AND p.age <= 40 "
                   "RETURN count(p) AS n")

    #: live first arm (restricted counts all 1), empty join -- the n=0 probe branch
    Q_N0_PROBE = ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
                  "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
                  "WHERE toLower(i.interest) = 'tennis' "
                  "AND p.age >= 20 AND p.age <= 40 AND c.city = 'NoSuchCity' "
                  "RETURN count(p) AS n")

    def _graph(self):
        return TestFusedTwoStarLane._q7_shape_graph(TestFusedTwoStarLane())

    @staticmethod
    def _semis(joins):
        return [(left, right) for how, left, right, _ in joins if how == "semi"]

    @staticmethod
    def _lookup_inners(joins):
        return [j for j in joins if j[0] == "inner" and j[2] == _LOOKUP_KEY]

    # --- (b) KEPT side: without group properties the second-leaf semi STAYS -----------

    @requires_polars
    def test_without_group_properties_the_second_leaf_semi_is_emitted(self):
        """PLAN pin: the ungrouped count has no group-property lookup join, so nothing
        subsumes the second-leaf semi-join and the fused plan must still emit it --
        semis are EXACTLY (first-leaf dst, shared-domain src, second-leaf dst), no
        lookup join, and no probe rebuild (the shared-domain semi appears once; the
        n=0 boundary would add a second). Value parity vs the eager twin and the
        pandas oracle. (Raw collect counts are not asserted on this branch: polars
        implements eager DataFrame ops via internal collects, so only the join shape
        is a stable signal here.)"""
        g = self._graph()
        nodes, served, joins, eager_gbs, _collects = _run_with_fused_plan_spies(g, self.Q_UNGROUPED)
        assert served == [True], "fused lane did not serve"
        assert self._semis(joins) == [("dst", "node_id"), ("src", "node_id"), ("dst", "node_id")], (
            f"expected the second-leaf semi to be EMITTED without group properties: {joins}")
        assert self._lookup_inners(joins) == [], (
            f"no group properties, yet a lookup join appeared: {joins}")
        assert eager_gbs == []
        assert _to_records(nodes) == [{"n": 3}]
        assert _to_records(nodes) == _to_records(_eager_twin_nodes(g, self.Q_UNGROUPED))
        gpd = graphistry.nodes(g._nodes.to_pandas(), "node_id").edges(
            g._edges.to_pandas(), "src", "dst")
        assert _to_records(nodes) == gpd.gfql(self.Q_UNGROUPED, engine="pandas")._nodes.to_dict("records")

    # --- (b) SERVED side: the lookup join subsumes the semi, explicitly --------------

    @requires_polars
    def test_with_group_properties_the_lookup_join_subsumes_the_second_leaf_semi(self):
        """PLAN pin for what the existing anatomy test asserts only by value: with group
        properties the second-leaf semi is NOT emitted (exactly two semis remain:
        first-leaf dst + shared-domain src) and exactly one unique-keyed lookup INNER
        join replaces it."""
        g = self._graph()
        nodes, served, joins, eager_gbs, collects = _run_with_fused_plan_spies(g, self.Q_GROUPED)
        assert served == [True], "fused lane did not serve"
        assert self._semis(joins) == [("dst", "node_id"), ("src", "node_id")], (
            f"the second-leaf semi must be SUBSUMED under group properties: {joins}")
        assert len(self._lookup_inners(joins)) == 1, f"expected one lookup join: {joins}"
        assert _to_records(nodes) == [{"city": "London", "country": "United Kingdom", "n": 3}]
        assert _to_records(nodes) == _to_records(_eager_twin_nodes(g, self.Q_GROUPED))

    # --- (a) SERVED side: populated match, out-of-domain multiplicity ----------------

    @requires_polars
    def test_populated_match_out_of_domain_counts_stay_correct_ungrouped(self):
        """The dropped left-arm semi, OBSERVABLE on a populated match: the fixture's
        unrestricted per-src counts carry an out-of-domain group (p3, age 55) that the
        restricted counts do not -- asserted directly so the discriminator cannot rot.
        The fused answer must still be 3 (p1's 2 + p2's 1): a plan that let the dropped
        semi widen the final join would answer 4 (p3's Paris row joins its count of 1).
        Complements the existing grouped anatomy pin with the ungrouped served side."""
        g = self._graph()
        ndf, edf = g._nodes, g._edges
        fine_ids = ndf.filter(
            pl.col("interest").str.to_lowercase() == "fine dining")["node_id"].to_list()
        first_arm = edf.filter(
            (pl.col("rel") == "HAS_INTEREST") & pl.col("dst").is_in(fine_ids))
        unrestricted = dict(
            first_arm.group_by("src").len().rows())
        in_domain = set(ndf.filter(
            (pl.col("node_type") == "Person")
            & (pl.col("age") >= 20) & (pl.col("age") <= 40))["node_id"].to_list())
        restricted = {k: v for k, v in unrestricted.items() if k in in_domain}
        assert set(unrestricted) != set(restricted), (
            "fixture no longer discriminates: every first-arm src is in-domain, so the "
            "dropped shared-domain semi is unobservable here")
        nodes, served, _joins, _gbs, _collects = _run_with_fused_plan_spies(g, self.Q_UNGROUPED)
        assert served == [True], "fused lane did not serve"
        assert _to_records(nodes) == [{"n": 3}]
        assert _to_records(nodes) == _to_records(_eager_twin_nodes(g, self.Q_UNGROUPED))
        gpd = graphistry.nodes(ndf.to_pandas(), "node_id").edges(
            edf.to_pandas(), "src", "dst")
        assert _to_records(nodes) == gpd.gfql(self.Q_UNGROUPED, engine="pandas")._nodes.to_dict("records")

    # --- (a) KEPT side: the n=0 probe REBUILDS the shared-domain semi ----------------

    @requires_polars
    def test_empty_match_probe_rebuilds_the_shared_domain_semi(self):
        """PLAN pin for the boundary branch the value-level probe test already guards:
        on the empty-match n=0 probe the fused helper must emit the shared-domain
        semi a SECOND time (the probe's restricted-count rebuild) -- the hot plan's
        unrestricted counts alone would flip the all-==-1 probe. Its populated twin
        above pins the served side: exactly ONE src-side semi, no rebuild."""
        g = self._graph()
        nodes, served, joins, _gbs, _collects = _run_with_fused_plan_spies(g, self.Q_N0_PROBE)
        assert served == [True], "fused lane did not serve"
        shared_semis = [s for s in self._semis(joins) if s == ("src", "node_id")]
        assert len(shared_semis) == 2, (
            f"the n=0 probe must REBUILD the shared-domain semi (expected 2 src-side "
            f"semis, hot plan + probe): {joins}")
        assert _to_records(nodes) == [{"n": 0}]
        assert _to_records(nodes) == _to_records(_eager_twin_nodes(g, self.Q_N0_PROBE))

    # --- (c) SERVED side: grouped-with-results tail is lazy (no eager group_by) ------

    @requires_polars
    def test_grouped_with_results_has_no_eager_group_by(self):
        """The one-collect pin's structural complement: when the grouped tail rides the
        single collect, NO eager DataFrame.group_by may run inside the helper."""
        g = self._graph()
        nodes, served, _joins, eager_gbs, collects = _run_with_fused_plan_spies(g, self.Q_GROUPED)
        assert served == [True], "fused lane did not serve"
        assert eager_gbs == [], f"grouped tail fell off the lazy plan: {eager_gbs}"
        assert collects == 1, f"expected the single fused collect, saw {collects}"
        assert _to_records(nodes) == [{"city": "London", "country": "United Kingdom", "n": 3}]

    # --- (c) KEPT side: grouped LIMIT 0 takes the eager tail --------------------------

    @requires_polars
    def test_grouped_limit_zero_takes_the_eager_tail(self):
        """The LIMIT 0 guard branch must still run the EAGER grouped tail (that is what
        preserves the 0-row WITH-columns contract): exactly one eager group_by on the
        group keys, maintain_order preserved -- plus the value contract vs the twin."""
        g = self._graph()
        q = self.Q_GROUPED.replace("LIMIT 1", "LIMIT 0")
        nodes, served, _joins, eager_gbs, _collects = _run_with_fused_plan_spies(g, q)
        assert served == [True], "fused lane did not serve"
        assert len(eager_gbs) == 1, f"LIMIT 0 must keep the eager grouped tail: {eager_gbs}"
        gb_args, gb_kwargs = eager_gbs[0]
        assert gb_args == (["city", "country"],)
        assert gb_kwargs.get("maintain_order") is True
        assert len(nodes) == 0 and sorted(_col_names(nodes)) == ["city", "country", "n"]
        eager = _eager_twin_nodes(g, q)
        assert len(eager) == 0 and sorted(_col_names(eager)) == sorted(_col_names(nodes))

    # --- (c) ungrouped shapes: unchanged behavior -------------------------------------

    @requires_polars
    @pytest.mark.parametrize("suffix", [" LIMIT 0", " LIMIT 2"])
    def test_ungrouped_limit_shapes_never_reach_the_fused_lane(self, suffix):
        """Ungrouped count + LIMIT declines the two-star fast path BEFORE either lane
        (unchanged by #1846): the fused helper is never called, and the answer is
        byte-identical to the forced-decline twin and the pandas oracle."""
        g = self._graph()
        q = self.Q_UNGROUPED + suffix
        nodes, served, joins, _gbs, _collects = _run_with_fused_plan_spies(g, q)
        assert served == [] and joins == [], f"ungrouped{suffix} unexpectedly reached the fused lane"
        eager = _eager_twin_nodes(g, q)
        assert nodes.equals(eager) and dict(nodes.schema) == dict(eager.schema)
        gpd = graphistry.nodes(g._nodes.to_pandas(), "node_id").edges(
            g._edges.to_pandas(), "src", "dst")
        oracle = gpd.gfql(q, engine="pandas")._nodes
        assert _col_names(nodes) == _col_names(oracle)
        assert _to_records(nodes) == oracle.to_dict("records")


class TestMinimalJoinDifferential:
    """Seeded differential across the three #1846 boundary axes.

    ~8 seeded graphs x (group properties yes/no) x (empty/populated match) x
    (LIMIT 0/None/positive): the fused lane, the forced-decline eager twin, and the
    pandas oracle must agree byte-for-byte -- fused vs eager as full polars frame
    equality (schema included), vs pandas as column-order + record equality (the
    established cross-engine comparison in this file). Every seeded graph carries a
    deterministic core that keeps the populated combos non-empty and keeps an
    out-of-domain shared node in the first arm, so the dropped shared-domain semi
    stays observable on every seed.
    """

    LIMITS = ["", " LIMIT 0", " LIMIT 2"]

    @staticmethod
    def _seeded_graph(seed):
        import numpy as np
        pl2 = pytest.importorskip("polars")
        rng = np.random.default_rng(seed)
        # deterministic core: p1/p2 in the age domain (counts 2 and 1 on 'fine dining'),
        # p3 OUT of domain but present in the first arm; two cities, one unmatched
        node_ids = [1, 2, 3, 4, 5, 6, 7]
        node_types = ["Person", "Person", "Person", "Interest", "Interest", "City", "City"]
        ages: list = [25, 30, 55, None, None, None, None]
        interests: list = [None, None, None, "Fine Dining", "tennis", None, None]
        cities: list = [None, None, None, None, None, "London", "Paris"]
        countries: list = [None, None, None, None, None, "United Kingdom", "France"]
        srcs = [1, 1, 2, 3, 2, 3, 3]
        dsts = [4, 4, 4, 4, 5, 5, 5]
        rels = ["HAS_INTEREST"] * 7
        srcs += [1, 2, 3]
        dsts += [6, 6, 7]
        rels += ["LIVES_IN"] * 3
        # seeded noise: extra persons/interests/cities and multi-edges
        interest_pool = ["fine dining", "Fine Dining", "chess", "tennis"]
        n_interest = int(rng.integers(1, 4))
        extra_interests = list(range(200, 200 + n_interest))
        for j, nid in enumerate(extra_interests):
            node_ids.append(nid)
            node_types.append("Interest")
            ages.append(None)
            interests.append(interest_pool[int(rng.integers(0, len(interest_pool)))])
            cities.append(None)
            countries.append(None)
        n_city = int(rng.integers(1, 4))
        extra_cities = list(range(300, 300 + n_city))
        for k, nid in enumerate(extra_cities):
            node_ids.append(nid)
            node_types.append("City")
            ages.append(None)
            interests.append(None)
            cities.append(f"city_{seed}_{k}")
            countries.append("United Kingdom" if rng.integers(0, 2) else "France")
        n_person = int(rng.integers(3, 7))
        all_interests = [4, 5] + extra_interests
        all_cities = [6, 7] + extra_cities
        for i in range(n_person):
            nid = 100 + i
            node_ids.append(nid)
            node_types.append("Person")
            ages.append(int(rng.integers(15, 65)))
            interests.append(None)
            cities.append(None)
            countries.append(None)
            for _ in range(int(rng.integers(1, 4))):
                target = all_interests[int(rng.integers(0, len(all_interests)))]
                for _dup in range(int(rng.integers(1, 3))):
                    srcs.append(nid)
                    dsts.append(target)
                    rels.append("HAS_INTEREST")
            srcs.append(nid)
            dsts.append(all_cities[int(rng.integers(0, len(all_cities)))])
            rels.append("LIVES_IN")
        ndf = pl2.DataFrame({
            "node_id": node_ids, "node_type": node_types, "age": ages,
            "interest": interests, "city": cities, "country": countries,
        })
        edf = pl2.DataFrame({"src": srcs, "dst": dsts, "rel": rels})
        return graphistry.nodes(ndf, "node_id").edges(edf, "src", "dst")

    @staticmethod
    def _query(grouped, literal, limit_suffix):
        ret = ("c.city AS city, count(p) AS n ORDER BY n DESC, city ASC" if grouped
               else "count(p) AS n")
        return ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
                "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
                f"WHERE toLower(i.interest) = '{literal}' "
                "AND p.age >= 20 AND p.age <= 40 "
                f"RETURN {ret}{limit_suffix}")

    @requires_polars
    @pytest.mark.parametrize("seed", range(8))
    def test_fused_eager_and_pandas_agree_on_every_boundary_combo(self, seed):
        g = self._seeded_graph(seed)
        gpd = graphistry.nodes(g._nodes.to_pandas(), "node_id").edges(
            g._edges.to_pandas(), "src", "dst")
        populated_nonempty = 0
        for grouped in (True, False):
            for literal, populated in (("fine dining", True), ("no such interest", False)):
                for limit_suffix in self.LIMITS:
                    q = self._query(grouped, literal, limit_suffix)
                    label = f"seed={seed} grouped={grouped} lit={literal!r} lim={limit_suffix!r}"
                    fused, served, _joins, _gbs, _collects = _run_with_fused_plan_spies(g, q)
                    # ungrouped + LIMIT declines the two-star fast path upstream of the
                    # lane; every other combo must be SERVED by the fused helper
                    if grouped or limit_suffix == "":
                        assert served == [True], f"{label}: fused lane did not serve"
                    else:
                        assert served == [], f"{label}: unexpectedly reached the fused lane"
                    eager = _eager_twin_nodes(g, q)
                    assert dict(fused.schema) == dict(eager.schema), (
                        f"{label}: schema drift fused={fused.schema} eager={eager.schema}")
                    assert fused.equals(eager), (
                        f"{label}: fused != eager twin\nfused:\n{fused}\neager:\n{eager}")
                    oracle = gpd.gfql(q, engine="pandas")._nodes
                    assert _col_names(fused) == _col_names(oracle), (
                        f"{label}: column drift vs pandas oracle")
                    assert _to_records(fused) == oracle.to_dict("records"), (
                        f"{label}: fused != pandas oracle")
                    if populated and limit_suffix != " LIMIT 0" and len(fused) > 0:
                        populated_nonempty += 1
        # non-vacuity: the deterministic core guarantees populated combos return rows
        assert populated_nonempty >= 4, (
            f"seed={seed}: populated combos unexpectedly empty (vacuous differential)")


class TestFusedLaneNanGuardScoping:
    """#1832 follow-up: the fused lane skips the IEEE NaN mask for BARE COLUMN operands only.

    Mechanism: the general row lowering wraps every float comparison in
    ``& col.is_nan().not()`` so NaN compares IEEE-style rather than polars-style
    (NaN = largest). On the connected-join fused lane that mask is provably dead --
    gfql ingest ran ``_pl_nan_to_null`` over the frame -- and it measurably doubled the
    cost of the graph benchmark's two ``p.age`` comparisons. Suppression is opt-in, is
    scoped to column reads, and must never leak to the general lowering.
    """

    def _schema(self):
        return dict(_float_nodes().schema)

    @requires_polars
    def test_fused_lane_drops_the_mask_for_a_bare_column(self):
        """The board's own shape: float COLUMN vs int literal, the whole mask goes."""
        e = fp._residual_polars_expr("(a.score >= 1)", "a", self._schema())
        assert e is not None
        assert "is_nan" not in str(e), f"fused lane still carries the NaN mask: {e}"

    @requires_polars
    def test_only_the_column_side_loses_its_mask(self):
        """SCOPE pin: a float LITERAL operand is not a column read, so its own is_nan()
        term is untouched. Only the column term is dropped, and only here."""
        e = fp._residual_polars_expr("(a.score >= 1.0)", "a", self._schema())
        assert e is not None
        assert 'col("score").is_nan()' not in str(e), f"column term survived: {e}"
        assert "is_nan" in str(e), f"literal term was also dropped (out of scope): {e}"

    @requires_polars
    def test_column_vs_column_drops_both_terms(self):
        e = fp._residual_polars_expr("(a.score >= a.other)", "a", self._schema())
        assert e is not None
        assert "is_nan" not in str(e), f"a column-vs-column compare kept a mask: {e}"

    @requires_polars
    def test_general_row_lowering_still_emits_the_mask(self):
        """The DEFAULT (no opt-in) must stay guarded, for BOTH entry points."""
        from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as rp

        # 1. the same seam without the opt-in
        e = rp.lower_single_alias_predicate("(a.score >= 1)", "a", self._schema())
        assert e is not None and 'col("score").is_nan()' in str(e), f"default lost the mask: {e}"

        # 2. the general row-table lowering (`where_rows_polars`'s own path)
        table = _float_nodes().rename({c: f"a.{c}" for c in _float_nodes().columns})
        general = rp._lower_with_schema(
            table, lambda: rp.lower_expr_str("a.score >= 1", list(table.columns))
        )
        assert general is not None and "is_nan" in str(general), f"row table lost it: {general}"

    @requires_polars
    def test_computed_float_operand_keeps_the_mask_even_on_the_fused_lane(self):
        """In-query math manufactures NaN (0.0/0.0) that ingest cannot have removed."""
        e = fp._residual_polars_expr("((a.num / a.other) >= 1)", "a", self._schema())
        assert e is not None
        assert "is_nan" in str(e), f"computed operand lost its NaN mask: {e}"

    @requires_polars
    def test_a_computed_nan_is_still_answered_ieee_style_on_the_fused_lane(self):
        """VALUE proof for the exclusion above: `other` holds a 0.0, so `score/other` is a
        genuine in-query NaN on row 2. NaN >= 1 must be FALSE (IEEE/pandas), not TRUE
        (polars NaN = largest)."""
        nodes = _float_nodes()
        e = fp._residual_polars_expr("((a.num / a.other) >= 1)", "a", dict(nodes.schema))
        assert e is not None
        kept = nodes.filter(e)["node_id"].to_list()
        assert 2 not in kept, f"the in-query NaN row survived a >= compare: {kept}"
        pdf = nodes.to_pandas()
        assert kept == sorted(pdf[(pdf["num"] / pdf["other"]) >= 1]["node_id"].tolist())
        assert kept, "vacuous (empty) comparison"

    @requires_polars
    def test_int_and_string_columns_are_unaffected(self):
        for expr in ("(a.node_id >= 2)", "(a.label = 'x')"):
            e = fp._residual_polars_expr(expr, "a", self._schema())
            assert e is not None and "is_nan" not in str(e)

    @requires_polars
    def test_contextvar_is_restored_after_the_call(self):
        from graphistry.compute.gfql.lazy.engine.polars.lowering_context import COLUMNS_NAN_FREE

        assert COLUMNS_NAN_FREE.get() is False
        fp._residual_polars_expr("(a.score >= 1.0)", "a", self._schema())
        assert COLUMNS_NAN_FREE.get() is False, "opt-in leaked out of the fused lane"

    @requires_polars
    def test_mask_free_expr_matches_the_masked_one_on_ingested_data(self):
        """VALUE gate, not just a repr gate: same rows, mask or no mask."""
        from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as rp

        nodes = _float_nodes()
        for expr in ("(a.score >= 1)", "(a.score < 2)", "(a.score = 1.5)", "(a.score <> 1.5)"):
            fast = nodes.filter(fp._residual_polars_expr(expr, "a", dict(nodes.schema)))
            guarded = nodes.filter(
                rp.lower_single_alias_predicate(expr, "a", dict(nodes.schema)))
            assert _canon(fast).equals(_canon(guarded)), f"{expr}: mask changed the answer"

    @requires_polars
    def test_genuine_nan_bypassing_pandas_is_normalized_by_ingest(self):
        """The suppression's PREMISE, end to end, ON THE LANE THAT USES IT.

        A natively-built polars frame is the only way to carry a real NaN into gfql (the
        pandas path converts at `from_pandas(nan_to_null=True)`), and `_coerce_input_formats`
        -> `_pl_nan_to_null` normalizes it to null on the way in. Three assertions, in
        increasing strength: the residual lane is actually reached; the frame it is handed
        carries no NaN in any float column; and the answer equals the pandas oracle.
        """
        pl2 = pytest.importorskip("polars")
        ndf = pl2.DataFrame({
            "node_id": list(range(1, 11)),
            "node_type": ["Person"] * 4 + ["Interest"] * 3 + ["City"] * 3,
            "interest": [None] * 4 + ["Fine Dining", "fine dining", "tennis"] + [None] * 3,
            "city": [None] * 7 + ["London", "london", "Paris"],
            # float column WITH A GENUINE NaN (rows 2 and 3), built natively -- no pandas hop
            "score": [1.5, float("nan"), float("nan"), 2.5] + [None] * 6,
        })
        assert ndf.get_column("score").is_nan().sum() == 2, "fixture lost its NaN"
        edf = pl2.DataFrame({
            "src": [1, 1, 2, 2, 3, 4, 1, 2, 3, 4],
            "dst": [5, 6, 5, 7, 6, 5, 8, 8, 9, 10],
            "rel": ["HAS_INTEREST"] * 6 + ["LIVES_IN"] * 4,
        })
        q = ("MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
             "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
             "WHERE toLower(i.interest) = 'fine dining' AND p.score >= 1 "
             "RETURN c.city AS city, count(p) AS n ORDER BY n DESC, city LIMIT 5")

        seen = []
        orig = fp._residual_polars_expr

        def spy(expr, alias, schema):
            seen.append(expr)
            return orig(expr, alias, schema)

        g = graphistry.nodes(ndf, "node_id").edges(edf, "src", "dst")
        try:
            fp._residual_polars_expr = spy
            got = g.gfql(q, engine="polars")._nodes
        finally:
            fp._residual_polars_expr = orig
        assert any("score" in e for e in seen), f"float residual never translated: {seen}"
        got = (got.to_pandas() if hasattr(got, "to_pandas") else got).to_dict("records")

        gp = graphistry.nodes(ndf.to_pandas(), "node_id").edges(edf.to_pandas(), "src", "dst")
        assert got == gp.gfql(q, engine="pandas")._nodes.to_dict("records")
        assert got, "vacuous (empty) comparison"

        # the load-bearing one: the ingested frame this lane reads has no NaN left, so the
        # dropped mask had nothing to mask.
        from graphistry.compute.ComputeMixin import _coerce_input_formats
        ingested = _coerce_input_formats(g, Engine.POLARS)._nodes
        floats = [c for c, dt in ingested.schema.items() if dt in (pl.Float32, pl.Float64)]
        assert floats, "fixture has no float column (vacuous)"
        for c in floats:
            assert not ingested.get_column(c).is_nan().any(), f"ingest left a raw NaN in {c}"


class TestNanFreeOperandPredicate:
    """Direct unit tests for `_operand_is_nan_free_column`, the one place the suppression is
    decided. The fused lane only ever reaches its `Identifier` arm (its predicates have been
    rewritten to bare columns by `_bare_column_ast`), so the `PropertyAccessExpr` arm --
    which is what a future caller opting in on a PREFIXED row table would hit -- is exercised
    here rather than left as an untested branch that a later change could silently break.
    """

    @requires_polars
    def _run(self, node, columns, nan_free):
        from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as rp
        from graphistry.compute.gfql.lazy.engine.polars.lowering_context import COLUMNS_NAN_FREE

        token = COLUMNS_NAN_FREE.set(nan_free)
        try:
            return rp._operand_is_nan_free_column(node, columns)
        finally:
            COLUMNS_NAN_FREE.reset(token)

    @requires_polars
    def test_optin_off_means_never(self):
        """Conjunct 1: without the opt-in nothing is nan-free, whatever the node is."""
        from graphistry.compute.gfql.expr_parser import Identifier, PropertyAccessExpr

        assert self._run(Identifier(name="score"), ["score"], False) is False
        assert self._run(
            PropertyAccessExpr(value=Identifier(name="a"), property="score"),
            ["a.score", "a"], False) is False

    @requires_polars
    def test_bare_identifier_arm(self):
        from graphistry.compute.gfql.expr_parser import Identifier

        assert self._run(Identifier(name="score"), ["score"], True) is True
        assert self._run(Identifier(name="absent"), ["score"], True) is False

    @requires_polars
    def test_property_access_arm(self):
        """A PREFIXED row table: `a.score` resolves, `b.score` does not, and a property
        access whose base is not a plain Identifier is not a column read at all."""
        from graphistry.compute.gfql.expr_parser import Identifier, Literal, PropertyAccessExpr

        cols = ["a.score", "a"]
        assert self._run(
            PropertyAccessExpr(value=Identifier(name="a"), property="score"), cols, True) is True
        assert self._run(
            PropertyAccessExpr(value=Identifier(name="b"), property="score"), cols, True) is False
        assert self._run(
            PropertyAccessExpr(value=Literal(value=1), property="score"), cols, True) is False

    @requires_polars
    def test_computed_and_missing_nodes_are_never_nan_free(self):
        from graphistry.compute.gfql.expr_parser import BinaryOp, Identifier, Literal

        computed = BinaryOp(op="/", left=Identifier(name="score"), right=Identifier(name="other"))
        assert self._run(computed, ["score", "other"], True) is False
        assert self._run(Literal(value=1.0), ["score"], True) is False
        assert self._run(None, ["score"], True) is False


class TestSingleAliasLoweringMemo:
    """#1832 follow-up: the lowering is memoized, and the key is complete.

    A stale key here is a silent wrong answer, so the negative cases (dtype change, column
    change, alias change, opt-in change) matter more than the positive one.
    """

    @requires_polars
    def test_same_key_returns_the_identical_expr(self):
        from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as rp

        schema = dict(_float_nodes().schema)
        a = rp.lower_single_alias_predicate("(a.score >= 1.0)", "a", schema)
        b = rp.lower_single_alias_predicate("(a.score >= 1.0)", "a", dict(schema))
        assert a is b, "memo did not hit for an identical key"

    @requires_polars
    def test_a_dtype_change_alone_gives_a_different_expr(self):
        """Same predicate, same column NAMES, different dtype -> the mask appears/disappears."""
        from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as rp

        float_schema = {"node_id": pl.Int64, "score": pl.Float64}
        int_schema = {"node_id": pl.Int64, "score": pl.Int64}
        f = rp.lower_single_alias_predicate("(a.score >= 1)", "a", float_schema)
        i = rp.lower_single_alias_predicate("(a.score >= 1)", "a", int_schema)
        assert f is not None and i is not None
        assert "is_nan" in str(f) and "is_nan" not in str(i), (
            f"dtype not reflected in the memo key: float={f} int={i}")

    @requires_polars
    def test_a_column_set_change_gives_a_different_result(self):
        from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as rp

        present = rp.lower_single_alias_predicate(
            "(a.score >= 1.0)", "a", {"node_id": pl.Int64, "score": pl.Float64})
        absent = rp.lower_single_alias_predicate(
            "(a.score >= 1.0)", "a", {"node_id": pl.Int64})
        assert present is not None
        assert absent is None, "an absent column must still decline under the memo"

    @requires_polars
    def test_alias_and_optin_are_both_in_the_key(self):
        from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as rp

        schema = {"node_id": pl.Int64, "score": pl.Float64}
        assert rp.lower_single_alias_predicate("(a.score >= 1.0)", "b", schema) is None
        assert rp.lower_single_alias_predicate("(a.score >= 1.0)", "a", schema) is not None
        guarded = rp.lower_single_alias_predicate("(a.score >= 1)", "a", schema)
        free = rp.lower_single_alias_predicate(
            "(a.score >= 1)", "a", schema, columns_nan_free=True)
        assert "is_nan" in str(guarded) and "is_nan" not in str(free), (
            "columns_nan_free is missing from the memo key")

    @requires_polars
    def test_memo_matches_the_uncached_lowering_for_every_board_shape(self):
        """The memo is a cache, not a behaviour change: identical repr on every shape."""
        from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as rp

        schema = dict(_float_nodes().schema)
        shapes = [
            "(a.score >= 1.0)", "(a.score <= 2.0)", "(a.label = 'x')",
            "(tolower(a.label) = 'x')", "(a.node_id >= 2)", "(a.score IS NULL)",
            "(a.label IN ['x', 'y'])", "(a.score >= 1.0 AND a.label = 'x')",
            "(NOT (a.label = 'x'))", "((a.num / a.other) >= 1)",
            "(a.missing = 1)", "(b.score >= 1.0)",
        ]
        for s in shapes:
            for opt in (False, True):
                memo = rp.lower_single_alias_predicate(s, "a", schema, columns_nan_free=opt)
                raw = rp._lower_single_alias_predicate_uncached(s, "a", schema, opt)
                assert (memo is None) == (raw is None), f"{s} (opt={opt}): decline mismatch"
                if memo is not None:
                    assert str(memo) == str(raw), f"{s} (opt={opt}): {memo} != {raw}"

    @requires_polars
    def test_cache_is_bounded(self):
        from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as rp

        for i in range(rp._SINGLE_ALIAS_CACHE_MAX * 2 + 5):
            rp.lower_single_alias_predicate(
                f"(a.score >= {i}.0)", "a", {"node_id": pl.Int64, "score": pl.Float64})
        assert len(rp._SINGLE_ALIAS_CACHE) <= rp._SINGLE_ALIAS_CACHE_MAX

    @requires_polars
    def test_a_cached_expr_is_reusable_across_frames(self):
        """The memo hands the SAME pl.Expr to different frames; polars exprs are immutable
        plan fragments, so each frame must still get its own answer."""
        from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as rp

        schema = {"node_id": pl.Int64, "score": pl.Float64}
        f1 = pl.DataFrame({"node_id": [1, 2], "score": [0.5, 2.5]})
        f2 = pl.DataFrame({"node_id": [3, 4], "score": [5.5, 0.1]})
        e = rp.lower_single_alias_predicate("(a.score >= 1.0)", "a", schema)
        assert e is not None
        assert f1.filter(e)["node_id"].to_list() == [2]
        assert f2.filter(e)["node_id"].to_list() == [3]
        assert rp.lower_single_alias_predicate("(a.score >= 1.0)", "a", schema) is e
        assert f1.filter(e)["node_id"].to_list() == [2], "expr mutated by use"
