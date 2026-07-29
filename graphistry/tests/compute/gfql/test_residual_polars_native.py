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
