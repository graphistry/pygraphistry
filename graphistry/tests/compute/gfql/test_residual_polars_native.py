"""#1729/#1755: native polars translation of simple connected-join residuals.

Covers `_residual_polars_expr` (the string→pl.Expr translator) and the fast-lane /
chain-fallback split in `_connected_join_apply_node_residuals`:
- positive: every covered shape translates and filters byte-identically to the
  chain fallback (the previous behavior), including nulls and case folding
- negative: unsupported shapes, alias mismatches, and absent columns decline
  (translator returns None); a group with ANY untranslatable expr falls back whole
- cross-engine: pandas frames never enter the fast lane (chain fallback only)
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
        "(a.name <> 'x')",              # unsupported operator
        "(a.name CONTAINS 'x')",        # unsupported predicate
        "(tolower(a.name) = tolower('x'))",   # two-sided: the lowering folds it away first
        "(tolower(b.name) = 'alice')",  # alias mismatch (checked with alias='a')
        "(toupper(b.name) = 'ALICE')",  # alias mismatch, other case fn
        "(upper(zz.name) = 'ALICE')",   # alias mismatch, GQL alias spelling
        "('x' = tolower(a.name))",      # reversed operand order
        "(tolower(a.name) = b.name)",   # rhs is a column, not a literal
        "(tolower(a.name) = 25)",       # rhs is a number, not a string literal
        "(substring(a.name, 0, 2) = 'x')",  # a foldable fn on the COLUMN is not this shape
        "((a.age = 25) AND (a.age = 30))",  # compound
        "a.age = 25",                   # missing outer parens
        "(b.age = 25)",                 # alias mismatch (checked with alias='a')
        "(a.missing = 25)",             # absent column
    ])
    def test_unsupported_shapes_decline(self, bad):
        assert fp._residual_polars_expr(bad, "a", COLS()) is None


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
    """Review-skill wave (#1763): escaped literals + dtype mismatches must DECLINE
    so the chain fallback keeps the evaluator's exact semantics (unescaping, or the
    designed parity-or-error NotImplementedError) instead of raw polars behavior."""

    @requires_polars
    def test_escaped_literal_declines(self):
        # renderer escapes ' \\ \n etc to \uXXXX text; raw regex compare would mismatch
        assert fp._residual_polars_expr(
            "(tolower(a.name) = 'it\\u0027s')", "a", COLS()) is None
        assert fp._residual_polars_expr(
            "(a.name = 'C:\\u005Cx')", "a", COLS()) is None

    @requires_polars
    @pytest.mark.parametrize("expr", [
        "(a.age = 'thirty')",           # string literal vs numeric column
        "(tolower(a.age) = 'x')",       # tolower on numeric column
        "(a.name >= 25)",               # numeric literal vs string column
    ])
    def test_dtype_mismatch_declines(self, expr):
        assert fp._residual_polars_expr(expr, "a", COLS()) is None

    @requires_polars
    def test_categorical_column_declines(self):
        nodes = pl.DataFrame({"node_id": [1], "cat": ["x"]}).with_columns(
            pl.col("cat").cast(pl.Categorical))
        assert fp._residual_polars_expr(
            "(tolower(a.cat) = 'x')", "a", dict(nodes.schema)) is None
        assert fp._residual_polars_expr("(a.cat = 'x')", "a", dict(nodes.schema)) is None

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
    def test_non_ascii_two_sided_declines_the_lane_but_not_the_answer(self, monkeypatch):
        """DISCLOSED NARROWING. A non-ASCII two-sided literal is outside the region
        where the engines provably agree, so the fold DECLINES, the residual stays
        two-sided, and the fused lane declines with it. That costs speed on an exotic
        shape and buys back a Python-vs-Rust case-table assumption master was making
        silently. The ANSWER must be unchanged -- that is what is asserted."""
        g = self._star_graph()
        q = self.Q_ONE_SIDED.replace("toLower(i.interest) = 'fine dining'",
                                     "toLower(i.interest) = toLower('FINE DINİNG')")
        calls = self._spy_fused(monkeypatch)
        declined = g.gfql(q, engine="polars")
        assert not any(calls), "non-ASCII two-sided residual unexpectedly served the fused lane"
        monkeypatch.setattr(fp, "_residual_polars_expr", lambda *a, **k: None)
        assert self._rows(declined) == self._rows(g.gfql(q, engine="polars"))
