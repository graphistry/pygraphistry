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
        expr = fp._residual_polars_expr("(tolower(a.name) = tolower('ALICE'))", "a", COLS())
        assert expr is not None
        out = _pl_nodes().filter(expr)
        assert sorted(out["node_id"].to_list()) == [1, 2]

    @requires_polars
    def test_tolower_eq_null_dropped(self):
        expr = fp._residual_polars_expr("(tolower(a.name) = tolower('bob'))", "a", COLS())
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
        "(tolower(a.name) = 'x')",      # rhs not tolower-wrapped
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
        exprs = ["(tolower(a.name) = tolower('Alice'))", "(a.age >= 25)"]
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
        exprs = ["(a.age >= 25)", "(tolower(a.name) = tolower('alice'))"]
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
            g, nodes, "a", ["(tolower(a.name) = tolower('alice'))"], "node_id",
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
            "(tolower(a.name) = tolower('It\\u0027s'))", "a", COLS()) is None
        assert fp._residual_polars_expr(
            "(a.name = 'C:\\u005Cx')", "a", COLS()) is None

    @requires_polars
    @pytest.mark.parametrize("expr", [
        "(a.age = 'thirty')",           # string literal vs numeric column
        "(tolower(a.age) = tolower('x'))",  # tolower on numeric column
        "(a.name >= 25)",               # numeric literal vs string column
    ])
    def test_dtype_mismatch_declines(self, expr):
        assert fp._residual_polars_expr(expr, "a", COLS()) is None

    @requires_polars
    def test_categorical_column_declines(self):
        nodes = pl.DataFrame({"node_id": [1], "cat": ["x"]}).with_columns(
            pl.col("cat").cast(pl.Categorical))
        assert fp._residual_polars_expr(
            "(tolower(a.cat) = tolower('x'))", "a", dict(nodes.schema)) is None
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
