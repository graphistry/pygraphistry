"""Pins for the temporal/error-leak family: #1915 B-5/B-7/B-8 + A-4, #1880 temporal half.

Oracles are hand-computed from openCypher CIP2016-06-14 ("Comparability and equality"):

- "Temporal instant values with timezone (`DateTime` and `LocalTime`) are compared on a
  global timeline, as if the instants were normalized to UTC." -> same instant under
  different offsets IS equal (B-5).
- "Two given instants `a` and `b` are equal if any only if they are of the same type and
  neither of them is _before_ or _after_ the other." and "Temporal instant values are
  only comparable within types." -> `datetime(...) = localdatetime(...)` is false and
  their ordering is null (B-5, the both-engines-wrong case).

B-7/#1880: temporal-vs-string comparisons must never leak raw backend errors
(polars InvalidOperationError, numpy ufunc TypeError) — they either answer with the
pandas-parity row set or raise a typed GFQL error / typed engine decline.

B-8: non-reserved keywords are valid property names (`n.when`, `n.order`, ...).

A-4: UNION branches projecting the SAME names in a different order align by name
(Neo4j semantics; output keeps the first branch's order); different name sets stay
a typed decline.
"""
from __future__ import annotations

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError, GFQLSyntaxError, GFQLValidationError

try:
    import polars as pl
except ImportError:  # pragma: no cover - polars-lane file also runs in core lane
    pl = None  # type: ignore[assignment]

try:
    import cudf
except ImportError:
    cudf = None  # type: ignore[assignment]


def _nodes_pd() -> pd.DataFrame:
    ts = pd.to_datetime([
        "2020-06-15T08:30:00", "2021-06-15T08:30:00",
        "2022-01-01T00:00:00", "2019-01-01T00:00:00", None,
    ])
    return pd.DataFrame({
        "id": ["p", "q", "r", "s", "t"],
        "ts": ts,
        "ts_aw": ts.tz_localize("UTC"),
        "ts_aw_lag": ts.tz_localize("UTC") - pd.Timedelta(hours=1),
        "dur": pd.to_timedelta(["1 days", "2 days", "3 days", "4 days", None]),
        "when": [1, 2, 3, 4, 5],
        "i": [7, 8, 9, 10, 11],
    })


def _edges_pd() -> pd.DataFrame:
    return pd.DataFrame({"s": ["p"], "d": ["q"]})


def _graph(engine: str):
    nodes, edges = _nodes_pd(), _edges_pd()
    if engine == "polars":
        assert pl is not None
        return graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    if engine == "cudf":
        assert cudf is not None
        return graphistry.nodes(cudf.from_pandas(nodes), "id").edges(cudf.from_pandas(edges), "s", "d")
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _rows(g, query: str, engine: str) -> pd.DataFrame:
    out = g.gfql(query, engine=engine)._nodes
    if pl is not None and isinstance(out, pl.LazyFrame):
        out = out.collect()
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


def _ids(g, query: str, engine: str) -> list:
    out = _rows(g, query, engine)
    col = "n.id" if "n.id" in out.columns else out.columns[0]
    return sorted(out[col].tolist())


ENGINES = [
    "pandas",
    pytest.param("polars", marks=pytest.mark.skipif(pl is None, reason="polars not installed")),
    pytest.param("cudf", marks=pytest.mark.skipif(cudf is None, reason="cudf not installed")),
]


# ---------------------------------------------------------------------------
# B-5: literal temporal comparisons
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
class TestB5LiteralTemporalComparison:
    def test_same_instant_different_offsets_equal(self, engine):
        """CIP: zoned values compare 'on a global timeline'. Red-at-master on polars
        (text equality on rendered literals gave False)."""
        g = _graph(engine)
        out = _rows(g, "MATCH (n) WHERE n.id = 'p' RETURN "
                       "datetime('2020-01-02T05:00:00+05:00') = datetime('2020-01-02T00:00:00Z') AS eq", engine)
        assert out["eq"].tolist() == [True]

    def test_same_instant_different_offsets_not_unequal(self, engine):
        g = _graph(engine)
        out = _rows(g, "MATCH (n) WHERE n.id = 'p' RETURN "
                       "datetime('2020-01-02T05:00:00+05:00') <> datetime('2020-01-02T00:00:00Z') AS eq", engine)
        assert out["eq"].tolist() == [False]

    def test_offset_vs_offset_same_instant_equal(self, engine):
        g = _graph(engine)
        out = _rows(g, "MATCH (n) WHERE n.id = 'p' RETURN "
                       "datetime('2020-01-02T05:00:00+05:00') = datetime('2020-01-01T19:00:00-05:00') AS eq", engine)
        assert out["eq"].tolist() == [True]

    def test_distinct_instants_stay_unequal(self, engine):
        """Anti-vacuity: the fold is a real instant comparison, not a constant True."""
        g = _graph(engine)
        out = _rows(g, "MATCH (n) WHERE n.id = 'p' RETURN "
                       "datetime('2020-01-02T05:00:00+05:00') = datetime('2020-01-02T00:00:01Z') AS eq", engine)
        assert out["eq"].tolist() == [False]

    def test_same_type_ordering_answers(self, engine):
        """Ordering two zoned literals compares instants (polars declined this NIE before)."""
        g = _graph(engine)
        out = _rows(g, "MATCH (n) WHERE n.id = 'p' RETURN "
                       "datetime('2020-01-02T05:00:00+05:00') < datetime('2020-01-02T00:00:01Z') AS eq", engine)
        assert out["eq"].tolist() == [True]

    def test_zoned_vs_local_equality_is_false(self, engine):
        """CIP: equal 'if any only if they are of the same type ...'. Red-at-master on
        BOTH pandas and polars (each answered True)."""
        g = _graph(engine)
        out = _rows(g, "MATCH (n) WHERE n.id = 'p' RETURN "
                       "datetime('2020-01-02T00:00:00Z') = localdatetime('2020-01-02T00:00:00') AS eq", engine)
        assert out["eq"].tolist() == [False]
        out = _rows(g, "MATCH (n) WHERE n.id = 'p' RETURN "
                       "localdatetime('2020-01-02T00:00:00') = datetime('2020-01-02T00:00:00Z') AS eq", engine)
        assert out["eq"].tolist() == [False]

    def test_zoned_vs_local_inequality_is_true(self, engine):
        g = _graph(engine)
        out = _rows(g, "MATCH (n) WHERE n.id = 'p' RETURN "
                       "datetime('2020-01-02T00:00:00Z') <> localdatetime('2020-01-02T00:00:00') AS eq", engine)
        assert out["eq"].tolist() == [True]

    def test_zoned_vs_local_ordering_is_null(self, engine):
        """CIP: 'Temporal instant values are only comparable within types.'"""
        g = _graph(engine)
        out = _rows(g, "MATCH (n) WHERE n.id = 'p' RETURN "
                       "datetime('2020-01-02T00:00:00Z') < localdatetime('2020-01-03T00:00:00') AS eq", engine)
        assert out["eq"].isna().tolist() == [True]

    def test_where_form_is_row_set_visible(self, engine):
        """The audit's B-5 wrong answer changed row sets, not just projections."""
        g = _graph(engine)
        ids = _ids(g, "MATCH (n) WHERE datetime('2020-01-02T05:00:00+05:00') = datetime('2020-01-02T00:00:00Z') "
                      "RETURN n.id", engine)
        assert ids == ["p", "q", "r", "s", "t"]  # anti-vacuity: all 5 rows survive a true WHERE
        ids = _ids(g, "MATCH (n) WHERE datetime('2020-01-02T00:00:00Z') = localdatetime('2020-01-02T00:00:00') "
                      "RETURN n.id", engine)
        assert ids == []

    def test_plain_string_equality_untouched(self, engine):
        """Mutation guard: non-temporal string literals keep plain string semantics."""
        g = _graph(engine)
        out = _rows(g, "MATCH (n) WHERE n.id = 'p' RETURN 'a' = 'b' AS eq, 'a' = 'a' AS eq2", engine)
        assert out["eq"].tolist() == [False]
        assert out["eq2"].tolist() == [True]


class TestB5FoldUnits:
    """Direct pins on the fold — the arms end-to-end queries cannot isolate."""

    def _fold(self, op: str, left: str, right: str):
        from graphistry.compute.gfql.expr_parser import BinaryOp, Literal
        from graphistry.compute.gfql.temporal.folding import _fold_temporal_comparison
        return _fold_temporal_comparison(BinaryOp(op=op, left=Literal(left), right=Literal(right)))

    def test_same_type_instant_comparison(self):
        assert self._fold("=", "2020-01-02T05:00:00+05:00", "2020-01-02T00:00:00Z").value is True
        assert self._fold("<", "2020-01-01", "2020-01-02").value is True
        assert self._fold(">=", "12:00:00", "12:00:01").value is False

    def test_cross_type_matrix(self):
        """Every distinct-kind pair: eq false, neq true, ordering null."""
        exemplars = {
            "datetime": "2020-01-02T00:00:00Z",
            "localdatetime": "2020-01-02T00:00:00",
            "date": "2020-01-02",
            "time": "12:00:00Z",
            "localtime": "12:00:00",
        }
        kinds = list(exemplars)
        checked = 0
        for i, a in enumerate(kinds):
            for b in kinds[i + 1:]:
                assert self._fold("=", exemplars[a], exemplars[b]).value is False, (a, b)
                assert self._fold("<>", exemplars[a], exemplars[b]).value is True, (a, b)
                assert self._fold("<", exemplars[a], exemplars[b]).value is None, (a, b)
                checked += 1
        assert checked == 10  # anti-vacuity: all C(5,2) pairs exercised

    def test_zone_name_without_offset_declines_fold(self):
        from graphistry.compute.gfql.temporal.values import _parse_temporal_value
        from graphistry.compute.gfql.temporal.folding import _temporal_instant_key
        value = _parse_temporal_value("2020-01-02T00:00:00[Europe/Paris]")
        assert value is not None and _temporal_instant_key(value) is None

    def test_offset_with_zone_name_still_folds(self):
        assert self._fold("=", "2020-01-02T02:00:00+02:00[Europe/Paris]", "2020-01-02T00:00:00Z").value is True

    def test_non_temporal_strings_do_not_fold(self):
        assert self._fold("=", "hello", "2020-01-02") is None
        assert self._fold("=", "2020-01-02", "world") is None

    def test_durations_do_not_fold(self):
        assert self._fold("=", "P1D", "PT24H") is None

    def test_seconds_offset_parsed(self):
        assert self._fold("=", "2020-01-02T00:00:30+00:00:30", "2020-01-02T00:00:00Z").value is True


# ---------------------------------------------------------------------------
# B-7 / #1880: temporal-vs-string never leaks raw backend errors
# ---------------------------------------------------------------------------

class TestB7TemporalStringLeaks:
    def test_pandas_zoned_string_vs_naive_column_answers(self):
        """Red-at-master: raw numpy `ufunc 'bitwise_and'` TypeError from the pushdown."""
        g = _graph("pandas")
        assert _ids(g, "MATCH (n) WHERE n.ts > '2021-01-01T00:00:00Z' RETURN n.id", "pandas") == ["q", "r"]

    def test_pandas_zoned_string_equality_matches_instant(self):
        """Red-at-master: the pushed equality silently matched ZERO rows."""
        g = _graph("pandas")
        assert _ids(g, "MATCH (n) WHERE n.ts = '2021-06-15T08:30:00Z' RETURN n.id", "pandas") == ["q"]

    @pytest.mark.skipif(cudf is None, reason="cudf not installed")
    def test_cudf_zoned_string_vs_naive_column_answers(self):
        g = _graph("cudf")
        assert _ids(g, "MATCH (n) WHERE n.ts > '2021-01-01T00:00:00Z' RETURN n.id", "cudf") == ["q", "r"]

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_polars_zoned_string_vs_naive_column_typed_decline(self):
        """The where_rows residual declines typed on polars — never the raw
        InvalidOperationError this leaked at master."""
        g = _graph("polars")
        with pytest.raises(NotImplementedError, match="where_rows"):
            g.gfql("MATCH (n) WHERE n.ts > '2021-01-01T00:00:00Z' RETURN n.id", engine="polars")

    @pytest.mark.parametrize("engine", [
        "pandas",
        pytest.param("polars", marks=pytest.mark.skipif(pl is None, reason="polars not installed")),
    ])
    def test_naive_string_vs_datetime_column_parity(self, engine):
        """#1880: `a.dt = '...'` leaked a raw polars InvalidOperationError; both engines
        must answer the same rows."""
        g = _graph(engine)
        assert _ids(g, "MATCH (n) WHERE n.ts = '2021-06-15T08:30:00' RETURN n.id", engine) == ["q"]
        assert _ids(g, "MATCH (n) WHERE n.ts > '2021-01-01T00:00:00' RETURN n.id", engine) == ["q", "r"]

    @pytest.mark.parametrize("engine", [
        "pandas",
        pytest.param("polars", marks=pytest.mark.skipif(pl is None, reason="polars not installed")),
    ])
    def test_string_vs_duration_column_parity(self, engine):
        """#1880: `a.dur = '1 days'` leaked a raw polars cast InvalidOperationError."""
        g = _graph(engine)
        assert _ids(g, "MATCH (n) WHERE n.dur = '1 days' RETURN n.id", engine) == ["p"]
        assert _ids(g, "MATCH (n) WHERE n.dur > '2 days' RETURN n.id", engine) == ["r", "s"]

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_polars_unparseable_string_typed_schema_error(self):
        """Fail-closed (#1880): what cannot compare declines with the SAME typed error
        family the scalar half raises (E302), never a raw polars exception."""
        from graphistry.compute.ast import n
        from graphistry.compute.predicates.comparison import gt
        g = _graph("polars")
        for chain in ([n({"ts": "not-a-timestamp"})], [n({"ts": gt("not-a-timestamp")})],
                      [n({"dur": "not-a-duration"})], [n({"ts": gt("2021-01-01T00:00:00Z")})]):
            with pytest.raises(GFQLSchemaError) as excinfo:
                g.gfql(chain, engine="polars")
            assert excinfo.value.code == ErrorCode.E302

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_polars_chain_predicate_parity_rows(self):
        """Anti-vacuity for the parse-and-compare lowering: real rows, pandas-identical."""
        from graphistry.compute.ast import n
        from graphistry.compute.predicates.comparison import gt, le
        for chain, expect in (
            ([n({"ts": gt("2021-01-01T00:00:00")})], ["q", "r"]),
            ([n({"ts": le("2020-06-15T08:30:00")})], ["p", "s"]),
            ([n({"dur": "2 days"})], ["q"]),
        ):
            got = {}
            for engine in ("pandas", "polars"):
                out = _graph(engine).gfql(chain, engine=engine)._nodes
                if isinstance(out, pl.LazyFrame):
                    out = out.collect()
                if hasattr(out, "to_pandas"):
                    out = out.to_pandas()
                got[engine] = sorted(out["id"].tolist())
            assert got["pandas"] == got["polars"] == expect

    def test_pandas_connected_join_zoned_string_answers(self):
        """The connected-join lowering must also keep tz-suffixed text a residual."""
        g = _graph("pandas")
        out = _rows(g, "MATCH (a)-[]->(b) WHERE b.ts > '2021-01-01T00:00:00Z' RETURN a.id, b.id",
                    "pandas")
        assert out.to_dict("records") == [{"a.id": "p", "b.id": "q"}]

    def test_pandas_optional_match_zoned_string_answers(self):
        """Red-at-master: the connected OPTIONAL MATCH lowering pushed the tz-suffixed
        literal into a filter dict too, leaking the same raw numpy TypeError."""
        g = _graph("pandas")
        out = _rows(g, "MATCH (a) WHERE a.id IN ['p', 'q'] OPTIONAL MATCH (a)-[]->(b) "
                       "WHERE b.ts > '2021-01-01T00:00:00Z' RETURN a.id, b.id", "pandas")
        out = out.where(out.notna(), None)
        got = sorted(out.to_dict("records"), key=lambda r: r["a.id"])
        assert got == [{"a.id": "p", "b.id": "q"}, {"a.id": "q", "b.id": None}]

    @pytest.mark.parametrize("engine", [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.skipif(cudf is None, reason="cudf not installed")),
    ])
    def test_naive_vs_aware_columns_answer(self, engine):
        """Red-at-master: raw pandas TypeError ('Invalid comparison ...') from the
        same-path WHERE. GFQL's extension reads naive datetimes as UTC, so the
        equal-instant pair yields no rows and the lagged pair yields all non-null rows."""
        g = _graph(engine)
        assert _ids(g, "MATCH (n) WHERE n.ts > n.ts_aw RETURN n.id", engine) == []
        # mutation guard: the comparison is real, not constant-empty
        assert _ids(g, "MATCH (n) WHERE n.ts > n.ts_aw_lag RETURN n.id", engine) == ["p", "q", "r", "s"]

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_polars_naive_vs_aware_columns_typed_decline(self):
        g = _graph("polars")
        with pytest.raises(NotImplementedError, match="polars engine does not yet natively support"):
            g.gfql("MATCH (n) WHERE n.ts > n.ts_aw RETURN n.id", engine="polars")


class TestB7Units:
    def test_zoned_iso_regex_accepts_and_rejects(self):
        from graphistry.compute.gfql.cypher.lowering import _ZONED_ISO_TEMPORAL_TEXT_RE as rx
        for text in ("2021-01-01T00:00:00Z", "2021-01-01T00:00:00+05:00", "2021-01-01 00:00:00-0500",
                     "2021-01-01T00:00Z", "12:30:00Z", "12:30:00.5+02:00"):
            assert rx.match(text), text
        for text in ("2021-01-01T00:00:00", "2021-01-01", "1 days", "P1D", "hello",
                     "12:30:00", "2021-01-01T00:00:00Zx"):
            assert not rx.match(text), text

    def test_align_mixed_tz_converts_only_mixed_datetime_pairs(self):
        from graphistry.compute.gfql.same_path.df_utils import _align_mixed_tz_datetimes
        naive = pd.Series(pd.to_datetime(["2021-01-01"]))
        aware = pd.Series(pd.to_datetime(["2021-01-01"]).tz_localize("US/Eastern"))
        left, right = _align_mixed_tz_datetimes(naive, aware)
        assert right.dt.tz is None and right.iloc[0] == pd.Timestamp("2021-01-01T05:00:00")
        assert left is naive
        left, right = _align_mixed_tz_datetimes(aware, naive)
        assert left.dt.tz is None and right is naive
        ints = pd.Series([1])
        assert _align_mixed_tz_datetimes(ints, naive) == (ints, naive)
        left, right = _align_mixed_tz_datetimes(naive, naive)
        assert left is naive and right is naive

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_parse_temporal_filter_scalar_safe_subset(self):
        import datetime as dt
        from graphistry.compute.gfql.lazy.engine.polars.predicates import _parse_temporal_filter_scalar
        assert _parse_temporal_filter_scalar("2021-01-01T00:00:00", pl.Datetime("ns")) == dt.datetime(2021, 1, 1)
        assert _parse_temporal_filter_scalar("2021-01-01T00:00:00Z", pl.Datetime("ns")) is None  # tz-suffixed
        assert _parse_temporal_filter_scalar("2021-01-01T00:00:00.000000001", pl.Datetime("ns")) is None  # sub-us
        assert _parse_temporal_filter_scalar("junk", pl.Datetime("ns")) is None
        assert _parse_temporal_filter_scalar("2021-01-01", pl.Datetime("ns", "UTC")) is None  # aware column
        assert _parse_temporal_filter_scalar("1 days", pl.Duration("ns")) == dt.timedelta(days=1)
        assert _parse_temporal_filter_scalar("junk", pl.Duration("ns")) is None
        assert _parse_temporal_filter_scalar("2021-01-01", pl.Date) == dt.date(2021, 1, 1)
        assert _parse_temporal_filter_scalar("12:30:00", pl.Time) == dt.time(12, 30)
        assert _parse_temporal_filter_scalar("2021-01-01", pl.Int64()) is None


# ---------------------------------------------------------------------------
# B-8: non-reserved keywords as property names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
class TestB8KeywordPropertyNames:
    def test_where_on_keyword_property(self, engine):
        """Red-at-master: `n.when > 3` raised GFQLSyntaxError."""
        g = _graph(engine)
        assert _ids(g, "MATCH (n) WHERE n.when > 3 RETURN n.id", engine) == ["s", "t"]

    def test_return_and_order_by_keyword_property(self, engine):
        g = _graph(engine)
        out = _rows(g, "MATCH (n) RETURN n.when AS w ORDER BY n.when DESC LIMIT 2", engine)
        assert out["w"].tolist() == [5, 4]

    def test_property_map_keyword_key(self, engine):
        g = _graph(engine)
        assert _ids(g, "MATCH (n {when: 4}) RETURN n.id", engine) == ["s"]


class TestB8Grammar:
    @pytest.mark.parametrize("prop", ["when", "then", "end", "order", "is", "all", "any", "contains"])
    def test_issue_keywords_parse_in_where(self, prop):
        """Every keyword the audit listed parses as a property name (pandas run)."""
        nodes = pd.DataFrame({"id": ["a", "b"], prop: [1, 5]})
        g = graphistry.nodes(nodes, "id").edges(_edges_pd(), "s", "d")
        out = g.gfql(f"MATCH (n) WHERE n.{prop} > 3 RETURN n.id", engine="pandas")._nodes
        assert out["n.id"].tolist() == ["b"]

    def test_keyword_property_keeps_filter_pushdown(self):
        """The WHERE-chain grammar (property_ref) must also accept keyword property
        names, or the conjunct silently loses its filter_dict pushdown."""
        import warnings
        from graphistry.compute.ast import ASTNode
        from graphistry.compute.gfql.cypher.api import compile_cypher
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            compiled = compile_cypher("MATCH (n) WHERE n.when > 3 RETURN n.id")
        node_ops = [op for op in compiled.chain.chain if isinstance(op, ASTNode)]
        assert node_ops and node_ops[0].filter_dict and "when" in node_ops[0].filter_dict

    def test_keywords_stay_reserved_outside_property_position(self):
        """Mutation guard: only the dot/map-key position was unreserved."""
        g = _graph("pandas")
        with pytest.raises(GFQLSyntaxError):
            g.gfql("MATCH (n) RETURN when", engine="pandas")


# ---------------------------------------------------------------------------
# A-4: UNION name alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
class TestA4UnionNameAlignment:
    def test_same_names_different_order_align(self, engine):
        """Red-at-master: typed decline. Neo4j aligns by name; the output keeps the
        first branch's column order."""
        g = _graph(engine)
        out = _rows(g, "MATCH (n) WHERE n.id='p' RETURN n.id AS a, n.i AS b "
                       "UNION MATCH (n) WHERE n.id='q' RETURN n.i AS b, n.id AS a", engine)
        assert list(out.columns) == ["a", "b"]
        assert sorted(out.to_dict("records"), key=lambda r: r["a"]) == [
            {"a": "p", "b": 7}, {"a": "q", "b": 8},
        ]

    def test_union_all_alignment(self, engine):
        g = _graph(engine)
        out = _rows(g, "MATCH (n) WHERE n.id='p' RETURN n.id AS a, n.i AS b "
                       "UNION ALL MATCH (n) WHERE n.id='p' RETURN n.i AS b, n.id AS a", engine)
        assert list(out.columns) == ["a", "b"]
        assert out.to_dict("records") == [{"a": "p", "b": 7}, {"a": "p", "b": 7}]

    def test_different_name_sets_stay_typed_decline(self, engine):
        """The genuine error half of the old message survives, by name."""
        g = _graph(engine)
        with pytest.raises(GFQLValidationError, match="must project the same output names"):
            g.gfql("MATCH (n) RETURN n.id AS a UNION MATCH (n) RETURN n.i AS c", engine=engine)
