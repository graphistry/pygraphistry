"""searchAny renders datetimes as the viz inspector does, so you search what you can see.

Expected strings are not hand-written: they are what real ``moment-timezone`` produces for
``'MMM D YYYY, h:mm:ss a z'``, the format the inspector's ``formatDate`` uses.
"""

import pandas as pd
import pytest

from graphistry.compute.gfql.search_any import search_any_mask, search_candidate_columns
from graphistry.compute.gfql.wysiwyg import DEFAULT_TEMPORAL_TZ, render_datetime_pandas


def dt_series(*stamps):
    return pd.Series(pd.to_datetime(list(stamps)))


# (utc stamp, tz, what moment-timezone renders)
MOMENT_CASES = [
    ("2024-01-05T03:04:05", "UTC", "Jan 5 2024, 3:04:05 am UTC"),
    ("2024-01-05T15:04:05", "UTC", "Jan 5 2024, 3:04:05 pm UTC"),
    ("2024-01-05T00:00:00", "UTC", "Jan 5 2024, 12:00:00 am UTC"),
    ("2024-01-05T12:00:00", "UTC", "Jan 5 2024, 12:00:00 pm UTC"),
    ("2024-12-25T23:59:59", "UTC", "Dec 25 2024, 11:59:59 pm UTC"),
    ("1999-11-30T13:45:07", "UTC", "Nov 30 1999, 1:45:07 pm UTC"),
    ("2024-02-29T06:07:08", "UTC", "Feb 29 2024, 6:07:08 am UTC"),
    ("2024-01-05T03:04:05", "America/New_York", "Jan 4 2024, 10:04:05 pm EST"),
    ("2024-07-04T09:05:00", "America/New_York", "Jul 4 2024, 5:05:00 am EDT"),
    ("2024-01-05T03:04:05", "Asia/Kolkata", "Jan 5 2024, 8:34:05 am IST"),
    ("2024-06-01T12:00:00", "Europe/London", "Jun 1 2024, 1:00:00 pm BST"),
    ("2024-01-01T12:00:00", "Europe/London", "Jan 1 2024, 12:00:00 pm GMT"),
]


@pytest.mark.parametrize("stamp,tz,expected", MOMENT_CASES,
                         ids=[f"{tz}-{s}" for s, tz, _ in MOMENT_CASES])
def test_render_matches_moment(stamp, tz, expected):
    assert render_datetime_pandas(dt_series(stamp), tz).iloc[0] == expected


def test_midnight_and_noon_are_twelve_not_zero():
    """moment's `h` is a 12-hour clock: hour 0 shows as 12, and so does hour 12."""
    rendered = render_datetime_pandas(dt_series("2024-01-05T00:00:00", "2024-01-05T12:00:00"))
    assert [r.split(", ")[1].split(":")[0] for r in rendered] == ["12", "12"]


def test_day_and_hour_carry_no_leading_zero_but_minute_and_second_do():
    rendered = render_datetime_pandas(dt_series("2024-01-05T03:04:05")).iloc[0]
    assert rendered == "Jan 5 2024, 3:04:05 am UTC"


def test_null_renders_as_nothing_and_never_matches():
    df = pd.DataFrame({"when": dt_series("2024-01-05T03:04:05", None)})
    assert render_datetime_pandas(df["when"]).tolist()[1] is None
    assert search_any_mask(df, "2024").tolist() == [True, False]


def test_an_empty_column_renders_empty():
    assert render_datetime_pandas(pd.Series([], dtype="datetime64[ns]")).tolist() == []


def test_a_tz_aware_column_is_converted_not_rejected():
    aware = pd.Series(pd.to_datetime(["2024-01-05T03:04:05Z"]))
    assert render_datetime_pandas(aware).iloc[0] == "Jan 5 2024, 3:04:05 am UTC"


def test_datetime_columns_are_searched_only_for_a_numeric_term():
    """The inspector's shouldSearch fires on date columns iff the term is numeric."""
    df = pd.DataFrame({"when": dt_series("2024-01-05T03:04:05")})
    assert search_candidate_columns(df, "2024", None) == ["when"]
    assert search_candidate_columns(df, "jan", None) == []


def test_the_term_matches_the_rendered_text():
    df = pd.DataFrame({"when": dt_series("2024-01-05T03:04:05", "2023-07-04T09:05:00")})
    assert search_any_mask(df, "2024").tolist() == [True, False]
    assert search_any_mask(df, "2023").tolist() == [False, True]
    assert search_any_mask(df, "9999").tolist() == [False, False]


def test_the_timezone_changes_which_rows_match():
    """tz is not cosmetic: it decides which day a timestamp lands on, so it decides matches."""
    df = pd.DataFrame({"when": dt_series("2024-01-01T02:00:00")})
    assert search_any_mask(df, "2024", temporal_tz="UTC").tolist() == [True]
    assert search_any_mask(df, "2023", temporal_tz="UTC").tolist() == [False]
    # two hours after midnight UTC is still 2023 in New York
    assert search_any_mask(df, "2023", temporal_tz="America/New_York").tolist() == [True]


def test_default_tz_is_utc():
    assert DEFAULT_TEMPORAL_TZ == "UTC"
    df = pd.DataFrame({"when": dt_series("2024-01-01T02:00:00")})
    assert (search_any_mask(df, "2023").tolist()
            == search_any_mask(df, "2023", temporal_tz="UTC").tolist())


class TestCudf:
    """cuDF matches pandas at UTC and declines any other zone rather than mislabel it."""

    def test_cudf_matches_pandas_at_utc(self):
        cudf = pytest.importorskip("cudf", reason="cuDF lane needs a GPU box")
        from graphistry.compute.gfql.wysiwyg import render_datetime_cudf
        stamps = ["2024-01-05T03:04:05", "2024-01-05T00:00:00", "2024-12-25T23:59:59"]
        want = render_datetime_pandas(pd.Series(pd.to_datetime(stamps))).tolist()
        got = render_datetime_cudf(cudf.Series(cudf.to_datetime(stamps))).to_pandas().tolist()
        assert got == want

    def test_cudf_declines_a_non_utc_zone(self):
        cudf = pytest.importorskip("cudf", reason="cuDF lane needs a GPU box")
        from graphistry.compute.gfql.wysiwyg import (
            CudfTemporalTzUnsupported, render_datetime_cudf)
        s = cudf.Series(cudf.to_datetime(["2024-01-05T03:04:05"]))
        with pytest.raises(CudfTemporalTzUnsupported):
            render_datetime_cudf(s, "America/New_York")


class TestPolars:
    """polars renders natively, so the same query answers the same rows as pandas."""

    def test_polars_render_matches_pandas(self):
        pl = pytest.importorskip("polars")
        from graphistry.compute.gfql.wysiwyg import datetime_render_expr_polars

        stamps = ["2024-01-05T03:04:05", "2024-01-05T00:00:00", "2024-01-05T12:00:00",
                  "2024-12-25T23:59:59", "1999-11-30T13:45:07"]
        want = render_datetime_pandas(pd.Series(pd.to_datetime(stamps))).tolist()
        df = pl.DataFrame({"t": pl.Series(stamps).str.to_datetime()})
        got = df.select(
            datetime_render_expr_polars(pl.col("t"), df["t"].dtype).alias("o"))["o"].to_list()
        assert got == want

    def test_polars_render_honours_the_timezone(self):
        pl = pytest.importorskip("polars")
        from graphistry.compute.gfql.wysiwyg import datetime_render_expr_polars

        df = pl.DataFrame({"t": pl.Series(["2024-01-05T03:04:05"]).str.to_datetime()})
        got = df.select(datetime_render_expr_polars(
            pl.col("t"), df["t"].dtype, "America/New_York").alias("o"))["o"][0]
        assert got == "Jan 4 2024, 10:04:05 pm EST"

    def test_polars_search_answers_the_same_rows_as_pandas_end_to_end(self):
        """The divergence this guards against is silent: polars used to skip datetime
        columns and answer all-False where pandas answered a match."""
        pl = pytest.importorskip("polars")
        import graphistry
        from graphistry.compute.ast import n, search_any as search_any_op

        stamps = ["2024-01-05T03:04:05", "2023-07-04T09:05:00", "2024-12-25T23:59:59"]
        edges_pd = pd.DataFrame({"s": [0, 1], "d": [1, 2]})
        nodes_pd = pd.DataFrame({"id": [0, 1, 2], "when": pd.to_datetime(stamps)})
        nodes_pl = pl.DataFrame({"id": [0, 1, 2], "when": pl.Series(stamps).str.to_datetime()})
        edges_pl = pl.DataFrame({"s": [0, 1], "d": [1, 2]})

        def hits(nodes, edges, engine, term):
            g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
            out = g.gfql([n(name="a"), search_any_op(alias="a", out_col="__hit__", term=term)],
                         engine=engine)._nodes
            out = out.to_pandas() if hasattr(out, "to_pandas") else out
            return out.sort_values("id")["__hit__"].tolist()

        for term in ["2024", "2023", "05", "9999"]:
            assert (hits(nodes_pl, edges_pl, "polars", term)
                    == hits(nodes_pd, edges_pd, "pandas", term)), f"term={term!r}"

    def test_polars_converts_a_tz_aware_column_instead_of_relabelling_it(self):
        """Stamping UTC onto a column that already carries a zone moves the instant."""
        pl = pytest.importorskip("polars")
        from graphistry.compute.gfql.wysiwyg import datetime_render_expr_polars

        df = pl.DataFrame({"t": pl.Series(["2024-01-05T03:04:05"]).str.to_datetime()
                           .dt.replace_time_zone("America/New_York")})
        got = df.select(
            datetime_render_expr_polars(pl.col("t"), df["t"].dtype).alias("o"))["o"][0]
        assert got == "Jan 5 2024, 8:04:05 am UTC", "03:04 in New York is 08:04 UTC"

    def test_polars_date_columns_render_at_midnight_like_pandas(self):
        """A polars Date has no time; pandas widens the same value to midnight."""
        pl = pytest.importorskip("polars")
        from graphistry.compute.gfql.wysiwyg import datetime_render_expr_polars

        df = pl.DataFrame({"t": pl.Series(["2024-01-05"]).str.to_date()})
        got = df.select(
            datetime_render_expr_polars(pl.col("t"), df["t"].dtype).alias("o"))["o"][0]
        assert got == render_datetime_pandas(pd.Series(pd.to_datetime(["2024-01-05"]))).iloc[0]

    def test_polars_null_rows_render_as_nothing(self):
        pl = pytest.importorskip("polars")
        from graphistry.compute.gfql.wysiwyg import datetime_render_expr_polars

        df = pl.DataFrame({"t": pl.Series(["2024-01-05T03:04:05", None]).str.to_datetime()})
        got = df.select(
            datetime_render_expr_polars(pl.col("t"), df["t"].dtype).alias("o"))["o"].to_list()
        assert got[1] is None


@pytest.mark.parametrize("term,label", [("2024", "numeric"), ("alpha", "alphabetic")])
def test_the_polars_dtype_gate_agrees_with_the_pandas_one(term, label):
    """The two engines pick the same columns, or one of them answers rows the other cannot.

    `lazy/engine/polars/search.py::auto_search_columns` restates the pandas auto-gate rather
    than sharing it, so the two drift apart whenever a dtype is added to one. That drift is
    silent -- the engine that skips a column answers all-False against the other's match --
    and it has happened for float and again for datetime. This is the pin that catches it.
    """
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql.lazy.engine.polars.search import auto_search_columns

    pdf = pd.DataFrame({
        "s_str": ["alpha", "beta"],
        "i_int": [1, 2],
        "f_float": [1.5, 2.5],
        "b_bool": [True, False],
        "t_datetime": pd.to_datetime(["2024-01-05", "2023-07-04"]),
    })
    pldf = pl.DataFrame({
        "s_str": ["alpha", "beta"],
        "i_int": pl.Series([1, 2], dtype=pl.Int64),
        "f_float": pl.Series([1.5, 2.5], dtype=pl.Float64),
        "b_bool": [True, False],
        "t_datetime": pl.Series(["2024-01-05", "2023-07-04"]).str.to_datetime(),
    })

    from_pandas = set(search_candidate_columns(pdf, term, None) or [])
    schema = dict(zip(pldf.columns, pldf.dtypes))
    from_polars = set(auto_search_columns(schema, list(pldf.columns), term) or [])

    assert from_pandas == from_polars, (
        f"{label} term {term!r}: pandas chose {sorted(from_pandas)}, "
        f"polars chose {sorted(from_polars)} -- the duplicated dtype gates have drifted")


def test_an_alphabetic_term_does_not_touch_datetime_columns_at_all(monkeypatch):
    """A term that cannot match a datetime must not cause one to be read.

    Spied rather than timed, so it cannot go flaky on a loaded CI box. A numeric term now takes
    the component index rather than the render, so both routes are watched.
    """
    import graphistry.compute.gfql.datetime_search_index as index_mod
    import graphistry.compute.gfql.wysiwyg as wy

    touched = []
    original_render, original_index = wy.render_datetime_pandas, index_mod.index_for
    monkeypatch.setattr(
        wy, "render_datetime_pandas",
        lambda s, tz=DEFAULT_TEMPORAL_TZ: (touched.append("render"), original_render(s, tz))[1])
    monkeypatch.setattr(
        index_mod, "index_for",
        lambda s, tz: (touched.append("index"), original_index(s, tz))[1])

    df = pd.DataFrame({
        "name": ["alpha", "beta"],
        "when": dt_series("2024-01-05T03:04:05", "2023-07-04T09:05:00"),
    })

    search_any_mask(df, "alpha")
    assert touched == [], "an alphabetic term read a datetime column"

    search_any_mask(df, "2024")
    assert touched == ["index"], f"a numeric term should take the index, got {touched}"


def test_the_index_is_built_once_per_datetime_column(monkeypatch):
    """Cost scales with the number of datetime columns, linearly and no worse."""
    import graphistry.compute.gfql.datetime_search_index as index_mod

    built = []
    original = index_mod.index_for

    def spy(s, tz):
        built.append(s.name)
        return original(s, tz)

    monkeypatch.setattr(index_mod, "index_for", spy)

    df = pd.DataFrame({
        "a": dt_series("2024-01-05T03:04:05", "2023-07-04T09:05:00"),
        "b": dt_series("2024-02-05T03:04:05", "2022-07-04T09:05:00"),
        "c": ["x", "y"],
    })
    search_any_mask(df, "2024")
    assert sorted(built) == ["a", "b"], f"expected one index per datetime column, got {built}"
