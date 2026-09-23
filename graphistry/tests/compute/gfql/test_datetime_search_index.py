"""The index must answer exactly what searching the rendered text answers.

Everything here is differential: the render in ``wysiwyg`` is the oracle, and the index is only
allowed to reach the same answer by another route, never a different one.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from graphistry.compute.gfql import datetime_search_index as dsi
from graphistry.compute.gfql.datetime_search_index import (
    DatetimeIndexCacheThrashWarning, DatetimeSearchIndex, clear_cache, index_for,
    thrash_events)
from graphistry.compute.gfql.search_any import search_any_mask
from graphistry.compute.gfql.wysiwyg import render_datetime_pandas


def stamps(n, span_s, start="2015-01-01", seed=5):
    rng = np.random.default_rng(seed)
    return pd.Series(pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, span_s, n), unit="s"))


SHAPES = {
    "three years of seconds": stamps(400, 3 * 10 ** 8),
    "a single day": stamps(400, 86_400),
    "seventy years": stamps(400, 70 * 365 * 86_400, "1970-01-01"),
    "every row identical": pd.Series(pd.to_datetime(["2024-01-05T03:04:05"] * 60)),
    "some rows null": pd.concat(
        [stamps(80, 10 ** 7), pd.Series(pd.to_datetime([None] * 40))], ignore_index=True),
    "every row null": pd.Series(pd.to_datetime([None] * 20)),
}

# Casablanca and Chatham carry zone labels with digits and a sign, which a numeric term may match
ZONES = ["UTC", "America/New_York", "Asia/Kolkata", "Africa/Casablanca", "Pacific/Chatham"]

TERMS = ([str(d) for d in range(10)]
         + ["10", "12", "20", "24", "31", "59", "60", "00", "01", "05"]
         + ["202", "2024", "1999", "2038", "1970", "9999", "123"]
         + ["-", "-0", "-03", "+01", "+1245", ".", "1.5", "0.5"])


def rendered_answer(s, tz, term):
    text = render_datetime_pandas(s, tz)
    return (text.notna() & text.fillna("").str.contains(term, regex=False)).to_numpy()


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_index_answers_what_the_render_answers(shape):
    s = SHAPES[shape]
    for tz, term in itertools.product(ZONES, TERMS):
        want = rendered_answer(s, tz, term)
        got = index_for(s, tz).matches(term)
        assert np.array_equal(want, got), (
            f"shape={shape!r} tz={tz} term={term!r}: index and render disagree")


def test_a_zone_label_with_digits_selects_only_its_own_dst_regime():
    """The bug this pins: treating the zone as a whole-column property.

    Africa/Casablanca spans both +00 and +01, so '+01' must select the rows in that regime and
    not the entire column.
    """
    s = stamps(400, 3 * 10 ** 8)
    idx = index_for(s, "Africa/Casablanca")
    got = idx.matches("+01")
    assert got.any(), "expected some rows in the +01 regime"
    assert not got.all(), "'+01' must not select rows that render +00"
    assert np.array_equal(got, rendered_answer(s, "Africa/Casablanca", "+01"))


def test_searchany_uses_the_index_and_agrees_with_the_render(monkeypatch):
    df = pd.DataFrame({"when": stamps(300, 10 ** 8)})
    import graphistry.compute.gfql.datetime_search_index as mod

    used = []
    original = mod.index_for
    monkeypatch.setattr(mod, "index_for", lambda s, tz: (used.append(tz), original(s, tz))[1])

    for term in ["2024", "05", "9999"]:
        got = search_any_mask(df, term).to_numpy()
        assert np.array_equal(got, rendered_answer(df["when"], "UTC", term)), term
    assert used, "searchAny did not take the index path"


@pytest.mark.parametrize("kwargs", [{"regex": True}, {"case_sensitive": True}])
def test_regex_and_case_sensitive_searches_still_agree(kwargs):
    """Those take the render path; they must not drift from it either."""
    df = pd.DataFrame({"when": stamps(200, 10 ** 8)})
    got = search_any_mask(df, "2024", **kwargs)
    text = render_datetime_pandas(df["when"], "UTC")
    want = text.notna() & text.fillna("").str.contains(
        "2024", regex=bool(kwargs.get("regex")), case=bool(kwargs.get("case_sensitive")))
    assert np.array_equal(got.to_numpy(), want.to_numpy())


class TestCache:

    def test_the_same_column_is_not_rebuilt(self):
        clear_cache()
        s = stamps(100, 10 ** 7)
        assert index_for(s, "UTC") is index_for(s, "UTC")

    def test_a_mutated_column_does_not_hit(self):
        clear_cache()
        s = stamps(100, 10 ** 7)
        first = index_for(s, "UTC")
        edited = s.copy()
        edited.iloc[0] = pd.Timestamp("1999-01-01")
        assert index_for(edited, "UTC") is not first

    def test_a_reordered_column_does_not_hit(self):
        """The index is row-ordered, so a permutation is a different index.

        Rules out any digest a permutation leaves untouched, such as a sum or an xor.
        """
        clear_cache()
        s = stamps(100, 10 ** 7)
        first = index_for(s, "UTC")
        assert index_for(s.iloc[::-1].reset_index(drop=True), "UTC") is not first

    def test_the_same_bytes_under_a_different_unit_do_not_hit(self):
        """The buffer alone does not say what its integers MEAN.

        The same int64s read as nanoseconds and as microseconds are different instants, so a
        digest of the bytes is only a key once the dtype is part of it. The value is chosen so
        both readings land in a year pandas can format.
        """
        clear_cache()
        shared = np.array([1_000_000_000_000_000], dtype="int64")
        nanos = pd.Series(pd.array(shared, dtype="datetime64[ns]"))
        micros = pd.Series(pd.array(shared, dtype="datetime64[us]"))
        assert np.array_equal(nanos.to_numpy().view("int64"), micros.to_numpy().view("int64")), (
            "this test is vacuous unless the two columns share their bytes")
        assert nanos.iloc[0].year != micros.iloc[0].year, (
            "same bytes, different unit, so they are different instants")
        assert index_for(nanos, "UTC") is not index_for(micros, "UTC")

    def test_a_different_zone_does_not_hit(self):
        clear_cache()
        s = stamps(100, 10 ** 7)
        assert index_for(s, "UTC") is not index_for(s, "America/New_York")

    def test_clearing_drops_what_was_held(self):
        clear_cache()
        s = stamps(100, 10 ** 7)
        first = index_for(s, "UTC")
        clear_cache()
        assert index_for(s, "UTC") is not first


def test_an_empty_column_builds_and_matches_nothing():
    s = pd.Series([], dtype="datetime64[ns]")
    idx = DatetimeSearchIndex(s, "UTC")
    assert idx.matches("2024").tolist() == []


@pytest.mark.parametrize("tz", ["UTC", "Africa/Casablanca"])
@pytest.mark.parametrize("sequence", [
    ["2", "20", "202", "2024"], ["0", "05"], ["1", "12", "123"], ["5", "59"],
    ["2", "20", "202", "2024", "20245"],
], ids=lambda s: s[-1])
def test_typing_a_term_one_character_at_a_time_agrees_with_the_render(sequence, tz):
    """Each keystroke is its own query; none of them may drift from the rendered answer."""
    s = stamps(600, 3 * 10 ** 8)
    idx = index_for(s, tz)
    for term in sequence:
        assert np.array_equal(idx.matches(term), rendered_answer(s, tz, term)), term


class TestTheShortcutsAreActuallyTaken:
    """The three shortcuts are invisible in the ANSWER: a version that quietly stopped taking
    them would still be exact, and every differential test above would still pass. These pin
    the route instead, by counting the whole-column passes ``matches`` performs.
    """

    @staticmethod
    def passes(monkeypatch, index, term):
        counts = {"equal": 0, "take": 0, "logical_and": 0}
        real = {name: getattr(np, name) for name in counts}

        def counting(name):
            def wrapper(*args, **kwargs):
                counts[name] += 1
                return real[name](*args, **kwargs)
            return wrapper

        for name in counts:
            monkeypatch.setattr(np, name, counting(name))
        index.matches(term)
        return counts

    def test_a_field_every_value_of_which_matches_reads_no_rows(self, monkeypatch):
        """Every year in this column contains '2', so every present row matches whatever the
        other fields hold -- the answer is the present mask and no field is evaluated."""
        column = stamps(300, 9 * 365 * 86_400, "2020-01-01")
        index = DatetimeSearchIndex(column, "UTC")
        assert self.passes(monkeypatch, index, "2") == {
            "equal": 0, "take": 0, "logical_and": 0}
        assert index.matches("2").all()

    def test_a_term_no_value_can_contain_reads_no_rows(self, monkeypatch):
        index = DatetimeSearchIndex(stamps(300, 10 ** 7), "UTC")
        # no scratch buffer and no final AND either: the empty answer is allocated and returned
        assert self.passes(monkeypatch, index, "202411") == {
            "equal": 0, "take": 0, "logical_and": 0}
        assert not index.matches("202411").any()

    def test_one_selected_value_is_a_comparison_not_a_gather(self, monkeypatch):
        """'2024' can only be a year, and only one year renders it, so the whole search is a
        single comparison against that year's code."""
        column = stamps(300, 700 * 86_400, "2023-06-01")
        index = DatetimeSearchIndex(column, "UTC")
        assert self.passes(monkeypatch, index, "2024") == {
            "equal": 1, "take": 0, "logical_and": 1}

    def test_many_selected_values_switch_to_a_gather(self, monkeypatch):
        """Fourteen of the sixty minutes render a '5', which is past the measured crossover, so
        that field is served by a lookup table instead of fourteen comparisons."""
        column = stamps(300, 300 * 86_400, "2024-01-01")
        index = DatetimeSearchIndex(column, "UTC")
        counts = self.passes(monkeypatch, index, "5")
        assert counts["take"] >= 1
        assert counts["equal"] < 14

    def test_the_crossover_is_a_threshold_both_sides_of_which_are_reachable(self, monkeypatch):
        """Guards against a threshold edited to a value that disables one branch entirely."""
        column = stamps(300, 700 * 86_400, "2023-06-01")
        index = DatetimeSearchIndex(column, "UTC")
        assert 0 < index._COMPARE_UPTO < 60

class TestCacheBudgetAndThrash:
    """Two columns whose indexes together exceed the budget evict each other on every search,
    so each keystroke rebuilds both. That is invisible in the answers -- only in the time -- so
    the cache has to say so itself, and the budget has to be something an operator can raise
    without a code change.
    """

    @staticmethod
    def two_columns():
        a = stamps(2_000, 10 ** 7)
        return a, a + pd.Timedelta(seconds=1)

    def test_the_budget_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv(dsi._CACHE_BUDGET_ENV, "12345")
        assert dsi._cache_budget_bytes() == 12345

    def test_an_unusable_budget_falls_back_to_the_default(self, monkeypatch):
        for bad in ("", "  ", "lots", "-1", "0"):
            monkeypatch.setenv(dsi._CACHE_BUDGET_ENV, bad)
            assert dsi._cache_budget_bytes() == dsi._CACHE_BUDGET_BYTES
        monkeypatch.delenv(dsi._CACHE_BUDGET_ENV)
        assert dsi._cache_budget_bytes() == dsi._CACHE_BUDGET_BYTES

    def test_alternating_columns_under_budget_is_reported_as_thrash(self, monkeypatch):
        a, b = self.two_columns()
        clear_cache()
        one = DatetimeSearchIndex(a, "UTC").nbytes
        monkeypatch.setenv(dsi._CACHE_BUDGET_ENV, str(one + one // 2))   # room for one, not two
        with pytest.warns(DatetimeIndexCacheThrashWarning):
            for _ in range(3):
                index_for(a, "UTC")
                index_for(b, "UTC")
        assert thrash_events() >= 1
        assert len(dsi._CACHE) == 1

    def test_the_warning_fires_once_but_the_count_keeps_going(self, monkeypatch):
        a, b = self.two_columns()
        clear_cache()
        one = DatetimeSearchIndex(a, "UTC").nbytes
        monkeypatch.setenv(dsi._CACHE_BUDGET_ENV, str(one + one // 2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(4):
                index_for(a, "UTC")
                index_for(b, "UTC")
        ours = [w for w in caught if issubclass(w.category, DatetimeIndexCacheThrashWarning)]
        assert len(ours) == 1
        assert thrash_events() > 1

    def test_a_budget_that_fits_both_keeps_both_and_stays_quiet(self, monkeypatch):
        a, b = self.two_columns()
        clear_cache()
        one = DatetimeSearchIndex(a, "UTC").nbytes
        monkeypatch.setenv(dsi._CACHE_BUDGET_ENV, str(3 * one))
        with warnings.catch_warnings():
            warnings.simplefilter("error", DatetimeIndexCacheThrashWarning)
            first_a = index_for(a, "UTC")
            first_b = index_for(b, "UTC")
            for _ in range(3):
                assert index_for(a, "UTC") is first_a
                assert index_for(b, "UTC") is first_b
        assert thrash_events() == 0
        assert len(dsi._CACHE) == 2

    def test_clearing_resets_the_detector(self, monkeypatch):
        a, b = self.two_columns()
        clear_cache()
        one = DatetimeSearchIndex(a, "UTC").nbytes
        monkeypatch.setenv(dsi._CACHE_BUDGET_ENV, str(one + one // 2))
        with pytest.warns(DatetimeIndexCacheThrashWarning):
            for _ in range(2):
                index_for(a, "UTC")
                index_for(b, "UTC")
        clear_cache()
        assert thrash_events() == 0
        with pytest.warns(DatetimeIndexCacheThrashWarning):      # can fire again
            for _ in range(2):
                index_for(a, "UTC")
                index_for(b, "UTC")
