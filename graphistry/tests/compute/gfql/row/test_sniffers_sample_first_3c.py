"""The row evaluator's text sniffers decide on a sample before scanning a column (#2116 item 3c).

A string equality such as ``b.kind = 'zzz'`` on a 100k-row object column paid ~670 ms in
``order_detect_temporal_mode`` (eight regex full-matches over every value) plus ~240 ms in
the list-like and mapping-like probes, for both operands, before the compare itself. Each
probe is an all-rows conjunction, so a failing 16-value sample settles it exactly; the column
is rendered and scanned only when the sample passes.
"""
import numpy as np
import pandas as pd
import pytest

from graphistry.compute.gfql.row.ordering import order_detect_temporal_mode
from graphistry.compute.gfql.row.pipeline import RowPipelineMixin

_DATES = [f"2024-01-{d:02d}" for d in range(1, 29)]
_DTS = [f"2024-01-{d:02d}T10:00:00" for d in range(1, 29)]


def _probe(series):
    return (
        order_detect_temporal_mode(series),
        RowPipelineMixin._gfql_series_is_list_like(series),
        RowPipelineMixin._gfql_series_is_mapping_like(series),
    )


@pytest.mark.parametrize("values,expected", [
    (_DATES, ("date", False, False)),
    (_DTS, ("datetime", False, False)),
    (["10:00:00", "11:30:00"], ("time", False, False)),
    (["date({year: 2024, month: 1, day: 1})"], ("date_constructor", False, False)),
    (["localdatetime({year: 2024, month: 1, day: 1, hour: 10, minute: 0})", "localdatetime({year: 2024, month: 1, day: 2, hour: 11, minute: 30, second: 5})"], ("datetime_constructor", False, False)),
    (["time({hour: 10, minute: 0, timezone: 'Z'})", "localtime({hour: 11, minute: 30, second: 1})"], ("time_constructor", False, False)),
    ([[1, 2], (3,), []], (None, True, False)),
    ([{"a": 1}, {}], (None, False, True)),
    (["zzz", "person"], (None, False, False)),
    (["[1, 2]", "(3, 4)"], (None, False, False)),     # list-looking TEXT is still a string
    (["{a: 1}"], (None, False, False)),
    (_DATES[:3] + _DTS[:3], (None, False, False)),    # mixed temporal families
    ([[1], "x"], (None, False, False)),
    ([None, None], (None, False, False)),
    ([], (None, False, False)),
])
def test_sniffers_keep_their_answers(values, expected):
    series = pd.Series(values, dtype=object)
    assert _probe(series) == expected
    assert _probe(pd.Series(list(values) + [None], dtype=object)) == expected


def test_a_failing_row_past_the_sample_still_counts():
    # 25 dates then one word: the sample passes, the full scan must still say "not temporal"
    assert order_detect_temporal_mode(pd.Series(_DATES[:25] + ["w"], dtype=object)) is None
    assert order_detect_temporal_mode(pd.Series(["w"] + _DATES[:25], dtype=object)) is None
    assert order_detect_temporal_mode(pd.Series(_DATES[:25], dtype=object)) == "date"
    big = pd.Series(np.random.default_rng(3).choice(_DATES, 100000), dtype=object)
    assert order_detect_temporal_mode(big) == "date"


def test_native_dtypes_are_untouched():
    assert _probe(pd.Series([1, 2, 3])) == (None, False, False)
    # the TEXT detector sees a datetime64 column through astype(str), as before; the native detector is separate
    assert _probe(pd.Series(pd.to_datetime(_DATES[:5]))) == ("date", False, False)


def test_a_string_column_sniff_does_not_scan_every_row(monkeypatch):
    # count full-match calls: a 100k string column must sniff on the sample, not the column
    from graphistry.compute.gfql.row import ordering
    calls = []
    real = ordering.series_str_fullmatch

    def counting(values, pattern, na=False):
        calls.append(len(values))
        return real(values, pattern, na=na)

    monkeypatch.setattr(ordering, "series_str_fullmatch", counting)
    col = pd.Series(np.random.default_rng(5).choice(["a", "b"], 100000), dtype=object)
    assert ordering.order_detect_temporal_mode(col) is None
    assert calls and max(calls) <= ordering._GFQL_TEMPORAL_SNIFF_SAMPLE


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("values,expected", [
    ([[1], [2]], (None, True, False)),
    (["[1]", "[2]"], (None, False, False)),                  # list-looking TEXT is text
    ([{"a": 1}, {"b": 2}], (None, False, True)),
    (_DATES[:4], ("date", False, False)),
    (["alpha", "beta"], (None, False, False)),
    ([None, None], (None, False, False)),
])
def test_the_sniffers_answer_the_same_on_cudf(engine, values, expected):
    series = pd.Series(values, dtype=object)
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        pytest.importorskip("cupy")
        series = cudf.Series(series)
    assert _probe(series) == expected


def test_an_array_column_with_list_looking_text_is_not_a_list_column():
    # the only answer this change corrects: `series == text` raises on the array, and the
    # swallowed error used to leave the string row undetected, so the column read as a list
    mixed = pd.Series([np.array([1, 2]), "[3, 4]"], dtype=object)
    assert RowPipelineMixin._gfql_series_is_list_like(mixed) is False
    assert RowPipelineMixin._gfql_series_is_list_like(pd.Series([np.array([1, 2])], dtype=object)) is True


@pytest.mark.parametrize("probe,pattern_rows", [
    (RowPipelineMixin._gfql_series_is_list_like, 16),
    (RowPipelineMixin._gfql_series_is_mapping_like, 32),
])
def test_the_list_and_mapping_probes_render_only_the_sample(monkeypatch, probe, pattern_rows):
    # count the rows handed to the regex: a column the sample rules out is never rendered
    from graphistry.compute.gfql.row import pipeline as pipeline_mod
    widths = []
    real = pipeline_mod.series_str_match

    def counting(values, pattern, na=False):
        widths.append(len(values))
        return real(values, pattern, na=na)

    monkeypatch.setattr(pipeline_mod, "series_str_match", counting)
    col = pd.Series(pd.to_datetime(_DATES).tolist() * 400, dtype=object)  # 11,200 rows, not list-like
    assert probe(col) is False
    assert widths and max(widths) <= pattern_rows
