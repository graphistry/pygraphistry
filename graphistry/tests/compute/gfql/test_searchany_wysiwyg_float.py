"""#1695: searchAny must match what the viz inspector DISPLAYS for float columns.

Oracle read from the shipping formatters (``apps/core/viz/src/formatters/``), not inferred:
fractional floats render ``sprintf('%.4f')`` == JS ``toFixed(4)``; whole-valued floats render
``String(v)`` with no trailing ``.0``; NaN and the Int32 sentinel ``2147483647`` render null and
must never match. Expected values below are hand-written from that rule.
"""
import pandas as pd
import pytest

from graphistry.compute.gfql.search_any import search_any_mask, search_candidate_columns
from graphistry.compute.gfql.wysiwyg import (
    DEFAULT_FLOAT_PRECISION, INSPECTOR_INT32_SENTINEL, render_float_pandas,
)

ENGINES = ["pandas", "cudf"]


def _frame(pdf, engine):
    if engine == "cudf":
        cudf = pytest.importorskip("cudf", reason="cuDF lane needs a GPU box")
        return cudf.from_pandas(pdf)
    return pdf


def _mask(df, term, **kw):
    m = search_any_mask(df, term, **kw)
    return m.to_pandas().tolist() if hasattr(m, "to_pandas") else m.tolist()


# --- the render itself, hand-computed from the inspector rule ---------------------------

@pytest.mark.parametrize("value,expected", [
    (7.25, "7.2500"),            # fractional -> fixed 4 decimals
    (1.5, "1.5000"),             # trailing zeros are PADDED (astype(str) would give '1.5')
    (-3.125, "-3.1250"),         # sign preserved
    (0.5, "0.5000"),
    (42.0, "42"),                # whole float -> String(v), NOT '42.0'
    (-42.0, "-42"),
    (0.0, "0"),                  # 0 is falsy in JS -> String branch
    (-0.0, "0"),                 # JS String(-0) is '0'
    (0.1 + 0.2, "0.3000"),       # the repr trap: NOT '0.30000000000000004'
    (1234.56789, "1234.5679"),
])
def test_render_matches_the_inspector(value, expected):
    assert render_float_pandas(pd.Series([value], dtype="float64")).tolist() == [expected]


def test_render_skips_what_the_inspector_shows_as_nothing():
    """NaN and the Int32 sentinel display nothing, so they must render null and never match."""
    s = pd.Series([float("nan"), float(INSPECTOR_INT32_SENTINEL), 1.5], dtype="float64")
    assert render_float_pandas(s).tolist() == [None, None, "1.5000"]


def test_render_precision_is_a_parameter():
    s = pd.Series([1.23456789], dtype="float64")
    assert render_float_pandas(s, 2).tolist() == ["1.23"]
    assert render_float_pandas(s, 6).tolist() == ["1.234568"]
    assert DEFAULT_FLOAT_PRECISION == 4


# --- the gate + end-to-end match, on every kernel engine --------------------------------

@pytest.mark.parametrize("engine", ENGINES)
def test_float_columns_join_the_auto_gate_for_numeric_terms(engine):
    df = _frame(pd.DataFrame({"s": ["a"], "i": [7], "f": [7.25], "b": [True]}), engine)
    assert search_candidate_columns(df, "7", None) == ["s", "i", "f"]   # float now included
    assert search_candidate_columns(df, "abc", None) == ["s"]            # non-numeric term


@pytest.mark.parametrize("engine", ENGINES)
def test_searching_a_float_column_matches_the_displayed_string(engine):
    df = _frame(pd.DataFrame({"f": [7.25, 0.5, 42.0, float("nan"), 0.1 + 0.2]}), engine)
    assert _mask(df, "7.2500") == [True, False, False, False, False]
    assert _mask(df, "7.25") == [True, False, False, False, False]   # substring of the render
    assert _mask(df, "0.3") == [False, False, False, False, True]    # displayed, not repr
    assert _mask(df, "0.30000000000000004") == [False] * 5           # the repr must NOT match
    assert _mask(df, "42") == [False, False, True, False, False]     # whole -> '42', not '42.0'
    assert _mask(df, "42.0") == [False] * 5


@pytest.mark.parametrize("engine", ENGINES)
def test_negative_large_and_sentinel_floats(engine):
    df = _frame(pd.DataFrame({
        "f": [-3.125, 1e6 + 0.5, float(INSPECTOR_INT32_SENTINEL), 1e15]
    }), engine)
    assert _mask(df, "-3.1250") == [True, False, False, False]
    assert _mask(df, "1000000.5000") == [False, True, False, False]
    # the sentinel displays nothing, so neither its digits nor a fragment may match
    assert _mask(df, "2147483647") == [False, False, False, False]


@pytest.mark.parametrize("engine", ENGINES)
def test_null_floats_never_match_any_term(engine):
    df = _frame(pd.DataFrame({"f": [None, 1.5]}, dtype="float64"), engine)
    for term in ["nan", "None", "1", "1.5000"]:
        got = _mask(df, term, columns=["f"])
        assert got[0] is False or got[0] == False, f"null matched {term!r}"  # noqa: E712
