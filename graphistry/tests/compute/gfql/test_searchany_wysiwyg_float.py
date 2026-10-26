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


def test_polars_render_skips_what_the_inspector_shows_as_nothing():
    """The polars arm of the same contract as ``test_render_skips_...``.

    Pinned separately because the polars renderer is a different expression with its own
    null/sentinel branch: a mutation that dropped the sentinel check there passed every
    other test in this file and in the conformance matrix.
    """
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql.wysiwyg import float_render_expr_polars

    df = pl.DataFrame({"x": pl.Series(
        [float("nan"), float(INSPECTOR_INT32_SENTINEL), 1.5, None], dtype=pl.Float64)})
    got = (df.select(float_render_expr_polars(pl.col("x"), pl.Float64).alias("o"))
           .to_series().to_list())
    assert got[0] is None, f"NaN must render nothing, got {got[0]!r}"
    assert got[1] is None, f"the Int32 sentinel must render nothing, got {got[1]!r}"
    assert got[2] == "1.5000"
    assert got[3] is None, f"null must render nothing, got {got[3]!r}"

    inf_df = pl.DataFrame({"x": pl.Series([float("inf"), float("-inf")], dtype=pl.Float64)})
    inf_got = (inf_df.select(float_render_expr_polars(pl.col("x"), pl.Float64).alias("o"))
               .to_series().to_list())
    assert inf_got == ["Infinity", "-Infinity"], inf_got


def test_polars_search_never_matches_the_sentinel_end_to_end():
    """The user-visible half of the pin: the sentinel's digits must not find its row."""
    pl = pytest.importorskip("polars")
    import graphistry
    from graphistry.compute.ast import n

    nodes = pl.DataFrame({
        "id": [0, 1],
        "f": pl.Series([float(INSPECTOR_INT32_SENTINEL), 1.5], dtype=pl.Float64),
    })
    edges = pl.DataFrame({"s": [0], "d": [1]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    from graphistry.compute.ast import search_any as search_any_op

    out = g.gfql([n(name="a"), search_any_op(alias="a", out_col="__hit__", term="2147483647")],
                 engine="polars")._nodes
    hits = out.to_pandas().sort_values("id")["__hit__"].tolist()
    assert hits == [False, False], f"sentinel matched: {hits}"


@pytest.mark.parametrize("engine", ENGINES)
def test_non_finite_floats_render_without_crashing_or_inventing_digits(engine):
    """NaN and the infinities are where the integer-cast renders go wrong.

    polars RAISED on NaN (it is a value there, not null, and the cast fails whatever the
    when/then guard says) and cuDF silently emitted int64-max digits with a doubled sign
    for the infinities ('--922337203685477.5808'), which a digit search would then MATCH.
    Both are pinned here on every kernel engine.
    """
    df = _frame(pd.DataFrame({"f": [float("nan"), float("inf"), float("-inf"), 1.5]}), engine)
    # must not raise, and the infinities must not turn into digits
    assert _mask(df, "922337") == [False, False, False, False]
    assert _mask(df, "1.5000") == [False, False, False, True]


def test_non_finite_render_matches_the_inspector_naming():
    s = pd.Series([float("nan"), float("inf"), float("-inf")], dtype="float64")
    assert render_float_pandas(s).tolist() == [None, "Infinity", "-Infinity"]


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
