"""The third-party dtype-classifier traps the pushdown planner has to dodge, as tests.

These lived as comments in ``lazy/engine/polars/dtypes.py``. Every one of them is a
checkable claim about pandas or polars behaviour, so every one of them belongs here: when
a future pandas/polars changes the behaviour this file FAILS, whereas the comment would
just have gone quietly stale while the planner started answering differently.
"""
import pandas as pd
import pytest

from graphistry.compute.gfql.lazy.engine.polars.dtypes import (
    dtype_text, is_numeric_dtype_safe, is_string_dtype_safe,
)

pl = pytest.importorskip("polars")


def test_pandas_numeric_classifier_says_false_for_polars_dtypes_and_does_not_raise():
    """THE TRAP, and the reason the polars branch must be consulted FIRST.

    pandas does not raise on a polars dtype -- it returns a confident, WRONG ``False``. So a
    "try pandas, fall back to polars on exception" ordering would never reach the fallback
    and would classify every polars numeric column as non-numeric.
    """
    for dtype in (pl.Int64, pl.Float64, pl.Decimal):
        result = pd.api.types.is_numeric_dtype(dtype)  # must NOT raise
        assert result is False or bool(result) is False, (
            f"pandas started classifying {dtype} correctly; the polars-first ordering in "
            "is_numeric_dtype_safe may no longer be load-bearing -- re-derive it")


@pytest.mark.parametrize("dtype,expected", [
    (pl.Int64, True),
    (pl.Float64, True),
    (pl.Decimal, True),      # Decimal IS numeric for this planner (polars' own is_numeric)
    (pl.String, False),
    (pl.Boolean, False),
])
def test_is_numeric_dtype_safe_classifies_polars_dtypes(dtype, expected):
    assert is_numeric_dtype_safe(dtype) is expected


def test_decimal_is_numeric_because_polars_own_is_numeric_is_used():
    """Pinned separately from the table above: ``pl.Decimal`` is exactly the dtype that
    distinguishes polars' ``DataType.is_numeric()`` from a hand-rolled int/float check."""
    assert pl.Decimal.is_numeric() is True
    assert is_numeric_dtype_safe(pl.Decimal) is True


@pytest.mark.parametrize("dtype_repr,expected", [
    ("str", True),           # pandas 3-era default string dtype reprs as bare "str"
    ("object", True),
    ("string[python]", True),
    ("large_string", True),
    ("struct<a:int>", False),  # "str" must match EXACTLY, else "struct" reads as string
    ("struct({'a': int64})", False),
    ("int64", False),
])
def test_string_classifier_fallback_matches_str_exactly(monkeypatch, dtype_repr, expected):
    """The text fallback only runs when pandas RAISES, which pandas 2.x never does for these
    inputs -- so the branch is forced here rather than left uncovered. The exact-match rule on
    "str" is what keeps "struct" from being classified as a string type."""
    def _raises(_dtype):
        raise TypeError("forced: exercise the text fallback")

    monkeypatch.setattr(pd.api.types, "is_string_dtype", _raises)
    assert is_string_dtype_safe(dtype_repr) is expected


def test_polars_refuses_an_int_to_float_join_key_where_pandas_coerces():
    """Why every endpoint<->node-id join in the polars engine casts first.

    pandas silently coerces an int64 key against a float64 key and matches; polars raises.
    A null endpoint is enough to promote an endpoint column to float while the node ids stay
    int, so without the cast the join dies on data pandas handles.
    """
    ints = pl.DataFrame({"k": [1, 2]}, schema={"k": pl.Int64})
    floats = pl.DataFrame({"k": [1.0, 2.0]}, schema={"k": pl.Float64})
    with pytest.raises(Exception) as excinfo:
        ints.join(floats, on="k", how="inner")
    assert "Schema" in type(excinfo.value).__name__ or "schema" in str(excinfo.value).lower()

    assert len(pd.DataFrame({"k": [1, 2]}).merge(pd.DataFrame({"k": [1.0, 2.0]}), on="k")) == 2

    cast_first = ints.join(floats.with_columns(pl.col("k").cast(pl.Int64)), on="k", how="inner")
    assert cast_first.height == 2


def test_polars_dtype_classes_and_instances_compare_equal():
    """Why ``PolarsDType`` is ``Union[DataType, Type[DataType]]``: schema values are INSTANCES
    while user code passes the bare CLASSES, and the metaclass makes the two compare equal --
    so a classifier may be handed either form."""
    assert pl.Utf8 == pl.String()
    assert isinstance(pl.String(), pl.DataType)
    assert type(pl.Utf8).__name__ == "DataTypeClass"
    assert is_string_dtype_safe(pl.Utf8) is is_string_dtype_safe(pl.String())


def test_dtype_text_is_lowercased_and_never_raises():
    class Explodes:
        def __str__(self):
            raise RuntimeError("no repr for you")

    assert dtype_text(pl.String) == str(pl.String).lower()
    assert dtype_text(Explodes()) == ""
