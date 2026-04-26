"""Direct unit tests for ``_StringAllowingComparisonMixin`` (#1217 wave-2).

EQ already had end-to-end string coverage in ``test_eq_types.py``; this
file covers the sibling comparison ops NE/GT/LT/GE/LE that gained string
support via the mixin.  Pandas Series natively supports lexicographic
``>``/``<``/``!=`` on strings — these tests lock in:

  1. ``validate()`` accepts raw-string ``val``.
  2. ``__call__`` returns the expected boolean Series under lexicographic
     ordering.
  3. ``to_json``/``from_json`` round-trip preserves the string ``val``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from graphistry.compute.predicates.comparison import (
    EQ, GE, GT, LE, LT, NE,
    eq, ge, gt, le, lt, ne,
)


_OPS = [
    pytest.param(GT, gt, id="GT"),
    pytest.param(LT, lt, id="LT"),
    pytest.param(GE, ge, id="GE"),
    pytest.param(LE, le, id="LE"),
    pytest.param(EQ, eq, id="EQ"),
    pytest.param(NE, ne, id="NE"),
]


@pytest.mark.parametrize("cls,factory", _OPS)
def test_string_val_validates(cls, factory) -> None:
    p = factory("hello")
    assert isinstance(p, cls)
    assert p.val == "hello"
    p.validate()


@pytest.mark.parametrize("cls,factory", _OPS)
def test_string_val_json_roundtrip(cls, factory) -> None:
    p = factory("zebra")
    j = p.to_json()
    assert j["val"] == "zebra"
    p2 = cls.from_json(j)
    assert isinstance(p2, cls)
    assert p2.val == "zebra"


def test_gt_string_lexicographic_apply() -> None:
    s = pd.Series(["apple", "banana", "cherry"])
    result = gt("banana")(s)
    assert list(result) == [False, False, True]


def test_lt_string_lexicographic_apply() -> None:
    s = pd.Series(["apple", "banana", "cherry"])
    result = lt("banana")(s)
    assert list(result) == [True, False, False]


def test_ge_string_lexicographic_apply() -> None:
    s = pd.Series(["apple", "banana", "cherry"])
    result = ge("banana")(s)
    assert list(result) == [False, True, True]


def test_le_string_lexicographic_apply() -> None:
    s = pd.Series(["apple", "banana", "cherry"])
    result = le("banana")(s)
    assert list(result) == [True, True, False]


def test_ne_string_apply() -> None:
    s = pd.Series(["a", "b", "a"])
    result = ne("a")(s)
    assert list(result) == [False, True, False]
