"""String-val coverage for ``_StringAllowingComparisonMixin`` on EQ/NE/GT/LT/GE/LE.

EQ already had multi-type coverage in ``test_eq_types.py``; this file
parametrizes the mixin contract over the sibling ops gained in #1217.
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

_SERIES = pd.Series(["apple", "banana", "cherry"])
_LEX_EXPECTED = {
    gt: [False, False, True],
    lt: [True, False, False],
    ge: [False, True, True],
    le: [True, True, False],
    eq: [False, True, False],
    ne: [True, False, True],
}


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


@pytest.mark.parametrize("cls,factory", _OPS)
def test_string_lexicographic_apply(cls, factory) -> None:
    result = factory("banana")(_SERIES)
    assert list(result) == _LEX_EXPECTED[factory]
