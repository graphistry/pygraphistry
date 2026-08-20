"""Unit pins for native-datetime resolution handling (#1915).

``astype('int64')`` on a datetime column yields ticks in the dtype's OWN unit, so
every consumer must scale by that unit. pandas 3 made this visible by defaulting
``to_datetime`` to ``datetime64[us]`` where pandas 2 gave ``[ns]``, but the hazard
is a RESOLUTION one and reproduces on pandas 2 with an explicit ``[us]`` column.
These pin the probe itself, including the arms the end-to-end query tests cannot
reach (the string-fallback path and unknown units).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graphistry.compute.gfql.row.ordering import (
    _native_epoch_ticks,
    _native_temporal_unit_nanoseconds,
)


@pytest.mark.parametrize("unit,expected", [
    ("s", 1_000_000_000),
    ("ms", 1_000_000),
    ("us", 1_000),
    ("ns", 1),
])
def test_unit_nanoseconds_from_numpy_dtype(unit, expected):
    assert _native_temporal_unit_nanoseconds(np.dtype(f"datetime64[{unit}]")) == expected


@pytest.mark.parametrize("unit,expected", [
    ("s", 1_000_000_000),
    ("ms", 1_000_000),
    ("us", 1_000),
    ("ns", 1),
])
def test_unit_nanoseconds_from_tz_aware_dtype(unit, expected):
    """Tz-aware dtypes expose ``.unit`` directly rather than through the string."""
    dtype = pd.DatetimeTZDtype(unit=unit, tz="UTC")
    assert _native_temporal_unit_nanoseconds(dtype) == expected


def test_unit_nanoseconds_falls_back_to_the_dtype_string():
    """An object exposing no ``.unit`` must still resolve via its repr — the arm a
    real pandas/cuDF dtype never exercises."""
    class _OpaqueDtype:
        def __repr__(self) -> str:
            return "datetime64[ms]"
        __str__ = __repr__

    assert _native_temporal_unit_nanoseconds(_OpaqueDtype()) == 1_000_000


def test_unit_nanoseconds_defaults_to_one_when_unparseable():
    """No unit anywhere: default to 1 so ticks are treated as nanoseconds rather
    than silently scaling by an invented factor."""
    assert _native_temporal_unit_nanoseconds(np.dtype("int64")) == 1
    assert _native_temporal_unit_nanoseconds(None) == 1


def test_unit_nanoseconds_ignores_a_non_string_unit_attribute():
    class _WeirdDtype:
        unit = 7  # not a str -> must fall through to the string probe

        def __repr__(self) -> str:
            return "datetime64[s]"
        __str__ = __repr__

    assert _native_temporal_unit_nanoseconds(_WeirdDtype()) == 1_000_000_000


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_epoch_ticks_are_in_the_dtypes_own_unit(unit):
    """The contract the fix depends on: ticks scale with the dtype, so
    ticks * unit_nanoseconds is invariant across resolutions."""
    s = pd.Series(pd.to_datetime(["2020-01-02 03:04:05"])).astype(f"datetime64[{unit}]")
    ticks = int(_native_epoch_ticks(s).iloc[0])
    assert ticks * _native_temporal_unit_nanoseconds(s.dtype) == 1_577_934_245_000_000_000


def test_epoch_ticks_normalises_tz_aware_to_utc():
    """A tz-aware instant and its UTC twin must yield identical ticks."""
    aware = pd.Series(pd.to_datetime(["2020-01-02T05:00:00+05:00"]))
    utc = pd.Series(pd.to_datetime(["2020-01-02T00:00:00Z"]))
    assert int(_native_epoch_ticks(aware).iloc[0]) == int(_native_epoch_ticks(utc).iloc[0])
