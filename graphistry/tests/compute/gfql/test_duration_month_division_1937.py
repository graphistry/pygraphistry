"""End-to-end pins for scaling a duration whose month group has to be SPLIT (#1937).

Scaling used to decline whenever the result was fractional in month-space, so
``duration('P1M') / 2`` raised instead of answering while ``duration('P2M') / 2``
answered ``P1M``. openCypher resolves the split at the average month of 30.436875 days
(365.2425 / 12, the same constant this codebase already used for ``duration('P0.5M')``),
cascading months -> days -> seconds and truncating toward zero at each step. The
truncation is pinned alongside because the seconds group used to ROUND, which is a
visible 1ns disagreement (``PT2S / 3`` was ...667S where openCypher says ...666S).

Every expected value below is an ORACLE taken from Neo4j 5.26.26 via ``cypher-shell``,
not from this implementation. The two halves are pinned together on purpose: the split
cases are the new behaviour, and the exact cases are the fence that keeps the average
month out of results that are exact in month-space.
"""
from __future__ import annotations

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLTypeError
from graphistry.Plottable import Plottable

pl = pytest.importorskip("polars")

ENGINES = ["pandas", "polars"]


def _one_row_graph() -> Plottable:
    nodes = pd.DataFrame({"id": ["a"]})
    edges = pd.DataFrame({"src": ["a"], "dst": ["a"]})
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def _value(query: str, engine: str) -> object:
    column = _one_row_graph().gfql(f"RETURN {query} AS x", engine=engine)._nodes["x"]
    return (column.to_list() if hasattr(column, "to_list") else column.tolist())[0]


# Half a month is 30.436875 / 2 == 15.2184375 days == 15 days + 18873 seconds.
SPLIT_MONTH = [
    ("div_month_by_two", "duration('P1M') / 2", "P15DT5H14M33S"),
    ("mul_month_by_half", "duration('P1M') * 0.5", "P15DT5H14M33S"),
    ("half_times_month_commutes", "0.5 * duration('P1M')", "P15DT5H14M33S"),
    ("div_month_by_three", "duration('P1M') / 3", "P10DT3H29M42S"),
    # The day remainder carries on into a fractional SECOND, not just whole seconds.
    ("div_month_by_four", "duration('P1M') / 4", "P7DT14H37M16.5S"),
    # Whole months survive the split; only the leftover fraction becomes days.
    ("div_three_months_by_two", "duration('P3M') / 2", "P1M15DT5H14M33S"),
    ("div_year_and_month_by_two", "duration('P1Y1M') / 2", "P6M15DT5H14M33S"),
    ("mul_month_by_one_and_a_half", "duration('P1M') * 1.5", "P1M15DT5H14M33S"),
    # A day group already present is scaled first and the month spill lands on top.
    ("div_month_and_days_by_two", "duration('P1M2D') / 2", "P16DT5H14M33S"),
    ("div_month_by_six", "duration('P1M') / 6", "P5DT1H44M51S"),
    # The month divides evenly, so only the DAY group splits and no month is spilled.
    ("div_two_months_and_a_day_by_two", "duration('P2M1D') / 2", "P1MT12H"),
    ("div_map_form_by_three", "duration({days: 14, minutes: 12, seconds: 70, nanoseconds: 1}) / 3", "P4DT16H4M23.333333333S"),
]

# Every cascade step truncates toward zero; it never rounds to nearest.
TRUNCATION = [
    ("two_seconds_by_three", "duration('PT2S') / 3", "PT0.666666666S"),
    ("eight_seconds_by_nine", "duration('PT8S') / 9", "PT0.888888888S"),
    ("one_second_by_seven", "duration('PT1S') / 7", "PT0.142857142S"),
    ("six_seconds_by_seven", "duration('PT6S') / 7", "PT0.857142857S"),
    ("one_second_by_three", "duration('PT1S') / 3", "PT0.333333333S"),
    ("negative_two_seconds_by_three", "duration('PT2S') / -3", "PT-0.666666666S"),
    ("negative_one_second_by_three", "duration('PT1S') / -3", "PT-0.333333333S"),
    # Observable at 1ns even where the exact answer is a round 38.4 seconds: the double
    # residue lands just under 384000000ns and truncation keeps it there.
    ("month_by_two_and_a_half", "duration('P1M') / 2.5", "P12DT4H11M38.399999999S"),
]

# Truncation toward zero, so a negative scale mirrors its positive twin exactly.
SPLIT_MONTH_SIGNS = [
    ("negative_divisor", "duration('P1M') / -2", "P-15DT-5H-14M-33S"),
    ("negative_duration", "duration('-P1M') / 2", "P-15DT-5H-14M-33S"),
    ("negative_both", "duration('-P1M') / -2", "P15DT5H14M33S"),
    ("negative_fractional_factor", "duration('P1M') * -0.5", "P-15DT-5H-14M-33S"),
    ("negative_fractional_second", "duration('P1M') / -4", "P-7DT-14H-37M-16.5S"),
    ("negative_whole_month_remainder", "duration('-P3M') / 2", "P-1M-15DT-5H-14M-33S"),
    ("negative_divisor_whole_month_remainder", "duration('P3M') / -2", "P-1M-15DT-5H-14M-33S"),
    ("negative_duration_with_days", "duration('-P1M2D') / 2", "P-16DT-5H-14M-33S"),
    ("negative_divisor_with_days", "duration('P1M2D') / -2", "P-16DT-5H-14M-33S"),
]

# The fence: exact in month-space stays in month-space, no average month anywhere.
EXACT_MONTH = [
    ("div_two_months_by_two", "duration('P2M') / 2", "P1M"),
    ("div_month_by_one", "duration('P1M') / 1", "P1M"),
    ("div_year_by_two", "duration('P1Y') / 2", "P6M"),
    ("mul_month_by_two", "duration('P1M') * 2", "P2M"),
    ("mul_month_by_two_point_zero", "duration('P1M') * 2.0", "P2M"),
    ("div_month_by_one_negative", "duration('P1M') / -1", "P-1M"),
    ("div_twelve_months_by_two", "duration('P12M') / 2", "P6M"),
    ("div_seven_months_by_seven", "duration('P7M') / 7", "P1M"),
    ("mul_month_by_zero", "duration('P1M') * 0", "PT0S"),
    ("mul_month_by_minus_one", "duration('P1M') * -1", "P-1M"),
    # ... and the non-month groups keep the fixed ratios they always had.
    ("div_days_by_two", "duration('P3D') / 2", "P1DT12H"),
    ("div_hour_by_two", "duration('PT1H') / 2", "PT30M"),
    ("date_plus_month_clamps", "date('2026-01-31') + duration('P1M')", "2026-02-28"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "name,query,expected", SPLIT_MONTH, ids=[case[0] for case in SPLIT_MONTH]
)
def test_splitting_a_month_resolves_at_the_average_month(engine, name, query, expected):
    assert _value(query, engine) == expected


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "name,query,expected", SPLIT_MONTH_SIGNS, ids=[case[0] for case in SPLIT_MONTH_SIGNS]
)
def test_splitting_a_month_truncates_toward_zero(engine, name, query, expected):
    assert _value(query, engine) == expected


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "name,query,expected", TRUNCATION, ids=[case[0] for case in TRUNCATION]
)
def test_every_cascade_step_truncates_rather_than_rounds(engine, name, query, expected):
    assert _value(query, engine) == expected


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "name,query,expected", EXACT_MONTH, ids=[case[0] for case in EXACT_MONTH]
)
def test_a_month_that_divides_evenly_stays_exact(engine, name, query, expected):
    assert _value(query, engine) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_split_month_agrees_with_the_fractional_month_literal(engine):
    """The constructor already split fractional months at 30.436875 days; scaling must
    land on the same value rather than introduce a second average month."""
    assert _value("duration('P1M') / 2", engine) == _value("duration('P0.5M')", engine)
    assert _value("duration('P1M') * 1.5", engine) == _value("duration('P1.5M')", engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_splitting_a_month_does_not_round_trip(engine):
    """ACCEPTED, not a defect: the average month is one-way. Halving then doubling a
    month lands on days+time and must NOT be 'restored' to P1M."""
    assert _value("(duration('P1M') / 2) * 2", engine) == "P30DT10H29M6S"


def test_dividing_a_duration_by_zero_still_declines():
    """Splitting a month resolves; dividing by zero is still not a duration. Pinned on
    pandas alone because the polars row path declines the unfolded node its own way."""
    with pytest.raises(GFQLTypeError):
        _value("duration('P1M') / 0", "pandas")
