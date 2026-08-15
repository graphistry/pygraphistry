"""Temporal-comparison / temporal-arithmetic / UNION semantics pins for #1915.

Round-010 probe (~282 queries x both engines) against oracles hand-written BEFORE the
sweep. Five defects, each pinned on BOTH engines with hand-computed openCypher oracles:

B-1  ``WHERE n.ts > datetime('...')`` on a tz-naive ``datetime64`` column matched ZERO
     rows across ~10 spellings. ``datetime()`` lowers to Z-suffixed ISO text; the
     temporal comparison path only recognised TEXT temporals, so pandas raised
     ``TypeError`` comparing ``datetime64`` to ``str`` and the mixed-comparison handler
     swallowed it into ``False``. openCypher says an incomparable comparison is NULL
     ("Comparability ... otherwise the result is null"), which is why ``NOT (...)``
     returned every non-null row instead of the complement.
B-2  ``date('2020-01-02') + duration('P1D')`` was Python string concatenation
     (``'2020-01-02P1D'``), and inside WHERE the concatenated text changed the ROW SET.
     Its sibling ``-`` already declined typed.
B-3  polars compared a String ISO column against the Z-suffixed literal
     lexicographically, so ``=`` never matched and ``>=`` degraded to ``>``.
B-4  ``IN [datetime('...')]`` returned nothing on BOTH engines (invisible to
     differential testing -- pinned per engine here, not by parity).
A-1/A-2/A-3  UNION: pandas failed to dedup ``NaN`` against ``None``; polars'
     ``vertical_relaxed`` concat stringified a non-string branch (an EMPTY branch alone
     was enough) and then deleted rows via the DISTINCT; and BOTH engines conflated
     BOOLEAN with INTEGER (openCypher ``true = 1`` is false).

Sibling spellings that were already correct (``localdatetime()``, ``date()``, a plain
ISO string, and a tz-AWARE column) are pinned alongside as the regression fence.
"""
from __future__ import annotations

import pandas as pd
import pytest

import graphistry

pl = pytest.importorskip("polars")

ENGINES = ["pandas", "polars"]


# ---------------------------------------------------------------- fixtures

# n1 2020-01-02 03:04:05, n2 2020-03-04 00:00:00, n3 2021-06-07 23:59:59, n4 null.
# The same three instants are carried as a real datetime64 column (``dt_native``), a
# tz-aware column (``dt_utc``) and as ISO TEXT (``d_str`` / ``dt_str``).
TEMPORAL_NODES = pd.DataFrame({
    "id": ["n1", "n2", "n3", "n4"],
    "d_str": ["2020-01-02", "2020-03-04", "2021-06-07", None],
    "dt_str": ["2020-01-02T03:04:05", "2020-03-04T00:00:00", "2021-06-07T23:59:59", None],
    "dt_native": pd.to_datetime(
        ["2020-01-02 03:04:05", "2020-03-04 00:00:00", "2021-06-07 23:59:59", None]
    ),
    "dt_utc": pd.to_datetime(
        ["2020-01-02 03:04:05Z", "2020-03-04 00:00:00Z", "2021-06-07 23:59:59Z", None],
        utc=True,
    ),
    "v": [1, 2, 3, 4],
})
TEMPORAL_EDGES = pd.DataFrame({
    "src": ["n1", "n2", "n3"],
    "dst": ["n2", "n3", "n4"],
})

# txt is the STRING spelling of what i holds as an INTEGER, so a UNION that stringifies
# one branch collapses six openCypher-distinct values into three.
UNION_NODES = pd.DataFrame({
    "id": ["a", "b", "c"],
    "txt": ["7", "8", "9"],
    "i": [7, 8, 9],
    "v": [1, 2, 3],
    "flag": [True, False, True],
    "fnan": [float("nan"), 1.5, float("nan")],
})
UNION_EDGES = pd.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]})


def temporal_graph():
    return graphistry.nodes(TEMPORAL_NODES, "id").edges(TEMPORAL_EDGES, "src", "dst")


def union_graph():
    return graphistry.nodes(UNION_NODES, "id").edges(UNION_EDGES, "src", "dst")


def _ids(result, col="id"):
    nodes = result._nodes
    if hasattr(nodes, "to_pandas"):
        nodes = nodes.to_pandas()
    return sorted(nodes[col].tolist())


def _values(result, col="x"):
    nodes = result._nodes
    if hasattr(nodes, "to_pandas"):
        nodes = nodes.to_pandas()
    return nodes[col].tolist()


# ---------------------------------------------------------------- B-1

# All ten spellings of the SAME predicate over the SAME tz-naive datetime column.
# dt_native = 2020-01-02 03:04:05 | 2020-03-04 | 2021-06-07 23:59:59 | NaT
B1_SPELLINGS = [
    ("gt", "MATCH (n) WHERE n.dt_native > datetime('2020-02-01T00:00:00') RETURN n.id AS id", ["n2", "n3"]),
    ("gt_z", "MATCH (n) WHERE n.dt_native > datetime('2020-02-01T00:00:00Z') RETURN n.id AS id", ["n2", "n3"]),
    ("lt", "MATCH (n) WHERE n.dt_native < datetime('2020-02-01T00:00:00') RETURN n.id AS id", ["n1"]),
    ("ge_boundary", "MATCH (n) WHERE n.dt_native >= datetime('2020-01-02T03:04:05') RETURN n.id AS id", ["n1", "n2", "n3"]),
    ("eq", "MATCH (n) WHERE n.dt_native = datetime('2020-03-04T00:00:00') RETURN n.id AS id", ["n2"]),
    ("property_map", "MATCH (n {dt_native: datetime('2020-03-04T00:00:00')}) RETURN n.id AS id", ["n2"]),
    ("and_chained", "MATCH (n) WHERE n.v > 0 AND n.dt_native > datetime('2020-02-01T00:00:00') RETURN n.id AS id", ["n2", "n3"]),
    ("or_chained", "MATCH (n) WHERE n.id = 'n1' OR n.dt_native > datetime('2020-02-01T00:00:00') RETURN n.id AS id", ["n1", "n2", "n3"]),
    ("edge_endpoint", "MATCH (n)-[e]->(m) WHERE m.dt_native > datetime('2020-02-01T00:00:00') RETURN n.id AS id", ["n1", "n2"]),
]


@pytest.mark.parametrize("name,query,expected", B1_SPELLINGS, ids=[c[0] for c in B1_SPELLINGS])
def test_b1_datetime_vs_native_datetime_column_matches(name, query, expected):
    """#1915 B-1: the canonical Cypher spelling matched ZERO rows on a datetime64 column."""
    assert _ids(temporal_graph().gfql(query, engine="pandas")) == expected


def test_b1_count_star_spelling():
    result = temporal_graph().gfql(
        "MATCH (n) WHERE n.dt_native > datetime('2020-02-01T00:00:00') RETURN count(*) AS c",
        engine="pandas",
    )
    assert _values(result, "c") == [2]


def test_b1_not_of_the_predicate_is_the_complement_not_everything():
    """openCypher: ``NOT`` of a null is null, so only rows where the predicate is
    genuinely FALSE come back. The old silent-False made this return all 4 non-null
    rows -- the tell that the comparison had evaluated false rather than null."""
    result = temporal_graph().gfql(
        "MATCH (n) WHERE NOT (n.dt_native > datetime('2020-02-01T00:00:00')) RETURN n.id AS id",
        engine="pandas",
    )
    assert _ids(result) == ["n1"]


# Regression fence: spellings that were ALREADY correct on the same column.
B1_FENCE = [
    ("localdatetime", "MATCH (n) WHERE n.dt_native > localdatetime('2020-02-01T00:00:00') RETURN n.id AS id", ["n2", "n3"]),
    ("date", "MATCH (n) WHERE n.dt_native > date('2020-02-01') RETURN n.id AS id", ["n2", "n3"]),
    ("plain_string", "MATCH (n) WHERE n.dt_native > '2020-02-01T00:00:00' RETURN n.id AS id", ["n2", "n3"]),
    ("tz_aware_column", "MATCH (n) WHERE n.dt_utc > datetime('2020-02-01T00:00:00Z') RETURN n.id AS id", ["n2", "n3"]),
    ("is_null", "MATCH (n) WHERE n.dt_native IS NULL RETURN n.id AS id", ["n4"]),
    ("is_not_null", "MATCH (n) WHERE n.dt_native IS NOT NULL RETURN n.id AS id", ["n1", "n2", "n3"]),
]


@pytest.mark.parametrize("name,query,expected", B1_FENCE, ids=[c[0] for c in B1_FENCE])
def test_b1_sibling_spellings_stay_correct(name, query, expected):
    assert _ids(temporal_graph().gfql(query, engine="pandas")) == expected


def test_b1_order_by_and_aggregates_on_native_datetime_unchanged():
    """ORDER BY keeps its native-dtype sort path (null LAST on ASC) and min/max/count
    keep skipping nulls -- the comparison fix must not reroute them."""
    ordered = temporal_graph().gfql("MATCH (n) RETURN n.dt_native AS t ORDER BY t", engine="pandas")
    values = _values(ordered, "t")
    assert [str(v) for v in values[:3]] == [
        "2020-01-02 03:04:05",
        "2020-03-04 00:00:00",
        "2021-06-07 23:59:59",
    ]
    assert pd.isna(values[3])

    agg = temporal_graph().gfql(
        "MATCH (n) RETURN max(n.dt_native) AS mx, min(n.dt_native) AS mn, count(n.dt_native) AS c",
        engine="pandas",
    )
    assert str(_values(agg, "mx")[0]) == "2021-06-07 23:59:59"
    assert str(_values(agg, "mn")[0]) == "2020-01-02 03:04:05"
    assert _values(agg, "c")[0] == 3


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_b1_native_datetime_comparison_is_resolution_independent(unit):
    """``astype('int64')`` returns the column's OWN ticks, not nanoseconds. Keying the
    native-datetime comparison as if the ticks were always ns collapsed every row onto
    the same Julian day for any coarser resolution, so the predicate matched nothing.

    pandas 3 made this the DEFAULT (``pd.to_datetime`` yields datetime64[us], not
    datetime64[ns]), but the same column shape is reachable on pandas 2 -- hence the
    pin is over resolutions, not over pandas versions."""
    nodes = TEMPORAL_NODES.assign(dt_native=TEMPORAL_NODES["dt_native"].astype(f"datetime64[{unit}]"))
    g = graphistry.nodes(nodes, "id").edges(TEMPORAL_EDGES, "src", "dst")
    assert _ids(g.gfql(
        "MATCH (n) WHERE n.dt_native > datetime('2020-02-01T00:00:00') RETURN n.id AS id",
        engine="pandas",
    )) == ["n2", "n3"]
    assert _ids(g.gfql(
        "MATCH (n) WHERE n.dt_native = datetime('2020-03-04T00:00:00') RETURN n.id AS id",
        engine="pandas",
    )) == ["n2"]


def test_b1_incomparable_mixed_comparison_is_null_not_false():
    """The general trap behind B-1: an element-wise TypeError inside the mixed-type
    comparison handler became ``False``. openCypher orders only within a type; anything
    else is null, so BOTH the predicate and its negation must drop the row."""
    nodes = pd.DataFrame({"id": ["a", "b"], "mixed": [1, "z"]})
    edges = pd.DataFrame({"src": ["a"], "dst": ["b"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    assert _ids(g.gfql("MATCH (n) WHERE n.mixed > 0 RETURN n.id AS id", engine="pandas")) == ["a"]
    # 'z' > 0 is incomparable -> null -> dropped by WHERE and by NOT alike.
    assert _ids(g.gfql("MATCH (n) WHERE NOT (n.mixed > 0) RETURN n.id AS id", engine="pandas")) == []


# ---------------------------------------------------------------- B-2

# Hand-computed against openCypher temporal arithmetic. A Duration keeps its month /
# day / second groups SEPARATE, so date + PT25H advances nothing while date + P1D does.
B2_ARITHMETIC = [
    ("date_plus_day", "RETURN date('2020-01-02') + duration('P1D') AS x", "2020-01-03"),
    ("date_plus_month_clamps", "RETURN date('2020-01-31') + duration('P1M') AS x", "2020-02-29"),
    ("date_plus_subday_is_noop", "RETURN date('2020-01-02') + duration('PT25H') AS x", "2020-01-02"),
    ("date_minus_day", "RETURN date('2020-01-02') - duration('P1D') AS x", "2020-01-01"),
    ("date_minus_month_clamps", "RETURN date('2020-03-31') - duration('P1M') AS x", "2020-02-29"),
    ("datetime_plus_hour", "RETURN datetime('2020-01-02T03:04:05Z') + duration('PT1H') AS x", "2020-01-02T04:04:05Z"),
    ("duration_plus_duration", "RETURN duration('P1D') + duration('PT1H') AS x", "P1DT1H"),
    ("duration_year_plus_month", "RETURN duration('P1Y') + duration('P1M') AS x", "P1Y1M"),
    ("duration_minus_duration", "RETURN duration('P2D') - duration('P1D') AS x", "P1D"),
    ("duration_times_int", "RETURN duration('P1D') * 2 AS x", "P2D"),
    ("int_times_duration", "RETURN 2 * duration('PT30M') AS x", "PT1H"),
    ("duration_div_int", "RETURN duration('PT1H') / 2 AS x", "PT30M"),
    ("duration_plus_date_commutes", "RETURN duration('P1D') + date('2020-01-02') AS x", "2020-01-03"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name,query,expected", B2_ARITHMETIC, ids=[c[0] for c in B2_ARITHMETIC])
def test_b2_temporal_arithmetic_is_arithmetic_not_concatenation(engine, name, query, expected):
    """#1915 B-2: ``+`` was Python ``str + str``. Folded at lowering, so both engines agree."""
    assert _values(union_graph().gfql(query, engine=engine)) == [expected]


@pytest.mark.parametrize("engine", ENGINES)
def test_b2_temporal_arithmetic_changes_the_row_set(engine):
    """Row-set-visible, not just display: comparing against ``'2020-01-01P1Y'`` admitted
    2020-03-04 lexicographically. Oracle: > 2021-01-01 keeps only n3."""
    result = temporal_graph().gfql(
        "MATCH (n) WHERE n.d_str > date('2020-01-01') + duration('P1Y') RETURN n.id AS id",
        engine=engine,
    )
    assert _ids(result) == ["n3"]


def test_b2_column_form_declines_typed_like_its_minus_sibling():
    """Not foldable (a column operand), so ``+`` must decline exactly as ``-`` already
    does. Silently concatenating is not an option."""
    from graphistry.compute.exceptions import GFQLValidationError
    for op in ("+", "-"):
        with pytest.raises(GFQLValidationError):
            temporal_graph().gfql(
                f"MATCH (n) RETURN n.d_str {op} duration('P1D') AS x", engine="pandas"
            )


def test_b2_ordinary_string_concatenation_is_untouched():
    """Only ISO-duration-shaped operands engage the fold/decline."""
    result = union_graph().gfql("MATCH (n) RETURN n.id + 'x' AS x", engine="pandas")
    assert _values(result) == ["ax", "bx", "cx"]


def test_b2_order_by_column_plus_duration_still_works():
    """ORDER BY has its own temporal-duration path and must keep serving the column form."""
    result = temporal_graph().gfql(
        "MATCH (n) WITH n ORDER BY n.d_str + duration('P1D') RETURN n.id AS id", engine="pandas"
    )
    nodes = result._nodes
    assert nodes["id"].tolist()[:3] == ["n1", "n2", "n3"]


# ---------------------------------------------------------------- B-3 / B-4

B3_TEXT_TEMPORAL = [
    ("eq", "MATCH (n) WHERE n.dt_str = datetime('2020-03-04T00:00:00') RETURN n.id AS id", ["n2"]),
    ("ge_includes_boundary", "MATCH (n) WHERE n.dt_str >= datetime('2020-03-04T00:00:00') RETURN n.id AS id", ["n2", "n3"]),
    ("gt_excludes_boundary", "MATCH (n) WHERE n.dt_str > datetime('2020-03-04T00:00:00') RETURN n.id AS id", ["n3"]),
    ("ne", "MATCH (n) WHERE n.dt_str <> datetime('2020-03-04T00:00:00') RETURN n.id AS id", ["n1", "n3"]),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name,query,expected", B3_TEXT_TEMPORAL, ids=[c[0] for c in B3_TEXT_TEMPORAL])
def test_b3_z_suffixed_literal_vs_text_temporal_column(engine, name, query, expected):
    """#1915 B-3: on polars ``=`` never matched and ``>=`` silently degraded to ``>``."""
    assert _ids(temporal_graph().gfql(query, engine=engine)) == expected


B4_IN_SPELLINGS = [
    ("datetime", "MATCH (n) WHERE n.dt_str IN [datetime('2020-03-04T00:00:00')] RETURN n.id AS id", ["n2"]),
    ("localdatetime", "MATCH (n) WHERE n.dt_str IN [localdatetime('2020-03-04T00:00:00')] RETURN n.id AS id", ["n2"]),
    ("plain_string", "MATCH (n) WHERE n.dt_str IN ['2020-03-04T00:00:00'] RETURN n.id AS id", ["n2"]),
    ("date", "MATCH (n) WHERE n.d_str IN [date('2020-03-04')] RETURN n.id AS id", ["n2"]),
    ("two_elements", "MATCH (n) WHERE n.dt_str IN [datetime('2020-03-04T00:00:00'), datetime('2020-01-02T03:04:05')] RETURN n.id AS id", ["n1", "n2"]),
    ("no_match", "MATCH (n) WHERE n.dt_str IN [datetime('2019-01-01T00:00:00')] RETURN n.id AS id", []),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name,query,expected", B4_IN_SPELLINGS, ids=[c[0] for c in B4_IN_SPELLINGS])
def test_b4_in_list_of_temporal_literals(engine, name, query, expected):
    """#1915 B-4: BOTH engines returned nothing for ``IN [datetime(...)]`` while ``=``
    against the same literal matched. Engine agreement cannot catch this, so it is
    pinned per engine against the hand oracle."""
    assert _ids(temporal_graph().gfql(query, engine=engine)) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_b4_non_temporal_in_is_untouched(engine):
    assert _ids(temporal_graph().gfql(
        "MATCH (n) WHERE n.id IN ['n1','n2'] RETURN n.id AS id", engine=engine
    )) == ["n1", "n2"]


# ---------------------------------------------------------------- A-1 / A-2 / A-3

@pytest.mark.parametrize("engine", ENGINES)
def test_a2_empty_branch_does_not_retype_the_surviving_branch(engine):
    """#1915 A-2: an EMPTY String branch alone made polars' ``vertical_relaxed`` concat
    stringify the surviving Int64 branch (``['7','8','9']`` instead of ``[7,8,9]``).
    openCypher: an empty branch contributes no rows -- and no type either."""
    result = union_graph().gfql(
        "MATCH (n) WHERE n.v > 100 RETURN n.txt AS x UNION ALL MATCH (n) RETURN n.i AS x",
        engine=engine,
    )
    assert _values(result) == [7, 8, 9]


def test_a2_mixed_type_union_keeps_all_six_values_on_pandas():
    """branch1 = {'7','8','9'} STRINGs, branch2 = {7,8,9} INTEGERs; no cross-type
    equality in openCypher, so DISTINCT keeps all six."""
    result = union_graph().gfql(
        "MATCH (n) RETURN n.txt AS x UNION MATCH (n) RETURN n.i AS x", engine="pandas"
    )
    assert _values(result) == ["7", "8", "9", 7, 8, 9]


def test_a2_mixed_type_union_declines_typed_on_polars():
    """A polars column cannot hold both branches' values; declining is the only honest
    answer left (the old behaviour stringified and then deleted three rows)."""
    with pytest.raises(NotImplementedError):
        union_graph().gfql(
            "MATCH (n) RETURN n.txt AS x UNION MATCH (n) RETURN n.i AS x", engine="polars"
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_a2_numeric_widening_union_stays_served(engine):
    """Regression fence: openCypher ``1 = 1.0`` is true, so Int/Float branches may widen."""
    result = union_graph().gfql(
        "MATCH (n) RETURN n.v AS x UNION MATCH (n) RETURN n.fnan AS x", engine=engine
    )
    values = _values(result)
    assert [v for v in values if v == v] == [1, 2, 3, 1.5]


def test_a1_pandas_union_dedups_nan_against_none():
    """#1915 A-1: ``n.fnan`` for row a is a MISSING value; both branches produce null,
    so openCypher UNION yields one row. The object-column concat hashed float NaN and
    None apart, giving two."""
    result = union_graph().gfql(
        "MATCH (n) WHERE n.id = 'a' RETURN n.fnan AS x UNION RETURN null AS x", engine="pandas"
    )
    assert len(result._nodes) == 1


def test_a3_boolean_and_integer_are_distinct_across_union_branches_pandas():
    """#1915 A-3: openCypher ``true = 1`` is FALSE, so both survive DISTINCT. pandas'
    concat upcast ``True`` to ``1`` and the dedup then deleted a row."""
    scalar = union_graph().gfql("RETURN true AS x UNION RETURN 1 AS x", engine="pandas")
    assert _values(scalar) == [True, 1]

    column = union_graph().gfql(
        "MATCH (n) RETURN n.flag AS x UNION MATCH (n) WHERE n.v < 3 RETURN n.v AS x",
        engine="pandas",
    )
    assert _values(column) == [True, False, 1, 2]


def test_a3_boolean_vs_integer_union_declines_typed_on_polars():
    with pytest.raises(NotImplementedError):
        union_graph().gfql("RETURN true AS x UNION RETURN 1 AS x", engine="polars")


@pytest.mark.parametrize("engine", ENGINES)
def test_a_union_regression_fence(engine):
    """Shapes the probe found CORRECT: same-type dedup, both-branches-empty schema,
    null == null dedup, and UNION ALL multiplicity."""
    g = union_graph()
    assert len(g.gfql(
        "MATCH (n) RETURN n.txt AS x UNION MATCH (n) RETURN n.txt AS x", engine=engine
    )._nodes) == 3
    both_empty = g.gfql(
        "MATCH (n) WHERE n.v > 100 RETURN n.id AS x UNION MATCH (n) WHERE n.v > 100 RETURN n.id AS x",
        engine=engine,
    )
    assert len(both_empty._nodes) == 0 and "x" in list(both_empty._nodes.columns)
    assert len(g.gfql("RETURN null AS x UNION RETURN null AS x", engine=engine)._nodes) == 1
    assert len(g.gfql("RETURN null AS x UNION ALL RETURN null AS x", engine=engine)._nodes) == 2
