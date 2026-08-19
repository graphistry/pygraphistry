"""Cross-alias grouped aggregates: sole min()/max() lowering + sum(bool) dtype (#1821).

Two families, every expected value HAND-COMPUTED on the fixtures below:

1. A grouped aggregate whose argument lives on a DIFFERENT MATCH alias than its
   group key (``RETURN c.city, min(p.age)``). The multiplicity-sensitive gate
   already routed sum/avg/collect/count(*) to binding rows; a SOLE
   multiplicity-INSENSITIVE aggregate (min/max/count DISTINCT) fell through to
   the one-source path and refused to lower. Binding rows are sound for those
   (their value is multiplicity-invariant), so the same route now serves them.

2. pandas ``sum()`` over an object-dtype boolean column: the kernel returned a
   lone-row group as the raw bool, mixing ``2`` and ``False`` in one output
   column. sum(bool) is the documented GFQL extension (see agg_types), so the
   accepted result must be a well-typed numeric column.

Fixture A (persons -> cities): persons 0(20), 1(30), 2(40); LIVES_IN edges
0->NYC, 1->NYC, 2->LA. Groups: NYC={20, 30}, LA={40}.
  min per city: LA=40, NYC=20;  max: LA=40, NYC=30;  sum: LA=40, NYC=50;
  count(DISTINCT p.age): LA=1, NYC=2.

Fixture B (path graph): nodes 0..5, grp x={0,1,2} y={3,4,5},
bool_col=[T,F,T,T,F,F], int_col=[1..6]; edges 0->1,1->2,2->3,3->4.
``WITH a.grp AS k, b.col AS v``: k=x pairs b in {1,2,3}, k=y pairs b in {4}.
  sum(int): x=2+3+4=9, y=5;  sum(bool): x=F+T+T=2, y=F=0.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]


NODES_A = pd.DataFrame({
    "id": [0, 1, 2, 10, 11],
    "node_type": ["Person", "Person", "Person", "City", "City"],
    "age": [20.0, 30.0, 40.0, None, None],
    "city": [None, None, None, "NYC", "LA"],
})
EDGES_A = pd.DataFrame({"s": [0, 1, 2], "d": [10, 10, 11], "rel": ["LIVES_IN"] * 3})

MATCH_A = ("MATCH (p {node_type:'Person'})-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) ")


def _graph_a(engine: str):
    if engine == "polars":
        return graphistry.nodes(pl.from_pandas(NODES_A), "id").edges(pl.from_pandas(EDGES_A), "s", "d")
    return graphistry.nodes(NODES_A, "id").edges(EDGES_A, "s", "d")


def _run(g, query: str, engine: str) -> pd.DataFrame:
    out = g.gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


def _records(df: pd.DataFrame):
    return [
        {k: (None if pd.isna(v) else (int(v) if isinstance(v, float) and float(v).is_integer() else v))
         for k, v in row.items()}
        for row in df.to_dict("records")
    ]


# ===========================================================================
# 1. Sole multiplicity-insensitive aggregates now lower onto binding rows
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("returns,expected", [
    ("RETURN c.city AS city, min(p.age) AS m ORDER BY city",
     [{"city": "LA", "m": 40}, {"city": "NYC", "m": 20}]),
    ("RETURN c.city AS city, max(p.age) AS m ORDER BY city",
     [{"city": "LA", "m": 40}, {"city": "NYC", "m": 30}]),
    ("RETURN c.city AS city, count(DISTINCT p.age) AS n ORDER BY city",
     [{"city": "LA", "n": 1}, {"city": "NYC", "n": 2}]),
    ("RETURN c.city AS city, min(p.age) AS m, max(p.age) AS x ORDER BY city",
     [{"city": "LA", "m": 40, "x": 40}, {"city": "NYC", "m": 20, "x": 30}]),
    ("WITH c.city AS city, min(p.age) AS m RETURN city, m ORDER BY city",
     [{"city": "LA", "m": 40}, {"city": "NYC", "m": 20}]),
], ids=["sole_min", "sole_max", "sole_count_distinct", "min_and_max", "with_stage_sole_min"])
def test_sole_insensitive_cross_alias_grouped_aggregate_serves(returns, expected, engine):
    assert _records(_run(_graph_a(engine), MATCH_A + returns, engine)) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_multiplicity_sensitive_route_is_value_identical(engine):
    """Anti-vacuity control: the sum route (already served at base) agrees with
    the newly served min route on the same binding rows."""
    out = _records(_run(
        _graph_a(engine),
        MATCH_A + "RETURN c.city AS city, sum(p.age) AS s, min(p.age) AS m ORDER BY city",
        engine))
    assert out == [{"city": "LA", "s": 40, "m": 40}, {"city": "NYC", "s": 50, "m": 20}]


@pytest.mark.parametrize("engine", ENGINES)
def test_single_alias_sole_min_still_uses_the_source_table_path(engine):
    """Anti-vacuity: one referenced alias never engages the cross-alias gate."""
    g = _graph_a(engine)
    assert _records(_run(g, MATCH_A + "RETURN min(p.age) AS m", engine)) == [{"m": 20}]
    assert _records(_run(g, MATCH_A + "RETURN max(c.city) AS m", engine)) == [{"m": "NYC"}]


def test_whole_row_grouped_sole_min_serves_on_pandas():
    """`RETURN c, min(p.age)` -- whole-row grouping with a sole insensitive
    aggregate rides the same binding-rows route on pandas."""
    out = _run(_graph_a("pandas"), MATCH_A + "RETURN c, min(p.age) AS m", "pandas")
    got = {row["c.city"]: int(row["m"]) for row in out.to_dict("records")}
    assert got == {"NYC": 20, "LA": 40}


@polars_only
def test_whole_row_grouped_sole_min_declines_typed_on_polars():
    """polars has no native whole-entity result projection for this shape;
    parity-or-error by design (NotImplementedError, never a silent wrong)."""
    with pytest.raises(NotImplementedError):
        _run(_graph_a("polars"), MATCH_A + "RETURN c, min(p.age) AS m", "polars")


@pytest.mark.parametrize("engine", ENGINES)
def test_mixed_aggregate_compound_item_keeps_the_fail_fast(engine):
    """The conservative one-source boundary survives the new gate: an item that
    COMBINES two aggregates in one expression still refuses to lower."""
    with pytest.raises(GFQLValidationError, match="one MATCH source alias at a time"):
        _run(_graph_a(engine),
             MATCH_A + "RETURN c.city AS city, min(p.age) + max(p.age) AS mm ORDER BY city",
             engine)


# ===========================================================================
# 2. pandas sum(bool) is a well-typed numeric column
# ===========================================================================


NODES_B = pd.DataFrame({
    "id": list(range(6)),
    "grp": ["x"] * 3 + ["y"] * 3,
    "int_col": [1, 2, 3, 4, 5, 6],
    "bool_col": [True, False, True, True, False, False],
})
EDGES_B = pd.DataFrame({"src": [0, 1, 2, 3], "dst": [1, 2, 3, 4]})

Q_B = "MATCH (a)-[e]->(b) WITH a.grp AS k, b.%s AS v RETURN k, sum(v) AS r ORDER BY k"


def _graph_b(engine: str):
    if engine == "polars":
        return graphistry.nodes(pl.from_pandas(NODES_B), "id").edges(pl.from_pandas(EDGES_B), "src", "dst")
    return graphistry.nodes(NODES_B, "id").edges(EDGES_B, "src", "dst")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("col,expected", [
    ("bool_col", [2, 0]),
    ("int_col", [9, 5]),
], ids=["sum_bool", "sum_int_control"])
def test_grouped_sum_column_is_numeric_not_mixed(col, expected, engine):
    out = _run(_graph_b(engine), Q_B % col, engine)
    values = out["r"].tolist()
    assert [int(v) for v in values] == expected
    # the defect was one column mixing int 2 and bool False: no bools, no object dtype
    assert all(not isinstance(v, bool) for v in values)
    assert str(out["r"].dtype) != "object"


def test_fast_path_sum_over_object_bools_is_numeric():
    """Twin pin for the OLAP single-hop grouped fast path's pandas branch: it
    must serve this labeled shape (engagement asserted, not assumed) and answer
    sum(object-of-bools) as ints. Oracle: NYC flags {True, True} -> 2, LA {False} -> 0."""
    import graphistry.compute.gfql_fast_paths as fast_paths
    import graphistry.compute.gfql_unified as unified

    nodes = NODES_A.assign(flag=pd.Series([True, True, False, None, None], dtype=object))
    g = graphistry.nodes(nodes, "id").edges(EDGES_A, "s", "d")
    served = []
    original = fast_paths._execute_single_hop_grouped_aggregate_fast_path

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        served.append(result is not None)
        return result

    unified._execute_single_hop_grouped_aggregate_fast_path = spy
    try:
        out = _run(g, MATCH_A + "RETURN c.city AS city, sum(p.flag) AS s ORDER BY city", "pandas")
    finally:
        unified._execute_single_hop_grouped_aggregate_fast_path = original
    assert served == [True]
    values = out["s"].tolist()
    assert [int(v) for v in values] == [0, 2]
    assert all(not isinstance(v, bool) for v in values)
    assert str(out["s"].dtype) != "object"
