"""`x IN [...]` is unknown on exactly the rows where `x = ...` is unknown (#2123).

A float column stores a Cypher null as NaN on pandas, and `IN` used to read NaN as a value while the
comparison operators read the same cell as null, so one row got two answers in one query:
`NOT (score IN [1, 2])` kept it and `NOT (score = 2)` dropped it. cuDF, which stores a real null,
dropped it both ways, so the engines disagreed too. A NaN *literal* is still a value, as under `=`.
"""
import math

import pandas as pd
import pytest

import graphistry

ENGINES = ["pandas", "polars", "cudf"]

_NODES = pd.DataFrame({"id": ["a", "b", "c", "tx1"], "score": [1.0, 2.0, None, 3.0]})
_EDGES = pd.DataFrame({"src": ["a", "b", "a"], "dst": ["b", "c", "tx1"]})


def _graph(engine):
    nodes, edges = _NODES, _EDGES
    if engine == "polars":
        pytest.importorskip("polars")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        pytest.importorskip("cupy")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.edges(edges, "src", "dst").nodes(nodes, "id")


def _ids(result):
    col = result._nodes["id"]
    col = col.to_pandas() if hasattr(col, "to_pandas") else col
    return sorted(col.tolist())


def _answer(engine, predicate):
    """Rows, or the typed decline polars raises for predicates it cannot lower."""
    query = f"MATCH (a)-[e]->(t) WHERE {predicate} RETURN t.id AS id"
    try:
        return _ids(_graph(engine).gfql(query, engine=engine))
    except NotImplementedError as exc:
        assert "natively" in str(exc), exc  # a decline, not a wrong answer
        return "declined"


@pytest.mark.parametrize("engine", ENGINES)
def test_in_and_equals_are_unknown_on_the_same_row(engine):
    # the null row (c) is unknown under both, so NOT drops it under both
    assert _answer(engine, "NOT (t.score IN [1, 2])") in (["tx1"], "declined")
    assert _answer(engine, "NOT (t.score = 2)") == ["tx1"]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_null_row_is_never_true_or_false_under_in(engine):
    assert _answer(engine, "t.score IN [1, 2]") in (["b"], "declined")          # c absent: unknown
    assert _answer(engine, "t.score IN [1, 2] OR t.score IS NULL") in (["b", "c"], "declined")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_a_null_element_lets_a_match_win_and_leaves_a_miss_unknown(engine):
    assert _answer(engine, "t.score IN [2, null]") in (["b"], "declined")       # match beats unknown
    assert _answer(engine, "t.score IN [9, null]") in ([], "declined")          # no match -> unknown


def test_a_null_element_in_the_list_is_a_typed_error_on_cudf_today():
    # Pre-existing and unchanged by this fix: measured on a GB10 with cudf 26.02, master and this
    # branch both raise here, while pandas answers ['b']. Pinned so the day cuDF lowers it we know.
    pytest.importorskip("cudf")
    pytest.importorskip("cupy")
    from graphistry.compute.exceptions import GFQLTypeError
    g = _graph("cudf")
    with pytest.raises(GFQLTypeError):
        g.gfql("MATCH (a)-[e]->(t) WHERE t.score IN [2, null] RETURN t.id AS id", engine="cudf")


@pytest.mark.parametrize("engine", ENGINES)
def test_an_empty_list_is_false_for_every_row_including_the_null_one(engine):
    assert _answer(engine, "t.score IN []") in ([], "declined")
    assert _answer(engine, "NOT (t.score IN [])") in (["b", "c", "tx1"], "declined")


def test_a_nan_literal_stays_a_value_under_in_as_it_is_under_equals():
    # the rule this issue did NOT change: a NaN produced by an expression is a value, and
    # `nan == nan` is False, so it matches nothing rather than making the row unknown
    nan = float("nan")
    assert math.isnan(nan)
    g = _graph("pandas")
    assert _ids(g.gfql("MATCH (a)-[e]->(t) WHERE t.score IN [0.0/0.0] RETURN t.id AS id", engine="pandas")) == []
    assert _ids(g.gfql("MATCH (a)-[e]->(t) WHERE NOT (t.score = 0.0/0.0) RETURN t.id AS id", engine="pandas")) \
        == ["b", "tx1"]


def test_the_reinstated_defect_would_fail_this_file():
    # Guard the fix at its seam: the isin lane must read a row's null-ness from the frame, not
    # from a per-scalar NaN rule. With the old narrowing the null row came back FALSE, not None.
    from graphistry.compute.gfql.row.pipeline import RowPipelineMixin
    series = pd.Series([1.0, 2.0, None, 3.0])
    out = RowPipelineMixin._gfql_in_literal_list_values(series, [1, 2])
    assert out == [True, True, None, False]
