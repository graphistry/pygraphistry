"""STARTS WITH / ENDS WITH / CONTAINS / =~ accept any right operand; null unless both sides are strings (#2374)."""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError


@pytest.fixture(params=["pandas", "cudf", "polars", "polars-gpu"])
def engine(request):
    if request.param == "cudf":
        pytest.importorskip("cudf")
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("cudf tests need TEST_CUDF=1")
    elif request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
    elif request.param == "polars":
        pytest.importorskip("polars")
    return request.param


_NODES = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "name": ["abcdef", "abcdef", "xyz", "q"],
    "pre": ["abc", "xyz", None, "q"],
    "suf": ["def", "abc", "z", None],
    "sub": ["cd", "zz", "y", "q"],
})
_EDGES = pd.DataFrame({"s": [1], "d": [2]})


def _graph(engine):
    nodes, edges = _NODES, _EDGES
    if engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _run(engine, query):
    out = _graph(engine).gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in out.to_dict("records")]


_POLARS = ("polars", "polars-gpu")
_UNWIND_DECLINE = "row op 'with_'"  # the Polars engines have no native WITH-of-a-list / UNWIND row op, with or without a string predicate


def _rows(engine, query):
    """Rows of the query. Every engine answers; the Polars engines lower the three operators natively."""
    return _run(engine, query)


def _rows_with_unwind(engine, query):
    """Rows of a query that builds its operands with WITH [...] / UNWIND: the Polars engines decline that exact row op, as on develop."""
    if engine in _POLARS:
        with pytest.raises(NotImplementedError, match=_UNWIND_DECLINE):
            _run(engine, query)
        return None
    return _run(engine, query)


@pytest.mark.parametrize("op,right", [("STARTS WITH", "n.pre"), ("ENDS WITH", "n.suf"), ("CONTAINS", "n.sub")])
def test_a_match_where_across_two_properties_still_declines_at_compile_time(engine, op, right):
    """The MATCH ... WHERE same-path form carries comparisons only; a string operator across properties stays an explicit compile-time decline."""
    with pytest.raises(GFQLValidationError, match="cross-property"):
        _graph(engine).gfql(f"MATCH (n) WHERE n.name {op} {right} RETURN n.id AS i", engine=engine)


@pytest.mark.parametrize("op,right,expected_ids", [
    ("STARTS WITH", "n.pre", [1, 4]),
    ("ENDS WITH", "n.suf", [1, 3]),
    ("CONTAINS", "n.sub", [1, 3, 4]),
])
def test_a_with_stage_filter_with_a_column_right_operand(engine, op, right, expected_ids):
    rows = _rows(engine, f"MATCH (n) WITH n.id AS i, n.name AS name, {right} AS other WHERE name {op} other RETURN i ORDER BY i")
    assert [row["i"] for row in rows] == expected_ids


def test_return_is_null_when_either_side_is_null(engine):
    rows = _rows(engine, "MATCH (n) RETURN n.id AS i, n.name STARTS WITH n.pre AS v ORDER BY i")
    assert [row["v"] for row in rows] == [True, False, None, True]


def test_literal_right_operands_are_unchanged(engine):
    rows = _rows(engine, "MATCH (n) WHERE n.name STARTS WITH 'abc' RETURN n.id AS i ORDER BY i")
    assert [row["i"] for row in rows] == [1, 2]


@pytest.mark.parametrize("op", ["STARTS WITH", "ENDS WITH", "CONTAINS"])
def test_every_pair_of_mixed_values_is_null_unless_both_are_strings(engine, op):
    query = (
        "WITH ['ab', 'b', 1, 3.14, true, [], {}, null] AS operands UNWIND operands AS op1 UNWIND operands AS op2 "
        f"WITH op1 {op} op2 AS v RETURN v, count(*) AS c"
    )
    rows = _rows_with_unwind(engine, query)
    if rows is None:
        return
    by_value = {row["v"]: row["c"] for row in rows}
    expected_true = {"STARTS WITH": 2, "ENDS WITH": 3, "CONTAINS": 3}[op]  # of the 4 string pairs: ab/ab and b/b, plus ab/b for ENDS WITH and CONTAINS
    assert by_value.get(True) == expected_true
    assert by_value.get(False) == 4 - expected_true
    assert by_value.get(None) == 64 - 4


@pytest.mark.parametrize("op", ["STARTS WITH", "ENDS WITH", "CONTAINS"])
def test_the_tck_non_string_operand_shape_is_all_null(engine, op):
    query = (
        "WITH [1, 3.14, true, [], {}, null] AS operands UNWIND operands AS op1 UNWIND operands AS op2 "
        f"WITH op1 {op} op2 AS v RETURN v, count(*) AS c"
    )
    rows = _rows_with_unwind(engine, query)
    assert rows is None or rows == [{"v": None, "c": 36}]


def test_a_regex_with_a_column_pattern_is_answered_by_pandas_and_declined_by_name_elsewhere(engine):
    query = "MATCH (n) WITH n.id AS i, n.name AS name, n.pre AS other WHERE name =~ other RETURN i ORDER BY i"
    if engine == "pandas":
        assert [row["i"] for row in _run(engine, query)] == [4]
    else:
        with pytest.raises(NotImplementedError):
            _run(engine, query)


def test_the_column_pattern_result_has_the_literal_operand_result_dtype(engine):
    if engine in _POLARS:
        pytest.skip("the Polars engines return a Polars Boolean column on both paths")
    literal = _graph(engine).gfql("MATCH (n) RETURN n.id AS i, n.name STARTS WITH 'abc' AS v ORDER BY i", engine=engine)._nodes
    column = _graph(engine).gfql("MATCH (n) RETURN n.id AS i, n.name STARTS WITH n.pre AS v ORDER BY i", engine=engine)._nodes
    assert str(column["v"].dtype) == str(literal["v"].dtype)
