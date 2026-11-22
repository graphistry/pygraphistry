"""String predicates over categorical strings, mixed object columns and regex predicates (#2335 review)."""
import os

import pandas as pd
import pytest

import graphistry
from graphistry import n
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError, GFQLTypeError
from graphistry.compute.predicates.str import contains, endswith, fullmatch, match, startswith


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


def _graph(engine, nodes):
    edges = pd.DataFrame({"src": [1], "dst": [2]})
    if engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.edges(edges, "src", "dst").nodes(nodes, "id")


def _rows(frame):
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    return frame.to_dict("records")


def _values(engine, nodes, query, key="r"):
    rows = _rows(_graph(engine, nodes).gfql(query, engine=engine)._nodes)
    return [None if pd.isna(row[key]) else row[key] for row in rows]


_CATEGORICAL = pd.DataFrame({"id": [1, 2, 3], "v": pd.Categorical(["a5", "b6", "a7"])})


@pytest.mark.parametrize("predicate,expected_rows,expected_ids", [
    ("n.v CONTAINS '5'", [True, False, False], [1]),
    ("n.v STARTS WITH 'a'", [True, False, True], [1, 3]),
    ("n.v ENDS WITH '7'", [False, False, True], [3]),
])
def test_categorical_string_column_matches_alike_in_return_and_where(engine, predicate, expected_rows, expected_ids):
    """A categorical-of-string column is string-valued: RETURN and WHERE give the same answer on every engine."""
    returned = _values(engine, _CATEGORICAL, f"MATCH (n) RETURN n.id AS id, {predicate} AS r ORDER BY id")
    assert returned == expected_rows
    filtered = _values(engine, _CATEGORICAL, f"MATCH (n) WHERE {predicate} RETURN n.id AS r ORDER BY r")
    assert filtered == expected_ids


_MIXED = pd.DataFrame({"id": [1, 2, 3], "v": pd.Series([5, "a5", None], dtype="object")})


@pytest.mark.parametrize("predicate,expected_rows,expected_ids", [
    ("n.v CONTAINS '5'", [None, True, None], [2]),
    ("n.v STARTS WITH 'a'", [None, True, None], [2]),
])
def test_mixed_object_column_never_stringifies_a_non_string_cell(predicate, expected_rows, expected_ids):
    """pandas only (the other engines cannot hold a mixed column): the int cell is null, never '5'."""
    returned = _values("pandas", _MIXED, f"MATCH (n) RETURN n.id AS id, {predicate} AS r ORDER BY id")
    assert returned == expected_rows
    filtered = _values("pandas", _MIXED, f"MATCH (n) WHERE {predicate} RETURN n.id AS r ORDER BY r")
    assert filtered == expected_ids


_NUMERIC = pd.DataFrame({"id": [1, 2, 3], "v": [5, 15, 25]})


@pytest.mark.parametrize("make_predicate", [
    lambda: contains("5"),
    lambda: startswith("5"),
    lambda: endswith("5"),
    lambda: match("5.*"),
    lambda: fullmatch("5"),
], ids=["contains", "startswith", "endswith", "match", "fullmatch"])
@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
def test_native_string_predicates_on_a_non_string_column_keep_the_typed_error(engine, make_predicate, strict):
    """The native chain API is unchanged: every string predicate on a non-string column raises E302, on every engine."""
    with pytest.raises(GFQLSchemaError) as exc:
        _graph(engine, _NUMERIC).gfql([n({"v": make_predicate()})], engine=engine, strict=strict)
    assert exc.value.code == ErrorCode.E302


@pytest.mark.parametrize("op", ["CONTAINS", "STARTS WITH", "ENDS WITH"])
def test_the_same_predicate_in_cypher_is_null_for_a_non_string_column(engine, op):
    """Only Cypher-lowered predicates carry nonstring='null': no rows in WHERE, null in RETURN."""
    graph = _graph(engine, _NUMERIC)
    assert _rows(graph.gfql(f"MATCH (n) WHERE n.v {op} '5' RETURN n.id AS id", engine=engine)._nodes) == []
    returned = _rows(graph.gfql(f"MATCH (n) RETURN n.id AS id, n.v {op} '5' AS r ORDER BY id", engine=engine)._nodes)
    assert [None if pd.isna(row["r"]) else row["r"] for row in returned] == [None, None, None]


def test_nonstring_field_round_trips_and_is_omitted_at_its_default():
    from graphistry.compute.predicates.from_json import from_json

    for build in (contains, startswith, endswith):
        default = build("x")
        assert "nonstring" not in default.to_json()  # existing serialized chains are unchanged
        assert from_json(default.to_json()).nonstring == "error"
        marked = build("x", nonstring="null")
        assert marked.to_json()["nonstring"] == "null"
        assert from_json(marked.to_json()).nonstring == "null"
    with pytest.raises(GFQLTypeError):
        contains("x", nonstring="maybe").validate()


def test_polars_row_lowering_declines_an_unknown_operand_dtype(monkeypatch):
    pytest.importorskip("polars")
    from graphistry.compute.gfql.expr_parser import parse_expr
    from graphistry.compute.gfql.lazy.engine.polars import row_pipeline as polars_rows

    monkeypatch.setattr(polars_rows, "_expr_output_dtype", lambda expr: None)
    assert polars_rows.lower_expr(parse_expr("a CONTAINS 'x'"), ["a"]) is None


def test_always_null_predicate_round_trips_through_json():
    from graphistry.compute.predicates.from_json import from_json
    from graphistry.compute.predicates.str import AlwaysNull

    rebuilt = from_json(AlwaysNull().to_json())
    assert isinstance(rebuilt, AlwaysNull)
    assert rebuilt(pd.Series(["a", None, "b"])).isna().all()
