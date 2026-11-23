"""STARTS WITH / ENDS WITH / CONTAINS / =~ accept any right operand; null unless both sides are strings (#2374)."""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.row import pipeline as row_pipeline
from graphistry.compute.gfql.row.dispatch import apply_string_predicate_scalar
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


def test_a_regex_with_a_column_pattern_is_answered_on_every_engine_that_lowers_it(engine):
    query = "MATCH (n) WITH n.id AS i, n.name AS name, n.pre AS other WHERE name =~ other RETURN i ORDER BY i"
    if engine in _POLARS:
        with pytest.raises(NotImplementedError):  # no per-row regex form that matches Python re; lowering it would risk a silent Rust-regex divergence
            _run(engine, query)
    else:
        assert [row["i"] for row in _run(engine, query)] == [4]


def test_the_column_pattern_result_has_the_literal_operand_result_dtype(engine):
    if engine in _POLARS:
        pytest.skip("the Polars engines return a Polars Boolean column on both paths")
    # both results contain a null (pre is null on one row), so they are compared like for like
    literal = _graph(engine).gfql("MATCH (n) RETURN n.id AS i, n.pre STARTS WITH 'a' AS v ORDER BY i", engine=engine)._nodes
    column = _graph(engine).gfql("MATCH (n) RETURN n.id AS i, n.name STARTS WITH n.pre AS v ORDER BY i", engine=engine)._nodes
    assert str(column["v"].dtype) == str(literal["v"].dtype)


# The column form on pandas: numpy string functions over both columns, checked against the scalar definition per pair.
_PAIRS = [
    ("abcdef", "abc"), ("abcdef", "def"), ("abcdef", "cd"), ("abcdef", "xyz"), ("abc", ""), ("", ""), ("", "a"), ("x", "xx"),
    ("a.b", "."), ("a+b", "a+"), ("é-ü", "ü"), ("日本語", "本"), ("日本語", "日本語x"), ("tab\there", "\t"),
]
_OPS = ["contains", "starts_with", "ends_with"]


def _column_form(left, right, op):
    return row_pipeline.RowPipelineMixin._gfql_string_predicate_per_row(pd.Series(left), pd.Series(right), op, "x")


@pytest.mark.parametrize("op", _OPS)
def test_the_pandas_column_form_equals_the_scalar_definition_for_every_pair(op):
    left, right = [pair[0] for pair in _PAIRS], [pair[1] for pair in _PAIRS]
    assert list(_column_form(left, right, op)) == [apply_string_predicate_scalar(l, r, op) for l, r in _PAIRS]


@pytest.mark.parametrize("op", _OPS)
def test_values_with_a_nul_character_take_the_row_by_row_path_and_stay_exact(op):
    pairs = [("ab\x00", "b\x00"), ("ab\x00", "b"), ("a\x00b", "\x00"), ("ab", "ab\x00")]
    result = _column_form([l for l, _ in pairs], [r for _, r in pairs], op)
    assert list(result) == [apply_string_predicate_scalar(l, r, op) for l, r in pairs]


@pytest.mark.parametrize("op", _OPS)
def test_a_padded_array_over_the_byte_budget_takes_the_row_by_row_path_and_stays_exact(monkeypatch, op):
    monkeypatch.setattr(row_pipeline, "_GFQL_STRING_PREDICATE_NUMPY_BYTES", 0)
    left, right = [pair[0] for pair in _PAIRS], [pair[1] for pair in _PAIRS]
    assert list(_column_form(left, right, op)) == [apply_string_predicate_scalar(l, r, op) for l, r in _PAIRS]


@pytest.mark.parametrize("op", _OPS + ["regex"])
def test_a_cell_that_is_not_a_string_is_null_and_never_stringified(op):
    result = _column_form(["5", 5, "abc", None, 3.5, True], ["5", "5", None, "a", "3", "T"], op)
    assert [None if pd.isna(v) else bool(v) for v in result] == [True, None, None, None, None, None]


@pytest.mark.parametrize("op", _OPS + ["regex"])
def test_a_column_with_no_strings_is_all_null(op):
    assert all(pd.isna(v) for v in _column_form([1, 2, 3], ["a", "b", "c"], op))


def test_a_regex_column_is_answered_one_pass_per_distinct_pattern_like_the_literal_operand():
    left = ["abc", "abd", "ABC", "xyz", "abc"]
    right = ["ab.", "ab.", "(?i)abc", "x.z", "a"]
    assert [None if pd.isna(v) else bool(v) for v in _column_form(left, right, "regex")] == [True, True, True, True, False]


def test_a_regex_column_of_many_distinct_patterns_goes_row_by_row_with_the_same_answer(monkeypatch):
    left, right = ["abc", "abd", "xyz", "abc"], ["ab.", "ab.", "x.z", "a"]
    expected = [bool(v) for v in _column_form(left, right, "regex")]
    monkeypatch.setattr(row_pipeline, "_GFQL_STRING_PREDICATE_MAX_PATTERNS", 1)
    assert [bool(v) for v in _column_form(left, right, "regex")] == expected


@pytest.mark.parametrize("max_patterns", [256, 1])
def test_an_invalid_regex_in_the_column_is_a_value_error_on_both_regex_paths(monkeypatch, max_patterns):
    monkeypatch.setattr(row_pipeline, "_GFQL_STRING_PREDICATE_MAX_PATTERNS", max_patterns)
    with pytest.raises(ValueError, match="invalid regex pattern"):
        _column_form(["abc", "abd"], ["(", "ab"], "regex")


def test_a_categorical_column_pattern_matches_like_strings():
    left = pd.Series(["abc", "abd", None]).astype("category")
    right = pd.Series(["ab", "x", "a"]).astype("category")
    result = row_pipeline.RowPipelineMixin._gfql_string_predicate_per_row(left, right, "starts_with", "x")
    assert [None if pd.isna(v) else bool(v) for v in result] == [True, False, None]
