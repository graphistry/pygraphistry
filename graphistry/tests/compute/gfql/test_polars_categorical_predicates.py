"""Categorical and Enum columns keep their dictionary comparisons; only string predicates decode them (#2335)."""
import os

import pandas as pd
import pytest

pl = pytest.importorskip("polars")

import graphistry
from graphistry import n

from graphistry.compute.gfql.lazy.engine.polars.predicates import predicate_to_expr
from graphistry.compute.predicates.comparison import eq, ge, gt, le, lt, ne
from graphistry.compute.predicates.is_in import is_in
from graphistry.compute.predicates.numeric import between
from graphistry.compute.predicates.str import contains, endswith, startswith

# The Enum's category order (c < b < a) differs from the lexical order (a < b < c), which is what a String cast would use.
_ENUM = pl.Enum(["c", "b", "a"])
_FRAME = pl.DataFrame(
    {
        "e": pl.Series(["a", "b", "c", None], dtype=_ENUM),
        "k": pl.Series(["a", "b", "c", None], dtype=pl.Categorical),
    }
)


def _answer(column, predicate):
    expr = predicate_to_expr(column, predicate, _FRAME.schema[column])
    assert expr is not None
    assert "cast" not in str(expr), f"a comparison or membership must not decode the column: {expr}"
    return _FRAME.filter(expr)[column].cast(pl.String).to_list()


# (predicate, Enum answer in category order, Categorical answer)
_COMPARISONS = [
    ("eq_b", eq("b"), ["b"], ["b"]),
    ("ne_b", ne("b"), ["a", "c"], ["a", "c"]),
    ("gt_b", gt("b"), ["a"], ["c"]),
    ("lt_b", lt("b"), ["c"], ["a"]),
    ("ge_b", ge("b"), ["a", "b"], ["b", "c"]),
    ("le_b", le("b"), ["b", "c"], ["a", "b"]),
    ("between_a_b", between("a", "b"), [], ["a", "b"]),
    ("is_in_ab", is_in(["a", "b"]), ["a", "b"], ["a", "b"]),
]


@pytest.mark.parametrize("name,predicate,enum_answer,_cat", _COMPARISONS, ids=[case[0] for case in _COMPARISONS])
def test_an_enum_column_compares_in_category_order(name, predicate, enum_answer, _cat):
    assert _answer("e", predicate) == enum_answer


@pytest.mark.parametrize("name,predicate,_enum,cat_answer", _COMPARISONS, ids=[case[0] for case in _COMPARISONS])
def test_a_categorical_column_keeps_its_dictionary_comparison(name, predicate, _enum, cat_answer):
    assert _answer("k", predicate) == cat_answer


@pytest.mark.parametrize("column", ["e", "k"])
@pytest.mark.parametrize(
    "predicate,expected",
    [(startswith("a"), ["a"]), (endswith("c"), ["c"]), (contains("b"), ["b"])],
    ids=["startswith", "endswith", "contains"],
)
def test_string_predicates_still_decode_a_categorical_or_enum_column(column, predicate, expected):
    expr = predicate_to_expr(column, predicate, _FRAME.schema[column])
    assert expr is not None
    assert _FRAME.filter(expr)[column].cast(pl.String).to_list() == expected


# The same predicates through the native GFQL form, n({column: predicate}), on every Polars engine: the answers must equal the
# lowered-expression answers above. The one tolerated decline is polars-gpu on an older cudf-polars, matched by exact message.
_GPU_DECLINES = ("not fully GPU-executable", "cannot execute this aggregation")


@pytest.fixture(params=["polars", "polars-gpu"])
def native_engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
    return request.param


# The native between() takes numeric bounds only (it validates them), so a string-bounded between has no native form.
_NATIVE_COMPARISONS = [case for case in _COMPARISONS if case[0] != "between_a_b"]


def _native_answer(native_engine, column, predicate):
    nodes = _FRAME.with_row_index("id")
    edges = pl.from_pandas(pd.DataFrame({"src": [0], "dst": [1]}))
    graph = graphistry.edges(edges, "src", "dst").nodes(nodes, "id")
    try:
        out = graph.gfql([n({column: predicate})], engine=native_engine)._nodes
    except NotImplementedError as exc:
        assert native_engine == "polars-gpu" and any(known in str(exc) for known in _GPU_DECLINES), str(exc)
        return None
    return out[column].cast(pl.String).to_list()


@pytest.mark.parametrize("name,predicate,enum_answer,_cat", _NATIVE_COMPARISONS, ids=[case[0] for case in _NATIVE_COMPARISONS])
def test_an_enum_column_through_the_native_form(native_engine, name, predicate, enum_answer, _cat):
    answer = _native_answer(native_engine, "e", predicate)
    assert answer is None or answer == enum_answer


@pytest.mark.parametrize("name,predicate,_enum,cat_answer", _NATIVE_COMPARISONS, ids=[case[0] for case in _NATIVE_COMPARISONS])
def test_a_categorical_column_through_the_native_form(native_engine, name, predicate, _enum, cat_answer):
    answer = _native_answer(native_engine, "k", predicate)
    assert answer is None or answer == cat_answer
