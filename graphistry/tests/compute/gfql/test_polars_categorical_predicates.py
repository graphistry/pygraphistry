"""Categorical and Enum columns keep their dictionary comparisons; only string predicates decode them (#2335)."""
import pytest

pl = pytest.importorskip("polars")

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
