"""``size()`` and its list-walking siblings must DECLINE a non-sequence column, never answer.

openCypher defines ``size()`` over lists and strings. Applied to a numeric/bool/temporal
column it is a type error. The pandas/cuDF row pipeline used to fall through to
``len(<containing frame>)``, so the answer was the TABLE ROW COUNT: it changed when unrelated
rows were added and was never a property of the data. ``WHERE size(n.age) = 3`` then kept
every row of a 3-row table and no row of a 4-row table.

Three call sites shared that swallow, and each gets both sides here:

==========================  ==========================  ==============================
surface                     declines (no defined size)  still serves (has a size)
==========================  ==========================  ==============================
``size(x)``                 int / float / bool column   string col, list col, literals
``any/all/none/single``     int column                  list column
``[x IN xs | ...]``         int column                  list column
``WHERE size(x) = k``       int column                  list column
==========================  ==========================  ==============================

The decline fires on EVIDENCE, not on dtype alone: a column with no non-null cell (an empty
zero-row intermediate, an all-null column) proves nothing about its element type — an empty
``collect()`` is still a list — so those answer null (or no rows) instead of being refused.

Every serving expectation is hand-computed from the fixtures below; ``size()`` over a
string column is CHARACTER length (openCypher), which all three engines already served and
which this decline must not touch. No engine is used as another engine's oracle.

The polars engine reaches these shapes through its own native lowering, which already
declines a non-sequence operand with a typed ``NotImplementedError``; it is pinned here so
the cross-engine story stays "value or typed decline, never the row count".
"""
from __future__ import annotations

import os
from typing import Any, List, Tuple, Union

import pandas as pd
import pytest

import graphistry
from graphistry.Plottable import Plottable
from graphistry.compute.exceptions import GFQLValidationError

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
cudf_only = pytest.mark.skipif(
    "TEST_CUDF" not in os.environ, reason="cuDF lane: set TEST_CUDF=1"
)

ENGINES = [
    "pandas",
    pytest.param("polars", marks=polars_only),
    pytest.param("cudf", marks=cudf_only),
]

#: Text every pandas/cuDF decline from the three fixed call sites must carry.
NAMED_LIMIT = "requires list/string input"

EDGES = pd.DataFrame({"s": [0], "d": [0]})


def _graph(engine: str, nodes: pd.DataFrame) -> Plottable:
    if engine == "polars":
        return graphistry.nodes(pl.from_pandas(nodes), "id").edges(
            pl.from_pandas(EDGES), "s", "d")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return graphistry.nodes(cudf.from_pandas(nodes), "id").edges(
            cudf.from_pandas(EDGES), "s", "d")
    return graphistry.nodes(nodes, "id").edges(EDGES, "s", "d")


def _ints(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({"id": list(range(n_rows)), "age": [10 + i for i in range(n_rows)]})


STRINGS = pd.DataFrame({"id": [0, 1, 2], "name": ["a", "bb", "ccc"]})
LISTS = pd.DataFrame({"id": [0, 1, 2], "tags": [["a"], ["a", "b"], []]})
FLOATS = pd.DataFrame({"id": [0, 1, 2], "v": [1.5, 2.5, 3.5]})
BOOLS = pd.DataFrame({"id": [0, 1, 2], "v": [True, False, True]})


def _cell(v: Any) -> Any:
    return None if isinstance(v, float) and v != v else v


def _run(g: Plottable, query: str, column: str, engine: str) -> Tuple[str, Union[str, List[Any]]]:
    """``("decline", message)`` or ``("values", [cells])`` — the only two acceptable outcomes."""
    try:
        out = g.gfql(query, engine=engine)
    except NotImplementedError as exc:
        return "decline", str(exc)
    except GFQLValidationError as exc:
        return "decline", str(exc)
    nodes = out._nodes
    if hasattr(nodes, "to_pandas"):
        nodes = nodes.to_pandas()
    return "values", [_cell(v) for v in list(nodes[column])]


def _assert_declines(outcome: Tuple[str, Any], engine: str, label: str) -> None:
    kind, payload = outcome
    assert kind == "decline", f"{engine}: {label} answered {payload!r} instead of declining"
    if engine != "polars":
        assert NAMED_LIMIT in payload, f"{engine}: {label} declined without naming the limit: {payload}"


class TestSizeOnNonSequenceColumnDeclines:

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize("n_rows", [3, 7])
    def test_size_of_int_column_declines_and_never_returns_the_row_count(
        self, engine: str, n_rows: int
    ) -> None:
        g = _graph(engine, _ints(n_rows))
        outcome = _run(g, "MATCH (n) RETURN size(n.age) AS z", "z", engine)
        _assert_declines(outcome, engine, f"size(int) over {n_rows} rows")

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize("nodes,label", [(FLOATS, "float"), (BOOLS, "bool")])
    def test_size_of_float_or_bool_column_declines(
        self, engine: str, nodes: pd.DataFrame, label: str
    ) -> None:
        g = _graph(engine, nodes)
        _assert_declines(_run(g, "MATCH (n) RETURN size(n.v) AS z", "z", engine), engine, f"size({label})")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_where_size_of_int_column_declines_instead_of_filtering_on_table_height(
        self, engine: str
    ) -> None:
        """The damaging form: the swallowed answer made this keep ALL rows of a 3-row table
        and NO row of a 4-row table, for the same data and the same predicate."""
        for n_rows in (3, 4):
            g = _graph(engine, _ints(n_rows))
            outcome = _run(g, "MATCH (n) WHERE size(n.age) = 3 RETURN n.id AS id", "id", engine)
            _assert_declines(outcome, engine, f"WHERE size(int)=3 over {n_rows} rows")


class TestSizeKeepsServingWhatHasASize:

    @pytest.mark.parametrize("engine", ENGINES)
    def test_size_of_string_column_is_character_length(self, engine: str) -> None:
        g = _graph(engine, STRINGS)
        assert _run(g, "MATCH (n) RETURN size(n.name) AS z", "z", engine) == ("values", [1, 2, 3])

    @pytest.mark.parametrize("engine", ENGINES)
    def test_size_of_list_column_is_element_count(self, engine: str) -> None:
        g = _graph(engine, LISTS)
        assert _run(g, "MATCH (n) RETURN size(n.tags) AS z", "z", engine) == ("values", [1, 2, 0])

    @pytest.mark.parametrize("engine", ENGINES)
    def test_size_of_all_null_column_is_null_not_a_decline(self, engine: str) -> None:
        nodes = pd.DataFrame({"id": [0, 1, 2], "nl": [None, None, None]})
        g = _graph(engine, nodes)
        assert _run(g, "MATCH (n) RETURN size(n.nl) AS z", "z", engine) == ("values", [None, None, None])

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize("literal,expected", [("[1,2,3]", 3), ("'abc'", 3)])
    def test_size_of_a_literal_still_counts_the_literal(
        self, engine: str, literal: str, expected: int
    ) -> None:
        g = _graph(engine, STRINGS)
        assert _run(g, f"MATCH (n) RETURN size({literal}) AS z", "z", engine) == (
            "values", [expected] * 3)


class TestUnknownElementTypeAnswersRatherThanDeclining:
    """A column with no non-null cell is UNKNOWN, not proven non-sequence — an empty
    ``collect()`` is still a list, so these must answer rather than be refused on dtype."""

    @pytest.mark.parametrize("engine", ENGINES)
    def test_size_of_all_null_float_column_is_null_not_the_row_count(self, engine: str) -> None:
        nodes = pd.DataFrame({"id": [0, 1, 2], "v": [float("nan")] * 3})
        g = _graph(engine, nodes)
        outcome = _run(g, "MATCH (n) RETURN size(n.v) AS z", "z", engine)
        if engine == "polars":
            _assert_declines(outcome, engine, "size(all-null float)")
        else:
            assert outcome == ("values", [None, None, None])

    @pytest.mark.parametrize("engine", ENGINES)
    def test_size_of_int_column_over_a_zero_row_table_serves_no_rows(self, engine: str) -> None:
        nodes = pd.DataFrame({"id": pd.Series([], dtype="int64"),
                              "age": pd.Series([], dtype="int64")})
        edges = pd.DataFrame({"s": pd.Series([], dtype="int64"),
                              "d": pd.Series([], dtype="int64")})
        if engine == "polars":
            g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
        elif engine == "cudf":
            cudf = pytest.importorskip("cudf")
            g = graphistry.nodes(cudf.from_pandas(nodes), "id").edges(
                cudf.from_pandas(edges), "s", "d")
        else:
            g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
        outcome = _run(g, "MATCH (n) RETURN size(n.age) AS z", "z", engine)
        if engine == "polars":
            _assert_declines(outcome, engine, "size(int) over 0 rows")
        else:
            assert outcome == ("values", [])

    @pytest.mark.parametrize("engine", ["pandas", pytest.param("cudf", marks=cudf_only)])
    def test_size_of_comprehension_over_an_empty_collect_is_zero(self, engine: str) -> None:
        """An OPTIONAL MATCH that binds nothing collects to ``[]``, whose size is 0."""
        nodes = pd.DataFrame({"id": ["n1"]})
        edges = pd.DataFrame({"s": [], "d": [], "type": []})
        if engine == "cudf":
            cudf = pytest.importorskip("cudf")
            g = graphistry.nodes(cudf.from_pandas(nodes), "id").edges(
                cudf.from_pandas(edges), "s", "d")
        else:
            g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
        query = ("MATCH (n) OPTIONAL MATCH (n)-[r]->(m) "
                 "RETURN size([x IN collect(r) WHERE x <> null]) AS cn")
        assert _run(g, query, "cn", engine) == ("values", [0])


class TestQuantifiersOverNonSequenceColumnDecline:

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize("fn", ["any", "all", "none", "single"])
    def test_quantifier_over_int_column_declines_instead_of_answering_from_zero_elements(
        self, engine: str, fn: str
    ) -> None:
        """The swallow made the element count 0, so any/single said False and all/none said
        True — four confident answers about a column that has no elements at all."""
        g = _graph(engine, _ints(3))
        outcome = _run(g, f"MATCH (n) RETURN {fn}(x IN n.age WHERE x > 0) AS z", "z", engine)
        _assert_declines(outcome, engine, f"{fn}(int)")
        if engine != "polars":
            assert f"{fn}()" in outcome[1], f"{engine}: decline must name {fn}(): {outcome[1]}"

    @pytest.mark.parametrize("engine", ENGINES)
    def test_quantifier_over_list_column_still_serves(self, engine: str) -> None:
        g = _graph(engine, LISTS)
        outcome = _run(g, "MATCH (n) RETURN any(x IN n.tags WHERE x = 'a') AS z", "z", engine)
        if engine == "polars":
            _assert_declines(outcome, engine, "any(list)")
        else:
            assert outcome == ("values", [True, True, False])


class TestListComprehensionOverNonSequenceColumnDeclines:

    @pytest.mark.parametrize("engine", ENGINES)
    def test_list_comprehension_over_int_column_declines_instead_of_yielding_empty(
        self, engine: str
    ) -> None:
        g = _graph(engine, _ints(3))
        outcome = _run(g, "MATCH (n) RETURN [x IN n.age | x] AS z", "z", engine)
        _assert_declines(outcome, engine, "comprehension(int)")
        if engine != "polars":
            assert "list comprehension" in outcome[1], \
                f"{engine}: decline must name the comprehension: {outcome[1]}"

    @pytest.mark.parametrize("engine", ENGINES)
    def test_list_comprehension_over_list_column_still_serves(self, engine: str) -> None:
        g = _graph(engine, LISTS)
        outcome = _run(g, "MATCH (n) RETURN [x IN n.tags | x] AS z", "z", engine)
        if engine == "polars":
            _assert_declines(outcome, engine, "comprehension(list)")
        else:
            assert outcome == ("values", [["a"], ["a", "b"], []])
