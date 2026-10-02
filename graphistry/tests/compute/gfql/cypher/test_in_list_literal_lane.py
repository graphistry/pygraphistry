"""``<column> IN [<literals>]`` is answered by one ``isin`` instead of a Python loop over
rows x elements, and must keep the loop's exact three-valued truth table.

Oracle: the element loop the lane replaces -- ``_gfql_cypher_value_equal`` folded with
Cypher's three-valued OR -- re-implemented here, so the lane is pinned to the OLD
semantics rather than to itself.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry import e_forward, n
from graphistry.compute.gfql.row.pipeline import RowPipelineMixin
from graphistry.compute.predicates.is_in import is_in


def _loop_oracle(lhs: Any, rhs: List[Any]) -> Optional[bool]:
    saw_unknown = False
    for element in rhs:
        verdict = RowPipelineMixin._gfql_cypher_value_equal(lhs, element)
        if verdict is True:
            return True
        if verdict is None:
            saw_unknown = True
    return None if saw_unknown else False


_SHAPES = [
    ("bool column, int element", pd.Series([True, False]), [1]),
    ("int column, bool element", pd.Series([1, 0, 2]), [True]),
    ("int column, float element", pd.Series([1, 2]), [1.0]),
    ("float column with NaN", pd.Series([np.nan, 1.0]), [1.0]),
    ("float NaN row, null in list", pd.Series([np.nan, 1.0]), [2.0, None]),
    ("object column with None", pd.Series(["a", None, "b"], dtype=object), ["b"]),
    ("object None row, null in list", pd.Series(["a", None, "b"], dtype=object), ["b", None]),
    ("object column holding a float NaN value", pd.Series(["a", float("nan"), None], dtype=object), ["b"]),
    ("str column, int element", pd.Series(["1", "2"]), [1]),
    ("nullable Int64", pd.Series([1, None, 3], dtype="Int64"), [3]),
    ("string dtype with NA", pd.Series(["a", None], dtype="string"), ["a"]),
    ("int beyond float precision", pd.Series([2**62, 5]), [2**62]),
    ("list-valued rows, scalar list", pd.Series([[1, 2], 3], dtype=object), [3]),
    ("map-valued rows, scalar list", pd.Series([{"a": 1}, 3], dtype=object), [3]),
    ("empty list, null row", pd.Series([1, None], dtype=object), []),
    ("list of only null", pd.Series([1, None], dtype=object), [None]),
    ("NaN element never matches", pd.Series([np.nan, 1.0]), [float("nan"), 1.0]),
    ("NaN element beside a null", pd.Series([np.nan, 1.0, 2.0]), [float("nan"), None, 2.0]),
    ("category column", pd.Series(["a", "b"], dtype="category"), ["a"]),
    ("datetime column with NaT", pd.Series(pd.to_datetime(["2020-01-01", None])), [pd.Timestamp("2020-01-01")]),
    ("NaT element is a null", pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01"])), [pd.NaT, pd.Timestamp("2020-01-01")]),
    ("pd.NA element is a null", pd.Series([1, 2]), [pd.NA, 2]),
]


@pytest.mark.parametrize("series,rhs", [(s, r) for _, s, r in _SHAPES], ids=[label for label, _, _ in _SHAPES])
def test_the_lane_matches_the_element_loop(series: pd.Series, rhs: List[Any]) -> None:
    expected = [_loop_oracle(value, rhs) for value in series.tolist()]
    assert RowPipelineMixin._gfql_in_literal_list_values(series, rhs) == expected


@pytest.mark.parametrize(
    "rhs",
    [[[1, 2]], [(1, 2)], [{"a": 1}], [1, [2]], pd.Series([[1]])],
    ids=["nested list", "nested tuple", "map element", "mixed scalar and list", "per-row list"],
)
def test_the_lane_declines_what_needs_structural_equality(rhs: Any) -> None:
    assert RowPipelineMixin._gfql_in_literal_list_values(pd.Series([1, 2]), rhs) is None


def _counting_equal(monkeypatch: pytest.MonkeyPatch) -> List[int]:
    calls = [0]
    original = RowPipelineMixin._gfql_cypher_value_equal

    def counted(left: Any, right: Any) -> Optional[bool]:
        calls[0] += 1
        return original(left, right)

    monkeypatch.setattr(RowPipelineMixin, "_gfql_cypher_value_equal", staticmethod(counted))
    return calls


def _graph() -> Any:
    rng = np.random.default_rng(7)
    nodes = pd.DataFrame({"id": np.arange(2_000), "kind": rng.choice(["person", "company"], 2_000)})
    edges = pd.DataFrame({"src": rng.integers(0, 2_000, 10_000), "dst": rng.integers(0, 2_000, 10_000)})
    return graphistry.edges(edges, "src", "dst").nodes(nodes, "id"), edges


def test_a_where_in_list_never_enters_the_element_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    g, edges = _graph()
    seeds = [3, 17, 99, 1500, 1999]
    calls = _counting_equal(monkeypatch)
    result = g.gfql(f"MATCH (a)-[e]->(b) WHERE a.id IN {seeds} RETURN b")
    assert calls[0] == 0, "the literal-list lane must answer IN without the per-element loop"
    expected = sorted(set(edges[edges["src"].isin(seeds)]["dst"].tolist()))
    assert sorted(result._nodes["b.id"].tolist()) == expected
    native = g.gfql([n({"id": is_in(seeds)}), e_forward(), n()])
    assert sorted(native._nodes[native._nodes["id"].isin(expected)]["id"].tolist()) == expected


def test_structural_equality_still_runs_through_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    g, _ = _graph()
    calls = _counting_equal(monkeypatch)
    result = g.gfql("RETURN [1, 2] IN [[1, 2]] AS hit")
    assert result._nodes["hit"].tolist() == [True]
    assert calls[0] > 0


def _scalar(g: Any, query: str) -> Any:
    frame = g.gfql(query)._nodes
    value = frame.iloc[0, 0]
    return None if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA else value


@pytest.mark.parametrize(
    "query,expected",
    [
        ("RETURN null IN [1] AS r", None),
        ("RETURN 1 IN [] AS r", False),
        ("RETURN null IN [] AS r", False),
        ("RETURN 1 IN [null] AS r", None),
        ("RETURN 2 IN [1, null, 2] AS r", True),
        ("RETURN 3 IN [1, null, 2] AS r", None),
        ("RETURN 'b' IN ['a', 'b'] AS r", True),
    ],
)
def test_three_valued_in_end_to_end(query: str, expected: Any) -> None:
    g, _ = _graph()
    assert _scalar(g, query) == expected


def test_where_drops_the_null_verdicts_and_keeps_the_true_ones() -> None:
    nodes = pd.DataFrame({"id": [0, 1, 2, 3], "v": pd.array([1, None, 3, 1], dtype="Int64")})
    g = graphistry.edges(pd.DataFrame({"s": [], "d": []}), "s", "d").nodes(nodes, "id")
    kept = g.gfql("MATCH (a) WHERE a.v IN [1, null] RETURN a.id AS id ORDER BY id")._nodes["id"].tolist()
    assert kept == [0, 3]
    kept_plain = g.gfql("MATCH (a) WHERE a.v IN [3] RETURN a.id AS id")._nodes["id"].tolist()
    assert kept_plain == [2]
