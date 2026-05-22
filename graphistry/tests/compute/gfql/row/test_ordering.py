from __future__ import annotations

import pandas as pd
import pytest

from graphistry.compute.ast import limit, order_by, rows, select
from graphistry.compute.exceptions import GFQLTypeError
from graphistry.tests.test_compute import CGFull


def _mk_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame | None = None):
    if edges_df is None:
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
    return CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")


def _self_loop_edges(nodes_df: pd.DataFrame) -> pd.DataFrame:
    if len(nodes_df) == 0:
        return pd.DataFrame({"s": [], "d": []})
    node_id = nodes_df["id"].iloc[0]
    return pd.DataFrame({"s": [node_id], "d": [node_id]})


def _run_node_steps(nodes_df: pd.DataFrame, steps: list[object], edges_df: pd.DataFrame | None = None):
    return _mk_graph(nodes_df, edges_df).gfql(steps)._nodes


def test_row_pipeline_order_by_supports_list_literal_and_subscript_expression_keys() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "list": [[2, -2], [1, 2], [300, 0], [1, -20], [2, -2, 100]],
            "list2": [[3, -2], [2, -2], [1, -2], [4, -2], [5, -2]],
        }
    )

    result = _run_node_steps(
        nodes_df,
        [
            rows(),
            order_by([("[list2[1], list2[0], list[1]] + list + list2", "asc")]),
            limit(3),
            select([("id", "id")]),
        ],
        edges_df=_self_loop_edges(nodes_df),
    )

    assert result.to_dict(orient="records") == [{"id": "c"}, {"id": "b"}, {"id": "a"}]


def test_row_pipeline_order_by_supports_stringified_list_subscript_expression_keys() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "list": pd.Series(
                ["[2, -2]", "[1, 2]", "[300, 0]", "[1, -20]", "[2, -2, 100]"],
                dtype="string",
            ),
            "list2": pd.Series(
                ["[3, -2]", "[2, -2]", "[1, -2]", "[4, -2]", "[5, -2]"],
                dtype="string",
            ),
        }
    )

    result = _run_node_steps(
        nodes_df,
        [
            rows(),
            order_by([("[list2[1], list2[0], list[1]] + list + list2", "asc")]),
            limit(3),
            select([("id", "id")]),
        ],
        edges_df=_self_loop_edges(nodes_df),
    )

    assert result.to_dict(orient="records") == [{"id": "c"}, {"id": "b"}, {"id": "a"}]


def test_row_pipeline_order_by_list_column_matches_opencypher_prefix_tie_break() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "list": [[2, -2], [1, 2], [300, 0], [1, -20], [2, -2, 100]],
        }
    )

    asc = _run_node_steps(
        nodes_df,
        [rows(), order_by([("list", "asc")]), limit(3), select([("id", "id")])],
        edges_df=_self_loop_edges(nodes_df),
    )
    assert asc.to_dict(orient="records") == [{"id": "d"}, {"id": "b"}, {"id": "a"}]

    desc = _run_node_steps(
        nodes_df,
        [rows(), order_by([("list", "desc")]), limit(3), select([("id", "id")])],
        edges_df=_self_loop_edges(nodes_df),
    )
    assert desc.to_dict(orient="records") == [{"id": "c"}, {"id": "e"}, {"id": "a"}]


def test_row_pipeline_order_by_stringified_list_column_uses_list_orderability() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "list": pd.Series(["[2, -2]", "[1, 2]", "[300, 0]", "[1, -20]", "[2, -2, 100]"], dtype="string"),
        }
    )

    asc = _run_node_steps(
        nodes_df,
        [rows(), order_by([("list", "asc")]), limit(3), select([("list", "list")])],
        edges_df=_self_loop_edges(nodes_df),
    )
    asc_values = sorted(str(v) for v in asc["list"].tolist())
    assert asc_values == sorted(["[1, -20]", "[1, 2]", "[2, -2]"])

    desc = _run_node_steps(
        nodes_df,
        [rows(), order_by([("list", "desc")]), limit(3), select([("list", "list")])],
        edges_df=_self_loop_edges(nodes_df),
    )
    desc_values = sorted(str(v) for v in desc["list"].tolist())
    assert desc_values == sorted(["[300, 0]", "[2, -2, 100]", "[2, -2]"])


def test_row_pipeline_order_by_partial_stringified_list_raises_mixed_family() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "list": pd.Series(["[1, -20]", "hello", "[2, 0]"], dtype="string"),
        }
    )

    with pytest.raises((GFQLTypeError, ValueError)) as exc:
        _run_node_steps(
            nodes_df,
            [rows(), order_by([("list", "asc")]), select([("id", "id"), ("list", "list")])],
            edges_df=_self_loop_edges(nodes_df),
        )
    assert "list_string" in str(exc.value) and "str" in str(exc.value)


def test_row_pipeline_order_by_stringified_list_with_nulls_returns_top_k_without_error() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "list": pd.Series(["[1, 2]", None, "[3, 4]", "[1, 1]"], dtype="object"),
        }
    )

    result = _run_node_steps(
        nodes_df,
        [rows(), order_by([("list", "asc")]), limit(3), select([("list", "list")])],
        edges_df=_self_loop_edges(nodes_df),
    )
    values = sorted(str(v) for v in result["list"].tolist())
    assert "[1, 1]" in values and "[1, 2]" in values


def test_row_pipeline_order_by_malformed_stringified_list_falls_back_to_lex_sort() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "list": pd.Series(["[1, 2", "abc"], dtype="string"),
        }
    )

    result = _run_node_steps(
        nodes_df,
        [rows(), order_by([("list", "asc")]), select([("list", "list")])],
        edges_df=_self_loop_edges(nodes_df),
    )
    values = [str(v) for v in result["list"].tolist()]
    assert values == ["[1, 2", "abc"]


def test_row_pipeline_order_by_multi_key_stringified_list_with_scalar() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "list": pd.Series(["[1, 2]", "[1, 2]", "[3, 4]", "[3, 4]"], dtype="string"),
            "num": [10, 20, 5, 15],
        }
    )

    result = _run_node_steps(
        nodes_df,
        [rows(), order_by([("list", "asc"), ("num", "desc")]), select([("list", "list"), ("num", "num")])],
        edges_df=_self_loop_edges(nodes_df),
    )
    assert result["num"].tolist() == [20, 10, 15, 5]
    leaked = [c for c in result.columns if "__gfql_sort_listparsed" in str(c)]
    assert leaked == [], f"aux columns leaked: {leaked}"
