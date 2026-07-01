from __future__ import annotations

import pandas as pd
import pytest

from graphistry.compute.ast import limit, order_by, rows, select
from graphistry.compute.exceptions import GFQLTypeError
from graphistry.compute.gfql.row.ordering import order_detect_temporal_mode
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


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1, 2, 3], dtype="int64"),
        pd.Series([1, 2, 3], dtype="uint32"),
        pd.Series([1.5, 2.5], dtype="float64"),
        pd.Series([True, False], dtype="bool"),
    ],
    ids=["int64", "uint32", "float64", "bool"],
)
def test_order_detect_temporal_mode_skips_non_text_dtypes(series: pd.Series) -> None:
    # Numeric/bool columns can never hold temporal *text*; the detector must
    # short-circuit without the astype(str) + multi-regex scan (issue #1650).
    assert order_detect_temporal_mode(series) is None


def test_order_detect_temporal_mode_still_detects_text_temporals() -> None:
    # Gate must not regress detection on object/string columns.
    assert order_detect_temporal_mode(pd.Series(["2020-01-01", "2020-02-02"], dtype="object")) == "date"
    assert order_detect_temporal_mode(pd.Series(["abc", "def"], dtype="object")) is None


def test_where_rows_numeric_filter_returns_correct_rows() -> None:
    # End-to-end: a numeric where_rows comparison still filters correctly with the
    # temporal-detection gate in place.
    nodes_df = pd.DataFrame({"id": [0, 1, 2, 3], "val": [10, 60, 51, 99]})
    edges_df = pd.DataFrame({"s": [0, 1], "d": [2, 3]})
    from graphistry.compute.ast import where_rows

    out = _mk_graph(nodes_df, edges_df).gfql([rows(), where_rows(expr="val > 50")])._nodes
    assert sorted(out["val"].tolist()) == [51, 60, 99]


@pytest.mark.skipif("TEST_CUDF" not in __import__("os").environ, reason="cuDF lane: set TEST_CUDF=1 (e.g. dgx-spark)")
def test_where_rows_numeric_filter_returns_correct_rows_cudf() -> None:
    # cuDF lane for the numeric dtype-gate end-to-end path: the gate is pure
    # dtype.kind inspection (engine-agnostic), but compute/gfql/row needs paired
    # cuDF coverage per the review conventions.
    cudf = pytest.importorskip("cudf")
    from graphistry.compute.ast import where_rows
    nodes_df = cudf.DataFrame({"id": [0, 1, 2, 3], "val": [10, 60, 51, 99]})
    edges_df = cudf.DataFrame({"s": [0, 1], "d": [2, 3]})
    out = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d").gfql(
        [rows(), where_rows(expr="val > 50")])._nodes
    assert sorted(out["val"].to_pandas().tolist()) == [51, 60, 99]


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1, 2, 3], dtype="int64"),
        pd.Series([1, 2, 3], dtype="uint32"),
        pd.Series([1.0, 2.0], dtype="float64"),
        pd.Series([True, False], dtype="bool"),
    ],
    ids=["int64", "uint32", "float64", "bool"],
)
def test_gfql_series_is_list_like_skips_non_listable_dtypes(series: pd.Series) -> None:
    # pipeline.py sibling of the ordering gate: numeric/bool columns can never be
    # list-like, so the detector short-circuits before the astype(str)+regex scan.
    from graphistry.compute.gfql.row.pipeline import RowPipelineMixin
    assert RowPipelineMixin._gfql_series_is_list_like(series) is False


def test_gfql_series_is_list_like_still_detects_real_lists() -> None:
    # Gate must not regress detection of real list/tuple-valued object columns.
    # (Stringified lists like "[1, 2]" are intentionally NOT list-like here — that
    # case is handled by order_detect_stringified_list_series, not this helper.)
    from graphistry.compute.gfql.row.pipeline import RowPipelineMixin
    assert RowPipelineMixin._gfql_series_is_list_like(pd.Series([[1, 2], [3, 4]], dtype="object")) is True
    assert RowPipelineMixin._gfql_series_is_list_like(pd.Series(["[1, 2]", "[3, 4]"], dtype="object")) is False
    assert RowPipelineMixin._gfql_series_is_list_like(pd.Series(["abc", "def"], dtype="object")) is False


@pytest.mark.parametrize(
    "dtype_str,expected",
    [
        ("int64", True), ("uint32", True), ("int8", True), ("float32", True),
        ("float64", True), ("bool", True), ("complex128", True),
        ("object", False), ("datetime64[ns]", False), ("timedelta64[ns]", False),
        ("<U5", False),
    ],
)
def test_is_non_textual_scalar_dtype(dtype_str: str, expected: bool) -> None:
    # Single source of truth for the #1650/#1651 dtype gate (shared by ordering,
    # pipeline, and cypher result post-processing).
    import numpy as np
    from graphistry.compute.gfql.series_str_compat import is_non_textual_scalar_dtype
    assert is_non_textual_scalar_dtype(np.dtype(dtype_str)) is expected


def test_is_non_textual_scalar_dtype_none() -> None:
    from graphistry.compute.gfql.series_str_compat import is_non_textual_scalar_dtype
    assert is_non_textual_scalar_dtype(None) is False


def test_dtype_gate_kind_set_matches_filter_by_dict_mirror() -> None:
    """Drift guard: `filter_by_dict._is_numeric_dtype_safe` keeps a SEPARATE copy of
    the dtype-kind set that `series_str_compat._NON_TEXTUAL_SCALAR_KINDS` is the SSOT
    for (a cross-layer import would risk an import cycle, so the copy stays — but it
    MUST NOT drift). Assert the two agree across every numpy dtype kind, so a future
    edit to one set fails here instead of silently desyncing the gate."""
    import numpy as np
    from graphistry.compute.gfql.series_str_compat import is_non_textual_scalar_dtype
    from graphistry.compute.filter_by_dict import _is_numeric_dtype_safe
    # One representative dtype per numpy kind, incl. text/temporal/extension kinds.
    samples = [
        np.dtype("int64"), np.dtype("uint8"), np.dtype("float64"), np.dtype("bool"),
        np.dtype("complex128"), np.dtype("object"), np.dtype("datetime64[ns]"),
        np.dtype("timedelta64[ns]"), np.dtype("<U5"), np.dtype("S3"),
    ]
    for dt in samples:
        assert is_non_textual_scalar_dtype(dt) == _is_numeric_dtype_safe(dt), \
            f"dtype-gate kind set drift for {dt!r}: SSOT and filter_by_dict mirror disagree"
