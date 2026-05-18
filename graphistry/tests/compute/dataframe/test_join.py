import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute.dataframe.join import (
    binding_join_columns,
    connected_inner_join_rows,
    ineq_eval_pairs,
    joined_alias_columns,
    joined_hidden_scalar_columns,
    project_node_attrs,
    semijoin_eval_pairs,
)


def _records(df):
    if hasattr(df, "to_pandas"):
        return df.to_pandas().to_dict(orient="records")
    return df.to_dict(orient="records")


def test_binding_join_columns_selects_dotted_columns_only() -> None:
    frame = pd.DataFrame(
        {
            "a.id": ["n1"],
            "a.label": ["A"],
            "plain": [1],
            5: [9],
        }
    )
    assert binding_join_columns(frame) == ["a.id", "a.label"]


def test_joined_hidden_scalar_columns_coalesces_suffix_variants() -> None:
    frame = pd.DataFrame(
        {
            "a.__cypher_reentry_property__": [None, 2],
            "b.__cypher_reentry_property__": [1, None],
            "a.id": ["a1", "a2"],
        }
    )
    out = joined_hidden_scalar_columns(frame)
    assert _records(out[["__cypher_reentry_property__"]]) == [
        {"__cypher_reentry_property__": 1.0},
        {"__cypher_reentry_property__": 2.0},
    ]


def test_joined_hidden_scalar_columns_preserves_existing_suffix_column() -> None:
    frame = pd.DataFrame(
        {
            "__cypher_reentry_property__": [7, 8],
            "a.__cypher_reentry_property__": [1, 2],
            "b.__cypher_reentry_property__": [3, 4],
        }
    )
    out = joined_hidden_scalar_columns(frame)
    assert _records(out[["__cypher_reentry_property__"]]) == [
        {"__cypher_reentry_property__": 7},
        {"__cypher_reentry_property__": 8},
    ]


def test_joined_alias_columns_recovers_alias_identity_columns() -> None:
    frame = pd.DataFrame(
        {
            "a.id": ["a1"],
            "b.b": ["b1"],
            "b.id": ["b1-id-fallback"],
        }
    )
    out = joined_alias_columns(frame)
    assert _records(out[["a", "b"]]) == [{"a": "a1", "b": "b1"}]


def test_joined_alias_columns_does_not_overwrite_existing_alias_column() -> None:
    frame = pd.DataFrame(
        {
            "a": ["already-present"],
            "a.id": ["fallback-id"],
            "a.a": ["alias-shape"],
        }
    )
    out = joined_alias_columns(frame)
    assert _records(out[["a"]]) == [{"a": "already-present"}]


def test_connected_inner_join_rows_pandas_path() -> None:
    joined_rows = pd.DataFrame(
        {
            "a.id": ["a1", "a2"],
            "a.num": [1, 2],
        }
    )
    pattern_rows = pd.DataFrame(
        {
            "a.id": ["a1", "a1", "a3"],
            "b.id": ["b1", "b2", "b3"],
        }
    )
    out = connected_inner_join_rows(
        joined_rows,
        pattern_rows,
        join_cols=["a.id"],
        keep_cols=["a.id", "b.id"],
        engine=Engine.PANDAS,
    )
    assert _records(out[["a.id", "b.id"]]) == [
        {"a.id": "a1", "b.id": "b1"},
        {"a.id": "a1", "b.id": "b2"},
    ]


def test_connected_inner_join_rows_cudf_path() -> None:
    cudf = pytest.importorskip("cudf")
    joined_rows = cudf.DataFrame(
        {
            "a.id": ["a1", "a2"],
            "a.num": [1, 2],
        }
    )
    pattern_rows = cudf.DataFrame(
        {
            "a.id": ["a1", "a1", "a3"],
            "b.id": ["b1", "b2", "b3"],
        }
    )
    out = connected_inner_join_rows(
        joined_rows,
        pattern_rows,
        join_cols=["a.id"],
        keep_cols=["a.id", "b.id"],
        engine=Engine.CUDF,
    )
    assert type(out).__module__.startswith("cudf")
    assert _records(out[["a.id", "b.id"]]) == [
        {"a.id": "a1", "b.id": "b1"},
        {"a.id": "a1", "b.id": "b2"},
    ]


def test_connected_inner_join_rows_empty_match_keeps_payload_schema() -> None:
    joined_rows = pd.DataFrame(
        {
            "a.id": ["a1", "a2"],
            "a.num": [1, 2],
        }
    )
    pattern_rows = pd.DataFrame(
        {
            "a.id": ["x1"],
            "b.id": ["b1"],
        }
    )
    out = connected_inner_join_rows(
        joined_rows,
        pattern_rows,
        join_cols=["a.id"],
        keep_cols=["a.id", "b.id"],
        engine=Engine.PANDAS,
    )
    assert list(out.columns) == ["a.id", "a.num", "b.id"]
    assert len(out) == 0


def test_project_node_attrs_filters_renames_dedupes_and_drops_nulls() -> None:
    frame = pd.DataFrame(
        {
            "id": ["n1", "n1", "n2", "n3"],
            "color": ["blue", "blue", None, "green"],
        }
    )
    out = project_node_attrs(
        frame,
        "id",
        ["id", "color"],
        id_label="__node_id__",
        labels=["node", "color_label"],
        node_domain=["n1", "n2"],
        dedupe=True,
        drop_nulls=True,
    )
    assert _records(out[["__node_id__", "node", "color_label"]]) == [
        {"__node_id__": "n1", "node": "n1", "color_label": "blue"}
    ]


def test_project_node_attrs_prefix_mode_without_labels() -> None:
    frame = pd.DataFrame({"id": ["n1"], "color": ["blue"]})
    out = project_node_attrs(
        frame,
        "id",
        ["id", "color"],
        id_label="node_id",
        prefix="L_",
    )
    assert _records(out[["node_id", "L_color"]]) == [{"node_id": "n1", "L_color": "blue"}]


def test_ineq_eval_pairs_filters_bounds_per_mid_group() -> None:
    left_pairs = pd.DataFrame(
        {"__mid__": ["m1", "m1", "m2"], "__left_val__": [1, 5, 2], "__left__": ["l1", "l2", "l3"]}
    )
    right_pairs = pd.DataFrame(
        {"__mid__": ["m1", "m2"], "__right_val__": [3, 1], "__right__": ["r1", "r2"]}
    )
    left_eval, right_eval = ineq_eval_pairs(
        left_pairs,
        right_pairs,
        "<",
        left_value="__left_val__",
        right_value="__right_val__",
    )
    assert _records(left_eval[["__mid__", "__left_val__", "__left__"]]) == [
        {"__mid__": "m1", "__left_val__": 1, "__left__": "l1"}
    ]
    assert _records(right_eval[["__mid__", "__right_val__", "__right__"]]) == [
        {"__mid__": "m1", "__right_val__": 3, "__right__": "r1"}
    ]


def test_ineq_eval_pairs_greater_than_branch() -> None:
    left_pairs = pd.DataFrame(
        {"__mid__": ["m1", "m1"], "__left_val__": [1, 5], "__left__": ["l1", "l2"]}
    )
    right_pairs = pd.DataFrame(
        {"__mid__": ["m1", "m1"], "__right_val__": [2, 4], "__right__": ["r1", "r2"]}
    )
    left_eval, right_eval = ineq_eval_pairs(
        left_pairs,
        right_pairs,
        ">",
        left_value="__left_val__",
        right_value="__right_val__",
    )
    assert _records(left_eval[["__left__"]]) == [{"__left__": "l2"}]
    assert _records(right_eval[["__right__"]]) == [{"__right__": "r1"}, {"__right__": "r2"}]


def test_semijoin_eval_pairs_eq_filters_to_shared_mid_values() -> None:
    left_pairs = pd.DataFrame(
        {"__mid__": ["m1", "m1", "m2"], "__left_val__": ["a", "b", "c"], "__left__": [1, 2, 3]}
    )
    right_pairs = pd.DataFrame(
        {"__mid__": ["m1", "m2"], "__right_val__": ["b", "d"], "__right__": [9, 10]}
    )
    left_eval, right_eval, mid_values = semijoin_eval_pairs(
        left_pairs,
        right_pairs,
        "==",
        left_value="__left_val__",
        right_value="__right_val__",
    )
    assert mid_values is not None
    assert _records(mid_values[["__mid__", "__value__"]]) == [{"__mid__": "m1", "__value__": "b"}]
    assert left_eval is not None
    assert right_eval is not None
    assert _records(left_eval[["__mid__", "__left_val__", "__left__"]]) == [
        {"__mid__": "m1", "__left_val__": "b", "__left__": 2}
    ]
    assert _records(right_eval[["__mid__", "__right_val__", "__right__"]]) == [
        {"__mid__": "m1", "__right_val__": "b", "__right__": 9}
    ]


def test_semijoin_eval_pairs_eq_returns_none_when_no_shared_values() -> None:
    left_pairs = pd.DataFrame({"__mid__": ["m1"], "__left_val__": ["a"]})
    right_pairs = pd.DataFrame({"__mid__": ["m1"], "__right_val__": ["b"]})
    left_eval, right_eval, mid_values = semijoin_eval_pairs(
        left_pairs,
        right_pairs,
        "==",
        left_value="__left_val__",
        right_value="__right_val__",
    )
    assert left_eval is None
    assert right_eval is None
    assert mid_values is not None
    assert len(mid_values) == 0


def test_semijoin_eval_pairs_neq_prunes_singleton_equal_values() -> None:
    left_pairs = pd.DataFrame(
        {"__mid__": ["m1", "m1", "m2"], "__left_val__": ["x", "y", "z"], "__left__": [1, 2, 3]}
    )
    right_pairs = pd.DataFrame(
        {"__mid__": ["m1", "m2"], "__right_val__": ["y", "z"], "__right__": [9, 10]}
    )
    left_eval, right_eval, _ = semijoin_eval_pairs(
        left_pairs,
        right_pairs,
        "!=",
        left_value="__left_val__",
        right_value="__right_val__",
    )
    assert left_eval is not None
    assert right_eval is not None
    assert _records(left_eval[["__mid__", "__left_val__", "__left__"]]) == [
        {"__mid__": "m1", "__left_val__": "x", "__left__": 1}
    ]
    assert _records(right_eval[["__mid__", "__right_val__", "__right__"]]) == [
        {"__mid__": "m1", "__right_val__": "y", "__right__": 9}
    ]


def test_semijoin_eval_pairs_delegates_inequality_ops() -> None:
    left_pairs = pd.DataFrame({"__mid__": ["m1", "m1"], "__left_val__": [1, 3], "__left__": ["l1", "l2"]})
    right_pairs = pd.DataFrame({"__mid__": ["m1"], "__right_val__": [2], "__right__": ["r1"]})
    left_eval, right_eval, mid_values = semijoin_eval_pairs(
        left_pairs,
        right_pairs,
        "<",
        left_value="__left_val__",
        right_value="__right_val__",
    )
    assert mid_values is None
    assert left_eval is not None
    assert right_eval is not None
    assert _records(left_eval[["__left__"]]) == [{"__left__": "l1"}]
    assert _records(right_eval[["__right__"]]) == [{"__right__": "r1"}]
