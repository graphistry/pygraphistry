from __future__ import annotations

import warnings

import pandas as pd
import pytest

import graphistry.compute.gfql.call.validation as call_safelist
import graphistry.compute.gfql.expr_parser as expr_parser
import graphistry.compute.gfql.row.pipeline as row_pipeline_mixin
from graphistry.compute.gfql.row.entity_props import entity_keys_series
from graphistry.compute.ast import (
    ASTCall,
    distinct,
    group_by,
    limit,
    n,
    order_by,
    return_,
    rows,
    select,
    skip,
    unwind,
    where_rows,
    with_,
)
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
from graphistry.compute.predicates.numeric import gt
from graphistry.compute.gfql.call.validation import validate_call_params
from graphistry.tests.test_compute import CGFull


def _mk_graph(nodes_df, edges_df=None):
    if edges_df is None:
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
    return CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")


def _normalize_expr_eval_output(value):
    if hasattr(value, "tolist"):
        out = []
        for item in value.tolist():
            if isinstance(item, (list, tuple, dict)):
                out.append(item)
                continue
            if pd.isna(item):
                out.append(None)
            else:
                out.append(item)
        return out
    if pd.isna(value):
        return None
    return value


def _normalize_record_value(value):
    if isinstance(value, list):
        return [_normalize_record_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_record_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_record_value(item) for key, item in value.items()}
    return _normalize_expr_eval_output(value)


def _normalize_records(records):
    return [{key: _normalize_record_value(value) for key, value in row.items()} for row in records]


def _self_loop_edges(nodes_df):
    if len(nodes_df) == 0:
        return pd.DataFrame({"s": [], "d": []})
    node_id = nodes_df["id"].iloc[0]
    return pd.DataFrame({"s": [node_id], "d": [node_id]})


def _run_node_steps(nodes_df, steps, edges_df=None):
    return _mk_graph(nodes_df, edges_df).gfql(steps)._nodes


def _assert_node_records(
    nodes_df,
    steps,
    expected_records,
    *,
    edges_df=None,
    expected_columns=None,
    normalize=False,
):
    result_nodes = _run_node_steps(nodes_df, steps, edges_df=edges_df)
    if expected_columns is not None:
        assert list(result_nodes.columns) == expected_columns
    records = result_nodes.reset_index(drop=True).to_dict(orient="records")
    if normalize:
        records = _normalize_records(records)
    assert records == expected_records
    return result_nodes


def _assert_single_row_select_records(items, expected_records, *, nodes_df=None):
    base_nodes = pd.DataFrame({"id": ["a"]}) if nodes_df is None else nodes_df
    return _assert_node_records(
        base_nodes,
        [rows(), select(items)],
        expected_records,
        edges_df=_self_loop_edges(base_nodes),
    )


def _assert_ordered_projection_values(nodes, column, asc_expected, desc_expected, *, limit_value=None):
    nodes_df = pd.DataFrame(nodes)
    g = _mk_graph(nodes_df)
    base_steps = [rows(), select([(column, column)])]
    limit_steps = [] if limit_value is None else [limit(limit_value)]

    asc_result = g.gfql(base_steps + [order_by([(column, "asc")])] + limit_steps)
    assert asc_result._nodes[column].tolist() == asc_expected

    desc_result = g.gfql(base_steps + [order_by([(column, "desc")])] + limit_steps)
    assert desc_result._nodes[column].tolist() == desc_expected


def _parse_identifier_score(_expr):
    return expr_parser.Identifier("score")


def _parse_score_gt_one(_expr):
    return expr_parser.BinaryOp(">", expr_parser.Identifier("score"), expr_parser.Literal(1))


def _parse_unknown_fn_score(_expr):
    return expr_parser.FunctionCall("unknown_fn", (expr_parser.Identifier("score"),))


def _parse_name_minus_x(_expr):
    return expr_parser.BinaryOp("-", expr_parser.Identifier("name"), expr_parser.Literal("x"))


def _raise_bad_parse(_expr):
    raise ValueError("bad parse")


def _capabilities_ok(_node):
    return []


def _capabilities_unsupported(_node):
    return ["unsupported"]


def _collect_score(_node):
    return {"score"}


def _collect_nested_score_vals(_node):
    return {"n.age", "score", "vals"}


def _where_rows_parser_bundle(parse=_parse_identifier_score, capabilities=_capabilities_ok, collect=_collect_score):
    return parse, capabilities, collect


def _runtime_parser_bundle(parse, capabilities=_capabilities_ok):
    return parse, capabilities, expr_parser


def _assert_ast_parity(nodes, cases):
    g = _mk_graph(pd.DataFrame(nodes))
    ctx = row_pipeline_mixin._RowPipelineAdapter(g)
    table_df = g._nodes

    for ast_node, expr in cases:
        ok, ast_out = ctx._gfql_eval_expr_ast(table_df, ast_node)
        assert ok, expr
        legacy_out = ctx._gfql_eval_string_expr(table_df, expr)
        assert _normalize_expr_eval_output(ast_out) == _normalize_expr_eval_output(legacy_out)


class TestRowPipelineASTPrimitives:
    @pytest.mark.parametrize(
        ("step", "function", "params"),
        [
            pytest.param(rows("nodes", source="a"), "rows", {"table": "nodes", "source": "a"}, id="rows"),
            pytest.param(
                select([("name", "name"), ("age", "age")]),
                "select",
                {"items": [("name", "name"), ("age", "age")]},
                id="select",
            ),
            pytest.param(with_([("name", "name")]), "with_", {"items": [("name", "name")]}, id="with_"),
            pytest.param(return_([("name", "name")]), "select", {"items": [("name", "name")]}, id="return_"),
            pytest.param(
                where_rows({"name": "alice"}),
                "where_rows",
                {"filter_dict": {"name": "alice"}},
                id="where-dict",
            ),
            pytest.param(where_rows(expr="score > 1"), "where_rows", {"expr": "score > 1"}, id="where-expr"),
            pytest.param(
                order_by([("name", "asc"), ("age", "desc")]),
                "order_by",
                {"keys": [("name", "asc"), ("age", "desc")]},
                id="order_by",
            ),
            pytest.param(skip(3), "skip", {"value": 3}, id="skip"),
            pytest.param(limit(10), "limit", {"value": 10}, id="limit"),
            pytest.param(distinct(), "distinct", {}, id="distinct"),
            pytest.param(unwind("vals", as_="v"), "unwind", {"expr": "vals", "as_": "v"}, id="unwind"),
            pytest.param(
                group_by(["grp"], [("cnt", "count"), ("sum_score", "sum", "score")]),
                "group_by",
                {
                    "keys": ["grp"],
                    "aggregations": [("cnt", "count"), ("sum_score", "sum", "score")],
                },
                id="group_by",
            ),
        ],
    )
    def test_row_pipeline_primitives_build_ast_calls(self, step, function, params):
        assert isinstance(step, ASTCall)
        assert step.function == function
        assert step.params == params


def test_row_pipeline_select_supports_range_scalar_function() -> None:
    nodes_df = pd.DataFrame({"id": ["a"]})

    result = _run_node_steps(nodes_df, [rows(), select([("vals", "range(0, 3)")])], edges_df=_self_loop_edges(nodes_df))

    assert _normalize_records(result.to_dict(orient="records")) == [{"vals": [0, 1, 2, 3]}]


def test_row_pipeline_select_supports_range_with_constant_series_bounds() -> None:
    nodes_df = pd.DataFrame({"id": ["a"], "num_of_values": [3]})

    result = _run_node_steps(
        nodes_df,
        [
            rows(),
            select([("ordered_x", "[0, 1, 2]"), ("num_of_values", "num_of_values")]),
            select([("equal", "ordered_x = range(0, num_of_values - 1)")]),
        ],
        edges_df=_self_loop_edges(nodes_df),
    )

    assert _normalize_records(result.to_dict(orient="records")) == [{"equal": True}]


def test_row_pipeline_select_supports_range_with_varying_row_bounds_and_steps() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "start": [0, 0, 10, 2],
            "stop": [3, -1, 4, 2],
            "step": [1, -1, -3, 5],
        }
    )

    result = _run_node_steps(
        nodes_df,
        [rows(), select([("id", "id"), ("vals", "range(start, stop, step)")]), order_by([("id", "asc")])],
        edges_df=_self_loop_edges(nodes_df),
    )

    assert _normalize_records(result.to_dict(orient="records")) == [
        {"id": "a", "vals": [0, 1, 2, 3]},
        {"id": "b", "vals": [0, -1]},
        {"id": "c", "vals": [10, 7, 4]},
        {"id": "d", "vals": [2]},
    ]


@pytest.mark.parametrize(
    ("expr", "pattern"),
    [
        ("range(2, 8, 0)", "range\\(\\) step must be non-zero"),
        ("range(true, 1, 1)", "range\\(\\) start must be an integer"),
        ("range(0, 1.0, 1)", "range\\(\\) stop must be an integer"),
    ],
)
def test_row_pipeline_select_rejects_invalid_range_arguments(expr: str, pattern: str) -> None:
    nodes_df = pd.DataFrame({"id": ["a"]})

    with pytest.raises(GFQLTypeError, match=pattern):
        _run_node_steps(nodes_df, [rows(), select([("vals", expr)])], edges_df=_self_loop_edges(nodes_df))


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


def test_row_pipeline_order_by_supports_temporal_duration_expression_keys() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "time": [
                "time({hour: 10, minute: 35, timezone: '-08:00'})",
                "time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})",
                "time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})",
                "time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})",
                "time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})",
            ],
        }
    )

    result = _run_node_steps(
        nodes_df,
        [
            rows(),
            order_by([("time + 'PT6M'", "asc")]),
            limit(3),
            select([("id", "id")]),
        ],
        edges_df=_self_loop_edges(nodes_df),
    )

    assert result.to_dict(orient="records") == [{"id": "d"}, {"id": "e"}, {"id": "b"}]


def test_row_pipeline_order_by_supports_date_duration_expression_keys() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "date": [
                "date({year: 1910, month: 5, day: 6})",
                "date({year: 1980, month: 12, day: 24})",
                "date({year: 1984, month: 10, day: 12})",
                "date({year: 1985, month: 5, day: 6})",
                "date({year: 1980, month: 10, day: 24})",
                "date({year: 1984, month: 10, day: 11})",
            ],
        }
    )

    result = _run_node_steps(
        nodes_df,
        [
            rows(),
            order_by([("date + 'P1M2D'", "asc")]),
            limit(2),
            select([("id", "id")]),
        ],
        edges_df=_self_loop_edges(nodes_df),
    )

    assert result.to_dict(orient="records") == [{"id": "a"}, {"id": "e"}]


def test_row_pipeline_select_supports_keys_for_map_literals_and_nulls() -> None:
    nodes_df = pd.DataFrame({"id": ["a"]})

    result = _run_node_steps(
        nodes_df,
        [rows(), select([("ks", "keys({k: 1, l: null})"), ("null_keys", "keys(null)")])],
        edges_df=_self_loop_edges(nodes_df),
    )

    assert _normalize_records(result.to_dict(orient="records")) == [{"ks": ["k", "l"], "null_keys": None}]


def test_row_pipeline_order_by_falls_back_to_string_sort_for_mixed_date_text_beyond_sample_window() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(129)],
            "date": [f"1984-10-{(i % 28) + 1:02d}" for i in range(128)] + ["not-a-date"],
        }
    )

    result = _run_node_steps(
        nodes_df,
        [rows(), order_by([("date", "asc")]), select([("date", "date")])],
        edges_df=_self_loop_edges(nodes_df),
    )

    assert result["date"].iloc[0] == "1984-10-01"
    assert result["date"].iloc[-1] == "not-a-date"


def test_row_pipeline_order_by_rejects_mixed_list_values_beyond_sample_window() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(129)],
            "vals": [[1]] * 128 + [1],
        },
        dtype="object",
    )

    with pytest.raises(Exception, match="unsupported order_by expression for vectorized execution"):
        _run_node_steps(
            nodes_df,
            [rows(), order_by([("vals", "asc")]), select([("vals", "vals")])],
            edges_df=_self_loop_edges(nodes_df),
        )


@pytest.mark.parametrize(
    ("values", "families"),
    [
        pytest.param([1] * 128 + [{}], "number, unsupported", id="number-unsupported"),
        pytest.param([True] * 128 + ["x"], "bool, str", id="bool-str"),
        pytest.param([1.5] * 128 + ["x"], "number, str", id="number-str"),
    ],
)
def test_row_pipeline_order_by_rejects_mixed_scalar_families_beyond_sample_window(
    values: list[object], families: str
) -> None:
    nodes_df = pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(len(values))],
            "v": values,
        },
        dtype="object",
    )

    with pytest.raises(Exception, match=rf"mixed/dynamic value families \({families}\)"):
        _run_node_steps(
            nodes_df,
            [rows(), order_by([("v", "asc")]), select([("v", "v")])],
            edges_df=_self_loop_edges(nodes_df),
        )


def test_entity_keys_series_supports_mixed_entity_property_sets() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "name": ["Alice", None],
            "score": [None, None],
        }
    )

    result = entity_keys_series(
        nodes_df,
        alias_col="id",
        table="nodes",
        excluded=(),
    )

    assert _normalize_expr_eval_output(result) == [["name"], []]


def test_row_pipeline_dynamic_subscript_uses_full_series_constant_check() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(129)],
            "vals": [[10, 20, 30]] * 129,
            "idx": [1] * 128 + [2],
        },
        dtype="object",
    )

    result = _run_node_steps(
        nodes_df,
        [rows(), select([("x", "vals[idx]")])],
        edges_df=_self_loop_edges(nodes_df),
    )

    assert result["x"].iloc[0] == 20
    assert result["x"].iloc[-1] == 30


def test_row_pipeline_dynamic_subscript_rejects_mixed_list_scalar_beyond_sample_window() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(129)],
            "vals": [[10, 20, 30]] * 128 + [1],
            "idx": [1] * 129,
        },
        dtype="object",
    )

    with pytest.raises(Exception, match="dynamic subscript requires list-like base"):
        _run_node_steps(
            nodes_df,
            [rows(), select([("x", "vals[idx]")])],
            edges_df=_self_loop_edges(nodes_df),
        )


def test_row_pipeline_dynamic_subscript_rejects_mixed_integer_key_beyond_sample_window() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(129)],
            "vals": [[10, 20, 30]] * 129,
            "idx": [1] * 128 + ["bad"],
        },
        dtype="object",
    )

    with pytest.raises(Exception, match="dynamic subscript keys must be integer typed"):
        _run_node_steps(
            nodes_df,
            [rows(), select([("x", "vals[idx]")])],
            edges_df=_self_loop_edges(nodes_df),
        )


def test_row_pipeline_property_access_rejects_mixed_map_scalar_beyond_sample_window() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(129)],
            "m": [{"a": 1}] * 128 + [1],
        },
        dtype="object",
    )

    with pytest.raises(Exception, match="property access requires a graph element alias, entity value, or map"):
        _run_node_steps(
            nodes_df,
            [rows(), select([("x", "m.a")])],
            edges_df=_self_loop_edges(nodes_df),
        )


def test_row_pipeline_labels_rejects_mixed_entity_scalar_beyond_sample_window() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(129)],
            "e": ["()"] * 128 + [1],
        },
        dtype="object",
    )

    with pytest.raises(Exception, match="labels\\(\\) requires a graph element, entity value, or null"):
        _run_node_steps(
            nodes_df,
            [rows(), select([("x", "labels(e)")])],
            edges_df=_self_loop_edges(nodes_df),
        )


def test_row_pipeline_range_rejects_mixed_integer_arg_beyond_sample_window() -> None:
    nodes_df = pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(129)],
            "start": [0] * 128 + ["bad"],
            "stop": [2] * 129,
            "step": [1] * 129,
        },
        dtype="object",
    )

    with pytest.raises(Exception, match="range\\(\\) start must be an integer"):
        _run_node_steps(
            nodes_df,
            [rows(), select([("vals", "range(start, stop, step)")])],
            edges_df=_self_loop_edges(nodes_df),
        )


def test_row_pipeline_select_supports_properties_for_map_literals_and_nulls() -> None:
    nodes_df = pd.DataFrame({"id": ["a"]})

    result = _run_node_steps(
        nodes_df,
        [rows(), select([("m", "properties({name: 'Popeye', level: 9001})"), ("null_props", "properties(null)")])],
        edges_df=_self_loop_edges(nodes_df),
    )

    assert _normalize_records(result.to_dict(orient="records")) == [
        {"m": "{name: 'Popeye', level: 9001}", "null_props": None}
    ]

    @pytest.mark.parametrize(
        ("step", "function", "params"),
        [
            pytest.param(select(["name", "age"]), "select", {"items": [("name", "name"), ("age", "age")]}, id="select"),
            pytest.param(with_(["name"]), "with_", {"items": [("name", "name")]}, id="with_"),
            pytest.param(return_(["name"]), "select", {"items": [("name", "name")]}, id="return_"),
        ],
    )
    def test_row_pipeline_projection_shorthand_builds_identity_pairs(self, step, function, params):
        assert isinstance(step, ASTCall)
        assert step.function == function
        assert step.params == params


class TestRowPipelineExecution:
    @staticmethod
    def _where_expr_ids(nodes_df, expr):
        result = _mk_graph(nodes_df).gfql(
            [rows(), where_rows(expr=expr), order_by([("id", "asc")]), return_([("id", "id")])]
        )
        return result._nodes["id"].tolist()

    def test_row_pipeline_exec_projection_sort_page_distinct(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c", "d"],
            "name": ["n2", "n1", "n2", "n3"],
            "score": [2, 3, 2, 1],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([("name", "name"), ("score", "score")]),
            distinct(),
            order_by([("score", "desc"), ("name", "asc")]),
            skip(1),
            limit(2),
        ])

        assert list(result._nodes.columns) == ["name", "score"]
        assert result._nodes.reset_index(drop=True).to_dict(orient="records") == [
            {"name": "n2", "score": 2},
            {"name": "n3", "score": 1},
        ]
        assert result._edges is not None
        assert len(result._edges) == 0

    def test_row_pipeline_exec_projection_shorthand_and_mixed(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "name": ["n1", "n2", "n3"],
            "score": [1, 2, 3],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select(["id", ("score_twice", "score * 2")]),
            order_by([("id", "asc")]),
        ])
        assert list(result._nodes.columns) == ["id", "score_twice"]
        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "score_twice": 2},
            {"id": "b", "score_twice": 4},
            {"id": "c", "score_twice": 6},
        ]

        ret = g.gfql([rows(), where_rows(expr="score >= 2"), return_(["id", "score"])])
        assert list(ret._nodes.columns) == ["id", "score"]
        assert ret._nodes.sort_values("id").to_dict(orient="records") == [
            {"id": "b", "score": 2},
            {"id": "c", "score": 3},
        ]

    @pytest.mark.parametrize(
        ("nodes", "steps", "expected_records", "expected_columns", "edges"),
        [
            pytest.param(
                {"id": ["a", "b", "c"], "vals": [[1, 2], [1, 2], [3]], "meta": [{"k": "v"}, {"k": "v"}, {"k": "z"}]},
                [rows(), select([("vals", "vals"), ("meta", "meta")]), distinct()],
                [{"vals": [1, 2], "meta": {"k": "v"}}, {"vals": [3], "meta": {"k": "z"}}],
                None,
                None,
                id="distinct-unhashable-cells",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "vals": [[1, 2], [1, 2], [3]]},
                [rows(), select([("vals", "vals")]), order_by([("vals", "asc")])],
                [{"vals": [1, 2]}, {"vals": [1, 2]}, {"vals": [3]}],
                ["vals"],
                None,
                id="order-by-list-values",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "grp": ["x", "x", "y"]},
                [n({"grp": "x"}, name="a"), rows(source="a"), return_([("id", "id")]), order_by([("id", "asc")])],
                [{"id": "a"}, {"id": "b"}],
                ["id"],
                {"s": ["a", "b"], "d": ["b", "c"]},
                id="match-alias-source",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "score": [1, 3, 2]},
                [rows(), where_rows({"score": gt(1)}), order_by([("id", "asc")]), return_([("id", "id"), ("score", "score")])],
                [{"id": "b", "score": 3}, {"id": "c", "score": 2}],
                None,
                None,
                id="where-rows-dict",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "vals": [[1], [1, 2], [1, 2, 3]], "name": ["n1", "n2", "n3"]},
                [rows(), where_rows(expr="size(vals) > 1 AND NOT (name = 'n3')"), order_by([("id", "asc")]), return_([("id", "id"), ("vals", "vals"), ("name", "name")])],
                [{"id": "b", "vals": [1, 2], "name": "n2"}],
                None,
                None,
                id="where-rows-expr",
            ),
            pytest.param(
                {"id": ["a", "b"], "name": ["rand()", "plain"]},
                [rows(), where_rows(expr="name = 'rand()'"), return_([("id", "id"), ("name", "name")])],
                [{"id": "a", "name": "rand()"}],
                None,
                None,
                id="where-rows-quoted-function-text",
            ),
            pytest.param(
                {"id": ["a", "b"], "name": ["n1", "n2"]},
                [rows(), where_rows(expr="size(['x)', 'y']) = 2 AND NOT (name = 'n2')"), return_([("id", "id")])],
                [{"id": "a"}],
                None,
                None,
                id="where-rows-size-literal-paren",
            ),
            pytest.param(
                {"id": ["a", "b"], "vals": [[1, 2], [3]]},
                [rows(), unwind("vals", as_="v"), select([("id", "id"), ("v", "v")]), order_by([("v", "asc")])],
                [{"id": "a", "v": 1}, {"id": "a", "v": 2}, {"id": "b", "v": 3}],
                None,
                None,
                id="unwind-column",
            ),
            pytest.param(
                {"id": ["a", "b"], "vals": [[], [3]]},
                [rows(), unwind("vals", as_="v"), select([("id", "id"), ("v", "v")]), order_by([("v", "asc")])],
                [{"id": "b", "v": 3}],
                None,
                None,
                id="unwind-column-empty-list-drops-row",
            ),
            pytest.param(
                {"id": ["a", "b"], "first": [[1, 2], [3]], "second": [[4], [5, 6]]},
                [rows(), unwind("first + second", as_="x"), select([("x", "x")]), order_by([("x", "asc")])],
                [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}, {"x": 6}],
                None,
                None,
                id="unwind-string-expr",
            ),
            pytest.param(
                {"id": ["a", "b"], "qrows": [[[2], [3, 4]], [[5], [6, 7]]]},
                [rows(), unwind("qrows[0]", as_="q"), select([("q", "q")]), order_by([("q", "asc")])],
                [{"q": 2}, {"q": 5}],
                None,
                None,
                id="unwind-subscript-expr",
            ),
            pytest.param(
                {"id": ["seed"]},
                [
                    rows(),
                    with_([("prows", [0, 1]), ("qrows", [[2], [3, 4]])]),
                    unwind("prows", as_="p"),
                    unwind("qrows[p]", as_="q"),
                    select([("p", "p"), ("q", "q")]),
                    order_by([("p", "asc"), ("q", "asc")]),
                ],
                [{"p": 0, "q": 2}, {"p": 1, "q": 3}, {"p": 1, "q": 4}],
                None,
                {"s": [], "d": []},
                id="unwind-dynamic-subscript-expr",
            ),
        ],
    )
    def test_row_pipeline_exact_record_cases_vectorized(
        self, nodes, steps, expected_records, expected_columns, edges
    ):
        edges_df = None if edges is None else pd.DataFrame(edges)
        _assert_node_records(
            pd.DataFrame(nodes),
            steps,
            expected_records,
            edges_df=edges_df,
            expected_columns=expected_columns,
        )

    @pytest.mark.parametrize(
        ("nodes", "expr", "expected_ids"),
        [
            pytest.param(
                {"id": ["a", "b", "c"], "name": ["a AND b", "z", "a OR b"], "size": [1, 2, 3]},
                "name = 'a AND b' OR (name = 'a OR b' AND size > 2)",
                ["a", "c"],
                id="keyword-in-string",
            ),
            pytest.param({"id": ["a", "b"], "size": [1, 3]}, "size > 1", ["b"], id="column-named-size"),
            pytest.param(
                {"id": ["a", "b"], "mp": [{"k": "v"}, {"k": "z"}]},
                "mp = {k: 'v'}",
                ["a"],
                id="map-literal",
            ),
            pytest.param(
                {"id": ["a", "b"], "txt": ["x5", "yy"]},
                "txt CONTAINS 5",
                ["a"],
                id="numeric-string-rhs",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "txt": ["aa", "bb", "cc"]},
                "txt CONTAINS 'a' OR txt CONTAINS 'b'",
                ["a", "b"],
                id="string-boolean-composition",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "txt": ["aa", "bb", "cc"]},
                "NOT txt CONTAINS 'a'",
                ["b", "c"],
                id="not-string-predicate",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "txt": ["aa", "bb", "bb"]},
                "id = 'a' OR id = 'b' AND txt CONTAINS 'b'",
                ["a", "b"],
                id="and-or-precedence",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "score": [1, 3, 2]},
                "CASE WHEN score > 2 THEN true ELSE false END",
                ["b"],
                id="case-when",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "vals": [[1, 2], [2, 3], [1]]},
                "size([x IN vals WHERE x > 1]) > 0",
                ["a", "b"],
                id="list-comprehension-bound-var",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "vals": [[1, 2], [2, 3], [1]]},
                "size([x IN vals WHERE x > 1 | x + 1]) > 0",
                ["a", "b"],
                id="list-comprehension-projection",
            ),
            pytest.param(
                {"id": ["a", "b", "c", "d"], "x": [1, 2, None, 4], "lst": [[1, 2], [2, 3], [], None]},
                "NOT (x IN lst) OR x IS NULL",
                ["c"],
                id="is-null-or-precedence",
            ),
        ],
    )
    def test_row_pipeline_where_rows_expr_vectorized_cases(self, nodes, expr, expected_ids):
        assert self._where_expr_ids(pd.DataFrame(nodes), expr) == expected_ids

    def test_row_pipeline_where_rows_expr_string_predicate_parenthesized_scalar_rhs(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "txt": ["abc5", "5abc", "none"]})

        cases = [
            ("txt CONTAINS (5)", ["a", "b"]),
            ("txt STARTS WITH (5)", ["b"]),
            ("txt ENDS WITH (5)", ["a"]),
        ]
        for expr, expected_ids in cases:
            assert self._where_expr_ids(nodes_df, expr) == expected_ids

    def test_row_pipeline_where_rows_expr_case_boolean_composition(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c", "d"], "score": [1, 2, 3, 4]})
        cases = [
            (
                "CASE WHEN score > 1 THEN true ELSE false END "
                "AND CASE WHEN score > 2 THEN true ELSE false END",
                ["c", "d"],
            ),
            (
                "CASE WHEN score > 1 THEN true ELSE false END "
                "OR CASE WHEN score > 2 THEN true ELSE false END",
                ["b", "c", "d"],
            ),
        ]
        for expr, expected_ids in cases:
            assert self._where_expr_ids(nodes_df, expr) == expected_ids

    def test_row_pipeline_where_rows_expr_xor_null_semantics(self):
        nodes_df = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e"],
                "lhs": [True, True, False, False, None],
                "rhs": [True, False, False, None, True],
            }
        )

        assert self._where_expr_ids(nodes_df, "lhs XOR rhs") == ["b"]

        result = _mk_graph(nodes_df).gfql(
            [
                rows(),
                select([("id", "id"), ("flag", "lhs XOR rhs")]),
                order_by([("id", "asc")]),
            ]
        )

        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "flag": False},
            {"id": "b", "flag": True},
            {"id": "c", "flag": False},
            {"id": "d", "flag": None},
            {"id": "e", "flag": None},
        ]

    def test_row_pipeline_where_rows_expr_quantifiers(self):
        nodes_df = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "vals": [[1, 2, 3], [2], []],
                "maps": [[{"a": 1}, {"a": 2}], [{"a": 0}], []],
            }
        )

        cases = [
            ("any(x IN vals WHERE x = 2)", ["a", "b"]),
            ("all(x IN vals WHERE x < 4)", ["a", "b", "c"]),
            ("none(x IN vals WHERE x < 0)", ["a", "b", "c"]),
            ("single(x IN vals WHERE x = 2)", ["a", "b"]),
            ("any(x IN maps WHERE x.a = 2)", ["a"]),
        ]
        for expr, expected_ids in cases:
            assert self._where_expr_ids(nodes_df, expr) == expected_ids

    def test_row_pipeline_where_rows_expr_avoids_fillna_downcast_futurewarning(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "x": [1, None, 3],
            "lst": [[1], [2], None],
        })
        g = _mk_graph(nodes_df)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = g.gfql(
                [
                    rows(),
                    where_rows(expr="x IN lst OR x IS NULL"),
                    order_by([("id", "asc")]),
                    return_([("id", "id")]),
                ]
            )

        assert result._nodes["id"].tolist() == ["a", "b"]
        assert not any(
            "Downcasting object dtype arrays on .fillna" in str(w.message)
            for w in caught
            if issubclass(w.category, FutureWarning)
        )

    def test_row_pipeline_where_rows_expr_or_short_circuits_mixed_type_compare_with_is_not_null(self):
        nodes_df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "var": ["text", 0],
            }
        )

        result = _mk_graph(nodes_df).gfql(
            [
                rows(),
                where_rows(expr="var > 'te' OR var IS NOT NULL"),
                order_by([("id", "asc")]),
                return_([("id", "id")]),
            ]
        )

        assert result._nodes["id"].tolist() == ["a", "b"]

    @pytest.mark.parametrize(
        ("nodes", "steps", "expected_records"),
        [
            pytest.param(
                {"id": ["a", "b", "c"], "grp": ["x", "x", "y"], "score": [1, 2, 5]},
                [
                    rows(),
                    group_by(
                        ["grp"],
                        [("cnt", "count"), ("sum_score", "sum", "score"), ("avg_score", "avg", "score")],
                    ),
                    order_by([("grp", "asc")]),
                ],
                [
                    {"grp": "x", "cnt": 2, "sum_score": 3, "avg_score": 1.5},
                    {"grp": "y", "cnt": 1, "sum_score": 5, "avg_score": 5.0},
                ],
                id="group-by",
            ),
            pytest.param(
                {"id": ["a", "b", "c", "d"], "grp": ["x", "x", "y", "y"], "score": [1, None, 5, 6]},
                [rows(), group_by(["grp"], [("scores", "collect", "score")]), order_by([("grp", "asc")])],
                [{"grp": "x", "scores": [1.0]}, {"grp": "y", "scores": [5.0, 6.0]}],
                id="group-by-collect",
            ),
            pytest.param(
                {"id": ["a", "b", "c", "d"], "grp": ["x", "x", "x", "y"], "score": [1, 1, 2, 3]},
                [rows(), group_by(["grp"], [("cnt_shifted", "count_distinct", "score + 1")]), order_by([("grp", "asc")])],
                [{"grp": "x", "cnt_shifted": 2}, {"grp": "y", "cnt_shifted": 1}],
                id="group-by-count-distinct-expr",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "grp": ["x", "x", "y"], "score": [1, 2, 5]},
                [rows(), group_by(["grp"], [("scores_shifted", "collect", "score + 1")]), order_by([("grp", "asc")])],
                [{"grp": "x", "scores_shifted": [2, 3]}, {"grp": "y", "scores_shifted": [6]}],
                id="group-by-collect-expr",
            ),
            pytest.param(
                {"id": ["a", "b"], "score": [2, 1]},
                [rows(), with_([("id2", "id"), ("score2", "score")]), order_by([("score2", "asc")])],
                [{"id2": "b", "score2": 1}, {"id2": "a", "score2": 2}],
                id="with-alias",
            ),
            pytest.param(
                {"id": ["a", "b"], "score": [2, 5]},
                [
                    rows(),
                    select([("score", "score"), ("score_plus_2", "score + 2"), ("neg_score", "-1 * score")]),
                    order_by([("score", "asc")]),
                ],
                [{"score": 2, "score_plus_2": 4, "neg_score": -2}, {"score": 5, "score_plus_2": 7, "neg_score": -5}],
                id="select-arithmetic",
            ),
            pytest.param(
                {"id": ["a", "b"], "score": [2, 5]},
                [rows(), select([("id", "id"), ("vals", "[score, score + 1, 99]")]), order_by([("id", "asc")])],
                [{"id": "a", "vals": [2, 3, 99]}, {"id": "b", "vals": [5, 6, 99]}],
                id="select-dynamic-list",
            ),
            pytest.param(
                {"id": ["a", "b", "c", "d"], "vals": [[1, 2, 3], [2], [], None]},
                [rows(), select([("id", "id"), ("vals2", "[x IN vals WHERE x > 1 | x + 10]")]), order_by([("id", "asc")])],
                [
                    {"id": "a", "vals2": [12, 13]},
                    {"id": "b", "vals2": [12]},
                    {"id": "c", "vals2": []},
                    {"id": "d", "vals2": None},
                ],
                id="select-list-comprehension",
            ),
        ],
    )
    def test_row_pipeline_exact_record_cases(self, nodes, steps, expected_records):
        _assert_node_records(pd.DataFrame(nodes), steps, expected_records)

    @pytest.mark.parametrize(
        ("nodes", "items", "expected_records"),
        [
            pytest.param({"id": ["a"]}, [("v", "-(3 + 2)")], [{"v": -5}], id="unary-neg"),
            pytest.param(
                {"id": ["a"]},
                [("eq", "[1, 2] = [1, 2]"), ("neq", "[1, 2] = [1]")],
                [{"eq": True, "neq": False}],
                id="list-literal-comparison",
            ),
            pytest.param(
                {"id": ["a"]},
                [("eq_nested", "[1, {'k': [2, null]}] = [1, {'k': [2, null]}]")],
                [{"eq_nested": True}],
                id="nested-json-like-literal",
            ),
            pytest.param(
                {"id": ["a"]},
                [
                    ("any_hit", "any(x IN [1, 2, 3] WHERE x = 2)"),
                    ("all_lt_5", "all(x IN [1, 2, 3] WHERE x < 5)"),
                    ("none_gt_3", "none(x IN [1, 2, 3] WHERE x > 3)"),
                    ("single_even", "single(x IN [1, 2, 3] WHERE x % 2 = 0)"),
                    ("size_list", "size([1, 2, 3])"),
                    ("abs_num", "abs(-7)"),
                    ("substring_val", "substring('0123456789', 1)"),
                    ("toint_val", "toInteger(82.9)"),
                    ("tofloat_val", "toFloat(82)"),
                ],
                [
                    {
                        "any_hit": True,
                        "all_lt_5": True,
                        "none_gt_3": True,
                        "single_even": True,
                        "size_list": 3,
                        "abs_num": 7,
                        "substring_val": "123456789",
                        "toint_val": 82,
                        "tofloat_val": 82.0,
                    }
                ],
                id="quantifier-literals",
            ),
            pytest.param(
                {"id": ["a"]},
                [("nested_none", "none(x IN [['abc'], ['abc', 'def']] WHERE all(y IN x WHERE y = 'def'))")],
                [{"nested_none": True}],
                id="nested-quantifier-list-literal",
            ),
            pytest.param(
                {"id": ["a"]},
                [
                    ("none_map_hit", "none(x IN [{a: 2, b: 5}, {a: 4}] WHERE x.a = 2)"),
                    ("any_map_hit", "any(x IN [{a: 2, b: 5}, {a: 4}] WHERE x.a = 2)"),
                    ("single_map_hit", "single(x IN [{a: 2, b: 5}, {a: 4}] WHERE x.a = 2)"),
                ],
                [{"none_map_hit": False, "any_map_hit": True, "single_map_hit": True}],
                id="quantifier-over-map-list-literal",
            ),
            pytest.param(
                {"id": ["a"]},
                [
                    ("picked", "[x IN [{a: 2, b: 5}, {a: 4}] WHERE x.a = 2 | x.a]"),
                    ("picked_count", "size([x IN [{a: 2, b: 5}, {a: 4}] WHERE x.a = 2 | x])"),
                ],
                [{"picked": [2], "picked_count": 1}],
                id="list-comprehension-map-literal",
            ),
            pytest.param(
                {"id": ["a"]},
                [
                    ("eq_negated_any", "none(x IN [1, 2, 3] WHERE x = 2) = (NOT any(x IN [1, 2, 3] WHERE x = 2))"),
                    ("all_and_any", "all(x IN [1, 2, 3] WHERE x < 5) AND any(x IN [1, 2, 3] WHERE x = 2)"),
                ],
                [{"eq_negated_any": True, "all_and_any": True}],
                id="quantifier-composed-expressions",
            ),
            pytest.param(
                {"id": ["a"]},
                [("none_outer_lt", "none(x IN [1, 2, 3] WHERE all(y IN [1, 2, 3] WHERE x < y))")],
                [{"none_outer_lt": True}],
                id="nested-quantifier-outer-var",
            ),
            pytest.param(
                {"id": ["a"], "txt": ["abcdef"]},
                [
                    ("mid", "txt[1..3]"),
                    ("left", "txt[..2]"),
                    ("right", "txt[2..]"),
                    ("empty", "txt[0..0]"),
                ],
                [{"mid": "bc", "left": "ab", "right": "cdef", "empty": ""}],
                id="slice-variants",
            ),
            pytest.param(
                {"id": ["a"], "vals": [[[1], [2, 3], [4, 5], 5, [6, 7], [8, 9], 10]]},
                [
                    ("mid", "vals[1..3]"),
                    ("tail", "vals[4..6]"),
                    ("empty", "vals[0..0]"),
                ],
                [{"mid": [[2, 3], [4, 5]], "tail": [[6, 7], [8, 9]], "empty": []}],
                id="list-slice-variants",
            ),
            pytest.param(
                {"id": ["a"]},
                [("v", "[[1, 2, 3]]"), ("has3", "3 IN [[1, 2, 3]][0]")],
                [{"v": [[1, 2, 3]], "has3": True}],
                id="preserves-nested-list-shape",
            ),
        ],
    )
    def test_row_pipeline_select_single_row_exact_records(self, nodes, items, expected_records):
        _assert_single_row_select_records(items, expected_records, nodes_df=pd.DataFrame(nodes))

    @pytest.mark.parametrize("expr", ["[1, (2, 3)]", "{1: 2}"])
    def test_row_pipeline_select_rejects_non_json_literal_shapes(self, expr):
        with pytest.raises(Exception, match="unsupported token in row expression|unsupported row expression"):
            _run_node_steps(
                pd.DataFrame({"id": ["a"]}),
                [rows(), select([("bad", expr)])],
                edges_df=pd.DataFrame({"s": ["a"], "d": ["a"]}),
            )

    @pytest.mark.parametrize(
        ("nodes", "steps", "expected_records", "expected_columns", "edges"),
        [
            pytest.param(
                {"id": ["a", "b", "c"], "s": ["true", "false", None]},
                [rows(), select([("b", "toBoolean(s)"), ("s2", "toString(coalesce(s, 'x'))")]), order_by([("s2", "asc")])],
                [{"b": False, "s2": "false"}, {"b": True, "s2": "true"}, {"b": None, "s2": "x"}],
                None,
                None,
                id="toboolean-tostring-coalesce",
            ),
            pytest.param(
                {"id": ["a", "b"], "a": [True, True], "num": [2, 5]},
                [rows(), select([("num_proj", "a.num"), ("num_plus_1", "a.num + 1")]), order_by([("num_proj", "asc")])],
                [{"num_proj": 2, "num_plus_1": 3}, {"num_proj": 5, "num_plus_1": 6}],
                None,
                None,
                id="alias-property-expression",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "score": [2, None, 7]},
                [
                    rows(),
                    select(
                        [
                            ("id", "id"),
                            ("is_null", "score IS NULL"),
                            ("is_big", "score > 5"),
                            ("flag", "(score IS NULL) OR (score > 5)"),
                        ]
                    ),
                    order_by([("id", "asc")]),
                ],
                [
                    {"id": "a", "is_null": False, "is_big": False, "flag": False},
                    {"id": "b", "is_null": True, "is_big": None, "flag": True},
                    {"id": "c", "is_null": False, "is_big": True, "flag": True},
                ],
                None,
                None,
                id="boolean-null-expression",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "vals": [[1, 2, 3], [2], []], "maps": [[{"a": 1}, {"a": 2}], [{"a": 0}], []]},
                [
                    rows(),
                    select(
                        [
                            ("id", "id"),
                            ("has_2", "any(x IN vals WHERE x = 2)"),
                            ("all_lt_4", "all(x IN vals WHERE x < 4)"),
                            ("any_map_a_2", "any(x IN maps WHERE x.a = 2)"),
                        ]
                    ),
                    order_by([("id", "asc")]),
                ],
                [
                    {"id": "a", "has_2": True, "all_lt_4": True, "any_map_a_2": True},
                    {"id": "b", "has_2": True, "all_lt_4": True, "any_map_a_2": False},
                    {"id": "c", "has_2": False, "all_lt_4": True, "any_map_a_2": False},
                ],
                None,
                None,
                id="quantifier-on-list-columns",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "score": [1, 3, 2]},
                [rows(), order_by([("score + 2", "desc")]), limit(2), select([("id", "id"), ("score", "score")])],
                [{"id": "b", "score": 3}, {"id": "c", "score": 2}],
                None,
                None,
                id="order-by-string-expression",
            ),
            pytest.param(
                {"id": ["a", "b"], "path_nodes": [["n1", "n2"], ["n3"]], "path_rels": [["r1"], []]},
                [rows(), select([("id", "id"), ("ns", "nodes(path_nodes)"), ("rs", "relationships(path_rels)")]), order_by([("id", "asc")])],
                [{"id": "a", "ns": ["n1", "n2"], "rs": ["r1"]}, {"id": "b", "ns": ["n3"], "rs": []}],
                None,
                None,
                id="nodes-relationships-passthrough",
            ),
            pytest.param(
                {"id": ["a", "b"], "vals": [[1, 2], [3]]},
                [
                    rows(),
                    select(
                        [
                            ("id", "id"),
                            ("has2", "2 IN vals"),
                            ("has5", "5 IN vals"),
                            ("append9", "vals + 9"),
                            ("prepend9", "9 + vals"),
                        ]
                    ),
                    order_by([("id", "asc")]),
                ],
                [
                    {"id": "a", "has2": True, "has5": False, "append9": [1, 2, 9], "prepend9": [9, 1, 2]},
                    {"id": "b", "has2": False, "has5": False, "append9": [3, 9], "prepend9": [9, 3]},
                ],
                None,
                None,
                id="in-operator-list-concat",
            ),
            pytest.param(
                {"id": ["a", "b"], "one_src": [0, 0]},
                [rows(), select([("id", "id"), ("one", 1), ("txt", "id")])],
                [{"id": "a", "one": 1, "txt": "a"}, {"id": "b", "one": 1, "txt": "b"}],
                None,
                None,
                id="literal-expressions",
            ),
            pytest.param(
                {"id": ["a", "b"]},
                [rows(), select([("id", "id"), ("lst", [1, 2]), ("mp", {"k": "v"})]), order_by([("id", "asc")])],
                [{"id": "a", "lst": [1, 2], "mp": {"k": "v"}}, {"id": "b", "lst": [1, 2], "mp": {"k": "v"}}],
                None,
                None,
                id="broadcast-list-map-literals",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "score": [1, 3, 2]},
                [rows(), select([("id", "id"), ("bucket", "CASE WHEN score > 2 THEN 'hi' ELSE 'lo' END")]), order_by([("id", "asc")])],
                [{"id": "a", "bucket": "lo"}, {"id": "b", "bucket": "hi"}, {"id": "c", "bucket": "lo"}],
                None,
                None,
                id="case-when-expression",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "score": [1, 3, 2]},
                [rows(), select([("id", "id"), ("bucket", "CASE score WHEN 1 THEN 'lo' WHEN 3 THEN 'hi' ELSE 'mid' END")]), order_by([("id", "asc")])],
                [{"id": "a", "bucket": "lo"}, {"id": "b", "bucket": "hi"}, {"id": "c", "bucket": "mid"}],
                None,
                None,
                id="simple-case-expression",
            ),
            pytest.param(
                {"id": ["a"], "flag": [True]},
                [rows(), select([("bucket", "CASE flag WHEN 1 THEN 'one' ELSE 'other' END")])],
                [{"bucket": "other"}],
                None,
                None,
                id="simple-case-bool-not-equal-int",
            ),
            pytest.param(
                {"id": ["a", "b", "c"]},
                [rows(table="edges"), select([("weight", "weight")]), order_by([("weight", "desc")])],
                [{"weight": 3}, {"weight": 2}, {"weight": 1}],
                ["weight"],
                {"s": ["a", "b", "a"], "d": ["b", "c", "c"], "weight": [1, 3, 2]},
                id="rows-edges-table-projection",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "division": ["x", "x", "y"], "age": [3, 7, 4]},
                [rows(), group_by(["division"], [("count(*)", "count"), ("max(n.age)", "max", "age")]), order_by([("count(*)", "asc"), ("max(n.age)", "desc")])],
                [{"division": "y", "count(*)": 1, "max(n.age)": 4}, {"division": "x", "count(*)": 2, "max(n.age)": 7}],
                None,
                None,
                id="order-by-aggregate-alias-columns",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "division": ["x", "x", "y"], "age": [3, 7, 4]},
                [
                    rows(),
                    group_by(["division"], [("count(*)", "count"), ("max(n.age)", "max", "age")]),
                    return_([("division", "division"), ("count(*)", "count(*)"), ("max(n.age)", "max(n.age)")]),
                    order_by([("division", "asc")]),
                ],
                [
                    {"division": "x", "count(*)": 2, "max(n.age)": 7},
                    {"division": "y", "count(*)": 1, "max(n.age)": 4},
                ],
                None,
                None,
                id="return-aggregate-alias-columns",
            ),
        ],
    )
    def test_row_pipeline_exact_record_cases_more(
        self, nodes, steps, expected_records, expected_columns, edges
    ):
        edges_df = None if edges is None else pd.DataFrame(edges)
        _assert_node_records(
            pd.DataFrame(nodes),
            steps,
            expected_records,
            edges_df=edges_df,
            expected_columns=expected_columns,
        )

    @pytest.mark.parametrize(
        ("nodes", "steps", "expected_records", "normalize", "edges"),
        [
            pytest.param(
                {"id": ["a"]},
                [
                    rows(),
                    select(
                        [
                            ("any_empty", "any(x IN [] WHERE x = 1)"),
                            ("all_empty", "all(x IN [] WHERE x = 1)"),
                            ("none_empty", "none(x IN [] WHERE x = 1)"),
                            ("single_empty", "single(x IN [] WHERE x = 1)"),
                            ("any_null", "any(x IN [null] WHERE x = 1)"),
                            ("all_null", "all(x IN [null] WHERE x = 1)"),
                            ("none_null", "none(x IN [null] WHERE x = 1)"),
                            ("single_null", "single(x IN [null] WHERE x = 1)"),
                        ]
                    ),
                ],
                [
                    {
                        "any_empty": False,
                        "all_empty": True,
                        "none_empty": True,
                        "single_empty": False,
                        "any_null": None,
                        "all_null": None,
                        "none_null": None,
                        "single_null": None,
                    }
                ],
                True,
                {"s": ["a"], "d": ["a"]},
                id="quantifier-empty-and-null",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "txt": ["abcdef", "xxabyy", None]},
                [
                    rows(),
                    select(
                        [
                            ("id", "id"),
                            ("has_ab", "txt CONTAINS 'ab'"),
                            ("starts_ab", "txt STARTS WITH 'ab'"),
                            ("ends_ef", "txt ENDS WITH 'ef'"),
                        ]
                    ),
                    order_by([("id", "asc")]),
                ],
                [
                    {"id": "a", "has_ab": True, "starts_ab": True, "ends_ef": True},
                    {"id": "b", "has_ab": True, "starts_ab": False, "ends_ef": False},
                    {"id": "c", "has_ab": None, "starts_ab": None, "ends_ef": None},
                ],
                True,
                None,
                id="string-predicate-ops",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "txt": ["abc STARTS WITH xyz", "abc xyz", None]},
                [rows(), select([("id", "id"), ("hit", "txt CONTAINS 'STARTS WITH'")]), order_by([("id", "asc")])],
                [{"id": "a", "hit": True}, {"id": "b", "hit": False}, {"id": "c", "hit": None}],
                True,
                None,
                id="string-predicate-keyword-in-literal",
            ),
            pytest.param(
                {"id": ["a", "b"], "txt": ["abcdef", None]},
                [
                    rows(),
                    select(
                        [
                            ("contains_null", "txt CONTAINS null"),
                            ("starts_null", "txt STARTS WITH null"),
                            ("ends_null", "txt ENDS WITH null"),
                        ]
                    ),
                ],
                [{"contains_null": None, "starts_null": None, "ends_null": None}] * 2,
                True,
                None,
                id="string-predicate-null-rhs",
            ),
            pytest.param(
                {"id": ["a", "b"], "txt": ["abcdef", "ghij"]},
                [rows(), select([("lhs_null", "txt[null..2]"), ("rhs_null", "txt[1..null]")])],
                [{"lhs_null": None, "rhs_null": None}, {"lhs_null": None, "rhs_null": None}],
                True,
                None,
                id="slice-null-bounds",
            ),
            pytest.param(
                {
                    "id": ["a", "b", "c"],
                    "vals": [[1, 2, 3], [4], []],
                    "txt": ["ab", "cd", ""],
                    "score": [-2, 0, 3],
                },
                [
                    rows(),
                    select(
                        [
                            ("id", "id"),
                            ("h_vals", "head(vals)"),
                            ("t_vals", "tail(vals)"),
                            ("r_vals", "reverse(vals)"),
                            ("h_txt", "head(txt)"),
                            ("t_txt", "tail(txt)"),
                            ("r_txt", "reverse(txt)"),
                            ("sgn", "sign(score)"),
                        ]
                    ),
                    order_by([("id", "asc")]),
                ],
                [
                    {
                        "id": "a",
                        "h_vals": 1,
                        "t_vals": [2, 3],
                        "r_vals": [3, 2, 1],
                        "h_txt": "a",
                        "t_txt": "b",
                        "r_txt": "ba",
                        "sgn": -1,
                    },
                    {
                        "id": "b",
                        "h_vals": 4,
                        "t_vals": [],
                        "r_vals": [4],
                        "h_txt": "c",
                        "t_txt": "d",
                        "r_txt": "dc",
                        "sgn": 0,
                    },
                    {
                        "id": "c",
                        "h_vals": None,
                        "t_vals": [],
                        "r_vals": [],
                        "h_txt": None,
                        "t_txt": "",
                        "r_txt": "",
                        "sgn": 1,
                    },
                ],
                True,
                None,
                id="sequence-function-family",
            ),
        ],
    )
    def test_row_pipeline_normalized_exact_record_cases(
        self, nodes, steps, expected_records, normalize, edges
    ):
        edges_df = None if edges is None else pd.DataFrame(edges)
        _assert_node_records(
            pd.DataFrame(nodes),
            steps,
            expected_records,
            edges_df=edges_df,
            normalize=normalize,
        )

    def test_row_pipeline_order_by_mixed_list_values(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c", "d", "e", "f", "g"],
            "v": [[], ["a"], [1], [None, 1], [1, None], float("nan"), None],
        })
        g = _mk_graph(nodes_df)

        with pytest.raises(Exception, match="unsupported order_by expression for vectorized execution"):
            g.gfql([
                rows(),
                order_by([("v", "asc")]),
                select([("v", "v")]),
            ])

    @pytest.mark.parametrize(
        ("nodes", "column", "asc_expected", "desc_expected", "limit_value"),
        [
            pytest.param(
                {
                    "id": list("abcdefgh"),
                    "lists": [[], ["a"], ["a", 1], [1], [1, "a"], [1, None], [None, 1], [None, 2]],
                },
                "lists",
                [[], ["a"], ["a", 1], [1], [1, "a"], [1, None], [None, 1], [None, 2]],
                [[None, 2], [None, 1], [1, None], [1, "a"], [1], ["a", 1], ["a"], []],
                None,
                id="cypher-list-value-semantics",
            ),
            pytest.param(
                {
                    "id": list("abcde"),
                    "times": [
                        "10:35-08:00",
                        "12:31:14.645876123+01:00",
                        "12:31:14.645876124+01:00",
                        "12:35:15+05:00",
                        "12:30:14.645876123+01:01",
                    ],
                },
                "times",
                ["12:35:15+05:00", "12:30:14.645876123+01:01", "12:31:14.645876123+01:00"],
                ["10:35-08:00", "12:31:14.645876124+01:00", "12:31:14.645876123+01:00"],
                3,
                id="time-offsets",
            ),
            pytest.param(
                {
                    "id": list("abcde"),
                    "datetimes": [
                        "1984-10-11T12:30:14.000000012+00:15",
                        "1984-10-11T12:31:14.645876123+00:17",
                        "0001-01-01T01:01:01.000000001-11:59",
                        "9999-09-09T09:59:59.999999999+11:59",
                        "1980-12-11T12:31:14-11:59",
                    ],
                },
                "datetimes",
                [
                    "0001-01-01T01:01:01.000000001-11:59",
                    "1980-12-11T12:31:14-11:59",
                    "1984-10-11T12:31:14.645876123+00:17",
                ],
                [
                    "9999-09-09T09:59:59.999999999+11:59",
                    "1984-10-11T12:30:14.000000012+00:15",
                    "1984-10-11T12:31:14.645876123+00:17",
                ],
                3,
                id="datetime-offsets",
            ),
        ],
    )
    def test_row_pipeline_order_by_value_semantics(
        self, nodes, column, asc_expected, desc_expected, limit_value
    ):
        _assert_ordered_projection_values(
            nodes,
            column,
            asc_expected,
            desc_expected,
            limit_value=limit_value,
        )

    @pytest.mark.parametrize(
        ("nodes", "steps", "pattern", "edges"),
        [
            pytest.param(
                {"id": ["a", "b"], "v": [1, 2]},
                [rows(source="missing")],
                "requires node column|alias column not found",
                None,
                id="bad-source-alias",
            ),
            pytest.param(
                {"id": ["a"], "x": [1]},
                [rows(), select([("bad", "reverse(x)")])],
                "reverse\\(\\) requires list/string input",
                {"s": ["a"], "d": ["a"]},
                id="reverse-invalid-input",
            ),
            pytest.param(
                {"id": ["a", "b"]},
                [rows("bad_table")],
                "table",
                None,
                id="invalid-rows-table",
            ),
            pytest.param(
                {"id": ["a", "b"]},
                [rows(), select([("x", "missing_col")])],
                "unsupported token in row expression|unsupported row expression",
                None,
                id="select-missing-column",
            ),
            pytest.param(
                {"id": ["a", "b"]},
                [rows(), order_by([("missing_col", "asc")])],
                "order_by column not found|unsupported token in row expression|unsupported row expression",
                None,
                id="order-by-missing-column",
            ),
        ],
    )
    def test_row_pipeline_runtime_error_cases(self, nodes, steps, pattern, edges):
        edges_df = None if edges is None else pd.DataFrame(edges)
        with pytest.raises(Exception, match=pattern):
            _run_node_steps(pd.DataFrame(nodes), steps, edges_df=edges_df)

    @pytest.mark.parametrize("builder", [skip, limit], ids=["skip", "limit"])
    @pytest.mark.parametrize("value", [-1, True, "1.5", "bad"])
    def test_row_pipeline_invalid_page_values_rejected(self, builder, value):
        with pytest.raises(Exception, match="Invalid type for parameter|non-negative integer|non-negative"):
            _run_node_steps(pd.DataFrame({"id": ["a", "b"]}), [rows(), builder(value)])

    def test_row_pipeline_vectorized_cudf_when_available(self):
        cudf = pytest.importorskip("cudf")

        nodes_pd = pd.DataFrame({"id": ["a", "b", "c"], "score": [3, 1, 2]})
        edges_pd = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(cudf.from_pandas(nodes_pd), "id").edges(cudf.from_pandas(edges_pd), "s", "d")

        result = g.gfql([
            rows(),
            order_by([("score", "asc")]),
            limit(2),
        ])

        assert type(result._nodes).__module__.startswith("cudf")
        assert result._nodes["score"].to_pandas().tolist() == [1, 2]

    def test_row_pipeline_cudf_where_unwind_group_by_when_available(self):
        cudf = pytest.importorskip("cudf")

        nodes_pd = pd.DataFrame({
            "id": ["a", "b", "c"],
            "grp": ["x", "x", "y"],
            "vals": [[1, 2], [3], [4, 5]],
            "score": [1, 2, 5],
        })
        edges_pd = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(cudf.from_pandas(nodes_pd), "id").edges(cudf.from_pandas(edges_pd), "s", "d")

        result = g.gfql([
            rows(),
            where_rows({"score": gt(1)}),
            unwind("vals", as_="v"),
            group_by(["grp"], [("cnt", "count"), ("sum_v", "sum", "v")]),
            order_by([("grp", "asc")]),
        ])
        assert type(result._nodes).__module__.startswith("cudf")
        pdf = result._nodes.to_pandas()
        assert pdf.to_dict(orient="records") == [
            {"grp": "x", "cnt": 1, "sum_v": 3},
            {"grp": "y", "cnt": 2, "sum_v": 9},
        ]


class TestRowPipelineSafelist:
    @staticmethod
    def _assert_e201(function, params):
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params(function, params)
        assert exc_info.value.code == ErrorCode.E201

    @staticmethod
    def _assert_e303(function, params):
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params(function, params)
        assert exc_info.value.code == ErrorCode.E303

    @staticmethod
    def _assert_valid(function, params):
        assert validate_call_params(function, params) == params

    @staticmethod
    def _patch_where_rows_parser(
        monkeypatch,
        *,
        parse=_parse_identifier_score,
        capabilities=_capabilities_ok,
        collect=_collect_score,
    ):
        monkeypatch.setattr(
            call_safelist,
            "_where_rows_expr_parser_fn",
            lambda: _where_rows_parser_bundle(parse, capabilities, collect),
        )

    @staticmethod
    def _assert_runtime_where_rows(
        monkeypatch,
        *,
        nodes,
        expr,
        bundle,
        expected_records=None,
        expected_message=None,
    ):
        g = _mk_graph(pd.DataFrame(nodes))
        monkeypatch.setattr(row_pipeline_mixin, "_gfql_expr_runtime_parser_bundle", lambda: bundle)
        steps = [rows(), where_rows(expr=expr)]
        if expected_message is None:
            result = g.gfql(steps)
            assert result._nodes.reset_index(drop=True).to_dict(orient="records") == expected_records
            return

        with pytest.raises(GFQLTypeError) as exc_info:
            g.gfql(steps + [return_([("id", "id")])])
        assert exc_info.value.code == ErrorCode.E303
        assert expected_message in exc_info.value.message

    def test_row_pipeline_rows_validation(self):
        self._assert_valid("rows", {})
        self._assert_valid("rows", {"table": "edges", "source": "rel"})

        self._assert_e201("rows", {"table": "bad"})

    def test_row_pipeline_select_validation(self):
        self._assert_valid("select", {"items": [("name", "name"), ("const", 1)]})
        self._assert_valid("select", {"items": ["name", ("const", 1)]})
        self._assert_valid("return_", {"items": [("name", "name")]})
        self._assert_valid("with_", {"items": ["name"]})

        for bad_items in [
            None,
            "name",
            [""],
            [("a",)],
            [("a", "b", "c")],
            [1],
            [(1, "name")],
            [("", "name")],
        ]:
            self._assert_e201("select", {"items": bad_items})
            self._assert_e201("return_", {"items": bad_items})

    def test_row_pipeline_with_where_rows_validation(self):
        self._assert_valid("with_", {"items": [("name", "name")]})
        self._assert_valid("where_rows", {"filter_dict": {"name": "alice"}})
        valid_exprs = [
            "score > 1 AND name != 'bob'",
            "name = 'rand()'",
            'name = "rand()"',
            *[f"txt {op}" for op in ["CONTAINS 5", "CONTAINS null", "CONTAINS (5)", "STARTS WITH (5)", "ENDS WITH (5)"]],
            "CASE WHEN score > 1 THEN true ELSE false END",
            "CASE WHEN score > 1 THEN true END",
            "CASE score WHEN 1 THEN true ELSE false END",
            *[f"{fn}(x IN vals WHERE x {pred})" for fn, pred in [("any", "= 2"), ("all", "> 1"), ("none", "< 0"), ("single", "= 2")]],
            "score > 1 AND CASE WHEN id = 'a' THEN true ELSE false END",
            "score > 1 AND CASE WHEN id = 'a' THEN true END",
            *[f"size([x IN vals WHERE x > 1{suffix}]) > 0" for suffix in ["", " | x"]],
        ]
        for expr in valid_exprs:
            self._assert_valid("where_rows", {"expr": expr})
        pred = gt(1)
        self._assert_valid("where_rows", {"filter_dict": {"score": pred}})
        self._assert_valid(
            "where_rows",
            {"filter_dict": {"score": pred}, "expr": "score > 1"},
        )

        bad_filter_dict_inputs = [{"filter_dict": "bad"}, {"filter_dict": {1: "x"}}]
        for bad_params in bad_filter_dict_inputs:
            self._assert_e201("where_rows", bad_params)

        invalid_exprs = [
            "rand() > 0.1",
            *[f"txt {op} rhs" for op in ["CONTAINS", "STARTS WITH", "ENDS WITH"]],
            *[f"txt CONTAINS {rhs}" for rhs in ["[1,2]", "{k: 'v'}", "(rhs)"]],
            "any(x IN vals WHERE x = 2",
            "any(x vals WHERE x = 2)",
            "any(x IN vals | WHERE x = 2)",
            *[
                f"size({expr}) > 0"
                for expr in [
                    "[x vals WHERE x > 1]",
                    "[x IN vals WHERE ]",
                    "[x IN vals WHERE x > 1 | ]",
                    "[x IN vals | ]",
                    "[x IN vals | WHERE x > 1]",
                    "[x IN vals | WHERE x > 1 | x + 1]",
                ]
            ],
            "(id = 'a') | (id = 'b')",
            "score > 1 THEN true",
            "END",
            "CASE WHEN score > 1 THEN true ELSE false END END",
            "any(x IN vals WHERE x = 2))",
            "any(x IN vals WHERE x = 2) foo",
            "id == 'a'",
            "id = 'a' -- comment",
            "id = 'a' /*x*/",
            "name = 'unterminated",
            'name = "unterminated',
            "meta = {k: }",
            "id = 'a' OR",
            "OR id = 'a'",
            "NOT",
            "id = 'a' AND OR id = 'b'",
            "id = 'a' OR AND id = 'b'",
            "id = 'a' AND NOT",
            "id = 'a' OR NOT",
            "id = 'a' AND (OR id = 'b')",
            "id = 'a' OR (AND id = 'b')",
            *["id = = 'a'", "score > = 2", "score < = 2", "score ! = 2", "score < > 2", "score =< 2", "score => 2"],
            "coalesce(id, 'x') y",
            "size(vals) z",
            "any(x IN vals WHERE x = 2) OR OR id = 'a'",
            *["size() > 0", "size( ) > 0", "size(,) > 0"],
            "coalesce() = id",
            "toString() = 'x'",
            "toBoolean()",
            "abs() = 1",
            "sign() = 1",
            "head() = 1",
            "tail() = []",
            "reverse() = []",
            *["size(( )) > 0", "coalesce(( )) = id", "coalesce(id score) = id", "size(vals,) > 0", "size(,vals) > 0"],
        ]
        for expr in invalid_exprs:
            self._assert_e201("where_rows", {"expr": expr})

    def test_row_pipeline_where_rows_parser_authority(self, monkeypatch):
        monkeypatch.setattr(call_safelist, "_where_rows_expr_parser_parse_ok", lambda _expr: False)
        self._patch_where_rows_parser(monkeypatch, parse=lambda _expr: object(), collect=lambda _node: set())
        self._assert_e201("where_rows", {"expr": "score > 1"})

    @pytest.mark.parametrize(
        ("capabilities", "expr", "expected"),
        [
            pytest.param(_capabilities_ok, "score == 1", {"expr": "score == 1"}, id="parser-controls-syntax"),
            pytest.param(_capabilities_unsupported, "score > 1", ErrorCode.E201, id="capability-fail"),
        ],
    )
    def test_row_pipeline_where_rows_strict_parser_authority(self, monkeypatch, capabilities, expr, expected):
        self._patch_where_rows_parser(monkeypatch, capabilities=capabilities)
        if isinstance(expected, dict):
            assert validate_call_params("where_rows", {"expr": expr}) == expected
            return
        self._assert_e201("where_rows", {"expr": expr})

    @pytest.mark.parametrize(
        ("bundle", "params", "expected"),
        [
            pytest.param(
                _where_rows_parser_bundle(lambda _expr: "node", collect=_collect_nested_score_vals),
                {"filter_dict": {"id": "a"}, "expr": "ignored"},
                ["id", "n", "score", "vals"],
                id="parser-derived-cols",
            ),
            pytest.param(None, {"expr": "any(x IN vals WHERE x > threshold)"}, [], id="parser-required"),
        ],
    )
    def test_row_pipeline_where_rows_required_cols(self, monkeypatch, bundle, params, expected):
        monkeypatch.setattr(call_safelist, "_where_rows_expr_parser_fn", lambda: bundle)
        assert call_safelist._where_rows_requires_node_cols(params) == expected

    def test_row_pipeline_where_rows_validator_rejects_without_parser(self, monkeypatch):
        monkeypatch.setattr(call_safelist, "_where_rows_expr_parser_fn", lambda: None)
        self._assert_e201("where_rows", {"expr": "score > 1"})

    @pytest.mark.parametrize(
        ("nodes", "expr", "bundle", "expected_records", "expected_message"),
        [
            pytest.param(
                {"id": ["a", "b", "c"], "score": [1, 2, 3]},
                "score > 1",
                None,
                None,
                "parser backend unavailable",
                id="parser-unavailable",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "score": [1, 2, 3]},
                "score > 1",
                _runtime_parser_bundle(_raise_bad_parse),
                None,
                "parser validation failed",
                id="parse-fail",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "score": [1, 2, 3]},
                "score > 1",
                _runtime_parser_bundle(_parse_score_gt_one),
                [{"id": "b", "score": 2}, {"id": "c", "score": 3}],
                None,
                id="ast-success",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "score": [1, 2, 3]},
                "score > 1",
                _runtime_parser_bundle(_parse_unknown_fn_score),
                None,
                "AST evaluator unsupported",
                id="unsupported-ast",
            ),
            pytest.param(
                {"id": ["a", "b", "c"], "name": ["x", "y", "z"]},
                "name - 'x'",
                _runtime_parser_bundle(_parse_name_minus_x),
                None,
                "AST evaluator unsupported",
                id="type-error-normalized",
            ),
        ],
    )
    def test_row_pipeline_runtime_parser_paths(
        self, monkeypatch, nodes, expr, bundle, expected_records, expected_message
    ):
        self._assert_runtime_where_rows(
            monkeypatch,
            nodes=nodes,
            expr=expr,
            bundle=bundle,
            expected_records=expected_records,
            expected_message=expected_message,
        )

    def test_row_pipeline_eval_expr_ast_subset_parity(self):
        _assert_ast_parity(
            {
                "id": ["a", "b", "c"],
                "score": [1, 2, 3],
                "name": ["a", "bb", "ccc"],
                "flag1": [True, True, False],
                "flag2": [True, False, False],
            },
            [
                (
                    expr_parser.BinaryOp(">", expr_parser.Identifier("score"), expr_parser.Literal(1)),
                    "score > 1",
                ),
                (
                    expr_parser.UnaryOp(
                        "not",
                        expr_parser.BinaryOp(">", expr_parser.Identifier("score"), expr_parser.Literal(1)),
                    ),
                    "NOT (score > 1)",
                ),
                (
                    expr_parser.BinaryOp(
                        "and",
                        expr_parser.BinaryOp(">", expr_parser.Identifier("score"), expr_parser.Literal(1)),
                        expr_parser.BinaryOp("<", expr_parser.Identifier("score"), expr_parser.Literal(3)),
                    ),
                    "score > 1 AND score < 3",
                ),
                (
                    expr_parser.BinaryOp(
                        "xor",
                        expr_parser.Identifier("flag1"),
                        expr_parser.Identifier("flag2"),
                    ),
                    "flag1 XOR flag2",
                ),
                (
                    expr_parser.BinaryOp(
                        "contains",
                        expr_parser.Identifier("name"),
                        expr_parser.Literal("b"),
                    ),
                    "name CONTAINS 'b'",
                ),
                (
                    expr_parser.BinaryOp(
                        ">=",
                        expr_parser.BinaryOp("+", expr_parser.Identifier("score"), expr_parser.Literal(1)),
                        expr_parser.Literal(3),
                    ),
                    "score + 1 >= 3",
                ),
            ],
        )

    def test_row_pipeline_eval_expr_ast_advanced_parity(self):
        _assert_ast_parity(
            {"id": ["a", "b", "c"], "score": [1, 3, 2], "vals": [[1], [1, 2], [2, 3]]},
            [
            (
                expr_parser.CaseWhen(
                    condition=expr_parser.BinaryOp(">", expr_parser.Identifier("score"), expr_parser.Literal(2)),
                    when_true=expr_parser.Literal(True),
                    when_false=expr_parser.Literal(False),
                ),
                "CASE WHEN score > 2 THEN true ELSE false END",
            ),
            (
                expr_parser.QuantifierExpr(
                    fn="any",
                    var="x",
                    source=expr_parser.Identifier("vals"),
                    predicate=expr_parser.BinaryOp(">", expr_parser.Identifier("x"), expr_parser.Literal(1)),
                ),
                "any(x IN vals WHERE x > 1)",
            ),
            (
                expr_parser.ListComprehension(
                    var="x",
                    source=expr_parser.Identifier("vals"),
                    predicate=expr_parser.BinaryOp(">", expr_parser.Identifier("x"), expr_parser.Literal(1)),
                    projection=expr_parser.BinaryOp("+", expr_parser.Identifier("x"), expr_parser.Literal(1)),
                ),
                "[x IN vals WHERE x > 1 | x + 1]",
            ),
            (
                expr_parser.SubscriptExpr(
                    value=expr_parser.Identifier("vals"),
                    key=expr_parser.Literal(1),
                ),
                "vals[1]",
            ),
            (
                expr_parser.SliceExpr(
                    value=expr_parser.Identifier("vals"),
                    start=expr_parser.Literal(0),
                    stop=expr_parser.Literal(2),
                ),
                "vals[0..2]",
            ),
            ],
        )

    def test_row_pipeline_order_by_validation(self):
        self._assert_valid("order_by", {"keys": [("name", "asc"), ("score", "desc")]})
        self._assert_valid("order_by", {"keys": [("count(*)", "asc"), ("max(n.age)", "desc")]})
        self._assert_valid("order_by", {"keys": [("[score, score + 1]", "asc")]})

        for bad_keys in [
            None,
            "name",
            [("a",)],
            [("a", "asc", "x")],
            [1],
            [(1, "asc")],
            [("a", "up")],
            [("unknown_fn(score)", "asc")],
        ]:
            self._assert_e201("order_by", {"keys": bad_keys})

    @pytest.mark.parametrize("function", ["skip", "limit"])
    def test_row_pipeline_skip_limit_validation(self, function):
        for value in [0, 2, 2.0, "3"]:
            params = validate_call_params(function, {"value": value})
            assert params == {"value": value}

        for bad_value in [True, -1, -1.0, "-1", "1.5", "abc"]:
            self._assert_e201(function, {"value": bad_value})

    def test_row_pipeline_distinct_validation(self):
        self._assert_valid("distinct", {})

        self._assert_e303("distinct", {"extra": True})

    def test_row_pipeline_unwind_group_by_validation(self):
        self._assert_valid("unwind", {"expr": "vals", "as_": "v"})
        self._assert_valid("unwind", {"expr": [1, 2, 3], "as_": "v"})
        self._assert_valid(
            "group_by",
            {"keys": ["grp"], "aggregations": [("cnt", "count"), ("sum_v", "sum", "v")]},
        )
        self._assert_valid(
            "group_by",
            {"keys": ["grp"], "aggregations": [("vals", "collect", "v")]},
        )
        self._assert_valid(
            "group_by",
            {"keys": ["grp"], "aggregations": [("vals", "collect", "v + 1")]},
        )

        self._assert_e201("unwind", {"expr": 1})
        self._assert_e201("unwind", {"expr": "vals", "as_": ""})
        self._assert_e201("group_by", {"keys": ["grp"], "aggregations": ["bad"]})
        self._assert_e201("group_by", {"keys": ["grp"], "aggregations": [("x", "median", "score")]})
        self._assert_e201("group_by", {"keys": [], "aggregations": [("x", "count")]})
