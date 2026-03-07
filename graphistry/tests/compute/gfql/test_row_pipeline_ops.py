import warnings

import pandas as pd
import pytest

import graphistry.compute.gfql.call_safelist as call_safelist
import graphistry.compute.gfql.expr_parser as expr_parser
import graphistry.compute.gfql.row_pipeline_mixin as row_pipeline_mixin
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
from graphistry.compute.gfql.call_safelist import validate_call_params
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


def _self_loop_edges(nodes_df):
    if len(nodes_df) == 0:
        return pd.DataFrame({"s": [], "d": []})
    node_id = nodes_df["id"].iloc[0]
    return pd.DataFrame({"s": [node_id], "d": [node_id]})


def _run_node_steps(nodes_df, steps, edges_df=None):
    return _mk_graph(nodes_df, edges_df).gfql(steps)._nodes


def _assert_node_records(nodes_df, steps, expected_records, *, edges_df=None, expected_columns=None):
    result_nodes = _run_node_steps(nodes_df, steps, edges_df=edges_df)
    if expected_columns is not None:
        assert list(result_nodes.columns) == expected_columns
    assert result_nodes.reset_index(drop=True).to_dict(orient="records") == expected_records
    return result_nodes


def _assert_single_row_select_records(items, expected_records, *, nodes_df=None):
    base_nodes = pd.DataFrame({"id": ["a"]}) if nodes_df is None else nodes_df
    return _assert_node_records(
        base_nodes,
        [rows(), select(items)],
        expected_records,
        edges_df=_self_loop_edges(base_nodes),
    )


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

    def test_row_pipeline_distinct_unhashable_cells(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "vals": [[1, 2], [1, 2], [3]],
            "meta": [{"k": "v"}, {"k": "v"}, {"k": "z"}],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([("vals", "vals"), ("meta", "meta")]),
            distinct(),
        ])

        assert result._nodes.reset_index(drop=True).to_dict(orient="records") == [
            {"vals": [1, 2], "meta": {"k": "v"}},
            {"vals": [3], "meta": {"k": "z"}},
        ]

    def test_row_pipeline_order_by_list_values_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "vals": [[1, 2], [1, 2], [3]],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([("vals", "vals")]),
            order_by([("vals", "asc")]),
        ])
        assert result._nodes["vals"].tolist() == [[1, 2], [1, 2], [3]]

    def test_row_pipeline_exec_with_match_alias_source(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "grp": ["x", "x", "y"],
        })
        edges_df = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            n({"grp": "x"}, name="a"),
            rows(source="a"),
            return_([("id", "id")]),
            order_by([("id", "asc")]),
        ])

        assert list(result._nodes.columns) == ["id"]
        assert result._nodes["id"].tolist() == ["a", "b"]

    def test_row_pipeline_where_rows_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "score": [1, 3, 2],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            where_rows({"score": gt(1)}),
            order_by([("id", "asc")]),
            return_([("id", "id"), ("score", "score")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"id": "b", "score": 3},
            {"id": "c", "score": 2},
        ]

    def test_row_pipeline_where_rows_expr_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "vals": [[1], [1, 2], [1, 2, 3]],
            "name": ["n1", "n2", "n3"],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            where_rows(expr="size(vals) > 1 AND NOT (name = 'n3')"),
            order_by([("id", "asc")]),
            return_([("id", "id"), ("vals", "vals"), ("name", "name")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"id": "b", "vals": [1, 2], "name": "n2"},
        ]

    def test_row_pipeline_where_rows_expr_quoted_function_text(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "name": ["rand()", "plain"]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            where_rows(expr="name = 'rand()'"),
            return_([("id", "id"), ("name", "name")]),
        ])

        assert result._nodes.to_dict(orient="records") == [{"id": "a", "name": "rand()"}]

    def test_row_pipeline_where_rows_expr_size_literal_with_paren(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "name": ["n1", "n2"]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            where_rows(expr="size(['x)', 'y']) = 2 AND NOT (name = 'n2')"),
            return_([("id", "id")]),
        ])

        assert result._nodes.to_dict(orient="records") == [{"id": "a"}]

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
                ["c", "d"],
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

    def test_row_pipeline_unwind_column_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b"],
            "vals": [[1, 2], [3]],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            unwind("vals", as_="v"),
            select([("id", "id"), ("v", "v")]),
            order_by([("v", "asc")]),
        ])
        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "v": 1},
            {"id": "a", "v": 2},
            {"id": "b", "v": 3},
        ]

    def test_row_pipeline_unwind_string_expression_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b"],
            "first": [[1, 2], [3]],
            "second": [[4], [5, 6]],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            unwind("first + second", as_="x"),
            select([("x", "x")]),
            order_by([("x", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"x": 1},
            {"x": 2},
            {"x": 3},
            {"x": 4},
            {"x": 5},
            {"x": 6},
        ]

    def test_row_pipeline_unwind_subscript_expression(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b"],
            "qrows": [[[2], [3, 4]], [[5], [6, 7]]],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            unwind("qrows[0]", as_="q"),
            select([("q", "q")]),
            order_by([("q", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"q": 2},
            {"q": 5},
        ]

    def test_row_pipeline_unwind_dynamic_subscript_expression(self):
        nodes_df = pd.DataFrame({
            "id": ["seed"],
        })
        edges_df = pd.DataFrame({"s": [], "d": []})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            with_([("prows", [0, 1]), ("qrows", [[2], [3, 4]])]),
            unwind("prows", as_="p"),
            unwind("qrows[p]", as_="q"),
            select([("p", "p"), ("q", "q")]),
            order_by([("p", "asc"), ("q", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"p": 0, "q": 2},
            {"p": 1, "q": 3},
            {"p": 1, "q": 4},
        ]

    def test_row_pipeline_group_by_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "grp": ["x", "x", "y"],
            "score": [1, 2, 5],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            group_by(
                ["grp"],
                [("cnt", "count"), ("sum_score", "sum", "score"), ("avg_score", "avg", "score")],
            ),
            order_by([("grp", "asc")]),
        ])
        assert result._nodes.to_dict(orient="records") == [
            {"grp": "x", "cnt": 2, "sum_score": 3, "avg_score": 1.5},
            {"grp": "y", "cnt": 1, "sum_score": 5, "avg_score": 5.0},
        ]

    def test_row_pipeline_group_by_collect_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c", "d"],
            "grp": ["x", "x", "y", "y"],
            "score": [1, None, 5, 6],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            group_by(
                ["grp"],
                [("scores", "collect", "score")],
            ),
            order_by([("grp", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"grp": "x", "scores": [1.0]},
            {"grp": "y", "scores": [5.0, 6.0]},
        ]

    def test_row_pipeline_group_by_expression_count_distinct_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c", "d"],
            "grp": ["x", "x", "x", "y"],
            "score": [1, 1, 2, 3],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            group_by(["grp"], [("cnt_shifted", "count_distinct", "score + 1")]),
            order_by([("grp", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"grp": "x", "cnt_shifted": 2},
            {"grp": "y", "cnt_shifted": 1},
        ]

    def test_row_pipeline_group_by_expression_collect_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "grp": ["x", "x", "y"],
            "score": [1, 2, 5],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            group_by(["grp"], [("scores_shifted", "collect", "score + 1")]),
            order_by([("grp", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"grp": "x", "scores_shifted": [2, 3]},
            {"grp": "y", "scores_shifted": [6]},
        ]

    def test_row_pipeline_with_alias(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "score": [2, 1]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            with_([("id2", "id"), ("score2", "score")]),
            order_by([("score2", "asc")]),
        ])
        assert result._nodes.to_dict(orient="records") == [
            {"id2": "b", "score2": 1},
            {"id2": "a", "score2": 2},
        ]

    def test_row_pipeline_select_string_arithmetic_expression(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "score": [2, 5]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([("score", "score"), ("score_plus_2", "score + 2"), ("neg_score", "-1 * score")]),
            order_by([("score", "asc")]),
        ])
        assert result._nodes.to_dict(orient="records") == [
            {"score": 2, "score_plus_2": 4, "neg_score": -2},
            {"score": 5, "score_plus_2": 7, "neg_score": -5},
        ]

    def test_row_pipeline_select_dynamic_list_expression_vectorized(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "score": [2, 5]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([("id", "id"), ("vals", "[score, score + 1, 99]")]),
            order_by([("id", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "vals": [2, 3, 99]},
            {"id": "b", "vals": [5, 6, 99]},
        ]

    def test_row_pipeline_select_list_comprehension_filter_projection_vectorized(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c", "d"], "vals": [[1, 2, 3], [2], [], None]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([("id", "id"), ("vals2", "[x IN vals WHERE x > 1 | x + 10]")]),
            order_by([("id", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "vals2": [12, 13]},
            {"id": "b", "vals2": [12]},
            {"id": "c", "vals2": []},
            {"id": "d", "vals2": None},
        ]

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
                ],
                [
                    {
                        "any_hit": True,
                        "all_lt_5": True,
                        "none_gt_3": True,
                        "single_even": True,
                        "size_list": 3,
                        "abs_num": 7,
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
                {"id": ["a"]},
                [("v", "[[1, 2, 3]]"), ("has3", "3 IN [[1, 2, 3]][0]")],
                [{"v": [[1, 2, 3]], "has3": True}],
                id="preserves-nested-list-shape",
            ),
        ],
    )
    def test_row_pipeline_select_single_row_exact_records(self, nodes, items, expected_records):
        _assert_single_row_select_records(items, expected_records, nodes_df=pd.DataFrame(nodes))

    def test_row_pipeline_select_rejects_non_json_literal_shapes(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="unsupported token in row expression|unsupported row expression"):
            g.gfql([rows(), select([("bad", "[1, (2, 3)]")])])

        with pytest.raises(Exception, match="unsupported token in row expression|unsupported row expression"):
            g.gfql([rows(), select([("bad", "{1: 2}")])])

    def test_row_pipeline_select_toboolean_tostring_coalesce(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "s": ["true", "false", None]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([
                ("b", "toBoolean(s)"),
                ("s2", "toString(coalesce(s, 'x'))"),
            ]),
            order_by([("s2", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"b": False, "s2": "false"},
            {"b": True, "s2": "true"},
            {"b": None, "s2": "x"},
        ]

    def test_row_pipeline_select_alias_property_expression(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "a": [True, True], "num": [2, 5]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([("num_proj", "a.num"), ("num_plus_1", "a.num + 1")]),
            order_by([("num_proj", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"num_proj": 2, "num_plus_1": 3},
            {"num_proj": 5, "num_plus_1": 6},
        ]

    def test_row_pipeline_select_boolean_null_expression(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "score": [2, None, 7]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([
                ("id", "id"),
                ("is_null", "score IS NULL"),
                ("is_big", "score > 5"),
                ("flag", "(score IS NULL) OR (score > 5)"),
            ]),
            order_by([("id", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "is_null": False, "is_big": False, "flag": False},
            {"id": "b", "is_null": True, "is_big": None, "flag": True},
            {"id": "c", "is_null": False, "is_big": True, "flag": True},
        ]

    def test_row_pipeline_select_quantifier_empty_and_null(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([
                ("any_empty", "any(x IN [] WHERE x = 1)"),
                ("all_empty", "all(x IN [] WHERE x = 1)"),
                ("none_empty", "none(x IN [] WHERE x = 1)"),
                ("single_empty", "single(x IN [] WHERE x = 1)"),
                ("any_null", "any(x IN [null] WHERE x = 1)"),
                ("all_null", "all(x IN [null] WHERE x = 1)"),
                ("none_null", "none(x IN [null] WHERE x = 1)"),
                ("single_null", "single(x IN [null] WHERE x = 1)"),
            ]),
        ])
        row = result._nodes.to_dict(orient="records")[0]

        assert row["any_empty"] is False
        assert row["all_empty"] is True
        assert row["none_empty"] is True
        assert row["single_empty"] is False
        assert pd.isna(row["any_null"])
        assert pd.isna(row["all_null"])
        assert pd.isna(row["none_null"])
        assert pd.isna(row["single_null"])

    def test_row_pipeline_select_quantifier_on_list_columns(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "vals": [[1, 2, 3], [2], []],
            "maps": [[{"a": 1}, {"a": 2}], [{"a": 0}], []],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([
                ("id", "id"),
                ("has_2", "any(x IN vals WHERE x = 2)"),
                ("all_lt_4", "all(x IN vals WHERE x < 4)"),
                ("any_map_a_2", "any(x IN maps WHERE x.a = 2)"),
            ]),
            order_by([("id", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "has_2": True, "all_lt_4": True, "any_map_a_2": True},
            {"id": "b", "has_2": True, "all_lt_4": True, "any_map_a_2": False},
            {"id": "c", "has_2": False, "all_lt_4": True, "any_map_a_2": False},
        ]

    def test_row_pipeline_order_by_string_expression(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "score": [1, 3, 2]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            order_by([("score + 2", "desc")]),
            limit(2),
            select([("id", "id"), ("score", "score")]),
        ])
        assert result._nodes.to_dict(orient="records") == [
            {"id": "b", "score": 3},
            {"id": "c", "score": 2},
        ]

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

    def test_row_pipeline_order_by_cypher_list_value_semantics(self):
        nodes_df = pd.DataFrame({
            "id": list("abcdefgh"),
            "lists": [[], ["a"], ["a", 1], [1], [1, "a"], [1, None], [None, 1], [None, 2]],
        })
        g = _mk_graph(nodes_df)

        asc_result = g.gfql([
            rows(),
            select([("lists", "lists")]),
            order_by([("lists", "asc")]),
        ])
        assert asc_result._nodes["lists"].tolist() == [
            [],
            ["a"],
            ["a", 1],
            [1],
            [1, "a"],
            [1, None],
            [None, 1],
            [None, 2],
        ]

        desc_result = g.gfql([
            rows(),
            select([("lists", "lists")]),
            order_by([("lists", "desc")]),
        ])
        assert desc_result._nodes["lists"].tolist() == [
            [None, 2],
            [None, 1],
            [1, None],
            [1, "a"],
            [1],
            ["a", 1],
            ["a"],
            [],
        ]

    def test_row_pipeline_order_by_time_offsets_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": list("abcde"),
            "times": [
                "10:35-08:00",
                "12:31:14.645876123+01:00",
                "12:31:14.645876124+01:00",
                "12:35:15+05:00",
                "12:30:14.645876123+01:01",
            ],
        })
        g = _mk_graph(nodes_df)

        asc_result = g.gfql([
            rows(),
            select([("times", "times")]),
            order_by([("times", "asc")]),
            limit(3),
        ])
        assert asc_result._nodes["times"].tolist() == [
            "12:35:15+05:00",
            "12:30:14.645876123+01:01",
            "12:31:14.645876123+01:00",
        ]

        desc_result = g.gfql([
            rows(),
            select([("times", "times")]),
            order_by([("times", "desc")]),
            limit(3),
        ])
        assert desc_result._nodes["times"].tolist() == [
            "10:35-08:00",
            "12:31:14.645876124+01:00",
            "12:31:14.645876123+01:00",
        ]

    def test_row_pipeline_order_by_datetime_offsets_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": list("abcde"),
            "datetimes": [
                "1984-10-11T12:30:14.000000012+00:15",
                "1984-10-11T12:31:14.645876123+00:17",
                "0001-01-01T01:01:01.000000001-11:59",
                "9999-09-09T09:59:59.999999999+11:59",
                "1980-12-11T12:31:14-11:59",
            ],
        })
        g = _mk_graph(nodes_df)

        asc_result = g.gfql([
            rows(),
            select([("datetimes", "datetimes")]),
            order_by([("datetimes", "asc")]),
            limit(3),
        ])
        assert asc_result._nodes["datetimes"].tolist() == [
            "0001-01-01T01:01:01.000000001-11:59",
            "1980-12-11T12:31:14-11:59",
            "1984-10-11T12:31:14.645876123+00:17",
        ]

        desc_result = g.gfql([
            rows(),
            select([("datetimes", "datetimes")]),
            order_by([("datetimes", "desc")]),
            limit(3),
        ])
        assert desc_result._nodes["datetimes"].tolist() == [
            "9999-09-09T09:59:59.999999999+11:59",
            "1984-10-11T12:30:14.000000012+00:15",
            "1984-10-11T12:31:14.645876123+00:17",
        ]

    def test_row_pipeline_bad_source_alias_raises(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "v": [1, 2]})
        g = _mk_graph(nodes_df)

        with pytest.raises(Exception, match="requires node column|alias column not found"):
            g.gfql([rows(source="missing")])

    def test_row_pipeline_rows_edges_table_projection(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c"]})
        edges_df = pd.DataFrame({
            "s": ["a", "b", "a"],
            "d": ["b", "c", "c"],
            "weight": [1, 3, 2],
        })
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(table="edges"),
            select([("weight", "weight")]),
            order_by([("weight", "desc")]),
        ])

        assert list(result._nodes.columns) == ["weight"]
        assert result._nodes["weight"].tolist() == [3, 2, 1]

    def test_row_pipeline_select_allows_literal_expressions(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([("id", "id"), ("one", 1), ("txt", "id")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "one": 1, "txt": "a"},
            {"id": "b", "one": 1, "txt": "b"},
        ]

    def test_row_pipeline_select_case_when_expression(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "score": [1, 3, 2]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select(
                [
                    ("id", "id"),
                    ("bucket", "CASE WHEN score > 2 THEN 'hi' ELSE 'lo' END"),
                ]
            ),
            order_by([("id", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "bucket": "lo"},
            {"id": "b", "bucket": "hi"},
            {"id": "c", "bucket": "lo"},
        ]

    def test_row_pipeline_select_sequence_function_family(self):
        nodes_df = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "vals": [[1, 2, 3], [4], []],
                "txt": ["ab", "cd", ""],
                "score": [-2, 0, 3],
            }
        )
        g = _mk_graph(nodes_df)

        result = g.gfql(
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
            ]
        )

        records = result._nodes.to_dict(orient="records")
        assert records[:2] == [
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
        ]
        assert records[2]["id"] == "c"
        assert pd.isna(records[2]["h_vals"])
        assert records[2]["t_vals"] == []
        assert records[2]["r_vals"] == []
        assert pd.isna(records[2]["h_txt"])
        assert records[2]["t_txt"] == ""
        assert records[2]["r_txt"] == ""
        assert records[2]["sgn"] == 1

    def test_row_pipeline_select_nodes_relationships_passthrough(self):
        nodes_df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "path_nodes": [["n1", "n2"], ["n3"]],
                "path_rels": [["r1"], []],
            }
        )
        g = _mk_graph(nodes_df)

        result = g.gfql(
            [
                rows(),
                select(
                    [
                        ("id", "id"),
                        ("ns", "nodes(path_nodes)"),
                        ("rs", "relationships(path_rels)"),
                    ]
                ),
                order_by([("id", "asc")]),
            ]
        )

        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "ns": ["n1", "n2"], "rs": ["r1"]},
            {"id": "b", "ns": ["n3"], "rs": []},
        ]

    def test_row_pipeline_select_in_operator_and_list_scalar_concat(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "vals": [[1, 2], [3]]})
        g = _mk_graph(nodes_df)

        result = g.gfql(
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
            ]
        )

        assert result._nodes.to_dict(orient="records") == [
            {
                "id": "a",
                "has2": True,
                "has5": False,
                "append9": [1, 2, 9],
                "prepend9": [9, 1, 2],
            },
            {
                "id": "b",
                "has2": False,
                "has5": False,
                "append9": [3, 9],
                "prepend9": [9, 3],
            },
        ]

    def test_row_pipeline_select_string_predicate_ops(self):
        nodes_df = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "txt": ["abcdef", "xxabyy", None],
            }
        )
        g = _mk_graph(nodes_df)

        result = g.gfql(
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
            ]
        )

        records = result._nodes.to_dict(orient="records")
        assert records[0] == {"id": "a", "has_ab": True, "starts_ab": True, "ends_ef": True}
        assert records[1] == {"id": "b", "has_ab": True, "starts_ab": False, "ends_ef": False}
        assert records[2]["id"] == "c"
        assert pd.isna(records[2]["has_ab"])
        assert pd.isna(records[2]["starts_ab"])
        assert pd.isna(records[2]["ends_ef"])

    def test_row_pipeline_select_string_predicate_keyword_in_literal(self):
        nodes_df = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "txt": ["abc STARTS WITH xyz", "abc xyz", None],
            }
        )
        g = _mk_graph(nodes_df)

        result = g.gfql(
            [
                rows(),
                select([("id", "id"), ("hit", "txt CONTAINS 'STARTS WITH'")]),
                order_by([("id", "asc")]),
            ]
        )

        records = result._nodes.to_dict(orient="records")
        assert records[0] == {"id": "a", "hit": True}
        assert records[1] == {"id": "b", "hit": False}
        assert records[2]["id"] == "c"
        assert pd.isna(records[2]["hit"])

    def test_row_pipeline_select_string_predicates_null_rhs(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "txt": ["abcdef", None]})
        g = _mk_graph(nodes_df)

        result = g.gfql(
            [
                rows(),
                select(
                    [
                        ("contains_null", "txt CONTAINS null"),
                        ("starts_null", "txt STARTS WITH null"),
                        ("ends_null", "txt ENDS WITH null"),
                    ]
                ),
            ]
        )

        for col in ["contains_null", "starts_null", "ends_null"]:
            assert all(pd.isna(v) for v in result._nodes[col].tolist())

    def test_row_pipeline_select_slice_null_bound_returns_null(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "txt": ["abcdef", "ghij"]})
        g = _mk_graph(nodes_df)

        result = g.gfql(
            [
                rows(),
                select(
                    [
                        ("lhs_null", "txt[null..2]"),
                        ("rhs_null", "txt[1..null]"),
                    ]
                ),
            ]
        )

        assert result._nodes["lhs_null"].tolist() == [None, None]
        assert result._nodes["rhs_null"].tolist() == [None, None]

    def test_row_pipeline_select_reverse_invalid_input_raises(self):
        nodes_df = pd.DataFrame({"id": ["a"], "x": [1]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="reverse\\(\\) requires list/string input"):
            g.gfql([rows(), select([("bad", "reverse(x)")])])

    def test_row_pipeline_select_broadcasts_list_map_literals(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            select([("id", "id"), ("lst", [1, 2]), ("mp", {"k": "v"})]),
            order_by([("id", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "lst": [1, 2], "mp": {"k": "v"}},
            {"id": "b", "lst": [1, 2], "mp": {"k": "v"}},
        ]

    def test_row_pipeline_invalid_rows_table_rejected(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        g = _mk_graph(nodes_df)

        with pytest.raises(Exception, match="table"):
            g.gfql([rows("bad_table")])

    def test_row_pipeline_select_missing_column_raises(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        g = _mk_graph(nodes_df)

        with pytest.raises(Exception, match="unsupported token in row expression|unsupported row expression"):
            g.gfql([rows(), select([("x", "missing_col")])])

    def test_row_pipeline_order_by_missing_column_raises(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        g = _mk_graph(nodes_df)

        with pytest.raises(Exception, match="order_by column not found|unsupported token in row expression|unsupported row expression"):
            g.gfql([rows(), order_by([("missing_col", "asc")])])

    @pytest.mark.parametrize("value", [-1, True, "1.5", "bad"])
    def test_row_pipeline_skip_invalid_values_rejected(self, value):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        g = _mk_graph(nodes_df)

        with pytest.raises(Exception, match="Invalid type for parameter|non-negative integer|non-negative"):
            g.gfql([rows(), skip(value)])

    @pytest.mark.parametrize("value", [-1, True, "1.5", "bad"])
    def test_row_pipeline_limit_invalid_values_rejected(self, value):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        g = _mk_graph(nodes_df)

        with pytest.raises(Exception, match="Invalid type for parameter|non-negative integer|non-negative"):
            g.gfql([rows(), limit(value)])

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

    def test_row_pipeline_rows_validation(self):
        params = validate_call_params("rows", {})
        assert params == {}

        params = validate_call_params("rows", {"table": "edges", "source": "rel"})
        assert params == {"table": "edges", "source": "rel"}

        self._assert_e201("rows", {"table": "bad"})

    def test_row_pipeline_select_validation(self):
        params = validate_call_params("select", {"items": [("name", "name"), ("const", 1)]})
        assert params == {"items": [("name", "name"), ("const", 1)]}
        params = validate_call_params("select", {"items": ["name", ("const", 1)]})
        assert params == {"items": ["name", ("const", 1)]}
        params = validate_call_params("return_", {"items": [("name", "name")]})
        assert params == {"items": [("name", "name")]}
        params = validate_call_params("with_", {"items": ["name"]})
        assert params == {"items": ["name"]}

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
        params = validate_call_params("with_", {"items": [("name", "name")]})
        assert params == {"items": [("name", "name")]}
        params = validate_call_params("where_rows", {"filter_dict": {"name": "alice"}})
        assert params == {"filter_dict": {"name": "alice"}}
        valid_exprs = [
            "score > 1 AND name != 'bob'",
            "name = 'rand()'",
            'name = "rand()"',
            "txt CONTAINS 5",
            "txt CONTAINS null",
            "txt CONTAINS (5)",
            "txt STARTS WITH (5)",
            "txt ENDS WITH (5)",
            "CASE WHEN score > 1 THEN true ELSE false END",
            "any(x IN vals WHERE x = 2)",
            "all(x IN vals WHERE x > 1)",
            "none(x IN vals WHERE x < 0)",
            "single(x IN vals WHERE x = 2)",
            "score > 1 AND CASE WHEN id = 'a' THEN true ELSE false END",
            "size([x IN vals WHERE x > 1]) > 0",
            "size([x IN vals WHERE x > 1 | x]) > 0",
        ]
        for expr in valid_exprs:
            assert validate_call_params("where_rows", {"expr": expr}) == {"expr": expr}
        pred = gt(1)
        params = validate_call_params("where_rows", {"filter_dict": {"score": pred}})
        assert params == {"filter_dict": {"score": pred}}
        params = validate_call_params(
            "where_rows",
            {"filter_dict": {"score": pred}, "expr": "score > 1"},
        )
        assert params == {"filter_dict": {"score": pred}, "expr": "score > 1"}

        bad_filter_dict_inputs = [{"filter_dict": "bad"}, {"filter_dict": {1: "x"}}]
        for bad_params in bad_filter_dict_inputs:
            self._assert_e201("where_rows", bad_params)

        invalid_exprs = [
            "rand() > 0.1",
            "txt CONTAINS rhs",
            "txt STARTS WITH rhs",
            "txt ENDS WITH rhs",
            "txt CONTAINS [1,2]",
            "txt CONTAINS {k: 'v'}",
            "txt CONTAINS (rhs)",
            "any(x IN vals WHERE x = 2",
            "any(x vals WHERE x = 2)",
            "any(x IN vals | WHERE x = 2)",
            "CASE WHEN score > 1 THEN true END",
            "score > 1 AND CASE WHEN id = 'a' THEN true END",
            "size([x vals WHERE x > 1]) > 0",
            "size([x IN vals WHERE ]) > 0",
            "size([x IN vals WHERE x > 1 | ]) > 0",
            "size([x IN vals | ]) > 0",
            "size([x IN vals | WHERE x > 1]) > 0",
            "size([x IN vals | WHERE x > 1 | x + 1]) > 0",
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
            "id = = 'a'",
            "score > = 2",
            "score < = 2",
            "score ! = 2",
            "score < > 2",
            "score =< 2",
            "score => 2",
            "coalesce(id, 'x') y",
            "size(vals) z",
            "any(x IN vals WHERE x = 2) OR OR id = 'a'",
            "size() > 0",
            "size( ) > 0",
            "size(,) > 0",
            "coalesce() = id",
            "toString() = 'x'",
            "toBoolean()",
            "abs() = 1",
            "sign() = 1",
            "head() = 1",
            "tail() = []",
            "reverse() = []",
            "size(( )) > 0",
            "coalesce(( )) = id",
            "coalesce(id score) = id",
            "size(vals,) > 0",
            "size(,vals) > 0",
        ]
        for expr in invalid_exprs:
            self._assert_e201("where_rows", {"expr": expr})

    def test_row_pipeline_where_rows_parser_authority(self, monkeypatch):
        monkeypatch.setattr(call_safelist, "_where_rows_expr_parser_parse_ok", lambda _expr: False)
        monkeypatch.setattr(
            call_safelist,
            "_where_rows_expr_parser_fn",
            lambda: (lambda _expr: object(), lambda _node: [], lambda _node: set()),
        )

        self._assert_e201("where_rows", {"expr": "score > 1"})

    def test_row_pipeline_where_rows_strict_parser_authority_when_available(self, monkeypatch):
        def fake_parse(_expr):
            return expr_parser.Identifier("score")

        def fake_capabilities(_node):
            return []

        monkeypatch.setattr(
            call_safelist,
            "_where_rows_expr_parser_fn",
            lambda: (fake_parse, fake_capabilities, lambda _node: {"score"}),
        )
        # Old lexical checks reject '==', but strict parser authority should control when parser is available.
        params = validate_call_params("where_rows", {"expr": "score == 1"})
        assert params == {"expr": "score == 1"}

    def test_row_pipeline_where_rows_strict_parser_authority_rejects_capability_fail(self, monkeypatch):
        def fake_parse(_expr):
            return expr_parser.Identifier("score")

        def fake_capabilities(_node):
            return ["unsupported"]

        monkeypatch.setattr(
            call_safelist,
            "_where_rows_expr_parser_fn",
            lambda: (fake_parse, fake_capabilities, lambda _node: {"score"}),
        )
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params("where_rows", {"expr": "score > 1"})
        assert exc_info.value.code == ErrorCode.E201

    def test_row_pipeline_where_rows_required_cols_from_parser(self, monkeypatch):
        def fake_parse(_expr):
            return "node"

        def fake_capabilities(_node):
            return []

        def fake_collect(_node):
            return {"n.age", "score", "vals"}

        monkeypatch.setattr(
            call_safelist,
            "_where_rows_expr_parser_fn",
            lambda: (fake_parse, fake_capabilities, fake_collect),
        )
        cols = call_safelist._where_rows_requires_node_cols(
            {"filter_dict": {"id": "a"}, "expr": "ignored"}
        )
        assert cols == ["id", "n", "score", "vals"]

    def test_row_pipeline_where_rows_required_cols_parser_required(self, monkeypatch):
        monkeypatch.setattr(call_safelist, "_where_rows_expr_parser_fn", lambda: None)
        cols = call_safelist._where_rows_requires_node_cols(
            {"expr": "any(x IN vals WHERE x > threshold)"}
        )
        assert cols == []

    def test_row_pipeline_where_rows_validator_rejects_without_parser(self, monkeypatch):
        monkeypatch.setattr(call_safelist, "_where_rows_expr_parser_fn", lambda: None)
        self._assert_e201("where_rows", {"expr": "score > 1"})

    def test_row_pipeline_runtime_parser_authority(self, monkeypatch):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "score": [1, 2, 3],
        })
        g = _mk_graph(nodes_df)

        # Parser unavailable -> runtime fails fast.
        monkeypatch.setattr(row_pipeline_mixin, "_gfql_expr_runtime_parser_bundle", lambda: None)
        with pytest.raises(GFQLTypeError) as exc_info:
            g.gfql([rows(), where_rows(expr="score > 1"), return_([("id", "id")])])
        assert exc_info.value.code == ErrorCode.E303
        assert "parser backend unavailable" in exc_info.value.message

        # Parser available + parse failure -> runtime failfast.
        def fake_parse(_expr):
            raise ValueError("bad parse")

        def fake_capabilities(_node):
            return []

        monkeypatch.setattr(
            row_pipeline_mixin,
            "_gfql_expr_runtime_parser_bundle",
            lambda: (fake_parse, fake_capabilities, expr_parser),
        )
        with pytest.raises(GFQLTypeError) as exc_info:
            g.gfql([rows(), where_rows(expr="score > 1"), return_([("id", "id")])])
        assert exc_info.value.code == ErrorCode.E303
        assert "parser validation failed" in exc_info.value.message

    def test_row_pipeline_eval_expr_ast_subset_parity(self, monkeypatch):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "score": [1, 2, 3],
            "name": ["a", "bb", "ccc"],
        })
        g = _mk_graph(nodes_df)
        table_df = g._nodes

        cases = [
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
        ]

        for ast_node, expr in cases:
            ctx = row_pipeline_mixin._RowPipelineAdapter(g)
            ok, ast_out = ctx._gfql_eval_expr_ast(table_df, ast_node)
            assert ok, expr
            legacy_out = ctx._gfql_eval_string_expr(table_df, expr)
            assert _normalize_expr_eval_output(ast_out) == _normalize_expr_eval_output(legacy_out)

    def test_row_pipeline_eval_expr_ast_advanced_parity(self, monkeypatch):
        nodes_df = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "score": [1, 3, 2],
                "vals": [[1], [1, 2], [2, 3]],
            }
        )
        g = _mk_graph(nodes_df)
        table_df = g._nodes

        cases = [
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
        ]

        for ast_node, expr in cases:
            ctx = row_pipeline_mixin._RowPipelineAdapter(g)
            ok, ast_out = ctx._gfql_eval_expr_ast(table_df, ast_node)
            assert ok, expr
            legacy_out = ctx._gfql_eval_string_expr(table_df, expr)
            assert _normalize_expr_eval_output(ast_out) == _normalize_expr_eval_output(legacy_out)

    def test_row_pipeline_runtime_ast_path_with_parser_bundle(self, monkeypatch):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "score": [1, 2, 3],
        })
        g = _mk_graph(nodes_df)

        def fake_parse(_expr):
            return expr_parser.BinaryOp(">", expr_parser.Identifier("score"), expr_parser.Literal(1))

        def fake_capabilities(_node):
            return []

        monkeypatch.setattr(
            row_pipeline_mixin,
            "_gfql_expr_runtime_parser_bundle",
            lambda: (fake_parse, fake_capabilities, expr_parser),
        )
        result = g.gfql([rows(), where_rows(expr="score > 1")])
        assert result._nodes[["id", "score"]].reset_index(drop=True).to_dict(orient="records") == [
            {"id": "b", "score": 2},
            {"id": "c", "score": 3},
        ]

    def test_row_pipeline_runtime_ast_unsupported(self, monkeypatch):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "score": [1, 2, 3],
        })
        g = _mk_graph(nodes_df)

        def fake_parse(_expr):
            return expr_parser.FunctionCall("unknown_fn", (expr_parser.Identifier("score"),))

        def fake_capabilities(_node):
            return []

        monkeypatch.setattr(
            row_pipeline_mixin,
            "_gfql_expr_runtime_parser_bundle",
            lambda: (fake_parse, fake_capabilities, expr_parser),
        )
        with pytest.raises(GFQLTypeError) as exc_info:
            g.gfql([rows(), where_rows(expr="score > 1")])
        assert exc_info.value.code == ErrorCode.E303
        assert "AST evaluator unsupported" in exc_info.value.message

    def test_row_pipeline_runtime_ast_type_error_normalized(self, monkeypatch):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "name": ["x", "y", "z"],
        })
        g = _mk_graph(nodes_df)

        def fake_parse(_expr):
            return expr_parser.BinaryOp("-", expr_parser.Identifier("name"), expr_parser.Literal("x"))

        def fake_capabilities(_node):
            return []

        monkeypatch.setattr(
            row_pipeline_mixin,
            "_gfql_expr_runtime_parser_bundle",
            lambda: (fake_parse, fake_capabilities, expr_parser),
        )
        with pytest.raises(GFQLTypeError) as exc_info:
            g.gfql([rows(), where_rows(expr="name - 'x'")])
        assert exc_info.value.code == ErrorCode.E303
        assert "AST evaluator unsupported" in exc_info.value.message

    def test_row_pipeline_order_by_validation(self):
        params = validate_call_params("order_by", {"keys": [("name", "asc"), ("score", "desc")]})
        assert params == {"keys": [("name", "asc"), ("score", "desc")]}
        params = validate_call_params("order_by", {"keys": [("count(*)", "asc"), ("max(n.age)", "desc")]})
        assert params == {"keys": [("count(*)", "asc"), ("max(n.age)", "desc")]}

        for bad_keys in [
            None,
            "name",
            [("a",)],
            [("a", "asc", "x")],
            [1],
            [(1, "asc")],
            [("a", "up")],
            [("[1, 2]", "asc")],
            [("unknown_fn(score)", "asc")],
        ]:
            self._assert_e201("order_by", {"keys": bad_keys})

    def test_row_pipeline_order_by_aggregate_alias_columns(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "division": ["x", "x", "y"],
            "age": [3, 7, 4],
        })
        g = _mk_graph(nodes_df)

        result = g.gfql([
            rows(),
            group_by(["division"], [("count(*)", "count"), ("max(n.age)", "max", "age")]),
            order_by([("count(*)", "asc"), ("max(n.age)", "desc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"division": "y", "count(*)": 1, "max(n.age)": 4},
            {"division": "x", "count(*)": 2, "max(n.age)": 7},
        ]

    @pytest.mark.parametrize("function", ["skip", "limit"])
    def test_row_pipeline_skip_limit_validation(self, function):
        for value in [0, 2, 2.0, "3"]:
            params = validate_call_params(function, {"value": value})
            assert params == {"value": value}

        for bad_value in [True, -1, -1.0, "-1", "1.5", "abc"]:
            self._assert_e201(function, {"value": bad_value})

    def test_row_pipeline_distinct_validation(self):
        params = validate_call_params("distinct", {})
        assert params == {}

        self._assert_e303("distinct", {"extra": True})

    def test_row_pipeline_unwind_group_by_validation(self):
        params = validate_call_params("unwind", {"expr": "vals", "as_": "v"})
        assert params == {"expr": "vals", "as_": "v"}
        params = validate_call_params("unwind", {"expr": [1, 2, 3], "as_": "v"})
        assert params == {"expr": [1, 2, 3], "as_": "v"}
        params = validate_call_params(
            "group_by",
            {"keys": ["grp"], "aggregations": [("cnt", "count"), ("sum_v", "sum", "v")]},
        )
        assert params == {
            "keys": ["grp"],
            "aggregations": [("cnt", "count"), ("sum_v", "sum", "v")],
        }
        params = validate_call_params(
            "group_by",
            {"keys": ["grp"], "aggregations": [("vals", "collect", "v")]},
        )
        assert params == {
            "keys": ["grp"],
            "aggregations": [("vals", "collect", "v")],
        }
        params = validate_call_params(
            "group_by",
            {"keys": ["grp"], "aggregations": [("vals", "collect", "v + 1")]},
        )
        assert params == {
            "keys": ["grp"],
            "aggregations": [("vals", "collect", "v + 1")],
        }

        self._assert_e201("unwind", {"expr": 1})
        self._assert_e201("unwind", {"expr": "vals", "as_": ""})
        self._assert_e201("group_by", {"keys": ["grp"], "aggregations": ["bad"]})
        self._assert_e201("group_by", {"keys": ["grp"], "aggregations": [("x", "median", "score")]})
        self._assert_e201("group_by", {"keys": [], "aggregations": [("x", "count")]})
