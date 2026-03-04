import pandas as pd
import pytest

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


class TestRowPipelineASTPrimitives:
    def test_row_pipeline_primitives_build_ast_calls(self):
        row_step = rows("nodes", source="a")
        assert isinstance(row_step, ASTCall)
        assert row_step.function == "rows"
        assert row_step.params == {"table": "nodes", "source": "a"}

        select_step = select([("name", "name"), ("age", "age")])
        assert isinstance(select_step, ASTCall)
        assert select_step.function == "select"
        assert select_step.params == {"items": [("name", "name"), ("age", "age")]}

        with_step = with_([("name", "name")])
        assert isinstance(with_step, ASTCall)
        assert with_step.function == "with_"
        assert with_step.params == {"items": [("name", "name")]}

        return_step = return_([("name", "name")])
        assert isinstance(return_step, ASTCall)
        assert return_step.function == "select"
        assert return_step.params == {"items": [("name", "name")]}

        where_step = where_rows({"name": "alice"})
        assert isinstance(where_step, ASTCall)
        assert where_step.function == "where_rows"
        assert where_step.params == {"filter_dict": {"name": "alice"}}

        order_step = order_by([("name", "asc"), ("age", "desc")])
        assert isinstance(order_step, ASTCall)
        assert order_step.function == "order_by"
        assert order_step.params == {"keys": [("name", "asc"), ("age", "desc")]}

        skip_step = skip(3)
        assert isinstance(skip_step, ASTCall)
        assert skip_step.function == "skip"
        assert skip_step.params == {"value": 3}

        limit_step = limit(10)
        assert isinstance(limit_step, ASTCall)
        assert limit_step.function == "limit"
        assert limit_step.params == {"value": 10}

        distinct_step = distinct()
        assert isinstance(distinct_step, ASTCall)
        assert distinct_step.function == "distinct"
        assert distinct_step.params == {}

        unwind_step = unwind("vals", as_="v")
        assert isinstance(unwind_step, ASTCall)
        assert unwind_step.function == "unwind"
        assert unwind_step.params == {"expr": "vals", "as_": "v"}

        group_step = group_by(["grp"], [("cnt", "count"), ("sum_score", "sum", "score")])
        assert isinstance(group_step, ASTCall)
        assert group_step.function == "group_by"
        assert group_step.params == {
            "keys": ["grp"],
            "aggregations": [("cnt", "count"), ("sum_score", "sum", "score")],
        }


class TestRowPipelineExecution:
    def test_row_pipeline_exec_projection_sort_page_distinct(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c", "d"],
            "name": ["n2", "n1", "n2", "n3"],
            "score": [2, 3, 2, 1],
        })
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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

    def test_row_pipeline_distinct_unhashable_cells(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "vals": [[1, 2], [1, 2], [3]],
            "meta": [{"k": "v"}, {"k": "v"}, {"k": "z"}],
        })
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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

    def test_row_pipeline_unwind_column_vectorized(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b"],
            "vals": [[1, 2], [3]],
        })
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([("score", "score"), ("score_plus_2", "score + 2"), ("neg_score", "-1 * score")]),
            order_by([("score", "asc")]),
        ])
        assert result._nodes.to_dict(orient="records") == [
            {"score": 2, "score_plus_2": 4, "neg_score": -2},
            {"score": 5, "score_plus_2": 7, "neg_score": -5},
        ]

    def test_row_pipeline_select_unary_neg_parenthesized_expression(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([("v", "-(3 + 2)")]),
        ])

        assert result._nodes.to_dict(orient="records") == [{"v": -5}]

    def test_row_pipeline_select_list_literal_comparison(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([("eq", "[1, 2] = [1, 2]"), ("neq", "[1, 2] = [1]")]),
        ])

        assert result._nodes.to_dict(orient="records") == [{"eq": True, "neq": False}]

    def test_row_pipeline_select_dynamic_list_expression_vectorized(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "score": [2, 5]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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

    def test_row_pipeline_select_nested_json_like_literal(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([("eq_nested", "[1, {'k': [2, null]}] = [1, {'k': [2, null]}]")]),
        ])

        assert result._nodes.to_dict(orient="records") == [{"eq_nested": True}]

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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

    def test_row_pipeline_select_quantifier_literals(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([
                ("any_hit", "any(x IN [1, 2, 3] WHERE x = 2)"),
                ("all_lt_5", "all(x IN [1, 2, 3] WHERE x < 5)"),
                ("none_gt_3", "none(x IN [1, 2, 3] WHERE x > 3)"),
                ("single_even", "single(x IN [1, 2, 3] WHERE x % 2 = 0)"),
                ("size_list", "size([1, 2, 3])"),
                ("abs_num", "abs(-7)"),
            ]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {
                "any_hit": True,
                "all_lt_5": True,
                "none_gt_3": True,
                "single_even": True,
                "size_list": 3,
                "abs_num": 7,
            }
        ]

    def test_row_pipeline_select_nested_quantifier_list_literal(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([
                ("nested_none", "none(x IN [['abc'], ['abc', 'def']] WHERE all(y IN x WHERE y = 'def'))"),
            ]),
        ])

        assert result._nodes.to_dict(orient="records") == [{"nested_none": True}]

    def test_row_pipeline_select_quantifier_over_map_list_literal(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([
                ("none_map_hit", "none(x IN [{a: 2, b: 5}, {a: 4}] WHERE x.a = 2)"),
                ("any_map_hit", "any(x IN [{a: 2, b: 5}, {a: 4}] WHERE x.a = 2)"),
                ("single_map_hit", "single(x IN [{a: 2, b: 5}, {a: 4}] WHERE x.a = 2)"),
            ]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"none_map_hit": False, "any_map_hit": True, "single_map_hit": True}
        ]

    def test_row_pipeline_select_list_comprehension_with_map_literal(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([
                ("picked", "[x IN [{a: 2, b: 5}, {a: 4}] WHERE x.a = 2 | x.a]"),
                ("picked_count", "size([x IN [{a: 2, b: 5}, {a: 4}] WHERE x.a = 2 | x])"),
            ]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"picked": [2], "picked_count": 1}
        ]

    def test_row_pipeline_select_quantifier_composed_expressions(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([
                ("eq_negated_any", "none(x IN [1, 2, 3] WHERE x = 2) = (NOT any(x IN [1, 2, 3] WHERE x = 2))"),
                ("all_and_any", "all(x IN [1, 2, 3] WHERE x < 5) AND any(x IN [1, 2, 3] WHERE x = 2)"),
            ]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"eq_negated_any": True, "all_and_any": True}
        ]

    def test_row_pipeline_select_nested_quantifier_outer_var_reference(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([
                ("none_outer_lt", "none(x IN [1, 2, 3] WHERE all(y IN [1, 2, 3] WHERE x < y))"),
            ]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"none_outer_lt": True}
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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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

    def test_row_pipeline_select_slice_variants(self):
        nodes_df = pd.DataFrame({"id": ["a"], "txt": ["abcdef"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql(
            [
                rows(),
                select(
                    [
                        ("mid", "txt[1..3]"),
                        ("left", "txt[..2]"),
                        ("right", "txt[2..]"),
                        ("empty", "txt[0..0]"),
                    ]
                ),
            ]
        )

        assert result._nodes.to_dict(orient="records") == [
            {
                "mid": "bc",
                "left": "ab",
                "right": "cdef",
                "empty": "",
            }
        ]

    def test_row_pipeline_select_slice_null_bound_returns_null(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "txt": ["abcdef", "ghij"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([("id", "id"), ("lst", [1, 2]), ("mp", {"k": "v"})]),
            order_by([("id", "asc")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "lst": [1, 2], "mp": {"k": "v"}},
            {"id": "b", "lst": [1, 2], "mp": {"k": "v"}},
        ]

    def test_row_pipeline_select_preserves_nested_list_literal_shape(self):
        nodes_df = pd.DataFrame({"id": ["a"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["a"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([rows(), select([("v", "[[1, 2, 3]]"), ("has3", "3 IN [[1, 2, 3]][0]")])])

        records = result._nodes.to_dict(orient="records")
        assert records == [{"v": [[1, 2, 3]], "has3": True}]

    def test_row_pipeline_invalid_rows_table_rejected(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="table"):
            g.gfql([rows("bad_table")])

    def test_row_pipeline_select_missing_column_raises(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="unsupported token in row expression|unsupported row expression"):
            g.gfql([rows(), select([("x", "missing_col")])])

    def test_row_pipeline_order_by_missing_column_raises(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="order_by column not found|unsupported token in row expression|unsupported row expression"):
            g.gfql([rows(), order_by([("missing_col", "asc")])])

    @pytest.mark.parametrize("value", [-1, True, "1.5", "bad"])
    def test_row_pipeline_skip_invalid_values_rejected(self, value):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="Invalid type for parameter|non-negative integer|non-negative"):
            g.gfql([rows(), skip(value)])

    @pytest.mark.parametrize("value", [-1, True, "1.5", "bad"])
    def test_row_pipeline_limit_invalid_values_rejected(self, value):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
    def test_row_pipeline_rows_validation(self):
        params = validate_call_params("rows", {})
        assert params == {}

        params = validate_call_params("rows", {"table": "edges", "source": "rel"})
        assert params == {"table": "edges", "source": "rel"}

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params("rows", {"table": "bad"})
        assert exc_info.value.code == ErrorCode.E201

    def test_row_pipeline_select_validation(self):
        params = validate_call_params("select", {"items": [("name", "name"), ("const", 1)]})
        assert params == {"items": [("name", "name"), ("const", 1)]}

        for bad_items in [None, "name", [("a",)], [("a", "b", "c")], [1], [(1, "name")], [("", "name")]]:
            with pytest.raises(GFQLTypeError) as exc_info:
                validate_call_params("select", {"items": bad_items})
            assert exc_info.value.code == ErrorCode.E201

    def test_row_pipeline_with_where_rows_validation(self):
        params = validate_call_params("with_", {"items": [("name", "name")]})
        assert params == {"items": [("name", "name")]}
        params = validate_call_params("where_rows", {"filter_dict": {"name": "alice"}})
        assert params == {"filter_dict": {"name": "alice"}}
        pred = gt(1)
        params = validate_call_params("where_rows", {"filter_dict": {"score": pred}})
        assert params == {"filter_dict": {"score": pred}}

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params("where_rows", {"filter_dict": "bad"})
        assert exc_info.value.code == ErrorCode.E201
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params("where_rows", {"filter_dict": {1: "x"}})
        assert exc_info.value.code == ErrorCode.E201

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
            with pytest.raises(GFQLTypeError) as exc_info:
                validate_call_params("order_by", {"keys": bad_keys})
            assert exc_info.value.code == ErrorCode.E201

    def test_row_pipeline_order_by_aggregate_alias_columns(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "division": ["x", "x", "y"],
            "age": [3, 7, 4],
        })
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

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
            with pytest.raises(GFQLTypeError) as exc_info:
                validate_call_params(function, {"value": bad_value})
            assert exc_info.value.code == ErrorCode.E201

    def test_row_pipeline_distinct_validation(self):
        params = validate_call_params("distinct", {})
        assert params == {}

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params("distinct", {"extra": True})
        assert exc_info.value.code == ErrorCode.E303

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

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params("unwind", {"expr": 1})
        assert exc_info.value.code == ErrorCode.E201
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params("unwind", {"expr": "vals", "as_": ""})
        assert exc_info.value.code == ErrorCode.E201

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params("group_by", {"keys": ["grp"], "aggregations": ["bad"]})
        assert exc_info.value.code == ErrorCode.E201
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params(
                "group_by",
                {"keys": ["grp"], "aggregations": [("x", "median", "score")]},
            )
        assert exc_info.value.code == ErrorCode.E201
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params(
                "group_by",
                {"keys": [], "aggregations": [("x", "count")]},
            )
        assert exc_info.value.code == ErrorCode.E201
