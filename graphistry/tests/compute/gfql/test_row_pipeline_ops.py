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

        with pytest.raises(Exception, match="select expression column not found"):
            g.gfql([rows(), select([("x", "missing_col")])])

    def test_row_pipeline_order_by_missing_column_raises(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="order_by column not found"):
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

        for bad_items in [None, "name", [("a",)], [("a", "b", "c")], [1]]:
            with pytest.raises(GFQLTypeError) as exc_info:
                validate_call_params("select", {"items": bad_items})
            assert exc_info.value.code == ErrorCode.E201

    def test_row_pipeline_with_where_rows_validation(self):
        params = validate_call_params("with_", {"items": [("name", "name")]})
        assert params == {"items": [("name", "name")]}
        params = validate_call_params("where_rows", {"filter_dict": {"name": "alice"}})
        assert params == {"filter_dict": {"name": "alice"}}

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params("where_rows", {"filter_dict": "bad"})
        assert exc_info.value.code == ErrorCode.E201

    def test_row_pipeline_order_by_validation(self):
        params = validate_call_params("order_by", {"keys": [("name", "asc"), ("score", "desc")]})
        assert params == {"keys": [("name", "asc"), ("score", "desc")]}

        for bad_keys in [None, "name", [("a",)], [("a", "asc", "x")], [1]]:
            with pytest.raises(GFQLTypeError) as exc_info:
                validate_call_params("order_by", {"keys": bad_keys})
            assert exc_info.value.code == ErrorCode.E201

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
        params = validate_call_params(
            "group_by",
            {"keys": ["grp"], "aggregations": [("cnt", "count"), ("sum_v", "sum", "v")]},
        )
        assert params == {
            "keys": ["grp"],
            "aggregations": [("cnt", "count"), ("sum_v", "sum", "v")],
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params("unwind", {"expr": 1})
        assert exc_info.value.code == ErrorCode.E201

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params("group_by", {"keys": ["grp"], "aggregations": ["bad"]})
        assert exc_info.value.code == ErrorCode.E201
