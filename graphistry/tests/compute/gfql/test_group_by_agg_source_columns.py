"""group_by aggregation sources that name an existing column validate as that column.

Both engines read an aggregation source that is an existing column as the column itself: pandas
checks the column before parsing an expression, and polars accepts only columns. The validator
parsed every source as an expression, splitting ``analysis.cohorts.Activist`` at the dot and
``x.Pro-Vax`` at the minus, so uploaded column names the executors would serve were rejected as
missing columns ``analysis`` and ``Vax``.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import ASTCall, drop_cols, e_forward, group_by, n, rows, select, with_
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError

try:
    import polars  # noqa: F401
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

ENGINES = [
    "pandas",
    pytest.param("cudf", marks=pytest.mark.skipif(
        os.environ.get("TEST_CUDF") != "1", reason="requires TEST_CUDF=1 and a CUDA device")),
    pytest.param("polars-gpu", marks=pytest.mark.skipif(
        os.environ.get("TEST_POLARS_GPU") != "1", reason="requires TEST_POLARS_GPU=1 and cudf-polars")),
    pytest.param("polars", marks=pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")),
]

COLUMNS = ["plain", "x.y.Dotted", "x.y.Pro-Vax", "x.user-y.Left Wing"]


def _graph():
    values = [0, 1, 2, 0, 3, 0]
    nodes = pd.DataFrame({
        "id": list(range(6)),
        "kind": ["a", "a", "b", "b", "a", "b"],
        **{c: values for c in COLUMNS},
    })
    return graphistry.nodes(nodes, "id")


def _max_by_kind(source):
    return [rows(), group_by(keys=["kind"], aggregations=[["m", "max", source]])]


def _validates_clean(g, query) -> bool:
    try:
        out = g.gfql_validate(query)
    except GFQLValidationError:
        return False
    return out["ok"] is True and out["diagnostics"] == []


def _max_per_kind(g, query, engine):
    df = g.gfql(query, engine=engine)._nodes
    expected_module = "polars" if engine.startswith("polars") else engine
    assert type(df).__module__.split(".")[0] == expected_module
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    return sorted((r["kind"], r["m"]) for r in pdf.to_dict("records"))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("column", COLUMNS)
def test_existing_column_validates_and_executes_as_that_column(column, engine):
    g = _graph()
    query = _max_by_kind(column)

    assert _validates_clean(g, query)
    assert _max_per_kind(g, query, engine) == [("a", 3), ("b", 2)]


def test_missing_column_is_still_diagnosed():
    g = _graph()
    query = _max_by_kind("x.y.Missing")

    assert not _validates_clean(g, query)
    with pytest.raises(GFQLValidationError) as exc:
        g.gfql(query, engine="pandas")
    assert exc.value.code == ErrorCode.E301


def test_expression_over_existing_columns_still_validates():
    g = _graph()
    query = _max_by_kind("plain * 2")

    assert _validates_clean(g, query)
    assert _max_per_kind(g, query, "pandas") == [("a", 6), ("b", 4)]


def _scope_graph(source, table, engine):
    active = pd.DataFrame({
        "id": [0, 1, 2, 3], "src": [0, 1, 2, 3], "dst": [1, 2, 3, 0],
        "kind": ["a", "a", "b", "b"], "a": [10, 20, 30, 40], "b": [2, 4, 6, 8],
        source: [101.0, 103.0, 107.0, None],
    })
    other = pd.DataFrame({
        "id": [0, 1, 2, 3], "src": [0, 1, 2, 3], "dst": [1, 2, 3, 0],
        "kind": ["a", "a", "b", "b"], "other.only": [1, 2, 3, 4],
    })
    if engine == "cudf":
        import cudf
        active, other = cudf.from_pandas(active), cudf.from_pandas(other)
    elif engine.startswith("polars"):
        import polars as pl
        active, other = pl.from_pandas(active), pl.from_pandas(other)
    nodes, edges = (active, other) if table == "nodes" else (other, active)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def _replacement(op_name):
    if op_name == "select":
        return select(["kind", "a", "b"])
    if op_name == "with":
        return with_(["kind", "a", "b"])
    if op_name == "return":
        return ASTCall("return_", {"items": [(c, c) for c in ["kind", "a", "b"]]})
    if op_name == "drop_cols":
        return drop_cols(["plain", "x.y.Dotted", "a + missing"])
    return group_by(keys=["kind"], aggregations=[("a", "sum", "a"), ("b", "sum", "b")])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("table", ["nodes", "edges"])
@pytest.mark.parametrize("op_name", ["select", "with", "return", "group_by", "drop_cols"])
@pytest.mark.parametrize("source,missing", [("plain", "plain"), ("x.y.Dotted", "x"), ("a + missing", "missing")])
def test_replaced_row_schema_rejects_removed_aggregation_source(engine, table, op_name, source, missing):
    g = _scope_graph(source, table, engine)
    query = [rows(table=table), _replacement(op_name),
             group_by(keys=["kind"], aggregations=[("m", "max", source)])]

    with pytest.raises(GFQLValidationError) as exc:
        g.gfql_validate(query)
    assert exc.value.code == ErrorCode.E301
    assert exc.value.context["value"] == missing

    from graphistry.compute.validate.validate_schema import validate_chain_schema
    errors = validate_chain_schema(g, query, collect_all=True)
    assert len(errors) == 1
    assert errors[0].code == ErrorCode.E301
    assert errors[0].context["operation_index"] == 2
    assert source in (g._nodes if table == "nodes" else g._edges).columns


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("table", ["nodes", "edges"])
@pytest.mark.parametrize("mode", ["keep", "rename", "extend"])
@pytest.mark.parametrize("source", ["x.y.Dotted", "a + b"])
def test_visible_literal_source_after_projection_keeps_precedence(engine, table, mode, source):
    g = _scope_graph(source, table, engine)
    if mode == "keep":
        op = select(["kind", source])
    elif mode == "rename":
        op = with_(["kind", ("renamed.dot", source)])
        source = "renamed.dot"
    else:
        op = with_([("new", "a")], extend=True)
    query = [rows(table=table), op, group_by(keys=["kind"], aggregations=[("m", "max", source)])]

    assert _validates_clean(g, query)
    assert _max_per_kind(g, query, engine) == [("a", 103.0), ("b", 107.0)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("table", ["nodes", "edges"])
def test_group_by_prefix_keys_remain_visible(engine, table):
    g = _scope_graph("entity.value", table, engine)
    query = [rows(table=table),
             group_by(keys=["kind"], aggregations=[("cnt", "count")], key_prefixes=["entity."]),
             group_by(keys=["kind"], aggregations=[("m", "max", "entity.value")])]

    assert _validates_clean(g, query)
    assert _max_per_kind(g, query, engine) == [("a", 103.0), ("b", 107.0)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("table", ["nodes", "edges"])
def test_removed_literal_source_falls_back_to_visible_expression_operands(engine, table):
    g = _scope_graph("a + b", table, engine)
    query = [rows(table=table), select(["kind", "a", "b"]),
             group_by(keys=["kind"], aggregations=[("m", "max", "a + b")])]
    assert _validates_clean(g, query)
    if engine.startswith("polars"):
        with pytest.raises(NotImplementedError):
            g.gfql(query, engine=engine)
    else:
        assert _max_per_kind(g, query, engine) == [("a", 24), ("b", 48)]


@pytest.mark.skipif(os.environ.get("TEST_POLARS_GPU") != "1", reason="requires TEST_POLARS_GPU=1 and cudf-polars")
def test_dotted_aggregate_after_graph_path_uses_strict_gpu_collect(monkeypatch):
    import polars as pl

    gpu_collects = []
    original_collect = pl.LazyFrame.collect
    original_collect_all = pl.collect_all

    def record_gpu(engine):
        if isinstance(engine, pl.GPUEngine):
            assert engine.config.get("raise_on_fail") is True
            gpu_collects.append(engine)

    def collect(frame, *args, **kwargs):
        result = original_collect(frame, *args, **kwargs)
        record_gpu(kwargs.get("engine"))
        return result

    def collect_all(frames, *args, **kwargs):
        result = original_collect_all(frames, *args, **kwargs)
        record_gpu(kwargs.get("engine"))
        return result

    monkeypatch.setattr(pl.LazyFrame, "collect", collect)
    monkeypatch.setattr(pl, "collect_all", collect_all)
    g = _scope_graph("x.y.Dotted", "nodes", "polars-gpu")
    query = [n(), e_forward(), n(), rows(),
             group_by(keys=["kind"], aggregations=[("m", "max", "x.y.Dotted")])]

    assert _validates_clean(g, query)
    assert _max_per_kind(g, query, "polars-gpu") == [("a", 103.0), ("b", 107.0)]
    assert gpu_collects


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("table", ["nodes", "edges"])
@pytest.mark.parametrize("source", ["with missing", "a +", "sum(", ")"])
def test_malformed_missing_aggregate_source_fails_validation(engine, table, source):
    from graphistry.compute.exceptions import GFQLSyntaxError
    from graphistry.compute.validate.validate_schema import validate_chain_schema

    g = _scope_graph("x.y.Dotted", table, engine)
    query = [rows(table=table), group_by(keys=["kind"], aggregations=[("m", "max", source)])]
    with pytest.raises(GFQLSyntaxError) as exc:
        g.gfql_validate(query)
    assert exc.value.code == ErrorCode.E107
    assert exc.value.context["value"] == source
    assert exc.value.context["operation_index"] == 1
    errors = validate_chain_schema(g, query, collect_all=True)
    assert errors is not None and len(errors) == 1
    assert errors[0].code == ErrorCode.E107
    assert errors[0].context["operation_index"] == 1


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source", ["with missing", "a +", "sum(", ")"])
def test_malformed_expression_spelling_is_valid_existing_literal(engine, source):
    g = _scope_graph(source, "nodes", engine)
    assert g.gfql_validate(_max_by_kind(source))["ok"]
    assert _max_per_kind(g, _max_by_kind(source), engine) == [("a", 103.0), ("b", 107.0)]


@pytest.mark.parametrize("source,expected", [("1", 1), ("1 + 2", 3)])
def test_valid_constant_aggregate_source_requires_no_columns(source, expected):
    g = _scope_graph("x.y.Dotted", "nodes", "pandas")
    assert g.gfql_validate(_max_by_kind(source))["ok"]
    assert _max_per_kind(g, _max_by_kind(source), "pandas") == [("a", expected), ("b", expected)]


def test_aggregate_source_unknown_schema_remains_deferred():
    from graphistry.compute.gfql.call.validation import _agg_source_required_cols

    assert _agg_source_required_cols("with missing") == []
    assert _agg_source_required_cols("a + missing") == ["a", "missing"]


def test_aggregate_source_without_parser_distinguishes_literal_and_expression(monkeypatch):
    from graphistry.compute.gfql.call import validation
    from graphistry.compute.exceptions import GFQLTypeError

    monkeypatch.setattr(validation, "_where_rows_expr_parser_fn", lambda: None)
    assert validation._agg_source_required_cols("with space", {"with space"}) == ["with space"]
    with pytest.raises(GFQLTypeError) as exc:
        validation._agg_source_required_cols("1 + 2", {"id"})
    assert exc.value.code == ErrorCode.E201
    assert exc.value.context["value"] == "1 + 2"


@pytest.mark.parametrize("func", ["sum", "avg", "mean"])
@pytest.mark.parametrize("source", ["'text'", "[1, 2]", "{value: 1}"])
def test_numeric_aggregate_constant_type_is_validated(func, source):
    from graphistry.compute.exceptions import GFQLTypeError

    g = _scope_graph("x.y.Dotted", "nodes", "pandas")
    query = [rows(), group_by([], [("answer", func, source)])]
    with pytest.raises(GFQLTypeError) as exc:
        g.gfql_validate(query)
    assert exc.value.code == ErrorCode.E302
    assert exc.value.context["field"] == source
    assert exc.value.context["operation_index"] == 1


@pytest.mark.parametrize("source", ["3", "1 + 2", "true", "null"])
def test_numeric_aggregate_valid_constant_global_validation(source):
    g = _scope_graph("x.y.Dotted", "nodes", "pandas")
    assert g.gfql_validate([rows(), group_by([], [("answer", "sum", source)])])["ok"]


@pytest.mark.parametrize("source", ["'text'", "[1, 2]", "{value: 1}"])
def test_numeric_aggregate_literal_column_precedes_constant_type(source):
    g = _scope_graph(source, "nodes", "pandas")
    assert g.gfql_validate([rows(), group_by([], [("answer", "sum", source)])])["ok"]
