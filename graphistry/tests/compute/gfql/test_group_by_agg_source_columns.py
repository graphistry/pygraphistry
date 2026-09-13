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


@pytest.mark.parametrize("connected", [False, True])
@pytest.mark.parametrize("source", ["plain", "x.y.Dotted", "a + b"])
@pytest.mark.skipif(os.environ.get("TEST_POLARS_GPU") != "1", reason="requires TEST_POLARS_GPU=1 and cudf-polars")
def test_aggregate_itself_collects_on_gpu(monkeypatch, connected, source):
    import polars as pl
    from graphistry.compute.gfql.lazy.engine.polars import row_pipeline

    aggregate_calls = []
    collecting_aggregate = False
    collect_depth = 0
    original_group_by = row_pipeline.group_by_polars
    original_collect = pl.LazyFrame.collect

    def group_by_on_device(*args, **kwargs):
        nonlocal collecting_aggregate
        collecting_aggregate = True
        try:
            return original_group_by(*args, **kwargs)
        finally:
            collecting_aggregate = False

    def collect(frame, *args, **kwargs):
        nonlocal collect_depth
        outer_aggregate = collecting_aggregate and collect_depth == 0
        collect_depth += 1
        try:
            result = original_collect(frame, *args, **kwargs)
        finally:
            collect_depth -= 1
        if outer_aggregate:
            backend = kwargs.get("engine")
            assert isinstance(backend, pl.GPUEngine)
            assert backend.config["raise_on_fail"] is True
            aggregate_calls.append(result)
        return result

    monkeypatch.setattr(row_pipeline, "group_by_polars", group_by_on_device)
    monkeypatch.setattr(pl.LazyFrame, "collect", collect)
    g = _scope_graph(source, "nodes", "polars-gpu")
    prefix = [n(), e_forward(), n()] if connected else []
    query = [*prefix, rows(), group_by(keys=["kind"], aggregations=[("m", "max", source)])]
    assert _max_per_kind(g, query, "polars-gpu") == [("a", 103.0), ("b", 107.0)]
    assert len(aggregate_calls) == 1


def test_gpu_aggregate_backend_refusal_is_structured(monkeypatch):
    pytest.importorskip("polars")
    from graphistry.compute.exceptions import GFQLUnsupportedError
    from graphistry.compute.gfql import lazy
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import group_by_polars

    calls = []

    def decline(plan):
        calls.append(plan)
        raise NotImplementedError("backend capability refusal")

    monkeypatch.setattr(lazy, "collect", decline)
    g = _scope_graph("x.y.Dotted", "nodes", "polars")
    with lazy.target_mode(lazy.ExecutionTarget.GPU):
        with pytest.raises(GFQLUnsupportedError) as exc:
            group_by_polars(g, ["kind"], [("m", "max", "x.y.Dotted")])
    assert len(calls) == 1
    assert isinstance(exc.value, NotImplementedError)
    assert exc.value.code == ErrorCode.E110
    assert exc.value.context["value"] == "group_by"
    assert exc.value.context["engine"] == "polars-gpu"
    assert isinstance(exc.value.__cause__, NotImplementedError)


def test_cpu_aggregate_does_not_require_gpu_collector(monkeypatch):
    pytest.importorskip("polars")
    from graphistry.compute.gfql import lazy

    def forbidden(plan):
        pytest.fail("CPU aggregate called the explicit device collector")

    monkeypatch.setattr(lazy, "collect", forbidden)
    g = _scope_graph("x.y.Dotted", "nodes", "polars")
    assert _max_per_kind(g, _max_by_kind("x.y.Dotted"), "polars") == [("a", 103.0), ("b", 107.0)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("empty", [False, True])
def test_numeric_empty_and_null_aggregate_schema(engine, empty):
    pdf = pd.DataFrame({
        "id": [0, 1, 2, 3], "kind": ["a", "a", "b", "b"],
        "x.y.Dotted": pd.Series([None, None, 3.0, None], dtype="float64"),
    })
    if empty:
        pdf = pdf.iloc[:0]
    g = graphistry.nodes(pdf, "id")
    query = [rows(), group_by(keys=["kind"], aggregations=[("m", "sum", "x.y.Dotted")])]
    assert _max_per_kind(g, query, engine) == ([] if empty else [("a", 0.0), ("b", 3.0)])


def test_gpu_aggregate_unlowerable_expression_is_structured(monkeypatch):
    pytest.importorskip("polars")
    from graphistry.compute.exceptions import GFQLUnsupportedError
    from graphistry.compute.gfql import lazy
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import group_by_polars

    g = _scope_graph("x.y.Dotted", "nodes", "polars")
    from graphistry.compute.gfql.lazy.engine.polars import row_pipeline

    monkeypatch.setattr(row_pipeline, "_lower_with_schema", lambda *args, **kwargs: None)
    args = (g, ["kind"], [("m", "max", "a + b")])
    with lazy.target_mode(lazy.ExecutionTarget.CPU):
        assert group_by_polars(*args) is None
    with lazy.target_mode(lazy.ExecutionTarget.GPU):
        with pytest.raises(GFQLUnsupportedError) as exc:
            group_by_polars(*args)
    assert exc.value.code == ErrorCode.E110
    assert exc.value.context["engine"] == "polars-gpu"


def test_gpu_aggregate_refusal_survives_native_call_boundary(monkeypatch):
    pytest.importorskip("polars")
    from graphistry.compute.exceptions import GFQLUnsupportedError
    from graphistry.compute.gfql import lazy
    from graphistry.compute.gfql.lazy.engine.polars.chain import chain_polars

    def decline(plan):
        raise NotImplementedError("backend capability refusal")

    monkeypatch.setattr(lazy, "collect", decline)
    g = _scope_graph("x.y.Dotted", "nodes", "polars")
    with lazy.target_mode(lazy.ExecutionTarget.GPU):
        with pytest.raises(GFQLUnsupportedError) as exc:
            chain_polars(g, _max_by_kind("x.y.Dotted"))
    assert exc.value.code == ErrorCode.E110
    assert exc.value.context["value"] == "group_by"
    assert isinstance(exc.value, NotImplementedError)


@pytest.mark.parametrize("target", ["cpu", "gpu"])
def test_aggregate_preparation_refusal_uses_decline_contract(monkeypatch, target):
    pl = pytest.importorskip("polars")
    from graphistry.compute.exceptions import GFQLUnsupportedError
    from graphistry.compute.gfql import lazy
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import group_by_polars

    refusal = NotImplementedError("controlled preparation refusal")
    calls = []

    def decline(plan):
        calls.append(plan)
        raise refusal

    monkeypatch.setattr(lazy, "collect", decline)
    g = graphistry.nodes(pl.DataFrame({"value": [1]}))
    with lazy.target_mode(lazy.ExecutionTarget(target)):
        if target == "cpu":
            assert group_by_polars(g, [], [("answer", "sum", "'text'")]) is None
        else:
            with pytest.raises(GFQLUnsupportedError) as exc:
                group_by_polars(g, [], [("answer", "sum", "'text'")])
            assert exc.value.code == ErrorCode.E110
            assert exc.value.context["engine"] == "polars-gpu"
            assert exc.value.__cause__ is refusal
    assert len(calls) == 1
