"""Exact native aggregate operand and global-row identities."""
import os

import pytest

import graphistry

pl = pytest.importorskip("polars")


@pytest.mark.parametrize("rows", [0, 3])
@pytest.mark.parametrize("keys", [[], ["key"]])
def test_native_aggregate_expression_and_constant_shapes(rows, keys):
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import group_by_polars

    table = pl.DataFrame({"key": [1] * rows, "value": [2, None, 4][:rows]},
                         schema={"key": pl.Int64, "value": pl.Int64})
    graph = graphistry.nodes(table)
    out = group_by_polars(graph, keys, [
        ("n", "count"), ("constant_n", "count", "3"),
        ("constant_sum", "sum", "3"), ("expression_sum", "sum", "value + 1"),
        ("mean", "mean", "value + 1"), ("minimum", "min", "value + 1"),
        ("maximum", "max", "value + 1"), ("distinct", "count_distinct", "value + 1"),
        ("values", "collect", "value + 1"), ("unique", "collect_distinct", "3"),
    ])
    assert out is not None
    expected = [] if keys and rows == 0 else [{
        **({"key": 1} if keys else {}),
        "n": rows, "constant_n": rows, "constant_sum": rows * 3,
        "expression_sum": 8 if rows else 0, "mean": 4.0 if rows else None,
        "minimum": 3 if rows else None, "maximum": 5 if rows else None,
        "distinct": 2 if rows else 0, "values": [3, 5] if rows else [],
        "unique": [3] if rows else [],
    }]
    assert out._nodes.to_dicts() == expected
    assert graph._nodes is table
    assert table.columns == ["key", "value"]


def test_native_aggregate_literal_precedes_expression():
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import group_by_polars

    graph = graphistry.nodes(pl.DataFrame({"value": [1, 2], "value + 1": [40, 60]}))
    out = group_by_polars(graph, [], [("total", "sum", "value + 1")])
    assert out is not None
    assert out._nodes.to_dicts() == [{"total": 100}]


@pytest.mark.parametrize("engine", ["pandas", "polars",
    pytest.param("cudf", marks=pytest.mark.skipif(os.environ.get("TEST_CUDF") != "1", reason="requires TEST_CUDF=1")),
    pytest.param("polars-gpu", marks=pytest.mark.skipif(os.environ.get("TEST_POLARS_GPU") != "1", reason="requires TEST_POLARS_GPU=1")),
])
@pytest.mark.parametrize("rows", [0, 3])
def test_public_global_aggregate_shapes(engine, rows):
    import pandas as pd
    from graphistry.compute.ast import group_by

    if engine in ("cudf", "polars-gpu"):
        pytest.importorskip("cudf")
    if engine == "polars-gpu":
        pytest.importorskip("cudf_polars")
    table = pd.DataFrame({"value": pd.Series([2, None, 4][:rows], dtype="float64")})
    table = table.assign(id=range(rows))
    graph = graphistry.nodes(table, "id")
    out = graph.gfql([group_by([], [
        ("n", "count"), ("total", "sum", "3"),
        ("minimum", "min", "value + 1"), ("values", "collect", "value + 1"),
    ])], engine=engine)._nodes
    assert type(out).__module__.split(".")[0] == ("polars" if engine.startswith("polars") else engine)
    records = out.to_dicts() if engine.startswith("polars") else (
        out.to_pandas().to_dict("records") if engine == "cudf" else out.to_dict("records"))
    assert len(records) == 1
    assert records[0]["n"] == rows
    assert records[0]["total"] == 3 * rows
    if rows:
        assert records[0]["minimum"] == 3
    else:
        assert pd.isna(records[0]["minimum"])
    assert list(records[0]["values"]) == ([3, 5] if rows else [])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_global_internal_key_does_not_shadow_output(engine):
    import pandas as pd
    from graphistry.compute.ast import group_by

    graph = graphistry.nodes(pd.DataFrame({"id": [0, 1]}), "id")
    out = graph.gfql([group_by([], [("__gfql_global_group__", "count")])], engine=engine)._nodes
    assert list(out.columns) == ["__gfql_global_group__"]
    assert out["__gfql_global_group__"].to_list() == [2]


@pytest.fixture
def strict_aggregate_device(monkeypatch):
    from graphistry.compute.gfql.lazy.engine.polars import row_pipeline

    original_group = row_pipeline.group_by_polars
    original_collect = pl.LazyFrame.collect
    active = False
    depth = 0
    receipts = []

    def group(*args, **kwargs):
        nonlocal active
        active = True
        try:
            return original_group(*args, **kwargs)
        finally:
            active = False

    def collect(frame, *args, **kwargs):
        nonlocal depth
        outer = active and depth == 0
        depth += 1
        try:
            result = original_collect(frame, *args, **kwargs)
        finally:
            depth -= 1
        if outer:
            backend = kwargs.get("engine")
            assert isinstance(backend, pl.GPUEngine)
            assert backend.config["raise_on_fail"] is True
            receipts.append(result)
        return result

    monkeypatch.setattr(row_pipeline, "group_by_polars", group)
    monkeypatch.setattr(pl.LazyFrame, "collect", collect)
    return receipts


@pytest.mark.skipif(os.environ.get("TEST_POLARS_GPU") != "1", reason="requires actual polars-gpu")
@pytest.mark.parametrize("connected", [False, True])
@pytest.mark.parametrize("keys", [[], ["key"], ["key", "other"]])
@pytest.mark.parametrize("func, expected", [
    ("count", 3), ("sum", 11), ("avg", 11 / 3), ("mean", 11 / 3),
    ("min", 3), ("max", 5), ("count_distinct", 2),
    ("collect", [3, 3, 5]), ("collect_distinct", [3, 5]),
])
def test_gpu_expression_aggregate_matrix(strict_aggregate_device, connected, keys, func, expected):
    import pandas as pd
    from graphistry.compute.ast import e_forward, group_by, n, rows

    nodes = pd.DataFrame({"id": [0, 1, 2, 3], "key": [None] * 4,
                          "other": [7] * 4, "value": [2.0, None, 2.0, 4.0]})
    edges = pd.DataFrame({"src": [0, 1, 2, 3], "dst": [0, 1, 2, 3]})
    graph = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    prefix = [n(), e_forward(), n()] if connected else []
    out = graph.gfql([*prefix, rows(), group_by(keys, [("answer", func, "value + 1")])],
                     engine="polars-gpu")._nodes
    assert type(out).__module__.split(".")[0] == "polars"
    assert out.height == 1
    answer = out.to_dicts()[0]["answer"]
    if func in ("avg", "mean"):
        assert answer == pytest.approx(expected)
    else:
        assert answer == expected
    for key in keys:
        assert out.to_dicts()[0][key] == (None if key == "key" else 7)
    assert strict_aggregate_device
    assert any("answer" in result.columns for result in strict_aggregate_device)


@pytest.mark.parametrize("values, expected", [([], 0), ([None, None], 0), ([2, None, 2, 4], 2), ([2, 2], 1)])
def test_gpu_distinct_count_plan_null_identity(monkeypatch, values, expected):
    from graphistry.compute.gfql import lazy
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import group_by_polars

    monkeypatch.setattr(lazy, "collect", lambda plan: plan.collect())
    graph = graphistry.nodes(pl.DataFrame({"value": values}, schema={"value": pl.Int64}))
    with lazy.target_mode(lazy.ExecutionTarget.GPU):
        out = group_by_polars(graph, [], [("n", "count_distinct", "value")])
    assert out is not None
    assert out._nodes.to_dicts() == [{"n": expected}]


@pytest.mark.parametrize("keys", [[], ["key"]])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("device", [False, pytest.param(True, marks=pytest.mark.skipif(
    os.environ.get("TEST_POLARS_GPU") != "1", reason="requires actual GPU collection identity execution"))])
def test_gpu_collection_plan_preserves_groups_and_identity(monkeypatch, request, keys, empty, device):
    from graphistry.compute.gfql import lazy
    from graphistry.compute.gfql.lazy.engine.polars import row_pipeline

    receipts = request.getfixturevalue("strict_aggregate_device") if device else None
    if not device:
        monkeypatch.setattr(lazy, "collect", lambda plan: plan.collect())
    table = pl.DataFrame({"key": [2, 1, 1, 1], "value": [None, 3, 3, 5]},
                         schema={"key": pl.Int64, "value": pl.Int64})
    if empty:
        table = table.head(0)
    with lazy.target_mode(lazy.ExecutionTarget.GPU):
        out = row_pipeline.group_by_polars(graphistry.nodes(table), keys, [
            ("values", "collect", "value"), ("n", "count"),
            ("unique", "collect_distinct", "value"),
        ])
    assert out is not None
    if keys:
        expected = [] if empty else [
            {"key": 2, "values": [], "n": 1, "unique": []},
            {"key": 1, "values": [3, 3, 5], "n": 3, "unique": [3, 5]},
        ]
    else:
        expected = [{"values": [] if empty else [3, 3, 5], "n": 0 if empty else 4,
                     "unique": [] if empty else [3, 5]}]
    assert out._nodes.to_dicts() == expected
    assert out._nodes.columns == [*keys, "values", "n", "unique"]
    if device:
        assert receipts
