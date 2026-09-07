import importlib

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.ast import ASTCall, e_forward, e_reverse, e_undirected, n, rows, select
from graphistry.compute.chain_specializations.admission import point_rows_admits
from graphistry.compute.chain_specializations.point_rows import _try_point_rows
from graphistry.compute.predicates.numeric import GT
from graphistry.tests.compute.gfql.routes.switch import ROUTES, routes_off

chain_mod = importlib.import_module("graphistry.compute.chain")


@pytest.fixture(params=["pandas", "cudf"])
def engine(request):
    from graphistry.tests.compute.gfql.routes.test_route_harness import _skip_unavailable
    _skip_unavailable(request.param)
    return request.param


def graph(engine, variant="base"):
    nodes = pd.DataFrame({"key": range(100), "id": range(1000, 1100), "kind": ["Person"] * 100,
                          "value": range(100), "content": ["hello"] * 100})
    edges = pd.DataFrame({"s": [0, 0, 1, 2, 4, 0], "d": [1, 2, 0, 0, 99, 1],
                          "type": ["X", "X", "X", "Y", "X", "X"], "eid": range(6), "weight": range(6)})
    if variant == "loop":
        edges.loc[len(edges)] = [0, 0, "X", 6, 6]
    elif variant == "dangling":
        edges.loc[len(edges)] = [0, 10000, "X", 6, 6]
    elif variant == "nonleading":
        nodes = nodes[["id", "value", "key", "kind", "content"]]
    elif variant == "empty":
        edges = edges.iloc[:0]
    elif variant == "nullable":
        nodes = nodes.astype({"value": "Int64"})
        nodes.loc[1, "value"] = pd.NA
    elif variant == "duplicate-index":
        nodes.index = [0] * len(nodes)
        edges.index = [0] * len(edges)
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "key").edges(edges, "s", "d", "eid").gfql_index_all(engine=engine)


def topd(frame):
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


def assert_result(actual, expected):
    for attr in ("_nodes", "_edges"):
        aa, ee = topd(getattr(actual, attr)), topd(getattr(expected, attr))
        if aa is None or ee is None:
            assert aa is ee
            continue
        assert list(aa.columns) == list(ee.columns)
        cols = list(aa.columns)
        if cols:
            aa = aa.sort_values(cols, na_position="last")
            ee = ee.sort_values(cols, na_position="last")
        pd.testing.assert_frame_equal(aa.reset_index(drop=True), ee.reset_index(drop=True))
    for attr in ("_node", "_edge", "_source", "_destination"):
        assert getattr(actual, attr) == getattr(expected, attr)
    assert actual._gfql_rows_base_graph is None
    assert actual._gfql_start_nodes is None


def point_ops(shape, projection=False, alias="a"):
    if shape == "single":
        ops = [n({"key": 0}, name=alias), rows(source=alias)]
    else:
        edge = e_reverse if shape == "reverse" else e_forward
        source = alias if shape == "seed" else "e" if shape == "edges" else "b"
        ops = [n({"key": 0}, name=alias), edge({"type": "X"}, name="e"),
               n({"kind": "Person"}, name="b"), rows(table="edges" if shape == "edges" else "nodes", source=source)]
    if projection:
        source = ops[-1].params["source"]
        prop = "weight" if shape == "edges" else "value"
        ops += [select([("out", f"{source}.{prop}")])]
    return ops


@pytest.mark.parametrize("shape", ["single", "tail", "seed", "edges", "reverse"])
@pytest.mark.parametrize("projection", [False, True])
@pytest.mark.parametrize("variant", ["base", "loop", "dangling", "nonleading", "empty", "nullable", "duplicate-index"])
def test_point_route_serves_without_chain_and_preserves_results(engine, shape, projection, variant, monkeypatch):
    g = graph(engine, variant)
    ops = point_ops(shape, projection)
    with routes_off(ROUTES):
        expected = g.gfql(ops, engine=engine)
    before_nodes, before_edges = topd(g._nodes).copy(), topd(g._edges).copy()
    def reject_chain(*args, **kwargs):
        pytest.fail("point query entered chain orchestration")
    monkeypatch.setattr(chain_mod, "_chain_impl", reject_chain)
    result = g.gfql(ops, engine=engine)
    assert_result(result, expected)
    pd.testing.assert_frame_equal(topd(g._nodes), before_nodes)
    pd.testing.assert_frame_equal(topd(g._edges), before_edges)


@pytest.mark.parametrize("shape", ["single", "tail", "seed", "edges"])
@pytest.mark.parametrize("alias", ["value", "kind", "content"])
def test_alias_property_collisions(engine, shape, alias, request):
    g = graph(engine)
    ops = point_ops(shape, False, alias)
    with routes_off(ROUTES):
        expected = g.gfql(ops, engine=engine)
    actual = _try_point_rows(g, ops, Engine(engine))
    assert actual is not None
    if engine == "cudf" and shape == "seed" and alias == "value":
        assert topd(actual._nodes)["__gfql_shadow_restore__value__"].tolist() == [0]
        request.applymarker(pytest.mark.xfail(strict=True, reason="general cuDF alias restoration uses reordered row indexes; local repro in plan"))
    assert_result(actual, expected)


@pytest.mark.parametrize("ops", [
    [n(), rows(source="a")],
    [n({"key": GT(0)}, name="a"), rows(source="a")],
    [n({"key": 0}, name="a", query="value > 1"), rows(source="a")],
    [n({"key": 0}, name="a"), e_undirected(), n(name="b"), rows(source="b")],
    [n({"key": 0}, name="a"), e_forward(hops=2), n(name="b"), rows(source="b")],
    [n({"key": 0}, name="a"), e_forward(prune_to_endpoints=True), n(name="b"), rows(source="b")],
    [n({"key": 0}, name="a"), e_forward(source_node_match={"value": 0}), n(name="b"), rows(source="b")],
    [n({"key": 0}, name="a"), rows()],
    [n({"key": 0}, name="a"), rows(source="missing")],
    [n({"key": 0}, name="a"), rows(table="edges", source="a")],
    [n({"key": 0}, name="a"), rows(source="a", alias_endpoints={"a": "src"})],
    [n({"key": 0}, name="a"), ASTCall("rows", {"table": "nodes", "source": "a", "unexpected": True})],
])
def test_admission_declines_unsupported_shapes(ops):
    assert point_rows_admits(ops, Engine.PANDAS, None) is None


@pytest.mark.parametrize("shape", ["single", "tail"])
@pytest.mark.parametrize("mode", ["off", "missing", "stale"])
def test_unusable_indexes_decline_without_scanning(engine, shape, mode, monkeypatch):
    g = graph(engine)
    if mode == "off":
        from graphistry.compute.gfql.index import with_index_policy
        g = with_index_policy(g, "off")
    elif mode == "missing":
        g = graphistry.nodes(g._nodes, "key").edges(g._edges, "s", "d", "eid")
    else:
        g = g.nodes(g._nodes.copy())
    bindings = importlib.import_module("graphistry.compute.gfql.index.bindings")
    def reject_scan(*args, **kwargs):
        pytest.fail("a declined point route scanned the node table")
    monkeypatch.setattr(bindings, "_filter_frame", reject_scan)
    assert _try_point_rows(g, point_ops(shape), Engine(engine)) is None


@pytest.mark.parametrize("shape", ["single", "tail"])
def test_start_nodes_and_other_engines_decline(shape):
    ops = point_ops(shape)
    assert point_rows_admits(ops, Engine.PANDAS, pd.DataFrame({"key": [0]})) is None
    for engine in (Engine.POLARS, Engine.POLARS_GPU):
        assert point_rows_admits(ops, engine, None) is None


@pytest.mark.parametrize("shape", ["single", "tail"])
def test_property_index_serves_with_residual_and_empty_filters(engine, shape, monkeypatch):
    g = graph(engine).gfql_index_node_props(["id"], engine=engine)
    for filters in ({"id": 1000, "kind": "Person"}, {"id": 9999}, {"id": 1000, "kind": "Absent"}):
        ops = point_ops(shape)
        ops[0] = n(filters, name="a")
        with routes_off(ROUTES):
            expected = g.gfql(ops, engine=engine)
        actual = _try_point_rows(g, ops, Engine(engine))
        assert actual is not None
        assert_result(actual, expected)


@pytest.mark.parametrize("shape", ["single", "tail"])
def test_policy_hooks_keep_the_canonical_dispatch(engine, shape, monkeypatch):
    g = graph(engine)
    seen = []
    def point_must_decline(*args, **kwargs):
        pytest.fail("policy-bearing query entered the point route")
    monkeypatch.setattr(chain_mod, "_try_point_rows", point_must_decline)
    g.gfql(point_ops(shape), engine=engine, policy={"prechain": lambda ctx: seen.append("prechain")})
    assert seen


@pytest.mark.parametrize("filters", [{"absent": 1}, {"key": "wrong-type"}])
def test_schema_errors_match_the_general_path(engine, filters):
    g = graph(engine)
    ops = [n(filters, name="a"), rows(source="a")]
    with routes_off(ROUTES), pytest.raises(Exception) as expected:
        g.gfql(ops, engine=engine, strict=True)
    with pytest.raises(type(expected.value)) as actual:
        g.gfql(ops, engine=engine, strict=True)
    assert actual.value.code == expected.value.code


@pytest.mark.parametrize("shape", ["single", "tail"])
def test_index_trace_reports_the_point_route(engine, shape):
    g = graph(engine)
    report = g.gfql_explain(point_ops(shape), engine=engine)
    assert report["used_index"]
    assert any(step.get("seam") == "point_rows" for step in report["steps"])


@pytest.mark.parametrize("source", ["value", "kind"])
@pytest.mark.parametrize("key", [0, 2, 4])
def test_colliding_source_property_stays_with_its_node(engine, source, key):
    g = graph(engine)
    ops = [n({"key": key}, name=source), rows(source=source), select([("out", f"{source}.{source}")])]
    result = g.gfql(ops, engine=engine)
    expected = key if source == "value" else "Person"
    assert topd(result._nodes)["out"].tolist() == [expected]
