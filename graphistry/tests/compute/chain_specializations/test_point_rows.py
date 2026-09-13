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
    elif variant == "categorical":
        nodes["value"] = nodes["value"].astype("category")
    elif variant == "timestamp":
        nodes["value"] = pd.date_range("2020-01-01", periods=len(nodes), tz="UTC")
    elif variant == "float-null":
        nodes["value"] = nodes["value"].astype(float)
        nodes.loc[1, "value"] = float("nan")
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
def test_alias_property_collisions(engine, shape, alias):
    g = graph(engine)
    ops = point_ops(shape, False, alias)
    with routes_off(ROUTES):
        expected = g.gfql(ops, engine=engine)
    actual = _try_point_rows(g, ops, Engine(engine))
    assert actual is not None
    if engine == "cudf" and shape == "seed" and alias == "value":
        assert topd(actual._nodes)["__gfql_shadow_restore__value__"].tolist() == [0]
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


@pytest.mark.parametrize("items", [
    [("out", "a.value + 1")], [("out", 7)], [("out", "a")],
    [("eid", "a.value")], [("key", "a.value")], [],
    [("out", "a.value"), ("out", "a.id")],
])
def test_projection_shortcut_and_fallback_preserve_binding_metadata(engine, items):
    g = graph(engine)
    ops = [n({"key": 2}, name="a"), rows(source="a"), select(items)]
    with routes_off(ROUTES):
        expected = g.gfql(ops, engine=engine)
    actual = _try_point_rows(g, ops, Engine(engine))
    assert actual is not None
    assert_result(actual, expected)


@pytest.mark.parametrize("source,expression", [("a.b", "a.b.value"), ("a", "a.two words")])
def test_projection_requires_parsing_for_non_identifier_properties(source, expression):
    from graphistry.compute.chain_specializations.point_rows import _project_point_columns
    frame = pd.DataFrame({"value": [1], "two words": [2]})
    assert _project_point_columns(frame, select([("out", expression)]), source, [source]) is None


@pytest.mark.parametrize("variant", ["base", "loop", "dangling", "empty", "nullable", "duplicate-index",
                                     "nonleading", "categorical", "timestamp", "float-null"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("named_edge", [False, True])
def test_joined_point_projection(engine, variant, reverse, named_edge, monkeypatch):
    g = graph(engine, variant)
    hop = e_reverse if reverse else e_forward
    items = [("seed", "a.value"), ("tail", "b.value"), ("id", "b.key")]
    if named_edge:
        items.append(("weight", "e.weight"))
    ops = [n({"key": 0}, name="a"), hop({"type": "X"}, name="e" if named_edge else None),
           n(name="b"), rows(), select(items)]
    with routes_off(ROUTES):
        expected = g.gfql(ops, engine=engine)
    def fail(*args, **kwargs):
        raise AssertionError("joined point fell back to chain execution")
    monkeypatch.setattr(chain_mod, "_chain_impl", fail)
    actual = g.gfql(ops, engine=engine)
    assert_result(actual, expected)


@pytest.mark.parametrize("items", [[("v", "a.value + b.value")], [("v", "a")], [],
                                   [("v", "a.missing")], [("v", "a.`value`")]])
def test_joined_point_projection_declines(engine, items):
    g = graph(engine)
    ops = [n({"key": 0}, name="a"), e_forward({"type": "X"}), n(name="b"), rows(), select(items)]
    assert _try_point_rows(g, ops, Engine(engine), validate_schema=False) is None


@pytest.mark.parametrize("seed,tail", [({"key": 999}, {}), ({"key": 0}, {"value": 1}),
                                      ({"key": 0}, {"value": 999}), ({"bucket": 0}, {"value": 1})])
def test_joined_point_filters_and_repeated_outputs(engine, seed, tail, monkeypatch):
    g = graph(engine)
    g = g.nodes(g._nodes.assign(bucket=g._nodes["key"] // 3)).gfql_index_all(engine=engine)
    g = g.gfql_index_node_props(["bucket"], engine=engine)
    from graphistry.compute.gfql.index import with_index_policy
    g = with_index_policy(g, "force")
    ops = [n(seed, name="a"), e_forward({"type": "X"}), n(tail, name="b"), rows(),
           select([("v", "a.value"), ("v", "b.value"), ("key", "a.key")])]
    with routes_off(ROUTES):
        expected = g.gfql(ops, engine=engine)
    def fail(*args, **kwargs):
        raise AssertionError("joined point fell back")
    monkeypatch.setattr(chain_mod, "_chain_impl", fail)
    assert_result(g.gfql(ops, engine=engine), expected)


@pytest.mark.parametrize("alias", ["value", "key", "content"])
def test_joined_point_alias_collision_declines(engine, alias):
    g = graph(engine)
    ops = [n({"key": 0}, name=alias), e_forward({"type": "X"}), n(name="b"), rows(),
           select([("v", f"{alias}.value"), ("tail", "b.value")])]
    assert _try_point_rows(g, ops, Engine(engine), validate_schema=False) is None


def test_joined_point_result_owns_projected_arrays():
    g = graph("pandas")
    ops = [n({"key": 0}, name="a"), e_forward({"type": "X"}, name="e"), n(name="b"), rows(),
           select([("seed", "a.value"), ("tail", "b.value"), ("weight", "e.weight")])]
    original_nodes, original_edges = g._nodes.copy(), g._edges.copy()
    result = g.gfql(ops, engine="pandas")
    result._nodes.loc[:, ["seed", "tail", "weight"]] = -1
    pd.testing.assert_frame_equal(g._nodes, original_nodes)
    pd.testing.assert_frame_equal(g._edges, original_edges)


@pytest.mark.parametrize("left,right,dtype", [
    ("value", "fallback", "object"), (None, "fallback", "object"),
    ("", "fallback", "object"), (None, None, "object"),
    (pd.NA, "fallback", "string"), (float("nan"), 2.0, "float64"),
    (pd.NA, 2, "Int64"), (pd.NA, False, "boolean"),
])
@pytest.mark.parametrize("empty", [False, True])
def test_point_coalesce_projection(engine, left, right, dtype, empty, monkeypatch):
    g = graph(engine)
    nodes = topd(g._nodes)
    nodes["lhs"] = pd.Series([left] * len(nodes), dtype=dtype)
    nodes["rhs"] = pd.Series([right] * len(nodes), dtype=dtype)
    if engine == "cudf":
        nodes = pytest.importorskip("cudf").from_pandas(nodes)
    g = g.nodes(nodes).gfql_index_all(engine=engine)
    ops = [n({"key": 999 if empty else 0}, name="a"), rows(source="a"),
           select([("key", "a.key"), ("v", "coalesce(a.lhs, a.rhs)")])]
    with routes_off(ROUTES):
        expected = g.gfql(ops, engine=engine)
    import graphistry.compute.chain_specializations.point_rows as point_mod
    def fail(*args, **kwargs):
        raise AssertionError("point projection used the general expression evaluator")
    monkeypatch.setattr(point_mod, "_restore_point_source", fail)
    assert_result(g.gfql(ops, engine=engine), expected)


@pytest.mark.parametrize("expr", ["coalesce(a.value)", "coalesce(a.value, a.key, 0)",
                                  "coalesce(a.value, 0)", "coalesce(a.value, b.value)",
                                  "coalesce(a.value, a.key) + 1", "coalesce(a.value, a.`key`)"])
def test_point_coalesce_unsupported_expression_falls_back(expr):
    from graphistry.compute.chain_specializations.point_rows import _project_point_columns
    assert _project_point_columns(graph("pandas")._nodes, select([("v", expr)]), "a", ["a", "b"]) is None


@pytest.mark.parametrize("where", ["seed", "tail", "edge"])
def test_nullable_equality_does_not_match_null(engine, where):
    g = graph(engine)
    nodes, edges = topd(g._nodes), topd(g._edges)
    nodes["value"] = nodes["value"].astype("Int64")
    edges["weight"] = edges["weight"].astype("Int64")
    if where == "seed":
        nodes.loc[0, "value"] = pd.NA
        ops = [n({"key": 0, "value": 1}, name="a"), rows(source="a")]
    else:
        if where == "tail":
            nodes.loc[[1, 2], "value"] = pd.NA
        else:
            edges.loc[[0, 1, 5], "weight"] = pd.NA
        ops = [n({"key": 0}, name="a"),
               e_forward({"type": "X", **({"weight": 1} if where == "edge" else {})}),
               n({"value": 1} if where == "tail" else {}, name="b"), rows(source="b")]
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = g.nodes(nodes).edges(edges).gfql_index_all(engine=engine)
    with routes_off(ROUTES):
        expected = g.gfql(ops, engine=engine, index_policy="off")
    assert len(expected._nodes) == 0
    assert_result(g.gfql(ops, engine=engine), expected)


@pytest.mark.parametrize("value,dtype,filter_value", [
    (1, "int64", 1), (1, "int64", 2), (1, "int64", "1"),
    (2**63 + 1, "uint64", 2**63), (2**63 + 1, "uint64", 2**63 + 1),
    (1.0, "float64", 1), (float("nan"), "float64", 1),
    (pd.NA, "Int64", 1), (True, "boolean", True), (pd.NA, "boolean", False),
    ("one", "string", "one"), (pd.NA, "string", "one"),
    ("one", "object", 1), (None, "object", "one"),
    (pd.Timestamp("2020-01-01"), "datetime64[ns]", "2020-01-01"),
    ("one", "category", "one"),
])
def test_single_row_scalar_filter_matches_vector_oracle(value, dtype, filter_value):
    from graphistry.compute.chain_fast_paths import _verify_scalar_filters_on_hit
    frame = pd.DataFrame({"value": pd.Series([value], dtype=dtype)})
    filters = {"value": filter_value}
    doubled = pd.concat([frame, frame], ignore_index=True)
    try:
        expected = _verify_scalar_filters_on_hit(doubled, filters, Engine.PANDAS)
    except Exception as error:
        with pytest.raises(type(error)):
            _verify_scalar_filters_on_hit(frame, filters, Engine.PANDAS)
    else:
        actual = _verify_scalar_filters_on_hit(frame, filters, Engine.PANDAS)
        assert actual is not None and expected is not None
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.head(1).reset_index(drop=True))


def test_single_row_false_filter_still_validates_later_predicates():
    from graphistry.compute.chain_fast_paths import _verify_scalar_filters_on_hit
    from graphistry.compute.exceptions import GFQLSchemaError
    frame = pd.DataFrame({"first": [0], "second": [1]})
    with pytest.raises(GFQLSchemaError):
        _verify_scalar_filters_on_hit(frame, {"first": 1, "second": "wrong type"}, Engine.PANDAS)


@pytest.mark.parametrize("expression", [
    "coalesce(a.lhs, a.rhs)", "COALESCE(a.lhs,a.rhs)",
    "CoAlEsCe ( a.lhs , a.rhs )", "coalesce(a . lhs, a.rhs)",
])
def test_point_coalesce_shared_grammar_keeps_null_semantics(engine, expression):
    from graphistry.compute.chain_specializations.point_rows import _project_point_columns
    frame = pd.DataFrame({"lhs": pd.Series([None, 3], dtype="Int64"),
                          "rhs": pd.Series([7, 8], dtype="Int64")})
    if engine == "cudf":
        frame = pytest.importorskip("cudf").from_pandas(frame)
    result = _project_point_columns(frame, select([("answer", expression)]), "a", ["a"])
    assert result is not None
    assert topd(result)["answer"].tolist() == [7, 3]
    assert topd(frame)["lhs"].isna().tolist() == [True, False]


@pytest.mark.parametrize("expression", [
    "coalesce(DISTINCT a.lhs, a.rhs)", "coalesce(a.lhs)",
    "coalesce(a.lhs, a.rhs, a.lhs)", "coalesce(a.lhs, 0)",
    "coalesce(a.lhs, a.rhs) + 1", "coalesce(a.lhs.deep, a.rhs)",
    "coalesce(a.lhs, coalesce(a.rhs, a.lhs))", "coalesce(a.`lhs`, a.rhs)",
    "coalesce(a.lhs, b.rhs)", "coalesce(a.lhs, a.rhs", "coalesceX(a.lhs,a.rhs)",
])
def test_point_coalesce_shared_grammar_declines_outside_boundary(expression):
    from graphistry.compute.chain_specializations.point_rows import _project_point_columns
    frame = pd.DataFrame({"lhs": [1], "rhs": [2]})
    assert _project_point_columns(frame, select([("answer", expression)]), "a", ["a", "b"]) is None


@pytest.mark.parametrize("table,source", [("nodes", "b"), ("edges", "e")])
@pytest.mark.parametrize("seed", [0, 1, 98])
@pytest.mark.parametrize("disable_routes", [False, True])
def test_empty_and_populated_traversals_keep_binding_first(engine, table, source, seed, disable_routes):
    g = graph(engine)
    ops = [n({"key": seed}, name="a"), e_forward(name="e"), n(name="b"),
           rows(table=table, source=source)]
    with routes_off(ROUTES if disable_routes else ()):
        result = g.gfql(ops, engine=engine)
    edge_columns = ["eid", "e", "s", "d", "type", "weight"]
    expected_columns = (["key", "a", "b", "id", "kind", "value", "content"]
                        if table == "nodes" else edge_columns)
    expected_rows = {0: 2 if table == "nodes" else 3, 1: 1, 98: 0}
    assert list(result._nodes.columns) == expected_columns
    assert list(result._edges.columns) == edge_columns
    assert len(result._nodes) == expected_rows[seed]
    assert len(result._edges) == 0
