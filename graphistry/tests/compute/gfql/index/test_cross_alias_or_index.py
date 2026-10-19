"""Selective cross-alias OR preserves path bags and the canonical row suffix."""
import numpy as np
import pytest

from graphistry.Engine import Engine, df_cons
from graphistry.tests.compute.gfql.index.test_float_property_index import assert_same_frame
from graphistry.tests.compute.gfql.test_polars_membership_seeded_hop import graph


@pytest.fixture(params=["pandas", "cudf", "polars", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def indexed_graph(engine, reverse=False, bound_edge=True):
    g = graph(engine, reverse)
    names = np.where(np.isin(np.arange(5000), [1000, 1999]), "rare", "other")
    series = df_cons(Engine(engine))({"firstName": names})["firstName"]
    if engine.startswith("polars"):
        g = g.nodes(g._nodes.with_columns(series))
    else:
        g = g.nodes(g._nodes.assign(firstName=series))
    if not bound_edge:
        g = g.bind(edge=None)
    return g.gfql_index_all(engine=engine).gfql_index_node_props(["id", "firstName"], engine=engine)


def query(reverse=False, swap=False, binding=False, predicate=None):
    left = "m.key IN [0]" if binding else "m.id IN [10000]"
    right = "p.firstName = 'rare'"
    condition = predicate or (f"{right} OR {left}" if swap else f"{left} OR {right}")
    pattern = "<-[r:HAS_CREATOR]-" if reverse else "-[r:HAS_CREATOR]->"
    return f"MATCH (m:Message){pattern}(p:Person) WHERE {condition} RETURN m.id AS mid, p.id AS pid, r.eid AS eid, r.weight AS weight"


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("binding", [False, True])
@pytest.mark.parametrize("bound_edge", [False, True])
@pytest.mark.parametrize("engagement", [False, pytest.param(True, marks=pytest.mark.route_engaged("cross-alias-or", "indexed-kernel", "cypher-fast"))])
def test_or_overlap_parallel_edges_order_and_owned_gathers(engine, reverse, swap, binding, bound_edge, engagement, monkeypatch):
    import graphistry.compute.gfql.index.or_bindings as route
    g = indexed_graph(engine, reverse, bound_edge)
    nodes, edges = g._nodes, g._edges
    q = query(reverse, swap, binding)
    expected = g.gfql(q, engine=engine, index_policy="off")
    assert len(expected._nodes) == 4
    for policy in ["use", "force"]:
        actual = g.gfql(q, engine=engine, index_policy=policy)
        assert_same_frame(actual._nodes, expected._nodes, engine)
        assert_same_frame(actual._edges, expected._edges, engine)
        assert actual._gfql_indexed_bindings_handoff is None
    assert g._nodes is nodes and g._edges is edges
    if not engagement:
        return
    gathers = []
    lookup = route.lookup_edge_rows

    def spy(*args, **kwargs):
        result = lookup(*args, **kwargs)
        gathers.append(len(result[0]))
        return result

    monkeypatch.setattr(route, "lookup_edge_rows", spy)
    report = g.gfql_explain(q, engine=engine, index_policy="use")
    assert report["error"] is None
    assert any(step.get("seam") == "cross_alias_or" and step.get("served") for step in report["steps"])
    assert sorted(gathers) == [2, 4]


@pytest.mark.parametrize("change", ["off", "no-index", "no-property", "no-incoming", "stale-nodes", "stale-edges", "dense"])
def test_or_unsafe_or_unindexed_shapes_keep_canonical_execution(engine, change, monkeypatch):
    import graphistry.compute.gfql.index.or_bindings as route
    g = indexed_graph(engine)
    if change == "no-index":
        g = g.drop_index()
    elif change == "no-property":
        g = g.drop_index("node_prop", column="firstName")
    elif change == "no-incoming":
        g = g.drop_index("edge_in_adj")
    elif change == "stale-nodes":
        g = g.nodes(g._nodes.clone() if engine.startswith("polars") else g._nodes.copy())
    elif change == "stale-edges":
        g = g.edges(g._edges.reverse() if engine.startswith("polars") else g._edges.iloc[::-1])
    q = query(predicate="m.id IN [10000] OR p.firstName = 'other'" if change == "dense" else None)
    policy = "off" if change == "off" else "use"
    actual = g.gfql(q, engine=engine, index_policy=policy)
    report = g.gfql_explain(q, engine=engine, index_policy=policy)
    assert not any(step.get("seam") == "cross_alias_or" and step.get("served") for step in report["steps"])
    monkeypatch.setattr(route, "prepare_indexed_or_bindings", lambda *args, **kwargs: None)
    expected = g.gfql(q, engine=engine, index_policy=policy)
    assert_same_frame(actual._nodes, expected._nodes, engine)
    assert_same_frame(actual._edges, expected._edges, engine)


@pytest.mark.parametrize("predicate,params", [
    ("m.id IN [] OR p.firstName = 'rare'", None),
    ("m.id IN [10000, NULL] OR p.firstName = 'rare'", None),
    ("m.id IN [10000, '10000'] OR p.firstName = 'rare'", None),
    ("m.id = '10000' OR p.firstName = 'rare'", None),
    ("m.id IN [TRUE] OR p.firstName = 'rare'", None),
    ("m.id IN [999999] OR p.firstName = 'missing'", None),
    ("m.id IN $ids OR p.firstName = $name", {"ids": [10000], "name": "rare"}),
    ("m.id IN $ids OR p.firstName = $name", {"ids": [10000]}),
    ("m.missing IN [10000] OR p.firstName = 'rare'", None),
])
def test_or_boundaries_keep_canonical_results_and_structured_errors(engine, predicate, params, monkeypatch):
    from graphistry.compute.exceptions import GFQLValidationError
    import graphistry.compute.gfql.index.or_bindings as route
    g = indexed_graph(engine)
    q = query(predicate=predicate)

    def outcome(policy):
        try:
            return g.gfql(q, engine=engine, index_policy=policy, params=params)
        except (GFQLValidationError, TypeError, ValueError, OverflowError, NotImplementedError) as error:
            return error

    for policy in ["off", "use", "force"]:
        actual = outcome(policy)
        with monkeypatch.context() as patch:
            patch.setattr(route, "prepare_indexed_or_bindings", lambda *args, **kwargs: None)
            expected = outcome(policy)
        if isinstance(expected, Exception):
            assert type(actual) is type(expected)
            assert getattr(actual, "code", None) == getattr(expected, "code", None)
            assert getattr(actual, "context", None) == getattr(expected, "context", None)
        else:
            assert_same_frame(actual._nodes, expected._nodes, engine)
            assert_same_frame(actual._edges, expected._edges, engine)


@pytest.mark.parametrize("returns", [
    "count(*) AS count",
    "DISTINCT p.id AS pid",
    "m.id AS mid, p.id AS pid ORDER BY mid DESC, pid SKIP 1 LIMIT 2",
])
def test_or_canonical_aggregate_distinct_and_order_limit_suffix(engine, returns):
    g = indexed_graph(engine)
    q = query().split(" RETURN ")[0] + " RETURN " + returns
    expected = g.gfql(q, engine=engine, index_policy="off")
    assert len(expected._nodes) > 0
    for policy in ["use", "force"]:
        actual = g.gfql(q, engine=engine, index_policy=policy)
        assert_same_frame(actual._nodes, expected._nodes, engine)
        assert_same_frame(actual._edges, expected._edges, engine)


def test_or_float_binding_ids_decline_before_branch_gathers(engine, monkeypatch):
    import graphistry.compute.gfql.index.or_bindings as route
    g = indexed_graph(engine)
    if engine.startswith("polars"):
        import polars as pl
        g = g.nodes(g._nodes.with_columns(pl.col("key").cast(pl.Float64)))
        g = g.edges(g._edges.with_columns(pl.col("s").cast(pl.Float64), pl.col("d").cast(pl.Float64)))
    else:
        g = g.nodes(g._nodes.assign(key=g._nodes["key"].astype("float64")))
        g = g.edges(g._edges.assign(s=g._edges["s"].astype("float64"), d=g._edges["d"].astype("float64")))
    g = g.gfql_index_all(engine=engine).gfql_index_node_props(["id", "firstName"], engine=engine)
    expected = g.gfql(query(), engine=engine, index_policy="off")
    assert len(expected._nodes) == 4
    gathers = []
    lookup = route.lookup_edge_rows

    def spy(*args, **kwargs):
        result = lookup(*args, **kwargs)
        gathers.append(len(result[0]))
        return result

    monkeypatch.setattr(route, "lookup_edge_rows", spy)
    actual = g.gfql(query(), engine=engine, index_policy="use")
    assert not gathers
    assert_same_frame(actual._nodes, expected._nodes, engine)


@pytest.mark.parametrize("requested", ["pandas", "cudf", "polars", "polars-gpu"])
def test_or_explicit_engine_conversion_keeps_canonical_execution(engine, requested, monkeypatch):
    if requested == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif requested != "pandas":
        pytest.importorskip(requested)
    import graphistry.compute.gfql.index.or_bindings as route
    g = indexed_graph(engine)
    def outcome():
        try:
            return g.gfql(query(), engine=requested, index_policy="use")
        except NotImplementedError as error:
            return error

    actual = outcome()
    monkeypatch.setattr(route, "prepare_indexed_or_bindings", lambda *args, **kwargs: None)
    expected = outcome()
    if isinstance(expected, NotImplementedError):
        assert isinstance(actual, NotImplementedError)
        assert type(actual.__cause__) is type(expected.__cause__)
        assert getattr(actual, "code", None) == getattr(expected, "code", None)
        assert getattr(actual, "context", None) == getattr(expected, "context", None)
        return
    assert len(actual._nodes) == len(expected._nodes) == 4
    assert_same_frame(actual._nodes, expected._nodes, requested)
    assert_same_frame(actual._edges, expected._edges, requested)


@pytest.mark.parametrize("engagement", [False, pytest.param(True, marks=pytest.mark.route_engaged("cross-alias-or", "indexed-kernel", "cypher-fast"))])
def test_or_unlabeled_seed_remains_bounded_and_canonical(engine, engagement):
    g = indexed_graph(engine)
    q = query().replace(":Message", "").replace(":Person", "")
    expected = g.gfql(q, engine=engine, index_policy="off")
    assert len(expected._nodes) == 4
    for policy in ["use", "force"]:
        actual = g.gfql(q, engine=engine, index_policy=policy)
        assert_same_frame(actual._nodes, expected._nodes, engine)
        assert_same_frame(actual._edges, expected._edges, engine)
    if engagement:
        report = g.gfql_explain(q, engine=engine, index_policy="use")
        assert any(step.get("seam") == "cross_alias_or" and step.get("served") for step in report["steps"])


@pytest.mark.parametrize("alias", ["m", "p", "r"])
def test_or_alias_named_like_payload_preserves_original_values(engine, alias):
    g = indexed_graph(engine)
    if engine.startswith("polars"):
        import polars as pl
        if alias == "r":
            g = g.edges(g._edges.with_columns((pl.col("eid") + 5000).alias(alias)))
        else:
            g = g.nodes(g._nodes.with_columns((pl.col("key") + 5000).alias(alias)))
    elif alias == "r":
        g = g.edges(g._edges.assign(**{alias: g._edges["eid"] + 5000}))
    else:
        g = g.nodes(g._nodes.assign(**{alias: g._nodes["key"] + 5000}))
    g = g.gfql_index_all(engine=engine).gfql_index_node_props(["id", "firstName"], engine=engine)
    q = query().split(" RETURN ")[0] + f" RETURN {alias}.{alias} AS value, r.eid AS eid"
    expected = g.gfql(q, engine=engine, index_policy="off")
    assert len(expected._nodes) == 4
    for policy in ["use", "force"]:
        actual = g.gfql(q, engine=engine, index_policy=policy)
        assert_same_frame(actual._nodes, expected._nodes, engine)
        assert_same_frame(actual._edges, expected._edges, engine)
