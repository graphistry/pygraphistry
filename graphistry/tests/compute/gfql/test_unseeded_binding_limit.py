"""Bound wide properties after an unfiltered walk without changing its row order."""
import numpy as np
import pytest

from graphistry.Engine import Engine, df_cons
import graphistry
from graphistry.tests.compute.gfql.index.test_float_property_index import assert_same_frame


@pytest.fixture(params=["pandas", "cudf", "polars", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def graph(engine, change=None):
    size = 500
    ids = np.arange(size)[::-1]
    if change == "duplicate":
        ids[1] = ids[0]
    nodes = df_cons(Engine(engine))({
        "id": ids, "a": [f"user-{i}" for i in range(size)],
        "wide": ["payload" * 20] * size, "value": np.arange(size),
    })
    edges = df_cons(Engine(engine))({
        "s": np.tile(np.arange(size), 4), "d": np.tile((np.arange(size) + 1) % size, 4),
        "e": np.arange(2000) + 10000, "weight": np.arange(2000, dtype=float),
    })
    if engine in ("pandas", "cudf"):
        edges.index = edges.index[::-1]
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _reference(g, query, engine, monkeypatch):
    import graphistry.compute.gfql.limit_bindings as route
    with monkeypatch.context() as patch:
        patch.setattr(route, "try_limited_bindings_state", lambda *a, **k: None)
        return g.gfql(query, engine=engine)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("projection", ["b", "b.id AS id", "a.a AS user, b.value AS value"])
@pytest.mark.parametrize("count", [0, 1, 5, 2500])
@pytest.mark.parametrize("engagement", [False, pytest.param(True, marks=pytest.mark.route_engaged("binding-limit", "cypher-fast"))])
def test_limited_entity_property_and_shadow_payload_keep_full_frame_order(
    engine, reverse, projection, count, engagement, monkeypatch,
):
    g = graph(engine)
    nodes, edges = g._nodes, g._edges
    pattern = "<-[e]-" if reverse else "-[e]->"
    query = f"MATCH (a){pattern}(b) RETURN {projection} LIMIT {count}"
    expected = _reference(g, query, engine, monkeypatch)
    assert len(expected._nodes) == min(count, len(edges))
    actual = g.gfql(query, engine=engine)
    assert_same_frame(actual._nodes, expected._nodes, engine)
    assert_same_frame(actual._edges, expected._edges, engine)
    assert g._nodes is nodes and g._edges is edges
    assert g._gfql_indexed_bindings_handoff is None
    assert actual._gfql_indexed_bindings_handoff is None
    if engagement:
        report = g.gfql_explain(query, engine=engine)
        served = any(s.get("seam") == "unseeded_binding_limit" and s.get("served")
                     for s in report["steps"])
        assert served is (engine in ("pandas", "cudf") and count < len(edges))


@pytest.mark.parametrize("suffix", [
    "b.id ORDER BY b.id DESC LIMIT 5", "DISTINCT b.id LIMIT 5", "count(*) AS n LIMIT 5",
    "b.id SKIP 2 LIMIT 5", "1 / b.value AS value LIMIT 5", "b.missing AS value LIMIT 5",
])
def test_unsafe_ordering_cardinality_and_expression_shapes_keep_existing_route(engine, suffix, monkeypatch):
    from graphistry.compute.exceptions import GFQLValidationError
    g = graph(engine)
    query = f"MATCH (a)-[e]->(b) RETURN {suffix}"
    def outcome(disabled):
        try:
            return _reference(g, query, engine, monkeypatch) if disabled else g.gfql(query, engine=engine)
        except (GFQLValidationError, NotImplementedError) as error:
            return error
    actual, expected = outcome(False), outcome(True)
    if isinstance(expected, Exception):
        assert type(actual) is type(expected)
        assert getattr(actual, "code", None) == getattr(expected, "code", None)
        assert getattr(actual, "context", None) == getattr(expected, "context", None)
        return
    assert_same_frame(actual._nodes, expected._nodes, engine)
    report = g.gfql_explain(query, engine=engine)
    assert not any(s.get("seam") == "unseeded_binding_limit" for s in report["steps"])


@pytest.mark.parametrize("change", ["duplicate", "seeded", "filtered", "typed", "multihop"])
def test_noncovered_graphs_and_patterns_keep_canonical_results(engine, change, monkeypatch):
    g = graph(engine, change)
    query = {
        "seeded": "MATCH (a {id: 499})-[e]->(b) RETURN b.id LIMIT 5",
        "filtered": "MATCH (a)-[e]->(b) WHERE b.value > 10 RETURN b.id LIMIT 5",
        "typed": "MATCH (a)-[e:TYPE]->(b) RETURN b.id LIMIT 5",
        "multihop": "MATCH (a)-[*1..2]->(b) RETURN b.id ORDER BY b.id LIMIT 5",
    }.get(change, "MATCH (a)-[e]->(b) RETURN b.id LIMIT 5")
    expected = _reference(g, query, engine, monkeypatch)
    actual = g.gfql(query, engine=engine)
    assert_same_frame(actual._nodes, expected._nodes, engine)
    report = g.gfql_explain(query, engine=engine)
    assert not any(s.get("seam") == "unseeded_binding_limit" for s in report["steps"])


@pytest.mark.route_engaged("binding-limit", "cypher-fast")
def test_public_limit_bounds_actual_wide_gathers_and_preserves_parallel_edges(engine, monkeypatch):
    if engine not in ("pandas", "cudf"):
        pytest.skip("initial LIMIT route is pandas/cuDF; native Polars parity is tested separately")
    import graphistry.compute.gfql.limit_bindings as route
    g = graph(engine)
    gathered = []
    real = route.take_rows
    def spy(frame, positions, requested):
        gathered.append(len(positions))
        return real(frame, positions, requested)
    monkeypatch.setattr(route, "take_rows", spy)
    result = g.gfql("MATCH (a)-[e]->(b) RETURN b.id LIMIT 5", engine=engine)
    assert len(result._nodes) == 5
    assert gathered == [5]
    assert list(result._nodes["b.id"].to_pandas() if engine == "cudf" else result._nodes["b.id"]) == [0, 0, 0, 0, 499]
    report = g.gfql_explain("MATCH (a)-[e]->(b) RETURN b.id LIMIT 5", engine=engine)
    assert report["used_index"] is False
    assert any(s.get("op") == "binding_limit" and s.get("served") for s in report["steps"])


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("policy", ["off", "use", "force"])
@pytest.mark.parametrize("edge_alias", [None, "e"])
def test_native_full_binding_payloads_and_index_policies_keep_canonical_answers(
    engine, bound, policy, edge_alias, monkeypatch,
):
    from graphistry.compute.ast import e_forward, limit, n, rows
    import graphistry.compute.gfql.limit_bindings as route
    g = graph(engine)
    if bound:
        g = g.gfql_index_all(engine=engine)
    ops = [n(name="a"), e_forward(name=edge_alias), n(name="b"), rows(), limit(5)]
    with monkeypatch.context() as patch:
        patch.setattr(route, "try_limited_bindings_state", lambda *a, **k: None)
        expected = g.gfql(ops, engine=engine, index_policy=policy)
    actual = g.gfql(ops, engine=engine, index_policy=policy)
    assert len(actual._nodes) == 5
    assert_same_frame(actual._nodes, expected._nodes, engine)
    assert_same_frame(actual._edges, expected._edges, engine)
    assert g._gfql_indexed_bindings_handoff is None
    assert actual._gfql_indexed_bindings_handoff is None


@pytest.mark.parametrize("requested", ["pandas", "cudf", "polars", "polars-gpu"])
def test_source_request_engine_conversions_preserve_canonical_outcomes(engine, requested, monkeypatch):
    if requested == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif requested != "pandas":
        pytest.importorskip(requested)
    g = graph(engine)
    query = "MATCH (a)-[e]->(b) RETURN b.id LIMIT 5"
    def outcome(disabled):
        try:
            return _reference(g, query, requested, monkeypatch) if disabled else g.gfql(
                query, engine=requested,
            )
        except NotImplementedError as error:
            return error
    actual, expected = outcome(False), outcome(True)
    if isinstance(expected, NotImplementedError):
        assert type(actual) is type(expected)
        assert type(actual.__cause__) is type(expected.__cause__)
        assert getattr(actual, "code", None) == getattr(expected, "code", None)
        assert getattr(actual, "context", None) == getattr(expected, "context", None)
    else:
        assert len(actual._nodes) == 5
        assert_same_frame(actual._nodes, expected._nodes, requested)
        assert_same_frame(actual._edges, expected._edges, requested)


def test_active_nested_policy_context_declines_the_limit_helper(monkeypatch):
    from graphistry.compute.ast import e_forward, limit, n, rows
    from graphistry.compute.gfql.call.executor import _thread_local
    from graphistry.compute.gfql.limit_bindings import try_limited_bindings_state
    monkeypatch.setattr(_thread_local, "policy", {}, raising=False)
    assert try_limited_bindings_state(
        graph("pandas"), [n(name="a"), e_forward(name="e"), n(name="b")],
        [rows(), limit(5)], Engine.PANDAS,
    ) is None


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("projection", [None, "b", "b.id AS id"])
@pytest.mark.parametrize("shadow", [False, True])
def test_bound_edge_identity_keeps_full_payload_and_input_bindings(
    engine, reverse, projection, shadow, monkeypatch,
):
    from graphistry.compute.ast import e_forward, e_reverse, limit, n, rows
    g = graph(engine)
    if not shadow:
        g = g.edges(g._edges.drop(columns=["e"]) if engine in ("pandas", "cudf") else g._edges.drop("e"))
    if engine in ("pandas", "cudf"):
        edges = g._edges.assign(eid=np.arange(len(g._edges)) + 30000)
    else:
        import polars as pl
        edges = g._edges.with_columns(pl.Series("eid", np.arange(len(g._edges)) + 30000))
    g = g.edges(edges).bind(edge="eid")
    nodes = g._nodes
    hop = e_reverse if reverse else e_forward
    query = ([n(name="a"), hop(name="e"), n(name="b"), rows(), limit(5)]
             if projection is None else
             f"MATCH (a){'<-[e]-' if reverse else '-[e]->'}(b) RETURN {projection} LIMIT 5")
    expected = _reference(g, query, engine, monkeypatch)
    actual = g.gfql(query, engine=engine)
    assert len(actual._nodes) == 5
    assert_same_frame(actual._nodes, expected._nodes, engine)
    assert_same_frame(actual._edges, expected._edges, engine)
    assert actual._edge == expected._edge
    assert g._edge == "eid" and g._nodes is nodes and g._edges is edges
    assert g._gfql_indexed_bindings_handoff is None
    assert actual._gfql_indexed_bindings_handoff is None


@pytest.mark.parametrize("domain", ["string", "uint64", "missing-endpoints"])
@pytest.mark.parametrize("edge_alias", [None, "e"])
def test_opaque_ids_and_missing_endpoints_keep_canonical_rows_and_schema(
    engine, domain, edge_alias, monkeypatch,
):
    from graphistry.compute.ast import e_forward, limit, n, rows
    ids = ([f"key-{i}" for i in range(500)] if domain == "string"
           else np.arange(500, dtype=np.uint64) + np.uint64(2**63 + 1))
    source = [ids[i % 500] for i in range(2000)]
    target = [ids[(i + 1) % 500] for i in range(2000)]
    if domain == "missing-endpoints":
        source[0], target[2] = 2**63 + 10000, 2**63 + 10001
    if domain != "string":
        source, target = np.asarray(source, dtype=np.uint64), np.asarray(target, dtype=np.uint64)
    g = graphistry.nodes(df_cons(Engine(engine))({"id": ids, "value": np.arange(500)}), "id").edges(
        df_cons(Engine(engine))({"s": source, "d": target, "weight": np.arange(2000)}), "s", "d",
    )
    query = [n(name="a"), e_forward(name=edge_alias), n(name="b"), rows(), limit(5)]
    expected = _reference(g, query, engine, monkeypatch)
    actual = g.gfql(query, engine=engine)
    assert len(actual._nodes) == 5
    assert_same_frame(actual._nodes, expected._nodes, engine)
    assert_same_frame(actual._edges, expected._edges, engine)


def test_unsigned_key_representation_keeps_canonical_results_or_errors(engine, monkeypatch):
    from graphistry.compute.ast import e_forward, limit, n, rows
    from graphistry.compute.exceptions import GFQLTypeError
    ids = np.arange(500, dtype=np.uint64) + (2**63 + 1)
    ends = np.asarray([ids[i % 500] for i in range(2000)], dtype=np.uint64)
    g = graphistry.nodes(df_cons(Engine(engine))({"id": ids}), "id").edges(
        df_cons(Engine(engine))({"s": ends, "d": ends, "weight": np.arange(2000)}), "s", "d",
    )
    query = [n(name="a"), e_forward(name="e"), n(name="b"), rows(), limit(5)]
    def outcome(disabled):
        try:
            return _reference(g, query, engine, monkeypatch) if disabled else g.gfql(query, engine=engine)
        except GFQLTypeError as error:
            return error
    actual, expected = outcome(False), outcome(True)
    if isinstance(expected, GFQLTypeError):
        assert type(actual) is type(expected)
        assert type(actual.__cause__) is type(expected.__cause__)
        assert actual.code == expected.code and actual.context == expected.context
    else:
        assert len(actual._nodes) == 5
        assert_same_frame(actual._nodes, expected._nodes, engine)
        assert_same_frame(actual._edges, expected._edges, engine)
