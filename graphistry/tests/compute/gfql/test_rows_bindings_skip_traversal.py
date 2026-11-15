"""A boundary rows() call that rebuilds bindings from the base graph skips the unread middle traversal."""
import pytest

import graphistry
from graphistry import e_forward, e_reverse, e_undirected, n
from graphistry.Engine import Engine, df_cons
from graphistry.compute.ast import rows, select
from graphistry.compute.gfql.index.api import index_trace
from graphistry.tests.compute.gfql.index.test_float_property_index import assert_same_frame
from graphistry.tests.compute.gfql.routes.switch import routes_off

SEAM = "rows_bindings_skip_traversal"
ROUTE = "bindings-skip-traversal"


@pytest.fixture(params=["pandas", "cudf", "polars", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def graph(engine):
    cons = df_cons(Engine(engine))
    nodes = cons({
        "id": [1, 2, 3, 4, 5, 6, 7],
        "label__Person": [True, True, True, True, False, False, True],
        "label__City": [False, False, False, False, True, True, False],
        "name": ["a", "b", None, "d", "x", "y", "lonely"],
        "score": [1.5, None, 3.0, 4.0, None, None, 7.0],
    })
    nodes = nodes.assign(blank=nodes["score"] * float("nan")) if engine in ("pandas", "cudf") else (
        nodes.with_columns(blank=nodes["score"] * None))
    edges = cons({
        "s": [1, 2, 3, 1, 2, 4, 4, 3],
        "d": [5, 5, 6, 2, 3, 1, 2, 4],
        "type": ["LOC", "LOC", "LOC", "KNOWS", "KNOWS", "KNOWS", "KNOWS", "KNOWS"],
        "since": [10, 20, 30, 40, 50, 60, 70, 80],
    })
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


PERSON = {"label__Person": True}
SHAPES = {
    "seeded_forward_select": (
        [n({"id": 1, **PERSON}, name="p"), e_forward({"type": "LOC"}), n({"label__City": True}, name="c"),
         rows(), select([("pid", "p.id"), ("pname", "p.name"), ("city", "c.name")])],
        1,
    ),
    "seeded_undirected_edge_alias": (
        [n({"id": 4, **PERSON}, name="p"), e_undirected({"type": "KNOWS"}, name="r"), n(PERSON, name="f"),
         rows(), select([("fid", "f.id"), ("since", "r.since"), ("score", "f.score")])],
        3,
    ),
    "unseeded_two_hop": (
        [n(PERSON, name="a"), e_forward({"type": "KNOWS"}), n(PERSON, name="b"), e_forward({"type": "LOC"}),
         n(name="c"), rows(), select([("a", "a.id"), ("b", "b.id"), ("c", "c.id")])],
        4,
    ),
    "reverse_all_null_property": (
        [n({"label__City": True}, name="c"), e_reverse({"type": "LOC"}), n(name="p"),
         rows(), select([("c", "c.id"), ("p", "p.id"), ("blank", "p.blank")])],
        3,
    ),
    "seed_absent_empty": (
        [n({"id": 99}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])],
        0,
    ),
    "explicit_binding_ops": (
        [n({"id": 1}, name="p"), e_forward({"type": "KNOWS"}, name="r"), n(name="q"),
         rows(binding_ops=[n({"id": 1}, name="p").to_json(), e_forward({"type": "KNOWS"}, name="r").to_json(),
                           n(name="q").to_json()]),
         select([("q", "q.id"), ("since", "r.since")])],
        1,
    ),
}
CYPHER = {
    "multi_alias_return": ("MATCH (p {id: 4})-[r]-(f) RETURN f.id AS fid, r.since AS since ORDER BY fid", 3),
    "var_length_zero_hop": ("MATCH (p)-[*0..2]->(q) WHERE p.id = 7 RETURN p.id AS p, q.id AS q", 1),
    "single_alias_prefilter": (
        "MATCH (p)-[r]->(q) WHERE p.score > 2 RETURN p.id AS p, q.id AS q ORDER BY p, q", 4),
}


def _run(g, query, engine):
    return g.gfql(query, engine=engine, index_policy="off")


def _consulted(g, query, engine, index_policy="off"):
    with index_trace() as steps:
        g.gfql(query, engine=engine, index_policy=index_policy)
    return {bool(s["served"]) for s in steps if s.get("op") == "fast_path" and s.get("seam") == SEAM}


def _assert_matches_traversal(g, query, engine, expected_rows):
    with routes_off([ROUTE]):
        expected = _run(g, query, engine)
    assert len(expected._nodes) == expected_rows
    actual = _run(g, query, engine)
    assert_same_frame(actual._nodes, expected._nodes, engine)
    assert_same_frame(actual._edges, expected._edges, engine)


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_chain_rows_match_the_traversal_path(engine, shape):
    query, expected_rows = SHAPES[shape]
    _assert_matches_traversal(graph(engine), query, engine, expected_rows)


@pytest.mark.parametrize("shape", sorted(CYPHER))
def test_cypher_rows_match_the_traversal_path(engine, shape):
    query, expected_rows = CYPHER[shape]
    _assert_matches_traversal(graph(engine), query, engine, expected_rows)


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_chain_rows_skip_the_traversal(engine, shape):
    assert _consulted(graph(engine), SHAPES[shape][0], engine) == {True}


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("shape", sorted(CYPHER))
def test_cypher_rows_skip_the_traversal(engine, shape):
    assert _consulted(graph(engine), CYPHER[shape][0], engine) == {True}


DECLINED = {
    "unnamed_middle_reads_traversal_nodes": (
        [n({"id": 1}), e_forward(), n(), rows()], 3),
    "edge_table_reads_traversal_edges": (
        [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(table="edges")], 2),
    "source_alias_reads_traversal_nodes": (
        [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(source="q")], 2),
}


@pytest.mark.parametrize("shape", sorted(DECLINED))
def test_rows_that_read_the_traversal_keep_it(engine, shape):
    query, expected_rows = DECLINED[shape]
    g = graph(engine)
    assert _consulted(g, query, engine) == set()
    assert len(_run(g, query, engine)._nodes) == expected_rows


UNBOUND_QUERY = [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]


@pytest.mark.route_engaged(ROUTE)
def test_unbound_node_frame_declines_the_skip(engine):
    assert _consulted(graphistry.edges(graph(engine)._edges, "s", "d"), UNBOUND_QUERY, engine) == {False}


def test_unbound_node_frame_keeps_the_traversal(engine):
    edges_only = graphistry.edges(graph(engine)._edges, "s", "d")
    query = UNBOUND_QUERY
    with routes_off([ROUTE]):
        expected = _run(edges_only, query, engine)
    assert len(expected._nodes) == 2
    assert_same_frame(_run(edges_only, query, engine)._nodes, expected._nodes, engine)


POLICIES = [("off", True), ("use", False), ("auto", False), ("force", False)]
NAMED_SEED = [n({"name": "a"}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("index_policy,served", POLICIES)
def test_index_policy_that_may_serve_keeps_the_traversal(engine, index_policy, served):
    g = graph(engine).create_index("node_prop", column="name", engine=engine)
    assert _consulted(g, NAMED_SEED, engine, index_policy) == {served}


@pytest.mark.parametrize("index_policy", [policy for policy, _ in POLICIES])
def test_every_index_policy_returns_the_same_rows(engine, index_policy):
    g = graph(engine).create_index("node_prop", column="name", engine=engine)
    expected = g.gfql(NAMED_SEED, engine=engine, index_policy="off")
    assert len(expected._nodes) == 2
    assert_same_frame(g.gfql(NAMED_SEED, engine=engine, index_policy=index_policy)._nodes, expected._nodes, engine)


@pytest.mark.route_engaged(ROUTE)
def test_default_policy_without_indexes_skips(engine):
    query = [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]
    assert _consulted(graph(engine), query, engine, "use") == {True}


def test_policy_keeps_the_traversal():
    g = graph("pandas")
    seen = []
    query = [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]
    with index_trace() as steps:
        out = g.gfql(query, engine="pandas", policy={"prechain": lambda ctx: seen.append(ctx.get("phase"))})
    assert not any(s.get("seam") == SEAM for s in steps)
    assert seen and len(out._nodes) == 2


@pytest.mark.parametrize("validate_schema", [True, False])
@pytest.mark.parametrize("bad_filter", [{"name": 5}, {"score": "high"}, {"missing": 1}])
def test_filter_errors_match_the_traversal(engine, bad_filter, validate_schema):
    g = graph(engine)
    query = [n(bad_filter, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]

    def outcome():
        try:
            out = g.gfql(query, engine=engine, index_policy="off", validate_schema=validate_schema)
        except Exception as error:  # the outcome under test is the error class and code
            return type(error), getattr(error, "code", None)
        return "rows", len(out._nodes)

    with routes_off([ROUTE]):
        expected = outcome()
    assert outcome() == expected


def test_cached_zero_row_twin_follows_input_schema_and_clears(engine):
    from graphistry.compute.gfql_unified import gfql_clear_caches

    query = SHAPES["seeded_undirected_edge_alias"][0]
    g = graph(engine)
    _assert_matches_traversal(g, query, engine, 3)
    widened = g.edges(g._edges.with_columns(weight=g._edges["since"] * 2) if engine.startswith("polars")
                      else g._edges.assign(weight=g._edges["since"] * 2))
    _assert_matches_traversal(widened, query, engine, 3)
    assert "weight" in _run(widened, query, engine)._edges.columns
    gfql_clear_caches()
    _assert_matches_traversal(g, query, engine, 3)
