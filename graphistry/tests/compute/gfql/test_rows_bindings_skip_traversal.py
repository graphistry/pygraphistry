"""A boundary rows() call that rebuilds bindings from the base graph skips the unread middle traversal."""
import inspect
import os

import pytest

import graphistry
from graphistry import e_forward, e_reverse, e_undirected, n
from graphistry.Engine import Engine, df_cons
from graphistry.compute.ast import rows, select
from graphistry.compute.gfql.index.api import index_trace, with_index_policy
from graphistry.compute.predicates.numeric import gt
from graphistry.tests.compute.gfql.index.test_float_property_index import assert_same_frame
from graphistry.tests.compute.gfql.routes.switch import routes_off

SEAM = "rows_bindings_skip_traversal"
ROUTE = "bindings-skip-traversal"


@pytest.fixture(params=[
    "pandas",
    "polars",
    pytest.param("cudf", marks=pytest.mark.skipif(os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1")),
    pytest.param("polars-gpu", marks=pytest.mark.skipif(
        os.environ.get("TEST_POLARS_GPU") != "1", reason="requires actual polars-gpu")),
])
def engine(request):
    if request.param != "pandas":
        pytest.importorskip("cudf_polars" if request.param == "polars-gpu" else request.param)
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
@pytest.mark.parametrize("bad_filter", [
    {"name": 5}, {"score": "high"}, {"missing": 1}, {"name": gt(1)},
])
@pytest.mark.parametrize("seed", [1, 99, None])
@pytest.mark.parametrize("string_ids", [False, True])
def test_filter_errors_match_the_traversal(engine, bad_filter, validate_schema, seed, string_ids):
    g = graph(engine)
    if string_ids:
        g = g.nodes(g._nodes.with_columns(g._nodes["id"].cast(str)) if engine.startswith("polars")
                    else g._nodes.assign(id=g._nodes["id"].astype(str)))
    g = with_index_policy(g, "off")
    first = n(name="p") if seed is None else n({"id": seed}, name="p")
    query = [first, e_forward(), n(bad_filter, name="q"), rows(), select([("q", "q.id")])]

    def outcome():
        try:
            out = g.chain(query, engine=engine, validate_schema=validate_schema)
        except Exception as error:  # the outcome under test is the error class and code
            return type(error), getattr(error, "code", None)
        return "rows", len(out._nodes)

    from graphistry.compute.chain import chain as chain_impl
    assert "validate_schema" in inspect.signature(inspect.unwrap(chain_impl)).parameters
    with routes_off([ROUTE]):
        expected = outcome()
    assert outcome() == expected


VAR_LENGTH = [
    "MATCH (p)-[*0..2]->(q) WHERE p.id = 7 RETURN p.id AS p, q.id AS q",
    "MATCH (p)-[*1..3]->(q) WHERE p.id = 1 RETURN p.id AS p, q.id AS q",
]


@pytest.mark.parametrize("query", VAR_LENGTH)
def test_variable_length_keeps_its_traversal_schema(engine, query):
    _assert_matches_traversal(graph(engine), query, engine, len(_run(graph(engine), query, engine)._nodes))


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("query", VAR_LENGTH)
def test_variable_length_declines_the_skip(engine, query):
    assert _consulted(graph(engine), query, engine) == {False}


def _widened(g, engine):
    since = g._edges["since"] * 2
    return g.edges(g._edges.with_columns(weight=since) if engine.startswith("polars") else g._edges.assign(weight=since))


def test_zero_row_twin_follows_input_schema(engine):
    query = SHAPES["seeded_undirected_edge_alias"][0]
    g = graph(engine)
    _assert_matches_traversal(g, query, engine, 3)
    _assert_matches_traversal(_widened(g, engine), query, engine, 3)
    assert "weight" in _run(_widened(g, engine), query, engine)._edges.columns


@pytest.mark.route_engaged(ROUTE)
def test_zero_row_twin_cache_fills_and_clears(engine):
    from graphistry.compute.gfql import rows_skip_traversal
    from graphistry.compute.gfql_unified import gfql_clear_caches

    gfql_clear_caches()
    assert not rows_skip_traversal._TWINS
    _run(graph(engine), SHAPES["seeded_undirected_edge_alias"][0], engine)
    assert len(rows_skip_traversal._TWINS) == 1
    _run(graph(engine), SHAPES["seeded_undirected_edge_alias"][0], engine)
    assert len(rows_skip_traversal._TWINS) == 1
    gfql_clear_caches()
    assert not rows_skip_traversal._TWINS


def test_cached_twin_keeps_each_graphs_categories():
    import pandas as pd

    query = SHAPES["seeded_undirected_edge_alias"][0]
    results = []
    for categories in (["x", "y"], ["p", "q", "z"]):
        g = graph("pandas")
        kinds = pd.Categorical([categories[i % 2] for i in range(len(g._edges))], categories=categories)
        g = g.edges(g._edges.assign(kind=kinds))
        _assert_matches_traversal(g, query, "pandas", 3)
        results.append(list(_run(g, query, "pandas")._edges["kind"].cat.categories))
    assert results == [["x", "y"], ["p", "q", "z"]]


def test_column_named_schema_keys_on_dtypes():
    query = SHAPES["seeded_undirected_edge_alias"][0]
    for values in ([1] * 8, ["a"] * 8):
        g = graph("pandas")
        g = g.edges(g._edges.assign(schema=values))
        _assert_matches_traversal(g, query, "pandas", 3)


def test_explain_trace_does_not_depend_on_the_twin_cache(engine):
    from graphistry.compute.gfql_unified import gfql_clear_caches

    query = SHAPES["seeded_forward_select"][0]
    g = graph(engine)
    gfql_clear_caches()
    cold = g.gfql_explain(query, engine=engine, index_policy="off")["steps"]
    warm = g.gfql_explain(query, engine=engine, index_policy="off")["steps"]
    assert cold == warm


def test_prefix_call_keeps_the_traversal():
    from graphistry.compute.ast import call

    g = graph("pandas")
    query = [call("get_degrees"), n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]
    with routes_off(["polars-point-rows"]), index_trace() as steps:
        out = g.gfql(query, engine="pandas", index_policy="off")
    assert not any(s.get("seam") == SEAM for s in steps)
    assert len(out._nodes) == 2


@pytest.mark.parametrize("unbind", ["node_column", "edges"])
def test_partially_bound_graph_matches_the_traversal(engine, unbind):
    g = graph(engine)
    g = g.bind(node=None) if unbind == "node_column" else graphistry.nodes(g._nodes, "id")
    query = [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]

    def outcome():
        try:
            return "rows", _run(g, query, engine)._nodes.shape
        except Exception as error:  # the outcome under test is the error class and code
            return type(error), getattr(error, "code", None)

    with routes_off([ROUTE]):
        expected = outcome()
    assert outcome() == expected


def test_default_engine_twins_stay_per_frame_type(engine):
    query = SHAPES["seeded_undirected_edge_alias"][0]
    graph("pandas").gfql(query, index_policy="off")
    g = graph(engine)
    with routes_off([ROUTE]):
        expected = g.gfql(query, index_policy="off")
    actual = g.gfql(query, index_policy="off")
    assert type(actual._edges) is type(g._edges)
    assert_same_frame(actual._edges, expected._edges, engine)
    assert_same_frame(actual._nodes, expected._nodes, engine)


def test_cached_twin_drops_attrs_and_keeps_category_order():
    import pandas as pd

    query = SHAPES["seeded_undirected_edge_alias"][0]
    for tenant, categories in (("A", ["x", "y"]), ("B", ["y", "x"])):
        g = graph("pandas")
        edges = g._edges.assign(kind=pd.Categorical(["x", "y"] * 4, categories=categories))
        edges.attrs = {"tenant": tenant}
        g = g.edges(edges)
        _assert_matches_traversal(g, query, "pandas", 3)
        out = _run(g, query, "pandas")._edges
        with routes_off([ROUTE]):
            expected = _run(g, query, "pandas")._edges
        assert list(out["kind"].cat.categories) == categories
        assert out.attrs == expected.attrs


@pytest.mark.parametrize("query_field", ["edge_query", "destination_node_query"])
def test_query_strings_keep_the_traversal(query_field):
    g = graph("pandas")
    g = g.edges(g._edges.assign(mixed=[1, "a", 2, "b", 3, "c", 4, "d"]))
    query = [n({"id": 99}, name="p"), e_forward(**{query_field: "mixed > 0" if query_field == "edge_query" else "id > 0"}),
             n(name="q"), rows(), select([("q", "q.id")])]

    def outcome():
        try:
            return "rows", len(_run(g, query, "pandas")._nodes)
        except Exception as error:  # the outcome under test is the error class
            return type(error), None

    with routes_off([ROUTE]):
        expected = outcome()
    assert outcome() == expected
