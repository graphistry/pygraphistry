"""A query must not mutate the ``Plottable`` it was run on (#1786).

``WITH`` re-entry seeds the follow-up ``MATCH`` from the carried nodes. That seed used to be
assigned to the CALLER's graph and never cleared, so the next -- entirely unrelated -- query on
the same object was answered against the stale seed: no error, just the previous query's count.
Engine-independent (wrong on pandas too), so an engine A/B never surfaces it.

Everything here is written to FAIL if the leak comes back: the assertions compare a query run on
a re-used graph against the SAME query run on a pristine one, and pin the absolute counts so a
fix that merely makes both sides equally wrong is caught too.
"""
import os

import pandas as pd
import pytest

import graphistry


def _engines():
    out = ["pandas"]
    try:
        import polars  # noqa: F401
        out.append("polars")
    except Exception:
        pass
    return out


ENGINES = _engines()

# The #1786 repro graph. `kind` alternates a/b/a/b/a/b/a, so `{kind:'a'}` selects 4 of 7 nodes.
NODES = pd.DataFrame({"id": list(range(7)), "kind": ["a", "b"] * 3 + ["a"]})
EDGES = pd.DataFrame(
    [(0, 3), (1, 2), (4, 6), (3, 5), (5, 6), (0, 6), (3, 4)], columns=["s", "d"]
)
ADVERSARIAL_EDGES = pd.concat(
    [EDGES, pd.DataFrame([(6, 1)], columns=["s", "d"])], ignore_index=True
)
PARALLEL_NODES = pd.DataFrame({"id": [1, 2, 3]})
PARALLEL_EDGES = pd.DataFrame({"s": [1, 1, 2], "d": [2, 2, 3]})

PLAIN = "MATCH (a)-[*]->(b) RETURN count(*) AS c"
REENTRY = "MATCH (a {kind:'a'}) WITH a MATCH (a)-[*]->(b) RETURN count(*) AS c"
NESTED = (
    "MATCH (a {kind:'a'}) WITH a MATCH (a)-[]->(b) WITH b MATCH (b)-[]->(c) "
    "RETURN count(*) AS c"
)
DUPLICATE_NESTED_NODES = pd.DataFrame({"id": [1, 2, 3]})
DUPLICATE_NESTED_EDGES = pd.DataFrame({"s": [1, 1, 2], "d": [2, 2, 3]})
DUPLICATE_NESTED = (
    "MATCH (a) WITH a MATCH (a)-[]->(b) WITH b MATCH (b)-[]->(c) "
    "RETURN count(*) AS c"
)

# Independently established on this fixed graph (identical on both engines).
PLAIN_COUNT = 13
REENTRY_COUNT = 7
NESTED_COUNT = 2


def _g():
    """A PRISTINE graph. Every oracle needs one: re-using a graph is the thing under test."""
    return graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")


def _count(g, query, engine):
    return int(g.gfql(query, engine=engine)._nodes["c"].to_list()[0])


def _aggregate_rows(g, query, engine="pandas"):
    nodes = g.gfql(query, engine=engine)._nodes
    to_dicts = getattr(nodes, "to_dicts", None)
    return to_dicts() if to_dicts is not None else nodes.to_dict(orient="records")


@pytest.mark.parametrize("engine", ENGINES)
def test_reentry_then_unrelated_query_is_unaffected(engine):
    """THE regression: two independent queries, one graph. The second must not see the first."""
    g = _g()
    assert _count(g, REENTRY, engine) == REENTRY_COUNT
    assert _count(g, PLAIN, engine) == PLAIN_COUNT
    # ... and identical to the same query on a graph that never ran the re-entry query.
    assert _count(g, PLAIN, engine) == _count(_g(), PLAIN, engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_reentry_leaves_no_execution_state_on_caller_graph(engine):
    """Directly: the per-execution fields on the caller's object are untouched."""
    g = _g()
    g.gfql(REENTRY, engine=engine)
    assert g._gfql_start_nodes is None
    assert g._gfql_rows_base_graph is None


@pytest.mark.parametrize("engine", ENGINES)
def test_reentry_still_works(engine):
    """Do not "fix" the leak by breaking WITH: the re-entry answer itself must survive."""
    assert _count(_g(), REENTRY, engine) == REENTRY_COUNT
    assert REENTRY_COUNT != PLAIN_COUNT  # else the test above proves nothing


@pytest.mark.parametrize("engine", ENGINES)
def test_repeated_reentry_is_stable(engine):
    """Re-entry twice on one graph: the second run must not inherit the first's seed."""
    g = _g()
    assert [_count(g, REENTRY, engine) for _ in range(3)] == [REENTRY_COUNT] * 3


@pytest.mark.parametrize("engine", ENGINES)
def test_nested_reentry(engine):
    """Two WITH boundaries in one query, then an unrelated query on the same graph.

    The original graph's node 6 is duplicated in the carried prefix but has no
    outgoing arm, so the old Polars route passed count 2 by coincidence. This
    graph keeps the historical count and caller-isolation controls; the adversarial
    multiplicity case is pinned separately. Polars still declines this multi-hop
    row-pipeline shape.
    """
    g = _g()
    if engine == "polars":
        with pytest.raises(NotImplementedError, match="cypher row op 'rows'"):
            _count(g, NESTED, engine)
    else:
        assert _count(g, NESTED, engine) == NESTED_COUNT
    assert g._gfql_start_nodes is None
    assert _count(g, PLAIN, engine) == PLAIN_COUNT


def test_nested_reentry_duplicate_prefix_declines_polars():
    """A duplicate-bearing nested control for the typed Polars boundary.

    The first traversal produces duplicate b=2 rows before the second WITH.
    Polars still declines this unsupported row-pipeline shape rather than
    relying on those duplicates having no outgoing arm.
    """
    pytest.importorskip("polars")
    g = graphistry.nodes(DUPLICATE_NESTED_NODES, "id").edges(
        DUPLICATE_NESTED_EDGES, "s", "d"
    )
    with pytest.raises(NotImplementedError, match="cypher row op 'rows'"):
        _count(g, DUPLICATE_NESTED, "polars")


@pytest.mark.parametrize("engine", ["pandas"])
def test_nested_reentry_preserves_seed_row_multiplicity(engine):
    """A duplicated reentry seed is a row, not the same identity twice.

    The first traversal produces b ids [3, 6, 3, 6, 1]. Direct enumeration
    therefore has five (b,c) rows: (3,5), (6,1), (3,4), (6,1), (1,2).
    """
    graph = graphistry.nodes(NODES, "id").edges(ADVERSARIAL_EDGES, "s", "d")
    projection = graph.gfql(
        "MATCH (a {kind:'a'}) WITH a MATCH (a)-[]->(b) WITH b "
        "MATCH (b)-[]->(c) RETURN b, c",
        engine=engine,
    )._nodes
    assert sorted(projection["b.id"].to_list()) == [1, 3, 3, 6, 6]
    assert _count(graph, NESTED, engine) == 5


@pytest.mark.parametrize("engine", ["pandas"])
def test_nested_reentry_distinguishes_bag_seed_from_distinct(engine):
    """WITH DISTINCT is an explicit contraction; plain WITH keeps both seed rows."""
    graph = graphistry.nodes(PARALLEL_NODES, "id").edges(PARALLEL_EDGES, "s", "d")
    bag = (
        "MATCH (a) WITH a MATCH (a)-[]->(b) WITH b "
        "MATCH (b)-[]->(c) RETURN count(*) AS c"
    )
    distinct = (
        "MATCH (a) WITH a MATCH (a)-[]->(b) WITH DISTINCT b "
        "MATCH (b)-[]->(c) RETURN count(*) AS c"
    )
    assert _count(graph, bag, engine) == 2
    assert _count(graph, distinct, engine) == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_nested_reentry_binds_incoming_edges_as_one_bag(engine):
    """A global aggregate runs after the combined multi-row reentry bag."""
    graph = graphistry.nodes(PARALLEL_NODES, "id").edges(PARALLEL_EDGES, "s", "d")
    query = (
        "MATCH (a {id:1}) WITH a MATCH (a)-[]->(b) WITH b "
        "MATCH (c)-[]->(b) RETURN count(*) AS c"
    )
    assert _aggregate_rows(graph, query, engine) == [{"c": 6}]
    projection = graph.gfql(
        "MATCH (a {id:1}) WITH a MATCH (a)-[]->(b) WITH b "
        "MATCH (c)-[]->(b) RETURN b.id AS b, c.id AS c",
        engine=engine,
    )._nodes
    assert sorted(zip(projection["b"], projection["c"])) == [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (3, 2),
        (3, 2),
    ]


@pytest.mark.skipif(
    os.environ.get("TEST_CUDF") != "1",
    reason="requires TEST_CUDF=1 and a CUDA device",
)
def test_nested_reentry_binds_incoming_edges_as_one_bag_cudf():
    """The same global-bag route keeps count and full projection parity on cuDF."""
    cudf = pytest.importorskip("cudf", reason="cudf engine lane requires a GPU box")
    graph = graphistry.nodes(cudf.from_pandas(PARALLEL_NODES), "id").edges(
        cudf.from_pandas(PARALLEL_EDGES), "s", "d"
    )
    bag = (
        "MATCH (a {id:1}) WITH a MATCH (a)-[]->(b) WITH b "
        "MATCH (c)-[]->(b) RETURN count(*) AS c"
    )
    projection = (
        "MATCH (a {id:1}) WITH a MATCH (a)-[]->(b) WITH b "
        "MATCH (c)-[]->(b) RETURN b.id AS b, c.id AS c"
    )
    assert graph.gfql(bag, engine="cudf")._nodes.to_pandas().to_dict(
        orient="records"
    ) == [{"c": 6}]
    rows = graph.gfql(projection, engine="cudf")._nodes.to_pandas()
    assert sorted(zip(rows["b"], rows["c"])) == [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (3, 2),
        (3, 2),
    ]
    distinct = (
        "MATCH (a {id:1}) WITH a MATCH (a)-[]->(b) WITH DISTINCT b "
        "MATCH (c)-[]->(b) RETURN count(*) AS c"
    )
    assert graph.gfql(distinct, engine="cudf")._nodes.to_pandas().to_dict(
        orient="records"
    ) == [{"c": 3}]


@pytest.mark.parametrize("edge_count", [0, 1])
@pytest.mark.parametrize("engine", ENGINES)
def test_freeform_reentry_suffix_aggregates_once(edge_count, engine):
    """Multiple carried rows feed one global suffix aggregate."""
    nodes = pd.DataFrame({"id": [1, 2, 3, 4], "kind": ["a", "a", "c", "d"]})
    edges = pd.DataFrame({"s": [3] * edge_count, "d": [4] * edge_count})
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    query = (
        "MATCH (a {kind:'a'}) WITH a "
        "MATCH (c {kind:'c'})-[]->(d {kind:'d'}) RETURN count(*) AS c"
    )
    assert _aggregate_rows(graph, query, engine) == [{"c": 2 * edge_count}]


def test_nested_reentry_preserves_carried_alias_associations():
    """Each duplicate b row keeps its source a while the next hop is traversed."""
    graph = graphistry.nodes(NODES, "id").edges(ADVERSARIAL_EDGES, "s", "d")
    projection = graph.gfql(
        "MATCH (a {kind:'a'}) MATCH (a)-[]->(b) WITH a,b "
        "MATCH (b)-[]->(c) RETURN a.id AS a, b.id AS b, c.id AS c",
        engine="pandas",
    )._nodes
    assert sorted(zip(projection["a"], projection["b"], projection["c"])) == [
        (0, 3, 4),
        (0, 3, 5),
        (0, 6, 1),
        (4, 6, 1),
        (6, 1, 2),
    ]


def test_nested_reentry_seed_multiplicity_typed_decline_polars():
    """The unsupported Polars row-pipeline route remains an explicit decline."""
    pytest.importorskip("polars")
    graph = graphistry.nodes(NODES, "id").edges(ADVERSARIAL_EDGES, "s", "d")
    with pytest.raises(NotImplementedError, match="cypher row op 'rows'"):
        _count(graph, NESTED, "polars")


@pytest.mark.parametrize("engine", ENGINES)
def test_query_order_does_not_change_answers(engine):
    """Whole point of immutability: the answers do not depend on execution order."""
    g1 = _g()
    plain_first = (_count(g1, PLAIN, engine), _count(g1, REENTRY, engine))
    g2 = _g()
    reentry_first = (_count(g2, REENTRY, engine), _count(g2, PLAIN, engine))
    assert plain_first == (PLAIN_COUNT, REENTRY_COUNT)
    assert reentry_first == (REENTRY_COUNT, PLAIN_COUNT)


@pytest.mark.parametrize("engine", ENGINES)
def test_reentry_result_does_not_carry_execution_state(engine):
    """The same defect one hop out: a RESULT that carries the seed poisons queries on IT."""
    out = _g().gfql("MATCH (a {kind:'a'}) WITH a MATCH (a)-[]->(b) RETURN a", engine=engine)
    assert out._gfql_start_nodes is None
    assert out._gfql_rows_base_graph is None


@pytest.mark.parametrize("engine", ENGINES)
def test_pure_call_chain_result_does_not_carry_execution_state(engine):
    """The all-calls boundary run (``let()`` bodies) hands its graph straight through.

    On polars that graph is the one the run started from, so the boundary's base-graph
    write lands on the returned object unless it is attached to a copy and cleared.
    """
    from graphistry.compute.ast import call

    out = _g().gfql([call("rows", {"table": "nodes"})], engine=engine)
    assert out._gfql_rows_base_graph is None
    assert out._gfql_start_nodes is None


@pytest.mark.parametrize("engine", ENGINES)
def test_shortest_path_backend_is_not_written_onto_caller_graph(engine):
    """A per-CALL argument is not a property of the graph; it must not persist on it."""
    g = _g()
    g.gfql(PLAIN, engine=engine, shortest_path_backend="bfs")
    assert g._gfql_shortest_path_backend == "auto"
