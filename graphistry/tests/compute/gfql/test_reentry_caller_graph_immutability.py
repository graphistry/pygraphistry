"""A query must not mutate the ``Plottable`` it was run on (#1786).

``WITH`` re-entry seeds the follow-up ``MATCH`` from the carried nodes. That seed used to be
assigned to the CALLER's graph and never cleared, so the next -- entirely unrelated -- query on
the same object was answered against the stale seed: no error, just the previous query's count.
Engine-independent (wrong on pandas too), so an engine A/B never surfaces it.

Everything here is written to FAIL if the leak comes back: the assertions compare a query run on
a re-used graph against the SAME query run on a pristine one, and pin the absolute counts so a
fix that merely makes both sides equally wrong is caught too.
"""
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

PLAIN = "MATCH (a)-[*]->(b) RETURN count(*) AS c"
REENTRY = "MATCH (a {kind:'a'}) WITH a MATCH (a)-[*]->(b) RETURN count(*) AS c"
NESTED = (
    "MATCH (a {kind:'a'}) WITH a MATCH (a)-[]->(b) WITH b MATCH (b)-[]->(c) "
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
    """Two WITH boundaries in one query, then an unrelated query on the same graph."""
    g = _g()
    assert _count(g, NESTED, engine) == NESTED_COUNT
    assert g._gfql_start_nodes is None
    assert _count(g, PLAIN, engine) == PLAIN_COUNT


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
