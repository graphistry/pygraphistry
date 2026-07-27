"""Native polars var-length binding rows: parity with the pandas oracle, or decline (#1787).

Three bounded shapes returned a DIFFERENT count from pandas with no error -- the contract is
parity-or-``NotImplementedError``, so they are now declined, mirroring what #1781 did for the
unbounded case:

  1. directed ``-[*k..m]->`` with ``min_hops >= 3`` (and ``min_hops >= 2`` off a filtered seed);
  2. undirected ``-[*1..1]-`` / ``-[*1]-``, the degenerate window, which HALVED the count;
  3. undirected ``-[*1..k]-`` that does not start from the full node set (filtered seed, or a
     non-first segment), which OVER-counted.

The differential fuzz at the bottom is what found all three (and what proves the gate is not
merely declining everything): it asserts served shapes still equal pandas AND that the
neighbouring shapes are still served, so a lazily over-broad gate fails it.
"""
import random

import pandas as pd
import pytest

import graphistry

pl = pytest.importorskip("polars")


def _pair(nodes, edges):
    return (
        graphistry.nodes(nodes, "id").edges(edges, "s", "d"),
        graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d"),
    )


def _count(g, query, engine):
    return int(g.gfql(query, engine=engine)._nodes["c"].to_list()[0])


# The fixtures from the issue, one per divergence.
BOUNDED_NODES = pd.DataFrame({"id": list(range(7))})
BOUNDED_EDGES = pd.DataFrame(
    [(4, 6), (0, 4), (1, 2), (4, 6), (5, 6), (4, 5), (4, 6)], columns=["s", "d"]
)
UNDIR_NODES = pd.DataFrame({"id": list(range(5))})
UNDIR_EDGES = pd.DataFrame(
    [(1, 2), (1, 2), (0, 2), (1, 0), (4, 2), (3, 2), (3, 4), (0, 3), (1, 2)], columns=["s", "d"]
)
SEEDED_NODES = pd.DataFrame({"id": list(range(7)), "kind": ["a", "b"] * 3 + ["a"]})
SEEDED_EDGES = pd.DataFrame(
    [(0, 3), (6, 3), (4, 2), (1, 3), (6, 3), (3, 1), (3, 5), (4, 1), (3, 3), (0, 3)],
    columns=["s", "d"],
)

DECLINED = [
    # (nodes, edges, query) -- each returned a wrong count before the gate.
    (BOUNDED_NODES, BOUNDED_EDGES, "MATCH (a)-[*3..3]->(b) RETURN count(*) AS c"),
    (BOUNDED_NODES, BOUNDED_EDGES, "MATCH (a)-[*3..4]->(b) RETURN count(*) AS c"),
    (SEEDED_NODES, SEEDED_EDGES, "MATCH (a {kind:'a'})-[*2..3]->(b) RETURN count(*) AS c"),
    (UNDIR_NODES, UNDIR_EDGES, "MATCH (a)-[*1..1]-(b) RETURN count(*) AS c"),
    (UNDIR_NODES, UNDIR_EDGES, "MATCH (a)-[*1]-(b) RETURN count(*) AS c"),
    (SEEDED_NODES, SEEDED_EDGES, "MATCH (a {kind:'a'})-[*1..2]-(b) RETURN count(*) AS c"),
    (SEEDED_NODES, SEEDED_EDGES, "MATCH (a)-[]->(b)-[*1..2]-(c) RETURN count(*) AS c"),
]

STILL_SERVED = [
    # Neighbours of each declined shape: the gate must not swallow these.
    (BOUNDED_NODES, BOUNDED_EDGES, "MATCH (a)-[*1..2]->(b) RETURN count(*) AS c"),
    (BOUNDED_NODES, BOUNDED_EDGES, "MATCH (a)-[*2..3]->(b) RETURN count(*) AS c"),
    (UNDIR_NODES, UNDIR_EDGES, "MATCH (a)-[]-(b) RETURN count(*) AS c"),
    (UNDIR_NODES, UNDIR_EDGES, "MATCH (a)-[*1..2]-(b) RETURN count(*) AS c"),
    (UNDIR_NODES, UNDIR_EDGES, "MATCH (a)-[*1..3]-(b) RETURN count(*) AS c"),
    (SEEDED_NODES, SEEDED_EDGES, "MATCH (a {kind:'a'})-[*1..2]->(b) RETURN count(*) AS c"),
    (SEEDED_NODES, SEEDED_EDGES, "MATCH (a)-[]->(b)-[*2..3]->(c) RETURN count(*) AS c"),
]


@pytest.mark.parametrize("nodes,edges,query", DECLINED)
def test_divergent_varlen_shapes_decline(nodes, edges, query):
    """A shape that cannot be reproduced must RAISE, never answer differently."""
    _, g_pl = _pair(nodes, edges)
    with pytest.raises(NotImplementedError):
        g_pl.gfql(query, engine="polars")


@pytest.mark.parametrize("nodes,edges,query", STILL_SERVED)
def test_neighbouring_varlen_shapes_still_served_and_match_pandas(nodes, edges, query):
    """The gate is a scalpel: every neighbouring shape still runs natively AND matches pandas."""
    g_pd, g_pl = _pair(nodes, edges)
    assert _count(g_pl, query, "polars") == _count(g_pd, query, "pandas")


@pytest.mark.parametrize("nodes,edges,query", DECLINED)
def test_declined_shapes_still_answerable_on_pandas(nodes, edges, query):
    """Declining is an ENGINE limit, not a query rejection: pandas still answers all of them."""
    g_pd, _ = _pair(nodes, edges)
    assert _count(g_pd, query, "pandas") >= 0


FUZZ_SHAPES = [
    "MATCH (a)-[*1..2]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*2..2]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*2..3]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*3..3]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*1..1]-(b) RETURN count(*) AS c",
    "MATCH (a)-[*1..2]-(b) RETURN count(*) AS c",
    "MATCH (a)-[*1..3]-(b) RETURN count(*) AS c",
    "MATCH (a {kind:'a'})-[*1..2]-(b) RETURN count(*) AS c",
    "MATCH (a {kind:'a'})-[*1..2]->(b) RETURN count(*) AS c",
    "MATCH (a {kind:'a'})-[*2..3]->(b) RETURN count(*) AS c",
    "MATCH (a)-[]->(b)-[*1..2]-(c) RETURN count(*) AS c",
    "MATCH (a)-[]->(b)-[*2..3]->(c) RETURN count(*) AS c",
]


def test_bounded_varlen_differential_fuzz_against_pandas_oracle():
    """Random small graphs (cyclic, parallel edges, self-loops): match pandas or raise.

    Bounded windows only -- undirected UNBOUNDED shapes through the pandas oracle can blow the
    box up (75 GiB RSS observed), and they are already declined elsewhere. Seeded so a failure
    is reproducible; the shape list spans both sides of every gate boundary, so a gate that
    declines too much shows up as the served-count assertion below going to zero.
    """
    rnd = random.Random(1787)
    served = 0
    for _ in range(25):
        n_nodes = rnd.randint(3, 7)
        nodes = pd.DataFrame({
            "id": list(range(n_nodes)),
            "kind": [rnd.choice("ab") for _ in range(n_nodes)],
        })
        edges = pd.DataFrame(
            [(rnd.randrange(n_nodes), rnd.randrange(n_nodes)) for _ in range(rnd.randint(3, 10))],
            columns=["s", "d"],
        )
        g_pd, g_pl = _pair(nodes, edges)
        for query in FUZZ_SHAPES:
            expected = _count(g_pd, query, "pandas")
            try:
                got = _count(g_pl, query, "polars")
            except NotImplementedError:
                continue  # an honest decline is allowed; a different answer is not
            served += 1
            assert got == expected, (
                f"polars diverged from the pandas oracle on {query!r}: "
                f"{got} != {expected}\nnodes={nodes.to_dict('list')}\nedges={edges.values.tolist()}"
            )
    assert served > 100, f"gate declines too much to be a useful parity check ({served})"
