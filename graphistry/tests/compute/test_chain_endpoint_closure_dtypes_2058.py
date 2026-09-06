"""The chain's endpoint-closure backfill keeps the node frame's dtypes (#2058).

The general path (every specialized route declined) used to concatenate id-only endpoint rows onto the
node frame and dedup afterwards, so an integer attribute came back float64 even when no endpoint
was missing. Pins: the general path keeps int64 on a closed graph on every eager engine; an edge
to an id absent from the node frame is dropped by the closure gate (no backfill, dtypes intact);
duplicate node-id rows still collapse.
"""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n
from graphistry.tests.compute.gfql.routes.switch import ROUTES, routes_off

NODES = pd.DataFrame({"id": [0, 1, 2], "v": [10, 20, 30], "flag": [True, False, True]})
CLOSED_EDGES = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0]})
DANGLING_EDGES = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 7]})


def _frames(engine, nodes, edges):
    if engine == "cudf":
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("cuDF lane runs with TEST_CUDF=1")
        cudf = pytest.importorskip("cudf")
        return cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return nodes, edges


def _pd(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_general_path_keeps_node_attribute_dtypes_on_a_closed_graph(engine):
    nodes, edges = _frames(engine, NODES, CLOSED_EDGES)
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    with routes_off(ROUTES):
        out = _pd(g.gfql([n(), e_forward(), n()], engine=engine)._nodes)
    assert str(out["v"].dtype) == "int64"
    assert str(out["flag"].dtype) == "bool"
    assert sorted(out["id"].tolist()) == [0, 1, 2]


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_edge_to_an_absent_node_is_dropped_and_dtypes_stay_intact(engine):
    nodes, edges = _frames(engine, NODES, DANGLING_EDGES)
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    with routes_off(ROUTES):
        res = g.gfql([n(), e_forward(), n()], engine=engine)
    out = _pd(res._nodes)
    assert sorted(_pd(res._edges)[["s", "d"]].values.tolist()) == [[0, 1], [1, 2]]
    assert sorted(out["id"].tolist()) == [0, 1, 2]
    assert str(out["v"].dtype) == "int64"


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_duplicate_node_id_rows_still_collapse_on_the_general_path(engine):
    dup = pd.concat([NODES, NODES.iloc[[0]]], ignore_index=True)
    nodes, edges = _frames(engine, dup, CLOSED_EDGES)
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    with routes_off(ROUTES):
        out = _pd(g.gfql([n(), e_forward(), n()], engine=engine)._nodes)
    assert sorted(out["id"].tolist()) == [0, 1, 2]
    assert str(out["v"].dtype) == "int64"


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_cypher_destination_projection_keeps_dtypes_on_every_route(engine):
    """The hop's endpoint backfill takes a backed id's full row, so a RETURN of destination
    properties has the source dtypes on the general path, the same as the seeded lanes."""
    nodes = pd.DataFrame({"id": [0, 1, 2], "rank": [5, 6, 7], "flag": [True, False, True], "public": [100, 1, 2]})
    edges = pd.DataFrame({"s": [0, 0, 1], "d": [1, 2, 2], "type": ["A", "A", "A"]})
    nodes, edges = _frames(engine, nodes, edges)
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    q = "MATCH (source {public: 100})-[:A]->(destination) RETURN destination.rank AS r, destination.flag AS f ORDER BY r"
    with routes_off(ROUTES):
        general = _pd(g.gfql(q, engine=engine)._nodes)
    lane = _pd(g.gfql(q, engine=engine)._nodes)
    assert general["r"].tolist() == lane["r"].tolist() == [6, 7]
    assert str(general["r"].dtype) == str(lane["r"].dtype) == "int64"
    assert str(general["f"].dtype) == str(lane["f"].dtype) == "bool"
