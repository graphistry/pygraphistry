"""Cross-alias filtering must preserve operands, physical edges and empty schemas."""

import os

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import series_to_pylist
from graphistry.compute import e_forward, e_undirected, n
from graphistry.compute.gfql.same_path_types import col, compare


@pytest.fixture(params=["pandas", "cudf", "polars"])
def frame_engine(request):
    if request.param == "cudf" and os.environ.get("TEST_CUDF") != "1":
        pytest.skip("cuDF coverage requires TEST_CUDF=1")
    if request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def _graph(frame_engine, nodes, edges, *, bind_edge=True):
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    if frame_engine == "cudf":
        import cudf

        nodes_df, edges_df = cudf.from_pandas(nodes_df), cudf.from_pandas(edges_df)
    elif frame_engine == "polars":
        import polars as pl

        nodes_df, edges_df = pl.from_pandas(nodes_df), pl.from_pandas(edges_df)
    g = graphistry.nodes(nodes_df, "id").edges(edges_df, "s", "d")
    return g.bind(edge="eid") if bind_edge else g


def _engine(frame_engine):
    # Native Polars same-path WHERE explicitly declines; its default route is supported.
    return "auto" if frame_engine == "polars" else frame_engine


@pytest.mark.parametrize(
    "predicate,expected",
    [
        ("a.t < b.t", ["rise"]),
        ("b.t > a.t", ["rise"]),
        ("a.t <= b.t", ["equal", "rise"]),
        ("b.t >= a.t", ["equal", "rise"]),
        ("a.t > b.t", ["fall"]),
        ("b.t < a.t", ["fall"]),
        ("a.t >= b.t", ["equal", "fall"]),
        ("b.t <= a.t", ["equal", "fall"]),
        ("a.t = b.t", ["equal"]),
        ("b.t = a.t", ["equal"]),
    ],
)
def test_cross_alias_operand_order(frame_engine, predicate, expected):
    g = _graph(
        frame_engine,
        {"id": ["u", "v", "x", "y", "p", "q", "missing"], "t": [1, 2, 5, 3, 4, 4, None]},
        {"s": ["u", "x", "p", "missing", "v"], "d": ["v", "y", "q", "v", "missing"],
         "eid": ["rise", "fall", "equal", "null_left", "null_right"]},
    )
    out = g.gfql(
        f"MATCH (a)-[r]->(b) WHERE {predicate} RETURN r.eid AS eid ORDER BY eid",
        engine=_engine(frame_engine),
    )
    assert series_to_pylist(out._nodes["eid"]) == expected


@pytest.mark.parametrize("bind_edge", [True, False])
@pytest.mark.parametrize("pattern", ["(a)-[r]->(b)", "(b)<-[r]-(a)", "(a)-[r]-(b)"])
def test_cross_alias_preserves_parallel_edges(frame_engine, bind_edge, pattern):
    g = _graph(
        frame_engine,
        {"id": ["w", "x", "y", "z"], "t": [1, 2, 3, 4]},
        {"s": ["w", "w", "y"], "d": ["x", "x", "z"], "eid": ["e1", "e2", "e3"]},
        bind_edge=bind_edge,
    )
    out = g.gfql(
        f"MATCH {pattern} WHERE a.t < b.t RETURN r.eid AS eid ORDER BY eid",
        engine=_engine(frame_engine),
    )
    assert series_to_pylist(out._nodes["eid"]) == ["e1", "e2", "e3"]
    if pattern == "(a)-[r]->(b)":
        rows = g.gfql(
            f"MATCH {pattern} WHERE a.t < b.t RETURN a.id AS id ORDER BY id",
            engine=_engine(frame_engine),
        )
        assert series_to_pylist(rows._nodes["id"]) == ["w", "w", "y"]


@pytest.mark.parametrize("projection", ["r.eid AS eid", "r.s AS s", "r"])
def test_empty_cross_alias_keeps_relationship_schema(frame_engine, projection):
    g = _graph(
        frame_engine,
        {"id": ["u", "v"], "t": [1, 2]},
        {"s": ["u"], "d": ["v"], "eid": ["k"]},
        bind_edge=False,
    )
    out = g.gfql(
        f"MATCH (a)-[r]->(b) WHERE a.t > b.t RETURN {projection}",
        engine=_engine(frame_engine),
    )
    assert len(out._nodes) == 0


def test_native_chain_reversed_operand_order(frame_engine):
    g = _graph(
        frame_engine,
        {"id": ["u", "v"], "t": [1, 2]},
        {"s": ["u", "v"], "d": ["v", "u"], "eid": ["rise", "fall"]},
    )
    out = g.gfql(
        [n(name="a"), e_forward(name="r"), n(name="b")],
        where=[compare(col("b", "t"), ">", col("a", "t"))],
        engine=_engine(frame_engine),
    )
    assert series_to_pylist(out._edges["eid"]) == ["rise"]


def test_undirected_filter_keeps_each_physical_edge_once(frame_engine):
    g = _graph(
        frame_engine,
        {"id": ["u", "v"], "t": [1, 1]},
        {"s": ["u", "u", "v", "u"], "d": ["v", "v", "u", "u"],
         "eid": ["parallel1", "parallel2", "reverse", "loop"]},
    )
    out = g.gfql(
        [n(name="a"), e_undirected(name="r"), n(name="b")],
        where=[compare(col("a", "t"), "==", col("b", "t"))],
        engine=_engine(frame_engine),
    )
    assert sorted(series_to_pylist(out._edges["eid"])) == ["loop", "parallel1", "parallel2", "reverse"]


def test_multihop_reversed_operand_order(frame_engine):
    g = _graph(
        frame_engine,
        {"id": ["u", "m", "v"], "t": [1, 2, 3]},
        {"s": ["u", "m"], "d": ["m", "v"], "eid": ["first", "second"]},
    )
    out = g.gfql(
        [n({"id": "u"}, name="a"), e_forward(min_hops=2, max_hops=2, name="r"), n(name="b")],
        where=[compare(col("b", "t"), ">", col("a", "t"))],
        engine=_engine(frame_engine),
    )
    assert sorted(series_to_pylist(out._edges["eid"])) == ["first", "second"]
