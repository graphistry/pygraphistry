"""Explicit relationship-type adjacency must preserve canonical GFQL results."""
from __future__ import annotations

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.ast import e_forward, e_undirected, n, rows
from graphistry.compute.gfql.index import get_registry
from graphistry.compute.gfql.index.registry import EDGE_IN_ADJ, EDGE_OUT_ADJ
from graphistry.compute.gfql.index.traverse import _indices_for_direction


def _graph(engine: str):
    nodes = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    # Duplicate links, two type values, and a competing nonmatching link.
    edges = pd.DataFrame({
        "src": [1, 1, 1, 2, 2, 3, 4, 5],
        "dst": [2, 2, 3, 4, 5, 4, 1, 1],
        "type": ["K", "K", "X", "T", "X", "T", "K", "T"],
    })
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.bind(node="id", source="src", destination="dst").nodes(nodes).edges(edges)


def _pandas(frame):
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


def _assert_same_graph(actual, expected) -> None:
    pd.testing.assert_frame_equal(_pandas(actual._nodes), _pandas(expected._nodes))
    assert (actual._edges is None) == (expected._edges is None)
    if actual._edges is not None:
        pd.testing.assert_frame_equal(_pandas(actual._edges), _pandas(expected._edges))


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize("query", [
    [n({"id": 1}, name="a"), e_forward({"type": "K"}, name="r"), n(name="b"),
     e_forward({"type": "T"}, name="s"), n(name="c"), rows()],
    [n({"id": 1}), e_forward({"type": "K"}), n()],
    [n({"id": 1}), e_undirected({"type": "K"}), n()],
])
def test_explicit_typed_adjacency_matches_scan(engine: str, query) -> None:
    graph = _graph(engine)
    expected = graph.gfql(query, engine=engine, index_policy="off")
    indexed = graph.gfql_index_all(engine=engine).gfql_index_edge_types("type", ["K", "T"], engine=engine)
    actual = indexed.gfql(query, engine=engine, index_policy="force")
    _assert_same_graph(actual, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
def test_typed_adjacency_lifecycle_and_fallback(engine: str) -> None:
    graph = _graph(engine)
    indexed = graph.gfql_index_all(engine=engine).gfql_index_edge_types("type", ["K"], engine=engine)
    registry = get_registry(indexed)
    edges = indexed._edges
    assert edges is not None
    eng = Engine(engine)

    selected = _indices_for_direction(registry, "forward", edges, ("src", "dst"), eng, {"type": "K"})
    assert selected is not None and selected[0] is registry.typed_adjacencies[("type", "K", EDGE_OUT_ADJ)]
    ordinary = _indices_for_direction(registry, "forward", edges, ("src", "dst"), eng, {"type": "T"})
    assert ordinary is not None and ordinary[0] is registry.get(EDGE_OUT_ADJ)
    both = _indices_for_direction(registry, "undirected", edges, ("src", "dst"), eng, {"type": "K"})
    assert both is not None and len(both) == 2
    assert both[1] is registry.typed_adjacencies[("type", "K", EDGE_IN_ADJ)]

    shown = indexed.show_indexes(engine=engine)
    typed = shown[shown["name"].str.startswith("typed:")]
    assert len(typed) == 2 and typed["valid"].all() and (typed["nbytes"] > 0).all()
    dropped = indexed.drop_index(EDGE_OUT_ADJ)
    assert ("type", "K", EDGE_OUT_ADJ) not in get_registry(dropped).typed_adjacencies
    assert ("type", "K", EDGE_IN_ADJ) in get_registry(dropped).typed_adjacencies

    # A new frame, even of equal shape, invalidates the old source-frame identity.
    stale = indexed.edges(edges.clone() if engine == "polars" else edges.copy())
    assert not stale.show_indexes(engine=engine)["valid"].all()
    stale_selected = _indices_for_direction(get_registry(stale), "forward", stale._edges,
                                            ("src", "dst"), eng, {"type": "K"})
    assert stale_selected is None
    query = [n({"id": 1}), e_forward({"type": "K"}), n()]
    _assert_same_graph(stale.gfql(query, engine=engine, index_policy="force"),
                       stale.gfql(query, engine=engine, index_policy="off"))
