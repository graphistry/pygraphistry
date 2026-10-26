"""Pins three layout paths whose locals were bound on only some routes.

Each of these read a name that an earlier branch may never have bound, so the failure
mode was UnboundLocalError rather than the intended behaviour.
"""

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.layout.graph.graphBase import GraphBase
from graphistry.layout.graph.vertex import Vertex
from graphistry.layout.mercator import mercator_layout
from graphistry.layout.utils.geometry import tangents


def test_tangents_returns_one_tangent_per_point_and_repeats_the_last():
    """The trailing tangent duplicates the final segment's, and needs no leaked loop variable."""
    for n in [2, 3, 5, 9]:
        pts = [np.array([float(k), float(k * k)]) for k in range(n)]
        Q, T = tangents(pts, n)
        assert len(Q) == n - 1
        assert len(T) == n
        assert np.allclose(T[-1], T[-2])
        for t in T:
            assert np.isclose(np.linalg.norm(t), 1.0), "tangents are unit vectors"


def test_graph_with_vertices_but_no_edges_is_rejected_as_unconnected():
    """Two vertices and no edges is unconnected; it must say so, not raise UnboundLocalError."""
    with pytest.raises(ValueError) as excinfo:
        GraphBase([Vertex("a"), Vertex("b")], [])
    assert "unconnected" in str(excinfo.value)


def test_single_vertex_graph_still_short_circuits():
    g = GraphBase([Vertex("a")], [])
    assert len(g.verticesPoset) == 1


def _geo_graph(frame):
    nodes = frame({
        "n": ["a", "b", "c"],
        "latitude": [37.7749, 40.7128, -33.8688],
        "longitude": [-122.4194, -74.0060, 151.2093],
    })
    edges = frame({"s": ["a", "b"], "d": ["b", "c"]})
    return graphistry.edges(edges, "s", "d").nodes(nodes, "n")


def test_mercator_projects_a_pandas_graph_without_cupy():
    """The CPU route must not depend on cupy having been imported."""
    out = mercator_layout(_geo_graph(pd.DataFrame))
    assert out._nodes["x"].notna().all()
    assert out._nodes["y"].notna().all()
    # x is a pure scaling of longitude, so ordering by longitude is preserved
    assert list(out._nodes.sort_values("longitude")["x"]) == sorted(out._nodes["x"])


def test_mercator_is_symmetric_about_the_equator():
    nodes = pd.DataFrame({
        "n": ["north", "south"], "latitude": [45.0, -45.0], "longitude": [10.0, 10.0]
    })
    edges = pd.DataFrame({"s": ["north"], "d": ["south"]})
    out = mercator_layout(graphistry.edges(edges, "s", "d").nodes(nodes, "n"))
    ys = out._nodes["y"].to_numpy()
    assert np.isclose(ys[0], -ys[1]), "equal latitudes north and south must mirror"
