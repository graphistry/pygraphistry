import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.cypher.procedures.networkx import NETWORKX_PROCEDURES
from graphistry.plugins.networkx import compute_algs


nx = pytest.importorskip("networkx")


def _graph(edges: pd.DataFrame):
    return graphistry.edges(edges, "s", "d").materialize_nodes()


def test_compute_networkx_degree_centrality_out_col():
    g = _graph(pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "a"]}))

    g2 = g.compute_networkx("degree_centrality", out_col="degree_score", directed=False)

    assert "degree_score" in g2._nodes.columns
    scores = dict(zip(g2._nodes[g2._node], g2._nodes["degree_score"]))
    assert scores == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_compute_networkx_connected_components_labels():
    g = _graph(pd.DataFrame({"s": ["a", "c"], "d": ["b", "d"]}))

    g2 = g.compute_networkx("connected_components", directed=False)

    assert "labels" in g2._nodes.columns
    labels = dict(zip(g2._nodes[g2._node], g2._nodes["labels"]))
    assert labels["a"] == labels["b"]
    assert labels["c"] == labels["d"]
    assert labels["a"] != labels["c"]


def test_compute_networkx_edge_betweenness_out_col_preserves_edges():
    g = _graph(pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "edge_attr": [10, 20]}))

    g2 = g.compute_networkx("edge_betweenness_centrality", out_col="edge_bc", directed=False)

    assert "edge_bc" in g2._edges.columns
    assert "edge_attr" in g2._edges.columns
    assert len(g2._edges) == 2
    assert set(g2._edges["edge_bc"]) == {2.0 / 3.0}


def test_compute_networkx_k_core_projects_graph_and_preserves_attrs():
    g = _graph(
        pd.DataFrame(
            {
                "s": ["a", "b", "c", "c"],
                "d": ["b", "c", "a", "d"],
                "edge_attr": [10, 20, 30, 40],
            }
        )
    )

    g2 = g.compute_networkx("k_core", params={"k": 2}, directed=False)

    assert set(g2._nodes[g2._node]) == {"a", "b", "c"}
    assert set(g2._edges["edge_attr"]) == {10, 20, 30}


def test_compute_networkx_rejects_unknown_algorithm():
    g = _graph(pd.DataFrame({"s": ["a"], "d": ["b"]}))

    with pytest.raises(ValueError, match="Unsupported NetworkX algorithm"):
        g.compute_networkx("not_an_algorithm")


def test_compute_networkx_exported_allowlist_includes_gfql_networkx_surface():
    assert set(compute_algs) == set(NETWORKX_PROCEDURES)
