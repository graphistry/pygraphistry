import math

import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.cypher.procedures.networkx import NETWORKX_PROCEDURES
from graphistry.plugins.networkx import compute_algs


nx = pytest.importorskip("networkx")


def _graph(edges: pd.DataFrame):
    return graphistry.edges(edges, "s", "d").materialize_nodes()


def _node_values(g, col):
    return dict(zip(g._nodes[g._node], g._nodes[col]))


def test_compute_networkx_degree_centrality_out_col():
    g = _graph(pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "a"]}))

    g2 = g.compute_networkx("degree_centrality", out_col="degree_score", directed=False)

    assert "degree_score" in g2._nodes.columns
    assert _node_values(g2, "degree_score") == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_compute_networkx_connected_components_labels():
    g = _graph(pd.DataFrame({"s": ["a", "c"], "d": ["b", "d"]}))

    g2 = g.compute_networkx("connected_components", directed=False)

    assert "labels" in g2._nodes.columns
    labels = _node_values(g2, "labels")
    assert labels["a"] == labels["b"]
    assert labels["c"] == labels["d"]
    assert labels["a"] != labels["c"]


def test_compute_networkx_node_algorithm_surface():
    g = _graph(pd.DataFrame({"s": ["a", "b", "c", "c"], "d": ["b", "c", "a", "d"]}))

    node_algorithms = [
        ("pagerank", "pagerank", {}),
        ("betweenness_centrality", "betweenness_centrality", {}),
        ("closeness_centrality", "closeness_centrality", {}),
        ("core_number", "core_number", {}),
        ("eigenvector_centrality", "eigenvector_centrality", {"max_iter": 200}),
        ("katz_centrality", "katz_centrality", {"alpha": 0.05, "max_iter": 200}),
    ]

    for alg, col, params in node_algorithms:
        g2 = g.compute_networkx(alg, params=params, directed=False)
        values = _node_values(g2, col)
        assert set(values) == {"a", "b", "c", "d"}
        assert all(pd.notna(value) for value in values.values())


def test_compute_networkx_hits_outputs_hubs_and_authorities():
    # NOT a pure cycle: a directed cycle's adjacency is a permutation matrix
    # (all singular values equal), so networkx HITS (via scipy ``svds(k=1)``)
    # returns an ARBITRARY vector from the degenerate singular space -- whose
    # components can sum to ~0, making its ``h /= h.sum()`` normalization blow
    # up to inf/nan. Which vector svds returns is backend/version dependent, so
    # a cycle graph makes this test flaky across scipy versions. A graph with a
    # well-defined hub/authority structure has a unique (Perron-Frobenius)
    # dominant singular vector -> stable, version-independent scores.
    g = _graph(pd.DataFrame({"s": ["a", "a", "b"], "d": ["b", "c", "c"]}))

    g2 = g.compute_networkx("hits", params={"max_iter": 200})

    hubs = _node_values(g2, "hubs")
    authorities = _node_values(g2, "authorities")
    assert set(hubs) == {"a", "b", "c"}
    assert set(authorities) == {"a", "b", "c"}
    assert math.isclose(sum(hubs.values()), 1.0)
    assert math.isclose(sum(authorities.values()), 1.0)


def test_compute_networkx_directed_component_algorithms():
    g = _graph(pd.DataFrame({"s": ["a", "b", "b", "c"], "d": ["b", "a", "c", "d"]}))

    weak = _node_values(g.compute_networkx("connected_components", directed=True), "labels")
    strong = _node_values(g.compute_networkx("strongly_connected_components", directed=True), "labels")

    assert weak["a"] == weak["b"] == weak["c"] == weak["d"]
    assert strong["a"] == strong["b"]
    assert strong["c"] != strong["d"]


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


def test_compute_networkx_cudf_input_restores_cudf_engine():
    cudf = pytest.importorskip("cudf")
    g = graphistry.edges(cudf.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "a"]}), "s", "d").materialize_nodes()

    g2 = g.compute_networkx("degree_centrality", out_col="degree_score", directed=False)

    assert isinstance(g2._nodes, cudf.DataFrame)
    assert isinstance(g2._edges, cudf.DataFrame)
    assert set(g2._nodes["degree_score"].to_arrow().to_pylist()) == {1.0}


def test_compute_networkx_rejects_unknown_algorithm():
    g = _graph(pd.DataFrame({"s": ["a"], "d": ["b"]}))

    with pytest.raises(ValueError, match="Unsupported NetworkX algorithm"):
        g.compute_networkx("not_an_algorithm")


def test_compute_networkx_rejects_unsupported_params_for_no_param_algorithm():
    g = _graph(pd.DataFrame({"s": ["a"], "d": ["b"]}))

    with pytest.raises(ValueError, match="unsupported algorithm parameters"):
        g.compute_networkx("degree_centrality", params={"weight": "w"})


def test_compute_networkx_rejects_out_col_for_multi_and_graph_results():
    g = _graph(pd.DataFrame({"s": ["a"], "d": ["b"]}))

    with pytest.raises(ValueError, match="multi-column algorithms"):
        g.compute_networkx("hits", out_col="score")
    with pytest.raises(ValueError, match="graph-returning algorithms"):
        g.compute_networkx("k_core", out_col="core")


def test_compute_networkx_exported_allowlist_includes_gfql_networkx_surface():
    assert set(compute_algs) == set(NETWORKX_PROCEDURES)
