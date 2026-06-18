from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest  # type: ignore[import-not-found]

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.call.executor import execute_call
from graphistry.compute.gfql.ir.types import ScalarType
from graphistry.compute.gfql.schema_effects import SchemaEffect
from graphistry.schema import EdgeType, GraphSchema, NodeType


def _bound_graph() -> Any:
    nodes = pd.DataFrame(
        [
            {"id": "a", "label__Node": True},
            {"id": "b", "label__Node": True},
            {"id": "c", "label__Node": True},
        ]
    )
    edges = pd.DataFrame(
        [
            {"src": "a", "dst": "b", "label__REL": True},
            {"src": "b", "dst": "c", "label__REL": True},
        ]
    )
    schema = GraphSchema(
        node_types=[NodeType("Node", {"id": str})],
        edge_types=[EdgeType("REL", "Node", "Node", {})],
        node_id_column="id",
        edge_source_column="src",
        edge_destination_column="dst",
    )
    return graphistry.bind(source="src", destination="dst", node="id", schema=schema).edges(edges).nodes(nodes)


def _assert_node_property_validates(g: Any, prop: str) -> None:
    report = g.gfql_validate(f"MATCH (n:Node) RETURN n.{prop} AS value")
    assert report["ok"] is True


def _assert_edge_property_validates(g: Any, prop: str) -> None:
    report = g.gfql_validate(f"MATCH (:Node)-[e:REL]->(:Node) RETURN e.{prop} AS value")
    assert report["ok"] is True


def test_schema_effect_model_is_internal_and_confidence_checked() -> None:
    assert not hasattr(graphistry, "SchemaEffect")

    effect = SchemaEffect(confidence="declared")
    assert effect.confidence == "declared"

    with pytest.raises(ValueError, match="SchemaEffect.confidence"):
        SchemaEffect(confidence="guessed")  # type: ignore[arg-type]


def test_execute_call_degree_updates_bound_schema_for_chain_continuation() -> None:
    g = _bound_graph()
    with pytest.raises(GFQLValidationError):
        _assert_node_property_validates(g, "degree")

    out = cast(Any, execute_call(g, "get_degrees", {}, Engine.PANDAS))

    _assert_node_property_validates(out, "degree")
    _assert_node_property_validates(out, "degree_in")
    assert out._gfql_schema.node_types[0].properties["degree"] == ScalarType("int32")
    assert "degree" not in g._gfql_schema.node_types[0].properties


def test_execute_call_cugraph_edge_write_updates_bound_schema_for_chain_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _bound_graph()

    def fake_compute_cugraph(self: Any, alg: str, out_col: str | None = None, **kwargs: object) -> Any:
        assert alg == "edge_betweenness_centrality"
        assert self._edges is not None
        return self.edges(
            self._edges.assign(**{out_col or "betweenness_centrality": [1.0, 2.0]}),
            self._source,
            self._destination,
        )

    monkeypatch.setattr(type(g), "compute_cugraph", fake_compute_cugraph)

    out = cast(
        Any,
        execute_call(
            g,
            "compute_cugraph",
            {"alg": "edge_betweenness_centrality", "out_col": "ebc"},
            Engine.PANDAS,
        ),
    )

    _assert_edge_property_validates(out, "ebc")
    assert out._gfql_schema.edge_types[0].properties["ebc"] == ScalarType("float64")


def test_execute_call_cugraph_hits_updates_all_bound_schema_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _bound_graph()

    def fake_compute_cugraph(self: Any, alg: str, out_col: str | None = None, **kwargs: object) -> Any:
        assert alg == "hits"
        assert self._nodes is not None
        return self.nodes(
            self._nodes.assign(**{out_col or alg: [0.1, 0.2, 0.3], "authorities": [0.4, 0.5, 0.6]}),
            self._node,
        )

    monkeypatch.setattr(type(g), "compute_cugraph", fake_compute_cugraph)

    out = cast(Any, execute_call(g, "compute_cugraph", {"alg": "hits"}, Engine.PANDAS))

    _assert_node_property_validates(out, "hits")
    _assert_node_property_validates(out, "authorities")
    assert out._gfql_schema.node_types[0].properties["hits"] == ScalarType("float64")
    assert out._gfql_schema.node_types[0].properties["authorities"] == ScalarType("float64")


def test_cypher_degree_write_updates_bound_schema_for_next_validation() -> None:
    enriched = _bound_graph().gfql("CALL graphistry.degree.write()")

    _assert_node_property_validates(enriched, "degree")
    assert enriched._gfql_schema.node_types[0].properties["degree_in"] == ScalarType("int32")


def test_cypher_igraph_pagerank_write_updates_bound_schema_for_next_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _bound_graph()

    def fake_compute_igraph(self: Any, alg: str, out_col: str | None = None, **kwargs: object) -> Any:
        assert alg == "pagerank"
        assert self._nodes is not None
        return self.nodes(
            self._nodes.assign(**{out_col or alg: [0.2, 0.3, 0.5]}),
            self._node,
        )

    monkeypatch.setattr(type(g), "compute_igraph", fake_compute_igraph)

    enriched = g.gfql("CALL graphistry.igraph.pagerank.write()")

    _assert_node_property_validates(enriched, "pagerank")
    assert enriched._gfql_schema.node_types[0].properties["pagerank"] == ScalarType("float64")


def test_cypher_cugraph_edge_write_updates_bound_schema_for_next_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _bound_graph()

    def fake_compute_cugraph(self: Any, alg: str, out_col: str | None = None, **kwargs: object) -> Any:
        assert alg == "edge_betweenness_centrality"
        assert self._edges is not None
        return self.edges(
            self._edges.assign(**{out_col or alg: [1.0, 2.0]}),
            self._source,
            self._destination,
        )

    monkeypatch.setattr(type(g), "compute_cugraph", fake_compute_cugraph)

    enriched = g.gfql("CALL graphistry.cugraph.edge_betweenness_centrality.write({out_col: 'ebc'})")

    _assert_edge_property_validates(enriched, "ebc")
    assert enriched._gfql_schema.edge_types[0].properties["ebc"] == ScalarType("float64")
