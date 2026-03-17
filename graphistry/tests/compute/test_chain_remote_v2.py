"""Tests for gfql_remote() v2: WHERE passthrough, Let support, Cypher string support."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from graphistry.compute.ast import ASTLet, ASTNode, ASTEdge
from graphistry.compute.chain import Chain
from graphistry.compute.chain_remote import chain_remote_generic
from graphistry.compute.gfql.same_path_types import col, compare


def _mock_plottable() -> MagicMock:
    mock = MagicMock()
    mock._dataset_id = "test-dataset-123"
    mock._edges = pd.DataFrame({"s": [0], "d": [1]})
    mock._nodes = pd.DataFrame({"id": [0, 1]})
    mock._privacy = None
    mock._url_params = {}
    mock.session.api_token = "test-token"
    mock.session.certificate_validation = True
    mock.base_url_server.return_value = "https://test.graphistry.com"
    return mock


_JSON_RESPONSE = MagicMock(
    ok=True, json=lambda: {"nodes": [], "edges": []},
    headers={"content-type": "application/json"},
)


def _send(query: Any, **kwargs: Any) -> Dict[str, Any]:
    """Send query through chain_remote_generic, return the request body JSON."""
    with patch("graphistry.compute.chain_remote.requests.post") as mock_post:
        mock_post.return_value = _JSON_RESPONSE
        chain_remote_generic(_mock_plottable(), query, format="json", **kwargs)
        return mock_post.call_args.kwargs["json"]


class TestWherePassthrough:
    """P0: WHERE clauses must be included in gfql_query field."""

    def test_chain_with_where(self) -> None:
        body = _send(Chain(
            [ASTNode(filter_dict={"type": "Person"}, name="a"), ASTEdge(), ASTNode(name="b")],
            where=[compare(col("a", "owner_id"), "==", col("b", "owner_id"))],
        ))
        assert body["gfql_query"]["type"] == "Chain"
        assert len(body["gfql_query"]["where"]) > 0
        assert isinstance(body["gfql_operations"], list)

    def test_chain_without_where(self) -> None:
        body = _send(Chain([ASTNode(filter_dict={"type": "Person"})]))
        assert body["gfql_query"]["type"] == "Chain"
        assert "gfql_operations" in body

    def test_list_input(self) -> None:
        body = _send([ASTNode(filter_dict={"type": "Person"})])
        assert body["gfql_query"]["type"] == "Chain"

    def test_cypher_with_cross_step_where(self) -> None:
        """Regression: Cypher cross-step WHERE must survive compile → serialize."""
        body = _send("MATCH (a:Person)-[r]->(b:Person) WHERE a.team = b.team RETURN a.name AS name")
        assert body["gfql_query"]["type"] == "Chain"
        assert len(body["gfql_query"].get("where", [])) > 0
        assert len(body["gfql_operations"]) > 0


class TestLetSupport:
    """P1: ASTLet and Let dicts must serialize correctly."""

    def test_astlet(self) -> None:
        body = _send(ASTLet({"people": ASTNode(filter_dict={"type": "person"})}))
        assert body["gfql_query"]["type"] == "Let"
        assert "people" in body["gfql_query"]["bindings"]
        assert body["gfql_operations"] == []

    def test_let_dict(self) -> None:
        body = _send({"type": "Let", "bindings": {"people": {"type": "Node", "filter_dict": {"type": "person"}}}})
        assert body["gfql_query"]["type"] == "Let"
        assert body["gfql_operations"] == []


class TestCypherStringSupport:
    """P2: Cypher strings must compile locally and serialize."""

    def test_simple_match_return(self) -> None:
        body = _send("MATCH (n:Person) RETURN n")
        assert body["gfql_query"]["type"] == "Chain"
        assert len(body["gfql_operations"]) > 0

    def test_graph_constructor(self) -> None:
        body = _send("GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'x' }")
        assert body["gfql_query"]["type"] == "Chain"
        assert len(body["gfql_operations"]) > 0

    def test_graph_binding_with_use(self) -> None:
        body = _send("GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) } USE g1 MATCH (x) RETURN x")
        assert body["gfql_query"]["type"] == "Let"
        assert "g1" in body["gfql_query"]["bindings"]
        assert "__result__" in body["gfql_query"]["bindings"]
        assert body["gfql_operations"] == []

    def test_graph_constructor_with_call_write(self) -> None:
        body = _send(
            "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) } "
            "GRAPH { USE g1 CALL graphistry.degree.write() }"
        )
        assert body["gfql_query"]["type"] == "Let"
        result = body["gfql_query"]["bindings"]["__result__"]
        assert result["type"] == "ChainRef"
        assert result["ref"] == "g1"

    def test_standalone_call_write(self) -> None:
        body = _send("GRAPH { CALL graphistry.degree.write() }")
        assert body["gfql_query"]["type"] == "Chain"

    def test_call_params_preserved(self) -> None:
        """Regression: CALL procedure params must not be dropped."""
        body = _send(
            "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) } "
            "GRAPH { USE g1 CALL graphistry.cugraph.pagerank.write({damping: 0.85}) }"
        )
        result = body["gfql_query"]["bindings"]["__result__"]
        call_ops = [op for op in result.get("chain", []) if isinstance(op, dict) and op.get("type") == "Call"]
        assert call_ops, f"Expected Call in result binding, got: {result}"
        params = call_ops[0]["params"]
        assert params.get("params", params).get("damping") == 0.85

    def test_cypher_with_params(self) -> None:
        body = _send("MATCH (n:Person) WHERE n.score > $cutoff RETURN n.id AS id", params={"cutoff": 10})
        assert body["gfql_query"]["type"] == "Chain"
        assert len(body["gfql_operations"]) > 0

    def test_union_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            _send("MATCH (n) RETURN n.id AS id UNION MATCH (m) RETURN m.id AS id")


class TestBackwardCompat:
    """Backward compatibility: old-format dict input still works."""

    def test_legacy_dict(self) -> None:
        d = {"type": "Chain", "chain": [{"type": "Node", "filter_dict": {"type": "Person"}}]}
        body = _send(d)
        assert body["gfql_operations"] == d["chain"]
        assert body["gfql_query"] == d

    def test_empty_chain(self) -> None:
        body = _send([])
        assert "gfql_query" in body
        assert body["gfql_operations"] == []


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_cypher(self) -> None:
        with pytest.raises(Exception):
            _send("hello world not cypher")

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="gfql_remote"):
            _send(42)  # type: ignore

    def test_let_emits_warning(self) -> None:
        captured: list = []
        import graphistry.compute.chain_remote as _cr
        _orig = _cr.warnings.warn
        _cr.warnings.warn = lambda *a, **kw: (captured.append(a), _orig(*a, **kw))  # type: ignore
        try:
            _send(ASTLet({"people": ASTNode(filter_dict={"type": "person"})}))
        finally:
            _cr.warnings.warn = _orig  # type: ignore
        assert any("Let/DAG" in str(a[0]) for a in captured)
