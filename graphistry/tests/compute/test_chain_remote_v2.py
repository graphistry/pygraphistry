"""Tests for gfql_remote() v2: WHERE passthrough, Let support, Cypher string support."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from graphistry.compute.ast import ASTLet, ASTNode
from graphistry.compute.chain import Chain
from graphistry.compute.chain_remote import chain_remote_generic


def _mock_plottable(**overrides: Any) -> MagicMock:
    """Build a minimal mock Plottable for chain_remote_generic."""
    mock = MagicMock()
    mock._dataset_id = "test-dataset-123"
    mock._edges = pd.DataFrame({"s": [0], "d": [1]})
    mock._nodes = pd.DataFrame({"id": [0, 1]})
    mock._privacy = None
    mock._url_params = {}
    mock.session.api_token = "test-token"
    mock.session.certificate_validation = True
    mock.base_url_server.return_value = "https://test.graphistry.com"
    for k, v in overrides.items():
        setattr(mock, k, v)
    return mock


class TestWherePassthrough:
    """P0: WHERE clauses must be included in gfql_query field."""

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_chain_with_where_includes_gfql_query(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        from graphistry.compute.ast import ASTNode, ASTEdge
        from graphistry.compute.gfql.same_path_types import col, compare
        chain = Chain(
            [ASTNode(filter_dict={"type": "Person"}, name="a"), ASTEdge(), ASTNode(name="b")],
            where=[compare(col("a", "owner_id"), "==", col("b", "owner_id"))],
        )

        chain_remote_generic(mock_self, chain, format="json")

        body = mock_post.call_args.kwargs["json"]
        # gfql_query must contain the full envelope with WHERE
        assert "gfql_query" in body
        assert body["gfql_query"]["type"] == "Chain"
        assert "where" in body["gfql_query"]
        assert len(body["gfql_query"]["where"]) > 0
        # gfql_operations must still be the flat array (backward compat)
        assert "gfql_operations" in body
        assert isinstance(body["gfql_operations"], list)

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_chain_without_where_still_sends_gfql_query(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        chain = Chain([ASTNode(filter_dict={"type": "Person"})])
        chain_remote_generic(mock_self, chain, format="json")

        body = mock_post.call_args.kwargs["json"]
        assert "gfql_query" in body
        assert body["gfql_query"]["type"] == "Chain"
        assert "gfql_operations" in body

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_list_input_sends_gfql_query(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        chain_remote_generic(mock_self, [ASTNode(filter_dict={"type": "Person"})], format="json")

        body = mock_post.call_args.kwargs["json"]
        assert "gfql_query" in body
        assert body["gfql_query"]["type"] == "Chain"


    @patch("graphistry.compute.chain_remote.requests.post")
    def test_cypher_with_cross_step_where_preserves_where(self, mock_post: MagicMock) -> None:
        """Regression: Cypher with cross-step WHERE must preserve WHERE in gfql_query.

        This is the exact P0 scenario end-to-end: a Cypher string that compiles
        to a Chain with WHERE constraints. The WHERE must appear in gfql_query.
        """
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        chain_remote_generic(
            mock_self,
            "MATCH (a:Person)-[r]->(b:Person) WHERE a.team = b.team RETURN a.name AS name",
            format="json",
        )

        body = mock_post.call_args.kwargs["json"]
        assert body["gfql_query"]["type"] == "Chain"
        assert "where" in body["gfql_query"], (
            f"Cypher cross-step WHERE must be in gfql_query, got keys: {list(body['gfql_query'].keys())}"
        )
        assert len(body["gfql_query"]["where"]) > 0, "WHERE clause should not be empty"
        # gfql_operations should also have content (the flat chain for old servers)
        assert len(body["gfql_operations"]) > 0


class TestLetSupport:
    """P1: ASTLet and Let dicts must serialize correctly."""

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_astlet_sends_let_envelope(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        let_query = ASTLet({"people": ASTNode(filter_dict={"type": "person"})})
        chain_remote_generic(mock_self, let_query, format="json")

        body = mock_post.call_args.kwargs["json"]
        assert body["gfql_query"]["type"] == "Let"
        assert "people" in body["gfql_query"]["bindings"]
        # Old servers get empty operations
        assert body["gfql_operations"] == []

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_let_dict_sends_let_envelope(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        let_dict = {"type": "Let", "bindings": {"people": {"type": "Node", "filter_dict": {"type": "person"}}}}
        chain_remote_generic(mock_self, let_dict, format="json")

        body = mock_post.call_args.kwargs["json"]
        assert body["gfql_query"]["type"] == "Let"
        assert body["gfql_operations"] == []


class TestCypherStringSupport:
    """P2: Cypher strings must compile locally and serialize."""

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_simple_cypher_sends_chain(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        chain_remote_generic(mock_self, "MATCH (n:Person) RETURN n", format="json")

        body = mock_post.call_args.kwargs["json"]
        assert "gfql_query" in body
        assert body["gfql_query"]["type"] == "Chain"
        assert len(body["gfql_operations"]) > 0

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_graph_constructor_sends_chain(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        chain_remote_generic(
            mock_self,
            "GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'x' }",
            format="json",
        )

        body = mock_post.call_args.kwargs["json"]
        assert "gfql_query" in body
        assert body["gfql_query"]["type"] == "Chain"
        assert len(body["gfql_operations"]) > 0

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_graph_binding_with_use_sends_let(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        chain_remote_generic(
            mock_self,
            "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) } "
            "USE g1 MATCH (x) RETURN x",
            format="json",
        )

        body = mock_post.call_args.kwargs["json"]
        assert body["gfql_query"]["type"] == "Let"
        assert "g1" in body["gfql_query"]["bindings"]
        assert "__result__" in body["gfql_query"]["bindings"]
        assert body["gfql_operations"] == []

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_graph_constructor_with_call_write_sends_let(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        chain_remote_generic(
            mock_self,
            "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) } "
            "GRAPH { USE g1 CALL graphistry.degree.write() }",
            format="json",
        )

        body = mock_post.call_args.kwargs["json"]
        assert body["gfql_query"]["type"] == "Let"
        assert "g1" in body["gfql_query"]["bindings"]
        # The final constructor should be a ChainRef with a CALL
        result_binding = body["gfql_query"]["bindings"]["__result__"]
        assert result_binding["type"] == "ChainRef"
        assert result_binding["ref"] == "g1"

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_standalone_call_write_sends_chain(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        chain_remote_generic(
            mock_self,
            "GRAPH { CALL graphistry.degree.write() }",
            format="json",
        )

        body = mock_post.call_args.kwargs["json"]
        assert "gfql_query" in body
        # Should be a Chain with a Call inside, not a Let (no bindings)
        assert body["gfql_query"]["type"] == "Chain"

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_call_with_params_preserves_params(self, mock_post: MagicMock) -> None:
        """Regression: CALL procedure params must not be dropped (was hardcoded to {})."""
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        chain_remote_generic(
            mock_self,
            "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) } "
            "GRAPH { USE g1 CALL graphistry.cugraph.pagerank.write({damping: 0.85}) }",
            format="json",
        )

        body = mock_post.call_args.kwargs["json"]
        # Find the CALL binding in the Let
        result_binding = body["gfql_query"]["bindings"]["__result__"]
        # The CALL should be inside the ChainRef chain
        call_items = result_binding.get("chain", [])
        call_ops = [op for op in call_items if isinstance(op, dict) and op.get("type") == "Call"]
        assert len(call_ops) > 0, f"Expected Call operation in result binding, got: {result_binding}"
        call_op = call_ops[0]
        # call_params nests algorithm params under 'params' key
        params = call_op["params"]
        nested = params.get("params", params)
        assert nested.get("damping") == 0.85, (
            f"CALL params should include damping=0.85, got: {params}"
        )

    def test_union_cypher_raises(self) -> None:
        mock_self = _mock_plottable()
        with pytest.raises((ValueError, Exception)):
            chain_remote_generic(
                mock_self,
                "MATCH (n) RETURN n.id AS id UNION MATCH (m) RETURN m.id AS id",
                format="json",
            )


class TestBackwardCompat:
    """Backward compatibility: old-format dict input still works."""

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_legacy_dict_still_works(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        legacy_dict = {
            "type": "Chain",
            "chain": [{"type": "Node", "filter_dict": {"type": "Person"}}]
        }
        chain_remote_generic(mock_self, legacy_dict, format="json")

        body = mock_post.call_args.kwargs["json"]
        assert body["gfql_operations"] == [{"type": "Node", "filter_dict": {"type": "Person"}}]
        assert body["gfql_query"] == legacy_dict

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_empty_chain_sends_gfql_query(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        chain_remote_generic(mock_self, [], format="json")

        body = mock_post.call_args.kwargs["json"]
        assert "gfql_query" in body
        assert body["gfql_operations"] == []


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_cypher_raises_parse_error(self) -> None:
        mock_self = _mock_plottable()
        with pytest.raises(Exception):
            chain_remote_generic(mock_self, "hello world not cypher", format="json")

    def test_invalid_type_raises_type_error(self) -> None:
        mock_self = _mock_plottable()
        with pytest.raises(TypeError, match="gfql_remote"):
            chain_remote_generic(mock_self, 42, format="json")  # type: ignore

    @patch("graphistry.compute.chain_remote.requests.post")
    def test_let_emits_warning(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(ok=True, json=lambda: {"nodes": [], "edges": []},
                                           headers={"content-type": "application/json"})
        mock_self = _mock_plottable()

        let_query = ASTLet({"people": ASTNode(filter_dict={"type": "person"})})
        # Capture warnings via monkeypatching (avoids Python's once-per-location filter)
        captured: list = []
        import graphistry.compute.chain_remote as _cr
        _orig = _cr.warnings.warn
        def _capture(*a: Any, **kw: Any) -> None:
            captured.append(a)
            _orig(*a, **kw)
        _cr.warnings.warn = _capture  # type: ignore
        try:
            chain_remote_generic(mock_self, let_query, format="json")
        finally:
            _cr.warnings.warn = _orig  # type: ignore
        assert any("Let/DAG" in str(a[0]) for a in captured), f"Expected Let/DAG warning, got: {captured}"
