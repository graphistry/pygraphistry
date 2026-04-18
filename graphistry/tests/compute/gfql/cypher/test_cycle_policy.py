"""Cycle-policy regression locks for M2-PR2 (issue #1135).

These tests lock the current lowering.py contract for:
  1. Same-MATCH repeated alias (connected comma pattern) — rewrite succeeds
  2. Disconnected comma pattern — raises stable error
  3. Cross-WITH scalar alias reused as node variable — raises stable compiler error message

All tests are regression locks: they must not change behavior as M2 work lands.
"""
from __future__ import annotations

import pandas as pd
import pytest
import graphistry

from graphistry.compute.ast import ASTNode, ASTEdgeForward
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher import lower_match_clause, parse_cypher
from graphistry.tests.test_compute import CGFull


class _G(CGFull):
    _dgl_graph = None


def _mk_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> _G:
    g = _G()
    return g.nodes(nodes, "id").edges(edges, "s", "d")  # type: ignore[return-value]


def _connected_graph() -> _G:
    return _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"], "label__A": [True, False, False], "label__B": [False, True, False], "label__C": [False, False, True]}),
        pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["R", "R"]}),
    )


def _scalar_reentry_graph() -> _G:
    return _mk_graph(
        pd.DataFrame({"id": ["a1", "a2", "b1", "b2"], "label__A": [True, True, False, False], "num": [1, 2, 1, 3]}),
        pd.DataFrame({"s": ["a1", "a2"], "d": ["b1", "b2"], "type": ["R", "R"]}),
    )


# ---------------------------------------------------------------------------
# Policy 1: same-MATCH repeated alias → connected join (success)
# ---------------------------------------------------------------------------

class TestConnectedJoinPolicy:
    def test_connected_comma_pattern_stitched_in_order(self) -> None:
        """MATCH (a)-->(b),(b)-->(c): shared alias 'b' → 5-op linear chain a→b→c."""
        parsed = parse_cypher("MATCH (a)-[:R]->(b), (b)-[:R]->(c) RETURN c")
        ops = lower_match_clause(parsed.match)  # type: ignore[union-attr]
        assert len(ops) == 5
        assert isinstance(ops[0], ASTNode)
        assert isinstance(ops[2], ASTNode)
        assert isinstance(ops[4], ASTNode)
        assert ops[0]._name == "a"
        assert ops[2]._name == "b"
        assert ops[4]._name == "c"

    def test_connected_comma_reversed_segment_stitched(self) -> None:
        """MATCH (a)-->(b),(c)<--(b): reversed second segment stitched, still 5 ops."""
        parsed = parse_cypher("MATCH (a)-[:R1]->(b), (c)<-[:R2]-(b) RETURN c")
        ops = lower_match_clause(parsed.match)  # type: ignore[union-attr]
        assert len(ops) == 5

    def test_connected_comma_pattern_executes(self) -> None:
        """End-to-end: connected comma pattern returns the joined node."""
        result = _connected_graph().gfql(
            "MATCH (a:A)-[:R]->(b:B), (b:B)-[:R]->(c:C) RETURN c.id AS cid"
        )
        assert "cid" in result._nodes.columns
        assert "c" in result._nodes["cid"].tolist()


# ---------------------------------------------------------------------------
# Policy 2: disconnected comma pattern → stable error
# ---------------------------------------------------------------------------

class TestDisconnectedCommaPolicy:
    def test_disconnected_comma_rejected(self) -> None:
        """MATCH (a)-->(b),(c)-->(d): no shared alias → stable GFQLValidationError."""
        parsed = parse_cypher("MATCH (a)-[:R]->(b), (c)-[:R]->(d) RETURN d")
        with pytest.raises(GFQLValidationError, match="single linear connected path"):
            lower_match_clause(parsed.match)  # type: ignore[union-attr]

    def test_disconnected_error_message_stable(self) -> None:
        """Error message substring must remain stable so callers can match it."""
        parsed = parse_cypher("MATCH (a)-[:R]->(b), (x)-[:R]->(y) RETURN y")
        with pytest.raises(GFQLValidationError) as exc_info:
            lower_match_clause(parsed.match)  # type: ignore[union-attr]
        assert "single linear connected path" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Policy 3: cross-WITH scalar alias reused as node variable → stable error
# ---------------------------------------------------------------------------

class TestCrossWithAliasReusePolicy:
    def test_scalar_alias_reused_as_node_variable_rejected(self) -> None:
        """WITH [a] AS users MATCH (users)...: stable compiler error."""
        with pytest.raises(
            GFQLValidationError,
            match="Cypher MATCH after WITH scalar-only prefix aliases cannot be reused as node variables",
        ):
            _scalar_reentry_graph().gfql(
                "MATCH (a:A) WITH [a] AS users MATCH (users)-->(b) RETURN b.id AS bid"
            )

    def test_cross_with_error_message_stable(self) -> None:
        """The error message substring must remain stable across M2 refactors."""
        with pytest.raises(GFQLValidationError) as exc_info:
            _scalar_reentry_graph().gfql(
                "MATCH (a:A) WITH [a] AS users MATCH (users)-->(b) RETURN b.id AS bid"
            )
        assert "scalar-only prefix aliases cannot be reused as node variables" in str(exc_info.value)
