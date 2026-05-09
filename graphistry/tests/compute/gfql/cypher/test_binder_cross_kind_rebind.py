"""Cross-kind alias rebind guards in FrontendBinder (#1357).

The binder rejects shapes where a previously-bound alias is re-used as a
different entity kind in a later MATCH pattern. The four transitions are:

- node ↔ edge: ``MATCH (a) MATCH ()-[a]->()`` and the reverse.
- node ↔ scalar (path / list-of-path / list-of-x via WITH): ``WITH [a] AS users
  MATCH (users)-->()``.
- edge ↔ scalar: ``MATCH ()-[r]->() WITH r AS path MATCH path = (a)-->(b)``
  (path alias rebinding an edge variable).
- edge ↔ node: covered by the relationship-pattern guard.

This module exercises both the direct ``FrontendBinder().bind()`` surface and
the ``compile_cypher_query`` path so the guard's reach is pinned at both
layers. The guard is the binder-layer half of #1357; full strict_name_resolution
rollout at the post-normalize site is deferred (see plan).
"""

from __future__ import annotations

import pytest

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.api import compile_cypher
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.compilation import PlanContext


# ---------------------------------------------------------------------------
# Direct binder surface — _bind_node_pattern guard
# ---------------------------------------------------------------------------


def test_node_pattern_rejects_rebind_of_existing_edge_alias() -> None:
    """``MATCH ()-[r]->() MATCH (r)`` — edge alias r re-used as a node."""
    query = "MATCH ()-[r]->() MATCH (r) RETURN r"
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert exc_info.value.code == ErrorCode.E204
    ctx = exc_info.value.context
    assert ctx["existing_kind"] == "edge"
    assert ctx["new_kind"] == "node"
    assert ctx["new_role"] == "node pattern"
    assert ctx["value"] == "r"


def test_node_pattern_rejects_rebind_of_existing_path_alias() -> None:
    """``MATCH p = ()-->() MATCH (p)`` — path alias p re-used as a node."""
    query = "MATCH p = ()-->() MATCH (p) RETURN p"
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert exc_info.value.code == ErrorCode.E204
    ctx = exc_info.value.context
    assert ctx["existing_kind"] == "scalar"
    assert ctx["new_kind"] == "node"


def test_node_pattern_admits_rebind_of_existing_node_alias_label_widening() -> None:
    """``MATCH (a:A) MATCH (a:B)`` — same-kind rebind is admitted; labels widen."""
    query = "MATCH (a:A) MATCH (a:B) RETURN a"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert "a" in bound.semantic_table.variables
    a_var = bound.semantic_table.variables["a"]
    assert a_var.entity_kind == "node"


# ---------------------------------------------------------------------------
# Direct binder surface — _bind_relationship_pattern guard
# ---------------------------------------------------------------------------


def test_relationship_pattern_rejects_rebind_of_existing_node_alias() -> None:
    """``MATCH (a) MATCH ()-[a]->()`` — node alias a re-used as an edge."""
    query = "MATCH (a) MATCH ()-[a]->() RETURN a"
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert exc_info.value.code == ErrorCode.E204
    ctx = exc_info.value.context
    assert ctx["existing_kind"] == "node"
    assert ctx["new_kind"] == "edge"
    assert ctx["new_role"] == "relationship pattern"
    assert ctx["value"] == "a"


def test_relationship_pattern_rejects_rebind_of_existing_path_alias() -> None:
    """``MATCH p = ()-->() MATCH ()-[p]->()`` — path alias p re-used as edge."""
    query = "MATCH p = ()-->() MATCH ()-[p]->() RETURN p"
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert exc_info.value.code == ErrorCode.E204
    ctx = exc_info.value.context
    assert ctx["existing_kind"] == "scalar"
    assert ctx["new_kind"] == "edge"


def test_relationship_pattern_admits_rebind_of_existing_edge_alias_type_widening() -> None:
    """``MATCH ()-[r:R]->() MATCH ()-[r:S]->()`` — same-kind rebind is admitted."""
    query = "MATCH ()-[r:R]->() MATCH ()-[r:S]->() RETURN r"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert "r" in bound.semantic_table.variables
    assert bound.semantic_table.variables["r"].entity_kind == "edge"


# ---------------------------------------------------------------------------
# Direct binder surface — _bind_path_alias guard
# ---------------------------------------------------------------------------


def test_path_alias_rejects_rebind_of_existing_node_alias() -> None:
    """``MATCH (a) MATCH a = ()-->()`` — node alias a re-used as a path."""
    query = "MATCH (a) MATCH a = ()-->() RETURN a"
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert exc_info.value.code == ErrorCode.E204
    ctx = exc_info.value.context
    assert ctx["existing_kind"] == "node"
    assert ctx["new_kind"] == "scalar"
    assert ctx["new_role"] == "path alias"
    assert ctx["value"] == "a"


def test_path_alias_rejects_rebind_of_existing_edge_alias() -> None:
    """``MATCH ()-[r]->() MATCH r = ()-->()`` — edge alias r re-used as a path."""
    query = "MATCH ()-[r]->() MATCH r = ()-->() RETURN r"
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert exc_info.value.code == ErrorCode.E204
    ctx = exc_info.value.context
    assert ctx["existing_kind"] == "edge"
    assert ctx["new_kind"] == "scalar"


def test_path_alias_admits_rebind_of_existing_path_alias() -> None:
    """``MATCH p = ()-->() MATCH p = ()-->()`` — same-kind path rebind is admitted."""
    query = "MATCH p = ()-->() MATCH p = ()-->() RETURN p"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert "p" in bound.semantic_table.variables
    assert bound.semantic_table.variables["p"].entity_kind == "scalar"


# ---------------------------------------------------------------------------
# Reach via compile_cypher_query — guard fires through the compile entrypoint
# ---------------------------------------------------------------------------


def test_compile_cypher_query_propagates_cross_kind_rebind_error() -> None:
    """The cross-kind guard reaches callers of compile_cypher_query, not just
    direct FrontendBinder().bind() invocations."""
    query = "MATCH (a) MATCH ()-[a]->() RETURN a"
    with pytest.raises(GFQLValidationError) as exc_info:
        compile_cypher(query)
    assert exc_info.value.code == ErrorCode.E204
    assert "Cypher alias rebound as a different entity kind" in str(exc_info.value)
    assert exc_info.value.context["existing_kind"] == "node"
    assert exc_info.value.context["new_kind"] == "edge"


# ---------------------------------------------------------------------------
# Negative cases — same-kind rebinds across multiple shapes must still admit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a) MATCH (a) RETURN a",
        "MATCH (a:A) MATCH (a:B) RETURN a",
        "MATCH (a)-[r]->(b) MATCH (a)-[r]->(c) RETURN a, b, c",
        "MATCH (a)-[:R]->(b) MATCH (b)-[:S]->(c) RETURN a, b, c",
    ],
)
def test_same_kind_rebind_admitted(query: str) -> None:
    """Same-kind rebinds across MATCH boundaries are unaffected by the guard."""
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert bound.semantic_table.variables  # non-empty
