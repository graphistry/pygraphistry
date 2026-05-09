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


def test_compile_graph_constructor_query_propagates_cross_kind_rebind_error() -> None:
    """The compile entrypoint should reject the same cross-kind shape when
    expressed as a GRAPH constructor query."""
    query = "GRAPH { MATCH (a) MATCH ()-[a]->() }"
    with pytest.raises(GFQLValidationError) as exc_info:
        compile_cypher(query)
    assert exc_info.value.code == ErrorCode.E204
    assert "Cypher alias rebound as a different entity kind" in str(exc_info.value)
    assert exc_info.value.context["existing_kind"] == "node"
    assert exc_info.value.context["new_kind"] == "edge"
    assert exc_info.value.context["value"] == "a"


# ---------------------------------------------------------------------------
# CypherGraphQuery coverage — graph constructor / graph binding paths route
# through _bind_graph_query/_bind_graph_constructor and must reject the same
# cross-kind rebind shapes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "GRAPH { MATCH (a) MATCH ()-[a]->() }",
        (
            "GRAPH g1 = GRAPH { MATCH (a) MATCH ()-[a]->() } "
            "GRAPH { USE g1 MATCH (x)-[r]->(y) }"
        ),
    ],
)
def test_graph_query_paths_reject_cross_kind_rebind(query: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert exc_info.value.code == ErrorCode.E204
    ctx = exc_info.value.context
    assert ctx["existing_kind"] == "node"
    assert ctx["new_kind"] == "edge"
    assert ctx["value"] == "a"


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


# ---------------------------------------------------------------------------
# Scalar-origin matrix — aliases bound as entity_kind="scalar" by callers
# *other than* path patterns (UNWIND output, WITH expression projection,
# CALL/YIELD output, list literals) must also reject re-bind as node/edge.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, alias",
    [
        # WITH expression projection — entity_kind="scalar" via _project_items.
        ("MATCH (a) WITH count(a) AS x MATCH (x) RETURN x", "x"),
        # WITH literal projection — same.
        ("MATCH (a) WITH 1 AS x MATCH (x) RETURN x", "x"),
        # NOTE: UNWIND-then-MATCH cases (e.g. ``UNWIND [1,2] AS x MATCH
        # (x)``) are NOT reliably rejected today — `_bind_graph_sequence`
        # binds `ast.matches` before `ast.unwinds`, so MATCH (x) binds x
        # as a fresh node *before* UNWIND runs and silently overwrites
        # entity_kind. That traversal-order bug is tracked in #1371 (P1).
        # When P1 lands, UNWIND-origin scalar→node should fall under this
        # matrix.
    ],
)
def test_scalar_origin_alias_rejected_as_node_rebind(query: str, alias: str) -> None:
    """Scalar-kind aliases from non-path origins are rejected when a later
    MATCH pattern uses them as a node variable."""
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert exc_info.value.code == ErrorCode.E204
    ctx = exc_info.value.context
    assert ctx["existing_kind"] == "scalar"
    assert ctx["new_kind"] == "node"
    assert ctx["value"] == alias


@pytest.mark.parametrize(
    "query, alias",
    [
        ("MATCH (a) WITH count(a) AS x MATCH ()-[x]->() RETURN x", "x"),
        ("MATCH (a) WITH 1 AS x MATCH ()-[x]->() RETURN x", "x"),
    ],
)
def test_scalar_origin_alias_rejected_as_edge_rebind(query: str, alias: str) -> None:
    """Scalar-kind aliases from non-path origins are rejected when a later
    MATCH pattern uses them as an edge variable."""
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert exc_info.value.code == ErrorCode.E204
    ctx = exc_info.value.context
    assert ctx["existing_kind"] == "scalar"
    assert ctx["new_kind"] == "edge"
    assert ctx["value"] == alias


# ---------------------------------------------------------------------------
# Reentry MATCH path coverage (binder.py:296-307) — structurally distinct
# from the plain-MATCH loop. _bind_match_clause is shared, so the guard
# fires identically, but pin a dedicated test so a future refactor of the
# reentry traversal can't silently regress.
# ---------------------------------------------------------------------------


def test_cross_kind_rebind_via_reentry_match_after_with_carry() -> None:
    """``MATCH (a) WITH a MATCH ()-[a]->()`` — reentry MATCH path; a
    survives the WITH carry as a node, then re-bind as edge is rejected."""
    query = "MATCH (a) WITH a MATCH ()-[a]->() RETURN a"
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher(query), PlanContext())
    ctx = exc_info.value.context
    assert ctx["existing_kind"] == "node"
    assert ctx["new_kind"] == "edge"


# ---------------------------------------------------------------------------
# Precedence: schema validation runs before the cross-kind guard inside
# _bind_match_clause. With a populated strict catalog, a label-miss should
# raise E301 before E204 even when both apply. Lock the order so a
# refactor can't silently swap precedence.
# ---------------------------------------------------------------------------


def test_schema_check_precedes_cross_kind_rebind_guard() -> None:
    """Pin the order: missing-label (E301) raises before cross-kind
    rebind (E204) when both conditions hold on the same MATCH."""
    from graphistry.compute.gfql.ir.compilation import GraphSchemaCatalog

    ctx = PlanContext(
        catalog=GraphSchemaCatalog.from_schema_parts(
            node_columns=["id", "label__Person"],
            edge_columns=["src", "dst"],
            metadata={"strict": True},
        )
    )
    # `a` first bound as node; second MATCH tries (a:Ghost) (label miss
    # under strict catalog) AND would-be cross-kind if it were instead
    # `[a]`. Variant: rebind a *as an edge* with a missing-label-bearing
    # adjacent node — schema check on the missing node label fires first.
    query = "MATCH (a:Person) MATCH (b:Ghost)-[a]->(c:Person) RETURN a"
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher(query), ctx, strict_name_resolution=True)
    assert exc_info.value.code == ErrorCode.E301
