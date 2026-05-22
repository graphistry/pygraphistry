"""Cross-kind alias rebind guards in FrontendBinder (#1357)."""

from __future__ import annotations

from typing import Mapping, Optional

import pytest

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.api import compile_cypher
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.compilation import GraphSchemaCatalog, PlanContext


def _bind(query: str, ctx: Optional[PlanContext] = None):
    return FrontendBinder().bind(parse_cypher(query), ctx or PlanContext())


def _assert_rebind_error(
    query: str,
    expected: Mapping[str, object],
    *,
    ctx: Optional[PlanContext] = None,
    via_compile: bool = False,
) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        if via_compile:
            compile_cypher(query)
        else:
            _bind(query, ctx)

    assert exc_info.value.code == ErrorCode.E204
    for key, value in expected.items():
        assert exc_info.value.context[key] == value


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "MATCH ()-[r]->() MATCH (r) RETURN r",
            {"existing_kind": "edge", "new_kind": "node", "new_role": "node pattern", "value": "r"},
        ),
        (
            "MATCH p = ()-->() MATCH (p) RETURN p",
            {"existing_kind": "scalar", "new_kind": "node", "value": "p"},
        ),
        (
            "MATCH (a) MATCH ()-[a]->() RETURN a",
            {"existing_kind": "node", "new_kind": "edge", "new_role": "relationship pattern", "value": "a"},
        ),
        (
            "MATCH p = ()-->() MATCH ()-[p]->() RETURN p",
            {"existing_kind": "scalar", "new_kind": "edge", "value": "p"},
        ),
        (
            "MATCH (a) MATCH a = ()-->() RETURN a",
            {"existing_kind": "node", "new_kind": "scalar", "new_role": "path alias", "value": "a"},
        ),
        (
            "MATCH ()-[r]->() MATCH r = ()-->() RETURN r",
            {"existing_kind": "edge", "new_kind": "scalar", "value": "r"},
        ),
    ],
)
def test_direct_pattern_rebind_rejections(query: str, expected: Mapping[str, object]) -> None:
    _assert_rebind_error(query, expected)


@pytest.mark.parametrize(
    ("query", "alias", "kind"),
    [
        ("MATCH (a) MATCH (a) RETURN a", "a", "node"),
        ("MATCH (a:A) MATCH (a:B) RETURN a", "a", "node"),
        ("MATCH ()-[r:R]->() MATCH ()-[r:S]->() RETURN r", "r", "edge"),
        ("MATCH (a)-[r]->(b) MATCH (a)-[r]->(c) RETURN r", "r", "edge"),
        ("MATCH (a)-[:R]->(b) MATCH (b)-[:S]->(c) RETURN a, b, c", "c", "node"),
        ("MATCH p = ()-->() MATCH p = ()-->() RETURN p", "p", "scalar"),
    ],
)
def test_same_kind_pattern_rebinds_are_admitted(query: str, alias: str, kind: str) -> None:
    bound = _bind(query)
    assert bound.semantic_table.variables[alias].entity_kind == kind


@pytest.mark.parametrize(
    ("query", "via_compile"),
    [
        ("MATCH (a) MATCH ()-[a]->() RETURN a", True),
        ("GRAPH { MATCH (a) MATCH ()-[a]->() }", True),
        ("GRAPH { MATCH (a) MATCH ()-[a]->() }", False),
        (
            "GRAPH g1 = GRAPH { MATCH (a) MATCH ()-[a]->() } "
            "GRAPH { USE g1 MATCH (x)-[r]->(y) }",
            False,
        ),
    ],
)
def test_compile_and_graph_query_paths_reject_cross_kind_rebind(query: str, via_compile: bool) -> None:
    _assert_rebind_error(
        query,
        {"existing_kind": "node", "new_kind": "edge", "value": "a"},
        via_compile=via_compile,
    )


@pytest.mark.parametrize(
    ("query", "alias", "new_kind"),
    [
        ("MATCH (a) WITH count(a) AS x MATCH (x) RETURN x", "x", "node"),
        ("MATCH (a) WITH 1 AS x MATCH (x) RETURN x", "x", "node"),
        ("MATCH (a) WITH count(a) AS x MATCH ()-[x]->() RETURN x", "x", "edge"),
        ("MATCH (a) WITH 1 AS x MATCH ()-[x]->() RETURN x", "x", "edge"),
    ],
)
def test_scalar_origin_alias_rejected_as_pattern_rebind(query: str, alias: str, new_kind: str) -> None:
    _assert_rebind_error(
        query,
        {"existing_kind": "scalar", "new_kind": new_kind, "value": alias},
    )


def test_cross_kind_rebind_via_reentry_match_after_with_carry() -> None:
    _assert_rebind_error(
        "MATCH (a) WITH a MATCH ()-[a]->() RETURN a",
        {"existing_kind": "node", "new_kind": "edge"},
    )


def test_schema_check_precedes_cross_kind_rebind_guard() -> None:
    ctx = PlanContext(
        catalog=GraphSchemaCatalog.from_schema_parts(
            node_columns=["id", "label__Person"],
            edge_columns=["src", "dst"],
            metadata={"strict": True},
        )
    )
    query = "MATCH (a:Person) MATCH (b:Ghost)-[a]->(c:Person) RETURN a"
    with pytest.raises(GFQLValidationError) as exc_info:
        _bind(query, ctx)
    assert exc_info.value.code == ErrorCode.E301
