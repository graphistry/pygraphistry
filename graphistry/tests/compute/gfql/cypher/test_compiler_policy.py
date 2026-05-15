from __future__ import annotations

from typing import List

import pandas as pd

import graphistry
from graphistry.compute.gfql.policy import PolicyContext


def _graph():
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "division": ["x", "x", "y"],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def test_string_cypher_postcompile_policy_reports_grouped_aggregate_summary() -> None:
    contexts: List[PolicyContext] = []

    def postcompile(context: PolicyContext) -> None:
        contexts.append(context)

    result = _graph().gfql(
        "MATCH (n) RETURN n.division AS division, count(*) AS cnt ORDER BY division",
        policy={"postcompile": postcompile},
    )

    assert result._nodes.to_dict(orient="records") == [
        {"division": "x", "cnt": 2},
        {"division": "y", "cnt": 1},
    ]
    assert len(contexts) == 1
    context = contexts[0]
    assert context["phase"] == "postcompile"
    assert context["hook"] == "postcompile"
    assert context["language"] == "cypher"
    assert "current_ast" not in context

    summary = context["compiler_summary"]
    assert summary is not None
    assert summary["phase"] == "postcompile"
    assert summary["language"] == "cypher"
    assert summary["query_type"] == "chain"
    assert len(summary["query_hash"]) == 64
    assert {"name": "n", "kind": "node", "nullable": False} in summary["aliases"]
    assert {"clause": "return", "output": "division", "expr": "n.division", "expr_kind": "property", "source": "n", "property": "division", "entity_kind": "node"} in summary["projections"]
    assert {"clause": "return", "output": "cnt", "expr": "count(*)", "expr_kind": "aggregate", "source": "*", "entity_kind": None} in summary["projections"]
    assert summary["aggregates"] == [{"clause": "return", "output": "cnt", "fn": "count", "input": "*", "distinct": False}]
    assert summary["group_keys"] == ["division"]


def test_string_cypher_postcompile_policy_fires_once_not_per_row() -> None:
    calls: List[PolicyContext] = []

    def postcompile(context: PolicyContext) -> None:
        calls.append(context)

    result = _graph().gfql(
        "MATCH (n) RETURN n.id AS id ORDER BY id",
        policy={"postcompile": postcompile},
    )

    assert result._nodes.to_dict(orient="records") == [
        {"id": "a"},
        {"id": "b"},
        {"id": "c"},
    ]
    assert len(calls) == 1


def test_string_cypher_no_policy_backward_compatibility() -> None:
    result = _graph().gfql(
        "MATCH (n) RETURN n.division AS division, count(*) AS cnt ORDER BY division"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"division": "x", "cnt": 2},
        {"division": "y", "cnt": 1},
    ]
