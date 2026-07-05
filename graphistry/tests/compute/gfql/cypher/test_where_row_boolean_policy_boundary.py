"""Policy-boundary audit for #1219 row-boolean WHERE behavior.

This suite maps the currently supported boundary before any future
"reject-subset" policy decision:

1) Bottom-up parser/binder routing contracts
2) Top-down execution behavior on representative shapes
3) Explicit rejected forms that remain unsupported
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, cast

import pandas as pd
import pytest

from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher import parse_cypher
from graphistry.compute.gfql.cypher.ast import CypherQuery, WherePatternPredicate
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.bound_ir import BoundQueryPart
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.tests.test_compute import CGFull


class _CypherBoundaryGraph(CGFull):
    _dgl_graph = None

    def search_graph(self, query: str, scale: float = 0.5, top_n: int = 100, thresh: float = 5000, broader: bool = False, inplace: bool = False):
        raise NotImplementedError

    def search(self, query: str, cols=None, thresh: float = 5000, fuzzy: bool = True, top_n: int = 10):
        raise NotImplementedError

    def embed(self, relation: str, *args, **kwargs):
        raise NotImplementedError


def _mk_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> _CypherBoundaryGraph:
    return cast(_CypherBoundaryGraph, _CypherBoundaryGraph().nodes(nodes_df, "id").edges(edges_df, "s", "d"))


def _rows(result: Any) -> List[Dict[str, Any]]:
    return cast(List[Dict[str, Any]], result._nodes.to_dict(orient="records"))


def _match_parts(query: str) -> List[BoundQueryPart]:
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())
    return [p for p in bound.query_parts if p.clause == "MATCH"]


@dataclass(frozen=True)
class _WhereRouteCase:
    name: str
    query: str
    expected_expr_op: str | None
    expected_pattern_pred_count: int
    expected_bound_pred_count: int


@pytest.mark.parametrize(
    "case",
    [
        _WhereRouteCase(
            name="structured-and",
            query="MATCH (n) WHERE n.x = 1 AND n.y = 2 RETURN n",
            expected_expr_op=None,
            expected_pattern_pred_count=0,
            expected_bound_pred_count=2,
        ),
        _WhereRouteCase(
            name="row-or-tree",
            query="MATCH (n) WHERE n.x = 1 OR n.y = 2 RETURN n",
            expected_expr_op="or",
            expected_pattern_pred_count=0,
            expected_bound_pred_count=1,
        ),
        _WhereRouteCase(
            name="row-not-tree",
            query="MATCH (n) WHERE NOT n.x = 1 RETURN n",
            expected_expr_op="not",
            expected_pattern_pred_count=0,
            expected_bound_pred_count=1,
        ),
        _WhereRouteCase(
            name="row-xor-tree",
            query="MATCH (n) WHERE n:Admin XOR n:Active RETURN n",
            expected_expr_op="xor",
            expected_pattern_pred_count=0,
            expected_bound_pred_count=1,
        ),
        _WhereRouteCase(
            name="mixed-pattern-and-row",
            query="MATCH (n) WHERE (n)-[:R]->() AND n.x = 1 RETURN n",
            expected_expr_op="atom",
            expected_pattern_pred_count=1,
            expected_bound_pred_count=2,
        ),
        _WhereRouteCase(
            name="pattern-or-row-stays-tree",
            query="MATCH (n) WHERE (n)-[:R]->() OR n.x = 1 RETURN n",
            expected_expr_op="or",
            expected_pattern_pred_count=0,
            expected_bound_pred_count=1,
        ),
    ],
    ids=lambda c: c.name,
)
def test_issue_1219_policy_boundary_parser_binder_routes(case: _WhereRouteCase) -> None:
    parsed = parse_cypher(case.query)
    assert isinstance(parsed, CypherQuery)
    assert parsed.where is not None

    where = parsed.where
    pattern_pred_count = len([p for p in where.predicates if isinstance(p, WherePatternPredicate)])

    assert pattern_pred_count == case.expected_pattern_pred_count
    if case.expected_expr_op is None:
        assert where.expr_tree is None
    else:
        assert where.expr_tree is not None
        assert where.expr_tree.op == case.expected_expr_op

    parts = _match_parts(case.query)
    assert len(parts) == 1
    assert len(parts[0].predicates) == case.expected_bound_pred_count


def test_issue_1219_policy_boundary_execution_matrix() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "z"],
                "x": [1, 9, 9, 0],
                "y": [9, 2, 9, 0],
                "label__Admin": [True, False, True, False],
                "label__Active": [False, True, True, False],
            }
        ),
        pd.DataFrame({"s": ["a", "c"], "d": ["b", "b"], "type": ["R", "R"]}),
    )

    out_or = graph.gfql("MATCH (n) WHERE n.x = 1 OR n.y = 2 RETURN n.id AS id ORDER BY id")
    assert [r["id"] for r in _rows(out_or)] == ["a", "b"]

    out_not = graph.gfql("MATCH (n) WHERE NOT n.x = 1 RETURN n.id AS id ORDER BY id")
    assert [r["id"] for r in _rows(out_not)] == ["b", "c", "z"]

    out_xor = graph.gfql("MATCH (n) WHERE n:Admin XOR n:Active RETURN n.id AS id ORDER BY id")
    assert [r["id"] for r in _rows(out_xor)] == ["a", "b"]

    out_pattern_or = graph.gfql(
        "MATCH (n) WHERE (n)-[:R]->() OR n.id = 'z' RETURN n.id AS id ORDER BY id"
    )
    assert [r["id"] for r in _rows(out_pattern_or)] == ["a", "c", "z"]


def test_issue_1219_policy_boundary_quoted_pattern_existence_lexemes_are_row_literals() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "txt": ["exists { marker }", "not((a)-[:R]->(b))", "plain"],
            }
        ),
        pd.DataFrame({"s": [], "d": [], "type": []}),
    )

    out_exists_text = graph.gfql(
        "MATCH (n) WHERE n.txt = 'exists { marker }' RETURN n.id AS id ORDER BY id"
    )
    assert [r["id"] for r in _rows(out_exists_text)] == ["a"]

    out_not_pattern_text = graph.gfql(
        "MATCH (n) WHERE n.txt = 'not((a)-[:R]->(b))' RETURN n.id AS id ORDER BY id"
    )
    assert [r["id"] for r in _rows(out_not_pattern_text)] == ["b"]


def test_issue_1219_policy_boundary_comment_lexemes_do_not_trip_unsupported_gate() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "z"],
                "x": [1, 9, 9, 0],
                "y": [9, 2, 9, 0],
            }
        ),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
    )

    out_inline_comment = graph.gfql(
        "MATCH (n) WHERE n.x = 1 /* exists { shadow } */ OR n.y = 2 RETURN n.id AS id ORDER BY id"
    )
    assert [r["id"] for r in _rows(out_inline_comment)] == ["a", "b"]

    out_line_comment = graph.gfql(
        "MATCH (n) WHERE n.x = 1 // not((a)-[:R]->(b))\nOR n.y = 2 RETURN n.id AS id ORDER BY id"
    )
    assert [r["id"] for r in _rows(out_line_comment)] == ["a", "b"]

    out_scalar_filter_with_comment = graph.gfql(
        "MATCH (n) WHERE n.id = 'a' /* exists { shadow } */ RETURN n.id AS id"
    )
    assert [r["id"] for r in _rows(out_scalar_filter_with_comment)] == ["a"]


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) WHERE NOT((n)-[:R]->()) RETURN n",
        "MATCH (n) WHERE NOT((n:Admin)-[:R]->()) RETURN n",
        "MATCH (n) WHERE NOT((n)<-[:R]-()) RETURN n",
        "MATCH (n) WHERE NOT((n)--()) RETURN n",
        "MATCH (n) WHERE NOT((n)-[:R*]->()) RETURN n",
    ],
)
def test_issue_1219_policy_boundary_pattern_existence_forms_still_rejected(query: str) -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
    )

    with pytest.raises(GFQLValidationError, match="Pattern existence expressions"):
        graph.gfql(query)
