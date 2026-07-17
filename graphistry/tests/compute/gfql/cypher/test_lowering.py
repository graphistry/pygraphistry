from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path
import pandas as pd
import pytest
import graphistry
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from graphistry.compute.ast import ASTCall, ASTNode, ASTEdge, ASTEdgeForward, ASTEdgeReverse, ASTEdgeUndirected
from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError, GFQLTypeError, GFQLValidationError
from graphistry.compute.predicates.is_in import IsIn
from graphistry.compute.gfql.same_path_types import NODE_IDENTITY_COLUMN, col, compare
from graphistry.compute.gfql.cypher import (
    CypherQuery,
    compile_cypher,
    cypher_to_gfql,
    CompiledCypherUnionQuery,
    CompiledCypherQuery,
    lower_cypher_query,
    lower_match_clause,
    lower_match_query,
    parse_cypher,
    WherePatternPredicate,
    projection_planning as _projection_planning,
)
from graphistry.compute.gfql.cypher.call_procedures import (
    CompiledCypherProcedureCall,
)
from graphistry.compute.gfql.cypher.procedures.networkx import (
    _ensure_networkx_feature,
    _ensure_networkx_version_policy,
    _ensure_scipy_version_policy,
    _networkx_component_labels,
    _networkx_hits_scores,
    _networkx_pagerank_scores,
    networkx_normalized_value_columns,
)
from graphistry.plugins.networkx.policy import NETWORKX_SCIPY_EXTRA_REQUIREMENTS, NETWORKX_VERSION_SPEC, SCIPY_VERSION_SPEC
from graphistry.compute.gfql.cypher.ast import ExpressionText, OrderByClause, OrderItem, ReturnClause, ReturnItem, SourceSpan
from graphistry.compute.gfql.cypher.lowering import CompiledCypherExecutionExtras, CompiledCypherGraphQuery
from graphistry.compute.gfql.cypher.lowering import _logical_plan_route_for_query
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundQueryPart, SemanticTable
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import CHILD_SLOTS, Filter, PatternMatch, ProcedureCall as LogicalProcedureCall
from graphistry.compute.gfql_unified import _node_dtypes_for_pushdown
from graphistry.compute.gfql.cypher.lowering import _connected_join_dtype_admits, _connected_join_dtype_classes
from graphistry.Plottable import Plottable
from graphistry.tests.test_compute import CGFull
from graphistry.tests.compute.gfql.cypher._whole_entity_compat import entity_text_records


class _CypherTestGraph(CGFull):
    _dgl_graph = None

    def search_graph(self, query: str, scale: float = 0.5, top_n: int = 100, thresh: float = 5000, broader: bool = False, inplace: bool = False):
        raise NotImplementedError

    def search(self, query: str, cols=None, thresh: float = 5000, fuzzy: bool = True, top_n: int = 10):
        raise NotImplementedError

    def embed(self, relation: str, *args, **kwargs):
        raise NotImplementedError


def _mk_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> _CypherTestGraph:
    return cast(_CypherTestGraph, _CypherTestGraph().nodes(nodes_df, "id").edges(edges_df, "s", "d"))


def _mk_cudf_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> _CypherTestGraph:
    cudf = pytest.importorskip("cudf")
    return cast(
        _CypherTestGraph,
        _CypherTestGraph().nodes(cudf.from_pandas(nodes_df), "id").edges(cudf.from_pandas(edges_df), "s", "d"),
    )


def _require_cudf_runtime() -> Any:
    cudf = pytest.importorskip("cudf")
    try:
        _ = cudf.Series([1, 2, 3])
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"cudf installed but runtime is unavailable: {exc}")
    return cudf


def _mk_simple_path_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]}),
    )


def _mk_simple_path_graph() -> _CypherTestGraph:
    return _mk_graph(*_mk_simple_path_data())


def _mk_triangle_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "b", "a"], "d": ["b", "c", "c"]}),
    )


def _projection_clause(expression: str, alias: Optional[str] = None) -> ReturnClause:
    span = SourceSpan(1, 1, 1, 1 + len(expression), 0, len(expression))
    return ReturnClause(
        items=(ReturnItem(ExpressionText(expression, span), alias, span),),
        distinct=False,
        kind="return",
        span=span,
    )


def _mk_cartesian_node_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "num": [1, 2],
                "ts": [10, 20],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )


def _mk_cartesian_dynamic_pattern_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["a1", "a2", "b1", "b2"],
                "label__A": [True, True, False, False],
                "label__B": [False, False, True, True],
                "num": [1, 2, 1, 3],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )


def _mk_path_with_isolate_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame({"id": ["a", "b", "c", "z"]}),
        pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]}),
    )


def _mk_path_with_isolate_graph() -> _CypherTestGraph:
    return _mk_graph(*_mk_path_with_isolate_data())


def _mk_simple_path_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(*_mk_simple_path_data())


def _mk_path_with_isolate_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(*_mk_path_with_isolate_data())


def _mk_empty_graph() -> _CypherTestGraph:
    return _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))


class _MiniNxGraph:
    def __init__(
        self,
        edges: List[Tuple[str, str]],
        *,
        nodes: List[str],
        directed: bool = True,
    ) -> None:
        self._edges = edges
        self._nodes = nodes
        self._directed = directed

    def nodes(self) -> List[str]:
        return self._nodes

    def is_directed(self) -> bool:
        return self._directed

    def predecessors(self, node: str) -> List[str]:
        return [src for src, dst in self._edges if dst == node]

    def successors(self, node: str) -> List[str]:
        return [dst for src, dst in self._edges if src == node]

    def out_degree(self, node: str) -> int:
        return len(self.successors(node))

    def neighbors(self, node: str) -> List[str]:
        neighbors = [dst for src, dst in self._edges if src == node]
        if not self._directed:
            neighbors.extend(src for src, dst in self._edges if dst == node)
        return neighbors

    def degree(self, node: str) -> int:
        return len(self.neighbors(node))


def _mk_reentry_carried_scalar_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            {
                "id": ["a1", "a2", "b1", "b2"],
                "label__A": [True, True, False, False],
                "num": [1, 2, 1, 3],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a1", "a2"],
                "d": ["b1", "b2"],
                "type": ["R", "R"],
            }
        ),
    )


def _mk_reentry_carried_scalar_graph() -> _CypherTestGraph:
    return _mk_graph(*_mk_reentry_carried_scalar_data())


def _mk_reentry_carried_scalar_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(*_mk_reentry_carried_scalar_data())


def _mk_connected_reentry_carried_scalar_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            {
                "id": ["a1", "a2", "b1", "b2", "c1", "c2"],
                "label__A": [True, True, False, False, False, False],
                "label__B": [False, False, True, True, False, False],
                "label__C": [False, False, False, False, True, True],
                "score": [None, None, 10, 20, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a1", "a2", "b1", "b2"],
                "d": ["b1", "b2", "c1", "c2"],
                "type": ["R", "R", "S", "S"],
            }
        ),
    )


def _mk_connected_reentry_carried_scalar_graph() -> _CypherTestGraph:
    return _mk_graph(*_mk_connected_reentry_carried_scalar_data())


def _mk_connected_reentry_carried_scalar_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(*_mk_connected_reentry_carried_scalar_data())


def _mk_optional_prefix_reentry_no_match_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["x1", "y1"],
                "label__A": [True, False],
                "label__B": [False, True],
                "label__C": [False, False],
            }
        ),
        pd.DataFrame(
            {
                "s": ["y1"],
                "d": ["x1"],
                "type": ["Q"],
            }
        ),
    )


def _mk_optional_prefix_reentry_match_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["a1", "b1", "c1"],
                "label__A": [True, False, False],
                "label__B": [False, True, False],
                "label__C": [False, False, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a1", "b1"],
                "d": ["b1", "c1"],
                "type": ["R", "S"],
            }
        ),
    )


def _mk_reentry_order_limit_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["p0", "f1", "f2", "f3", "f4", "u1", "u2", "u3", "u4", "u5"],
                "label__Person": [True, True, True, True, True, False, False, False, False, False],
                "label__University": [False, False, False, False, False, True, True, True, True, True],
                "firstName": ["Seed", "Amy", "Bea", "Cara", "Dan", None, None, None, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p0", "p0", "p0", "p0", "f1", "f2", "f2", "f3", "f4"],
                "d": ["f1", "f2", "f3", "f4", "u1", "u2", "u3", "u4", "u5"],
                "type": ["KNOWS", "KNOWS", "KNOWS", "KNOWS", "STUDY_AT", "STUDY_AT", "STUDY_AT", "STUDY_AT", "STUDY_AT"],
            }
        ),
    )


def _mk_reentry_order_limit_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(
        pd.DataFrame(
            {
                "id": ["p0", "f1", "f2", "f3", "f4", "u1", "u2", "u3", "u4", "u5"],
                "label__Person": [True, True, True, True, True, False, False, False, False, False],
                "label__University": [False, False, False, False, False, True, True, True, True, True],
                "firstName": ["Seed", "Amy", "Bea", "Cara", "Dan", None, None, None, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p0", "p0", "p0", "p0", "f1", "f2", "f2", "f3", "f4"],
                "d": ["f1", "f2", "f3", "f4", "u1", "u2", "u3", "u4", "u5"],
                "type": ["KNOWS", "KNOWS", "KNOWS", "KNOWS", "STUDY_AT", "STUDY_AT", "STUDY_AT", "STUDY_AT", "STUDY_AT"],
            }
        ),
    )


def _mk_connected_multi_pattern_reentry_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2"],
                "label__A": [True, True, False, False, False, False, False, False],
                "label__B": [False, False, True, True, False, False, False, False],
                "label__C": [False, False, False, False, True, True, False, False],
                "label__D": [False, False, False, False, False, False, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a1", "a2", "b1", "b2", "c1", "c2"],
                "d": ["b1", "b2", "c1", "c2", "d1", "d2"],
                "type": ["R", "R", "S", "S", "T", "T"],
            }
        ),
    )


def _mk_collect_unwind_reentry_graph() -> _CypherTestGraph:
    """Graph for WITH collect(...) UNWIND ... MATCH tests: s->b1->c1, s->b2->c2."""
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["s", "b1", "b2", "c1", "c2"],
                "label__S": [True, False, False, False, False],
                "label__B": [False, True, True, False, False],
                "label__C": [False, False, False, True, True],
                "val": [0, 10, 20, 100, 200],
            }
        ),
        pd.DataFrame(
            {
                "s": ["s", "s", "b1", "b2"],
                "d": ["b1", "b2", "c1", "c2"],
                "type": ["X", "X", "Y", "Y"],
            }
        ),
    )


def _mk_connected_multi_pattern_fanout_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            {
                "id": ["a1", "b1", "c1", "d1", "d2"],
                "label__A": [True, False, False, False, False],
                "label__B": [False, True, False, False, False],
                "label__C": [False, False, True, False, False],
                "label__D": [False, False, False, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a1", "b1", "c1", "c1"],
                "d": ["b1", "c1", "d1", "d2"],
                "type": ["R", "S", "T", "T"],
            }
        ),
    )


def _mk_connected_multi_pattern_fanout_graph() -> _CypherTestGraph:
    return _mk_graph(*_mk_connected_multi_pattern_fanout_data())


def _mk_connected_multi_pattern_fanout_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(*_mk_connected_multi_pattern_fanout_data())


def _mk_multi_stage_reentry_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e"],
                "label__A": [True, False, False, False, False],
                "label__B": [False, True, False, False, False],
                "label__C": [False, False, True, False, False],
                "label__D": [False, False, False, True, False],
                "label__E": [False, False, False, False, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a", "b", "c", "a", "b"],
                "d": ["b", "c", "d", "e", "e"],
                "type": ["R", "S", "T", "R", "S"],
            }
        ),
    )


def _mk_multi_stage_reentry_graph() -> _CypherTestGraph:
    return _mk_graph(*_mk_multi_stage_reentry_data())


def _mk_multi_stage_reentry_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(*_mk_multi_stage_reentry_data())


def _mk_multi_stage_reentry_graph_with_terminal_u() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e"],
                "label__A": [True, False, False, False, False],
                "label__B": [False, True, False, False, False],
                "label__C": [False, False, True, False, False],
                "label__D": [False, False, False, True, False],
                "label__E": [False, False, False, False, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a", "b", "c", "d", "a", "b"],
                "d": ["b", "c", "d", "e", "e", "e"],
                "type": ["R", "S", "T", "U", "R", "S"],
            }
        ),
    )


def _mk_prefix_scalar_reentry_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            {
                "id": ["tag1", "tag2", "post1", "post2", "post3"],
                "label__Tag": [True, True, False, False, False],
                "label__Post": [False, False, True, True, True],
                "name": ["topic", "other", None, None, None],
                "tagId": [101, 202, None, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["post1", "post2", "post3"],
                "d": ["tag1", "tag1", "tag2"],
                "type": ["HAS_TAG", "HAS_TAG", "HAS_TAG"],
            }
        ),
    )


def _mk_prefix_scalar_reentry_graph() -> _CypherTestGraph:
    return _mk_graph(*_mk_prefix_scalar_reentry_data())


def _mk_prefix_scalar_reentry_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(*_mk_prefix_scalar_reentry_data())


def _mk_prefix_scalar_reentry_duplicate_seed_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["tag1", "tag1b", "post1", "post2"],
                "label__Tag": [True, True, False, False],
                "label__Post": [False, False, True, True],
                "name": ["topic", "topic", None, None],
                "tagId": [101, 101, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["post1", "post2"],
                "d": ["tag1", "tag1b"],
                "type": ["HAS_TAG", "HAS_TAG"],
            }
        ),
    )


def _mk_connected_post_tag_fanout_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            {
                "id": ["tag_known", "tag_other", "person1", "post1"],
                "label__Tag": [True, True, False, False],
                "label__Person": [False, False, True, False],
                "label__Post": [False, False, False, True],
                "name": ["topic", "other", None, None],
                "tagId": [101, 202, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["post1", "post1", "post1"],
                "d": ["person1", "tag_known", "tag_other"],
                "type": ["HAS_CREATOR", "HAS_TAG", "HAS_TAG"],
            }
        ),
    )


def _mk_connected_post_tag_fanout_graph() -> _CypherTestGraph:
    return _mk_graph(*_mk_connected_post_tag_fanout_data())


def _mk_connected_post_tag_fanout_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(*_mk_connected_post_tag_fanout_data())


def _issue_1000_ic6_query() -> str:
    return """
MATCH (knownTag:Tag { name: $tagName })
WITH knownTag.id as knownTagId

MATCH (person:Person { id: $personId })-[:KNOWS*1..2]-(friend)
WHERE NOT person=friend
WITH
    knownTagId,
    collect(distinct friend) as friends
UNWIND friends as f
    MATCH (f)<-[:HAS_CREATOR]-(post:Post),
          (post)-[:HAS_TAG]->(t:Tag{id: knownTagId}),
          (post)-[:HAS_TAG]->(tag:Tag)
    WHERE NOT t = tag
    WITH
        tag.name as tagName,
        count(post) as postCount
RETURN
    tagName,
    postCount
ORDER BY
    postCount DESC,
    tagName ASC
LIMIT 10
"""


def _issue_1000_ic6_params() -> dict[str, object]:
    return {
        "personId": 4398046511333,
        "tagName": "Carl_Gustaf_Emil_Mannerheim",
    }


def _mk_issue_1000_ic6_minimal_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": [501, 502, 503, 4398046511333, 2, 9001, 9002],
                "label__Tag": [True, True, True, False, False, False, False],
                "label__Person": [False, False, False, True, True, False, False],
                "label__Post": [False, False, False, False, False, True, True],
                "name": [
                    "Carl_Gustaf_Emil_Mannerheim",
                    "Alpha",
                    "Beta",
                    None,
                    None,
                    None,
                    None,
                ],
            }
        ),
        pd.DataFrame(
            {
                "s": [4398046511333, 9001, 9001, 9002, 9002],
                "d": [2, 2, 501, 2, 501],
                "type": ["KNOWS", "HAS_CREATOR", "HAS_TAG", "HAS_CREATOR", "HAS_TAG"],
            }
        ).pipe(
            lambda df: pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "s": [9001, 9002],
                            "d": [503, 502],
                            "type": ["HAS_TAG", "HAS_TAG"],
                        }
                    ),
                ],
                ignore_index=True,
            )
        ),
    )


def _mk_issue_1000_ic6_minimal_graph_cudf() -> _CypherTestGraph:
    graph = _mk_issue_1000_ic6_minimal_graph()
    return _mk_cudf_graph(graph._nodes, graph._edges)


def _mk_issue_1396_tag_cooccurrence_join_aggregation_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": [501, 502, 503, 4398046511333, 2, 3, 9001, 9002, 9003],
                "label__Tag": [True, True, True, False, False, False, False, False, False],
                "label__Person": [False, False, False, True, True, True, False, False, False],
                "label__Post": [False, False, False, False, False, False, True, True, True],
                "name": [
                    "Carl_Gustaf_Emil_Mannerheim",
                    "Alpha",
                    "Beta",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            }
        ),
        pd.DataFrame(
            {
                "s": [4398046511333, 4398046511333, 9001, 9002, 9003, 9001, 9002, 9003, 9001, 9002, 9003],
                "d": [2, 3, 2, 2, 3, 501, 501, 501, 502, 502, 503],
                "type": [
                    "KNOWS",
                    "KNOWS",
                    "HAS_CREATOR",
                    "HAS_CREATOR",
                    "HAS_CREATOR",
                    "HAS_TAG",
                    "HAS_TAG",
                    "HAS_TAG",
                    "HAS_TAG",
                    "HAS_TAG",
                    "HAS_TAG",
                ],
            }
        ),
    )


def _mk_issue_1396_tag_cooccurrence_join_aggregation_graph_cudf() -> _CypherTestGraph:
    graph = _mk_issue_1396_tag_cooccurrence_join_aggregation_graph()
    return _mk_cudf_graph(graph._nodes, graph._edges)


def _prefix_scalar_reentry_query(
    *,
    tag_name: str = "topic",
    with_clause: str = "knownTag.tagId AS knownTagId",
    return_clause: str = "post.id AS id",
    order_by: Optional[str] = None,
) -> str:
    query = (
        f"MATCH (knownTag:Tag {{ name: '{tag_name}' }}) "
        f"WITH {with_clause} "
        "MATCH (post:Post)-[:HAS_TAG]->(t:Tag {tagId: knownTagId}) "
        f"RETURN {return_clause}"
    )
    if order_by is not None:
        query += f" ORDER BY {order_by}"
    return query


def _mk_multi_alias_edge_projection_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["a1", "a2", "b1", "b2"],
                "label__A": [True, True, False, False],
                "label__B": [False, False, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a1", "a2"],
                "d": ["b1", "b2"],
                "type": ["R", "R"],
                "creationDate": [10, 20],
            }
        ),
    )


def _mk_recent_message_reentry_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            {
                "id": ["viewer", "author1", "post1", "post2", "comment1"],
                "label__Person": [True, True, False, False, False],
                "label__Message": [False, False, True, True, True],
                "label__Post": [False, False, True, True, False],
                "label__Comment": [False, False, False, False, True],
                "creationDate": [None, None, 5, 20, 10],
                "imageFile": [None, None, None, None, None],
                "content": [None, None, "post1", "post2", "comment1"],
                "firstName": ["View", "Ada", None, None, None],
                "lastName": ["Er", "Lovelace", None, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["comment1", "post2", "post1", "comment1"],
                "d": ["viewer", "viewer", "author1", "post1"],
                "type": ["HAS_CREATOR", "HAS_CREATOR", "HAS_CREATOR", "REPLY_OF"],
            }
        ),
    )


def _mk_recent_message_reentry_graph() -> _CypherTestGraph:
    return _mk_graph(*_mk_recent_message_reentry_data())


def _mk_recent_message_reentry_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(*_mk_recent_message_reentry_data())


def _mk_recent_message_reentry_graph_branching() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["viewer", "author1", "author2", "post1", "post2", "comment1"],
                "label__Person": [True, True, True, False, False, False],
                "label__Message": [False, False, False, True, True, True],
                "label__Post": [False, False, False, True, True, False],
                "label__Comment": [False, False, False, False, False, True],
                "creationDate": [None, None, None, 5, 6, 10],
                "imageFile": [None, None, None, None, None, None],
                "content": [None, None, None, "post1", "post2", "comment1"],
                "firstName": ["View", "Ada", "Grace", None, None, None],
                "lastName": ["Er", "Lovelace", "Hopper", None, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["comment1", "post1", "post2", "comment1", "comment1"],
                "d": ["viewer", "author1", "author2", "post1", "post2"],
                "type": ["HAS_CREATOR", "HAS_CREATOR", "HAS_CREATOR", "REPLY_OF", "REPLY_OF"],
            }
        ),
    )


def _mk_multihop_row_binding_cycle_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "label__A": [True, False],
                "label__B": [False, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a", "b"],
                "d": ["b", "a"],
                "type": ["R", "R"],
            }
        ),
    )


def _compiled_reentry_projection_outputs(compiled: CompiledCypherQuery) -> Tuple[str, Tuple[str, ...]]:
    assert compiled.start_nodes_query is not None
    projection = compiled.start_nodes_query.result_projection
    assert projection is not None
    whole_row_columns = tuple(column.output_name for column in projection.columns if column.kind == "whole_row")
    carried_columns = tuple(column.output_name for column in projection.columns if column.kind != "whole_row")
    assert len(whole_row_columns) == 1
    return whole_row_columns[0], carried_columns


def _reentry_query(
    with_clause: str,
    *,
    return_clause: str,
    match_alias: str = "a",
    order_by: Optional[str] = None,
    where_clause: Optional[str] = None,
) -> str:
    parts = ["MATCH (a:A) ", f"WITH {with_clause} ", f"MATCH ({match_alias})-->(b) "]
    if where_clause is not None:
        parts.append(f"WHERE {where_clause} ")
    parts.append(f"RETURN {return_clause}")
    if order_by is not None:
        parts.append(f" ORDER BY {order_by}")
    return "".join(parts)


def _assert_query_rows(
    query: str,
    expected_rows: list[dict[str, object]],
    *,
    nodes_df: pd.DataFrame | None = None,
    edges_df: pd.DataFrame | None = None,
) -> None:
    graph = (
        _mk_empty_graph()
        if nodes_df is None and edges_df is None
        else _mk_graph(
            pd.DataFrame({"id": []}) if nodes_df is None else nodes_df,
            pd.DataFrame({"s": [], "d": []}) if edges_df is None else edges_df,
        )
    )
    result = graph.gfql(query)
    assert result._nodes.to_dict(orient="records") == expected_rows


def _to_pandas_df(df: Any) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    # RAPIDS 25.02 can segfault on some direct to_pandas() paths; prefer Arrow.
    if hasattr(df, "to_arrow"):
        return cast(pd.DataFrame, df.to_arrow().to_pandas())
    return cast(pd.DataFrame, df.to_pandas() if hasattr(df, "to_pandas") else df)


def _parse_query(query: str) -> CypherQuery:
    return cast(CypherQuery, parse_cypher(query))


def _compile_query(query: str) -> CompiledCypherQuery:
    return cast(CompiledCypherQuery, compile_cypher(query))


def _compiled_execution_extras(compiled: CompiledCypherQuery) -> CompiledCypherExecutionExtras:
    return compiled.execution_extras or CompiledCypherExecutionExtras()


def _logical_plan_route(compiled: CompiledCypherQuery) -> str:
    if compiled.logical_plan is not None:
        return "planned"
    if compiled.logical_plan_defer_reason is not None:
        return "deferred"
    return "none"


def test_compiled_query_escape_hatches_are_grouped() -> None:
    field_names = {f.name for f in dataclass_fields(CompiledCypherQuery)}
    assert "post_processing" in field_names
    assert "execution_extras" in field_names
    assert "result_projection" not in field_names
    assert "empty_result_row" not in field_names
    assert "connected_match_join" not in field_names
    assert "connected_optional_match" not in field_names


def test_compiled_query_execution_extras_retire_legacy_scalar_reentry_fields() -> None:
    field_names = {f.name for f in dataclass_fields(CompiledCypherExecutionExtras)}
    assert "reentry_plan" in field_names
    assert "scalar_reentry_alias" not in field_names
    assert "scalar_reentry_columns" not in field_names


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n:Person) RETURN n",
        "MATCH (n:Person) RETURN 1 AS x",
        "MATCH (n) RETURN DISTINCT n.id AS id",
        "MATCH (a:A) WITH a MATCH (a) RETURN a",
        "MATCH (a:A) WITH a MATCH (a)-->(b) RETURN b",
        "MATCH (a)-[r]->(b) RETURN r",
        "MATCH (a:A) WITH a MATCH (a)-[r]->(b) RETURN r",
        "UNWIND [1,2] AS n RETURN n",
        "WITH 1 AS n RETURN n",
        "MATCH (a:A) WITH a OPTIONAL MATCH (a)-->(b) RETURN b",
    ],
)
def test_compiled_query_sets_logical_plan_route_for_planned_shapes(query: str) -> None:
    compiled = _compile_query(query)
    assert _logical_plan_route(compiled) == "planned"
    assert compiled.logical_plan is not None
    assert compiled.logical_plan_defer_reason is None


def test_compiled_query_threads_bound_scope_stack_for_runtime_passes() -> None:
    compiled = _compile_query("MATCH (n:Person) RETURN n")
    scope_stack = _compiled_execution_extras(compiled).scope_stack
    assert scope_stack
    assert scope_stack[0].origin_clause.upper() == "MATCH"
    assert scope_stack[-1].origin_clause.upper() == "RETURN"
    assert "n" in scope_stack[-1].visible_vars


def test_compiled_query_sets_logical_plan_route_for_top_level_optional_shape() -> None:
    compiled = _compile_query("OPTIONAL MATCH (n:Person) RETURN n")
    assert _logical_plan_route(compiled) == "planned"
    assert compiled.logical_plan is not None
    assert compiled.logical_plan_defer_reason is None
    optional_match = compiled.logical_plan.input if hasattr(compiled.logical_plan, "input") else compiled.logical_plan
    assert isinstance(optional_match, PatternMatch)
    assert optional_match.optional is True
    assert optional_match.arm_id == "top_level_optional_0"


def test_logical_plan_route_for_query_defers_unknown_alias_match_shape_by_default() -> None:
    query = _parse_query("MATCH (n:Person) RETURN n")
    bound_ir = BoundIR(
        query_parts=[
            BoundQueryPart(clause="MATCH", outputs=frozenset({"ghost"})),
            BoundQueryPart(clause="RETURN", outputs=frozenset({"ghost"})),
        ],
        semantic_table=SemanticTable(variables={}),
    )
    logical_plan, defer_reason, defer_code = _logical_plan_route_for_query(query, bound_ir=bound_ir)
    assert logical_plan is None
    assert defer_reason is not None
    assert defer_code is None
    assert "present in semantic scope" in defer_reason


def test_logical_plan_route_for_query_allows_unknown_alias_match_shape_when_opted_in() -> None:
    query = _parse_query("MATCH (n:Person) RETURN n")
    bound_ir = BoundIR(
        query_parts=[
            BoundQueryPart(clause="MATCH", outputs=frozenset({"ghost"})),
            BoundQueryPart(clause="RETURN", outputs=frozenset({"ghost"})),
        ],
        semantic_table=SemanticTable(variables={}),
    )
    logical_plan, defer_reason, defer_code = _logical_plan_route_for_query(
        query, bound_ir=bound_ir, allow_unknown_match_aliases=True
    )
    assert logical_plan is not None
    assert defer_reason is None
    assert defer_code is None


def test_logical_plan_route_for_query_emits_filter_for_where_predicate() -> None:
    # Compilation emits a Filter node for the WHERE clause; predicate pushdown into
    # PatternMatch.predicates happens later in the runtime pass pipeline (gfql_unified.py).
    query = _parse_query("MATCH (a)-[r]->(b) WHERE r.weight > 5 RETURN b")
    bound_ir = FrontendBinder().bind(query, PlanContext())

    logical_plan, defer_reason, defer_code = _logical_plan_route_for_query(query, bound_ir=bound_ir)

    assert logical_plan is not None
    assert defer_reason is None
    assert defer_code is None

    def _walk(node):  # noqa: ANN001, ANN202
        yield node
        for slot in CHILD_SLOTS:
            child = getattr(node, slot, None)
            if child is not None:
                yield from _walk(child)

    nodes = list(_walk(logical_plan))
    # Predicate is in a Filter node — not yet pushed into PatternMatch
    assert any(isinstance(node, Filter) and "alias='r'" in node.predicate.expression for node in nodes)
    pattern_nodes = [node for node in nodes if isinstance(node, PatternMatch)]
    assert pattern_nodes
    assert not any("alias='r'" in pred.expression for pred in pattern_nodes[0].predicates)


def test_compiled_query_sets_logical_plan_route_for_call_shape() -> None:
    compiled = _compile_query("CALL graphistry.degree()")
    assert compiled.procedure_call is not None
    assert _logical_plan_route(compiled) == "planned"
    assert compiled.logical_plan is not None
    assert isinstance(compiled.logical_plan, LogicalProcedureCall)
    assert compiled.logical_plan.procedure == "graphistry.degree"
    assert compiled.logical_plan.result_kind == "rows"
    assert compiled.logical_plan_defer_reason is None


def test_connected_optional_query_sets_query_graph_and_logical_plan() -> None:
    compiled = _compile_query("MATCH (a)-[:A]->(b) OPTIONAL MATCH (b)-[:B]->(c) RETURN c")
    compiled_extras = _compiled_execution_extras(compiled)
    assert compiled_extras.connected_optional_match is not None
    assert _logical_plan_route(compiled) == "planned"
    assert compiled.logical_plan is not None
    assert compiled_extras.query_graph is not None
    assert len(compiled_extras.query_graph.components) == 2
    assert len(compiled_extras.query_graph.optional_arms) == 1


def test_compile_graph_query_sets_logical_plan_route_for_call_constructor() -> None:
    compiled = compile_cypher("GRAPH { CALL graphistry.degree.write() }")
    assert isinstance(compiled, CompiledCypherGraphQuery)
    assert compiled.procedure_call is not None
    assert compiled.logical_plan is not None
    assert compiled.logical_plan_defer_reason is None
    assert compiled.logical_plan is not None
    assert isinstance(compiled.logical_plan, LogicalProcedureCall)
    assert compiled.logical_plan.procedure == "graphistry.degree.write"
    assert compiled.logical_plan.result_kind == "graph"
    assert compiled.logical_plan_defer_reason is None


def test_compile_query_sets_logical_plan_route_for_graph_binding_call_constructor() -> None:
    compiled = _compile_query(
        "GRAPH g1 = GRAPH { CALL graphistry.degree.write() } "
        "USE g1 MATCH (n) RETURN n.id AS id ORDER BY id"
    )
    assert len(compiled.graph_bindings) == 1
    binding = compiled.graph_bindings[0]
    assert binding.procedure_call is not None
    assert binding.logical_plan is not None
    assert binding.logical_plan_defer_reason is None
    assert binding.logical_plan is not None
    assert isinstance(binding.logical_plan, LogicalProcedureCall)
    assert binding.logical_plan.procedure == "graphistry.degree.write"
    assert binding.logical_plan.result_kind == "graph"


def test_compile_graph_query_sets_logical_plan_route_for_match_constructor_shape() -> None:
    compiled = compile_cypher("GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'a' }")
    assert isinstance(compiled, CompiledCypherGraphQuery)
    assert compiled.logical_plan is not None
    assert compiled.logical_plan_defer_reason is None
    assert compiled.logical_plan is not None
    assert compiled.logical_plan_defer_reason is None


def test_lower_match_clause_to_gfql_ops() -> None:
    parsed = _parse_query(
        "MATCH (p:Person {id: $person_id})-[r:FOLLOWS]->(q:Person {active: true}) RETURN p"
    )

    assert parsed.match is not None
    ops = lower_match_clause(parsed.match, params={"person_id": 1})

    assert len(ops) == 3
    assert isinstance(ops[0], ASTNode)
    assert isinstance(ops[1], ASTEdgeForward)
    assert isinstance(ops[2], ASTNode)

    assert ops[0].filter_dict == {"label__Person": True, "id": 1}
    assert ops[0]._name == "p"
    assert ops[1].edge_match == {"type": "FOLLOWS"}
    assert ops[1]._name == "r"
    assert ops[2].filter_dict == {"label__Person": True, "active": True}
    assert ops[2]._name == "q"


def test_string_cypher_supports_same_path_alias_comparison_where_runtime() -> None:
    result = _mk_reentry_carried_scalar_graph().gfql(
        "MATCH (a:A)-[:R]->(b) WHERE b.num = a.num RETURN b.id AS bid ORDER BY bid"
    )

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1"}]


def test_string_cypher_preserves_alias_frames_for_same_path_multi_column_projection() -> None:
    result = _mk_reentry_carried_scalar_graph().gfql(
        "MATCH (a:A)-[:R]->(b) WHERE b.num = a.num RETURN a.id AS aid, b.id AS bid ORDER BY aid, bid"
    )

    assert result._nodes.to_dict(orient="records") == [{"aid": "a1", "bid": "b1"}]


def test_string_cypher_supports_same_path_expression_valued_pattern_property() -> None:
    parsed = _parse_query("MATCH (a:A)-[:R]->(b {num: a.num}) RETURN b.id AS bid ORDER BY bid")
    lowered = lower_match_query(parsed)

    assert lowered.row_where is None
    assert lowered.where == [compare(col("b", "num"), "==", col("a", "num"))]

    result = _mk_reentry_carried_scalar_graph().gfql(
        "MATCH (a:A)-[:R]->(b {num: a.num}) RETURN b.id AS bid ORDER BY bid"
    )

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1"}]


def test_lower_match_query_keeps_literal_pattern_filters_alongside_dynamic_property_entries() -> None:
    parsed = _parse_query("MATCH (a:A {id: 'a1'})-[:R]->(b {id: 'b1', num: a.num}) RETURN b.id AS bid")
    lowered = lower_match_query(parsed)

    assert lowered.row_where is None
    assert lowered.where == [compare(col("b", "num"), "==", col("a", "num"))]
    assert isinstance(lowered.query[0], ASTNode)
    assert isinstance(lowered.query[2], ASTNode)
    assert lowered.query[0].filter_dict == {"id": "a1", "label__A": True}
    assert lowered.query[2].filter_dict == {"id": "b1"}


def test_lower_match_query_falls_back_to_row_where_for_non_property_dynamic_pattern_expression() -> None:
    parsed = _parse_query("MATCH (a:A)-[:R]->(b {num: a.num + 0}) RETURN b.id AS bid")
    lowered = lower_match_query(parsed)

    assert lowered.where == []
    assert lowered.row_where is not None
    assert lowered.row_where.text == "b.num = (a.num + 0)"


def test_string_cypher_supports_non_property_dynamic_pattern_expression_runtime() -> None:
    result = _mk_reentry_carried_scalar_graph().gfql(
        "MATCH (a:A)-[:R]->(b {num: a.num + 0}) RETURN b.id AS bid ORDER BY bid"
    )

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1"}]


def test_lower_match_query_supports_relationship_expression_valued_pattern_property() -> None:
    parsed = _parse_query("MATCH (a:A)-[r:R {weight: a.num}]->(b) RETURN b.id AS bid ORDER BY bid")
    lowered = lower_match_query(parsed)

    assert lowered.row_where is None
    assert lowered.where == [compare(col("r", "weight"), "==", col("a", "num"))]


def test_string_cypher_supports_relationship_expression_valued_pattern_property_runtime() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a1", "a2", "b1", "b2"],
                "label__A": [True, True, False, False],
                "num": [1, 2, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a1", "a2"],
                "d": ["b1", "b2"],
                "type": ["R", "R"],
                "weight": [1, 99],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (a:A)-[r:R {weight: a.num}]->(b) RETURN b.id AS bid ORDER BY bid"
    )

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1"}]


def test_string_cypher_supports_reentry_identifier_valued_pattern_property() -> None:
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b:B {id: bid})-[:S]->(c:C) "
        "RETURN bid, c.id AS cid"
    )

    result = _mk_connected_reentry_carried_scalar_graph().gfql(query, params={"seed": "a1"})

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1", "cid": "c1"}]


def test_string_cypher_supports_reentry_relationship_expression_valued_pattern_property() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a1", "b1", "b2", "c1", "c2"],
                "label__A": [True, False, False, False, False],
                "label__B": [False, True, True, False, False],
                "label__C": [False, False, False, True, True],
                "score": [None, 10, 20, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a1", "a1", "b1", "b2"],
                "d": ["b1", "b2", "c1", "c2"],
                "type": ["R", "R", "S", "S"],
                "score": [None, None, 10, 99],
            }
        ),
    )
    query = (
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH b, b.score AS bscore "
        "MATCH (b)-[rel:S {score: bscore}]->(c:C) "
        "RETURN b.id AS bid, c.id AS cid "
        "ORDER BY bid, cid"
    )

    result = graph.gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1", "cid": "c1"}]


@pytest.mark.parametrize(
    "query,edge_type",
    [
        ("MATCH (a)<-[r:KNOWS]-(b) RETURN r", ASTEdgeReverse),
        ("MATCH (a)-[r:KNOWS]-(b) RETURN r", ASTEdgeUndirected),
        ("MATCH (a)-->(b) RETURN b", ASTEdgeForward),
    ],
)
def test_lower_match_clause_relationship_direction(query: str, edge_type: type) -> None:
    parsed = _parse_query(query)
    assert parsed.match is not None
    ops = lower_match_clause(parsed.match)
    assert isinstance(ops[1], edge_type)


@pytest.mark.parametrize(
    "query,edge_type,min_hops,max_hops,to_fixed_point,edge_match",
    [
        ("MATCH (a)-[*2]->(b) RETURN b", ASTEdgeForward, 2, 2, False, None),
        ("MATCH (a)<-[*3]-(b) RETURN b", ASTEdgeReverse, 3, 3, False, None),
        ("MATCH (a)-[:R*1..4]-(b) RETURN b", ASTEdgeUndirected, 1, 4, False, {"type": "R"}),
        ("MATCH (a)-[*]->(b) RETURN b", ASTEdgeForward, None, None, True, None),
    ],
)
def test_lower_match_clause_variable_length_relationships(
    query: str,
    edge_type: type,
    min_hops: int | None,
    max_hops: int | None,
    to_fixed_point: bool,
    edge_match: dict[str, object] | None,
) -> None:
    parsed = _parse_query(query)
    assert parsed.match is not None

    ops = lower_match_clause(parsed.match)

    assert isinstance(ops[1], edge_type)
    edge = ops[1]
    assert edge.min_hops == min_hops
    assert edge.max_hops == max_hops
    assert edge.to_fixed_point is to_fixed_point
    assert edge.edge_match == edge_match


def test_lower_match_clause_relationship_type_alternation_uses_is_in_predicate() -> None:
    parsed = _parse_query("MATCH (n)-[r:KNOWS|HATES]->(x) RETURN r")
    assert parsed.match is not None

    ops = lower_match_clause(parsed.match)

    assert isinstance(ops[1], ASTEdgeForward)
    assert isinstance(ops[1].edge_match, dict)
    type_predicate = ops[1].edge_match["type"]
    assert isinstance(type_predicate, IsIn)
    assert type_predicate.options == ["KNOWS", "HATES"]


def test_lower_match_clause_stitches_connected_comma_patterns() -> None:
    parsed = _parse_query("MATCH (a)-[:A]->(b), (b)-[:B]->(c) RETURN c")
    assert parsed.match is not None

    ops = lower_match_clause(parsed.match)

    assert len(ops) == 5
    assert isinstance(ops[0], ASTNode)
    assert isinstance(ops[1], ASTEdgeForward)
    assert isinstance(ops[2], ASTNode)
    assert isinstance(ops[3], ASTEdgeForward)
    assert isinstance(ops[4], ASTNode)
    assert ops[0]._name == "a"
    assert ops[2]._name == "b"
    assert ops[4]._name == "c"


def test_lower_match_clause_stitches_reversed_second_comma_segment() -> None:
    parsed = _parse_query("MATCH (a)-[:R1]->(b), (c)<-[:R2]-(b) RETURN c")
    assert parsed.match is not None

    ops = lower_match_clause(parsed.match)

    assert len(ops) == 5
    assert isinstance(ops[1], ASTEdgeForward)
    assert isinstance(ops[3], ASTEdgeForward)
    assert cast(ASTNode, ops[2])._name == "b"
    assert cast(ASTNode, ops[4])._name == "c"


def test_lower_match_clause_rejects_disconnected_comma_patterns() -> None:
    parsed = _parse_query("MATCH (a)-[:A]->(b), (c)-[:B]->(d) RETURN d")
    assert parsed.match is not None

    with pytest.raises(GFQLValidationError, match="single linear connected path"):
        lower_match_clause(parsed.match)


def test_lower_match_query_executes_seeded_repeated_match_for_connected_pattern() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "x1", "x2"],
            "name": ["A", "B", "x1", "x2"],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "a", "b", "b"],
            "d": ["x1", "x2", "x1", "x2"],
            "type": ["KNOWS", "KNOWS", "KNOWS", "KNOWS"],
        }
    )

    chain = cypher_to_gfql(
        "MATCH (a {name: 'A'}), (b {name: 'B'}) MATCH (a)-[:KNOWS]->(x)<-[:KNOWS]-(b) RETURN x.name ORDER BY x.name"
    )
    result = _mk_graph(nodes, edges).gfql(chain)

    assert result._nodes[["x.name"]].to_dict(orient="records") == [
        {"x.name": "x1"},
        {"x.name": "x2"},
    ]


def test_lower_match_query_rejects_seed_alias_not_used_by_final_pattern() -> None:
    parsed = _parse_query("MATCH (a {name: 'A'}), (c {name: 'C'}) MATCH (a)-->(b) RETURN b")

    with pytest.raises(GFQLValidationError, match="must participate in the final connected MATCH pattern"):
        lower_match_query(parsed)


def test_lower_match_query_rewrites_duplicate_node_aliases_to_internal_identity_checks() -> None:
    parsed = _parse_query("MATCH (n)-[r]-(n) RETURN count(r)")

    lowered = lower_match_query(parsed)

    assert len(lowered.query) == 3
    repeated = cast(ASTNode, lowered.query[0])
    assert isinstance(repeated, ASTNode)
    assert repeated._name is not None
    assert repeated._name.startswith("__cypher_aliasdup_n")
    assert lowered.where == [
        compare(
            col("n", NODE_IDENTITY_COLUMN),
            "==",
            col(repeated._name, NODE_IDENTITY_COLUMN),
        )
    ]


def test_string_cypher_duplicate_node_alias_identity_shape_uses_bound_node_column() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a"]}),
        pd.DataFrame({"s": ["a"], "d": ["a"], "type": ["LOOP"]}),
    )

    result = graph.gfql("MATCH (n)-[:LOOP]->(n) RETURN count(*) AS loops")

    assert result._nodes.to_dict(orient="records") == [{"loops": 1}]


def test_issue_1490_repeated_node_alias_uses_bound_node_identity() -> None:
    nodes = pd.DataFrame(
        {
            "__node_id__": ["node-a", "node-b"],
            "id": [0, 0],
            "name": ["A", "B"],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["node-a", "node-b", "node-a"],
            "d": ["node-a", "node-b", "node-b"],
            "type": ["LOOP", "LOOP", "R"],
        }
    )
    graph = cast(_CypherTestGraph, _CypherTestGraph().nodes(nodes, "__node_id__").edges(edges, "s", "d"))

    result = graph.gfql("MATCH (n)-[:LOOP]->(n) RETURN count(*) AS loops")

    assert result._nodes.to_dict(orient="records") == [{"loops": 2}]


def test_issue_1490_user_id_property_remains_projectable_with_custom_identity() -> None:
    nodes = pd.DataFrame(
        {
            "__node_id__": ["node-a", "node-b"],
            "id": [0, 0],
            "name": ["A", "B"],
        }
    )
    edges = pd.DataFrame({"s": [], "d": [], "type": []})
    graph = cast(_CypherTestGraph, _CypherTestGraph().nodes(nodes, "__node_id__").edges(edges, "s", "d"))

    result = graph.gfql(
        "MATCH (n) "
        "RETURN n.id AS user_id, n.__node_id__ AS identity "
        "ORDER BY identity"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"user_id": 0, "identity": "node-a"},
        {"user_id": 0, "identity": "node-b"},
    ]


def test_issue_1490_distinct_and_grouping_use_bound_identity_not_user_id() -> None:
    nodes = pd.DataFrame(
        {
            "__node_id__": ["node-a", "node-b"],
            "id": [0, 0],
            "name": ["A", "B"],
        }
    )
    edges = pd.DataFrame({"s": [], "d": [], "type": []})
    graph = cast(_CypherTestGraph, _CypherTestGraph().nodes(nodes, "__node_id__").edges(edges, "s", "d"))

    distinct_result = graph.gfql("MATCH (n) RETURN count(DISTINCT n) AS c")
    grouped_result = graph.gfql("MATCH (n) RETURN n AS node, count(*) AS c ORDER BY node")

    assert distinct_result._nodes.to_dict(orient="records") == [{"c": 2}]
    assert [row["c"] for row in grouped_result._nodes.to_dict(orient="records")] == [1, 1]


def test_issue_1490_custom_identity_user_id_property_cudf() -> None:
    cudf = _require_cudf_runtime()
    nodes = pd.DataFrame(
        {
            "__node_id__": ["node-a", "node-b"],
            "id": [0, 0],
            "name": ["A", "B"],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["node-a", "node-b", "node-a"],
            "d": ["node-a", "node-b", "node-b"],
            "type": ["LOOP", "LOOP", "R"],
        }
    )
    graph = cast(
        _CypherTestGraph,
        _CypherTestGraph()
        .nodes(cudf.from_pandas(nodes), "__node_id__")
        .edges(cudf.from_pandas(edges), "s", "d"),
    )

    loops = graph.gfql("MATCH (n)-[:LOOP]->(n) RETURN count(*) AS loops", engine="cudf")
    projected = graph.gfql(
        "MATCH (n) "
        "RETURN n.id AS user_id, n.__node_id__ AS identity "
        "ORDER BY identity",
        engine="cudf",
    )
    distinct_result = graph.gfql("MATCH (n) RETURN count(DISTINCT n) AS c", engine="cudf")

    assert _to_pandas_df(loops._nodes).to_dict(orient="records") == [{"loops": 2}]
    assert _to_pandas_df(projected._nodes).to_dict(orient="records") == [
        {"user_id": 0, "identity": "node-a"},
        {"user_id": 0, "identity": "node-b"},
    ]
    assert _to_pandas_df(distinct_result._nodes).to_dict(orient="records") == [{"c": 2}]


def test_parse_where_pattern_predicate() -> None:
    parsed = _parse_query("MATCH (n) WHERE (n)-[:R]->() RETURN n")

    assert parsed.where is not None
    assert len(parsed.where.predicates) == 1
    assert isinstance(parsed.where.predicates[0], WherePatternPredicate)


def test_lower_match_query_executes_connected_comma_pattern_with_unique_projection_alias() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "name": ["a", "b", "c"],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "b", "b"],
            "d": ["b", "a", "c"],
            "type": ["A", "B", "B"],
        }
    )

    chain = cypher_to_gfql("MATCH (a)-[:A]->(b), (b)-[:B]->(c) RETURN c.name")
    result = _mk_graph(nodes, edges).gfql(chain)

    assert result._nodes[["c.name"]].to_dict(orient="records") == [
        {"c.name": "a"},
        {"c.name": "c"},
    ]


def test_gfql_executes_positive_where_pattern_predicate_as_seeded_match() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["R", "X"]})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n) WHERE (n)-[:R]->() RETURN n.id AS id ORDER BY id"
    )

    assert result._nodes.to_dict(orient="records") == [{"id": "a"}]


def test_gfql_executes_reverse_fixed_point_where_pattern_predicate_as_seeded_match() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "x", "y"]})
    edges = pd.DataFrame(
        {
            "s": ["a", "b", "x"],
            "d": ["b", "c", "y"],
            "type": ["R", "R", "R"],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n) WHERE (n)<-[:R*]-() RETURN n.id AS id ORDER BY id"
    )

    assert result._nodes.to_dict(orient="records") == [{"id": "b"}, {"id": "c"}, {"id": "y"}]


def test_gfql_executes_undirected_fixed_point_where_pattern_predicate_as_seeded_match() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "x", "y"]})
    edges = pd.DataFrame(
        {
            "s": ["a", "b", "x"],
            "d": ["b", "c", "y"],
            "type": ["R", "R", "R"],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n) WHERE (n)-[:R*]-() RETURN n.id AS id ORDER BY id"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"id": "a"},
        {"id": "b"},
        {"id": "c"},
        {"id": "x"},
        {"id": "y"},
    ]


def test_gfql_executes_positive_where_pattern_predicate_between_bound_aliases_for_single_source_projection() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({"s": ["a", "a"], "d": ["b", "c"], "type": ["R", "X"]})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n), (m) "
        "WHERE (n)-[:R]->(m) "
        "RETURN n.id AS n_id "
        "ORDER BY n_id"
    )

    assert result._nodes.to_dict(orient="records") == [{"n_id": "a"}]


def test_gfql_executes_multi_positive_where_pattern_predicates_as_intersected_seed() -> None:
    # Slice 3 of #1031: AND-joined positive WHERE pattern predicates against a
    # single bound seed alias execute as the intersection (rows where ALL
    # patterns exist).  Fixture: ``a`` has both R and T outgoing; ``b`` has
    # both; ``c`` has only T; ``d`` has neither.  Expected result: only
    # rows where both relationships exist.
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    edges = pd.DataFrame(
        {
            "s": ["a", "a", "b", "b", "c"],
            "d": ["b", "c", "c", "d", "d"],
            "type": ["R", "T", "R", "T", "T"],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n) WHERE (n)-[:R]->() AND (n)-[:T]->() RETURN n.id AS id ORDER BY id"
    )

    assert result._nodes.to_dict(orient="records") == [{"id": "a"}, {"id": "b"}]


def test_compile_cypher_supports_cartesian_node_only_bindings_rows() -> None:
    compiled = _compile_query(
        "MATCH (n), (m) RETURN n.num AS n_num, m.num AS m_num ORDER BY n_num, m_num"
    )

    row_call = next(
        op for op in compiled.chain.chain
        if isinstance(op, ASTCall) and op.function == "rows"
    )
    binding_ops = row_call.params.get("binding_ops")

    assert isinstance(binding_ops, list)
    assert [op["type"] for op in binding_ops] == ["Node", "Node"]
    assert [op.get("name") for op in binding_ops] == ["n", "m"]


def test_compile_cypher_records_cartesian_multi_whole_row_projection_sources() -> None:
    compiled = _compile_query("MATCH (a), (b) WHERE a = b RETURN a, b")

    assert compiled.result_projection is not None
    assert [column.output_name for column in compiled.result_projection.columns] == ["a", "b"]
    assert [column.kind for column in compiled.result_projection.columns] == ["whole_row", "whole_row"]
    assert [column.source_name for column in compiled.result_projection.columns] == ["a", "b"]


def test_compile_cypher_records_cartesian_aliased_whole_row_projection_sources() -> None:
    compiled = _compile_query("MATCH (a), (b) WHERE a = b RETURN a AS left, b AS right")

    assert compiled.result_projection is not None
    assert [column.output_name for column in compiled.result_projection.columns] == ["left", "right"]
    assert [column.kind for column in compiled.result_projection.columns] == ["whole_row", "whole_row"]
    assert [column.source_name for column in compiled.result_projection.columns] == ["a", "b"]


def test_lower_match_query_converts_cartesian_property_join_to_row_where_expression() -> None:
    lowered = lower_match_query(_parse_query("MATCH (a:A), (b:B) WHERE a.k = b.k RETURN a, b"))

    assert lowered.where == []
    assert lowered.row_where is not None
    assert lowered.row_where.text == "a.k = b.k"


def test_lower_match_query_converts_cartesian_dynamic_pattern_property_to_row_where_expression() -> None:
    lowered = lower_match_query(
        _parse_query("MATCH (a:A), (b:B {num: a.num}) RETURN a.id AS aid, b.id AS bid")
    )

    assert lowered.where == []
    assert lowered.row_where is not None
    assert lowered.row_where.text == "b.num = (a.num)"


def test_string_cypher_supports_cartesian_node_only_scalar_projection() -> None:
    result = _mk_cartesian_node_graph().gfql(
        "MATCH (n), (m) "
        "RETURN n.num AS n_num, m.num AS m_num "
        "ORDER BY n_num, m_num"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"n_num": 1, "m_num": 1},
        {"n_num": 1, "m_num": 2},
        {"n_num": 2, "m_num": 1},
        {"n_num": 2, "m_num": 2},
    ]


def test_string_cypher_supports_cartesian_node_only_row_filter_between_aliases() -> None:
    result = _mk_cartesian_node_graph().gfql(
        "MATCH (n), (m) "
        "WHERE n.num < m.num "
        "RETURN n.num AS n_num, m.num AS m_num "
        "ORDER BY n_num, m_num"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"n_num": 1, "m_num": 2},
    ]


def test_string_cypher_rejects_cartesian_where_pattern_predicates_mixed_with_or() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": [0, 1, 2],
                "label__TheLabel": [False, True, False],
                "label__MissingLabel": [False, False, False],
            }
        ),
        pd.DataFrame({"s": [0, 0, 1], "d": [1, 2, 2], "type": ["T", "X", "T"]}),
    )
    with pytest.raises(GFQLValidationError, match="OR/XOR"):
        graph.gfql(
            "MATCH (a), (b) "
            "WHERE a.id = 0 AND (a)-[:T]->(b:TheLabel) OR (a)-[:T*]->(b:MissingLabel) "
            "RETURN DISTINCT b"
        )


def test_string_cypher_rejects_cartesian_where_pattern_predicates_mixed_with_xor() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": [0, 1], "label__TheLabel": [False, True]}),
        pd.DataFrame({"s": [0], "d": [1], "type": ["T"]}),
    )
    with pytest.raises(GFQLValidationError, match="OR/XOR"):
        graph.gfql(
            "MATCH (a), (b) "
            "WHERE (a)-[:T]->(b:TheLabel) XOR a.id = 0 "
            "RETURN DISTINCT b"
        )


def test_string_cypher_supports_cartesian_scalar_or_without_pattern_predicates() -> None:
    result = _mk_cartesian_node_graph().gfql(
        "MATCH (a), (b) "
        "WHERE a.num = 1 OR b.num = 1 "
        "RETURN a.num AS a_num, b.num AS b_num "
        "ORDER BY a_num, b_num"
    )
    assert result._nodes.to_dict(orient="records") == [
        {"a_num": 1, "b_num": 1},
        {"a_num": 1, "b_num": 2},
        {"a_num": 2, "b_num": 1},
    ]


def test_string_cypher_supports_cartesian_dynamic_pattern_property_projection() -> None:
    graph = _mk_cartesian_dynamic_pattern_graph()

    result = graph.gfql(
        "MATCH (a:A), (b:B {num: a.num}) "
        "RETURN a.id AS aid, b.id AS bid "
        "ORDER BY aid, bid"
    )

    assert result._nodes.to_dict(orient="records") == [{"aid": "a1", "bid": "b1"}]


def test_string_cypher_supports_cartesian_dynamic_pattern_property_global_count() -> None:
    graph = _mk_cartesian_dynamic_pattern_graph()

    result = graph.gfql("MATCH (a:A), (b:B {num: a.num}) RETURN count(*) AS cnt")

    assert result._nodes.to_dict(orient="records") == [{"cnt": 1}]


def test_string_cypher_supports_cartesian_dynamic_pattern_property_grouped_count() -> None:
    graph = _mk_cartesian_dynamic_pattern_graph()

    result = graph.gfql(
        "MATCH (a:A), (b:B {num: a.num}) "
        "RETURN a.id AS aid, count(*) AS cnt "
        "ORDER BY aid"
    )

    assert result._nodes.to_dict(orient="records") == [{"aid": "a1", "cnt": 1}]


def test_string_cypher_supports_cartesian_dynamic_pattern_property_with_stage_projection() -> None:
    graph = _mk_cartesian_dynamic_pattern_graph()

    result = graph.gfql(
        "MATCH (a:A), (b:B {num: a.num}) "
        "WITH a.id AS aid, b.id AS bid "
        "RETURN aid, bid "
        "ORDER BY aid, bid"
    )

    assert result._nodes.to_dict(orient="records") == [{"aid": "a1", "bid": "b1"}]


def test_string_cypher_supports_cartesian_node_only_global_count() -> None:
    result = _mk_cartesian_node_graph().gfql(
        "MATCH (n), (m) "
        "RETURN count(*) AS cnt"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"cnt": 4},
    ]


def test_string_cypher_supports_cartesian_node_only_grouped_count() -> None:
    result = _mk_cartesian_node_graph().gfql(
        "MATCH (n), (m) "
        "RETURN n.num AS n_num, count(*) AS cnt "
        "ORDER BY n_num"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"n_num": 1, "cnt": 2},
        {"n_num": 2, "cnt": 2},
    ]


def test_string_cypher_count_non_active_node_alias_in_degree() -> None:
    """#1708: `MATCH (a)-[e]->(b) RETURN b.id, count(a)` (graph-benchmark q1
    "top-k by in-degree") counts a bare NODE alias other than the projection's
    active alias. It must route to the bindings-row source and return the true
    in-degree per `b`, not fail with the misleading "one MATCH" error."""
    nodes = pd.DataFrame({"id": [0, 1, 2, 3, 4, 7, 8, 9]})
    # in-degree: node 9 <- {0,1,2,4}=4 ; node 8 <- {3,0}=2 ; node 7 <- {1}=1
    edges = pd.DataFrame({"s": [0, 1, 2, 3, 0, 1, 4], "d": [9, 9, 9, 8, 8, 7, 9]})
    graph = _mk_graph(nodes, edges)
    result = graph.gfql(
        "MATCH (a)-[e]->(b) RETURN b.id AS id, count(a) AS c ORDER BY c DESC LIMIT 3"
    )
    assert result._nodes.to_dict(orient="records") == [
        {"id": 9, "c": 4},
        {"id": 8, "c": 2},
        {"id": 7, "c": 1},
    ]


def test_string_cypher_count_non_active_node_alias_matches_count_star() -> None:
    """#1708 corollary: `count(<bare node alias>)` equals `count(*)` and
    `count(<active alias>)` for the same q1 shape (each row binds exactly one of
    each endpoint), so all three routes agree."""
    nodes = pd.DataFrame({"id": [0, 1, 2, 3, 8, 9]})
    edges = pd.DataFrame({"s": [0, 1, 2, 3], "d": [9, 9, 9, 8]})
    graph = _mk_graph(nodes, edges)
    tmpl = "MATCH (a)-[e]->(b) RETURN b.id AS id, count({arg}) AS c ORDER BY c DESC LIMIT 3"
    expected = [{"id": 9, "c": 3}, {"id": 8, "c": 1}]
    for arg in ("a", "b", "*"):
        got = graph.gfql(tmpl.format(arg=arg))._nodes.to_dict(orient="records")
        assert got == expected, f"count({arg}) disagreed: {got}"


def test_string_cypher_count_non_active_node_alias_cudf() -> None:
    """#1708: bindings-route count of a non-active node alias also runs on cudf."""
    _require_cudf_runtime()
    nodes = pd.DataFrame({"id": [0, 1, 2, 3, 8, 9]})
    edges = pd.DataFrame({"s": [0, 1, 2, 3], "d": [9, 9, 9, 8]})
    result = _mk_cudf_graph(nodes, edges).gfql(
        "MATCH (a)-[e]->(b) RETURN b.id AS id, count(a) AS c ORDER BY c DESC LIMIT 3"
    )
    assert result._nodes.to_pandas().to_dict(orient="records") == [
        {"id": 9, "c": 3},
        {"id": 8, "c": 1},
    ]


def test_string_cypher_supports_cartesian_node_only_non_simple_scalar_expression() -> None:
    result = _mk_cartesian_node_graph().gfql(
        "MATCH (n), (m) "
        "RETURN n.ts < m.ts AS lt "
        "ORDER BY lt"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"lt": False},
        {"lt": False},
        {"lt": False},
        {"lt": True},
    ]


def test_string_cypher_supports_cartesian_node_only_with_stage_filter() -> None:
    result = _mk_cartesian_node_graph().gfql(
        "MATCH (n), (m) "
        "WITH n.num AS n_num, m.num AS m_num "
        "WHERE n_num < m_num "
        "RETURN n_num, m_num "
        "ORDER BY n_num, m_num"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"n_num": 1, "m_num": 2},
    ]


def test_string_cypher_supports_cartesian_with_stage_identity_join_whole_row_projection() -> None:
    result = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "label__A": [True, False],
                "label__B": [False, True],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    ).gfql("MATCH (a), (b) WITH a, b WHERE a = b RETURN a, b ORDER BY a.id")

    assert entity_text_records(result, {"a": "nodes", "b": "nodes"}) == [
        {"a": "(:A)", "b": "(:A)"},
        {"a": "(:B)", "b": "(:B)"},
    ]


def test_string_cypher_supports_cartesian_with_stage_property_join_whole_row_projection() -> None:
    result = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a1", "a2", "b2", "b3"],
                "label__A": [True, True, False, False],
                "label__B": [False, False, True, True],
                "k": [1, 2, 2, 3],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    ).gfql("MATCH (a:A), (b:B) WITH a, b WHERE a.k = b.k RETURN a, b")

    assert entity_text_records(result, {"a": "nodes", "b": "nodes"}) == [
        {"a": "(:A {k: 2})", "b": "(:B {k: 2})"},
    ]


def test_string_cypher_supports_cartesian_with_stage_inequality_join_whole_row_projection() -> None:
    result = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "label__A": [True, False],
                "label__B": [False, True],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    ).gfql("MATCH (a), (b) WITH a, b WHERE a <> b RETURN a, b ORDER BY a.id, b.id")

    assert entity_text_records(result, {"a": "nodes", "b": "nodes"}) == [
        {"a": "(:A)", "b": "(:B)"},
        {"a": "(:B)", "b": "(:A)"},
    ]


def test_string_cypher_supports_cartesian_aggregate_order_by_expression() -> None:
    result = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a1", "x1", "x2"],
                "label__A": [True, False, False],
                "label__X": [False, True, True],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    ).gfql("MATCH (a:A), (b:X) RETURN count(a) * 10 + count(b) * 5 AS x ORDER BY x")

    assert result._nodes.to_dict(orient="records") == [
        {"x": 30},
    ]


def test_string_cypher_supports_cartesian_node_identity_join_with_whole_row_projection() -> None:
    result = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "label__A": [True, False],
                "label__B": [False, True],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    ).gfql("MATCH (a), (b) WHERE a = b RETURN a, b ORDER BY a.id")

    assert entity_text_records(result, {"a": "nodes", "b": "nodes"}) == [
        {"a": "(:A)", "b": "(:A)"},
        {"a": "(:B)", "b": "(:B)"},
    ]
    entity_meta = getattr(result, "_cypher_entity_projection_meta")
    assert entity_meta["a"]["alias"] == "a"
    assert entity_meta["a"]["ids"].tolist() == ["a", "b"]
    assert entity_meta["b"]["alias"] == "b"
    assert entity_meta["b"]["ids"].tolist() == ["a", "b"]


def test_string_cypher_supports_cartesian_node_property_join_with_whole_row_projection() -> None:
    result = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a1", "a2", "b2", "b3"],
                "label__A": [True, True, False, False],
                "label__B": [False, False, True, True],
                "k": [1, 2, 2, 3],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    ).gfql("MATCH (a:A), (b:B) WHERE a.k = b.k RETURN a, b")

    assert entity_text_records(result, {"a": "nodes", "b": "nodes"}) == [
        {"a": "(:A {k: 2})", "b": "(:B {k: 2})"},
    ]


def test_string_cypher_supports_cartesian_whole_row_projection_aliases() -> None:
    result = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "label__A": [True, False],
                "label__B": [False, True],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    ).gfql("MATCH (a), (b) WHERE a = b RETURN a AS left, b AS right ORDER BY left.id")

    assert entity_text_records(result, {"left": "nodes", "right": "nodes"}) == [
        {"left": "(:A)", "right": "(:A)"},
        {"left": "(:B)", "right": "(:B)"},
    ]
    entity_meta = getattr(result, "_cypher_entity_projection_meta")
    assert entity_meta["left"]["alias"] == "a"
    assert entity_meta["right"]["alias"] == "b"


def test_lower_match_query_rejects_bare_where_pattern_predicate_without_relationship() -> None:
    with pytest.raises(GFQLValidationError, match="must include a relationship"):
        lower_cypher_query(_parse_query("MATCH (n) WHERE (n) RETURN n"))


def test_lower_match_query_supports_multiple_where_pattern_predicates() -> None:
    # AND-joined positive WHERE pattern predicates compile through independent
    # semi-apply markers so each pattern remains an existence check.
    compiled = lower_cypher_query(_parse_query("MATCH (n) WHERE (n)-[:R]->() AND (n)-[:S]->() RETURN n"))
    assert compiled is not None


def test_lower_match_query_emits_row_marker_for_single_positive_where_pattern() -> None:
    lowered = lower_match_query(_parse_query("MATCH (n) WHERE (n)-[:R]->() RETURN n"))

    assert [type(op).__name__ for op in lowered.query] == ["ASTNode"]
    assert len(lowered.row_pre_filters) == 1
    marker = lowered.row_pre_filters[0]
    assert isinstance(marker, ASTCall)
    assert marker.function == "semi_apply_mark"
    assert marker.params.get("join_aliases") == ["n"]
    out_col = marker.params.get("out_col")
    assert isinstance(out_col, str) and out_col.startswith("__gfql_where_pattern_")
    assert lowered.row_where is not None
    assert lowered.row_where.text == out_col
    binding_ops = marker.params.get("binding_ops")
    assert isinstance(binding_ops, list)
    assert [op.get("type") for op in binding_ops] == ["Node", "Edge", "Node"]


def test_lower_match_query_exists_subquery_emits_same_marker_as_bare_pattern() -> None:
    """viz-filter L1: WHERE EXISTS { <pattern> } lowers to the IDENTICAL
    semi_apply_mark shape as the bare pattern predicate (same leaf, zero new ops);
    inline block comments and property maps inside the pattern are fine."""
    for q in [
        "MATCH (n) WHERE EXISTS { (n)-[:R]->() } RETURN n",
        "MATCH (n) WHERE exists/*inline*/{ (n)-[:R]->() } RETURN n",
        "MATCH (n) WHERE EXISTS { (n)-[:R {w: 1}]->() } RETURN n",
    ]:
        lowered = lower_match_query(_parse_query(q))
        assert len(lowered.row_pre_filters) == 1, q
        marker = lowered.row_pre_filters[0]
        assert isinstance(marker, ASTCall), q
        assert marker.function == "semi_apply_mark", q
        assert marker.params.get("join_aliases") == ["n"], q
        binding_ops = marker.params.get("binding_ops")
        assert [op.get("type") for op in binding_ops] == ["Node", "Edge", "Node"], q


def test_lower_match_query_not_exists_subquery_emits_anti_semi_apply() -> None:
    """viz-filter L1: NOT EXISTS { <pattern> } composes through the NOT tier into
    anti_semi_apply — the declarative prune-isolated building block."""
    lowered = lower_match_query(
        _parse_query("MATCH (n) WHERE NOT EXISTS { (n)-[:R]->() } RETURN n.id AS id")
    )
    assert len(lowered.row_pre_filters) == 1
    anti = lowered.row_pre_filters[0]
    assert isinstance(anti, ASTCall)
    assert anti.function == "anti_semi_apply"
    assert anti.params.get("join_aliases") == ["n"]


def test_lower_exists_drop_self_flavor_emits_neq_semi_apply() -> None:
    """viz-filter L1 drop-self prune-isolated: EXISTS { (n)--(m) WHERE m <> n } — the
    existential local alias m is ALLOWED (bindings project it away) and the endpoint
    inequality rides the op as neq=[m, n] (bindings-table filter / self-loop-edge
    exclusion on polars)."""
    lowered = lower_match_query(
        _parse_query("MATCH (n) WHERE EXISTS { (n)--(m) WHERE m <> n } RETURN n.id AS id")
    )
    assert len(lowered.row_pre_filters) == 1
    marker = lowered.row_pre_filters[0]
    assert isinstance(marker, ASTCall)
    assert marker.function == "semi_apply_mark"
    assert marker.params.get("join_aliases") == ["n"]
    assert sorted(marker.params.get("neq") or []) == ["m", "n"]

    anti = lower_match_query(
        _parse_query("MATCH (n) WHERE NOT EXISTS { (n)--(m) WHERE m <> n } RETURN n.id AS id")
    ).row_pre_filters[0]
    assert isinstance(anti, ASTCall)
    assert anti.function == "anti_semi_apply"
    assert sorted(anti.params.get("neq") or []) == ["m", "n"]


def test_lower_search_any_where_emits_marker_prefilter() -> None:
    """viz-filter L2-b: WHERE searchAny(a, 'x'[, {opts}]) lifts to a search_any row
    pre-filter + a fresh marker column in the residual boolean expression (the
    pattern-predicate marker mechanism), composing through AND/OR/NOT."""
    lowered = lower_match_query(_parse_query(
        "MATCH (a) WHERE searchAny(a, 'foo') AND a.x > 1 RETURN a.id AS id"))
    pre = [op for op in lowered.row_pre_filters
           if isinstance(op, ASTCall) and op.function == "search_any"]
    assert len(pre) == 1
    assert pre[0].params["alias"] == "a"
    assert pre[0].params["term"] == "foo"
    marker = pre[0].params["out_col"]
    assert marker.startswith("__gfql_search_any_")
    assert lowered.row_where is not None and marker in lowered.row_where.text
    assert "searchAny" not in lowered.row_where.text

    lowered2 = lower_match_query(_parse_query(
        "MATCH (a) WHERE searchAny(a, '7', {caseSensitive: true, regex: false, columns: ['name']}) RETURN a.id AS id"))
    op2 = [op for op in lowered2.row_pre_filters
           if isinstance(op, ASTCall) and op.function == "search_any"][0]
    assert op2.params.get("case_sensitive") is True
    assert op2.params.get("columns") == ["name"]

    for q, phrase in [
        ("MATCH (a) WHERE searchAny(a, 'x', {nope: true}) RETURN a", "unsupported option"),
        ("MATCH (a) WHERE searchAny(zzz, 'x') RETURN a", "not bound"),
        ("MATCH (a) WHERE searchAny(a, 'x', {caseSensitive: 'yes'}) RETURN a", "true or false"),
    ]:
        with pytest.raises(GFQLValidationError) as exc_info:
            lower_match_query(_parse_query(q))
        assert phrase in exc_info.value.message, (q, exc_info.value.message)


def test_exists_subquery_unsupported_bodies_decline_clearly() -> None:
    """viz-filter L1 v1 boundaries: inner WHERE, multi-pattern bodies, and full
    MATCH..RETURN subquery bodies decline with a clear message (never a wrong answer)."""
    for q, phrase in [
        ("MATCH (a) WHERE EXISTS { (a)--(m) WHERE m.x > 1 } RETURN a", "inner WHERE"),
        ("MATCH (a) WHERE EXISTS { (a)--(), (a)-->() } RETURN a", "multi-pattern"),
        ("MATCH (a) WHERE EXISTS { MATCH (a)--(b) RETURN b } RETURN a", "subquery body"),
        ("MATCH (a) WHERE EXISTS { (a)--(b) WHERE a <> c } RETURN a", "endpoint aliases"),
    ]:
        with pytest.raises(GFQLValidationError) as exc_info:
            _parse_query(q)
        assert phrase in exc_info.value.message, (q, exc_info.value.message)


def test_lower_match_query_emits_row_anti_semi_filter_for_negated_where_pattern() -> None:
    lowered = lower_match_query(_parse_query("MATCH (n) WHERE NOT (n)-[:R]->() RETURN n.id AS id"))

    assert len(lowered.row_pre_filters) == 1
    anti = lowered.row_pre_filters[0]
    assert isinstance(anti, ASTCall)
    assert anti.function == "anti_semi_apply"
    assert anti.params.get("join_aliases") == ["n"]
    binding_ops = anti.params.get("binding_ops")
    assert isinstance(binding_ops, list)
    assert [op.get("type") for op in binding_ops] == ["Node", "Edge", "Node"]


def test_lower_match_query_emits_row_anti_semi_filter_for_bound_alias_negated_where_pattern() -> None:
    lowered = lower_match_query(
        _parse_query("MATCH (a)-[:R]->(b) WHERE NOT (b)-[:R]->(a) RETURN a.id AS a_id, b.id AS b_id")
    )

    assert len(lowered.row_pre_filters) == 1
    anti = lowered.row_pre_filters[0]
    assert isinstance(anti, ASTCall)
    assert anti.function == "anti_semi_apply"
    assert anti.params.get("join_aliases") == ["b", "a"]
    binding_ops = anti.params.get("binding_ops")
    assert isinstance(binding_ops, list)
    assert [op.get("type") for op in binding_ops] == ["Node", "Edge", "Node"]


def test_lower_match_query_emits_row_anti_semi_filter_for_bound_alias_negated_bounded_varlen_where_pattern() -> None:
    lowered = lower_match_query(
        _parse_query("MATCH (a)-[:R]->(b) WHERE NOT (b)-[:R*1..2]->(a) RETURN a.id AS a_id, b.id AS b_id")
    )

    assert len(lowered.row_pre_filters) == 1
    anti = lowered.row_pre_filters[0]
    assert isinstance(anti, ASTCall)
    assert anti.function == "anti_semi_apply"
    assert anti.params.get("join_aliases") == ["b", "a"]
    binding_ops = anti.params.get("binding_ops")
    assert isinstance(binding_ops, list)
    assert [op.get("type") for op in binding_ops] == ["Node", "Edge", "Node"]
    edge = binding_ops[1]
    assert edge.get("min_hops") == 1
    assert edge.get("max_hops") == 2
    assert edge.get("to_fixed_point") is False


def test_lower_match_query_emits_row_marker_for_xor_wrapped_bounded_varlen_where_pattern() -> None:
    lowered = lower_match_query(
        _parse_query("MATCH (n) WHERE (n)-[:R*2]->() XOR n.id = 'd' RETURN n.id AS id")
    )

    assert len(lowered.row_pre_filters) == 1
    marker = lowered.row_pre_filters[0]
    assert isinstance(marker, ASTCall)
    assert marker.function == "semi_apply_mark"
    assert marker.params.get("join_aliases") == ["n"]
    out_col = marker.params.get("out_col")
    assert isinstance(out_col, str) and out_col.startswith("__gfql_where_pattern_")
    assert lowered.row_where is not None
    assert " XOR " in lowered.row_where.text
    assert out_col in lowered.row_where.text
    binding_ops = marker.params.get("binding_ops")
    assert isinstance(binding_ops, list)
    edge = binding_ops[1]
    assert edge.get("min_hops") == 2
    assert edge.get("max_hops") == 2
    assert edge.get("to_fixed_point") is False


def test_lower_match_query_rejects_where_pattern_predicate_introducing_new_aliases() -> None:
    with pytest.raises(GFQLValidationError, match="cannot introduce new aliases"):
        lower_cypher_query(_parse_query("MATCH (n) WHERE (n)-[r]->(a) RETURN n"))


def test_lower_match_clause_executes_through_gfql_runtime() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "type": ["Person", "Person", "Company"],
            "name": ["Alice", "Bob", "Acme"],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "b"],
            "d": ["b", "c"],
            "type": ["FOLLOWS", "WORKS_AT"],
        }
    )

    parsed = _parse_query("MATCH (p:Person {name: 'Alice'})-[r:FOLLOWS]->(q:Person) RETURN p, q")
    assert parsed.match is not None
    ops = lower_match_clause(parsed.match)
    result = _mk_graph(nodes, edges).gfql(ops)

    assert sorted(result._nodes["id"].tolist()) == ["a", "b"]
    assert result._edges[["s", "d", "type"]].to_dict(orient="records") == [
        {"s": "a", "d": "b", "type": "FOLLOWS"}
    ]


def test_lower_match_clause_executes_against_label_boolean_columns() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "label__Person": [True, True, False],
            "label__Company": [False, False, True],
            "name": ["Alice", "Bob", "Acme"],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "b"],
            "d": ["b", "c"],
            "type": ["FOLLOWS", "WORKS_AT"],
        }
    )

    parsed = _parse_query("MATCH (p:Person {name: 'Alice'})-[r:FOLLOWS]->(q:Person) RETURN p, q")
    assert parsed.match is not None
    ops = lower_match_clause(parsed.match)
    result = _mk_graph(nodes, edges).gfql(ops)

    assert sorted(result._nodes["id"].tolist()) == ["a", "b"]
    assert result._edges[["s", "d", "type"]].to_dict(orient="records") == [
        {"s": "a", "d": "b", "type": "FOLLOWS"}
    ]


def test_lower_match_clause_requires_parameters() -> None:
    parsed = _parse_query("MATCH (p {id: $person_id}) RETURN p")

    with pytest.raises(GFQLValidationError) as exc_info:
        assert parsed.match is not None
        lower_match_clause(parsed.match)

    assert exc_info.value.code == ErrorCode.E105


def test_lower_match_clause_supports_multi_label_nodes() -> None:
    parsed = _parse_query("MATCH (p:Person:Admin) RETURN p")

    assert parsed.match is not None
    ops = lower_match_clause(parsed.match)

    assert isinstance(ops[0], ASTNode)
    assert ops[0].filter_dict == {"label__Person": True, "label__Admin": True}


def test_lower_match_query_folds_relationship_type_where_into_match_filter() -> None:
    parsed = _parse_query("MATCH (n {name: 'A'})-[r]->(x) WHERE type(r) = 'KNOWS' RETURN x")

    lowered = lower_match_query(parsed)

    assert lowered.row_where is None
    assert isinstance(lowered.query[1], ASTEdgeForward)
    assert lowered.query[1].edge_match == {"type": "KNOWS"}


def test_string_cypher_executes_relationship_type_where_with_bound_relationship_alias() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "name": ["A", "B", "C"]})
    edges = pd.DataFrame(
        {
            "s": ["a", "a"],
            "d": ["b", "c"],
            "type": ["KNOWS", "LIKES"],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n {name: 'A'})-[r]->(x) WHERE type(r) = 'KNOWS' RETURN x.id AS id"
    )

    assert result._nodes.to_dict(orient="records") == [{"id": "b"}]


def test_string_cypher_executes_multi_label_node_patterns() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "label__A": [True, True, False],
            "label__B": [True, True, True],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql("MATCH (a:A:B) RETURN a.id AS id ORDER BY id")

    assert result._nodes.to_dict(orient="records") == [{"id": "a"}, {"id": "b"}]


def test_lower_match_query_folds_literal_where_into_filter_dicts() -> None:
    parsed = _parse_query("MATCH (p)-[r]->(q) WHERE p.name = 'Alice' AND q.name = 'Bob' RETURN p, q")

    lowered = lower_match_query(parsed)

    assert lowered.where == []
    assert isinstance(lowered.query[0], ASTNode)
    assert isinstance(lowered.query[2], ASTNode)
    assert lowered.query[0].filter_dict == {"name": "Alice"}
    assert lowered.query[2].filter_dict == {"name": "Bob"}


def test_lower_match_query_emits_same_path_where_for_alias_comparisons() -> None:
    parsed = _parse_query("MATCH (p)-[r]->(q) WHERE p.team = q.team RETURN p, q")

    lowered = lower_match_query(parsed)

    assert len(lowered.where) == 1
    clause = lowered.where[0]
    assert clause.left.alias == "p"
    assert clause.left.column == "team"
    assert clause.op == "=="
    assert clause.right.alias == "q"
    assert clause.right.column == "team"


def test_lower_match_query_executes_same_path_where() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "team": ["x", "x", "y"],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "a"],
            "d": ["b", "c"],
        }
    )

    parsed = _parse_query("MATCH (p)-[r]->(q) WHERE p.team = q.team RETURN p, q")
    lowered = lower_match_query(parsed)
    result = _mk_graph(nodes, edges).gfql(lowered.query, where=lowered.where)

    assert sorted(result._nodes["id"].tolist()) == ["a", "b"]
    assert result._edges[["s", "d"]].to_dict(orient="records") == [{"s": "a", "d": "b"}]


def test_lower_match_query_executes_null_predicates() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "deleted": [None, "yes", None],
            "name": ["Alice", "Bob", "Carol"],
        }
    )
    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})

    parsed = _parse_query("MATCH (p)-[r]->(q) WHERE p.deleted IS NULL AND q.name IS NOT NULL RETURN p, q")
    lowered = lower_match_query(parsed)
    result = _mk_graph(nodes, edges).gfql(lowered.query, where=lowered.where)

    assert sorted(result._nodes["id"].tolist()) == ["a", "b"]
    assert result._edges[["s", "d"]].to_dict(orient="records") == [{"s": "a", "d": "b"}]


def test_lower_match_query_executes_bracketless_relationship_and_label_where() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["root", "t1", "t2"],
            "type": ["Root", "TextNode", "Other"],
            "name": ["x", None, None],
            "score": [None, 7, 2],
        }
    )
    edges = pd.DataFrame({"s": ["root", "root"], "d": ["t1", "t2"]})

    chain = cypher_to_gfql(
        "MATCH (:Root {name: 'x'})-->(i) WHERE i.score > 5 AND i:TextNode RETURN i"
    )
    result = _mk_graph(nodes, edges).gfql(chain)

    assert result._nodes[["id", "type", "score"]].to_dict(orient="records") == [
        {"id": "t1", "type": "TextNode", "score": 7}
    ]


def test_lower_match_query_executes_bracketless_relationship_with_labeled_alias_projection() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "type": ["Start", "Foo", "Bar"],
        }
    )
    edges = pd.DataFrame({"s": ["a", "a"], "d": ["b", "c"]})

    chain = cypher_to_gfql("MATCH (a)-->(b:Foo) RETURN b")
    result = _mk_graph(nodes, edges).gfql(chain)

    assert result._nodes[["id", "type"]].to_dict(orient="records") == [
        {"id": "b", "type": "Foo"}
    ]


def test_cypher_to_gfql_executes_relationship_type_alternation() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame(
        {
            "s": ["a", "a", "a"],
            "d": ["b", "c", "b"],
            "type": ["KNOWS", "HATES", "LIKES"],
        }
    )

    chain = cypher_to_gfql("MATCH (n)-[r:KNOWS|HATES]->(x) RETURN r")
    result = _mk_graph(nodes, edges).gfql(chain)

    assert result._nodes[["s", "d", "type"]].to_dict(orient="records") == [
        {"s": "a", "d": "b", "type": "KNOWS"},
        {"s": "a", "d": "c", "type": "HATES"},
    ]


def test_lower_cypher_query_builds_row_pipeline_chain() -> None:
    parsed = _parse_query(
        "MATCH (p:Person) RETURN DISTINCT p.name AS person_name ORDER BY person_name DESC SKIP 1 LIMIT 2"
    )

    chain = lower_cypher_query(parsed)

    assert [type(op).__name__ for op in chain.chain[:2]] == ["ASTNode", "ASTCall"]
    assert isinstance(chain.chain[1], ASTCall)
    assert chain.chain[1].function == "rows"
    assert chain.chain[1].params == {"table": "nodes", "source": "p"}
    assert isinstance(chain.chain[2], ASTCall)
    assert chain.chain[2].function == "select"
    assert chain.chain[2].params["items"] == [("person_name", "p.name")]
    assert isinstance(chain.chain[3], ASTCall)
    assert chain.chain[3].function == "distinct"
    assert isinstance(chain.chain[4], ASTCall)
    assert chain.chain[4].function == "order_by"
    assert chain.chain[4].params["keys"] == [("person_name", "desc")]
    assert isinstance(chain.chain[5], ASTCall)
    assert chain.chain[5].function == "skip"
    assert chain.chain[5].params["value"] == 1
    assert isinstance(chain.chain[6], ASTCall)
    assert chain.chain[6].function == "limit"
    assert chain.chain[6].params["value"] == 2


def test_cypher_to_gfql_executes_property_projection_pipeline() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "type": ["Person", "Person", "Person"],
            "name": ["Alice", "Bob", "Alice"],
            "score": [9, 4, 7],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    chain = cypher_to_gfql(
        "MATCH (p:Person) RETURN DISTINCT p.name AS person_name ORDER BY p.name DESC SKIP 1 LIMIT $top_n",
        params={"top_n": 1},
    )
    result = _mk_graph(nodes, edges).gfql(chain)

    assert result._nodes.to_dict(orient="records") == [{"person_name": "Alice"}]


def test_cypher_to_gfql_executes_whole_row_alias_pipeline() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "type": ["Person", "Person", "Company"],
            "score": [2, 9, 7],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    chain = cypher_to_gfql("MATCH (p:Person) RETURN p ORDER BY p.score DESC LIMIT 1")
    result = _mk_graph(nodes, edges).gfql(chain)

    assert result._nodes[["id", "type", "score"]].to_dict(orient="records") == [
        {"id": "b", "type": "Person", "score": 9}
    ]


def test_string_cypher_formats_single_node_entity_projection() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a"],
            "type": ["Person"],
            "name": ["Alice"],
            "score": [2],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql("MATCH (p) RETURN p")

    assert entity_text_records(result, {"p": "nodes"}) == [
        {"p": "(:Person {name: 'Alice', score: 2})"}
    ]
    entity_meta = getattr(result, "_cypher_entity_projection_meta")
    assert entity_meta["p"]["table"] == "nodes"
    assert entity_meta["p"]["alias"] == "p"
    assert entity_meta["p"]["id_column"] == "id"
    assert entity_meta["p"]["ids"].tolist() == ["a"]


def test_string_cypher_formats_single_edge_entity_projection() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame(
        {
            "s": ["a"],
            "d": ["b"],
            "edge_id": ["e1"],
            "type": ["KNOWS"],
            "weight": [5],
        }
    )

    result = _mk_graph(nodes, edges).gfql("MATCH ()-[r]->() RETURN r")

    assert entity_text_records(result, {"r": "edges"}) == [
        {"r": "[:KNOWS {weight: 5}]"}
    ]


def test_standalone_graph_constructor_returns_subgraph() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "z"], "score": [10, 5, 1, 0]})
    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "weight": [7, 9]})
    result = _mk_graph(nodes, edges).gfql(
        "GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'a' }"
    )
    assert set(_to_pandas_df(result._nodes)["id"].tolist()) == {"a", "b"}
    assert _to_pandas_df(result._edges)[["s", "d", "weight"]].to_dict(orient="records") == [
        {"s": "a", "d": "b", "weight": 7}
    ]


def test_graph_binding_with_use_returns_rows() -> None:
    result = _mk_simple_path_graph().gfql(
        "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'a' } "
        "USE g1 MATCH (x) RETURN x.id AS id ORDER BY id"
    )
    assert sorted(_to_pandas_df(result._nodes)["id"].tolist()) == ["a", "b"]


def test_graph_constructor_empty_match_returns_empty_graph() -> None:
    result = _mk_simple_path_graph().gfql(
        "GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'nonexistent' }"
    )
    assert len(result._nodes) == 0
    assert len(result._edges) == 0


def test_graph_constructor_rejects_node_residual_on_multi_alias_pattern() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "score": [0.3, 0.1, None, 0.5],
        "name": ["alpha", "beta", "gamma", "delta"],
    })
    edges = pd.DataFrame({
        "s": ["a", "b", "c", "d"],
        "d": ["b", "c", "d", "a"],
        "weight": [7, 9, 11, 13],
    })

    with pytest.raises(GFQLValidationError, match="single-node GRAPH MATCH masks"):
        _mk_graph(nodes, edges).gfql(
            "GRAPH { MATCH (a)-[r]->(b) WHERE (a.score > 0.25 OR a.score IS NULL) }"
        )


def test_graph_constructor_single_node_residual_alias_marker_overrides_property_column_collision() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "a": ["shadow-a", "shadow-b", "shadow-c", "shadow-d"],
        "score": [0.3, 0.1, None, 0.5],
    })
    edges = pd.DataFrame({
        "s": ["a", "b", "c", "d"],
        "d": ["b", "c", "d", "a"],
        "weight": [7, 9, 11, 13],
    })

    result = _mk_graph(nodes, edges).gfql(
        "GRAPH { MATCH (a) WHERE (a.score > 0.25 OR a.score IS NULL) }"
    )

    assert set(_to_pandas_df(result._nodes)["id"].tolist()) == {"a", "c", "d"}


def test_graph_constructor_applies_single_node_residual_row_predicate_as_graph_mask_cudf() -> None:
    _require_cudf_runtime()
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "score": [0.3, 0.1, None, 0.5],
    })
    edges = pd.DataFrame({
        "s": ["a", "b", "c", "d"],
        "d": ["b", "c", "d", "a"],
        "weight": [7, 9, 11, 13],
    })

    result = _mk_cudf_graph(nodes, edges).gfql(
        "GRAPH { MATCH (a) WHERE (a.score > 0.25 OR a.score IS NULL) }",
        engine="cudf",
    )

    assert set(_to_pandas_df(result._nodes)["id"].tolist()) == {"a", "c", "d"}


def test_graph_constructor_residual_row_predicate_declines_on_polars() -> None:
    pl = pytest.importorskip("polars")
    nodes = pl.DataFrame({
        "id": ["a", "b", "c", "d"],
        "score": [0.3, 0.1, None, 0.5],
    })
    edges = pl.DataFrame({
        "s": ["a", "b", "c", "d"],
        "d": ["b", "c", "d", "a"],
        "weight": [7, 9, 11, 13],
    })

    with pytest.raises(GFQLValidationError, match="not yet supported on polars"):
        _CypherTestGraph().nodes(nodes, "id").edges(edges, "s", "d").gfql(
            "GRAPH { MATCH (a) WHERE (a.score > 0.25 OR a.score IS NULL) }",
            engine="polars",
        )


def test_graph_constructor_applies_search_any_residual_as_graph_mask() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "name": ["alpha", "beta", "gamma", "delta"],
    })
    edges = pd.DataFrame({
        "s": ["a", "b", "c", "d"],
        "d": ["b", "c", "d", "a"],
        "weight": [7, 9, 11, 13],
    })

    result = _mk_graph(nodes, edges).gfql(
        "GRAPH { MATCH (a) WHERE searchAny(a, 'l') }"
    )

    assert set(_to_pandas_df(result._nodes)["id"].tolist()) == {"a", "d"}


def test_graph_constructor_applies_edge_residual_row_predicate_as_graph_mask() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    edges = pd.DataFrame({
        "s": ["a", "b", "c", "d"],
        "d": ["b", "c", "d", "a"],
        "weight": [7, 9, 11, 13],
    })

    result = _mk_graph(nodes, edges).gfql(
        "GRAPH { MATCH (a)-[r]->(b) WHERE (r.weight > 8 OR r.weight IS NULL) }"
    )

    assert set(_to_pandas_df(result._nodes)["id"].tolist()) == {"a", "b", "c", "d"}
    assert _to_pandas_df(result._edges)[["s", "d", "weight"]].to_dict(orient="records") == [
        {"s": "b", "d": "c", "weight": 9},
        {"s": "c", "d": "d", "weight": 11},
        {"s": "d", "d": "a", "weight": 13},
    ]


def test_graph_constructor_edge_residual_alias_marker_overrides_property_column_collision() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    edges = pd.DataFrame({
        "s": ["a", "b", "c", "d"],
        "d": ["b", "c", "d", "a"],
        "r": ["shadow-r0", "shadow-r1", "shadow-r2", "shadow-r3"],
        "weight": [7, 9, 11, 13],
    })

    result = _mk_graph(nodes, edges).gfql(
        "GRAPH { MATCH (a)-[r]->(b) WHERE (r.weight > 8 OR r.weight IS NULL) }"
    )

    assert _to_pandas_df(result._edges)[["s", "d", "weight"]].to_dict(orient="records") == [
        {"s": "b", "d": "c", "weight": 9},
        {"s": "c", "d": "d", "weight": 11},
        {"s": "d", "d": "a", "weight": 13},
    ]


def test_graph_binding_applies_residual_mask_before_use() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "score": [0.3, 0.1, None, 0.5],
    })
    edges = pd.DataFrame({
        "s": ["a", "b", "c", "d"],
        "d": ["b", "c", "d", "a"],
    })

    result = _mk_graph(nodes, edges).gfql(
        "GRAPH g = GRAPH { "
        "MATCH (a) WHERE (a.score > 0.25 OR a.score IS NULL) "
        "} USE g MATCH (n) RETURN n.id AS id ORDER BY id"
    )

    assert _to_pandas_df(result._nodes)["id"].tolist() == ["a", "c", "d"]


def test_graph_constructor_rejects_pattern_residual_predicates() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["R", "S"]})

    with pytest.raises(GFQLValidationError, match="pattern-predicate residuals"):
        _mk_graph(nodes, edges).gfql(
            "GRAPH { MATCH (a)-[r]->(b) WHERE (a)-[:R]->() }"
        )


def test_standalone_graph_constructor_preserves_columns() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "score": [10, 5, 1]})
    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "weight": [7, 9]})
    result = _mk_graph(nodes, edges).gfql(
        "GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'a' }"
    )
    assert "score" in _to_pandas_df(result._nodes).columns
    assert "weight" in _to_pandas_df(result._edges).columns


def test_graph_constructor_cudf_support() -> None:
    result = _mk_path_with_isolate_graph_cudf().gfql(
        "GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'b' }",
        engine="cudf",
    )
    assert set(_to_pandas_df(result._nodes)["id"].tolist()) == {"b", "c"}
    assert _to_pandas_df(result._edges)[["s", "d"]].to_dict(orient="records") == [
        {"s": "b", "d": "c"}
    ]


def test_graph_constructor_with_call_write() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "score": [10, 5, 1]})
    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "weight": [7, 9]})
    result = _mk_graph(nodes, edges).gfql("GRAPH { CALL graphistry.degree.write() }")
    assert "degree" in _to_pandas_df(result._nodes).columns
    assert not result._edges.empty


def test_graph_constructor_call_with_use_pipeline() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "z"], "score": [10, 5, 1, 0]})
    edges = pd.DataFrame({"s": ["a", "b", "b"], "d": ["b", "c", "a"]})
    g = _mk_graph(nodes, edges)
    result = g.gfql(
        "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) WHERE a.score > 3 } "
        "GRAPH g2 = GRAPH { USE g1 CALL graphistry.degree.write() } "
        "USE g2 "
        "MATCH (n) RETURN n.id AS id, n.degree AS degree "
        "ORDER BY degree DESC, id ASC"
    )
    ids = _to_pandas_df(result._nodes)["id"].tolist()
    assert "a" in ids
    assert "b" in ids
    assert "degree" in _to_pandas_df(result._nodes).columns


def test_graph_constructor_rejects_non_write_call() -> None:
    with pytest.raises(GFQLValidationError):
        _mk_simple_path_graph().gfql("GRAPH { CALL graphistry.degree() }")


def test_cypher_to_gfql_supports_standalone_graph_constructor() -> None:
    chain = cypher_to_gfql("GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'a' }")
    result = _mk_simple_path_graph().gfql(chain)
    assert set(result._nodes["id"].tolist()) == {"a", "b"}
    assert result._edges[["s", "d"]].to_dict(orient="records") == [{"s": "a", "d": "b"}]


def test_string_cypher_formats_filtered_edge_entity_projection_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "labels": [[], [], []],
                "name": ["A", "B", "C"],
            }
        )
    )
    edges = cudf.from_pandas(
        pd.DataFrame(
            {
                "s": ["a", "a", "a"],
                "d": ["b", "c", "c"],
                "edge_id": ["rel_1", "rel_2", "rel_3"],
                "type": ["KNOWS", "HATES", "WONDERS"],
                "undirected": [False, False, False],
            }
        )
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n)-[r]->(x) "
        "WHERE type(r) = 'KNOWS' OR type(r) = 'HATES' "
        "RETURN r",
        engine="cudf",
    )

    assert sorted(
        entity_text_records(result, {"r": "edges"}), key=lambda row: row["r"]
    ) == [
        {"r": "[:HATES]"},
        {"r": "[:KNOWS]"},
    ]


def test_string_cypher_formats_optional_match_projection_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["s", "a", "b", "c"],
                "labels": [["Single"], ["A"], ["B"], ["C"]],
                "label__Single": [True, False, False, False],
                "label__A": [False, True, False, False],
                "label__B": [False, False, True, False],
                "label__C": [False, False, False, True],
                "num": pd.Series([pd.NA, 42, 46, pd.NA], dtype="Int64"),
            }
        )
    )
    edges = cudf.from_pandas(
        pd.DataFrame(
            {
                "s": ["s", "s", "a", "b"],
                "d": ["a", "b", "c", "b"],
                "edge_id": ["rel_1", "rel_2", "rel_3", "rel_4"],
                "type": ["REL", "REL", "REL", "LOOP"],
                "undirected": [False, False, False, False],
            }
        )
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n:Single) "
        "OPTIONAL MATCH (n)-[r]-(m) "
        "WHERE m.num = 42 "
        "RETURN m",
        engine="cudf",
    )

    assert entity_text_records(result, {"m": "nodes"}) == [
        {"m": "(:A {num: 42})"}
    ]


def test_string_cypher_formats_small_float_node_entity_projection_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["b"],
                "labels": [["B"]],
                "label__B": [True],
                "num": [30.94857],
                "num2": [0.00002],
            }
        )
    )
    edges = cudf.from_pandas(pd.DataFrame({"s": [], "d": []}))

    result = _mk_graph(nodes, edges).gfql("MATCH (a) RETURN a", engine="cudf")

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:B {num: 30.94857, num2: 0.00002})"}
    ]


def test_string_cypher_formats_single_node_entity_projection_with_alias() -> None:
    nodes = pd.DataFrame({"id": ["a"], "type": ["A"]})
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql("MATCH (a) RETURN a AS ColumnName")

    assert entity_text_records(result, {"ColumnName": "nodes"}) == [{"ColumnName": "(:A)"}]
    entity_meta = getattr(result, "_cypher_entity_projection_meta")
    assert entity_meta["ColumnName"]["ids"].tolist() == ["a"]


def test_string_cypher_supports_is1_seed_city_projection_shape() -> None:
    nodes = pd.DataFrame(
        [
            {"id": "p1", "labels": ["Person"], "label__Person": True, "firstName": "A"},
            {"id": "c1", "labels": ["Place"], "label__Place": True, "name": "City"},
        ]
    )
    edges = pd.DataFrame([{"s": "p1", "d": "c1", "type": "IS_LOCATED_IN"}])

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (person:Person {id: 'p1'}) "
        "MATCH (person)-[:IS_LOCATED_IN]->(city:Place) "
        "RETURN person.id AS personId, city.id AS cityId, person.firstName AS firstName"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"personId": "p1", "cityId": "c1", "firstName": "A"}
    ]


def test_issue_1411_connected_join_property_projection_shape() -> None:
    nodes = pd.DataFrame(
        [
            {
                "id": "p1",
                "labels": ["Person"],
                "label__Person": True,
                "firstName": "Seed",
                "name": None,
            },
            {
                "id": "p2",
                "labels": ["Person"],
                "label__Person": True,
                "firstName": "Friend",
                "name": None,
            },
            {
                "id": "c1",
                "labels": ["Place"],
                "label__Place": True,
                "firstName": None,
                "name": "City",
            },
        ]
    )
    edges = pd.DataFrame(
        [
            {"s": "p1", "d": "c1", "type": "IS_LOCATED_IN"},
            {"s": "p2", "d": "c1", "type": "IS_LOCATED_IN"},
        ]
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH "
        "(person:Person {id: 'p1'})-[:IS_LOCATED_IN]->(city:Place), "
        "(friend:Person)-[:IS_LOCATED_IN]->(city) "
        "RETURN friend.id AS friendId, friend.firstName AS friendFirstName, city.name AS cityName "
        "ORDER BY friendId"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"friendId": "p1", "friendFirstName": "Seed", "cityName": "City"},
        {"friendId": "p2", "friendFirstName": "Friend", "cityName": "City"},
    ]


def test_issue_1411_connected_join_whole_row_projection_shape() -> None:
    nodes = pd.DataFrame(
        [
            {
                "id": "p1",
                "labels": ["Person"],
                "label__Person": True,
                "firstName": "Seed",
                "name": None,
            },
            {
                "id": "p2",
                "labels": ["Person"],
                "label__Person": True,
                "firstName": "Friend",
                "name": None,
            },
            {
                "id": "c1",
                "labels": ["Place"],
                "label__Place": True,
                "firstName": None,
                "name": "City",
            },
        ]
    )
    edges = pd.DataFrame(
        [
            {"s": "p1", "d": "c1", "type": "IS_LOCATED_IN"},
            {"s": "p2", "d": "c1", "type": "IS_LOCATED_IN"},
        ]
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH "
        "(person:Person {id: 'p1'})-[:IS_LOCATED_IN]->(city:Place), "
        "(friend:Person {id: 'p2'})-[:IS_LOCATED_IN]->(city) "
        "RETURN city"
    )

    assert entity_text_records(result, {"city": "nodes"}) == [
        {"city": "(:Place {name: 'City'})"}
    ]
    entity_meta = getattr(result, "_cypher_entity_projection_meta")
    assert entity_meta["city"]["table"] == "nodes"
    assert entity_meta["city"]["alias"] == "city"
    assert entity_meta["city"]["id_column"] == "id"
    assert entity_meta["city"]["ids"].tolist() == ["c1"]


def test_string_cypher_supports_is3_seed_expand_projection_shape() -> None:
    nodes = pd.DataFrame(
        [
            {"id": "p1", "labels": ["Person"], "label__Person": True, "firstName": "Seed"},
            {"id": "p2", "labels": ["Person"], "label__Person": True, "firstName": "Friend"},
        ]
    )
    edges = pd.DataFrame([{"s": "p2", "d": "p1", "type": "KNOWS", "creationDate": 123}])

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (person:Person {id: 'p1'}) "
        "MATCH (person)-[k:KNOWS]-(friend:Person) "
        "RETURN friend.id AS friendId, friend.firstName AS firstName, k.creationDate AS friendshipCreationDate"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"friendId": "p2", "firstName": "Friend", "friendshipCreationDate": 123}
    ]


def test_string_cypher_supports_is6_open_range_continuation_projection_shape() -> None:
    nodes = pd.DataFrame(
        [
            {"id": "c1", "labels": ["Comment"], "label__Comment": True},
            {"id": "m1", "labels": ["Message"], "label__Message": True},
            {"id": "p1", "labels": ["Post"], "label__Post": True},
            {"id": "f1", "labels": ["Forum"], "label__Forum": True, "title": "Forum"},
            {
                "id": "u1",
                "labels": ["Person"],
                "label__Person": True,
                "firstName": "Mod",
                "lastName": "Erator",
            },
        ]
    )
    edges = pd.DataFrame(
        [
            {"s": "c1", "d": "m1", "type": "REPLY_OF"},
            {"s": "m1", "d": "p1", "type": "REPLY_OF"},
            {"s": "f1", "d": "p1", "type": "CONTAINER_OF"},
            {"s": "f1", "d": "u1", "type": "HAS_MODERATOR"},
        ]
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (message:Comment {id: 'c1'})-[:REPLY_OF*0..]->(post:Post)"
        "<-[:CONTAINER_OF]-(forum:Forum)-[:HAS_MODERATOR]->(moderator:Person) "
        "RETURN forum.id AS forumId, moderator.id AS moderatorId"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"forumId": "f1", "moderatorId": "u1"}
    ]


def test_compile_cypher_records_mixed_whole_row_projection_plan() -> None:
    compiled = _compile_query("MATCH (p:Person) RETURN p AS person, p.name AS person_name")

    assert compiled.result_projection is not None
    assert compiled.result_projection.alias == "p"
    assert compiled.result_projection.table == "nodes"
    assert [column.output_name for column in compiled.result_projection.columns] == [
        "person",
        "person_name",
    ]
    assert [column.kind for column in compiled.result_projection.columns] == [
        "whole_row",
        "property",
    ]


def test_string_cypher_formats_mixed_node_entity_projection() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "type": ["Person", "Person"],
            "name": ["Alice", "Bob"],
            "score": [2, 9],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (p:Person) RETURN p AS person, p.name AS person_name ORDER BY person_name DESC LIMIT 1"
    )

    assert entity_text_records(result, {"person": "nodes"}) == [
        {"person": "(:Person {name: 'Bob', score: 9})", "person_name": "Bob"}
    ]


def test_string_cypher_formats_mixed_node_entity_and_null_predicate_projection() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "label__X": [True, True],
            "prop": [42, pd.NA],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql("MATCH (n:X) RETURN n, n.prop IS NULL AS b")

    assert entity_text_records(result, {"n": "nodes"}) == [
        {"n": "(:X {prop: 42})", "b": False},
        {"n": "(:X)", "b": True},
    ]


def test_string_cypher_formats_mixed_node_entity_and_not_null_predicate_projection() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "label__X": [True, True],
            "prop": [42, pd.NA],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql("MATCH (n:X) RETURN n, n.prop IS NOT NULL AS b")

    assert entity_text_records(result, {"n": "nodes"}) == [
        {"n": "(:X {prop: 42})", "b": True},
        {"n": "(:X)", "b": False},
    ]


def test_string_cypher_formats_mixed_edge_entity_projection() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame(
        {
            "s": ["a"],
            "d": ["b"],
            "edge_id": ["e1"],
            "type": ["KNOWS"],
            "weight": [5],
        }
    )

    result = _mk_graph(nodes, edges).gfql("MATCH ()-[r]->() RETURN r AS rel, type(r) AS rel_type")

    assert entity_text_records(result, {"rel": "edges"}) == [
        {"rel": "[:KNOWS {weight: 5}]", "rel_type": "KNOWS"}
    ]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "MATCH (a) WHERE a.name STARTS WITH 'ABC' RETURN a.name AS name ORDER BY name",
            [{"name": "ABCDEF"}],
        ),
        (
            "MATCH (a) WHERE a.name ENDS WITH 'DEF' RETURN a.name AS name ORDER BY name",
            [{"name": "ABCDEF"}],
        ),
        (
            "MATCH (a) WHERE a.name CONTAINS 'CD' RETURN a.name AS name ORDER BY name",
            [{"name": "ABCDEF"}],
        ),
        (
            "MATCH (a) WHERE a.name STARTS WITH null RETURN a.name AS name ORDER BY name",
            [],
        ),
        (
            "MATCH (a) WHERE a.name ENDS WITH null RETURN a.name AS name ORDER BY name",
            [],
        ),
        (
            "MATCH (a) WHERE a.name CONTAINS null RETURN a.name AS name ORDER BY name",
            [],
        ),
        (
            "MATCH (a) WHERE a.name STARTS WITH 'A' AND a.name CONTAINS 'C' AND a.name ENDS WITH 'EF' "
            "RETURN a.name AS name ORDER BY name",
            [{"name": "ABCDEF"}],
        ),
        # openCypher/neo4j `=~` — Java-regex, FULL/anchored match (not partial), inline flags.
        (
            "MATCH (a) WHERE a.name =~ 'AB.*' RETURN a.name AS name ORDER BY name",
            [{"name": "AB"}, {"name": "ABCDEF"}],
        ),
        (  # full-match, not partial: 'AB' matches only 'AB', NOT 'ABCDEF'
            "MATCH (a) WHERE a.name =~ 'AB' RETURN a.name AS name ORDER BY name",
            [{"name": "AB"}],
        ),
        (  # inline case-insensitive flag
            "MATCH (a) WHERE a.name =~ '(?i)abcdef' RETURN a.name AS name ORDER BY name",
            [{"name": "ABCDEF"}, {"name": "abcdef"}],
        ),
        (  # null rhs → never matches (mirrors CONTAINS/STARTS WITH null)
            "MATCH (a) WHERE a.name =~ null RETURN a.name AS name ORDER BY name",
            [],
        ),
    ],
)
def test_string_cypher_executes_match_where_string_predicates(
    query: str, expected: list[dict[str, object]]
) -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "name": ["ABCDEF", "AB", "abcdef", "ab", "", None],
        }
    )
    result = _mk_graph(nodes, pd.DataFrame({"s": [], "d": []})).gfql(query)

    assert result._nodes.where(~result._nodes.isna(), None).to_dict(orient="records") == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # `=~` composes through OR / NOT / RETURN-expression (shared expr engine path),
        # not just the simple WHERE-predicate path.
        (
            "MATCH (a) WHERE a.name =~ 'al.*' OR a.name = 'bob' RETURN a.name AS name ORDER BY name",
            [{"name": "alba"}, {"name": "alice"}, {"name": "bob"}],
        ),
        (
            "MATCH (a) WHERE NOT (a.name =~ 'al.*') RETURN a.name AS name ORDER BY name",
            [{"name": "Alice"}, {"name": "bob"}],
        ),
        (  # regex leaf under both AND and NOT, with inline (?i)
            "MATCH (a) WHERE a.name =~ 'a.*' AND NOT a.name =~ '(?i)alice' RETURN a.name AS name ORDER BY name",
            [{"name": "alba"}],
        ),
    ],
)
def test_regex_operator_composes_or_not(query: str, expected: list[dict[str, object]]) -> None:
    nodes = pd.DataFrame({"id": ["p", "q", "r", "s"], "name": ["alice", "Alice", "bob", "alba"]})
    result = _mk_graph(nodes, pd.DataFrame({"s": [], "d": []})).gfql(query)
    assert result._nodes.where(~result._nodes.isna(), None).to_dict(orient="records") == expected


def test_regex_cudf_inline_flag_parity() -> None:
    """cuDF/libcudf rejects inline regex flags (``(?i)``/``(?m)``/``(?s)``) at ANY position
    ("invalid regex pattern"), so a bare ``=~ '(?i)…'`` used to CRASH on ``engine='cudf'``.
    The Match/Fullmatch cuDF path now translates a leading ``(?i)`` to the lowercase
    case-folding workaround (parity with pandas). Regression for viz-filter #1673."""
    pytest.importorskip("cudf")
    nodes = pd.DataFrame({"id": [0, 1, 2, 3], "name": ["bob", "ALICE", "abc", "a.c"]})
    g = _mk_graph(nodes, pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3]}))
    q = "MATCH (n) WHERE n.name =~ '(?i)a.c' RETURN n.id AS id ORDER BY id"
    oracle = g.gfql(q)._nodes["id"].tolist()  # pandas
    got = g.gfql(q, engine="cudf")._nodes.to_pandas()["id"].tolist()
    assert got == oracle == [2, 3]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        # openCypher/neo4j numeric functions (standard). Column form -> floats.
        ("floor(n.x)", [2.0, -3.0, 4.0]),
        ("ceil(n.x)", [3.0, -2.0, 4.0]),
        ("ceiling(n.x)", [3.0, -2.0, 4.0]),
        ("round(n.x)", [2.0, -3.0, 4.0]),
        ("round(n.x, 1)", [2.3, -2.7, 4.0]),
        ("sqrt(abs(n.x))", [pytest.approx(1.5165750888), pytest.approx(1.6431676725), 2.0]),
        ("sign(n.x)", [1, -1, 1]),              # neo4j sign() -> Integer (both engines)
    ],
)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_numeric_functions(engine: str, expr: str, expected: list[object]) -> None:
    if engine == "polars":
        pytest.importorskip("polars")
    nodes = pd.DataFrame({"id": [0, 1, 2], "x": [2.3, -2.7, 4.0]})
    g = _mk_graph(nodes, pd.DataFrame({"s": [], "d": []}))
    q = f"MATCH (n) RETURN {expr} AS v, n.id AS id ORDER BY id"
    col = g.gfql(q, engine=engine)._nodes["v"]
    got = col.to_list() if hasattr(col, "to_list") else col.tolist()
    assert got == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        # scalar (constant-folded) forms exercise the scalar code paths
        ("floor(2.7)", 2.0),
        ("ceil(2.1)", 3.0),
        ("ceiling(-2.1)", -2.0),
        ("round(2.5)", 3.0),        # neo4j: precision-0 ties round toward +inf
        ("round(-1.5)", -1.0),      # neo4j manual Example 8
        ("round(2.567, 2)", 2.57),
        ("round(-1.55, 1)", -1.6),  # neo4j manual Example 11: p>0 ties away from zero
        ("sign(-9)", -1),
        ("sqrt(9.0)", 3.0),
        ("toLower('ABC')", "abc"),
        ("toUpper('abc')", "ABC"),
        ("lower('ABC')", "abc"),    # GQL-conformance aliases (ISO GQL §20.24; neo4j accepts both)
        ("upper('abc')", "ABC"),
    ],
)
def test_numeric_functions_scalar(expr: str, expected: object) -> None:
    nodes = pd.DataFrame({"id": [0], "x": [1.0]})
    g = _mk_graph(nodes, pd.DataFrame({"s": [], "d": []}))
    assert g.gfql(f"MATCH (n) RETURN {expr} AS v, n.id AS id")._nodes["v"].tolist() == [expected]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_round_neo4j_tie_breaking(engine: str) -> None:
    """Standards-vetted (#1673): neo4j round() ties — precision 0 rounds ties toward
    +inf (round(-1.5) = -1.0, neo4j manual Ex. 8/10), precision > 0 rounds ties away
    from zero (HALF_UP: round(-1.55, 1) = -1.6, Ex. 11). The numpy/polars
    half-to-even defaults (round(2.5) -> 2.0) are wrong answers vs this spec."""
    if engine == "polars":
        pytest.importorskip("polars")

    def vals(nodes: pd.DataFrame, expr: str) -> List[Any]:
        g = _mk_graph(nodes, pd.DataFrame({"s": [], "d": []}))
        q = f"MATCH (n) RETURN {expr} AS v, n.id AS id ORDER BY id"
        col = g.gfql(q, engine=engine)._nodes["v"]
        return col.to_list() if hasattr(col, "to_list") else col.tolist()

    ties = pd.DataFrame({"id": [0, 1, 2, 3], "x": [2.5, -1.5, 0.5, -2.5]})
    assert vals(ties, "round(n.x)") == [3.0, -1.0, 1.0, -2.0]      # ties toward +inf
    assert vals(ties, "round(n.x, 0)") == [3.0, -1.0, 1.0, -2.0]   # p=0 aligns with 1-arg
    prec = pd.DataFrame({"id": [0, 1], "x": [1.25, -1.55]})
    assert vals(prec, "round(n.x, 1)") == [pytest.approx(1.3), pytest.approx(-1.6)]  # away from zero

    # 1-ulp-below-a-tie (JDK-6430675 class): a floor(x+0.5) kernel wrongly rounds UP;
    # the correct answer (Java Math.round post-fix, BigDecimal, polars native) is down.
    ulp = pd.DataFrame({"id": [0, 1], "x": [0.49999999999999994, 0.049999999999999996]})
    assert vals(ulp, "round(n.x)") == [0.0, 0.0]
    assert vals(ulp, "round(n.x, 1)") == [pytest.approx(0.5), pytest.approx(0.0)]
    # infinity passes through (rounding is the identity; no overflow to inf/crash)
    inf = pd.DataFrame({"id": [0, 1], "x": [float("inf"), 1e300]})
    assert vals(inf, "round(n.x)") == [float("inf"), 1e300]
    assert vals(inf, "round(n.x, 2)") == [float("inf"), 1e300]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_round_negative_precision_declines(engine: str) -> None:
    """neo4j raises on negative round() precision; we decline honestly (error, never a
    silent value — and never a raw polars OverflowError crash)."""
    if engine == "polars":
        pytest.importorskip("polars")
    g = _mk_graph(pd.DataFrame({"id": [0], "x": [25.0]}), pd.DataFrame({"s": [], "d": []}))
    with pytest.raises(Exception) as exc_info:
        g.gfql("MATCH (n) RETURN round(n.x, -1) AS v, n.id AS id", engine=engine)
    assert "OverflowError" not in type(exc_info.value).__name__


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("MATCH (n) WHERE lower(n.name) = 'bob' RETURN n.id AS id ORDER BY id", [0]),
        ("MATCH (n) WHERE upper(n.name) = 'BOB' RETURN n.id AS id ORDER BY id", [0]),
    ],
)
def test_lower_upper_gql_aliases(engine: str, query: str, expected: list[int]) -> None:
    if engine == "polars":
        pytest.importorskip("polars")
    nodes = pd.DataFrame({"id": [0, 1, 2], "name": ["BOB", "Alice", "carol"]})
    col = _mk_graph(nodes, pd.DataFrame({"s": [], "d": []})).gfql(query, engine=engine)._nodes["id"]
    got = col.to_list() if hasattr(col, "to_list") else col.tolist()
    assert got == expected


@pytest.mark.parametrize("expr", ["floor(null)", "ceil(null)", "round(null)", "toLower(null)", "toUpper(null)"])
def test_numeric_string_fns_null_scalar(expr: str) -> None:
    """null literal -> null (exercises the scalar null-guard branches)."""
    g = _mk_graph(pd.DataFrame({"id": [0]}), pd.DataFrame({"s": [], "d": []}))
    v = g.gfql(f"MATCH (n) RETURN {expr} AS v, n.id AS id")._nodes["v"].tolist()[0]
    assert v is None or pd.isna(v)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("MATCH (n) WHERE toLower(n.name) = 'bob' RETURN n.id AS id ORDER BY id", [0]),
        ("MATCH (n) WHERE toUpper(n.name) = 'BOB' RETURN n.id AS id ORDER BY id", [0]),
    ],
)
def test_tolower_toupper(engine: str, query: str, expected: list[int]) -> None:
    if engine == "polars":
        pytest.importorskip("polars")
    # node with matching name is id 0 after we make BOB id 0; keep ids stable for both engines
    nodes = pd.DataFrame({"id": [0, 1, 2], "name": ["BOB", "Alice", "carol"]})
    col = _mk_graph(nodes, pd.DataFrame({"s": [], "d": []})).gfql(query, engine=engine)._nodes["id"]
    got = col.to_list() if hasattr(col, "to_list") else col.tolist()
    assert got == expected


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a) WHERE a.name STARTS WITH null RETURN a.name AS name ORDER BY name",
        "MATCH (a) WHERE a.name ENDS WITH null RETURN a.name AS name ORDER BY name",
        "MATCH (a) WHERE a.name CONTAINS null RETURN a.name AS name ORDER BY name",
    ],
)
def test_string_cypher_string_predicates_with_null_rhs_on_arrow_string_dtype(
    query: str,
) -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "name": pd.Series(["ABCDEF", "AB", "abcdef", "ab", "", None], dtype="string[pyarrow]"),
        }
    )
    result = _mk_graph(nodes, pd.DataFrame({"s": [], "d": []})).gfql(query)

    assert result._nodes.to_dict(orient="records") == []


@pytest.mark.parametrize(
    ("query", "nodes_df", "edges_df", "expected"),
    [
        (
            "MATCH (n) RETURN 'keys(n)' AS s",
            pd.DataFrame({"id": ["a"]}),
            pd.DataFrame({"s": [], "d": []}),
            [{"s": "keys(n)"}],
        ),
        (
            "MATCH (n) RETURN 'n:Foo' AS s",
            pd.DataFrame({"id": ["a"]}),
            pd.DataFrame({"s": [], "d": []}),
            [{"s": "n:Foo"}],
        ),
        (
            "MATCH ()-[r]->() RETURN 'r:T' AS s",
            pd.DataFrame({"id": ["a", "b"]}),
            pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["T"]}),
            [{"s": "r:T"}],
        ),
    ],
)
def test_string_cypher_preserves_string_literals_during_rewrite_passes(
    query: str,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    expected: list[dict[str, object]],
) -> None:
    result = _mk_graph(nodes_df, edges_df).gfql(query)

    assert result._nodes.to_dict(orient="records") == expected


def test_string_cypher_returns_missing_property_as_null() -> None:
    node_graph = _mk_graph(
        pd.DataFrame({"id": ["a"], "num": [1]}),
        pd.DataFrame({"s": [], "d": []}),
    )
    edge_graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "name": [1]}),
    )

    node_result = node_graph.gfql("MATCH (a) RETURN a.name")
    edge_result = edge_graph.gfql("MATCH ()-[r]->() RETURN r.name2")

    assert node_result._nodes.where(~node_result._nodes.isna(), None).to_dict(orient="records") == [
        {"a.name": None}
    ]
    assert edge_result._nodes.where(~edge_result._nodes.isna(), None).to_dict(orient="records") == [
        {"r.name2": None}
    ]


def test_string_cypher_orders_distinct_whole_row_by_missing_property() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "label__A": [True, False],
                "label__B": [False, True],
            }
        ),
        pd.DataFrame({"s": ["a"], "d": ["b"]}),
    )

    result = graph.gfql("MATCH (a)-->(b) RETURN DISTINCT b ORDER BY b.name")

    assert entity_text_records(result, {"b": "nodes"}) == [{"b": "(:B)"}]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("RETURN time('2140-00:00') AS result", [{"result": "21:40Z"}]),
        ("RETURN datetime('2015-07-20T2140-00:00') AS result", [{"result": "2015-07-20T21:40Z"}]),
    ],
)
def test_string_cypher_normalizes_zero_offset_temporals(query: str, expected: list[dict[str, object]]) -> None:
    result = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []})).gfql(query)
    assert result._nodes.where(~result._nodes.isna(), None).to_dict(orient="records") == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("RETURN (0.0 / 0.0) = 1 AS isEqual, (0.0 / 0.0) <> 1 AS isNotEqual", [{"isEqual": False, "isNotEqual": True}]),
        ("RETURN (0.0 / 0.0) = 1.0 AS isEqual, (0.0 / 0.0) <> 1.0 AS isNotEqual", [{"isEqual": False, "isNotEqual": True}]),
        ("RETURN (0.0 / 0.0) = (0.0 / 0.0) AS isEqual, (0.0 / 0.0) <> (0.0 / 0.0) AS isNotEqual", [{"isEqual": False, "isNotEqual": True}]),
        ("RETURN (0.0 / 0.0) = 'a' AS isEqual, (0.0 / 0.0) <> 'a' AS isNotEqual", [{"isEqual": False, "isNotEqual": True}]),
    ],
)
def test_string_cypher_nan_comparisons_match_cypher_semantics(query: str, expected: list[dict[str, object]]) -> None:
    result = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []})).gfql(query)
    assert result._nodes.where(~result._nodes.isna(), None).to_dict(orient="records") == expected


def test_string_cypher_formats_match_node_without_null_type_label() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["n1", "n2"],
                "name": ["bar", "baz"],
                "type": [pd.NA, pd.NA],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n {name: 'bar'}) RETURN n")

    assert entity_text_records(result, {"n": "nodes"}) == [{"n": "({name: 'bar'})"}]


def test_string_cypher_ignores_placeholder_label_columns_in_entity_rendering() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["n1"],
                "name": ["bar"],
                "label__<NA>": [True],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n {name: 'bar'}) RETURN n")

    assert entity_text_records(result, {"n": "nodes"}) == [{"n": "({name: 'bar'})"}]


def test_string_cypher_formats_numeric_id_as_entity_property() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": [1, 10]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n) RETURN DISTINCT n ORDER BY n.id")

    assert entity_text_records(result, {"n": "nodes"}) == [{"n": "({id: 1})"}, {"n": "({id: 10})"}]


def test_string_cypher_formats_small_float_entity_properties_without_scientific_notation() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "num": [30.94857, 0.00002, -0.00002],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n) RETURN n ORDER BY n.num DESC")

    assert entity_text_records(result, {"n": "nodes"}) == [
        {"n": "({num: 30.94857})"},
        {"n": "({num: 0.00002})"},
        {"n": "({num: -0.00002})"},
    ]


def test_string_cypher_supports_return_star_with_order_by() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": [1, 10]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n) RETURN * ORDER BY n.id")

    assert entity_text_records(result, {"n": "nodes"}) == [{"n": "({id: 1})"}, {"n": "({id: 10})"}]


def test_string_cypher_supports_return_label_predicate_expression() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "label__Foo": [False, True],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n) RETURN (n:Foo)")

    assert result._nodes.to_dict(orient="records") == [{"(n:Foo)": False}, {"(n:Foo)": True}]


def test_string_cypher_supports_match_with_constant_projection_before_order_by() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["c1", "c2", "c3"],
                "label__Crew": [True, True, True],
                "name": ["Neo", "Neo", "Neo"],
                "rank": [1, 2, 3],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (c:Crew {name: 'Neo'}) WITH c, 0 AS relevance RETURN c.rank AS rank ORDER BY relevance, c.rank"
    )

    assert result._nodes.to_dict(orient="records") == [{"rank": 1}, {"rank": 2}, {"rank": 3}]


def test_string_cypher_supports_path_bound_match_when_path_variable_is_unused() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "id": ["r1"]}),
    )

    result = graph.gfql("MATCH p = (n)-[r]->(b) RETURN count(r) AS cnt")

    assert result._nodes.to_dict(orient="records") == [{"cnt": 1}]


def test_string_cypher_emits_global_aggregate_row_for_empty_match() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n {id: 'missing'}) RETURN aVg(    n.aGe     )")

    assert result._nodes.where(~result._nodes.isna(), None).to_dict(orient="records") == [
        {"aVg(    n.aGe     )": None}
    ]


def test_string_cypher_supports_generic_match_where_constant_false() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "name": ["a", "b"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n)\nWHERE 1 = 0\nRETURN n\nSKIP 0")

    assert result._nodes.to_dict(orient="records") == []


def test_string_cypher_supports_generic_match_where_constant_false_with_arrow_string_props() -> None:
    pytest.importorskip("pyarrow")

    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": pd.Series(["a", "b"], dtype="string[pyarrow]"),
                "name": pd.Series(["a", "b"], dtype="string[pyarrow]"),
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n)\nWHERE 1 = 0\nRETURN n\nSKIP 0")

    assert result._nodes.to_dict(orient="records") == []


def test_string_cypher_supports_generic_match_where_boolean_expression() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a"], "name": ["a"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n)\nWHERE NOT(n.name = 'apa' AND false)\nRETURN n")

    assert entity_text_records(result, {"n": "nodes"}) == [{"n": "({name: 'a'})"}]


def test_string_cypher_executes_searched_case_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"], "score": [1, 3, 2]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (n) RETURN n.id AS id, CASE WHEN n.score > 2 THEN 'hi' ELSE 'lo' END AS bucket ORDER BY id"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"id": "a", "bucket": "lo"},
        {"id": "b", "bucket": "hi"},
        {"id": "c", "bucket": "lo"},
    ]


def test_string_cypher_executes_simple_case_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"], "score": [1, 3, 2]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (n) RETURN n.id AS id, CASE n.score WHEN 1 THEN 'lo' WHEN 3 THEN 'hi' ELSE 'mid' END AS bucket ORDER BY id"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"id": "a", "bucket": "lo"},
        {"id": "b", "bucket": "hi"},
        {"id": "c", "bucket": "mid"},
    ]


def test_string_cypher_simple_case_does_not_match_bool_to_int() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a"], "flag": [True]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (n) RETURN CASE n.flag WHEN 1 THEN 'one' ELSE 'other' END AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": "other"}]


def test_string_cypher_simple_case_when_null_matches_null_value() -> None:
    """CASE x WHEN null THEN 'yes' ELSE 'no' END — x is null → 'yes'."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "val": [None, "v"]}),
        pd.DataFrame({"s": [], "d": []}),
    )
    result = graph.gfql(
        "MATCH (n) RETURN n.id AS id, CASE n.val WHEN null THEN 'yes' ELSE 'no' END AS out ORDER BY id"
    )
    rows = result._nodes.to_dict(orient="records")
    assert {"id": "a", "out": "yes"} in rows   # null → matches null arm
    assert {"id": "b", "out": "no"} in rows    # non-null → falls to ELSE


def test_string_cypher_simple_case_when_null_non_null_not_matched() -> None:
    """CASE x WHEN null — non-null x must NOT match."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a"], "val": ["present"]}),
        pd.DataFrame({"s": [], "d": []}),
    )
    result = graph.gfql(
        "MATCH (n) RETURN CASE n.val WHEN null THEN true ELSE false END AS is_null"
    )
    assert result._nodes.to_dict(orient="records") == [{"is_null": False}]


def test_string_cypher_simple_case_when_null_all_null() -> None:
    """All rows null → all match the null arm."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "val": [None, None]}),
        pd.DataFrame({"s": [], "d": []}),
    )
    result = graph.gfql(
        "MATCH (n) RETURN n.id AS id, CASE n.val WHEN null THEN 1 ELSE 0 END AS out ORDER BY id"
    )
    for row in result._nodes.to_dict(orient="records"):
        assert row["out"] == 1


def test_string_cypher_simple_case_when_null_mixed_series() -> None:
    """Mixed series: some null, some non-null, some other value."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"], "val": [None, "x", None]}),
        pd.DataFrame({"s": [], "d": []}),
    )
    result = graph.gfql(
        "MATCH (n) RETURN n.id AS id, CASE n.val WHEN null THEN 'null' ELSE 'set' END AS out ORDER BY id"
    )
    rows = {r["id"]: r["out"] for r in result._nodes.to_dict(orient="records")}
    assert rows == {"a": "null", "b": "set", "c": "null"}


def test_string_cypher_simple_case_when_null_no_else() -> None:
    """CASE x WHEN null THEN 'y' END — no ELSE; non-null should yield null."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "val": [None, "v"]}),
        pd.DataFrame({"s": [], "d": []}),
    )
    result = graph.gfql(
        "MATCH (n) RETURN n.id AS id, CASE n.val WHEN null THEN 'hit' END AS out ORDER BY id"
    )
    rows = {r["id"]: r["out"] for r in result._nodes.to_dict(orient="records")}
    assert rows["a"] == "hit"
    assert rows["b"] is None or (isinstance(rows["b"], float) and rows["b"] != rows["b"])


def test_string_cypher_simple_case_non_null_comparison_unaffected() -> None:
    """Non-null WHEN arm still works after null fix — regression guard."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"], "score": [1, 2, 3]}),
        pd.DataFrame({"s": [], "d": []}),
    )
    result = graph.gfql(
        "MATCH (n) RETURN n.id AS id, CASE n.score WHEN 2 THEN 'two' ELSE 'other' END AS out ORDER BY id"
    )
    rows = {r["id"]: r["out"] for r in result._nodes.to_dict(orient="records")}
    assert rows == {"a": "other", "b": "two", "c": "other"}


def test_string_cypher_simple_case_when_null_first_arm_then_value_arm() -> None:
    """Multiple arms: first is null, second is a value."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"], "val": [None, "x", "y"]}),
        pd.DataFrame({"s": [], "d": []}),
    )
    result = graph.gfql(
        "MATCH (n) RETURN n.id AS id, CASE n.val WHEN null THEN 'N' WHEN 'x' THEN 'X' ELSE 'E' END AS out ORDER BY id"
    )
    rows = {r["id"]: r["out"] for r in result._nodes.to_dict(orient="records")}
    assert rows == {"a": "N", "b": "X", "c": "E"}


def test_string_cypher_supports_generic_match_where_chained_comparison() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a"], "num": [5]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n)\nWHERE 10 < n.num <= 3\nRETURN n.num")

    assert result._nodes.to_dict(orient="records") == []


def test_issue_1413_searched_case_rewrites_multiple_chained_when_arms() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"], "score": [5, 15, 25]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (n) RETURN n.id AS id, "
        "CASE WHEN 0 <= n.score < 10 THEN 'low' "
        "WHEN 10 <= n.score < 20 THEN 'mid' "
        "ELSE 'high' END AS bucket ORDER BY id"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"id": "a", "bucket": "low"},
        {"id": "b", "bucket": "mid"},
        {"id": "c", "bucket": "high"},
    ]


def test_issue_1413_searched_case_preserves_unrelated_comparisons() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"], "score": [-1, 5, 15]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (n) RETURN n.id AS id, "
        "CASE WHEN n.score > 0 THEN n.score < 10 ELSE false END AS inRange "
        "ORDER BY id"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"id": "a", "inRange": False},
        {"id": "b", "inRange": True},
        {"id": "c", "inRange": False},
    ]


def test_string_cypher_supports_distinct_with_aggregate_grouping() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "color": ["red", "red", "blue"],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (a) RETURN DISTINCT a.color, count(*)")

    actual = sorted(
        result._nodes.where(~result._nodes.isna(), None).to_dict(orient="records"),
        key=lambda row: cast(str, row["a.color"]),
    )

    assert actual == [
        {"a.color": "blue", "count(*)": 1},
        {"a.color": "red", "count(*)": 2},
    ]


def test_string_cypher_collects_node_entities_in_aggregate_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["n1"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n) RETURN count(n), collect(n)")

    assert result._nodes.to_dict(orient="records") == [{"count(n)": 1, "collect(n)": ["()"]}]


def test_string_cypher_supports_post_aggregate_arithmetic_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["n1"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH () RETURN count(*) * 10 AS c")

    assert result._nodes.to_dict(orient="records") == [{"c": 10}]


def test_string_cypher_uses_integer_division_for_post_aggregate_expression() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": [f"n{i}" for i in range(7)]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n) RETURN count(n) / 3 / 2 AS count")

    assert result._nodes.to_dict(orient="records") == [{"count": 1}]


def test_string_cypher_uses_integer_division_for_literal_arithmetic_expression() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": []}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("RETURN 12 / 4 * 3 - 2 * 4")

    assert result._nodes.to_dict(orient="records") == [{"12 / 4 * 3 - 2 * 4": 1}]


def test_string_cypher_preserves_escaped_literal_backslash_pairs() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": []}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(r"""RETURN 'a\\bcn5t\'"\\//\\"\'' AS literal""")

    assert result._nodes.to_dict(orient="records") == [
        {"literal": "a\\\\bcn5t'\"\\\\//\\\\\"'"}
    ]


def test_string_cypher_preserves_unicode_string_literal() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": []}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(r"RETURN '\u01FF' AS literal")

    assert result._nodes.to_dict(orient="records") == [{"literal": "\u01ff"}]


def test_string_cypher_handles_standard_string_literal_escapes() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": []}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(r"""RETURN '\n\t\r\b\f\\\'\"' AS literal""")

    assert result._nodes.to_dict(orient="records") == [
        {"literal": "\n\t\r\b\f\\\\'\""}
    ]


@pytest.mark.parametrize("query", [r"RETURN '\uH' AS literal", "RETURN 'unterminated AS literal"])
def test_string_cypher_rejects_invalid_string_literals(query: str) -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": []}),
        pd.DataFrame({"s": [], "d": []}),
    )

    with pytest.raises(GFQLSyntaxError):
        graph.gfql(query)


@pytest.mark.parametrize(
    "query,expected_rows",
    [
        (
            "RETURN 4 * 2 + 3 / 2 AS a, 4 * 2 + (3 / 2) AS b, 4 * (2 + 3) / 2 AS c",
            [{"a": 9, "b": 9, "c": 10}],
        ),
        (
            "RETURN 4 * 2 - 3 / 2 AS a, 4 * 2 - (3 / 2) AS b, 4 * (2 - 3) / 2 AS c",
            [{"a": 7, "b": 7, "c": -2}],
        ),
        (
            "RETURN 4 / 2 + 3 * 2 AS a, 4 / 2 + (3 * 2) AS b, 4 / (2 + 3) * 2 AS c",
            [{"a": 8, "b": 8, "c": 0}],
        ),
        (
            "RETURN 4 / 2 + 3 / 2 AS a, 4 / 2 + (3 / 2) AS b, 4 / (2 + 3) / 2 AS c",
            [{"a": 3, "b": 3, "c": 0}],
        ),
        (
            "RETURN 4 / 2 + 3 % 2 AS a, 4 / 2 + (3 % 2) AS b, 4 / (2 + 3) % 2 AS c",
            [{"a": 3, "b": 3, "c": 0}],
        ),
        (
            "RETURN 4 / 2 - 3 * 2 AS a, 4 / 2 - (3 * 2) AS b, 4 / (2 - 3) * 2 AS c",
            [{"a": -4, "b": -4, "c": -8}],
        ),
        (
            "RETURN 4 / 2 - 3 / 2 AS a, 4 / 2 - (3 / 2) AS b, 4 / (2 - 3) / 2 AS c",
            [{"a": 1, "b": 1, "c": -2}],
        ),
        (
            "RETURN 4 / 2 - 3 % 2 AS a, 4 / 2 - (3 % 2) AS b, 4 / (2 - 3) % 2 AS c",
            [{"a": 1, "b": 1, "c": 0}],
        ),
        (
            "RETURN 4 % 2 + 3 / 2 AS a, 4 % 2 + (3 / 2) AS b, 4 % (2 + 3) / 2 AS c",
            [{"a": 1, "b": 1, "c": 2}],
        ),
        (
            "RETURN 4 % 2 - 3 / 2 AS a, 4 % 2 - (3 / 2) AS b, 4 % (2 - 3) / 2 AS c",
            [{"a": -1, "b": -1, "c": 0}],
        ),
    ],
)
def test_string_cypher_precedence_examples_use_integer_division(
    query: str,
    expected_rows: List[Dict[str, Any]],
) -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["n1"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(query)

    assert result._nodes.to_dict(orient="records") == expected_rows


def test_string_cypher_supports_whole_row_grouping_with_post_aggregate_expression() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["n1"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (a) RETURN a, count(a) + 3")

    # Aggregate/grouping projection still renders the entity as a single text
    # column via a separate path (not the structured #1650 terminal-RETURN path).
    assert result._nodes.to_dict(orient="records") == [{"a": "()", "count(a) + 3": 4}]


def test_string_cypher_supports_whole_row_grouping_with_count_star() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a"],
                "label__L": [True],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (a:L) RETURN a, count(*)")

    # Aggregate/grouping projection renders entity text via a separate path.
    assert result._nodes.to_dict(orient="records") == [{"a": "(:L)", "count(*)": 1}]


@pytest.mark.parametrize(
    "query,expected_rows",
    [
        ("MATCH (a:L)-[r]->(b) RETURN a.id AS aid, count(*) AS cnt", [{"aid": "a", "cnt": 2}]),
        ("MATCH (a)-[r]->(b) RETURN count(*) AS cnt", [{"cnt": 2}]),
        ("MATCH (a)-[r]->(b) RETURN count(a) AS cnt", [{"cnt": 2}]),
        ("MATCH (a)-[r]->(b) RETURN sum(1) AS total", [{"total": 2}]),
        ("MATCH (a)-[r]->(b) RETURN avg(r.weight) AS avg_w", [{"avg_w": 2.5}]),
    ],
)
def test_string_cypher_supports_relationship_row_multiplicity_sensitive_aggregates(
    query: str,
    expected_rows: List[Dict[str, Any]],
) -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "label__L": [True, False, False],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a", "a"],
                "d": ["b", "c"],
                "id": ["r1", "r2"],
                "weight": [2, 3],
            }
        ),
    )

    result = graph.gfql(query)
    assert result._nodes.to_dict(orient="records") == expected_rows


def test_string_cypher_failfast_relationship_whole_row_grouped_count_star_boundary() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "label__L": [True, False, False],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a", "a"],
                "d": ["b", "c"],
                "id": ["r1", "r2"],
            }
        ),
    )

    with pytest.raises(GFQLValidationError, match="repeated MATCH rows"):
        graph.gfql("MATCH (a:L)-[rel]->(b) RETURN a, count(*)")


def test_string_cypher_failfast_optional_match_collect_null_whole_row_return_boundary() -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["n1"]}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql("MATCH (n) OPTIONAL MATCH (n)-[:NOT_EXIST]->(x) RETURN n, collect(x)")
    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "return"
    assert exc_info.value.context["value"] == "x"


def test_string_cypher_failfast_optional_match_collect_null_whole_row_with_boundary() -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["n1"]}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(
            "MATCH (n) "
            "OPTIONAL MATCH (n)-[:NOT_EXIST]->(x) "
            "WITH n, collect(x) AS xs "
            "RETURN n, xs"
        )
    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "with"
    assert exc_info.value.context["value"] == ["n", "collect(x)"]


def test_string_cypher_supports_optional_match_collect_alias_property() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "u1", "u2"],
                "label__Person": [True, False, False],
                "label__University": [False, True, True],
                "name": ["P", "Uni1", "Uni2"],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p1"],
                "d": ["u1", "u2"],
                "id": ["e1", "e2"],
                "type": ["STUDY_AT", "STUDY_AT"],
                "classYear": [2001, 2002],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (p:Person {id: 'p1'}) "
        "OPTIONAL MATCH (p)-[rel:STUDY_AT]->(uni:University) "
        "WITH p, collect(uni.name) AS unis "
        "RETURN p.id AS personId, unis"
    )

    assert result._nodes.to_dict(orient="records") == [{"personId": "p1", "unis": ["Uni1", "Uni2"]}]


def test_string_cypher_supports_optional_match_collect_case_relationship_rows() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "u1", "u2"],
                "label__Person": [True, False, False],
                "label__University": [False, True, True],
                "name": ["P", "Uni1", "Uni2"],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p1"],
                "d": ["u1", "u2"],
                "id": ["e1", "e2"],
                "type": ["STUDY_AT", "STUDY_AT"],
                "classYear": [2001, 2002],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (p:Person {id: 'p1'}) "
        "OPTIONAL MATCH (p)-[rel:STUDY_AT]->(uni:University) "
        "WITH p, collect(CASE uni.name WHEN null THEN null ELSE [uni.name, rel.classYear] END) AS unis "
        "RETURN p.id AS personId, unis"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"personId": "p1", "unis": [["Uni1", 2001], ["Uni2", 2002]]}
    ]


def test_string_cypher_supports_relationship_row_grouped_count_sum_and_avg() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame(
            {
                "s": ["a", "a"],
                "d": ["b", "c"],
                "id": ["r1", "r2"],
                "type": ["R", "R"],
                "weight": [2, 3],
            }
        ),
    )

    count_result = graph.gfql(
        "MATCH (a)-[r:R]->(b) "
        "WITH a, count(r) AS deg "
        "RETURN a.id AS aid, deg"
    )
    sum_result = graph.gfql(
        "MATCH (a)-[r:R]->(b) "
        "WITH a, sum(r.weight) AS total "
        "RETURN a.id AS aid, total"
    )
    avg_result = graph.gfql(
        "MATCH (a)-[r:R]->(b) "
        "WITH a, avg(r.weight) AS avg_w "
        "RETURN a.id AS aid, avg_w"
    )

    assert count_result._nodes.to_dict(orient="records") == [{"aid": "a", "deg": 2}]
    assert sum_result._nodes.to_dict(orient="records") == [{"aid": "a", "total": 5}]
    assert avg_result._nodes.to_dict(orient="records") == [{"aid": "a", "avg_w": 2.5}]


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a:L)-[r]->(b) RETURN a.id AS aid, count(r) AS cnt",
        "MATCH (a:L)-[r]->(b) RETURN a.id AS aid, sum(r.weight) AS total",
        "MATCH (a:L)-[r]->(b) RETURN a.id AS aid, avg(r.weight) AS avg_w",
    ],
)
def test_string_cypher_direct_return_grouped_relationship_aggregate_one_source_boundary(
    query: str,
) -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "label__L": [True, False, False],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a", "a"],
                "d": ["b", "c"],
                "id": ["r1", "r2"],
                "type": ["R", "R"],
                "weight": [2, 3],
            }
        ),
    )

    with pytest.raises(GFQLValidationError, match="one MATCH source alias at a time"):
        graph.gfql(query)


def test_string_cypher_keeps_single_edge_relationship_grouped_count_star() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "a"], "d": ["b", "c"], "id": ["r1", "r2"]}),
    )

    result = graph.gfql("MATCH (a)-[r]->(b) RETURN r.id AS rid, count(*) AS cnt")

    assert result._nodes.to_dict(orient="records") == [{"rid": "r1", "cnt": 1}, {"rid": "r2", "cnt": 1}]


def test_string_cypher_keeps_single_edge_relationship_global_count() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "a"], "d": ["b", "c"], "id": ["r1", "r2"]}),
    )

    result = graph.gfql("MATCH (a)-[r]->(b) RETURN count(r) AS cnt")

    assert result._nodes.to_dict(orient="records") == [{"cnt": 2}]


def test_string_cypher_supports_with_whole_row_grouping_then_return() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["x", "y"],
                "label__X": [True, True],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (x:X) "
        "WITH x, count(*) AS c "
        "ORDER BY c LIMIT 1 "
        "RETURN x, c"
    )

    # Aggregate/grouping projection renders entity text via a separate path.
    assert result._nodes.to_dict(orient="records") == [{"x": "(:X)", "c": 1}]


def test_string_cypher_supports_post_aggregate_size_collect_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["n1", "n2", "n3"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (a) RETURN size(collect(a)) AS n")

    assert result._nodes.to_dict(orient="records") == [{"n": 3}]


def test_string_cypher_supports_post_aggregate_size_collect_projection_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(pd.DataFrame({"id": [f"n{i}" for i in range(11)], "labels": [[] for _ in range(11)]}))
    edges = cudf.from_pandas(pd.DataFrame({"s": [], "d": []}))

    result = _mk_graph(nodes, edges).gfql("MATCH (a) RETURN size(collect(a)) AS n", engine="cudf")

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"n": 11}]


def test_issue_1367_empty_optional_match_post_aggregate_list_comprehension_size() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": []}),
        pd.DataFrame({"s": [], "d": [], "type": []}),
    )

    result = graph.gfql(
        "MATCH (n) "
        "OPTIONAL MATCH (n)-[r]->(m) "
        "RETURN size([x IN collect(r) WHERE x <> null]) AS cn"
    )

    assert result._nodes.to_dict(orient="records") == [{"cn": 0}]


def test_issue_1367_empty_optional_match_multiple_post_aggregate_projections() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": []}),
        pd.DataFrame({"s": [], "d": [], "type": []}),
    )

    result = graph.gfql(
        "MATCH (n) "
        "OPTIONAL MATCH (n)-[r]->(m) "
        "RETURN count(r) AS total, "
        "size([x IN collect(r) WHERE x <> null]) AS nonnull"
    )

    assert result._nodes.to_dict(orient="records") == [{"total": 0, "nonnull": 0}]


def test_issue_1367_grouped_aggregate_does_not_synthesize_empty_row() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": []}),
        pd.DataFrame({"s": [], "d": [], "type": []}),
    )

    result = graph.gfql(
        "MATCH (n) "
        "RETURN n.id AS id, size([x IN collect(n) WHERE x <> null]) AS cn"
    )

    assert result._nodes.to_dict(orient="records") == []


def test_issue_1367_optional_match_existing_node_missing_relationship_counts_zero() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["n1"]}),
        pd.DataFrame({"s": [], "d": [], "type": []}),
    )

    result = graph.gfql(
        "MATCH (n) "
        "OPTIONAL MATCH (n)-[r]->(m) "
        "RETURN size([x IN collect(r) WHERE x <> null]) AS cn"
    )

    assert result._nodes.to_dict(orient="records") == [{"cn": 0}]


def test_issue_1367_post_aggregate_list_projection_counts_existing_relationships() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["n1", "n2"]}),
        pd.DataFrame({"s": ["n1"], "d": ["n2"], "type": ["REL"]}),
    )

    result = graph.gfql(
        "MATCH (n)-[r]->(m) "
        "RETURN size([x IN collect(r) WHERE x = x]) AS cn"
    )

    assert result._nodes.to_dict(orient="records") == [{"cn": 1}]


def test_issue_1367_empty_optional_match_post_aggregate_list_comprehension_size_on_cudf() -> None:
    pytest.importorskip("cudf")
    graph = _mk_cudf_graph(
        pd.DataFrame({"id": pd.Series(dtype="object")}),
        pd.DataFrame(
            {
                "s": pd.Series(dtype="object"),
                "d": pd.Series(dtype="object"),
                "type": pd.Series(dtype="object"),
            }
        ),
    )

    result = graph.gfql(
        "MATCH (n) "
        "OPTIONAL MATCH (n)-[r]->(m) "
        "RETURN size([x IN collect(r) WHERE x <> null]) AS cn",
        engine="cudf",
    )

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"cn": 0}]


def test_issue_1367_post_aggregate_list_projection_counts_existing_relationships_on_cudf() -> None:
    pytest.importorskip("cudf")
    graph = _mk_cudf_graph(
        pd.DataFrame({"id": ["n1", "n2"]}),
        pd.DataFrame({"s": ["n1"], "d": ["n2"], "type": ["REL"]}),
    )

    result = graph.gfql(
        "MATCH (n)-[r]->(m) "
        "RETURN size([x IN collect(r) WHERE x = x]) AS cn",
        engine="cudf",
    )

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"cn": 1}]


def test_string_cypher_supports_grouped_post_aggregate_size_collect_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["andres", "anna"],
                "name": ["Andres", "Anna"],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (a) "
        "RETURN a.name AS name, size(collect(a.name)) AS n "
        "ORDER BY name"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"name": "Andres", "n": 1},
        {"name": "Anna", "n": 1},
    ]


def test_string_cypher_orders_on_aggregate_expression_using_projected_outputs() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["m1", "m2", "m3", "m4"],
                "label__Person": [True, True, True, True],
                "age": [20, 20, 30, 30],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (me:Person) "
        "RETURN me.age AS age, count(*) AS cnt "
        "ORDER BY age, age + count(*)"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"age": 20, "cnt": 2},
        {"age": 30, "cnt": 2},
    ]


def test_string_cypher_supports_bare_label_predicate_in_with_where() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["root", "t1", "t2"],
                "label__Root": [True, False, False],
                "label__TextNode": [False, True, True],
                "name": ["x", None, None],
                "var": ["aa", "tf", "td"],
            }
        ),
        pd.DataFrame({"s": ["root", "root"], "d": ["t1", "t2"]}),
    )

    result = graph.gfql("MATCH (:Root {name: 'x'})-->(i:TextNode) WITH i WHERE i.var > 'te' AND i:TextNode RETURN i")

    assert entity_text_records(result, {"i": "nodes"}) == [{"i": "(:TextNode {var: 'tf'})"}]


# OR/NOT WHERE shapes — Earley admits them where LALR rejected.  Pandas
# 3-valued NaN semantics: NaN comparisons → NaN (falsy), NaN under NOT →
# NaN (still filtered).  Tests lock that behavior.


def _or_where_graph() -> "_CypherTestGraph":
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "label__A": [True, False, False],
                "label__B": [False, True, False],
                "label__C": [False, False, True],
                "p1": [12.0, float("nan"), float("nan")],
                "p2": [float("nan"), 13.0, float("nan")],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )


def test_string_cypher_executes_disjunctive_property_predicate_returns_union() -> None:
    result = _or_where_graph().gfql("MATCH (n) WHERE n.p1 = 12 OR n.p2 = 13 RETURN n")

    rendered = sorted(row["n"] for row in entity_text_records(result, {"n": "nodes"}))
    assert rendered == ["(:A {p1: 12})", "(:B {p2: 13})"]


def test_string_cypher_executes_disjunctive_same_alias_property_predicate() -> None:
    result = _or_where_graph().gfql("MATCH (n) WHERE n.p1 = 12 OR n.p1 = 99 RETURN n")

    rendered = sorted(row["n"] for row in entity_text_records(result, {"n": "nodes"}))
    assert rendered == ["(:A {p1: 12})"]


def test_string_cypher_executes_negation_property_predicate_returns_complement() -> None:
    result = _or_where_graph().gfql("MATCH (n) WHERE NOT n.p1 = 12 RETURN n")

    rendered = sorted(row["n"] for row in entity_text_records(result, {"n": "nodes"}))
    assert rendered == []


def test_string_cypher_executes_disjunctive_then_conjunction() -> None:
    result = _or_where_graph().gfql(
        "MATCH (n) WHERE (n.p1 = 12 OR n.p2 = 13) AND n.id = 'a' RETURN n"
    )

    rendered = [row["n"] for row in entity_text_records(result, {"n": "nodes"})]
    assert rendered == ["(:A {p1: 12})"]


def test_string_cypher_executes_disjunction_returns_correct_count_with_more_rows() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["x1", "x2", "y1", "y2", "z", "w"],
                "p1": [1.0, 1.0, float("nan"), float("nan"), 1.0, float("nan")],
                "p2": [float("nan"), float("nan"), 2.0, 2.0, 2.0, float("nan")],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n) WHERE n.p1 = 1 OR n.p2 = 2 RETURN n.id AS id")

    ids = sorted(row["id"] for row in result._nodes.to_dict(orient="records"))
    assert ids == ["x1", "x2", "y1", "y2", "z"]


# Compositional row-boolean shapes that Earley admits (LALR rejected).
# Locked behavior — see #1219 for the broader unverified-cases ledger.


def test_string_cypher_executes_cross_alias_or_returns_correct_union() -> None:
    # Cross-alias OR with 2x2 join topology: each a-node connects to
    # multiple b-nodes (and vice versa) so that "distribute-OR-then-
    # push-each-disjunct" pushdown variants would produce a different
    # row set than the correct row-wise post-join evaluation.
    #
    # Topology:
    #   a1 (x=1) -> b1 (y=99), b2 (y=2)
    #   a2 (x=99)-> b1 (y=99), b2 (y=2)
    #   a3 (x=99)-> b3 (y=99)  ← neither branch holds; must be excluded
    #
    # Correct row-wise OR evaluates 5 joined rows, keeps where (a.x=1
    # OR b.y=2) — that's (a1,b1), (a1,b2), (a2,b2).  Excludes (a2,b1)
    # and (a3,b3).  Wrong-pushdown variants:
    #   * push a.x=1 only: rows (a1,b1), (a1,b2) — 2 rows, missing (a2,b2)
    #   * push b.y=2 only: rows (a1,b2), (a2,b2) — 2 rows, missing (a1,b1)
    #   * push as AND: rows (a1,b2) — 1 row
    #   * distribute-OR pushdown (union of both branches pre-join):
    #     {(a1,b1),(a1,b2)} ∪ {(a1,b2),(a2,b2)} = same 3 rows BUT only
    #     because a-side dropping a2/a3 still produces (a2,b2) via the
    #     b-branch — confirms distribute-OR converges to correct answer
    #     here, which is the underlying OR-distributivity-over-join
    #     property (correct).
    nodes = pd.DataFrame({
        "id":       ["a1", "a2", "a3", "b1", "b2", "b3"],
        "label__A": [True, True, True, False, False, False],
        "label__B": [False, False, False, True, True, True],
        "x":        [1.0, 99.0, 99.0, float("nan"), float("nan"), float("nan")],
        "y":        [float("nan"), float("nan"), float("nan"), 99.0, 2.0, 99.0],
    })
    edges = pd.DataFrame({
        "s": ["a1", "a1", "a2", "a2", "a3"],
        "d": ["b1", "b2", "b1", "b2", "b3"],
    })
    graph = _mk_graph(nodes, edges)

    result = graph.gfql(
        "MATCH (a:A)-->(b:B) WHERE a.x = 1 OR b.y = 2 RETURN a.id AS a_id, b.id AS b_id"
    )

    rows = sorted(
        (row["a_id"], row["b_id"])
        for row in result._nodes.to_dict(orient="records")
    )
    assert rows == [("a1", "b1"), ("a1", "b2"), ("a2", "b2")]


def test_string_cypher_executes_cross_alias_or_on_cartesian_product_topology() -> None:
    # Residual #1219 frontier: explicit cartesian-product topology stress.
    # Ensure row-wise OR semantics stay correct when A/B are disconnected in
    # MATCH and joined only by cartesian expansion.
    nodes = pd.DataFrame({
        "id":       ["a1", "a2", "a3", "b1", "b2", "b3"],
        "label__A": [True, True, True, False, False, False],
        "label__B": [False, False, False, True, True, True],
        "x":        [1.0, 99.0, 99.0, float("nan"), float("nan"), float("nan")],
        "y":        [float("nan"), float("nan"), float("nan"), 2.0, 99.0, 2.0],
    })
    graph = _mk_graph(nodes, pd.DataFrame({"s": [], "d": []}))

    result = graph.gfql(
        "MATCH (a:A), (b:B) WHERE a.x = 1 OR b.y = 2 RETURN a.id AS a_id, b.id AS b_id"
    )

    rows = sorted(
        (row["a_id"], row["b_id"])
        for row in result._nodes.to_dict(orient="records")
    )
    assert rows == [
        ("a1", "b1"),
        ("a1", "b2"),
        ("a1", "b3"),
        ("a2", "b1"),
        ("a2", "b3"),
        ("a3", "b1"),
        ("a3", "b3"),
    ]


def test_string_cypher_executes_cross_alias_or_on_two_hop_fanout_topology() -> None:
    # Residual #1219 frontier: broader fan-out topology than the 2x2 lock.
    # Stress a two-hop many-to-many path where both disjunct branches should
    # independently retain rows across different path instances.
    nodes = pd.DataFrame({
        "id":       ["a1", "a2", "m1", "m2", "b1", "b2", "b3"],
        "label__A": [True, True, False, False, False, False, False],
        "label__M": [False, False, True, True, False, False, False],
        "label__B": [False, False, False, False, True, True, True],
        "x":        [1.0, 99.0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
        "y":        [float("nan"), float("nan"), float("nan"), float("nan"), 99.0, 2.0, 99.0],
    })
    edges = pd.DataFrame({
        "s": ["a1", "a1", "a2", "a2", "m1", "m1", "m2", "m2"],
        "d": ["m1", "m2", "m1", "m2", "b1", "b2", "b2", "b3"],
    })
    graph = _mk_graph(nodes, edges)

    result = graph.gfql(
        "MATCH (a:A)-->(m:M)-->(b:B) "
        "WHERE a.x = 1 OR b.y = 2 "
        "RETURN a.id AS a_id, m.id AS m_id, b.id AS b_id"
    )

    rows = sorted(
        (row["a_id"], row["m_id"], row["b_id"])
        for row in result._nodes.to_dict(orient="records")
    )
    assert rows == [
        ("a1", "m1", "b1"),
        ("a1", "m1", "b2"),
        ("a1", "m2", "b2"),
        ("a1", "m2", "b3"),
        ("a2", "m1", "b2"),
        ("a2", "m2", "b2"),
    ]


@pytest.mark.parametrize("where_clause", [
    "",  # no WHERE at all
    "WHERE b.x = 1",  # simple WHERE
    "WHERE b.x = 1 OR b IS NULL",  # WHERE with OR (the Earley-admitted shape)
])
def test_string_cypher_rejects_optional_match_seed_only_projection(where_clause: str) -> None:
    # The existing OPTIONAL-MATCH-projection validator gates this shape
    # regardless of whether/how WHERE is structured.  Locks that the OR
    # variant doesn't slip past the validator into a silent wrong-rows
    # path — the gate fires identically across all three WHERE shapes.
    graph = _mk_graph(
        pd.DataFrame({
            "id":       ["a1", "a2", "b1"],
            "label__A": [True, True, False],
            "label__B": [False, False, True],
            "x":        [float("nan"), float("nan"), 1.0],
        }),
        pd.DataFrame({"s": ["a1"], "d": ["b1"]}),
    )

    # Pin the specific substring of the seed-only-projection validator
    # so the test cannot accidentally pass on a different OPTIONAL-MATCH
    # error site (the lowering layer has 6+ distinct error messages
    # mentioning "OPTIONAL MATCH" — they're not interchangeable).
    with pytest.raises(GFQLValidationError, match="seed alias are not yet supported"):
        graph.gfql(
            f"MATCH (a:A) OPTIONAL MATCH (a)-->(b:B) {where_clause} "
            "RETURN a.id AS a_id, b.id AS b_id"
        )


def test_string_cypher_executes_mixed_type_or_with_null_safe_semantics() -> None:
    # Mixed-type OR now evaluates row-wise with null-safe comparison
    # semantics: incomparable rows are non-matching instead of raising.
    graph = _mk_graph(
        pd.DataFrame({
            "id":       ["n1", "n2", "n3"],
            "label__N": [True, True, True],
            "var":      ["text", 0, 5],
        }),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n:N) WHERE n.var > 'te' OR n.var > 0 RETURN n.id AS id")
    ids = sorted(row["id"] for row in result._nodes.to_dict(orient="records"))
    assert ids == ["n1", "n3"]


def test_string_cypher_executes_homogeneous_or_returns_correct_union() -> None:
    # Sanity check: simple homogeneous-type OR continues to work post-#1217.
    graph = _mk_graph(
        pd.DataFrame({
            "id":       ["n1", "n2", "n3"],
            "label__N": [True, True, True],
            "var":      [1, 5, 10],
        }),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n:N) WHERE n.var > 8 OR n.var < 3 RETURN n.id AS id")

    ids = sorted(row["id"] for row in result._nodes.to_dict(orient="records"))
    assert ids == ["n1", "n3"]


# Compositional row-boolean shapes (#1219 residual matrix).  Each shape locks
# Cypher 3VL semantics + boolean-tree composition correctness against
# fixtures designed to discriminate against subtle bugs.


def _three_valued_logic_fixture_graph() -> _CypherTestGraph:
    # 4-row fixture mixing actual and projected NaN over (x, y) for 3VL tests.
    return _mk_graph(
        pd.DataFrame({
            "id":       ["n1", "n2", "n3", "n4"],
            "label__N": [True, True, True, True],
            "x":        [1.0, 2.0, float("nan"), float("nan")],
            "y":        [10.0, float("nan"), 20.0, float("nan")],
        }),
        pd.DataFrame({"s": [], "d": []}),
    )


def _rows_for_issue_1219_engine(result: Any, engine: str | None) -> List[Dict[str, Any]]:
    frame = _to_pandas_df(result._nodes) if engine == "cudf" else result._nodes
    return cast(List[Dict[str, Any]], frame.to_dict(orient="records"))


def _engine_kwargs_for_issue_1219(engine: str | None) -> Dict[str, Any]:
    if engine == "cudf":
        _require_cudf_runtime()
        return {"engine": "cudf"}
    return {}


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_string_cypher_executes_nullable_not_or_uses_three_valued_logic(engine: str | None) -> None:
    # `WHERE NOT n.x = 1 OR n.y IS NULL` against a fixture mixing actual
    # and projected nulls.  Cypher 3VL truth table:
    #   n1{x=1, y=10}:   NOT(1=1)=F,    y IS NULL=F → F OR F = F → drop
    #   n2{x=2, y=NaN}:  NOT(2=1)=T,    y IS NULL=T → T OR T = T → keep
    #   n3{x=NaN,y=20}:  NOT(NaN=1)=NULL, y IS NULL=F → NULL OR F = NULL → drop
    #   n4{x=NaN,y=NaN}: NOT(NaN=1)=NULL, y IS NULL=T → NULL OR T = T → keep
    # Locks that the pandas-backed row-evaluator preserves NULL OR T = T.
    graph = _three_valued_logic_fixture_graph()

    result = graph.gfql(
        "MATCH (n:N) WHERE NOT n.x = 1 OR n.y IS NULL RETURN n.id AS id",
        **_engine_kwargs_for_issue_1219(engine),
    )

    ids = sorted(row["id"] for row in _rows_for_issue_1219_engine(result, engine))
    assert ids == ["n2", "n4"]


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_string_cypher_executes_nary_or_returns_full_union(engine: str | None) -> None:
    # `WHERE n.x = 1 OR n.x = 2 OR n.x = 3` — three OR branches against a
    # 5-row fixture where each value matches a unique row.  Locks that the
    # binder's parse evaluates ALL branches; silently dropping any one
    # branch yields a 2-row result and fails the assertion.
    graph = _mk_graph(
        pd.DataFrame({
            "id":       ["n1", "n2", "n3", "n4", "n5"],
            "label__N": [True, True, True, True, True],
            "x":        [1, 2, 3, 4, 5],
        }),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (n:N) WHERE n.x = 1 OR n.x = 2 OR n.x = 3 RETURN n.id AS id",
        **_engine_kwargs_for_issue_1219(engine),
    )

    ids = sorted(row["id"] for row in _rows_for_issue_1219_engine(result, engine))
    assert ids == ["n1", "n2", "n3"]


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_string_cypher_executes_nary_or_with_duplicate_branch_locks_specific_associativity(engine: str | None) -> None:
    # Companion to test_string_cypher_executes_nary_or_returns_full_union:
    # `WHERE n.x = 1 OR n.x = 1 OR n.x = 3` has a duplicated leftmost
    # branch.  If the binder silently dropped the rightmost branch under
    # an associativity bug, the result would be `[n1]` only.  If it
    # silently dropped one of the duplicates, the result is still `[n1, n3]`
    # (correct) — so this isolates the rightmost-drop case from the
    # any-branch-drop case the previous test covers.
    graph = _mk_graph(
        pd.DataFrame({
            "id":       ["n1", "n2", "n3"],
            "label__N": [True, True, True],
            "x":        [1, 2, 3],
        }),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (n:N) WHERE n.x = 1 OR n.x = 1 OR n.x = 3 RETURN n.id AS id",
        **_engine_kwargs_for_issue_1219(engine),
    )

    ids = sorted(row["id"] for row in _rows_for_issue_1219_engine(result, engine))
    assert ids == ["n1", "n3"]


def _de_morgan_fixture_graph() -> _CypherTestGraph:
    # 4-row fixture covering all (x∈{1,2}, y∈{2,3}) combos.
    return _mk_graph(
        pd.DataFrame({
            "id":       ["n1", "n2", "n3", "n4"],
            "label__N": [True, True, True, True],
            "x":        [1, 1, 2, 2],
            "y":        [2, 3, 2, 3],
        }),
        pd.DataFrame({"s": [], "d": []}),
    )


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
@pytest.mark.parametrize("compound,distributed,expected", [
    # NOT(A OR B) ≡ NOT(A) AND NOT(B) — both forms must return {n4}
    (
        "MATCH (n:N) WHERE NOT (n.x = 1 OR n.y = 2) RETURN n.id AS id",
        "MATCH (n:N) WHERE NOT n.x = 1 AND NOT n.y = 2 RETURN n.id AS id",
        ["n4"],
    ),
    # NOT(A AND B) ≡ NOT(A) OR NOT(B) — both forms must return {n2,n3,n4}
    (
        "MATCH (n:N) WHERE NOT (n.x = 1 AND n.y = 2) RETURN n.id AS id",
        "MATCH (n:N) WHERE NOT n.x = 1 OR NOT n.y = 2 RETURN n.id AS id",
        ["n2", "n3", "n4"],
    ),
])
def test_string_cypher_executes_de_morgan_compositions(
    compound: str, distributed: str, expected: List[str], engine: str | None,
) -> None:
    # Each NOT-of-compound and its De-Morganed equivalent must return the
    # same row set AND that row set must equal the hardcoded expected.
    graph = _de_morgan_fixture_graph()

    compound_result = graph.gfql(compound, **_engine_kwargs_for_issue_1219(engine))
    distributed_result = graph.gfql(distributed, **_engine_kwargs_for_issue_1219(engine))
    compound_ids = sorted(row["id"] for row in _rows_for_issue_1219_engine(compound_result, engine))
    distributed_ids = sorted(row["id"] for row in _rows_for_issue_1219_engine(distributed_result, engine))

    assert compound_ids == expected
    assert distributed_ids == expected
    assert compound_ids == distributed_ids  # De Morgan equivalence


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_string_cypher_executes_xor_with_null_uses_three_valued_logic(engine: str | None) -> None:
    # XOR + IS NULL on the 3VL fixture.  IS NULL is deterministic
    # (NaN → TRUE, non-null → FALSE; no NULL output), so XOR's NULL
    # comes only from the comparison branch.
    #
    #   n1{x=1, y=10}:   x=1=T,    y IS NULL=F → T XOR F = T → keep
    #   n2{x=2, y=NaN}:  x=1=F,    y IS NULL=T → F XOR T = T → keep
    #   n3{x=NaN,y=20}:  x=1=NULL, y IS NULL=F → NULL XOR F = NULL → drop
    #   n4{x=NaN,y=NaN}: x=1=NULL, y IS NULL=T → NULL XOR T = NULL → drop
    graph = _three_valued_logic_fixture_graph()

    result = graph.gfql(
        "MATCH (n:N) WHERE n.x = 1 XOR n.y IS NULL RETURN n.id AS id",
        **_engine_kwargs_for_issue_1219(engine),
    )

    ids = sorted(row["id"] for row in _rows_for_issue_1219_engine(result, engine))
    assert ids == ["n1", "n2"]


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_string_cypher_executes_xor_returns_symmetric_difference(engine: str | None) -> None:
    # Sibling to the OR/AND/NOT runtime locks: XOR(A, B) ≡ (A AND NOT B) OR (NOT A AND B).
    # Locks pandas-backed evaluator returns the symmetric-difference row set
    # rather than treating XOR as OR (the boolean_expr_to_text and parse-tree
    # tests already cover structure; this is the runtime sibling).
    #
    #   n1{x=1, y=2}: x=1=T, y=2=T → T XOR T = F → drop
    #   n2{x=1, y=3}: x=1=T, y=2=F → T XOR F = T → keep
    #   n3{x=2, y=2}: x=1=F, y=2=T → F XOR T = T → keep
    #   n4{x=2, y=3}: x=1=F, y=2=F → F XOR F = F → drop
    graph = _de_morgan_fixture_graph()

    result = graph.gfql(
        "MATCH (n:N) WHERE n.x = 1 XOR n.y = 2 RETURN n.id AS id",
        **_engine_kwargs_for_issue_1219(engine),
    )

    ids = sorted(row["id"] for row in _rows_for_issue_1219_engine(result, engine))
    assert ids == ["n2", "n3"]


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_string_cypher_executes_double_negation_returns_original(engine: str | None) -> None:
    # NOT(NOT A) ≡ A.  Locks compound-NOT lowering doesn't drop one negation.
    graph = _de_morgan_fixture_graph()

    plain_result = graph.gfql("MATCH (n:N) WHERE n.x = 1 RETURN n.id AS id", **_engine_kwargs_for_issue_1219(engine))
    double_neg_result = graph.gfql(
        "MATCH (n:N) WHERE NOT NOT n.x = 1 RETURN n.id AS id",
        **_engine_kwargs_for_issue_1219(engine),
    )
    plain_ids = sorted(row["id"] for row in _rows_for_issue_1219_engine(plain_result, engine))
    double_neg_ids = sorted(row["id"] for row in _rows_for_issue_1219_engine(double_neg_result, engine))

    assert plain_ids == ["n1", "n2"]
    assert double_neg_ids == plain_ids


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_string_cypher_executes_mixed_string_numeric_and_inside_or(engine: str | None) -> None:
    # `WHERE (n.s > 'a' AND n.x > 0) OR n.x < -1` — exercises the
    # `_StringAllowingComparisonMixin` (#1217: extended GT/LT/GE/LE/NE
    # to strings) paired with OR composition.  The string GT branch
    # `n.s > 'a'` is the mixin-specific path; plain EQ on strings was
    # already supported pre-#1217 and would not exercise the mixin.
    # Truth table over the 5-row fixture:
    #   n1{s='b', x=5}:  ('b'>'a' AND 5>0)=T;  T OR (5<-1)=T → keep
    #   n2{s='b', x=-5}: ('b'>'a' AND -5>0)=F; F OR (-5<-1)=T → keep
    #   n3{s='a', x=5}:  ('a'>'a' AND 5>0)=F;  F OR (5<-1)=F → drop
    #   n4{s='a', x=-5}: ('a'>'a' AND -5>0)=F; F OR (-5<-1)=T → keep
    #   n5{s='a', x=0}:  ('a'>'a' AND 0>0)=F;  F OR (0<-1)=F → drop
    graph = _mk_graph(
        pd.DataFrame({
            "id":       ["n1", "n2", "n3", "n4", "n5"],
            "label__N": [True, True, True, True, True],
            "s":        ["b", "b", "a", "a", "a"],
            "x":        [5, -5, 5, -5, 0],
        }),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(
        "MATCH (n:N) WHERE (n.s > 'a' AND n.x > 0) OR n.x < -1 RETURN n.id AS id",
        **_engine_kwargs_for_issue_1219(engine),
    )

    ids = sorted(row["id"] for row in _rows_for_issue_1219_engine(result, engine))
    assert ids == ["n1", "n2", "n4"]


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
@pytest.mark.parametrize(
    "query,expected_rows",
    [
        # TCK-like disjunction baseline (match-where1-10 shape)
        (
            "MATCH (n) WHERE n.p1 = 12 OR n.p2 = 13 RETURN n.id AS id ORDER BY id",
            [{"id": "a"}, {"id": "b"}],
        ),
        # Nested boolean mix: OR group narrowed by AND
        (
            "MATCH (n) WHERE (n.p1 = 12 OR n.p2 = 13) AND n.id = 'a' RETURN n.id AS id ORDER BY id",
            [{"id": "a"}],
        ),
        # NOT over disjunction; current pandas-null semantics produce empty set
        (
            "MATCH (n) WHERE NOT (n.p1 = 12 OR n.p2 = 13) RETURN n.id AS id ORDER BY id",
            [],
        ),
    ],
    ids=[
        "tck-match-where1-10-disjunction",
        "audit-or-then-and-narrowing",
        "audit-negated-disjunction",
    ],
)
def test_issue_1219_row_boolean_audit_base_match_matrix(
    query: str,
    expected_rows: List[Dict[str, Any]],
    engine: str | None,
) -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "p1": [12.0, float("nan"), float("nan")],
                "p2": [float("nan"), 13.0, float("nan")],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql(query, **_engine_kwargs_for_issue_1219(engine))
    assert _rows_for_issue_1219_engine(result, engine) == expected_rows


def _normalize_nullable_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        out: Dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                out[key] = None
            else:
                # Normalize all scalar null sentinels (np.nan/pd.NA/cudf nulls)
                # so assertions remain stable across dataframe backends.
                out[key] = None if pd.isna(value) else value
        normalized.append(out)
    return normalized


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
@pytest.mark.parametrize(
    "query,expected_rows",
    [
        (
            "MATCH (x)-[r1:R1]->(y) "
            "OPTIONAL MATCH (y)-[r2:T]->(z) "
            "WHERE z:Z "
            "RETURN y.id AS yid, z.id AS zid "
            "ORDER BY yid",
            [{"yid": "y1", "zid": "z1"}, {"yid": "y2", "zid": None}],
        ),
        (
            "MATCH (x)-[r1:R1]->(y) "
            "OPTIONAL MATCH (y)-[r2:T]->(z) "
            "WHERE z.id IS NULL "
            "RETURN y.id AS yid, z.id AS zid "
            "ORDER BY yid",
            [{"yid": "y1", "zid": None}, {"yid": "y2", "zid": None}],
        ),
        (
            "MATCH (x)-[r1:R1]->(y) "
            "OPTIONAL MATCH (y)-[r2:T]->(z) "
            "WHERE z.id IS NOT NULL "
            "RETURN y.id AS yid, z.id AS zid "
            "ORDER BY yid",
            [{"yid": "y1", "zid": "z1"}, {"yid": "y2", "zid": None}],
        ),
    ],
    ids=[
        "tck-match-where6-structured-label-on-optional",
        "audit-optional-is-null",
        "audit-optional-is-not-null",
    ],
)
def test_issue_1219_row_boolean_audit_connected_optional_structured_matrix(
    query: str,
    expected_rows: List[Dict[str, Any]],
    engine: str | None,
) -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["s1", "y1", "z1", "s2", "y2"],
                "label__Z": [False, False, True, False, False],
            }
        ),
        pd.DataFrame(
            {
                "s": ["s1", "y1", "s2"],
                "d": ["y1", "z1", "y2"],
                "type": ["R1", "T", "R1"],
            }
        ),
    )

    rows = _rows_for_issue_1219_engine(
        graph.gfql(query, **_engine_kwargs_for_issue_1219(engine)),
        engine,
    )
    assert _normalize_nullable_rows(rows) == expected_rows


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
@pytest.mark.parametrize(
    "query,expected_rows",
    [
        (
            "MATCH (x)-[r1:R1]->(y) "
            "OPTIONAL MATCH (y)-[r2:T]->(z) "
            "WHERE z:Z OR z.id IS NULL "
            "RETURN y.id AS yid, z.id AS zid "
            "ORDER BY yid",
            [{"yid": "y1", "zid": "z1"}, {"yid": "y2", "zid": None}],
        ),
        (
            "MATCH (x)-[r1:R1]->(y) "
            "OPTIONAL MATCH (y)-[r2:T]->(z) "
            "WHERE NOT z:Z "
            "RETURN y.id AS yid, z.id AS zid "
            "ORDER BY yid",
            [{"yid": "y1", "zid": None}, {"yid": "y2", "zid": None}],
        ),
        (
            "MATCH (x)-[r1:R1]->(y) "
            "OPTIONAL MATCH (y)-[r2:T]->(z) "
            "WHERE z:Z XOR z.id IS NULL "
            "RETURN y.id AS yid, z.id AS zid "
            "ORDER BY yid",
            [{"yid": "y1", "zid": "z1"}, {"yid": "y2", "zid": None}],
        ),
        (
            "MATCH (x)-[r1:R1]->(y) "
            "WHERE y.id = 'y1' OR y.id = 'y2' "
            "OPTIONAL MATCH (y)-[r2:T]->(z) "
            "RETURN y.id AS yid, z.id AS zid "
            "ORDER BY yid",
            [{"yid": "y1", "zid": "z1"}, {"yid": "y2", "zid": None}],
        ),
    ],
    ids=[
        "audit-optional-or-row-expr-supported",
        "audit-optional-not-row-expr-supported",
        "audit-optional-xor-row-expr-supported",
        "audit-base-or-row-expr-supported-with-optional-arm",
    ],
)
def test_issue_1219_row_boolean_audit_connected_optional_row_expr_matrix(
    query: str,
    expected_rows: List[Dict[str, Any]],
    engine: str | None,
) -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["s1", "y1", "z1", "s2", "y2"],
                "label__Z": [False, False, True, False, False],
            }
        ),
        pd.DataFrame(
            {
                "s": ["s1", "y1", "s2"],
                "d": ["y1", "z1", "y2"],
                "type": ["R1", "T", "R1"],
            }
        ),
    )

    rows = _rows_for_issue_1219_engine(
        graph.gfql(query, **_engine_kwargs_for_issue_1219(engine)),
        engine,
    )
    assert _normalize_nullable_rows(rows) == expected_rows


def test_string_cypher_supports_list_slice_precedence_with_concat() -> None:
    graph = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = graph.gfql(
        "RETURN [[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10][1..3] AS a, "
        "[[1], [2, 3], [4, 5]] + ([5, [6, 7], [8, 9], 10][1..3]) AS b, "
        "([[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10])[1..3] AS c"
    )

    assert result._nodes.to_dict(orient="records") == [
        {
            "a": [[1], [2, 3], [4, 5], [6, 7], [8, 9]],
            "b": [[1], [2, 3], [4, 5], [6, 7], [8, 9]],
            "c": [[2, 3], [4, 5]],
        }
    ]


def test_string_cypher_supports_constant_limit_expressions() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["n1", "n2", "n3"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n)\nWITH n LIMIT toInteger(ceil(1.7))\nRETURN count(*) AS count")

    assert result._nodes.to_dict(orient="records") == [{"count": 2}]


def test_string_cypher_supports_integer_division_in_limit_expression() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["n1", "n2", "n3", "n4", "n5"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n) WITH n ORDER BY n.id LIMIT 7 / 3 RETURN count(*) AS count")

    assert result._nodes.to_dict(orient="records") == [{"count": 2}]


def test_string_cypher_uses_integer_division_for_literal_expression() -> None:
    graph = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = graph.gfql("RETURN 12 / 4 * 3 - 2 * 4 AS v")
    rows = result._nodes.to_dict(orient="records")
    assert rows == [{"v": 1}]
    assert not isinstance(rows[0]["v"], float)


@pytest.mark.parametrize(
    "query,params,expected",
    [
        ("RETURN $elt IN $coll AS result", {"elt": 2, "coll": [1, 2, 3]}, True),
        ("RETURN $elt IN $coll AS result", {"elt": 4, "coll": [1, 2, 3]}, False),
        ("RETURN $elt IN [1, 2, 3] AS result", {"elt": 2}, True),
        ("RETURN 2 IN $coll AS result", {"coll": [1, 2, 3]}, True),
    ],
)
def test_string_cypher_supports_parameterized_row_expressions(
    query: str,
    params: dict[str, object],
    expected: object,
) -> None:
    result = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []})).gfql(query, params=params)

    assert result._nodes.to_dict(orient="records") == [{"result": expected}]


def test_string_cypher_supports_parameterized_row_expr_null_propagation() -> None:
    result = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []})).gfql(
        "RETURN $elt IN $coll AS result",
        params={"elt": 4, "coll": [1, None, 3]},
    )

    assert pd.isna(result._nodes.iloc[0]["result"])


def test_string_cypher_supports_static_row_expr_null_propagation_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    graph = _mk_graph(
        cudf.from_pandas(pd.DataFrame({"id": pd.Series(dtype="object")})),
        cudf.from_pandas(pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")})),
    )

    result = graph.gfql("RETURN 4 IN [1, null, 3] AS result", engine="cudf")

    assert pd.isna(_to_pandas_df(result._nodes).iloc[0]["result"])


def test_string_cypher_supports_list_append_precedence_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    graph = _mk_graph(
        cudf.from_pandas(pd.DataFrame({"id": pd.Series(dtype="object")})),
        cudf.from_pandas(pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")})),
    )

    result = graph.gfql(
        "RETURN [1]+2 IN [3]+4 AS a, ([1]+2) IN ([3]+4) AS b, [1]+(2 IN [3])+4 AS c",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"a": False, "b": False, "c": [1, False, 4]}
    ]


def test_string_cypher_supports_list_membership_append_precedence_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    graph = _mk_graph(
        cudf.from_pandas(pd.DataFrame({"id": pd.Series(dtype="object")})),
        cudf.from_pandas(pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")})),
    )

    result = graph.gfql(
        "RETURN [1]+[2] IN [3]+[4] AS a, ([1]+[2]) IN ([3]+[4]) AS b, (([1]+[2]) IN [3])+[4] AS c, [1]+([2] IN [3])+[4] AS d",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"a": False, "b": False, "c": [False, 4], "d": [1, False, 4]}
    ]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("WITH null AS expr, 'x' AS idx RETURN expr[idx] AS value", [{"value": None}]),
        ("WITH {name: 'Mats'} AS expr, null AS idx RETURN expr[idx] AS value", [{"value": None}]),
        ("WITH {name: 'Mats', Name: 'Pontus'} AS map RETURN map['nAMe'] AS result", [{"result": None}]),
        ("WITH {name: 'Mats', nome: 'Pontus'} AS map RETURN map['null'] AS result", [{"result": None}]),
        ("WITH {null: 'Mats', NULL: 'Pontus'} AS map RETURN map['null'] AS result", [{"result": "Mats"}]),
        ("WITH {null: 'Mats', NULL: 'Pontus'} AS map RETURN map['NULL'] AS result", [{"result": "Pontus"}]),
    ],
)
def test_string_cypher_supports_dynamic_map_subscripts(query: str, expected: list[dict[str, object]]) -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(query)

    assert result._nodes.where(~result._nodes.isna(), None).to_dict(orient="records") == expected


def test_string_cypher_supports_list_subscript_with_integer_index() -> None:
    _assert_query_rows(
        "WITH [10, 20, 30] AS list, 1 AS idx RETURN list[idx] AS value",
        [{"value": 20}],
    )


def test_string_cypher_rejects_string_subscript_with_integer_index() -> None:
    g = _mk_empty_graph()

    with pytest.raises(Exception, match="dynamic subscript requires list-like base"):
        g.gfql("WITH '1' AS list, 0 AS idx RETURN list[idx] AS value")


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "UNWIND [date({year: 1910, month: 5, day: 6}), date({year: 1980, month: 10, day: 24})] AS dates "
            "WITH dates ORDER BY dates ASC LIMIT 2 RETURN dates",
            [{"dates": "1910-05-06"}, {"dates": "1980-10-24"}],
        ),
        ("RETURN date({year: 1817, week: 1, dayOfWeek: 2}) AS d", [{"d": "1816-12-31"}]),
        ("RETURN localtime('214032.142') AS result", [{"result": "21:40:32.142"}]),
        (
            "RETURN datetime('2015-07-21T21:40:32.142[Europe/London]') AS result",
            [{"result": "2015-07-21T21:40:32.142+01:00[Europe/London]"}],
        ),
        (
            "RETURN datetime('1818-07-21T21:40:32.142[Europe/Stockholm]') AS result",
            [{"result": "1818-07-21T21:40:32.142+00:53:28[Europe/Stockholm]"}],
        ),
        (
            "RETURN "
            "datetime('1818-07-21T21:40:32.142[Europe/Stockholm]') = "
            "datetime('1818-07-21T21:40:32.142+00:53:28[Europe/Stockholm]') AS b",
            [{"b": True}],
        ),
        (
            "RETURN time({hour: 12, minute: 34, second: 56, timezone: '+02:05:00'}) AS result",
            [{"result": "12:34:56+02:05"}],
        ),
        (
            "RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31}) AS result",
            [{"result": "1984-10-11T12:31Z"}],
        ),
        (
            "RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS result",
            [{"result": "1984-10-11T12:00+01:00"}],
        ),
        (
            "RETURN "
            "localtime({hour: 12, minute: 31, second: 14, millisecond: 645, microsecond: 876, nanosecond: 123}) AS t, "
            "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, millisecond: 645, microsecond: 876, nanosecond: 123, timezone: '+01:00'}) AS dt",
            [{"t": "12:31:14.645876123", "dt": "1984-10-11T12:31:14.645876123+01:00"}],
        ),
        (
            "RETURN datetime.fromepoch(416779, 999999999) AS d1, "
            "datetime.fromepochmillis(237821673987) AS d2",
            [{"d1": "1970-01-05T19:46:19.999999999Z", "d2": "1977-07-15T13:34:33.987Z"}],
        ),
        ("RETURN time({hour: 12, minute: 31, second: 14}) AS result", [{"result": "12:31:14Z"}]),
        (
            "RETURN date({date: date('1816-12-31'), year: 1817, week: 2}) AS d",
            [{"d": "1817-01-07"}],
        ),
        (
            "RETURN "
            "date({date: date('1816-12-30'), week: 2, dayOfWeek: 3}) AS d1, "
            "localdatetime({date: date('1816-12-31'), week: 2}) AS d2, "
            "datetime({date: date('1816-12-30'), week: 2, dayOfWeek: 3}) AS d3",
            [{"d1": "1817-01-08", "d2": "1817-01-07T00:00", "d3": "1817-01-08T00:00Z"}],
        ),
        (
            "WITH date({year: 1984, month: 11, day: 11}) AS other "
            "RETURN date({date: other, year: 28}) AS result",
            [{"result": "0028-11-11"}],
        ),
        (
            "WITH date({year: 1984, month: 10, day: 11}) AS other "
            "RETURN datetime({date: other, hour: 10, minute: 10, second: 10}) AS result",
            [{"result": "1984-10-11T10:10:10Z"}],
        ),
        (
            "WITH datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'}) AS other "
            "RETURN date(other) AS result",
            [{"result": "1984-11-11"}],
        ),
        (
            "WITH time({hour: 12, minute: 31, second: 14, nanosecond: 645876, timezone: '+01:00'}) AS other "
            "RETURN localtime(other) AS result",
            [{"result": "12:31:14.000645876"}],
        ),
        (
            "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other "
            "RETURN localdatetime({datetime: other}) AS result",
            [{"result": "1984-10-11T12:00"}],
        ),
        (
            "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other "
            "RETURN "
            "time(other) AS cast_result, "
            "time({time: other}) AS ctor_result, "
            "time({time: other, timezone: '+05:00'}) AS converted_result, "
            "time({time: other, second: 42, timezone: '+05:00'}) AS converted_second_result",
            [{
                "cast_result": "12:00+01:00",
                "ctor_result": "12:00+01:00",
                "converted_result": "16:00+05:00",
                "converted_second_result": "16:00:42+05:00",
            }],
        ),
        (
            "WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other "
            "RETURN "
            "time({time: other, timezone: '+05:00'}) AS converted_result, "
            "time({time: other, second: 42, timezone: '+05:00'}) AS converted_second_result",
            [{"converted_result": "16:31:14.645876+05:00", "converted_second_result": "16:31:42.645876+05:00"}],
        ),
        (
            "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other "
            "RETURN "
            "datetime({year: 1984, month: 10, day: 11, time: other}) AS preserved_result, "
            "datetime({year: 1984, month: 10, day: 11, time: other, second: 42}) AS preserved_second_result, "
            "datetime({year: 1984, month: 10, day: 11, time: other, timezone: '+05:00'}) AS converted_result, "
            "datetime({year: 1984, month: 10, day: 11, time: other, second: 42, timezone: 'Pacific/Honolulu'}) AS converted_named_result",
            [{
                "preserved_result": "1984-10-11T12:00+01:00[Europe/Stockholm]",
                "preserved_second_result": "1984-10-11T12:00:42+01:00[Europe/Stockholm]",
                "converted_result": "1984-10-11T16:00+05:00",
                "converted_named_result": "1984-10-11T01:00:42-10:00[Pacific/Honolulu]",
            }],
        ),
        (
            "WITH "
            "localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, "
            "datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime "
            "RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
            [{"result": "1984-03-28T12:00:42+02:00[Europe/Stockholm]"}],
        ),
        (
            "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other "
            "RETURN "
            "datetime({datetime: other}) AS preserved_result, "
            "datetime({datetime: other, timezone: '+05:00'}) AS converted_result, "
            "datetime({datetime: other, day: 28, second: 42}) AS recomputed_result",
            [{
                "preserved_result": "1984-10-11T12:00+01:00[Europe/Stockholm]",
                "converted_result": "1984-10-11T16:00+05:00",
                "recomputed_result": "1984-10-28T12:00:42+01:00[Europe/Stockholm]",
            }],
        ),
    ],
)
def test_string_cypher_temporal_constructor_and_cast_cases(
    query: str,
    expected: list[dict[str, object]],
) -> None:
    _assert_query_rows(query, expected)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "RETURN date.truncate('decade', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
            [{"result": "1980-01-02"}],
        ),
        (
            "RETURN date.truncate("
            "'decade', "
            "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), "
            "{day: 2}"
            ") AS result",
            [{"result": "1980-01-02"}],
        ),
        (
            "RETURN datetime.truncate("
            "'weekYear', "
            "localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
            "{timezone: 'Europe/Stockholm'}"
            ") AS result",
            [{"result": "1983-01-03T00:00+01:00[Europe/Stockholm]"}],
        ),
    ],
)
def test_string_cypher_executes_temporal_truncation_cases(
    query: str,
    expected: list[dict[str, object]],
) -> None:
    _assert_query_rows(query, expected)


@pytest.mark.parametrize(
    ("query", "expected_result"),
    [
        ("RETURN duration('P5M1.5D') AS result", "P5M1DT12H"),
        ("RETURN duration('P0.75M') AS result", "P22DT19H51M49.5S"),
        ("RETURN duration('PT0.75M') AS result", "PT45S"),
        ("RETURN duration('P2.5W') AS result", "P17DT12H"),
        ("RETURN duration('P12Y5M14DT16H12M70S') AS result", "P12Y5M14DT16H13M10S"),
        ("RETURN duration('P2012-02-02T14:37:21.545') AS result", "P2012Y2M2DT14H37M21.545S"),
    ],
)
def test_string_cypher_executes_temporal_duration_string_canonicalization(
    query: str,
    expected_result: str,
) -> None:
    _assert_query_rows(query, [{"result": expected_result}])


@pytest.mark.parametrize(
    ("query", "expected_ts", "expected_b"),
    [
        # openCypher TCK Temporal6 [6] examples 2 and 8 (pygraphistry #1361 / #1353 item #2):
        # toString preserves the months/days/seconds-nanos components separately.
        # Negative days alongside positive hours stay distinct ('-14D + 16H', not collapsed).
        (
            "WITH duration({years: 12, months: 5, days: -14, hours: 16}) AS d "
            "RETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "P12Y5M-14DT16H",
            True,
        ),
        # 1 day plus a negative millisecond stays as 'P1DT-0.001S', not 'PT23H59M59.999S'.
        (
            "WITH duration({days: 1, milliseconds: -1}) AS d "
            "RETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "P1DT-0.001S",
            True,
        ),
    ],
)
def test_string_cypher_duration_tostring_preserves_components(
    query: str,
    expected_ts: str,
    expected_b: bool,
) -> None:
    _assert_query_rows(query, [{"ts": expected_ts, "b": expected_b}])


def test_string_cypher_duration_equality_is_component_wise() -> None:
    # openCypher TCK Temporal7 [6] example 8 (pygraphistry #1361 / #1353 item #2):
    # Two durations with equal total seconds but different (days, seconds) component
    # shapes are NOT equal. Equality compares months/days/seconds-nanos components,
    # not just totals.
    _assert_query_rows(
        "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, "
        "duration({years: 12, months: 5, days: 13, hours: 40, minutes: 13, seconds: 10}) AS d "
        "RETURN x = d AS eq",
        [{"eq": False}],
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "RETURN localdatetime.truncate("
            "'weekYear', "
            "localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
            "{day: 5}"
            ") AS result",
            [{"result": "1983-01-05T00:00"}],
        ),
        (
            "RETURN time.truncate("
            "'hour', "
            "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
            "{timezone: '+01:00'}"
            ") AS result",
            [{"result": "12:00+01:00"}],
        ),
        (
            "RETURN datetime.truncate("
            "'hour', "
            "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
            "{nanosecond: 2}"
            ") AS result",
            [{"result": "1984-10-11T12:00:00.000000002Z"}],
        ),
        (
            "RETURN "
            "datetime.truncate('millisecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS dt_ms, "
            "datetime.truncate('microsecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS dt_us, "
            "time.truncate('microsecond', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS t_us",
            [{
                "dt_ms": "1984-10-11T12:31:14.645000002+01:00",
                "dt_us": "1984-10-11T12:31:14.645876002Z",
                "t_us": "12:31:14.645876002+01:00",
            }],
        ),
    ],
)
def test_string_cypher_executes_temporal_truncation_override_cases(
    query: str,
    expected: list[dict[str, object]],
) -> None:
    _assert_query_rows(query, expected)


def test_string_cypher_executes_duration_between_with_alias_properties() -> None:
    _assert_query_rows(
        "WITH duration.between(localdatetime('2018-01-01T12:00'), localdatetime('2018-01-02T10:00')) AS dur "
        "RETURN dur, dur.days, dur.seconds, dur.nanosecondsOfSecond",
        [{"dur": "PT22H", "dur.days": 0, "dur.seconds": 79200, "dur.nanosecondsOfSecond": 0}],
    )


def test_string_cypher_folds_temporal_constructor_before_property_access() -> None:
    _assert_query_rows("RETURN duration({days: 1}).days AS days", [{"days": 1}])


def test_string_cypher_executes_negative_duration_between_day_time_components() -> None:
    _assert_query_rows(
        "RETURN duration.between("
        "localdatetime('2015-07-21T21:40:32.142'), "
        "date('2015-06-24')"
        ") AS duration",
        [{"duration": "P-27DT-21H-40M-32.142S"}],
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "RETURN duration.inMonths(date('1984-10-11'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
            "P31Y9M",
        ),
        (
            "RETURN duration.inDays(date('1984-10-11'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
            "P11606D",
        ),
        (
            "RETURN duration.inSeconds(date('1984-10-11'), localtime('16:30')) AS duration",
            "PT16H30M",
        ),
    ],
)
def test_string_cypher_executes_duration_unit_functions(query: str, expected: str) -> None:
    _assert_query_rows(query, [{"duration": expected}])


def test_string_cypher_executes_duration_in_seconds_with_dst_anchor_and_mixed_temporals() -> None:
    _assert_query_rows(
        "RETURN "
        "duration.inSeconds(datetime({year: 2017, month: 10, day: 29, hour: 0, timezone: 'Europe/Stockholm'}), localdatetime({year: 2017, month: 10, day: 29, hour: 4})) AS d1, "
        "duration.inSeconds(datetime({year: 2017, month: 10, day: 29, hour: 0, timezone: 'Europe/Stockholm'}), localtime({hour: 4})) AS d2, "
        "duration.inSeconds(date({year: 2017, month: 10, day: 29}), datetime({year: 2017, month: 10, day: 29, hour: 4, timezone: 'Europe/Stockholm'})) AS d3, "
        "duration.inSeconds(datetime({year: 2017, month: 10, day: 29, hour: 0, timezone: 'Europe/Stockholm'}), date({year: 2017, month: 10, day: 30})) AS d4",
        [
        {
            "d1": "PT5H",
            "d2": "PT5H",
            "d3": "PT5H",
            "d4": "PT25H",
        }
        ],
    )


def test_string_cypher_executes_extreme_year_duration_functions() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "RETURN "
        "duration.between(date('-999999999-01-01'), date('+999999999-12-31')) AS d1, "
        "duration.inSeconds(localdatetime('-999999999-01-01'), localdatetime('+999999999-12-31T23:59:59')) AS d2"
    )

    assert result._nodes.to_dict(orient="records") == [
        {
            "d1": "P1999999998Y11M30D",
            "d2": "PT17531639991215H59M59S",
        }
    ]


@pytest.mark.parametrize(
    ("query", "nodes_df", "edges_df"),
    [
        (
            "MATCH (a) WITH a, count(*) RETURN a",
            pd.DataFrame({"id": []}),
            pd.DataFrame({"s": [], "d": []}),
        ),
        (
            "WITH [1, 2, 3, 4, 5] AS list, true AS idx RETURN list[idx]",
            pd.DataFrame({"id": []}),
            pd.DataFrame({"s": [], "d": []}),
        ),
        (
            "MATCH (n) RETURN [x IN [1, 2, 3, 4, 5] | count(*)]",
            pd.DataFrame({"id": []}),
            pd.DataFrame({"s": [], "d": []}),
        ),
        (
            "RETURN 1 IN 'foo' AS res",
            pd.DataFrame({"id": []}),
            pd.DataFrame({"s": [], "d": []}),
        ),
        (
            "RETURN 1 IN {x: []} AS res",
            pd.DataFrame({"id": []}),
            pd.DataFrame({"s": [], "d": []}),
        ),
        (
            "RETURN 9223372036854775808 AS literal",
            pd.DataFrame({"id": []}),
            pd.DataFrame({"s": [], "d": []}),
        ),
        (
            "RETURN -9223372036854775809 AS literal",
            pd.DataFrame({"id": []}),
            pd.DataFrame({"s": [], "d": []}),
        ),
        (
            "RETURN 0x8000000000000000 AS literal",
            pd.DataFrame({"id": []}),
            pd.DataFrame({"s": [], "d": []}),
        ),
        (
            "RETURN -0x8000000000000001 AS literal",
            pd.DataFrame({"id": []}),
            pd.DataFrame({"s": [], "d": []}),
        ),
        (
            "RETURN 0o1000000000000000000000 AS literal",
            pd.DataFrame({"id": []}),
            pd.DataFrame({"s": [], "d": []}),
        ),
        (
            "RETURN -0o1000000000000000000001 AS literal",
            pd.DataFrame({"id": []}),
            pd.DataFrame({"s": [], "d": []}),
        ),
        (
            "MATCH p = (n)-[r:T]->() RETURN [x IN [1.0, true] | toFloat(x) ] AS list",
            pd.DataFrame({"id": ["n1", "n2"]}),
            pd.DataFrame({"s": ["n1"], "d": ["n2"], "type": ["T"]}),
        ),
        (
            "MATCH p = (n)-[r:T]->() RETURN [x IN [1, '', r] | toString(x) ] AS list",
            pd.DataFrame({"id": ["n1", "n2"]}),
            pd.DataFrame({"s": ["n1"], "d": ["n2"], "type": ["T"]}),
        ),
        (
            "MATCH p = (n)-[r:T]->() RETURN [x IN [1, '', []] | toString(x) ] AS list",
            pd.DataFrame({"id": ["n1", "n2"]}),
            pd.DataFrame({"s": ["n1"], "d": ["n2"], "type": ["T"]}),
        ),
        (
            "MATCH p = (n)-[r:T]->() RETURN [x IN [1, '', {}] | toString(x) ] AS list",
            pd.DataFrame({"id": ["n1", "n2"]}),
            pd.DataFrame({"s": ["n1"], "d": ["n2"], "type": ["T"]}),
        ),
    ],
)
def test_string_cypher_failfast_rejects_invalid_supported_overlap_queries(
    query: str,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> None:
    g = _mk_graph(nodes_df, edges_df)

    with pytest.raises(GFQLValidationError):
        g.gfql(query)


@pytest.mark.parametrize(
    ("query", "missing_name"),
    [
        (
            "MATCH (n)\nRETURN n\nORDER BY n.name ASC\nSKIP $skipAmount",
            "skipAmount",
        ),
        (
            "MATCH (n)\nRETURN n\nORDER BY n.name ASC\nSKIP $s\nLIMIT $l",
            "s",
        ),
        (
            "MATCH (n)\nWITH n\nORDER BY n.name ASC\nSKIP $s\nLIMIT $l\nRETURN n",
            "s",
        ),
        (
            "RETURN $elt IN $coll AS result",
            "elt",
        ),
    ],
)
def test_string_cypher_failfast_rejects_missing_overlap_parameters(
    query: str,
    missing_name: str,
) -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql(query)

    assert exc_info.value.code == ErrorCode.E105
    assert missing_name in str(exc_info.value)


@pytest.mark.parametrize(
    "query",
    [
        (
            "WITH [1, null, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true, c: 15.2}, {}, [], [null], [[{b: [null]}]]] AS inputList\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "WITH list WHERE size(list) > 0\n"
            "WITH none(x IN list WHERE false) AS result, count(*) AS cnt\n"
            "RETURN result"
        ),
        (
            "WITH [1, null, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true, c: 15.2}, {}, [], [null], [[{b: [null]}]]] AS inputList\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "WITH list WHERE size(list) > 0\n"
            "WITH single(x IN list WHERE false) AS result, count(*) AS cnt\n"
            "RETURN result"
        ),
        (
            "WITH [1, null, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true, c: 15.2}, {}, [], [null], [[{b: [null]}]]] AS inputList\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "WITH list WHERE size(list) > 0\n"
            "WITH any(x IN list WHERE false) AS result, count(*) AS cnt\n"
            "RETURN result"
        ),
        (
            "WITH [1, null, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true, c: 15.2}, {}, [], [null], [[{b: [null]}]]] AS inputList\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "UNWIND inputList AS x\n"
            "WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
            "WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
            "WITH list WHERE size(list) > 0\n"
            "WITH all(x IN list WHERE false) AS result, count(*) AS cnt\n"
            "RETURN result"
        ),
    ],
)
def test_string_cypher_failfast_rejects_rand_quantifier_overlap_queries(query: str) -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLValidationError, match="currently supported local GFQL subset"):
        g.gfql(query)


def test_string_cypher_rejects_placeholder_quantifier_overlap_query_as_syntax() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))
    query = (
        "UNWIND [{list: [2], fixed: true}, {list: [6], fixed: true}, {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n"
        "WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n"
        "     CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n"
        "UNWIND inputList AS x\n"
        "WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
        "WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
        "UNWIND inputList AS x\n"
        "WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
        "WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
        "UNWIND inputList AS x\n"
        "WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n"
        "WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n"
        "WITH coalesce(fixedList, list) AS list\n"
        "WITH list WHERE single(<operands>) OR all(x IN list WHERE x < 7)\n"
        "WITH any(x IN list WHERE x < 7) AS result, count(*) AS cnt\n"
        "RETURN result"
    )

    with pytest.raises(GFQLSyntaxError):
        g.gfql(query)


@pytest.mark.parametrize(
    "query, expected_message, expect_issue_ref",
    [
        (
            "MATCH (a {name: 'Andres'})<-[:FATHER]-(child)\nRETURN a.name, {foo: a.name='Andres', kids: collect(child.name)}",
            "aggregate expressions inside map literals",
            False,
        ),
        (
            "MATCH (me: Person)--(you: Person)\nWITH me.age AS age, you\nRETURN age, age + count(you.age)",
            "one MATCH source alias at a time",
            True,
        ),
        (
            "MATCH (me: Person)--(you: Person)\nRETURN me.age, me.age + count(you.age)",
            "one MATCH source alias at a time",
            True,
        ),
        (
            "MATCH (me: Person)--(you: Person)\nRETURN me.age AS age, count(you.age) AS cnt\nORDER BY age, age + count(you.age)",
            "one MATCH source alias at a time",
            True,
        ),
    ],
)
def test_string_cypher_rejects_unsound_multi_source_aggregate_overlap_queries(
    query: str,
    expected_message: str,
    expect_issue_ref: bool,
) -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLValidationError, match=expected_message) as exc_info:
        g.gfql(query)
    if expect_issue_ref:
        assert "#1273" in exc_info.value.message
    else:
        assert "#1273" not in exc_info.value.message


@pytest.mark.parametrize(
    "query",
    [
        (
            "MATCH (david {name: 'David'})--(otherPerson)-->() "
            "WITH otherPerson, count(*) AS foaf "
            "WHERE foaf > 1 "
            "WITH otherPerson "
            "WHERE otherPerson.name <> 'NotOther' "
            "RETURN count(*)"
        ),
        "MATCH (a)-->() WITH a, count(*) AS relCount WHERE relCount > 1 RETURN a",
    ],
)
def test_string_cypher_failfast_rejects_with_stage_unsound_relationship_multiplicity_aggregates(query: str) -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "anon_5", "anon_6", "anon_7", "anon_8", "anon_9"],
                "name": ["David", "Other", "NotOther", "NotOther2", None, None, None, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["a", "a", "a", "b", "b", "c", "c", "d"],
                "d": ["b", "c", "d", "anon_5", "anon_6", "anon_7", "anon_8", "anon_9"],
                "type": ["REL"] * 8,
            }
        ),
    )

    with pytest.raises(GFQLValidationError, match="repeated MATCH rows"):
        graph.gfql(query)


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n:Single) OPTIONAL MATCH (n)-[r]-(m) WHERE m:NonExistent RETURN r",
        "MATCH (a:A), (c:C) OPTIONAL MATCH (a)-->(b)-->(c) RETURN b",
    ],
)
def test_string_cypher_failfast_rejects_optional_match_null_extension_shapes_without_safe_alignment(query: str) -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["s", "a", "b", "c"],
                "label__Single": [True, False, False, False],
                "label__A": [False, True, False, False],
                "label__B": [False, False, True, False],
                "label__C": [False, False, False, True],
                "num": [None, 42, 46, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["s", "s", "a", "b"],
                "d": ["a", "b", "c", "b"],
                "type": ["REL", "REL", "REL", "LOOP"],
            }
        ),
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(query)

    assert exc_info.value.code == ErrorCode.E108


def test_string_cypher_failfast_rejects_graph_backed_unwind_after_with_as_validation_error() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["s", "n", "e"],
                "label__S": [True, False, False],
                "label__E": [False, False, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["s", "s", "n"],
                "d": ["e", "e", "e"],
                "type": ["X", "Y", "Y"],
            }
        ),
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql("MATCH (a:S)-[:X]->(b1) WITH a, collect(b1) AS bees UNWIND bees AS b2 MATCH (a)-[:Y]->(b2) RETURN a, b2")

    assert exc_info.value.code == ErrorCode.E108
    assert "UNWIND after WITH/RETURN" in exc_info.value.message


def test_string_cypher_executes_graph_backed_unwind_after_with_into_post_with_match() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["s", "b1", "b2", "c1", "c2"],
                "label__S": [True, False, False, False, False],
                "label__B": [False, True, True, False, False],
                "label__C": [False, False, False, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["s", "s", "b1", "b2"],
                "d": ["b1", "b2", "c1", "c2"],
                "type": ["X", "X", "Y", "Y"],
            }
        ),
    )

    _assert_query_rows(
        "MATCH (root:S)-[:X]->(b1:B) "
        "WITH collect(b1) AS bees "
        "UNWIND bees AS b2 "
        "MATCH (b2)-[:Y]->(c:C) "
        "RETURN c.id AS id "
        "ORDER BY id",
        [{"id": "c1"}, {"id": "c2"}],
        nodes_df=graph._nodes,
        edges_df=graph._edges,
    )


def test_string_cypher_executes_graph_backed_distinct_unwind_after_with_into_post_with_match() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["s", "b1", "b2", "c1", "c2"],
                "label__S": [True, False, False, False, False],
                "label__B": [False, True, True, False, False],
                "label__C": [False, False, False, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["s", "s", "s", "b1", "b2"],
                "d": ["b1", "b1", "b2", "c1", "c2"],
                "type": ["X", "X", "X", "Y", "Y"],
            }
        ),
    )

    _assert_query_rows(
        "MATCH (root:S)-[:X]->(b1:B) "
        "WITH collect(DISTINCT b1) AS bees "
        "UNWIND bees AS b2 "
        "MATCH (b2)-[:Y]->(c:C) "
        "RETURN c.id AS id "
        "ORDER BY id",
        [{"id": "c1"}, {"id": "c2"}],
        nodes_df=graph._nodes,
        edges_df=graph._edges,
    )


def test_string_cypher_executes_graph_backed_unwind_with_carried_scalar_into_post_with_match() -> None:
    """IC-6 shape: WITH scalar, collect(alias) AS list UNWIND list AS alias MATCH ... RETURN (#1000)."""
    graph = _mk_collect_unwind_reentry_graph()
    _assert_query_rows(
        "MATCH (root:S)-[:X]->(b1:B) "
        "WITH root.val AS rv, collect(b1) AS bees "
        "UNWIND bees AS b2 "
        "MATCH (b2)-[:Y]->(c:C) "
        "RETURN rv, c.id AS cid "
        "ORDER BY cid",
        [{"rv": 0, "cid": "c1"}, {"rv": 0, "cid": "c2"}],
        nodes_df=graph._nodes,
        edges_df=graph._edges,
    )


def test_string_cypher_executes_graph_backed_unwind_with_multiple_carried_scalars() -> None:
    """Multiple carried scalars alongside collect (#1000)."""
    graph = _mk_collect_unwind_reentry_graph()
    _assert_query_rows(
        "MATCH (root:S)-[:X]->(b1:B) "
        "WITH root.id AS rid, root.val AS rv, collect(b1) AS bees "
        "UNWIND bees AS b2 "
        "MATCH (b2)-[:Y]->(c:C) "
        "RETURN rid, rv, c.id AS cid "
        "ORDER BY cid",
        [{"rid": "s", "rv": 0, "cid": "c1"}, {"rid": "s", "rv": 0, "cid": "c2"}],
        nodes_df=graph._nodes,
        edges_df=graph._edges,
    )


def test_string_cypher_executes_graph_backed_distinct_unwind_with_carried_scalar() -> None:
    """DISTINCT collect with carried scalar (#1000).

    Uses a graph with duplicate s->b1 edges to exercise the DISTINCT path.
    """
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["s", "b1", "b2", "c1", "c2"],
                "label__S": [True, False, False, False, False],
                "label__B": [False, True, True, False, False],
                "label__C": [False, False, False, True, True],
                "val": [0, 10, 20, 100, 200],
            }
        ),
        pd.DataFrame(
            {
                "s": ["s", "s", "s", "b1", "b2"],
                "d": ["b1", "b1", "b2", "c1", "c2"],
                "type": ["X", "X", "X", "Y", "Y"],
            }
        ),
    )
    _assert_query_rows(
        "MATCH (root:S)-[:X]->(b1:B) "
        "WITH root.val AS rv, collect(DISTINCT b1) AS bees "
        "UNWIND bees AS b2 "
        "MATCH (b2)-[:Y]->(c:C) "
        "RETURN rv, c.id AS cid "
        "ORDER BY cid",
        [{"rv": 0, "cid": "c1"}, {"rv": 0, "cid": "c2"}],
        nodes_df=graph._nodes,
        edges_df=graph._edges,
    )


def test_string_cypher_executes_graph_backed_unwind_after_with_into_post_with_match_on_cudf() -> None:
    graph = _mk_cudf_graph(
        pd.DataFrame(
            {
                "id": ["s", "b1", "b2", "c1", "c2"],
                "label__S": [True, False, False, False, False],
                "label__B": [False, True, True, False, False],
                "label__C": [False, False, False, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["s", "s", "b1", "b2"],
                "d": ["b1", "b2", "c1", "c2"],
                "type": ["X", "X", "Y", "Y"],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (root:S)-[:X]->(b1:B) "
        "WITH collect(b1) AS bees "
        "UNWIND bees AS b2 "
        "MATCH (b2)-[:Y]->(c:C) "
        "RETURN c.id AS id "
        "ORDER BY id",
        engine="cudf",
    )

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"id": "c1"},
        {"id": "c2"},
    ]


def test_string_cypher_with_unwind_reentry_progresses_past_parser_to_row_scope_boundary() -> None:
    query = (
        "MATCH (root:S)-[:X]->(friend:B) "
        "WITH collect(friend) AS friends "
        "UNWIND friends AS friend "
        "MATCH (friend)-[:Y]->(foaf:C) "
        "WHERE NOT friend = root "
        "RETURN foaf"
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        compile_cypher(query)

    assert exc_info.value.code == ErrorCode.E108
    assert "one MATCH source alias at a time" in exc_info.value.message
    assert "#1273" in exc_info.value.message


def test_string_cypher_rejects_with_unwind_reentry_when_unwind_source_is_not_collected_alias() -> None:
    query = (
        "MATCH (root:S)-[:X]->(b1:B) "
        "WITH collect(b1) AS bees "
        "UNWIND other_bees AS b2 "
        "MATCH (b2)-[:Y]->(c:C) "
        "RETURN c"
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        compile_cypher(query)

    assert exc_info.value.code == ErrorCode.E204
    assert exc_info.value.context.get("field") == "identifier"
    assert exc_info.value.context.get("value") == "other_bees"


def test_string_cypher_executes_exact_multihop_relationship_pattern() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d", "e", "f"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c", "d", "e"],
                "d": ["b", "c", "d", "e", "f"],
                "type": ["R", "R", "R", "R", "R"],
            }
        ),
    )

    result = graph.gfql("MATCH (a {id: 'a'})-[*2]->(b) RETURN b.id AS id ORDER BY id")

    assert result._nodes.to_dict(orient="records") == [{"id": "c"}]


def test_string_cypher_executes_bounded_multihop_relationship_pattern() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d", "e"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c", "a"],
                "d": ["b", "c", "d", "e"],
                "type": ["R", "R", "R", "S"],
            }
        ),
    )

    result = graph.gfql("MATCH (a {id: 'a'})-[:R*1..3]->(b) RETURN b.id AS id ORDER BY id")

    assert result._nodes.to_dict(orient="records") == [{"id": "b"}, {"id": "c"}, {"id": "d"}]


def test_string_cypher_executes_fixed_point_relationship_pattern() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d", "e"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c", "a"],
                "d": ["b", "c", "d", "e"],
                "type": ["R", "R", "R", "S"],
            }
        ),
    )

    result = graph.gfql("MATCH (a {id: 'a'})-[*]->(b) RETURN b.id AS id ORDER BY id")

    assert result._nodes.to_dict(orient="records") == [{"id": "b"}, {"id": "c"}, {"id": "d"}, {"id": "e"}]


def test_string_cypher_executes_typed_reverse_fixed_point_relationship_pattern() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c"],
                "d": ["b", "c", "d"],
                "type": ["R", "R", "R"],
            }
        ),
    )

    result = graph.gfql("MATCH (a {id: 'd'})<-[:R*]-(b) RETURN b.id AS id ORDER BY id")

    assert result._nodes.to_dict(orient="records") == [{"id": "a"}, {"id": "b"}, {"id": "c"}]


def test_string_cypher_executes_reverse_multihop_relationship_pattern() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c"],
                "d": ["b", "c", "d"],
                "type": ["R", "R", "R"],
            }
        ),
    )

    result = graph.gfql("MATCH (a {id: 'c'})<-[*2]-(b) RETURN b.id AS id ORDER BY id")

    assert result._nodes.to_dict(orient="records") == [{"id": "a"}]


def test_string_cypher_executes_undirected_multihop_relationship_pattern() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c"],
                "d": ["b", "c", "d"],
                "type": ["R", "R", "R"],
            }
        ),
    )

    result = graph.gfql("MATCH (a {id: 'a'})-[:R*1..2]-(b) RETURN b.id AS id ORDER BY id")

    assert result._nodes.to_dict(orient="records") == [{"id": "b"}, {"id": "c"}]


def test_string_cypher_executes_undirected_fixed_point_relationship_pattern_without_zero_hop_backtracking() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c"],
                "d": ["b", "c", "d"],
                "type": ["R", "R", "R"],
            }
        ),
    )

    result = graph.gfql("MATCH (a {id: 'a'})-[:R*]-(b) RETURN b.id AS id ORDER BY id")

    assert result._nodes.to_dict(orient="records") == [{"id": "b"}, {"id": "c"}, {"id": "d"}]


def test_string_cypher_executes_undirected_fixed_point_relationship_pattern_on_cycle_includes_seed() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c"],
                "d": ["b", "c", "a"],
                "type": ["R", "R", "R"],
            }
        ),
    )

    result = graph.gfql("MATCH (a {id: 'a'})-[:R*]-(b) RETURN b.id AS id ORDER BY id")

    assert result._nodes.to_dict(orient="records") == [{"id": "a"}, {"id": "b"}, {"id": "c"}]


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a)-[r*1..1]->(b) RETURN r",
        "MATCH (a:A) MATCH (a)-[r*2]->() RETURN r",
        "MATCH (a)-[r:REL*2..2]->(b:End) RETURN r",
        "MATCH (a)-[r:REL*2..2]-(b:End) RETURN r",
        "MATCH (a:Start)-[r:REL*2..2]-(b) RETURN r",
        "MATCH (a:Blue)-[r*]->(b:Green) RETURN count(r)",
        "MATCH (a)-[r*]->(b) RETURN b.id AS id ORDER BY r",
        "MATCH (a)-[r*]->(b) WITH b, r WHERE r IS NOT NULL RETURN b.id AS id",
        "MATCH (a)-[r*]->(b) WITH b, r ORDER BY r RETURN b.id AS id",
        "MATCH (a)-[r*]->(b) WITH b, type(r) AS t RETURN b.id AS id",
    ],
)
def test_string_cypher_failfast_rejects_variable_length_relationship_alias_path_carrier_forms(
    query: str,
) -> None:
    graph = _mk_empty_graph()

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(query)

    assert exc_info.value.code == ErrorCode.E108
    assert "path/list carriers" in exc_info.value.message


@pytest.mark.parametrize(
    "query",
    [
        "MATCH p = (a)-[:R*2]->(b) RETURN p",
        "MATCH p = (a)-[:R*]->(b) RETURN p",
        "MATCH p = (a)-[:R*2]->(b) RETURN length(p) AS n",
        "MATCH p = (a)-[:R*]->(b) RETURN relationships(p) AS rels",
        "MATCH p = (a)-[:R*]->(b) WHERE p IS NOT NULL RETURN b",
        "MATCH p = (a)-[:R*]->(b) WITH p RETURN p",
        "MATCH p = (a)-[:R*]->(b) RETURN b.id AS id ORDER BY p",
    ],
)
def test_string_cypher_failfast_rejects_variable_length_named_path_alias_references(
    query: str,
) -> None:
    graph = _mk_empty_graph()

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(query)

    assert exc_info.value.code == ErrorCode.E108
    assert "named path aliases" in exc_info.value.message


def test_string_cypher_supports_unused_named_path_alias_for_endpoint_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame(
            {
                "s": ["a", "b"],
                "d": ["b", "c"],
                "type": ["R", "R"],
            }
        ),
    )

    result = graph.gfql("MATCH p = (a {id: 'a'})-[:R*2]->(b) RETURN b.id AS id")

    assert result._nodes.to_dict(orient="records") == [{"id": "c"}]


@pytest.mark.parametrize(
    "query,expected_names",
    [
        ("MATCH (a:A) MATCH (a)-[:LIKES*1]->()-[:LIKES]->(c) RETURN c.name AS name", ["C"]),
        ("MATCH (a:A) MATCH (a)-[:LIKES*2]->()-[:LIKES]->(c) RETURN c.name AS name", ["D"]),
        ("MATCH (a:A) MATCH (a)-[:LIKES]->()-[:LIKES*3]->(c) RETURN c.name AS name", []),
        ("MATCH (a:A) MATCH (a)<-[:LIKES]-()-[:LIKES*3]->(c) RETURN c.name AS name", []),
    ],
)
def test_string_cypher_accepts_nonterminal_variable_length_relationship_patterns(
    query: str,
    expected_names: List[str],
) -> None:
    """Connected reentry forms with varlen rels return expected rows."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"], "label__A": [True, False, False, False], "name": ["A", "B", "C", "D"]}),
        pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"], "type": ["LIKES", "LIKES", "LIKES"]}),
    )
    result = graph.gfql(query)
    got_names = _to_pandas_df(result._nodes)["name"].tolist()
    assert got_names == expected_names


def test_string_cypher_two_match_reentry_varlen_forward_matches_connected_shape() -> None:
    """Regression for #1001: split MATCH form should match connected form rows."""
    nodes = []
    edges = []
    for level in range(5):
        for idx in range(2 ** level):
            node_id = f"n{level}_{idx}"
            nodes.append({"id": node_id, "label__A": level == 0, "name": node_id})
            if level > 0:
                parent = f"n{level - 1}_{idx // 2}"
                edges.append({"s": parent, "d": node_id, "type": "LIKES"})

    graph = _mk_graph(pd.DataFrame(nodes), pd.DataFrame(edges))
    split_query = "MATCH (a:A) MATCH (a)-[:LIKES]->()-[:LIKES*3]->(c) RETURN c.name AS name ORDER BY name"
    connected_query = "MATCH (a:A)-[:LIKES]->()-[:LIKES*3]->(c) RETURN c.name AS name ORDER BY name"

    split_rows = graph.gfql(split_query)._nodes.to_dict(orient="records")
    connected_rows = graph.gfql(connected_query)._nodes.to_dict(orient="records")

    assert split_rows == connected_rows
    assert split_rows == [{"name": name} for name in sorted(f"n4_{i}" for i in range(16))]


def test_string_cypher_two_match_reentry_varlen_reverse_matches_connected_shape() -> None:
    """Regression for #1001 reverse direction shape from TCK-style query."""
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "x", "y1", "y2", "c1", "c2"],
                "label__A": [True, False, False, False, False, False],
                "name": ["a", "x", "y1", "y2", "c1", "c2"],
            }
        ),
        pd.DataFrame(
            {
                "s": ["x", "x", "y1", "y2", "y2"],
                "d": ["a", "y1", "y2", "c1", "c2"],
                "type": ["LIKES", "LIKES", "LIKES", "LIKES", "LIKES"],
            }
        ),
    )

    split_query = "MATCH (a:A) MATCH (a)<-[:LIKES]-()-[:LIKES*3]->(c) RETURN c.name AS name ORDER BY name"
    connected_query = "MATCH (a:A)<-[:LIKES]-()-[:LIKES*3]->(c) RETURN c.name AS name ORDER BY name"

    split_rows = graph.gfql(split_query)._nodes.to_dict(orient="records")
    connected_rows = graph.gfql(connected_query)._nodes.to_dict(orient="records")

    assert split_rows == connected_rows
    assert split_rows == [{"name": "c1"}, {"name": "c2"}]


def _mk_chain_graph():
    """a->b->c->d->e linear chain with LIKES edges."""
    return _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d", "e"], "label__S": [True, False, False, False, False]}),
        pd.DataFrame({"s": ["a", "b", "c", "d"], "d": ["b", "c", "d", "e"], "type": ["R", "R", "R", "R"]}),
    )


@pytest.mark.parametrize(
    "query,expected_ids",
    [
        ("MATCH (s:S)-[:R*1]->()-[:R]->(c) RETURN c.id AS id", ["c"]),
        ("MATCH (s:S)-[:R*2]->()-[:R]->(c) RETURN c.id AS id", ["d"]),
        ("MATCH (s:S)-[:R*3]->()-[:R]->(c) RETURN c.id AS id", ["e"]),
        ("MATCH (s:S)-[:R*1..2]->()-[:R]->(c) RETURN c.id AS id ORDER BY id", ["c", "d"]),
        ("MATCH (s:S)-[:R*1..3]->()-[:R]->(c) RETURN c.id AS id ORDER BY id", ["c", "d", "e"]),
        ("MATCH (s:S)-[:R]->()-[:R*2]->(c) RETURN c.id AS id", ["d"]),
        ("MATCH (s:S)-[:R]->()-[:R*1..2]->(c) RETURN c.id AS id ORDER BY id", ["c", "d"]),
        ("MATCH (s:S)-[:R]->()-[:R*2]->()-[:R]->(c) RETURN c.id AS id", ["e"]),
    ],
)
def test_connected_variable_length_exact_results(query: str, expected_ids: list) -> None:
    """Connected patterns with variable-length rels produce correct endpoint results."""
    g = _mk_chain_graph()
    result = g.gfql(query)
    ids = sorted(_to_pandas_df(result._nodes)["id"].tolist())
    assert ids == expected_ids


def test_connected_variable_length_branching() -> None:
    """Star graph: *1 then fixed reaches all leaf neighbors."""
    g = _mk_graph(
        pd.DataFrame({"id": ["root", "a", "b", "c", "d"], "label__Root": [True, False, False, False, False]}),
        pd.DataFrame({"s": ["root", "root", "a", "b"], "d": ["a", "b", "c", "d"], "type": ["R", "R", "R", "R"]}),
    )
    result = g.gfql("MATCH (r:Root)-[:R*1]->()-[:R]->(c) RETURN c.id AS id ORDER BY id")
    ids = sorted(_to_pandas_df(result._nodes)["id"].tolist())
    assert ids == ["c", "d"]


def test_connected_variable_length_reverse() -> None:
    """Reverse direction: <-[:R*2]-()<-[:R]- works."""
    g = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"], "label__E": [False, False, False, True]}),
        pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"], "type": ["R", "R", "R"]}),
    )
    result = g.gfql("MATCH (e:E)<-[:R*2]-()<-[:R]-(c) RETURN c.id AS id")
    ids = _to_pandas_df(result._nodes)["id"].tolist()
    assert ids == ["a"]


def test_connected_variable_length_no_match() -> None:
    """Non-matching type produces empty result."""
    g = _mk_chain_graph()
    result = g.gfql("MATCH (s:S)-[:X*2]->()-[:R]->(c) RETURN c.id AS id")
    assert len(result._nodes) == 0


def test_connected_variable_length_typed_mixed() -> None:
    """Typed varlen followed by different-type fixed hop."""
    g = _mk_graph(
        pd.DataFrame({"id": list("abcde"), "label__S": [True, False, False, False, False]}),
        pd.DataFrame({"s": list("abcd"), "d": list("bcde"), "type": ["A", "A", "B", "B"]}),
    )
    result = g.gfql("MATCH (s:S)-[:A*2]->()-[:B]->(c) RETURN c.id AS id")
    ids = _to_pandas_df(result._nodes)["id"].tolist()
    assert ids == ["d"]


def _mk_tck_pattern_predicate_graph(engine: str | None = None) -> _CypherTestGraph:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "label__A": [True, False, False, False],
            "label__B": [False, True, False, False],
            "label__C": [False, False, True, False],
            "label__D": [False, False, False, True],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "b", "a", "a"],
            "d": ["b", "a", "c", "d"],
            "type": ["REL1", "REL2", "REL3", "REL1"],
        }
    )
    if engine == "cudf":
        _require_cudf_runtime()
        return _mk_cudf_graph(nodes, edges)
    return _mk_graph(nodes, edges)


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
@pytest.mark.parametrize(
    "query,expected_rows",
    [
        (
            "MATCH (n) WHERE (n)-[:REL1*2]-() RETURN n ORDER BY n.id",
            [{"n": "(:B)"}, {"n": "(:D)"}],
        ),
        (
            "MATCH (n), (m) WHERE (n)-[:REL1|REL2|REL3|REL4]-(m) RETURN n, m ORDER BY n.id, m.id",
            [
                {"n": "(:A)", "m": "(:B)"},
                {"n": "(:A)", "m": "(:C)"},
                {"n": "(:A)", "m": "(:D)"},
                {"n": "(:B)", "m": "(:A)"},
                {"n": "(:C)", "m": "(:A)"},
                {"n": "(:D)", "m": "(:A)"},
            ],
        ),
        (
            "MATCH (n), (m) WHERE (n)-[:REL1*2]-(m) RETURN n, m ORDER BY n.id, m.id",
            [
                {"n": "(:B)", "m": "(:D)"},
                {"n": "(:D)", "m": "(:B)"},
            ],
        ),
    ],
)
def test_string_cypher_pattern_predicates_are_existence_checks_not_row_expansions(
    engine: str | None,
    query: str,
    expected_rows: list[dict[str, object]],
) -> None:
    graph = _mk_tck_pattern_predicate_graph(engine)
    kwargs = {"engine": "cudf"} if engine == "cudf" else {}

    result = graph.gfql(query, **kwargs)

    if engine == "cudf":
        assert type(result._nodes).__module__.startswith("cudf")
    _entities = {
        name: ("edges" if str(val).startswith("[") else "nodes")
        for name, val in (expected_rows[0].items() if expected_rows else [])
        if str(val).startswith(("(", "["))
    }
    assert entity_text_records(result, _entities) == expected_rows


@pytest.mark.parametrize(
    "query,expected_rows",
    [
        (
            "MATCH (n) WHERE (n)-[:REL1*2]->() RETURN n.id AS id ORDER BY id",
            [{"id": "a"}, {"id": "b"}],
        ),
        (
            "MATCH (n) WHERE (n)-[*2]->() RETURN n.id AS id ORDER BY id",
            [{"id": "a"}, {"id": "b"}],
        ),
        (
            "MATCH (n) WHERE (n)<-[:REL1*1..2]-() RETURN n.id AS id ORDER BY id",
            [{"id": "b"}, {"id": "c"}, {"id": "d"}],
        ),
        (
            "MATCH (n) WHERE (n)-[:REL1*2]->() AND n.id <> 'a' RETURN n.id AS id ORDER BY id",
            [{"id": "b"}],
        ),
    ],
)
def test_string_cypher_executes_bounded_variable_length_where_pattern_predicates(
    query: str,
    expected_rows: list[dict[str, object]],
) -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c"],
                "d": ["b", "c", "d"],
                "type": ["REL1", "REL1", "REL1"],
            }
        ),
    )

    result = graph.gfql(query)
    assert result._nodes.to_dict(orient="records") == expected_rows


@pytest.mark.parametrize(
    "query,expected_rows",
    [
        (
            "MATCH (n) WHERE (n)-[:REL1*2]->() OR n.id = 'd' RETURN n.id AS id ORDER BY id",
            [{"id": "a"}, {"id": "b"}, {"id": "d"}],
        ),
        (
            "MATCH (n) WHERE (n)-[:REL1*2]->() XOR n.id = 'd' RETURN n.id AS id ORDER BY id",
            [{"id": "a"}, {"id": "b"}, {"id": "d"}],
        ),
        (
            "MATCH (n) WHERE NOT (n)-[:REL1*2]->() RETURN n.id AS id ORDER BY id",
            [{"id": "c"}, {"id": "d"}],
        ),
    ],
)
def test_string_cypher_executes_bounded_variable_length_where_pattern_boolean_wrappers(
    query: str,
    expected_rows: list[dict[str, object]],
) -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c"],
                "d": ["b", "c", "d"],
                "type": ["REL1", "REL1", "REL1"],
            }
        ),
    )

    result = graph.gfql(query)
    assert result._nodes.to_dict(orient="records") == expected_rows


def test_string_cypher_executes_conjoined_bounded_varlen_where_predicates_across_edge_types() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d", "e"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c", "a", "b"],
                "d": ["b", "c", "d", "e", "e"],
                "type": ["REL1", "REL1", "REL1", "REL2", "REL2"],
            }
        ),
    )

    rows_forward = graph.gfql(
        "MATCH (n) WHERE (n)-[:REL1*2]->() AND (n)-[:REL2*1]->() RETURN n.id AS id ORDER BY id"
    )._nodes.to_dict(orient="records")
    rows_reverse = graph.gfql(
        "MATCH (n) WHERE (n)-[:REL2*1]->() AND (n)-[:REL1*2]->() RETURN n.id AS id ORDER BY id"
    )._nodes.to_dict(orient="records")

    assert rows_forward == [{"id": "a"}, {"id": "b"}]
    assert rows_reverse == [{"id": "a"}, {"id": "b"}]
    assert rows_forward == rows_reverse


def test_string_cypher_executes_xor_between_bounded_reverse_and_forward_where_patterns() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c"],
                "d": ["b", "c", "d"],
                "type": ["REL1", "REL1", "REL1"],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (n) WHERE (n)<-[:REL1*1..2]-() XOR (n)-[:REL1*2]->() RETURN n.id AS id ORDER BY id"
    )
    assert result._nodes.to_dict(orient="records") == [
        {"id": "a"},
        {"id": "c"},
        {"id": "d"},
    ]



@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a)-[:KNOWS]-(b) RETURN not((a)-[:KNOWS]-(b)) AS isNew",
        "MATCH (a)-[:KNOWS]-(b) RETURN not((a:Person)-[:KNOWS]-(b)) AS isNew",
        "MATCH (a)-[:KNOWS]-(b) RETURN not((a)-[:KNOWS {w: 1}]-(b)) AS isNew",
        "MATCH (a)-[:KNOWS]-(b) RETURN not((a)-[:KNOWS]-(:Person)) AS isNew",
        "MATCH (a)-[:KNOWS]-(b) RETURN not((a)<-[:KNOWS]-(b)) AS isNew",
        "MATCH (a)-[:KNOWS]-(b) RETURN not((a)--(b)) AS isNew",
        "MATCH (a)-[:KNOWS]-(b) RETURN not((a)-[:KNOWS*]->(b)) AS isNew",
        "MATCH (a)-[:KNOWS]-(b) RETURN not((a)-[k:KNOWS]->(b)) AS isNew",
        "MATCH (a)-[:KNOWS]-(b) RETURN not/*inline*/((a)-[:KNOWS]-(b)) AS isNew",
        "MATCH (a) RETURN exists { (a)-[:KNOWS]-() } AS has",
        "MATCH (a) RETURN exists/*inline*/{ (a)-[:KNOWS]-() } AS has",
        "MATCH (a) RETURN not exists { (a)-[:KNOWS]-() } AS no",
        "MATCH (a) RETURN not/*inline*/exists/*inline*/{ (a)-[:KNOWS]-() } AS no",
    ],
)
def test_string_cypher_failfast_rejects_pattern_existence(query: str) -> None:
    """#998: not((..)) pattern-expression spellings and RETURN/WITH-position
    EXISTS {{ }} fail-fast with a clear message. WHERE-position EXISTS {{ }} is
    SUPPORTED since viz-filter L1 (see the exists_subquery lowering tests)."""
    graph = _mk_empty_graph()
    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(query)
    assert "pattern existence" in exc_info.value.message.lower()


@pytest.mark.parametrize(
    "query,expected_rows",
    [
        (
            "MATCH (n) WHERE (n)-[:R*]->() AND n.id <> 'a' RETURN n.id AS id ORDER BY id",
            [{"id": "b"}, {"id": "c"}],
        ),
        (
            "MATCH (n) WHERE n.id <> 'a' AND (n)-[:R*]->() RETURN n.id AS id ORDER BY id",
            [{"id": "b"}, {"id": "c"}],
        ),
        (
            "MATCH (n) WHERE n.id <> 'a' AND (n)-[:R*]->() AND n.kind = 'x' RETURN n.id AS id ORDER BY id",
            [{"id": "b"}, {"id": "c"}],
        ),
        (
            "MATCH (n) WHERE n.kind = 'x' AND (n)-[:R*]->() AND n.id <> 'a' RETURN n.id AS id ORDER BY id",
            [{"id": "b"}, {"id": "c"}],
        ),
        (
            "MATCH (n) WHERE (n)-[:R*]->() AND n.id <> 'a' AND n.id <> 'b' RETURN n.id AS id ORDER BY id",
            [{"id": "c"}],
        ),
        (
            "MATCH (n) WHERE n.id <> 'a' AND n.kind = 'x' AND (n)-[:R*]->() RETURN n.id AS id ORDER BY id",
            [{"id": "b"}, {"id": "c"}],
        ),
        (
            "MATCH (n) WHERE (n)-[:R*]->() AND (n.id = 'b' OR n.id = 'c') RETURN n.id AS id ORDER BY id",
            [{"id": "b"}, {"id": "c"}],
        ),
    ],
)
def test_string_cypher_executes_where_pattern_predicate_and_expr_mix(
    query: str,
    expected_rows: list[dict[str, object]],
) -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"], "kind": ["x", "x", "x", "y"]}),
        pd.DataFrame(
            {
                "s": ["a", "a", "b", "c"],
                "d": ["b", "c", "c", "d"],
                "type": ["R", "R", "R", "R"],
            }
        ),
    )

    result = graph.gfql(query)

    assert result._nodes.to_dict(orient="records") == expected_rows


@pytest.mark.parametrize(
    "query,expected_rows",
    [
        (
            "MATCH (n) WHERE (n)-[:R*]->() OR n.id = 'd' RETURN n.id AS id ORDER BY id",
            [{"id": "a"}, {"id": "b"}, {"id": "d"}],
        ),
        (
            "MATCH (n) WHERE (n)-[:R]->() XOR n.id = 'd' RETURN n.id AS id ORDER BY id",
            [{"id": "a"}, {"id": "b"}, {"id": "d"}],
        ),
        (
            "MATCH (n) WHERE (n)-[:R]->() XOR n.id = 'a' RETURN n.id AS id ORDER BY id",
            [{"id": "b"}],
        ),
    ],
)
def test_string_cypher_executes_or_xor_around_pattern_predicates(
    query: str,
    expected_rows: list[dict[str, object]],
) -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "a", "b"],
                "d": ["b", "c", "c"],
                "type": ["R", "R", "R"],
            }
        ),
    )
    result = graph.gfql(query)
    assert result._nodes.to_dict(orient="records") == expected_rows


def test_string_cypher_failfast_rejects_not_over_pattern_or_expr_compound() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "a", "b"],
                "d": ["b", "c", "c"],
                "type": ["R", "R", "R"],
            }
        ),
    )

    with pytest.raises(GFQLValidationError, match="Pattern existence expressions"):
        graph.gfql("MATCH (n) WHERE NOT ((n)-[:R]->() OR n.id = 'd') RETURN n.id AS id")


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) WHERE NOT (n)-[:R*]->() RETURN n.id AS id ORDER BY id",
        "MATCH (n) WHERE NOT (n)-[:R]->() RETURN n.id AS id ORDER BY id",
    ],
)
def test_string_cypher_executes_negated_pattern_where_predicate(query: str) -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "a", "b"],
                "d": ["b", "c", "c"],
                "type": ["R", "R", "R"],
            }
        ),
    )

    result = graph.gfql(query)
    assert result._nodes.to_dict(orient="records") == [
        {"id": "c"},
        {"id": "d"},
    ]


def test_string_cypher_executes_mixed_row_and_negated_pattern_where_predicate() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "a", "b"],
                "d": ["b", "c", "c"],
                "type": ["R", "R", "R"],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (n) WHERE n.id <> 'd' AND NOT (n)-[:R]->() RETURN n.id AS id ORDER BY id"
    )

    assert result._nodes.to_dict(orient="records") == [{"id": "c"}]


def test_string_cypher_executes_bound_alias_negated_pattern_where_predicate() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c"],
                "d": ["b", "a", "d"],
                "type": ["R", "R", "R"],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (a)-[:R]->(b) "
        "WHERE NOT (b)-[:R]->(a) "
        "RETURN a.id AS a_id, b.id AS b_id"
    )

    assert result._nodes.to_dict(orient="records") == [{"a_id": "c", "b_id": "d"}]


def test_string_cypher_executes_mixed_row_and_bound_alias_negated_pattern_where_predicate() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d", "e"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c", "d"],
                "d": ["b", "a", "d", "e"],
                "type": ["R", "R", "R", "R"],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (a)-[:R]->(b) "
        "WHERE a.id <> 'd' AND NOT (b)-[:R]->(a) "
        "RETURN a.id AS a_id, b.id AS b_id"
    )

    assert result._nodes.to_dict(orient="records") == [{"a_id": "c", "b_id": "d"}]


def test_string_cypher_executes_ic10_shaped_bound_alias_negated_pattern_where_predicate() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d", "e"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "b", "a", "c"],
                "d": ["b", "c", "d", "d", "e"],
                "type": ["R", "R", "R", "R", "R"],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (root {id: 'a'})-[:R]->(mid)-[:R]->(cand) "
        "WHERE NOT (root)-[:R]->(cand) "
        "RETURN cand.id AS cand_id ORDER BY cand_id"
    )

    assert result._nodes.to_dict(orient="records") == [{"cand_id": "c"}]


def test_string_cypher_failfast_rejects_multi_alias_return_star_projection() -> None:
    graph = _mk_empty_graph()

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql("MATCH (a)-[]->(b) RETURN *")

    assert exc_info.value.code == ErrorCode.E108
    assert "RETURN * currently requires a single MATCH alias" in exc_info.value.message


def test_projection_plan_defaults_return_star_to_single_alias_without_active_alias() -> None:
    plan = _projection_planning._build_projection_plan(
        _projection_clause("*"),
        alias_targets={"a": ASTNode(name="a")},
    )

    assert plan.source_alias == "a"
    assert plan.whole_row_output_names == ["a"]


def test_projection_plan_defaults_scalar_expression_to_single_alias_without_active_alias() -> None:
    plan = _projection_planning._build_projection_plan(
        _projection_clause("1 + 2", "x"),
        alias_targets={"a": ASTNode(name="a")},
    )

    assert plan.source_alias == "a"
    assert plan.projection_items == [("x", "(1 + 2)")]
    assert [(column.output_name, column.kind, column.source_name) for column in plan.projection_columns] == [
        ("x", "expr", "(1 + 2)")
    ]


def test_string_cypher_projection_duplicate_name_error_preserves_context() -> None:
    graph = _mk_empty_graph()

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql("MATCH (n) RETURN n.id AS x, n.name AS x")

    err = exc_info.value
    assert err.code == ErrorCode.E108
    assert err.context["field"] == "return.items"
    assert err.context["value"] == "x"
    assert err.context["line"] == 1
    assert err.context["column"] == 29
    assert err.context["language"] == "cypher"


def test_string_cypher_projection_type_node_error_preserves_context() -> None:
    graph = _mk_empty_graph()

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql("MATCH (n) RETURN type(n) AS t")

    err = exc_info.value
    assert err.code == ErrorCode.E108
    assert err.context["field"] == "return.items"
    assert err.context["value"] == "type(n)"
    assert err.context["line"] == 1
    assert err.context["column"] == 18
    assert err.context["language"] == "cypher"


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n:A) WITH n MATCH (m:B), (n)-->(x:X) RETURN *",
        "MATCH (n:A) WITH n LIMIT 1 MATCH (m:B), (n)-->(x:X) RETURN *",
        "MATCH (n:A) WITH n SKIP 0 LIMIT 1 MATCH (m:B), (n)-->(x:X) RETURN *",
    ],
)
def test_string_cypher_failfast_rejects_multi_pattern_reentry_match_as_unsupported(query: str) -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["anon_1", "anon_2", "anon_3"],
                "label__A": [True, False, False],
                "label__X": [False, True, False],
                "label__B": [False, False, True],
            }
        ),
        pd.DataFrame({"s": ["anon_1"], "d": ["anon_2"], "type": ["REL"]}),
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(query)

    assert exc_info.value.code == ErrorCode.E108


def test_string_cypher_failfast_rejects_multi_alias_return_star_after_with_reentry() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "label__A": [True, False, False],
                "label__B": [False, True, True],
            }
        ),
        pd.DataFrame({"s": ["a", "a"], "d": ["b", "c"], "type": ["REL", "REL"]}),
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql("MATCH (a:A) WITH a MATCH (a)-->(b) RETURN *")

    assert exc_info.value.code == ErrorCode.E108


def test_string_cypher_supports_optional_match_optional_alias_projection_when_all_rows_match() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["s", "a", "b", "c"],
                "label__Single": [True, False, False, False],
                "label__A": [False, True, False, False],
                "label__B": [False, False, True, False],
                "label__C": [False, False, False, True],
                "num": [None, 42, 46, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["s", "s", "a", "b"],
                "d": ["a", "b", "c", "b"],
                "type": ["REL", "REL", "REL", "LOOP"],
            }
        ),
    )

    result = graph.gfql("MATCH (a:Single), (c:C) OPTIONAL MATCH (a)-->(b)-->(c) RETURN b")

    # OPTIONAL row-guard path keeps whole entities as text (structured #1650 flip
    # is gated off for the reentry/optional machinery; unify in follow-up).
    assert result._nodes.to_dict(orient="records") == [{"b": "(:A {num: 42})"}]


@pytest.mark.parametrize(
    "query",
    [
        "RETURN duration.inSeconds(localtime(), localtime()) AS duration",
        "RETURN duration.inSeconds(time(), time()) AS duration",
        "RETURN duration.inSeconds(date(), date()) AS duration",
        "RETURN duration.inSeconds(localdatetime(), localdatetime()) AS duration",
        "RETURN duration.inSeconds(datetime(), datetime()) AS duration",
    ],
)
def test_string_cypher_executes_duration_unit_functions_with_current_temporals(query: str) -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"duration": "PT0S"}]


@pytest.mark.parametrize(
    ("query", "column"),
    [
        ("RETURN date(null) AS t", "t"),
        ("RETURN localtime(null) AS t", "t"),
        ("RETURN time(null) AS t", "t"),
        ("RETURN localdatetime(null) AS t", "t"),
        ("RETURN datetime(null) AS t", "t"),
        ("RETURN duration(null) AS t", "t"),
    ],
)
def test_string_cypher_temporal_constructor_null_propagation(query: str, column: str) -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(query)

    assert pd.isna(result._nodes.iloc[0][column])


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "WITH date({year: 1984, month: 10, day: 11}) AS d "
            "RETURN toString(d) AS ts, date(toString(d)) = d AS b",
            [{"ts": "1984-10-11", "b": True}],
        ),
        (
            "WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d "
            "RETURN toString(d) AS ts, localtime(toString(d)) = d AS b",
            [{"ts": "12:31:14.645876123", "b": True}],
        ),
        (
            "WITH time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS d "
            "RETURN toString(d) AS ts, time(toString(d)) = d AS b",
            [{"ts": "12:31:14.645876123+01:00", "b": True}],
        ),
        (
            "WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d "
            "RETURN toString(d) AS ts, localdatetime(toString(d)) = d AS b",
            [{"ts": "1984-10-11T12:31:14.645876123", "b": True}],
        ),
        (
            "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS d "
            "RETURN toString(d) AS ts, datetime(toString(d)) = d AS b",
            [{"ts": "1984-10-11T12:31:14.645876123+01:00", "b": True}],
        ),
    ],
)
def test_string_cypher_temporal_tostring_roundtrip(
    query: str, expected: list[dict[str, object]]
) -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(query)

    assert result._nodes.to_dict(orient="records") == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("RETURN duration({months: 5, days: 1.5}) AS result", "P5M1DT12H"),
        ("RETURN duration({months: 0.75}) AS result", "P22DT19H51M49.5S"),
        ("RETURN duration({weeks: 2.5}) AS result", "P17DT12H"),
    ],
)
def test_string_cypher_executes_fractional_duration_map_literals(query: str, expected: str) -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"result": expected}]


def test_string_cypher_formats_temporal_constructor_properties_in_entity_projection() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "label__A": [True, True, True],
            "date": [
                "date({year: 1910, month: 5, day: 6})",
                "date({year: 1980, month: 10, day: 24})",
                "date({year: 1984, month: 10, day: 12})",
            ],
        }
    ).astype({"date": "string"})
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) WITH a, a.date AS date WITH a, date ORDER BY date ASC LIMIT 2 RETURN a, date"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:A {date: '1910-05-06'})", "date": "1910-05-06"},
        {"a": "(:A {date: '1980-10-24'})", "date": "1980-10-24"},
    ]


def test_string_cypher_orders_temporal_constructor_time_properties() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "label__A": [True, True, True, True, True],
            "time": [
                "time({hour: 10, minute: 35, timezone: '-08:00'})",
                "time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})",
                "time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})",
                "time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})",
                "time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})",
            ],
        }
    ).astype({"time": "string"})
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) WITH a, a.time AS time WITH a, time ORDER BY time ASC LIMIT 3 RETURN a, time"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:A {time: '12:35:15+05:00'})", "time": "12:35:15+05:00"},
        {"a": "(:A {time: '12:30:14.645876123+01:01'})", "time": "12:30:14.645876123+01:01"},
        {"a": "(:A {time: '12:31:14.645876123+01:00'})", "time": "12:31:14.645876123+01:00"},
    ]


def test_string_cypher_order_by_falls_back_to_string_sort_for_mixed_time_constructor_text() -> None:
    nodes = pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(129)],
            "label__A": [True] * 129,
            "time": ["time({hour: 10, minute: 35, timezone: '-08:00'})"] * 128 + ["zz-not-a-time"],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) WITH a, a.time AS time WITH a, time ORDER BY time ASC RETURN time"
    )

    assert result._nodes["time"].iloc[0] == "time({hour: 10, minute: 35, timezone: '-08:00'})"
    assert result._nodes["time"].iloc[-1] == "zz-not-a-time"


def test_string_cypher_orders_time_plus_duration_expression() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "label__A": [True, True, True, True, True],
            "time": [
                "time({hour: 10, minute: 35, timezone: '-08:00'})",
                "time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})",
                "time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})",
                "time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})",
                "time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})",
            ],
        }
    ).astype({"time": "string"})
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) WITH a ORDER BY a.time + duration({minutes: 6}) ASC LIMIT 3 RETURN a"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:A {time: '12:35:15+05:00'})"},
        {"a": "(:A {time: '12:30:14.645876123+01:01'})"},
        {"a": "(:A {time: '12:31:14.645876123+01:00'})"},
    ]


def test_string_cypher_orders_datetime_plus_duration_expression() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "label__A": [True, True, True, True, True],
            "datetime": [
                "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})",
                "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})",
                "datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})",
                "datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})",
                "datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})",
            ],
        }
    ).astype({"datetime": "string"})
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) WITH a ORDER BY a.datetime + duration({days: 4, minutes: 6}) ASC LIMIT 3 RETURN a"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:A {datetime: '0001-01-01T01:01:01.000000001-11:59'})"},
        {"a": "(:A {datetime: '1980-12-11T12:31:14-11:59'})"},
        {"a": "(:A {datetime: '1984-10-11T12:31:14.645876123+00:17'})"},
    ]


def test_string_cypher_orders_date_plus_duration_expression() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "label__A": [True, True, True, True, True, True],
            "date": [
                "date({year: 1910, month: 5, day: 6})",
                "date({year: 1980, month: 12, day: 24})",
                "date({year: 1984, month: 10, day: 12})",
                "date({year: 1985, month: 5, day: 6})",
                "date({year: 1980, month: 10, day: 24})",
                "date({year: 1984, month: 10, day: 11})",
            ],
        }
    ).astype({"date": "string"})
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) WITH a ORDER BY a.date + duration({months: 1, days: 2}) ASC LIMIT 2 RETURN a"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:A {date: '1910-05-06'})"},
        {"a": "(:A {date: '1980-10-24'})"},
    ]


def test_string_cypher_formats_list_literal_strings_in_entity_projection() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "label__A": [True, True],
            "list": ["[1, 2]", "[2, -2]"],
        }
    ).astype({"list": "string"})
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) WITH a, a.list AS list WITH a, list ORDER BY list ASC LIMIT 2 RETURN a, list"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:A {list: [1, 2]})", "list": "[1, 2]"},
        {"a": "(:A {list: [2, -2]})", "list": "[2, -2]"},
    ]


def test_string_cypher_unwinds_node_keys_without_alias_heuristics() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a"],
            "type": ["Person"],
            "name": ["Alice"],
            "active": [True],
            "score": [5],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n:Person) UNWIND keys(n) AS x RETURN DISTINCT x AS theProps ORDER BY theProps"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"theProps": "active"},
        {"theProps": "name"},
        {"theProps": "score"},
    ]


def test_string_cypher_supports_in_keys_for_node_properties() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a"],
            "type": ["Person"],
            "name": ["Alice"],
            "active": [True],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n:Person) RETURN 'active' IN keys(n) AS has_active, 'missing' IN keys(n) AS missing"
    )

    assert result._nodes.to_dict(orient="records") == [{"has_active": True, "missing": False}]


def test_string_cypher_supports_keys_for_mixed_node_property_sets() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "type": ["Person", "Person"],
            "name": ["Alice", None],
            "score": [None, None],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n:Person) RETURN n.id AS id, keys(n) AS ks ORDER BY id"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"id": "a", "ks": ["name"]},
        {"id": "b", "ks": []},
    ]


def test_string_cypher_supports_unwind_keys_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["a"],
                "type": ["Person"],
                "name": ["Alice"],
                "score": [1],
                "active": [True],
            }
        )
    )
    edges = cudf.from_pandas(pd.DataFrame({"s": [], "d": []}))

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n:Person) UNWIND keys(n) AS x RETURN DISTINCT x AS theProps ORDER BY theProps",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"theProps": "active"},
        {"theProps": "name"},
        {"theProps": "score"},
    ]


def test_string_cypher_supports_properties_for_node_relationship_map_and_null() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "label__Person": [True, False],
            "name": ["Popeye", None],
            "level": [9001, None],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a"],
            "d": ["b"],
            "type": ["R"],
            "name": ["Popeye"],
            "level": [9001],
        }
    )
    graph = _mk_graph(nodes, edges)

    node_result = graph.gfql("MATCH (p:Person) RETURN properties(p) AS m")
    assert node_result._nodes.to_dict(orient="records") == [{"m": "{name: 'Popeye', level: 9001}"}]

    edge_result = graph.gfql("MATCH ()-[r:R]->() RETURN properties(r) AS m")
    assert edge_result._nodes.to_dict(orient="records") == [{"m": "{name: 'Popeye', level: 9001}"}]

    map_result = graph.gfql("RETURN properties({name: 'Popeye', level: 9001}) AS m, properties(null) AS n")
    assert map_result._nodes.to_dict(orient="records") == [
        {"m": "{name: 'Popeye', level: 9001}", "n": None}
    ]


def test_string_cypher_supports_property_access_on_null_map_alias() -> None:
    graph = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": [], "type": []}))

    result = graph.gfql("WITH null AS m RETURN m.missing AS out, m.missing IS NULL AS is_null")

    assert result._nodes.to_dict(orient="records") == [{"out": None, "is_null": True}]


def test_string_cypher_supports_property_access_on_graph_alias_in_projection() -> None:
    graph = cast(
        _CypherTestGraph,
        _CypherTestGraph().nodes(pd.DataFrame({"node_id": ["n1"]}), "node_id").edges(
            pd.DataFrame({"s": [], "d": [], "type": []}), "s", "d"
        ),
    )

    result = graph.gfql("MATCH (a) RETURN a.id IS NOT NULL AS a, a IS NOT NULL AS b")

    assert result._nodes.to_dict(orient="records") == [{"a": False, "b": True}]


def test_string_cypher_supports_labels_projection_and_relationship_label_predicate() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "label__Foo": [True, True, False],
            "label__Bar": [False, True, False],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "a"],
            "d": ["b", "c"],
            "type": ["T1", "T2"],
        }
    )
    graph = _mk_graph(nodes, edges)

    labels_result = graph.gfql("MATCH (n) RETURN labels(n) AS ls ORDER BY ls")
    actual_label_rows = sorted(
        labels_result._nodes.to_dict(orient="records"),
        key=lambda row: (len(row["ls"]), row["ls"]),
    )
    assert actual_label_rows == [
        {"ls": "[]"},
        {"ls": "['Foo']"},
        {"ls": "['Foo', 'Bar']"},
    ]

    rel_result = graph.gfql("MATCH ()-[r]->() RETURN r, r:T2 AS result ORDER BY result")
    assert sorted(entity_text_records(rel_result, {"r": "edges"}), key=lambda row: row["r"]) == [
        {"r": "[:T1]", "result": False},
        {"r": "[:T2]", "result": True},
    ]


def test_string_cypher_supports_graph_functions_on_list_wrapped_entities() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "label__Foo": [True, True],
            "label__Bar": [False, True],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a"],
            "d": ["b"],
            "type": ["T"],
        }
    )
    graph = _mk_graph(nodes, edges)

    labels_result = graph.gfql("MATCH (a) WITH [a, 1] AS list RETURN labels(list[0]) AS l ORDER BY l")
    assert sorted(labels_result._nodes.to_dict(orient="records"), key=lambda row: (len(row["l"]), row["l"])) == [
        {"l": "['Foo']"},
        {"l": "['Foo', 'Bar']"},
    ]

    type_result = graph.gfql("MATCH ()-[r]->() WITH [r, 1] AS list RETURN type(list[0]) AS t")
    assert type_result._nodes.to_dict(orient="records") == [{"t": "T"}]


def test_string_cypher_supports_graph_functions_on_list_wrapped_entities_on_cudf(monkeypatch) -> None:
    cudf = pytest.importorskip("cudf")
    import graphistry.compute.gfql.row.pipeline as row_pipeline

    monkeypatch.setattr(
        row_pipeline,
        "_gfql_bridge_cudf_df_to_pandas",
        lambda _df: (_ for _ in ()).throw(AssertionError("projection should not bridge the full cuDF row table")),
    )

    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "label__Foo": [True, True],
                "label__Bar": [False, True],
            }
        )
    )
    edges = cudf.from_pandas(
        pd.DataFrame(
            {
                "s": ["a"],
                "d": ["b"],
                "type": ["T"],
            }
        )
    )
    graph = _mk_graph(nodes, edges)

    labels_result = graph.gfql(
        "MATCH (a) WITH [a, 1] AS list RETURN labels(list[0]) AS l",
        engine="cudf",
    )
    assert sorted(
        _to_pandas_df(labels_result._nodes).to_dict(orient="records"),
        key=lambda row: (len(row["l"]), row["l"]),
    ) == [
        {"l": "['Foo']"},
        {"l": "['Foo', 'Bar']"},
    ]

    type_result = graph.gfql(
        "MATCH ()-[r]->() WITH [r, 1] AS list RETURN type(list[0]) AS t",
        engine="cudf",
    )
    assert _to_pandas_df(type_result._nodes).to_dict(orient="records") == [{"t": "T"}]


def test_string_cypher_supports_null_graph_functions_in_multi_alias_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "type": ["Seed", "Seed"]}),
        pd.DataFrame(
            {
                "s": ["a"],
                "d": ["b"],
                "type": ["REL"],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (a)-[r]->(b) "
        "RETURN labels(null) AS nn, properties(null) AS q, type(r) AS tr, type(null) AS tn"
    )
    assert result._nodes.to_dict(orient="records") == [{"nn": None, "q": None, "tr": "REL", "tn": None}]


def test_string_cypher_supports_top_level_optional_match_null_rows_for_labels() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": pd.Series(dtype="object"), "type": pd.Series(dtype="object")}),
        pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")}),
    )

    result = graph.gfql("OPTIONAL MATCH (n:DoesNotExist) RETURN labels(n) AS ln, labels(null) AS nn")

    assert result._nodes.to_dict(orient="records") == [{"ln": None, "nn": None}]


def test_string_cypher_supports_top_level_optional_match_null_rows_for_labels_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    graph = _mk_graph(
        cudf.from_pandas(
            pd.DataFrame({"id": pd.Series(dtype="object"), "labels": pd.Series(dtype="object")})
        ),
        cudf.from_pandas(
            pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")})
        ),
    )

    result = graph.gfql(
        "OPTIONAL MATCH (n:DoesNotExist) RETURN labels(n) AS ln, labels(null) AS nn",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"ln": None, "nn": None}]


def test_string_cypher_supports_optional_match_inline_missing_label_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    graph = _mk_graph(
        cudf.from_pandas(
            pd.DataFrame(
                {
                    "id": ["a"],
                    "labels": [["Single"]],
                    "label__Single": [True],
                }
            )
        ),
        cudf.from_pandas(pd.DataFrame({"s": [], "d": []})),
    )

    result = graph.gfql(
        "MATCH (n:Single)\nOPTIONAL MATCH (n)-[r]-(m:NonExistent)\nRETURN r",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"r": None}]


def test_string_cypher_supports_top_level_optional_match_null_rows_for_property_access() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": pd.Series(dtype="object")}),
        pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")}),
    )

    node_result = graph.gfql("OPTIONAL MATCH (n) RETURN n.missing AS m")
    rel_result = graph.gfql("OPTIONAL MATCH ()-[r]->() RETURN r.missing AS m")

    assert node_result._nodes.to_dict(orient="records") == [{"m": None}]
    assert rel_result._nodes.to_dict(orient="records") == [{"m": None}]


def test_string_cypher_supports_bound_optional_match_null_rows_for_type() -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["a"]}), pd.DataFrame({"s": [], "d": [], "type": []}))

    result = graph.gfql("MATCH (a) OPTIONAL MATCH (a)-[r:NOT_THERE]->() RETURN type(r) AS tr, type(null) AS tn")

    assert result._nodes.to_dict(orient="records") == [{"tr": None, "tn": None}]


def test_string_cypher_supports_bound_optional_match_whole_row_with_scalar_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["T"]}),
    )

    result = graph.gfql("MATCH (a) OPTIONAL MATCH (a)-[r:T]->(b) RETURN b, b.id + '!' AS label")

    # OPTIONAL-MATCH null-fill path still renders whole entities as text (the
    # structured #1650 flip is gated off for reentry; unification is a follow-up).
    assert result._nodes.to_dict(orient="records") == [
        {"b": "()", "label": "b!"},
        {"b": None, "label": None},
    ]


def test_string_cypher_supports_top_level_optional_match_null_rows_for_whole_row_edge_projection() -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["a"]}), pd.DataFrame({"s": [], "d": [], "type": []}))

    result = graph.gfql("OPTIONAL MATCH ()-[r]->() RETURN r")

    assert result._nodes.to_dict(orient="records") == [{"r": None}]


def test_string_cypher_supports_bound_optional_match_mixed_null_and_non_null_rows_for_type() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["T"]}),
    )

    result = graph.gfql("MATCH (a) OPTIONAL MATCH (a)-[r:T]->() RETURN type(r) AS tr")

    assert sorted(
        result._nodes.to_dict(orient="records"),
        key=lambda row: (row["tr"] is None, str(row["tr"])),
    ) == [
        {"tr": "T"},
        {"tr": None},
    ]


def test_string_cypher_preserves_bound_optional_match_row_order_for_optional_alias_outputs() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["b"], "d": ["c"], "type": ["T"]}),
    )

    result = graph.gfql("MATCH (a) OPTIONAL MATCH (a)-[r:T]->() RETURN type(r) AS tr")

    assert result._nodes.to_dict(orient="records") == [
        {"tr": None},
        {"tr": "T"},
        {"tr": None},
    ]


def test_string_cypher_rejects_bound_optional_match_seed_only_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["b"], "d": ["c"], "type": ["T"]}),
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql("MATCH (a) OPTIONAL MATCH (a)-[r:T]->() RETURN a.id AS id")

    assert exc_info.value.code == ErrorCode.E108
    assert "bound seed alias" in exc_info.value.message


def test_string_cypher_supports_label_expression_on_null_with_reserved_keyword_labels() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["s"], "label__Single": [True]}),
        pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object"), "type": pd.Series(dtype="object")}),
    )

    result = graph.gfql("MATCH (n:Single) OPTIONAL MATCH (n)-[r:TYPE]-(m) RETURN m:TYPE")

    assert result._nodes.to_dict(orient="records") == [{"m:TYPE": None}]


def test_string_cypher_supports_dynamic_graph_property_lookup() -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["a"], "name": ["Apa"]}), pd.DataFrame({"s": [], "d": []}))

    result = graph.gfql("MATCH (n {name: 'Apa'}) RETURN n['nam' + 'e'] AS value")

    assert result._nodes.to_dict(orient="records") == [{"value": "Apa"}]


def test_string_cypher_supports_dynamic_graph_property_lookup_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    graph = _mk_graph(
        cudf.from_pandas(pd.DataFrame({"id": ["a"], "name": ["Apa"]})),
        cudf.from_pandas(pd.DataFrame({"s": [], "d": []})),
    )

    result = graph.gfql("MATCH (n {name: 'Apa'}) RETURN n['nam' + 'e'] AS value", engine="cudf")

    nodes_df = _to_pandas_df(result._nodes)
    assert nodes_df.to_dict(orient="records") == [{"value": "Apa"}]


def test_string_cypher_supports_dynamic_graph_property_lookup_with_param() -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["a"], "name": ["Apa"]}), pd.DataFrame({"s": [], "d": []}))

    result = graph.gfql("MATCH (n {name: 'Apa'}) RETURN n[$idx] AS value", params={"idx": "name"})

    assert result._nodes.to_dict(orient="records") == [{"value": "Apa"}]


def test_string_cypher_supports_property_access_on_list_wrapped_node_and_relationship_entities() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "existing": [42, None],
            "missing": [None, None],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a"],
            "d": ["b"],
            "type": ["REL"],
            "existing": [42],
            "missing": [None],
        }
    )
    graph = _mk_graph(nodes, edges)

    node_result = graph.gfql(
        "MATCH (n) WITH [123, n] AS list RETURN (list[1]).missing, (list[1]).missingToo, (list[1]).existing"
    )
    assert node_result._nodes.to_dict(orient="records") == [
        {"(list[1]).missing": None, "(list[1]).missingToo": None, "(list[1]).existing": 42},
        {"(list[1]).missing": None, "(list[1]).missingToo": None, "(list[1]).existing": None},
    ]

    rel_result = graph.gfql(
        "MATCH ()-[r]->() WITH [123, r] AS list RETURN (list[1]).missing, (list[1]).missingToo, (list[1]).existing"
    )
    assert rel_result._nodes.to_dict(orient="records") == [
        {"(list[1]).missing": None, "(list[1]).missingToo": None, "(list[1]).existing": 42},
    ]


def test_string_cypher_supports_property_access_on_list_wrapped_node_and_relationship_entities_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "existing": [42, None],
                "missing": [None, None],
            }
        )
    )
    edges = cudf.from_pandas(
        pd.DataFrame(
            {
                "s": ["a"],
                "d": ["b"],
                "type": ["REL"],
                "existing": [42],
                "missing": [None],
            }
        )
    )
    graph = _mk_graph(nodes, edges)

    node_result = graph.gfql(
        "MATCH (n) WITH [123, n] AS list RETURN (list[1]).missing, (list[1]).missingToo, (list[1]).existing",
        engine="cudf",
    )
    assert _to_pandas_df(node_result._nodes).to_dict(orient="records") == [
        {"(list[1]).missing": None, "(list[1]).missingToo": None, "(list[1]).existing": 42},
        {"(list[1]).missing": None, "(list[1]).missingToo": None, "(list[1]).existing": None},
    ]

    rel_result = graph.gfql(
        "MATCH ()-[r]->() WITH [123, r] AS list RETURN (list[1]).missing, (list[1]).missingToo, (list[1]).existing",
        engine="cudf",
    )
    assert _to_pandas_df(rel_result._nodes).to_dict(orient="records") == [
        {"(list[1]).missing": None, "(list[1]).missingToo": None, "(list[1]).existing": 42},
    ]


def test_string_cypher_supports_property_access_on_list_wrapped_map_values() -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["a"]}), pd.DataFrame({"s": [], "d": []}))

    result = graph.gfql(
        "WITH [123, {existing: 42, notMissing: null}] AS list RETURN (list[1]).missing, (list[1]).notMissing, (list[1]).existing"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"(list[1]).missing": None, "(list[1]).notMissing": None, "(list[1]).existing": 42}
    ]


def test_string_cypher_supports_property_access_on_list_wrapped_map_values_on_cudf(monkeypatch) -> None:
    cudf = pytest.importorskip("cudf")
    import graphistry.compute.gfql.row.pipeline as row_pipeline

    monkeypatch.setattr(
        row_pipeline,
        "_gfql_bridge_cudf_df_to_pandas",
        lambda _df: (_ for _ in ()).throw(AssertionError("projection should not bridge the full cuDF row table")),
    )

    graph = _mk_graph(
        cudf.from_pandas(pd.DataFrame({"id": pd.Series(dtype="object")})),
        cudf.from_pandas(pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")})),
    )

    result = graph.gfql(
        "WITH [123, {existing: 42, notMissing: null}] AS list RETURN (list[1]).missing, (list[1]).notMissing, (list[1]).existing",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"(list[1]).missing": None, "(list[1]).notMissing": None, "(list[1]).existing": 42}
    ]


@pytest.mark.parametrize("bad_literal", ["0", "1.0", "true"])
def test_string_cypher_rejects_type_on_mixed_entity_and_scalar_list_values(bad_literal: str) -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["a", "b"]}), pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["T"]}))

    with pytest.raises(GFQLTypeError, match="type\\(\\) requires a graph element, entity value, or null"):
        graph.gfql(f"MATCH ()-[r:T]->() RETURN [x IN [r, {bad_literal}] | type(x)] AS list")


@pytest.mark.parametrize(
    "query",
    [
        "RETURN properties(1)",
        "RETURN properties('Cypher')",
        "RETURN properties([true, false])",
    ],
)
def test_string_cypher_rejects_invalid_properties_arguments(query: str) -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["a"]}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLTypeError, match="properties\\(\\) requires a node, relationship, map, or null argument"):
        graph.gfql(query)


def test_string_cypher_rejects_invalid_graph_functions_on_list_wrapped_scalars() -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["a"]}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLTypeError, match="labels\\(\\) requires a graph element, entity value, or null"):
        graph.gfql("MATCH (a) WITH [a, 1] AS list RETURN labels(list[1]) AS l")

    with pytest.raises(GFQLTypeError, match="type\\(\\) requires a graph element, entity value, or null"):
        graph.gfql("MATCH (a) WITH [a, 1] AS list RETURN type(list[1]) AS t")


def test_string_cypher_rejects_type_on_node_alias_even_when_match_is_empty() -> None:
    graph = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLValidationError, match="relationship aliases"):
        graph.gfql("MATCH (r) RETURN type(r)")


@pytest.mark.parametrize(
    "query",
    [
        "WITH 123 AS nonGraphElement RETURN nonGraphElement.num AS v",
        "WITH 42.45 AS nonGraphElement RETURN nonGraphElement.num AS v",
        "WITH true AS nonGraphElement RETURN nonGraphElement.num AS v",
        "WITH 'abc' AS nonGraphElement RETURN nonGraphElement.num AS v",
        "WITH [1, 2] AS nonGraphElement RETURN nonGraphElement.num AS v",
    ],
)
def test_string_cypher_rejects_property_access_on_non_graph_values(query: str) -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["a"]}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLTypeError, match="property access requires a graph element alias"):
        graph.gfql(query)


def test_string_cypher_unwind_null_returns_no_rows() -> None:
    graph = _mk_graph(pd.DataFrame({"id": ["a"]}), pd.DataFrame({"s": [], "d": []}))

    result = graph.gfql("UNWIND null AS nil RETURN nil")

    assert result._nodes.to_dict(orient="records") == []


def test_string_cypher_unwinds_edge_keys_without_internal_columns() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"], "type": ["Person", "Person"]})
    edges = pd.DataFrame(
        {
            "s": ["a"],
            "d": ["b"],
            "type": ["KNOWS"],
            "weight": [5],
            "active": [True],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH ()-[r:KNOWS]-() UNWIND keys(r) AS x RETURN DISTINCT x AS theProps ORDER BY theProps"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"theProps": "active"},
        {"theProps": "weight"},
    ]


def test_cypher_to_gfql_executes_relationship_type_projection() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["KNOWS"]})

    chain = cypher_to_gfql("MATCH ()-[r]->() RETURN type(r) AS rel_type")
    result = _mk_graph(nodes, edges).gfql(chain)

    assert result._nodes.to_dict(orient="records") == [{"rel_type": "KNOWS"}]


def test_cypher_to_gfql_preserves_default_property_output_name() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a"],
            "type": ["Person"],
            "name": ["Alice"],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    chain = cypher_to_gfql("MATCH (p) RETURN p.name")
    result = _mk_graph(nodes, edges).gfql(chain)

    assert result._nodes.to_dict(orient="records") == [{"p.name": "Alice"}]


def test_cypher_to_gfql_preserves_default_relationship_type_output_name() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["KNOWS"]})

    chain = cypher_to_gfql("MATCH ()-[r]->() RETURN type(r)")
    result = _mk_graph(nodes, edges).gfql(chain)

    assert result._nodes.to_dict(orient="records") == [{"type(r)": "KNOWS"}]


def test_compile_cypher_union_returns_union_program() -> None:
    compiled = compile_cypher("RETURN 1 AS x UNION RETURN 2 AS x")

    assert isinstance(compiled, CompiledCypherUnionQuery)
    assert compiled.union_kind == "distinct"
    assert len(compiled.branches) == 2


def test_gfql_executes_union_distinct_query() -> None:
    _assert_query_rows(
        "RETURN 1 AS x UNION RETURN 1 AS x UNION RETURN 2 AS x",
        [{"x": 1}, {"x": 2}],
    )


def test_gfql_executes_union_all_query() -> None:
    _assert_query_rows(
        "RETURN 1 AS x UNION ALL RETURN 1 AS x UNION ALL RETURN 2 AS x",
        [{"x": 1}, {"x": 1}, {"x": 2}],
    )


def test_cypher_to_gfql_rejects_union_programs() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        cypher_to_gfql("RETURN 1 AS x UNION RETURN 2 AS x")

    assert exc_info.value.code == ErrorCode.E108


@pytest.mark.parametrize(
    ("query", "procedure", "result_kind"),
    [
        ("CALL graphistry.degree()", "graphistry.degree", "rows"),
        ("CALL graphistry.igraph.pagerank()", "graphistry.igraph.pagerank", "rows"),
        ("CALL graphistry.igraph.spanning_tree.write()", "graphistry.igraph.spanning_tree.write", "graph"),
        ("CALL graphistry.cugraph.louvain()", "graphistry.cugraph.louvain", "rows"),
        ("CALL graphistry.cugraph.edge_betweenness_centrality()", "graphistry.cugraph.edge_betweenness_centrality", "rows"),
        ("CALL graphistry.cugraph.k_core.write()", "graphistry.cugraph.k_core.write", "graph"),
        ("CALL graphistry.nx.pagerank()", "graphistry.nx.pagerank", "rows"),
        ("CALL graphistry.nx.betweenness_centrality()", "graphistry.nx.betweenness_centrality", "rows"),
        ("CALL graphistry.nx.connected_components()", "graphistry.nx.connected_components", "rows"),
        ("CALL graphistry.nx.core_number()", "graphistry.nx.core_number", "rows"),
        ("CALL graphistry.nx.hits()", "graphistry.nx.hits", "rows"),
        ("CALL graphistry.nx.edge_betweenness_centrality()", "graphistry.nx.edge_betweenness_centrality", "rows"),
        ("CALL graphistry.degree.write()", "graphistry.degree.write", "graph"),
        ("CALL graphistry.igraph.betweenness.write()", "graphistry.igraph.betweenness.write", "graph"),
        ("CALL graphistry.cugraph.louvain.write()", "graphistry.cugraph.louvain.write", "graph"),
        ("CALL graphistry.nx.k_core.write()", "graphistry.nx.k_core.write", "graph"),
        ("CALL graphistry.nx.pagerank.write()", "graphistry.nx.pagerank.write", "graph"),
    ],
)
def test_compile_cypher_call_returns_procedure_program(query: str, procedure: str, result_kind: str) -> None:
    compiled = _compile_query(query)

    assert compiled.procedure_call is not None
    assert compiled.procedure_call.procedure == procedure
    assert compiled.procedure_call.result_kind == result_kind
    assert compiled.logical_plan is not None
    assert compiled.logical_plan_defer_reason is None


def test_cypher_to_gfql_rejects_call_programs() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        cypher_to_gfql("CALL graphistry.degree()")

    assert exc_info.value.code == ErrorCode.E108


@pytest.mark.parametrize(
    ("module_name", "query", "expected_rows", "expected_columns"),
    [
        (
            None,
            "CALL graphistry.degree() "
            "YIELD nodeId, degree "
            "RETURN nodeId, degree "
            "ORDER BY degree DESC, nodeId ASC",
            [
                {"nodeId": "b", "degree": 2},
                {"nodeId": "a", "degree": 1},
                {"nodeId": "c", "degree": 1},
            ],
            None,
        ),
        (
            None,
            "CALL graphistry.degree()",
            [
                {"nodeId": "a", "degree": 1, "degree_in": 0, "degree_out": 1},
                {"nodeId": "b", "degree": 2, "degree_in": 1, "degree_out": 1},
                {"nodeId": "c", "degree": 1, "degree_in": 1, "degree_out": 0},
            ],
            ["nodeId", "degree", "degree_in", "degree_out"],
        ),
        (
            None,
            "CALL graphistry.degree YIELD nodeId RETURN nodeId ORDER BY nodeId ASC",
            [{"nodeId": "a"}, {"nodeId": "b"}, {"nodeId": "c"}],
            None,
        ),
        (
            "igraph",
            "CALL graphistry.igraph.pagerank() "
            "YIELD nodeId, pagerank "
            "RETURN nodeId "
            "ORDER BY pagerank DESC, nodeId ASC "
            "LIMIT 1",
            [{"nodeId": "c"}],
            None,
        ),
        (
            "networkx",
            "CALL graphistry.nx.pagerank() "
            "YIELD nodeId, pagerank "
            "RETURN nodeId "
            "ORDER BY pagerank DESC, nodeId ASC "
            "LIMIT 1",
            [{"nodeId": "c"}],
            None,
        ),
    ],
)
def test_string_cypher_executes_graphistry_call(
    module_name: str | None,
    query: str,
    expected_rows: list[dict[str, object]],
    expected_columns: list[str] | None,
) -> None:
    if module_name is not None:
        pytest.importorskip(module_name)

    result = _mk_simple_path_graph().gfql(query)

    if expected_columns is not None:
        assert list(result._nodes.columns) == expected_columns
    assert result._node == result._nodes.columns[0]
    assert result._edges.empty
    assert result._nodes.to_dict(orient="records") == expected_rows


@pytest.mark.parametrize(
    "query",
    [
        "CALL graphistry.unknown()",
        "CALL test.my.proc YIELD out RETURN out",
        "CALL graphistry.degree() YIELD score RETURN score",
        "CALL graphistry.nx.betweenness()",
    ],
)
def test_string_cypher_call_rejects_invalid_procedure_or_yield(query: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_empty_graph().gfql(query)

    assert exc_info.value.code == ErrorCode.E108


@pytest.mark.parametrize("procedure", ["jaccard", "louvain.write", "minimum_spanning_tree.write"])
def test_compile_cypher_call_rejects_unsupported_networkx_subset_structured(procedure: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _compile_query(f"CALL graphistry.nx.{procedure}()")

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "call"
    assert exc_info.value.context["value"] == f"graphistry.nx.{procedure}"


def test_string_cypher_executes_bare_pagerank_call_in_row_state() -> None:
    pytest.importorskip("igraph")

    result = _mk_simple_path_graph().gfql("CALL graphistry.igraph.pagerank()")

    assert list(result._nodes.columns) == ["nodeId", "pagerank"]
    assert result._node == "nodeId"
    assert result._edges.empty
    assert set(result._nodes["nodeId"]) == {"a", "b", "c"}
    assert result._nodes["pagerank"].gt(0).all()


def test_string_cypher_executes_graph_preserving_degree_call() -> None:
    result = _mk_simple_path_graph().gfql("CALL graphistry.degree.write()")

    assert set(result._nodes.columns) == {"id", "degree", "degree_in", "degree_out"}
    assert result._nodes.sort_values("id").to_dict(orient="records") == [
        {"id": "a", "degree_in": 0, "degree_out": 1, "degree": 1},
        {"id": "b", "degree_in": 1, "degree_out": 1, "degree": 2},
        {"id": "c", "degree_in": 1, "degree_out": 0, "degree": 1},
    ]
    assert result._edges.to_dict(orient="records") == [
        {"s": "a", "d": "b"},
        {"s": "b", "d": "c"},
    ]


@pytest.mark.parametrize(
    ("module_name", "query", "expected_col"),
    [
        ("igraph", "CALL graphistry.igraph.pagerank.write()", "pagerank"),
        ("networkx", "CALL graphistry.nx.pagerank.write()", "pagerank"),
    ],
)
def test_string_cypher_executes_graph_preserving_pagerank_write_call(
    module_name: str,
    query: str,
    expected_col: str,
) -> None:
    pytest.importorskip(module_name)

    result = _mk_simple_path_graph().gfql(query)

    assert expected_col in result._nodes.columns
    assert result._nodes[expected_col].gt(0).all()
    assert result._edges.to_dict(orient="records") == [
        {"s": "a", "d": "b"},
        {"s": "b", "d": "c"},
    ]


def test_string_cypher_write_call_can_feed_match_query() -> None:
    pytest.importorskip("igraph")

    enriched = _mk_simple_path_graph().gfql("CALL graphistry.igraph.pagerank.write()")
    assert "pagerank" in enriched._nodes.columns

    result = enriched.gfql(
        "MATCH (n) "
        "WHERE n.pagerank > 0 "
        "RETURN n.id AS nodeId, n.pagerank AS pagerank "
        "ORDER BY pagerank DESC, nodeId ASC "
        "LIMIT 1"
    )

    assert result._nodes.to_dict(orient="records") == [{"nodeId": "c", "pagerank": pytest.approx(0.47441217150760717)}]


@pytest.mark.parametrize(
    "query",
    [
        "CALL graphistry.cugraph.pagerank.write({out_col: 'pr'}) YIELD nodeId RETURN nodeId",
        "CALL graphistry.degree.write() RETURN * ORDER BY degree DESC",
        "CALL graphistry.cugraph.pagerank.write('pagerank')",
    ],
)
def test_string_cypher_graph_write_call_rejects_row_mode_forms(query: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_simple_path_graph().gfql(query)

    assert exc_info.value.code == ErrorCode.E108


def test_compile_cypher_call_flattens_algorithm_options_into_params() -> None:
    compiled = _compile_query(
        "CALL graphistry.cugraph.betweenness_centrality({directed: false, k: 2, params: {normalized: true}})"
    )

    assert compiled.procedure_call is not None
    assert compiled.procedure_call.call_params == {
        "alg": "betweenness_centrality",
        "directed": False,
        "params": {"normalized": True, "k": 2},
    }


def test_compile_cypher_call_flattens_networkx_options_into_params() -> None:
    compiled = _compile_query(
        "CALL graphistry.nx.edge_betweenness_centrality({directed: false, normalized: false})"
    )

    assert compiled.procedure_call is not None
    assert compiled.procedure_call.call_params == {
        "alg": "edge_betweenness_centrality",
        "directed": False,
        "params": {"normalized": False},
    }


@pytest.mark.parametrize(
    ("query", "expected_outputs"),
    [
        (
            "CALL graphistry.degree()",
            (("nodeId", "nodeId"), ("degree", "degree"), ("degree_in", "degree_in"), ("degree_out", "degree_out")),
        ),
        (
            "CALL graphistry.igraph.pagerank({out_col: 'pr'})",
            (("nodeId", "nodeId"), ("pr", "pr")),
        ),
        (
            "CALL graphistry.nx.pagerank({out_col: 'pr'})",
            (("nodeId", "nodeId"), ("pr", "pr")),
        ),
        (
            "CALL graphistry.nx.edge_betweenness_centrality({out_col: 'ebc'})",
            (("source", "source"), ("destination", "destination"), ("ebc", "ebc")),
        ),
        (
            "CALL graphistry.nx.connected_components()",
            (("nodeId", "nodeId"), ("labels", "labels")),
        ),
        (
            "CALL graphistry.nx.hits()",
            (("nodeId", "nodeId"), ("hubs", "hubs"), ("authorities", "authorities")),
        ),
        (
            "CALL graphistry.cugraph.hits()",
            (("nodeId", "nodeId"), ("hits", "hits"), ("authorities", "authorities")),
        ),
        ("CALL graphistry.nx.k_core.write()", ()),
    ],
)
def test_compile_cypher_call_output_columns_are_backend_stable(query: str, expected_outputs: Tuple[Tuple[str, str], ...]) -> None:
    compiled = _compile_query(query)

    assert compiled.procedure_call is not None
    assert tuple(
        (output.source_name, output.output_name)
        for output in compiled.procedure_call.output_columns
    ) == expected_outputs


def test_compile_cypher_call_rejects_multi_column_out_col_structured() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _compile_query("CALL graphistry.cugraph.hits({out_col: 'score'})")

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "call.args.out_col"
    assert exc_info.value.context["value"] == "score"


def test_compile_cypher_call_rejects_multi_column_networkx_out_col_structured() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _compile_query("CALL graphistry.nx.hits({out_col: 'score'})")

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "call.args.out_col"
    assert exc_info.value.context["value"] == "score"


def test_networkx_backend_rejects_multi_column_out_col_structured() -> None:
    compiled_call = CompiledCypherProcedureCall(
        procedure="graphistry.nx.hits",
        backend="networkx",
        algorithm="hits",
        call_params={"out_col": "score"},
        row_kind="node",
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        networkx_normalized_value_columns(compiled_call)

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "call.args.out_col"
    assert exc_info.value.context["value"] == "score"


def test_compile_cypher_call_rejects_graph_only_cugraph_rows() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _compile_query("CALL graphistry.cugraph.k_core()")

    assert exc_info.value.code == ErrorCode.E108


def test_compile_cypher_call_rejects_graph_only_networkx_rows() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _compile_query("CALL graphistry.nx.k_core()")

    assert exc_info.value.code == ErrorCode.E108


def test_string_cypher_executes_networkx_call_with_option_map() -> None:
    pytest.importorskip("networkx")

    result = _mk_simple_path_graph().gfql(
        "CALL graphistry.nx.pagerank({alpha: 0.9, out_col: 'pr'}) "
        "YIELD nodeId, pr "
        "RETURN nodeId, pr "
        "ORDER BY pr DESC, nodeId ASC "
        "LIMIT 1"
    )

    assert result._nodes.to_dict(orient="records")[0]["nodeId"] == "c"
    assert result._nodes.to_dict(orient="records")[0]["pr"] > 0


def test_string_cypher_executes_networkx_betweenness_row_call() -> None:
    pytest.importorskip("networkx")

    result = _mk_simple_path_graph().gfql(
        "CALL graphistry.nx.betweenness_centrality() "
        "YIELD nodeId, betweenness_centrality "
        "RETURN nodeId, betweenness_centrality "
        "ORDER BY betweenness_centrality DESC, nodeId ASC "
        "LIMIT 1"
    )

    assert result._edges.empty
    assert result._nodes.to_dict(orient="records") == [{"nodeId": "b", "betweenness_centrality": 0.5}]


def test_string_cypher_executes_networkx_betweenness_row_call_with_out_col() -> None:
    pytest.importorskip("networkx")

    result = _mk_simple_path_graph().gfql(
        "CALL graphistry.nx.betweenness_centrality({out_col: 'bc'}) "
        "YIELD nodeId, bc "
        "RETURN nodeId, bc "
        "ORDER BY bc DESC, nodeId ASC "
        "LIMIT 1"
    )

    assert result._edges.empty
    assert result._nodes.to_dict(orient="records") == [{"nodeId": "b", "bc": 0.5}]


@pytest.mark.parametrize(
    ("procedure", "out_col"),
    [
        ("closeness_centrality", "closeness_centrality"),
        ("core_number", "core_number"),
        ("degree_centrality", "degree_centrality"),
        ("eigenvector_centrality({directed: false})", "eigenvector_centrality"),
        ("katz_centrality({directed: false, alpha: 0.1})", "katz_centrality"),
    ],
)
def test_string_cypher_executes_networkx_node_scalar_parity_calls(procedure: str, out_col: str) -> None:
    pytest.importorskip("networkx")

    result = _mk_simple_path_graph().gfql(
        f"CALL graphistry.nx.{procedure} "
        f"YIELD nodeId, {out_col} "
        f"RETURN nodeId, {out_col} "
        "ORDER BY nodeId ASC"
    )

    assert result._edges.empty
    assert list(result._nodes.columns) == ["nodeId", out_col]
    assert result._nodes["nodeId"].tolist() == ["a", "b", "c"]
    assert result._nodes[out_col].notna().all()


def test_string_cypher_executes_networkx_connected_components_row_call() -> None:
    pytest.importorskip("networkx")

    result = _mk_path_with_isolate_graph().gfql(
        "CALL graphistry.nx.connected_components({directed: false}) "
        "YIELD nodeId, labels "
        "RETURN nodeId, labels "
        "ORDER BY nodeId ASC"
    )

    assert result._edges.empty
    rows = result._nodes.to_dict(orient="records")
    assert rows[:3] == [
        {"nodeId": "a", "labels": 0},
        {"nodeId": "b", "labels": 0},
        {"nodeId": "c", "labels": 0},
    ]
    assert rows[3] == {"nodeId": "z", "labels": 1}


def test_string_cypher_executes_networkx_strongly_connected_components_row_call() -> None:
    pytest.importorskip("networkx")

    result = _mk_simple_path_graph().gfql(
        "CALL graphistry.nx.strongly_connected_components() "
        "YIELD nodeId, labels "
        "RETURN nodeId, labels "
        "ORDER BY nodeId ASC"
    )

    assert result._edges.empty
    assert result._nodes["nodeId"].tolist() == ["a", "b", "c"]
    assert len(set(result._nodes["labels"].tolist())) == 3


def test_string_cypher_executes_networkx_hits_row_and_write_calls() -> None:
    pytest.importorskip("networkx")

    rows = _mk_simple_path_graph().gfql(
        "CALL graphistry.nx.hits() "
        "YIELD nodeId, hubs, authorities "
        "RETURN nodeId, hubs, authorities "
        "ORDER BY nodeId ASC"
    )

    assert rows._edges.empty
    assert list(rows._nodes.columns) == ["nodeId", "hubs", "authorities"]
    assert rows._nodes[["hubs", "authorities"]].notna().all().all()

    graph = _mk_simple_path_graph().gfql("CALL graphistry.nx.hits.write()")
    assert {"hubs", "authorities"}.issubset(set(graph._nodes.columns))
    assert graph._nodes[["hubs", "authorities"]].notna().all().all()
    assert not graph._edges.empty


def test_string_cypher_executes_networkx_edge_row_call() -> None:
    pytest.importorskip("networkx")

    result = _mk_simple_path_graph().gfql(
        "CALL graphistry.nx.edge_betweenness_centrality() "
        "YIELD source, destination, edge_betweenness_centrality "
        "RETURN source, destination, edge_betweenness_centrality "
        "ORDER BY source, destination"
    )

    assert result._edges.empty
    assert result._nodes.to_dict(orient="records") == [
        {"source": "a", "destination": "b", "edge_betweenness_centrality": pytest.approx(1.0 / 3.0)},
        {"source": "b", "destination": "c", "edge_betweenness_centrality": pytest.approx(1.0 / 3.0)},
    ]


def test_string_cypher_executes_networkx_edge_write_call() -> None:
    pytest.importorskip("networkx")

    result = _mk_simple_path_graph().gfql("CALL graphistry.nx.edge_betweenness_centrality.write()")

    assert "edge_betweenness_centrality" in result._edges.columns
    assert result._edges["edge_betweenness_centrality"].gt(0).all()


def test_string_cypher_executes_networkx_edge_write_call_with_out_col() -> None:
    pytest.importorskip("networkx")

    result = _mk_simple_path_graph().gfql("CALL graphistry.nx.edge_betweenness_centrality.write({out_col: 'ebc'})")

    assert "ebc" in result._edges.columns
    assert "edge_betweenness_centrality" not in result._edges.columns
    assert result._edges["ebc"].gt(0).all()


def test_string_cypher_networkx_bad_params_raise_structured_error() -> None:
    pytest.importorskip("networkx")

    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_simple_path_graph().gfql("CALL graphistry.nx.betweenness_centrality({bogus_option: 1})")

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "call.args"
    assert exc_info.value.context["value"] == {"bogus_option": 1}


@pytest.mark.parametrize("procedure", ["degree_centrality", "connected_components", "core_number", "strongly_connected_components"])
def test_string_cypher_networkx_no_param_algorithms_reject_extra_params(procedure: str) -> None:
    pytest.importorskip("networkx")

    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_simple_path_graph().gfql(f"CALL graphistry.nx.{procedure}({{bogus_option: 1}})")

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "call.args"
    assert exc_info.value.context["value"] == {"bogus_option": 1}


def test_string_cypher_executes_networkx_graph_write_call() -> None:
    pytest.importorskip("networkx")

    result = _mk_path_with_isolate_graph().gfql(
        "CALL graphistry.nx.k_core.write({k: 1, directed: false})"
    )

    assert set(result._nodes["id"]) == {"a", "b", "c"}
    assert result._edges.to_dict(orient="records") == [
        {"s": "a", "d": "b"},
        {"s": "b", "d": "c"},
    ]


def test_networkx_component_labels_assigns_dense_labels_without_networkx_dependency() -> None:
    labels = _networkx_component_labels([{"a", "b"}, {"z"}])

    assert labels["a"] == labels["b"]
    assert labels["z"] == 1
    assert set(labels.values()) == {0, 1}


def test_networkx_feature_guard_reports_installed_version_structurally() -> None:
    compiled_call = CompiledCypherProcedureCall(
        procedure="graphistry.nx.future_algorithm",
        backend="networkx",
        algorithm="future_algorithm",
        line=4,
        column=9,
    )
    nx_stub = type("NetworkXStub", (), {"__version__": "1.2.3"})()

    with pytest.raises(GFQLValidationError) as exc_info:
        _ensure_networkx_feature(
            False,
            compiled_call,
            "networkx.future_algorithm",
            nx_stub,
        )

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "call"
    assert exc_info.value.context["value"] == "graphistry.nx.future_algorithm"
    assert "1.2.3" in exc_info.value.context["suggestion"]
    assert exc_info.value.context["line"] == 4
    assert exc_info.value.context["column"] == 9


def test_networkx_optional_dependency_policy_matches_setup_extras() -> None:
    setup_py = Path(__file__).parents[5] / "setup.py"
    setup_text = setup_py.read_text()

    for requirement in ("networkx" + NETWORKX_VERSION_SPEC, *NETWORKX_SCIPY_EXTRA_REQUIREMENTS):
        assert requirement in setup_text


def test_networkx_version_policy_accepts_supported_lower_bound_structurally() -> None:
    compiled_call = CompiledCypherProcedureCall(
        procedure="graphistry.nx.pagerank",
        backend="networkx",
        algorithm="pagerank",
        line=2,
        column=5,
    )
    nx_stub = type("NetworkXStub", (), {"__version__": "2.5"})()

    _ensure_networkx_version_policy(compiled_call, nx_stub)


def test_networkx_version_policy_rejects_unsupported_combination_structured() -> None:
    compiled_call = CompiledCypherProcedureCall(
        procedure="graphistry.nx.pagerank",
        backend="networkx",
        algorithm="pagerank",
        line=2,
        column=5,
    )
    nx_stub = type("NetworkXStub", (), {"__version__": "2.4"})()

    with pytest.raises(GFQLValidationError) as exc_info:
        _ensure_networkx_version_policy(compiled_call, nx_stub)

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "call"
    assert exc_info.value.context["value"] == "graphistry.nx.pagerank"
    assert f"networkx{NETWORKX_VERSION_SPEC}" in exc_info.value.context["suggestion"]
    assert exc_info.value.context["line"] == 2
    assert exc_info.value.context["column"] == 5


def test_networkx_scipy_policy_rejects_unsupported_installed_scipy_structured() -> None:
    compiled_call = CompiledCypherProcedureCall(
        procedure="graphistry.nx.hits",
        backend="networkx",
        algorithm="hits",
        line=3,
        column=7,
    )
    scipy_stub = type("SciPyStub", (), {"__version__": "2.0.0"})()

    with pytest.raises(GFQLValidationError) as exc_info:
        _ensure_scipy_version_policy(compiled_call, scipy_stub)

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "call"
    assert exc_info.value.context["value"] == "graphistry.nx.hits"
    assert f"scipy{SCIPY_VERSION_SPEC}" in exc_info.value.context["suggestion"]
    assert exc_info.value.context["line"] == 3
    assert exc_info.value.context["column"] == 7


def test_networkx_hits_fallback_scores_directed_graph_without_scipy_dependency() -> None:
    graph = _MiniNxGraph(
        [("a", "b"), ("a", "c"), ("b", "c")],
        nodes=["a", "b", "c"],
    )

    hubs, authorities = _networkx_hits_scores(graph, max_iter=50)

    assert set(hubs) == {"a", "b", "c"}
    assert set(authorities) == {"a", "b", "c"}
    assert sum(hubs.values()) == pytest.approx(1.0)
    assert sum(authorities.values()) == pytest.approx(1.0)
    assert hubs["a"] >= hubs["c"]
    assert authorities["c"] >= authorities["a"]


def test_networkx_hits_fallback_handles_empty_and_zero_start_graphs() -> None:
    assert _networkx_hits_scores(_MiniNxGraph([], nodes=[])) == ({}, {})

    graph = _MiniNxGraph(
        [("a", "b"), ("b", "c")],
        nodes=["a", "b", "c"],
    )

    hubs, authorities = _networkx_hits_scores(
        graph,
        max_iter=10,
        nstart={"a": 0, "b": 0, "c": 0},
        normalized=False,
    )

    assert set(hubs) == {"a", "b", "c"}
    assert set(authorities) == {"a", "b", "c"}


def test_networkx_pagerank_fallback_scores_directed_path_without_networkx_dependency() -> None:
    graph = _MiniNxGraph(
        [("a", "b"), ("b", "c")],
        nodes=["a", "b", "c"],
    )

    scores = _networkx_pagerank_scores(graph, max_iter=50)

    assert set(scores) == {"a", "b", "c"}
    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores["c"] > scores["b"] > scores["a"]


def test_string_cypher_executes_real_cugraph_node_row_call_on_cudf() -> None:
    pytest.importorskip("cugraph")

    result = _mk_simple_path_graph_cudf().gfql(
        "CALL graphistry.cugraph.pagerank() "
        "YIELD nodeId, pagerank "
        "RETURN nodeId, pagerank "
        "ORDER BY pagerank DESC, nodeId ASC"
    )

    rows = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert result._edges.empty
    assert [row["nodeId"] for row in rows] == ["c", "b", "a"]
    assert rows[0]["pagerank"] > rows[1]["pagerank"] > rows[2]["pagerank"] > 0


def test_string_cypher_executes_real_cugraph_node_multi_column_row_call_on_cudf() -> None:
    pytest.importorskip("cugraph")

    result = _mk_simple_path_graph_cudf().gfql(
        "CALL graphistry.cugraph.hits() "
        "YIELD nodeId, hits, authorities "
        "RETURN nodeId, hits, authorities "
        "ORDER BY nodeId"
    )

    rows = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert result._edges.empty
    assert [row["nodeId"] for row in rows] == ["a", "b", "c"]
    assert all("hits" in row and "authorities" in row for row in rows)


def test_string_cypher_executes_real_cugraph_edge_row_call_on_cudf() -> None:
    pytest.importorskip("cugraph")

    result = _mk_simple_path_graph_cudf().gfql(
        "CALL graphistry.cugraph.edge_betweenness_centrality() "
        "YIELD source, destination, edge_betweenness_centrality "
        "RETURN source, destination, edge_betweenness_centrality "
        "ORDER BY source, destination"
    )

    rows = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert result._edges.empty
    assert [(row["source"], row["destination"]) for row in rows] == [("a", "b"), ("b", "c")]
    assert rows[0]["edge_betweenness_centrality"] == pytest.approx(1.0 / 3.0)
    assert rows[1]["edge_betweenness_centrality"] == pytest.approx(1.0 / 3.0)


def test_string_cypher_executes_real_cugraph_edge_write_call_on_cudf() -> None:
    pytest.importorskip("cugraph")

    result = _mk_simple_path_graph_cudf().gfql("CALL graphistry.cugraph.edge_betweenness_centrality.write()")

    edges_pdf = _to_pandas_df(result._edges).sort_values(["s", "d"]).reset_index(drop=True)
    assert "edge_betweenness_centrality" in edges_pdf.columns
    assert edges_pdf["edge_betweenness_centrality"].tolist() == [pytest.approx(1.0 / 3.0), pytest.approx(1.0 / 3.0)]


def test_string_cypher_executes_real_cugraph_node_multi_column_write_call_on_cudf() -> None:
    pytest.importorskip("cugraph")

    result = _mk_simple_path_graph_cudf().gfql("CALL graphistry.cugraph.hits.write()")

    nodes_pdf = _to_pandas_df(result._nodes).sort_values("id").reset_index(drop=True)
    assert "hits" in nodes_pdf.columns
    assert "authorities" in nodes_pdf.columns


def test_string_cypher_executes_real_cugraph_graph_write_call_on_cudf() -> None:
    pytest.importorskip("cugraph")

    result = _mk_path_with_isolate_graph_cudf().gfql(
        "CALL graphistry.cugraph.k_core.write({k: 1, directed: false})"
    )

    nodes_pdf = _to_pandas_df(result._nodes).sort_values("id").reset_index(drop=True)
    edges_pdf = _to_pandas_df(result._edges).sort_values(["s", "d"]).reset_index(drop=True)
    assert set(nodes_pdf["id"]) == {"a", "b", "c", "z"}
    assert edges_pdf.to_dict(orient="records") == [
        {"s": "a", "d": "b"},
        {"s": "b", "d": "c"},
    ]


def test_string_cypher_executes_real_igraph_graph_write_call() -> None:
    pytest.importorskip("igraph")

    result = _mk_triangle_graph().gfql("CALL graphistry.igraph.spanning_tree.write()")

    nodes_pdf = _to_pandas_df(result._nodes).sort_values("id").reset_index(drop=True)
    edges_pdf = _to_pandas_df(result._edges).sort_values(["s", "d"]).reset_index(drop=True)
    edge_pairs = set(zip(edges_pdf["s"], edges_pdf["d"]))
    assert set(nodes_pdf["id"]) == {"a", "b", "c"}
    assert len(edges_pdf) == 2
    assert edge_pairs.issubset({("a", "b"), ("a", "c"), ("b", "c")})


def test_string_cypher_executes_cugraph_node_row_call_via_shared_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_compute_cugraph(self, alg: str, out_col: str | None = None, **kwargs):
        assert alg == "louvain"
        assert kwargs == {"params": {"resolution": 1.0}}
        graph_with_nodes = self.materialize_nodes()
        assert graph_with_nodes._nodes is not None
        assert graph_with_nodes._node is not None
        out_name = out_col or alg
        nodes_df = graph_with_nodes._nodes.assign(**{out_name: [0, 1, 1]})
        return graph_with_nodes.nodes(nodes_df, graph_with_nodes._node)

    monkeypatch.setattr(_CypherTestGraph, "compute_cugraph", fake_compute_cugraph)

    result = _mk_simple_path_graph().gfql(
        "CALL graphistry.cugraph.louvain({resolution: 1.0}) "
        "YIELD nodeId, louvain "
        "RETURN nodeId, louvain "
        "ORDER BY nodeId"
    )

    assert result._edges.empty
    assert result._nodes.to_dict(orient="records") == [
        {"nodeId": "a", "louvain": 0},
        {"nodeId": "b", "louvain": 1},
        {"nodeId": "c", "louvain": 1},
    ]


def test_string_cypher_executes_cugraph_node_multi_column_row_call_via_shared_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_compute_cugraph(self, alg: str, out_col: str | None = None, **kwargs):
        assert alg == "hits"
        assert out_col is None
        assert kwargs == {}
        graph_with_nodes = self.materialize_nodes()
        assert graph_with_nodes._nodes is not None
        assert graph_with_nodes._node is not None
        nodes_df = graph_with_nodes._nodes.assign(hubs=[0.1, 0.7, 0.2], authorities=[0.2, 0.6, 0.2])
        return graph_with_nodes.nodes(nodes_df, graph_with_nodes._node)

    monkeypatch.setattr(_CypherTestGraph, "compute_cugraph", fake_compute_cugraph)

    result = _mk_simple_path_graph().gfql(
        "CALL graphistry.cugraph.hits() "
        "YIELD nodeId, hits, authorities "
        "RETURN nodeId, hits, authorities "
        "ORDER BY nodeId"
    )

    assert result._edges.empty
    assert result._nodes.to_dict(orient="records") == [
        {"nodeId": "a", "hits": 0.1, "authorities": 0.2},
        {"nodeId": "b", "hits": 0.7, "authorities": 0.6},
        {"nodeId": "c", "hits": 0.2, "authorities": 0.2},
    ]


def test_string_cypher_executes_cugraph_edge_row_call_via_shared_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_compute_cugraph(self, alg: str, out_col: str | None = None, **kwargs):
        assert alg == "edge_betweenness_centrality"
        assert out_col is None
        assert kwargs == {}
        assert self._edges is not None
        edges_df = self._edges.assign(edge_betweenness_centrality=[0.25, 0.75])
        return self.edges(edges_df, self._source, self._destination)

    monkeypatch.setattr(_CypherTestGraph, "compute_cugraph", fake_compute_cugraph)

    result = _mk_simple_path_graph().gfql(
        "CALL graphistry.cugraph.edge_betweenness_centrality() "
        "YIELD source, destination, edge_betweenness_centrality "
        "RETURN source, destination, edge_betweenness_centrality "
        "ORDER BY source, destination"
    )

    assert result._edges.empty
    assert result._nodes.to_dict(orient="records") == [
        {"source": "a", "destination": "b", "edge_betweenness_centrality": 0.25},
        {"source": "b", "destination": "c", "edge_betweenness_centrality": 0.75},
    ]


def test_string_cypher_executes_cugraph_edge_write_call_via_shared_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_compute_cugraph(self, alg: str, out_col: str | None = None, **kwargs):
        assert alg == "edge_betweenness_centrality"
        assert kwargs == {"params": {"k": 2}}
        assert self._edges is not None
        out_name = out_col or alg
        edges_df = self._edges.assign(**{out_name: [0.25, 0.75]})
        return self.edges(edges_df, self._source, self._destination)

    monkeypatch.setattr(_CypherTestGraph, "compute_cugraph", fake_compute_cugraph)

    result = _mk_simple_path_graph().gfql(
        "CALL graphistry.cugraph.edge_betweenness_centrality.write({k: 2})"
    )

    assert "edge_betweenness_centrality" in result._edges.columns
    assert result._edges["edge_betweenness_centrality"].tolist() == [0.25, 0.75]


def test_string_cypher_executes_cugraph_graph_write_call_via_shared_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_compute_cugraph(self, alg: str, out_col: str | None = None, **kwargs):
        assert alg == "k_core"
        assert out_col is None
        assert kwargs == {}
        trimmed_nodes = pd.DataFrame({"id": ["a", "b", "c"]})
        trimmed_edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
        return self.nodes(trimmed_nodes, "id").edges(trimmed_edges, "s", "d")

    monkeypatch.setattr(_CypherTestGraph, "compute_cugraph", fake_compute_cugraph)

    result = _mk_path_with_isolate_graph().gfql("CALL graphistry.cugraph.k_core.write()")

    assert set(result._nodes["id"]) == {"a", "b", "c"}
    assert result._edges.to_dict(orient="records") == [
        {"s": "a", "d": "b"},
        {"s": "b", "d": "c"},
    ]


def test_string_cypher_rejects_igraph_graph_only_row_call_after_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_compute_igraph(self, alg: str, out_col: str | None = None, **kwargs):
        assert alg == "spanning_tree"
        assert out_col is None
        assert kwargs == {}
        trimmed_edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
        return self.nodes(pd.DataFrame({"id": ["a", "b", "c"]}), "id").edges(trimmed_edges, "s", "d")

    monkeypatch.setattr(_CypherTestGraph, "compute_igraph", fake_compute_igraph)

    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_triangle_graph().gfql("CALL graphistry.igraph.spanning_tree()")

    assert exc_info.value.code == ErrorCode.E108


def test_string_cypher_executes_igraph_graph_write_call_via_shared_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_compute_igraph(self, alg: str, out_col: str | None = None, **kwargs):
        assert alg == "spanning_tree"
        assert out_col is None
        assert kwargs == {}
        trimmed_edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
        return self.nodes(pd.DataFrame({"id": ["a", "b", "c"]}), "id").edges(trimmed_edges, "s", "d")

    monkeypatch.setattr(_CypherTestGraph, "compute_igraph", fake_compute_igraph)

    result = _mk_triangle_graph().gfql("CALL graphistry.igraph.spanning_tree.write()")

    assert set(result._nodes["id"]) == {"a", "b", "c"}
    assert result._edges.to_dict(orient="records") == [
        {"s": "a", "d": "b"},
        {"s": "b", "d": "c"},
    ]


def test_cypher_to_gfql_uses_terminal_with_projection() -> None:
    chain = cypher_to_gfql("MATCH (p) WITH p.name AS person_name ORDER BY person_name ASC LIMIT 1")

    assert isinstance(chain.chain[1], ASTCall)
    assert chain.chain[1].function == "rows"
    assert isinstance(chain.chain[2], ASTCall)
    assert chain.chain[2].function == "with_"
    assert chain.chain[2].params["items"] == [("person_name", "p.name")]


def test_cypher_to_gfql_supports_with_then_return_pipeline() -> None:
    chain = cypher_to_gfql("UNWIND [1, 3, 2] AS ints WITH ints ORDER BY ints DESC LIMIT 2 RETURN ints")

    functions = [cast(ASTCall, step).function for step in chain.chain]
    assert functions == ["rows", "unwind", "with_", "order_by", "limit", "select"]
    assert cast(ASTCall, chain.chain[2]).params["items"] == [("ints", "ints")]
    assert cast(ASTCall, chain.chain[3]).params["keys"] == [("ints", "desc")]


def test_string_cypher_executes_with_then_return_pipeline() -> None:
    _assert_query_rows(
        "UNWIND [1, 3, 2] AS ints WITH ints ORDER BY ints DESC LIMIT 2 RETURN ints",
        [{"ints": 3}, {"ints": 2}],
    )


def test_string_cypher_executes_multiple_with_stages() -> None:
    _assert_query_rows(
        "WITH 1 AS a, 'b' AS b WITH a ORDER BY a ASCENDING WITH a RETURN a",
        [{"a": 1}],
    )


def test_string_cypher_executes_interleaved_row_only_with_unwind_pipeline() -> None:
    _assert_query_rows(
        "WITH [0, 1] AS prows, [[2], [3, 4]] AS qrows "
        "UNWIND prows AS p "
        "UNWIND qrows[p] AS q "
        "WITH p, count(q) AS rng "
        "RETURN p "
        "ORDER BY rng",
        [{"p": 0}, {"p": 1}],
    )


def test_string_cypher_supports_range_comparison_after_collect() -> None:
    _assert_query_rows(
        "WITH [1, 3, 2] AS values "
        "WITH values, size(values) AS numOfValues "
        "UNWIND values AS value "
        "WITH size([x IN values WHERE x < value]) AS x, value, numOfValues "
        "ORDER BY value "
        "WITH numOfValues, collect(x) AS orderedX "
        "RETURN orderedX = range(0, numOfValues - 1) AS equal",
        [{"equal": True}],
    )


def test_string_cypher_supports_range_with_varying_row_bounds_and_steps() -> None:
    _assert_query_rows(
        "WITH ["
        "{id: 'a', start: 0, stop: 3, step: 1}, "
        "{id: 'b', start: 0, stop: -1, step: -1}, "
        "{id: 'c', start: 10, stop: 4, step: -3}, "
        "{id: 'd', start: 2, stop: 2, step: 5}"
        "] AS rows "
        "UNWIND rows AS row "
        "RETURN row.id AS id, range(row.start, row.stop, row.step) AS vals "
        "ORDER BY id",
        [
        {"id": "a", "vals": [0, 1, 2, 3]},
        {"id": "b", "vals": [0, -1]},
        {"id": "c", "vals": [10, 7, 4]},
        {"id": "d", "vals": [2]},
        ],
    )


@pytest.mark.parametrize(
    ("query", "pattern"),
    [
        ("RETURN range(2, 8, 0)", "range\\(\\) step must be non-zero"),
        ("RETURN range(true, 1, 1)", "range\\(\\) start must be an integer"),
        ("RETURN range(0, 1.0, 1)", "range\\(\\) stop must be an integer"),
    ],
)
def test_string_cypher_rejects_invalid_range_arguments(query: str, pattern: str) -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLTypeError, match=pattern):
        g.gfql(query)


_TEMPORAL7_COMPARISON_CASES: Tuple[Tuple[str, str, Dict[str, bool]], ...] = (
    (
        "date-before",
        "WITH date({year: 1980, month: 12, day: 24}) AS x, "
        "date({year: 1984, month: 10, day: 11}) AS d "
        "RETURN x > d, x < d, x >= d, x <= d, x = d",
        {"x > d": False, "x < d": True, "x >= d": False, "x <= d": True, "x = d": False},
    ),
    (
        "date-equal",
        "WITH date({year: 1984, month: 10, day: 11}) AS x, "
        "date({year: 1984, month: 10, day: 11}) AS d "
        "RETURN x > d, x < d, x >= d, x <= d, x = d",
        {"x > d": False, "x < d": False, "x >= d": True, "x <= d": True, "x = d": True},
    ),
    (
        "localtime-before",
        "WITH localtime({hour: 10, minute: 35}) AS x, "
        "localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d "
        "RETURN x > d, x < d, x >= d, x <= d, x = d",
        {"x > d": False, "x < d": True, "x >= d": False, "x <= d": True, "x = d": False},
    ),
    (
        "localtime-equal",
        "WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS x, "
        "localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d "
        "RETURN x > d, x < d, x >= d, x <= d, x = d",
        {"x > d": False, "x < d": False, "x >= d": True, "x <= d": True, "x = d": True},
    ),
    (
        "time-before-offset",
        "WITH time({hour: 10, minute: 0, timezone: '+01:00'}) AS x, "
        "time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'}) AS d "
        "RETURN x > d, x < d, x >= d, x <= d, x = d",
        {"x > d": False, "x < d": True, "x >= d": False, "x <= d": True, "x = d": False},
    ),
    (
        "time-equal-offset",
        "WITH time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'}) AS x, "
        "time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'}) AS d "
        "RETURN x > d, x < d, x >= d, x <= d, x = d",
        {"x > d": False, "x < d": False, "x >= d": True, "x <= d": True, "x = d": True},
    ),
    (
        "localdatetime-before",
        "WITH localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14}) AS x, "
        "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d "
        "RETURN x > d, x < d, x >= d, x <= d, x = d",
        {"x > d": False, "x < d": True, "x >= d": False, "x <= d": True, "x = d": False},
    ),
    (
        "localdatetime-equal",
        "WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS x, "
        "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d "
        "RETURN x > d, x < d, x >= d, x <= d, x = d",
        {"x > d": False, "x < d": False, "x >= d": True, "x <= d": True, "x = d": True},
    ),
    (
        "datetime-before-offset",
        "WITH datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '+00:00'}) AS x, "
        "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'}) AS d "
        "RETURN x > d, x < d, x >= d, x <= d, x = d",
        {"x > d": False, "x < d": True, "x >= d": False, "x <= d": True, "x = d": False},
    ),
    (
        "datetime-equal-offset",
        "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'}) AS x, "
        "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'}) AS d "
        "RETURN x > d, x < d, x >= d, x <= d, x = d",
        {"x > d": False, "x < d": False, "x >= d": True, "x <= d": True, "x = d": True},
    ),
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [(query, expected) for _, query, expected in _TEMPORAL7_COMPARISON_CASES],
    ids=[case_id for case_id, _, _ in _TEMPORAL7_COMPARISON_CASES],
)
def test_string_cypher_temporal7_comparison_truth_table(
    query: str,
    expected: dict[str, bool],
) -> None:
    _assert_query_rows(query, [expected])


_TEMPORAL_LITERAL_CONSTRUCTOR_COMPARISON_CASES: Tuple[Tuple[str, str, Dict[str, bool]], ...] = (
    (
        "date-literal-constructor-equal",
        "WITH date('1984-10-11') AS x, "
        "date({year: 1984, month: 10, day: 11}) AS d "
        "RETURN x = d AS eq, x <= d AS le, x >= d AS ge",
        {"eq": True, "le": True, "ge": True},
    ),
    (
        "time-literal-constructor-equal-offset",
        "WITH time('10:00:00+01:00') AS x, "
        "time({hour: 9, minute: 0, timezone: '+00:00'}) AS d "
        "RETURN x = d AS eq, x <= d AS le, x >= d AS ge",
        {"eq": True, "le": True, "ge": True},
    ),
    (
        "datetime-literal-constructor-equal-offset",
        "WITH datetime('1984-10-11T12:31:14+05:00') AS x, "
        "datetime({year: 1984, month: 10, day: 11, hour: 7, minute: 31, second: 14, timezone: '+00:00'}) AS d "
        "RETURN x = d AS eq, x <= d AS le, x >= d AS ge",
        {"eq": True, "le": True, "ge": True},
    ),
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [(query, expected) for _, query, expected in _TEMPORAL_LITERAL_CONSTRUCTOR_COMPARISON_CASES],
    ids=[case_id for case_id, _, _ in _TEMPORAL_LITERAL_CONSTRUCTOR_COMPARISON_CASES],
)
def test_string_cypher_temporal_comparison_mixes_literal_and_constructor_forms(
    query: str,
    expected: dict[str, bool],
) -> None:
    _assert_query_rows(query, [expected])


_CUDF_TEMPORAL_COMPARISON_CASES = (
    _TEMPORAL7_COMPARISON_CASES
    + _TEMPORAL_LITERAL_CONSTRUCTOR_COMPARISON_CASES
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [(query, expected) for _, query, expected in _CUDF_TEMPORAL_COMPARISON_CASES],
    ids=[case_id for case_id, _, _ in _CUDF_TEMPORAL_COMPARISON_CASES],
)
def test_string_cypher_temporal_comparison_cases_cudf(
    query: str,
    expected: dict[str, bool],
) -> None:
    _require_cudf_runtime()
    graph = _mk_cudf_graph(
        pd.DataFrame({"id": []}),
        pd.DataFrame({"s": [], "d": []}),
    )

    assert type(graph._nodes).__module__.startswith("cudf")
    assert type(graph._edges).__module__.startswith("cudf")
    result = graph.gfql(query, engine="cudf")
    assert type(result._nodes).__module__.startswith("cudf")

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [expected]


_TEMPORAL_PROPERTY_ORDER_BY_CASES: Tuple[Tuple[str, pd.DataFrame, List[str]], ...] = (
    (
        "date-property-order",
        pd.DataFrame(
            {
                "id": ["early", "same", "late"],
                "v": [
                    "date({year: 1980, month: 12, day: 24})",
                    "date({year: 1984, month: 10, day: 11})",
                    "date({year: 1985, month: 5, day: 6})",
                ],
            }
        ),
        ["early", "same", "late"],
    ),
    (
        "time-property-order-offset",
        pd.DataFrame(
            {
                "id": ["offset", "later", "earliest"],
                "v": [
                    "time({hour: 10, minute: 0, timezone: '+01:00'})",
                    "time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'})",
                    "time({hour: 8, minute: 0, timezone: '+00:00'})",
                ],
            }
        ),
        ["earliest", "offset", "later"],
    ),
    (
        "datetime-property-order-offset",
        pd.DataFrame(
            {
                "id": ["offset", "after", "earliest"],
                "v": [
                    "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'})",
                    "datetime({year: 1984, month: 10, day: 11, hour: 8, minute: 31, second: 14, timezone: '+00:00'})",
                    "datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '+00:00'})",
                ],
            }
        ),
        ["earliest", "offset", "after"],
    ),
)


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
@pytest.mark.parametrize(
    ("nodes_df", "expected"),
    [(nodes_df, expected) for _, nodes_df, expected in _TEMPORAL_PROPERTY_ORDER_BY_CASES],
    ids=[case_id for case_id, _, _ in _TEMPORAL_PROPERTY_ORDER_BY_CASES],
)
def test_string_cypher_temporal_property_order_by_cases_on_engine(
    nodes_df: pd.DataFrame,
    expected: list[str],
    engine: Optional[str],
) -> None:
    edges_df = pd.DataFrame({"s": [], "d": []})
    query = "MATCH (n) RETURN n.id AS id, n.v AS v ORDER BY v ASC"
    if engine == "cudf":
        _require_cudf_runtime()
        graph = _mk_cudf_graph(nodes_df, edges_df)
        assert type(graph._nodes).__module__.startswith("cudf")
        assert type(graph._edges).__module__.startswith("cudf")
        result = graph.gfql(query, engine="cudf")
        assert type(result._nodes).__module__.startswith("cudf")
    else:
        result = _mk_graph(nodes_df, edges_df).gfql(query)

    assert _to_pandas_df(result._nodes)["id"].tolist() == expected


def test_string_cypher_supports_time_comparison_consistent_with_sort_order() -> None:
    _assert_query_rows(
        "WITH ["
        "time({hour: 10, minute: 35, timezone: '-08:00'}), "
        "time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), "
        "time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}), "
        "time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}), "
        "time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'}), "
        "time({hour: 12, minute: 35, second: 15, timezone: '+01:00'})"
        "] AS values "
        "WITH values, size(values) AS numOfValues "
        "UNWIND values AS value "
        "WITH size([x IN values WHERE x < value]) AS x, value, numOfValues "
        "ORDER BY value "
        "WITH numOfValues, collect(x) AS orderedX "
        "RETURN orderedX = range(0, numOfValues - 1) AS equal",
        [{"equal": True}],
    )


def test_string_cypher_supports_datetime_comparison_consistent_with_sort_order() -> None:
    _assert_query_rows(
        "WITH ["
        "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}), "
        "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}), "
        "datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}), "
        "datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}), "
        "datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})"
        "] AS values "
        "WITH values, size(values) AS numOfValues "
        "UNWIND values AS value "
        "WITH size([x IN values WHERE x < value]) AS x, value, numOfValues "
        "ORDER BY value "
        "WITH numOfValues, collect(x) AS orderedX "
        "RETURN orderedX = range(0, numOfValues - 1) AS equal",
        [{"equal": True}],
    )


def test_string_cypher_supports_datetime_comparison_with_extended_positive_years() -> None:
    _assert_query_rows(
        "RETURN datetime('9999-01-01T00:00:00Z') < datetime('+10000-01-01T00:00:00Z') AS b",
        [{"b": True}],
    )


def test_string_cypher_orders_extended_positive_year_datetimes_temporally() -> None:
    _assert_query_rows(
        "WITH [datetime('9999-01-01T00:00:00Z'), datetime('+10000-01-01T00:00:00Z')] AS values "
        "UNWIND values AS value "
        "RETURN value ORDER BY value ASC",
        [
            {"value": "9999-01-01T00:00:00Z"},
            {"value": "10000-01-01T00:00:00Z"},
        ],
    )


def test_string_cypher_matches_labels_column_even_with_node_type_property() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "labels": [["Person"], ["Animal"]],
            "type": ["node", "node"],
            "name": ["alice", "bear"],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql("MATCH (n:Person) RETURN n.name AS name")

    assert result._nodes.to_dict(orient="records") == [{"name": "alice"}]


def test_string_cypher_supports_date_comparison_consistent_with_sort_order() -> None:
    _assert_query_rows(
        "WITH ["
        "date({year: 1910, month: 5, day: 6}), "
        "date({year: 1980, month: 12, day: 24}), "
        "date({year: 1984, month: 10, day: 12}), "
        "date({year: 1985, month: 5, day: 6}), "
        "date({year: 1980, month: 10, day: 24}), "
        "date({year: 1984, month: 10, day: 11})"
        "] AS values "
        "WITH values, size(values) AS numOfValues "
        "UNWIND values AS value "
        "WITH size([x IN values WHERE x < value]) AS x, value, numOfValues "
        "ORDER BY value "
        "WITH numOfValues, collect(x) AS orderedX "
        "RETURN orderedX = range(0, numOfValues - 1) AS equal",
        [{"equal": True}],
    )


def test_string_cypher_order_by_stringified_list_column_uses_list_orderability() -> None:
    """Thin cypher integration smoke for stringified-list ORDER BY semantics.

    Row-orderability specifics are covered in row-pipeline tests under
    `graphistry/tests/compute/gfql/row/test_ordering.py`.
    """
    g = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e"],
                "list": pd.Series(
                    ["[2, -2]", "[1, 2]", "[300, 0]", "[1, -20]", "[2, -2, 100]"],
                    dtype="string",
                ),
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    asc = g.gfql(
        "MATCH (a) "
        "WITH a, a.list AS list "
        "WITH a, list ORDER BY list ASC LIMIT 3 "
        "RETURN a, list"
    )
    asc_values = sorted(str(v) for v in asc._nodes["list"].tolist())
    assert asc_values == sorted(["[1, -20]", "[1, 2]", "[2, -2]"])

    desc = g.gfql(
        "MATCH (a) "
        "WITH a, a.list AS list "
        "WITH a, list ORDER BY list DESC LIMIT 3 "
        "RETURN a, list"
    )
    desc_values = sorted(str(v) for v in desc._nodes["list"].tolist())
    assert desc_values == sorted(["[300, 0]", "[2, -2, 100]", "[2, -2]"])


def test_string_cypher_supports_return_star_after_with_distinct_row_projection() -> None:
    g = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"], "name": ["A", "B", "C"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = g.gfql(
        "MATCH (a) "
        "WITH DISTINCT a.name AS name "
        "ORDER BY a.name DESC "
        "LIMIT 1 "
        "RETURN *"
    )

    assert result._nodes.to_dict(orient="records") == [{"name": "C"}]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "MATCH (a) "
            "WITH DISTINCT a.name2 AS name "
            "WHERE a.name2 = 'B' "
            "RETURN *",
            [{"name": "B"}],
        ),
        (
            "MATCH (a) "
            "WITH a.name2 AS name "
            "WHERE name = 'B' OR a.name2 = 'C' "
            "RETURN * "
            "ORDER BY name",
            [{"name": "B"}, {"name": "C"}],
        ),
    ],
)
def test_string_cypher_supports_with_where_using_projected_source_properties(
    query: str,
    expected: List[Dict[str, Any]],
) -> None:
    result = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"], "name2": ["A", "B", "C"]}),
        pd.DataFrame({"s": [], "d": []}),
    ).gfql(query)

    assert result._nodes.to_dict(orient="records") == expected


def test_string_cypher_rejects_out_of_scope_order_by_after_multiple_with_stages() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql("WITH 1 AS a, 'b' AS b, 3 AS c WITH a, b WITH a ORDER BY a, c RETURN a")

    assert exc_info.value.code == ErrorCode.E204
    assert exc_info.value.context["field"] == "identifier"
    assert exc_info.value.context["value"] == "c"
    assert exc_info.value.context["visible_scope"] == ["a", "b"]


def test_string_cypher_executes_row_column_expression_order_after_with() -> None:
    _assert_query_rows(
        "UNWIND [1, 2, 3] AS a WITH a ORDER BY a + 2 DESC, a ASC LIMIT 1 RETURN a",
        [{"a": 3}],
    )


def test_string_cypher_executes_match_with_then_return_pipeline() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "label__A": [True, True, True],
            "score": [5, 9, 1],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) WITH a ORDER BY a.score DESC LIMIT 2 RETURN a"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:A {score: 9})"},
        {"a": "(:A {score: 5})"},
    ]


def test_string_cypher_executes_match_with_expression_order_pipeline() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "type": ["A", "B", "C", "D", "E"],
            "bool": [True, False, False, True, True],
            "bool2": [True, False, True, True, False],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) WITH a ORDER BY NOT (a.bool AND a.bool2) LIMIT 2 RETURN a"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:A {bool: true, bool2: true})"},
        {"a": "(:D {bool: true, bool2: true})"},
    ]


def test_string_cypher_executes_match_with_arithmetic_order_pipeline() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "type": ["A", "B", "C", "D", "E"],
            "num": [9, 5, 30, -11, 7054],
            "num2": [5, 4, 3, 2, 1],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) WITH a ORDER BY (a.num2 + (a.num * 2)) * -1 LIMIT 3 RETURN a.id AS id"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"id": "e"},
        {"id": "c"},
        {"id": "a"},
    ]


def test_string_cypher_executes_match_with_constant_expression_order_pipeline() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b"],
            "num": [1, 1, 2, 2, 3, 3, 4, 4],
            "text": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) "
        "WITH a "
        "ORDER BY 4 + ((a.num * 2) % 2) ASC, a.num ASC, a.text ASC "
        "LIMIT 1 "
        "RETURN a"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [{"a": "({num: 1, text: 'a'})"}]


def test_string_cypher_executes_match_with_mixed_whole_row_bool_alias_pipeline() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "type": ["A", "B", "C", "D", "E"],
            "bool": [True, False, False, True, False],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) "
        "WITH a, a.bool AS bool "
        "WITH a, bool "
        "ORDER BY bool ASC "
        "LIMIT 3 "
        "RETURN a, bool"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:B {bool: false})", "bool": False},
        {"a": "(:C {bool: false})", "bool": False},
        {"a": "(:E {bool: false})", "bool": False},
    ]


def test_string_cypher_executes_match_with_mixed_whole_row_numeric_alias_pipeline() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "type": ["A", "B", "C", "D", "E"],
            "num": [9, 5, 30, -11, 7054],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) "
        "WITH a, a.num AS num "
        "WITH a, num "
        "ORDER BY num DESC "
        "LIMIT 3 "
        "RETURN a, num"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:E {num: 7054})", "num": 7054},
        {"a": "(:C {num: 30})", "num": 30},
        {"a": "(:A {num: 9})", "num": 9},
    ]


def test_string_cypher_executes_match_with_mixed_whole_row_computed_alias_pipeline() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "type": ["A", "B", "C"],
            "num": [1, 2, 3],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) "
        "WITH a, a.num + 1 AS score "
        "WITH a, score "
        "ORDER BY score DESC "
        "RETURN a, score"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:C {num: 3})", "score": 4},
        {"a": "(:B {num: 2})", "score": 3},
        {"a": "(:A {num: 1})", "score": 2},
    ]


def test_string_cypher_executes_with_orderby4_style_mixed_whole_row_pipeline() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "type": ["A", "A", "A", "A"],
            "num": [1, 4, 3, 2],
            "num2": [5, 1, 4, 2],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) "
        "WITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod "
        "ORDER BY mod, sum "
        "LIMIT 3 "
        "RETURN a, sum, mod"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [
        {"a": "(:A {num: 4, num2: 1})", "sum": 5, "mod": 1},
        {"a": "(:A {num: 3, num2: 4})", "sum": 7, "mod": 1},
        {"a": "(:A {num: 2, num2: 2})", "sum": 4, "mod": 2},
    ]


def test_string_cypher_executes_with_match_reentry_limit_shape() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a1", "a2", "b1", "b2"],
            "label__A": [True, True, False, False],
            "name": ["alpha", "beta", None, None],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a1", "a2"],
            "d": ["b1", "b2"],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) WITH a ORDER BY a.name LIMIT 1 MATCH (a)-->(b) RETURN a"
    )

    assert entity_text_records(result, {"a": "nodes"}) == [{"a": "(:A {name: 'alpha'})"}]


def test_string_cypher_executes_with_match_reentry_ordered_topk_multi_row_shape() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a1", "a2", "a3", "b1", "b2", "b3"],
            "label__A": [True, True, True, False, False, False],
            "num": [30, 20, 10, 1, 1, 1],
            "name": ["gamma", "beta", "alpha", None, None, None],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a1", "a2", "a3"],
            "d": ["b1", "b2", "b3"],
            "type": ["R", "R", "R"],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) "
        "WITH a "
        "ORDER BY a.num DESC "
        "LIMIT 2 "
        "MATCH (a)-->(b) "
        "RETURN a.id AS aid, b.id AS bid"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"aid": "a1", "bid": "b1"},
        {"aid": "a2", "bid": "b2"},
    ]


def test_string_cypher_executes_with_match_reentry_ordered_topk_multicolumn_shape() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a1", "a2", "a3", "b1", "b2", "b3"],
            "label__A": [True, True, True, False, False, False],
            "name": ["alpha", "alpha", "beta", None, None, None],
            "num": [2, 1, 99, 0, 0, 0],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a1", "a2", "a3"],
            "d": ["b1", "b2", "b3"],
            "type": ["R", "R", "R"],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) "
        "WITH a, a.name AS aname "
        "ORDER BY aname ASC, a.num DESC "
        "LIMIT 2 "
        "MATCH (a)-->(b) "
        "RETURN a.id AS aid, aname, b.id AS bid"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"aid": "a1", "aname": "alpha", "bid": "b1"},
        {"aid": "a2", "aname": "alpha", "bid": "b2"},
    ]


def test_string_cypher_executes_with_match_reentry_ordered_topk_with_carried_scalar_shape() -> None:
    result = _mk_reentry_carried_scalar_graph().gfql(
        "MATCH (a:A) "
        "WITH a, a.num AS property "
        "ORDER BY property DESC "
        "LIMIT 2 "
        "MATCH (a)-->(b) "
        "RETURN property, b.id AS bid "
        "ORDER BY bid"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"property": 1, "bid": "b1"},
        {"property": 2, "bid": "b2"},
    ]


def test_string_cypher_executes_with_match_reentry_ordered_limit_zero_shape() -> None:
    result = _mk_reentry_carried_scalar_graph().gfql(
        "MATCH (a:A) "
        "WITH a "
        "ORDER BY a.num DESC "
        "LIMIT 0 "
        "MATCH (a)-->(b) "
        "RETURN b.id AS bid"
    )

    assert result._nodes.to_dict(orient="records") == []


def test_string_cypher_executes_with_match_reentry_parameterized_limit_shape() -> None:
    """Parameterized LIMIT should be treated as bounded when params resolve to int."""
    nodes = pd.DataFrame(
        {
            "id": ["a1", "a2", "b1"],
            "label__A": [True, True, False],
            "name": ["alpha", "beta", None],
        }
    )
    edges = pd.DataFrame({"s": ["a1", "a2"], "d": ["b1", "b1"]})
    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) WITH a ORDER BY a.name LIMIT $n MATCH (a)-->(b) RETURN a",
        params={"n": 1},
    )
    assert entity_text_records(result, {"a": "nodes"}) == [{"a": "(:A {name: 'alpha'})"}]


def test_string_cypher_rejects_reentry_with_parameterized_non_int_limit_and_order() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a1", "a2", "b1"],
            "label__A": [True, True, False],
            "name": ["alpha", "beta", None],
        }
    )
    edges = pd.DataFrame({"s": ["a1", "a2"], "d": ["b1", "b1"]})
    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_graph(nodes, edges).gfql(
            "MATCH (a:A) WITH a ORDER BY a.name LIMIT $n MATCH (a)-->(b) RETURN a",
            params={"n": "1"},
        )
    assert "integer" in exc_info.value.message.lower()


def test_string_cypher_executes_with_match_reentry_limit_shape_on_cudf() -> None:
    cudf = _require_cudf_runtime()

    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["a1", "a2", "b1", "b2"],
                "label__A": [True, True, False, False],
                "name": ["alpha", "beta", None, None],
            }
        )
    )
    edges = cudf.from_pandas(
        pd.DataFrame(
            {
                "s": ["a1", "a2"],
                "d": ["b1", "b2"],
            }
        )
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) WITH a ORDER BY a.name LIMIT 1 MATCH (a)-->(b) RETURN a",
        engine="cudf",
    )

    assert type(result._nodes).__module__.startswith("cudf")
    assert entity_text_records(result, {"a": "nodes"}) == [{"a": "(:A {name: 'alpha'})"}]


def test_string_cypher_executes_with_match_reentry_ordered_topk_multi_row_shape_on_cudf() -> None:
    cudf = _require_cudf_runtime()

    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["a1", "a2", "a3", "b1", "b2", "b3"],
                "label__A": [True, True, True, False, False, False],
                "num": [30, 20, 10, 1, 1, 1],
                "name": ["gamma", "beta", "alpha", None, None, None],
            }
        )
    )
    edges = cudf.from_pandas(
        pd.DataFrame(
            {
                "s": ["a1", "a2", "a3"],
                "d": ["b1", "b2", "b3"],
                "type": ["R", "R", "R"],
            }
        )
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) "
        "WITH a "
        "ORDER BY a.num DESC "
        "LIMIT 2 "
        "MATCH (a)-->(b) "
        "RETURN a.id AS aid, b.id AS bid",
        engine="cudf",
    )

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"aid": "a1", "bid": "b1"},
        {"aid": "a2", "bid": "b2"},
    ]


def test_string_cypher_executes_with_match_reentry_parameterized_limit_shape_on_cudf() -> None:
    cudf = _require_cudf_runtime()

    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["a1", "a2", "b1"],
                "label__A": [True, True, False],
                "name": ["alpha", "beta", None],
            }
        )
    )
    edges = cudf.from_pandas(
        pd.DataFrame(
            {
                "s": ["a1", "a2"],
                "d": ["b1", "b1"],
            }
        )
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) WITH a ORDER BY a.name LIMIT $n MATCH (a)-->(b) RETURN a",
        params={"n": 1},
        engine="cudf",
    )

    assert type(result._nodes).__module__.startswith("cudf")
    assert entity_text_records(result, {"a": "nodes"}) == [{"a": "(:A {name: 'alpha'})"}]


def test_string_cypher_failfast_rejects_with_match_reentry_ordered_skip_shape() -> None:
    with pytest.raises(GFQLValidationError, match="preserve prefix WITH row ordering"):
        _mk_reentry_carried_scalar_graph().gfql(
            "MATCH (a:A) "
            "WITH a "
            "ORDER BY a.num DESC "
            "SKIP 1 "
            "LIMIT 1 "
            "MATCH (a)-->(b) "
            "RETURN b.id AS bid"
        )


def test_string_cypher_executes_with_match_reentry_multihop_shape() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "label__A": [True, False, False, False],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "b", "c"],
            "d": ["b", "c", "d"],
            "type": ["R", "R", "R"],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) WITH a MATCH (a)-[:R*2]->(b) RETURN b.id AS id"
    )

    assert result._nodes.to_dict(orient="records") == [{"id": "c"}]


@pytest.mark.parametrize(
    ("query", "expected_whole_row_output", "expected_columns"),
    [
        (
            _reentry_query("a, a.num AS property", return_clause="property", order_by="property DESC"),
            "a",
            ("property",),
        ),
        (
            _reentry_query(
                "a, a.num AS property, a.num + 10 AS property2",
                return_clause="property, property2",
                order_by="property DESC",
            ),
            "a",
            ("property", "property2"),
        ),
        (
            _reentry_query(
                "a AS x, a.num AS property",
                match_alias="x",
                return_clause="property",
                order_by="property DESC",
            ),
            "x",
            ("property",),
        ),
    ],
)
def test_compile_cypher_tracks_reentry_carried_scalar_columns(
    query: str,
    expected_whole_row_output: str,
    expected_columns: Tuple[str, ...],
) -> None:
    compiled = _compile_query(query)
    whole_row_output, carried_columns = _compiled_reentry_projection_outputs(compiled)
    plan = compiled.reentry_plan

    assert whole_row_output == expected_whole_row_output
    assert carried_columns == expected_columns
    assert plan is not None
    assert plan.reentry_alias_name == expected_whole_row_output
    assert tuple(plan.scalar_columns) == expected_columns


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            _reentry_query("a AS x", match_alias="x", return_clause="b.id AS bid", order_by="bid"),
            [{"bid": "b1"}, {"bid": "b2"}],
        ),
        (
            _reentry_query("a, a.num AS property", return_clause="property", order_by="property DESC"),
            [{"property": 2}, {"property": 1}],
        ),
        (
            _reentry_query("a, a.num AS property", return_clause="a", order_by="property DESC"),
            [{"a": "(:A {num: 2})"}, {"a": "(:A {num: 1})"}],
        ),
        (
            _reentry_query(
                "a, a.num AS property, a.num + 10 AS property2",
                return_clause="property, property2",
                order_by="property DESC",
            ),
            [{"property": 2, "property2": 12}, {"property": 1, "property2": 11}],
        ),
        (
            _reentry_query(
                "a AS x, a.num AS property",
                match_alias="x",
                return_clause="x, property",
                order_by="property DESC",
            ),
            [{"x": "(:A {num: 2})", "property": 2}, {"x": "(:A {num: 1})", "property": 1}],
        ),
    ],
)
def test_string_cypher_executes_with_match_reentry_carried_scalar_shapes(query: str, expected: List[Dict[str, Any]]) -> None:
    result = _mk_reentry_carried_scalar_graph().gfql(query)
    _entities = {
        name: ("edges" if str(val).startswith("[") else "nodes")
        for name, val in (expected[0].items() if expected else [])
        if str(val).startswith(("(", "["))
    }
    assert entity_text_records(result, _entities) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            _reentry_query("a, a.num AS property", return_clause="property", order_by="property DESC"),
            [{"property": 2}, {"property": 1}],
        ),
        (
            _reentry_query(
                "a AS x, a.num AS property",
                match_alias="x",
                return_clause="x, property",
                order_by="property DESC",
            ),
            [{"x": "(:A {num: 2})", "property": 2}, {"x": "(:A {num: 1})", "property": 1}],
        ),
    ],
)
def test_string_cypher_executes_with_match_reentry_carried_scalar_shapes_on_cudf(
    query: str,
    expected: List[Dict[str, Any]],
) -> None:
    pytest.importorskip("cudf")

    result = _mk_reentry_carried_scalar_graph_cudf().gfql(query, engine="cudf")

    assert type(result._nodes).__module__.startswith("cudf")
    _entities = {
        name: ("edges" if str(val).startswith("[") else "nodes")
        for name, val in (expected[0].items() if expected else [])
        if str(val).startswith(("(", "["))
    }
    assert entity_text_records(result, _entities) == expected


def test_string_cypher_executes_with_match_reentry_carried_scalars_from_connected_prefix_shape() -> None:
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN bid, c.id AS cid"
    )

    result = _mk_connected_reentry_carried_scalar_graph().gfql(query, params={"seed": "a1"})

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1", "cid": "c1"}]


def test_string_cypher_executes_with_match_reentry_multiple_carried_scalars_from_connected_prefix_shape() -> None:
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH b, b.id AS bid, b.score AS bscore "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN bid, bscore, c.id AS cid"
    )

    result = _mk_connected_reentry_carried_scalar_graph().gfql(query, params={"seed": "a1"})

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1", "bscore": 10, "cid": "c1"}]


def test_string_cypher_executes_with_match_reentry_carried_scalars_from_connected_prefix_shape_on_cudf() -> None:
    pytest.importorskip("cudf")

    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN bid, c.id AS cid"
    )

    result = _mk_connected_reentry_carried_scalar_graph_cudf().gfql(query, params={"seed": "a1"}, engine="cudf")

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"bid": "b1", "cid": "c1"}]


def test_string_cypher_executes_plain_connected_multi_pattern_scalar_projection() -> None:
    result = _mk_connected_multi_pattern_reentry_graph().gfql(
        "MATCH (b:B)-[:S]->(c:C), (c)-[:T]->(d:D) RETURN c.id AS cid, d.id AS did ORDER BY cid"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"cid": "c1", "did": "d1"},
        {"cid": "c2", "did": "d2"},
    ]


def test_string_cypher_executes_plain_multi_alias_edge_scalar_projection() -> None:
    result = _mk_multi_alias_edge_projection_graph().gfql(
        "MATCH (a:A)-[r:R]->(b:B) RETURN a.id AS aid, r.creationDate AS created, b.id AS bid ORDER BY aid"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"aid": "a1", "created": 10, "bid": "b1"},
        {"aid": "a2", "created": 20, "bid": "b2"},
    ]


def test_string_cypher_executes_with_match_reentry_whole_row_into_connected_multi_pattern_shape() -> None:
    result = _mk_connected_multi_pattern_reentry_graph().gfql(
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C), (c)-[:T]->(d:D) "
        "RETURN d.id AS did",
        params={"seed": "a1"},
    )

    assert result._nodes.to_dict(orient="records") == [{"did": "d1"}]


def test_string_cypher_executes_with_match_reentry_carried_scalar_into_connected_multi_pattern_shape() -> None:
    result = _mk_connected_multi_pattern_reentry_graph().gfql(
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C), (c)-[:T]->(d:D) "
        "RETURN bid, d.id AS did",
        params={"seed": "a1"},
    )

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1", "did": "d1"}]


def test_string_cypher_executes_plain_connected_multi_pattern_scalar_projection_fanout() -> None:
    result = _mk_connected_multi_pattern_fanout_graph().gfql(
        "MATCH (b:B)-[:S]->(c:C), (c)-[:T]->(d:D) RETURN c.id AS cid, d.id AS did ORDER BY did"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"cid": "c1", "did": "d1"},
        {"cid": "c1", "did": "d2"},
    ]


def test_string_cypher_executes_with_match_reentry_carried_scalar_into_connected_multi_pattern_fanout_shape() -> None:
    result = _mk_connected_multi_pattern_fanout_graph().gfql(
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C), (c)-[:T]->(d:D) "
        "RETURN bid, d.id AS did "
        "ORDER BY did",
        params={"seed": "a1"},
    )

    assert result._nodes.to_dict(orient="records") == [
        {"bid": "b1", "did": "d1"},
        {"bid": "b1", "did": "d2"},
    ]


def test_string_cypher_executes_with_match_reentry_carried_scalar_into_connected_multi_pattern_fanout_shape_on_cudf() -> None:
    pytest.importorskip("cudf")

    result = _mk_connected_multi_pattern_fanout_graph_cudf().gfql(
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C), (c)-[:T]->(d:D) "
        "RETURN bid, d.id AS did "
        "ORDER BY did",
        params={"seed": "a1"},
        engine="cudf",
    )

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"bid": "b1", "did": "d1"},
        {"bid": "b1", "did": "d2"},
    ]


def test_string_cypher_executes_recent_message_reentry_multihop_scalar_projection() -> None:
    result = _mk_recent_message_reentry_graph().gfql(
        "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(message) "
        "WITH message, message.id AS messageId, message.creationDate AS messageCreationDate "
        "ORDER BY messageCreationDate DESC, messageId ASC "
        "LIMIT 10 "
        "MATCH (message)-[:REPLY_OF*0..]->(post:Post), (post)-[:HAS_CREATOR]->(person) "
        "RETURN messageId, messageCreationDate, post.id AS postId, person.id AS personId "
        "ORDER BY messageCreationDate DESC, messageId ASC",
        params={"personId": "viewer"},
    )

    assert result._nodes.to_dict(orient="records") == [
        {"messageId": "post2", "messageCreationDate": 20, "postId": "post2", "personId": "viewer"},
        {"messageId": "comment1", "messageCreationDate": 10, "postId": "post1", "personId": "author1"},
    ]


def test_string_cypher_executes_recent_message_reentry_multihop_scalar_projection_on_cudf() -> None:
    pytest.importorskip("cudf")

    result = _mk_recent_message_reentry_graph_cudf().gfql(
        "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(message) "
        "WITH message, message.id AS messageId, message.creationDate AS messageCreationDate "
        "ORDER BY messageCreationDate DESC, messageId ASC "
        "LIMIT 10 "
        "MATCH (message)-[:REPLY_OF*0..]->(post:Post), (post)-[:HAS_CREATOR]->(person) "
        "RETURN messageId, messageCreationDate, post.id AS postId, person.id AS personId "
        "ORDER BY messageCreationDate DESC, messageId ASC",
        params={"personId": "viewer"},
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"messageId": "post2", "messageCreationDate": 20, "postId": "post2", "personId": "viewer"},
        {"messageId": "comment1", "messageCreationDate": 10, "postId": "post1", "personId": "author1"},
    ]

def test_string_cypher_executes_recent_message_reentry_multihop_branching_row_bindings() -> None:
    query = (
        "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(message) "
        "WITH message, message.id AS messageId "
        "MATCH (message)-[:REPLY_OF*0..]->(post:Post), (post)-[:HAS_CREATOR]->(person) "
        "RETURN messageId, post.id AS postId, person.id AS personId "
        "ORDER BY messageId, postId, personId"
    )

    result = _mk_recent_message_reentry_graph_branching().gfql(query, params={"personId": "viewer"})

    assert result._nodes.to_dict(orient="records") == [
        {"messageId": "comment1", "postId": "post1", "personId": "author1"},
        {"messageId": "comment1", "postId": "post2", "personId": "author2"},
    ]


def test_string_cypher_executes_undirected_multihop_row_bindings() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c"],
                "d": ["b", "c", "d"],
                "type": ["R", "R", "R"],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (a {id: 'a'})-[:R*1..2]-(b) RETURN a.id AS aid, b.id AS bid ORDER BY aid, bid"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"aid": "a", "bid": "b"},
        {"aid": "a", "bid": "c"},
    ]


def test_string_cypher_executes_undirected_multihop_row_bindings_on_cudf() -> None:
    pytest.importorskip("cudf")

    graph = _mk_cudf_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c"],
                "d": ["b", "c", "d"],
                "type": ["R", "R", "R"],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (a {id: 'a'})-[:R*1..2]-(b) RETURN a.id AS aid, b.id AS bid ORDER BY aid, bid",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"aid": "a", "bid": "b"},
        {"aid": "a", "bid": "c"},
    ]


@pytest.mark.parametrize(
    ("graph_factory", "query", "params", "match"),
    [
        (
            _mk_multihop_row_binding_cycle_graph,
            "MATCH (a:A)-[r:R*1..2]->(b) RETURN a.id AS aid, b.id AS bid",
            None,
            "do not yet support variable-length relationship aliases",
        ),
        (
            _mk_multihop_row_binding_cycle_graph,
            "MATCH (a:A)-[:R*0..]->(b) RETURN a.id AS aid, b.id AS bid",
            None,
            "currently require terminating variable-length segments",
        ),
    ],
)
def test_string_cypher_failfast_rejects_remaining_unsupported_multihop_row_bindings(
    graph_factory: Callable[[], _CypherTestGraph],
    query: str,
    params: Optional[Dict[str, Any]],
    match: str,
) -> None:
    with pytest.raises(GFQLValidationError, match=match):
        graph_factory().gfql(query, params=params)


def test_compile_cypher_records_reentry_plan_for_multi_whole_row_prefix() -> None:
    """#989 slice 4.1: ``compiled_query.reentry_plan`` is populated with one
    CarriedAlias per prefix whole-row, exactly one marked as the reentry source.

    Without this assertion, a future slice could regress the plan structure
    while user-facing behavior keeps working through incidental fallback paths.
    """
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(b) "
        "RETURN b.id AS bid"
    )
    compiled = cast(CompiledCypherQuery, compile_cypher(query))
    plan = compiled.reentry_plan
    assert plan is not None
    assert plan.reentry_alias_name == "a"
    assert plan.scalar_only is False
    assert tuple(alias.output_name for alias in plan.aliases) == ("a", "x")
    source = plan.reentry_alias
    assert source is not None and source.output_name == "a"
    non_source = plan.non_source_aliases
    assert len(non_source) == 1 and non_source[0].output_name == "x"


def test_compile_cypher_records_freeform_reentry_plan_contract() -> None:
    """#989 follow-through: free-form intermediate MATCH must not degrade to
    scalar-only reentry plan routing.
    """
    query = (
        "MATCH (a:A {id: 'a'}) "
        "WITH a "
        "MATCH (n:B) "
        "RETURN n.id AS nid"
    )
    compiled = cast(CompiledCypherQuery, compile_cypher(query))
    plan = compiled.reentry_plan
    assert plan is not None
    assert plan.reentry_alias_name == "n"
    assert plan.scalar_only is False
    assert plan.free_form is True
    assert tuple(alias.output_name for alias in plan.aliases) == ("a",)
    assert plan.reentry_alias is None


def test_compile_cypher_records_non_source_carried_properties_on_reentry_plan() -> None:
    """#989 row-carrier contract: non-source aliases record property-level carry deps."""
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(b) "
        "RETURN b.id AS bid, x.id AS xid"
    )
    compiled = cast(CompiledCypherQuery, compile_cypher(query))
    plan = compiled.reentry_plan
    assert plan is not None
    non_source = {alias.output_name: alias for alias in plan.non_source_aliases}
    assert "x" in non_source
    assert non_source["x"].carried_properties == ("id",)


def test_compile_cypher_records_freeform_non_source_carried_properties_on_reentry_plan() -> None:
    """#989 free-form lane: plan metadata keeps property carries for non-source aliases."""
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (n:B) "
        "RETURN n.id AS nid, x.id AS xid"
    )
    compiled = cast(CompiledCypherQuery, compile_cypher(query))
    plan = compiled.reentry_plan
    assert plan is not None
    assert plan.free_form is True
    aliases = {alias.output_name: alias for alias in plan.aliases}
    assert "x" in aliases
    assert aliases["x"].carried_properties == ("id",)


def test_string_cypher_admits_multi_whole_row_prefix_when_non_source_aliases_are_unused() -> None:
    """#989 slice 4.3: admit `WITH a, x` prefix when only `a` is referenced downstream."""
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(b) "
        "RETURN b.id AS bid ORDER BY bid"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    records = result._nodes.to_dict(orient="records")
    assert records == [{"bid": "b"}, {"bid": "e"}]


def test_string_cypher_admits_multi_whole_row_prefix_with_downstream_stage_where() -> None:
    """#989 slice 4.3: regression for ProjectionStage.where vs WhereClause type mismatch.

    The non-source-alias scanner walks `query.with_stages[1:]`, each of which has a
    `where: Optional[ExpressionText]` (a raw text node, NOT a `WhereClause`). An
    earlier draft passed it to a `WhereClause`-shaped helper and would have raised
    `AttributeError: 'ExpressionText' object has no attribute 'expr_tree'` on any
    multi-whole-row prefix WITH followed by a downstream stage that carries a WHERE.
    """
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(b) "
        "WITH b "
        "WHERE b.id = 'b' "
        "RETURN b.id AS bid"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"bid": "b"}]


def test_string_cypher_admits_non_source_alias_property_carry_through_reentry() -> None:
    """#989 slice 4.3b: property references on non-source aliases (`x.id`) are
    admitted. The prefix WITH is rewritten to project ``x.id AS <carry_name>``;
    trailing ``x.id`` references rewrite to a property access on the
    reentry-alias's hidden column. Carries the value end-to-end.
    """
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(b) "
        "RETURN b.id AS bid, x.id AS xid ORDER BY bid"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    # x.id == 'b' for every row; R-neighbors of 'a' are 'b' and 'e'.
    assert result._nodes.to_dict(orient="records") == [
        {"bid": "b", "xid": "b"},
        {"bid": "e", "xid": "b"},
    ]


def test_string_cypher_failfast_rejects_multi_whole_row_prefix_when_non_source_alias_is_bare_referenced_in_order_by() -> None:
    """#989 slice 4.3b failfast scope: ORDER BY referencing a non-source alias bare.

    Slice 4.3b admits property access (`x.id`); a bare ORDER BY on the alias
    itself still requires the full row-carrier rewrite.
    """
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(b) "
        "RETURN b.id AS bid ORDER BY x"
    )
    with pytest.raises(
        GFQLValidationError,
        match=r"(bare references like|whole-row outputs; reference them by property only)",
    ):
        _mk_multi_stage_reentry_graph().gfql(query)


def test_string_cypher_failfast_rejects_multi_whole_row_prefix_when_non_source_alias_is_bare_referenced() -> None:
    """#989 slice 4.3b: bare references to non-source whole-row aliases (no
    property access) still failfast — that requires the full row-carrier IR
    rewrite, not just per-property hidden-column carry.

    Note: pure forwarding patterns like ``WITH b, x`` (where ``x`` is just being
    re-projected for downstream stages) are NOT bare uses — slice 4.3c drops
    those at compile time. This test uses ``WITH a, x AS xref`` which renames
    the bare alias — that's a use, not a forward, and still fails.
    """
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(b) "
        "WHERE x = b "
        "RETURN b.id AS bid"
    )
    with pytest.raises(
        GFQLValidationError,
        match=r"(bare references like|whole-row outputs; reference them by property only)",
    ):
        _mk_multi_stage_reentry_graph().gfql(query)


def test_string_cypher_chained_reentry_with_repeated_primary_preserves_prefix_row_bag_semantics() -> None:
    """#1394: repeated-primary chained reentry with duplicate carried ids executes.

    Prefix rows can carry the same reentry alias id multiple times (one per
    prior match row). Runtime should execute suffix semantics per prefix row
    and preserve bag semantics instead of failfasting on duplicate carried ids.
    """
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(friend) "
        "WITH a, x, friend "
        "MATCH (a)-[:R]->(other) "
        "WHERE other.id <> friend.id "
        "RETURN other.id AS oid, x.id AS xid"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    assert sorted(result._nodes.to_dict(orient="records"), key=lambda row: row["oid"]) == [
        {"oid": "b", "xid": "b"},
        {"oid": "e", "xid": "b"},
    ]


def test_string_cypher_chained_reentry_with_repeated_primary_preserves_duplicate_output_rows() -> None:
    """#1394 amplification: duplicate carried-id fallback preserves per-prefix-row multiplicity.

    The second reentry runs once per prefix row. Here prefix produces two rows
    over the same carried `a`; suffix yields two rows each time, so output must
    contain four `(friend, other)` pairs (cartesian per prefix row).
    """
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(friend) "
        "WITH a, x, friend "
        "MATCH (a)-[:R]->(other) "
        "RETURN other.id AS oid, friend.id AS fid"
    )
    records = _mk_multi_stage_reentry_graph().gfql(query)._nodes.to_dict(orient="records")
    assert len(records) == 4
    assert {(row["fid"], row["oid"]) for row in records} == {
        ("b", "b"),
        ("b", "e"),
        ("e", "b"),
        ("e", "e"),
    }


def test_string_cypher_chained_reentry_with_repeated_primary_preserves_duplicate_output_rows_on_cudf() -> None:
    """#1394 amplification cuDF parity for per-prefix-row multiplicity lock."""
    pytest.importorskip("cudf")
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(friend) "
        "WITH a, x, friend "
        "MATCH (a)-[:R]->(other) "
        "RETURN other.id AS oid, friend.id AS fid"
    )
    result = _mk_multi_stage_reentry_graph_cudf().gfql(query, engine="cudf")
    nodes_pd = _to_pandas_df(result._nodes)
    records = nodes_pd.to_dict(orient="records")
    assert len(records) == 4
    assert {(row["fid"], row["oid"]) for row in records} == {
        ("b", "b"),
        ("b", "e"),
        ("e", "b"),
        ("e", "e"),
    }


def test_string_cypher_admits_secondary_alias_carry_across_reentry_source_rebinding() -> None:
    """#1256 slice 4.3d.2: secondary alias carry survives a reentry-source rebinding.

    `WITH a, x MATCH (a)-[:R]->(friend) WITH friend, x MATCH (friend)-[:S]->(c)
    RETURN c.id, x.id` rebinds the trailing-MATCH source from `a` to `friend`
    between the two reentry boundaries. The carry `x.id` lives as a hidden
    column on `a`'s row table; after the join with friend, on friend's table;
    the second boundary's prefix WITH must continue to forward it as a scalar
    so the inner compile can resolve it.
    """
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(friend) "
        "WITH friend, x "
        "MATCH (friend)-[:S]->(c) "
        "RETURN c.id AS cid, x.id AS xid ORDER BY cid"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [
        {"cid": "c", "xid": "b"},
        {"cid": "e", "xid": "b"},
    ]


def test_string_cypher_admits_multi_alias_distinct_forwarding_through_reentry() -> None:
    """#1256 slice 4.3d.1: bare-identifier projection items in downstream
    ``WITH a, x, friend`` (pure forwarding through a reentry boundary) are
    dropped at compile time so the bare-ref scanner does not false-positive.

    Before slice 4.3d.1, the active rewrite path
    (``_demote_secondary_whole_row_aliases``) failed at line ~7900
    ("does not yet support carrying secondary whole-row aliases as whole-row
    outputs") because the bare ``x`` in stage[1] was treated as a true USE.
    """
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(friend) "
        "WITH DISTINCT a, x, friend "
        "RETURN friend.id AS fid, x.id AS xid ORDER BY fid"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [
        {"fid": "b", "xid": "b"},
        {"fid": "e", "xid": "b"},
    ]


def test_string_cypher_chained_reentry_carry_with_aggregate_relationship_match() -> None:
    """Closed #1256 aggregate lane: hidden secondary carry survives an
    aggregating downstream WITH stage after a relationship-pattern MATCH.
    """
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a)-[:R]->(friend) "
        "WITH a, x, count(*) AS n "
        "RETURN x.id AS xid, n"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"xid": "b", "n": 2}]


def test_string_cypher_chained_reentry_carry_with_aggregate_node_only_match() -> None:
    """Closed #1256 aggregate lane: hidden secondary carry survives an
    aggregating downstream WITH stage after a node-only MATCH.
    """
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (a) "
        "WITH a, x, count(*) AS n "
        "RETURN x.id AS xid, n"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"xid": "b", "n": 1}]


def test_string_cypher_executes_intermediate_reentry_match_with_carried_property_bridge_where() -> None:
    """#1275: free-form intermediate MATCH admits bridge WHERE predicates that
    reference both trailing aliases and carried-alias properties."""
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (c:C)-[:T]->(d:D) "
        "WHERE c.id = x.id OR c.id = 'c' "
        "RETURN d.id AS did"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"did": "d"}]


def test_string_cypher_executes_intermediate_reentry_match_with_carried_property_bridge_where_on_cudf() -> None:
    query = (
        "MATCH (a:A {id: 'a'}), (x:B {id: 'b'}) "
        "WITH a, x "
        "MATCH (c:C)-[:T]->(d:D) "
        "WHERE c.id = x.id OR c.id = 'c' "
        "RETURN d.id AS did"
    )
    _require_cudf_runtime()
    result = _mk_multi_stage_reentry_graph_cudf().gfql(query, engine="cudf")
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert records == [{"did": "d"}]


def test_string_cypher_executes_intermediate_reentry_match_with_carried_property_bridge_where_and_return() -> None:
    """#1275 issue-shape: bridge predicate + trailing RETURN include carried alias property."""
    query = (
        "MATCH (a:A {id: 'a'}), (x:C {id: 'c'}) "
        "WITH a, x "
        "MATCH (c:C)-[:T]->(d:D) "
        "WHERE c.id = x.id "
        "RETURN c.id AS cid, d.id AS did, x.id AS xid"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"cid": "c", "did": "d", "xid": "c"}]


def test_string_cypher_executes_simple_freeform_intermediate_reentry_match() -> None:
    """#1263 conservative admit: trailing MATCH whose first alias is NOT in
    the prefix's carried whole-row set executes correctly when no
    carried-alias property is referenced in the trailing scope.

    Single-alias prefix WITH (no demote interaction); the runtime broadcasts
    carried hidden columns onto every base node and the trailing MATCH binds
    fresh aliases (``c``, ``d``).
    """
    query = (
        "MATCH (a:A {id: 'a'}) "
        "WITH a "
        "MATCH (c:C)-[:T]->(d:D) "
        "RETURN d.id AS did, c.id AS cid"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"did": "d", "cid": "c"}]


def test_string_cypher_executes_freeform_intermediate_reentry_match_with_multi_carried_aliases() -> None:
    """#1263 Wave 2 amplification: free-form admit composes with multi-alias
    prefix `WITH a, b` when no carried-alias property is referenced in the
    trailing scope. The demote (#1071) bails out (free-form path doesn't anchor
    on a carried alias) and the runtime broadcasts both carried whole-row rows
    onto every base node uniformly.
    """
    query = (
        "MATCH (a:A {id: 'a'}), (b:B {id: 'b'}) "
        "WITH a, b "
        "MATCH (c:C)-[:T]->(d:D) "
        "RETURN d.id AS did, c.id AS cid"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"did": "d", "cid": "c"}]


def test_string_cypher_executes_freeform_intermediate_reentry_match_with_empty_prefix() -> None:
    """#1263 Wave 2 amplification: when the prefix MATCH yields zero rows the
    runtime helper short-circuits to an empty graph dispatch and the trailing
    MATCH produces zero rows. Locks `_compiled_query_freeform_reentry_state`'s
    early-return path at gfql_unified.py.
    """
    query = (
        "MATCH (a:A {id: 'NONEXISTENT'}) "
        "WITH a "
        "MATCH (c:C)-[:T]->(d:D) "
        "RETURN d.id AS did, c.id AS cid"
    )
    result = _mk_multi_stage_reentry_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == []


def test_string_cypher_freeform_intermediate_reentry_preserves_bag_semantics_multi_row_prefix() -> None:
    """#1263/#1285 multi-prefix-row free-form intermediate MATCH executes via
    per-row union (mirror of the scalar-only multi-row pattern from #1047).

    Prefix yields 2 ``A`` rows; trailing MATCH ``(c:C)-[:T]->(d:D)`` yields 1
    pair globally → cartesian product = 2 result rows (one per prefix row,
    bag semantics preserved). Originally landed in #1287 as the bag-semantics
    lock; #1285 reuses the same fixture/shape for the multi-row admit.
    """
    nodes = pd.DataFrame(
        {
            "id": ["a", "a2", "b", "c", "d"],
            "label__A": [True, True, False, False, False],
            "label__B": [False, False, True, False, False],
            "label__C": [False, False, False, True, False],
            "label__D": [False, False, False, False, True],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "a2", "b", "c"],
            "d": ["b", "b", "c", "d"],
            "type": ["R", "R", "S", "T"],
        }
    )
    graph = _mk_graph(nodes, edges)
    query = (
        "MATCH (a:A) "
        "WITH a "
        "MATCH (c:C)-[:T]->(d:D) "
        "RETURN d.id AS did"
    )
    result = graph.gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"did": "d"}, {"did": "d"}]



def test_string_cypher_executes_freeform_intermediate_reentry_match_on_multi_row_prefix_cartesian() -> None:
    """#1285: confirm cartesian semantics — 2 prefix rows × 2 trailing-MATCH
    pairs = 4 result rows. Locks per-row union behaves like a Cartesian
    product over (prefix_row, trailing_row)."""
    nodes = pd.DataFrame(
        {
            "id": ["a1", "a2", "c1", "d1", "c2", "d2"],
            "label__A": [True, True, False, False, False, False],
            "label__C": [False, False, True, False, True, False],
            "label__D": [False, False, False, True, False, True],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["c1", "c2"],
            "d": ["d1", "d2"],
            "type": ["T", "T"],
        }
    )
    graph = _mk_graph(nodes, edges)
    query = (
        "MATCH (a:A) "
        "WITH a "
        "MATCH (c:C)-[:T]->(d:D) "
        "RETURN d.id AS did, c.id AS cid"
    )
    result = graph.gfql(query)
    records = result._nodes.to_dict(orient="records")
    # 2 prefix rows × 2 trailing pairs = 4 rows; per-row union runs the
    # trailing MATCH once per prefix row, so each (cid, did) pair appears twice.
    assert len(records) == 4
    counts = {(r["cid"], r["did"]) for r in records}
    assert counts == {("c1", "d1"), ("c2", "d2")}
    assert sum(1 for r in records if (r["cid"], r["did"]) == ("c1", "d1")) == 2
    assert sum(1 for r in records if (r["cid"], r["did"]) == ("c2", "d2")) == 2


def test_string_cypher_failfast_rejects_freeform_multi_row_prefix_with_optional_reentry() -> None:
    """#1285: multi-prefix-row free-form combined with OPTIONAL MATCH is not
    yet supported — the per-row union path returns early before any
    null-fill branch, which would silently produce wrong results for prefix
    rows that match nothing. Mirrors the scalar-only guard locked by
    ``test_issue_1047_multi_row_scalar_prefix_with_optional_reentry_raises``.
    """
    nodes = pd.DataFrame(
        {
            "id": ["a", "a2", "c", "d"],
            "label__A": [True, True, False, False],
            "label__C": [False, False, True, False],
            "label__D": [False, False, False, True],
        }
    )
    edges = pd.DataFrame({"s": ["c"], "d": ["d"], "type": ["T"]})
    graph = _mk_graph(nodes, edges)
    query = (
        "MATCH (a:A) "
        "WITH a "
        "OPTIONAL MATCH (c:C)-[:T]->(d:D) "
        "RETURN d.id AS did"
    )
    with pytest.raises(Exception, match="optional"):
        graph.gfql(query)


def test_string_cypher_executes_freeform_intermediate_reentry_match_on_multi_row_prefix_on_cudf_when_available() -> None:
    """#1285 cuDF parity for the multi-prefix-row free-form admit."""
    cudf = pytest.importorskip("cudf")
    nodes_pd = pd.DataFrame(
        {
            "id": ["a", "a2", "c", "d"],
            "label__A": [True, True, False, False],
            "label__C": [False, False, True, False],
            "label__D": [False, False, False, True],
        }
    )
    edges_pd = pd.DataFrame(
        {
            "s": ["c"],
            "d": ["d"],
            "type": ["T"],
        }
    )
    graph = _mk_graph(cudf.from_pandas(nodes_pd), cudf.from_pandas(edges_pd))
    query = (
        "MATCH (a:A) "
        "WITH a "
        "MATCH (c:C)-[:T]->(d:D) "
        "RETURN d.id AS did"
    )
    result = graph.gfql(query)
    nodes_pd_out = _to_pandas_df(result._nodes)
    assert nodes_pd_out.to_dict(orient="records") == [{"did": "d"}, {"did": "d"}]


def test_string_cypher_executes_simple_freeform_intermediate_reentry_match_on_cudf_when_available() -> None:
    """#1263 cuDF parity for the simple free-form admit (paired with the
    pandas case above)."""
    cudf = pytest.importorskip("cudf")
    base_graph = _mk_multi_stage_reentry_graph()
    cudf_graph = base_graph.nodes(
        cudf.from_pandas(base_graph._nodes), base_graph._node
    ).edges(
        cudf.from_pandas(base_graph._edges),
        base_graph._source,
        base_graph._destination,
    )
    query = (
        "MATCH (a:A {id: 'a'}) "
        "WITH a "
        "MATCH (c:C)-[:T]->(d:D) "
        "RETURN d.id AS did, c.id AS cid"
    )
    result = cudf_graph.gfql(query)
    nodes_pd = _to_pandas_df(result._nodes)
    assert nodes_pd.to_dict(orient="records") == [{"did": "d", "cid": "c"}]


def test_string_cypher_failfast_rejects_with_match_reentry_multiple_trailing_match_clauses() -> None:
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "MATCH (c)-[:T]->(d:D) "
        "RETURN bid, d.id AS did"
    )

    with pytest.raises(GFQLSyntaxError, match="alternating MATCH \\.\\.\\. WITH \\.\\.\\. MATCH"):
        _mk_connected_multi_pattern_reentry_graph().gfql(query, params={"seed": "a1"})


def test_string_cypher_connected_multi_pattern_relationship_alias_projection_second_edge() -> None:
    """Multi-hop: project edge alias from second hop (#880)."""
    query = (
        "MATCH (b:B)-[r:S]->(c:C), (c)-[t:T]->(d:D) "
        "RETURN t.type AS tt, d.id AS did"
    )
    result = _mk_connected_multi_pattern_reentry_graph().gfql(query)
    records = sorted(result._nodes.to_dict(orient="records"), key=lambda r: r["did"])
    assert records == [{"tt": "T", "did": "d1"}, {"tt": "T", "did": "d2"}]


def test_string_cypher_connected_multi_pattern_relationship_alias_projection_first_edge() -> None:
    """Multi-hop: project edge alias from first hop (#880)."""
    query = (
        "MATCH (b:B)-[r:S]->(c:C), (c)-[t:T]->(d:D) "
        "RETURN r.type AS rt, d.id AS did"
    )
    result = _mk_connected_multi_pattern_reentry_graph().gfql(query)
    records = sorted(result._nodes.to_dict(orient="records"), key=lambda r: r["did"])
    assert records == [{"rt": "S", "did": "d1"}, {"rt": "S", "did": "d2"}]


def test_string_cypher_connected_multi_pattern_relationship_alias_projection_both_edges() -> None:
    """Multi-hop: project both edge aliases (#880)."""
    query = (
        "MATCH (b:B)-[r:S]->(c:C), (c)-[t:T]->(d:D) "
        "RETURN r.type AS rt, t.type AS tt, d.id AS did"
    )
    result = _mk_connected_multi_pattern_reentry_graph().gfql(query)
    records = sorted(result._nodes.to_dict(orient="records"), key=lambda r: r["did"])
    assert records == [
        {"rt": "S", "tt": "T", "did": "d1"},
        {"rt": "S", "tt": "T", "did": "d2"},
    ]


def test_string_cypher_connected_multi_pattern_mixed_node_and_edge_alias_projection() -> None:
    """Multi-hop: mixed node + edge alias projection (#880)."""
    query = (
        "MATCH (b:B)-[r:S]->(c:C), (c)-[t:T]->(d:D) "
        "RETURN b.id AS bid, r.type AS rt, c.id AS cid, t.type AS tt, d.id AS did"
    )
    result = _mk_connected_multi_pattern_reentry_graph().gfql(query)
    records = sorted(result._nodes.to_dict(orient="records"), key=lambda r: r["did"])
    assert len(records) == 2
    assert records[0]["bid"] == "b1"
    assert records[0]["rt"] == "S"
    assert records[0]["cid"] == "c1"
    assert records[0]["tt"] == "T"
    assert records[0]["did"] == "d1"


def test_string_cypher_with_match_reentry_multi_whole_row_alias_unreferenced_secondary() -> None:
    """An unreferenced secondary whole-row alias in the prefix WITH is dropped (#1071)."""
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH a, b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN bid, c.id AS cid"
    )

    result = _mk_connected_reentry_carried_scalar_graph().gfql(query, params={"seed": "a1"})

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1", "cid": "c1"}]


def test_string_cypher_with_match_reentry_multi_whole_row_alias_property_carry() -> None:
    """Secondary whole-row alias is referenced by property in the trailing
    RETURN; rewritten to a hidden carry column (#1071)."""
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN a.id AS aid, b.id AS bid, c.id AS cid"
    )

    result = _mk_connected_reentry_carried_scalar_graph().gfql(query, params={"seed": "a1"})

    assert result._nodes.to_dict(orient="records") == [{"aid": "a1", "bid": "b1", "cid": "c1"}]


def test_string_cypher_executes_ic1_shaped_multi_alias_with_match_reentry() -> None:
    """LDBC SNB IC1-shape: variable-length ``KNOWS*1..3`` followed by
    ``WITH p, friend MATCH (friend)-...`` with property references to the
    secondary alias ``p`` in RETURN (#1071)."""
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["seed", "alice", "bob", "city1", "city2"],
                "label__Person": [True, True, True, False, False],
                "label__Place": [False, False, False, True, True],
                "firstName": ["Seed", "Alice", "Bob", None, None],
                "name": [None, None, None, "Springfield", "Shelbyville"],
            }
        ),
        pd.DataFrame(
            {
                "s": ["seed", "alice", "bob", "alice", "bob"],
                "d": ["alice", "bob", "city1", "city1", "city2"],
                "type": ["KNOWS", "KNOWS", "KNOWS", "IS_LOCATED_IN", "IS_LOCATED_IN"],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (p:Person {id: $pid})-[:KNOWS*1..3]->(friend:Person) "
        "WITH p, friend "
        "MATCH (friend)-[:IS_LOCATED_IN]->(city:Place) "
        "RETURN friend.id AS fid, p.firstName AS pfn, city.name AS cname "
        "ORDER BY fid",
        params={"pid": "seed"},
    )

    assert result._nodes.to_dict(orient="records") == [
        {"fid": "alice", "pfn": "Seed", "cname": "Springfield"},
        {"fid": "bob", "pfn": "Seed", "cname": "Shelbyville"},
    ]


def test_string_cypher_executes_with_match_reentry_secondary_alias_user_scalar_collision() -> None:
    """Defense-in-depth: a user-named carried scalar (``b.id AS a_id``) and a
    demoted secondary property ref (``a.id`` on secondary alias ``a``) BOTH
    feed an output named ``a_id`` in the prefix WITH. The double-prefix
    wrapping in ``_reentry_hidden_column_name`` keeps the in-table columns
    distinct, so values flow correctly to RETURN (#1071, Wave 3 regression)."""
    nodes = pd.DataFrame(
        {
            "id": ["a1", "b1", "c1"],
            "label__A": [True, False, False],
            "label__B": [False, True, False],
            "label__C": [False, False, True],
        }
    )
    edges = pd.DataFrame(
        {"s": ["a1", "b1"], "d": ["b1", "c1"], "type": ["R", "S"]}
    )
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH a, b, b.id AS a_id "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN a.id AS aid_demoted, a_id AS a_id_user, c.id AS cid"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"aid_demoted": "a1", "a_id_user": "b1", "cid": "c1"},
    ]


def test_string_cypher_executes_with_match_reentry_secondary_alias_inline_pattern_property() -> None:
    """Inline node-pattern property map referencing a secondary alias property
    (``MATCH (c {tag: a.tag})``) is rewritten via the pattern-element walk in
    ``_rewrite_reentry_match_clause`` (#1071, Wave 3 regression)."""
    nodes = pd.DataFrame(
        {
            "id": ["a1", "b1", "c1"],
            "label__A": [True, False, False],
            "label__B": [False, True, False],
            "label__C": [False, False, True],
            "tag": ["t1", "t1", "t1"],
        }
    )
    edges = pd.DataFrame(
        {"s": ["a1", "b1"], "d": ["b1", "c1"], "type": ["R", "S"]}
    )
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C {tag: a.tag}) "
        "RETURN c.id AS cid"
    )

    assert result._nodes.to_dict(orient="records") == [{"cid": "c1"}]


def test_string_cypher_executes_with_match_reentry_secondary_alias_property_in_or_where() -> None:
    """Secondary alias property used in a tree-shape (OR) trailing WHERE: the
    demoter must rewrite ``a.score`` inside the OR atom so the trailing row
    pre-filter sees the carried hidden column (#1071, Wave 2 regression)."""
    nodes = pd.DataFrame(
        {
            "id": ["a1", "b1", "c1", "c2"],
            "label__A": [True, False, False, False],
            "label__B": [False, True, False, False],
            "label__C": [False, False, True, True],
            "score": [5, None, None, None],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a1", "b1", "b1"],
            "d": ["b1", "c1", "c2"],
            "type": ["R", "S", "S"],
        }
    )
    g = _mk_graph(nodes, edges)

    # a.score = 5; first arm `a.score > 10` is false, so only c1 passes via second arm.
    result = g.gfql(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE a.score > 10 OR c.id = 'c1' "
        "RETURN c.id AS cid ORDER BY cid"
    )

    assert result._nodes.to_dict(orient="records") == [{"cid": "c1"}]


@pytest.mark.parametrize(
    ("predicate", "expected_cid", "expected_null"),
    [("IS NULL", "c_null", True), ("IS NOT NULL", "c_value", False)],
)
def test_string_cypher_executes_with_match_reentry_secondary_alias_null_property_where(
    predicate: str,
    expected_cid: str,
    expected_null: bool,
) -> None:
    """Demoted secondary alias properties preserve null semantics in trailing WHERE."""
    nodes = pd.DataFrame(
        {
            "id": ["a_null", "a_value", "b_null", "b_value", "c_null", "c_value"],
            "label__A": [True, True, False, False, False, False],
            "label__B": [False, False, True, True, False, False],
            "label__C": [False, False, False, False, True, True],
            "score": pd.Series([None, 7, None, None, None, None], dtype="Int64"),
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a_null", "a_value", "b_null", "b_value"],
            "d": ["b_null", "b_value", "c_null", "c_value"],
            "type": ["R", "R", "S", "S"],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        f"WHERE a.score {predicate} "
        "RETURN c.id AS cid, a.score AS score"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["cid"] == expected_cid
    assert bool(pd.isna(rows[0]["score"])) is expected_null


@pytest.mark.parametrize(
    ("predicate", "expected_cid", "expected_null"),
    [("IS NULL", "c_null", True), ("IS NOT NULL", "c_value", False)],
)
def test_string_cypher_executes_with_match_reentry_secondary_alias_null_property_where_on_cudf(
    predicate: str,
    expected_cid: str,
    expected_null: bool,
) -> None:
    """cuDF parity for demoted secondary alias null filtering."""
    cudf = _require_cudf_runtime()
    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["a_null", "a_value", "b_null", "b_value", "c_null", "c_value"],
                "label__A": [True, True, False, False, False, False],
                "label__B": [False, False, True, True, False, False],
                "label__C": [False, False, False, False, True, True],
                "score": pd.Series([None, 7, None, None, None, None], dtype="Int64"),
            }
        )
    )
    edges = cudf.from_pandas(
        pd.DataFrame(
            {
                "s": ["a_null", "a_value", "b_null", "b_value"],
                "d": ["b_null", "b_value", "c_null", "c_value"],
                "type": ["R", "R", "S", "S"],
            }
        )
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        f"WHERE a.score {predicate} "
        "RETURN c.id AS cid, a.score AS score",
        engine="cudf",
    )

    rows = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["cid"] == expected_cid
    assert bool(pd.isna(rows[0]["score"])) is expected_null


def test_string_cypher_executes_three_alias_with_match_reentry() -> None:
    """Three-alias carry through WITH; secondaries referenced by property in RETURN (#1071)."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["x1", "y1", "z1"], "label": ["X", "Y", "Z"]}),
        pd.DataFrame(
            {
                "s": ["x1", "y1"],
                "d": ["y1", "z1"],
                "type": ["R", "R"],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (x)-[:R]->(y)-[:R]->(z) "
        "WITH x, y, z "
        "MATCH (z) "
        "RETURN x.label AS xl, y.label AS yl, z.label AS zl"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"xl": "X", "yl": "Y", "zl": "Z"},
    ]


def test_string_cypher_failfast_rejects_with_match_reentry_secondary_whole_row_return() -> None:
    """Returning a secondary whole-row alias is unsupported in MVP (#1071)."""
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN a, c.id AS cid"
    )

    with pytest.raises(GFQLValidationError, match="secondary whole-row aliases as whole-row outputs"):
        _mk_connected_reentry_carried_scalar_graph().gfql(query, params={"seed": "a1"})


def test_string_cypher_failfast_rejects_with_match_reentry_carried_relationship_alias() -> None:
    """#1358: carrying a relationship variable across re-entry must surface as a
    clean scope error citing the unsupported alias kind, not silently fall into
    untested code paths in the multi-whole-row prefix rewriter.

    The trailing MATCH binds a fresh node alias `c`, so the #1341 flattener
    does not admit this query — it falls through to the existing reentry path
    where the new classifier check fires.
    """
    query = (
        "MATCH (a:A {id: $seed})-[r:R]->(b:B) "
        "WITH a, r "
        "MATCH (a)-[:S]->(c:C) "
        "RETURN r.weight, c.id AS cid"
    )

    with pytest.raises(
        GFQLValidationError,
        match="does not yet support carrying a relationship variable",
    ):
        _mk_connected_reentry_carried_scalar_graph().gfql(query, params={"seed": "a1"})


def test_string_cypher_failfast_rejects_with_match_reentry_carried_path_alias() -> None:
    """#1358: carrying a named path alias across re-entry must surface as a
    clean scope error. Mirrors the relationship-variable case via the path-alias
    branch of ``MatchClause.pattern_aliases``.
    """
    query = (
        "MATCH path = (a:A {id: $seed})-[:R]->(b:B) "
        "WITH path, b "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN length(path), c.id AS cid"
    )

    with pytest.raises(
        GFQLValidationError,
        match="does not yet support carrying a named path alias",
    ):
        _mk_connected_reentry_carried_scalar_graph().gfql(query, params={"seed": "a1"})


def test_unit_all_match_alias_kinds_lets_rel_kind_win_over_node() -> None:
    """#1358: when a name is bound as both a node and a relationship variable
    across patterns (parser-permitted), the alias-kinds classifier must record
    the relationship kind so the pre-flight still flags the unsupported carry
    rather than silently admitting via the node fallback.

    Bypasses lowering (which rejects the multi-pattern shape under a different
    rule) and exercises the classifier helper directly on the parsed AST.
    """
    from graphistry.compute.gfql.cypher.reentry.lowering_support import _all_match_alias_kinds

    parsed = parse_cypher(
        "MATCH (x:X) "
        "MATCH (a:A)-[x:R]->(b:B) "
        "WITH a, x "
        "MATCH (a)-[:S]->(c:C) "
        "RETURN x.weight, c.id"
    )
    assert isinstance(parsed, CypherQuery)
    kinds = _all_match_alias_kinds(parsed)
    assert kinds.get("x") == "rel", (
        "rel binding must override the prior node binding so the pre-flight still "
        "flags the unsupported carry"
    )
    assert kinds.get("a") == "node"
    assert kinds.get("b") == "node"


def test_string_cypher_executes_with_match_reentry_secondary_alias_rebinding() -> None:
    """Re-binding a carried secondary alias as a node variable in the trailing
    MATCH is admitted by flattening when the trailing pattern adds structure
    (#1341). The flattener merges prefix and trailing patterns into a single
    MATCH so the supported single-MATCH paths handle the rebind directly."""
    nodes = pd.DataFrame(
        {
            "id": ["a1", "b1", "a2", "b2"],
            "label__A": [True, False, True, False],
            "label__B": [False, True, False, True],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a1", "b1", "a2"],
            "d": ["b1", "a1", "b2"],
            "type": ["R", "S", "R"],
        }
    )
    g = _mk_graph(nodes, edges)

    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id AS bid"
    )

    result = g.gfql(query, params={"seed": "a1"})
    assert result._nodes.to_dict(orient="records") == [{"bid": "b1"}]


def test_string_cypher_executes_with_match_reentry_secondary_alias_rebinding_empty_when_no_back_edge() -> None:
    """Same flattened rebind shape returns empty when no back-edge exists (#1341)."""
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id AS bid"
    )

    result = _mk_connected_reentry_carried_scalar_graph().gfql(query, params={"seed": "a1"})
    assert result._nodes.to_dict(orient="records") == []


def _mk_ic1_shortest_path_graph() -> _CypherTestGraph:
    """LDBC SNB IC1 fixture: 4-person social graph with KNOWS edges."""
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4"],
                "label__Person": [True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p1"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_string_cypher_executes_ic1_shortest_path_with_carried_endpoints_rebound_length_dtype_parity(
    engine: str,
) -> None:
    """IC1 carried-endpoint shortestPath parity on values + numeric dtype (#1354)."""
    nodes = pd.DataFrame(
        {
            "id": ["p1", "p2", "p3", "p4"],
            "label__Person": [True, True, True, True],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["p1", "p2", "p1"],
            "d": ["p2", "p3", "p4"],
            "type": ["KNOWS", "KNOWS", "KNOWS"],
        }
    )
    query = (
        "MATCH (p:Person {id: 'p1'}), (friend:Person) "
        "WHERE NOT p = friend "
        "WITH p, friend "
        "MATCH path = shortestPath((p)-[:KNOWS*1..3]-(friend)) "
        "RETURN friend.id AS friendId, length(path) AS dist "
        "ORDER BY friendId"
    )
    if engine == "cudf":
        _require_cudf_runtime()
        result = _mk_cudf_graph(nodes, edges).gfql(query, engine="cudf")
        assert type(result._nodes).__module__.startswith("cudf")
        rows_df = result._nodes
        friend_ids = rows_df["friendId"].to_arrow().to_pylist()
        dists = rows_df["dist"].astype("float64").to_arrow().to_pylist()
        assert rows_df["dist"].dtype.kind in ("i", "u", "f")
    else:
        rows_df = _mk_graph(nodes, edges).gfql(query)._nodes
        rows = rows_df.to_dict(orient="records")
        friend_ids = [r["friendId"] for r in rows]
        dists = [float(r["dist"]) for r in rows]
        assert pd.api.types.is_numeric_dtype(rows_df["dist"])

    assert friend_ids == ["p2", "p3", "p4"]
    assert dists == [1.0, 2.0, 1.0]


def test_string_cypher_flatten_admit_matches_hand_flattened_oracle_ic1() -> None:
    """Round-001 amplification (#1341): admit-side correctness oracle.

    Compares the IC1 shape (which flatten admits and lowers via the merged
    single-MATCH path) against the manually-flattened equivalent that the
    same path would produce. Locks in admit-side correctness as more than
    "doesn't error" — admit must produce semantically identical results to
    the hand-written single-MATCH form."""
    g = _mk_ic1_shortest_path_graph()
    with_form = (
        "MATCH (p:Person {id: 'p1'}), (friend:Person) "
        "WHERE NOT p = friend "
        "WITH p, friend "
        "MATCH path = shortestPath((p)-[:KNOWS*1..3]-(friend)) "
        "RETURN friend.id AS friendId, length(path) AS dist "
        "ORDER BY friendId"
    )
    flat_form = (
        "MATCH (p:Person {id: 'p1'}), (friend:Person), "
        "  path = shortestPath((p)-[:KNOWS*1..3]-(friend)) "
        "WHERE NOT p = friend "
        "RETURN friend.id AS friendId, length(path) AS dist "
        "ORDER BY friendId"
    )
    via_flatten = g.gfql(with_form)._nodes.to_dict(orient="records")
    via_oracle = g.gfql(flat_form)._nodes.to_dict(orient="records")
    assert via_flatten == via_oracle


def test_string_cypher_flatten_admit_matches_hand_flattened_oracle_simple_rebind() -> None:
    """Same admit-side oracle for the simpler rebind shape (no shortestPath)."""
    nodes = pd.DataFrame(
        {
            "id": ["a1", "b1", "a2", "b2"],
            "label__A": [True, False, True, False],
            "label__B": [False, True, False, True],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a1", "b1", "a2"],
            "d": ["b1", "a1", "b2"],
            "type": ["R", "S", "R"],
        }
    )
    g = _mk_graph(nodes, edges)
    with_form = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(a) "
        "RETURN b.id AS bid"
    )
    flat_form = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B), (b)-[:S]->(a) "
        "RETURN b.id AS bid"
    )
    via_flatten = g.gfql(with_form, params={"seed": "a1"})._nodes.to_dict(orient="records")
    via_oracle = g.gfql(flat_form, params={"seed": "a1"})._nodes.to_dict(orient="records")
    assert via_flatten == via_oracle


def test_string_cypher_executes_with_match_reentry_relationship_variable_carried_on_cudf() -> None:
    """cuDF parity + no-crash regression for #1355.

    Covers both:
    - WITH-form that can flatten to connected comma-pattern
    - Hand-flattened single-MATCH connected comma-pattern
    """
    pytest.importorskip("cudf")
    nodes = pd.DataFrame(
        {
            "id": ["a1", "b1"],
            "label__A": [True, False],
            "label__B": [False, True],
        }
    )
    edges = pd.DataFrame(
        {"s": ["a1", "b1"], "d": ["b1", "a1"], "type": ["R", "S"], "weight": [7, 9]}
    )
    g = _mk_cudf_graph(nodes, edges)
    with_form = (
        "MATCH (a:A {id: 'a1'})-[r:R]->(b:B) "
        "WITH a, b, r "
        "MATCH (b)-[:S]->(a) "
        "RETURN r.weight AS w"
    )
    hand_flattened = (
        "MATCH (a:A {id: 'a1'})-[r:R]->(b:B), (b)-[:S]->(a) "
        "RETURN r.weight AS w"
    )

    for query in (with_form, hand_flattened):
        result = g.gfql(query, engine="cudf")
        assert type(result._nodes).__module__.startswith("cudf")
        assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"w": 7}]


def test_string_cypher_executes_with_match_reentry_multi_whole_row_alias_property_carry_on_cudf() -> None:
    """cuDF parity for the property-carry path (#1071)."""
    pytest.importorskip("cudf")
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN a.id AS aid, b.id AS bid, c.id AS cid"
    )

    result = _mk_connected_reentry_carried_scalar_graph_cudf().gfql(query, params={"seed": "a1"}, engine="cudf")

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"aid": "a1", "bid": "b1", "cid": "c1"}]


def test_string_cypher_executes_with_match_reentry_cross_alias_carried_scalars_from_connected_prefix_shape() -> None:
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH b, a.id AS aid "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN aid, c.id AS cid"
    )

    result = _mk_connected_reentry_carried_scalar_graph().gfql(query, params={"seed": "a1"})

    assert result._nodes.to_dict(orient="records") == [{"aid": "a1", "cid": "c1"}]


def test_string_cypher_reentry_carried_scalars_ignore_internal_hidden_column_collisions() -> None:
    g = _mk_reentry_carried_scalar_graph()
    g._nodes = g._nodes.assign(__cypher_reentry_property__=["orig1", "orig2", None, None])

    result = g.gfql(_reentry_query("a, a.num AS property", return_clause="property", order_by="property DESC"))

    assert result._nodes.to_dict(orient="records") == [{"property": 2}, {"property": 1}]


def test_string_cypher_reentry_carried_scalars_ignore_internal_hidden_column_collisions_on_cudf() -> None:
    pytest.importorskip("cudf")

    g = _mk_reentry_carried_scalar_graph_cudf()
    g._nodes = g._nodes.assign(__cypher_reentry_property__=["orig1", "orig2", None, None])

    result = g.gfql(
        _reentry_query("a, a.num AS property", return_clause="property", order_by="property DESC"),
        engine="cudf",
    )

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"property": 2}, {"property": 1}]


def test_string_cypher_executes_with_match_reentry_carried_scalar_where() -> None:
    query = _reentry_query(
        "a, a.num AS property",
        return_clause="property, b.id AS id",
        where_clause="property = b.num",
        order_by="id",
    )

    result = _mk_reentry_carried_scalar_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"property": 1, "id": "b1"}]


def test_string_cypher_executes_with_match_reentry_carried_scalar_where_on_cudf() -> None:
    pytest.importorskip("cudf")

    query = _reentry_query(
        "a, a.num AS property",
        return_clause="property, b.id AS id",
        where_clause="property = b.num",
        order_by="id",
    )

    result = _mk_reentry_carried_scalar_graph_cudf().gfql(query, engine="cudf")

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"property": 1, "id": "b1"}]


def test_string_cypher_executes_with_match_reentry_secondary_alias_property_where() -> None:
    query = (
        "MATCH (a:A {id: 'a1'})-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE a.id = 'a1' "
        "RETURN a.id AS aid, c.id AS cid "
        "ORDER BY cid"
    )

    result = _mk_connected_reentry_carried_scalar_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"aid": "a1", "cid": "c1"}]


def test_string_cypher_executes_with_match_reentry_where_or_on_carried_and_trailing_alias_props() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE a.id = 'a1' OR c.id = 'missing' "
        "RETURN a.id AS aid, c.id AS cid "
        "ORDER BY cid"
    )

    result = _mk_connected_reentry_carried_scalar_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"aid": "a1", "cid": "c1"}]


def test_string_cypher_executes_with_match_reentry_where_xor_on_carried_and_trailing_alias_props() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE a.id = 'a1' XOR c.id = 'c1' "
        "RETURN a.id AS aid, c.id AS cid"
    )

    result = _mk_connected_reentry_carried_scalar_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == []


def test_string_cypher_executes_with_match_reentry_preserves_orderby_single_column_limit_prefix() -> None:
    query = (
        "MATCH (p:Person {id: 'p0'})-[:KNOWS]->(friend:Person) "
        "WITH friend "
        "ORDER BY friend.firstName ASC "
        "LIMIT 2 "
        "MATCH (friend)-[:STUDY_AT]->(uni:University) "
        "RETURN friend.id AS friendId, uni.id AS uniId "
        "ORDER BY friendId, uniId"
    )
    result = _mk_reentry_order_limit_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [
        {"friendId": "f1", "uniId": "u1"},
        {"friendId": "f2", "uniId": "u2"},
        {"friendId": "f2", "uniId": "u3"},
    ]


def test_string_cypher_executes_with_match_reentry_preserves_orderby_multi_column_limit_prefix() -> None:
    query = (
        "MATCH (p:Person {id: 'p0'})-[:KNOWS]->(friend:Person) "
        "WITH friend "
        "ORDER BY friend.firstName DESC, friend.id ASC "
        "LIMIT 2 "
        "MATCH (friend)-[:STUDY_AT]->(uni:University) "
        "RETURN friend.id AS friendId, uni.id AS uniId "
        "ORDER BY friendId, uniId"
    )
    result = _mk_reentry_order_limit_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [
        {"friendId": "f3", "uniId": "u4"},
        {"friendId": "f4", "uniId": "u5"},
    ]


def test_string_cypher_executes_with_match_reentry_preserves_orderby_desc_limit_prefix() -> None:
    query = (
        "MATCH (p:Person {id: 'p0'})-[:KNOWS]->(friend:Person) "
        "WITH friend "
        "ORDER BY friend.firstName DESC "
        "LIMIT 1 "
        "MATCH (friend)-[:STUDY_AT]->(uni:University) "
        "RETURN friend.id AS friendId, uni.id AS uniId "
        "ORDER BY friendId, uniId"
    )
    result = _mk_reentry_order_limit_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"friendId": "f4", "uniId": "u5"}]


def test_string_cypher_executes_with_match_reentry_preserves_orderby_limit_prefix_on_cudf() -> None:
    pytest.importorskip("cudf")
    query = (
        "MATCH (p:Person {id: 'p0'})-[:KNOWS]->(friend:Person) "
        "WITH friend "
        "ORDER BY friend.firstName ASC "
        "LIMIT 2 "
        "MATCH (friend)-[:STUDY_AT]->(uni:University) "
        "RETURN friend.id AS friendId, uni.id AS uniId "
        "ORDER BY friendId, uniId"
    )
    result = _mk_reentry_order_limit_graph_cudf().gfql(query, engine="cudf")
    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"friendId": "f1", "uniId": "u1"},
        {"friendId": "f2", "uniId": "u2"},
        {"friendId": "f2", "uniId": "u3"},
    ]


@pytest.mark.parametrize(
    "query",
    [
        (
            "MATCH (p:Person {id: 'p0'})-[:KNOWS]->(friend:Person) "
            "WITH friend "
            "ORDER BY friend.firstName ASC "
            "MATCH (friend)-[:STUDY_AT]->(uni:University) "
            "RETURN friend.id AS friendId"
        ),
        (
            "MATCH (p:Person {id: 'p0'})-[:KNOWS]->(friend:Person) "
            "WITH friend "
            "ORDER BY friend.firstName ASC "
            "SKIP 1 LIMIT 2 "
            "MATCH (friend)-[:STUDY_AT]->(uni:University) "
            "RETURN friend.id AS friendId"
        ),
    ],
)
def test_string_cypher_failfast_rejects_with_match_reentry_unbounded_or_skip_order_shapes(query: str) -> None:
    with pytest.raises(GFQLValidationError, match="requires bounded literal LIMIT"):
        _mk_reentry_order_limit_graph().gfql(query)


def test_string_cypher_reentry_prefix_order_error_preserves_context() -> None:
    query = (
        "MATCH (p:Person {id: 'p0'})-[:KNOWS]->(friend:Person) "
        "WITH friend "
        "ORDER BY friend.firstName ASC "
        "MATCH (friend)-[:STUDY_AT]->(uni:University) "
        "RETURN friend.id AS friendId"
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_reentry_order_limit_graph().gfql(query)

    err = exc_info.value
    assert err.code == ErrorCode.E108
    assert err.context["field"] == "with.order_by"
    assert err.context["value"] == ["friend.firstName"]
    assert err.context["line"] == 1
    assert err.context["column"] == 67
    assert err.context["language"] == "cypher"


def test_reentry_order_by_rewrite_aborts_when_expression_cannot_rewrite() -> None:
    from graphistry.compute.gfql.cypher.reentry.lowering_support import _rewrite_order_by_expressions

    span = SourceSpan(1, 1, 1, 17, 0, 16)
    order_by = OrderByClause(
        items=(OrderItem(ExpressionText("friend.firstName", span), "asc", span),),
        span=span,
    )
    seen: List[Tuple[str, str]] = []

    def _cannot_rewrite(expr: ExpressionText, field: str) -> Optional[ExpressionText]:
        seen.append((expr.text, field))
        return None

    assert _rewrite_order_by_expressions(order_by, _cannot_rewrite) is None
    assert seen == [("friend.firstName", "order_by")]


def test_reentry_first_pattern_node_alias_handles_empty_pattern_list() -> None:
    from graphistry.compute.gfql.cypher.ast import MatchClause
    from graphistry.compute.gfql.cypher.reentry.lowering_support import _first_pattern_node_alias

    span = SourceSpan(1, 1, 1, 1, 0, 0)

    assert _first_pattern_node_alias(MatchClause(patterns=(), span=span)) is None


def test_reentry_where_predicate_text_rejects_unrenderable_structured_predicates() -> None:
    from graphistry.compute.gfql.cypher.ast import LabelRef, PropertyRef, WherePredicate
    from graphistry.compute.gfql.cypher.lowering import _row_where_predicate_text

    span = SourceSpan(1, 1, 1, 1, 0, 0)

    assert _row_where_predicate_text(
        WherePredicate(
            left=LabelRef(alias="a", labels=("A",), span=span),
            op="has_labels",
            right=None,
            span=span,
        )
    ) is None
    assert _row_where_predicate_text(
        WherePredicate(
            left=PropertyRef(alias="a", property="score", span=span),
            op="is_null",
            right=PropertyRef(alias="b", property="score", span=span),
            span=span,
        )
    ) is None


def test_reentry_secondary_alias_rewrite_respects_shadowed_expression_vars() -> None:
    from graphistry.compute.gfql.cypher.reentry.lowering_support import _collect_secondary_property_refs

    expressions = (
        "[x IN [1, 2] | x]",
        "ANY(x IN [1, 2] WHERE x > 1)",
    )
    for expression in expressions:
        span = SourceSpan(1, 1, 1, 1 + len(expression), 0, len(expression))
        expr = ExpressionText(expression, span)

        rewritten, refs, bare = _collect_secondary_property_refs(
            expr,
            secondary_aliases={"x"},
            field="return",
        )

        assert rewritten == expr
        assert refs == set()
        assert bare == set()


def test_string_cypher_executes_seeded_multihop_then_with_match_reentry_shape() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "label__A": [True, False, False, False],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "b", "c"],
            "d": ["b", "c", "d"],
            "type": ["R", "R", "R"],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) MATCH (a)-[:R*2]->(b) WITH b MATCH (b)-[:R]->(c) RETURN c.id AS id"
    )

    assert result._nodes.to_dict(orient="records") == [{"id": "d"}]


def _job_referral_employment_company_shape() -> Tuple[pd.DataFrame, pd.DataFrame, str, List[Dict[str, Any]]]:
    nodes = pd.DataFrame(
        [
            {"id": 1, "label__Person": True},
            {"id": 2, "label__Person": True, "firstName": "Bob", "lastName": "Baker"},
            {"id": 3, "label__Person": True, "firstName": "Carol", "lastName": "Clark"},
            {"id": 4, "label__Company": True, "name": "Zoo"},
            {"id": 5, "label__Company": True, "name": "Alpha"},
            {"id": 6, "label__Country": True, "name": "Hungary"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"s": 1, "d": 2, "type": "KNOWS"},
            {"s": 2, "d": 3, "type": "KNOWS"},
            {"s": 2, "d": 4, "type": "WORK_AT", "workFrom": 2010},
            {"s": 3, "d": 5, "type": "WORK_AT", "workFrom": 2009},
            {"s": 4, "d": 6, "type": "IS_LOCATED_IN"},
            {"s": 5, "d": 6, "type": "IS_LOCATED_IN"},
        ]
    )
    query = (
        "MATCH (person:Person {id: 1})-[:KNOWS*1..2]-(friend:Person) "
        "WHERE NOT(person=friend) "
        "WITH DISTINCT friend "
        "MATCH (friend)-[workAt:WORK_AT]->(company:Company)-[:IS_LOCATED_IN]->(:Country {name: 'Hungary'}) "
        "WHERE workAt.workFrom < 2011 "
        "RETURN "
        "friend.id AS personId, "
        "friend.firstName AS personFirstName, "
        "friend.lastName AS personLastName, "
        "company.name AS organizationName, "
        "workAt.workFrom AS organizationWorkFromYear "
        "ORDER BY organizationWorkFromYear ASC, toInteger(personId) ASC, organizationName DESC "
        "LIMIT 10"
    )
    expected = [
        {
            "personId": 3,
            "personFirstName": "Carol",
            "personLastName": "Clark",
            "organizationName": "Alpha",
            "organizationWorkFromYear": 2009.0,
        },
        {
            "personId": 2,
            "personFirstName": "Bob",
            "personLastName": "Baker",
            "organizationName": "Zoo",
            "organizationWorkFromYear": 2010.0,
        },
    ]
    return nodes, edges, query, expected


def test_string_cypher_executes_job_referral_employment_company_row_join_shape() -> None:
    nodes, edges, query, expected = _job_referral_employment_company_shape()

    result = _mk_graph(nodes, edges).gfql(query)

    assert result._nodes.to_dict(orient="records") == expected


def test_string_cypher_executes_job_referral_employment_company_row_join_shape_on_cudf() -> None:
    _require_cudf_runtime()
    nodes, edges, query, expected = _job_referral_employment_company_shape()

    result = _mk_cudf_graph(nodes, edges).gfql(query, engine="cudf")

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == expected


def test_string_cypher_executes_seeded_multihop_then_with_optional_match_reentry_shape() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "label__A": [True, False, False, False],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "b", "c"],
            "d": ["b", "c", "d"],
            "type": ["R", "R", "R"],
        }
    )

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a:A) MATCH (a)-[:R*2]->(b) WITH b OPTIONAL MATCH (b)-[:R]->(c) RETURN c.id AS id"
    )

    assert result._nodes.to_dict(orient="records") == [{"id": "d"}]


def test_string_cypher_executes_connected_whole_row_plus_scalar_projection() -> None:
    result = _mk_multi_stage_reentry_graph().gfql(
        "MATCH (b:B)-[:S]->(c:C) "
        "RETURN c, b.id AS bid"
    )

    assert entity_text_records(result, {"c": "nodes"}) == [{"c": "(:C)", "bid": "b"}]


def test_string_cypher_executes_multi_stage_with_match_reentry_connected_shape() -> None:
    result = _mk_multi_stage_reentry_graph().gfql(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH c "
        "MATCH (c)-[:T]->(d:D) "
        "RETURN d.id AS id"
    )

    assert result._nodes.to_dict(orient="records") == [{"id": "d"}]


def test_string_cypher_executes_multi_stage_with_match_reentry_carried_scalar_shape() -> None:
    result = _mk_multi_stage_reentry_graph().gfql(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "WITH c, bid "
        "MATCH (c)-[:T]->(d:D) "
        "RETURN bid, d.id AS id"
    )

    assert result._nodes.to_dict(orient="records") == [{"bid": "b", "id": "d"}]


def test_string_cypher_executes_multi_stage_with_match_reentry_connected_shape_on_cudf() -> None:
    pytest.importorskip("cudf")

    result = _mk_multi_stage_reentry_graph_cudf().gfql(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH c "
        "MATCH (c)-[:T]->(d:D) "
        "RETURN d.id AS id",
        engine="cudf",
    )

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"id": "d"}]


def test_string_cypher_executes_multi_stage_with_match_reentry_empty_result_shape() -> None:
    result = _mk_multi_stage_reentry_graph().gfql(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH c "
        "MATCH (c)-[:X]->(z:D) "
        "RETURN z.id AS id"
    )

    assert result._nodes.to_dict(orient="records") == []


def test_string_cypher_executes_multi_stage_with_match_reentry_with_intermediate_where() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "WITH c "
        "MATCH (c)-[:T]->(d:D) "
        "RETURN d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"id": "d"}]


def test_string_cypher_executes_post_with_reentry_where_direct_return() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "RETURN c.id AS cid"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"cid": "c"}]


def test_string_cypher_executes_multi_stage_with_match_reentry_with_intermediate_where_and_carried_scalar() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "WITH c, bid "
        "MATCH (c)-[:T]->(d:D) "
        "RETURN bid, d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"bid": "b", "id": "d"}]


def test_string_cypher_executes_multi_stage_with_match_reentry_with_intermediate_where_and_order_by() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "WITH c, bid "
        "MATCH (c)-[:T]->(d:D) "
        "RETURN bid, d.id AS id "
        "ORDER BY id DESC, bid ASC"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"bid": "b", "id": "d"}]


def test_string_cypher_executes_multi_stage_with_match_reentry_with_intermediate_where_empty_result() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'x' "
        "WITH c, bid "
        "MATCH (c)-[:T]->(d:D) "
        "RETURN bid, d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == []


def test_string_cypher_failfast_rejects_direct_match_after_post_with_where_without_intervening_with() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "MATCH (c)-[:T]->(d:D) "
        "RETURN d.id AS id"
    )

    with pytest.raises(
        GFQLSyntaxError,
        match="Cypher MATCH after post-WITH WHERE is not yet supported",
    ):
        _mk_multi_stage_reentry_graph().gfql(query)


def test_string_cypher_executes_multiple_post_with_where_clauses() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "WITH c "
        "MATCH (c)-[:T]->(d:D) "
        "WHERE d.id = 'd' "
        "WITH d "
        "RETURN d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"id": "d"}]


def test_string_cypher_executes_multiple_post_with_where_clauses_with_carried_scalar() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "WITH c, bid "
        "MATCH (c)-[:T]->(d:D) "
        "WHERE bid = 'b' AND d.id = 'd' "
        "WITH d, bid "
        "RETURN bid, d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"bid": "b", "id": "d"}]


def test_string_cypher_executes_sparse_multiple_post_with_where_clauses() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH c "
        "MATCH (c)-[:T]->(d:D) "
        "WHERE d.id = 'd' "
        "WITH d "
        "RETURN d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"id": "d"}]


def test_string_cypher_executes_multiple_post_with_where_clauses_empty_second_stage() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "WITH c "
        "MATCH (c)-[:T]->(d:D) "
        "WHERE d.id = 'x' "
        "WITH d "
        "RETURN d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == []


def test_string_cypher_executes_multiple_post_with_where_clauses_on_cudf() -> None:
    pytest.importorskip("cudf")

    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "WITH c "
        "MATCH (c)-[:T]->(d:D) "
        "WHERE d.id = 'd' "
        "WITH d "
        "RETURN d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph_cudf().gfql(query, engine="cudf")

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"id": "d"}]


def test_string_cypher_failfast_rejects_direct_match_after_second_post_with_where_without_intervening_with() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "WITH c "
        "MATCH (c)-[:T]->(d:D) "
        "WHERE d.id = 'd' "
        "MATCH (d)-[:U]->(e) "
        "RETURN e.id AS id"
    )

    with pytest.raises(
        GFQLSyntaxError,
        match="Cypher MATCH after post-WITH WHERE is not yet supported",
    ):
        _mk_multi_stage_reentry_graph_with_terminal_u().gfql(query)


def test_string_cypher_executes_post_with_match_collect_unwind_match_before_return() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH collect(distinct c) AS cs "
        "UNWIND cs AS c2 "
        "MATCH (c2)-[:T]->(d:D) "
        "RETURN d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"id": "d"}]


def test_string_cypher_executes_post_with_match_collect_unwind_match_with_carried_scalar() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "WITH bid, collect(distinct c) AS cs "
        "UNWIND cs AS c2 "
        "MATCH (c2)-[:T]->(d:D) "
        "RETURN bid, d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"bid": "b", "id": "d"}]


def test_string_cypher_executes_post_with_match_collect_unwind_match_final_with_carried_scalar() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "WITH bid, collect(distinct c) AS cs "
        "UNWIND cs AS c2 "
        "MATCH (c2)-[:T]->(d:D) "
        "WITH d, bid "
        "RETURN bid, d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"bid": "b", "id": "d"}]


def test_string_cypher_executes_post_with_match_collect_unwind_match_final_with_order_by_limit() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "WITH bid, collect(distinct c) AS cs "
        "UNWIND cs AS c2 "
        "MATCH (c2)-[:T]->(d:D) "
        "WITH d, bid "
        "ORDER BY d.id DESC "
        "LIMIT 1 "
        "RETURN bid, d.id AS id"
    )

    result = _mk_connected_multi_pattern_fanout_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1", "id": "d2"}]


def test_string_cypher_executes_post_with_match_collect_unwind_match_empty_result() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH collect(distinct c) AS cs "
        "UNWIND cs AS c2 "
        "MATCH (c2)-[:X]->(d:D) "
        "RETURN d.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == []


def test_string_cypher_failfast_rejects_post_with_match_non_collect_unwind_match_shape() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH c "
        "UNWIND [c] AS c2 "
        "MATCH (c2)-[:T]->(d:D) "
        "RETURN d.id AS id"
    )

    with pytest.raises(
        GFQLValidationError,
        match="supports only a single WITH collect\\(\\[distinct\\] alias\\) AS list UNWIND list AS alias MATCH \\.\\.\\. RETURN shape",
    ):
        _mk_multi_stage_reentry_graph().gfql(query)


def test_string_cypher_executes_post_with_match_unwind_after_reentry_passthrough_with() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "WITH c, bid "
        "UNWIND [c] AS c2 "
        "RETURN bid, c2.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"bid": "b", "id": "c"}]


def test_string_cypher_executes_post_with_match_unwind_after_reentry_passthrough_with_on_cudf() -> None:
    pytest.importorskip("cudf")

    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "WITH c, bid "
        "UNWIND [c] AS c2 "
        "RETURN bid, c2.id AS id"
    )

    result = _mk_multi_stage_reentry_graph_cudf().gfql(query, engine="cudf")

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"bid": "b", "id": "c"}]


def test_string_cypher_failfast_rejects_multiple_post_with_match_unwinds() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "UNWIND [c] AS c2 "
        "UNWIND [c2] AS c3 "
        "RETURN c3.id AS id"
    )

    with pytest.raises(
        GFQLSyntaxError,
        match="Cypher only supports one UNWIND after post-WITH MATCH",
    ):
        _mk_multi_stage_reentry_graph().gfql(query)


def test_string_cypher_failfast_rejects_match_after_post_with_match_unwind() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH collect(distinct c) AS cs "
        "UNWIND cs AS c2 "
        "MATCH (c2)-[:T]->(d:D) "
        "MATCH (d)-[:Z]->(e) "
        "RETURN e.id AS id"
    )

    with pytest.raises(
        GFQLSyntaxError,
        match="Cypher MATCH after post-WITH MATCH UNWIND is not yet supported",
    ):
        _mk_multi_stage_reentry_graph().gfql(query)


def test_string_cypher_executes_post_with_match_with_before_return() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH c "
        "RETURN c.id AS id"
    )

    result = _mk_multi_stage_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"id": "c"}]


def test_issue_1000_ic6_compiles_after_phase6_connected_star_fanout() -> None:
    compiled = compile_cypher(_issue_1000_ic6_query(), params=_issue_1000_ic6_params())

    assert compiled is not None


def test_string_cypher_executes_issue_1000_ic6_exact_runtime_minimal() -> None:
    result = _mk_issue_1000_ic6_minimal_graph().gfql(
        _issue_1000_ic6_query(),
        params=_issue_1000_ic6_params(),
    )

    assert result._nodes.to_dict(orient="records") == [
        {"tagName": "Alpha", "postCount": 1},
        {"tagName": "Beta", "postCount": 1},
    ]


def test_string_cypher_executes_issue_1000_ic6_exact_runtime_minimal_on_cudf() -> None:
    pytest.importorskip("cudf")

    result = _mk_issue_1000_ic6_minimal_graph_cudf().gfql(
        _issue_1000_ic6_query(),
        params=_issue_1000_ic6_params(),
        engine="cudf",
    )

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"tagName": "Alpha", "postCount": 1},
        {"tagName": "Beta", "postCount": 1},
    ]


def test_issue_1396_issue_1415_tag_cooccurrence_join_aggregation_counts() -> None:
    """IC6 tag-cooccurrence join+aggregation shape keeps grouped post cardinality."""
    result = _mk_issue_1396_tag_cooccurrence_join_aggregation_graph().gfql(
        _issue_1000_ic6_query(),
        params=_issue_1000_ic6_params(),
    )

    assert result._nodes.to_dict(orient="records") == [
        {"tagName": "Alpha", "postCount": 2},
        {"tagName": "Beta", "postCount": 1},
    ]


def test_issue_1396_issue_1415_tag_cooccurrence_join_aggregation_counts_on_cudf() -> None:
    pytest.importorskip("cudf")

    result = _mk_issue_1396_tag_cooccurrence_join_aggregation_graph_cudf().gfql(
        _issue_1000_ic6_query(),
        params=_issue_1000_ic6_params(),
        engine="cudf",
    )

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"tagName": "Alpha", "postCount": 2},
        {"tagName": "Beta", "postCount": 1},
    ]


def test_string_cypher_executes_scalar_only_prefix_with_match_reentry() -> None:
    query = _prefix_scalar_reentry_query(order_by="id")

    result = _mk_prefix_scalar_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"id": "post1"}, {"id": "post2"}]


def test_string_cypher_executes_scalar_only_prefix_with_match_reentry_scalar_projection() -> None:
    query = _prefix_scalar_reentry_query(
        return_clause="knownTagId, post.id AS postId",
        order_by="postId",
    )

    result = _mk_prefix_scalar_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [
        {"knownTagId": 101, "postId": "post1"},
        {"knownTagId": 101, "postId": "post2"},
    ]


def test_string_cypher_executes_scalar_only_prefix_with_match_reentry_multiple_scalars() -> None:
    query = _prefix_scalar_reentry_query(
        with_clause="knownTag.tagId AS knownTagId, knownTag.name AS knownTagName",
        return_clause="knownTagId, knownTagName, post.id AS postId",
        order_by="postId",
    )

    result = _mk_prefix_scalar_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [
        {"knownTagId": 101, "knownTagName": "topic", "postId": "post1"},
        {"knownTagId": 101, "knownTagName": "topic", "postId": "post2"},
    ]


def test_string_cypher_executes_scalar_only_prefix_with_match_reentry_empty_prefix() -> None:
    query = _prefix_scalar_reentry_query(tag_name="missing")

    result = _mk_prefix_scalar_reentry_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == []


def test_string_cypher_executes_scalar_only_prefix_with_match_reentry_on_cudf() -> None:
    pytest.importorskip("cudf")

    query = _prefix_scalar_reentry_query(order_by="id")

    result = _mk_prefix_scalar_reentry_graph_cudf().gfql(query, engine="cudf")

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"id": "post1"}, {"id": "post2"}]


def test_string_cypher_executes_scalar_only_prefix_with_match_reentry_multi_row_prefix() -> None:
    """#1047: multi-row scalar prefix now executes (was rejected before fix)."""
    query = _prefix_scalar_reentry_query(order_by="id")

    result = _mk_prefix_scalar_reentry_duplicate_seed_graph().gfql(query)
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    # Both tag1 and tag1b have tagId=101; post1→tag1, post2→tag1b, so both match
    assert set(ids) == {"post1", "post2"}


def test_string_cypher_failfast_rejects_scalar_only_prefix_with_match_reentry_prefix_ordering() -> None:
    query = (
        "MATCH (knownTag:Tag { name: 'topic' }) "
        "WITH knownTag.tagId AS knownTagId "
        "ORDER BY knownTagId DESC "
        "MATCH (post:Post)-[:HAS_TAG]->(t:Tag {tagId: knownTagId}) "
        "RETURN post.id AS id"
    )

    with pytest.raises(
        GFQLValidationError,
        match="Cypher MATCH after WITH scalar-only prefix stages do not yet support ORDER BY, SKIP, or LIMIT",
    ):
        _mk_prefix_scalar_reentry_graph().gfql(query)


def test_string_cypher_failfast_rejects_scalar_only_prefix_with_match_reentry_prefix_limit() -> None:
    query = (
        "MATCH (knownTag:Tag { name: 'topic' }) "
        "WITH knownTag.tagId AS knownTagId "
        "LIMIT 1 "
        "MATCH (post:Post)-[:HAS_TAG]->(t:Tag {tagId: knownTagId}) "
        "RETURN post.id AS id"
    )

    with pytest.raises(
        GFQLValidationError,
        match="Cypher MATCH after WITH scalar-only prefix stages do not yet support ORDER BY, SKIP, or LIMIT",
    ):
        _mk_prefix_scalar_reentry_graph().gfql(query)

def test_string_cypher_failfast_rejects_scalar_only_prefix_alias_reused_as_node_variable() -> None:
    # #1357: cross-kind rebind is now caught at the binder via the
    # _bind_node_pattern entity_kind guard rather than in the re-entry
    # compiletime path. The intent (reject scalar→node rebind across WITH)
    # is preserved; the error message and surface moved earlier.
    with pytest.raises(
        GFQLValidationError,
        match="Cypher alias rebound as a different entity kind",
    ) as exc_info:
        _mk_reentry_carried_scalar_graph().gfql(
            "MATCH (a:A) WITH [a] AS users MATCH (users)-->(messages) RETURN messages.id AS mid"
        )
    assert exc_info.value.context["existing_kind"] == "scalar"
    assert exc_info.value.context["new_kind"] == "node"


def test_string_cypher_executes_scalar_prefix_reentry_connected_star_comma_fanout() -> None:
    query = (
        "MATCH (knownTag:Tag { name: 'topic' }) "
        "WITH knownTag.tagId AS knownTagId "
        "MATCH (f:Person)<-[:HAS_CREATOR]-(post:Post), "
        "(post)-[:HAS_TAG]->(t:Tag {tagId: knownTagId}), "
        "(post)-[:HAS_TAG]->(tag:Tag) "
        "WHERE NOT t = tag "
        "RETURN post.id AS postId, tag.name AS tagName "
        "ORDER BY postId, tagName"
    )

    result = _mk_connected_post_tag_fanout_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"postId": "post1", "tagName": "other"}]


def test_string_cypher_executes_scalar_prefix_reentry_connected_star_comma_fanout_grouped_count() -> None:
    query = (
        "MATCH (knownTag:Tag { name: 'topic' }) "
        "WITH knownTag.tagId AS knownTagId "
        "MATCH (f:Person)<-[:HAS_CREATOR]-(post:Post), "
        "(post)-[:HAS_TAG]->(t:Tag {tagId: knownTagId}), "
        "(post)-[:HAS_TAG]->(tag:Tag) "
        "WHERE NOT t = tag "
        "WITH tag.name AS tagName, count(post) AS postCount "
        "RETURN tagName, postCount"
    )

    result = _mk_connected_post_tag_fanout_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"tagName": "other", "postCount": 1}]


def test_string_cypher_executes_connected_star_comma_fanout_grouped_count_without_reentry() -> None:
    query = (
        "MATCH (f:Person)<-[:HAS_CREATOR]-(post:Post), "
        "(post)-[:HAS_TAG]->(t:Tag {tagId: 101}), "
        "(post)-[:HAS_TAG]->(tag:Tag) "
        "WHERE NOT t = tag "
        "RETURN tag.name AS tagName, count(post) AS postCount"
    )

    result = _mk_connected_post_tag_fanout_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"tagName": "other", "postCount": 1}]


def test_string_cypher_executes_scalar_prefix_reentry_connected_star_comma_fanout_on_cudf() -> None:
    pytest.importorskip("cudf")

    query = (
        "MATCH (knownTag:Tag { name: 'topic' }) "
        "WITH knownTag.tagId AS knownTagId "
        "MATCH (f:Person)<-[:HAS_CREATOR]-(post:Post), "
        "(post)-[:HAS_TAG]->(t:Tag {tagId: knownTagId}), "
        "(post)-[:HAS_TAG]->(tag:Tag) "
        "WHERE NOT t = tag "
        "RETURN post.id AS postId, tag.name AS tagName "
        "ORDER BY postId, tagName"
    )

    result = _mk_connected_post_tag_fanout_graph_cudf().gfql(query, engine="cudf")

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"postId": "post1", "tagName": "other"}]


def test_string_cypher_executes_connected_multi_pattern_grouped_aggregate_overlap() -> None:
    query = (
        "MATCH (b:B)-[:S]->(c:C), (c)-[:T]->(d:D) "
        "RETURN c.id AS cid, count(d) AS cnt "
        "ORDER BY cid"
    )

    result = _mk_connected_multi_pattern_fanout_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"cid": "c1", "cnt": 2}]


def test_string_cypher_executes_connected_multi_pattern_grouped_distinct_aggregate_overlap() -> None:
    query = (
        "MATCH (b:B)-[:S]->(c:C), (c)-[:T]->(d:D) "
        "RETURN c.id AS cid, count(DISTINCT d) AS cnt "
        "ORDER BY cid"
    )

    result = _mk_connected_multi_pattern_fanout_graph().gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"cid": "c1", "cnt": 2}]


def test_string_cypher_executes_with_match_reentry_connected_multi_pattern_grouped_aggregate_overlap() -> None:
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C), (c)-[:T]->(d:D) "
        "RETURN bid, count(d) AS cnt "
        "ORDER BY bid"
    )

    result = _mk_connected_multi_pattern_fanout_graph().gfql(query, params={"seed": "a1"})

    assert result._nodes.to_dict(orient="records") == [{"bid": "b1", "cnt": 2}]


def test_string_cypher_executes_connected_multi_pattern_grouped_aggregate_overlap_on_cudf() -> None:
    pytest.importorskip("cudf")

    query = (
        "MATCH (b:B)-[:S]->(c:C), (c)-[:T]->(d:D) "
        "RETURN c.id AS cid, count(d) AS cnt "
        "ORDER BY cid"
    )

    result = _mk_connected_multi_pattern_fanout_graph_cudf().gfql(query, engine="cudf")

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"cid": "c1", "cnt": 2}]
def test_cypher_to_gfql_supports_multi_alias_scalar_projection() -> None:
    """Multi-alias scalar projections are supported via bindings table."""
    chain = cypher_to_gfql("MATCH (p)-[r]->(q) RETURN p.id, q.id")
    assert chain is not None


def test_multi_alias_return_basic() -> None:
    """MATCH (a)-[:R]->(b) RETURN a.id, b.id produces one row per binding."""
    g = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "label__A": [True, False], "label__B": [False, True]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
    )
    result = g.gfql("MATCH (a:A)-[:R]->(b:B) RETURN a.id AS a_id, b.id AS b_id")
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(records) == 1
    assert records[0]["a_id"] == "a"
    assert records[0]["b_id"] == "b"


def test_multi_alias_return_multiple_bindings() -> None:
    """Multiple edges produce multiple binding rows."""
    g = _mk_graph(
        pd.DataFrame({"id": ["x", "y", "z"], "label__X": [True, False, False], "label__Y": [False, True, True]}),
        pd.DataFrame({"s": ["x", "x"], "d": ["y", "z"], "type": ["R", "R"]}),
    )
    result = g.gfql("MATCH (a:X)-[:R]->(b:Y) RETURN a.id AS a_id, b.id AS b_id ORDER BY b_id")
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(records) == 2
    assert records[0]["b_id"] == "y"
    assert records[1]["b_id"] == "z"


def test_multi_alias_return_empty_match() -> None:
    """No matching pattern produces empty result."""
    g = _mk_graph(
        pd.DataFrame({"id": ["a"], "label__A": [True]}),
        pd.DataFrame({"s": pd.Series(dtype="str"), "d": pd.Series(dtype="str"), "type": pd.Series(dtype="str")}),
    )
    result = g.gfql("MATCH (a:A)-[:R]->(b) RETURN a.id AS a_id, b.id AS b_id")
    assert len(result._nodes) == 0


def test_multi_alias_return_with_edge_alias_property() -> None:
    """Edge alias properties are accessible in multi-alias RETURN (#982)."""
    g = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "label__A": [True, False], "label__B": [False, True], "firstName": ["Alice", "Bob"]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["KNOWS"], "creationDate": [123]}),
    )
    result = g.gfql("MATCH (a:A)-[r:KNOWS]->(b:B) RETURN a.id AS a_id, r.creationDate AS cd, b.firstName AS name")
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(records) == 1
    assert records[0]["a_id"] == "a"
    assert records[0]["cd"] == 123
    assert records[0]["name"] == "Bob"


def test_multi_alias_return_missing_property_yields_null() -> None:
    """Missing alias-prefixed properties should project as null, not error."""
    g = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "label__A": [True, False], "label__B": [False, True]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
    )
    result = g.gfql("MATCH (a:A)-[:R]->(b:B) RETURN a.id AS a_id, a.nonexistent AS missing")
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(records) == 1
    assert records[0]["a_id"] == "a"
    assert records[0]["missing"] is None or pd.isna(records[0]["missing"])


def test_multi_alias_undirected_incoming_edge_returns_peer_not_seed() -> None:
    """#994: undirected MATCH with incoming edge must return peer, not seed."""
    g = _mk_graph(
        pd.DataFrame({"id": [1, 2], "label__Person": [True, True], "firstName": ["Alice", "Bob"]}),
        pd.DataFrame({"s": [2], "d": [1], "type": ["KNOWS"], "creationDate": [10]}),
    )
    result = g.gfql(
        "MATCH (n:Person {id: $pid})-[r:KNOWS]-(friend) "
        "RETURN friend.id AS fid, friend.firstName AS fname, r.creationDate AS cd",
        params={"pid": 1},
    )
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(records) == 1
    assert records[0]["fid"] == 2
    assert records[0]["fname"] == "Bob"
    assert records[0]["cd"] == 10


def test_multi_alias_undirected_outgoing_edge_returns_peer() -> None:
    """#994 regression: outgoing edge should still return peer correctly."""
    g = _mk_graph(
        pd.DataFrame({"id": [1, 2], "label__Person": [True, True], "firstName": ["Alice", "Bob"]}),
        pd.DataFrame({"s": [1], "d": [2], "type": ["KNOWS"], "creationDate": [10]}),
    )
    result = g.gfql(
        "MATCH (n:Person {id: $pid})-[r:KNOWS]-(friend) "
        "RETURN friend.id AS fid, friend.firstName AS fname, r.creationDate AS cd",
        params={"pid": 1},
    )
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(records) == 1
    assert records[0]["fid"] == 2
    assert records[0]["fname"] == "Bob"


def test_multi_alias_undirected_bidirectional_edges() -> None:
    """#994 adversarial: both incoming and outgoing edges to same peer."""
    g = _mk_graph(
        pd.DataFrame({"id": [1, 2], "label__Person": [True, True], "firstName": ["Alice", "Bob"]}),
        pd.DataFrame({"s": [1, 2], "d": [2, 1], "type": ["KNOWS", "KNOWS"], "creationDate": [10, 20]}),
    )
    result = g.gfql(
        "MATCH (n:Person {id: $pid})-[r:KNOWS]-(friend) "
        "RETURN friend.id AS fid, r.creationDate AS cd "
        "ORDER BY cd",
        params={"pid": 1},
    )
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(records) == 2
    assert all(r["fid"] == 2 for r in records), f"Both rows should reference Bob, got {records}"


def test_multi_alias_undirected_multiple_edges_same_nodes() -> None:
    """#994 amplification: 3 edges between same pair, undirected query returns all 3."""
    g = _mk_graph(
        pd.DataFrame({"id": [1, 2], "label__Person": [True, True], "firstName": ["Alice", "Bob"]}),
        pd.DataFrame({
            "s": [1, 1, 2],
            "d": [2, 2, 1],
            "type": ["KNOWS", "LIKES", "KNOWS"],
            "weight": [10, 20, 30],
        }),
    )
    result = g.gfql(
        "MATCH (n:Person {id: $pid})-[r]-(friend) "
        "RETURN friend.id AS fid, r.weight AS w "
        "ORDER BY w",
        params={"pid": 1},
    )
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(records) == 3, f"Expected 3 rows (one per edge), got {records}"
    assert all(r["fid"] == 2 for r in records), f"All rows should reference Bob, got {records}"
    assert [r["w"] for r in records] == [10, 20, 30]


def test_multi_alias_undirected_star_multiple_peers() -> None:
    """#994 amplification: undirected star — seed node 1 has edges to peers 2, 3, 4."""
    g = _mk_graph(
        pd.DataFrame({
            "id": [1, 2, 3, 4],
            "label__Person": [True, True, True, True],
            "firstName": ["Alice", "Bob", "Carol", "Dave"],
        }),
        pd.DataFrame({
            "s": [2, 1, 4],
            "d": [1, 3, 1],
            "type": ["KNOWS", "KNOWS", "KNOWS"],
        }),
    )
    result = g.gfql(
        "MATCH (n:Person {id: $pid})-[r:KNOWS]-(friend) "
        "RETURN friend.id AS fid, friend.firstName AS fname "
        "ORDER BY fid",
        params={"pid": 1},
    )
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(records) == 3, f"Expected 3 peers, got {records}"
    assert [r["fid"] for r in records] == [2, 3, 4]
    assert [r["fname"] for r in records] == ["Bob", "Carol", "Dave"]


def test_multi_alias_undirected_self_loop() -> None:
    """#994 amplification: self-loop edge where src==dst."""
    g = _mk_graph(
        pd.DataFrame({"id": [1, 2], "label__Person": [True, True], "firstName": ["Alice", "Bob"]}),
        pd.DataFrame({
            "s": [1, 1],
            "d": [1, 2],
            "type": ["SELF", "KNOWS"],
            "weight": [99, 10],
        }),
    )
    result = g.gfql(
        "MATCH (n:Person {id: $pid})-[r]-(friend) "
        "RETURN friend.id AS fid, r.weight AS w "
        "ORDER BY w",
        params={"pid": 1},
    )
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    # Self-loop in undirected traversal matches both directions → 2 rows for the self-loop.
    # KNOWS edge: 1 row with fid=2. Total: 3 rows.
    assert len(records) == 3, f"Expected 3 rows (self-loop×2 + KNOWS), got {records}"
    assert records[0]["w"] == 10  # KNOWS edge first (weight 10)
    assert records[0]["fid"] == 2
    assert all(r["fid"] == 1 for r in records[1:]), f"Self-loop rows should reference self, got {records}"


def test_multi_alias_return_star_graph() -> None:
    """Star graph: 1 hub -> 3 leaves produces 3 binding rows."""
    g = _mk_graph(
        pd.DataFrame({"id": ["hub", "a", "b", "c"], "label__Hub": [True, False, False, False], "label__Leaf": [False, True, True, True]}),
        pd.DataFrame({"s": ["hub", "hub", "hub"], "d": ["a", "b", "c"], "type": ["R", "R", "R"]}),
    )
    result = g.gfql("MATCH (h:Hub)-[:R]->(l:Leaf) RETURN h.id AS hub, l.id AS leaf ORDER BY leaf")
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(records) == 3
    assert [r["leaf"] for r in records] == ["a", "b", "c"]
    assert all(r["hub"] == "hub" for r in records)


def test_multi_alias_return_bidirectional() -> None:
    """Bidirectional edges produce one binding per directed edge."""
    g = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "label__X": [True, True], "val": [1, 2]}),
        pd.DataFrame({"s": ["a", "b"], "d": ["b", "a"], "type": ["R", "R"]}),
    )
    result = g.gfql("MATCH (x:X)-[:R]->(y:X) RETURN x.id AS x_id, y.id AS y_id, x.val AS x_val, y.val AS y_val ORDER BY x_id")
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert len(records) == 2
    assert records[0]["x_id"] == "a"
    assert records[0]["y_val"] == 2
    assert records[1]["x_id"] == "b"
    assert records[1]["y_val"] == 1


def test_multi_alias_return_duplicate_edges() -> None:
    """Duplicate edges produce one binding per edge."""
    g = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "label__X": [True, False], "label__Y": [False, True]}),
        pd.DataFrame({"s": ["a", "a"], "d": ["b", "b"], "type": ["R", "R"]}),
    )
    result = g.gfql("MATCH (x:X)-[:R]->(y:Y) RETURN x.id AS x_id, y.id AS y_id")
    assert len(result._nodes) == 2


def test_multi_alias_with_stage_scalar_projection_executes() -> None:
    """#1273 shape A: WITH multi-alias scalar projections execute on bindings-row path."""
    g = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "label__A": [True, False], "label__B": [False, True]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
    )
    result = g.gfql("MATCH (a:A)-[:R]->(b:B) WITH a.id AS a_id, b.id AS b_id RETURN a_id, b_id")
    assert result._nodes.to_dict(orient="records") == [{"a_id": "a", "b_id": "b"}]


def test_multi_alias_with_stage_scalar_projection_with_where_executes() -> None:
    """#1273 shape A: scalar aliases from WITH can drive same-stage WHERE filtering."""
    g = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b1", "b2"],
                "label__A": [True, False, False],
                "label__B": [False, True, True],
            }
        ),
        pd.DataFrame({"s": ["a", "a"], "d": ["b1", "b2"], "type": ["R", "R"]}),
    )
    result = g.gfql(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH a.id AS a_id, b.id AS b_id "
        "WHERE b_id = 'b2' "
        "RETURN a_id, b_id"
    )
    assert result._nodes.to_dict(orient="records") == [{"a_id": "a", "b_id": "b2"}]


def test_multi_alias_with_stage_whole_row_projection_executes_for_joined_row_projection_1393() -> None:
    """#1393: multi-alias whole-row WITH projection executes on bindings-row path."""
    g = _mk_graph(
        pd.DataFrame(
            {
                "id": ["n1", "n2", "x1", "x2"],
                "animal": ["cat", "dog", "cat", "wolf"],
            }
        ),
        pd.DataFrame({"s": ["n1", "n2"], "d": ["x1", "x2"], "type": ["R", "R"]}),
    )
    result = g.gfql("MATCH (n)-[rel]->(x) WITH n, x WHERE n.animal = x.animal RETURN n, x")
    records = entity_text_records(result, {"n": "nodes", "x": "nodes"})
    assert len(records) == 1
    assert "cat" in records[0]["n"]
    assert "cat" in records[0]["x"]


def test_string_cypher_executes_connected_multi_pattern_multi_whole_row_joined_projection_1393() -> None:
    result = _mk_connected_multi_pattern_reentry_graph().gfql(
        "MATCH (b:B)-[:S]->(c:C), (c)-[:T]->(d:D) RETURN b, c, d.id AS did ORDER BY did"
    )

    records = entity_text_records(result, {"b": "nodes", "c": "nodes"})
    assert records == [
        {"b": "(:B)", "c": "(:C)", "did": "d1"},
        {"b": "(:B)", "c": "(:C)", "did": "d2"},
    ]


def test_multi_alias_connected_whole_row_return_with_cross_alias_where_executes_for_joined_row_projection_1393() -> None:
    g = _mk_graph(
        pd.DataFrame(
            {
                "id": ["n1", "n2", "x1", "x2"],
                "animal": ["cat", "dog", "cat", "wolf"],
            }
        ),
        pd.DataFrame({"s": ["n1", "n2"], "d": ["x1", "x2"], "type": ["R", "R"]}),
    )
    result = g.gfql("MATCH (n)-[rel]->(x) WHERE n.animal = x.animal RETURN n, x")
    assert entity_text_records(result, {"n": "nodes", "x": "nodes"}) == [{"n": "({animal: 'cat'})", "x": "({animal: 'cat'})"}]


def test_multi_alias_connected_cross_alias_where_scalar_projection_remains_supported() -> None:
    g = _mk_graph(
        pd.DataFrame(
            {
                "id": ["n1", "n2", "x1", "x2"],
                "animal": ["cat", "dog", "cat", "wolf"],
            }
        ),
        pd.DataFrame({"s": ["n1", "n2"], "d": ["x1", "x2"], "type": ["R", "R"]}),
    )
    result = g.gfql("MATCH (n)-[rel]->(x) WHERE n.animal = x.animal RETURN n.id AS n_id, x.id AS x_id")
    assert result._nodes.to_dict(orient="records") == [{"n_id": "n1", "x_id": "x1"}]


def test_multi_alias_connected_cross_alias_where_single_whole_row_projection_remains_supported() -> None:
    g = _mk_graph(
        pd.DataFrame(
            {
                "id": ["n1", "n2", "x1", "x2"],
                "animal": ["cat", "dog", "cat", "wolf"],
            }
        ),
        pd.DataFrame({"s": ["n1", "n2"], "d": ["x1", "x2"], "type": ["R", "R"]}),
    )
    result = g.gfql("MATCH (n)-[rel]->(x) WHERE n.animal = x.animal RETURN n, x.id AS x_id")
    assert entity_text_records(result, {"n": "nodes"}) == [{"n": "({animal: 'cat'})", "x_id": "x1"}]


def test_compile_cypher_tracks_seeded_top_level_row_query() -> None:
    compiled = _compile_query("UNWIND [1, 2, 3] AS x RETURN x ORDER BY x DESC LIMIT 2")

    assert compiled.seed_rows is True
    first = cast(ASTCall, compiled.chain.chain[0])
    second = cast(ASTCall, compiled.chain.chain[1])
    assert isinstance(first, ASTCall)
    assert isinstance(second, ASTCall)
    assert first.function == "rows"
    assert second.function == "unwind"
    assert second.params == {"expr": "[1, 2, 3]", "as_": "x"}


def _assert_lowered_calls(query: str, expected_calls: List[Tuple[int, str, Dict[str, Any]]]) -> None:
    chain = lower_cypher_query(_parse_query(query))
    for idx, function, expected_params in expected_calls:
        call = cast(ASTCall, chain.chain[idx])
        assert call.function == function
        for key, value in expected_params.items():
            assert call.params[key] == value


@pytest.mark.parametrize(
    ("query", "expected_calls"),
    [
        (
            "MATCH (n) RETURN n.division AS division, count(*) AS cnt, max(n.age) AS max_age "
            "ORDER BY division ASC, cnt DESC",
            [
                (1, "rows", {}),
                (2, "with_", {"items": [("division", "n.division"), ("__cypher_agg__", "n.age")]}),
                (
                    3,
                    "group_by",
                    {"keys": ["division"], "aggregations": [("cnt", "count"), ("max_age", "max", "__cypher_agg__")]},
                ),
                (4, "order_by", {"keys": [("division", "asc"), ("cnt", "desc")]}),
            ],
        ),
        (
            "UNWIND [null, 1, null, 2, 1] AS x RETURN count(DISTINCT x) AS cnt, collect(DISTINCT x) AS vals",
            [
                (2, "with_", {"items": [("__cypher_group__", 1), ("__cypher_agg__", "x"), ("__cypher_agg__1", "x")]}),
                (
                    3,
                    "group_by",
                    {
                        "keys": ["__cypher_group__"],
                        "aggregations": [
                            ("cnt", "count_distinct", "__cypher_agg__"),
                            ("vals", "collect_distinct", "__cypher_agg__1"),
                        ],
                    },
                ),
                (4, "select", {}),
            ],
        ),
        (
            "UNWIND [1, 2, 2] AS x WITH collect(DISTINCT x) AS xs RETURN size(xs) AS n",
            [
                (2, "with_", {"items": [("__cypher_group__", 1), ("__cypher_agg__", "x")]}),
                (
                    3,
                    "group_by",
                    {"keys": ["__cypher_group__"], "aggregations": [("xs", "collect_distinct", "__cypher_agg__")]},
                ),
                (4, "with_", {"items": [("xs", "xs")]}),
                (5, "select", {"items": [("n", "size(xs)")]}),
            ],
        ),
        (
            "MATCH (a) WITH a.name AS name ORDER BY a.name + 'C' ASC LIMIT 2 RETURN name",
            [
                (1, "rows", {}),
                (2, "with_", {"items": [("name", "a.name")]}),
                (3, "order_by", {"keys": [("(name + 'C')", "asc")]}),
                (4, "limit", {"value": 2}),
                (5, "select", {"items": [("name", "name")]}),
            ],
        ),
        (
            "MATCH (a) WITH a.name AS name, count(*) AS cnt ORDER BY a.name + 'C' DESC LIMIT 1 RETURN name, cnt",
            [
                (1, "rows", {}),
                (2, "with_", {"items": [("name", "a.name")]}),
                (3, "group_by", {"keys": ["name"], "aggregations": [("cnt", "count")]}),
                (4, "order_by", {"keys": [("(name + 'C')", "desc")]}),
                (5, "limit", {"value": 1}),
                (6, "select", {"items": [("name", "name"), ("cnt", "cnt")]}),
            ],
        ),
        (
            "MATCH (a)-[r]->(b) RETURN count(DISTINCT r)",
            [
                (3, "rows", {"table": "edges", "source": "r"}),
                (4, "with_", {"items": [("__cypher_group__", 1), ("__cypher_agg__", "__gfql_edge_index_0__")]}),
                (
                    5,
                    "group_by",
                    {
                        "keys": ["__cypher_group__"],
                        "aggregations": [("count(DISTINCT r)", "count_distinct", "__cypher_agg__")],
                    },
                ),
            ],
        ),
    ],
)
def test_lower_cypher_query_builds_row_pipeline_shapes(query, expected_calls) -> None:
    _assert_lowered_calls(query, expected_calls)


def test_gfql_executes_top_level_unwind_query() -> None:
    _assert_query_rows("UNWIND [3, 1, 2] AS x RETURN x ORDER BY x ASC LIMIT 2", [{"x": 1}, {"x": 2}])


def test_gfql_executes_match_then_unwind_query() -> None:
    _assert_query_rows(
        "MATCH (n) UNWIND n.vals AS v RETURN v ORDER BY v ASC",
        [{"v": 1}, {"v": 2}, {"v": 3}],
        nodes_df=pd.DataFrame({"id": ["a", "b"], "vals": [[2, 1], [3]]}),
    )


def test_gfql_executes_aggregate_return_query() -> None:
    _assert_query_rows(
        "MATCH (n) RETURN n.division AS division, count(*) AS cnt, max(n.age) AS max_age ORDER BY division ASC",
        [{"division": "x", "cnt": 2, "max_age": 7}, {"division": "y", "cnt": 1, "max_age": 4}],
        nodes_df=pd.DataFrame({"id": ["a", "b", "c"], "division": ["x", "x", "y"], "age": [3, 7, 4]}),
    )


def test_gfql_executes_aggregate_order_by_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["anon_1", "anon_2", "anon_3", "anon_4"],
                "labels": [[], [], [], []],
                "division": ["A", "B", "B", "C"],
                "age": [22, 33, 44, 55],
            }
        )
    )
    edges = cudf.from_pandas(pd.DataFrame({"s": [], "d": []}))

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n)\nRETURN n.division, max(n.age)\nORDER BY max(n.age)",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"n.division": "A", "max(n.age)": 22},
        {"n.division": "B", "max(n.age)": 44},
        {"n.division": "C", "max(n.age)": 55},
    ]


def test_gfql_preserves_group_order_for_aggregate_order_ties_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "labels": [["L1"], ["L2"], ["L3"]],
                "label__L1": [True, False, False],
                "label__L2": [False, True, False],
                "label__L3": [False, False, True],
            }
        )
    )
    edges = cudf.from_pandas(pd.DataFrame({"s": [], "d": []}))

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a)\nRETURN a, count(*)\nORDER BY count(*)",
        engine="cudf",
    )

    # Aggregate/grouping projection renders entity text via a separate path.
    assert result._nodes.to_pandas().to_dict(orient="records") == [
        {"a": "(:L1)", "count(*)": 1},
        {"a": "(:L2)", "count(*)": 1},
        {"a": "(:L3)", "count(*)": 1},
    ]


@pytest.mark.parametrize(
    "values",
    [
        "[true, false]",
        "[351, -3974856, 93, -3, 123, 0, 3, -2, 20934587, 1, 20934585, 20934586, -10]",
        "[[2, 2], [2, -2], [1, 2], [], [1], [300, 0], [1, -20], [2, -2, 100]]",
    ],
)
def test_gfql_executes_list_comprehension_order_check_on_cudf(values: str) -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(pd.DataFrame({"id": []}))
    edges = cudf.from_pandas(pd.DataFrame({"s": [], "d": []}))

    result = _mk_graph(nodes, edges).gfql(
        f"WITH {values} AS values\n"
        "WITH values, size(values) AS numOfValues\n"
        "UNWIND values AS value\n"
        "WITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n"
        "  ORDER BY value\n"
        "WITH numOfValues, collect(x) AS orderedX\n"
        "RETURN orderedX = range(0, numOfValues-1) AS equal",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"equal": True}]


def test_gfql_executes_loop_edge_count_queries() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame(
        {
            "s": ["a", "a"],
            "d": ["a", "b"],
            "type": ["LOOP", "LINK"],
        }
    )
    g = _mk_graph(nodes, edges)

    count_result = g.gfql("MATCH (n)-[r]-(n) RETURN count(r) AS cnt")
    distinct_result = g.gfql("MATCH (n)-[r]-(n) RETURN count(DISTINCT r) AS cnt")

    assert count_result._nodes.to_dict(orient="records") == [{"cnt": 1}]
    assert distinct_result._nodes.to_dict(orient="records") == [{"cnt": 1}]


def test_gfql_executes_distinct_edge_count_with_bound_edge_ids() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame(
        {
            "edge_id": ["e1", "e2"],
            "s": ["a", "a"],
            "d": ["a", "b"],
            "type": ["LOOP", "LINK"],
        }
    )
    g = cast(_CypherTestGraph, _CypherTestGraph().nodes(nodes, "id").edges(edges, "s", "d", edge="edge_id"))

    result = g.gfql("MATCH (n)-[r]-(n) RETURN count(DISTINCT r) AS cnt")

    assert result._nodes.to_dict(orient="records") == [{"cnt": 1}]


def test_gfql_rejects_repeated_node_alias_row_projection() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "m", "z"],
            "name": ["a", "mid", "other"],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "m", "z"],
            "d": ["m", "a", "m"],
            "type": ["A", "B", "B"],
        }
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_graph(nodes, edges).gfql("MATCH (a)-[:A]->()-[:B]->(a) RETURN a.name")

    assert exc_info.value.code == ErrorCode.E108


def test_gfql_rejects_connected_comma_cycle_row_projection_on_repeated_alias() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "name": ["a", "b", "c"],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "b", "b"],
            "d": ["b", "a", "c"],
            "type": ["A", "B", "B"],
        }
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_graph(nodes, edges).gfql("MATCH (a)-[:A]->(b), (b)-[:B]->(a) RETURN a.name")

    assert exc_info.value.code == ErrorCode.E108


def test_gfql_executes_distinct_aggregate_return_query() -> None:
    _assert_query_rows(
        "UNWIND [null, 1, null, 2, 1] AS x RETURN count(DISTINCT x) AS cnt, collect(DISTINCT x) AS vals",
        [{"cnt": 2, "vals": [1, 2]}],
    )


def test_gfql_executes_string_min_max_aggregate_return_query_with_nulls() -> None:
    _assert_query_rows(
        "UNWIND ['a', 'b', 'B', null, 'abc', 'abc1'] AS i "
        "RETURN max(i) AS max_i, min(i) AS min_i",
        [{"max_i": "b", "min_i": "B"}],
    )


def test_gfql_executes_collect_distinct_all_null_return_query() -> None:
    _assert_query_rows("UNWIND [null, null] AS x RETURN collect(DISTINCT x) AS c", [{"c": []}])


def test_gfql_executes_count_distinct_missing_property_as_zero() -> None:
    _assert_query_rows(
        "MATCH (a) RETURN count(DISTINCT a.name) AS cnt",
        [{"cnt": 0}],
        nodes_df=pd.DataFrame({"id": ["a", "b"]}),
    )


def test_cypher_to_gfql_rejects_multi_source_aggregate_expr() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        cypher_to_gfql("MATCH (a)-[r]->(b) RETURN a.id AS a_id, max(b.score) AS max_b")

    assert exc_info.value.code == ErrorCode.E108
    assert exc_info.value.context["field"] == "return"
    assert exc_info.value.context["value"] == "b.score"
    assert "value: 'b.score'" in str(exc_info.value)


def test_gfql_executes_top_level_quantifier_expression() -> None:
    _assert_query_rows("RETURN none(x IN [true, false] WHERE x) AS result", [{"result": False}])


def test_gfql_executes_top_level_quantifier_composed_expression() -> None:
    _assert_query_rows(
        "RETURN none(x IN [1, 2, 3] WHERE x = 2) = (NOT any(x IN [1, 2, 3] WHERE x = 2)) AS result",
        [{"result": True}],
    )


def test_gfql_executes_top_level_single_quantifier_null_semantics() -> None:
    _assert_query_rows(
        "RETURN "
        "single(x IN [2, null] WHERE x = 2) AS left_result, "
        "single(x IN [null, 2] WHERE x = 2) AS right_result",
        [{"left_result": None, "right_result": None}],
    )


def test_gfql_executes_top_level_membership_and_null_expression() -> None:
    _assert_query_rows("RETURN 3 IN [1, 2, 3] AS hit, null IS NULL AS empty", [{"hit": True, "empty": True}])


def test_gfql_executes_top_level_list_equality_with_nested_null_returns_null() -> None:
    _assert_query_rows(
        "RETURN [[1], [2]] = [[1], [null]] AS result",
        [{"result": None}],
    )


def test_gfql_executes_top_level_map_equality_with_null_values_returns_null() -> None:
    _assert_query_rows(
        "RETURN {k: null} = {k: null} AS both_null, {k: 1} = {k: null} AS mixed",
        [{"both_null": None, "mixed": None}],
    )


def test_gfql_executes_top_level_membership_nested_null_propagates_unknown() -> None:
    _assert_query_rows(
        "RETURN "
        "[null] IN [[null]] AS list_null, "
        "[1, 2] IN [[null, 2]] AS needs_null_compare, "
        "[1, 2] IN [[null, 2], [1, 3]] AS all_unknown_or_false",
        [{"list_null": None, "needs_null_compare": None, "all_unknown_or_false": None}],
    )


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_gfql_executes_top_level_list_map_nan_comparisons_on_engines(engine: str | None) -> None:
    if engine == "cudf":
        _require_cudf_runtime()

    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "WITH (0.0 / 0.0) AS z "
        "RETURN "
        "[z] = [z] AS list_eq, "
        "[z] <> [z] AS list_neq, "
        "[z] IN [[z]] AS list_in, "
        "{k: z} = {k: z} AS map_eq, "
        "{k: z} <> {k: z} AS map_neq",
        **({"engine": engine} if engine is not None else {}),
    )

    rows = _to_pandas_df(result._nodes).to_dict(orient="records") if engine == "cudf" else result._nodes.to_dict(orient="records")
    if engine == "cudf":
        # cuDF currently canonicalizes arithmetic NaN to null in this path.
        assert rows == [
            {
                "list_eq": None,
                "list_neq": None,
                "list_in": None,
                "map_eq": None,
                "map_neq": None,
            }
        ]
    else:
        assert rows == [
            {
                "list_eq": False,
                "list_neq": True,
                "list_in": False,
                "map_eq": False,
                "map_neq": True,
            }
        ]


def test_gfql_executes_size_null_and_sqrt_constant_expressions() -> None:
    _assert_query_rows(
        "WITH null AS l RETURN size(l) AS size_l, size(null) AS size_null, sqrt(12.96) AS root",
        [{"size_l": None, "size_null": None, "root": 3.6}],
    )


def test_gfql_executes_double_quoted_string_literal_projection() -> None:
    _assert_query_rows('RETURN "a" AS literal', [{"literal": "a"}])


def test_gfql_executes_bool_and_null_list_literal_projections() -> None:
    _assert_query_rows(
        "RETURN [true] AS trues, [false] AS falses, [null] AS nulls",
        [{"trues": [True], "falses": [False], "nulls": [None]}],
    )


def test_gfql_preserves_single_alias_list_projection() -> None:
    _assert_query_rows(
        "MATCH (n) WITH [n] AS ns RETURN size(ns) AS size",
        [{"size": 1}, {"size": 1}],
        nodes_df=pd.DataFrame({"id": ["a", "b"]}),
    )


def test_gfql_rejects_unresolved_single_alias_list_projection() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _mk_empty_graph().gfql("RETURN [missingAlias] AS xs")

    assert "missingAlias" in str(exc_info.value)


def test_gfql_executes_substring_and_tointeger_expressions() -> None:
    _assert_query_rows(
        "WITH 82.9 AS weight "
        "RETURN substring('0123456789', 1) AS s, toInteger(weight) AS int_weight",
        [{"s": "123456789", "int_weight": 82}],
    )


def test_gfql_executes_top_level_xor_expression_and_precedence() -> None:
    _assert_query_rows(
        "RETURN "
        "true XOR false AS tf, "
        "false XOR false AS ff, "
        "true XOR null AS tn, "
        "true OR true XOR true AS or_xor, "
        "true XOR false AND false AS xor_and",
        [
        {
            "tf": True,
            "ff": False,
            "tn": None,
            "or_xor": True,
            "xor_and": True,
        }
        ],
    )


def test_gfql_executes_top_level_xor_literal_expression_on_cudf() -> None:
    cudf = _require_cudf_runtime()
    graph = _mk_graph(
        cudf.from_pandas(pd.DataFrame({"id": pd.Series(dtype="object")})),
        cudf.from_pandas(pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")})),
    )

    result = graph.gfql(
        "RETURN true XOR false AS tf, true XOR null AS tn",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"tf": True, "tn": None}
    ]


def test_gfql_executes_with_where_xor_null_pipeline() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "UNWIND [true, false, null] AS a "
        "UNWIND [true, false, null] AS b "
        "WITH a, b WHERE a IS NULL OR b IS NULL "
        "RETURN a, b, (a XOR b) IS NULL = (b XOR a) IS NULL AS result "
        "ORDER BY a, b"
    )

    actual_rows = result._nodes.to_dict(orient="records")
    expected_rows = [
        {"a": False, "b": None, "result": True},
        {"a": None, "b": False, "result": True},
        {"a": None, "b": None, "result": True},
        {"a": None, "b": True, "result": True},
        {"a": True, "b": None, "result": True},
    ]

    def _row_key(row):
        return (str(row["a"]), str(row["b"]), str(row["result"]))

    assert sorted(actual_rows, key=_row_key) == sorted(expected_rows, key=_row_key)


def test_gfql_handles_empty_match_label_filter_with_limit_zero() -> None:
    g = _mk_graph(
        pd.DataFrame({"id": [], "labels": []}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = g.gfql(
        "MATCH (p:Person) "
        "RETURN p.name AS name "
        "ORDER BY p.name "
        "LIMIT 0"
    )

    assert result._nodes.to_dict(orient="records") == []


def test_gfql_executes_optional_match_null_projections_on_non_empty_graph() -> None:
    g = _mk_graph(
        pd.DataFrame({"id": ["n1"], "exists": [42]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = g.gfql(
        "OPTIONAL MATCH (n) "
        "RETURN n.missing IS NULL AS missing_is_null, "
        "n.exists IS NOT NULL AS exists_is_not_null"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"missing_is_null": True, "exists_is_not_null": True}
    ]


def test_gfql_executes_optional_match_null_projections_on_empty_graph() -> None:
    g = _mk_graph(
        pd.DataFrame({"id": [], "exists": []}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = g.gfql(
        "OPTIONAL MATCH (n) "
        "RETURN n.missing IS NULL AS missing_is_null, "
        "n.missing IS NOT NULL AS missing_is_not_null"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"missing_is_null": True, "missing_is_not_null": False}
    ]


def test_gfql_executes_with_where_or_short_circuit_over_mixed_type_compare() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["root", "child1", "child2"],
            "label__Root": [True, False, False],
            "label__TextNode": [False, True, False],
            "label__IntNode": [False, False, True],
            "name": ["x", None, None],
            "var": [None, "text", 0],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["root", "root"],
            "d": ["child1", "child2"],
            "type": ["T", "T"],
        }
    )
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (:Root {name: 'x'})-->(i) "
        "WITH i "
        "WHERE i.var > 'te' OR i.var IS NOT NULL "
        "RETURN i "
        "ORDER BY i.id"
    )

    assert entity_text_records(result, {"i": "nodes"}) == [
        {"i": "(:TextNode {var: 'text'})"},
        {"i": "(:IntNode {var: 0})"},
    ]


def test_gfql_executes_with_where_null_filter_over_mixed_type_compare_on_pandas() -> None:
    nodes = pd.DataFrame(
        {
            "id": [
                "root",
                "child_text",
                "child_equal",
                "child_less",
                "child_int",
                "child_float",
                "child_true",
                "child_false",
                "child_none",
                "child_pdna",
                "child_nan",
                "child_nat",
                "child_zz",
            ],
            "label__Root": [True] + [False] * 12,
            "label__Node": [False] + [True] * 12,
            "name": ["x"] + [None] * 12,
            "var": [
                None,
                "text",
                "te",
                "aa",
                0,
                1.5,
                True,
                False,
                None,
                pd.NA,
                float("nan"),
                pd.NaT,
                "zz",
            ],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["root"] * 12,
            "d": [
                "child_text",
                "child_equal",
                "child_less",
                "child_int",
                "child_float",
                "child_true",
                "child_false",
                "child_none",
                "child_pdna",
                "child_nan",
                "child_nat",
                "child_zz",
            ],
            "type": ["T"] * 12,
        }
    )
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (:Root {name: 'x'})-->(i)\n"
        "WITH i\n"
        "WHERE i.var > 'te'\n"
        "RETURN i.id AS id\n"
        "ORDER BY id"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"id": "child_text"},
        {"id": "child_zz"},
    ]


def test_gfql_executes_with_where_null_filter_over_mixed_type_compare_on_cudf() -> None:
    _require_cudf_runtime()

    nodes = pd.DataFrame(
        {
            "id": [
                "root",
                "child_text",
                "child_equal",
                "child_less",
                "child_int",
                "child_float",
                "child_true",
                "child_false",
                "child_none",
                "child_pdna",
                "child_nan",
                "child_nat",
                "child_zz",
            ],
            "label__Root": [True] + [False] * 12,
            "label__Node": [False] + [True] * 12,
            "name": ["x"] + [None] * 12,
            "var": [
                None,
                "text",
                "te",
                "aa",
                0,
                1.5,
                True,
                False,
                None,
                pd.NA,
                float("nan"),
                pd.NaT,
                "zz",
            ],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["root"] * 12,
            "d": [
                "child_text",
                "child_equal",
                "child_less",
                "child_int",
                "child_float",
                "child_true",
                "child_false",
                "child_none",
                "child_pdna",
                "child_nan",
                "child_nat",
                "child_zz",
            ],
            "type": ["T"] * 12,
        }
    )
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (:Root {name: 'x'})-->(i)\n"
        "WITH i\n"
        "WHERE i.var > 'te'\n"
        "RETURN i.id AS id\n"
        "ORDER BY id",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"id": "child_text"},
        {"id": "child_zz"},
    ]


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_gfql_executes_with_where_is_null_over_mixed_null_sentinels_on_engines(
    engine: str | None,
) -> None:
    if engine == "cudf":
        _require_cudf_runtime()

    nodes = pd.DataFrame(
        {
            "id": [
                "root",
                "child_text",
                "child_int",
                "child_float",
                "child_none",
                "child_pdna",
                "child_nan",
                "child_nat",
            ],
            "label__Root": [True] + [False] * 7,
            "label__Node": [False] + [True] * 7,
            "name": ["x"] + [None] * 7,
            "var": [None, "text", 1, 1.5, None, pd.NA, float("nan"), pd.NaT],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["root"] * 7,
            "d": ["child_text", "child_int", "child_float", "child_none", "child_pdna", "child_nan", "child_nat"],
            "type": ["T"] * 7,
        }
    )
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (:Root {name: 'x'})-->(i)\n"
        "WITH i\n"
        "WHERE i.var IS NULL\n"
        "RETURN i.id AS id\n"
        "ORDER BY id",
        **({"engine": engine} if engine is not None else {}),
    )
    rows = _to_pandas_df(result._nodes).to_dict(orient="records") if engine == "cudf" else result._nodes.to_dict(orient="records")
    assert rows == [
        {"id": "child_nan"},
        {"id": "child_nat"},
        {"id": "child_none"},
        {"id": "child_pdna"},
    ]


@pytest.mark.parametrize(
    "operator,expected_ids",
    (
        (">", ["child1"]),
        (">=", ["child1"]),
        ("<", ["child3"]),
        ("<=", ["child3"]),
    ),
)
def test_gfql_executes_with_where_mixed_type_compare_operator_matrix_on_pandas(
    operator: str,
    expected_ids: list[str],
) -> None:
    nodes = pd.DataFrame(
        {
            "id": ["root", "child1", "child2", "child3", "child4"],
            "label__Root": [True, False, False, False, False],
            "label__TextNode": [False, True, False, True, True],
            "label__IntNode": [False, False, True, False, False],
            "name": ["x", None, None, None, None],
            "var": [None, "text", 0, "aa", None],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["root", "root", "root", "root"],
            "d": ["child1", "child2", "child3", "child4"],
            "type": ["T", "T", "T", "T"],
        }
    )
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (:Root {name: 'x'})-->(i)\n"
        "WITH i\n"
        f"WHERE i.var {operator} 'te'\n"
        "RETURN i.id AS id\n"
        "ORDER BY id"
    )
    assert result._nodes.to_dict(orient="records") == [{"id": node_id} for node_id in expected_ids]


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
@pytest.mark.parametrize(
    "where_expr,expected_ids",
    (
        ("i.n > '1'", []),
        ("i.n <= '1'", []),
        ("i.s > 1", []),
        ("i.s <= 1", []),
        ("i.n > 1", ["child1"]),
        ("i.s > 'te'", ["child1"]),
    ),
)
def test_gfql_executes_with_where_cross_type_comparison_conformance_on_engines(
    engine: str | None,
    where_expr: str,
    expected_ids: list[str],
) -> None:
    if engine == "cudf":
        _require_cudf_runtime()

    nodes = pd.DataFrame(
        {
            "id": ["root", "child1", "child2", "child3"],
            "label__Root": [True, False, False, False],
            "label__Node": [False, True, True, True],
            "name": ["x", None, None, None],
            "n": [None, 2, 0, None],
            "s": [None, "text", "aa", None],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["root", "root", "root"],
            "d": ["child1", "child2", "child3"],
            "type": ["T", "T", "T"],
        }
    )
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (:Root {name: 'x'})-->(i)\n"
        "WITH i\n"
        f"WHERE {where_expr}\n"
        "RETURN i.id AS id\n"
        "ORDER BY id",
        **({"engine": engine} if engine is not None else {}),
    )

    rows = _to_pandas_df(result._nodes).to_dict(orient="records") if engine == "cudf" else result._nodes.to_dict(orient="records")
    assert rows == [{"id": node_id} for node_id in expected_ids]


def test_gfql_executes_optional_match_count_distinct_on_empty_graph() -> None:
    g = _mk_graph(
        pd.DataFrame({"id": []}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = g.gfql("OPTIONAL MATCH (a) RETURN count(DISTINCT a)")

    assert result._nodes.to_dict(orient="records") == [
        {"count(DISTINCT a)": 0}
    ]


def test_gfql_executes_with_aggregate_precedence_pipeline() -> None:
    _assert_query_rows(
        "UNWIND [true, false, null] AS a "
        "UNWIND [true, false, null] AS b "
        "UNWIND [true, false, null] AS c "
        "WITH collect((a OR b XOR c) = (a OR (b XOR c))) AS eq, "
        "collect((a OR b XOR c) <> ((a OR b) XOR c)) AS neq "
        "RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result",
        [{"result": True}],
    )


def test_gfql_executes_top_level_list_comprehension_expression() -> None:
    _assert_query_rows("RETURN [x IN [1, 2, 3] WHERE x > 1 | x + 10] AS vals", [{"vals": [12, 13]}])


def test_string_cypher_supports_empty_map_quantifier_predicates() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": [], "type": []}))

    result = g.gfql(
        "RETURN none(x IN [] WHERE x.a = 2) AS none_result, "
        "any(x IN [] WHERE x.a = 2) AS any_result, "
        "single(x IN [] WHERE x.a = 2) AS single_result"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"none_result": True, "any_result": False, "single_result": False}
    ]


def test_string_cypher_supports_map_quantifier_predicates_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    g = _mk_graph(
        cudf.from_pandas(pd.DataFrame({"id": pd.Series(dtype="object")})),
        cudf.from_pandas(pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object"), "type": pd.Series(dtype="object")})),
    )

    result = g.gfql(
        "RETURN none(x IN [{a: 2, b: 5}] WHERE x.a = 2) AS result",
        engine="cudf",
    )

    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {"result": False}
    ]


@pytest.mark.parametrize(
    "query",
    [
        "RETURN 123 AND true AS out",
        "RETURN 123.4 OR false AS out",
        "RETURN 'foo' XOR true AS out",
        "RETURN NOT [] AS out",
    ],
)
def test_string_cypher_rejects_obviously_non_boolean_operands_in_boolean_ops(query: str) -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": [], "type": []}))

    with pytest.raises(GFQLValidationError, match="requires boolean or null operands"):
        g.gfql(query)


# ── Multi-alias WITH projection from connected MATCH (#880 / IC-4 shape) ──


def _mk_ic3_cross_country_shape_graph() -> _CypherTestGraph:
    """Graph for IC-3-style carried row + collected city list reentry tests."""
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "cityA", "friend1", "friend2", "friend3", "cityB", "cityC"],
                "label__Person": [True, False, True, True, True, False, False],
                "label__City": [False, True, False, False, False, True, True],
                "name": ["", "CityA", "", "", "", "CityB", "CityC"],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p1", "p1", "p1", "friend1", "friend2", "friend3"],
                "d": ["cityA", "friend1", "friend2", "friend3", "cityB", "cityA", "cityC"],
                "type": [
                    "IS_LOCATED_IN",
                    "KNOWS",
                    "KNOWS",
                    "KNOWS",
                    "IS_LOCATED_IN",
                    "IS_LOCATED_IN",
                    "IS_LOCATED_IN",
                ],
            }
        ),
    )


def _mk_ic4_shape_graph() -> _CypherTestGraph:
    """Graph for IC-4 multi-alias WITH tests: person-KNOWS-friend, post-HAS_CREATOR->friend, post-HAS_TAG->tag."""
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "f1", "f2", "post1", "post2", "post3", "tag1", "tag2"],
                "label__Person": [True, True, True, False, False, False, False, False],
                "label__Post": [False, False, False, True, True, True, False, False],
                "label__Tag": [False, False, False, False, False, False, True, True],
                "name": ["", "", "", "", "", "", "TagA", "TagB"],
                "creationDate": [0, 0, 0, 100, 200, 300, 0, 0],
            }
        ),
        pd.DataFrame(
            {
                # KNOWS: p1↔f1, p1↔f2 (undirected)
                # HAS_CREATOR: post→friend (post is creator-of friend's content)
                # HAS_TAG: post→tag
                "s": ["p1", "p1", "post1", "post2", "post3", "post1", "post2", "post3"],
                "d": ["f1", "f2", "f1", "f1", "f2", "tag1", "tag1", "tag2"],
                "type": [
                    "KNOWS", "KNOWS",
                    "HAS_CREATOR", "HAS_CREATOR", "HAS_CREATOR",
                    "HAS_TAG", "HAS_TAG", "HAS_TAG",
                ],
            }
        ),
    )


def _assert_ic4_shape_query_rows(
    query: str,
    expected_rows: list[dict[str, object]],
    *,
    params: dict[str, object] | None = None,
) -> None:
    graph = _mk_ic4_shape_graph()
    result = graph.gfql(query, params={"pid": "p1"} if params is None else params)
    assert result._nodes.to_dict(orient="records") == expected_rows


def _mk_ic4_id_collision_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """IC4-shaped graph with LDBC-style cross-label numeric id collisions (#1496)."""
    return (
        pd.DataFrame(
            {
                "id": [1, 2, 10, 20, 30, 10, 20],
                "label__Person": [True, True, False, False, False, False, False],
                "label__Post": [False, False, True, True, True, False, False],
                "label__Tag": [False, False, False, False, False, True, True],
                "name": ["", "", "", "", "", "TagA", "TagB"],
                "creationDate": [0, 0, 100, 200, 300, 0, 0],
            }
        ),
        pd.DataFrame(
            {
                "s": [1, 10, 20, 30, 10, 20, 30],
                "d": [2, 2, 2, 2, 10, 10, 20],
                "type": [
                    "KNOWS",
                    "HAS_CREATOR",
                    "HAS_CREATOR",
                    "HAS_CREATOR",
                    "HAS_TAG",
                    "HAS_TAG",
                    "HAS_TAG",
                ],
            }
        ),
    )


_ISSUE_1496_IC4_QUERY = """
MATCH (person:Person {id: $personId })-[:KNOWS]-(friend:Person),
      (friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag)
WITH DISTINCT tag, post
WITH tag,
     CASE
       WHEN $startDate <= post.creationDate < $endDate THEN 1
       ELSE 0
     END AS valid,
     CASE
       WHEN post.creationDate < $startDate THEN 1
       ELSE 0
     END AS inValid
WITH tag, sum(valid) AS postCount, sum(inValid) AS inValidPostCount
WHERE postCount>0 AND inValidPostCount=0
RETURN tag.name AS tagName, postCount
ORDER BY postCount DESC, tagName ASC
LIMIT 10
"""


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_issue_1496_ic4_unlabeled_has_tag_destination_disambiguates_colliding_ids(engine: Optional[str]) -> None:
    """Official IC4 shape: HAS_TAG should bind the tag row when ids collide across labels."""
    nodes, edges = _mk_ic4_id_collision_data()
    if engine == "cudf":
        _require_cudf_runtime()
        graph = _mk_cudf_graph(nodes, edges)
    else:
        graph = _mk_graph(nodes, edges)
    params = {"personId": 1, "startDate": 150, "endDate": 350}
    result = (
        graph.gfql(_ISSUE_1496_IC4_QUERY, params=params, engine=engine)
        if engine is not None
        else graph.gfql(_ISSUE_1496_IC4_QUERY, params=params)
    )
    records = _to_pandas_df(result._nodes).to_dict(orient="records")
    assert records == [{"tagName": "TagB", "postCount": 1}]


@pytest.mark.parametrize(
    "query,expected_rows",
    [
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "RETURN tag.name AS tagName, post.id AS postId "
            "ORDER BY tagName, postId",
            [
                {"tagName": "TagA", "postId": "post1"},
                {"tagName": "TagA", "postId": "post2"},
                {"tagName": "TagB", "postId": "post3"},
            ],
            id="distinct_scalar_projection",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            'WHERE tag.name = "TagA" '
            "RETURN post.id AS postId ORDER BY postId",
            [{"postId": "post1"}, {"postId": "post2"}],
            id="distinct_property_where",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "RETURN tag.name AS tagName, count(post) AS postCount "
            "ORDER BY postCount DESC, tagName ASC",
            [{"tagName": "TagA", "postCount": 2}, {"tagName": "TagB", "postCount": 1}],
            id="distinct_count_projection",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "RETURN tag.name AS tagName, sum(post.creationDate) AS totalDate "
            "ORDER BY tagName",
            [{"tagName": "TagA", "totalDate": 300}, {"tagName": "TagB", "totalDate": 300}],
            id="distinct_sum_aggregation",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "RETURN tag.name AS tagName, count(*) AS cnt "
            "ORDER BY cnt DESC, tagName ASC",
            [{"tagName": "TagA", "cnt": 2}, {"tagName": "TagB", "cnt": 1}],
            id="distinct_count_star",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd "
            "RETURN tag.name AS tagName, sum(cd) AS totalDate "
            "ORDER BY tagName",
            [{"tagName": "TagA", "totalDate": 300}, {"tagName": "TagB", "totalDate": 300}],
            id="three_stage_chain",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd, post.id AS pid "
            "RETURN tag.name AS tn, cd, pid ORDER BY tn, pid",
            [
                {"tn": "TagA", "cd": 100, "pid": "post1"},
                {"tn": "TagA", "cd": 200, "pid": "post2"},
                {"tn": "TagB", "cd": 300, "pid": "post3"},
            ],
            id="two_scalars_extend",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd "
            "WITH tag, cd WHERE cd > 150 "
            "RETURN tag.name AS tn, cd ORDER BY tn, cd",
            [{"tn": "TagA", "cd": 200}, {"tn": "TagB", "cd": 300}],
            id="extend_scalar_where",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd "
            "WITH tag, cd AS creationDate "
            "RETURN tag.name AS tn, creationDate ORDER BY tn, creationDate",
            [
                {"tn": "TagA", "creationDate": 100},
                {"tn": "TagA", "creationDate": 200},
                {"tn": "TagB", "creationDate": 300},
            ],
            id="extend_scalar_renames",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd "
            "RETURN tag.name AS tn, "
            "CASE WHEN cd > 150 THEN 'recent' ELSE 'old' END AS era "
            "ORDER BY tn, era",
            [
                {"tn": "TagA", "era": "old"},
                {"tn": "TagA", "era": "recent"},
                {"tn": "TagB", "era": "recent"},
            ],
            id="extend_scalar_in_case",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd "
            "WITH tag, sum(cd) AS total "
            "RETURN tag.name AS tn, total ORDER BY tn",
            [{"tn": "TagA", "total": 300}, {"tn": "TagB", "total": 300}],
            id="four_stage_chain",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH post, count(tag) AS tagCount "
            "RETURN post.id AS postId, post.creationDate AS cd, tagCount ORDER BY postId",
            [
                {"postId": "post1", "cd": 100, "tagCount": 1},
                {"postId": "post2", "cd": 200, "tagCount": 1},
                {"postId": "post3", "cd": 300, "tagCount": 1},
            ],
            id="non_active_whole_row_grouping",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH post, count(tag) AS tagCount "
            "WITH post.id AS postId, post.creationDate AS cd, tagCount "
            "RETURN postId, cd, tagCount ORDER BY postId",
            [
                {"postId": "post1", "cd": 100.0, "tagCount": 1},
                {"postId": "post2", "cd": 200.0, "tagCount": 1},
                {"postId": "post3", "cd": 300.0, "tagCount": 1},
            ],
            id="non_active_grouping_then_property_projection",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH post, tag, count(*) AS cnt "
            "RETURN post.id AS postId, tag.name AS tagName, cnt "
            "ORDER BY postId, tagName",
            [
                {"postId": "post1", "tagName": "TagA", "cnt": 1},
                {"postId": "post2", "tagName": "TagA", "cnt": 1},
                {"postId": "post3", "tagName": "TagB", "cnt": 1},
            ],
            id="non_active_multi_whole_row_grouping",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd "
            "WITH tag, sum(cd) AS total, count(*) AS cnt "
            "RETURN tag.name AS tn, total, cnt ORDER BY tn",
            [{"tn": "TagA", "total": 300, "cnt": 2}, {"tn": "TagB", "total": 300, "cnt": 1}],
            id="non_final_agg_multiple_funcs",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd "
            "WITH tag, sum(cd) AS total "
            "WITH tag.name AS tn, total "
            "RETURN tn, total ORDER BY tn",
            [{"tn": "TagA", "total": 300}, {"tn": "TagB", "total": 300}],
            id="non_final_agg_then_scalar_stage",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd "
            "WITH tag, sum(cd) AS total "
            "RETURN tag.name AS tn, total ORDER BY total DESC, tn",
            [{"tn": "TagA", "total": 300}, {"tn": "TagB", "total": 300}],
            id="non_final_agg_order_by_alias_property",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd "
            "RETURN tag.name AS tn, min(cd) AS earliest ORDER BY tn",
            [{"tn": "TagA", "earliest": 100}, {"tn": "TagB", "earliest": 300}],
            id="extend_min_aggregation",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd "
            "RETURN tag.name AS tn, count(*) AS cnt ORDER BY tn",
            [{"tn": "TagA", "cnt": 2}, {"tn": "TagB", "cnt": 1}],
            id="extend_count_star",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "WITH tag, post.creationDate AS cd "
            "RETURN tag.name AS tn, sum(cd) AS total",
            [],
            id="extend_empty_result",
        ),
        pytest.param(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "RETURN tag.name AS tagName, "
            "sum(CASE WHEN post.creationDate > 150 THEN 1 ELSE 0 END) AS recentCount "
            "ORDER BY recentCount DESC, tagName ASC",
            [{"tagName": "TagA", "recentCount": 1}, {"tagName": "TagB", "recentCount": 1}],
            id="case_in_return_aggregation",
        ),
    ],
)
def test_string_cypher_ic4_multi_alias_rows(query: str, expected_rows: list[dict[str, object]]) -> None:
    """IC-4 multi-alias WITH DISTINCT / scalar / aggregation regression cases (#880, #1045, #1392, #1054)."""
    params = {"pid": "nonexistent"} if expected_rows == [] else {"pid": "p1"}
    _assert_ic4_shape_query_rows(query, expected_rows, params=params)


def test_string_cypher_multi_alias_with_distinct_simple_two_hop() -> None:
    """Simpler shape: single connected pattern, two aliases, WITH DISTINCT (#880)."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"], "label__A": [True, False, False], "label__B": [False, True, False], "label__C": [False, False, True], "val": [1, 2, 3]}),
        pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["R", "S"]}),
    )
    result = graph.gfql(
        "MATCH (a:A)-[:R]->(b:B)-[:S]->(c:C) "
        "WITH DISTINCT b, c "
        "RETURN b.val AS bv, c.val AS cv",
    )
    assert result._nodes.to_dict(orient="records") == [{"bv": 2, "cv": 3}]


def test_string_cypher_multi_alias_with_distinct_where_filter() -> None:
    """Multi-alias WITH DISTINCT + WHERE filter on projected alias (#880)."""
    graph = _mk_ic4_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
        "WITH DISTINCT tag, post "
        "WHERE post.creationDate > 100 "
        "RETURN tag.name AS tagName, post.id AS postId "
        "ORDER BY tagName, postId",
        params={"pid": "p1"},
    )
    records = result._nodes.to_dict(orient="records")
    # post1 has creationDate=100 (excluded by >100), post2=200, post3=300
    assert all(r["postId"] != "post1" for r in records)
    assert len(records) >= 1


def test_string_cypher_multi_alias_with_distinct_order_by_limit() -> None:
    """Multi-alias WITH DISTINCT + ORDER BY + LIMIT (#880)."""
    graph = _mk_ic4_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
        "WITH DISTINCT tag, post "
        "RETURN tag.name AS tagName, post.id AS postId "
        "ORDER BY postId "
        "LIMIT 2",
        params={"pid": "p1"},
    )
    records = result._nodes.to_dict(orient="records")
    assert len(records) == 2
    assert records[0]["postId"] == "post1"
    assert records[1]["postId"] == "post2"


def test_string_cypher_multi_alias_with_distinct_three_aliases() -> None:
    """Three-alias WITH DISTINCT from single connected pattern (#880)."""
    graph = _mk_graph(
        pd.DataFrame({
            "id": ["a", "b", "c", "d"],
            "label__A": [True, False, False, False],
            "val": [1, 2, 3, 4],
        }),
        pd.DataFrame({
            "s": ["a", "a", "b", "c"],
            "d": ["b", "c", "d", "d"],
            "type": ["R", "R", "S", "S"],
        }),
    )
    result = graph.gfql(
        "MATCH (a:A)-[:R]->(b)-[:S]->(c) "
        "WITH DISTINCT a, b, c "
        "RETURN a.id AS aid, b.id AS bid, c.id AS cid "
        "ORDER BY bid",
    )
    assert result._nodes.to_dict(orient="records") == [
        {"aid": "a", "bid": "b", "cid": "d"},
        {"aid": "a", "bid": "c", "cid": "d"},
    ]


def test_string_cypher_multi_alias_with_extend_scalar_order_by() -> None:
    """Extended scalar visible in ORDER BY on the next WITH stage (#1045)."""
    graph = _mk_ic4_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
        "WITH DISTINCT tag, post "
        "WITH tag, post.creationDate AS cd "
        "WITH tag, cd ORDER BY cd DESC "
        "RETURN tag.name AS tn, cd",
        params={"pid": "p1"},
    )
    records = result._nodes.to_dict(orient="records")
    assert [r["cd"] for r in records] == [300, 200, 100]


def test_string_cypher_multi_alias_with_non_final_agg_two_aliases_survive() -> None:
    """Non-final WITH aggregate: both grouping alias AND a second passed-through alias survive (#1054)."""
    graph = _mk_ic4_shape_graph()
    # post is included so that after aggregation we can still access post.id in the RETURN —
    # but post is NOT a group key here, only tag is.  The test verifies that tag.* columns
    # survive the non-final aggregate stage (post.* are not group keys so they drop, which is correct).
    result = graph.gfql(
        "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
        "WITH DISTINCT tag, post "
        "WITH tag, post.creationDate AS cd "
        "WITH tag, sum(cd) AS total "
        "RETURN tag.name AS tn, tag.id AS tid, total ORDER BY tn",
        params={"pid": "p1"},
    )
    records = result._nodes.to_dict(orient="records")
    assert len(records) == 2
    assert records[0]["tn"] == "TagA"
    assert records[0]["tid"] == "tag1"
    assert records[0]["total"] == 300
    assert records[1]["tn"] == "TagB"
    assert records[1]["tid"] == "tag2"


def test_string_cypher_multi_alias_with_non_final_agg_where_on_alias_property() -> None:
    """WHERE on alias.property in RETURN after non-final WITH aggregate is supported (#1054)."""
    graph = _mk_ic4_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
        "WITH DISTINCT tag, post "
        "WITH tag, post.creationDate AS cd "
        "WITH tag, sum(cd) AS total "
        "RETURN tag.name AS tn, total ORDER BY tn",
        params={"pid": "p1"},
    )
    # Filter in Python to confirm both aliases are accessible post-agg
    records = result._nodes.to_dict(orient="records")
    assert {"tn": "TagA", "total": 300} in records
    assert {"tn": "TagB", "total": 300} in records


def test_string_cypher_multi_alias_with_three_stage_case_aggregation() -> None:
    """IC-4 full shape with CASE in intermediate WITH (#880)."""
    graph = _mk_ic4_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
        "WITH DISTINCT tag, post "
        "WITH tag, "
        "CASE WHEN $lo <= post.creationDate AND post.creationDate < $hi THEN 1 ELSE 0 END AS valid, "
        "CASE WHEN post.creationDate < $lo THEN 1 ELSE 0 END AS inValid "
        "WITH tag, sum(valid) AS postCount, sum(inValid) AS inValidPostCount "
        "WHERE postCount > 0 AND inValidPostCount = 0 "
        "RETURN tag.name AS tagName, postCount "
        "ORDER BY postCount DESC, tagName ASC "
        "LIMIT 10",
        params={"pid": "p1", "lo": 150, "hi": 350},
    )
    records = result._nodes.to_dict(orient="records")
    assert len(records) >= 1
    assert all(r["postCount"] > 0 for r in records)


def test_issue_1413_ic4_new_topics_exact_ldbc_reference_query() -> None:
    """IC-4 new-topics exact LDBC reference shape: joined rows + CASE aggregation (#1413, #880)."""
    graph = _mk_ic4_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $personId})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag) "
        "WITH DISTINCT tag, post "
        "WITH tag, "
        "CASE WHEN $startDate <= post.creationDate < $endDate THEN 1 ELSE 0 END AS valid, "
        "CASE WHEN post.creationDate < $startDate THEN 1 ELSE 0 END AS inValid "
        "WITH tag, sum(valid) AS postCount, sum(inValid) AS inValidPostCount "
        "WHERE postCount>0 AND inValidPostCount=0 "
        "RETURN tag.name AS tagName, postCount "
        "ORDER BY postCount DESC, tagName ASC "
        "LIMIT 10",
        params={"personId": "p1", "startDate": 150, "endDate": 350},
    )
    assert result._nodes.to_dict(orient="records") == [
        {"tagName": "TagB", "postCount": 1},
    ]


def test_issue_1413_ic4_new_topics_multiple_chained_case_flags() -> None:
    graph = _mk_ic4_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $personId})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag) "
        "WITH DISTINCT tag, post "
        "WITH tag, "
        "CASE WHEN $startDate <= post.creationDate < $endDate THEN 1 ELSE 0 END AS valid, "
        "CASE WHEN $invalidStart <= post.creationDate < $startDate THEN 1 ELSE 0 END AS inValid "
        "WITH tag, sum(valid) AS postCount, sum(inValid) AS inValidPostCount "
        "WHERE postCount > 0 AND inValidPostCount = 0 "
        "RETURN tag.name AS tagName, postCount "
        "ORDER BY postCount DESC, tagName ASC "
        "LIMIT 10",
        params={"personId": "p1", "invalidStart": 0, "startDate": 150, "endDate": 350},
    )
    assert result._nodes.to_dict(orient="records") == [
        {"tagName": "TagB", "postCount": 1},
    ]


def test_issue_1413_ic3_cross_country_carried_row_collect_list_reentry_case_sum() -> None:
    graph = _mk_ic3_cross_country_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $personId})-[:IS_LOCATED_IN]->(city:City) "
        "WITH person, collect(city) AS cities "
        "MATCH (person)-[:KNOWS]-(friend:Person)-[:IS_LOCATED_IN]->(friendCity:City) "
        "WHERE NOT person = friend AND NOT friendCity IN cities "
        "WITH friend, "
        "CASE WHEN friendCity.name = $countryXName THEN 1 ELSE 0 END AS messageX, "
        "CASE WHEN friendCity.name = $countryYName THEN 1 ELSE 0 END AS messageY "
        "WITH friend, sum(messageX) AS xCount, sum(messageY) AS yCount "
        "RETURN friend.id AS friendId, xCount, yCount "
        "ORDER BY friendId ASC",
        params={"personId": "p1", "countryXName": "CityB", "countryYName": "CityC"},
    )

    assert result._nodes.to_dict(orient="records") == [
        {"friendId": "friend1", "xCount": 1, "yCount": 0},
        {"friendId": "friend3", "xCount": 0, "yCount": 1},
    ]


def test_issue_1413_ic3_collect_distinct_entity_membership_with_post_aggregate_where() -> None:
    graph = _mk_ic3_cross_country_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $personId})-[:IS_LOCATED_IN]->(city:City) "
        "WITH person, collect(DISTINCT city) AS cities "
        "MATCH (person)-[:KNOWS]-(friend:Person)-[:IS_LOCATED_IN]->(friendCity:City) "
        "WHERE NOT (friendCity IN cities) "
        "WITH friend, "
        "CASE WHEN friendCity.name = $countryXName THEN 1 ELSE 0 END AS messageX, "
        "CASE WHEN friendCity.name = $countryYName THEN 1 ELSE 0 END AS messageY "
        "WITH friend, sum(messageX) AS xCount, sum(messageY) AS yCount "
        "WHERE xCount > 0 OR yCount > 0 "
        "RETURN friend.id AS friendId, xCount, yCount "
        "ORDER BY yCount DESC, friendId ASC",
        params={"personId": "p1", "countryXName": "CityB", "countryYName": "CityC"},
    )

    assert result._nodes.to_dict(orient="records") == [
        {"friendId": "friend3", "xCount": 0, "yCount": 1},
        {"friendId": "friend1", "xCount": 1, "yCount": 0},
    ]


def _mk_1712_graph() -> "_CypherTestGraph":
    # p0,p1 have interest Books; p2 has Music; all three LIVES_IN NYC.
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 10, 20, 21],
        "node_type": ["Person", "Person", "Person", "City", "Interest", "Interest"],
        "interest": [None, None, None, None, "Books", "Music"],
    })
    edges = pd.DataFrame({
        "s": [0, 1, 2, 0, 1, 2],
        "d": [10, 10, 10, 20, 20, 21],
        "rel": ["LIVES_IN", "LIVES_IN", "LIVES_IN", "HAS_INTEREST", "HAS_INTEREST", "HAS_INTEREST"],
    })
    return cast(_CypherTestGraph, _CypherTestGraph().nodes(nodes, "id").edges(edges, "s", "d"))


@pytest.mark.parametrize("carry", ["WITH p", "WITH p, collect(i.interest) AS ii"])
def test_issue_1712_subset_with_carry_restricts_second_match(carry: str) -> None:
    """#1712: the graph-benchmark q5/q6/q7 shape — filter a SUBSET of Person in the
    first MATCH, carry via WITH, re-MATCH from the carried nodes — must count only the
    carried subset (p0,p1 have Books → numPersons=2, not 3). The bug: the reentry
    seed was never wired to the binding-ops build, so `p` re-matched the whole graph.
    (The coverage gap that let the benchmark shortcuts hide it.)"""
    result = _mk_1712_graph().gfql(
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}) "
        "WHERE i.interest = 'Books' "
        f"{carry} "
        "MATCH (p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "RETURN count(p) AS numPersons"
    )
    assert result._nodes.to_dict(orient="records") == [{"numPersons": 2}]


@pytest.mark.parametrize("agg,expected", [
    ("avg(p.age)", [{"city": "LA", "v": 40.0}, {"city": "NYC", "v": 25.0}]),
    ("sum(p.age)", [{"city": "LA", "v": 40}, {"city": "NYC", "v": 50}]),
    ("count(p.age)", [{"city": "LA", "v": 1}, {"city": "NYC", "v": 2}]),
])
def test_issue_1273_multi_source_grouped_aggregate(agg: str, expected: list) -> None:
    """#1273: a CLEAN grouped aggregate `func(<alias>.<prop>)` grouped by another
    alias's property (graph-benchmark q3 `RETURN c.city, avg(p.age)`) routes to the
    bindings-row table instead of NIE-ing with 'one MATCH source alias'. p0(20),p1(30)
    live in NYC; p2(40) lives in LA."""
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 10, 11],
        "node_type": ["Person", "Person", "Person", "City", "City"],
        "age": [20, 30, 40, 0, 0],
        "city": [None, None, None, "NYC", "LA"],
    })
    edges = pd.DataFrame({"s": [0, 1, 2], "d": [10, 10, 11],
                          "rel": ["LIVES_IN", "LIVES_IN", "LIVES_IN"]})
    graph = cast(_CypherTestGraph, _CypherTestGraph().nodes(nodes, "id").edges(edges, "s", "d"))
    result = graph.gfql(
        "MATCH (p {node_type:'Person'})-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        f"RETURN c.city AS city, {agg} AS v ORDER BY city"
    )
    assert result._nodes.to_dict(orient="records") == expected



def test_issue_1712_connected_comma_pattern_where_intersects() -> None:
    """#1712: a connected comma-pattern sharing a node alias with a WHERE on a leaf
    alias must intersect both patterns (the WHERE was silently dropped on the
    structured-predicate path). Same expected count as the WITH form."""
    result = _mk_1712_graph().gfql(
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "WHERE i.interest = 'Books' "
        "RETURN count(p) AS numPersons"
    )
    assert result._nodes.to_dict(orient="records") == [{"numPersons": 2}]


def _mk_graph_benchmark_t1_shape_graph() -> "_CypherTestGraph":
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 10, 11, 20, 21],
        "node_type": ["Person", "Person", "Person", "City", "City", "Interest", "Interest"],
        "gender": ["MALE", "female", "male", None, None, None, None],
        "age": [25, 27, 35, None, None, None, None],
        "city": [None, None, None, "London", "Paris", None, None],
        "country": [None, None, None, "United Kingdom", "France", None, None],
        "state": [None, None, None, "England", "Ile-de-France", None, None],
        "interest": [None, None, None, None, None, "Fine Dining", "photography"],
    })
    edges = pd.DataFrame({
        "s": [0, 1, 2, 0, 1, 2],
        "d": [20, 20, 21, 10, 10, 11],
        "rel": ["HAS_INTEREST", "HAS_INTEREST", "HAS_INTEREST", "LIVES_IN", "LIVES_IN", "LIVES_IN"],
    })
    return _mk_graph(nodes, edges)


def _compiled_connected_join_filters(query: str) -> list[dict[str, Any]]:
    compiled = _compile_query(query)
    plan = _compiled_execution_extras(compiled).connected_match_join
    assert plan is not None
    out: list[dict[str, Any]] = []
    for chain in plan.pattern_chains:
        for op in chain.chain:
            if isinstance(op, ASTNode) and isinstance(op._name, str):
                out.append({op._name: dict(op.filter_dict or {})})
    return out


def _compiled_connected_join_plan(query: str) -> Any:
    compiled = _compile_query(query)
    plan = _compiled_execution_extras(compiled).connected_match_join
    assert plan is not None
    return plan


def _post_join_functions(query: str) -> list[str]:
    return [op.function for op in _compiled_connected_join_plan(query).post_join_chain.chain if isinstance(op, ASTCall)]


def _mk_graph_benchmark_t1_labelled_shape_graph() -> "_CypherTestGraph":
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 10, 11, 20, 21],
        "node_type": ["Person", "Person", "Person", "City", "City", "Interest", "Interest"],
        "age": [25, 27, 35, None, None, None, None],
        "city": [None, None, None, "London", "Paris", None, None],
        "interest": [None, None, None, None, None, "Fine Dining", "photography"],
        "label__Person": [True, True, True, False, False, False, False],
        "label__City": [False, False, False, True, True, False, False],
        "label__Interest": [False, False, False, False, False, True, True],
    })
    edges = pd.DataFrame({
        "s": [0, 1, 2, 0, 1, 2],
        "d": [20, 20, 21, 10, 10, 11],
        "rel": ["HAS_INTEREST", "HAS_INTEREST", "HAS_INTEREST", "LIVES_IN", "LIVES_IN", "LIVES_IN"],
    })
    return _mk_graph(nodes, edges)


def test_t1_connected_comma_pushes_label_predicate_with_property_filter() -> None:
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "WHERE p:Person AND p.age >= 26 "
        "RETURN count(p) AS numPersons"
    )

    result = _mk_graph_benchmark_t1_labelled_shape_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"numPersons": 2}]

    filters_by_alias = _compiled_connected_join_filters(query)
    assert any("label__Person" in entry.get("p", {}) for entry in filters_by_alias)
    assert any("age" in entry.get("p", {}) for entry in filters_by_alias)


def test_t1_connected_comma_mixes_signed_literal_pushdown_and_in_residual() -> None:
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "WHERE toLower(i.interest) = toLower('fine dining') "
        "AND p.age >= -1 AND p.age IN [25, 27] "
        "RETURN count(p) AS numPersons"
    )

    result = _mk_graph_benchmark_t1_shape_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"numPersons": 2}]

    plan = _compiled_connected_join_plan(query)
    assert "where_rows" in _post_join_functions(query)
    assert plan.pattern_attach_prop_aliases == (("i", "p"), ("p",))
    filters_by_alias = _compiled_connected_join_filters(query)
    assert any("age" in entry.get("p", {}) for entry in filters_by_alias)


def _mk_graph_benchmark_t1_nullable_shape_graph() -> "_CypherTestGraph":
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 3, 10, 11, 20, 21],
        "node_type": ["Person"] * 4 + ["City"] * 2 + ["Interest"] * 2,
        "age": [25, 27, 35, None, None, None, None, None],
        "nick": ["a", "b", "c", None, None, None, None, None],
        "flag": [True, False, True, None, None, None, None, None],
        "city": [None] * 4 + ["London", "Paris", None, None],
        "interest": [None] * 6 + ["Fine Dining", "photography"],
    })
    edges = pd.DataFrame({
        "s": [0, 1, 2, 3, 0, 1, 2, 3],
        "d": [20, 20, 21, 20, 10, 10, 11, 11],
        "rel": ["HAS_INTEREST"] * 4 + ["LIVES_IN"] * 4,
    })
    return _mk_graph(nodes, edges)


def _real_t1_nullable_graph() -> Plottable:
    # A real Plottable, not `_CypherTestGraph`: the executor's serialize/deserialize
    # round-trip is where dropped-null and revived-predicate filters actually bite, and
    # the test double does not reproduce it.
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 3, 10, 11, 20, 21],
        "node_type": ["Person"] * 4 + ["City"] * 2 + ["Interest"] * 2,
        "age": [25, 27, 35, None, None, None, None, None],
        "nick": ["a", "b", "c", None, None, None, None, None],
        "flag": [True, False, True, None, None, None, None, None],
        "city": [None] * 4 + ["London", "Paris", None, None],
        "interest": [None] * 6 + ["Fine Dining", "photography"],
    })
    edges = pd.DataFrame({
        "s": [0, 1, 2, 3, 0, 1, 2, 3],
        "d": [20, 20, 21, 20, 10, 10, 11, 11],
        "rel": ["HAS_INTEREST"] * 4 + ["LIVES_IN"] * 4,
    })
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _t1_nullable_query(predicate: str) -> str:
    return (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        f"WHERE {predicate} "
        "RETURN count(p) AS n"
    )


def test_connected_join_does_not_push_null_equality() -> None:
    # `filter_dict` serialization drops null values, so a pushed `nick = null` would
    # vanish and return every row unfiltered. It must stay a residual instead.
    query = _t1_nullable_query("p.nick = $v")

    result = _mk_graph_benchmark_t1_nullable_shape_graph().gfql(query, params={"v": None})
    assert result._nodes.to_dict(orient="records") == []

    compiled = cast(CompiledCypherQuery, compile_cypher(query, params={"v": None}))
    plan = _compiled_execution_extras(compiled).connected_match_join
    assert plan is not None
    pushed_props = {
        prop
        for chain in plan.pattern_chains
        for op in chain.chain
        if isinstance(op, ASTNode)
        for prop in (op.filter_dict or {})
    }
    assert "nick" not in pushed_props
    residuals = [op.function for op in plan.post_join_chain.chain if isinstance(op, ASTCall)]
    assert "where_rows" in residuals


@pytest.mark.parametrize(
    "predicate,expected",
    [
        ("p.nick <> 'a'", 2),
        ("i.interest <> 'photography'", 3),
        ("p.nick < 'c'", 2),
        ("p.flag >= true", 2),
    ],
)
def test_connected_join_does_not_push_non_numeric_ordering(predicate: str, expected: int) -> None:
    # Ordering/inequality ops lower to numeric-only predicates; pushing a string or bool
    # raises, so these must fall back to a residual and still answer correctly.
    query = _t1_nullable_query(predicate)

    result = _mk_graph_benchmark_t1_nullable_shape_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"n": expected}]

    assert "where_rows" in _post_join_functions(query)


@pytest.mark.parametrize(
    "value",
    [
        {"type": "GT", "val": 26},
        {"type": "Between", "lower": 0, "upper": 100, "inclusive": True},
        [25, 27],
    ],
)
def test_connected_join_does_not_push_non_scalar_param(value: Any) -> None:
    # `_filter_dict_to_json` passes a dict through verbatim and `maybe_filter_dict_from_json`
    # revives it through `predicates_from_json`, so pushing a caller-supplied map would run
    # an arbitrary predicate (`age > 26`) instead of an equality against a map.
    # Must use a real Plottable: the serialization round-trip that resurrects the predicate
    # only happens on the real executor, so `_CypherTestGraph` cannot observe this.
    query = _t1_nullable_query("p.age = $v")

    result = _real_t1_nullable_graph().gfql(query, params={"v": value})
    assert result._nodes.to_dict(orient="records") == []


@pytest.mark.parametrize(
    "predicate,expected",
    [
        ("p.nick STARTS WITH 'a'", 1),
        ("p.nick CONTAINS 'a'", 1),
        ("p.nick ENDS WITH 'a'", 1),
        ("p.nick =~ 'a'", 1),
    ],
)
def test_connected_join_pushes_string_predicates(predicate: str, expected: int) -> None:
    # These lower to real round-trippable ASTPredicates, and the residual cannot render
    # them at all, so they must push rather than fall back.
    query = _t1_nullable_query(predicate)

    result = _mk_graph_benchmark_t1_nullable_shape_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"n": expected}]

    filters_by_alias = _compiled_connected_join_filters(query)
    assert any("nick" in entry.get("p", {}) for entry in filters_by_alias)


@pytest.mark.parametrize(
    "predicate,expected",
    [
        # string literal vs numeric column, and numeric literal vs string column: the
        # residual answers these leniently, so pushing them must not turn them into errors
        ("p.age = 'foo'", []),
        ("p.age = date('2020-01-01')", []),
        ("p.nick >= 26", []),
        ("p.nick < 26", []),
        ("p.nick <> 26", [{"n": 3}]),
    ],
)
def test_connected_join_does_not_push_dtype_incompatible_atoms(predicate: str, expected: Any) -> None:
    query = _t1_nullable_query(predicate)

    result = _real_t1_nullable_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == expected


def _real_bool_shape_graph() -> Plottable:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "flag": pd.Series([True, False, True, False], dtype="bool"),
        "age": pd.Series([25, 27, 35, 40], dtype="int64"),
    })
    edges = pd.DataFrame({"s": ["a", "b", "c", "d"], "d": ["b", "c", "d", "a"]})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize(
    "predicate,expected",
    [
        # `bool` counts as numeric to the live validator, so a string value against a bool
        # column must stay residual rather than push into a type error.
        ("p.flag = 'yes'", []),
        ("p.flag = true", [{"n": 2}]),
        ("p.age >= 26", [{"n": 3}]),
    ],
)
def test_connected_join_bool_column_matches_validator(predicate: str, expected: Any) -> None:
    query = f"MATCH (p)-[]->(x), (p)-[]->(y) WHERE {predicate} RETURN count(p) AS n"

    result = _real_bool_shape_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == expected


@pytest.mark.parametrize(
    "predicate",
    [
        "p.age = 99999999999999999999",
        "p.age <> 99999999999999999999",
        "p.flag = 99999999999999999999",
    ],
)
def test_connected_join_out_of_range_int_still_reports_range_error(predicate: str) -> None:
    # The 64-bit literal guard lives on the row-expr path; pushing an out-of-range int
    # would evade it and surface a raw OverflowError from pandas instead.
    query = f"MATCH (p)-[]->(x), (p)-[]->(y) WHERE {predicate} RETURN count(p) AS n"

    with pytest.raises(GFQLValidationError):
        _real_bool_shape_graph().gfql(query)


class _FakeDtype:
    """Stands in for a non-pandas dtype (polars/cuDF) without importing that engine."""

    def __init__(self, text: str, kind: Optional[str] = None) -> None:
        self._text = text
        if kind is not None:
            self.kind = kind

    def __str__(self) -> str:
        return self._text


class _UnprintableDtype:
    def __str__(self) -> str:
        raise RuntimeError("dtype has no text form")


@pytest.mark.parametrize(
    "dtype,expected",
    [
        # `pd.api.types` returns False (not raises) for these, so classification has to
        # fall back to kind/text or the gate would read them as "safe to push".
        (_FakeDtype("Int64"), (True, False)),
        (_FakeDtype("Float64"), (True, False)),
        (_FakeDtype("Boolean"), (True, False)),
        (_FakeDtype("Decimal(38,10)"), (True, False)),
        (_FakeDtype("String"), (False, True)),
        (_FakeDtype("Utf8"), (False, True)),
        (_FakeDtype("Categorical(ordering='physical')"), (False, True)),
        (_FakeDtype("i-am-not-a-dtype", kind="i"), (True, False)),
        # Container dtypes embed their element type, so a substring match reads
        # `interval[int64, right]` / `List(Int64)` as numeric and `struct` as string.
        # They are never scalar-comparable and must fail closed.
        (_FakeDtype("interval[int64, right]"), (False, False)),
        (_FakeDtype("List(Int64)"), (False, False)),
        (_FakeDtype("Array(Float64, 2)"), (False, False)),
        (_FakeDtype("struct"), (False, False)),
        (_FakeDtype("Struct({'a': Int64})"), (False, False)),
        (_FakeDtype("Binary"), (False, False)),
        # Unrecognized must fail closed, not be guessed at.
        (_FakeDtype("Date"), (False, False)),
        (_FakeDtype("Duration(time_unit='us')"), (False, False)),
        (_UnprintableDtype(), (False, False)),
    ],
)
def test_connected_join_dtype_classes_falls_back_for_non_pandas_dtypes(dtype: Any, expected: Any) -> None:
    assert _connected_join_dtype_classes(dtype) == expected


@pytest.mark.parametrize(
    "op,value,dtype",
    [
        ("==", "foo", _FakeDtype("Date")),
        (">=", 26, _FakeDtype("Date")),
        ("contains", "o", _FakeDtype("Date")),
    ],
)
def test_connected_join_unknown_dtype_never_pushes(op: str, value: Any, dtype: Any) -> None:
    assert _connected_join_dtype_admits(op, value, dtype) is False


def test_connected_join_fake_dtype_verdicts_match_pandas() -> None:
    numeric_fake, string_fake = _FakeDtype("Int64"), _FakeDtype("String")
    numeric_pd, string_pd = pd.Series([1]).dtype, pd.Series(["a"]).dtype

    for numeric, string in [(numeric_pd, string_pd), (numeric_fake, string_fake)]:
        assert _connected_join_dtype_admits("==", "foo", numeric) is False
        assert _connected_join_dtype_admits(">=", 26, numeric) is True
        assert _connected_join_dtype_admits("contains", "o", string) is True
        assert _connected_join_dtype_admits("==", 26, string) is False


@pytest.mark.parametrize("predicate,expected", [("p.iv > 1", []), ("p.iv >= 1", []), ("p.iv <> 1", [{"n": 8}]), ("p.iv = 1", [])])
def test_connected_join_interval_column_matches_master(predicate: str, expected: Any) -> None:
    # `interval[int64, right]` contains "int": classified numeric, a comparison pushed onto
    # it raised a raw ValueError out of pandas where the residual answers correctly.
    nodes = pd.DataFrame({"id": ["p1", "p2", "a1", "b1", "a2", "b2"]})
    nodes["iv"] = pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3, 4, 5, 6])
    edges = pd.DataFrame({"s": ["p1", "p1", "p2", "p2"], "d": ["a1", "b1", "a2", "b2"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    query = f"MATCH (p)-[]->(x), (p)-[]->(y) WHERE {predicate} RETURN count(p) AS n"

    assert g.gfql(query)._nodes.to_dict(orient="records") == expected


def _real_labelled_inline_graph() -> Plottable:
    nodes = pd.DataFrame([
        {"id": "p1", "label__Person": True, "label__Place": False, "nick": "aa", "age": 30},
        {"id": "p2", "label__Person": True, "label__Place": False, "nick": "bb", "age": 40},
        {"id": "c1", "label__Person": False, "label__Place": True, "nick": None, "age": None},
    ])
    edges = pd.DataFrame([{"s": "p1", "d": "c1", "type": "L"}, {"s": "p2", "d": "c1", "type": "L"}])
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize(
    "inline,predicate,expected",
    [
        # Merging onto an existing STRING inline value wraps it with comparison.eq, which
        # serializes to {'type':'EQ'} -- a tag from_json binds to the numeric-only EQ, so the
        # executor raises when it rehydrates. Don't create that shape; stay residual.
        ("{nick:'aa'}", "friend.nick = 'aa'", [{"n": 1}]),
        ("{nick:'aa'}", "friend.nick = 'bb'", []),
        # A different property never merges, and a numeric inline value rehydrates fine.
        ("{nick:'aa'}", "friend.age > 20", [{"n": 1}]),
        ("{age:30}", "friend.age > 20", [{"n": 1}]),
        ("{}", "friend.nick = 'aa'", [{"n": 1}]),
    ],
)
def test_connected_join_inline_string_property_merge_matches_master(
    inline: str, predicate: str, expected: Any
) -> None:
    query = (
        "MATCH (person:Person {id:'p1'})-[:L]->(city:Place), "
        f"(friend:Person {inline})-[:L]->(city) "
        f"WHERE {predicate} RETURN count(friend) AS n"
    )

    result = _real_labelled_inline_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == expected


def _real_string_merge_graph() -> Plottable:
    nodes = pd.DataFrame({
        "id": ["p1", "p2", "p3", "aa", "bb"],
        "name": ["alice", "bob", "carol", "aa", "bb"],
        "age": [30, 40, 50, 1, 2],
    })
    edges = pd.DataFrame({
        "s": ["p1", "p1", "p2", "p2", "p3", "p3"],
        "d": ["aa", "bb", "aa", "bb", "aa", "bb"],
    })
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize(
    "predicate,expected",
    [
        # filter_dict is mutated as earlier atoms push, so a previously-pushed string
        # predicate must not green-light merging a raw string behind it. Checking only the
        # existing side made this order-dependent.
        ("p.name CONTAINS 'al' AND p.name = 'alice'", [{"n": 4}]),
        ("p.name STARTS WITH 'al' AND p.name = 'alice'", [{"n": 4}]),
        ("p.name CONTAINS 'a' AND p.name CONTAINS 'l'", [{"n": 8}]),
        ("p.name CONTAINS 'al'", [{"n": 4}]),
        # Numeric merges are representable and must still push.
        ("p.age > 20 AND p.age = 30", [{"n": 4}]),
    ],
)
def test_connected_join_string_predicate_merge_matches_cypher(predicate: str, expected: Any) -> None:
    query = f"MATCH (p)-[]->(a), (p)-[]->(b) WHERE {predicate} RETURN count(p) AS n"

    result = _real_string_merge_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == expected


def test_connected_join_dtype_classes_handles_non_pandas_dtypes() -> None:
    # `pd.api.types.is_numeric_dtype(pl.Int64())` returns False rather than raising, so
    # filter_by_dict's helpers never reach their fallback and a polars column would look
    # neither-numeric-nor-string -- which the gate would read as "safe to push".
    pl = pytest.importorskip("polars")

    assert _connected_join_dtype_classes(pl.Int64()) == (True, False)
    assert _connected_join_dtype_classes(pl.Float64()) == (True, False)
    assert _connected_join_dtype_classes(pl.String()) == (False, True)
    assert _connected_join_dtype_classes(pd.Series([1]).dtype) == (True, False)
    assert _connected_join_dtype_classes(pd.Series(["a"]).dtype) == (False, True)
    # Unrecognized dtype must fail closed rather than be treated as pushable.
    assert _connected_join_dtype_classes(pl.Date()) == (False, False)
    assert _connected_join_dtype_admits("==", "foo", pl.Date()) is False


def test_connected_join_polars_nodes_match_pandas_gate_verdicts() -> None:
    pl = pytest.importorskip("polars")

    for numeric_dtype, string_dtype in [
        (pd.Series([1]).dtype, pd.Series(["a"]).dtype),
        (pl.Int64(), pl.String()),
    ]:
        assert _connected_join_dtype_admits("==", "foo", numeric_dtype) is False
        assert _connected_join_dtype_admits(">=", 26, numeric_dtype) is True
        assert _connected_join_dtype_admits("contains", "o", string_dtype) is True
        assert _connected_join_dtype_admits("==", 26, string_dtype) is False


def test_connected_join_polars_backed_graph_matches_pandas_results() -> None:
    # The whole dtype gate was inert on polars-typed nodes: `p.age = 'foo'` pushed and
    # raised where a pandas-backed graph correctly answers [].
    pl = pytest.importorskip("polars")

    nodes = pd.DataFrame({"id": ["p1", "p2", "p3", "x1", "y1"], "age": [25, 26, 30, 1, 2]})
    edges = pd.DataFrame({"s": ["p1", "p1", "p2", "p2", "p3", "p3"], "d": ["x1", "y1", "x1", "y1", "x1", "y1"]})
    query = "MATCH (p)-->(a), (p)-->(b) WHERE p.age = 'foo' RETURN count(p) AS n"

    g_polars = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    result = g_polars.gfql(query)._nodes
    records = result.to_dict(orient="records") if hasattr(result, "to_dict") else result.to_dicts()
    assert records == []


def test_node_dtypes_for_pushdown_reads_columns_not_items() -> None:
    # pandas/cuDF expose a column-indexed Series but polars exposes a plain list, so the
    # mapping must be built by zipping columns with dtypes rather than calling `.items()`.
    g = _real_bool_shape_graph()
    dtypes = _node_dtypes_for_pushdown(g)
    assert dtypes is not None
    assert set(dtypes) == {"id", "flag", "age"}
    assert _connected_join_dtype_admits("==", "yes", dtypes["flag"]) is False
    assert _connected_join_dtype_admits("==", "yes", dtypes["id"]) is True


def test_connected_join_dtype_schema_selects_pushdown() -> None:
    # With dtypes the planner pushes what the column admits and leaves the rest residual.
    g = _real_t1_nullable_graph()
    dtypes = _node_dtypes_for_pushdown(g)
    assert dtypes is not None

    def pushed_props(predicate: str) -> set:
        compiled = cast(CompiledCypherQuery, compile_cypher(_t1_nullable_query(predicate), _node_dtypes=dtypes))
        plan = _compiled_execution_extras(compiled).connected_match_join
        assert plan is not None
        return {
            prop
            for chain in plan.pattern_chains
            for op in chain.chain
            if isinstance(op, ASTNode)
            for prop in (op.filter_dict or {})
        } - {"node_type"}

    assert pushed_props("p.age >= 26") == {"age"}
    assert pushed_props("p.nick = 'a'") == {"nick"}
    assert pushed_props("p.nick STARTS WITH 'a'") == {"nick"}
    assert pushed_props("p.nick >= 26") == set()
    assert pushed_props("p.age = 'foo'") == set()


@pytest.mark.parametrize("node_id_column", ["id", "nid"])
def test_connected_join_count_distinct_alias_is_node_identity(node_id_column: str) -> None:
    # A bare alias must lower to its identity column. Left unrewritten it only resolves when
    # the node id column happens to be named `id`/`p`, and otherwise degrades to a constant,
    # collapsing count(DISTINCT p) to 1.
    nodes = pd.DataFrame({
        node_id_column: [0, 1, 2, 10, 11, 20, 21],
        "node_type": ["Person"] * 3 + ["City"] * 2 + ["Interest"] * 2,
        "age": [25, 27, 35, None, None, None, None],
        "interest": [None] * 5 + ["Fine Dining", "photography"],
    })
    edges = pd.DataFrame({
        "s": [0, 1, 2, 0, 1, 2],
        "d": [20, 20, 21, 10, 10, 11],
        "rel": ["HAS_INTEREST"] * 3 + ["LIVES_IN"] * 3,
    })
    g = graphistry.nodes(nodes, node_id_column).edges(edges, "s", "d")
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "RETURN count(DISTINCT p) AS n"
    )

    assert g.gfql(query)._nodes.to_dict(orient="records") == [{"n": 3}]


@pytest.mark.parametrize("predicate", ["p.age >= - 26", "p.age >= -(26)", "p.age >= +(26)"])
def test_t1_connected_comma_pushes_unfolded_unary_literal(predicate: str) -> None:
    # `NUMBER` carries its sign as a lexer terminal, so only a lexically adjacent sign
    # folds into the literal. A spaced or parenthesised sign stays a UnaryOp node and
    # must still push down rather than degrade to a row residual.
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        f"WHERE {predicate} "
        "RETURN count(p) AS numPersons"
    )

    expected = 3 if predicate != "p.age >= +(26)" else 2
    result = _mk_graph_benchmark_t1_shape_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"numPersons": expected}]

    filters_by_alias = _compiled_connected_join_filters(query)
    assert any("age" in entry.get("p", {}) for entry in filters_by_alias)


def test_t1_connected_comma_q5_retains_lower_residual_and_props() -> None:
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "WHERE toLower(i.interest) = toLower('fine dining') "
        "AND toLower(p.gender) = toLower('male') "
        "AND c.city = 'London' AND c.country = 'United Kingdom' "
        "RETURN count(p) AS numPersons"
    )

    plan = _compiled_connected_join_plan(query)
    assert "where_rows" in _post_join_functions(query)
    assert plan.pattern_attach_prop_aliases == (("i", "p"), ("p",))


def test_t1_connected_comma_grouped_projection_attaches_only_city_props() -> None:
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "WHERE toLower(i.interest) = toLower('fine dining') "
        "AND p.age >= 23 AND p.age <= 30 "
        "AND c.country = 'United Kingdom' "
        "RETURN count(p) AS numPersons, c.state AS state, c.country AS country "
        "ORDER BY state"
    )

    plan = _compiled_connected_join_plan(query)
    assert "where_rows" in _post_join_functions(query)
    assert plan.pattern_attach_prop_aliases == (("i",), ("c",))


def test_t1_connected_comma_pushes_q5_literal_filters_and_retains_lower_residual() -> None:
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "WHERE toLower(i.interest) = toLower('fine dining') "
        "AND toLower(p.gender) = toLower('male') "
        "AND c.city = 'London' AND c.country = 'United Kingdom' "
        "RETURN count(p) AS numPersons"
    )

    result = _mk_graph_benchmark_t1_shape_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"numPersons": 1}]

    plan = _compiled_connected_join_plan(query)
    assert "where_rows" in _post_join_functions(query)
    assert plan.pattern_attach_prop_aliases == (("i", "p"), ("p",))

    filters_by_alias = _compiled_connected_join_filters(query)
    assert not any("interest" in entry.get("i", {}) for entry in filters_by_alias)
    assert not any("gender" in entry.get("p", {}) for entry in filters_by_alias)
    assert any(entry.get("c", {}).get("city") == "London" for entry in filters_by_alias)
    assert any(entry.get("c", {}).get("country") == "United Kingdom" for entry in filters_by_alias)


def test_t1_connected_comma_pushes_reversed_single_alias_filters_before_join() -> None:
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "WHERE toLower('fine dining') = toLower(i.interest) "
        "AND 23 <= p.age AND 30 >= p.age "
        "AND 'London' = c.city "
        "RETURN count(p) AS numPersons"
    )

    result = _mk_graph_benchmark_t1_shape_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"numPersons": 2}]

    plan = _compiled_connected_join_plan(query)
    assert "where_rows" in _post_join_functions(query)
    assert plan.pattern_attach_prop_aliases == (("i",), ())

    filters_by_alias = _compiled_connected_join_filters(query)
    assert not any("interest" in entry.get("i", {}) for entry in filters_by_alias)
    assert any("age" in entry.get("p", {}) for entry in filters_by_alias)
    assert any(entry.get("c", {}).get("city") == "London" for entry in filters_by_alias)


def test_t1_connected_comma_retains_lower_property_plain_lowercase_literal() -> None:
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "WHERE toLower(i.interest) = 'fine dining' "
        "AND p.age >= 23 AND p.age <= 30 AND c.city = 'London' "
        "RETURN count(p) AS numPersons"
    )

    result = _mk_graph_benchmark_t1_shape_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == [{"numPersons": 2}]

    plan = _compiled_connected_join_plan(query)
    assert "where_rows" in _post_join_functions(query)
    assert plan.pattern_attach_prop_aliases == (("i",), ())


def test_t1_connected_comma_retains_reversed_uppercase_plain_literal_residual() -> None:
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "WHERE 'FINE DINING' = toLower(i.interest) "
        "AND p.age >= 23 AND p.age <= 30 AND c.city = 'London' "
        "RETURN count(p) AS numPersons"
    )

    result = _mk_graph_benchmark_t1_shape_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == []

    plan = _compiled_connected_join_plan(query)
    assert "where_rows" in _post_join_functions(query)
    assert plan.pattern_attach_prop_aliases == (("i",), ())


@pytest.mark.parametrize(
    "where_expr",
    [
        "toLower(i.interest) = toLower('i')",
        "toLower('i') = toLower(i.interest)",
    ],
)
def test_t1_connected_comma_retains_unicode_lower_equality_residual(where_expr: str) -> None:
    values = ["İ", "i", "I", "ı"]
    person_ids = list(range(4))
    interest_ids = list(range(10, 14))
    city_id = 20
    nodes = pd.DataFrame({
        "id": person_ids + interest_ids + [city_id],
        "node_type": ["Person"] * 4 + ["Interest"] * 4 + ["City"],
        "interest": [None] * 4 + values + [None],
        "city": [None] * 8 + ["London"],
    })
    edges = pd.DataFrame({
        "s": person_ids + person_ids,
        "d": interest_ids + [city_id] * 4,
        "rel": ["HAS_INTEREST"] * 4 + ["LIVES_IN"] * 4,
    })
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        f"WHERE {where_expr} AND c.city = 'London' "
        "RETURN count(p) AS n"
    )

    graph = _mk_graph(nodes, edges)
    oracle_query = (
        "MATCH (i {node_type:'Interest'}) "
        f"WHERE {where_expr} "
        "RETURN count(i) AS n"
    )
    oracle_compiled = cast(CompiledCypherQuery, compile_cypher(oracle_query))
    assert oracle_compiled.execution_extras.connected_match_join is None
    oracle_rows = graph.gfql(oracle_query)._nodes.to_dict(orient="records")
    assert oracle_rows[0]["n"] != len(values)

    result = graph.gfql(query)
    assert result._nodes.to_dict(orient="records") == oracle_rows

    plan = _compiled_connected_join_plan(query)
    assert "where_rows" in _post_join_functions(query)
    assert plan.pattern_attach_prop_aliases == (("i",), ())


def test_t1_connected_comma_pushes_q7_range_filters_before_join() -> None:
    query = (
        "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
        "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
        "WHERE toLower(i.interest) = toLower('photography') "
        "AND p.age >= 23 AND p.age <= 30 "
        "AND c.country = 'France' "
        "RETURN count(p) AS numPersons, c.state AS state, c.country AS country"
    )

    result = _mk_graph_benchmark_t1_shape_graph().gfql(query)
    assert result._nodes.to_dict(orient="records") == []

    plan = _compiled_connected_join_plan(query)
    assert "where_rows" in _post_join_functions(query)
    assert plan.pattern_attach_prop_aliases == (("i",), ("c",))

    filters_by_alias = _compiled_connected_join_filters(query)
    assert any("age" in entry.get("p", {}) for entry in filters_by_alias)
    assert not any("interest" in entry.get("i", {}) for entry in filters_by_alias)
    assert any(entry.get("c", {}).get("country") == "France" for entry in filters_by_alias)



def test_issue_1413_ic3_entity_membership_positive_same_city_friend_only() -> None:
    graph = _mk_ic3_cross_country_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $personId})-[:IS_LOCATED_IN]->(city:City) "
        "WITH person, collect(city) AS cities "
        "MATCH (person)-[:KNOWS]-(friend:Person)-[:IS_LOCATED_IN]->(friendCity:City) "
        "WHERE friendCity IN cities "
        "RETURN friend.id AS friendId "
        "ORDER BY friendId ASC",
        params={"personId": "p1"},
    )

    assert result._nodes.to_dict(orient="records") == [
        {"friendId": "friend2"},
    ]


def test_issue_1413_ic3_collect_whole_row_entities_render_after_binding_grouping() -> None:
    graph = _mk_ic3_cross_country_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $personId})-[:IS_LOCATED_IN]->(city:City) "
        "WITH person, collect(city) AS cities "
        "RETURN person.id AS personId, cities",
        params={"personId": "p1"},
    )

    assert result._nodes.to_dict(orient="records") == [
        {"personId": "p1", "cities": ["(:City {name: 'CityA'})"]},
    ]


def test_issue_1038_ic4_return_side_case_expression_regression_lock() -> None:
    """Regression lock for #1038: RETURN-side CASE over IC4-shaped post timestamp ranges."""
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "f1", "f2", "post1", "post2", "post3", "tag1", "tag2"],
                "label__Person": [True, True, True, False, False, False, False, False],
                "label__Post": [False, False, False, True, True, True, False, False],
                "label__Tag": [False, False, False, False, False, False, True, True],
                "name": ["", "", "", "", "", "", "TagA", "TagB"],
                "creationDate": [
                    0,
                    0,
                    0,
                    1275350400000,
                    1275264000000,
                    1306799999999,
                    0,
                    0,
                ],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p1", "post1", "post2", "post3", "post1", "post2", "post3"],
                "d": ["f1", "f2", "f1", "f1", "f2", "tag1", "tag1", "tag2"],
                "type": [
                    "KNOWS",
                    "KNOWS",
                    "HAS_CREATOR",
                    "HAS_CREATOR",
                    "HAS_CREATOR",
                    "HAS_TAG",
                    "HAS_TAG",
                    "HAS_TAG",
                ],
            }
        ),
    )

    result = graph.gfql(
        "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
        "WITH DISTINCT tag, post "
        "RETURN tag.name AS tagName, post.id AS rawPostId, "
        "CASE WHEN 1275350400000 <= post.creationDate AND post.creationDate < 1306886400000 "
        "THEN post.id ELSE null END AS postId "
        "ORDER BY tagName ASC, rawPostId ASC",
        params={"pid": "p1"},
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 3
    assert rows[0] == {"tagName": "TagA", "rawPostId": "post1", "postId": "post1"}
    assert rows[1]["tagName"] == "TagA"
    assert rows[1]["rawPostId"] == "post2"
    assert pd.isna(rows[1]["postId"])
    assert rows[2] == {"tagName": "TagB", "rawPostId": "post3", "postId": "post3"}


def test_issue_1038_rejects_aggregate_inside_row_case_expression() -> None:
    graph = _mk_ic4_shape_graph()

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(
            "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
            "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
            "WITH DISTINCT tag, post "
            "RETURN CASE WHEN post.creationDate > 150 THEN count(*) ELSE 0 END AS out",
            params={"pid": "p1"},
        )

    assert exc_info.value.code == ErrorCode.E108


def test_issue_1469_rejects_aggregate_inside_literal_map_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "label__A": [True, False],
                "label__B": [False, True],
                "num": [None, 42],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(
            "MATCH (a:A), (b:B) "
            "RETURN coalesce(a.num, b.num) AS foo, "
            "b.num AS bar, "
            "{name: count(b)} AS baz"
        )

    assert exc_info.value.code == ErrorCode.E108
    assert "aggregate expressions inside map literals" in str(exc_info.value)

    with pytest.raises(GFQLValidationError) as with_exc_info:
        graph.gfql(
            "MATCH (a:A), (b:B) "
            "WITH coalesce(a.num, b.num) AS foo, "
            "b.num AS bar, "
            "{name: count(b)} AS baz "
            "RETURN foo, bar, baz"
        )

    assert with_exc_info.value.code == ErrorCode.E108
    assert "aggregate expressions inside map literals" in str(with_exc_info.value)


# ---------------------------------------------------------------------------
# Issue #996: MATCH (connected) OPTIONAL MATCH ... RETURN mixed + CASE
# ---------------------------------------------------------------------------


def _mk_996_graph() -> _CypherTestGraph:
    """Small graph exercising connected-path MATCH + OPTIONAL MATCH.

    Nodes: a->b->c with types R1, R2.  a->c with type T (the "optional" edge).
    Node 'd' is connected a->d via R1 but has NO T edge to c, so the optional
    match should produce null for that row.
    """
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
    })
    edges = pd.DataFrame({
        "s": ["a", "b", "a", "a"],
        "d": ["b", "c", "c", "d"],
        "type": ["R1", "R2", "T", "R1"],
    })
    return _mk_graph(nodes, edges)


def test_issue_996_connected_match_optional_match_simple_projection() -> None:
    graph = _mk_996_graph()

    result = graph.gfql(
        "MATCH (a)-[r1:R1]->(b) "
        "OPTIONAL MATCH (a)-[r2:T]->(c) "
        "RETURN a.id AS aid, b.id AS bid, c.id AS cid"
    )

    rows = sorted(
        result._nodes.to_dict(orient="records"),
        key=lambda r: (str(r.get("aid", "")), str(r.get("bid", ""))),
    )
    assert len(rows) == 2
    assert rows[0]["aid"] == "a"
    assert rows[0]["bid"] == "b"
    assert rows[0]["cid"] == "c"
    assert rows[1]["aid"] == "a"
    assert rows[1]["bid"] == "d"
    assert rows[1]["cid"] == "c"


def test_issue_996_connected_match_optional_match_case_null() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "a", "b"],
        "d": ["b", "c", "c"],
        "type": ["R1", "R1", "T"],
    })
    graph2 = _mk_graph(nodes, edges)

    result = graph2.gfql(
        "MATCH (a)-[r1:R1]->(b) "
        "OPTIONAL MATCH (b)-[r2:T]->(c) "
        "RETURN b.id AS bid, "
        "CASE WHEN r2 IS NULL THEN false ELSE true END AS has_t"
    )

    rows = sorted(
        result._nodes.to_dict(orient="records"),
        key=lambda r: str(r.get("bid", "")),
    )
    assert len(rows) == 2
    assert rows[0] == {"bid": "b", "has_t": True}
    assert rows[1] == {"bid": "c", "has_t": False}


# ---------------------------------------------------------------------------
# Issue #996 amplification: expression breadth + join edge cases
# ---------------------------------------------------------------------------


def _mk_996_rich_graph() -> _CypherTestGraph:
    nodes = pd.DataFrame({
        "id": ["p1", "p2", "p3", "p4"],
        "score": [10, 20, 30, 40],
        "label__Person": [True, True, True, True],
    })
    edges = pd.DataFrame({
        "s": ["p1", "p1", "p2", "p1", "p4"],
        "d": ["p2", "p3", "p3", "p4", "p1"],
        "type": ["FRIEND", "FRIEND", "KNOWS", "FRIEND", "KNOWS"],
        "weight": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    return _mk_graph(nodes, edges)


def _assert_996_rich_rows(query: str, expected_rows: list[dict[str, object]], *, sort_key: str | None = None) -> None:
    g = _mk_996_rich_graph()
    result = g.gfql(
        "MATCH (a)-[r1:FRIEND]->(b) "
        "OPTIONAL MATCH (b)-[r2:KNOWS]->(c) "
        f"{query}"
    )
    frame = result._nodes.astype(object)
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records")
    if sort_key is not None:
        rows = sorted(rows, key=lambda row: str(row[sort_key]))
    assert rows == expected_rows


@pytest.mark.parametrize(
    "query,expected_rows,sort_key",
    [
        pytest.param(
            "RETURN a.id AS aid, b.id AS bid, type(r2) AS t",
            [
                {"aid": "p1", "bid": "p2", "t": "KNOWS"},
                {"aid": "p1", "bid": "p3", "t": None},
                {"aid": "p1", "bid": "p4", "t": "KNOWS"},
            ],
            "bid",
            id="type_function_on_optional_edge",
        ),
        pytest.param(
            "RETURN b.id AS bid, coalesce(c.id, 'none') AS target",
            [{"bid": "p2", "target": "p3"}, {"bid": "p3", "target": "none"}, {"bid": "p4", "target": "p1"}],
            "bid",
            id="coalesce_on_optional_property",
        ),
        pytest.param(
            "RETURN b.id AS bid, b.score + 1 AS bumped",
            [{"bid": "p2", "bumped": 21}, {"bid": "p3", "bumped": 31}, {"bid": "p4", "bumped": 41}],
            "bid",
            id="arithmetic_in_return",
        ),
        pytest.param(
            "RETURN b.id AS bid, c.id AS cid ORDER BY bid DESC",
            [
                {"bid": "p4", "cid": "p1"},
                {"bid": "p3", "cid": None},
                {"bid": "p2", "cid": "p3"},
            ],
            None,
            id="order_by_desc",
        ),
        pytest.param(
            "RETURN b.id AS bid ORDER BY bid LIMIT 2",
            [{"bid": "p2"}, {"bid": "p3"}],
            None,
            id="limit",
        ),
        pytest.param(
            "RETURN b.id AS bid ORDER BY bid SKIP 1 LIMIT 1",
            [{"bid": "p3"}],
            None,
            id="skip_limit",
        ),
    ],
)
def test_issue_996_rich_optional_match_rows(
    query: str,
    expected_rows: list[dict[str, object]],
    sort_key: str | None,
) -> None:
    _assert_996_rich_rows(query, expected_rows, sort_key=sort_key)


def test_issue_996_distinct() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "a", "b"],
        "d": ["b", "c", "c"],
        "type": ["R1", "R1", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (y)-[r2:T]->(z) "
        "RETURN DISTINCT z.id AS zid"
    )

    rows = result._nodes.to_dict(orient="records")
    non_null_zids = [r["zid"] for r in rows if r["zid"] is not None and not (isinstance(r["zid"], float) and r["zid"] != r["zid"])]
    null_count = len(rows) - len(non_null_zids)
    assert sorted(non_null_zids) == ["c"]
    assert null_count == 1


def test_issue_996_no_optional_matches() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R1"]})
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (y)-[r2:NONEXISTENT]->(z) "
        "RETURN x.id AS xid, y.id AS yid, z.id AS zid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0] == {"xid": "a", "yid": "b", "zid": None}


def test_issue_996_all_rows_match_optional() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "b"],
        "d": ["b", "c"],
        "type": ["R1", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (y)-[r2:T]->(z) "
        "RETURN y.id AS yid, z.id AS zid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert rows == [{"yid": "b", "zid": "c"}]


def test_issue_996_multi_row_optional_match_per_base() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    edges = pd.DataFrame({
        "s": ["a", "b", "b"],
        "d": ["b", "c", "d"],
        "type": ["R1", "T", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (y)-[r2:T]->(z) "
        "RETURN y.id AS yid, z.id AS zid "
        "ORDER BY zid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 2
    assert rows[0] == {"yid": "b", "zid": "c"}
    assert rows[1] == {"yid": "b", "zid": "d"}


def test_issue_996_property_access_on_null_optional_node() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"], "name": ["alice", "bob"]})
    edges = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R1"]})
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (y)-[r2:NOPE]->(z) "
        "RETURN y.id AS yid, z.name AS zname"
    )

    rows = result._nodes.to_dict(orient="records")
    assert rows == [{"yid": "b", "zname": None}]


def test_issue_996_is_not_null_on_optional_edge() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "a", "b"],
        "d": ["b", "c", "c"],
        "type": ["R1", "R1", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (y)-[r2:T]->(z) "
        "RETURN y.id AS yid, "
        "CASE WHEN r2 IS NOT NULL THEN true ELSE false END AS has_edge"
    )

    rows = sorted(result._nodes.to_dict(orient="records"), key=lambda r: str(r["yid"]))
    assert rows[0] == {"yid": "b", "has_edge": True}
    assert rows[1] == {"yid": "c", "has_edge": False}


def test_issue_996_empty_base_match() -> None:
    nodes = pd.DataFrame({"id": ["a"]})
    edges = pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object"), "type": pd.Series(dtype="object")})
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (y)-[r2:T]->(z) "
        "RETURN x.id AS xid"
    )

    assert result._nodes.to_dict(orient="records") == []


def test_issue_996_two_shared_node_aliases() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "b", "a"],
        "d": ["b", "c", "c"],
        "type": ["R1", "R1", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a)-[r1:R1]->(b) "
        "OPTIONAL MATCH (a)-[r2:T]->(b) "
        "RETURN a.id AS aid, b.id AS bid, "
        "CASE WHEN r2 IS NULL THEN false ELSE true END AS has_t"
    )

    rows = sorted(
        result._nodes.to_dict(orient="records"),
        key=lambda r: (str(r["aid"]), str(r["bid"])),
    )
    assert len(rows) == 2
    assert rows[0] == {"aid": "a", "bid": "b", "has_t": False}
    assert rows[1] == {"aid": "b", "bid": "c", "has_t": False}


def test_issue_996_integer_node_ids() -> None:
    nodes = pd.DataFrame({"id": [1, 2, 3]})
    edges = pd.DataFrame({
        "s": [1, 2],
        "d": [2, 3],
        "type": ["R1", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a)-[r1:R1]->(b) "
        "OPTIONAL MATCH (b)-[r2:T]->(c) "
        "RETURN a.id AS aid, b.id AS bid, c.id AS cid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["aid"] == 1
    assert rows[0]["bid"] == 2
    assert rows[0]["cid"] == 3


def test_issue_996_custom_node_column_name() -> None:
    nodes = pd.DataFrame({"nid": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "b"],
        "d": ["b", "c"],
        "type": ["R1", "T"],
    })
    g = cast(_CypherTestGraph, _CypherTestGraph().nodes(nodes, "nid").edges(edges, "s", "d"))

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (y)-[r2:T]->(z) "
        "RETURN x.nid AS xid, y.nid AS yid, z.nid AS zid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["xid"] == "a"
    assert rows[0]["yid"] == "b"
    assert rows[0]["zid"] == "c"


def test_issue_996_longer_optional_chain() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    edges = pd.DataFrame({
        "s": ["a", "b", "c"],
        "d": ["b", "c", "d"],
        "type": ["R1", "T", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (y)-[r2:T]->(z)-[r3:T]->(w) "
        "RETURN y.id AS yid, z.id AS zid, w.id AS wid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["yid"] == "b"
    assert rows[0]["zid"] == "c"
    assert rows[0]["wid"] == "d"


# ---------------------------------------------------------------------------
# Issue #1024: WHERE clauses on OPTIONAL MATCH results
# ---------------------------------------------------------------------------


def test_issue_1024_where_label_on_base_match() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c"],
        "label__B": [False, True, False],
    })
    edges = pd.DataFrame({
        "s": ["a", "a"],
        "d": ["b", "c"],
        "type": ["R1", "R1"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "WHERE y:B "
        "OPTIONAL MATCH (y)-[r2:R1]->(z) "
        "RETURN x.id AS xid, y.id AS yid, z.id AS zid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["xid"] == "a"
    assert rows[0]["yid"] == "b"


def test_issue_1024_where_label_on_optional_match() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "label__C": [False, False, True, False],
    })
    edges = pd.DataFrame({
        "s": ["a", "b", "b"],
        "d": ["b", "c", "d"],
        "type": ["R1", "T", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (y)-[r2:T]->(z) "
        "WHERE z:C "
        "RETURN y.id AS yid, z.id AS zid"
    )

    rows = sorted(result._nodes.to_dict(orient="records"), key=lambda r: str(r["yid"]))
    assert len(rows) == 1
    assert rows[0]["yid"] == "b"
    assert rows[0]["zid"] == "c"


def test_issue_1024_where_not_pattern_on_optional_match_filters_candidates() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d", "e"]})
    edges = pd.DataFrame({
        "s": ["a", "b", "b", "c"],
        "d": ["b", "c", "d", "e"],
        "type": ["R1", "T", "T", "U"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[:R1]->(y) "
        "OPTIONAL MATCH (y)-[:T]->(z) "
        "WHERE NOT (z)-[:U]->() "
        "RETURN y.id AS yid, z.id AS zid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert rows == [{"yid": "b", "zid": "d"}]


def test_issue_1024_where_not_pattern_on_optional_match_null_fills_when_all_filtered() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "e"]})
    edges = pd.DataFrame({
        "s": ["a", "b", "c"],
        "d": ["b", "c", "e"],
        "type": ["R1", "T", "U"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[:R1]->(y) "
        "OPTIONAL MATCH (y)-[:T]->(z) "
        "WHERE NOT (z)-[:U]->() "
        "RETURN y.id AS yid, z.id AS zid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert rows == [{"yid": "b", "zid": None}]


def test_issue_1024_where_on_both_match_and_optional() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "label__B": [False, True, False, False],
        "label__C": [False, False, True, False],
        "label__D": [False, False, False, True],
        "name": ["A", "B", "C", "D"],
    })
    edges = pd.DataFrame({
        "s": ["a", "a", "a"],
        "d": ["b", "c", "d"],
        "type": ["T", "T", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a)-->(b) "
        "WHERE b:B "
        "OPTIONAL MATCH (a)-->(c) "
        "WHERE c:C "
        "RETURN a.name AS aname"
    )

    rows = result._nodes.to_dict(orient="records")
    assert rows == [{"aname": "A"}]


def test_issue_1024_where_property_on_optional_null_safe() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b"],
        "name": ["Alice", "Bob"],
    })
    edges = pd.DataFrame({
        "s": ["a"],
        "d": ["b"],
        "type": ["R1"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (n)-->(x0) "
        "OPTIONAL MATCH (x0)-->(x1) "
        "WHERE x1.name = 'bar' "
        "RETURN x0.name AS name"
    )

    rows = result._nodes.to_dict(orient="records")
    assert rows == [{"name": "Bob"}]


def test_issue_1024_where_base_eliminates_all_rows() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b"],
        "label__X": [False, False],
    })
    edges = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R1"]})
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "WHERE y:X "
        "OPTIONAL MATCH (y)-[r2:R1]->(z) "
        "RETURN x.id AS xid"
    )

    assert result._nodes.to_dict(orient="records") == []


def test_issue_1024_where_optional_no_matches() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c"],
        "label__Z": [False, False, False],
    })
    edges = pd.DataFrame({
        "s": ["a", "b"],
        "d": ["b", "c"],
        "type": ["R1", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (y)-[r2:T]->(z) "
        "WHERE z:Z "
        "RETURN y.id AS yid, z.id AS zid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["yid"] == "b"


def test_issue_1024_where_combined_with_case_and_order_by() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "label__Target": [False, True, True, False],
    })
    edges = pd.DataFrame({
        "s": ["a", "a", "b", "c"],
        "d": ["b", "c", "d", "d"],
        "type": ["R1", "R1", "T", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "WHERE y:Target "
        "OPTIONAL MATCH (y)-[r2:T]->(z) "
        "RETURN y.id AS yid, "
        "CASE WHEN r2 IS NULL THEN false ELSE true END AS has_t "
        "ORDER BY yid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 2
    assert rows[0] == {"yid": "b", "has_t": True}
    assert rows[1] == {"yid": "c", "has_t": True}


def test_issue_1024_where_property_comparison_on_base() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c"],
        "score": [10, 20, 30],
    })
    edges = pd.DataFrame({
        "s": ["a", "a"],
        "d": ["b", "c"],
        "type": ["R1", "R1"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "WHERE y.score > 15 "
        "OPTIONAL MATCH (y)-[r2:R1]->(z) "
        "RETURN y.id AS yid"
    )

    rows = sorted(result._nodes.to_dict(orient="records"), key=lambda r: str(r["yid"]))
    assert len(rows) == 2
    assert rows[0]["yid"] == "b"
    assert rows[1]["yid"] == "c"


# ---------------------------------------------------------------------------
# Issue #1025: Multiple OPTIONAL MATCH clauses
# ---------------------------------------------------------------------------


def test_issue_1025_two_optional_match_clauses() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    edges = pd.DataFrame({
        "s": ["a", "a", "a"],
        "d": ["b", "c", "d"],
        "type": ["R1", "T1", "T2"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a)-[r:R1]->(b) "
        "OPTIONAL MATCH (a)-[r1:T1]->(c) "
        "OPTIONAL MATCH (a)-[r2:T2]->(d) "
        "RETURN b.id AS bid, c.id AS cid, d.id AS did"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["bid"] == "b"
    assert rows[0]["cid"] == "c"
    assert rows[0]["did"] == "d"


def test_issue_1025_two_optional_match_one_misses() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "a"],
        "d": ["b", "c"],
        "type": ["R1", "T1"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a)-[r:R1]->(b) "
        "OPTIONAL MATCH (a)-[r1:T1]->(c) "
        "OPTIONAL MATCH (a)-[r2:T2]->(d) "
        "RETURN b.id AS bid, c.id AS cid, d.id AS did"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["bid"] == "b"
    assert rows[0]["cid"] == "c"
    assert rows[0]["did"] is None or (isinstance(rows[0]["did"], float) and rows[0]["did"] != rows[0]["did"])


@pytest.mark.parametrize(
    "optional_clauses",
    [
        "OPTIONAL MATCH (m)-[:T1]->(a:A) OPTIONAL MATCH (m)-[:T2]->(b:B)",
        "OPTIONAL MATCH (m)-[:T2]->(b:B) OPTIONAL MATCH (m)-[:T1]->(a:A)",
    ],
)
def test_issue_1472_independent_optional_arms_preserve_per_row_nulls(optional_clauses: str) -> None:
    """Two independent OPTIONAL arms must not bleed matches across base rows."""
    nodes = pd.DataFrame(
        {
            "id": ["m1", "m2", "a1", "b2"],
            "label__M": [True, True, False, False],
            "label__A": [False, False, True, False],
            "label__B": [False, False, False, True],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["m1", "m2"],
            "d": ["a1", "b2"],
            "type": ["T1", "T2"],
        }
    )
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (m:M) "
        f"{optional_clauses} "
        "RETURN m.id AS mid, "
        "CASE a WHEN null THEN 'no-a' ELSE a.id END AS aid, "
        "CASE b WHEN null THEN 'no-b' ELSE b.id END AS bid "
        "ORDER BY mid, aid, bid"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"mid": "m1", "aid": "a1", "bid": "no-b"},
        {"mid": "m2", "aid": "no-a", "bid": "b2"},
    ]


def test_issue_1025_single_node_base_two_optionals() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "a"],
        "d": ["b", "c"],
        "type": ["T1", "T2"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a) "
        "OPTIONAL MATCH (a)-[r1:T1]->(b) "
        "OPTIONAL MATCH (a)-[r2:T2]->(c) "
        "RETURN a.id AS aid, b.id AS bid, c.id AS cid"
    )

    rows = sorted(result._nodes.to_dict(orient="records"), key=lambda r: str(r["aid"]))
    a_rows = [r for r in rows if r["aid"] == "a"]
    assert len(a_rows) == 1
    assert a_rows[0]["bid"] == "b"
    assert a_rows[0]["cid"] == "c"


def test_issue_1025_chained_optionals_with_case() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "a"],
        "d": ["b", "c"],
        "type": ["R1", "T1"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a)-[r:R1]->(b) "
        "OPTIONAL MATCH (a)-[r1:T1]->(c) "
        "OPTIONAL MATCH (a)-[r2:NOPE]->(d) "
        "RETURN b.id AS bid, "
        "CASE WHEN r1 IS NULL THEN false ELSE true END AS has_t1, "
        "CASE WHEN r2 IS NULL THEN false ELSE true END AS has_nope"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["bid"] == "b"
    assert rows[0]["has_t1"] is True
    assert rows[0]["has_nope"] is False


def test_audit_chained_optionals_share_non_base_alias() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    edges = pd.DataFrame({
        "s": ["a", "b", "c"],
        "d": ["b", "c", "d"],
        "type": ["R1", "T1", "T2"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a)-[r:R1]->(b) "
        "OPTIONAL MATCH (b)-[r1:T1]->(c) "
        "OPTIONAL MATCH (c)-[r2:T2]->(d) "
        "RETURN a.id AS aid, b.id AS bid, c.id AS cid, d.id AS did"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["aid"] == "a"
    assert rows[0]["bid"] == "b"
    assert rows[0]["cid"] == "c"
    assert rows[0]["did"] == "d"


def test_audit_chained_optionals_transitive_miss() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "b"],
        "d": ["b", "c"],
        "type": ["R1", "T1"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a)-[r:R1]->(b) "
        "OPTIONAL MATCH (b)-[r1:T1]->(c) "
        "OPTIONAL MATCH (c)-[r2:NOPE]->(d) "
        "RETURN b.id AS bid, c.id AS cid, d.id AS did"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["bid"] == "b"
    assert rows[0]["cid"] == "c"


def test_audit_where_on_three_optionals() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d", "e"],
        "label__X": [True, False, False, False, False],
        "label__Y": [False, True, False, False, False],
        "label__Z": [False, False, True, False, False],
    })
    edges = pd.DataFrame({
        "s": ["a", "a", "a", "a"],
        "d": ["b", "c", "d", "e"],
        "type": ["R1", "R1", "T1", "T2"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r:R1]->(y) "
        "WHERE y:Y "
        "OPTIONAL MATCH (x)-[r1:T1]->(z1) "
        "OPTIONAL MATCH (x)-[r2:T2]->(z2) "
        "RETURN y.id AS yid, z1.id AS z1id, z2.id AS z2id"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["yid"] == "b"
    assert rows[0]["z1id"] == "d"
    assert rows[0]["z2id"] == "e"


def test_audit_single_node_base_three_optionals() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    edges = pd.DataFrame({
        "s": ["a", "a", "a"],
        "d": ["b", "c", "d"],
        "type": ["T1", "T2", "T3"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a) "
        "OPTIONAL MATCH (a)-[r1:T1]->(b) "
        "OPTIONAL MATCH (a)-[r2:T2]->(c) "
        "OPTIONAL MATCH (a)-[r3:T3]->(d) "
        "RETURN a.id AS aid, b.id AS bid, c.id AS cid, d.id AS did"
    )

    a_rows = [r for r in result._nodes.to_dict(orient="records") if r["aid"] == "a"]
    assert len(a_rows) == 1
    assert a_rows[0]["bid"] == "b"
    assert a_rows[0]["cid"] == "c"
    assert a_rows[0]["did"] == "d"


def test_audit_multi_optional_partial_null_fill() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "a"],
        "d": ["b", "c"],
        "type": ["T1", "T3"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a) "
        "OPTIONAL MATCH (a)-[r1:T1]->(x1) "
        "OPTIONAL MATCH (a)-[r2:T2]->(x2) "
        "OPTIONAL MATCH (a)-[r3:T3]->(x3) "
        "RETURN a.id AS aid, x1.id AS x1id, x2.id AS x2id, x3.id AS x3id"
    )

    a_rows = [r for r in result._nodes.to_dict(orient="records") if r["aid"] == "a"]
    assert len(a_rows) == 1
    assert a_rows[0]["x1id"] == "b"
    assert a_rows[0]["x3id"] == "c"


def test_audit_where_property_comparison_filters_optional_results() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c"],
        "score": [10, 20, 5],
    })
    edges = pd.DataFrame({
        "s": ["a", "a"],
        "d": ["b", "c"],
        "type": ["R1", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "OPTIONAL MATCH (x)-[r2:T]->(z) "
        "WHERE z.score > 3 "
        "RETURN y.id AS yid, z.id AS zid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["yid"] == "b"
    assert rows[0]["zid"] == "c"


def test_audit_multi_optional_order_by_limit() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    edges = pd.DataFrame({
        "s": ["a", "a", "b", "c"],
        "d": ["b", "c", "d", "d"],
        "type": ["R1", "R1", "T1", "T2"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r:R1]->(y) "
        "OPTIONAL MATCH (y)-[r1:T1]->(z1) "
        "OPTIONAL MATCH (y)-[r2:T2]->(z2) "
        "RETURN y.id AS yid "
        "ORDER BY yid "
        "LIMIT 1"
    )

    rows = result._nodes.to_dict(orient="records")
    assert rows == [{"yid": "b"}]


def test_audit_single_node_base_where_plus_three_optionals() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c", "d", "e"],
        "label__X": [True, True, False, False, False],
    })
    edges = pd.DataFrame({
        "s": ["a", "a", "a", "b", "b"],
        "d": ["c", "d", "e", "c", "d"],
        "type": ["T1", "T2", "T3", "T1", "T2"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x) "
        "WHERE x:X "
        "OPTIONAL MATCH (x)-[r1:T1]->(y1) "
        "OPTIONAL MATCH (x)-[r2:T2]->(y2) "
        "OPTIONAL MATCH (x)-[r3:T3]->(y3) "
        "RETURN x.id AS xid, y1.id AS y1id, y2.id AS y2id, y3.id AS y3id "
        "ORDER BY xid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 2
    assert rows[0]["xid"] == "a"
    assert rows[0]["y1id"] == "c"
    assert rows[0]["y2id"] == "d"
    assert rows[0]["y3id"] == "e"
    assert rows[1]["xid"] == "b"
    assert rows[1]["y1id"] == "c"
    assert rows[1]["y2id"] == "d"


def test_audit_cross_alias_property_where_on_base() -> None:
    nodes = pd.DataFrame({
        "id": ["a", "b", "c"],
        "score": [100, 50, 200],
    })
    edges = pd.DataFrame({
        "s": ["a", "a"],
        "d": ["b", "c"],
        "type": ["R1", "R1"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x)-[r1:R1]->(y) "
        "WHERE x.score > y.score "
        "OPTIONAL MATCH (y)-[r2:R1]->(z) "
        "RETURN x.id AS xid, y.id AS yid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0] == {"xid": "a", "yid": "b"}


# ---------------------------------------------------------------------------
# Issue #1026: OPTIONAL MATCH with WITH/UNWIND stages
# ---------------------------------------------------------------------------


def test_issue_1026_with_optional_match_null_fill() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["T"]})
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x) WITH x "
        "OPTIONAL MATCH (x)-->(y) "
        "RETURN x.id AS xid, y.id AS yid "
        "ORDER BY xid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 3
    matched = [r for r in rows if r["yid"] == "b"]
    assert len(matched) == 1
    assert matched[0]["xid"] == "a"


def test_issue_1026_with_limit_optional_match_null_fill() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object"), "type": pd.Series(dtype="object")})
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x) WITH x LIMIT 2 "
        "OPTIONAL MATCH (x)-->(y) "
        "RETURN x.id AS xid, y.id AS yid "
        "ORDER BY xid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 2


def test_issue_1026_multi_alias_with_optional_match_carries_secondary_property() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "num": [1, 2, 3]})
    edges = pd.DataFrame({
        "s": ["a", "b"],
        "d": ["b", "c"],
        "type": ["R", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a)-[r:R]->(b) "
        "WITH a, b "
        "OPTIONAL MATCH (b)-[r2:T]->(c) "
        "RETURN a.id AS aid, b.id AS bid, c.id AS cid"
    )
    assert result._nodes.to_dict(orient="records") == [{"aid": "a", "bid": "b", "cid": "c"}]


def test_issue_1026_with_limit_optional_match_all_match() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "a"], "type": ["T", "T"]})
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x) WITH x LIMIT 2 "
        "OPTIONAL MATCH (x)-->(y) "
        "RETURN x.id AS xid, y.id AS yid "
        "ORDER BY xid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 2
    assert rows[0] == {"xid": "a", "yid": "b"}
    assert rows[1] == {"xid": "b", "yid": "a"}


def test_issue_1026_with_optional_match_property_on_optional() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["T"]})
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x) WITH x "
        "OPTIONAL MATCH (x)-[:T]->(y) "
        "RETURN y.id AS yid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 3
    non_null = [r for r in rows if r["yid"] is not None and not (isinstance(r["yid"], float) and r["yid"] != r["yid"])]
    assert len(non_null) == 1
    assert non_null[0]["yid"] == "b"


def test_issue_1026_with_optional_match_empty_graph() -> None:
    nodes = pd.DataFrame({"id": pd.Series(dtype="object")})
    edges = pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object"), "type": pd.Series(dtype="object")})
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x) WITH x "
        "OPTIONAL MATCH (x)-->(y) "
        "RETURN x.id AS xid"
    )

    assert result._nodes.to_dict(orient="records") == []


def test_issue_1026_with_optional_match_multi_row_optional() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    edges = pd.DataFrame({
        "s": ["a", "a", "b"],
        "d": ["b", "c", "d"],
        "type": ["T", "T", "T"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (x) WITH x "
        "OPTIONAL MATCH (x)-[:T]->(y) "
        "RETURN x.id AS xid, y.id AS yid "
        "ORDER BY xid, yid"
    )

    rows = result._nodes.to_dict(orient="records")
    matched = [r for r in rows if r["yid"] is not None and not (isinstance(r["yid"], float) and r["yid"] != r["yid"])]
    assert len(matched) == 3
    assert {"xid": "a", "yid": "b"} in matched
    assert {"xid": "a", "yid": "c"} in matched


# ---------------------------------------------------------------------------
# Issue #996: connected MATCH + OPTIONAL MATCH + CASE (IS7 shape)
# ---------------------------------------------------------------------------

def test_issue_996_connected_match_optional_match_case_edge_alias() -> None:
    """
    IS7 shape: MATCH (m)<-[:R]-(c)-[:H]->(p)
               OPTIONAL MATCH (m)-[:H]->(a)-[r:K]-(p)
               RETURN ..., CASE r WHEN null THEN false ELSE true END AS knows

    Left-join semantics: all (m,c,p) rows preserved; r=null when OPTIONAL arm misses.
    """
    nodes = pd.DataFrame({
        "id": ["m", "c", "c2", "p", "p2", "a"],
        "label__Message": [True, False, False, False, False, False],
        "label__Comment": [False, True, True, False, False, False],
        "label__Person":  [False, False, False, True, True, True],
    })
    edges = pd.DataFrame({
        "s":    ["c",  "c",            "c2", "c2",           "m",           "a"],
        "d":    ["m",  "p",            "m",  "p2",           "a",           "p"],
        "type": ["REPLY_OF", "HAS_CREATOR", "REPLY_OF", "HAS_CREATOR", "HAS_CREATOR", "KNOWS"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (m:Message)<-[:REPLY_OF]-(c:Comment)-[:HAS_CREATOR]->(p:Person) "
        "OPTIONAL MATCH (m)-[:HAS_CREATOR]->(a:Person)-[r:KNOWS]-(p) "
        "RETURN c.id AS commentId, p.id AS replyAuthorId, "
        "CASE r WHEN null THEN false ELSE true END AS knows "
        "ORDER BY commentId"
    )

    rows = result._nodes[["commentId", "replyAuthorId", "knows"]].to_dict(orient="records")
    assert len(rows) == 2
    assert {"commentId": "c", "replyAuthorId": "p", "knows": True} in rows
    assert {"commentId": "c2", "replyAuthorId": "p2", "knows": False} in rows


def test_issue_996_connected_match_optional_match_all_rows_match() -> None:
    """All base rows have a matching OPTIONAL arm — no nulls expected."""
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "label__N": [True, True, True]})
    edges = pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "b"], "type": ["T", "T", "T"]})
    g = _mk_graph(nodes, edges)
    result = g.gfql(
        "MATCH (x:N)-[:T]->(y:N) "
        "OPTIONAL MATCH (x)-[:T]->(z:N) "
        "RETURN x.id AS xid, y.id AS yid, z.id AS zid ORDER BY xid, yid"
    )
    rows = result._nodes[["xid", "yid", "zid"]].to_dict(orient="records")
    assert all(r["zid"] is not None for r in rows)


def test_issue_996_connected_match_optional_match_no_rows_match() -> None:
    """No base rows have a matching OPTIONAL arm — all optional cols null."""
    nodes = pd.DataFrame({"id": ["a", "b"], "label__N": [True, True]})
    edges = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["T"]})
    g = _mk_graph(nodes, edges)
    result = g.gfql(
        "MATCH (x:N)-[:T]->(y:N) "
        "OPTIONAL MATCH (x)-[:MISSING]->(z:N) "
        "RETURN x.id AS xid, CASE z WHEN null THEN 'none' ELSE 'found' END AS found"
    )
    rows = result._nodes[["xid", "found"]].to_dict(orient="records")
    assert rows == [{"xid": "a", "found": "none"}]


def test_issue_996_connected_match_optional_match_node_alias_null_in_return() -> None:
    """Node alias (not edge) from OPTIONAL arm is null when arm misses."""
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "label__N": [True, True, True]})
    edges2 = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["T", "T"]})
    g2 = _mk_graph(nodes, edges2)
    result = g2.gfql(
        "MATCH (x:N)-[:T]->(y:N) "
        "OPTIONAL MATCH (y)-[:T]->(z:N) "
        "RETURN x.id AS xid, y.id AS yid, z.id AS zid ORDER BY xid, yid"
    )
    rows = result._nodes[["xid", "yid", "zid"]].to_dict(orient="records")
    assert len(rows) == 2
    by_x = {r["xid"]: r for r in rows}
    assert by_x["a"]["yid"] == "b" and by_x["a"]["zid"] == "c"
    assert by_x["b"]["yid"] == "c" and (by_x["b"]["zid"] is None or (isinstance(by_x["b"]["zid"], float) and by_x["b"]["zid"] != by_x["b"]["zid"]))


def test_issue_996_connected_match_optional_match_order_by_optional_col() -> None:
    """ORDER BY on an optional-arm column (may be null) must not crash."""
    nodes = pd.DataFrame({
        "id": ["m", "c1", "c2", "p1", "p2", "a"],
        "label__Message": [True, False, False, False, False, False],
        "label__Comment": [False, True, True, False, False, False],
        "label__Person":  [False, False, False, True, True, True],
    })
    edges = pd.DataFrame({
        "s":    ["c1", "c1",  "c2", "c2",  "m",  "a"],
        "d":    ["m",  "p1",  "m",  "p2",  "a",  "p1"],
        "type": ["REPLY_OF", "HAS_CREATOR", "REPLY_OF", "HAS_CREATOR", "HAS_CREATOR", "KNOWS"],
    })
    g = _mk_graph(nodes, edges)
    result = g.gfql(
        "MATCH (m:Message)<-[:REPLY_OF]-(c:Comment)-[:HAS_CREATOR]->(p:Person) "
        "OPTIONAL MATCH (m)-[:HAS_CREATOR]->(a:Person)-[r:KNOWS]-(p) "
        "RETURN c.id AS cid, CASE r WHEN null THEN false ELSE true END AS knows "
        "ORDER BY cid"
    )
    rows = result._nodes[["cid", "knows"]].to_dict(orient="records")
    assert len(rows) == 2
    assert rows[0]["cid"] == "c1"
    assert rows[1]["cid"] == "c2"
    assert rows[0]["knows"] is True   # c1→p1, a KNOWS p1
    assert rows[1]["knows"] is False  # c2→p2, a does NOT know p2


def test_issue_996_case_r_when_null_searched_form_still_works() -> None:
    """Regression: searched CASE WHEN r IS NULL form not broken by fix."""
    nodes = pd.DataFrame({
        "id": ["m", "c", "p", "a"],
        "label__Message": [True, False, False, False],
        "label__Comment": [False, True, False, False],
        "label__Person":  [False, False, True, True],
    })
    edges = pd.DataFrame({
        "s": ["c", "c", "m"], "d": ["m", "p", "a"], "type": ["REPLY_OF", "HAS_CREATOR", "HAS_CREATOR"]
    })
    g = _mk_graph(nodes, edges)
    # No KNOWS edge → r is null → CASE WHEN r IS NULL THEN false
    result = g.gfql(
        "MATCH (m:Message)<-[:REPLY_OF]-(c:Comment)-[:HAS_CREATOR]->(p:Person) "
        "OPTIONAL MATCH (m)-[:HAS_CREATOR]->(a:Person)-[r:KNOWS]-(p) "
        "RETURN CASE WHEN r IS NULL THEN false ELSE true END AS knows"
    )
    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["knows"] is False


# ---------------------------------------------------------------------------
# Issue #1395: sequential MATCH reply-author row-shaping joins (IC8 / IS7)
# ---------------------------------------------------------------------------

def _mk_issue_1395_reply_author_ic8_graph(*, cudf_mode: bool = False) -> _CypherTestGraph:
    nodes = pd.DataFrame({
        "id": [
            "viewer",
            "m1",
            "m2",
            "c1",
            "c2",
            "c3",
            "author1",
            "author2",
        ],
        "label__Person": [True, False, False, False, False, False, True, True],
        "label__Message": [False, True, True, False, False, False, False, False],
        "label__Comment": [False, False, False, True, True, True, False, False],
        "firstName": [None, None, None, None, None, None, "Ann", "Bob"],
        "lastName": [None, None, None, None, None, None, "One", "Two"],
        "creationDate": [None, 100, 90, 110, 105, 80, None, None],
        "content": [None, "post-1", "post-2", "reply-1", "reply-2", "nested-reply", None, None],
    })
    edges = pd.DataFrame({
        "s": ["m1", "m2", "c1", "c1", "c2", "c2", "c3", "c3"],
        "d": ["viewer", "viewer", "m1", "author1", "m2", "author2", "c1", "author2"],
        "type": [
            "HAS_CREATOR",
            "HAS_CREATOR",
            "REPLY_OF",
            "HAS_CREATOR",
            "REPLY_OF",
            "HAS_CREATOR",
            "REPLY_OF",
            "HAS_CREATOR",
        ],
    })
    if cudf_mode:
        pytest.importorskip("cudf")
        import cudf  # type: ignore
        nodes = cudf.DataFrame.from_pandas(nodes)
        edges = cudf.DataFrame.from_pandas(edges)
    return _mk_graph(nodes, edges)


def test_issue_1395_sequential_match_recent_replies_row_shaping_ic8() -> None:
    """IC8 shape: two non-optional MATCH clauses preserve reply-author projection fields."""
    g = _mk_issue_1395_reply_author_ic8_graph()

    query = (
        "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(message:Message) "
        "MATCH (message)<-[:REPLY_OF]-(comment:Comment)-[:HAS_CREATOR]->(commentAuthor:Person) "
        "RETURN "
        "commentAuthor.id AS commentAuthorId, "
        "commentAuthor.firstName AS commentAuthorFirstName, "
        "commentAuthor.lastName AS commentAuthorLastName, "
        "comment.creationDate AS commentCreationDate, "
        "comment.id AS commentId, "
        "comment.content AS commentContent "
        "ORDER BY commentCreationDate DESC, commentId ASC "
        "LIMIT 20"
    )
    result = g.gfql(query, params={"personId": "viewer"})

    assert result._nodes.to_dict(orient="records") == [
        {
            "commentAuthorId": "author1",
            "commentAuthorFirstName": "Ann",
            "commentAuthorLastName": "One",
            "commentCreationDate": 110.0,
            "commentId": "c1",
            "commentContent": "reply-1",
        },
        {
            "commentAuthorId": "author2",
            "commentAuthorFirstName": "Bob",
            "commentAuthorLastName": "Two",
            "commentCreationDate": 105.0,
            "commentId": "c2",
            "commentContent": "reply-2",
        },
    ]


def test_issue_1395_sequential_match_recent_replies_row_shaping_ic8_on_cudf() -> None:
    """cuDF parity: IC8 sequential MATCH row-shaping stays aligned on GPU engine."""
    g = _mk_issue_1395_reply_author_ic8_graph(cudf_mode=True)

    query = (
        "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(message:Message) "
        "MATCH (message)<-[:REPLY_OF]-(comment:Comment)-[:HAS_CREATOR]->(commentAuthor:Person) "
        "RETURN "
        "commentAuthor.id AS commentAuthorId, "
        "commentAuthor.firstName AS commentAuthorFirstName, "
        "commentAuthor.lastName AS commentAuthorLastName, "
        "comment.creationDate AS commentCreationDate, "
        "comment.id AS commentId, "
        "comment.content AS commentContent "
        "ORDER BY commentCreationDate DESC, commentId ASC "
        "LIMIT 20"
    )
    result = g.gfql(query, params={"personId": "viewer"}, engine="cudf")

    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [
        {
            "commentAuthorId": "author1",
            "commentAuthorFirstName": "Ann",
            "commentAuthorLastName": "One",
            "commentCreationDate": 110.0,
            "commentId": "c1",
            "commentContent": "reply-1",
        },
        {
            "commentAuthorId": "author2",
            "commentAuthorFirstName": "Bob",
            "commentAuthorLastName": "Two",
            "commentCreationDate": 105.0,
            "commentId": "c2",
            "commentContent": "reply-2",
        },
    ]


def test_issue_1395_sequential_match_where_boundary_lock_ic8() -> None:
    """Boundary lock: intermediate WHERE stays explicitly unsupported for sequential MATCH merge."""
    g = _mk_issue_1395_reply_author_ic8_graph()

    query = (
        "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(message:Message) "
        "WHERE message.creationDate >= 95 "
        "MATCH (message)<-[:REPLY_OF]-(comment:Comment)-[:HAS_CREATOR]->(commentAuthor:Person) "
        "RETURN comment.id AS commentId, commentAuthor.id AS commentAuthorId "
        "ORDER BY commentId"
    )
    with pytest.raises(
        GFQLValidationError,
        match="WHERE on intermediate MATCH clauses is not yet supported for sequential MATCH merge",
    ):
        _ = g.gfql(query, params={"personId": "viewer"})


def test_issue_1395_sequential_match_equivalent_to_single_match_comma_pattern() -> None:
    """Shape equivalence: sequential MATCH and single-MATCH comma pattern return identical rows."""
    g = _mk_issue_1395_reply_author_ic8_graph()

    query_sequential = (
        "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(message:Message) "
        "MATCH (message)<-[:REPLY_OF]-(comment:Comment)-[:HAS_CREATOR]->(commentAuthor:Person) "
        "RETURN comment.id AS commentId, commentAuthor.id AS commentAuthorId "
        "ORDER BY commentId"
    )
    query_single_match = (
        "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(message:Message), "
        "(message)<-[:REPLY_OF]-(comment:Comment)-[:HAS_CREATOR]->(commentAuthor:Person) "
        "RETURN comment.id AS commentId, commentAuthor.id AS commentAuthorId "
        "ORDER BY commentId"
    )

    seq_rows = g.gfql(query_sequential, params={"personId": "viewer"})._nodes.to_dict(orient="records")
    comma_rows = g.gfql(query_single_match, params={"personId": "viewer"})._nodes.to_dict(orient="records")
    assert seq_rows == comma_rows


def test_issue_1395_sequential_match_message_replies_row_shaping_is7() -> None:
    """IS7 shape: sequential MATCH keeps comment + replyAuthor + messageAuthor rows aligned."""
    nodes = pd.DataFrame({
        "id": ["m1", "message_author", "reply_author", "c1", "c2"],
        "label__Message": [True, False, False, False, False],
        "label__Person": [False, True, True, False, False],
        "label__Comment": [False, False, False, True, True],
        "firstName": [None, "Main", "Peer", None, None],
        "lastName": [None, "Author", "One", None, None],
        "creationDate": [None, None, None, 20, 10],
        "content": [None, None, None, "reply-from-peer", "reply-from-main"],
    })
    edges = pd.DataFrame({
        "s": ["m1", "c1", "c1", "c2", "c2"],
        "d": ["message_author", "m1", "reply_author", "m1", "message_author"],
        "type": ["HAS_CREATOR", "REPLY_OF", "HAS_CREATOR", "REPLY_OF", "HAS_CREATOR"],
    })
    g = _mk_graph(nodes, edges)

    query = (
        "MATCH (message:Message {id: $messageId})<-[:REPLY_OF]-(comment:Comment)-[:HAS_CREATOR]->(replyAuthor:Person) "
        "MATCH (message)-[:HAS_CREATOR]->(messageAuthor:Person) "
        "RETURN "
        "comment.id AS commentId, "
        "comment.content AS commentContent, "
        "comment.creationDate AS commentCreationDate, "
        "replyAuthor.id AS replyAuthorId, "
        "replyAuthor.firstName AS replyAuthorFirstName, "
        "replyAuthor.lastName AS replyAuthorLastName, "
        "messageAuthor.id AS messageAuthorId "
        "ORDER BY commentCreationDate DESC, replyAuthorId ASC"
    )
    result = g.gfql(query, params={"messageId": "m1"})

    assert result._nodes.to_dict(orient="records") == [
        {
            "commentId": "c1",
            "commentContent": "reply-from-peer",
            "commentCreationDate": 20.0,
            "replyAuthorId": "reply_author",
            "replyAuthorFirstName": "Peer",
            "replyAuthorLastName": "One",
            "messageAuthorId": "message_author",
        },
        {
            "commentId": "c2",
            "commentContent": "reply-from-main",
            "commentCreationDate": 10.0,
            "replyAuthorId": "message_author",
            "replyAuthorFirstName": "Main",
            "replyAuthorLastName": "Author",
            "messageAuthorId": "message_author",
        },
    ]


# ── Issue #1052: OPTIONAL MATCH semi-join — opt arm must be pre-filtered ──────


def test_issue_1488_optional_match_seeds_shared_first_alias_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IS7 OOM guard: bound OPTIONAL arm traversal before rows(binding_ops)."""
    n = 20
    msg_ids = [f"m{i}" for i in range(1, n + 1)]
    nodes = pd.DataFrame({
        "id": msg_ids + ["c1", "p1"] + [f"a{i}" for i in range(1, n + 1)] + [f"p{i}" for i in range(2, n + 1)],
        "label__M": [True] * n + [False] * (2 * n + 1),
        "label__C": [False] * n + [True] + [False] * (2 * n),
        "label__P": [False] * (n + 1) + [True] + [False] * n + [True] * (n - 1),
        "label__A": [False] * (n + 2) + [True] * n + [False] * (n - 1),
    })
    edge_rows = [
        {"s": "c1", "d": "m1", "type": "REPLY_OF"},
        {"s": "c1", "d": "p1", "type": "HAS_CREATOR"},
        {"s": "m1", "d": "a1", "type": "HAS_CREATOR"},
        {"s": "a1", "d": "p1", "type": "KNOWS"},
    ]
    for i in range(2, n + 1):
        edge_rows.extend([
            {"s": f"m{i}", "d": f"a{i}", "type": "HAS_CREATOR"},
            {"s": f"a{i}", "d": f"p{i}", "type": "KNOWS"},
        ])
    edges = pd.DataFrame(edge_rows)
    g = _mk_graph(nodes, edges)

    original_execute = ASTEdge.execute
    optional_start_sizes: List[int] = []

    def spy_execute(
        self: ASTEdge,
        g: Any,
        prev_node_wavefront: Any,
        target_wave_front: Any,
        engine: Any,
    ) -> Any:
        if (
            self.edge_match == {"type": "HAS_CREATOR"}
            and prev_node_wavefront is not None
            and "id" in prev_node_wavefront.columns
        ):
            prev_ids = [str(value) for value in prev_node_wavefront["id"].tolist()]
            if prev_ids and all(value.startswith("m") for value in prev_ids):
                optional_start_sizes.append(len(prev_ids))
        return original_execute(self, g, prev_node_wavefront, target_wave_front, engine)

    monkeypatch.setattr(ASTEdge, "execute", spy_execute)

    result = g.gfql(
        "MATCH (m:M {id: $mid})<-[:REPLY_OF]-(c:C)-[:HAS_CREATOR]->(p:P) "
        "OPTIONAL MATCH (m)-[:HAS_CREATOR]->(a:A)-[r:KNOWS]-(p) "
        "RETURN c.id AS cid, CASE r WHEN null THEN false ELSE true END AS knows",
        params={"mid": "m1"},
    )

    assert result._nodes[["cid", "knows"]].to_dict(orient="records") == [{"cid": "c1", "knows": True}]
    assert optional_start_sizes
    assert max(optional_start_sizes) == 1


def test_issue_1052_optional_match_semijoin_filters_opt_arm() -> None:
    """OPTIONAL MATCH opt arm must only join rows whose keys appear in base result."""
    nodes = pd.DataFrame({
        "id": ["m1", "m2", "c1", "c2", "p1", "p2", "a1", "a2"],
        "label__Message": [True, True, False, False, False, False, False, False],
        "label__Comment": [False, False, True, True, False, False, False, False],
        "label__Person":  [False, False, False, False, True, True, True, True],
    })
    edges = pd.DataFrame({
        "s":    ["c1",       "c1",            "c2",       "c2",            "m1",            "a1",    "m2",            "a2"],
        "d":    ["m1",       "p1",            "m2",       "p2",            "a1",            "p1",    "a2",            "p2"],
        "type": ["REPLY_OF", "HAS_CREATOR",   "REPLY_OF", "HAS_CREATOR",   "HAS_CREATOR",   "KNOWS", "HAS_CREATOR",   "KNOWS"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (m:Message {id: $mid})<-[:REPLY_OF]-(c:Comment)-[:HAS_CREATOR]->(p:Person) "
        "OPTIONAL MATCH (m)-[:HAS_CREATOR]->(a:Person)-[r:KNOWS]-(p) "
        "RETURN c.id AS cid, CASE r WHEN null THEN false ELSE true END AS knows "
        "ORDER BY cid",
        params={"mid": "m1"},
    )
    rows = result._nodes[["cid", "knows"]].to_dict(orient="records")
    assert rows == [{"cid": "c1", "knows": True}]


def test_issue_1052_semijoin_two_shared_aliases_multi_col_path() -> None:
    """Multi-column semi-join (2 shared node aliases) preserves correct rows.

    MATCH (a)-[r1:R1]->(b)-[r2:R2]->(c)
    OPTIONAL MATCH (a)-[r3:R3]->(b)-[r4:R4]->(d)
    """
    nodes = pd.DataFrame({
        "id":        ["a",  "b",  "c",  "d",  "b2", "d2"],
        "label__N":  [True, True, True, True, True,  True],
    })
    edges = pd.DataFrame({
        "s":    ["a",  "b",  "a",  "b",  "a",  "b2"],
        "d":    ["b",  "c",  "b",  "d",  "b2", "d2"],
        "type": ["R1", "R2", "R3", "R4", "R3", "R4"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (a:N)-[:R1]->(b:N)-[:R2]->(c:N) "
        "OPTIONAL MATCH (a)-[:R3]->(b)-[:R4]->(d:N) "
        "RETURN a.id AS aid, b.id AS bid, c.id AS cid, d.id AS did"
    )
    rows = result._nodes[["aid", "bid", "cid", "did"]].to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["aid"] == "a"
    assert rows[0]["bid"] == "b"
    assert rows[0]["cid"] == "c"
    assert rows[0]["did"] == "d"
    cols = list(result._nodes.columns)
    assert len(cols) == len(set(cols)), f"Duplicate columns: {cols}"


def test_issue_1052_semijoin_multi_arm_second_arm_uses_updated_joined() -> None:
    """With 2 OPTIONAL MATCH arms, arm-2 semi-join uses joined updated by arm-1."""
    nodes = pd.DataFrame({
        "id": ["m", "c", "a", "p"],
        "label__M": [True,  False, False, False],
        "label__C": [False, True,  False, False],
        "label__A": [False, False, True,  False],
        "label__P": [False, False, False, True],
    })
    edges = pd.DataFrame({
        "s":    ["c", "m", "a"],
        "d":    ["m", "a", "p"],
        "type": ["R", "H", "K"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (m:M)<-[:R]-(c:C) "
        "OPTIONAL MATCH (m)-[:H]->(a:A) "
        "OPTIONAL MATCH (a)-[:K]->(p:P) "
        "RETURN c.id AS cid, a.id AS aid, p.id AS pid"
    )
    rows = result._nodes[["cid", "aid", "pid"]].to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["cid"] == "c"
    assert rows[0]["aid"] == "a"
    assert rows[0]["pid"] == "p"


def test_issue_1052_semijoin_multi_arm_second_arm_null_when_first_misses() -> None:
    """Arm-2 null-fills when arm-1 finds no match (join key for arm-2 is absent)."""
    nodes = pd.DataFrame({
        "id": ["m", "c", "a", "p"],
        "label__M": [True,  False, False, False],
        "label__C": [False, True,  False, False],
        "label__A": [False, False, True,  False],
        "label__P": [False, False, False, True],
    })
    # No H edge from m — arm-1 misses, so a.id is null; arm-2 also misses
    edges = pd.DataFrame({
        "s":    ["c", "a"],
        "d":    ["m", "p"],
        "type": ["R", "K"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (m:M)<-[:R]-(c:C) "
        "OPTIONAL MATCH (m)-[:H]->(a:A) "
        "OPTIONAL MATCH (a)-[:K]->(p:P) "
        "RETURN c.id AS cid, "
        "CASE a WHEN null THEN 'no-a' ELSE 'has-a' END AS a_status, "
        "CASE p WHEN null THEN 'no-p' ELSE 'has-p' END AS p_status"
    )
    rows = result._nodes[["cid", "a_status", "p_status"]].to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["cid"] == "c"
    assert rows[0]["a_status"] == "no-a"
    assert rows[0]["p_status"] == "no-p"


def test_issue_1052_semijoin_edge_alias_synthesis_after_filter() -> None:
    """Edge alias bare-form synthesis works correctly after semi-join row filtering."""
    nodes = pd.DataFrame({
        "id": ["m1", "m2", "a1", "a2", "p1", "p2"],
        "label__M": [True,  True,  False, False, False, False],
        "label__A": [False, False, True,  True,  False, False],
        "label__P": [False, False, False, False, True,  True],
    })
    edges = pd.DataFrame({
        "s":    ["m1", "a1",  "m2", "a2"],
        "d":    ["a1", "p1",  "a2", "p2"],
        "type": ["H",  "K",   "H",  "K"],
    })
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (m:M {id: $mid})-[:H]->(a:A) "
        "OPTIONAL MATCH (a)-[r:K]->(p:P) "
        "RETURN m.id AS mid, p.id AS pid, "
        "CASE r WHEN null THEN false ELSE true END AS knows",
        params={"mid": "m1"},
    )
    rows = result._nodes[["mid", "pid", "knows"]].to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["mid"] == "m1"
    assert rows[0]["pid"] == "p1"
    assert rows[0]["knows"] is True


def test_issue_1052_semijoin_no_bleed_from_unscoped_opt_rows() -> None:
    """Opt rows whose join-key values are NOT in base must not appear in result."""
    n = 5
    msg_ids = [f"m{i}" for i in range(1, n + 1)]
    comment_ids = [f"c{i}" for i in range(1, n + 1)]
    person_ids = [f"p{i}" for i in range(1, n + 1)]
    author_ids = [f"a{i}" for i in range(1, n + 1)]

    nodes = pd.DataFrame({
        "id": msg_ids + comment_ids + person_ids + author_ids,
        "label__M": [True] * n + [False] * (3 * n),
        "label__C": [False] * n + [True] * n + [False] * (2 * n),
        "label__P": [False] * (2 * n) + [True] * n + [False] * n,
        "label__A": [False] * (3 * n) + [True] * n,
    })
    edges_rows = []
    for i in range(1, n + 1):
        edges_rows += [
            {"s": f"c{i}", "d": f"m{i}", "type": "REPLY_OF"},
            {"s": f"c{i}", "d": f"p{i}", "type": "HAS_CREATOR"},
            {"s": f"m{i}", "d": f"a{i}", "type": "HAS_CREATOR"},
            {"s": f"a{i}", "d": f"p{i}", "type": "KNOWS"},
        ]
    edges = pd.DataFrame(edges_rows)
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (m:M {id: $mid})<-[:REPLY_OF]-(c:C)-[:HAS_CREATOR]->(p:P) "
        "OPTIONAL MATCH (m)-[:HAS_CREATOR]->(a:A)-[r:KNOWS]-(p) "
        "RETURN c.id AS cid, p.id AS pid, "
        "CASE r WHEN null THEN false ELSE true END AS knows "
        "ORDER BY cid",
        params={"mid": "m1"},
    )
    rows = result._nodes[["cid", "pid", "knows"]].to_dict(orient="records")
    # Only m1's reply chain should appear, not m2..m5
    assert rows == [{"cid": "c1", "pid": "p1", "knows": True}]


# ── Issue #983: bounded zero-min variable-length relationships (*0..N) ────────


def test_issue_983_bounded_zero_min_includes_zero_hop() -> None:
    """MATCH (a {id:'a'})-[*0..2]->(b) must include a itself (0-hop)."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"]}),
        pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"], "type": ["R", "R", "R"]}),
    )
    result = graph.gfql("MATCH (a {id: 'a'})-[*0..2]->(b) RETURN b.id AS id ORDER BY id")
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    # 0-hop: a itself; 1-hop: b; 2-hop: c
    assert ids == ["a", "b", "c"]


def test_issue_983_bounded_zero_min_typed_rel() -> None:
    """[:R*0..3] bounded zero-min with type filter returns 0- through 3-hop nodes."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d", "e"]}),
        pd.DataFrame(
            {
                "s": ["a", "b", "c", "a"],
                "d": ["b", "c", "d", "e"],
                "type": ["R", "R", "R", "S"],
            }
        ),
    )
    result = graph.gfql("MATCH (a {id: 'a'})-[:R*0..3]->(b) RETURN b.id AS id ORDER BY id")
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    # 0-hop: a; 1-hop: b; 2-hop: c; 3-hop: d (not e, wrong type)
    assert ids == ["a", "b", "c", "d"]


def test_issue_983_bounded_zero_min_multi_type() -> None:
    """[:HAS_TYPE|IS_SUBCLASS_OF*0..3] IC12 shape works end-to-end."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["tag", "sub1", "sub2", "leaf"]}),
        pd.DataFrame(
            {
                "s": ["tag", "sub1", "sub2"],
                "d": ["sub1", "sub2", "leaf"],
                "type": ["IS_SUBCLASS_OF", "IS_SUBCLASS_OF", "IS_SUBCLASS_OF"],
            }
        ),
    )
    result = graph.gfql(
        "MATCH (t {id: 'tag'})-[:IS_SUBCLASS_OF|HAS_TYPE*0..3]->(x) RETURN x.id AS id ORDER BY id"
    )
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    # 0-hop: tag; 1-hop: sub1; 2-hop: sub2; 3-hop: leaf
    assert ids == ["leaf", "sub1", "sub2", "tag"]


def test_issue_983_bounded_zero_min_max_zero_is_still_rejected() -> None:
    """*0 (exact zero hops) is still a parse error — degenerate, not supported."""
    with pytest.raises(Exception):
        _mk_simple_path_graph().gfql("MATCH (a)-[*0]->(b) RETURN b")


# ── Issue #1047: multi-row WITH prefix for scalar reentry ─────────────────────


def _mk_multi_row_scalar_prefix_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Two tags with distinct tagIds; posts connect to exactly one tag each."""
    return (
        pd.DataFrame(
            {
                "id": ["tagA", "tagB", "post1", "post2", "post3"],
                "label__Tag": [True, True, False, False, False],
                "label__Post": [False, False, True, True, True],
                "name": ["topicA", "topicB", None, None, None],
                "tagId": [1, 2, None, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["post1", "post2", "post3"],
                "d": ["tagA", "tagA", "tagB"],
                "type": ["HAS_TAG", "HAS_TAG", "HAS_TAG"],
            }
        ),
    )


def _mk_multi_row_scalar_prefix_graph() -> _CypherTestGraph:
    return _mk_graph(*_mk_multi_row_scalar_prefix_data())


def _mk_multi_row_scalar_prefix_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(*_mk_multi_row_scalar_prefix_data())


def test_issue_1047_multi_row_scalar_prefix_both_tags_matched() -> None:
    """Prefix WITH produces 2 rows (one per tag); suffix should union results from both."""
    # The prefix matches both Tag nodes (no name filter), producing 2 rows.
    # Each suffix run finds posts connected to that tag's tagId.
    result = _mk_multi_row_scalar_prefix_graph().gfql(
        "MATCH (t:Tag) "
        "WITH t.tagId AS knownTagId "
        "MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
        "RETURN post.id AS id ORDER BY id"
    )
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    # tagId=1 → post1, post2; tagId=2 → post3
    assert ids == ["post1", "post2", "post3"]


def test_issue_1047_multi_row_scalar_prefix_carried_scalar_visible_in_return() -> None:
    """Carried scalar from each prefix row is visible in the RETURN clause."""
    result = _mk_multi_row_scalar_prefix_graph().gfql(
        "MATCH (t:Tag) "
        "WITH t.tagId AS knownTagId "
        "MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
        "RETURN knownTagId, post.id AS postId ORDER BY knownTagId, postId"
    )
    rows = result._nodes.to_dict(orient="records")
    assert rows == [
        {"knownTagId": 1, "postId": "post1"},
        {"knownTagId": 1, "postId": "post2"},
        {"knownTagId": 2, "postId": "post3"},
    ]


def test_issue_1047_multi_row_scalar_prefix_empty_prefix_still_returns_empty() -> None:
    """If prefix produces 0 rows, result is still empty (unchanged)."""
    result = _mk_multi_row_scalar_prefix_graph().gfql(
        "MATCH (t:Tag {name: 'missing'}) "
        "WITH t.tagId AS knownTagId "
        "MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
        "RETURN post.id AS id"
    )
    assert result._nodes.to_dict(orient="records") == []


def test_issue_1047_existing_single_row_prefix_still_works() -> None:
    """Single-row prefix (original case) continues to work after multi-row fix."""
    result = _mk_multi_row_scalar_prefix_graph().gfql(
        "MATCH (t:Tag {name: 'topicA'}) "
        "WITH t.tagId AS knownTagId "
        "MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
        "RETURN post.id AS id ORDER BY id"
    )
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    assert ids == ["post1", "post2"]


def test_issue_1047_duplicate_seed_graph_now_works() -> None:
    """Previously rejected: duplicate seed produces 2 rows, now both should match.

    Cypher bag semantics: each of the 2 prefix rows (tag1, tag1b — both tagId=101)
    is a distinct context.  The suffix finds {post1, post2} for *each* row, so the
    union produces 4 rows: [post1, post1, post2, post2].
    """
    # Both tag1 and tag1b have tagId=101, posts connect to one each.
    result = _mk_prefix_scalar_reentry_duplicate_seed_graph().gfql(
        "MATCH (knownTag:Tag {name: 'topic'}) "
        "WITH knownTag.tagId AS knownTagId "
        "MATCH (post:Post)-[:HAS_TAG]->(t:Tag {tagId: knownTagId}) "
        "RETURN post.id AS id ORDER BY id"
    )
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    # Two prefix rows × two matching posts each = 4 rows (bag semantics, no implicit DISTINCT).
    # Row order within each fan-out iteration is preserved; iterations are concatenated in order.
    assert ids == ["post1", "post2", "post1", "post2"]


def test_issue_1047_multi_row_scalar_prefix_on_cudf() -> None:
    """Multi-row prefix works on cuDF path."""
    pytest.importorskip("cudf")
    result = _mk_multi_row_scalar_prefix_graph_cudf().gfql(
        "MATCH (t:Tag) "
        "WITH t.tagId AS knownTagId "
        "MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
        "RETURN post.id AS id ORDER BY id",
        engine="cudf",
    )
    assert type(result._nodes).__module__.startswith("cudf")
    ids = [r["id"] for r in _to_pandas_df(result._nodes).to_dict(orient="records")]
    assert ids == ["post1", "post2", "post3"]


def test_issue_1047_multi_row_scalar_prefix_partial_hit() -> None:
    """Prefix rows where only some produce non-empty suffix results.

    tagId=1 matches post1 and post2 (via tagA).
    tagId=99 matches nothing — no post connects to a tag with tagId=99.
    Only the hits from tagId=1 should appear; no crash, no spurious rows.
    """
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["tagA", "tagC", "post1", "post2"],
                "label__Tag": [True, True, False, False],
                "label__Post": [False, False, True, True],
                "tagId": [1, 99, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["post1", "post2"],
                "d": ["tagA", "tagA"],
                "type": ["HAS_TAG", "HAS_TAG"],
            }
        ),
    )
    result = graph.gfql(
        "MATCH (t:Tag) "
        "WITH t.tagId AS knownTagId "
        "MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
        "RETURN post.id AS id ORDER BY id"
    )
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    # tagId=1 → post1, post2; tagId=99 → empty (no crash)
    assert ids == ["post1", "post2"]


def test_issue_1047_multi_row_scalar_prefix_with_optional_reentry_raises() -> None:
    """Multi-row scalar prefix combined with optional_reentry is not yet supported.

    The multi-row fan-out path returns early before the optional_reentry null-fill
    branch, which would silently produce wrong results (missing null rows for prefix
    rows that matched nothing).  Until null-fill is implemented for multi-row, the
    engine must raise rather than silently return incomplete results.
    """
    with pytest.raises(Exception, match="optional"):
        _mk_multi_row_scalar_prefix_graph().gfql(
            "MATCH (t:Tag) "
            "WITH t.tagId AS knownTagId "
            "OPTIONAL MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
            "RETURN post.id AS id"
        )


# ── Round 2 amplification: #1047 bag semantics, boundaries, #983 bounds ───────


def test_issue_1047_distinct_prefix_deduplicates_bag() -> None:
    """WITH DISTINCT collapses the 2 duplicate-seed prefix rows to 1.

    Both tag1 and tag1b carry tagId=101, but DISTINCT on the scalar
    produces a single row.  The suffix runs once and finds {post1, post2}
    — exactly 2 rows, not 4.
    """
    result = _mk_prefix_scalar_reentry_duplicate_seed_graph().gfql(
        "MATCH (knownTag:Tag {name: 'topic'}) "
        "WITH DISTINCT knownTag.tagId AS knownTagId "
        "MATCH (post:Post)-[:HAS_TAG]->(t:Tag {tagId: knownTagId}) "
        "RETURN post.id AS id ORDER BY id"
    )
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    assert ids == ["post1", "post2"]


def test_issue_1047_single_row_prefix_with_optional_match_still_works() -> None:
    """Single-row scalar prefix + OPTIONAL MATCH must not be blocked by the multi-row guard.

    The guard at line ~709 fires only when prefix_row_count > 1.  A single-row prefix
    (topicA has exactly one matching Tag) must continue to work with OPTIONAL MATCH.
    """
    result = _mk_multi_row_scalar_prefix_graph().gfql(
        "MATCH (t:Tag {name: 'topicA'}) "
        "WITH t.tagId AS knownTagId "
        "OPTIONAL MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
        "RETURN post.id AS id ORDER BY id"
    )
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    assert ids == ["post1", "post2"]


def test_issue_1461_optional_reentry_null_extension_preserves_carried_scalar() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "label__Seed": [True, True]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
    )

    result = graph.gfql(
        "MATCH (a:Seed {id: 'b'}) "
        "WITH a, a.id AS aid "
        "OPTIONAL MATCH (a)-[:R]->(b) "
        "RETURN aid, b.id AS bid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert rows == [{"aid": "b", "bid": None}]


def test_issue_1461_optional_reentry_null_extension_does_not_leak_unprojected_scalar() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "label__Seed": [True, True]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
    )

    result = graph.gfql(
        "MATCH (a:Seed {id: 'b'}) "
        "WITH a, a.id AS aid "
        "OPTIONAL MATCH (a)-[:R]->(b) "
        "RETURN b.id AS bid"
    )

    rows = result._nodes.to_dict(orient="records")
    assert rows == [{"bid": None}]


def test_issue_1461_optional_reentry_mixed_carried_rows_preserve_null_extended_prefixes() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "label__Seed": [True, True, True, False],
            }
        ),
        pd.DataFrame({"s": ["a", "c"], "d": ["b", "d"], "type": ["R", "R"]}),
    )

    result = graph.gfql(
        "MATCH (a:Seed) "
        "WITH a, a.id AS aid "
        "OPTIONAL MATCH (a)-[:R]->(b) "
        "RETURN aid, b.id AS bid"
    )

    rows = sorted(result._nodes.to_dict(orient="records"), key=lambda row: row["aid"])
    assert rows == [
        {"aid": "a", "bid": "b"},
        {"aid": "b", "bid": None},
        {"aid": "c", "bid": "d"},
    ]


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_issue_1461_optional_reentry_preserves_multiple_carried_scalars_on_null_rows(
    engine: Optional[str],
) -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "label__Seed": [True, True, True, False],
            "bucket": ["alpha", "beta", "gamma", "sink"],
        }
    )
    edges = pd.DataFrame({"s": ["a", "c"], "d": ["d", "b"], "type": ["R", "R"]})
    if engine == "cudf":
        _require_cudf_runtime()
        graph = _mk_cudf_graph(nodes, edges)
    else:
        graph = _mk_graph(nodes, edges)

    query = (
        "MATCH (a:Seed) "
        "WITH a, a.id AS aid, a.bucket AS bucket "
        "OPTIONAL MATCH (a)-[:R]->(b) "
        "RETURN aid, bucket, b.id AS bid"
    )
    result = graph.gfql(query, engine=engine) if engine == "cudf" else graph.gfql(query)

    if engine == "cudf":
        assert type(result._nodes).__module__.startswith("cudf")
    frame = _to_pandas_df(result._nodes)
    rows = sorted(
        frame.where(pd.notna(frame), None).to_dict(orient="records"),
        key=lambda row: row["aid"],
    )
    assert rows == [
        {"aid": "a", "bucket": "alpha", "bid": "d"},
        {"aid": "b", "bucket": "beta", "bid": None},
        {"aid": "c", "bucket": "gamma", "bid": "b"},
    ]


def test_issue_1461_optional_reentry_matched_single_prefix_remains_native_semantics() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "label__Seed": [True, False]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
    )

    result = graph.gfql(
        "MATCH (a:Seed {id: 'a'}) "
        "WITH a, a.id AS aid "
        "OPTIONAL MATCH (a)-[:R]->(b) "
        "RETURN aid, b.id AS bid"
    )

    assert result._nodes.to_dict(orient="records") == [{"aid": "a", "bid": "b"}]


def test_issue_1047_optional_reentry_raises_is_gfql_validation_error() -> None:
    """The optional_reentry + multi-row guard raises GFQLValidationError specifically."""
    with pytest.raises(GFQLValidationError, match="optional"):
        _mk_multi_row_scalar_prefix_graph().gfql(
            "MATCH (t:Tag) "
            "WITH t.tagId AS knownTagId "
            "OPTIONAL MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
            "RETURN post.id AS id"
        )


@pytest.mark.parametrize(
    "graph_factory, expected_rows",
    [
        (_mk_optional_prefix_reentry_no_match_graph, []),
        (_mk_optional_prefix_reentry_match_graph, [{"cid": "c1"}]),
    ],
)
def test_issue_1356_optional_prefix_reentry_handles_no_match_semantics(
    graph_factory: Callable[[], _CypherTestGraph],
    expected_rows: List[Dict[str, Any]],
) -> None:
    """OPTIONAL prefix reentry should not fail identity recovery on no-match.

    #1356 regression guard: when the OPTIONAL prefix has no matches, the
    prefix stage may produce a null carry row without whole-row metadata.
    Reentry should treat it as an empty seed set (no crash), while matched
    fixtures continue to produce expected rows.
    """
    result = graph_factory().gfql(
        "OPTIONAL MATCH (a:A)-[:R]->(b:B) "
        "WITH a, b "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN c.id AS cid ORDER BY cid"
    )
    assert result._nodes.to_dict(orient="records") == expected_rows


def test_issue_1047_partial_hit_zero_contribution_has_no_null_rows() -> None:
    """The empty-suffix iteration contributes 0 rows, not null/NaN rows.

    When tagId=99 matches nothing, the fan-out loop appends an empty result.
    The union must contain exactly the rows from the matching iteration,
    with no null-padded entries from the non-matching one.
    """
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["tagA", "tagC", "post1"],
                "label__Tag": [True, True, False],
                "label__Post": [False, False, True],
                "tagId": [1, 99, None],
            }
        ),
        pd.DataFrame({"s": ["post1"], "d": ["tagA"], "type": ["HAS_TAG"]}),
    )
    result = graph.gfql(
        "MATCH (t:Tag) "
        "WITH t.tagId AS knownTagId "
        "MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
        "RETURN post.id AS id"
    )
    rows = result._nodes.to_dict(orient="records")
    # Exactly 1 row; no NaN/None from the empty tagId=99 iteration
    assert len(rows) == 1
    assert rows[0]["id"] == "post1"
    assert rows[0]["id"] is not None


def test_issue_1047_empty_base_graph_with_multi_row_prefix() -> None:
    """Multi-row prefix against an empty base graph returns empty, no crash."""
    graph = _mk_graph(
        pd.DataFrame({"id": pd.Series(dtype="object"), "label__Tag": pd.Series(dtype="bool"), "tagId": pd.Series(dtype="object")}),
        pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object"), "type": pd.Series(dtype="object")}),
    )
    result = graph.gfql(
        "MATCH (t:Tag) "
        "WITH t.tagId AS knownTagId "
        "MATCH (post)-[:HAS_TAG]->(x {tagId: knownTagId}) "
        "RETURN post.id AS id"
    )
    assert result._nodes.to_dict(orient="records") == []


def test_issue_983_zero_one_hop_includes_seed_and_neighbors() -> None:
    """`*0..1` must include the seed itself (0-hop) and its direct neighbors (1-hop)."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["R", "R"]}),
    )
    result = graph.gfql("MATCH (a {id: 'a'})-[*0..1]->(b) RETURN b.id AS id ORDER BY id")
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    # 0-hop: a; 1-hop: b (c is 2 hops away)
    assert ids == ["a", "b"]


def test_issue_983_large_upper_bound_stops_at_graph_depth() -> None:
    """`*0..100` on a 3-node chain returns all reachable nodes, no crash at the bound."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["R", "R"]}),
    )
    result = graph.gfql("MATCH (a {id: 'a'})-[*0..100]->(b) RETURN b.id AS id ORDER BY id")
    ids = [r["id"] for r in result._nodes.to_dict(orient="records")]
    # Reachable from a: a (0), b (1), c (2) — upper bound 100 is not an issue
    assert ids == ["a", "b", "c"]


@pytest.mark.parametrize("engine", [None, "cudf"], ids=["pandas", "cudf"])
def test_issue_1369_zero_zero_hop_returns_seed_engine_parity(engine: Optional[str]) -> None:
    """`*0..0` is a valid zero-length path and returns the seed node."""
    if engine == "cudf":
        _require_cudf_runtime()
        graph = _mk_simple_path_graph_cudf()
    else:
        graph = _mk_simple_path_graph()

    query = "MATCH (a {id: 'a'})-[*0..0]->(b) RETURN b.id AS id"
    result = graph.gfql(query, engine=engine) if engine == "cudf" else graph.gfql(query)
    rows = _to_pandas_df(result._nodes).to_dict(orient="records") if engine == "cudf" else result._nodes.to_dict(orient="records")
    if engine == "cudf":
        assert type(result._nodes).__module__.startswith("cudf")
    assert rows == [{"id": "a"}]


def test_issue_1369_one_one_hop_does_not_include_seed() -> None:
    """The Cypher zero-hop seed opt-in must not leak into positive-hop traversals."""
    result = _mk_simple_path_graph().gfql("MATCH (a {id: 'a'})-[*1..1]->(b) RETURN b.id AS id")
    assert result._nodes.to_dict(orient="records") == [{"id": "b"}]


def test_issue_1369_empty_graph_post_aggregate_boolean_projection() -> None:
    """Empty global aggregates must still project post-aggregate expressions."""
    graph = _mk_graph(
        pd.DataFrame({"id": pd.Series(dtype="object")}),
        pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")}),
    )
    result = graph.gfql("MATCH (a) RETURN count(a) > 0")
    assert result._nodes.to_dict(orient="records") == [{"count(a) > 0": False}]


def test_issue_1369_empty_graph_post_aggregate_multiple_expressions() -> None:
    """Empty global aggregates project each final expression name and value."""
    graph = _mk_graph(
        pd.DataFrame({"id": pd.Series(dtype="object")}),
        pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")}),
    )
    result = graph.gfql("MATCH (a) RETURN count(a) = 0 AS is_empty, count(a) + 1 AS plus_one")
    assert result._nodes.to_dict(orient="records") == [{"is_empty": True, "plus_one": 1}]


def test_issue_1369_empty_graph_post_aggregate_boolean_projection_on_cudf() -> None:
    """cuDF empty global aggregates match pandas post-aggregate projection semantics."""
    _require_cudf_runtime()
    graph = _mk_cudf_graph(
        pd.DataFrame({"id": pd.Series(dtype="object")}),
        pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")}),
    )
    result = graph.gfql("MATCH (a) RETURN count(a) > 0 AS any_nodes", engine="cudf")
    assert type(result._nodes).__module__.startswith("cudf")
    assert _to_pandas_df(result._nodes).to_dict(orient="records") == [{"any_nodes": False}]


# ── Issue #977: cudf SIGSEGV — safe_map_series regression guards ──────────────


def test_issue_977_cudf_label_filter_no_sigsegv() -> None:
    """cudf: label filter must not SIGSEGV (safe_map_series / filter_by_dict)."""
    cudf = pytest.importorskip("cudf")
    from graphistry.Engine import EngineAbstract
    nodes = cudf.DataFrame(pd.DataFrame({
        "id": ["a", "b", "c"],
        "label__Person": [True, True, False],
    }))
    edges = cudf.DataFrame(pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["T"]}))
    g_cu = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    result = g_cu.gfql("MATCH (p:Person) RETURN p.id AS pid ORDER BY pid", engine=EngineAbstract.CUDF)
    ids = sorted(_to_pandas_df(result._nodes)["pid"].tolist())
    assert ids == ["a", "b"]


def test_issue_977_cudf_single_hop_no_sigsegv() -> None:
    """cudf: single hop traversal exercises df_executor safe_map_series."""
    cudf = pytest.importorskip("cudf")
    from graphistry.Engine import EngineAbstract
    nodes = cudf.DataFrame(pd.DataFrame({"id": ["a", "b", "c"]}))
    edges = cudf.DataFrame(pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["T", "T"]}))
    g_cu = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    result = g_cu.gfql("MATCH (a)-[:T]->(b) RETURN a.id AS aid ORDER BY aid", engine=EngineAbstract.CUDF)
    assert sorted(_to_pandas_df(result._nodes)["aid"].tolist()) == ["a", "b"]


def test_issue_977_pandas_label_filter_regression() -> None:
    """pandas: label filter still works after safe_map_series change (regression guard)."""
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "label__Person": [True, True, False]})
    edges = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["T"]})
    g = _mk_graph(nodes, edges)
    result = g.gfql("MATCH (p:Person) RETURN p.id AS pid ORDER BY pid")
    ids = sorted(result._nodes["pid"].tolist())
    assert ids == ["a", "b"]


def test_order_by_multi_column_no_crash() -> None:
    """ORDER BY with two columns must not crash and must sort correctly."""
    nodes = pd.DataFrame({
        "id": ["c", "a", "b", "a"],
        "score": [3, 1, 2, 1],
        "name": ["charlie", "alice", "bob", "anna"],
    })
    edges = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["T"]})
    g = _mk_graph(nodes, edges)

    result = g.gfql(
        "MATCH (n) RETURN n.name AS name, n.score AS score ORDER BY score, name"
    )

    rows = result._nodes[["score", "name"]].to_dict(orient="records")
    assert len(rows) > 0, "Expected non-empty result from ORDER BY multi-column query"

    # Verify the result is sorted: primary key score asc, secondary key name asc
    for i in range(len(rows) - 1):
        s_cur, s_nxt = rows[i]["score"], rows[i + 1]["score"]
        assert s_cur <= s_nxt, f"Rows not sorted by score: {rows}"
        if s_cur == s_nxt:
            assert rows[i]["name"] <= rows[i + 1]["name"], (
                f"Rows with equal score not sorted by name: {rows}"
            )


# ---------------------------------------------------------------------------
# count(*) short-circuit (count_table row op) — pandas-lane coverage
# ---------------------------------------------------------------------------


def test_count_star_node_shortcircuit_lowers_to_count_table() -> None:
    chain = cypher_to_gfql("MATCH (n) RETURN count(*) AS c")
    ops = chain.chain if hasattr(chain, "chain") else chain
    calls = [op for op in ops if isinstance(op, ASTCall) and op.function == "count_table"]
    assert len(calls) == 1
    assert calls[0].params["alias"] == "c"
    assert calls[0].params.get("table", "nodes") == "nodes"


def test_count_star_node_shortcircuit_executes_pandas() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
    )

    result = graph.gfql("MATCH (n) RETURN count(*) AS c")

    assert result._nodes.to_dict(orient="records") == [{"c": 3}]


def test_count_star_edge_shortcircuit_counts_only_valid_endpoint_edges() -> None:
    # 3 edges but 'zz' is not a node: MATCH ()-[r]->() requires both endpoints
    # to exist, so the dangling edge is excluded (count is 2, not 3).
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"]}),
        pd.DataFrame({"s": ["a", "b", "a"], "d": ["b", "a", "zz"], "type": ["R", "R", "R"]}),
    )

    result = graph.gfql("MATCH ()-[r]->() RETURN count(*) AS c")

    assert result._nodes.to_dict(orient="records") == [{"c": 2}]


def test_count_star_with_group_key_does_not_shortcircuit() -> None:
    # A grouped count is NOT the pure-count shape: it must keep the
    # rows + group_by pipeline, not lower to count_table.
    chain = cypher_to_gfql("MATCH (n) RETURN n.id AS i, count(*) AS c")
    ops = chain.chain if hasattr(chain, "chain") else chain
    assert not any(isinstance(op, ASTCall) and op.function == "count_table" for op in ops)


def test_count_table_row_op_direct_pandas() -> None:
    from graphistry.compute.ast import ASTNode, count_table, rows

    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b"]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
    )

    result = graph.gfql([ASTNode(), rows(), count_table(table="nodes", alias="cnt")])

    assert result._nodes.to_dict(orient="records") == [{"cnt": 2}]


def test_count_table_row_op_rejects_unknown_table() -> None:
    from graphistry.compute.ast import ASTNode, count_table, rows

    graph = _mk_graph(
        pd.DataFrame({"id": ["a"]}),
        pd.DataFrame({"s": [], "d": [], "type": []}),
    )

    with pytest.raises(GFQLValidationError):
        graph.gfql([ASTNode(), rows(), count_table(table="everything", alias="cnt")])


def test_count_table_frame_op_error_and_empty_paths() -> None:
    # Direct frame-op coverage (the GFQL call path validates params BEFORE the
    # frame op, so these branches need direct exercise): bad table name, missing
    # source column, and the no-table 0-count fallback.
    from graphistry.compute.gfql.row import frame_ops

    nodes = pd.DataFrame({"id": ["a", "b"], "__mask__": [True, None]})
    edges = pd.DataFrame({"s": ["a"], "d": ["b"]})
    ctx = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

    with pytest.raises(ValueError, match="must be one of"):
        frame_ops.count_table(ctx, table="everything", alias="c")
    with pytest.raises(ValueError, match="not found"):
        frame_ops.count_table(ctx, table="nodes", source="nope", alias="c")

    # source mask: null counts as False
    out = frame_ops.count_table(ctx, table="nodes", source="__mask__", alias="c")
    assert out._nodes.to_dict(orient="records") == [{"c": 1}]

    # no node table -> 0-count row templated from the sibling edges frame
    ctx2 = graphistry.edges(edges, "s", "d")
    assert ctx2._nodes is None
    out2 = frame_ops.count_table(ctx2, table="nodes", alias="c")
    assert out2._nodes.to_dict(orient="records") == [{"c": 0}]

    # no tables at all -> plain pandas 0-count row
    ctx3 = graphistry.bind()
    assert ctx3._nodes is None and ctx3._edges is None
    out3 = frame_ops.count_table(ctx3, table="nodes", alias="c")
    assert out3._nodes.to_dict(orient="records") == [{"c": 0}]

def test_string_cypher_parenthesized_and_preserves_missing_property_as_null() -> None:
    # A parenthesized AND must stay on the row-filter path, not the filter_dict path:
    # the row engine treats an absent property as null (Cypher semantics), whereas
    # filter_dict requires the column to exist and would raise.
    g = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"], "x": [1, 2, 1, 1]}),  # no 'missing' col
        pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"]}),
    )
    got = g.gfql("MATCH (n) WHERE n.missing IS NULL AND (n.x = 1) RETURN n.id AS id")
    assert sorted(got._nodes["id"].tolist()) == ["a", "c", "d"]
    # IS NOT NULL on the same absent property yields no rows (absent -> null)
    none = g.gfql("MATCH (n) WHERE n.missing IS NOT NULL AND (n.x = 1) RETURN n.id AS id")
    assert none._nodes["id"].tolist() == [] if "id" in none._nodes.columns else len(none._nodes) == 0
