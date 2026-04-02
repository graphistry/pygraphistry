from __future__ import annotations

import pandas as pd
import pytest
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from graphistry.compute.ast import ASTCall, ASTNode, ASTEdgeForward, ASTEdgeReverse, ASTEdgeUndirected
from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError, GFQLTypeError, GFQLValidationError
from graphistry.compute.predicates.is_in import IsIn
from graphistry.compute.gfql.same_path_types import col, compare
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
)
from graphistry.tests.test_compute import CGFull


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


def _mk_simple_path_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]}),
    )


def _mk_triangle_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "b", "a"], "d": ["b", "c", "c"]}),
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


def _mk_path_with_isolate_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "z"]}),
        pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]}),
    )


def _mk_simple_path_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]}),
    )


def _mk_path_with_isolate_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(
        pd.DataFrame({"id": ["a", "b", "c", "z"]}),
        pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]}),
    )


def _mk_empty_graph() -> _CypherTestGraph:
    return _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))


def _mk_reentry_carried_scalar_graph() -> _CypherTestGraph:
    return _mk_graph(
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


def _mk_reentry_carried_scalar_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(
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


def _mk_connected_reentry_carried_scalar_graph() -> _CypherTestGraph:
    return _mk_graph(
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


def _mk_connected_reentry_carried_scalar_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(
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


def _mk_connected_multi_pattern_fanout_graph() -> _CypherTestGraph:
    return _mk_graph(
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


def _mk_connected_multi_pattern_fanout_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(
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


def _mk_multi_stage_reentry_graph() -> _CypherTestGraph:
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
                "s": ["a", "b", "c", "a", "b"],
                "d": ["b", "c", "d", "e", "e"],
                "type": ["R", "S", "T", "R", "S"],
            }
        ),
    )


def _mk_multi_stage_reentry_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(
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


def _mk_recent_message_reentry_graph() -> _CypherTestGraph:
    return _mk_graph(
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


def _mk_recent_message_reentry_graph_cudf() -> _CypherTestGraph:
    return _mk_cudf_graph(
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
    return cast(pd.DataFrame, df.to_pandas() if hasattr(df, "to_pandas") else df)


def _parse_query(query: str) -> CypherQuery:
    return cast(CypherQuery, parse_cypher(query))


def _compile_query(query: str) -> CompiledCypherQuery:
    return cast(CompiledCypherQuery, compile_cypher(query))


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
    assert lowered.where == [compare(col("n", "id"), "==", col(repeated._name, "id"))]


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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
        {"left": "(:A)", "right": "(:A)"},
        {"left": "(:B)", "right": "(:B)"},
    ]
    entity_meta = getattr(result, "_cypher_entity_projection_meta")
    assert entity_meta["left"]["alias"] == "a"
    assert entity_meta["right"]["alias"] == "b"


def test_lower_match_query_rejects_bare_where_pattern_predicate_without_relationship() -> None:
    with pytest.raises(GFQLValidationError, match="must include a relationship"):
        lower_cypher_query(_parse_query("MATCH (n) WHERE (n) RETURN n"))


def test_lower_match_query_rejects_multiple_where_pattern_predicates() -> None:
    with pytest.raises(GFQLValidationError, match="one positive pattern predicate at a time"):
        lower_cypher_query(_parse_query("MATCH (n) WHERE (n)-[:R]->() AND (n)-[:S]->() RETURN n"))


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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    pdf = result._nodes.to_pandas().sort_values("r").reset_index(drop=True)
    assert pdf.to_dict(orient="records") == [
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

    assert result._nodes.to_pandas().to_dict(orient="records") == [
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

    assert result._nodes.to_pandas().to_dict(orient="records") == [
        {"a": "(:B {num: 30.94857, num2: 0.00002})"}
    ]


def test_string_cypher_formats_single_node_entity_projection_with_alias() -> None:
    nodes = pd.DataFrame({"id": ["a"], "type": ["A"]})
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql("MATCH (a) RETURN a AS ColumnName")

    assert result._nodes.to_dict(orient="records") == [{"ColumnName": "(:A)"}]
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [{"b": "(:B)"}]


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

    assert result._nodes.to_dict(orient="records") == [{"n": "({name: 'bar'})"}]


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

    assert result._nodes.to_dict(orient="records") == [{"n": "({name: 'bar'})"}]


def test_string_cypher_formats_numeric_id_as_entity_property() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": [1, 10]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n) RETURN DISTINCT n ORDER BY n.id")

    assert result._nodes.to_dict(orient="records") == [{"n": "({id: 1})"}, {"n": "({id: 10})"}]


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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [{"n": "({id: 1})"}, {"n": "({id: 10})"}]


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

    assert result._nodes.to_dict(orient="records") == [{"n": "({name: 'a'})"}]


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


def test_string_cypher_supports_generic_match_where_chained_comparison() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["a"], "num": [5]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (n)\nWHERE 10 < n.num <= 3\nRETURN n.num")

    assert result._nodes.to_dict(orient="records") == []


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


def test_string_cypher_supports_whole_row_grouping_with_post_aggregate_expression() -> None:
    graph = _mk_graph(
        pd.DataFrame({"id": ["n1"]}),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = graph.gfql("MATCH (a) RETURN a, count(a) + 3")

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

    assert result._nodes.to_dict(orient="records") == [{"a": "(:L)", "count(*)": 1}]


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a:L)-[r]->(b) RETURN a, count(*) AS cnt",
        "MATCH (a:L)-[r]->(b) RETURN a.id AS aid, count(*) AS cnt",
        "MATCH (a)-[r]->(b) RETURN count(*) AS cnt",
        "MATCH (a)-[r]->(b) RETURN count(a) AS cnt",
        "MATCH (a)-[r]->(b) RETURN sum(1) AS total",
    ],
)
def test_string_cypher_rejects_unsound_node_carrier_multiplicity_sensitive_aggregates(query: str) -> None:
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

    assert result._nodes.to_pandas().to_dict(orient="records") == [{"n": 11}]


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

    assert result._nodes.to_dict(orient="records") == [{"i": "(:TextNode {var: 'tf'})"}]


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

    assert pd.isna(result._nodes.to_pandas().iloc[0]["result"])


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

    assert result._nodes.to_pandas().to_dict(orient="records") == [
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

    assert result._nodes.to_pandas().to_dict(orient="records") == [
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


def test_string_cypher_executes_unwind_temporal_date_literals() -> None:
    _assert_query_rows(
        "UNWIND [date({year: 1910, month: 5, day: 6}), date({year: 1980, month: 10, day: 24})] AS dates "
        "WITH dates ORDER BY dates ASC LIMIT 2 RETURN dates",
        [{"dates": "1910-05-06"}, {"dates": "1980-10-24"}],
    )


def test_string_cypher_parses_week_date_literals() -> None:
    _assert_query_rows("RETURN date({year: 1817, week: 1, dayOfWeek: 2}) AS d", [{"d": "1816-12-31"}])


def test_string_cypher_parses_localtime_compact_literals() -> None:
    _assert_query_rows("RETURN localtime('214032.142') AS result", [{"result": "21:40:32.142"}])


def test_string_cypher_parses_datetime_named_zone_literals() -> None:
    _assert_query_rows(
        "RETURN datetime('2015-07-21T21:40:32.142[Europe/London]') AS result",
        [{"result": "2015-07-21T21:40:32.142+01:00[Europe/London]"}],
    )


def test_string_cypher_normalizes_time_offset_seconds() -> None:
    _assert_query_rows(
        "RETURN time({hour: 12, minute: 34, second: 56, timezone: '+02:05:00'}) AS result",
        [{"result": "12:34:56+02:05"}],
    )


def test_string_cypher_defaults_datetime_map_to_utc() -> None:
    _assert_query_rows(
        "RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31}) AS result",
        [{"result": "1984-10-11T12:31Z"}],
    )


def test_string_cypher_parses_datetime_map_with_quoted_offset_timezone() -> None:
    _assert_query_rows(
        "RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS result",
        [{"result": "1984-10-11T12:00+01:00"}],
    )


def test_string_cypher_combines_fractional_temporal_fields() -> None:
    _assert_query_rows(
        "RETURN "
        "localtime({hour: 12, minute: 31, second: 14, millisecond: 645, microsecond: 876, nanosecond: 123}) AS t, "
        "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, millisecond: 645, microsecond: 876, nanosecond: 123, timezone: '+01:00'}) AS dt",
        [
        {
            "t": "12:31:14.645876123",
            "dt": "1984-10-11T12:31:14.645876123+01:00",
        }
        ],
    )


def test_string_cypher_supports_datetime_fromepoch_functions() -> None:
    _assert_query_rows(
        "RETURN datetime.fromepoch(416779, 999999999) AS d1, "
        "datetime.fromepochmillis(237821673987) AS d2",
        [
        {
            "d1": "1970-01-05T19:46:19.999999999Z",
            "d2": "1977-07-15T13:34:33.987Z",
        }
        ],
    )


def test_string_cypher_defaults_time_map_to_utc() -> None:
    _assert_query_rows(
        "RETURN time({hour: 12, minute: 31, second: 14}) AS result",
        [{"result": "12:31:14Z"}],
    )


def test_string_cypher_supports_nested_temporal_base_date_overrides() -> None:
    _assert_query_rows(
        "RETURN date({date: date('1816-12-31'), year: 1817, week: 2}) AS d",
        [{"d": "1817-01-07"}],
    )


def test_string_cypher_inherits_iso_week_year_from_base_date() -> None:
    _assert_query_rows(
        "RETURN "
        "date({date: date('1816-12-30'), week: 2, dayOfWeek: 3}) AS d1, "
        "localdatetime({date: date('1816-12-31'), week: 2}) AS d2, "
        "datetime({date: date('1816-12-30'), week: 2, dayOfWeek: 3}) AS d3",
        [
        {
            "d1": "1817-01-08",
            "d2": "1817-01-07T00:00",
            "d3": "1817-01-08T00:00Z",
        }
        ],
    )


def test_string_cypher_executes_temporal_date_casts_from_with_alias() -> None:
    _assert_query_rows(
        "WITH date({year: 1984, month: 11, day: 11}) AS other "
        "RETURN date({date: other, year: 28}) AS result",
        [{"result": "0028-11-11"}],
    )


def test_string_cypher_executes_temporal_datetime_casts_from_with_alias() -> None:
    _assert_query_rows(
        "WITH date({year: 1984, month: 10, day: 11}) AS other "
        "RETURN datetime({date: other, hour: 10, minute: 10, second: 10}) AS result",
        [{"result": "1984-10-11T10:10:10Z"}],
    )


def test_string_cypher_executes_temporal_date_cast_from_aware_datetime_alias() -> None:
    _assert_query_rows(
        "WITH datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'}) AS other "
        "RETURN date(other) AS result",
        [{"result": "1984-11-11"}],
    )


def test_string_cypher_executes_temporal_localtime_cast_from_aware_time_alias() -> None:
    _assert_query_rows(
        "WITH time({hour: 12, minute: 31, second: 14, nanosecond: 645876, timezone: '+01:00'}) AS other "
        "RETURN localtime(other) AS result",
        [{"result": "12:31:14.000645876"}],
    )


def test_string_cypher_executes_temporal_localdatetime_cast_from_aware_datetime_alias() -> None:
    _assert_query_rows(
        "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other "
        "RETURN localdatetime({datetime: other}) AS result",
        [{"result": "1984-10-11T12:00"}],
    )


def test_string_cypher_executes_temporal_time_cast_and_constructor_from_named_zone_datetime_alias() -> None:
    _assert_query_rows(
        "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other "
        "RETURN "
        "time(other) AS cast_result, "
        "time({time: other}) AS ctor_result, "
        "time({time: other, timezone: '+05:00'}) AS converted_result, "
        "time({time: other, second: 42, timezone: '+05:00'}) AS converted_second_result",
        [
        {
            "cast_result": "12:00+01:00",
            "ctor_result": "12:00+01:00",
            "converted_result": "16:00+05:00",
            "converted_second_result": "16:00:42+05:00",
        }
        ],
    )


def test_string_cypher_executes_temporal_time_constructor_converts_aware_time_alias_timezone() -> None:
    _assert_query_rows(
        "WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other "
        "RETURN "
        "time({time: other, timezone: '+05:00'}) AS converted_result, "
        "time({time: other, second: 42, timezone: '+05:00'}) AS converted_second_result",
        [
        {
            "converted_result": "16:31:14.645876+05:00",
            "converted_second_result": "16:31:42.645876+05:00",
        }
        ],
    )


def test_string_cypher_executes_temporal_datetime_constructor_preserves_and_converts_named_zone_time_alias() -> None:
    _assert_query_rows(
        "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other "
        "RETURN "
        "datetime({year: 1984, month: 10, day: 11, time: other}) AS preserved_result, "
        "datetime({year: 1984, month: 10, day: 11, time: other, second: 42}) AS preserved_second_result, "
        "datetime({year: 1984, month: 10, day: 11, time: other, timezone: '+05:00'}) AS converted_result, "
        "datetime({year: 1984, month: 10, day: 11, time: other, second: 42, timezone: 'Pacific/Honolulu'}) AS converted_named_result",
        [
        {
            "preserved_result": "1984-10-11T12:00+01:00[Europe/Stockholm]",
            "preserved_second_result": "1984-10-11T12:00:42+01:00[Europe/Stockholm]",
            "converted_result": "1984-10-11T16:00+05:00",
            "converted_named_result": "1984-10-11T01:00:42-10:00[Pacific/Honolulu]",
        }
        ],
    )


def test_string_cypher_executes_temporal_datetime_constructor_recomputes_named_zone_offset_for_new_date() -> None:
    _assert_query_rows(
        "WITH "
        "localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, "
        "datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime "
        "RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        [{"result": "1984-03-28T12:00:42+02:00[Europe/Stockholm]"}],
    )


def test_string_cypher_executes_temporal_datetime_constructor_from_named_zone_datetime_alias() -> None:
    _assert_query_rows(
        "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other "
        "RETURN "
        "datetime({datetime: other}) AS preserved_result, "
        "datetime({datetime: other, timezone: '+05:00'}) AS converted_result, "
        "datetime({datetime: other, day: 28, second: 42}) AS recomputed_result",
        [
        {
            "preserved_result": "1984-10-11T12:00+01:00[Europe/Stockholm]",
            "converted_result": "1984-10-11T16:00+05:00",
            "recomputed_result": "1984-10-28T12:00:42+01:00[Europe/Stockholm]",
        }
        ],
    )


def test_string_cypher_executes_temporal_date_truncate() -> None:
    _assert_query_rows(
        "RETURN date.truncate('decade', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        [{"result": "1980-01-02"}],
    )


def test_string_cypher_executes_temporal_date_truncate_from_aware_datetime_constructor() -> None:
    _assert_query_rows(
        "RETURN date.truncate("
        "'decade', "
        "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), "
        "{day: 2}"
        ") AS result",
        [{"result": "1980-01-02"}],
    )


def test_string_cypher_executes_temporal_datetime_truncate_with_named_zone() -> None:
    _assert_query_rows(
        "RETURN datetime.truncate("
        "'weekYear', "
        "localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
        "{timezone: 'Europe/Stockholm'}"
        ") AS result",
        [{"result": "1983-01-03T00:00+01:00[Europe/Stockholm]"}],
    )


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


def test_string_cypher_executes_temporal_localdatetime_weekyear_truncate_day_override() -> None:
    _assert_query_rows(
        "RETURN localdatetime.truncate("
        "'weekYear', "
        "localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
        "{day: 5}"
        ") AS result",
        [{"result": "1983-01-05T00:00"}],
    )


def test_string_cypher_executes_temporal_time_truncate_with_timezone_override() -> None:
    _assert_query_rows(
        "RETURN time.truncate("
        "'hour', "
        "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
        "{timezone: '+01:00'}"
        ") AS result",
        [{"result": "12:00+01:00"}],
    )


def test_string_cypher_executes_temporal_datetime_hour_truncate_from_localdatetime() -> None:
    _assert_query_rows(
        "RETURN datetime.truncate("
        "'hour', "
        "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
        "{nanosecond: 2}"
        ") AS result",
        [{"result": "1984-10-11T12:00:00.000000002Z"}],
    )


def test_string_cypher_preserves_truncated_fraction_when_overriding_lower_precision_fields() -> None:
    _assert_query_rows(
        "RETURN "
        "datetime.truncate('millisecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS dt_ms, "
        "datetime.truncate('microsecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS dt_us, "
        "time.truncate('microsecond', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS t_us",
        [
        {
            "dt_ms": "1984-10-11T12:31:14.645000002+01:00",
            "dt_us": "1984-10-11T12:31:14.645876002Z",
            "t_us": "12:31:14.645876002+01:00",
        }
        ],
    )


def test_string_cypher_executes_duration_between_with_alias_properties() -> None:
    _assert_query_rows(
        "WITH duration.between(localdatetime('2018-01-01T12:00'), localdatetime('2018-01-02T10:00')) AS dur "
        "RETURN dur, dur.days, dur.seconds, dur.nanosecondsOfSecond",
        [{"dur": "PT22H", "dur.days": 0, "dur.seconds": 79200, "dur.nanosecondsOfSecond": 0}],
    )


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
    "query",
    [
        "MATCH (a {name: 'Andres'})<-[:FATHER]-(child)\nRETURN a.name, {foo: a.name='Andres', kids: collect(child.name)}",
        "MATCH (me: Person)--(you: Person)\nWITH me.age AS age, you\nRETURN age, age + count(you.age)",
        "MATCH (me: Person)--(you: Person)\nRETURN me.age, me.age + count(you.age)",
        "MATCH (me: Person)--(you: Person)\nRETURN me.age AS age, count(you.age) AS cnt\nORDER BY age, age + count(you.age)",
    ],
)
def test_string_cypher_rejects_unsound_multi_source_aggregate_overlap_queries(query: str) -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLValidationError, match="one MATCH source alias at a time"):
        g.gfql(query)


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
    assert result._nodes.to_pandas().to_dict(orient="records") == [
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

    assert exc_info.value.code == ErrorCode.E108
    assert (
        "currently supports only a single WITH collect([distinct] alias) AS list "
        "UNWIND list AS alias MATCH ... RETURN shape"
    ) in exc_info.value.message


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
    "query",
    [
        "MATCH (a:A) MATCH (a)-[:LIKES*1]->()-[:LIKES]->(c) RETURN c.name",
        "MATCH (a:A) MATCH (a)-[:LIKES*2]->()-[:LIKES]->(c) RETURN c.name",
        "MATCH (a:A) MATCH (a)-[:LIKES]->()-[:LIKES*3]->(c) RETURN c.name",
        "MATCH (a:A) MATCH (a)<-[:LIKES]-()-[:LIKES*3]->(c) RETURN c.name",
    ],
)
def test_string_cypher_accepts_nonterminal_variable_length_relationship_patterns(
    query: str,
) -> None:
    """Connected patterns with non-terminal variable-length relationships are now supported."""
    graph = _mk_graph(
        pd.DataFrame({"id": ["a", "b", "c", "d"], "label__A": [True, False, False, False], "name": ["A", "B", "C", "D"]}),
        pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"], "type": ["LIKES", "LIKES", "LIKES"]}),
    )
    result = graph.gfql(query)
    assert result._nodes is not None


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


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) WHERE (n)-[:REL1*2]-() RETURN n",
        "MATCH (n) WHERE (n)-[*2]-() RETURN n",
        "MATCH (n) WHERE (n)<-[:REL1*1..2]-() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1*2]-() AND n.id <> 'a' RETURN n",
    ],
)
def test_string_cypher_failfast_rejects_bounded_variable_length_where_pattern_predicates(query: str) -> None:
    graph = _mk_empty_graph()

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(query)

    assert exc_info.value.code == ErrorCode.E108
    assert "WHERE pattern predicates" in exc_info.value.message


@pytest.mark.parametrize(
    "query",
    [
        "MATCH path = shortestPath((a)-[:KNOWS*]-(b)) RETURN length(path)",
        "MATCH (a), path = shortestPath((a)-[:KNOWS*]-(b)) RETURN a.id",
        "MATCH path = allShortestPaths((a)-[:KNOWS*]-(b)) RETURN length(path)",
    ],
)
def test_string_cypher_failfast_rejects_shortest_path(query: str) -> None:
    """#997: shortestPath/allShortestPaths parse but fail-fast with clear message."""
    graph = _mk_empty_graph()
    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(query)
    assert "shortestpath" in exc_info.value.message.lower() or "allshortestpaths" in exc_info.value.message.lower()


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a)-[:KNOWS]-(b) RETURN not((a)-[:KNOWS]-(b)) AS isNew",
        "MATCH (a) RETURN exists { (a)-[:KNOWS]-() } AS has",
        "MATCH (a) RETURN not exists { (a)-[:KNOWS]-() } AS no",
        "MATCH (a) WHERE exists { (a)-[:KNOWS]-() } RETURN a.id",
    ],
)
def test_string_cypher_failfast_rejects_pattern_existence(query: str) -> None:
    """#998: pattern existence expressions fail-fast with clear message."""
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
    "query",
    [
        "MATCH (n) WHERE (n)-[:R*]->() OR n.id = 'z' RETURN n",
        "MATCH (n) WHERE NOT (n)-[:R*]->() RETURN n",
    ],
)
def test_string_cypher_failfast_rejects_unsupported_mixed_variable_length_where_pattern_predicates(query: str) -> None:
    graph = _mk_empty_graph()

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(query)

    assert exc_info.value.code == ErrorCode.E108
    assert "mixed with generic row expressions" in exc_info.value.message


def test_string_cypher_failfast_rejects_multi_alias_return_star_projection() -> None:
    graph = _mk_empty_graph()

    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql("MATCH (a)-[]->(b) RETURN *")

    assert exc_info.value.code == ErrorCode.E108
    assert "RETURN * currently requires a single MATCH alias" in exc_info.value.message


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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_pandas().to_dict(orient="records") == [
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
    assert sorted(rel_result._nodes.to_dict(orient="records"), key=lambda row: row["r"]) == [
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


def test_string_cypher_supports_graph_functions_on_list_wrapped_entities_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

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
        "MATCH (a) WITH [a, 1] AS list RETURN labels(list[0]) AS l ORDER BY l",
        engine="cudf",
    )
    assert sorted(
        labels_result._nodes.to_pandas().to_dict(orient="records"),
        key=lambda row: (len(row["l"]), row["l"]),
    ) == [
        {"l": "['Foo']"},
        {"l": "['Foo', 'Bar']"},
    ]

    type_result = graph.gfql(
        "MATCH ()-[r]->() WITH [r, 1] AS list RETURN type(list[0]) AS t",
        engine="cudf",
    )
    assert type_result._nodes.to_pandas().to_dict(orient="records") == [{"t": "T"}]


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

    assert result._nodes.to_pandas().to_dict(orient="records") == [{"ln": None, "nn": None}]


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

    assert result._nodes.to_pandas().to_dict(orient="records") == [{"r": None}]


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

    nodes_df = result._nodes.to_pandas() if hasattr(result._nodes, "to_pandas") else result._nodes
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
    assert node_result._nodes.to_pandas().to_dict(orient="records") == [
        {"(list[1]).missing": None, "(list[1]).missingToo": None, "(list[1]).existing": 42},
        {"(list[1]).missing": None, "(list[1]).missingToo": None, "(list[1]).existing": None},
    ]

    rel_result = graph.gfql(
        "MATCH ()-[r]->() WITH [123, r] AS list RETURN (list[1]).missing, (list[1]).missingToo, (list[1]).existing",
        engine="cudf",
    )
    assert rel_result._nodes.to_pandas().to_dict(orient="records") == [
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


def test_string_cypher_supports_property_access_on_list_wrapped_map_values_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    graph = _mk_graph(
        cudf.from_pandas(pd.DataFrame({"id": pd.Series(dtype="object")})),
        cudf.from_pandas(pd.DataFrame({"s": pd.Series(dtype="object"), "d": pd.Series(dtype="object")})),
    )

    result = graph.gfql(
        "WITH [123, {existing: 42, notMissing: null}] AS list RETURN (list[1]).missing, (list[1]).notMissing, (list[1]).existing",
        engine="cudf",
    )

    assert result._nodes.to_pandas().to_dict(orient="records") == [
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


def test_string_cypher_supports_order_by_list_literal_and_subscript_expression() -> None:
    g = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e"],
                "list": [[2, -2], [1, 2], [300, 0], [1, -20], [2, -2, 100]],
                "list2": [[3, -2], [2, -2], [1, -2], [4, -2], [5, -2]],
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = g.gfql(
        "MATCH (a) "
        "WITH a "
        "ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 "
        "LIMIT 3 "
        "RETURN a"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"a": "({list: [300, 0], list2: [1, -2]})"},
        {"a": "({list: [1, 2], list2: [2, -2]})"},
        {"a": "({list: [2, -2], list2: [3, -2]})"},
    ]


def test_string_cypher_supports_order_by_stringified_list_subscript_expression() -> None:
    g = _mk_graph(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e"],
                "list": pd.Series(
                    ["[2, -2]", "[1, 2]", "[300, 0]", "[1, -20]", "[2, -2, 100]"],
                    dtype="string",
                ),
                "list2": pd.Series(
                    ["[3, -2]", "[2, -2]", "[1, -2]", "[4, -2]", "[5, -2]"],
                    dtype="string",
                ),
            }
        ),
        pd.DataFrame({"s": [], "d": []}),
    )

    result = g.gfql(
        "MATCH (a) "
        "WITH a "
        "ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 "
        "LIMIT 3 "
        "RETURN a"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"a": "({list: [300, 0], list2: [1, -2]})"},
        {"a": "({list: [1, 2], list2: [2, -2]})"},
        {"a": "({list: [2, -2], list2: [3, -2]})"},
    ]


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

    with pytest.raises(GFQLValidationError, match="ORDER BY column must exist after RETURN/WITH projection"):
        g.gfql("WITH 1 AS a, 'b' AS b, 3 AS c WITH a, b WITH a ORDER BY a, c RETURN a")


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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [{"a": "({num: 1, text: 'a'})"}]


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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [
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

    assert result._nodes.to_dict(orient="records") == [{"a": "(:A {name: 'alpha'})"}]


def test_string_cypher_rejects_reentry_with_parameterized_limit_and_order() -> None:
    """Regression for 992f2fc1: ParameterRef in LIMIT must not crash _literal_limit_value."""
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
            params={"n": 1},
        )
    assert "order" in exc_info.value.message.lower()


def test_string_cypher_executes_with_match_reentry_limit_shape_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

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
    assert result._nodes.to_pandas().to_dict(orient="records") == [{"a": "(:A {name: 'alpha'})"}]


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

    assert whole_row_output == expected_whole_row_output
    assert carried_columns == expected_columns


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
    assert result._nodes.to_dict(orient="records") == expected


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
    assert result._nodes.to_pandas().to_dict(orient="records") == expected


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
    assert result._nodes.to_pandas().to_dict(orient="records") == [{"bid": "b1", "cid": "c1"}]


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
    assert result._nodes.to_pandas().to_dict(orient="records") == [
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

    assert result._nodes.to_pandas().to_dict(orient="records") == [
        {"messageId": "post2", "messageCreationDate": 20, "postId": "post2", "personId": "viewer"},
        {"messageId": "comment1", "messageCreationDate": 10, "postId": "post1", "personId": "author1"},
    ]


@pytest.mark.parametrize(
    ("graph_factory", "query", "params", "match"),
    [
        (
            _mk_recent_message_reentry_graph_branching,
            "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(message) "
            "WITH message, message.id AS messageId "
            "MATCH (message)-[:REPLY_OF*0..]->(post:Post), (post)-[:HAS_CREATOR]->(person) "
            "RETURN messageId, post.id AS postId, person.id AS personId",
            {"personId": "viewer"},
            "variable-length segments with at most one outgoing match per source row",
        ),
        (
            _mk_multihop_row_binding_cycle_graph,
            "MATCH (a:A)-[r:R*1..2]->(b) RETURN a.id AS aid, b.id AS bid",
            None,
            "do not yet support variable-length relationship aliases",
        ),
        (
            _mk_multihop_row_binding_cycle_graph,
            "MATCH (a:A)-[:R*1..2]-(b) RETURN a.id AS aid, b.id AS bid",
            None,
            "do not yet support undirected variable-length segments",
        ),
        (
            _mk_multihop_row_binding_cycle_graph,
            "MATCH (a:A)-[:R*0..]->(b) RETURN a.id AS aid, b.id AS bid",
            None,
            "currently require terminating variable-length segments",
        ),
    ],
)
def test_string_cypher_failfast_rejects_unsupported_multihop_row_bindings(
    graph_factory: Callable[[], _CypherTestGraph],
    query: str,
    params: Optional[Dict[str, Any]],
    match: str,
) -> None:
    with pytest.raises(GFQLValidationError, match=match):
        graph_factory().gfql(query, params=params)


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


def test_string_cypher_failfast_rejects_with_match_reentry_multiple_whole_row_aliases_with_carried_scalars() -> None:
    query = (
        "MATCH (a:A {id: $seed})-[:R]->(b:B) "
        "WITH a, b, b.id AS bid "
        "MATCH (b)-[:S]->(c:C) "
        "RETURN bid, c.id AS cid"
    )

    with pytest.raises(GFQLValidationError, match="one MATCH source alias at a time"):
        _mk_connected_reentry_carried_scalar_graph().gfql(query, params={"seed": "a1"})


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
    assert result._nodes.to_pandas().to_dict(orient="records") == [{"property": 2}, {"property": 1}]


@pytest.mark.parametrize(
    ("query", "match"),
    [
        (
            _reentry_query(
                "a, a.num AS property",
                return_clause="b.id AS id",
                where_clause="property = b.num",
            ),
            "one MATCH source alias at a time",
        ),
        (
            "MATCH (a:A) "
            "WITH a "
            "ORDER BY a.num DESC "
            "MATCH (a)-->(b) "
            "RETURN b.id AS bid",
            "does not yet preserve prefix WITH row ordering",
        ),
        (
            "MATCH (a:A) "
            "WITH a, a.num AS property "
            "ORDER BY property DESC "
            "MATCH (a)-->(b) "
            "RETURN b.id AS bid",
            "does not yet preserve prefix WITH row ordering",
        ),
    ],
)
def test_string_cypher_failfast_rejects_with_match_reentry_unsupported_shapes(query: str, match: str) -> None:
    with pytest.raises(GFQLValidationError, match=match):
        _mk_reentry_carried_scalar_graph().gfql(query)


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

    assert result._nodes.to_dict(orient="records") == [{"c": "(:C)", "bid": "b"}]


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
    assert result._nodes.to_pandas().to_dict(orient="records") == [{"id": "d"}]


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


def test_string_cypher_failfast_rejects_multi_stage_with_match_reentry_with_intermediate_where() -> None:
    query = (
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "WITH c "
        "MATCH (c)-[:T]->(d:D) "
        "RETURN d.id AS id"
    )

    with pytest.raises(
        GFQLSyntaxError,
        match="Cypher WITH after post-WITH MATCH WHERE is not yet supported",
    ):
        _mk_multi_stage_reentry_graph().gfql(query)


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


def test_multi_alias_with_stage_still_rejected() -> None:
    """WITH multi-alias scalar projections are not yet supported (separate code path)."""
    g = _mk_graph(
        pd.DataFrame({"id": ["a", "b"], "label__A": [True, False], "label__B": [False, True]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["R"]}),
    )
    with pytest.raises(GFQLValidationError, match="one MATCH source alias"):
        g.gfql("MATCH (a:A)-[:R]->(b:B) WITH a.id AS a_id, b.id AS b_id RETURN a_id, b_id")


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


def test_lower_cypher_query_builds_group_by_pipeline() -> None:
    parsed = _parse_query(
        "MATCH (n) RETURN n.division AS division, count(*) AS cnt, max(n.age) AS max_age ORDER BY division ASC, cnt DESC"
    )

    chain = lower_cypher_query(parsed)

    row_call = cast(ASTCall, chain.chain[1])
    with_call = cast(ASTCall, chain.chain[2])
    group_call = cast(ASTCall, chain.chain[3])
    order_call = cast(ASTCall, chain.chain[4])
    assert [row_call.function, with_call.function, group_call.function, order_call.function] == [
        "rows",
        "with_",
        "group_by",
        "order_by",
    ]
    assert with_call.params["items"] == [
        ("division", "n.division"),
        ("__cypher_agg__", "n.age"),
    ]
    assert group_call.params == {
        "keys": ["division"],
        "aggregations": [("cnt", "count"), ("max_age", "max", "__cypher_agg__")],
    }
    assert order_call.params["keys"] == [("division", "asc"), ("cnt", "desc")]


def test_lower_cypher_query_builds_distinct_aggregate_pipeline() -> None:
    parsed = _parse_query(
        "UNWIND [null, 1, null, 2, 1] AS x RETURN count(DISTINCT x) AS cnt, collect(DISTINCT x) AS vals"
    )

    chain = lower_cypher_query(parsed)

    with_call = cast(ASTCall, chain.chain[2])
    group_call = cast(ASTCall, chain.chain[3])
    return_call = cast(ASTCall, chain.chain[4])
    assert [with_call.function, group_call.function, return_call.function] == [
        "with_",
        "group_by",
        "select",
    ]
    assert with_call.params["items"] == [("__cypher_group__", 1), ("__cypher_agg__", "x"), ("__cypher_agg__1", "x")]
    assert group_call.params == {
        "keys": ["__cypher_group__"],
        "aggregations": [("cnt", "count_distinct", "__cypher_agg__"), ("vals", "collect_distinct", "__cypher_agg__1")],
    }


def test_lower_cypher_query_builds_with_aggregate_pipeline() -> None:
    parsed = _parse_query(
        "UNWIND [1, 2, 2] AS x WITH collect(DISTINCT x) AS xs RETURN size(xs) AS n"
    )

    chain = lower_cypher_query(parsed)

    with_call = cast(ASTCall, chain.chain[2])
    group_call = cast(ASTCall, chain.chain[3])
    with_output_call = cast(ASTCall, chain.chain[4])
    final_call = cast(ASTCall, chain.chain[5])

    assert [with_call.function, group_call.function, with_output_call.function, final_call.function] == [
        "with_",
        "group_by",
        "with_",
        "select",
    ]
    assert with_call.params["items"] == [("__cypher_group__", 1), ("__cypher_agg__", "x")]
    assert group_call.params == {
        "keys": ["__cypher_group__"],
        "aggregations": [("xs", "collect_distinct", "__cypher_agg__")],
    }
    assert with_output_call.params["items"] == [("xs", "xs")]
    assert final_call.params["items"] == [("n", "size(xs)")]


def test_lower_cypher_query_builds_match_alias_with_expression_order_pipeline() -> None:
    parsed = _parse_query(
        "MATCH (a) WITH a.name AS name ORDER BY a.name + 'C' ASC LIMIT 2 RETURN name"
    )

    chain = lower_cypher_query(parsed)

    row_call = cast(ASTCall, chain.chain[1])
    with_call = cast(ASTCall, chain.chain[2])
    order_call = cast(ASTCall, chain.chain[3])
    limit_call = cast(ASTCall, chain.chain[4])
    final_call = cast(ASTCall, chain.chain[5])

    assert [row_call.function, with_call.function, order_call.function, limit_call.function, final_call.function] == [
        "rows",
        "with_",
        "order_by",
        "limit",
        "select",
    ]
    assert with_call.params["items"] == [("name", "a.name")]
    assert order_call.params["keys"] == [("(name + 'C')", "asc")]
    assert limit_call.params["value"] == 2
    assert final_call.params["items"] == [("name", "name")]


def test_lower_cypher_query_builds_match_alias_with_aggregate_group_by_pipeline() -> None:
    parsed = _parse_query(
        "MATCH (a) WITH a.name AS name, count(*) AS cnt ORDER BY a.name + 'C' DESC LIMIT 1 RETURN name, cnt"
    )

    chain = lower_cypher_query(parsed)

    row_call = cast(ASTCall, chain.chain[1])
    with_call = cast(ASTCall, chain.chain[2])
    group_call = cast(ASTCall, chain.chain[3])
    order_call = cast(ASTCall, chain.chain[4])
    limit_call = cast(ASTCall, chain.chain[5])
    final_call = cast(ASTCall, chain.chain[6])

    assert [row_call.function, with_call.function, group_call.function, order_call.function, limit_call.function, final_call.function] == [
        "rows",
        "with_",
        "group_by",
        "order_by",
        "limit",
        "select",
    ]
    assert with_call.params["items"] == [("name", "a.name")]
    assert group_call.params == {
        "keys": ["name"],
        "aggregations": [("cnt", "count")],
    }
    assert order_call.params["keys"] == [("(name + 'C')", "desc")]
    assert limit_call.params["value"] == 1
    assert final_call.params["items"] == [("name", "name"), ("cnt", "cnt")]


def test_lower_cypher_query_maps_count_distinct_edge_alias_to_identity() -> None:
    parsed = _parse_query("MATCH (a)-[r]->(b) RETURN count(DISTINCT r)")

    chain = lower_cypher_query(parsed)

    row_call = cast(ASTCall, chain.chain[3])
    with_call = cast(ASTCall, chain.chain[4])
    group_call = cast(ASTCall, chain.chain[5])
    assert row_call.params == {"table": "edges", "source": "r"}
    assert with_call.params["items"] == [("__cypher_group__", 1), ("__cypher_agg__", "__gfql_edge_index_0__")]
    assert group_call.params == {
        "keys": ["__cypher_group__"],
        "aggregations": [("count(DISTINCT r)", "count_distinct", "__cypher_agg__")],
    }


def test_gfql_executes_top_level_unwind_query() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("UNWIND [3, 1, 2] AS x RETURN x ORDER BY x ASC LIMIT 2")

    assert result._nodes.to_dict(orient="records") == [{"x": 1}, {"x": 2}]


def test_gfql_executes_match_then_unwind_query() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "vals": [[2, 1], [3]],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n) UNWIND n.vals AS v RETURN v ORDER BY v ASC"
    )

    assert result._nodes.to_dict(orient="records") == [{"v": 1}, {"v": 2}, {"v": 3}]


def test_gfql_executes_aggregate_return_query() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "division": ["x", "x", "y"],
            "age": [3, 7, 4],
        }
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (n) RETURN n.division AS division, count(*) AS cnt, max(n.age) AS max_age ORDER BY division ASC"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"division": "x", "cnt": 2, "max_age": 7},
        {"division": "y", "cnt": 1, "max_age": 4},
    ]


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

    assert result._nodes.to_pandas().to_dict(orient="records") == [
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

    assert result._nodes.to_pandas().to_dict(orient="records") == [
        {"a": "(:L1)", "count(*)": 1},
        {"a": "(:L2)", "count(*)": 1},
        {"a": "(:L3)", "count(*)": 1},
    ]


def test_gfql_executes_boolean_list_comprehension_order_check_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(pd.DataFrame({"id": []}))
    edges = cudf.from_pandas(pd.DataFrame({"s": [], "d": []}))

    result = _mk_graph(nodes, edges).gfql(
        "WITH [true, false] AS values\n"
        "WITH values, size(values) AS numOfValues\n"
        "UNWIND values AS value\n"
        "WITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n"
        "  ORDER BY value\n"
        "WITH numOfValues, collect(x) AS orderedX\n"
        "RETURN orderedX = range(0, numOfValues-1) AS equal",
        engine="cudf",
    )

    assert result._nodes.to_pandas().to_dict(orient="records") == [{"equal": True}]


def test_gfql_executes_integer_list_comprehension_order_check_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(pd.DataFrame({"id": []}))
    edges = cudf.from_pandas(pd.DataFrame({"s": [], "d": []}))

    result = _mk_graph(nodes, edges).gfql(
        "WITH [351, -3974856, 93, -3, 123, 0, 3, -2, 20934587, 1, 20934585, 20934586, -10] AS values\n"
        "WITH values, size(values) AS numOfValues\n"
        "UNWIND values AS value\n"
        "WITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n"
        "  ORDER BY value\n"
        "WITH numOfValues, collect(x) AS orderedX\n"
        "RETURN orderedX = range(0, numOfValues-1) AS equal",
        engine="cudf",
    )

    assert result._nodes.to_pandas().to_dict(orient="records") == [{"equal": True}]


def test_gfql_executes_nested_list_comprehension_order_check_on_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    nodes = cudf.from_pandas(pd.DataFrame({"id": []}))
    edges = cudf.from_pandas(pd.DataFrame({"s": [], "d": []}))

    result = _mk_graph(nodes, edges).gfql(
        "WITH [[2, 2], [2, -2], [1, 2], [], [1], [300, 0], [1, -20], [2, -2, 100]] AS values\n"
        "WITH values, size(values) AS numOfValues\n"
        "UNWIND values AS value\n"
        "WITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n"
        "  ORDER BY value\n"
        "WITH numOfValues, collect(x) AS orderedX\n"
        "RETURN orderedX = range(0, numOfValues-1) AS equal",
        engine="cudf",
    )

    assert result._nodes.to_pandas().to_dict(orient="records") == [{"equal": True}]


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
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql("MATCH (a) RETURN count(DISTINCT a.name) AS cnt")

    assert result._nodes.to_dict(orient="records") == [{"cnt": 0}]


def test_cypher_to_gfql_rejects_multi_source_aggregate_expr() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        cypher_to_gfql("MATCH (a)-[r]->(b) RETURN a.id AS a_id, max(b.score) AS max_b")

    assert exc_info.value.code == ErrorCode.E108


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


def test_gfql_executes_size_null_and_sqrt_constant_expressions() -> None:
    _assert_query_rows(
        "WITH null AS l RETURN size(l) AS size_l, size(null) AS size_null, sqrt(12.96) AS root",
        [{"size_l": None, "size_null": None, "root": 3.6}],
    )


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

    assert result._nodes.to_dict(orient="records") == [
        {"i": "(:TextNode {var: 'text'})"},
        {"i": "(:IntNode {var: 0})"},
    ]


def test_gfql_executes_with_where_null_filter_over_mixed_type_compare_on_cudf() -> None:
    pytest.importorskip("cudf")

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
        "MATCH (:Root {name: 'x'})-->(i:TextNode)\n"
        "WITH i\n"
        "WHERE i.var > 'te'\n"
        "RETURN i",
        engine="cudf",
    )

    assert result._nodes.to_pandas().to_dict(orient="records") == [
        {"i": "(:TextNode {var: 'text'})"}
    ]


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

    assert result._nodes.to_pandas().to_dict(orient="records") == [
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


def test_string_cypher_multi_alias_with_distinct_scalar_projection() -> None:
    """IC-4 shape: MATCH multi-pattern, WITH DISTINCT two aliases, RETURN scalars (#880)."""
    graph = _mk_ic4_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
        "WITH DISTINCT tag, post "
        "RETURN tag.name AS tagName, post.id AS postId "
        "ORDER BY tagName, postId",
        params={"pid": "p1"},
    )
    assert result._nodes.to_dict(orient="records") == [
        {"tagName": "TagA", "postId": "post1"},
        {"tagName": "TagA", "postId": "post2"},
        {"tagName": "TagB", "postId": "post3"},
    ]


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


def test_string_cypher_multi_alias_with_distinct_property_where() -> None:
    """Multi-alias WITH DISTINCT + WHERE on alias property (#880)."""
    graph = _mk_ic4_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
        "WITH DISTINCT tag, post "
        'WHERE tag.name = "TagA" '
        "RETURN post.id AS postId ORDER BY postId",
        params={"pid": "p1"},
    )
    assert result._nodes.to_dict(orient="records") == [
        {"postId": "post1"},
        {"postId": "post2"},
    ]


def test_string_cypher_multi_alias_with_distinct_count_projection() -> None:
    """IC-4 shape: WITH DISTINCT two aliases, then count aggregation (#880)."""
    graph = _mk_ic4_shape_graph()
    result = graph.gfql(
        "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
        "WITH DISTINCT tag, post "
        "RETURN tag.name AS tagName, count(post) AS postCount "
        "ORDER BY postCount DESC, tagName ASC",
        params={"pid": "p1"},
    )
    assert result._nodes.to_dict(orient="records") == [
        {"tagName": "TagA", "postCount": 2},
        {"tagName": "TagB", "postCount": 1},
    ]


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
    """Simplest #996 shape: connected first MATCH + OPTIONAL MATCH + mixed RETURN."""
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
    # a->b via R1, a->c via T exists => cid='c'
    # a->d via R1, a->c via T exists => cid='c'
    assert len(rows) == 2
    assert rows[0]["aid"] == "a"
    assert rows[0]["bid"] == "b"
    assert rows[0]["cid"] == "c"
    assert rows[1]["aid"] == "a"
    assert rows[1]["bid"] == "d"
    assert rows[1]["cid"] == "c"


def test_issue_996_connected_match_optional_match_case_null() -> None:
    """#996 with CASE expression over nullable optional binding."""
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    # b has a T-edge to c, but c does not
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
    """Graph for #996 amplification tests.

    Nodes: p1, p2, p3, p4 with ``score`` and ``label__Person``.
    Edges:
      p1 -[FRIEND]-> p2
      p1 -[FRIEND]-> p3
      p2 -[KNOWS]-> p3     (optional edge present for p2)
      p3 has no KNOWS edge (optional will be null for p3)
      p1 -[FRIEND]-> p4
      p4 -[KNOWS]-> p1     (optional present, different target)
    """
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


def test_issue_996_type_function_on_optional_edge() -> None:
    """type() on an optional relationship alias — exercises row pipeline function dispatch."""
    g = _mk_996_rich_graph()

    result = g.gfql(
        "MATCH (a)-[r1:FRIEND]->(b) "
        "OPTIONAL MATCH (b)-[r2:KNOWS]->(c) "
        "RETURN a.id AS aid, b.id AS bid, type(r2) AS t"
    )

    rows = sorted(result._nodes.to_dict(orient="records"), key=lambda r: str(r["bid"]))
    assert len(rows) == 3
    # p2 has KNOWS->p3
    assert rows[0]["bid"] == "p2"
    assert rows[0]["t"] == "KNOWS"
    # p3 has no KNOWS edge — null may surface as None or NaN after left join
    assert rows[1]["bid"] == "p3"
    assert rows[1]["t"] is None or (isinstance(rows[1]["t"], float) and rows[1]["t"] != rows[1]["t"])
    # p4 has KNOWS->p1
    assert rows[2]["bid"] == "p4"
    assert rows[2]["t"] == "KNOWS"


def test_issue_996_coalesce_on_optional_property() -> None:
    """coalesce() over an optional property — null fallback via row pipeline."""
    g = _mk_996_rich_graph()  # fresh graph — do not reuse across tests

    result = g.gfql(
        "MATCH (a)-[r1:FRIEND]->(b) "
        "OPTIONAL MATCH (b)-[r2:KNOWS]->(c) "
        "RETURN b.id AS bid, coalesce(c.id, 'none') AS target"
    )

    rows = sorted(result._nodes.to_dict(orient="records"), key=lambda r: str(r["bid"]))
    assert rows[0] == {"bid": "p2", "target": "p3"}
    assert rows[1] == {"bid": "p3", "target": "none"}
    assert rows[2] == {"bid": "p4", "target": "p1"}


def test_issue_996_arithmetic_in_return() -> None:
    """Arithmetic expression over base + optional properties."""
    g = _mk_996_rich_graph()

    result = g.gfql(
        "MATCH (a)-[r1:FRIEND]->(b) "
        "OPTIONAL MATCH (b)-[r2:KNOWS]->(c) "
        "RETURN b.id AS bid, b.score + 1 AS bumped"
    )

    rows = sorted(result._nodes.to_dict(orient="records"), key=lambda r: str(r["bid"]))
    assert rows[0] == {"bid": "p2", "bumped": 21}
    assert rows[1] == {"bid": "p3", "bumped": 31}
    assert rows[2] == {"bid": "p4", "bumped": 41}


def test_issue_996_order_by_desc() -> None:
    """ORDER BY DESC on a base-alias column."""
    g = _mk_996_rich_graph()

    result = g.gfql(
        "MATCH (a)-[r1:FRIEND]->(b) "
        "OPTIONAL MATCH (b)-[r2:KNOWS]->(c) "
        "RETURN b.id AS bid, c.id AS cid "
        "ORDER BY bid DESC"
    )

    rows = result._nodes.to_dict(orient="records")
    assert [r["bid"] for r in rows] == ["p4", "p3", "p2"]


def test_issue_996_limit() -> None:
    """LIMIT on connected optional match results."""
    g = _mk_996_rich_graph()

    result = g.gfql(
        "MATCH (a)-[r1:FRIEND]->(b) "
        "OPTIONAL MATCH (b)-[r2:KNOWS]->(c) "
        "RETURN b.id AS bid "
        "ORDER BY bid "
        "LIMIT 2"
    )

    rows = result._nodes.to_dict(orient="records")
    assert len(rows) == 2
    assert rows[0]["bid"] == "p2"
    assert rows[1]["bid"] == "p3"


def test_issue_996_skip_limit() -> None:
    """SKIP + LIMIT on connected optional match results."""
    g = _mk_996_rich_graph()

    result = g.gfql(
        "MATCH (a)-[r1:FRIEND]->(b) "
        "OPTIONAL MATCH (b)-[r2:KNOWS]->(c) "
        "RETURN b.id AS bid "
        "ORDER BY bid "
        "SKIP 1 LIMIT 1"
    )

    rows = result._nodes.to_dict(orient="records")
    assert rows == [{"bid": "p3"}]


def test_issue_996_distinct() -> None:
    """DISTINCT deduplicates when multiple base rows map to the same optional."""
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
    # c appears for y=b, null for y=c — DISTINCT should collapse duplicates
    # Null may surface as None or NaN after left join.
    non_null_zids = [r["zid"] for r in rows if r["zid"] is not None and not (isinstance(r["zid"], float) and r["zid"] != r["zid"])]
    null_count = len(rows) - len(non_null_zids)
    assert sorted(non_null_zids) == ["c"]
    assert null_count == 1


def test_issue_996_no_optional_matches() -> None:
    """When OPTIONAL MATCH matches nothing, all optional aliases are null."""
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
    """When every base row has an optional match, no null-fill needed."""
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
    """Multiple optional matches for a single base row produce multiple output rows."""
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
    """Property access on a null optional node returns null, not an error."""
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
    """IS NOT NULL check on an optional edge alias."""
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
    """When the base MATCH returns no rows, the result is empty."""
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
    """Two shared aliases between MATCH and OPTIONAL MATCH — composite join key."""
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
    edges = pd.DataFrame({
        "s": ["a", "b", "a"],
        "d": ["b", "c", "c"],
        "type": ["R1", "R1", "T"],
    })
    g = _mk_graph(nodes, edges)

    # Both a and b are shared between base and optional.
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
    # a->b via R1, a->b via T? No — T goes a->c. So (a,b) has no T edge.
    # a->c via R1? No — only R1 edges are a->b and b->c. So base has (a,b) and (b,c).
    # (a,b): optional T from a->b? No T edge a->b exists. has_t=false
    # (b,c): optional T from b->c? No T edge b->c exists. has_t=false
    assert len(rows) == 2
    assert rows[0] == {"aid": "a", "bid": "b", "has_t": False}
    assert rows[1] == {"aid": "b", "bid": "c", "has_t": False}


def test_issue_996_integer_node_ids() -> None:
    """Integer node IDs — join must handle non-string keys."""
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
    """Non-default node ID column name (nid instead of id)."""
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
    """Optional MATCH with a longer path (two hops) — multiple optional-only aliases."""
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
