import pandas as pd
import pytest
from typing import cast

from graphistry.compute.ast import ASTCall, ASTNode, ASTEdgeForward, ASTEdgeReverse, ASTEdgeUndirected
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.predicates.is_in import IsIn
from graphistry.compute.gfql.same_path_types import col, compare
from graphistry.compute.gfql.cypher import (
    compile_cypher,
    cypher_to_gfql,
    lower_cypher_query,
    lower_match_clause,
    lower_match_query,
    parse_cypher,
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


def test_lower_match_clause_to_gfql_ops() -> None:
    parsed = parse_cypher(
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
    parsed = parse_cypher(query)
    assert parsed.match is not None
    ops = lower_match_clause(parsed.match)
    assert isinstance(ops[1], edge_type)


def test_lower_match_clause_relationship_type_alternation_uses_is_in_predicate() -> None:
    parsed = parse_cypher("MATCH (n)-[r:KNOWS|HATES]->(x) RETURN r")
    assert parsed.match is not None

    ops = lower_match_clause(parsed.match)

    assert isinstance(ops[1], ASTEdgeForward)
    assert isinstance(ops[1].edge_match, dict)
    type_predicate = ops[1].edge_match["type"]
    assert isinstance(type_predicate, IsIn)
    assert type_predicate.options == ["KNOWS", "HATES"]


def test_lower_match_clause_stitches_connected_comma_patterns() -> None:
    parsed = parse_cypher("MATCH (a)-[:A]->(b), (b)-[:B]->(c) RETURN c")
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
    parsed = parse_cypher("MATCH (a)-[:R1]->(b), (c)<-[:R2]-(b) RETURN c")
    assert parsed.match is not None

    ops = lower_match_clause(parsed.match)

    assert len(ops) == 5
    assert isinstance(ops[1], ASTEdgeForward)
    assert isinstance(ops[3], ASTEdgeForward)
    assert cast(ASTNode, ops[2])._name == "b"
    assert cast(ASTNode, ops[4])._name == "c"


def test_lower_match_clause_rejects_disconnected_comma_patterns() -> None:
    parsed = parse_cypher("MATCH (a)-[:A]->(b), (c)-[:B]->(d) RETURN d")
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
    parsed = parse_cypher("MATCH (a {name: 'A'}), (c {name: 'C'}) MATCH (a)-->(b) RETURN b")

    with pytest.raises(GFQLValidationError, match="must participate in the final connected MATCH pattern"):
        lower_match_query(parsed)


def test_lower_match_query_rewrites_duplicate_node_aliases_to_internal_identity_checks() -> None:
    parsed = parse_cypher("MATCH (n)-[r]-(n) RETURN count(r)")

    lowered = lower_match_query(parsed)

    assert len(lowered.query) == 3
    repeated = cast(ASTNode, lowered.query[0])
    assert isinstance(repeated, ASTNode)
    assert repeated._name is not None
    assert repeated._name.startswith("__cypher_aliasdup_n")
    assert lowered.where == [compare(col("n", "id"), "==", col(repeated._name, "id"))]


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

    parsed = parse_cypher("MATCH (p:Person {name: 'Alice'})-[r:FOLLOWS]->(q:Person) RETURN p, q")
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

    parsed = parse_cypher("MATCH (p:Person {name: 'Alice'})-[r:FOLLOWS]->(q:Person) RETURN p, q")
    assert parsed.match is not None
    ops = lower_match_clause(parsed.match)
    result = _mk_graph(nodes, edges).gfql(ops)

    assert sorted(result._nodes["id"].tolist()) == ["a", "b"]
    assert result._edges[["s", "d", "type"]].to_dict(orient="records") == [
        {"s": "a", "d": "b", "type": "FOLLOWS"}
    ]


def test_lower_match_clause_requires_parameters() -> None:
    parsed = parse_cypher("MATCH (p {id: $person_id}) RETURN p")

    with pytest.raises(GFQLValidationError) as exc_info:
        assert parsed.match is not None
        lower_match_clause(parsed.match)

    assert exc_info.value.code == ErrorCode.E105


def test_lower_match_clause_rejects_multi_label_nodes() -> None:
    parsed = parse_cypher("MATCH (p:Person:Admin) RETURN p")

    with pytest.raises(GFQLValidationError) as exc_info:
        assert parsed.match is not None
        lower_match_clause(parsed.match)

    assert exc_info.value.code == ErrorCode.E108


def test_lower_match_query_folds_literal_where_into_filter_dicts() -> None:
    parsed = parse_cypher("MATCH (p)-[r]->(q) WHERE p.name = 'Alice' AND q.name = 'Bob' RETURN p, q")

    lowered = lower_match_query(parsed)

    assert lowered.where == []
    assert isinstance(lowered.query[0], ASTNode)
    assert isinstance(lowered.query[2], ASTNode)
    assert lowered.query[0].filter_dict == {"name": "Alice"}
    assert lowered.query[2].filter_dict == {"name": "Bob"}


def test_lower_match_query_emits_same_path_where_for_alias_comparisons() -> None:
    parsed = parse_cypher("MATCH (p)-[r]->(q) WHERE p.team = q.team RETURN p, q")

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

    parsed = parse_cypher("MATCH (p)-[r]->(q) WHERE p.team = q.team RETURN p, q")
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

    parsed = parse_cypher("MATCH (p)-[r]->(q) WHERE p.deleted IS NULL AND q.name IS NOT NULL RETURN p, q")
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
    parsed = parse_cypher(
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


def test_string_cypher_formats_single_node_entity_projection_with_alias() -> None:
    nodes = pd.DataFrame({"id": ["a"], "type": ["A"]})
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql("MATCH (a) RETURN a AS ColumnName")

    assert result._nodes.to_dict(orient="records") == [{"ColumnName": "(:A)"}]


def test_compile_cypher_records_mixed_whole_row_projection_plan() -> None:
    compiled = compile_cypher("MATCH (p:Person) RETURN p AS person, p.name AS person_name")

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
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "UNWIND [date({year: 1910, month: 5, day: 6}), date({year: 1980, month: 10, day: 24})] AS dates "
        "WITH dates ORDER BY dates ASC LIMIT 2 RETURN dates"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"dates": "1910-05-06"},
        {"dates": "1980-10-24"},
    ]


def test_string_cypher_parses_week_date_literals() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("RETURN date({year: 1817, week: 1, dayOfWeek: 2}) AS d")

    assert result._nodes.to_dict(orient="records") == [{"d": "1816-12-31"}]


def test_string_cypher_parses_localtime_compact_literals() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("RETURN localtime('214032.142') AS result")

    assert result._nodes.to_dict(orient="records") == [{"result": "21:40:32.142"}]


def test_string_cypher_parses_datetime_named_zone_literals() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("RETURN datetime('2015-07-21T21:40:32.142[Europe/London]') AS result")

    assert result._nodes.to_dict(orient="records") == [
        {"result": "2015-07-21T21:40:32.142+01:00[Europe/London]"}
    ]


def test_string_cypher_normalizes_time_offset_seconds() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("RETURN time({hour: 12, minute: 34, second: 56, timezone: '+02:05:00'}) AS result")

    assert result._nodes.to_dict(orient="records") == [{"result": "12:34:56+02:05"}]


def test_string_cypher_defaults_datetime_map_to_utc() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31}) AS result")

    assert result._nodes.to_dict(orient="records") == [{"result": "1984-10-11T12:31Z"}]


def test_string_cypher_parses_datetime_map_with_quoted_offset_timezone() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS result")

    assert result._nodes.to_dict(orient="records") == [{"result": "1984-10-11T12:00+01:00"}]


def test_string_cypher_defaults_time_map_to_utc() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("RETURN time({hour: 12, minute: 31, second: 14}) AS result")

    assert result._nodes.to_dict(orient="records") == [{"result": "12:31:14Z"}]


def test_string_cypher_supports_nested_temporal_base_date_overrides() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("RETURN date({date: date('1816-12-31'), year: 1817, week: 2}) AS d")

    assert result._nodes.to_dict(orient="records") == [{"d": "1817-01-07"}]


def test_string_cypher_executes_temporal_date_casts_from_with_alias() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "WITH date({year: 1984, month: 11, day: 11}) AS other "
        "RETURN date({date: other, year: 28}) AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": "0028-11-11"}]


def test_string_cypher_executes_temporal_datetime_casts_from_with_alias() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "WITH date({year: 1984, month: 10, day: 11}) AS other "
        "RETURN datetime({date: other, hour: 10, minute: 10, second: 10}) AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": "1984-10-11T10:10:10Z"}]


def test_string_cypher_executes_temporal_date_cast_from_aware_datetime_alias() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "WITH datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'}) AS other "
        "RETURN date(other) AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": "1984-11-11"}]


def test_string_cypher_executes_temporal_localtime_cast_from_aware_time_alias() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "WITH time({hour: 12, minute: 31, second: 14, nanosecond: 645876, timezone: '+01:00'}) AS other "
        "RETURN localtime(other) AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": "12:31:14.000645876"}]


def test_string_cypher_executes_temporal_localdatetime_cast_from_aware_datetime_alias() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other "
        "RETURN localdatetime({datetime: other}) AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": "1984-10-11T12:00"}]


def test_string_cypher_executes_temporal_date_truncate() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "RETURN date.truncate('decade', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": "1980-01-02"}]


def test_string_cypher_executes_temporal_date_truncate_from_aware_datetime_constructor() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "RETURN date.truncate("
        "'decade', "
        "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), "
        "{day: 2}"
        ") AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": "1980-01-02"}]


def test_string_cypher_executes_temporal_datetime_truncate_with_named_zone() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "RETURN datetime.truncate("
        "'weekYear', "
        "localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
        "{timezone: 'Europe/Stockholm'}"
        ") AS result"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"result": "1983-01-03T00:00+01:00[Europe/Stockholm]"}
    ]


def test_string_cypher_executes_temporal_localdatetime_weekyear_truncate_day_override() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "RETURN localdatetime.truncate("
        "'weekYear', "
        "localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
        "{day: 5}"
        ") AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": "1983-01-05T00:00"}]


def test_string_cypher_executes_temporal_time_truncate_with_timezone_override() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "RETURN time.truncate("
        "'hour', "
        "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
        "{timezone: '+01:00'}"
        ") AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": "12:00+01:00"}]


def test_string_cypher_executes_temporal_datetime_hour_truncate_from_localdatetime() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "RETURN datetime.truncate("
        "'hour', "
        "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), "
        "{nanosecond: 2}"
        ") AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": "1984-10-11T12:00:00.000000002Z"}]


def test_string_cypher_executes_duration_between_with_alias_properties() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "WITH duration.between(localdatetime('2018-01-01T12:00'), localdatetime('2018-01-02T10:00')) AS dur "
        "RETURN dur, dur.days, dur.seconds, dur.nanosecondsOfSecond"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"dur": "PT22H", "dur.days": 0, "dur.seconds": 79200, "dur.nanosecondsOfSecond": 0}
    ]


def test_string_cypher_executes_negative_duration_between_day_time_components() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "RETURN duration.between("
        "localdatetime('2015-07-21T21:40:32.142'), "
        "date('2015-06-24')"
        ") AS duration"
    )

    assert result._nodes.to_dict(orient="records") == [{"duration": "P-27DT-21H-40M-32.142S"}]


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
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(query)

    assert result._nodes.to_dict(orient="records") == [{"duration": expected}]


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
    )
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
    )
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql(
        "MATCH (a) WITH a, a.time AS time WITH a, time ORDER BY time ASC LIMIT 3 RETURN a, time"
    )

    assert result._nodes.to_dict(orient="records") == [
        {"a": "(:A {time: '12:35:15+05:00'})", "time": "12:35:15+05:00"},
        {"a": "(:A {time: '12:30:14.645876123+01:01'})", "time": "12:30:14.645876123+01:01"},
        {"a": "(:A {time: '12:31:14.645876123+01:00'})", "time": "12:31:14.645876123+01:00"},
    ]


def test_string_cypher_formats_list_literal_strings_in_entity_projection() -> None:
    nodes = pd.DataFrame(
        {
            "id": ["a", "b"],
            "label__A": [True, True],
            "list": ["[1, 2]", "[2, -2]"],
        }
    )
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
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("UNWIND [1, 3, 2] AS ints WITH ints ORDER BY ints DESC LIMIT 2 RETURN ints")

    assert result._nodes.to_dict(orient="records") == [{"ints": 3}, {"ints": 2}]


def test_string_cypher_executes_multiple_with_stages() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("WITH 1 AS a, 'b' AS b WITH a ORDER BY a ASCENDING WITH a RETURN a")

    assert result._nodes.to_dict(orient="records") == [{"a": 1}]


def test_string_cypher_rejects_out_of_scope_order_by_after_multiple_with_stages() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    with pytest.raises(GFQLValidationError, match="ORDER BY column must exist after RETURN/WITH projection"):
        g.gfql("WITH 1 AS a, 'b' AS b, 3 AS c WITH a, b WITH a ORDER BY a, c RETURN a")


def test_string_cypher_executes_row_column_expression_order_after_with() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("UNWIND [1, 2, 3] AS a WITH a ORDER BY a + 2 DESC, a ASC LIMIT 1 RETURN a")

    assert result._nodes.to_dict(orient="records") == [{"a": 3}]


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


def test_cypher_to_gfql_rejects_multi_alias_projection() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        cypher_to_gfql("MATCH (p)-[r]->(q) RETURN p.id, q.id")

    assert exc_info.value.code == ErrorCode.E108


def test_compile_cypher_tracks_seeded_top_level_row_query() -> None:
    compiled = compile_cypher("UNWIND [1, 2, 3] AS x RETURN x ORDER BY x DESC LIMIT 2")

    assert compiled.seed_rows is True
    first = cast(ASTCall, compiled.chain.chain[0])
    second = cast(ASTCall, compiled.chain.chain[1])
    assert isinstance(first, ASTCall)
    assert isinstance(second, ASTCall)
    assert first.function == "rows"
    assert second.function == "unwind"
    assert second.params == {"expr": "[1, 2, 3]", "as_": "x"}


def test_lower_cypher_query_builds_group_by_pipeline() -> None:
    parsed = parse_cypher(
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
    parsed = parse_cypher(
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
    parsed = parse_cypher(
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
    parsed = parse_cypher(
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
    parsed = parse_cypher(
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
    parsed = parse_cypher("MATCH (a)-[r]->(b) RETURN count(DISTINCT r)")

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
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("UNWIND [null, 1, null, 2, 1] AS x RETURN count(DISTINCT x) AS cnt, collect(DISTINCT x) AS vals")

    assert result._nodes.to_dict(orient="records") == [{"cnt": 2, "vals": [1, 2]}]


def test_gfql_executes_collect_distinct_all_null_return_query() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("UNWIND [null, null] AS x RETURN collect(DISTINCT x) AS c")

    assert result._nodes.to_dict(orient="records") == [{"c": []}]


def test_gfql_executes_count_distinct_missing_property_as_zero() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame({"s": [], "d": []})

    result = _mk_graph(nodes, edges).gfql("MATCH (a) RETURN count(DISTINCT a.name) AS cnt")

    assert result._nodes.to_dict(orient="records") == [{"cnt": 0}]


def test_cypher_to_gfql_rejects_multi_source_aggregate_expr() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        cypher_to_gfql("MATCH (a)-[r]->(b) RETURN a.id AS a_id, max(b.id) AS max_b")

    assert exc_info.value.code == ErrorCode.E108


def test_gfql_executes_top_level_quantifier_expression() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("RETURN none(x IN [true, false] WHERE x) AS result")

    assert result._nodes.to_dict(orient="records") == [{"result": False}]


def test_gfql_executes_top_level_membership_and_null_expression() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("RETURN 3 IN [1, 2, 3] AS hit, null IS NULL AS empty")

    assert result._nodes.to_dict(orient="records") == [{"hit": True, "empty": True}]


def test_gfql_executes_top_level_xor_expression_and_precedence() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "RETURN "
        "true XOR false AS tf, "
        "false XOR false AS ff, "
        "true XOR null AS tn, "
        "true OR true XOR true AS or_xor, "
        "true XOR false AND false AS xor_and"
    )

    assert result._nodes.to_dict(orient="records") == [
        {
            "tf": True,
            "ff": False,
            "tn": None,
            "or_xor": True,
            "xor_and": True,
        }
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


def test_gfql_executes_with_aggregate_precedence_pipeline() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql(
        "UNWIND [true, false, null] AS a "
        "UNWIND [true, false, null] AS b "
        "UNWIND [true, false, null] AS c "
        "WITH collect((a OR b XOR c) = (a OR (b XOR c))) AS eq, "
        "collect((a OR b XOR c) <> ((a OR b) XOR c)) AS neq "
        "RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result"
    )

    assert result._nodes.to_dict(orient="records") == [{"result": True}]


def test_gfql_executes_top_level_list_comprehension_expression() -> None:
    g = _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

    result = g.gfql("RETURN [x IN [1, 2, 3] WHERE x > 1 | x + 10] AS vals")

    assert result._nodes.to_dict(orient="records") == [{"vals": [12, 13]}]
