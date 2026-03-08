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
    assert chain.chain[2].params["items"] == [("person_name", "name")]
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
    assert chain.chain[2].params["items"] == [("person_name", "name")]


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
