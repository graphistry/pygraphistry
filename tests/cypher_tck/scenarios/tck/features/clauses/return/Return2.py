from graphistry.compute import e_forward, e_undirected, n

from tests.cypher_tck.models import Expected, GraphFixture, Scenario
from tests.cypher_tck.parse_cypher import graph_fixture_from_create

from tests.cypher_tck.scenarios.fixtures import (
    MATCH5_GRAPH,
    MATCH7_GRAPH_SINGLE,
    MATCH7_GRAPH_AB,
    MATCH7_GRAPH_ABC,
    MATCH7_GRAPH_REL,
    MATCH7_GRAPH_X,
    MATCH7_GRAPH_AB_X,
    MATCH7_GRAPH_LABELS,
    MATCH7_GRAPH_PLAYER_TEAM_BOTH,
    MATCH7_GRAPH_PLAYER_TEAM_SINGLE,
    MATCH7_GRAPH_PLAYER_TEAM_DIFF,
    WITH_ORDERBY4_GRAPH,
    BINARY_TREE_1_GRAPH,
    BINARY_TREE_2_GRAPH,
)


SCENARIOS = [
    Scenario(
        key="return2-1",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[1] Arithmetic expressions should propagate null values",
        cypher="RETURN 1 + (2 - (3 * (4 / (5 ^ (6 % null))))) AS a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN expression evaluation and null arithmetic semantics are not supported",
        tags=("return", "expression", "null", "xfail"),
    ),

    Scenario(
        key="return2-2",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[2] Returning a node property value",
        cypher="MATCH (a)\nRETURN a.num",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a.num": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN property projections are not supported",
        tags=("return", "property", "xfail"),
    ),

    Scenario(
        key="return2-3",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[3] Missing node property should become null",
        cypher="MATCH (a)\nRETURN a.name",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a.name": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN property projections and null semantics are not supported",
        tags=("return", "property", "null", "xfail"),
    ),

    Scenario(
        key="return2-4",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[4] Returning a relationship property value",
        cypher="MATCH ()-[r]->()\nRETURN r.num",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T {num: 1}]->()
            """
        ),
        expected=Expected(
            rows=[
                {"r.num": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN relationship property projections are not supported",
        tags=("return", "relationship", "property", "xfail"),
    ),

    Scenario(
        key="return2-5",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[5] Missing relationship property should become null",
        cypher="MATCH ()-[r]->()\nRETURN r.name2",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T {name: 1}]->()
            """
        ),
        expected=Expected(
            rows=[
                {"r.name2": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN relationship property projections and null semantics are not supported",
        tags=("return", "relationship", "property", "null", "xfail"),
    ),

    Scenario(
        key="return2-6",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[6] Adding a property and a literal in projection",
        cypher="MATCH (a)\nRETURN a.num + 1 AS foo",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"foo": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN arithmetic expression evaluation is not supported",
        tags=("return", "expression", "arithmetic", "xfail"),
    ),

    Scenario(
        key="return2-7",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[7] Adding list properties in projection",
        cypher="MATCH (a)\nRETURN a.list2 + a.list1 AS foo",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "list1": [1, 2, 3], "list2": [4, 5]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"foo": "[4, 5, 1, 2, 3]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN list expressions and list property projections are not supported",
        tags=("return", "list", "expression", "xfail"),
    ),

    Scenario(
        key="return2-8",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[8] Returning label predicate expression",
        cypher="MATCH (n)\nRETURN (n:Foo)",
        graph=graph_fixture_from_create(
            """
            CREATE (), (:Foo)
            """
        ),
        expected=Expected(
            rows=[
                {"(n:Foo)": "true"},
                {"(n:Foo)": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN label predicate expressions are not supported",
        tags=("return", "label-predicate", "xfail"),
    ),

    Scenario(
        key="return2-9",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[9] Returning a projected map",
        cypher="RETURN {a: 1, b: 'foo'}",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "numbers": [1, 2, 3]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"{a: 1, b: 'foo'}": "{a: 1, b: 'foo'}"},
                {"{a: 1, b: 'foo'}": "{a: 1, b: 'foo'}"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN map projections are not supported",
        tags=("return", "map", "xfail"),
    ),

    Scenario(
        key="return2-10",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[10] Return count aggregation over an empty graph",
        cypher="MATCH (a)\nRETURN count(a) > 0",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"count(a) > 0": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and boolean projections are not supported",
        tags=("return", "aggregation", "xfail"),
    ),

    Scenario(
        key="return2-11",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[11] RETURN does not lose precision on large integers",
        cypher="MATCH (p:TheLabel)\nRETURN p.id",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {id: 4611686018427387905})
            """
        ),
        expected=Expected(
            rows=[
                {"p.id": 4611686018427387905},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN property projections are not supported",
        tags=("return", "property", "big-int", "xfail"),
    ),

    Scenario(
        key="return2-12",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[12] Projecting a list of nodes and relationships",
        cypher="MATCH (n)-[r]->(m)\nRETURN [n, r, m] AS r",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[(:A), [:T], (:B)]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN list projections of nodes and relationships are not supported",
        tags=("return", "list", "projection", "xfail"),
    ),

    Scenario(
        key="return2-13",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[13] Projecting a map of nodes and relationships",
        cypher="MATCH (n)-[r]->(m)\nRETURN {node1: n, rel: r, node2: m} AS m",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"m": "{node1: (:A), rel: [:T], node2: (:B)}"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN map projections of nodes and relationships are not supported",
        tags=("return", "map", "projection", "xfail"),
    ),

    Scenario(
        key="return2-14",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[14] Do not fail when returning type of deleted relationships",
        cypher="MATCH ()-[r]->()\nDELETE r\nRETURN type(r)",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
                {"type(r)": "'T'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DELETE semantics and return expressions on deleted relationships are not supported",
        tags=("return", "delete", "type", "xfail"),
    ),

    Scenario(
        key="return2-15",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[15] Fail when returning properties of deleted nodes",
        cypher="MATCH (n)\nDELETE n\nRETURN n.num",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 0})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="DELETE semantics and deleted-entity runtime errors are not supported",
        tags=("return", "delete", "runtime-error", "xfail"),
    ),

    Scenario(
        key="return2-16",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[16] Fail when returning labels of deleted nodes",
        cypher="MATCH (n)\nDELETE n\nRETURN labels(n)",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="DELETE semantics and deleted-entity runtime errors are not supported",
        tags=("return", "delete", "runtime-error", "xfail"),
    ),

    Scenario(
        key="return2-17",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[17] Fail when returning properties of deleted relationships",
        cypher="MATCH ()-[r]->()\nDELETE r\nRETURN r.num",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T {num: 0}]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="DELETE semantics and deleted-entity runtime errors are not supported",
        tags=("return", "delete", "runtime-error", "xfail"),
    ),

    Scenario(
        key="return2-18",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[18] Fail on projecting a non-existent function",
        cypher="MATCH (a)\nRETURN foo(a)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for unknown functions is not enforced",
        tags=("return", "syntax-error", "function", "xfail"),
    ),
]
