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
        key="with-where3-1",
        feature_path="tck/features/clauses/with-where/WithWhere3.feature",
        scenario="[1] Join between node identities",
        cypher="MATCH (a), (b)\nWITH a, b\nWHERE a = b\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:A)"},
                {"a": "(:B)", "b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, variable equality joins, and row projections are not supported",
        tags=("with", "join", "xfail"),
    ),

    Scenario(
        key="with-where3-2",
        feature_path="tck/features/clauses/with-where/WithWhere3.feature",
        scenario="[2] Join between node properties of disconnected nodes",
        cypher="MATCH (a:A), (b:B)\nWITH a, b\nWHERE a.id = b.id\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {id: 1}),
                   (:A {id: 2}),
                   (:B {id: 2}),
                   (:B {id: 3})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {id: 2})", "b": "(:B {id: 2})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, variable comparison joins, and row projections are not supported",
        tags=("with", "join", "xfail"),
    ),

    Scenario(
        key="with-where3-3",
        feature_path="tck/features/clauses/with-where/WithWhere3.feature",
        scenario="[3] Join between node properties of adjacent nodes",
        cypher="MATCH (n)-[rel]->(x)\nWITH n, x\nWHERE n.animal = x.animal\nRETURN n, x",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {animal: 'monkey'}),
              (b:B {animal: 'cow'}),
              (c:C {animal: 'monkey'}),
              (d:D {animal: 'cow'}),
              (a)-[:KNOWS]->(b),
              (a)-[:KNOWS]->(c),
              (d)-[:KNOWS]->(b),
              (d)-[:KNOWS]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"n": "(:A {animal: 'monkey'})", "x": "(:C {animal: 'monkey'})"},
                {"n": "(:D {animal: 'cow'})", "x": "(:B {animal: 'cow'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, variable comparison joins, and row projections are not supported",
        tags=("with", "join", "xfail"),
    ),
]
