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
        key='merge6-1',
        feature_path='tck/features/clauses/merge/Merge6.feature',
        scenario='[1] Using ON CREATE on a node',
        cypher='MATCH (a:A), (b:B)\n      MERGE (a)-[:KNOWS]->(b)\n        ON CREATE SET b.created = 1',
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge6-2',
        feature_path='tck/features/clauses/merge/Merge6.feature',
        scenario='[2] Using ON CREATE on a relationship',
        cypher="MATCH (a:A), (b:B)\n      MERGE (a)-[r:TYPE]->(b)\n        ON CREATE SET r.name = 'Lola'\n      RETURN count(r)",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
            {'count(r)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge6-3',
        feature_path='tck/features/clauses/merge/Merge6.feature',
        scenario='[3] Updating one property with ON CREATE',
        cypher="MATCH (a {name: 'A'}), (b {name: 'B'})\n      MERGE (a)-[r:TYPE]->(b)\n        ON CREATE SET r.name = 'foo'",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'A'}), (:B {name: 'B'})
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge6-4',
        feature_path='tck/features/clauses/merge/Merge6.feature',
        scenario='[4] Null-setting one property with ON CREATE',
        cypher="MATCH (a {name: 'A'}), (b {name: 'B'})\n      MERGE (a)-[r:TYPE]->(b)\n        ON CREATE SET r.name = null",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'A'}), (:B {name: 'B'})
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge6-6',
        feature_path='tck/features/clauses/merge/Merge6.feature',
        scenario='[6] Copying properties from node with ON CREATE',
        cypher="MATCH (a {name: 'A'}), (b {name: 'B'})\n      MERGE (a)-[r:TYPE]->(b)\n        ON CREATE SET r = a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'A'}), (:B {name: 'B'})
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge6-7',
        feature_path='tck/features/clauses/merge/Merge6.feature',
        scenario='[7] Copying properties from literal map with ON CREATE',
        cypher="MATCH (a {name: 'A'}), (b {name: 'B'})\n      MERGE (a)-[r:TYPE]->(b)\n      ON CREATE SET r += {name: 'bar', name2: 'baz'}",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'A'}), (:B {name: 'B'})
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),
]
