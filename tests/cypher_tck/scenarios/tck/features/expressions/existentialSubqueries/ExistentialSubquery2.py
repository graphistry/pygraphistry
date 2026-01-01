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
        key='expr-existentialsubquery2-1',
        feature_path='tck/features/expressions/existentialSubqueries/ExistentialSubquery2.feature',
        scenario='[1] Full existential subquery',
        cypher='MATCH (n) WHERE exists {\n        MATCH (n)-->()\n        RETURN true\n      }\n      RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {prop: 1})-[:R]->(b:B {prop: 1}), 
                         (a)-[:R]->(:C {prop: 2}), 
                         (a)-[:R]->(:D {prop: 3})
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A {prop:1})'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'existentialSubqueries', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-existentialsubquery2-2',
        feature_path='tck/features/expressions/existentialSubqueries/ExistentialSubquery2.feature',
        scenario='[2] Full existential subquery with aggregation',
        cypher='MATCH (n) WHERE exists {\n        MATCH (n)-->(m)\n        WITH n, count(*) AS numConnections\n        WHERE numConnections = 3\n        RETURN true\n      }\n      RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {prop: 1})-[:R]->(b:B {prop: 1}), 
                         (a)-[:R]->(:C {prop: 2}), 
                         (a)-[:R]->(d:D {prop: 3}), 
                         (b)-[:R]->(d)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A {prop:1})'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'existentialSubqueries', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-existentialsubquery2-3',
        feature_path='tck/features/expressions/existentialSubqueries/ExistentialSubquery2.feature',
        scenario='[3] Full existential subquery with update clause should fail',
        cypher="MATCH (n) WHERE exists {\n        MATCH (n)-->(m)\n        SET m.prop='fail'\n      }\n      RETURN n",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'existentialSubqueries', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
