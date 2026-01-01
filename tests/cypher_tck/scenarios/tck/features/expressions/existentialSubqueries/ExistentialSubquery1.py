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
        key='expr-existentialsubquery1-1',
        feature_path='tck/features/expressions/existentialSubqueries/ExistentialSubquery1.feature',
        scenario='[1] Simple subquery without WHERE clause',
        cypher='MATCH (n) WHERE exists {\n        (n)-->()\n      }\n      RETURN n',
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
        key='expr-existentialsubquery1-2',
        feature_path='tck/features/expressions/existentialSubqueries/ExistentialSubquery1.feature',
        scenario='[2] Simple subquery with WHERE clause',
        cypher='MATCH (n) WHERE exists {\n        (n)-->(m) WHERE n.prop = m.prop\n      }\n      RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {prop: 1})-[:R]->(b:B {prop: 1}), 
                         (a)-[:R]->(:C {prop: 2}), 
                         (a)-[:R]->(:D {prop: 3}), 
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
        key='expr-existentialsubquery1-3',
        feature_path='tck/features/expressions/existentialSubqueries/ExistentialSubquery1.feature',
        scenario='[3] Simple subquery without WHERE clause, not existing pattern',
        cypher='MATCH (n) WHERE exists {\n        (n)-[:NA]->()\n      }\n      RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {prop: 1})-[:R]->(b:B {prop: 1}), 
                         (a)-[:R]->(:C {prop: 2}), 
                         (a)-[:R]->(:D {prop: 3})
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'existentialSubqueries', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-existentialsubquery1-4',
        feature_path='tck/features/expressions/existentialSubqueries/ExistentialSubquery1.feature',
        scenario='[4] Simple subquery with WHERE clause, not existing pattern',
        cypher="MATCH (n) WHERE exists {\n        (n)-[r]->() WHERE type(r) = 'NA'\n      }\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {prop: 1})-[:R]->(b:B {prop: 1}), 
                         (a)-[:R]->(:C {prop: 2}), 
                         (a)-[:R]->(:D {prop: 3})
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'existentialSubqueries', 'meta-xfail', 'xfail'),
    ),
]
