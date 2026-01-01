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
        key='expr-existentialsubquery3-1',
        feature_path='tck/features/expressions/existentialSubqueries/ExistentialSubquery3.feature',
        scenario='[1] Nested simple existential subquery',
        cypher='MATCH (n) WHERE exists {\n        MATCH (m) WHERE exists {\n          (n)-[]->(m) WHERE n.prop = m.prop\n        }\n        RETURN true\n      }\n      RETURN n',
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
        key='expr-existentialsubquery3-2',
        feature_path='tck/features/expressions/existentialSubqueries/ExistentialSubquery3.feature',
        scenario='[2] Nested full existential subquery',
        cypher='MATCH (n) WHERE exists {\n        MATCH (m) WHERE exists {\n          MATCH (l)<-[:R]-(n)-[:R]->(m) RETURN true\n        }\n        RETURN true\n      }\n      RETURN n',
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
        key='expr-existentialsubquery3-3',
        feature_path='tck/features/expressions/existentialSubqueries/ExistentialSubquery3.feature',
        scenario='[3] Nested full existential subquery with pattern predicate',
        cypher='MATCH (n) WHERE exists {\n        MATCH (m) WHERE exists {\n          MATCH (l) WHERE (l)<-[:R]-(n)-[:R]->(m) RETURN true\n        }\n        RETURN true\n      }\n      RETURN n',
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
]
