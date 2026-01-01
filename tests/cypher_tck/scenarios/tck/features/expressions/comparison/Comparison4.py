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
        key='expr-comparison4-1',
        feature_path='tck/features/expressions/comparison/Comparison4.feature',
        scenario='[1] Handling long chains of operators',
        cypher='MATCH (n)-->(m)\n      WHERE n.prop1 < m.prop1 = n.prop2 <> m.prop2\n      RETURN labels(m)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {prop1: 3, prop2: 4})
                  CREATE (b:B {prop1: 4, prop2: 5})
                  CREATE (c:C {prop1: 4, prop2: 4})
                  CREATE (a)-[:R]->(b)
                  CREATE (b)-[:R]->(c)
                  CREATE (c)-[:R]->(a)
            """
        ),
        expected=Expected(
            rows=[
            {'labels(m)': "['B']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),
]
