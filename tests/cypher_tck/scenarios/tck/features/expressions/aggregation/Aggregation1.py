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
        key='expr-aggregation1-1',
        feature_path='tck/features/expressions/aggregation/Aggregation1.feature',
        scenario='[1] Count only non-null values',
        cypher='MATCH (n)\n      RETURN n.name, count(n.num)',
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'a', num: 33})
                  CREATE ({name: 'a'})
                  CREATE ({name: 'b', num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'n.name': "'a'", 'count(n.num)': 1},
            {'n.name': "'b'", 'count(n.num)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation1-2',
        feature_path='tck/features/expressions/aggregation/Aggregation1.feature',
        scenario='[2] Counting loop relationships',
        cypher='MATCH ()-[r]-()\n      RETURN count(r)',
        graph=graph_fixture_from_create(
            """
            CREATE (a), (a)-[:R]->(a)
            """
        ),
        expected=Expected(
            rows=[
            {'count(r)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),
]
