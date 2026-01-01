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
        key='expr-aggregation3-1',
        feature_path='tck/features/expressions/aggregation/Aggregation3.feature',
        scenario='[1] Sum only non-null values',
        cypher='MATCH (n)\n      RETURN n.name, sum(n.num)',
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'a', num: 33})
                  CREATE ({name: 'a'})
                  CREATE ({name: 'a', num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'n.name': "'a'", 'sum(n.num)': 75}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation3-2',
        feature_path='tck/features/expressions/aggregation/Aggregation3.feature',
        scenario='[2] No overflow during summation',
        cypher='UNWIND range(1000000, 2000000) AS i\n      WITH i\n      LIMIT 3000\n      RETURN sum(i)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'sum(i)': 3004498500}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),
]
