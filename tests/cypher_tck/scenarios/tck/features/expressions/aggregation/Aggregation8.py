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
        key='expr-aggregation8-1',
        feature_path='tck/features/expressions/aggregation/Aggregation8.feature',
        scenario='[1] Distinct on unbound node',
        cypher='OPTIONAL MATCH (a)\n      RETURN count(DISTINCT a)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'count(DISTINCT a)': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation8-2',
        feature_path='tck/features/expressions/aggregation/Aggregation8.feature',
        scenario='[2] Distinct on null',
        cypher='MATCH (a)\n      RETURN count(DISTINCT a.name)',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
            {'count(DISTINCT a.name)': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation8-3',
        feature_path='tck/features/expressions/aggregation/Aggregation8.feature',
        scenario='[3] Collect distinct nulls',
        cypher='UNWIND [null, null] AS x\n      RETURN collect(DISTINCT x) AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'c': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation8-4',
        feature_path='tck/features/expressions/aggregation/Aggregation8.feature',
        scenario='[4] Collect distinct values mixed with nulls',
        cypher='UNWIND [null, 1, null] AS x\n      RETURN collect(DISTINCT x) AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'c': '[1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),
]
