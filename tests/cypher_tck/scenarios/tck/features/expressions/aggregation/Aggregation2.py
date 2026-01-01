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
        key='expr-aggregation2-1',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[1] `max()` over integers',
        cypher='UNWIND [1, 2, 0, null, -1] AS x\n      RETURN max(x)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'max(x)': 2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation2-2',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[2] `min()` over integers',
        cypher='UNWIND [1, 2, 0, null, -1] AS x\n      RETURN min(x)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'min(x)': -1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation2-3',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[3] `max()` over floats',
        cypher='UNWIND [1.0, 2.0, 0.5, null] AS x\n      RETURN max(x)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'max(x)': 2.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation2-4',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[4] `min()` over floats',
        cypher='UNWIND [1.0, 2.0, 0.5, null] AS x\n      RETURN min(x)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'min(x)': 0.5}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation2-5',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[5] `max()` over mixed numeric values',
        cypher='UNWIND [1, 2.0, 5, null, 3.2, 0.1] AS x\n      RETURN max(x)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'max(x)': 5}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation2-6',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[6] `min()` over mixed numeric values',
        cypher='UNWIND [1, 2.0, 5, null, 3.2, 0.1] AS x\n      RETURN min(x)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'min(x)': 0.1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation2-7',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[7] `max()` over strings',
        cypher="UNWIND ['a', 'b', 'B', null, 'abc', 'abc1'] AS i\n      RETURN max(i)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'max(i)': "'b'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation2-8',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[8] `min()` over strings',
        cypher="UNWIND ['a', 'b', 'B', null, 'abc', 'abc1'] AS i\n      RETURN min(i)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'min(i)': "'B'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation2-9',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[9] `max()` over list values',
        cypher='UNWIND [[1], [2], [2, 1]] AS x\n      RETURN max(x)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'max(x)': '[2, 1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation2-10',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[10] `min()` over list values',
        cypher='UNWIND [[1], [2], [2, 1]] AS x\n      RETURN min(x)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'min(x)': '[1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation2-11',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[11] `max()` over mixed values',
        cypher="UNWIND [1, 'a', null, [1, 2], 0.2, 'b'] AS x\n      RETURN max(x)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'max(x)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation2-12',
        feature_path='tck/features/expressions/aggregation/Aggregation2.feature',
        scenario='[12] `min()` over mixed values',
        cypher="UNWIND [1, 'a', null, [1, 2], 0.2, 'b'] AS x\n      RETURN min(x)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'min(x)': '[1, 2]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),
]
