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
        key='expr-list2-1',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[1] List slice',
        cypher='WITH [1, 2, 3, 4, 5] AS list\n      RETURN list[1..3] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': '[2, 3]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-2',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[2] List slice with implicit end',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[1..] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': '[2, 3]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-3',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[3] List slice with implicit start',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[..2] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': '[1, 2]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-4',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[4] List slice with singleton range',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[0..1] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': '[1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-5',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[5] List slice with empty range',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[0..0] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-6',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[6] List slice with negative range',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[-3..-1] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': '[1, 2]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-7',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[7] List slice with invalid range',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[3..1] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-8',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[8] List slice with exceeding range',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[-5..5] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': '[1, 2, 3]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-9-1',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[9] List slice with null range (example 1)',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[null..null] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-9-2',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[9] List slice with null range (example 2)',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[1..null] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-9-3',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[9] List slice with null range (example 3)',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[null..3] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-9-4',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[9] List slice with null range (example 4)',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[..null] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-9-5',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[9] List slice with null range (example 5)',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[null..] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list2-10',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[10] List slice with parameterised range',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[$from..$to] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': '[2, 3]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-list2-11',
        feature_path='tck/features/expressions/list/List2.feature',
        scenario='[11] List slice with parameterised invalid range',
        cypher='WITH [1, 2, 3] AS list\n      RETURN list[$from..$to] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'xfail'),
    ),
]
