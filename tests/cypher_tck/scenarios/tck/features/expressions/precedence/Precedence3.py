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
        key='expr-precedence3-1',
        feature_path='tck/features/expressions/precedence/Precedence3.feature',
        scenario='[1] List element access takes precedence over list appending',
        cypher='RETURN [[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10][3] AS a,\n             [[1], [2, 3], [4, 5]] + ([5, [6, 7], [8, 9], 10][3]) AS b,\n             ([[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10])[3] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': '[[1], [2, 3], [4, 5], 10]', 'b': '[[1], [2, 3], [4, 5], 10]', 'c': 5}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence3-2',
        feature_path='tck/features/expressions/precedence/Precedence3.feature',
        scenario='[2] List element access takes precedence over list concatenation',
        cypher='RETURN [[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10][2] AS a,\n             [[1], [2, 3], [4, 5]] + ([5, [6, 7], [8, 9], 10][2]) AS b,\n             ([[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10])[2] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': '[[1], [2, 3], [4, 5], 8, 9]', 'b': '[[1], [2, 3], [4, 5], 8, 9]', 'c': '[4, 5]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence3-3',
        feature_path='tck/features/expressions/precedence/Precedence3.feature',
        scenario='[3] List slicing takes precedence over list concatenation',
        cypher='RETURN [[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10][1..3] AS a,\n             [[1], [2, 3], [4, 5]] + ([5, [6, 7], [8, 9], 10][1..3]) AS b,\n             ([[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10])[1..3] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': '[[1], [2, 3], [4, 5], [6, 7], [8, 9]]', 'b': '[[1], [2, 3], [4, 5], [6, 7], [8, 9]]', 'c': '[[2, 3], [4, 5]]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence3-4',
        feature_path='tck/features/expressions/precedence/Precedence3.feature',
        scenario='[4] List appending takes precedence over list element containment',
        cypher='RETURN [1]+2 IN [3]+4 AS a,\n             ([1]+2) IN ([3]+4) AS b,\n             [1]+(2 IN [3])+4 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': '[1, false, 4]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence3-5',
        feature_path='tck/features/expressions/precedence/Precedence3.feature',
        scenario='[5] List concatenation takes precedence over list element containment',
        cypher='RETURN [1]+[2] IN [3]+[4] AS a,\n             ([1]+[2]) IN ([3]+[4]) AS b,\n             (([1]+[2]) IN [3])+[4] AS c,\n             [1]+([2] IN [3])+[4] AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': '[false, 4]', 'd': '[1, false, 4]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence3-6-1',
        feature_path='tck/features/expressions/precedence/Precedence3.feature',
        scenario='[6] List element containment takes precedence over comparison operator (example 1)',
        cypher='RETURN [1, 2] = [3, 4] IN [[3, 4], false] AS a,\n             [1, 2] = ([3, 4] IN [[3, 4], false]) AS b,\n             ([1, 2] = [3, 4]) IN [[3, 4], false] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence3-6-2',
        feature_path='tck/features/expressions/precedence/Precedence3.feature',
        scenario='[6] List element containment takes precedence over comparison operator (example 2)',
        cypher='RETURN [1, 2] <> [3, 4] IN [[3, 4], false] AS a,\n             [1, 2] <> ([3, 4] IN [[3, 4], false]) AS b,\n             ([1, 2] <> [3, 4]) IN [[3, 4], false] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence3-6-3',
        feature_path='tck/features/expressions/precedence/Precedence3.feature',
        scenario='[6] List element containment takes precedence over comparison operator (example 3)',
        cypher='RETURN [1, 2] < [3, 4] IN [[3, 4], false] AS a,\n             [1, 2] < ([3, 4] IN [[3, 4], false]) AS b,\n             ([1, 2] < [3, 4]) IN [[3, 4], false] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'null', 'b': 'null', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence3-6-4',
        feature_path='tck/features/expressions/precedence/Precedence3.feature',
        scenario='[6] List element containment takes precedence over comparison operator (example 4)',
        cypher='RETURN [1, 2] > [3, 4] IN [[3, 4], false] AS a,\n             [1, 2] > ([3, 4] IN [[3, 4], false]) AS b,\n             ([1, 2] > [3, 4]) IN [[3, 4], false] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'null', 'b': 'null', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence3-6-5',
        feature_path='tck/features/expressions/precedence/Precedence3.feature',
        scenario='[6] List element containment takes precedence over comparison operator (example 5)',
        cypher='RETURN [1, 2] <= [3, 4] IN [[3, 4], false] AS a,\n             [1, 2] <= ([3, 4] IN [[3, 4], false]) AS b,\n             ([1, 2] <= [3, 4]) IN [[3, 4], false] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'null', 'b': 'null', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence3-6-6',
        feature_path='tck/features/expressions/precedence/Precedence3.feature',
        scenario='[6] List element containment takes precedence over comparison operator (example 6)',
        cypher='RETURN [1, 2] >= [3, 4] IN [[3, 4], false] AS a,\n             [1, 2] >= ([3, 4] IN [[3, 4], false]) AS b,\n             ([1, 2] >= [3, 4]) IN [[3, 4], false] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'null', 'b': 'null', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),
]
