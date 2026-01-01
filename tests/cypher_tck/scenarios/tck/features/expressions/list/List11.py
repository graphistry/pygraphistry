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
        key='expr-list11-1-1',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 1)',
        cypher='RETURN range(-1236, -1234) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-1236, -1235, -1234]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-2',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 2)',
        cypher='RETURN range(-1234, -1234) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-1234]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-3',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 3)',
        cypher='RETURN range(-10, -3) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-10, -9, -8, -7, -6, -5, -4, -3]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-4',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 4)',
        cypher='RETURN range(-10, 0) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-5',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 5)',
        cypher='RETURN range(-1, 0) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-1, 0]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-6',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 6)',
        cypher='RETURN range(0, -123) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-7',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 7)',
        cypher='RETURN range(0, -1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-8',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 8)',
        cypher='RETURN range(-1, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-1, 0, 1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-9',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 9)',
        cypher='RETURN range(0, 0) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-10',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 10)',
        cypher='RETURN range(0, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0, 1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-11',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 11)',
        cypher='RETURN range(0, 10) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-12',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 12)',
        cypher='RETURN range(6, 10) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[6, 7, 8, 9, 10]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-13',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 13)',
        cypher='RETURN range(1234, 1234) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[1234]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-1-14',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[1] Create list from `range()` with default step (example 14)',
        cypher='RETURN range(1234, 1236) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[1234, 1235, 1236]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-1',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 1)',
        cypher='RETURN range(1381, -3412, -1298) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[1381, 83, -1215, -2513]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-2',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 2)',
        cypher='RETURN range(0, -2000, -1298) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0, -1298]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-3',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 3)',
        cypher='RETURN range(10, -10, -3) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[10, 7, 4, 1, -2, -5, -8]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-4',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 4)',
        cypher='RETURN range(0, -10, -3) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0, -3, -6, -9]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-5',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 5)',
        cypher='RETURN range(0, -20, -2) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0, -2, -4, -6, -8, -10, -12, -14, -16, -18, -20]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-6',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 6)',
        cypher='RETURN range(0, -10, -1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-7',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 7)',
        cypher='RETURN range(0, -1, -1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0, -1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-8',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 8)',
        cypher='RETURN range(-1236, -1234, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-1236, -1235, -1234]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-9',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 9)',
        cypher='RETURN range(-10, 0, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-10',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 10)',
        cypher='RETURN range(-1, 0, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-1, 0]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-11',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 11)',
        cypher='RETURN range(0, 1, -123) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-12',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 12)',
        cypher='RETURN range(0, 1, -1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-13',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 13)',
        cypher='RETURN range(0, -123, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-14',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 14)',
        cypher='RETURN range(0, -1, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-15',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 15)',
        cypher='RETURN range(0, 0, -1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-16',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 16)',
        cypher='RETURN range(0, 0, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-17',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 17)',
        cypher='RETURN range(0, 1, 2) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-18',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 18)',
        cypher='RETURN range(0, 1, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0, 1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-19',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 19)',
        cypher='RETURN range(0, 10, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-20',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 20)',
        cypher='RETURN range(6, 10, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[6, 7, 8, 9, 10]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-21',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 21)',
        cypher='RETURN range(1234, 1234, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[1234]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-22',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 22)',
        cypher='RETURN range(1234, 1236, 1) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[1234, 1235, 1236]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-23',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 23)',
        cypher='RETURN range(-10, 0, 3) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-10, -7, -4, -1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-24',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 24)',
        cypher='RETURN range(-10, 10, 3) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-10, -7, -4, -1, 2, 5, 8]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-25',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 25)',
        cypher='RETURN range(-2000, 0, 1298) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-2000, -702]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-2-26',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[2] Create list from `range()` with explicitly given step (example 26)',
        cypher='RETURN range(-3412, 1381, 1298) AS list',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': '[-3412, -2114, -816, 482]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-3',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[3] Create an empty list if range direction and step direction are inconsistent',
        cypher='WITH 0 AS start, [1, 2, 500, 1000, 1500] AS stopList, [-1000, -3, -2, -1, 1, 2, 3, 1000] AS stepList\n      UNWIND stopList AS stop\n      UNWIND stepList AS step\n      WITH start, stop, step, range(start, stop, step) AS list\n      WITH start, stop, step, list, sign(stop-start) <> sign(step) AS empty\n      RETURN ALL(ok IN collect((size(list) = 0) = empty) WHERE ok) AS okay',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'okay': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-4-1',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[4] Fail on invalid arguments for `range()` (example 1)',
        cypher='RETURN range(2, 8, 0)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-4-2',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[4] Fail on invalid arguments for `range()` (example 2)',
        cypher='RETURN range(2, 8, 0)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-4-3',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[4] Fail on invalid arguments for `range()` (example 3)',
        cypher='RETURN range(2, 8, 0)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-4-4',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[4] Fail on invalid arguments for `range()` (example 4)',
        cypher='RETURN range(2, 8, 0)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-1',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 1)',
        cypher='RETURN range(true, 1, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-2',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 2)',
        cypher='RETURN range(0, true, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-3',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 3)',
        cypher='RETURN range(0, 1, true)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-4',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 4)',
        cypher='RETURN range(-1.1, 1, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-5',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 5)',
        cypher='RETURN range(-0.0, 1, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-6',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 6)',
        cypher='RETURN range(0.0, 1, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-7',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 7)',
        cypher='RETURN range(1.1, 1, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-8',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 8)',
        cypher='RETURN range(0, -1.1, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-9',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 9)',
        cypher='RETURN range(0, -0.0, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-10',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 10)',
        cypher='RETURN range(0, 0.0, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-11',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 11)',
        cypher='RETURN range(0, 1.1, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-12',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 12)',
        cypher='RETURN range(0, 1, -1.1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-13',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 13)',
        cypher='RETURN range(0, 1, 1.1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-14',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 14)',
        cypher="RETURN range('xyz', 1, 1)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-15',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 15)',
        cypher="RETURN range(0, 'xyz', 1)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-16',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 16)',
        cypher="RETURN range(0, 1, 'xyz')",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-17',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 17)',
        cypher='RETURN range([0], 1, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-18',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 18)',
        cypher='RETURN range(0, [1], 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-19',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 19)',
        cypher='RETURN range(0, 1, [1])',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-20',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 20)',
        cypher='RETURN range({start: 0}, 1, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-21',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 21)',
        cypher='RETURN range(0, {end: 1}, 1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list11-5-22',
        feature_path='tck/features/expressions/list/List11.feature',
        scenario='[5] Fail on invalid argument types for `range()` (example 22)',
        cypher='RETURN range(0, 1, {step: 1})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),
]
