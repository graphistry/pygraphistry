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
        key='expr-list3-1',
        feature_path='tck/features/expressions/list/List3.feature',
        scenario='[1] Equality between list and literal should return false',
        cypher="RETURN [1, 2] = 'foo' AS res",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list3-2',
        feature_path='tck/features/expressions/list/List3.feature',
        scenario='[2] Equality of lists of different length should return false despite nulls',
        cypher='RETURN [1] = [1, null] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list3-3',
        feature_path='tck/features/expressions/list/List3.feature',
        scenario='[3] Equality between different lists with null should return false',
        cypher="RETURN [1, 2] = [null, 'foo'] AS res",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list3-4',
        feature_path='tck/features/expressions/list/List3.feature',
        scenario='[4] Equality between almost equal lists with null should return null',
        cypher='RETURN [1, 2] = [null, 2] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list3-5',
        feature_path='tck/features/expressions/list/List3.feature',
        scenario='[5] Equality of nested lists of different length should return false despite nulls',
        cypher='RETURN [[1]] = [[1], [null]] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list3-6',
        feature_path='tck/features/expressions/list/List3.feature',
        scenario='[6] Equality between different nested lists with null should return false',
        cypher="RETURN [[1, 2], [1, 3]] = [[1, 2], [null, 'foo']] AS res",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list3-7',
        feature_path='tck/features/expressions/list/List3.feature',
        scenario='[7] Equality between almost equal nested lists with null should return null',
        cypher="RETURN [[1, 2], ['foo', 'bar']] = [[1, 2], [null, 'bar']] AS res",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),
]
