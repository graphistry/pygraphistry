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
        key='expr-list5-1',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[1] IN should work with nested list subscripting',
        cypher='WITH [[1, 2, 3]] AS list\n      RETURN 3 IN list[0] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-2',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[2] IN should work with nested literal list subscripting',
        cypher='RETURN 3 IN [[1, 2, 3]][0] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-3',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[3] IN should work with list slices',
        cypher='WITH [1, 2, 3] AS list\n      RETURN 3 IN list[0..1] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-4',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[4] IN should work with literal list slices',
        cypher='RETURN 3 IN [1, 2, 3][0..1] AS r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-5',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[5] IN should return false when matching a number with a string',
        cypher="RETURN 1 IN ['1', 2] AS res",
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
        key='expr-list5-6',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[6] IN should return false when matching a number with a string - list version',
        cypher="RETURN [1, 2] IN [1, [1, '2']] AS res",
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
        key='expr-list5-7',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario="[7] IN should return false when types of LHS and RHS don't match - singleton list",
        cypher='RETURN [1] IN [1, 2] AS res',
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
        key='expr-list5-8',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario="[8] IN should return false when types of LHS and RHS don't match - list",
        cypher='RETURN [1, 2] IN [1, 2] AS res',
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
        key='expr-list5-9',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[9] IN should return true when types of LHS and RHS match - singleton list',
        cypher='RETURN [1] IN [1, 2, [1]] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-10',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[10] IN should return true when types of LHS and RHS match - list',
        cypher='RETURN [1, 2] IN [1, [1, 2]] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-11',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario="[11] IN should return false when order of elements in LHS list and RHS list don't match",
        cypher='RETURN [1, 2] IN [1, [2, 1]] AS res',
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
        key='expr-list5-12',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[12] IN with different length lists should return false',
        cypher='RETURN [1, 2] IN [1, [1, 2, 3]] AS res',
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
        key='expr-list5-13',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[13] IN should return false when matching a list with a nested list with same elements',
        cypher='RETURN [1, 2] IN [1, [[1, 2]]] AS res',
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
        key='expr-list5-14',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[14] IN should return true when both LHS and RHS contain nested lists',
        cypher='RETURN [[1, 2], [3, 4]] IN [5, [[1, 2], [3, 4]]] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-15',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[15] IN should return true when both LHS and RHS contain a nested list alongside a scalar element',
        cypher='RETURN [[1, 2], 3] IN [1, [[1, 2], 3]] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-16',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[16] IN should return true when LHS and RHS contain a nested list - singleton version',
        cypher='RETURN [[1]] IN [2, [[1]]] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-17',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[17] IN should return true when LHS and RHS contain a nested list',
        cypher='RETURN [[1, 3]] IN [2, [[1, 3]]] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-18',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[18] IN should return false when LHS contains a nested list and type mismatch on RHS - singleton version',
        cypher='RETURN [[1]] IN [2, [1]] AS res',
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
        key='expr-list5-19',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[19] IN should return false when LHS contains a nested list and type mismatch on RHS',
        cypher='RETURN [[1, 3]] IN [2, [1, 3]] AS res',
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
        key='expr-list5-20',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[20] IN should return null if LHS and RHS are null',
        cypher='RETURN null IN [null] AS res',
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
        key='expr-list5-21',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[21] IN should return null if LHS and RHS are null - list version',
        cypher='RETURN [null] IN [[null]] AS res',
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
        key='expr-list5-22',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[22] IN should return null when LHS and RHS both ultimately contain null, even if LHS and RHS are of different types (nested list and flat list)',
        cypher='RETURN [null] IN [null] AS res',
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
        key='expr-list5-23',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[23] IN with different length lists should return false despite nulls',
        cypher='RETURN [1] IN [[1, null]] AS res',
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
        key='expr-list5-24',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[24] IN should return true if match despite nulls',
        cypher='RETURN 3 IN [1, null, 3] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-25',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[25] IN should return null if comparison with null is required',
        cypher='RETURN 4 IN [1, null, 3] AS res',
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
        key='expr-list5-26',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[26] IN should return true if correct list found despite other lists having nulls',
        cypher="RETURN [1, 2] IN [[null, 'foo'], [1, 2]] AS res",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-27',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[27] IN should return true if correct list found despite null being another element within containing list',
        cypher='RETURN [1, 2] IN [1, [1, 2], null] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-28',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[28] IN should return false if no match can be found, despite nulls',
        cypher="RETURN [1, 2] IN [[null, 'foo']] AS res",
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
        key='expr-list5-29',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[29] IN should return null if comparison with null is required, list version',
        cypher='RETURN [1, 2] IN [[null, 2]] AS res',
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
        key='expr-list5-30',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[30] IN should return false if different length lists compared, even if the extra element is null',
        cypher='RETURN [1, 2] IN [1, [1, 2, null]] AS res',
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
        key='expr-list5-31',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[31] IN should return null when comparing two so-called identical lists where one element is null',
        cypher='RETURN [1, 2, null] IN [1, [1, 2, null]] AS res',
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
        key='expr-list5-32',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[32] IN should return true with previous null match, list version',
        cypher='RETURN [1, 2] IN [[null, 2], [1, 2]] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-33',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[33] IN should return false if different length lists with nested elements compared, even if the extra element is null',
        cypher='RETURN [[1, 2], [3, 4]] IN [5, [[1, 2], [3, 4], null]] AS res',
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
        key='expr-list5-34',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[34] IN should return null if comparison with null is required, list version 2',
        cypher='RETURN [1, 2] IN [[null, 2], [1, 3]] AS res',
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
        key='expr-list5-35',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[35] IN should work with an empty list',
        cypher='RETURN [] IN [[]] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-36',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[36] IN should return false for the empty list if the LHS and RHS types differ',
        cypher='RETURN [] IN [] AS res',
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
        key='expr-list5-37',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[37] IN should work with an empty list in the presence of other list elements: matching',
        cypher='RETURN [] IN [1, []] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-38',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[38] IN should work with an empty list in the presence of other list elements: not matching',
        cypher='RETURN [] IN [1, 2] AS res',
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
        key='expr-list5-39',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[39] IN should work with an empty list when comparing nested lists',
        cypher='RETURN [[]] IN [1, [[]]] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-40',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[40] IN should return null if comparison with null is required for empty list',
        cypher='RETURN [] IN [1, 2, null] AS res',
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
        key='expr-list5-41',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[41] IN should return true when LHS and RHS contain nested list with multiple empty lists',
        cypher='RETURN [[], []] IN [1, [[], []]] AS res',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'res': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list5-42-1',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[42] Failing when using IN on a non-list literal (example 1)',
        cypher='RETURN 1 IN true',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list5-42-2',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[42] Failing when using IN on a non-list literal (example 2)',
        cypher='RETURN 1 IN 123',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list5-42-3',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[42] Failing when using IN on a non-list literal (example 3)',
        cypher='RETURN 1 IN 123.4',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list5-42-4',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[42] Failing when using IN on a non-list literal (example 4)',
        cypher="RETURN 1 IN 'foo'",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list5-42-5',
        feature_path='tck/features/expressions/list/List5.feature',
        scenario='[42] Failing when using IN on a non-list literal (example 5)',
        cypher='RETURN 1 IN {x: []}',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
