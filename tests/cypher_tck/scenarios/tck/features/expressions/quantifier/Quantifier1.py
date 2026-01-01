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
        key='expr-quantifier1-1',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[1] None quantifier is always true on empty list',
        cypher='RETURN none(x IN [] WHERE true) AS a, none(x IN [] WHERE false) AS b, none(x IN [] WHERE x) AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-2-1',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[2] None quantifier on list literal containing booleans (example 1)',
        cypher='RETURN none(x IN [] WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-2-2',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[2] None quantifier on list literal containing booleans (example 2)',
        cypher='RETURN none(x IN [true] WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-2-3',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[2] None quantifier on list literal containing booleans (example 3)',
        cypher='RETURN none(x IN [false] WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-2-4',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[2] None quantifier on list literal containing booleans (example 4)',
        cypher='RETURN none(x IN [true, false] WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-2-5',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[2] None quantifier on list literal containing booleans (example 5)',
        cypher='RETURN none(x IN [false, true] WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-2-6',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[2] None quantifier on list literal containing booleans (example 6)',
        cypher='RETURN none(x IN [true, false, true] WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-2-7',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[2] None quantifier on list literal containing booleans (example 7)',
        cypher='RETURN none(x IN [false, true, false] WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-2-8',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[2] None quantifier on list literal containing booleans (example 8)',
        cypher='RETURN none(x IN [true, true, true] WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-2-9',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[2] None quantifier on list literal containing booleans (example 9)',
        cypher='RETURN none(x IN [false, false, false] WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-1',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 1)',
        cypher='RETURN none(x IN [] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-2',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 2)',
        cypher='RETURN none(x IN [1] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-3',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 3)',
        cypher='RETURN none(x IN [1, 3] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-4',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 4)',
        cypher='RETURN none(x IN [1, 3, 20, 5000] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-5',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 5)',
        cypher='RETURN none(x IN [20, 3, 5000, -2] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-6',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 6)',
        cypher='RETURN none(x IN [2] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-7',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 7)',
        cypher='RETURN none(x IN [1, 2] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-8',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 8)',
        cypher='RETURN none(x IN [1, 2, 3] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-9',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 9)',
        cypher='RETURN none(x IN [2, 2] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-10',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 10)',
        cypher='RETURN none(x IN [2, 3] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-11',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 11)',
        cypher='RETURN none(x IN [3, 2, 3] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-12',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 12)',
        cypher='RETURN none(x IN [2, 3, 2] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-13',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 13)',
        cypher='RETURN none(x IN [2, -10, 3, 9, 0] WHERE x < 10) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-14',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 14)',
        cypher='RETURN none(x IN [2, -10, 3, 2, 10] WHERE x < 10) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-15',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 15)',
        cypher='RETURN none(x IN [2, -10, 3, 21, 10] WHERE x < 10) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-16',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 16)',
        cypher='RETURN none(x IN [200, -10, 36, 21, 10] WHERE x < 10) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-3-17',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[3] None quantifier on list literal containing integers (example 17)',
        cypher='RETURN none(x IN [200, 15, 36, 21, 10] WHERE x < 10) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-1',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 1)',
        cypher='RETURN none(x IN [] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-2',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 2)',
        cypher='RETURN none(x IN [1.1] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-3',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 3)',
        cypher='RETURN none(x IN [1.1, 3.5] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-4',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 4)',
        cypher='RETURN none(x IN [1.1, 3.5, 20.0, 50.42435] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-5',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 5)',
        cypher='RETURN none(x IN [20.0, 3.4, 50.2, -2.1] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-6',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 6)',
        cypher='RETURN none(x IN [2.1] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-7',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 7)',
        cypher='RETURN none(x IN [1.43, 2.1] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-8',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 8)',
        cypher='RETURN none(x IN [1.43, 2.1, 3.5] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-9',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 9)',
        cypher='RETURN none(x IN [2.1, 2.1] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-10',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 10)',
        cypher='RETURN none(x IN [2.1, 3.5] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-11',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 11)',
        cypher='RETURN none(x IN [3.5, 2.1, 3.5] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-4-12',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[4] None quantifier on list literal containing floats (example 12)',
        cypher='RETURN none(x IN [2.1, 3.5, 2.1] WHERE x = 2.1) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-5-1',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[5] None quantifier on list literal containing strings (example 1)',
        cypher='RETURN none(x IN [] WHERE size(x) = 3) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-5-2',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[5] None quantifier on list literal containing strings (example 2)',
        cypher="RETURN none(x IN ['abc'] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-5-3',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[5] None quantifier on list literal containing strings (example 3)',
        cypher="RETURN none(x IN ['ef'] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-5-4',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[5] None quantifier on list literal containing strings (example 4)',
        cypher="RETURN none(x IN ['abc', 'ef'] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-5-5',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[5] None quantifier on list literal containing strings (example 5)',
        cypher="RETURN none(x IN ['ef', 'abc'] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-5-6',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[5] None quantifier on list literal containing strings (example 6)',
        cypher="RETURN none(x IN ['abc', 'ef', 'abc'] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-5-7',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[5] None quantifier on list literal containing strings (example 7)',
        cypher="RETURN none(x IN ['ef', 'abc', 'ef'] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-5-8',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[5] None quantifier on list literal containing strings (example 8)',
        cypher="RETURN none(x IN ['abc', 'abc', 'abc'] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-5-9',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[5] None quantifier on list literal containing strings (example 9)',
        cypher="RETURN none(x IN ['ef', 'ef', 'ef'] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-6-1',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[6] None quantifier on list literal containing lists (example 1)',
        cypher='RETURN none(x IN [] WHERE size(x) = 3) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-6-2',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[6] None quantifier on list literal containing lists (example 2)',
        cypher='RETURN none(x IN [[1, 2, 3]] WHERE size(x) = 3) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-6-3',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[6] None quantifier on list literal containing lists (example 3)',
        cypher="RETURN none(x IN [['a']] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-6-4',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[6] None quantifier on list literal containing lists (example 4)',
        cypher="RETURN none(x IN [[1, 2, 3], ['a']] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-6-5',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[6] None quantifier on list literal containing lists (example 5)',
        cypher="RETURN none(x IN [['a'], [1, 2, 3]] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-6-6',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[6] None quantifier on list literal containing lists (example 6)',
        cypher="RETURN none(x IN [[1, 2, 3], ['a'], [1, 2, 3]] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-6-7',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[6] None quantifier on list literal containing lists (example 7)',
        cypher="RETURN none(x IN [['a'], [1, 2, 3], ['a']] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-6-8',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[6] None quantifier on list literal containing lists (example 8)',
        cypher='RETURN none(x IN [[1, 2, 3], [1, 2, 3], [1, 2, 3]] WHERE size(x) = 3) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-6-9',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[6] None quantifier on list literal containing lists (example 9)',
        cypher="RETURN none(x IN [['a'], ['a'], ['a']] WHERE size(x) = 3) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-7-1',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[7] None quantifier on list literal containing maps (example 1)',
        cypher='RETURN none(x IN [] WHERE x.a = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-7-2',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[7] None quantifier on list literal containing maps (example 2)',
        cypher='RETURN none(x IN [{a: 2, b: 5}] WHERE x.a = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-7-3',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[7] None quantifier on list literal containing maps (example 3)',
        cypher='RETURN none(x IN [{a: 4}] WHERE x.a = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-7-4',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[7] None quantifier on list literal containing maps (example 4)',
        cypher='RETURN none(x IN [{a: 2, b: 5}, {a: 4}] WHERE x.a = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-7-5',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[7] None quantifier on list literal containing maps (example 5)',
        cypher='RETURN none(x IN [{a: 4}, {a: 2, b: 5}] WHERE x.a = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-7-6',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[7] None quantifier on list literal containing maps (example 6)',
        cypher='RETURN none(x IN [{a: 2, b: 5}, {a: 4}, {a: 2, b: 5}] WHERE x.a = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-7-7',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[7] None quantifier on list literal containing maps (example 7)',
        cypher='RETURN none(x IN [{a: 4}, {a: 2, b: 5}, {a: 4}] WHERE x.a = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-7-8',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[7] None quantifier on list literal containing maps (example 8)',
        cypher='RETURN none(x IN [{a: 2, b: 5}, {a: 2, b: 5}, {a: 2, b: 5}] WHERE x.a = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-7-9',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[7] None quantifier on list literal containing maps (example 9)',
        cypher='RETURN none(x IN [{a: 4}, {a: 4}, {a: 4}] WHERE x.a = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-8',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[8] None quantifier on list containing nodes',
        cypher="MATCH p = (:SNodes)-[*0..3]->(x)\n      WITH tail(nodes(p)) AS nodes\n      RETURN nodes, none(x IN nodes WHERE x.name = 'a') AS result",
        graph=graph_fixture_from_create(
            """
            CREATE (s1:SRelationships), (s2:SNodes)
                  CREATE (a:A {name: 'a'}), (b:B {name: 'b'})
                  CREATE (aa:A {name: 'a'}), (ab:B {name: 'b'}),
                         (ba:A {name: 'a'}), (bb:B {name: 'b'})
                  CREATE (aaa:A {name: 'a'}), (aab:B {name: 'b'}),
                         (aba:A {name: 'a'}), (abb:B {name: 'b'}),
                         (baa:A {name: 'a'}), (bab:B {name: 'b'}),
                         (bba:A {name: 'a'}), (bbb:B {name: 'b'})
                  CREATE (s1)-[:I]->(s2),
                         (s2)-[:RA {name: 'a'}]->(a), (s2)-[:RB {name: 'b'}]->(b)
                  CREATE (a)-[:RA {name: 'a'}]->(aa), (a)-[:RB {name: 'b'}]->(ab),
                         (b)-[:RA {name: 'a'}]->(ba), (b)-[:RB {name: 'b'}]->(bb)
                  CREATE (aa)-[:RA {name: 'a'}]->(aaa), (aa)-[:RB {name: 'b'}]->(aab),
                         (ab)-[:RA {name: 'a'}]->(aba), (ab)-[:RB {name: 'b'}]->(abb),
                         (ba)-[:RA {name: 'a'}]->(baa), (ba)-[:RB {name: 'b'}]->(bab),
                         (bb)-[:RA {name: 'a'}]->(bba), (bb)-[:RB {name: 'b'}]->(bbb)
            """
        ),
        expected=Expected(
            rows=[
            {'nodes': '[]', 'result': 'true'},
            {'nodes': "[(:A {name: 'a'})]", 'result': 'false'},
            {'nodes': "[(:A {name: 'a'}), (:A {name: 'a'})]", 'result': 'false'},
            {'nodes': "[(:A {name: 'a'}), (:A {name: 'a'}), (:A {name: 'a'})]", 'result': 'false'},
            {'nodes': "[(:A {name: 'a'}), (:A {name: 'a'}), (:B {name: 'b'})]", 'result': 'false'},
            {'nodes': "[(:A {name: 'a'}), (:B {name: 'b'})]", 'result': 'false'},
            {'nodes': "[(:A {name: 'a'}), (:B {name: 'b'}), (:A {name: 'a'})]", 'result': 'false'},
            {'nodes': "[(:A {name: 'a'}), (:B {name: 'b'}), (:B {name: 'b'})]", 'result': 'false'},
            {'nodes': "[(:B {name: 'b'})]", 'result': 'true'},
            {'nodes': "[(:B {name: 'b'}), (:A {name: 'a'})]", 'result': 'false'},
            {'nodes': "[(:B {name: 'b'}), (:A {name: 'a'}), (:A {name: 'a'})]", 'result': 'false'},
            {'nodes': "[(:B {name: 'b'}), (:A {name: 'a'}), (:B {name: 'b'})]", 'result': 'false'},
            {'nodes': "[(:B {name: 'b'}), (:B {name: 'b'})]", 'result': 'true'},
            {'nodes': "[(:B {name: 'b'}), (:B {name: 'b'}), (:A {name: 'a'})]", 'result': 'false'},
            {'nodes': "[(:B {name: 'b'}), (:B {name: 'b'}), (:B {name: 'b'})]", 'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-9',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[9] None quantifier on list containing relationships',
        cypher="MATCH p = (:SRelationships)-[*0..4]->(x)\n      WITH tail(relationships(p)) AS relationships, COUNT(*) AS c\n      RETURN relationships, none(x IN relationships WHERE x.name = 'a') AS result",
        graph=graph_fixture_from_create(
            """
            CREATE (s1:SRelationships), (s2:SNodes)
                  CREATE (a:A {name: 'a'}), (b:B {name: 'b'})
                  CREATE (aa:A {name: 'a'}), (ab:B {name: 'b'}),
                         (ba:A {name: 'a'}), (bb:B {name: 'b'})
                  CREATE (aaa:A {name: 'a'}), (aab:B {name: 'b'}),
                         (aba:A {name: 'a'}), (abb:B {name: 'b'}),
                         (baa:A {name: 'a'}), (bab:B {name: 'b'}),
                         (bba:A {name: 'a'}), (bbb:B {name: 'b'})
                  CREATE (s1)-[:I]->(s2),
                         (s2)-[:RA {name: 'a'}]->(a), (s2)-[:RB {name: 'b'}]->(b)
                  CREATE (a)-[:RA {name: 'a'}]->(aa), (a)-[:RB {name: 'b'}]->(ab),
                         (b)-[:RA {name: 'a'}]->(ba), (b)-[:RB {name: 'b'}]->(bb)
                  CREATE (aa)-[:RA {name: 'a'}]->(aaa), (aa)-[:RB {name: 'b'}]->(aab),
                         (ab)-[:RA {name: 'a'}]->(aba), (ab)-[:RB {name: 'b'}]->(abb),
                         (ba)-[:RA {name: 'a'}]->(baa), (ba)-[:RB {name: 'b'}]->(bab),
                         (bb)-[:RA {name: 'a'}]->(bba), (bb)-[:RB {name: 'b'}]->(bbb)
            """
        ),
        expected=Expected(
            rows=[
            {'relationships': '[]', 'result': 'true'},
            {'relationships': "[[:RA {name: 'a'}]]", 'result': 'false'},
            {'relationships': "[[:RA {name: 'a'}], [:RA {name: 'a'}]]", 'result': 'false'},
            {'relationships': "[[:RA {name: 'a'}], [:RA {name: 'a'}], [:RA {name: 'a'}]]", 'result': 'false'},
            {'relationships': "[[:RA {name: 'a'}], [:RA {name: 'a'}], [:RB {name: 'b'}]]", 'result': 'false'},
            {'relationships': "[[:RA {name: 'a'}], [:RB {name: 'b'}]]", 'result': 'false'},
            {'relationships': "[[:RA {name: 'a'}], [:RB {name: 'b'}], [:RA {name: 'a'}]]", 'result': 'false'},
            {'relationships': "[[:RA {name: 'a'}], [:RB {name: 'b'}], [:RB {name: 'b'}]]", 'result': 'false'},
            {'relationships': "[[:RB {name: 'b'}]]", 'result': 'true'},
            {'relationships': "[[:RB {name: 'b'}], [:RA {name: 'a'}]]", 'result': 'false'},
            {'relationships': "[[:RB {name: 'b'}], [:RA {name: 'a'}], [:RA {name: 'a'}]]", 'result': 'false'},
            {'relationships': "[[:RB {name: 'b'}], [:RA {name: 'a'}], [:RB {name: 'b'}]]", 'result': 'false'},
            {'relationships': "[[:RB {name: 'b'}], [:RB {name: 'b'}]]", 'result': 'true'},
            {'relationships': "[[:RB {name: 'b'}], [:RB {name: 'b'}], [:RA {name: 'a'}]]", 'result': 'false'},
            {'relationships': "[[:RB {name: 'b'}], [:RB {name: 'b'}], [:RB {name: 'b'}]]", 'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-10-1',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[10] None quantifier on lists containing nulls (example 1)',
        cypher='RETURN none(x IN [null] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-10-2',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[10] None quantifier on lists containing nulls (example 2)',
        cypher='RETURN none(x IN [null, null] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-10-3',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[10] None quantifier on lists containing nulls (example 3)',
        cypher='RETURN none(x IN [0, null] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-10-4',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[10] None quantifier on lists containing nulls (example 4)',
        cypher='RETURN none(x IN [2, null] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-10-5',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[10] None quantifier on lists containing nulls (example 5)',
        cypher='RETURN none(x IN [null, 2] WHERE x = 2) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-10-6',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[10] None quantifier on lists containing nulls (example 6)',
        cypher='RETURN none(x IN [34, 0, null, 5, 900] WHERE x < 10) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-10-7',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[10] None quantifier on lists containing nulls (example 7)',
        cypher='RETURN none(x IN [34, 10, null, 15, 900] WHERE x < 10) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-10-8',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[10] None quantifier on lists containing nulls (example 8)',
        cypher='RETURN none(x IN [4, 0, null, -15, 9] WHERE x < 10) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-1',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 1)',
        cypher='RETURN none(x IN [] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-2',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 2)',
        cypher='RETURN none(x IN [0] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-3',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 3)',
        cypher='RETURN none(x IN [34, 0, 8, 900] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-4',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 4)',
        cypher='RETURN none(x IN [null] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-5',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 5)',
        cypher='RETURN none(x IN [null, null] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-6',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 6)',
        cypher='RETURN none(x IN [0, null] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-7',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 7)',
        cypher='RETURN none(x IN [2, null] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-8',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 8)',
        cypher='RETURN none(x IN [null, 2] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-9',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 9)',
        cypher='RETURN none(x IN [34, 0, null, 8, 900] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-10',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 10)',
        cypher='RETURN none(x IN [34, 0, null, 8, null] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-11',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 11)',
        cypher='RETURN none(x IN [null, 123, null, null] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-11-12',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[11] None quantifier with IS NULL predicate (example 12)',
        cypher='RETURN none(x IN [null, null, null, null] WHERE x IS NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-1',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 1)',
        cypher='RETURN none(x IN [] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-2',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 2)',
        cypher='RETURN none(x IN [0] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-3',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 3)',
        cypher='RETURN none(x IN [34, 0, 8, 900] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-4',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 4)',
        cypher='RETURN none(x IN [null] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-5',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 5)',
        cypher='RETURN none(x IN [null, null] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-6',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 6)',
        cypher='RETURN none(x IN [0, null] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-7',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 7)',
        cypher='RETURN none(x IN [2, null] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-8',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 8)',
        cypher='RETURN none(x IN [null, 2] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-9',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 9)',
        cypher='RETURN none(x IN [34, 0, null, 8, 900] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-10',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 10)',
        cypher='RETURN none(x IN [34, 0, null, 8, null] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-11',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 11)',
        cypher='RETURN none(x IN [null, 123, null, null] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-12-12',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[12] None quantifier with IS NOT NULL predicate (example 12)',
        cypher='RETURN none(x IN [null, null, null, null] WHERE x IS NOT NULL) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-13',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[13] None quantifier is true if the predicate is statically false and the list is not empty',
        cypher="RETURN none(x IN [1, null, true, 4.5, 'abc', false] WHERE false) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-14',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[14] None quantifier is false if the predicate is statically true and the list is not empty',
        cypher="RETURN none(x IN [1, null, true, 4.5, 'abc', false] WHERE true) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-15-1',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[15] Fail none quantifier on type mismatch between list elements and predicate (example 1)',
        cypher="RETURN none(x IN ['Clara'] WHERE x % 2 = 0) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-15-2',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[15] Fail none quantifier on type mismatch between list elements and predicate (example 2)',
        cypher='RETURN none(x IN [false, true] WHERE x % 2 = 0) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier1-15-3',
        feature_path='tck/features/expressions/quantifier/Quantifier1.feature',
        scenario='[15] Fail none quantifier on type mismatch between list elements and predicate (example 3)',
        cypher="RETURN none(x IN ['Clara', 'Bob', 'Dave', 'Alice'] WHERE x % 2 = 0) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
