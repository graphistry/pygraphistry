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
        key='expr-quantifier6-1-1',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[1] Single quantifier can nest itself and other quantifiers on nested lists (example 1)',
        cypher="RETURN single(x IN [['abc'], ['abc', 'def']] WHERE none(y IN x WHERE y = 'def')) AS result",
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
        key='expr-quantifier6-1-2',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[1] Single quantifier can nest itself and other quantifiers on nested lists (example 2)',
        cypher="RETURN single(x IN [['abc'], ['abc', 'def']] WHERE none(y IN x WHERE y = 'ghi')) AS result",
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
        key='expr-quantifier6-1-3',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[1] Single quantifier can nest itself and other quantifiers on nested lists (example 3)',
        cypher="RETURN single(x IN [['abc'], ['abc', 'def']] WHERE single(y IN x WHERE y = 'def')) AS result",
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
        key='expr-quantifier6-1-4',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[1] Single quantifier can nest itself and other quantifiers on nested lists (example 4)',
        cypher="RETURN single(x IN [['abc'], ['abc', 'def']] WHERE single(y IN x WHERE y = 'abc')) AS result",
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
        key='expr-quantifier6-1-5',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[1] Single quantifier can nest itself and other quantifiers on nested lists (example 5)',
        cypher="RETURN single(x IN [['abc'], ['abc', 'def']] WHERE any(y IN x WHERE y = 'def')) AS result",
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
        key='expr-quantifier6-1-6',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[1] Single quantifier can nest itself and other quantifiers on nested lists (example 6)',
        cypher="RETURN single(x IN [['abc'], ['abc', 'def']] WHERE any(y IN x WHERE y = 'abc')) AS result",
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
        key='expr-quantifier6-1-7',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[1] Single quantifier can nest itself and other quantifiers on nested lists (example 7)',
        cypher="RETURN single(x IN [['abc'], ['abc', 'def']] WHERE all(y IN x WHERE y = 'abc')) AS result",
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
        key='expr-quantifier6-1-8',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[1] Single quantifier can nest itself and other quantifiers on nested lists (example 8)',
        cypher="RETURN single(x IN [['abc'], ['abc', 'def']] WHERE all(y IN x WHERE y = 'def')) AS result",
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
        key='expr-quantifier6-2-1',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[2] Single quantifier can nest itself and other quantifiers on the same list (example 1)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN single(x IN list WHERE none(y IN list WHERE x < y)) AS result',
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
        key='expr-quantifier6-2-2',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[2] Single quantifier can nest itself and other quantifiers on the same list (example 2)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN single(x IN list WHERE none(y IN list WHERE x % y = 0)) AS result',
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
        key='expr-quantifier6-2-3',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[2] Single quantifier can nest itself and other quantifiers on the same list (example 3)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN single(x IN list WHERE single(y IN list WHERE x + y < 5)) AS result',
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
        key='expr-quantifier6-2-4',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[2] Single quantifier can nest itself and other quantifiers on the same list (example 4)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN single(x IN list WHERE single(y IN list WHERE x % y = 1)) AS result',
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
        key='expr-quantifier6-2-5',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[2] Single quantifier can nest itself and other quantifiers on the same list (example 5)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN single(x IN list WHERE any(y IN list WHERE 2 * x + y > 25)) AS result',
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
        key='expr-quantifier6-2-6',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[2] Single quantifier can nest itself and other quantifiers on the same list (example 6)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN single(x IN list WHERE any(y IN list WHERE x < y)) AS result',
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
        key='expr-quantifier6-2-7',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[2] Single quantifier can nest itself and other quantifiers on the same list (example 7)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN single(x IN list WHERE all(y IN list WHERE x <= y)) AS result',
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
        key='expr-quantifier6-2-8',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[2] Single quantifier can nest itself and other quantifiers on the same list (example 8)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN single(x IN list WHERE all(y IN list WHERE x <= y + 1)) AS result',
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
        key='expr-quantifier6-3-1',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[3] Single quantifier is equal whether the size of the list filtered with same the predicate is one (example 1)',
        cypher='RETURN single(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x = 2) = (size([x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x = 2 | x]) = 1) AS result',
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
        key='expr-quantifier6-3-2',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[3] Single quantifier is equal whether the size of the list filtered with same the predicate is one (example 2)',
        cypher='RETURN single(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 2 = 0) = (size([x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 2 = 0 | x]) = 1) AS result',
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
        key='expr-quantifier6-3-3',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[3] Single quantifier is equal whether the size of the list filtered with same the predicate is one (example 3)',
        cypher='RETURN single(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 3 = 0) = (size([x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 3 = 0 | x]) = 1) AS result',
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
        key='expr-quantifier6-3-4',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[3] Single quantifier is equal whether the size of the list filtered with same the predicate is one (example 4)',
        cypher='RETURN single(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x < 7) = (size([x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x < 7 | x]) = 1) AS result',
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
        key='expr-quantifier6-3-5',
        feature_path='tck/features/expressions/quantifier/Quantifier6.feature',
        scenario='[3] Single quantifier is equal whether the size of the list filtered with same the predicate is one (example 5)',
        cypher='RETURN single(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x >= 3) = (size([x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x >= 3 | x]) = 1) AS result',
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
]
