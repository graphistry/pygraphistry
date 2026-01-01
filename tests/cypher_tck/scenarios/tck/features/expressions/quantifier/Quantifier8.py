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
        key='expr-quantifier8-1-1',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[1] All quantifier can nest itself and other quantifiers on nested lists (example 1)',
        cypher="RETURN all(x IN [['abc'], ['abc', 'def']] WHERE none(y IN x WHERE y = 'ghi')) AS result",
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
        key='expr-quantifier8-1-2',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[1] All quantifier can nest itself and other quantifiers on nested lists (example 2)',
        cypher="RETURN all(x IN [['abc'], ['abc', 'def']] WHERE none(y IN x WHERE y = 'def')) AS result",
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
        key='expr-quantifier8-1-3',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[1] All quantifier can nest itself and other quantifiers on nested lists (example 3)',
        cypher="RETURN all(x IN [['abc'], ['abc', 'def']] WHERE single(y IN x WHERE y = 'abc')) AS result",
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
        key='expr-quantifier8-1-4',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[1] All quantifier can nest itself and other quantifiers on nested lists (example 4)',
        cypher="RETURN all(x IN [['abc'], ['abc', 'def']] WHERE single(y IN x WHERE y = 'ghi')) AS result",
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
        key='expr-quantifier8-1-5',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[1] All quantifier can nest itself and other quantifiers on nested lists (example 5)',
        cypher="RETURN all(x IN [['abc'], ['abc', 'def']] WHERE any(y IN x WHERE y = 'abc')) AS result",
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
        key='expr-quantifier8-1-6',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[1] All quantifier can nest itself and other quantifiers on nested lists (example 6)',
        cypher="RETURN all(x IN [['abc'], ['abc', 'def']] WHERE any(y IN x WHERE y = 'ghi')) AS result",
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
        key='expr-quantifier8-1-7',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[1] All quantifier can nest itself and other quantifiers on nested lists (example 7)',
        cypher="RETURN all(x IN [['abc'], ['abc', 'def']] WHERE all(y IN x WHERE y <> 'ghi')) AS result",
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
        key='expr-quantifier8-1-8',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[1] All quantifier can nest itself and other quantifiers on nested lists (example 8)',
        cypher="RETURN all(x IN [['abc'], ['abc', 'def']] WHERE all(y IN x WHERE y = 'abc')) AS result",
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
        key='expr-quantifier8-2-1',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[2] All quantifier can nest itself and other quantifiers on the same list (example 1)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN all(x IN list WHERE none(y IN list WHERE x = 10 * y)) AS result',
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
        key='expr-quantifier8-2-2',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[2] All quantifier can nest itself and other quantifiers on the same list (example 2)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN all(x IN list WHERE none(y IN list WHERE x = y)) AS result',
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
        key='expr-quantifier8-2-3',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[2] All quantifier can nest itself and other quantifiers on the same list (example 3)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN all(x IN list WHERE single(y IN list WHERE x = y)) AS result',
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
        key='expr-quantifier8-2-4',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[2] All quantifier can nest itself and other quantifiers on the same list (example 4)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN all(x IN list WHERE single(y IN list WHERE x < y)) AS result',
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
        key='expr-quantifier8-2-5',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[2] All quantifier can nest itself and other quantifiers on the same list (example 5)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN all(x IN list WHERE any(y IN list WHERE x % y = 0)) AS result',
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
        key='expr-quantifier8-2-6',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[2] All quantifier can nest itself and other quantifiers on the same list (example 6)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN all(x IN list WHERE any(y IN list WHERE x < y)) AS result',
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
        key='expr-quantifier8-2-7',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[2] All quantifier can nest itself and other quantifiers on the same list (example 7)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN all(x IN list WHERE all(y IN list WHERE abs(x - y) < 10)) AS result',
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
        key='expr-quantifier8-2-8',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[2] All quantifier can nest itself and other quantifiers on the same list (example 8)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS list\n      RETURN all(x IN list WHERE all(y IN list WHERE x < y + 7)) AS result',
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
        key='expr-quantifier8-3-1',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[3] All quantifier is equal the none quantifier on the boolean negative of the predicate (example 1)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x = 2) = none(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE NOT (x = 2)) AS result',
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
        key='expr-quantifier8-3-2',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[3] All quantifier is equal the none quantifier on the boolean negative of the predicate (example 2)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 2 = 0) = none(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE NOT (x % 2 = 0)) AS result',
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
        key='expr-quantifier8-3-3',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[3] All quantifier is equal the none quantifier on the boolean negative of the predicate (example 3)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 3 = 0) = none(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE NOT (x % 3 = 0)) AS result',
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
        key='expr-quantifier8-3-4',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[3] All quantifier is equal the none quantifier on the boolean negative of the predicate (example 4)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x < 7) = none(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE NOT (x < 7)) AS result',
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
        key='expr-quantifier8-3-5',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[3] All quantifier is equal the none quantifier on the boolean negative of the predicate (example 5)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x >= 3) = none(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE NOT (x >= 3)) AS result',
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
        key='expr-quantifier8-4-1',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[4] All quantifier is equal the boolean negative of the any quantifier on the boolean negative of the predicate (example 1)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x = 2) = (NOT any(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE NOT (x = 2))) AS result',
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
        key='expr-quantifier8-4-2',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[4] All quantifier is equal the boolean negative of the any quantifier on the boolean negative of the predicate (example 2)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 2 = 0) = (NOT any(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE NOT (x % 2 = 0))) AS result',
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
        key='expr-quantifier8-4-3',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[4] All quantifier is equal the boolean negative of the any quantifier on the boolean negative of the predicate (example 3)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 3 = 0) = (NOT any(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE NOT (x % 3 = 0))) AS result',
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
        key='expr-quantifier8-4-4',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[4] All quantifier is equal the boolean negative of the any quantifier on the boolean negative of the predicate (example 4)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x < 7) = (NOT any(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE NOT (x < 7))) AS result',
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
        key='expr-quantifier8-4-5',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[4] All quantifier is equal the boolean negative of the any quantifier on the boolean negative of the predicate (example 5)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x >= 3) = (NOT any(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE NOT (x >= 3))) AS result',
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
        key='expr-quantifier8-5-1',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[5] All quantifier is equal whether the size of the list filtered with same the predicate is equal the size of the unfiltered list (example 1)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x = 2) = (size([x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x = 2 | x]) = size([1, 2, 3, 4, 5, 6, 7, 8, 9])) AS result',
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
        key='expr-quantifier8-5-2',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[5] All quantifier is equal whether the size of the list filtered with same the predicate is equal the size of the unfiltered list (example 2)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 2 = 0) = (size([x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 2 = 0 | x]) = size([1, 2, 3, 4, 5, 6, 7, 8, 9])) AS result',
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
        key='expr-quantifier8-5-3',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[5] All quantifier is equal whether the size of the list filtered with same the predicate is equal the size of the unfiltered list (example 3)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 3 = 0) = (size([x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x % 3 = 0 | x]) = size([1, 2, 3, 4, 5, 6, 7, 8, 9])) AS result',
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
        key='expr-quantifier8-5-4',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[5] All quantifier is equal whether the size of the list filtered with same the predicate is equal the size of the unfiltered list (example 4)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x < 7) = (size([x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x < 7 | x]) = size([1, 2, 3, 4, 5, 6, 7, 8, 9])) AS result',
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
        key='expr-quantifier8-5-5',
        feature_path='tck/features/expressions/quantifier/Quantifier8.feature',
        scenario='[5] All quantifier is equal whether the size of the list filtered with same the predicate is equal the size of the unfiltered list (example 5)',
        cypher='RETURN all(x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x >= 3) = (size([x IN [1, 2, 3, 4, 5, 6, 7, 8, 9] WHERE x >= 3 | x]) = size([1, 2, 3, 4, 5, 6, 7, 8, 9])) AS result',
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
