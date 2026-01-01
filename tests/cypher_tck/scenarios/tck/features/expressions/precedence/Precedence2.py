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
        key='expr-precedence2-1-1',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 1)',
        cypher='RETURN 4 * 2 + 3 * 2 AS a,\n             4 * 2 + (3 * 2) AS b,\n             4 * (2 + 3) * 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 14, 'b': 14, 'c': 40}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-2',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 2)',
        cypher='RETURN 4 * 2 + 3 / 2 AS a,\n             4 * 2 + (3 / 2) AS b,\n             4 * (2 + 3) / 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 9, 'b': 9, 'c': 10}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-3',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 3)',
        cypher='RETURN 4 * 2 + 3 % 2 AS a,\n             4 * 2 + (3 % 2) AS b,\n             4 * (2 + 3) % 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 9, 'b': 9, 'c': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-4',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 4)',
        cypher='RETURN 4 * 2 - 3 * 2 AS a,\n             4 * 2 - (3 * 2) AS b,\n             4 * (2 - 3) * 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 2, 'b': 2, 'c': -8}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-5',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 5)',
        cypher='RETURN 4 * 2 - 3 / 2 AS a,\n             4 * 2 - (3 / 2) AS b,\n             4 * (2 - 3) / 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 7, 'b': 7, 'c': -2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-6',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 6)',
        cypher='RETURN 4 * 2 - 3 % 2 AS a,\n             4 * 2 - (3 % 2) AS b,\n             4 * (2 - 3) % 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 7, 'b': 7, 'c': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-7',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 7)',
        cypher='RETURN 4 / 2 + 3 * 2 AS a,\n             4 / 2 + (3 * 2) AS b,\n             4 / (2 + 3) * 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 8, 'b': 8, 'c': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-8',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 8)',
        cypher='RETURN 4 / 2 + 3 / 2 AS a,\n             4 / 2 + (3 / 2) AS b,\n             4 / (2 + 3) / 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 3, 'b': 3, 'c': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-9',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 9)',
        cypher='RETURN 4 / 2 + 3 % 2 AS a,\n             4 / 2 + (3 % 2) AS b,\n             4 / (2 + 3) % 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 3, 'b': 3, 'c': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-10',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 10)',
        cypher='RETURN 4 / 2 - 3 * 2 AS a,\n             4 / 2 - (3 * 2) AS b,\n             4 / (2 - 3) * 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': -4, 'b': -4, 'c': -8}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-11',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 11)',
        cypher='RETURN 4 / 2 - 3 / 2 AS a,\n             4 / 2 - (3 / 2) AS b,\n             4 / (2 - 3) / 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 1, 'b': 1, 'c': -2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-12',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 12)',
        cypher='RETURN 4 / 2 - 3 % 2 AS a,\n             4 / 2 - (3 % 2) AS b,\n             4 / (2 - 3) % 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 1, 'b': 1, 'c': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-13',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 13)',
        cypher='RETURN 4 % 2 + 3 * 2 AS a,\n             4 % 2 + (3 * 2) AS b,\n             4 % (2 + 3) * 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 6, 'b': 6, 'c': 8}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-14',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 14)',
        cypher='RETURN 4 % 2 + 3 / 2 AS a,\n             4 % 2 + (3 / 2) AS b,\n             4 % (2 + 3) / 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 1, 'b': 1, 'c': 2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-15',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 15)',
        cypher='RETURN 4 % 2 + 3 % 2 AS a,\n             4 % 2 + (3 % 2) AS b,\n             4 % (2 + 3) % 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 1, 'b': 1, 'c': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-16',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 16)',
        cypher='RETURN 4 % 2 - 3 * 2 AS a,\n             4 % 2 - (3 * 2) AS b,\n             4 % (2 - 3) * 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': -6, 'b': -6, 'c': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-17',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 17)',
        cypher='RETURN 4 % 2 - 3 / 2 AS a,\n             4 % 2 - (3 / 2) AS b,\n             4 % (2 - 3) / 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': -1, 'b': -1, 'c': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-1-18',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[1] Numeric multiplicative operations takes precedence over numeric additive operations (example 18)',
        cypher='RETURN 4 % 2 - 3 % 2 AS a,\n             4 % 2 - (3 % 2) AS b,\n             4 % (2 - 3) % 2 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': -1, 'b': -1, 'c': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-2-1',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[2] Exponentiation takes precedence over numeric multiplicative operations (example 1)',
        cypher='RETURN 4 ^ 3 * 2 ^ 3 AS a,\n             (4 ^ 3) * (2 ^ 3) AS b,\n             4 ^ (3 * 2) ^ 3 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 512.0, 'b': 512.0, 'c': 68719476736.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-2-2',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[2] Exponentiation takes precedence over numeric multiplicative operations (example 2)',
        cypher='RETURN 4 ^ 3 / 2 ^ 3 AS a,\n             (4 ^ 3) / (2 ^ 3) AS b,\n             4 ^ (3 / 2) ^ 3 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 8.0, 'b': 8.0, 'c': 64.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-2-3',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[2] Exponentiation takes precedence over numeric multiplicative operations (example 3)',
        cypher='RETURN 4 ^ 3 % 2 ^ 3 AS a,\n             (4 ^ 3) % (2 ^ 3) AS b,\n             4 ^ (3 % 2) ^ 3 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 0.0, 'b': 0.0, 'c': 64.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-3-1',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[3] Exponentiation takes precedence over numeric additive operations (example 1)',
        cypher='RETURN 4 ^ 3 + 2 ^ 3 AS a,\n             (4 ^ 3) + (2 ^ 3) AS b,\n             4 ^ (3 + 2) ^ 3 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 72.0, 'b': 72.0, 'c': 1073741824.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-3-2',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[3] Exponentiation takes precedence over numeric additive operations (example 2)',
        cypher='RETURN 4 ^ 3 - 2 ^ 3 AS a,\n             (4 ^ 3) - (2 ^ 3) AS b,\n             4 ^ (3 - 2) ^ 3 AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 56.0, 'b': 56.0, 'c': 64.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-4',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[4] Numeric unary negative takes precedence over exponentiation',
        cypher='RETURN -3 ^ 2 AS a,\n             (-3) ^ 2 AS b,\n             -(3 ^ 2) AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 9.0, 'b': 9.0, 'c': -9.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-5-1',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[5] Numeric unary negative takes precedence over numeric additive operations (example 1)',
        cypher='RETURN -3 + 2 AS a,\n             (-3) + 2 AS b,\n             -(3 + 2) AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': -1, 'b': -1, 'c': -5}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence2-5-2',
        feature_path='tck/features/expressions/precedence/Precedence2.feature',
        scenario='[5] Numeric unary negative takes precedence over numeric additive operations (example 2)',
        cypher='RETURN -3 - 2 AS a,\n             (-3) - 2 AS b,\n             -(3 - 2) AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': -5, 'b': -5, 'c': -1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),
]
