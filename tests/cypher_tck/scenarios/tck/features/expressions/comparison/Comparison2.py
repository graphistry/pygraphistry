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
        key='expr-comparison2-1',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario="[1] Comparing strings and integers using > in an AND'd predicate",
        cypher="MATCH (:Root)-->(i:Child)\n      WHERE i.var IS NOT NULL AND i.var > 'x'\n      RETURN i.var",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root)-[:T]->(:Child {var: 0}),
                         (root)-[:T]->(:Child {var: 'xx'}),
                         (root)-[:T]->(:Child)
            """
        ),
        expected=Expected(
            rows=[
            {'i.var': "'xx'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-2',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario="[2] Comparing strings and integers using > in a OR'd predicate",
        cypher="MATCH (:Root)-->(i:Child)\n      WHERE i.var IS NULL OR i.var > 'x'\n      RETURN i.var",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root)-[:T]->(:Child {var: 0}),
                         (root)-[:T]->(:Child {var: 'xx'}),
                         (root)-[:T]->(:Child)
            """
        ),
        expected=Expected(
            rows=[
            {'i.var': "'xx'"},
            {'i.var': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-3-1',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[3] Comparing across types yields null, except numbers (example 1)',
        cypher="MATCH p = (n)-[r]->()\n      WITH [n, r, p, '', 1, 3.14, true, null, [], {}] AS types\n      UNWIND range(0, size(types) - 1) AS i\n      UNWIND range(0, size(types) - 1) AS j\n      WITH types[i] AS lhs, types[j] AS rhs\n      WHERE i <> j\n      WITH lhs, rhs, lhs < rhs AS result\n      WHERE result\n      RETURN lhs, rhs",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
            {'lhs': 1, 'rhs': 3.14}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-3-2',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[3] Comparing across types yields null, except numbers (example 2)',
        cypher="MATCH p = (n)-[r]->()\n      WITH [n, r, p, '', 1, 3.14, true, null, [], {}] AS types\n      UNWIND range(0, size(types) - 1) AS i\n      UNWIND range(0, size(types) - 1) AS j\n      WITH types[i] AS lhs, types[j] AS rhs\n      WHERE i <> j\n      WITH lhs, rhs, lhs <= rhs AS result\n      WHERE result\n      RETURN lhs, rhs",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
            {'lhs': 1, 'rhs': 3.14}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-3-3',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[3] Comparing across types yields null, except numbers (example 3)',
        cypher="MATCH p = (n)-[r]->()\n      WITH [n, r, p, '', 1, 3.14, true, null, [], {}] AS types\n      UNWIND range(0, size(types) - 1) AS i\n      UNWIND range(0, size(types) - 1) AS j\n      WITH types[i] AS lhs, types[j] AS rhs\n      WHERE i <> j\n      WITH lhs, rhs, lhs >= rhs AS result\n      WHERE result\n      RETURN lhs, rhs",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
            {'lhs': 3.14, 'rhs': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-3-4',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[3] Comparing across types yields null, except numbers (example 4)',
        cypher="MATCH p = (n)-[r]->()\n      WITH [n, r, p, '', 1, 3.14, true, null, [], {}] AS types\n      UNWIND range(0, size(types) - 1) AS i\n      UNWIND range(0, size(types) - 1) AS j\n      WITH types[i] AS lhs, types[j] AS rhs\n      WHERE i <> j\n      WITH lhs, rhs, lhs > rhs AS result\n      WHERE result\n      RETURN lhs, rhs",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
            {'lhs': 3.14, 'rhs': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-4-1',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[4] Comparing lists (example 1)',
        cypher='RETURN [1, 0] >= [1] AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-4-2',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[4] Comparing lists (example 2)',
        cypher='RETURN [1, null] >= [1] AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-4-3',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[4] Comparing lists (example 3)',
        cypher='RETURN [1, 2] >= [1, null] AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-4-4',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[4] Comparing lists (example 4)',
        cypher="RETURN [1, 'a'] >= [1, null] AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-4-5',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[4] Comparing lists (example 5)',
        cypher='RETURN [1, 2] >= [3, null] AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-5-1',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[5] Comparing NaN (example 1)',
        cypher='RETURN 0.0 / 0.0 > 1 AS gt, 0.0 / 0.0 >= 1 AS gtE, 0.0 / 0.0 < <rhs> AS lt, 0.0 / 0.0 <= <rhs> AS ltE',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'gt': 'false', 'gtE': 'false', 'lt': 'false', 'ltE': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-5-2',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[5] Comparing NaN (example 2)',
        cypher='RETURN 0.0 / 0.0 > 1.0 AS gt, 0.0 / 0.0 >= 1.0 AS gtE, 0.0 / 0.0 < <rhs> AS lt, 0.0 / 0.0 <= <rhs> AS ltE',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'gt': 'false', 'gtE': 'false', 'lt': 'false', 'ltE': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-5-3',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[5] Comparing NaN (example 3)',
        cypher='RETURN 0.0 / 0.0 > 0.0 / 0.0 AS gt, 0.0 / 0.0 >= 0.0 / 0.0 AS gtE, 0.0 / 0.0 < <rhs> AS lt, 0.0 / 0.0 <= <rhs> AS ltE',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'gt': 'false', 'gtE': 'false', 'lt': 'false', 'ltE': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-5-4',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[5] Comparing NaN (example 4)',
        cypher="RETURN 0.0 / 0.0 > 'a' AS gt, 0.0 / 0.0 >= 'a' AS gtE, 0.0 / 0.0 < <rhs> AS lt, 0.0 / 0.0 <= <rhs> AS ltE",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'gt': 'null', 'gtE': 'null', 'lt': 'null', 'ltE': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-6-1',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[6] Comparability between numbers and strings (example 1)',
        cypher='RETURN 1.0 < <rhs> AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-6-2',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[6] Comparability between numbers and strings (example 2)',
        cypher='RETURN 1 < <rhs> AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-6-3',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[6] Comparability between numbers and strings (example 3)',
        cypher="RETURN '1.0' < <rhs> AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison2-6-4',
        feature_path='tck/features/expressions/comparison/Comparison2.feature',
        scenario='[6] Comparability between numbers and strings (example 4)',
        cypher="RETURN '1' < <rhs> AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),
]
