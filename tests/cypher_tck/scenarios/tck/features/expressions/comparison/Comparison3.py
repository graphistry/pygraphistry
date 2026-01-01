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
        key='expr-comparison3-1',
        feature_path='tck/features/expressions/comparison/Comparison3.feature',
        scenario='[1] Handling numerical ranges 1',
        cypher='MATCH (n)\n      WHERE 1 < n.num < 3\n      RETURN n.num',
        graph=graph_fixture_from_create(
            """
            UNWIND [1, 2, 3] AS i
                  CREATE ({num: i})
            """
        ),
        expected=Expected(
            rows=[
            {'n.num': 2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison3-2',
        feature_path='tck/features/expressions/comparison/Comparison3.feature',
        scenario='[2] Handling numerical ranges 2',
        cypher='MATCH (n)\n      WHERE 1 < n.num <= 3\n      RETURN n.num',
        graph=graph_fixture_from_create(
            """
            UNWIND [1, 2, 3] AS i
                  CREATE ({num: i})
            """
        ),
        expected=Expected(
            rows=[
            {'n.num': 2},
            {'n.num': 3}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison3-3',
        feature_path='tck/features/expressions/comparison/Comparison3.feature',
        scenario='[3] Handling numerical ranges 3',
        cypher='MATCH (n)\n      WHERE 1 <= n.num < 3\n      RETURN n.num',
        graph=graph_fixture_from_create(
            """
            UNWIND [1, 2, 3] AS i
                  CREATE ({num: i})
            """
        ),
        expected=Expected(
            rows=[
            {'n.num': 1},
            {'n.num': 2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison3-4',
        feature_path='tck/features/expressions/comparison/Comparison3.feature',
        scenario='[4] Handling numerical ranges 4',
        cypher='MATCH (n)\n      WHERE 1 <= n.num <= 3\n      RETURN n.num',
        graph=graph_fixture_from_create(
            """
            UNWIND [1, 2, 3] AS i
                  CREATE ({num: i})
            """
        ),
        expected=Expected(
            rows=[
            {'n.num': 1},
            {'n.num': 2},
            {'n.num': 3}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison3-5',
        feature_path='tck/features/expressions/comparison/Comparison3.feature',
        scenario='[5] Handling string ranges 1',
        cypher="MATCH (n)\n      WHERE 'a' < n.name < 'c'\n      RETURN n.name",
        graph=graph_fixture_from_create(
            """
            UNWIND ['a', 'b', 'c'] AS c
                  CREATE ({name: c})
            """
        ),
        expected=Expected(
            rows=[
            {'n.name': "'b'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison3-6',
        feature_path='tck/features/expressions/comparison/Comparison3.feature',
        scenario='[6] Handling string ranges 2',
        cypher="MATCH (n)\n      WHERE 'a' < n.name <= 'c'\n      RETURN n.name",
        graph=graph_fixture_from_create(
            """
            UNWIND ['a', 'b', 'c'] AS c
                  CREATE ({name: c})
            """
        ),
        expected=Expected(
            rows=[
            {'n.name': "'b'"},
            {'n.name': "'c'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison3-7',
        feature_path='tck/features/expressions/comparison/Comparison3.feature',
        scenario='[7] Handling string ranges 3',
        cypher="MATCH (n)\n      WHERE 'a' <= n.name < 'c'\n      RETURN n.name",
        graph=graph_fixture_from_create(
            """
            UNWIND ['a', 'b', 'c'] AS c
                  CREATE ({name: c})
            """
        ),
        expected=Expected(
            rows=[
            {'n.name': "'a'"},
            {'n.name': "'b'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison3-8',
        feature_path='tck/features/expressions/comparison/Comparison3.feature',
        scenario='[8] Handling string ranges 4',
        cypher="MATCH (n)\n      WHERE 'a' <= n.name <= 'c'\n      RETURN n.name",
        graph=graph_fixture_from_create(
            """
            UNWIND ['a', 'b', 'c'] AS c
                  CREATE ({name: c})
            """
        ),
        expected=Expected(
            rows=[
            {'n.name': "'a'"},
            {'n.name': "'b'"},
            {'n.name': "'c'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison3-9',
        feature_path='tck/features/expressions/comparison/Comparison3.feature',
        scenario='[9] Handling empty range',
        cypher='MATCH (n)\n      WHERE 10 < n.num <= 3\n      RETURN n.num',
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3})
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),
]
