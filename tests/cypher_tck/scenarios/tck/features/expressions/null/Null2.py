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
        key='expr-null2-1',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[1] Property not null check on non-null node',
        cypher='MATCH (n)\n      RETURN n.missing IS NOT NULL,\n             n.exists IS NOT NULL',
        graph=graph_fixture_from_create(
            """
            CREATE ({exists: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'n.missing IS NOT NULL': 'false', 'n.exists IS NOT NULL': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-2',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[2] Property not null check on optional non-null node',
        cypher='OPTIONAL MATCH (n)\n      RETURN n.missing IS NOT NULL,\n             n.exists IS NOT NULL',
        graph=graph_fixture_from_create(
            """
            CREATE ({exists: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'n.missing IS NOT NULL': 'false', 'n.exists IS NOT NULL': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-3',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[3] Property not null check on null node',
        cypher='OPTIONAL MATCH (n)\n      RETURN n.missing IS NOT NULL',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'n.missing IS NOT NULL': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-4',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[4] A literal null is not IS NOT null',
        cypher='RETURN null IS NOT NULL AS value',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-1',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 1)',
        cypher="WITH {name: 'Mats', name2: 'Pontus'} AS map\n      RETURN map.name IS NOT NULL AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-2',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 2)',
        cypher="WITH {name: 'Mats', name2: 'Pontus'} AS map\n      RETURN map.name2 IS NOT NULL AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-3',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 3)',
        cypher="WITH {name: 'Mats', name2: null} AS map\n      RETURN map.name IS NOT NULL AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-4',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 4)',
        cypher="WITH {name: 'Mats', name2: null} AS map\n      RETURN map.name2 IS NOT NULL AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-5',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 5)',
        cypher='WITH {name: null} AS map\n      RETURN map.name IS NOT NULL AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-6',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 6)',
        cypher='WITH {name: null, name2: null} AS map\n      RETURN map.name IS NOT NULL AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-7',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 7)',
        cypher='WITH {name: null, name2: null} AS map\n      RETURN map.name2 IS NOT NULL AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-8',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 8)',
        cypher='WITH {notName: null, notName2: null} AS map\n      RETURN map.name IS NOT NULL AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-9',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 9)',
        cypher='WITH {notName: 0, notName2: null} AS map\n      RETURN map.name IS NOT NULL AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-10',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 10)',
        cypher='WITH {notName: 0} AS map\n      RETURN map.name IS NOT NULL AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-11',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 11)',
        cypher='WITH {} AS map\n      RETURN map.name IS NOT NULL AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-5-12',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[5] IS NOT NULL on a map (example 12)',
        cypher='WITH null AS map\n      RETURN map.name IS NOT NULL AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null2-6',
        feature_path='tck/features/expressions/null/Null2.feature',
        scenario='[6] IS NOT NULL is case insensitive',
        cypher='MATCH (n:X)\n      RETURN n, n.prop Is noT nULl AS b',
        graph=graph_fixture_from_create(
            """
            CREATE (a:X {prop: 42}), (:X)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:X {prop: 42})', 'b': 'true'},
            {'n': '(:X)', 'b': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),
]
