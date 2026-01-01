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
        key='expr-string11-1',
        feature_path='tck/features/expressions/string/String11.feature',
        scenario='[1] Combining prefix and suffix search',
        cypher="MATCH (a)\n      WHERE a.name STARTS WITH 'a'\n        AND a.name ENDS WITH 'f'\n      RETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[
            {'a': "(:TheLabel {name: 'abcdef'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-string11-2',
        feature_path='tck/features/expressions/string/String11.feature',
        scenario='[2] Combining prefix, suffix, and substring search',
        cypher="MATCH (a)\n      WHERE a.name STARTS WITH 'A'\n        AND a.name CONTAINS 'C'\n        AND a.name ENDS WITH 'EF'\n      RETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[
            {'a': "(:TheLabel {name: 'ABCDEF'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),
]
