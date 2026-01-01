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
        key='expr-mathematical2-1',
        feature_path='tck/features/expressions/mathematical/Mathematical2.feature',
        scenario='[1] Allow addition',
        cypher='MATCH (a)\n      WHERE a.id = 1337\n      RETURN a.version + 5',
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 1337, version: 99})
            """
        ),
        expected=Expected(
            rows=[
            {'a.version + 5': 104}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'mathematical', 'meta-xfail', 'xfail'),
    ),
]
