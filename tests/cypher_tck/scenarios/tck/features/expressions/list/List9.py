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
        key='expr-list9-1',
        feature_path='tck/features/expressions/list/List9.feature',
        scenario='[1] Returning nested expressions based on list property',
        cypher='MATCH (n:TheLabel)\n      SET n.array = [1, 2, 3, 4, 5]\n      RETURN tail(tail(n.array))',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[
            {'tail(tail(n.array))': '[3, 4, 5]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),
]
