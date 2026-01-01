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
        key='delete3-1',
        feature_path='tck/features/clauses/delete/Delete3.feature',
        scenario='[1] Detach deleting paths',
        cypher='MATCH p = (:X)-->()-->()-->()\n      DETACH DELETE p',
        graph=graph_fixture_from_create(
            """
            CREATE (x:X), (n1), (n2), (n3)
                  CREATE (x)-[:R]->(n1)
                  CREATE (n1)-[:R]->(n2)
                  CREATE (n2)-[:R]->(n3)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete3-2',
        feature_path='tck/features/clauses/delete/Delete3.feature',
        scenario='[2] Delete on null path',
        cypher='OPTIONAL MATCH p = ()-->()\n      DETACH DELETE p',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),
]
