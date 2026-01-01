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
        key='delete4-1',
        feature_path='tck/features/clauses/delete/Delete4.feature',
        scenario='[1] Undirected expand followed by delete and count',
        cypher='MATCH (a)-[r]-(b)\n      DELETE r, a, b\n      RETURN count(*) AS c',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:R]->()
            """
        ),
        expected=Expected(
            rows=[
            {'c': 2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete4-2',
        feature_path='tck/features/clauses/delete/Delete4.feature',
        scenario='[2] Undirected variable length expand followed by delete and count',
        cypher='MATCH (a)-[*]-(b)\n      DETACH DELETE a, b\n      RETURN count(*) AS c',
        graph=graph_fixture_from_create(
            """
            CREATE (n1), (n2), (n3)
                  CREATE (n1)-[:R]->(n2)
                  CREATE (n2)-[:R]->(n3)
            """
        ),
        expected=Expected(
            rows=[
            {'c': 6}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete4-3',
        feature_path='tck/features/clauses/delete/Delete4.feature',
        scenario='[3] Create and delete in same query',
        cypher='MATCH ()\n      CREATE (n)\n      DELETE n',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),
]
