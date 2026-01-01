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
        key='delete1-1',
        feature_path='tck/features/clauses/delete/Delete1.feature',
        scenario='[1] Delete nodes',
        cypher='MATCH (n)\n      DELETE n',
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

    Scenario(
        key='delete1-2',
        feature_path='tck/features/clauses/delete/Delete1.feature',
        scenario='[2] Detach delete node',
        cypher='MATCH (n)\n      DETACH DELETE n',
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

    Scenario(
        key='delete1-3',
        feature_path='tck/features/clauses/delete/Delete1.feature',
        scenario='[3] Detach deleting connected nodes and relationships',
        cypher='MATCH (n:X)\n      DETACH DELETE n',
        graph=graph_fixture_from_create(
            """
            CREATE (x:X)
                  CREATE (x)-[:R]->()
                  CREATE (x)-[:R]->()
                  CREATE (x)-[:R]->()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete1-4',
        feature_path='tck/features/clauses/delete/Delete1.feature',
        scenario='[4] Delete on null node',
        cypher='OPTIONAL MATCH (n)\n      DELETE n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete1-5',
        feature_path='tck/features/clauses/delete/Delete1.feature',
        scenario='[5] Ignore null when deleting node',
        cypher='OPTIONAL MATCH (a:DoesNotExist)\n      DELETE a\n      RETURN a',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete1-6',
        feature_path='tck/features/clauses/delete/Delete1.feature',
        scenario='[6] Detach delete on null node',
        cypher='OPTIONAL MATCH (n)\n      DETACH DELETE n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete1-7',
        feature_path='tck/features/clauses/delete/Delete1.feature',
        scenario='[7] Failing when deleting connected nodes',
        cypher='MATCH (n:X)\n      DELETE n',
        graph=graph_fixture_from_create(
            """
            CREATE (x:X)
                  CREATE (x)-[:R]->()
                  CREATE (x)-[:R]->()
                  CREATE (x)-[:R]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('delete', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='delete1-8',
        feature_path='tck/features/clauses/delete/Delete1.feature',
        scenario='[8] Failing when deleting a label',
        cypher='MATCH (n)\n      DELETE n:Person',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('delete', 'syntax-error', 'xfail'),
    ),
]
